"""完成后核验八时点选模；供训练session代替单一best_epoch的检查。"""
import argparse
import math
from pathlib import Path
import sys

import torch
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'common'), str(ROOT / 'evaluation')]
from artifacts import read_json, sha256, now
from early_stopping import select_epoch, patience_epochs, should_stop
from baseline_evaluation import evaluate
from v3301_checkpoint import atomic_json
from v3301_text_artifact import state_digest


def verify_run(run):
    run = Path(run).resolve()
    manifest = read_json(run / 'manifest.json')
    selection = read_json(run / 'selection.json')
    config, protocol = read_json(run / 'config.json'), read_json(run / 'protocol.json')
    if manifest['status'] != 'complete' or manifest['stage'] != 'complete' or selection['best_epoch'] is not None:
        raise ValueError('分时点运行未完成或伪造了单一最佳轮数')
    if manifest['selection'] != selection or sha256(run / 'best.pt') != selection['model_sha256']:
        raise ValueError('完整选择记录或权重改变')
    chosen = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    if chosen['config'] != config or chosen['clock_selections'] != selection['clock_selections'] or len(selection['clock_selections']) != 8:
        raise ValueError('八时点checkpoint与选择记录不同')
    if not all(torch.isfinite(v).all() for v in chosen['state_dict'].values()):
        raise ValueError('权重非有限')
    encoder = {k.removeprefix('encoder.'): v for k, v in chosen['state_dict'].items() if k.startswith('encoder.')}
    encoder_digest = state_digest(encoder)
    consistency = read_json(run / 'loss-consistency.json')
    if (encoder_digest != consistency['encoder_tensor_sha256'] or
            consistency['parent_weight_sha256'] != config['model']['parent_model_sha256']):
        raise ValueError('共享编码器或父模型身份改变')
    history = pd.read_csv(run / 'history.csv')
    for q, record in enumerate(selection['clock_selections']):
        directory = run / 'clocks' / str(q)
        if record['clock_id'] != q or sha256(directory / 'best.pt') != record['checkpoint_sha256']:
            raise ValueError('时点最佳文件或顺序改变')
        component = torch.load(directory / 'best.pt', map_location='cpu', weights_only=True)
        saved = {k.removeprefix(f'readouts.heads.{q}.'): v for k, v in chosen['state_dict'].items() if k.startswith(f'readouts.heads.{q}.')}
        if state_digest(saved) != state_digest(component['state_dict']) or state_digest(saved) != record['tensor_sha256']:
            raise ValueError('最终组合未使用各时点选定权重')
        initial = read_json(directory / 'initial-validation.json')['metrics']['ic_mean']
        state = {'best': initial, 'best_epoch': 0, 'stale': 0}
        rows = history.loc[history.clock_id == q]
        if rows.epoch.tolist() != list(range(1, record['epochs_completed'] + 1)):
            raise ValueError('时点逐轮记录不连续')
        for row in rows.itertuples():
            if should_stop(state):
                raise ValueError('早停之后仍记录额外训练轮数')
            improved = select_epoch(state, row.validation_ic_mean, row.epoch, bool(row.selection_eligible))
            if improved != bool(row.improved) or state['best_epoch'] != row.best_epoch or patience_epochs(state['best_epoch']) != row.patience_epochs:
                raise ValueError('独立早停或验证选模无法重放')
        if state['best_epoch'] != record['best_epoch'] or not math.isclose(state['best'], record['best_validation_ic'], abs_tol=1e-12):
            raise ValueError('独立时点选择与重放结果不同')
        if record['kept_parent'] != (record['best_epoch'] == 0):
            raise ValueError('父读出保留标志不符')
    checks = {}
    caches = read_json(run / 'feature-caches.json')
    for split in ('validation', 'test'):
        record = caches[split]
        directory = Path(record['path'])
        if sha256(directory / 'manifest.json') != record['manifest_sha256']:
            raise ValueError('特征清单改变')
        metadata = read_json(directory / 'manifest.json')
        if (metadata['identity']['encoder_tensor_sha256'] != encoder_digest or
                metadata['identity']['input_source'] != read_json(run / 'data-manifest.json') or
                metadata['identity']['parent_model_sha256'] != config['model']['parent_model_sha256']):
            raise ValueError('评测缓存与冻结编码器或数据来源不一致')
        if sha256(directory / 'keys.parquet') != metadata['keys_sha256']:
            raise ValueError('评测原始键或标签改变')
        keys = pd.read_parquet(directory / 'keys.parquet')
        predictions = pd.read_parquet(run / 'evaluation' / split / 'predictions.parquet')
        metrics, sections, _ = evaluate(predictions, keys, protocol)
        saved = read_json(run / 'evaluation' / split / 'metrics.json')
        for k, value in metrics.items():
            if isinstance(value, (float, int)):
                if not math.isclose(value, saved[k], rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError('正式评测指标不一致: ' + k)
            elif value != saved[k]:
                raise ValueError('正式评测字段不一致: ' + k)
        if metrics['coverage'] != 1. or metrics['min_cross_section_coverage'] != 1.:
            raise ValueError('组合覆盖不足')
        if split == 'validation':
            grouped = sections.groupby(sections.date.dt.strftime('%H:%M')).ic.mean()
            for item in selection['clock_selections']:
                if not math.isclose(grouped[item['clock']], item['best_validation_ic'], abs_tol=1e-12):
                    raise ValueError('组合验证与时点选择分数不一致')
        checks[split] = {'rows': len(predictions), 'metrics_recomputed': True, 'coverage': metrics['coverage']}
    for path, digest in manifest['source']['files'].items():
        if sha256(run / 'source' / path) != digest:
            raise ValueError('运行冻结源码改变')
    report = {'status': 'passed', 'checked_at': now(), 'selection_kind': 'independent_clock_validation',
              'all_eight_selections_replayed': True, 'selected_model_sha256': selection['model_sha256'],
              'final_components_exact': True, 'frozen_encoder_exact': True,
              'splits': checks, 'formal_training_performed': False}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = verify_run(args.run_dir)
    if args.output:
        atomic_json(args.output, report)
    print(report)
