"""从两个已完成运行冻结组合并评估双集；不启动任何基础模型训练。"""

import argparse
from pathlib import Path
import shutil
import sys
import time

import torch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (HERE, ROOT / 'common', ROOT / 'evaluation', ROOT / 'scripts'):
    sys.path.insert(0, str(directory))
from artifacts import read_json, write_json, now, sha256, environment, snapshot_sources
from compare import compare
from baseline_evaluation import evaluate, _correlation
from signal_diagnostics import intraday_diagnostics
from v1401_combine import align_components, combine_scores, KEYS


def materialize(run, config):
    if run.exists():
        raise FileExistsError('组合运行目录已存在，不能覆盖既有评估')
    components = config['model']['components']
    parents = [ROOT / 'outputs/runs' / item['run_id'] for item in components]
    for split in ('validation', 'test'):
        compare(parents, split)
    for parent, item in zip(parents, components):
        if sha256(parent / 'best.pt') != item['model_sha256'] or sha256(parent / 'config.json') != item['config_sha256']:
            raise ValueError('组件权重或配置与预声明不符')
        for split in ('validation', 'test'):
            if sha256(parent / 'evaluation' / split / 'predictions.parquet') != item['prediction_sha256'][split]:
                raise ValueError('组件预测文件在预声明后改变')
    protocol = read_json(parents[0] / 'protocol.json')
    data = read_json(parents[0] / 'data-manifest.json')
    data_root = Path(data['root'])
    tables = {protocol['universe_table'], protocol['label_table']}
    for item in data['files']:
        if Path(item['path']).stem in tables and sha256(data_root / item['path']) != item['sha256']:
            raise ValueError('股票池或标签文件与冻结评估不一致')
    component_configs = [read_json(parent / 'config.json') for parent in parents]
    preprocessing = [read_json(parent / 'source/solutions' / item['solution_id'] / 'config.json')
                     for parent, item in zip(parents, components)]
    # 元数据标识不同；数据和处理政策必须一致，才共享平台预处理。
    for key in ('data', 'processing'):
        if preprocessing[0][key] != preprocessing[1][key]:
            raise ValueError('组件预处理政策不同，不能共享处理')
    for frequency in ('bar1m', 'bar5m', 'bar30m'):
        if sha256(parents[0] / 'normalization' / (frequency + '.json')) != sha256(parents[1] / 'normalization' / (frequency + '.json')):
            raise ValueError('组件归一化统计量不同')
    run.mkdir(parents=True)
    write_json(run / 'config.json', config)
    shutil.copyfile(parents[0] / 'protocol.json', run / 'protocol.json')
    shutil.copyfile(parents[0] / 'data-manifest.json', run / 'data-manifest.json')
    shutil.copytree(parents[0] / 'normalization', run / 'normalization')
    source = snapshot_sources(ROOT, run, 'v1401')
    for parent, item in zip(parents, components):
        relative = Path('solutions') / item['solution_id']
        original = parent / 'source' / relative
        shutil.copytree(original, run / 'source' / relative)
        for path in (run / 'source' / relative).rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.json', '.txt'):
                source['files'][path.relative_to(run / 'source').as_posix()] = sha256(path)
        sys.path.insert(0, str(run / 'source' / relative))
    manifest = {'run_id': run.name, 'solution_id': 'v1401', 'status': 'running', 'stage': 'composition_evaluation',
                'created_at': now(), 'source': source, 'parent_run_id': config['parent_run_id'],
                'parent_solution_id': config['parent_solution_id'], 'references': config['references'],
                'additional_training': False, 'components': components}
    write_json(run / 'manifest.json', manifest)
    from v1401_model import FrozenEnsemble
    torch.set_num_threads(config['training']['cpu_threads'])
    model = FrozenEnsemble(component_configs).eval()
    for model_part, parent in zip(model.components, parents):
        checkpoint = torch.load(parent / 'best.pt', map_location='cpu', weights_only=True)
        model_part.load_state_dict(checkpoint['state_dict'], strict=True)
    torch.save({'state_dict': model.state_dict(), 'config': config, 'component_configs': component_configs,
                'epoch': None, 'weight_source': 'frozen_component_ema'}, run / 'best.pt')
    env = environment()
    env.update(device='cpu', device_name='CPU frozen-prediction composition', parameters=sum(p.numel() for p in model.parameters()),
               additional_training=False, trainable_parameters=0, model_family='hierarchical_gru',
               inference_composition='two_frozen_backbones_shared_preprocessing')
    write_json(run / 'environment.json', env)
    selection = {'best_epoch': None, 'epochs_completed': 0, 'selection_metric': 'ic_mean',
                 'method': 'predeclared_frozen_equal_composition', 'evaluation_weights': 'frozen_component_ema',
                 'components': components, 'model_sha256': sha256(run / 'best.pt'), 'selected_at': now(),
                 'training_seconds': 0.0, 'optimizer_steps': 0, 'stop_reason': 'no_additional_training'}
    # 权重与组合规则在双集计算前已经冻结。
    write_json(run / 'selection.json', selection)
    pd.DataFrame(columns=['epoch', 'seconds', 'train_loss', 'validation_loss', 'ic_mean', 'rank_ic_mean']).to_csv(
        run / 'history.csv', index=False, encoding='utf-8-sig')
    metrics, diagnostics = {}, {}
    started = time.perf_counter()
    for split in ('validation', 'test'):
        interval = protocol['splits'][split]
        filters = [('date', '>=', pd.Timestamp(interval['start'])), ('date', '<', pd.Timestamp(interval['stop']))]
        keys = pd.read_parquet(data_root / (protocol['universe_table'] + '.parquet'), columns=KEYS, filters=filters)
        labels = pd.read_parquet(data_root / (protocol['label_table'] + '.parquet'), columns=KEYS + [protocol['label_column']], filters=filters)
        keys = keys.merge(labels, on=KEYS, how='left', validate='one_to_one').sort_values(KEYS).reset_index(drop=True)
        horizon = keys.date + pd.Timedelta(minutes=protocol['label_horizon_minutes'])
        admissible = (horizon < pd.Timestamp(interval['stop'])) & (horizon.dt.normalize() == keys.date.dt.normalize())
        keys.loc[~admissible, protocol['label_column']] = np.nan
        frames = [pd.read_parquet(parent / 'evaluation' / split / 'predictions.parquet') for parent in parents]
        # 验证标签重建口径；复用原冻结预测，不重新执行市场窗口推理。
        reference_metrics, _, _ = evaluate(frames[0], keys, protocol)
        expected = read_json(parents[0] / 'evaluation' / split / 'metrics.json')
        for key in ('ic_mean', 'rank_ic_mean', 'mse', 'mae', 'coverage', 'expected_rows', 'evaluated_rows'):
            if not np.isclose(reference_metrics[key], expected[key], rtol=0, atol=1e-12):
                raise ValueError('重建标签口径与父版评估不符: ' + key)
        prediction_keys, scores = align_components(*frames)
        predictions = combine_scores(prediction_keys, scores)
        metrics[split], sections, monthly = evaluate(predictions, keys, protocol)
        directory = run / 'evaluation' / split
        directory.mkdir(parents=True)
        write_json(directory / 'metrics.json', metrics[split])
        predictions.to_parquet(directory / 'predictions.parquet', index=False)
        sections.to_parquet(directory / 'cross-sections.parquet', index=False)
        monthly.to_csv(directory / 'monthly.csv', index=False, encoding='utf-8-sig')
        correlation = _correlation(pd.Series(scores[:, 0]), pd.Series(scores[:, 1]), prediction_keys.date.reset_index(drop=True))
        signal, transitions = intraday_diagnostics(predictions)
        transitions.to_csv(directory / 'signal-transitions.csv', index=False, encoding='utf-8-sig')
        diagnostics[split] = {'member_prediction_correlation_mean': float(correlation.mean()),
            'member_prediction_correlation_undefined_sections': int(correlation.isna().sum()), 'signal_proxy': signal,
            'parent_label_reconstruction_matches': True,
            'prediction_sha256': sha256(directory / 'predictions.parquet')}
        print({'split': split, **{key: metrics[split][key] for key in ('ic_mean','rank_ic_mean','ic_ir','rank_ic_ir','coverage')},
               'member_prediction_correlation_mean': float(correlation.mean()), 'signal_proxy': signal}, flush=True)
    write_json(run / 'composition-diagnostics.json', diagnostics)
    write_json(run / 'composition.json', {'declared_config_sha256': sha256(run / 'config.json'), 'additional_training': False,
        'component_weights_unchanged': True, 'component_local_predictions_reused': True,
        'market_inference_repeated': False, 'composition_and_evaluation_seconds': time.perf_counter() - started,
        'aggregation': config['aggregation'], 'input_and_normalization_compatible': True})
    selection['best_validation_metric'] = metrics['validation']['ic_mean']
    write_json(run / 'selection.json', selection)
    manifest.update(status='complete', stage='complete', completed_at=now(), selection=selection, metrics=metrics)
    write_json(run / 'manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'model-config.json')
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'outputs/runs/v1401-frozen-equal-ensemble-s42')
    args = parser.parse_args()
    materialize(args.run_dir.resolve(), read_json(args.config))


if __name__ == '__main__':
    main()
