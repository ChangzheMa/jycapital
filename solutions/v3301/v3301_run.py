"""冻结表示提取、八时点串行训练、验证组合与一次冻结测试。"""
import argparse
import gc
import os
from pathlib import Path
import random
import shutil
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'common'), str(ROOT / 'evaluation')]
from artifacts import now, read_json, sha256, environment, snapshot_sources, log_event
from research_queue import validate_spec, readiness_error, source_files
from early_stopping import validate_early_stopping
from protocol_compatibility import validate_protocol_compatibility
from training_resources import record_resources
from price_memory_inputs import base_data_manifest
import price_memory_resident as resident
import price_memory_evaluation_cache as evaluation_cache
import v3201_train as parent_train
from v3201_data import ProcessedStore
from v3301_model import MultiFrequencyClockSpecialists
from v3301_cache import FeatureCache, identity_for, build_features
from v3301_checkpoint import atomic_json
from v3301_text_artifact import state_digest
from v3301_fit import fit_readouts, evaluate_bank, save_evaluation, clock_training_step


def validate_training(config, protocol):
    if config.get('solution_id') != 'v3301' or config['model']['family'] != 'clock_specialists':
        raise ValueError('v3301模型类型不匹配')
    parent = config['backbone_config']
    parent_train.validate_training(parent, protocol)
    if config['data'] != parent['data'] or config['model']['feature_dim'] != parent['model']['fusion_hidden_dim']:
        raise ValueError('冻结父输入或表示维度改变')
    if config['model']['clock_count'] != 8 or config['model']['output_heads'] != 2 or config['model']['feature_layout'] != 'parent_fusion_before_experts':
        raise ValueError('八时点读出布局不一致')
    spec = config['training']
    if spec['batching'] != 'serial_clock_cached_features' or spec['resident_cache']['memory_reserve_gib'] != 2.:
        raise ValueError('要求独立时点缓存采样及2GiB预留')
    validate_early_stopping(spec)
    for key in ('epochs', 'batch_size', 'checkpoint_batches', 'feature_checkpoint_batches',
                'feature_inference_batch_size', 'clock_evaluation_batch_size', 'cpu_threads', 'ema_reference_samples'):
        if type(spec.get(key)) is not int or spec[key] < 1:
            raise ValueError('正整数训练配置非法: ' + key)
    for key in ('learning_rate', 'weight_decay', 'gradient_clip'):
        if not np.isfinite(spec.get(key, np.nan)) or spec[key] <= 0:
            raise ValueError('训练数值配置非法: ' + key)
    retained = ('loss', 'huber_delta', 'auxiliary_rank_weight', 'auxiliary_rank_huber_delta',
                'expert_loss_weight', 'router_loss_weight', 'assignment_temperature', 'assignment_iterations', 'temporal_weighting')
    if any(spec[k] != parent['training'][k] for k in retained):
        raise ValueError('分时点训练保留原目标定义和日期权重')
    avg = spec['weight_averaging']
    if avg['method'] != 'sample_count_ema' or not 0 < avg['decay'] < 1 or type(avg['warmup_epochs']) is not int or avg['warmup_epochs'] < 0:
        raise ValueError('样本量EMA配置非法')
    if any(avg.get(k) != v for k, v in {'evaluation_weights': 'ema', 'update_after_optimizer_step': True, 'accumulation_dtype': 'float32'}.items()):
        raise ValueError('分时点EMA执行约定改变')
    count = config['inference'].get('stock_day_batch_size')
    if type(count) is not int or count < 1 or config['inference']['batch_size'] != 8 * count:
        raise ValueError('fast推理名义样本数必须是股票日批量的八倍')
    if protocol['selection_metric'] != 'ic_mean' or config['inference']['factor_scale'] != 1.:
        raise ValueError('只能按验证IC选模并保留父预测单位')


def configure(config, device_name=None):
    torch.set_num_threads(config['training']['cpu_threads'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.use_deterministic_algorithms(config['training']['deterministic'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(device_name or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用')
    return device


def input_directory(store, source, split):
    if split == 'train':
        key = resident.cache_key(store, source, parent_train.prepare_training_targets)
    else:
        key = resident.side_key(evaluation_cache.base_evaluation.cache_key(store.base, split, base_data_manifest(source)))
    return store.training_run.parent / 'price-memory-cache' / key


def training_keys(store):
    interval, protocol = store.protocol['splits']['train'], store.protocol
    keys = store.universe.loc[(store.universe.date >= interval['start']) & (store.universe.date < interval['stop']),
                              ['date', 'instrument']].sort_values(['date', 'instrument']).reset_index(drop=True)
    labels = pq.read_table(store.data_root / (protocol['label_table'] + '.parquet'),
        columns=['date', 'instrument', protocol['label_column']],
        filters=[('date', '>=', pd.Timestamp(interval['start'])), ('date', '<', pd.Timestamp(interval['stop']))]).to_pandas()
    keys = keys.merge(labels, on=['date', 'instrument'], how='left', validate='one_to_one')
    horizon = keys.date + pd.Timedelta(minutes=protocol['label_horizon_minutes'])
    keys.loc[(horizon >= pd.Timestamp(interval['stop'])) | (horizon.dt.normalize() != keys.date.dt.normalize()), protocol['label_column']] = np.nan
    return keys


def draw_history(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history = pd.read_csv(run / 'history.csv')
    selection = read_json(run / 'selection.json')
    figure, axes = plt.subplots(2, 4, figsize=(16, 7))
    for q, axis in enumerate(axes.flat):
        rows = history.loc[history.clock_id == q]
        chosen = selection['clock_selections'][q]
        axis.plot(rows.epoch, rows.validation_ic_mean, color='tab:blue', label='validation IC')
        axis.axhline(chosen['parent_validation_ic'], color='gray', linestyle='--', label='parent')
        loss_axis = axis.twinx()
        loss_axis.plot(rows.epoch, rows.train_huber_loss, color='tab:orange', alpha=.7)
        axis.set(title=chosen['clock'], xlabel='Clock epoch', ylabel='Validation IC')
        loss_axis.set_ylabel('Train Huber')
    figure.tight_layout()
    figure.savefig(run / 'learning-curves.png', dpi=150)
    plt.close(figure)


def extract_split(store, encoder, config, source, split, run, cache_root, mode):
    input_path = input_directory(store, source, split)
    report = lambda event, **fields: log_event(run, event, **fields)
    encoder_digest = state_digest(encoder.state_dict())
    records_path = run / 'feature-caches.json'
    records = read_json(records_path) if records_path.exists() else {}
    if (input_path / 'manifest.json').exists():
        key, identity = identity_for(config, source, read_json(input_path / 'manifest.json'), split, encoder_digest)
        destination = cache_root / key
        if destination.exists():
            result = FeatureCache(destination, identity)
            records[split] = {'path': str(destination), 'cache_key': key, 'manifest_sha256': sha256(destination / 'manifest.json')}
            atomic_json(records_path, records)
            return result
    started = time.perf_counter()
    block = wrapper = None
    try:
        if split == 'train':
            wrapper = resident.open_resident_store(store, run, parent_train.prepare_training_targets, report=report)
            block, keys = wrapper.block, training_keys(store)
        else:
            wrapper = evaluation_cache.open_evaluation_data(store, split, source, mode=mode, report=report)
            block, keys = wrapper.block, wrapper.keys
        key, identity = identity_for(config, source, block.manifest, split, encoder_digest)
        destination = cache_root / key
        build_features(destination, identity, block, keys, encoder, config, store.protocol, split, report)
        records[split] = {'path': str(destination), 'cache_key': key, 'manifest_sha256': sha256(destination / 'manifest.json')}
        atomic_json(records_path, records)
    finally:
        if block is not None:
            block.close()
        block = wrapper = None
        gc.collect()
        if next(encoder.parameters()).device.type == 'cuda':
            torch.cuda.empty_cache()
    report('clock_features_ready', split=split, seconds=time.perf_counter() - started)
    return FeatureCache(destination, identity)


def execute(args):
    if not args.training_confirmed:
        raise ValueError('v3301正式训练必须由训练session显式确认')
    config, protocol = read_json(args.config), read_json(args.protocol)
    validate_training(config, protocol)
    run = args.run_dir.resolve()
    if not args.reproduction_only:
        candidate = validate_spec(ROOT, HERE / 'experiment.json')
        if candidate['status'] != 'ready' or candidate['run_id'] != run.name or readiness_error(ROOT, candidate):
            raise ValueError('候选未就绪或源码指纹改变')
    parent_run = args.parent_run.resolve() if args.parent_run else ROOT / 'outputs/runs' / config['parent_run_id']
    parent_manifest = read_json(parent_run / 'manifest.json')
    if parent_manifest['status'] != 'complete' or sha256(parent_run / 'best.pt') != config['model']['parent_model_sha256']:
        raise ValueError('冻结父模型或完成状态不符')
    if read_json(parent_run / 'config.json') != config['backbone_config']:
        raise ValueError('父模型配置不符')
    validate_protocol_compatibility(protocol, read_json(parent_run / 'protocol.json'))
    for relative, digest in parent_manifest['source']['files'].items():
        if relative.startswith('solutions/v3201/') and relative.endswith('.py') and sha256(HERE / Path(relative).name) != digest:
            raise ValueError('父版推理／数据源代码不是实际冻结版本: ' + relative)
    parent_checkpoint = torch.load(parent_run / 'best.pt', map_location='cpu', weights_only=True)
    if parent_checkpoint['config'] != config['backbone_config'] or parent_checkpoint.get('weight_source') != 'ema':
        raise ValueError('父checkpoint配置或EMA来源不一致')
    if args.resume:
        manifest = read_json(run / 'manifest.json')
        if manifest['status'] == 'complete':
            raise ValueError('已完成运行不重复训练')
        if read_json(run / 'config.json') != config or read_json(run / 'protocol.json') != protocol:
            raise ValueError('恢复配置或协议改变')
        for p, digest in source_files(ROOT, 'v3301').items():
            if manifest['source']['files'].get(p) != digest or sha256(run / 'source' / p) != digest:
                raise ValueError('恢复源码与冻结运行不同: ' + p)
    else:
        run.mkdir(parents=True, exist_ok=False)
        atomic_json(run / 'config.json', config)
        atomic_json(run / 'protocol.json', protocol)
        manifest = {'run_id': run.name, 'solution_id': 'v3301', 'parent_solution_id': 'v3201',
            'parent_run_id': config['parent_run_id'], 'status': 'running', 'stage': 'feature_extraction',
            'created_at': now(), 'formal_attempts': 1, 'selection_kind': 'independent_clock_validation',
            'source': snapshot_sources(ROOT, run, 'v3301')}
        atomic_json(run / 'manifest.json', manifest)
    device = configure(config, args.device)
    model = MultiFrequencyClockSpecialists(config)
    model.initialize_from_parent(parent_checkpoint['state_dict'])
    del parent_checkpoint
    encoder_digest = state_digest(model.encoder.state_dict())
    original_readout = state_digest(model.readouts.heads[0].state_dict())
    env = environment()
    env.update(device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        parameters=sum(p.numel() for p in model.parameters()), model_family='clock_specialists',
        shared_frozen_parameters=sum(p.numel() for p in model.encoder.parameters()),
        trainable_parameters_per_stage=sum(p.numel() for p in model.readouts.heads[0].parameters()),
        configured_host_memory_reserve_gib=2., evaluation_precision='float32')
    atomic_json(run / 'environment.json', env)
    (run / 'requirements-lock.txt').write_text('\n'.join(f'{k}=={v}' for k, v in env['packages'].items()) + '\n', encoding='utf-8')
    log_event(run, 'training_resumed' if args.resume else 'training_started', parameters=env['parameters'], formal_attempts=1)
    caches = []
    try:
        store = ProcessedStore(args.training_run, args.holdout_run, args.data_root, config['backbone_config'], protocol)
        verification = run / 'input-verification'
        verification.mkdir(exist_ok=True)
        atomic_json(verification / 'protocol.json', protocol)
        parent_train.data_manifest(store, verification)
        source = read_json(verification / 'data-manifest.json')
        if source != read_json(parent_run / 'data-manifest.json'):
            raise ValueError('子运行与冻结父版的数据来源不同')
        if (run / 'data-manifest.json').exists() and source != read_json(run / 'data-manifest.json'):
            raise ValueError('恢复数据身份不同，原记录保持不变')
        if not (run / 'data-manifest.json').exists():
            shutil.copyfile(verification / 'data-manifest.json', run / 'data-manifest.json')
        (run / 'normalization').mkdir(exist_ok=True)
        for p in (verification / 'normalization').glob('*.json'):
            target = run / 'normalization' / p.name
            if target.exists() and sha256(target) != sha256(p):
                raise ValueError('冻结归一化参数改变')
            if not target.exists():
                shutil.copyfile(p, target)
        cache_root = (args.feature_cache_root or args.training_run.parent / 'clock-feature-cache').resolve()
        if args.resume and manifest.get('stage') == 'test_evaluation':
            selection = read_json(run / 'selection.json')
            if sha256(run / 'best.pt') != selection['model_sha256']:
                raise ValueError('测试恢复时组合权重改变')
            selected = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
            if selected['config'] != config:
                raise ValueError('测试恢复时权重配置不同')
            model.load_state_dict(selected['state_dict'])
        else:
            model.encoder.to(device)
            train = extract_split(store, model.encoder, config, source, 'train', run, cache_root, args.evaluation_cache_mode)
            caches.append(train)
            validation = extract_split(store, model.encoder, config, source, 'validation', run, cache_root, args.evaluation_cache_mode)
            caches.append(validation)
            model.encoder.cpu()
            manifest.update(status='running', stage='clock_training')
            atomic_json(run / 'manifest.json', manifest)
            selection = fit_readouts(model, train, validation, config, protocol, run, device, args.resume,
                                    lambda event, **fields: log_event(run, event, **fields))
            for cache in caches:
                cache.close()
            caches.clear()
            draw_history(run)
        manifest.update(stage='test_evaluation', selection=selection)
        atomic_json(run / 'manifest.json', manifest)
        model.encoder.to(device)
        test = extract_split(store, model.encoder, config, source, 'test', run, cache_root, args.evaluation_cache_mode)
        caches.append(test)
        frame, result = evaluate_bank(model.readouts, test, config, protocol, device)
        save_evaluation(run, 'test', frame, result)
        if state_digest(model.encoder.state_dict()) != encoder_digest or sha256(parent_run / 'best.pt') != config['model']['parent_model_sha256']:
            raise ValueError('冻结父权重改变')
        atomic_json(run / 'loss-consistency.json', {'loss': config['training']['loss'],
            'provided_label_unchanged': True, 'clock_count': 8, 'encoder_unchanged': True,
            'encoder_tensor_sha256': encoder_digest, 'initial_readout_tensor_sha256': original_readout,
            'parent_weight_sha256': config['model']['parent_model_sha256'],
            'selection': 'per_clock_validation_ic_with_parent_fallback', 'test_used_for_selection': False})
        manifest.update(status='complete', stage='complete', completed_at=now(),
                        metrics={s: read_json(run / 'evaluation' / s / 'metrics.json') for s in ('validation', 'test')})
        atomic_json(run / 'manifest.json', manifest)
        record_resources(run, device)
        log_event(run, 'completed', validation_ic=selection['best_validation_ic'], test_ic=result[0]['ic_mean'])
        return manifest
    except BaseException as error:
        # 保留恢复阶段，允许测试期中断时直接恢复冻结后的测试。
        manifest.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error), stopped_at=now())
        atomic_json(run / 'manifest.json', manifest)
        raise
    finally:
        for cache in caches:
            cache.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--config', type=Path, default=HERE / 'model-config.json')
    result.add_argument('--protocol', type=Path, default=ROOT / 'evaluation/protocols/local-v2-ic.json')
    result.add_argument('--parent-run', type=Path)
    result.add_argument('--training-run', type=Path, required=True)
    result.add_argument('--holdout-run', type=Path, required=True)
    result.add_argument('--data-root', type=Path, required=True)
    result.add_argument('--run-dir', type=Path, required=True)
    result.add_argument('--feature-cache-root', type=Path)
    result.add_argument('--evaluation-cache-mode', choices=('auto', 'resident', 'stream'), default='stream')
    result.add_argument('--device', choices=('cpu', 'cuda'))
    result.add_argument('--training-confirmed', action='store_true')
    result.add_argument('--reproduction-only', action='store_true')
    result.add_argument('--resume', action='store_true')
    return result


def main():
    execute(parser().parse_args())
