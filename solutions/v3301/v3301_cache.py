"""FP32冻结表示缓存，按股票日恢复提取；监督目标来自提供的标签键。"""
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch
import numpy as np
import pandas as pd

from artifacts import read_json, sha256, now
from v3201_grouped import StockDayIndex
from v3201_resident import ResidentBatches, require_memory, CLOCK_MINUTES, MINUTE_NS
from v3201_train import prepare_training_targets
from v3201_targets import prepare_rank_training_targets
from v3301_checkpoint import atomic_json
from v3301_text_artifact import state_digest
import v3301_features

FORMAT = 'jycapital-clock-feature-cache-v1'
ARRAYS = {'features': np.dtype('float32'), 'targets': np.dtype('float64'),
          'weights': np.dtype('float64'), 'clock': np.dtype('uint8')}


class ZeroTargets:
    def __getitem__(self, indices):
        return np.zeros(len(indices), np.float32)


def identity_for(config, source, block_manifest, split, encoder_digest):
    identity = {'format': FORMAT, 'split': split, 'parent_model_sha256': config['model']['parent_model_sha256'],
                'encoder_tensor_sha256': encoder_digest, 'feature_dim': config['model']['feature_dim'],
                'input_source': source, 'input_cache_manifest': block_manifest,
                'feature_code_sha256': sha256(v3301_features.__file__), 'builder_sha256': sha256(__file__)}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    return key, identity


def record_array(path):
    value = np.load(path, mmap_mode='r', allow_pickle=False)
    try:
        return {'shape': list(value.shape), 'dtype': value.dtype.str, 'bytes': path.stat().st_size, 'sha256': sha256(path)}
    finally:
        value._mmap.close()


def verify_keys(block, keys, protocol, split):
    keys = keys[['date', 'instrument', protocol['label_column']]].copy().reset_index(drop=True)
    if keys.duplicated(['date', 'instrument']).any() or keys[['date', 'instrument']].isna().any().any():
        raise ValueError('特征样本键重复或为空')
    interval = protocol['splits'][split]
    if not keys.date.between(pd.Timestamp(interval['start']), pd.Timestamp(interval['stop']), inclusive='left').all():
        raise ValueError('特征提取超出固定划分')
    if hasattr(block, 'raw_batch'):
        stamps = block.arrays['days'][block.day].astype('datetime64[ns]').astype(np.int64) + CLOCK_MINUTES[block.clock] * MINUTE_NS
        stocks = pd.Index(block.manifest['stocks']).get_indexer(keys.instrument)
        if not np.array_equal(stamps, keys.date.to_numpy(dtype='datetime64[ns]').astype(np.int64)) or not np.array_equal(stocks, block.stock):
            raise ValueError('表示与原始样本键或行顺序不匹配')
    elif not keys[['date', 'instrument']].equals(block.keys[['date', 'instrument']].reset_index(drop=True)):
        raise ValueError('月内特征样本键不匹配')
    return keys


def weights_for(keys, protocol, split, policy):
    if split != 'train':
        return np.ones(len(keys), np.float64)
    days = keys.date.to_numpy(dtype='datetime64[D]')
    age = (np.datetime64(protocol['splits']['train']['stop'], 'D') - days).astype(np.float64)
    return policy['floor'] + (1 - policy['floor']) * np.exp2(-age / policy['half_life_days'])


def rows_digest(matrix, index, start, stop):
    digest = hashlib.sha256()
    for offset in range(start, stop, 1024):
        rows = index.samples[offset:min(offset + 1024, stop)]
        rows = rows[rows >= 0]
        digest.update(np.ascontiguousarray(matrix[rows]).tobytes())
    return digest.hexdigest()


def build_features(directory, identity, block, keys, encoder, config, protocol, split, report=None):
    directory = Path(directory).resolve()
    if directory.exists():
        saved = read_json(directory / 'manifest.json')
        if saved.get('status') != 'complete' or saved.get('identity') != identity:
            raise ValueError('特征缓存身份不同或尚未完成')
        return saved
    keys = verify_keys(block, keys, protocol, split)
    index = StockDayIndex(block)
    rows, width = len(keys), config['model']['feature_dim']
    staging = directory.with_name(directory.name + '.building')
    if staging.parent.resolve() != directory.parent.resolve():
        raise ValueError('特征临时目录不在目标父目录')
    directory.parent.mkdir(parents=True, exist_ok=True)
    before = state_digest(encoder.state_dict())
    if before != identity['encoder_tensor_sha256']:
        raise ValueError('表示提取器与缓存身份不同')
    progress = {'format': FORMAT, 'status': 'building', 'identity': identity, 'rows': rows,
                'feature_dim': width, 'membership_sha256': index.digest(), 'cursor_groups': 0, 'chunks': []}
    if staging.exists():
        saved = read_json(staging / 'progress.json')
        for k in ('format', 'identity', 'rows', 'feature_dim', 'membership_sha256'):
            if saved.get(k) != progress[k]:
                raise ValueError('未完成特征缓存的来源或成员改变')
        progress = saved
        if progress.get('status') not in ('initializing', 'building'):
            raise ValueError('未知特征缓存准备阶段')
    else:
        staging.mkdir()
        progress['status'] = 'initializing'
        atomic_json(staging / 'progress.json', progress)
    if progress['status'] == 'initializing':
        require_memory(rows * 24 + config['training']['feature_inference_batch_size'] * width * 8 * 4 * 3, 2.)
        if shutil.disk_usage(directory.parent).free < rows * (width * 4 + 32) + 2**30:
            raise OSError('冻结表示缓存磁盘余量不足')
        primary, skipped = prepare_training_targets(keys, protocol)
        ranks, rank_skipped = prepare_rank_training_targets(keys, protocol)
        if skipped != rank_skipped or not np.array_equal(np.isfinite(primary), np.isfinite(ranks)):
            raise ValueError('收益及排名目标有效性不同')
        if split == 'train' and hasattr(block, 'training_targets'):
            cached, _ = block.training_targets()
            if not np.array_equal(cached, primary, equal_nan=True):
                raise ValueError('提供标签与原训练缓存监督不一致')
        # 留出打包缓存可能保存零占位目标；这里始终按真实keys标签转换。
        np.save(staging / 'targets.npy', np.stack([primary, ranks], axis=-1), allow_pickle=False)
        np.save(staging / 'weights.npy', weights_for(keys, protocol, split, config['training']['temporal_weighting']), allow_pickle=False)
        clocks = keys.date.dt.strftime('%H:%M').map({s: i for i, s in enumerate(config['data']['prediction_clock']['times'])})
        if clocks.isna().any():
            raise ValueError('特征包含未知预测时点')
        np.save(staging / 'clock.npy', clocks.to_numpy(np.uint8), allow_pickle=False)
        keys.to_parquet(staging / 'keys.parquet', index=False)
        progress['metadata_hashes'] = {n: sha256(staging / n) for n in ('targets.npy', 'weights.npy', 'clock.npy', 'keys.parquet')}
        matrix = np.lib.format.open_memmap(staging / 'features.npy', mode='w+', dtype=np.float32, shape=(rows, width))
        matrix.flush()
        progress['status'] = 'building'
        atomic_json(staging / 'progress.json', progress)
    else:
        for name, digest in progress['metadata_hashes'].items():
            if sha256(staging / name) != digest:
                raise ValueError('未完成缓存的键或监督数据改变')
        matrix = np.load(staging / 'features.npy', mmap_mode='r+', allow_pickle=False)
        if matrix.shape != (rows, width) or matrix.dtype != np.float32:
            matrix._mmap.close()
            raise ValueError('未完成表示布局改变')
    try:
        cursor = 0
        for chunk in progress['chunks']:
            if chunk['start'] != cursor or chunk['stop'] > len(index.samples) or chunk['stop'] <= cursor or rows_digest(matrix, index, cursor, chunk['stop']) != chunk['sha256']:
                raise ValueError('已完成特征块被修改或游标不连续')
            cursor = chunk['stop']
        if cursor != progress['cursor_groups']:
            raise ValueError('特征恢复游标与完成块不一致')
        batch = config['training']['feature_inference_batch_size']
        device = next(encoder.parameters()).device
        if hasattr(block, 'raw_batch'):
            iterator = ResidentBatches(block, index.representatives, ZeroTargets(), cursor, batch, device)
        else:
            def native():
                for offset in range(cursor, len(index.samples), batch):
                    selected = index.representatives[offset:offset + batch]
                    windows, clock = block.batch(selected)
                    yield offset, selected, {f: torch.from_numpy(v).to(device) for f, v in windows.items()}, torch.from_numpy(clock).to(device), None
            iterator = native()
        pending_start = cursor
        for number, (offset, selected, windows, clock, _) in enumerate(iterator, 1):
            with torch.inference_mode():
                values = encoder.group_features({f: v.contiguous() for f, v in windows.items()}, clock).cpu().numpy()
            positions = index.samples[offset:offset + len(selected)]
            valid = positions >= 0
            if not np.isfinite(values[valid]).all():
                raise FloatingPointError('冻结表示含非有限数')
            matrix[positions[valid]] = values[valid]
            cursor = offset + len(selected)
            if number % config['training']['feature_checkpoint_batches'] == 0 or cursor == len(index.samples):
                matrix.flush()
                progress['chunks'].append({'start': pending_start, 'stop': cursor,
                    'sha256': rows_digest(matrix, index, pending_start, cursor)})
                progress['cursor_groups'] = cursor
                atomic_json(staging / 'progress.json', progress)
                pending_start = cursor
                if report:
                    report('clock_features_progress', split=split, completed_groups=cursor, groups=len(index.samples))
        if cursor != len(index.samples) or state_digest(encoder.state_dict()) != before:
            raise ValueError('冻结表示提取未完整或父权重改变')
    finally:
        matrix.flush()
        matrix._mmap.close()
    saved = {'format': FORMAT, 'status': 'complete', 'created_at': now(), 'identity': identity,
             'rows': rows, 'feature_dim': width, 'encoder_unchanged': True, 'encoder_tensor_sha256': before,
             'targets_from_provided_labels': True, 'files': {n: record_array(staging / (n + '.npy')) for n in ARRAYS},
             'keys_sha256': sha256(staging / 'keys.parquet'), 'progress_sha256': sha256(staging / 'progress.json')}
    atomic_json(staging / 'manifest.json', saved)
    os.replace(staging, directory)
    return saved


class FeatureCache:
    def __init__(self, directory, identity=None):
        self.directory = Path(directory)
        self.manifest = read_json(self.directory / 'manifest.json')
        self.arrays = {}
        saved = self.manifest
        if saved.get('format') != FORMAT or saved.get('status') != 'complete' or (identity is not None and saved['identity'] != identity):
            raise ValueError('特征缓存状态或身份不符')
        require_memory(saved['rows'] * 120, 2.)
        try:
            if set(saved['files']) != set(ARRAYS):
                raise ValueError('特征数组集合不符')
            for name, record in saved['files'].items():
                p = self.directory / (name + '.npy')
                if p.stat().st_size != record['bytes'] or sha256(p) != record['sha256']:
                    raise ValueError('特征或监督数组指纹改变')
                value = np.load(p, mmap_mode='r', allow_pickle=False)
                self.arrays[name] = value
                shape = (saved['rows'], saved['feature_dim']) if name == 'features' else ((saved['rows'], 2) if name == 'targets' else (saved['rows'],))
                if value.shape != shape or value.dtype != ARRAYS[name] or list(value.shape) != record['shape']:
                    raise ValueError('特征数组布局不符')
            if sha256(self.directory / 'keys.parquet') != saved['keys_sha256']:
                raise ValueError('特征原始键或标签改变')
            self.keys = pd.read_parquet(self.directory / 'keys.parquet')
            if len(self.keys) != saved['rows']:
                raise ValueError('缓存键数量不同')
        except BaseException:
            self.close()
            raise

    def clock_rows(self, clock, finite_targets=False):
        selected = self.arrays['clock'] == clock
        if finite_targets:
            selected = selected & np.isfinite(self.arrays['targets']).all(axis=1)
        return np.flatnonzero(selected)

    def close(self):
        for value in self.arrays.values():
            value._mmap.close()
        self.arrays.clear()
        self.keys = None
