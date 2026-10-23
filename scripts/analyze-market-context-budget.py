"""只读训练标签和位置元数据，估算完整市场上下文及日期内更新数量。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'solutions/v2601'))
from v2601_temporal import training_calendar_normalizer


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/v2601-market-context-budget.json')
    args = parser.parse_args()
    run = ROOT/'outputs/runs/v1601-price-geometry-s42'
    directory = Path(read(run/'resident-cache.json')['path'])
    manifest, config, protocol = read(directory/'manifest.json'), read(ROOT/'solutions/v2601/model-config.json'), read(run/'protocol.json')
    arrays, hashes = {}, {}
    try:
        for name in ('sample_day', 'sample_stock', 'targets', 'days'):
            path = directory/(name+'.npy')
            expected = manifest['files'][path.name]
            hashes[path.name] = digest(path)
            if hashes[path.name] != expected['sha256']:
                raise ValueError('冻结标签或位置数组改变')
            arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
        day, stock, target = arrays['sample_day'], arrays['sample_stock'], arrays['targets']
        if np.any(day[1:] < day[:-1]):
            raise ValueError('训练样本必须按日期排序')
        bounds = np.r_[0, np.flatnonzero(day[1:] != day[:-1])+1, len(day)]
        context, supervised = [], []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            context.append(len(np.unique(stock[lo:hi])))
            supervised.append(len(np.unique(stock[lo:hi][np.isfinite(target[lo:hi])])))
        context, supervised = np.array(context, np.int64), np.array(supervised, np.int64)
        batch = config['training']['stock_day_batch_size']
        snapshots = sum(config['model']['market_conditioning']['snapshot_bars'].values()) * 75
        calendar = training_calendar_normalizer(SimpleNamespace(days=arrays['days']), config, protocol)
        result = {'computed_at': datetime.now().astimezone().isoformat(timespec='seconds'),
            'script_sha256': digest(__file__), 'temporal_source_sha256': digest(ROOT/'solutions/v2601/v2601_temporal.py'),
            'model_config_sha256': digest(ROOT/'solutions/v2601/model-config.json'),
            'base_cache_key': manifest['cache_key'], 'source_sha256': hashes,
            'training_dates': len(context), 'context_stock_days': int(context.sum()),
            'supervised_stock_days': int(supervised.sum()),
            'minimum_context_stocks': int(context.min()), 'maximum_context_stocks': int(context.max()),
            'target_stock_day_batch_size': batch,
            'optimizer_steps_per_full_pass': int(((supervised+batch-1)//batch).sum()),
            'parent_globally_mixed_steps_per_full_pass': int((supervised.sum()+batch-1)//batch),
            'snapshot_features_per_clock': snapshots,
            'maximum_raw_context_tensor_bytes': int(context.max()) * 8 * snapshots * 4,
            'raw_context_dtype': 'float32', 'calendar_weighting': calendar,
            'model_training_performed': False, 'input_numeric_arrays_read': False,
            'validation_or_test_labels_read': False, 'returns_recomputed': False, 'exposure_read': False,
            'scope': '只读取父训练缓存的标签与位置，按当前成员建立完整上下文计数；标签仅判断监督目标资格。',
            'limitations': ['原始上下文张量大小不是GPU/进程峰值。',
                '每轮更新数只描述日期内分批，不是耗时或IC改善证据。',
                '日期顺序、日期归一化和目标批次均为新组合的研究变量。']}
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        print({key:result[key] for key in ('training_dates','context_stock_days','supervised_stock_days','maximum_context_stocks','optimizer_steps_per_full_pass','parent_globally_mixed_steps_per_full_pass','maximum_raw_context_tensor_bytes','calendar_weighting')})
    finally:
        for value in arrays.values():
            value._mmap.close()


if __name__ == '__main__':
    main()
