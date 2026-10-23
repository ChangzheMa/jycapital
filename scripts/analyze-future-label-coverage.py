"""仅检查冻结训练期已有标签的同日后续监督覆盖，不生成新收益标签。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLOCKS = ('09:30', '10:00', '10:30', '11:00', '11:30', '13:30', '14:00', '14:30')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def analyze(run):
    record = read(run / 'resident-cache.json')
    directory = Path(record['path'])
    metadata = read(directory / 'manifest.json')
    protocol = read(run / 'protocol.json')
    arrays, hashes = {}, {}
    try:
        for name in ('sample_day', 'sample_stock', 'sample_clock', 'targets', 'days'):
            path = directory / (name + '.npy')
            expected = metadata['files'][path.name]
            actual = digest(path)
            if actual != expected['sha256']:
                raise ValueError('冻结训练标签或位置数组改变: ' + name)
            value = np.load(path, mmap_mode='r', allow_pickle=False)
            if list(value.shape) != expected['shape'] or value.dtype.str != expected['dtype']:
                raise ValueError('训练数组布局不一致')
            arrays[name], hashes[path.name] = value, actual
        day, stock, clock, target = (arrays[name] for name in ('sample_day', 'sample_stock', 'sample_clock', 'targets'))
        if not (len(day) == len(stock) == len(clock) == len(target)) or np.any(day[1:] < day[:-1]):
            raise ValueError('训练样本位置未按交易日排序')
        days = arrays['days'].astype('datetime64[ns]')
        used = days[np.unique(day)]
        if (used < np.datetime64(protocol['splits']['train']['start'])).any() or (used >= np.datetime64(protocol['splits']['train']['stop'])).any():
            raise ValueError('分析只允许训练期标签')
        starts = np.r_[0, np.flatnonzero(day[1:] != day[:-1]) + 1, len(day)]
        pairs = {(q, k): {'query_clock': CLOCKS[q], 'target_clock': CLOCKS[k], 'clock_steps_ahead': k-q,
                          'valid_pairs': 0, 'defined_daily_correlations': 0, 'correlation_sum': 0., 'negative_daily_correlations': 0}
                 for q in range(8) for k in range(q+1, 8)}
        query_counts = np.zeros(8, np.int64)
        groups, eligible_groups, primary_rows = 0, 0, 0
        for start, stop in zip(starts[:-1], starts[1:]):
            slots = stock[start:stop].astype(np.int64) * 8 + clock[start:stop]
            if len(np.unique(slots)) != len(slots):
                raise ValueError('同日股票时点重复')
            table = np.full((len(metadata['stocks']), 8), np.nan)
            present = np.zeros(table.shape, bool)
            table.reshape(-1)[slots] = target[start:stop]
            present.reshape(-1)[slots] = True
            finite = np.isfinite(table)
            eligible = finite.any(axis=1)
            groups += int(present.any(axis=1).sum())
            eligible_groups += int(eligible.sum())
            primary_rows += int(finite.sum())
            for q in range(7):
                query_counts[q] += int((eligible & present[:, q] & finite[:, q+1:].any(axis=1)).sum())
                for k in range(q+1, 8):
                    row = pairs[q, k]
                    valid = eligible & present[:, q] & finite[:, k]
                    row['valid_pairs'] += int(valid.sum())
                    overlap = valid & finite[:, q]
                    if overlap.sum() >= protocol['min_cross_section_size']:
                        left, right = table[overlap, q], table[overlap, k]
                        left, right = left-left.mean(), right-right.mean()
                        denominator = np.sqrt(np.dot(left, left) * np.dot(right, right))
                        if denominator > 0:
                            correlation = float(np.dot(left, right) / denominator)
                            row['defined_daily_correlations'] += 1
                            row['correlation_sum'] += correlation
                            row['negative_daily_correlations'] += int(correlation < 0)
        rows = []
        for row in pairs.values():
            count = row['defined_daily_correlations']
            row['mean_daily_label_correlation'] = row.pop('correlation_sum') / count if count else None
            rows.append(row)
        return {'computed_at': datetime.now().astimezone().isoformat(timespec='seconds'),
            'script_sha256': digest(__file__), 'run_id': run.name, 'cache_key': metadata['cache_key'],
            'source_sha256': hashes, 'cache_manifest_sha256': digest(directory / 'manifest.json'),
            'protocol_sha256': digest(run / 'protocol.json'), 'training_interval': protocol['splits']['train'],
            'training_days': len(used), 'stock_day_groups': groups, 'eligible_stock_day_groups': eligible_groups,
            'finite_primary_rows': primary_rows, 'additional_strict_future_supervision_pairs': sum(r['valid_pairs'] for r in rows),
            'queries_with_any_strict_future_label': dict(zip(CLOCKS, map(int, query_counts))), 'pairs': rows,
            'input_numeric_arrays_read': False, 'validation_or_test_labels_read': False,
            'returns_recomputed': False, 'exposure_read': False, 'model_training_performed': False,
            'method': '读取父缓存中已由官方vwap_return标准化的训练目标。查询须在当时股票池，目标为同日严格后续时点的原有目标；组资格与父训练保持。',
            'limitations': ['监督对重复利用原有标签，不是新增独立样本。',
                '标签之间的相关性不是模型预测能力或正式IC。',
                '后续目标只允许在训练损失中使用，不能进入模型输入、验证选模依据或平台推理。']}
    finally:
        for value in arrays.values():
            value._mmap.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/future-label-coverage-20261023.json')
    args = parser.parse_args()
    result = analyze(ROOT / 'outputs/runs/v1601-price-geometry-s42')
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print({k: result[k] for k in ('training_days', 'stock_day_groups', 'eligible_stock_day_groups', 'finite_primary_rows', 'additional_strict_future_supervision_pairs', 'queries_with_any_strict_future_label')})


if __name__ == '__main__':
    main()
