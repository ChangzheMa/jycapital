"""只读冻结训练统计，量化价格时间路径的坐标尺度；不读取标签或训练模型。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FREQUENCIES = ('bar1m', 'bar5m', 'bar30m')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    if not np.isfinite(values).all():
        raise ValueError('汇总值非有限')
    return dict(zip(('min', 'p05', 'p25', 'median', 'p75', 'p95', 'max'),
                    map(float, np.quantile(values, [0, .05, .25, .5, .75, .95, 1]))))


def analyze(prepared, cache_manifest, coordinate_summary):
    config_path = prepared / 'config.json'
    config = read(config_path)['normalization']
    if config['price_transform'] != 'log' or config['ddof'] != 0:
        raise ValueError('分析要求现有对数价格、总体标准差契约')
    cache = read(cache_manifest)
    stocks = cache['stocks']
    minimum, epsilon = config['min_observations'], config['scale_epsilon']
    results = []
    hashes = {str(config_path.relative_to(ROOT)): digest(config_path),
              str(cache_manifest.relative_to(ROOT)): digest(cache_manifest)}
    for frequency in FREQUENCIES:
        path = prepared / 'normalization' / (frequency + '.json')
        stats = read(path)
        if stats['fit_interval']['stop'] != '2024-01-01' or stats['ddof'] != 0:
            raise ValueError('仅接受冻结训练范围统计')
        hashes[str(path.relative_to(ROOT))] = digest(path)
        rows = []
        for j, field in enumerate(stats['fields']):
            if field not in ('open', 'high', 'low', 'close') and 'price' not in field:
                continue
            scales = []
            for stock in stocks:
                item = stats['stocks'].get(stock)
                if item is None or item['count'][j] < minimum:
                    continue
                scale = np.sqrt(item['m2'][j] / item['count'][j])
                if np.isfinite(scale) and scale > epsilon:
                    scales.append(scale)
            scales = np.asarray(scales, dtype=np.float64)
            rows.append({'field': field, 'trusted_fixed_stocks': len(scales),
                         'fallback_stocks_excluded': len(stocks) - len(scales),
                         'log_price_standard_deviation': quantiles(scales),
                         'uniform_percent_log_coordinate_to_z_delta_ratio': quantiles(100 * scales),
                         'z_delta_for_hypothetical_one_percent_price_change': quantiles(np.log1p(.01) / scales)})
        results.append({'frequency': frequency, 'cache_stocks': len(stocks),
                        'fit_interval': stats['fit_interval'], 'fields': rows})
    hashes[str(coordinate_summary.relative_to(ROOT))] = digest(coordinate_summary)
    preprocessing = ROOT / 'solutions/v1601/v1601_preprocessing.py'
    hashes[str(preprocessing.relative_to(ROOT))] = digest(preprocessing)
    coordinate = read(coordinate_summary)
    anchors = []
    for entry in coordinate['coordinate_input_verification']['cache_verification']:
        path = ROOT / 'outputs/preprocessing/coordinate-cache' / entry['cache_key'] / 'manifest.json'
        metadata = read(path)
        if digest(path) != entry['manifest_sha256']:
            raise ValueError('旁路清单与冻结结果不一致')
        records = {name: value for name, value in metadata['files'].items() if name.endswith('-anchor-values.npy')}
        if len(records) != 3 or any(v['dtype'] != '<f4' for v in records.values()):
            raise ValueError('预期三频FP32价格中心数组')
        anchors.append({'split': entry['split'], 'cache_key': entry['cache_key'],
                        'manifest_sha256': entry['manifest_sha256'], 'files': records,
                        'anchor_array_bytes': sum(v['array_bytes'] for v in records.values()),
                        'scope': '仅核对冻结清单；没有本次重新读取或哈希全量数值数组'})
    return {'computed_at': datetime.now().astimezone().isoformat(timespec='seconds'),
            'script_sha256': digest(__file__), 'model_training_performed': False,
            'raw_market_data_read': False, 'label_table_read': False, 'exposure_read': False,
            'method': {'population': '父几何训练缓存股票；仅汇总满足原固定统计资格的价格字段，回退股票单列并排除',
                       'formula': 'delta_z = delta_log_price / frozen_stock_field_std; delta_u = 100 * delta_log_price',
                       'one_percent_move': '假设输入变化，用于表达尺度比较，不是读取、计算或替代任何收益标签'},
            'source_sha256': hashes, 'results': results, 'existing_anchor_storage': anchors,
            'existing_anchor_array_bytes_all_splits': sum(x['anchor_array_bytes'] for x in anchors),
            'limitations': ['这不是收益预测试验，也不证明哪个尺度最优。',
                '同字段固定z原则上保留时间变化，尺度差异不等于信息不可恢复。',
                '回退分支的统计随历史变化，本报告不把固定公式外推到这些样本。',
                '存储数值是已有FP32中心数组的体积，不是新加载器已实现的内存峰值。']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/price-path-scale-audit-20261023.json')
    args = parser.parse_args()
    record = read(ROOT / 'outputs/runs/v1601-price-geometry-s42/resident-cache.json')
    result = analyze(ROOT / 'outputs/preprocessing/v2-20261010', Path(record['path']) / 'manifest.json',
                     ROOT / 'experiments/v2001-gru.json')
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for row in result['results']:
        close = next(x for x in row['fields'] if x['field'] == 'close')
        print(json.dumps({'frequency': row['frequency'], **close}, ensure_ascii=False))
    print({'existing_anchor_gib_all_splits': result['existing_anchor_array_bytes_all_splits'] / 2 ** 30})


if __name__ == '__main__':
    main()
