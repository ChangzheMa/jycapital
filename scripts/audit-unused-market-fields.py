"""固定训练样本检查未使用行情字段；不读标签、不训练、不重算收益。"""
import argparse
import hashlib
from pathlib import Path
import sys

import torch
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
for directory in ('common', 'solutions/v2401'):
    sys.path.insert(0, str(ROOT / directory))
from artifacts import now, read_json, sha256, write_json
from price_geometry import FIELDS
from v1601_data import minute_slots

KEYS = ['date', 'instrument']
EXTRA = ['deal_number', 'pre_close', 'adjust_factor']
COLUMNS = KEYS + EXTRA + ['close', 'volume', 'instrument_id']
FREQUENCIES = ('bar1m', 'bar5m', 'bar30m')


def digest(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


def regular(frame, frequency):
    frame = frame.copy()
    frame['date'] = pd.to_datetime(frame.date)
    slots = {value: i for i, value in enumerate(minute_slots(frequency))}
    minutes = frame.date.dt.hour * 60 + frame.date.dt.minute
    frame['slot'] = minutes.map(slots)
    frame = frame.loc[frame.slot.notna() & frame.date.eq(frame.date.dt.floor('min'))].copy()
    assert not frame.duplicated(KEYS).any() and not frame[KEYS].isna().any().any()
    frame['day'] = frame.date.dt.normalize()
    frame['slot'] = frame.slot.astype('int64')
    return frame.sort_values(['instrument', 'date']).reset_index(drop=True)


def quality(frame):
    records = {}
    for field in EXTRA:
        values = frame[field].to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        positive = values[finite]
        finite_frame = frame[['instrument', 'day', field]].copy()
        finite_frame.loc[~finite, field] = np.nan
        nunique = finite_frame.groupby(['instrument', 'day'])[field].nunique()
        records[field] = {'rows': len(values), 'finite_rows': int(finite.sum()),
            'zero_rows': int((finite & (values == 0)).sum()), 'negative_rows': int((finite & (values < 0)).sum()),
            'noninteger_rows': int((finite & (values != np.floor(values))).sum()),
            'quantiles': {str(q): float(np.quantile(positive, q)) if len(positive) else None for q in (0, .01, .5, .99, 1)},
            'stock_days': len(nunique), 'stock_days_with_multiple_values': int((nunique > 1).sum()),
            'stock_days_without_finite_values': int((nunique == 0).sum())}
    valid = np.isfinite(frame.deal_number) & np.isfinite(frame.volume) & (frame.deal_number >= 0) & (frame.volume >= 0)
    records['deal_number_vs_volume'] = {
        'valid_rows': int(valid.sum()),
        'log1p_pearson_correlation': float(np.corrcoef(np.log1p(frame.loc[valid, 'deal_number']),
                                                     np.log1p(frame.loc[valid, 'volume']))[0, 1]),
        'count_zero_volume_positive_rows': int(((frame.deal_number == 0) & (frame.volume > 0)).sum()),
        'count_positive_volume_zero_rows': int(((frame.deal_number > 0) & (frame.volume == 0)).sum())}
    return records


def comparison(left, right, eligible=None):
    left, right = np.asarray(left, np.float64), np.asarray(right, np.float64)
    valid = np.isfinite(left) & np.isfinite(right) & (left > 0) & (right > 0)
    if eligible is not None:
        valid &= np.asarray(eligible, bool)
    count = int(valid.sum())
    exact = int((left[valid] == right[valid]).sum())
    close = int(np.isclose(left[valid], right[valid], rtol=1e-6, atol=1e-8).sum())
    return {'valid_pairs': count, 'exact_equal_pairs': exact, 'tolerance_equal_pairs': close,
            'exact_equal_fraction': exact / count if count else None,
            'tolerance_equal_fraction': close / count if count else None}


def predecessor_diagnostics(frame, first_day):
    result = frame.copy()
    daily = result.groupby(['instrument', 'day'], sort=True).agg(
        last_close=('close', 'last'), last_adjust_factor=('adjust_factor', 'last'))
    previous = daily.groupby(level=0).shift(1).rename(columns={
        'last_close': 'previous_observed_day_close', 'last_adjust_factor': 'previous_observed_day_adjust_factor'})
    result = result.merge(previous.reset_index(), on=['instrument', 'day'], how='left', validate='many_to_one')
    previous_bar = result.groupby(['instrument', 'day']).close.shift(1)
    adjacent = result.groupby(['instrument', 'day']).slot.diff().eq(1)
    use = result.day > first_day
    first = result.loc[use].groupby(['instrument', 'day']).first()
    valid = np.isfinite(first[['pre_close', 'adjust_factor', 'previous_observed_day_close',
                              'previous_observed_day_adjust_factor']]).all(axis=1)
    first = first.loc[valid]
    price_differs = ~np.isclose(first.pre_close, first.previous_observed_day_close, rtol=1e-6, atol=1e-8)
    factor_changes = ~np.isclose(first.adjust_factor, first.previous_observed_day_adjust_factor, rtol=1e-6, atol=1e-8)
    return {
        'day_change_association': {'stock_days_with_valid_comparison':len(first),
            'pre_close_differs_from_previous_close':int(price_differs.sum()),
            'adjust_factor_changes':int(factor_changes.sum()), 'both':int((price_differs & factor_changes).sum()),
            'pre_close_differs_without_factor_change':int((price_differs & ~factor_changes).sum()),
            'factor_changes_without_pre_close_difference':int((factor_changes & ~price_differs).sum())},
        'pre_close_vs_previous_observed_day_close': comparison(result.pre_close, result.previous_observed_day_close, use),
        'pre_close_vs_previous_adjacent_same_day_bar_close': comparison(result.pre_close, previous_bar, use & adjacent),
        'adjust_factor_vs_previous_observed_day_factor': comparison(result.adjust_factor, result.previous_observed_day_adjust_factor, use)}


def additive_diagnostics(one, coarse, frequency, first_day):
    size = {'bar5m': 5, 'bar30m': 30}[frequency]
    one = one.copy()
    one['coarse_slot'] = one.slot // size
    grouped = one.groupby(['instrument', 'day', 'coarse_slot']).agg(
        observed_bars=('slot', 'size'), summed_deal_number=('deal_number', 'sum'), summed_volume=('volume', 'sum'))
    joined = coarse.merge(grouped.reset_index(), left_on=['instrument', 'day', 'slot'],
                          right_on=['instrument', 'day', 'coarse_slot'], how='left', validate='one_to_one')
    joined = joined.loc[joined.day > first_day]
    complete = joined.observed_bars.eq(size)
    result = {'coarse_rows': len(joined), 'complete_one_minute_intervals': int(complete.sum())}
    for field in ('deal_number', 'volume'):
        a, b = joined[field].to_numpy(), joined['summed_' + field].to_numpy()
        valid = complete.to_numpy() & np.isfinite(a) & np.isfinite(b)
        result[field] = {'valid_intervals': int(valid.sum()), 'exact_equal_intervals': int((a[valid] == b[valid]).sum()),
                         'max_absolute_difference': float(np.abs(a[valid] - b[valid]).max()) if valid.any() else None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/unused-market-fields-audit-20261023.json')
    args = parser.parse_args()
    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)
    torch.set_num_threads(1)
    reference = ROOT / 'outputs/runs/v1601-price-geometry-s42/data-manifest.json'
    manifest = read_json(reference)
    raw_root = Path(manifest['root'])
    sources = {r['path']: r for r in manifest['files']}
    prior = read_json(ROOT / 'experiments/price-geometry-audit-20261022.json')
    samples, pooled, schema_records = [], {f: [] for f in FREQUENCIES}, {}
    for year in (2020, 2021, 2022, 2023):
        month = f'{year}-06'
        universe = pd.read_parquet(raw_root / 'cpt_jyc_2026_instruments.parquet', columns=KEYS,
            filters=[('date', '>=', pd.Timestamp(month + '-01')), ('date', '<', pd.Timestamp(f'{year}-07-01'))])
        days = pd.DatetimeIndex(sorted(universe.date.dt.normalize().unique())[:5])
        pool = sorted(universe.loc[universe.date.dt.normalize().eq(days[0]), 'instrument'].unique())
        stocks = sorted(np.random.default_rng(np.random.SeedSequence([20261022, year])).choice(pool, 64, replace=False).tolist())
        stock_sha = hashlib.sha256('\n'.join(stocks).encode()).hexdigest()
        previous = next(r for r in prior['samples'] if r['month'] == month and r['frequency'] == 'bar1m')
        assert stock_sha == previous['stock_selection_sha256']
        assert days[-1] < pd.Timestamp('2024-01-01')
        filters = [('date', '>=', days[0]), ('date', '<', days[-1] + pd.Timedelta(days=1)), ('instrument', 'in', stocks)]
        frames = {}
        for frequency in FREQUENCIES:
            path = raw_root / ('cpt_jyc_2026_stock_' + frequency) / (month + '.parquet')
            schema = pq.read_schema(path)
            unused = sorted(set(schema.names) - set(KEYS) - set(FIELDS))
            assert len(schema.names) == 42 and unused == sorted(EXTRA + ['instrument_id'])
            schema_records[frequency] = {'columns': len(schema.names), 'used_raw_fields': list(FIELDS),
                'unused_raw_fields': unused, 'unused_actual_types': {f:str(schema.field(f).type) for f in unused}}
            raw = pd.read_parquet(path, columns=COLUMNS, filters=filters)
            frame = regular(raw, frequency)
            frames[frequency] = frame
            selected = frame.loc[frame.day > days[0]].copy()
            assert selected.groupby('instrument').instrument_id.nunique().max() <= 1
            pooled[frequency].append(selected)
            expected = sources['raw/' + path.relative_to(raw_root).as_posix()]
            assert path.stat().st_size == expected['bytes']
            row = {'month':month, 'frequency':frequency, 'rows':len(selected),
                'stock_selection_sha256':stock_sha, 'warmup_date':str(days[0].date()),
                'dates':[str(day.date()) for day in days[1:]], 'quality':quality(selected),
                'predecessors':predecessor_diagnostics(frame, days[0]),
                'source':{'path':path.relative_to(ROOT).as_posix(), 'bytes':path.stat().st_size,
                    'expected_full_sha256_from_parent_manifest':expected['sha256'],
                    'full_file_sha256_recomputed':False, 'sample_content_sha256':digest(raw)}}
            if frequency != 'bar1m':
                row['coarse_aggregation'] = additive_diagnostics(frames['bar1m'], frame, frequency, days[0])
            samples.append(row)
            print({'month':month,'frequency':frequency,'rows':len(selected)}, flush=True)
    aggregate = {f:quality(pd.concat(pooled[f],ignore_index=True)) for f in FREQUENCIES}
    write_json(args.output, {'checked_at':now(), 'script_sha256':sha256(__file__),
        'data_manifest_sha256':sha256(reference), 'source_reference_run':reference.parent.name,
        'sample_protocol':'复用2020—2023各六月首五交易日、每年固定64股；首日只供前日对照，主诊断为16日、最多1024股票日。',
        'labels_read':False, 'exposure_read':False, 'formal_training_performed':False, 'new_candidate_created':False,
        'schema':schema_records, 'aggregate':aggregate, 'samples':samples,
        'equality_tolerance':{'rtol':1e-6,'atol':1e-8},
        'limitations':['只描述固定训练样本的输入属性，未测收益增益；不能以成交量相关性代替收益IC。',
            '前日对照使用各频率上一有记录日最后非空close，不代表官方pre_close定义；不据此自动复权或更改原价格。',
            '聚合仅检查完整规则交易槽位；不外推盘前、缺失Bar或盘口聚合规则。',
            '字段字典保持官方104条不变；instrument_id只作为元数据检查，不把ID数值顺序当连续信号。',
            'deal_number来自已闭合行情Bar；不读取标签表未来窗口的num_trades。']})


if __name__ == '__main__':
    main()
