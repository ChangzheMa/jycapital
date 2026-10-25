"""仅用冻结逐截面指标检查验证选时点的互补性，不生成正式组合或重算标签。"""

import argparse
from pathlib import Path
import sys

import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'common'), str(ROOT / 'scripts')]
from artifacts import read_json, write_json, sha256, now
from compare import compare


def check_frames(frames):
    reference = next(iter(frames.values())).index
    if reference.has_duplicates or reference.empty:
        raise ValueError('逐截面日期必须非空且唯一')
    for name, frame in frames.items():
        if not frame.index.equals(reference) or not np.isfinite(frame[['ic', 'rank_ic']]).all().all():
            raise ValueError('逐截面日期不一致或指标无定义: ' + name)


def choose_by_validation(frames, baseline):
    check_frames(frames)
    order = [baseline] + sorted(set(frames) - {baseline})
    means = {name: frame.groupby(frame.index.strftime('%H:%M')).ic.mean() for name, frame in frames.items()}
    return {clock: max(order, key=lambda name: means[name].loc[clock]) for clock in means[baseline].index}


def apply_choice(frames, choices):
    check_frames(frames)
    index = next(iter(frames.values())).index
    clocks = index.strftime('%H:%M')
    if set(clocks) != set(choices):
        raise ValueError('时点规则未完整覆盖请求')
    result = pd.DataFrame(index=index, columns=['ic', 'rank_ic'], dtype=float)
    for clock, name in choices.items():
        selected = index[clocks == clock]
        result.loc[selected] = frames[name].loc[selected, ['ic', 'rank_ic']]
    return result


def summary(frame):
    return {'ic_mean': float(frame.ic.mean()), 'rank_ic_mean': float(frame.rank_ic.mean()),
            'ic_ir': float(frame.ic.mean() / frame.ic.std(ddof=1)),
            'rank_ic_ir': float(frame.rank_ic.mean() / frame.rank_ic.std(ddof=1))}


def held_months(frames, baseline):
    check_frames(frames)
    index = frames[baseline].index
    months = index.strftime('%Y-%m')
    records, pieces = [], []
    for month in sorted(set(months)):
        train = {n: f.loc[months != month] for n, f in frames.items()}
        held = {n: f.loc[months == month] for n, f in frames.items()}
        choices = choose_by_validation(train, baseline)
        routed = apply_choice(held, choices)
        records.append({'held_month': month, 'choices_from_other_validation_months': choices,
                        'held_ic_delta': float(routed.ic.mean() - held[baseline].ic.mean())})
        pieces.append(routed)
    return {'months': records, 'summary': summary(pd.concat(pieces).sort_index()),
            'improved_months': sum(r['held_ic_delta'] > 0 for r in records)}


def conditional_test_interval(delta, repetitions=2000, block=5):
    daily = delta.groupby(delta.index.normalize()).mean().to_numpy()
    rng = np.random.default_rng(42)
    estimates = []
    for _ in range(repetitions):
        starts = rng.integers(0, len(daily), size=(len(daily) + block - 1) // block)
        indices = (starts[:, None] + np.arange(block)).reshape(-1)[:len(daily)] % len(daily)
        estimates.append(float(daily[indices].mean()))
    return {'block_trading_days': block, 'repetitions': repetitions, 'seed': 42,
            'mean_delta': float(delta.mean()), 'percentile_95_interval': np.quantile(estimates, [.025, .975]).tolist(),
            'scope': '固定已选权重和验证时点规则的条件区间，未校正历史模型搜索及选择偏差。'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = {name: ROOT / 'outputs/runs' / name for name in args.runs}
    if args.baseline not in paths or len(paths) < 2:
        raise ValueError('需要基线及至少另一个完整运行')
    frames, hashes, metrics = {}, {}, {}
    for split in ('validation', 'test'):
        compare(list(paths.values()), split, allow_selection_metric_change=True)
        frames[split], hashes[split], metrics[split] = {}, {}, {}
        for name, run in paths.items():
            p = run / 'evaluation' / split / 'cross-sections.parquet'
            f = pd.read_parquet(p).set_index('date').sort_index()
            m = read_json(p.parent / 'metrics.json')
            for key in ('ic', 'rank_ic'):
                if not np.isclose(f[key].mean(), m[key + '_mean'], rtol=0, atol=1e-12):
                    raise ValueError('冻结指标与逐截面均值不一致')
            frames[split][name], hashes[split][name], metrics[split][name] = f, sha256(p), m
    choices = choose_by_validation(frames['validation'], args.baseline)
    report = {'computed_at': now(), 'script_sha256': sha256(__file__), 'baseline': args.baseline,
              'source_hashes': hashes, 'choices_from_validation_only': choices, 'results': {},
              'formal_model_created': False, 'model_training_performed': False,
              'raw_data_or_label_table_read': False, 'predictions_modified': False,
              'limitations': ['权重已按整个验证期选择，留月分析不是完全独立的外层验证。',
                             '完整验证选择结果有选择偏差，测试只套用固定规则；研究已观察测试期。',
                             '现有模型按时点切换的互补性不证明独立训练八个网络一定有效。',
                             '这只是逐截面分解，不是正式组合结果或平台提交；未评价切换后的换手。']}
    for split, split_frames in frames.items():
        routed = apply_choice(split_frames, choices)
        base = summary(split_frames[args.baseline])
        result = {'baseline': base, 'validation_selected_route': summary(routed),
                  'route_minus_baseline': {k: v - base[k] for k, v in summary(routed).items()},
                  'clock_metrics': {name: f.groupby(f.index.strftime('%H:%M'))[['ic', 'rank_ic']].mean().to_dict(orient='index') for name, f in split_frames.items()}}
        if split == 'test':
            result['conditional_ic_interval'] = conditional_test_interval(routed.ic - split_frames[args.baseline].ic)
        report['results'][split] = result
    report['leave_one_validation_month_out'] = held_months(frames['validation'], args.baseline)
    write_json(args.output, report)
    print({'choices': choices, 'deltas': {s: r['route_minus_baseline'] for s, r in report['results'].items()},
           'held_month_summary': report['leave_one_validation_month_out']['summary'],
           'test_conditional_interval': report['results']['test']['conditional_ic_interval']})


if __name__ == '__main__':
    main()
