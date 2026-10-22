"""固定分位数组合的标签收益、风险与调仓代理；不模拟平台回测。"""

import numpy as np
import pandas as pd

from signal_diagnostics import tail_weights

KEYS = ['date', 'instrument']


def finite_number(value):
    return float(value) if np.isfinite(value) else None


def mean_std_ratio(values):
    values = pd.Series(values).dropna()
    if len(values) < 2:
        return None
    deviation = values.std(ddof=1)
    return finite_number(values.mean() / deviation) if deviation > 0 else None


def summarize(days, sections, costs_bps):
    valid = days.loc[days.defined]
    result = {'days': len(days), 'defined_days': len(valid), 'sections': len(sections),
        'undefined_sections': int((~sections.defined).sum()),
        'missing_held_label_sections': int((sections.missing_label_mass > 0).sum()),
        'known_held_label_mass_mean': finite_number(sections.known_label_mass.mean()),
        'known_held_label_mass_minimum': finite_number(sections.known_label_mass.min()),
        'gross_label_return_mean_per_section': finite_number(sections.gross_label_return.mean()),
        'section_label_return_ir': mean_std_ratio(sections.gross_label_return),
        'daily_label_return_mean': finite_number(valid.gross_label_return.sum() / len(valid)) if len(valid) else None,
        'daily_label_return_std': finite_number(valid.gross_label_return.std(ddof=1)),
        'daily_label_return_ir': mean_std_ratio(valid.gross_label_return),
        'daily_positive_fraction': float((valid.gross_label_return > 0).mean()) if len(valid) else None,
        'daily_label_return_q05': finite_number(valid.gross_label_return.quantile(.05)) if len(valid) else None,
        'one_way_intraday_turnover_mean': finite_number(sections.one_way_intraday_turnover.mean()),
        'daily_traded_notional_mean_including_open_close': finite_number(valid.traded_notional.mean()),
        'cost_scenarios': []}
    denominator = valid.traded_notional.sum()
    result['zero_mean_cost_threshold_bps'] = finite_number(10000 * valid.gross_label_return.sum() / denominator) if denominator > 0 else None
    for cost in costs_bps:
        net = valid.gross_label_return - cost / 10000 * valid.traded_notional
        result['cost_scenarios'].append({'cost_bps_per_unit_traded_notional': cost,
            'daily_net_label_return_mean': finite_number(net.mean()), 'daily_net_label_return_ir': mean_std_ratio(net),
            'daily_net_positive_fraction': float((net > 0).mean()) if len(net) else None})
    return result


def portfolio_diagnostics(predictions, universe_labels, label_column='vwap_return', fraction=.1, costs_bps=(0.,1.,5.,10.)):
    """先从全截面预测构造权重，再处理未知标签；不按未来可用标签筛股票。"""
    if list(predictions.columns) != KEYS + ['factor']:
        raise ValueError('预测格式必须为 date、instrument、factor')
    if predictions[KEYS].isna().any().any() or predictions.duplicated(KEYS).any() or universe_labels.duplicated(KEYS).any():
        raise ValueError('键为空或重复')
    if not np.isfinite(predictions.factor).all():
        raise ValueError('预测分数必须有限')
    costs_bps = tuple(float(cost) for cost in costs_bps)
    if not costs_bps or not all(np.isfinite(cost) and cost >= 0 for cost in costs_bps):
        raise ValueError('成本场景须为有限非负值')
    frame = predictions.merge(universe_labels[KEYS + [label_column]], on=KEYS, how='outer', indicator=True, validate='one_to_one')
    if not frame['_merge'].eq('both').all():
        raise ValueError('组合诊断要求完整相同股票池，不改变覆盖')
    frame['date'] = pd.to_datetime(frame.date)
    section_rows, day_rows = [], []
    for day, daily in frame.sort_values(KEYS).groupby(frame.date.dt.normalize(), sort=True):
        previous, totals = None, {'gross_label_return':0., 'rebalance_notional':0., 'opening_notional':0.,
                                 'closing_notional':0., 'defined':True, 'sections':0}
        for stamp, section in daily.groupby('date', sort=True):
            weights = pd.Series(tail_weights(section.factor.to_numpy(), fraction), index=section.instrument)
            gross = float(weights.abs().sum())
            defined = bool(np.isclose(gross,1))
            labels = section[label_column].to_numpy(dtype=np.float64)
            known = np.isfinite(labels)
            # 未知收益的持仓贡献不估计、不重新归一化；另报未知持仓质量。
            value = float(np.dot(weights.to_numpy()[known], labels[known])) if defined else np.nan
            known_mass = float(weights.abs().to_numpy()[known].sum()/gross) if defined else np.nan
            missing_mass = float(weights.abs().to_numpy()[~known].sum()/gross) if defined else np.nan
            if previous is None:
                opening, traded, turnover = gross, 0., np.nan
            else:
                before, after = previous.align(weights, join='outer', fill_value=0)
                opening = 0.
                traded = float((after-before).abs().sum())
                turnover = traded/2 if defined and np.isclose(previous.abs().sum(),1) else np.nan
            section_rows.append({'date':stamp, 'stocks':len(section), 'defined':defined, 'gross_exposure':gross,
                'net_exposure':float(weights.sum()), 'gross_label_return':value, 'known_label_mass':known_mass,
                'missing_label_mass':missing_mass,
                'opening_notional':opening, 'rebalance_notional':traded, 'one_way_intraday_turnover':turnover})
            totals['defined'] &= defined
            totals['gross_label_return'] += value
            totals['opening_notional'] += opening
            totals['rebalance_notional'] += traded
            totals['closing_notional'] = gross
            totals['sections'] += 1
            previous = weights
        totals['traded_notional'] = sum(totals[key] for key in ('opening_notional','rebalance_notional','closing_notional'))
        day_rows.append({'date':day, **totals})
    sections, days = pd.DataFrame(section_rows), pd.DataFrame(day_rows)
    if sections.empty:
        raise ValueError('没有可诊断截面')
    summary = summarize(days, sections, costs_bps)
    monthly = []
    for month, part in days.groupby(days.date.dt.strftime('%Y-%m')):
        local_sections = sections.loc[sections.date.dt.strftime('%Y-%m').eq(month)]
        monthly.append({'month':month, **summarize(part,local_sections,costs_bps)})
    return summary, monthly, sections, days
