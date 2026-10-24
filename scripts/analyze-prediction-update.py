"""分解两个冻结预测的IC变化；按月读取原标签，不训练或生成新候选。"""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import sys

import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'common'), str(ROOT / 'scripts')]
from artifacts import now, read_json, sha256, write_json
from compare import compare

KEYS = ['date', 'instrument']
BANDS = ('central_80pct', 'next_15pct', 'next_4pct', 'extreme_1pct')


def normalized_terms(prediction, target):
    p, y = np.asarray(prediction, np.float64), np.asarray(target, np.float64)
    p, y = p - p.mean(), y - y.mean()
    vp, vy = np.dot(p, p), np.dot(y, y)
    if min(vp, vy) <= 0:
        raise ValueError('贡献分解需要非恒定截面')
    return p * y / math.sqrt(vp * vy), p * p / vp, y * y / vy


def decompose_section(parent, child, target):
    """只由两组预测求线性投影；没有用标签拟合校准器。"""
    parent, child, target = [np.asarray(x, np.float64) for x in (parent, child, target)]
    if not (parent.shape == child.shape == target.shape) or parent.ndim != 1:
        raise ValueError('同一截面的向量形状必须一致')
    if not all(np.isfinite(x).all() for x in (parent, child, target)):
        raise ValueError('截面输入必须有限')
    p, q, y = [x - x.mean() for x in (parent, child, target)]
    pp, qq, yy = [np.dot(x, x) for x in (p, q, y)]
    if min(pp, qq, yy) <= 0:
        raise ValueError('贡献分解需要非恒定截面')
    update = q - p
    beta = np.dot(p, update) / pp
    orthogonal = update - beta * p
    coefficient = (1 + beta) * math.sqrt(pp / qq)
    parent_ic = np.dot(p, y) / math.sqrt(pp * yy)
    child_ic = np.dot(q, y) / math.sqrt(qq * yy)
    extra = np.dot(orthogonal, y) / math.sqrt(qq * yy)
    dilution = (coefficient - 1) * parent_ic
    correlation = np.dot(p, q) / math.sqrt(pp * qq)
    if not np.isclose(extra + dilution, child_ic - parent_ic, atol=1e-12, rtol=0):
        raise ValueError('正交分解未还原IC变化')
    independent_variance = np.dot(orthogonal, orthogonal) / qq
    if not np.isclose(independent_variance, 1 - correlation**2, atol=1e-12, rtol=0):
        raise ValueError('正交方差未还原预测相关性')
    return {'parent_ic': parent_ic, 'child_ic': child_ic, 'ic_delta': child_ic - parent_ic,
            'prediction_correlation': correlation, 'centered_update_to_parent_rms': math.sqrt(np.dot(update, update) / pp),
            'parent_direction_coefficient': 1 + beta,
            'orthogonal_child_variance_share': independent_variance,
            'orthogonal_ic_contribution': extra, 'parent_direction_ic_change': dilution,
            'intercept_update': float((child - parent).mean())}


def moving_block_interval(daily_delta, *, seed=20261024, block_days=5, draws=2000):
    values = np.asarray(daily_delta, np.float64)
    if values.ndim != 1 or len(values) < block_days or not np.isfinite(values).all():
        raise ValueError('区间估计需要按日期排序的有限日均差值')
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values) - block_days + 1, (draws, math.ceil(len(values) / block_days)))
    indices = (starts[..., None] + np.arange(block_days)).reshape(draws, -1)[:, :len(values)]
    sampled = values[indices].mean(1)
    return {'mean': float(values.mean()), 'percentile_95': np.quantile(sampled, [.025, .975]).tolist(),
            'block_days': block_days, 'draws': draws, 'seed': seed,
            'interpretation': '探索性日期块重采样；未校正历史选模和多次研究观察，不构成独立显著性证明。'}


def summarize_trees(run):
    config = read_json(run / 'config.json')
    if config['model']['family'] != 'neural_boosted_trees':
        return None
    saved = read_json(run / 'manifest.json')
    if sha256(run / 'best.pt') != saved['selection']['model_sha256']:
        raise ValueError('冻结权重指纹不一致')
    checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    document = checkpoint['booster_json']
    import json
    document = json.loads(document) if isinstance(document, (str, bytes, bytearray)) else document
    from price_geometry import VALUE_COLUMNS, PRICE_FIELDS
    if sha256(ROOT / 'common/price_geometry.py') != saved['source']['files']['common/price_geometry.py']:
        raise ValueError('字段命名须与冻结源码一致')
    assert config['model']['feature_dim'] == 624 and config['backbone_config']['model']['fusion_hidden_dim'] == 384
    names = [f'fusion_{i}' for i in range(384)]
    groups = ['fusion'] * 384
    columns = VALUE_COLUMNS + ['non_book_complete', 'book_complete', 'bar_available', 'log_price_reference', 'anchor_available']
    for frequency in ('bar1m', 'bar5m', 'bar30m'):
        for name in columns:
            names.append(f'{frequency}.{name}')
            if name.startswith('z_'):
                group = 'price_z' if name[2:] in PRICE_FIELDS else 'quantity_z'
            elif name.startswith('g_'):
                group = 'price_geometry'
            elif name.startswith('c_'):
                group = 'quantity_cross_section'
            elif name == 'log_price_reference':
                group = 'log_price_level'
            else:
                group = 'availability'
            groups.append(group)
    names += ['clock_' + value for value in config['data']['prediction_clock']['times']] + ['parent_prediction']
    groups += ['clock'] * 8 + ['parent_prediction']
    gain, counts, roots = np.zeros(624), np.zeros(624, np.int64), np.zeros(624, np.int64)
    trees = document['learner']['gradient_booster']['model']['trees']
    assert len(trees) == checkpoint['selected_tree_count']
    for tree in trees:
        for n, left in enumerate(tree['left_children']):
            if left < 0:
                continue
            feature = tree['split_indices'][n]
            value = tree['loss_changes'][n]
            if not 0 <= feature < len(names) or not math.isfinite(value) or value < 0:
                raise ValueError('树训练增益或特征索引不合法')
            gain[feature] += value
            counts[feature] += 1
            roots[feature] += n == 0
    result = defaultdict(lambda: {'split_count': 0, 'root_count': 0, 'gain': 0.})
    for i, group in enumerate(groups):
        row = result[group]
        row['split_count'] += int(counts[i]); row['root_count'] += int(roots[i]); row['gain'] += float(gain[i])
    for row in result.values():
        row['gain_share'] = row['gain'] / float(gain.sum())
    top = np.argsort(-gain)[:15]
    return {'selected_trees': len(trees), 'unique_split_features': int((counts > 0).sum()),
            'groups': dict(result), 'top_features': [dict(name=names[i], group=groups[i],
                split_count=int(counts[i]), root_count=int(roots[i]), gain_share=float(gain[i]/gain.sum())) for i in top],
            'limitation': '训练分裂增益受相关输入和先前树影响，不等于样本外贡献或因果重要性；没有删除特征重训。'}


def analyze(parent, child, output):
    parent, child = Path(parent).resolve(), Path(child).resolve()
    protocol = read_json(parent / 'protocol.json')
    data = read_json(parent / 'data-manifest.json')
    hashes = {}
    for table in (protocol['universe_table'], protocol['label_table']):
        path = Path(data['root']) / (table + '.parquet')
        expected = next(row for row in data['files'] if row['path'] == path.name)
        hashes[path.name] = sha256(path)
        if hashes[path.name] != expected['sha256']:
            raise ValueError('标签或股票池与冻结数据不同')
    result = {'computed_at': now(), 'script_sha256': sha256(__file__),
        'parent_run_id': parent.name, 'child_run_id': child.name, 'source_hashes': hashes,
        'method': '逐截面将中心化预测差正交分解为父方向和新增方向，并固定按真实标签幅度四档分解完整截面IC。',
        'band_cutoffs': [.8, .95, .99], 'bands': list(BANDS), 'tie_method': 'average',
        'model_training_performed': False, 'new_prediction_candidate': False,
        'formal_metrics_modified': False, 'exposure_read': False,
        'label_source': '直接使用已提供的vwap_return；不重算、替换或反推公式。',
        'limitations': ['标签幅度分档只用于事后解释，不能用于部署筛选或模型切换。',
            '预测差包括两次冻结推理的数值差异，不强称逐行等于原生树输出。',
            '正交方向表示与父分数线性不重合，不能据此证明加入了新的原始信息。',
            '完整截面贡献包含预测分母变化，不是子组独立IC。',
            '验证/测试均已在持续研究中观察；重采样未校正选模，不提供独立显著性结论。'],
        'results': {}, 'tree_training_structure': summarize_trees(child)}
    for split in ('validation', 'test'):
        compare([parent, child], split)
        interval = protocol['splits'][split]
        start, stop = pd.Timestamp(interval['start']), pd.Timestamp(interval['stop'])
        frozen = {name: read_json(run/'evaluation'/split/'metrics.json') for name, run in (('parent',parent),('child',child))}
        frozen_sections = {name: pd.read_parquet(run/'evaluation'/split/'cross-sections.parquet').set_index('date')
                           for name, run in (('parent',parent),('child',child))}
        fingerprints = {name: sha256(run/'evaluation'/split/'predictions.parquet') for name, run in (('parent',parent),('child',child))}
        section_rows, band_rows, finite_rows, total_rows = [], [], 0, 0
        for month in pd.period_range(start, stop-pd.Timedelta(1,'ns'), freq='M'):
            filters = [('date','>=', max(start,month.start_time)),('date','<',min(stop,(month+1).start_time))]
            universe = pd.read_parquet(Path(data['root'])/(protocol['universe_table']+'.parquet'),columns=KEYS,filters=filters)
            labels = pd.read_parquet(Path(data['root'])/(protocol['label_table']+'.parquet'),columns=KEYS+[protocol['label_column']],filters=filters)
            frame = universe.merge(labels,on=KEYS,how='left',validate='one_to_one')
            for name,run in (('parent',parent),('child',child)):
                prediction = pd.read_parquet(run/'evaluation'/split/'predictions.parquet',filters=filters).rename(columns={'factor':name})
                frame = frame.merge(prediction,on=KEYS,how='outer',validate='one_to_one',indicator=True)
                if not frame['_merge'].eq('both').all() or not np.isfinite(frame[name]).all():
                    raise ValueError('预测覆盖或有限性不同')
                frame = frame.drop(columns='_merge')
            total_rows += len(frame)
            horizon = frame.date + pd.Timedelta(minutes=protocol['label_horizon_minutes'])
            allowed = (horizon < stop) & (horizon.dt.normalize() == frame.date.dt.normalize())
            frame = frame.loc[allowed & np.isfinite(frame[protocol['label_column']])]
            finite_rows += len(frame)
            for date, section in frame.groupby('date',sort=True):
                if len(section) < protocol['min_cross_section_size']:
                    raise ValueError('本次分解要求全部截面满足正式最小样本数')
                target = section[protocol['label_column']].to_numpy(np.float64)
                p,q = section.parent.to_numpy(np.float64),section.child.to_numpy(np.float64)
                row = decompose_section(p,q,target);row['date'] = date
                percentile = pd.Series(np.abs(target-target.mean())).rank(method='average',pct=True).to_numpy()
                band = np.searchsorted([.8,.95,.99],percentile,side='left')
                ry = pd.Series(target).rank(method='average').to_numpy()
                for name,prediction in (('parent',p),('child',q)):
                    ic,pvar,yvar = normalized_terms(prediction,target)
                    rank,_,_ = normalized_terms(pd.Series(prediction).rank(method='average').to_numpy(),ry)
                    row[name+'_rank_ic'] = float(rank.sum())
                    expected = frozen_sections[name].loc[date]
                    if not np.allclose([ic.sum(),rank.sum()],[expected.ic,expected.rank_ic],atol=1e-12,rtol=0):
                        raise ValueError('贡献未还原冻结逐截面指标')
                    for number,label in enumerate(BANDS):
                        selected = band == number
                        band_rows.append({'date':date,'model':name,'band':label,'ic':float(ic[selected].sum()),
                            'rank_ic':float(rank[selected].sum()),'prediction_variance':float(pvar[selected].sum()),
                            'label_variance':float(yvar[selected].sum()),'sample_fraction':float(selected.mean())})
                row['rank_ic_delta'] = row['child_rank_ic'] - row['parent_rank_ic']
                section_rows.append(row)
        sections,bands = pd.DataFrame(section_rows),pd.DataFrame(band_rows)
        mean = sections.drop(columns='date').mean().to_dict()
        for name in ('parent','child'):
            if total_rows != frozen[name]['predicted_rows'] or finite_rows != frozen[name]['evaluated_rows']:
                raise ValueError('正式样本数未还原')
            for metric in ('ic','rank_ic'):
                if not np.isclose(mean[name+'_'+metric],frozen[name][metric+'_mean'],atol=1e-12,rtol=0):
                    raise ValueError('总体指标未还原')
        grouped = bands.groupby(['model','band']).mean(numeric_only=True)
        summary = {label:{'parent':grouped.loc[('parent',label)].to_dict(),
            'child':grouped.loc[('child',label)].to_dict(),
            'delta':(grouped.loc[('child',label)]-grouped.loc[('parent',label)]).to_dict()} for label in BANDS}
        daily = sections.groupby(sections.date.dt.normalize()).mean(numeric_only=True)
        monthly = sections.groupby(sections.date.dt.strftime('%Y-%m')).mean(numeric_only=True)
        result['results'][split] = {'prediction_sha256':fingerprints,'predicted_rows':total_rows,
            'evaluated_rows':finite_rows,'cross_sections':len(sections),'frozen_sections_reconstructed':True,
            'mean':mean,'bands':summary,'monthly':monthly.to_dict('index'),
            'exploratory_intervals':{key:moving_block_interval(daily[key].to_numpy()) for key in ('ic_delta','rank_ic_delta')}}
        directory = ROOT/'outputs/analysis'/output.stem
        directory.mkdir(parents=True,exist_ok=True)
        sections.to_parquet(directory/(split+'-sections.parquet'),index=False)
        bands.to_parquet(directory/(split+'-bands.parquet'),index=False)
        print({'split':split,**{k:mean[k] for k in ('ic_delta','rank_ic_delta','prediction_correlation','orthogonal_ic_contribution','parent_direction_ic_change')}},flush=True)
    write_json(output,result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run',type=Path,required=True)
    parser.add_argument('--child-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    analyze(args.parent_run,args.child_run,args.output)
