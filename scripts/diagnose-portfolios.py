"""对冻结预测计算固定组合代理和成本敏感性，不搜索策略或修改选模。"""

import argparse
from pathlib import Path
import sys

import torch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
for directory in (ROOT/'common',ROOT/'evaluation',ROOT/'scripts'):
    sys.path.insert(0,str(directory))
from artifacts import read_json,write_json,sha256,now
from compare import compare
from portfolio_diagnostics import portfolio_diagnostics


def read_labels(run,split):
    protocol=read_json(run/'protocol.json')
    data=read_json(run/'data-manifest.json')
    source=Path(data['root'])
    tables={protocol['universe_table'],protocol['label_table']}
    for item in data['files']:
        if Path(item['path']).stem in tables and sha256(source/item['path'])!=item['sha256']:
            raise ValueError('股票池或标签文件与冻结运行不同')
    interval=protocol['splits'][split]
    filters=[('date','>=',pd.Timestamp(interval['start'])),('date','<',pd.Timestamp(interval['stop']))]
    keys=pd.read_parquet(source/(protocol['universe_table']+'.parquet'),columns=['date','instrument'],filters=filters)
    labels=pd.read_parquet(source/(protocol['label_table']+'.parquet'),columns=['date','instrument',protocol['label_column']],filters=filters)
    keys=keys.merge(labels,on=['date','instrument'],how='left',validate='one_to_one')
    horizon=keys.date+pd.Timedelta(minutes=protocol['label_horizon_minutes'])
    admissible=(horizon<pd.Timestamp(interval['stop'])) & horizon.dt.normalize().eq(keys.date.dt.normalize())
    keys.loc[~admissible,protocol['label_column']]=np.nan
    metrics=read_json(run/'evaluation'/split/'metrics.json')
    if len(keys)!=metrics['expected_rows'] or int(np.isfinite(keys[protocol['label_column']]).sum())!=metrics['valid_label_rows']:
        raise ValueError('标签边界或股票池口径与冻结评估不符')
    return keys,protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--platform-observation',type=Path)
    args=parser.parse_args()
    runs=[run.resolve() for run in args.runs]
    platform=read_json(args.platform_observation) if args.platform_observation else None
    results=[]
    for split in ('validation','test'):
        compare(runs,split,allow_selection_metric_change=True)
        labels,protocol=read_labels(runs[0],split)
        for run in runs:
            manifest=read_json(run/'manifest.json')
            if sha256(run/'best.pt')!=manifest['selection']['model_sha256']:
                raise ValueError('冻结权重已改变')
            path=run/'evaluation'/split/'predictions.parquet'
            predictions=pd.read_parquet(path)
            summary,monthly,sections,days=portfolio_diagnostics(predictions,labels,protocol['label_column'])
            directory=ROOT/'outputs/analysis/portfolio-diagnostics-20261022'/run.name/split
            directory.mkdir(parents=True,exist_ok=True)
            sections.to_parquet(directory/'sections.parquet',index=False)
            days.to_csv(directory/'days.csv',index=False,encoding='utf-8-sig')
            item={'solution_id':manifest['solution_id'],'run_id':run.name,'split':split,
                  'model_sha256':manifest['selection']['model_sha256'],'predictions_sha256':sha256(path),
                  'protocol_sha256':sha256(run/'protocol.json'),'data_manifest_sha256':sha256(run/'data-manifest.json'),
                  'summary':summary,'monthly':monthly,'local_metrics':read_json(run/'evaluation'/split/'metrics.json')}
            if platform:
                matching=[row for row in platform['submissions'] if row.get('binding',{}).get('candidate_solution_id')==manifest['solution_id']]
                if len(matching)==1:
                    row=matching[0]
                    item['platform_file_label_observation']={'submission_id':row['id'],'public_score':row['public_score'],
                        'public_score_data':row['public_score_data'],'binding_status':'unconfirmed',
                        'scope':'只按附件名作描述性对照；平台区间和上传内容与本地 run 的精确映射未确认。'}
            results.append(item)
            print({'solution_id':manifest['solution_id'],'split':split,**{key:summary[key] for key in
                ('daily_label_return_mean','daily_label_return_ir','zero_mean_cost_threshold_bps','one_way_intraday_turnover_mean','known_held_label_mass_minimum')}},flush=True)
    associations=[]
    for split in ('validation','test'):
        rows=[]
        for item in results:
            observed=item.get('platform_file_label_observation')
            if item['split']!=split or not observed:
                continue
            rows.append({'solution_id':item['solution_id'],
                'local_ic':item['local_metrics']['ic_mean'],'platform_ic':observed['public_score_data']['ic_mean'],
                'local_return_ir':item['summary']['daily_label_return_ir'],'platform_sharpe':observed['public_score_data']['sharpe_ratio'],
                'local_turnover':item['summary']['one_way_intraday_turnover_mean'],'platform_turnover':observed['public_score_data']['turnover']})
        frame=pd.DataFrame(rows)
        if len(frame)>=3:
            associations.append({'split':split,'file_label_pairs':len(frame),'binding_status':'unconfirmed',
                'rank_correlations':{name:float(frame[left].rank().corr(frame[right].rank())) for name,left,right in
                    [('ic','local_ic','platform_ic'),('return_ir_vs_sharpe','local_return_ir','platform_sharpe'),('turnover','local_turnover','platform_turnover')]},
                'pairs':rows})
    result={'schema_version':1,'computed_at':now(),'formal_training_performed':False,'new_model_created':False,
        'method':{'tail_fraction':.1,'long_gross':.5,'short_gross':.5,'gross_exposure':1.,
            'score_to_weights':'existing tail_weights with fractional allocation of tied boundaries',
            'return_proxy':'sum of weight times finite official forward label, no renormalization after missing labels',
            'unknown_labels':'unpriced contribution omitted and held-label mass explicitly reported; not assumed known zero return',
            'one_way_turnover':'.5 * sum(abs(target_weight_t - target_weight_previous)) within day',
            'daily_cost_notional':'sum(abs(initial weights)) + sum intraday abs changes + sum(abs(final weights))',
            'cost_scenarios_bps_per_unit_traded_notional':[0.,1.,5.,10.],
            'risk_ratio':'mean / sample standard deviation of daily sums; no annualization',
            'price_drift_in_weights':False,'overnight_holdings':'none; daily hypothetical reset to cash',
            'selection_uses_labels':False,'strategy_or_weight_search':False},
        'results':results,'descriptive_platform_associations':associations,
        'sources':{'script_sha256':sha256(__file__),'portfolio_diagnostics_sha256':sha256(ROOT/'evaluation/portfolio_diagnostics.py'),
                   'tail_weights_source_sha256':sha256(ROOT/'evaluation/signal_diagnostics.py'),
                   'platform_observation_sha256':sha256(args.platform_observation) if args.platform_observation else None},
        'limitations':['这些是官方未来标签加权的研究代理，不是可实现资金收益、净值或平台夏普。',
            '未模拟实际持仓漂移、VWAP执行、融资、冲击、BARRA或官方压力场景。',
            '固定成本场景仅用于敏感性分析，不能当作平台费用或据测试结果选择成本参数。',
            '平台附件名对应关系尚未绑定权重，排名相关仅是少量模型的描述性线索。',
            '测试期已被持续观察，非新盲测。']}
    write_json(args.output,result)
    print({'saved':str(args.output),'models':len(runs),'results':len(results),'descriptive_platform_associations':associations})


if __name__=='__main__':
    main()
