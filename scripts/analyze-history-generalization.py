"""按训练监督天数分解冻结IC，并计算仅改善短历史股票的事后理想上界。"""
import argparse
from pathlib import Path
import sys

import torch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
for directory in ('common','scripts'):
    sys.path.insert(0,str(ROOT/directory))
from artifacts import now,read_json,write_json,sha256
from compare import compare

VERSIONS=('v1102','v1601','v1901')
BANDS=('no_supervised_day','1_59_days','60_251_days','252_plus_days')
CACHE_KEY='c760f3882029e185c69b39c92509beeeea3995e2214360a6196046d70158563b'
KEYS=['date','instrument']


def history_band(counts):
    values=np.asarray(counts)
    if np.any(values<0) or np.any(values!=np.floor(values)):
        raise ValueError('监督天数必须为非负整数')
    return np.searchsorted(np.array([0,59,251]),values,side='left')


def contributions(scores,labels,dates):
    dx=scores.astype(np.float64)-scores.groupby(dates).transform('mean')
    dy=labels.astype(np.float64)-labels.groupby(dates).transform('mean')
    xx=dx*dx
    yy=dy*dy
    vx=xx.groupby(dates).transform('sum')
    vy=yy.groupby(dates).transform('sum')
    denominator=np.sqrt(vx*vy).where((vx>0)&(vy>0))
    return dx*dy/denominator,xx/vx.where(vx>0),yy/vy.where(vy>0)


def restricted_oracle_ceiling(scores,labels,replaceable):
    """其余股票保留原分数的正向仿射关系，允许指定股票使用未来真值；只作上界。"""
    x,y=np.asarray(scores,np.float64),np.asarray(labels,np.float64)
    replaceable=np.asarray(replaceable,bool)
    if x.ndim!=1 or x.shape!=y.shape or x.shape!=replaceable.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('上界要求同形有限一维向量')
    total=float(((y-y.mean())**2).sum())
    if total<=0:
        return None
    known=~replaceable
    if known.sum()<2:
        return 1.0
    a=x[known]-x[known].mean();b=y[known]-y[known].mean()
    variance=float(a@a);covariance=float(a@b)
    explained=max(covariance,0.)**2/variance if variance>0 else 0.
    residual=max(0.,float(b@b)-explained)
    return float(np.sqrt(np.clip(1-residual/total,0,1)))


def training_history():
    directory=ROOT/'outputs/preprocessing/resident-cache'/CACHE_KEY
    manifest=read_json(directory/'manifest.json')
    if manifest['status']!='complete' or manifest['cache_key']!=CACHE_KEY:
        raise ValueError('训练缓存身份不符')
    names=('targets','sample_day','sample_stock','days')
    arrays={}
    fingerprints={}
    for name in names:
        path=directory/(name+'.npy');expected=manifest['files'][path.name]
        digest=sha256(path)
        if digest!=expected['sha256']:
            raise ValueError('训练索引或已有目标指纹改变')
        fingerprints[path.name]=digest
        arrays[name]=np.load(path,mmap_mode='r',allow_pickle=False)
    dates=arrays['days'].astype('datetime64[ns]')
    if np.any(dates<np.datetime64('2020-01-01')) or np.any(dates>=np.datetime64('2024-01-01')):
        raise ValueError('训练日期边界不符')
    seen=np.zeros((len(dates),len(manifest['stocks'])),bool)
    valid=np.isfinite(arrays['targets'])
    seen[arrays['sample_day'][valid],arrays['sample_stock'][valid]]=True
    counts=seen.sum(axis=0)
    mapping=dict(zip(manifest['stocks'],map(int,counts)))
    return mapping,{'cache_key':CACHE_KEY,'manifest_sha256':sha256(directory/'manifest.json'),
                    'source_arrays':fingerprints,'stocks':len(mapping),'calendar_days':len(dates),
                    'supervised_stock_days':int(counts.sum()),'uses_provided_label_validity_only':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/history-generalization-20261023.json')
    args=parser.parse_args()
    summaries={v:read_json(ROOT/f'experiments/{v}-gru.json') for v in VERSIONS}
    runs=[ROOT/summaries[v]['run_path'] for v in VERSIONS]
    counts,history=training_history()
    reference=ROOT/summaries['v1601']['run_path']
    protocol=read_json(reference/'protocol.json');sources=read_json(reference/'data-manifest.json')
    fingerprints={}
    for table in (protocol['universe_table'],protocol['label_table']):
        path=Path(sources['root'])/(table+'.parquet')
        fingerprints[path.name]=sha256(path)
        expected=next(row['sha256'] for row in sources['files'] if row['path']==path.name)
        if fingerprints[path.name]!=expected: raise ValueError('原股票池或提供标签指纹不符')
    report={'computed_at':now(),'script_sha256':sha256(__file__),'versions_fixed_before_analysis':list(VERSIONS),
      'bands':list(BANDS),'training_history':history,'source_hashes':fingerprints,
      'method':'监督天数仅按2020—2023已有合法收益标签的有限性计算；固定分组后，在完整截面中心化和分母不变的条件下分解IC及Rank IC。',
      'oracle_method':'指定短历史股票可获得未来标签，其余股票保持原预测分数的正向仿射关系。用总标签方差减去其余股票正向线性回归残差，再开方；这是事后理想上界，不是可部署结果。',
      'formal_metrics_modified':False,'model_training_performed':False,'official_return_recomputed':False,'exposure_read':False,
      'results':[],'deltas':[],
      'limitations':['分组依据训练期信息，但贡献和上界使用已观察验证/测试标签，只用于结果诊断。',
                     '贡献是全截面IC的加法分解，不是子组自身的IC。',
                     '理想上界允许未来真值和逐截面理想校准，不能成为推理方法，也不是整个任务的可预测性上限。',
                     '不同监督历史组的成分、波动和数量不同，贡献差不能直接归因为历史长短。']}
    for split in ('validation','test'):
        compare(runs,split,allow_selection_metric_change=True)
        interval=protocol['splits'][split]
        filters=[('date','>=',pd.Timestamp(interval['start'])),('date','<',pd.Timestamp(interval['stop']))]
        pool=pd.read_parquet(Path(sources['root'])/(protocol['universe_table']+'.parquet'),columns=KEYS,filters=filters)
        labels=pd.read_parquet(Path(sources['root'])/(protocol['label_table']+'.parquet'),columns=KEYS+[protocol['label_column']],filters=filters)
        pool=pool.merge(labels,on=KEYS,how='left',validate='one_to_one');del labels
        horizon=pool.date+pd.Timedelta(minutes=protocol['label_horizon_minutes'])
        allowed=(horizon<pd.Timestamp(interval['stop']))&(horizon.dt.normalize()==pool.date.dt.normalize())
        pool.loc[~allowed,protocol['label_column']]=np.nan
        pool['supervised_days']=pool.instrument.map(counts).fillna(0).astype(np.int64)
        pool['band']=history_band(pool.supervised_days.to_numpy())
        for version,run in zip(VERSIONS,runs):
            path=run/'evaluation'/split/'predictions.parquet'
            predictions=pd.read_parquet(path)
            if len(predictions)!=len(pool) or not np.isfinite(predictions.factor).all():raise ValueError('预测覆盖或有限性不符')
            frame=pool.merge(predictions,on=KEYS,how='outer',validate='one_to_one',indicator=True);del predictions
            if not frame._merge.eq('both').all():raise ValueError('预测键不匹配')
            frame=frame.loc[np.isfinite(frame[protocol['label_column']])].copy()
            dates,y=frame.date,frame[protocol['label_column']].astype(np.float64)
            ic,xvar,yvar=contributions(frame.factor,y,dates)
            rank_ic,_,_=contributions(frame.factor.groupby(dates).rank(method='average'),y.groupby(dates).rank(method='average'),dates)
            sections=pd.read_parquet(run/'evaluation'/split/'cross-sections.parquet').set_index('date')
            for name,values in [('ic',ic),('rank_ic',rank_ic)]:
                if not np.allclose(values.groupby(dates).sum().reindex(sections.index),sections[name],rtol=0,atol=1e-12,equal_nan=True):
                    raise ValueError('贡献未还原冻结逐截面指标')
            size=frame.groupby('date').size()
            detail=pd.DataFrame({'date':dates,'band':frame.band,'ic':ic,'rank_ic':rank_ic,
                                 'factor_variance_share':xvar,'label_variance_share':yvar,'sample_fraction':1/dates.map(size)})
            grouped=detail.groupby(['date','band']).sum().reindex(pd.MultiIndex.from_product([sections.index,range(4)],names=['date','band']),fill_value=0)
            means=grouped.groupby('band').mean()
            frozen=read_json(run/'evaluation'/split/'metrics.json')
            for metric in ('ic','rank_ic'):
                if not np.isclose(means[metric].sum(),frozen[metric+'_mean'],atol=1e-12,rtol=0):raise ValueError('贡献未还原冻结均值')
            monthly=grouped.reset_index();monthly['month']=monthly.date.dt.strftime('%Y-%m')
            monthly=monthly.groupby(['month','band'])[means.columns].mean().reset_index()
            monthly['band']=monthly.band.map(dict(enumerate(BANDS)))
            oracle=[]
            for stamp,part in frame.groupby('date',sort=True):
                row={'date':stamp,'month':stamp.strftime('%Y-%m'),'clock':stamp.strftime('%H:%M')}
                for cutoff in (1,60,252):
                    row['days_lt_'+str(cutoff)]=restricted_oracle_ceiling(part.factor,part[protocol['label_column']],part.supervised_days<cutoff)
                oracle.append(row)
            oracle=pd.DataFrame(oracle)
            numeric=['days_lt_1','days_lt_60','days_lt_252']
            result={'solution_id':version,'run_id':run.name,'split':split,'predictions_sha256':sha256(path),
                    'ic_mean':frozen['ic_mean'],'rank_ic_mean':frozen['rank_ic_mean'],'evaluated_rows':len(frame),
                    'contributions_match_frozen_sections':True,'bands':{BANDS[i]:means.loc[i].to_dict() for i in range(4)},
                    'unique_instruments_by_band':{BANDS[i]:int(frame.loc[frame.band==i,'instrument'].nunique()) for i in range(4)},
                    'monthly':monthly.to_dict('records'),'restricted_oracle_ceiling':oracle[numeric].mean().to_dict(),
                    'oracle_by_month':oracle.groupby('month')[numeric].mean().reset_index().to_dict('records'),
                    'oracle_by_clock':oracle.groupby('clock')[numeric].mean().reset_index().to_dict('records')}
            report['results'].append(result)
            print({'version':version,'split':split,'oracle_ceiling':result['restricted_oracle_ceiling']},flush=True)
    for split in ('validation','test'):
        base=next(r for r in report['results'] if r['solution_id']=='v1102' and r['split']==split)
        for new in [r for r in report['results'] if r['solution_id']!='v1102' and r['split']==split]:
            report['deltas'].append({'solution_id':new['solution_id'],'reference':'v1102','split':split,
                'ic_delta':new['ic_mean']-base['ic_mean'],'bands':{band:{key:new['bands'][band][key]-base['bands'][band][key]
                    for key in ('ic','rank_ic','factor_variance_share')} for band in BANDS}})
    write_json(args.output,report)


if __name__=='__main__':main()
