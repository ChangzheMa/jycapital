"""冻结 v1601 的训练/留出抽样拟合差距；CPU只推理，不更新参数。"""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys

import torch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
for directory in ('common','solutions/v1601'):
    sys.path.insert(0,str(ROOT/directory))
from artifacts import now,read_json,sha256,write_json
from v1601_model import MultiFrequencyGRU

spec=importlib.util.spec_from_file_location('temporal_input_reader',ROOT/'scripts/audit-temporal-sensitivity.py')
reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
RUN=ROOT/'outputs/runs/v1601-price-geometry-s42'
EVAL_KEYS={'validation':'b51e456d06d3a546b5de6b9cd0b5a68ff30677e16b8a0bef73868c43452d800c',
           'test':'749e20cfa1b1429cf312269707e42948e45dac6983b486914997ba32869b6da5'}


def section_metrics(scores,targets,minimum):
    valid=np.isfinite(scores)&np.isfinite(targets)
    x,y=np.asarray(scores,np.float64)[valid],np.asarray(targets,np.float64)[valid]
    result={'valid_rows':int(valid.sum()),'ic':None,'rank_ic':None}
    if len(x)<minimum or np.ptp(x)==0 or np.ptp(y)==0:return result
    result['ic']=float(np.corrcoef(x,y)[0,1])
    result['rank_ic']=float(np.corrcoef(pd.Series(x).rank(method='average'),pd.Series(y).rank(method='average'))[0,1])
    return result


def aggregate(rows,seed):
    frame=pd.DataFrame(rows)
    result={'sampled_sections':len(rows),'finite_label_rows':int(frame.valid_rows.sum()),
            'defined_sections':int(frame.ic.notna().sum()),'undefined_sections':int(frame.ic.isna().sum()),
            'ic_mean':float(frame.ic.mean()),'rank_ic_mean':float(frame.rank_ic.mean())}
    daily=frame.groupby('day')[['ic','rank_ic']].agg(['sum','count'])
    rng=np.random.default_rng(seed)
    picks=rng.integers(0,len(daily),size=(2000,len(daily)))
    for name in ('ic','rank_ic'):
        sums=daily[name]['sum'].to_numpy();counts=daily[name]['count'].to_numpy()
        boot=sums[picks].sum(axis=1)/counts[picks].sum(axis=1)
        result[name+'_day_resampling_95pct_interval']=[float(v) for v in np.quantile(boot,[.025,.975])]
    frame['year']=frame.day.str[:4]
    result['by_year']=frame.groupby('year')[['ic','rank_ic']].mean().reset_index().to_dict('records')
    result['by_clock']=frame.groupby('clock')[['ic','rank_ic']].mean().reset_index().to_dict('records')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/frozen-fit-gap-20261023.json')
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config,protocol,selection=[read_json(RUN/(name+'.json')) for name in ('config','protocol','selection')]
    if sha256(RUN/'best.pt')!=selection['model_sha256']:raise ValueError('冻结权重改变')
    source=read_json(RUN/'manifest.json')['source']['files']
    for name in ('solutions/v1601/v1601_model.py','solutions/v1601/v1601_schema.py','common/geometry_resident.py'):
        if sha256(ROOT/name)!=source[name]:raise ValueError('推理依赖与冻结源码不符')
    model=MultiFrequencyGRU(config).eval()
    model.load_state_dict(torch.load(RUN/'best.pt',map_location='cpu',weights_only=True)['state_dict'])
    model.requires_grad_(False)
    cache=ROOT/'outputs/preprocessing/resident-cache'/reader.CACHE_KEY
    block=reader.read_only_block(cache)
    if sha256(cache/'targets.npy')!=block.manifest['files']['targets.npy']['sha256']:raise ValueError('已有训练标签加工结果改变')
    targets=np.load(cache/'targets.npy',mmap_mode='r',allow_pickle=False)
    days=block.arrays['days'].astype('datetime64[ns]').astype('datetime64[D]').astype(str)
    choices=[]
    for year in range(2020,2024):
        dates=np.flatnonzero(np.char.startswith(days,str(year)+'-'))
        choices.extend(sorted(map(int,np.random.default_rng(np.random.SeedSequence([20261023,0,year])).choice(dates,8,replace=False))))
    train_rows,selected_dates=[],[]
    with torch.inference_mode():
        for number,day in enumerate(choices):
            indexes=np.flatnonzero(block.day==day)
            stocks=np.unique(block.stock[indexes])
            chosen=np.sort(np.random.default_rng(np.random.SeedSequence([20261023,1,int(day)])).choice(stocks,128,replace=False))
            mapping=np.full((len(block.manifest['stocks']),8),-1,np.int64)
            mapping[block.stock[indexes],block.clock[indexes]]=indexes
            sample_rows=mapping[chosen]
            latest=7-(sample_rows>=0)[:,::-1].argmax(axis=1)
            representatives=sample_rows[np.arange(len(chosen)),latest]
            values=np.empty((len(chosen),8),np.float32)
            input_hash=hashlib.sha256()
            for start in range(0,len(chosen),16):
                windows,clock=block.batch(representatives[start:start+16])
                for frequency,array in windows.items():input_hash.update(frequency.encode());input_hash.update(array.tobytes())
                prediction=model.forward_group({f:torch.from_numpy(x) for f,x in windows.items()},torch.from_numpy(clock))[...,0].numpy()
                values[start:start+len(prediction)]=prediction
            actual=np.where(sample_rows>=0,targets[sample_rows.clip(0)],np.nan)
            for clock_index,clock in enumerate(reader.CLOCKS):
                train_rows.append({'day':days[day],'clock':clock,**section_metrics(values[:,clock_index],actual[:,clock_index],protocol['min_cross_section_size'])})
            selected_dates.append({'day':days[day],'stock_days':len(chosen),'stock_indices_sha256':hashlib.sha256(chosen.tobytes()).hexdigest(),
                                   'model_inputs_sha256':input_hash.hexdigest(),'selected_targets_sha256':hashlib.sha256(actual.tobytes()).hexdigest()})
            if (number+1)%8==0:print({'completed_training_dates':number+1,'total':len(choices)},flush=True)
    report={'computed_at':now(),'script_sha256':sha256(__file__),
       'reader_script_sha256':sha256(ROOT/'scripts/audit-temporal-sensitivity.py'),
       'run_id':RUN.name,'model_sha256':selection['model_sha256'],'best_epoch':selection['best_epoch'],
       'training_cache_manifest_sha256':sha256(cache/'manifest.json'),
       'model_training_performed':False,'weight_updates':0,'exposure_read':False,'provided_return_formula_changed':False,
       'train_device':'cpu_float32_single_thread','train_input_mode':'read_only_mmap_batch16',
       'train_target_source':'existing_frozen_cross_section_zscore_of_provided_vwap_return',
       'sampling':'训练期每年均匀随机8个交易日、共32日；每日期从当日成员并集随机128只。验证和测试分别均匀随机32日、各128只，复用冻结预测。全部选择不看标签或预测。',
       'training_date_selection':selected_dates,'results':[{'split':'train',**aggregate(train_rows,20261023)}],
       'limitations':['本次是128股票截面的抽样诊断，不能代替1000股票完整协议IC。',
         '训练期使用已拟合过的数据，所得相关性不代表泛化；冻结权重仍由原验证IC选择。',
         '训练用CPU FP32新推理，留出复用原正式冻结预测；未进行新的留出模型选择。',
         '日期重采样区间只反映所选样本变化，不保证时间相关样本的总体置信覆盖。',
         '当前训练样本分数不是最优可达分数或任务可预测性上限。']}
    for split_index,split in enumerate(('validation','test')):
        directory=ROOT/'outputs/preprocessing/evaluation-cache'/EVAL_KEYS[split]
        meta=read_json(directory/'evaluation.json')
        if meta['cache_key']!=EVAL_KEYS[split] or sha256(directory/'keys.parquet')!=meta['keys_sha256']:
            raise ValueError('留出缓存标签来源不符')
        keys=pd.read_parquet(directory/'keys.parquet')
        path=RUN/'evaluation'/split/'predictions.parquet'
        prediction=pd.read_parquet(path)
        frame=keys.merge(prediction,on=['date','instrument'],how='outer',validate='one_to_one',indicator=True)
        if not frame._merge.eq('both').all():raise ValueError('留出预测键不符')
        frame['day']=frame.date.dt.normalize()
        dates=np.sort(frame.day.unique())
        selected=np.sort(np.random.default_rng(np.random.SeedSequence([20261023,2,split_index])).choice(dates,32,replace=False))
        rows,records=[],[]
        for number,day in enumerate(selected):
            daily=frame.loc[frame.day==day]
            stocks=np.sort(daily.instrument.unique())
            chosen=np.random.default_rng(np.random.SeedSequence([20261023,3,split_index,number])).choice(stocks,128,replace=False)
            part=daily.loc[daily.instrument.isin(chosen)]
            stamp=str(pd.Timestamp(day).date())
            records.append({'day':stamp,'stock_days':len(chosen),'stocks_sha256':hashlib.sha256('\n'.join(sorted(chosen)).encode()).hexdigest()})
            for date,section in part.groupby('date',sort=True):
                rows.append({'day':stamp,'clock':date.strftime('%H:%M'),**section_metrics(section.factor.to_numpy(),section[protocol['label_column']].to_numpy(),protocol['min_cross_section_size'])})
        section_path=RUN/'evaluation'/split/'cross-sections.parquet'
        full_sections=pd.read_parquet(section_path)
        selected_sections=full_sections.loc[full_sections.date.dt.normalize().isin(selected)]
        report['results'].append({'split':split,'source_keys_sha256':meta['keys_sha256'],'predictions_sha256':sha256(path),
            'date_selection':records,**aggregate(rows,20261024+split_index),
            'full_universe_context':{'cross_sections_sha256':sha256(section_path),
                'selected_dates_sections':len(selected_sections),
                'same_selected_dates_full_universe_ic':float(selected_sections.ic.mean()),
                'same_selected_dates_full_universe_rank_ic':float(selected_sections.rank_ic.mean()),
                'all_dates_formal_ic':float(full_sections.ic.mean()),
                'all_dates_formal_rank_ic':float(full_sections.rank_ic.mean())}})
    write_json(args.output,report)
    for row in report['results']:print({key:row[key] for key in ('split','ic_mean','rank_ic_mean','defined_sections','ic_day_resampling_95pct_interval')},flush=True)


if __name__=='__main__':main()
