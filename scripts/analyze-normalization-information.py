"""只读训练期固定样本，核对数量绝对尺度的可恢复性和精度代价。"""
import argparse
import hashlib
from pathlib import Path
import sys

import torch
import numpy as np
import pandas as pd
import pyarrow

ROOT=Path(__file__).resolve().parents[1]
for directory in ('common','solutions/v2401'):
    sys.path.insert(0,str(ROOT/directory))
from artifacts import read_json,write_json,sha256,now
from v2401_preprocessing import FIELDS,PRICE_FIELDS,unpack_moments,population_std,empty_moments
from v2401_data import minute_slots

KEYS=['date','instrument']
QUANTITIES=[f for f in FIELDS if f not in PRICE_FIELDS]
GROUPS={'volume':['volume'],'amount':['amount'],
    'book_volume':[f'{side}_volume{level}' for side in ('ask','bid') for level in range(1,6)],
    'book_order_count':[f'{side}_num_orders{level}' for side in ('ask','bid') for level in range(1,6)]}


def half_round(values):
    with np.errstate(over='ignore',invalid='ignore'):
        rounded=values.astype(np.float16).astype(np.float32)
    return np.where(np.isfinite(values)&~np.isfinite(rounded),values,rounded)


def inverse_fixed(values, means, scales, trusted):
    values=np.asarray(values,np.float32)
    restored=values*np.asarray(scales,np.float32)+np.asarray(means,np.float32)
    return np.where(np.asarray(trusted,bool)&np.isfinite(values),restored,np.nan).astype(np.float32)


def distribution(values):
    values=np.asarray(values)
    values=values[np.isfinite(values)]
    return {'count':int(len(values)), 'quantiles':{str(q):float(np.quantile(values,q)) if len(values) else None
            for q in (0,.01,.05,.5,.95,.99,1)}}


def regular(frame, frequency):
    minutes=frame.date.dt.hour*60+frame.date.dt.minute
    return frame.loc[minutes.isin(minute_slots(frequency)) & frame.date.eq(frame.date.dt.floor('min'))].copy()


def digest(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame,index=False).to_numpy().tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/normalization-information-audit-20261024.json')
    args=parser.parse_args()
    pyarrow.set_cpu_count(1);pyarrow.set_io_thread_count(1);torch.set_num_threads(1)
    run=ROOT/'outputs/runs/v2401-price-memory-s42'
    manifest=read_json(run/'data-manifest.json')
    preprocessing=read_json(run/'source/solutions/v2401/config.json')
    policy=preprocessing['normalization']
    raw_root=Path(manifest['root']);prepared=Path(manifest['preprocessing']['train']['path'])
    files={r['path']:r for r in manifest['files']}
    reference=read_json(ROOT/'experiments/price-geometry-audit-20261022.json')
    normalization={};moments={};metadata={}
    positions=[FIELDS.index(f) for f in QUANTITIES]
    parent_stocks=read_json(ROOT/'outputs/preprocessing/resident-cache/c760f3882029e185c69b39c92509beeeea3995e2214360a6196046d70158563b/manifest.json')['stocks']
    for frequency in ('bar1m','bar5m','bar30m'):
        path=run/'normalization'/(frequency+'.json')
        statistics=read_json(path)
        assert statistics['fields']==FIELDS and pd.Timestamp(statistics['fit_interval']['stop'])<=pd.Timestamp('2024-01-01')
        assert sha256(path)==sha256(prepared/'normalization'/(frequency+'.json'))
        table={stock:unpack_moments(value) for stock,value in statistics['stocks'].items()}
        blank=empty_moments(len(FIELDS))
        entries=[table.get(s,blank) for s in parent_stocks]
        means=np.stack([item['mean'][positions] for item in entries])
        scales=np.stack([population_std(item)[positions] for item in entries])
        counts=np.stack([item['count'][positions] for item in entries])
        trusted=(counts>=policy['min_observations']) & (scales>policy['scale_epsilon'])
        metadata[frequency]={'normalization_sha256':sha256(path),'fit_interval':statistics['fit_interval'],
            'training_cache_stocks':len(parent_stocks),'normalization_stocks':len(table),
            'stocks_without_fixed_statistics':sum(s not in table for s in parent_stocks),
            'field_statistics':{field:{'trusted_stock_fields':int(trusted[:,j].sum()),
                'untrusted_stock_fields':int((~trusted[:,j]).sum()),
                'mean_log1p_distribution':distribution(means[trusted[:,j],j]),
                'log1p_scale_distribution':distribution(scales[trusted[:,j],j])} for j,field in enumerate(QUANTITIES)},
            'float32_mean_scale_and_bool_trust_bytes':len(parent_stocks)*len(QUANTITIES)*9}
        normalization[frequency]=statistics;moments[frequency]=table
    samples=[]
    collected={f:{g:{m:[] for m in ('float32_z','fp16_z_with_overflow')} for g in GROUPS}
               for f in ('bar1m','bar5m','bar30m')}
    for year in (2020,2021,2022,2023):
        month=f'{year}-06'
        universe=pd.read_parquet(raw_root/'cpt_jyc_2026_instruments.parquet',columns=KEYS,
            filters=[('date','>=',pd.Timestamp(month+'-01')),('date','<',pd.Timestamp(f'{year}-07-01'))])
        days=pd.DatetimeIndex(sorted(universe.date.dt.normalize().unique())[:5])
        pool=sorted(universe.loc[universe.date.dt.normalize().eq(days[0]),'instrument'].unique())
        stocks=sorted(np.random.default_rng(np.random.SeedSequence([20261022,year])).choice(pool,64,replace=False).tolist())
        stock_digest=hashlib.sha256('\n'.join(stocks).encode()).hexdigest()
        assert stock_digest==next(s for s in reference['samples'] if s['month']==month and s['frequency']=='bar1m')['stock_selection_sha256']
        filters=[('date','>=',days[0]),('date','<',days[-1]+pd.Timedelta(days=1)),('instrument','in',stocks)]
        for frequency in collected:
            raw_path=raw_root/('cpt_jyc_2026_stock_'+frequency)/(month+'.parquet')
            raw=pd.read_parquet(raw_path,columns=KEYS+QUANTITIES,filters=filters)
            processed=pd.read_parquet(prepared/'data'/frequency/(month+'.parquet'),
                columns=KEYS+['z_'+f for f in QUANTITIES],filters=filters)
            frame=regular(raw,frequency).merge(regular(processed,frequency),on=KEYS,how='outer',validate='one_to_one',indicator=True)
            assert frame._merge.eq('both').all()
            frame=frame.loc[frame.date.dt.normalize()>days[0]].copy()
            assert (frame.date<pd.Timestamp('2024-01-01')).all()
            table=moments[frequency]
            codes,unique_stocks=pd.factorize(frame.instrument,sort=False)
            entries=[table.get(s,empty_moments(len(FIELDS))) for s in unique_stocks]
            means=np.stack([item['mean'][positions] for item in entries])[codes]
            scales=np.stack([population_std(item)[positions] for item in entries])[codes]
            counts=np.stack([item['count'][positions] for item in entries])[codes]
            trusted=(counts>=policy['min_observations']) & (scales>policy['scale_epsilon'])
            values=frame[QUANTITIES].to_numpy(dtype=np.float64)
            valid=np.isfinite(values)&(values>=0)
            truth=np.log1p(np.where(valid,values,0))
            z=frame[['z_'+f for f in QUANTITIES]].to_numpy(dtype=np.float32)
            views={'float32_z':inverse_fixed(z,means,scales,trusted),
                   'fp16_z_with_overflow':inverse_fixed(half_round(z),means,scales,trusted)}
            groups={}
            for name,fields in GROUPS.items():
                indices=[QUANTITIES.index(f) for f in fields]
                mask=valid[:,indices]&trusted[:,indices]&np.isfinite(z[:,indices])
                detail={'raw_finite_nonnegative_values':int(valid[:,indices].sum()),'recoverable_values':int(mask.sum()),
                    'untrusted_finite_values_not_inverted':int((valid[:,indices]&~trusted[:,indices]).sum()),'views':{}}
                for mode,decoded in views.items():
                    predicted=decoded[:,indices]
                    error=(predicted-truth[:,indices])[mask]
                    collected[frequency][name][mode].append(error.astype(np.float32))
                    detail['views'][mode]={'absolute_log1p_error':distribution(np.abs(error)),
                        'raw_zero_values':int((mask&(values[:,indices]==0)).sum()),
                        'raw_zero_reconstructed_nonzero':int((mask&(values[:,indices]==0)&(predicted!=0)).sum()),
                        'raw_positive_reconstructed_nonpositive':int((mask&(values[:,indices]>0)&(predicted<=0)).sum()),
                        'absolute_log1p_error_above_0_001':int((np.abs(error)>.001).sum())}
                groups[name]=detail
            expected=files['raw/'+raw_path.relative_to(raw_root).as_posix()]
            assert raw_path.stat().st_size==expected['bytes']
            samples.append({'month':month,'frequency':frequency,'rows':len(frame),'dates':[str(d.date()) for d in days[1:]],
                'stock_selection_sha256':stock_digest,'groups':groups,
                'raw_source':{'path':raw_path.relative_to(ROOT).as_posix(),'bytes':expected['bytes'],
                    'expected_sha256_from_parent_manifest':expected['sha256'],'full_file_sha256_recomputed':False,
                    'sample_content_sha256':digest(raw)},'processed_sample_content_sha256':digest(processed)})
            print({'month':month,'frequency':frequency,'rows':len(frame)},flush=True)
    aggregate={}
    for frequency in collected:
        rows=[r for r in samples if r['frequency']==frequency]
        aggregate[frequency]={}
        for name in GROUPS:
            item={k:sum(r['groups'][name][k] for r in rows) for k in ('raw_finite_nonnegative_values','recoverable_values','untrusted_finite_values_not_inverted')}
            item['views']={}
            for mode in collected[frequency][name]:
                error=np.concatenate(collected[frequency][name][mode])
                detail={k:sum(r['groups'][name]['views'][mode][k] for r in rows) for k in (
                    'raw_zero_values','raw_zero_reconstructed_nonzero','raw_positive_reconstructed_nonpositive','absolute_log1p_error_above_0_001')}
                detail['absolute_log1p_error']=distribution(np.abs(error))
                detail['relative_error_of_quantity_plus_one']=distribution(np.abs(np.expm1(error.astype(np.float64))))
                item['views'][mode]=detail
            aggregate[frequency][name]=item
    write_json(args.output,{'computed_at':now(),'script_sha256':sha256(__file__),
        'parent_run_id':run.name,'parent_data_manifest_sha256':sha256(run/'data-manifest.json'),
        'normalization_policy':policy,'quantities':QUANTITIES,'groups':GROUPS,'metadata':metadata,
        'sample_protocol':'复用2020—2023各六月首五日和每年64股；首日按既有审计约定排除，主诊断16日1024股票日。',
        'inverse_arithmetic':'FP32 z乘FP32固定尺度，再加FP32固定均值；不对未知／动态回退字段假设固定参数。',
        'aggregate':aggregate,'samples':samples,'labels_read':False,'exposure_read':False,
        'price_fields_read':False,'model_training_performed':False,'cache_modified':False,'new_candidate_created':False,
        'limitations':['固定归一化在已知参数时可逆；本诊断不证明原模型完全丢失信息或绝对尺度可预测收益。',
            '误差包含已有FP32存储、FP16舍入和FP32反变换；恢复值不是原始数据的逐位副本，合法零值也可能产生微小残差。',
            '仅训练期固定抽样；统计表按训练缓存股票等权统计，不能解释为验证／测试覆盖率。',
            'v2001已经检验共同盘口数量坐标且整体失败；本轮不将该方向重新标记为完全未探索。',
            '没有构造成交均价、量额比、收益公式或新标签，没有查询训练以外的行情。']})


if __name__=='__main__':main()
