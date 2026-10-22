"""只读训练预处理样本，比较绝对值 FP16 与因果前日锚点差值编码的信息损失。"""

import argparse
import hashlib
from pathlib import Path
import sys

import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for folder in ('common', 'solutions/v1002'):
    sys.path.insert(0, str(ROOT/folder))
from artifacts import read_json, write_json, sha256, now
from v1002_preprocessing import FIELDS, PRICE_FIELDS, output_columns
from v1002_data import minute_slots


def rounded(values):
    with np.errstate(over='ignore', invalid='ignore'):
        half = values.astype(np.float16).astype(np.float32)
    return np.where(np.isfinite(values) & ~np.isfinite(half), values, half)


def inspect_frame(frame, frequency):
    columns = output_columns()[2:]
    slots = {int(value): index for index, value in enumerate(minute_slots(frequency))}
    minutes = frame.date.dt.hour * 60 + frame.date.dt.minute
    frame = frame.loc[minutes.isin(slots) & frame.date.eq(frame.date.dt.floor('min'))].copy()
    if frame.empty:
        raise ValueError('诊断样本没有模型规则网格内的 Bar')
    totals = {name: {'finite_pairs':0, 'nonzero_changes':0, 'absolute_erased':0, 'delta_erased':0,
                     'absolute_change_error_sum':0., 'delta_change_error_sum':0., 'reference_change_abs_sum':0.,
                     'finite_values':0, 'absolute_value_error_sum':0., 'delta_value_error_sum':0.}
              for name in columns}
    scored_days = set()
    first_day = frame.date.dt.normalize().min()
    for _, stock in frame.sort_values(['instrument','date']).groupby('instrument', sort=False):
        anchor = np.zeros(len(columns), dtype=np.float32)
        for day, part in stock.groupby(stock.date.dt.normalize(), sort=True):
            values = part[columns].to_numpy(dtype=np.float32)
            absolute = rounded(values)
            delta = (rounded((values - anchor).astype(np.float32)) + anchor).astype(np.float32)
            finite = np.isfinite(values)
            if day > first_day:
                scored_days.add(str(day.date()))
                position = np.array([slots[int(value)] for value in (part.date.dt.hour * 60 + part.date.dt.minute)])
                adjacent = np.diff(position) == 1
                pair = finite[1:] & finite[:-1] & adjacent[:,None]
                reference_change = np.diff(values.astype(np.float64),axis=0)
                absolute_change = np.diff(absolute.astype(np.float64),axis=0)
                delta_change = np.diff(delta.astype(np.float64),axis=0)
                for index,name in enumerate(columns):
                    record=totals[name]
                    valid=pair[:,index]
                    moved=valid & (reference_change[:,index] != 0)
                    record['finite_pairs'] += int(valid.sum())
                    record['nonzero_changes'] += int(moved.sum())
                    record['absolute_erased'] += int((moved & (absolute_change[:,index] == 0)).sum())
                    record['delta_erased'] += int((moved & (delta_change[:,index] == 0)).sum())
                    record['reference_change_abs_sum'] += float(np.abs(reference_change[valid,index]).sum())
                    record['absolute_change_error_sum'] += float(np.abs(absolute_change[valid,index]-reference_change[valid,index]).sum())
                    record['delta_change_error_sum'] += float(np.abs(delta_change[valid,index]-reference_change[valid,index]).sum())
                    mask=finite[:,index]
                    record['finite_values'] += int(mask.sum())
                    record['absolute_value_error_sum'] += float(np.abs(absolute[mask,index].astype(np.float64)-values[mask,index]).sum())
                    record['delta_value_error_sum'] += float(np.abs(delta[mask,index].astype(np.float64)-values[mask,index]).sum())
            # 仅在本日计算结束后更新锚点；每个字段取本日前最后一个有限值供下一日使用。
            available=finite.any(axis=0)
            last=len(values)-1-finite[::-1].argmax(axis=0)
            anchor[available]=values[last[available],np.flatnonzero(available)]
    return totals,sorted(scored_days)


def summarize(record):
    result=dict(record)
    for kind in ('absolute','delta'):
        result[kind+'_nonzero_change_erased_fraction']=record[kind+'_erased']/record['nonzero_changes'] if record['nonzero_changes'] else None
        result[kind+'_change_relative_l1_error']=record[kind+'_change_error_sum']/record['reference_change_abs_sum'] if record['reference_change_abs_sum'] else None
        result[kind+'_value_mae']=record[kind+'_value_error_sum']/record['finite_values'] if record['finite_values'] else None
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/input-precision-audit-20261022.json')
    args=parser.parse_args()
    data=read_json(ROOT/'outputs/runs/v1002-current-temporal-experts-s42/data-manifest.json')
    preprocessing=Path(data['preprocessing']['train']['path'])
    universe_path=Path(data['root'])/'cpt_jyc_2026_instruments.parquet'
    records=[]
    for year in (2020,2021,2022,2023):
        month=f'{year}-06'
        start,stop=pd.Timestamp(month+'-01'),pd.Timestamp(f'{year}-07-01')
        universe=pd.read_parquet(universe_path,columns=['date','instrument'],filters=[('date','>=',start),('date','<',stop)])
        days=pd.DatetimeIndex(sorted(universe.date.dt.normalize().unique())[:5])
        pool=sorted(universe.loc[universe.date.dt.normalize().eq(days[0]),'instrument'].unique())
        stocks=sorted(np.random.default_rng(np.random.SeedSequence([20261022,year])).choice(pool,size=min(64,len(pool)),replace=False).tolist())
        digest=hashlib.sha256('\n'.join(stocks).encode('utf-8')).hexdigest()
        for frequency in ('bar1m','bar5m','bar30m'):
            path=preprocessing/'data'/frequency/(month+'.parquet')
            frame=pd.read_parquet(path,columns=output_columns(),filters=[('date','>=',days[0]),('date','<',days[-1]+pd.Timedelta(days=1)),('instrument','in',stocks)])
            values,scored=inspect_frame(frame,frequency)
            grouped={}
            for kind in ('z','c'):
                for group in ('price','quantity'):
                    selected=[values[kind+'_'+field] for field in FIELDS if (field in PRICE_FIELDS)==(group=='price')]
                    total={key:sum(item[key] for item in selected) for key in selected[0]}
                    grouped[kind+'_'+group]=summarize(total)
            records.append({'month':month,'frequency':frequency,'sample_stocks':len(stocks),'stock_selection_sha256':digest,
                'warmup_date':str(days[0].date()),'scored_dates':scored,'source_sha256':sha256(path),
                'rows_read':len(frame),'groups':grouped,'fields':{key:summarize(value) for key,value in values.items()}})
            print({'month':month,'frequency':frequency,'groups':{key:{field:record[field] for field in
                ('absolute_nonzero_change_erased_fraction','delta_nonzero_change_erased_fraction','absolute_change_relative_l1_error','delta_change_relative_l1_error')}
                for key,record in grouped.items()}},flush=True)
            del frame
    aggregate={}
    for frequency in ('bar1m','bar5m','bar30m'):
        aggregate[frequency]={}
        for group in ('z_price','c_price','z_quantity','c_quantity'):
            selected=[item['groups'][group] for item in records if item['frequency']==frequency]
            keys=[key for key in selected[0] if key.endswith('_sum') or key in ('finite_pairs','nonzero_changes','absolute_erased','delta_erased','finite_values')]
            aggregate[frequency][group]=summarize({key:sum(item[key] for item in selected) for key in keys})
    write_json(args.output,{'schema_version':1,'checked_at':now(),'source_reference_run':'v1002-current-temporal-experts-s42',
        'scope':'四个训练年份各取六月首五个交易日、64只确定随机股票；首日只暖启动，后三频日内相邻规则槽位用于诊断。',
        'formal_training_performed':False,'labels_read':False,'cache_modified':False,
        'absolute_codec':'float32 normalized values -> float16 -> float32; preserve float32 overflow',
        'alternative_codec':'previous-observed-day last-finite float32 anchor + rounded float16 difference; anchor updated only after each day',
        'precision_reference':'frozen preprocessing float32, not raw market source or float64 ground truth',
        'diagnostic_not_model_input':True,'new_candidate_created':False,
        'limitations':['样本诊断，不代表全数据统计或训练收益。','更好保留细微变化可能保留信号，也可能保留噪声，须后续独立模型检验。','差值编码仅用于数值重建诊断，当前训练缓存和平台版本未改变。'],
        'aggregate':aggregate,'samples':records,'script_sha256':sha256(__file__)})


if __name__=='__main__':
    main()
