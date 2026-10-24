"""比较两份已提供vwap_return的数值与时点变化，不重算标签，不读取模型。"""
import argparse
from collections import defaultdict
from datetime import date
from pathlib import Path
import sys
import tempfile

import duckdb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'common'))
from artifacts import now,read_json,sha256,write_json


def section_similarity(frame,minimum):
    paired=frame.loc[np.isfinite(frame.old) & np.isfinite(frame.new)].copy()
    rows=[]
    for stamp,group in paired.groupby('date',sort=True):
        if len(group)<minimum:continue
        old,new=group.old.to_numpy(np.float64),group.new.to_numpy(np.float64)
        def correlation(a,b):
            a,b=a-a.mean(),b-b.mean();denominator=np.linalg.norm(a)*np.linalg.norm(b)
            return float(np.dot(a,b)/denominator) if denominator>0 else None
        rows.append({'date':stamp,'pearson':correlation(old,new),
                     'rank':correlation(pd.Series(old).rank(method='average').to_numpy(),
                                        pd.Series(new).rank(method='average').to_numpy())})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();proof=read_json(args.comparison)
    if proof['status']!='complete' or not proof['covers_required_protocol']:raise ValueError('先完成完整数据比较')
    protocol=read_json(ROOT/'evaluation/protocols/local-v2-ic.json')
    old_root,new_root=Path(proof['identity']['old_root']),Path(proof['identity']['new_root'])
    table=protocol['label_table'];pool=protocol['universe_table'];column=protocol['label_column']
    if column!='vwap_return':raise ValueError('本诊断仅直接比较提供的vwap_return')
    hashes={}
    for side,root in (('old',old_root),('new',new_root)):
        path=root/(table+'.parquet');digest=sha256(path)
        row=next(v for v in proof['partitions'].values() if v['table']==table)
        if digest!=row[side+'_sha256']:raise ValueError('标签来源改变')
        hashes[side]=digest
    pool_path=new_root/(pool+'.parquet')
    expected=next(v for v in proof['partitions'].values() if v['table']==pool)
    if sha256(pool_path)!=expected['new_sha256']:raise ValueError('股票池来源改变')
    if any(v['rows'].get('added_keys',0) or v['rows'].get('removed_keys',0) for v in proof['partitions'].values() if v['table']==pool):
        raise ValueError('本诊断要求原股票池键一致')
    output={'computed_at':now(),'script_sha256':sha256(__file__),'source_label_sha256':hashes,
        'pool_sha256':expected['new_sha256'],'comparison_sha256':sha256(args.comparison),
        'label_formula_recomputed':False,'models_read_or_rerun':False,'exposure_read':False,
        'scope':'固定协议股票池；只比较直接提供的两份标签，不包含池外指数。相似度使用双方共同有限的样本。',
        'results':{},'limitations':['标签相关性不是模型IC，也不能证明新标签更容易预测。',
            '变化数量包括缺失位置变化；不反推具体标签公式，不将修订归因为未核实的机制。']}
    with tempfile.TemporaryDirectory(prefix='label-revision-') as temporary:
        connection=duckdb.connect();connection.execute('SET threads=1');connection.execute("SET memory_limit='256MB'")
        connection.execute("SET max_temp_directory_size='1GB'");connection.execute('SET temp_directory=?',[temporary])
        try:
            for split,interval in protocol['splits'].items():
                grouped=defaultdict(lambda:{'rows':0,'old_finite':0,'new_finite':0,'both_finite':0,
                    'finite_to_missing':0,'missing_to_finite':0,'changed_finite':0,'absolute_sum':0.,'square_sum':0.,'changed_absolute':[]})
                sections=[]
                for month in pd.period_range(interval['start'],pd.Timestamp(interval['stop'])-pd.Timedelta(1,'ns'),freq='M'):
                    lo,hi=month.start_time,(month+1).start_time
                    query='''WITH p AS(SELECT date,instrument FROM read_parquet(?) WHERE date>=? AND date<?),
                        a AS(SELECT date,instrument,vwap_return FROM read_parquet(?) WHERE date>=? AND date<?),
                        b AS(SELECT date,instrument,vwap_return FROM read_parquet(?) WHERE date>=? AND date<?)
                        SELECT p.date,a.vwap_return AS old,b.vwap_return AS new FROM p
                        LEFT JOIN a USING(date,instrument) LEFT JOIN b USING(date,instrument) ORDER BY p.date,p.instrument'''
                    frame=connection.execute(query,[str(pool_path),lo,hi,str(old_root/(table+'.parquet')),lo,hi,
                                                    str(new_root/(table+'.parquet')),lo,hi]).fetchdf()
                    horizon=frame.date+pd.Timedelta(minutes=protocol['label_horizon_minutes'])
                    frame=frame.loc[(horizon<pd.Timestamp(interval['stop'])) & (horizon.dt.normalize()==frame.date.dt.normalize())]
                    sections+=section_similarity(frame,protocol['min_cross_section_size'])
                    for clock,g in frame.groupby(frame.date.dt.strftime('%H:%M')):
                        row=grouped[clock];a,b=g.old.to_numpy(np.float64),g.new.to_numpy(np.float64)
                        va,vb=np.isfinite(a),np.isfinite(b);both=va&vb;delta=b[both]-a[both]
                        changed=np.abs(delta[delta!=0])
                        for key,value in {'rows':len(g),'old_finite':va.sum(),'new_finite':vb.sum(),'both_finite':both.sum(),
                            'finite_to_missing':(va&~vb).sum(),'missing_to_finite':(~va&vb).sum(),'changed_finite':len(changed)}.items():row[key]+=int(value)
                        row['absolute_sum']+=float(np.abs(delta).sum());row['square_sum']+=float(np.dot(delta,delta))
                        row['changed_absolute'].append(changed)
                similarity=pd.DataFrame(sections);rows={}
                for clock,r in sorted(grouped.items()):
                    changed=np.concatenate(r.pop('changed_absolute'))
                    selected=similarity.loc[similarity.date.dt.strftime('%H:%M')==clock]
                    r.update(changed_or_finiteness_fraction=(r['changed_finite']+r['finite_to_missing']+r['missing_to_finite'])/r['rows'],
                        mean_absolute_difference_on_common=r.pop('absolute_sum')/r['both_finite'],
                        rms_difference_on_common=np.sqrt(r.pop('square_sum')/r['both_finite']),
                        changed_absolute_quantiles=dict(zip(('p50','p90','p99','max'),np.quantile(changed,[.5,.9,.99,1]).tolist())) if len(changed) else {},
                        mean_section_label_pearson=float(selected.pearson.mean()),mean_section_label_rank_correlation=float(selected['rank'].mean()),
                        undefined_pearson_sections=int(selected.pearson.isna().sum()))
                    rows[clock]=r
                output['results'][split]={'clocks':rows,'mean_section_label_pearson':float(similarity.pearson.mean()),
                    'mean_section_label_rank_correlation':float(similarity['rank'].mean()),'cross_sections':len(similarity)}
                print({'split':split,'label_pearson':output['results'][split]['mean_section_label_pearson'],
                       'changed_by_clock':{k:v['changed_or_finiteness_fraction'] for k,v in rows.items()}},flush=True)
        finally:connection.close()
    write_json(args.output,output)


if __name__=='__main__':main()
