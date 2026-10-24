"""按日精确比较新旧原始数据；区分内容、schema与Parquet字节变化，不训练或改写数据。"""
import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
import json
from pathlib import Path
import sys
import tempfile

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'common'))
from artifacts import now,read_json,sha256
from research_queue import atomic_json

KEYS=('date','instrument')
ALLOWED_TABLES={'cpt_jyc_2026_stock_'+f for f in ('bar1m','bar5m','bar15m','bar30m')}
ALLOWED_TABLES.update(('cpt_jyc_2026_instruments','cpt_jyc_2026_vwap'))
ACTIVE_BARS={f'cpt_jyc_2026_stock_{f}' for f in ('bar1m','bar5m','bar30m')}
BASE_FIELDS=set(read_json(ROOT/'solutions/v3001/config.json')['data']['fields'])
ADDITIONAL_FIELDS={'deal_number','pre_close','adjust_factor'}


def identifier(value):
    return '"'+value.replace('"','""')+'"'


def literal(value):
    return "'"+str(value).replace("'","''")+"'"


def schema(path):
    result={}
    for field in pq.read_schema(path):
        if field.name=='__PARTITION__':continue
        dtype=field.type.value_type if pa.types.is_dictionary(field.type) else field.type
        result[field.name]='string' if pa.types.is_string(dtype) or pa.types.is_large_string(dtype) else str(dtype)
    if set(KEYS)-set(result):raise ValueError('原始数据缺少date/instrument键')
    return result


def months(start,stop):
    cursor=start
    while cursor<stop:
        end=min(date(cursor.year+(cursor.month==12),cursor.month%12+1,1),stop)
        yield cursor,end
        cursor=end


def compare_partition(connection,old_file,new_file,start,stop):
    """日期分块full join；NULL与NaN区分，NaN位模式与正负零按逻辑值处理。"""
    old_schema,new_schema=schema(old_file),schema(new_file)
    if any(old_schema[k]!=new_schema[k] for k in KEYS):
        raise ValueError('键类型改变，不能隐式转换后声称可比')
    added=sorted(set(new_schema)-set(old_schema));removed=sorted(set(old_schema)-set(new_schema))
    changed_types={k:{'old':old_schema[k],'new':new_schema[k]} for k in old_schema.keys()&new_schema.keys()
                   if old_schema[k]!=new_schema[k]}
    comparable=sorted((old_schema.keys()&new_schema.keys())-set(KEYS)-set(changed_types))
    for name,path in (('old_data',old_file),('new_data',new_file)):
        connection.execute(f'CREATE OR REPLACE TEMP VIEW {name} AS SELECT * FROM read_parquet({literal(path)})')
        invalid=connection.execute(f'''SELECT count(*) FROM {name}
            WHERE date IS NULL OR (date>=? AND date<? AND instrument IS NULL)''',[start,stop]).fetchone()[0]
        if invalid:raise ValueError('原始数据包含不能比较的空键')
    days=connection.execute('''SELECT DISTINCT CAST(date AS DATE) AS day FROM (
        SELECT date FROM old_data WHERE date >= ? AND date < ?
        UNION ALL SELECT date FROM new_data WHERE date >= ? AND date < ?
        ) ORDER BY day''',[start,stop,start,stop]).fetchall()
    rows=Counter();fields=Counter();daily=[]
    for (day,) in days:
        end=day+timedelta(days=1)
        counts={}
        for name in ('old_data','new_data'):
            total,date_count,instrument_count,unique=connection.execute(f'''SELECT count(*),count(date),count(instrument),
                count(DISTINCT (date,instrument)) FROM {name} WHERE date>=? AND date<?''',[day,end]).fetchone()
            if total!=date_count or total!=instrument_count or total!=unique:
                raise ValueError(f'{name} {day}: 空键或重复键，不能进行一对一比较')
            counts[name]=total
        expressions=[f'count(*) FILTER (WHERE a.date IS NOT NULL AND b.date IS NOT NULL AND a.{identifier(f)} IS DISTINCT FROM b.{identifier(f)}) AS c{i}'
                     for i,f in enumerate(comparable)]
        sql='''WITH a AS(SELECT * FROM old_data WHERE date>=? AND date<?),
                    b AS(SELECT * FROM new_data WHERE date>=? AND date<?)
            SELECT count(*) FILTER (WHERE a.date IS NULL) AS added,
                   count(*) FILTER (WHERE b.date IS NULL) AS removed,
                   count(*) FILTER (WHERE a.date IS NOT NULL AND b.date IS NOT NULL) AS matched'''
        if expressions:sql+=', '+', '.join(expressions)
        sql+=' FROM a FULL OUTER JOIN b USING(date,instrument)'
        values=connection.execute(sql,[day,end,day,end]).fetchone()
        n_added,n_removed,n_matched=map(int,values[:3])
        if n_matched+n_removed!=counts['old_data'] or n_matched+n_added!=counts['new_data']:
            raise ValueError('连接行数未还原原始键数')
        delta={f:int(n) for f,n in zip(comparable,values[3:]) if n}
        detail={'date':day.isoformat(),'old_rows':counts['old_data'],'new_rows':counts['new_data'],
                'added_keys':n_added,'removed_keys':n_removed,'matched_keys':n_matched,'changed_cells':delta}
        daily.append(detail);fields.update(delta)
        rows.update({k:v for k,v in detail.items() if k not in ('date','changed_cells')})
    same=not (added or removed or changed_types or fields or rows['added_keys'] or rows['removed_keys'])
    return {'start':start.isoformat(),'stop':stop.isoformat(),'logical_content_equal':same,
            'schema_added':added,'schema_removed':removed,'schema_type_changes':changed_types,
            'changed_cells':dict(fields),'rows':dict(rows),'days':daily}


def summarize(partitions,protocol):
    tables=defaultdict(lambda:{'changed_cells':Counter(),'added_keys':0,'removed_keys':0,'schema_changed':False})
    impact={split:{'base_market_values_or_keys_changed':False,'extra_trade_reference_changed':False,
                   'label_values_or_keys_changed':False,'universe_keys_changed':False} for split in protocol['splits']}
    for item in partitions.values():
        table=item['table'];row=tables[table]
        structural=set(item['schema_added'])|set(item['schema_removed'])|set(item['schema_type_changes'])
        row['schema_changed']|=bool(structural)
        row['changed_cells'].update(item['changed_cells'])
        row['added_keys']+=item['rows'].get('added_keys',0);row['removed_keys']+=item['rows'].get('removed_keys',0)
        for d in item['days']:
            for split,interval in protocol['splits'].items():
                if not interval['start'][:10]<=d['date']<interval['stop'][:10]:continue
                keys=d['added_keys']>0 or d['removed_keys']>0
                changed=set(d['changed_cells'])|structural
                if table in ACTIVE_BARS:
                    impact[split]['base_market_values_or_keys_changed']|=keys or bool(changed&BASE_FIELDS)
                    impact[split]['extra_trade_reference_changed']|=keys or bool(changed&ADDITIONAL_FIELDS)
                elif table=='cpt_jyc_2026_vwap':
                    impact[split]['label_values_or_keys_changed']|=keys or 'vwap_return' in changed
                elif table=='cpt_jyc_2026_instruments':
                    impact[split]['universe_keys_changed']|=keys
    for value in tables.values():value['changed_cells']=dict(value['changed_cells'])
    return {'tables':dict(tables),'impact_by_split':impact,
        'note':'变化标志只提示需核对的组件，不自动复用或替换缓存；未使用字段变化不等于当前模型输入变化。'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-root',type=Path,default=ROOT/'data')
    parser.add_argument('--new-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--memory-limit',default='512MB')
    parser.add_argument('--max-temp-size',default='1GB')
    args=parser.parse_args()
    old,new,output=args.old_root.resolve(),args.new_root.resolve(),args.output.resolve()
    if output.is_relative_to(old) or output.is_relative_to(new):raise ValueError('比较报告须位于原始数据目录之外')
    manifest_path=new/'download-manifest.json';manifest=read_json(manifest_path)
    if manifest.get('format')!='jycapital-raw-download-v1' or manifest.get('status')!='complete':
        raise ValueError('新数据必须具有完整下载清单')
    if old==new:raise ValueError('必须指定不同的新旧数据目录')
    request=manifest['request'];start=date.fromisoformat(request['start']);stop=date.fromisoformat(request['stop'])
    if start>=stop:raise ValueError('数据范围为空')
    expected={f'cpt_jyc_2026_stock_{f}/{lo:%Y-%m}.parquet':('cpt_jyc_2026_stock_'+f,lo,hi)
              for f in request['frequencies'] for lo,hi in months(start,stop)}
    expected.update({table+'.parquet':(table,start,stop) for table in ('cpt_jyc_2026_instruments','cpt_jyc_2026_vwap')})
    if set(expected)!=set(manifest['files']):raise ValueError('下载清单缺少请求范围内的文件或包含额外文件')
    for relative,(table,lo,hi) in expected.items():
        record=manifest['files'][relative]
        if (record['table'],record['start'],record['stop'])!=(table,lo.isoformat(),hi.isoformat()):
            raise ValueError('下载清单的文件/表/日期对应关系不一致')
    protocol=read_json(ROOT/'evaluation/protocols/local-v2-ic.json')
    identity={'old_root':str(old),'new_root':str(new),'new_manifest_sha256':sha256(manifest_path),
              'script_sha256':sha256(__file__),'protocol_sha256':sha256(ROOT/'evaluation/protocols/local-v2-ic.json')}
    if output.exists():
        if not args.resume:raise FileExistsError('已有比较记录；续比较请传--resume')
        report=read_json(output)
        if report['identity']!=identity:raise ValueError('比较来源或实现发生变化，请使用新输出文件')
    else:
        report={'created_at':now(),'identity':identity,'status':'comparing','partitions':{},
                'scope':request,'model_training_performed':False,'exposure_read':False,
                'logical_equality':'精确比较键、业务字段类型及值；忽略行序、列序、Parquet编码/元数据、NaN位模式和正负零。NULL与NaN分别处理。',
                'completed_models_retrained':False,'data_or_caches_modified':False}
    output.parent.mkdir(parents=True,exist_ok=True)
    report.update(status='comparing',last_started_at=now())
    atomic_json(output,report)
    hashed={}
    def file_digest(path):
        if path not in hashed:hashed[path]=(sha256(path),path.stat().st_size,path.stat().st_mtime_ns)
        digest,size,mtime=hashed[path]
        if (path.stat().st_size,path.stat().st_mtime_ns)!=(size,mtime):raise ValueError('比较期间源文件发生变化')
        return digest
    try:
        with tempfile.TemporaryDirectory(prefix='raw-comparison-',dir=output.parent) as temporary:
            assert Path(temporary).resolve().is_relative_to(output.parent)
            connection=duckdb.connect()
            connection.execute('SET threads=1')
            connection.execute('SET memory_limit=?',[args.memory_limit])
            connection.execute('SET max_temp_directory_size=?',[args.max_temp_size])
            connection.execute('SET temp_directory=?',[temporary])
            try:
                for relative,record in sorted(manifest['files'].items()):
                    table=record['table']
                    if table not in ALLOWED_TABLES:raise ValueError('下载清单含未允许的表，不读取：'+table)
                    old_file,new_file=(root/relative for root in (old,new))
                    if not old_file.resolve().is_relative_to(old) or not new_file.resolve().is_relative_to(new):
                        raise ValueError('清单路径超出原始数据目录')
                    if not old_file.is_file() or not new_file.is_file():raise FileNotFoundError('缺少新旧对应文件：'+relative)
                    old_sha,new_sha=file_digest(old_file),file_digest(new_file)
                    if new_sha!=record['sha256'] or new_file.stat().st_size!=record['bytes']:
                        raise ValueError('新数据文件与下载清单不一致：'+relative)
                    lo=max(start,date.fromisoformat(record['start']));hi=min(stop,date.fromisoformat(record['stop']))
                    for left,right in months(lo,hi):
                        key=table+'/'+left.strftime('%Y-%m')
                        saved=report['partitions'].get(key)
                        if saved:
                            if saved['old_sha256']!=old_sha or saved['new_sha256']!=new_sha:raise ValueError('已比较的来源文件改变')
                            continue
                        part=compare_partition(connection,old_file,new_file,left,right)
                        part.update(table=table,relative_file=relative,old_sha256=old_sha,new_sha256=new_sha,
                                    file_bytes_equal=old_sha==new_sha)
                        file_digest(old_file);file_digest(new_file)
                        report['partitions'][key]=part
                        atomic_json(output,report)
                        print({'partition':key,'logical_equal':part['logical_content_equal'],
                               'changed_fields':list(part['changed_cells']),'added_keys':part['rows'].get('added_keys',0),
                               'removed_keys':part['rows'].get('removed_keys',0)},flush=True)
            finally:
                connection.close()
        required=ACTIVE_BARS|{'cpt_jyc_2026_instruments','cpt_jyc_2026_vwap'}
        actual={v['table'] for v in report['partitions'].values()}
        report.update(status='complete',completed_at=now(),summary=summarize(report['partitions'],protocol),
            all_compared_content_equal=all(v['logical_content_equal'] for v in report['partitions'].values()),
            covers_required_protocol=(start<=date.fromisoformat(protocol['splits']['train']['start'][:10])
                and stop>=date.fromisoformat(protocol['splits']['test']['stop'][:10]) and required<=actual))
        report.pop('failure',None)
        atomic_json(output,report)
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                      failure={'at':now(),'type':type(error).__name__})
        atomic_json(output,report)
        raise
    print(json.dumps(report['summary'],ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
