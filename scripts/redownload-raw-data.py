"""在BigQuant Notebook重新导出原始数据；保持原值和类型，不训练模型。"""
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

FORMAT = 'jycapital-raw-download-v1'
FREQUENCIES = ('bar1m', 'bar5m', 'bar15m', 'bar30m')
BAR_FIELDS = ['date', 'instrument', 'instrument_id', 'adjust_factor', 'pre_close',
              'open', 'high', 'low', 'close', 'deal_number', 'volume', 'amount'] + [
    prefix + str(level) for prefix in ('ask_price', 'bid_price', 'ask_volume', 'bid_volume',
                                     'ask_num_orders', 'bid_num_orders') for level in range(1, 6)]
AUX_FIELDS = {'cpt_jyc_2026_instruments': ['date', 'instrument', 'name'],
              'cpt_jyc_2026_vwap': ['date', 'instrument', 'vwap_return', 'vwap', 'end_price',
                                   'volume', 'amount', 'num_trades']}


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


@contextmanager
def download_lock(directory):
    with (directory / '.download.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def month_intervals(start, stop):
    cursor = start
    while cursor < stop:
        following = date(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1)
        end = min(following, stop)
        yield cursor, end
        cursor = end


def query_arrow(client, sql, start, stop):
    # filters上界也用下一日/月起点；SQL严格<负责排除上界，避免漏掉月末日内数据。
    value = client.query(sql, filters={'date': [start.isoformat(), stop.isoformat()]}, compression=False).arrow()
    if isinstance(value, pa.RecordBatchReader):
        value = value.read_all()
    if not isinstance(value, pa.Table):
        raise TypeError('DAI .arrow()必须返回Arrow Table或RecordBatchReader，不经过DataFrame转换原数值')
    return value


def query_partition(client, table, start, stop):
    sql = (f"SELECT * FROM {table} WHERE date >= '{start.isoformat()}' "
           f"AND date < '{stop.isoformat()}' ORDER BY date, instrument")
    value = query_arrow(client, sql, start, stop)
    columns = [name for name in value.column_names if name != '__PARTITION__']
    value = value.select(columns).replace_schema_metadata(None)
    required = BAR_FIELDS if table.startswith('cpt_jyc_2026_stock_bar') else AUX_FIELDS[table]
    if len(set(columns)) != len(columns) or set(required) - set(columns):
        raise ValueError(f'{table} 缺少原始五档/辅助字段或列名重复；不自动改用e2e压缩表')
    if not pa.types.is_timestamp(value.schema.field('date').type):
        raise ValueError('date必须保留带日内时间的原始timestamp类型')
    # 只对键做检查；原始数值、null和NaN不清洗、不填充、不换精度。
    keys = value.select(['date', 'instrument']).to_pandas()
    if keys.isna().any().any() or keys.duplicated().any():
        raise ValueError(f'{table} 出现空键或重复键，保持失败状态，不静默去重')
    stamps = keys.date
    if stamps.dt.tz is not None:
        stamps = stamps.dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
    if not ((stamps >= pd.Timestamp(start)) & (stamps < pd.Timestamp(stop))).all():
        raise ValueError('查询返回区间外的行')
    if not pd.MultiIndex.from_frame(keys).is_monotonic_increasing:
        raise ValueError('查询结果未按date/instrument排序')
    return value


def export_file(client, table, start, stop, destination, *, is_bar):
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix('.parquet.partial')
    writer, schema, rows, chunks = None, None, 0, []
    try:
        for lo, hi in month_intervals(start, stop):
            count_sql = (f"SELECT CAST(date AS DATE) AS day, count(*) AS row_count FROM {table} "
                         f"WHERE date >= '{lo.isoformat()}' AND date < '{hi.isoformat()}' "
                         'GROUP BY 1 ORDER BY 1')
            counts = query_arrow(client, count_sql, lo, hi).to_pydict()
            days = [pd.Timestamp(value).date() for value in counts['day']]
            sizes = [int(value) for value in counts['row_count']]
            if not days or len(set(days)) != len(days) or any(n <= 0 for n in sizes):
                raise ValueError(f'{table} {lo:%Y-%m}没有完整可用日期计数，不能标记下载成功')
            if any(not lo <= d < hi for d in days):
                raise ValueError('日期计数超出查询区间')
            slices = [(d, d + timedelta(days=1), n) for d, n in zip(days, sizes)] if is_bar else [(lo, hi, sum(sizes))]
            for left, right, expected in slices:
                arrow = query_partition(client, table, left, right)
                if arrow.num_rows != expected:
                    raise ValueError(f'{table} {left}计数与导出行数不同；源表可能在下载中变化')
                if schema is None:
                    schema = arrow.schema
                    writer = pq.ParquetWriter(partial, schema, compression='zstd', compression_level=3,
                                              version='2.6', use_dictionary=True)
                elif not schema.equals(arrow.schema, check_metadata=False):
                    raise ValueError(f'{table}下载期间字段/类型发生变化，不混合不同schema')
                writer.write_table(arrow, row_group_size=100_000)
                rows += arrow.num_rows
                chunks.append({'start': left.isoformat(), 'stop': right.isoformat(), 'rows': arrow.num_rows})
                del arrow
    finally:
        if writer is not None:
            writer.close()
    if schema is None or pq.ParquetFile(partial).metadata.num_rows != rows:
        raise ValueError('Parquet写入行数不一致')
    if destination.exists():
        raise FileExistsError(f'拒绝覆盖已有文件：{destination}')
    record = {'table': table, 'start': start.isoformat(), 'stop': stop.isoformat(), 'rows': rows,
            'bytes': partial.stat().st_size, 'sha256': sha256(partial),
            'schema': [{'name': f.name, 'type': str(f.type)} for f in schema], 'chunks': chunks,
            'completed_at': now()}
    # 在最终文件发布前保存记录；进程若在rename与总清单更新之间退出，续传可恢复。
    atomic_json(destination.with_suffix('.download.json'), record)
    os.replace(partial, destination)
    return record


def download_snapshot(output, *, start='2020-01-01', stop='2025-01-01',
                      frequencies=FREQUENCIES, resume=False, retries=3, client=None):
    """输出新的完整原始目录；resume仅续同一下载，重新获取必须用新目录。"""
    start, stop = date.fromisoformat(start), date.fromisoformat(stop)
    frequencies = tuple(frequencies)
    if start >= stop or not frequencies or len(set(frequencies)) != len(frequencies) or set(frequencies) - set(FREQUENCIES):
        raise ValueError('日期区间或行情频率不合法')
    if type(retries) is not int or retries < 1:
        raise ValueError('retries必须为正整数')
    if client is None:
        import dai
        client = dai
    if getattr(client, 'IS_LOCAL_SHIM', False):
        raise RuntimeError('当前导入的是本地dai.py兼容层，只会读取旧文件。请在BigQuant Notebook运行本脚本。')
    output = Path(output).expanduser().resolve()
    manifest_path = output / 'download-manifest.json'
    request = {'start': start.isoformat(), 'stop': stop.isoformat(), 'frequencies': list(frequencies),
               'auxiliary_tables': list(AUX_FIELDS), 'raw_values': 'unchanged', 'exposure': False}
    if output.exists() and not resume:
        raise FileExistsError('本次重新下载必须使用新目录；中断续传才设置resume=True')
    if resume and not manifest_path.is_file():
        raise FileNotFoundError('续传目录没有本脚本的下载清单')
    output.mkdir(parents=True, exist_ok=True)
    with download_lock(output):
        if resume:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if manifest.get('format') != FORMAT or manifest.get('request') != request:
                raise ValueError('不能在已有下载中更改日期、频率或数据口径')
        else:
            manifest = {'format': FORMAT, 'snapshot_id': output.name, 'created_at': now(),
                'request': request, 'status': 'downloading', 'files': {},
                'source_access': 'BigQuant authorized native DAI',
                'source_atomic_snapshot': False,
                'scope': '各分片在记录时段独立查询；DAI未提供全表统一快照令牌。保留业务列原值/类型，仅去掉__PARTITION__系统字段。'}
        manifest.update(status='downloading', last_started_at=now())
        atomic_json(manifest_path, manifest)
        plan = [(f'cpt_jyc_2026_stock_{f}', lo, hi,
                 f'cpt_jyc_2026_stock_{f}/{lo:%Y-%m}.parquet', True)
                for f in frequencies for lo, hi in month_intervals(start, stop)]
        plan += [(table, start, stop, table+'.parquet', False) for table in AUX_FIELDS]
        try:
            for number, (table, lo, hi, relative, is_bar) in enumerate(plan, 1):
                path = output / relative
                previous = manifest['files'].get(relative)
                recovery = path.with_suffix('.download.json')
                if previous is None and recovery.is_file():
                    recovered = json.loads(recovery.read_text(encoding='utf-8'))
                    staged = path if path.is_file() else path.with_suffix('.parquet.partial')
                    if (recovered.get('table') == table and recovered.get('start') == lo.isoformat()
                            and recovered.get('stop') == hi.isoformat() and staged.is_file()
                            and staged.stat().st_size == recovered['bytes'] and sha256(staged) == recovered['sha256']):
                        if staged != path:
                            os.replace(staged, path)
                        previous = manifest['files'][relative] = recovered
                        atomic_json(manifest_path, manifest)
                if previous is not None:
                    if not path.is_file() or path.stat().st_size != previous['bytes'] or sha256(path) != previous['sha256']:
                        raise ValueError('已下载文件丢失或改变：'+relative)
                    print(f'[{number}/{len(plan)}] 已核对，跳过 {relative}', flush=True)
                    continue
                if path.exists():
                    raise FileExistsError('已有文件缺少可验证下载记录，拒绝覆盖：'+relative)
                for attempt in range(1, retries+1):
                    try:
                        record = export_file(client, table, lo, hi, path, is_bar=is_bar)
                        break
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except Exception:
                        if attempt == retries or path.exists():
                            raise
                        print(f'重试 {relative}（{attempt}/{retries}）；仅重做未完成分片', flush=True)
                        time.sleep(min(2**attempt, 10))
                manifest['files'][relative] = record
                atomic_json(manifest_path, manifest)
                print(f'[{number}/{len(plan)}] {relative}: {record["rows"]:,} 行，{record["bytes"]/2**20:.1f} MiB', flush=True)
            if set(manifest['files']) != {row[3] for row in plan}:
                raise ValueError('已完成文件与请求范围不同')
            manifest.update(status='complete', completed_at=now(), file_count=len(plan),
                total_bytes=sum(r['bytes'] for r in manifest['files'].values()))
            manifest.pop('failure', None)
            atomic_json(manifest_path, manifest)
        except BaseException as error:
            manifest.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                failure={'at': now(), 'type': type(error).__name__})
            atomic_json(manifest_path, manifest)
            raise
    print(f'下载完成：{output}\n共{manifest["file_count"]}个Parquet文件；原vwap_return未重算，未访问exposure。', flush=True)
    return manifest


def pack_snapshot(output, archive_directory, *, max_part_gib=4):
    """可选：按文件分卷打包已压缩的Parquet；不删除原文件，目录须新建。"""
    output, archives = Path(output).resolve(), Path(archive_directory).resolve()
    manifest_path = output / 'download-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('format') != FORMAT or manifest.get('status') != 'complete':
        raise ValueError('只能打包完整下载')
    limit = int(max_part_gib * 2**30)
    if limit <= 0 or archives == output or output in archives.parents:
        raise ValueError('分卷大小须为正，归档目录须在原始数据目录之外')
    archives.mkdir(parents=True, exist_ok=False)
    groups, group, size = [], ['download-manifest.json'], manifest_path.stat().st_size
    for relative, record in sorted(manifest['files'].items()):
        path = (output / relative).resolve()
        if not path.is_relative_to(output) or not path.is_file() or sha256(path) != record['sha256']:
            raise ValueError('文件路径或内容与下载清单不符')
        if size + record['bytes'] > limit and group:
            groups.append(group); group=[]; size=0
        group.append(relative); size+=record['bytes']
    if group:
        groups.append(group)
    result=[]
    for number, members in enumerate(groups, 1):
        path = archives/f'{output.name}-part{number:03d}.zip'
        temporary = path.with_suffix('.zip.partial')
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for relative in members:
                archive.write(output/relative, arcname=output.name+'/'+relative)
        os.replace(temporary, path)
        result.append(path)
        print(f'归档完成：{path.name}（{path.stat().st_size/2**30:.2f} GiB）',flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--start',default='2020-01-01')
    parser.add_argument('--stop',default='2025-01-01',help='结束日不包含')
    parser.add_argument('--frequencies',nargs='+',choices=FREQUENCIES,default=list(FREQUENCIES))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--archive-directory',type=Path,help='可选打包，需要额外磁盘空间')
    args=parser.parse_args()
    download_snapshot(args.output,start=args.start,stop=args.stop,frequencies=args.frequencies,resume=args.resume)
    if args.archive_directory:
        pack_snapshot(args.output,args.archive_directory)


if __name__ == '__main__':
    main()
