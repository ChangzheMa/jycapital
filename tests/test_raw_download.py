"""小型Arrow数据验证原值保留、月末边界、续传和新训练入口等待。"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]


def load(name, filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


download=load('raw_download','redownload-raw-data.py')
queue=load('refresh_queue','research-queue.py')


class FakeDai:
    def __init__(self):
        self.calls=[];self.bad_count=False
        stamps=pa.array(pd.to_datetime(['2020-01-31 14:30','2020-01-31 15:00','2020-02-01 09:30']),type=pa.timestamp('ns'))
        def table(fields):
            result={}
            for name in fields:
                if name=='date':result[name]=stamps
                elif name in ('instrument','name'):result[name]=pa.array(['000001.SZ']*3)
                elif name in ('volume','deal_number'):result[name]=pa.array([0,2**40,None],type=pa.int64())
                elif name=='amount':result[name]=pa.array([1.234567890123456,None,-3.5],type=pa.float64())
                elif name=='vwap_return':result[name]=pa.array([.125,None,-.375],type=pa.float32())
                else:result[name]=pa.array([0.125,None,-2.5],type=pa.float32())
            result['__PARTITION__']=pa.array([1,1,2],type=pa.int64())
            return pa.table(result)
        self.tables={'cpt_jyc_2026_stock_bar1m':table(download.BAR_FIELDS),
                     **{n:table(f) for n,f in download.AUX_FIELDS.items()}}

    def query(self,sql,filters,compression):
        assert compression is False
        assert 'exposure' not in sql and 'e2e_' not in sql
        name=re.search(r'FROM (\w+)',sql).group(1)
        lo,hi=re.findall(r"date (?:>=|<) '([^']+)'",sql)
        assert filters=={'date':[lo,hi]}
        self.calls.append((name,lo,hi,'count(*)' in sql))
        table=self.tables[name]
        mask=pc.and_(pc.greater_equal(table['date'],pa.scalar(pd.Timestamp(lo),type=pa.timestamp('ns'))),
                     pc.less(table['date'],pa.scalar(pd.Timestamp(hi),type=pa.timestamp('ns'))))
        table=table.filter(mask)
        if 'count(*)' in sql:
            counts=table.select(['date']).to_pandas().groupby(lambda i:pd.Timestamp(table['date'][i].as_py()).date()).size()
            table=pa.table({'day':pa.array(list(counts.index),type=pa.date32()),
                            'row_count':pa.array([int(v)+int(self.bad_count) for v in counts],type=pa.int64())})
        return SimpleNamespace(arrow=lambda:table)


class DownloadTests(unittest.TestCase):
    def run_download(self,path,client,**kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return download.download_snapshot(path,start='2020-01-01',stop='2020-03-01',
                frequencies=('bar1m',),retries=1,client=client,**kwargs)

    def test_month_end_original_types_values_aux_and_archives(self):
        client=FakeDai()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'snapshot';manifest=self.run_download(path,client)
            self.assertEqual(manifest['status'],'complete');self.assertEqual(manifest['file_count'],4)
            parts=[pq.read_table(path/'cpt_jyc_2026_stock_bar1m'/f'{m}.parquet') for m in ('2020-01','2020-02')]
            expected=client.tables['cpt_jyc_2026_stock_bar1m'].drop_columns(['__PARTITION__'])
            self.assertTrue(pa.concat_tables(parts).equals(expected))
            for name,original in client.tables.items():
                if name in download.AUX_FIELDS:self.assertTrue(pq.read_table(path/(name+'.parquet')).equals(original.drop_columns(['__PARTITION__'])))
            self.assertIn(('cpt_jyc_2026_stock_bar1m','2020-01-31','2020-02-01',False),client.calls)
            with contextlib.redirect_stdout(io.StringIO()):
                archives=download.pack_snapshot(path,Path(temporary)/'archives',max_part_gib=.00003)
            members={}
            for archive in archives:
                with zipfile.ZipFile(archive) as file:members.update({n:file.read(n) for n in file.namelist()})
            self.assertEqual(set(members),{path.name+'/'+n for n in ['download-manifest.json',*manifest['files']]})
            for relative in manifest['files']:self.assertEqual(members[path.name+'/'+relative],(path/relative).read_bytes())

    def test_resume_recovers_published_file_without_manifest_entry(self):
        client=FakeDai()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'snapshot';manifest=self.run_download(path,client)
            relative='cpt_jyc_2026_stock_bar1m/2020-01.parquet'
            manifest['files'].pop(relative);manifest['status']='interrupted'
            (path/'download-manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            count=len(client.calls);restored=self.run_download(path,client,resume=True)
            self.assertEqual(len(client.calls),count)
            self.assertEqual(restored['status'],'complete');self.assertIn(relative,restored['files'])
            # 已完成文件遭修改时，拒绝将它当作有效续传或覆盖原文件。
            (path/relative).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'改变'):self.run_download(path,client,resume=True)

    def test_failed_count_stays_incomplete_and_can_resume(self):
        client=FakeDai();client.bad_count=True
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'snapshot'
            with self.assertRaisesRegex(ValueError,'行数不同'):self.run_download(path,client)
            self.assertEqual(json.loads((path/'download-manifest.json').read_text(encoding='utf-8'))['status'],'failed')
            self.assertFalse((path/'cpt_jyc_2026_stock_bar1m/2020-01.parquet').exists())
            client.bad_count=False
            self.assertEqual(self.run_download(path,client,resume=True)['status'],'complete')

    def test_existing_directory_local_shim_and_duplicate_keys_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary);sentinel=path/'old-data';sentinel.write_text('keep')
            with self.assertRaises(FileExistsError):self.run_download(path,FakeDai())
            with self.assertRaisesRegex(RuntimeError,'兼容层'):
                self.run_download(path/'new',SimpleNamespace(IS_LOCAL_SHIM=True))
            self.assertEqual(sentinel.read_text(),'keep');self.assertFalse((path/'new').exists())
            client=FakeDai();name='cpt_jyc_2026_stock_bar1m'
            client.tables[name]=pa.concat_tables([client.tables[name].slice(0,1)]*2)
            with self.assertRaisesRegex(ValueError,'重复键'):self.run_download(path/'duplicate',client)

    def test_gate_only_blocks_new_dispatch(self):
        original={'next':{'run_id':'unstarted'},'active_count':1,'unevaluated_count':3,'candidates':[{'run_status':'complete'}]}
        gated=queue.apply_data_refresh_gate(original,{'data_refresh':{'block_new_training':True,'status':'awaiting_user_download'}})
        self.assertIsNone(gated['next']);self.assertTrue(gated['data_refresh_gate']['blocked'])
        self.assertEqual(gated['active_count'],original['active_count']);self.assertEqual(gated['candidates'],original['candidates'])
        self.assertEqual(original['next'],{'run_id':'unstarted'})
        self.assertIs(queue.apply_data_refresh_gate(original,{}),original)

    def test_gate_blocks_controller_fallback_from_candidate_list(self):
        candidates=[{'run_id':'new','candidate_status':'ready','run_status':'not_started','readiness_error':None},
                    {'run_id':'active','candidate_status':'ready','run_status':'running','readiness_error':None},
                    {'run_id':'done','candidate_status':'ready','run_status':'complete','readiness_error':None}]
        original={'next':candidates[0],'candidates':candidates,'active_count':1,'unevaluated_count':2}
        gated=queue.apply_data_refresh_gate(original,{'data_refresh':{'block_new_training':True}})
        ready=[row for row in gated['candidates'] if row.get('candidate_status')=='ready'
               and row['run_status']=='not_started' and not row.get('readiness_error')]
        selected=gated['next'] if gated['next'] else (ready[0] if ready else None)
        self.assertIsNone(selected)
        self.assertEqual(gated['candidates'][0]['source_candidate_status'],'ready')
        self.assertEqual(gated['candidates'][1:],candidates[1:])
        self.assertEqual(candidates[0]['candidate_status'],'ready')

    def test_standalone_notebook_embeds_current_script_and_runs_fixture(self):
        notebook=json.loads((ROOT/'examples/redownload-raw-data.ipynb').read_text(encoding='utf-8'))
        source=(ROOT/'scripts/redownload-raw-data.py').read_text(encoding='utf-8')
        self.assertEqual(''.join(notebook['cells'][2]['source']),source.rsplit("\nif __name__ == '__main__':",1)[0]+'\n')
        scope={'__name__':'notebook_fixture'};client=FakeDai()
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                self.assertEqual(cell['outputs'],[]);self.assertIsNone(cell['execution_count'])
                compile(''.join(cell['source']),'<download-notebook>','exec')
        exec(''.join(notebook['cells'][1]['source']),scope)
        exec(''.join(notebook['cells'][2]['source']),scope)
        self.assertEqual(client.calls,[])
        original=scope['download_snapshot']
        scope['download_snapshot']=lambda *args,**kwargs:original(*args,client=client,**kwargs)
        with tempfile.TemporaryDirectory() as temporary,contextlib.redirect_stdout(io.StringIO()):
            scope.update(OUTPUT_DIR=Path(temporary)/'notebook-data',START_DATE='2020-01-01',STOP_DATE='2020-03-01',DOWNLOAD_FREQUENCIES=('bar1m',))
            exec(''.join(notebook['cells'][3]['source']),scope)
            exec(''.join(notebook['cells'][4]['source']),scope)
            self.assertEqual(scope['manifest']['status'],'complete')
            self.assertFalse(scope['MAKE_ARCHIVES'])


if __name__=='__main__':
    unittest.main()
