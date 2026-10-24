"""内容比较不把重编码当成行情更新，并保留标签与股票池变化的区别。"""
import contextlib
from datetime import date
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('raw_comparison',ROOT/'scripts/compare-raw-snapshots.py')
comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)


def sample():
    return pa.table({'date':pa.array(pd.to_datetime(['2024-01-02 09:30']*3),type=pa.timestamp('ns')),
        'instrument':['A','B','C'],'close':pa.array([1.,2.,3.],type=pa.float32()),
        'volume':pa.array([2**40,None,0],type=pa.int64())})


class SnapshotTests(unittest.TestCase):
    def compare(self,left,right):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a.parquet',Path(tmp)/'b.parquet'
            pq.write_table(left,a,compression='NONE');pq.write_table(right,b,compression='zstd',row_group_size=1)
            with duckdb.connect() as connection:
                result=comparison.compare_partition(connection,a,b,date(2024,1,1),date(2024,2,1))
            self.assertNotEqual(comparison.sha256(a),comparison.sha256(b))
            return result

    def test_reordered_rows_columns_and_encoding_have_same_values(self):
        left=sample();right=left.take(pa.array([2,0,1])).select(['volume','close','instrument','date'])
        right=right.set_column(2,'instrument',right['instrument'].dictionary_encode())
        result=self.compare(left,right)
        self.assertTrue(result['logical_content_equal']);self.assertEqual(result['changed_cells'],{})
        self.assertEqual(result['rows']['matched_keys'],3)

    def test_added_removed_keys_and_exact_value_changes(self):
        left=sample();right=sample().set_column(1,'instrument',pa.array(['A','B','D']))
        right=right.set_column(2,'close',pa.array([1.000001,2.,3.],type=pa.float32()))
        right=right.set_column(3,'volume',pa.array([2**40,0,0],type=pa.int64()))
        result=self.compare(left,right)
        self.assertFalse(result['logical_content_equal'])
        self.assertEqual(result['rows']['added_keys'],1);self.assertEqual(result['rows']['removed_keys'],1)
        self.assertEqual(result['changed_cells'],{'close':1,'volume':1})

    def test_null_nan_and_signed_zero_have_explicit_semantics(self):
        left=sample().set_column(2,'close',pa.array([None,float('nan'),0.],type=pa.float32()))
        right=sample().set_column(2,'close',pa.array([float('nan'),None,-0.],type=pa.float32()))
        result=self.compare(left,right);self.assertEqual(result['changed_cells'],{'close':2})
        self.assertTrue(self.compare(right,right)['logical_content_equal'])

    def test_schema_changes_are_not_silently_cast_and_bad_keys_fail(self):
        left=sample();right=left.set_column(2,'close',left['close'].cast(pa.float64()))
        result=self.compare(left,right)
        self.assertFalse(result['logical_content_equal']);self.assertIn('close',result['schema_type_changes'])
        self.assertNotIn('close',result['changed_cells'])
        with self.assertRaisesRegex(ValueError,'重复键'):
            self.compare(pa.concat_tables([left,left.slice(0,1)]),right)
        missing=left.set_column(0,'date',pa.array([None,pd.Timestamp('2024-01-02'),pd.Timestamp('2024-01-02')],type=pa.timestamp('ns')))
        with self.assertRaisesRegex(ValueError,'空键'):self.compare(missing,right)

    def test_scope_impact_distinguishes_label_auxiliary_and_membership(self):
        protocol={'splits':{'train':{'start':'2020-01-01','stop':'2024-01-01'},'validation':{'start':'2024-01-01','stop':'2024-07-01'}}}
        row={'table':'cpt_jyc_2026_vwap','schema_added':[],'schema_removed':[],'schema_type_changes':{},
             'changed_cells':{'end_price':3},'rows':{'added_keys':0,'removed_keys':0},
             'days':[{'date':'2024-01-02','changed_cells':{'end_price':3},'added_keys':0,'removed_keys':0}]}
        report=comparison.summarize({'a':row},protocol)
        self.assertFalse(report['impact_by_split']['validation']['label_values_or_keys_changed'])
        row['days'][0]['changed_cells']['vwap_return']=1
        self.assertTrue(comparison.summarize({'a':row},protocol)['impact_by_split']['validation']['label_values_or_keys_changed'])
        self.assertFalse(comparison.summarize({'a':row},protocol)['impact_by_split']['train']['label_values_or_keys_changed'])

    def test_cli_verifies_manifest_and_resumes_without_changing_data(self):
        from test_raw_download import FakeDai,download
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            base=Path(tmp);old,new=base/'old',base/'new';old.mkdir()
            manifest=download.download_snapshot(new,start='2020-01-01',stop='2020-03-01',frequencies=('bar1m',),client=FakeDai())
            for relative in manifest['files']:
                target=old/relative;target.parent.mkdir(parents=True,exist_ok=True)
                table=pq.read_table(new/relative)
                pq.write_table(table.take(pa.array(list(reversed(range(len(table)))))),target,compression='NONE')
            original={p:comparison.sha256(p) for root in (old,new) for p in root.rglob('*.parquet')}
            output=base/'reports/result.json'
            argv=['compare','--old-root',str(old),'--new-root',str(new),'--output',str(output)]
            with patch.object(sys,'argv',argv):comparison.main()
            result=json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(result['status'],'complete');self.assertTrue(result['all_compared_content_equal'])
            self.assertFalse(result['covers_required_protocol'])
            result.update(status='interrupted',failure={'type':'KeyboardInterrupt'})
            output.write_text(json.dumps(result),encoding='utf-8')
            with patch.object(sys,'argv',argv+['--resume']):comparison.main()
            self.assertNotIn('failure',json.loads(output.read_text(encoding='utf-8')))
            self.assertEqual(original,{p:comparison.sha256(p) for p in original})
            manifest['files'].pop('cpt_jyc_2026_stock_bar1m/2020-01.parquet')
            (new/'download-manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            with patch.object(sys,'argv',argv+['--resume']),self.assertRaisesRegex(ValueError,'缺少请求范围'):
                comparison.main()


if __name__=='__main__':
    unittest.main()
