"""新运行路径解析不得覆盖已有启动，也不得把旧输入伪装成新绑定。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('training_data_paths',ROOT/'scripts/training-data-paths.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


class DataPathsTests(unittest.TestCase):
    def fixture(self,root):
        solution=root/'solutions/v3001';solution.mkdir(parents=True)
        (solution/'v3001_train.py').write_text('# fixture',encoding='utf-8');write(solution/'model-config.json',{})
        write(solution/'experiment.json',{'solution_id':'v3001','run_id':'new-run','status':'ready','formal_attempts':1,
            'priority':1,'entrypoint':'solutions/v3001/v3001_train.py','config':'solutions/v3001/model-config.json'})
        raw=root/'raw';write(raw/'download-manifest.json',{'status':'complete','files':{
            'pool.parquet':{'sha256':'pool'},'labels.parquet':{'sha256':'new-labels'}}})
        protocol={'protocol_id':'fixture','universe_table':'pool','label_table':'labels'};write(root/'evaluation/protocols/local-v2-ic.json',protocol)
        proto_sha=module.sha256(root/'evaluation/protocols/local-v2-ic.json')
        proof={'status':'complete','covers_required_protocol':True,'identity':{'new_root':str(raw),
            'new_manifest_sha256':module.sha256(raw/'download-manifest.json'),'protocol_sha256':proto_sha}}
        write(root/'comparison.json',proof)
        records={}
        for stage in ('train','holdout'):
            folder=root/'prepared'/stage
            for name,value in (('config.json',{}),('protocol.json',protocol),('data-manifest.json',{'root':str(raw)}),('summary.json',{})):
                write(folder/name,value)
            manifest={'status':'complete',**{key:module.sha256(folder/name) for name,key in
                (('config.json','config_sha256'),('protocol.json','protocol_sha256'),('data-manifest.json','data_manifest_sha256'),('summary.json','summary_sha256'))}}
            write(folder/'manifest.json',manifest)
            records[stage]={'path':str(folder),'manifest_sha256':module.sha256(folder/'manifest.json'),
                'data_manifest_sha256':manifest['data_manifest_sha256'],'summary_sha256':manifest['summary_sha256']}
        binding={'status':'prepared_for_scheduler','raw_root':str(raw),
            'raw_manifest_sha256':module.sha256(raw/'download-manifest.json'),'comparison_path':str(root/'comparison.json'),
            'comparison_sha256':module.sha256(root/'comparison.json'),'protocol_sha256':proto_sha,
            'training_run':records['train']['path'],'holdout_run':records['holdout']['path'],
            'preprocessing':records,'snapshot_id':'new-data'}
        write(root/'binding.json',binding)
        write(root/'experiments/research-policy.json',{'data_refresh':{'raw_snapshot_root':'raw',
            'future_data_binding':'binding.json','block_new_training':True}})
        write(root/'experiments/data-snapshots.json',{'snapshots':[{'id':'new-data',
            'evaluation_sources':{'pool':'pool','labels':'new-labels'}}]})
        (root/'experiments/runs.csv').write_text('run_id,data_snapshot_id\nold,legacy\n',encoding='utf-8-sig')

    def test_new_paths_are_explicit_and_gate_is_not_opened(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(module,'readiness_error',return_value=None):
            root=Path(tmp).resolve();self.fixture(root);value=module.resolve_new_run(root,'v3001')
            self.assertEqual(value['data_root'],str(root/'raw'));self.assertEqual(value['training_run'],str(root/'prepared/train'))
            self.assertFalse(value['dispatch_allowed']);self.assertIsNone(value['resident_cache_argument'])
            self.assertFalse(value['formal_training_performed'])
            self.assertTrue(value['first_cohort_baseline']);self.assertEqual(value['comparable_completed_run_ids'],[])

    def test_existing_run_or_launch_record_cannot_be_rebound(self):
        for marker in ('outputs/runs/new-run/manifest.json','outputs/run-control/new-run/attempt.json'):
            with self.subTest(marker=marker),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();self.fixture(root);write(root/marker,{'keep':'original'})
                with self.assertRaisesRegex(ValueError,'已启动'):module.resolve_new_run(root,'v3001')
                self.assertEqual(module.read_json(root/marker),{'keep':'original'})

    def test_raw_or_preprocessing_mutation_is_rejected(self):
        for relative in ('raw/download-manifest.json','prepared/train/data-manifest.json','prepared/holdout/config.json'):
            with self.subTest(relative=relative),tempfile.TemporaryDirectory() as tmp,patch.object(module,'readiness_error',return_value=None):
                root=Path(tmp).resolve();self.fixture(root);write(root/relative,{'changed':True})
                with self.assertRaises(ValueError):module.resolve_new_run(root,'v3001')


if __name__=='__main__':unittest.main()
