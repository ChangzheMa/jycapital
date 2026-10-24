"""新快照预处理编排只绑定新路径，不触发模型训练或改写旧运行。"""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prepare_refresh',ROOT/'scripts/prepare-refreshed-data.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class PrepareRefreshTests(unittest.TestCase):
    def test_execution_limits_preserve_data_and_normalization(self):
        original=module.read_json(ROOT/'solutions/v3001/config.json');saved=copy.deepcopy(original)
        result=module.processing_config(original,2,'1GB',32768)
        self.assertEqual(original,saved)
        self.assertEqual(result['data'],original['data']);self.assertEqual(result['normalization'],original['normalization'])
        delta={k for k in result['processing'] if result['processing'][k]!=original['processing'][k]}
        self.assertEqual(delta,{'threads','memory_limit','batch_rows'})

    def test_commands_only_prepare_new_training_and_holdout_data(self):
        paths=module.commands(Path('python'),Path('frozen'), 'v3001',Path('new-raw'),Path('new-prepared'),Path('config.json'))
        self.assertEqual([p[0] for p in paths],['train','holdout'])
        for name,folder,command in paths:
            self.assertNotIn('--training-confirmed',command)
            self.assertFalse(any(str(value).endswith('_train.py') for value in command))
            self.assertEqual(command[command.index('--data-root')+1],'new-raw')
            self.assertEqual(Path(command[command.index('--run-dir')+1]),Path('new-prepared')/name)
        second=paths[1][2]
        self.assertEqual(Path(second[second.index('--training-run')+1]),Path('new-prepared/train'))

    def test_complete_artifact_must_bind_exact_new_file_and_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';prepared=root/'prepared';raw.mkdir();prepared.mkdir()
            expected={'bar.parquet':{'bytes':5,'sha256':'market'},'pool.parquet':{'bytes':4,'sha256':'pool'}}
            module.atomic_json(raw/'download-manifest.json',{'files':expected})
            data={'root':str(raw),'files':{'bar.parquet':expected['bar.parquet']},'universe':{'path':'pool.parquet','sha256':'pool'}}
            module.atomic_json(prepared/'data-manifest.json',data);module.atomic_json(prepared/'summary.json',{'complete':True})
            manifest={'status':'complete','summary_sha256':module.sha256(prepared/'summary.json'),'data_manifest_sha256':module.sha256(prepared/'data-manifest.json')}
            module.atomic_json(prepared/'manifest.json',manifest)
            self.assertEqual(module.verify_complete(prepared,raw)['path'],str(prepared))
            data['files']['bar.parquet']['sha256']='old-content'
            module.atomic_json(prepared/'data-manifest.json',data)
            manifest['data_manifest_sha256']=module.sha256(prepared/'data-manifest.json');module.atomic_json(prepared/'manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'不属于已核对的新快照'):module.verify_complete(prepared,raw)


if __name__=='__main__':unittest.main()
