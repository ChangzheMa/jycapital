"""临时合成原始数据上的完整入口、测试恢复、分片材料与隔离推理。"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np
import pandas as pd

from test_v3301 import ROOT, small_config, model_from_parent
import v3301_run as run_api
import v3301_fit as fit
import v3301_verify as completion
import v3201_prepare as prep
import v3201_apply as apply
from v3201_data import regular_times, ProcessedStore
from v3301_text_artifact import read_bundle, decode_state, FORMAT, encode_state


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EntrypointTests(unittest.TestCase):
    def test_full_synthetic_pipeline_test_resume_and_isolated_export(self):
        config = small_config()
        config['training']['epochs'] = 1
        config['backbone_config']['training']['resident_cache'].update(build_batch_rows=256, evaluation_days=1)
        protocol = json.loads((ROOT / 'evaluation/protocols/local-v2-ic.json').read_text(encoding='utf-8'))
        protocol['min_cross_section_size'] = 3
        protocol['splits'] = {s: {'start': a, 'stop': b} for s, a, b in
            [('train', '2020-01-01', '2020-02-01'), ('validation', '2020-02-01', '2020-03-01'), ('test', '2020-03-01', '2020-04-01')]}
        with tempfile.TemporaryDirectory(prefix='v3301-entry-') as temporary:
            root = Path(temporary).resolve()
            data = root / 'data'; data.mkdir()
            days = pd.to_datetime(['2020-01-02', '2020-02-03', '2020-03-02'])
            stocks = [f'S{i}' for i in range(6)]
            minutes = (570, 600, 630, 660, 690, 810, 840, 870)
            keys = pd.DataFrame([(d + pd.Timedelta(minutes=m), s) for d in days for m in minutes for s in stocks], columns=['date', 'instrument'])
            keys.to_parquet(data / 'cpt_jyc_2026_instruments.parquet', index=False)
            rng = np.random.default_rng(161)
            labels = keys.copy(); labels['vwap_return'] = rng.normal(0, .001, len(keys))
            labels.loc[10, 'vwap_return'] = np.nan
            labels.to_parquet(data / 'cpt_jyc_2026_vwap.parquet', index=False)
            for frequency in prep.FREQUENCIES:
                directory = data / ('cpt_jyc_2026_stock_' + frequency); directory.mkdir()
                for day in days:
                    frame = pd.DataFrame([(pd.Timestamp(t), s) for t in regular_times([day], frequency) for s in stocks], columns=['date', 'instrument'])
                    for field in prep.FIELDS:
                        frame[field] = rng.uniform(1, 20, len(frame))
                    frame.to_parquet(directory / (str(day.to_period('M')) + '.parquet'), index=False)
            preprocessing = json.loads((ROOT / 'solutions/v3301/config.json').read_text(encoding='utf-8'))
            preprocessing['processing'].update(threads=1, batch_rows=256)
            training, holdout, parent_run, run = [root / s for s in ('training', 'holdout', 'parent', 'child')]
            parent_run.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                prep.prepare(training, data, preprocessing, protocol)
                apply.apply_holdout(training, holdout, data)
            model, parent = model_from_parent(config)
            torch.save({'config': config['backbone_config'], 'state_dict': parent.state_dict(), 'weight_source': 'ema', 'epoch': 4, 'ema_updates': 1}, parent_run / 'best.pt')
            config['model']['parent_model_sha256'] = run_api.sha256(parent_run / 'best.pt')
            for name, value in [('config.json', config['backbone_config']), ('protocol.json', protocol)]:
                run_api.atomic_json(parent_run / name, value)
            run_api.atomic_json(parent_run / 'manifest.json', {'status': 'complete', 'source': {'files': {
                'solutions/v3201/v3201_model.py': run_api.sha256(ROOT / 'solutions/v3301/v3201_model.py')}}})
            store = ProcessedStore(training, holdout, data, config['backbone_config'], protocol)
            with contextlib.redirect_stdout(io.StringIO()):
                run_api.parent_train.data_manifest(store, parent_run)
            config_path, protocol_path = root / 'config.json', root / 'protocol.json'
            run_api.atomic_json(config_path, config); run_api.atomic_json(protocol_path, protocol)
            arguments = ['--config', str(config_path), '--protocol', str(protocol_path), '--parent-run', str(parent_run),
                         '--training-run', str(training), '--holdout-run', str(holdout), '--data-root', str(data),
                         '--run-dir', str(run), '--training-confirmed', '--reproduction-only', '--device', 'cpu']
            evaluate_bank = run_api.evaluate_bank
            def test_interrupt(bank, cache, *args, **kwargs):
                if cache.manifest['identity']['split'] == 'test':
                    selection = run_api.read_json(run / 'selection.json')
                    self.assertEqual(len(selection['clock_selections']), 8)
                    raise KeyboardInterrupt
                return evaluate_bank(bank, cache, *args, **kwargs)
            with patch.object(run_api, 'environment', lambda: {'packages': {}}), patch.object(run_api, 'record_resources'), contextlib.redirect_stdout(io.StringIO()):
                with patch.object(run_api, 'evaluate_bank', test_interrupt), self.assertRaises(KeyboardInterrupt):
                    run_api.execute(run_api.parser().parse_args(arguments))
                selected_hash = run_api.sha256(run / 'best.pt')
                history = (run / 'history.csv').read_bytes()
                with patch.object(run_api, 'fit_readouts', side_effect=AssertionError('测试恢复不得再次训练')):
                    manifest = run_api.execute(run_api.parser().parse_args(arguments + ['--resume']))
            self.assertEqual(manifest['status'], 'complete')
            self.assertTrue((run / 'learning-curves.png').is_file())
            self.assertEqual(run_api.sha256(run / 'best.pt'), selected_hash)
            self.assertEqual((run / 'history.csv').read_bytes(), history)
            self.assertEqual(run_api.sha256(parent_run / 'best.pt'), config['model']['parent_model_sha256'])
            checked = completion.verify_run(run)
            self.assertTrue(checked['all_eight_selections_replayed'])
            self.assertEqual(checked['splits']['test']['coverage'], 1.)
            for split in ('validation', 'test'):
                frame = pd.read_parquet(run / 'evaluation' / split / 'predictions.parquet')
                self.assertEqual(len(frame), 48)
                self.assertTrue(np.isfinite(frame.factor).all())
                self.assertFalse(frame.duplicated(['date', 'instrument']).any())
            # 临时合成导出不写正式submissions或实验索引。
            exporter = load_script('export_v3301_test', ROOT / 'scripts/export-v3301.py')
            verifier = load_script('verify_v3301_test', ROOT / 'scripts/verify-v3301-submission.py')
            comparison = {'candidate': True, 'config_sha256': run_api.sha256(run / 'config.json'),
                'data_manifest_sha256': run_api.sha256(run / 'data-manifest.json'), 'decision': 'synthetic_check',
                'criterion': {'synthetic': True}, 'metrics': manifest['metrics']}
            run_api.atomic_json(run / 'comparison.json', comparison)
            package = root / 'package'
            with patch.object(exporter, 'ROOT', root), patch.object(verifier, 'ROOT', root):
                exporter.export(run, package)
                report = verifier.verify(run, package)
            self.assertTrue(report['weights_bit_exact'])
            self.assertTrue(report['synthetic_check']['isolated_sharded_model_inference'])
            bundle = read_bundle(package / 'v3301-model.json')
            self.assertTrue(bundle['state_dict_shards'])
            restored = decode_state(bundle)
            chosen = torch.load(run / 'best.pt', weights_only=True)
            for k, value in chosen['state_dict'].items():
                torch.testing.assert_close(value, restored[k], rtol=0, atol=0)

    def test_multiple_parameter_shards_roundtrip(self):
        exporter = load_script('export_v3301_shard_test', ROOT / 'scripts/export-v3301.py')
        state = {f'w{i}': torch.linspace(-1, 1, 70) + i for i in range(5)}
        bundle = {'format': FORMAT, 'state_dict': encode_state(state), 'sources': {}}
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            exporter.write_sharded_bundle(folder, bundle, max_bytes=2000)
            loaded = read_bundle(folder / 'v3301-model.json')
            self.assertGreater(len(loaded['state_dict_shards']), 1)
            for path in folder.iterdir():
                self.assertLessEqual(path.stat().st_size, 2000)
            for k, value in decode_state(loaded).items():
                torch.testing.assert_close(value, state[k], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
