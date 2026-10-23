"""树训练库与便携推理对照，以及冻结历史表示的因果一致性。"""
import copy
import json
from pathlib import Path
import sys
import unittest

import torch
import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
for directory in ('common', 'solutions/v2801'):
    sys.path.insert(0, str(ROOT / directory))
from v2801_trees import TreeEnsemble
from v2801_features import FrozenHistoryEncoder
from v2401_model import MultiFrequencyGRU
from v2401_train import preflight_inputs
from v2401_text_artifact import FORMAT, encode_state, decode_state

torch.set_num_threads(1)


class TreeTests(unittest.TestCase):
    def test_declared_full_tree_budget_and_partitioned_prediction(self):
        rng = np.random.default_rng(2801)
        x = rng.normal(size=(192,7)).astype(np.float32)
        y = np.sin(x[:,0])+.3*x[:,1]
        data = xgb.DMatrix(x,label=y,nthread=1)
        booster = xgb.train({'objective':'reg:squarederror','base_score':0.,'max_depth':6,
                             'eta':.03,'tree_method':'hist','max_bin':64,'nthread':1},data,1200)
        portable = TreeEnsemble(7,1200,6)
        portable.import_booster(booster.save_raw(raw_format='json'))
        expected = booster.predict(data)
        actual = portable(torch.from_numpy(x)).numpy()
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=1e-5)
        parts = np.concatenate([portable(torch.from_numpy(part)).numpy() for part in np.array_split(x,7)])
        np.testing.assert_array_equal(parts,actual)

    def test_native_threshold_missing_margin_and_text_roundtrip(self):
        rng = np.random.default_rng(42)
        values = rng.normal(size=(128, 5)).astype(np.float32)
        values[::7, 2] = np.nan
        labels = (values[:, 0] > .1).astype(np.float32) + .5 * np.nan_to_num(values[:, 2])
        margin = rng.normal(0, .05, len(values)).astype(np.float32)
        matrix = xgb.QuantileDMatrix(values, label=labels, base_margin=margin, max_bin=16, nthread=1)
        params = {'objective':'reg:squarederror','tree_method':'hist','device':'cpu','max_depth':3,
                  'max_bin':16,'eta':.2,'base_score':0.,'nthread':1,'seed':42,'seed_per_iteration':True}
        booster = xgb.train(params, matrix, num_boost_round=9)
        document = json.loads(booster.save_raw(raw_format='json'))
        portable = TreeEnsemble(5, 12, 3, tree_batch_size=4)
        portable.import_booster(document)
        boundary = []
        for tree in document['learner']['gradient_booster']['model']['trees']:
            for index, left in enumerate(tree['left_children']):
                if left < 0:
                    continue
                threshold = np.float32(tree['split_conditions'][index])
                for value in (np.nextafter(threshold, np.float32(-np.inf)), threshold,
                              np.nextafter(threshold, np.float32(np.inf)), np.nan):
                    row = np.zeros(5, np.float32)
                    row[tree['split_indices'][index]] = value
                    boundary.append(row)
        check = np.concatenate((values, np.asarray(boundary)))
        base = np.linspace(-.2,.2,len(check),dtype=np.float32)
        expected = booster.predict(xgb.DMatrix(check,base_margin=base,nthread=1))
        actual = portable(torch.from_numpy(check)).numpy() + base
        np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-6)
        text = json.loads(json.dumps({'format':FORMAT,'state_dict':encode_state(portable.state_dict())}))
        restored = TreeEnsemble(5,12,3,tree_batch_size=4)
        restored.load_state_dict(decode_state(text))
        torch.testing.assert_close(portable(torch.from_numpy(check)),restored(torch.from_numpy(check)),rtol=0,atol=0)
        empty = TreeEnsemble(5,12,3)
        self.assertEqual(empty(torch.from_numpy(check)).count_nonzero().item(),0)

    def test_scalar_tree_contract_and_truncation(self):
        x = np.arange(80,dtype=np.float32).reshape(40,2)
        matrix = xgb.DMatrix(x,label=np.sin(x[:,0]),nthread=1)
        booster = xgb.train({'objective':'reg:squarederror','base_score':0.,'max_depth':2,'nthread':1},matrix,5)
        limited = booster[:3]
        portable = TreeEnsemble(2,5,2)
        document = json.loads(limited.save_raw(raw_format='json'))
        portable.import_booster(document)
        np.testing.assert_allclose(portable(torch.from_numpy(x)).numpy(),limited.predict(matrix),atol=1e-7,rtol=1e-6)
        wrong = copy.deepcopy(document)
        wrong['learner']['learner_model_param']['base_score'] = '[1E0]'
        with self.assertRaises(ValueError):portable.import_booster(wrong)
        wrong = copy.deepcopy(document)
        wrong['learner']['gradient_booster']['model']['trees'][0]['left_children'][0] = 0
        with self.assertRaises(ValueError):portable.import_booster(wrong)


class FeatureTests(unittest.TestCase):
    def test_frozen_parent_shared_single_and_future_boundaries(self):
        config = json.loads((ROOT/'solutions/v2801/parent-model-config.json').read_text(encoding='utf-8'))
        torch.manual_seed(42)
        parent = MultiFrequencyGRU(config).eval()
        encoder = FrozenHistoryEncoder(config)
        encoder.load_state_dict(parent.state_dict())
        self.assertEqual(encoder.feature_dim,624)
        self.assertFalse(encoder.training)
        windows, latest = preflight_inputs(config,2,torch.device('cpu'))
        windows = {f:x.requires_grad_(True) for f,x in windows.items()}
        features, baseline = encoder.group_features(windows,latest)
        with torch.no_grad():
            _, inference_baseline = encoder.group_features(windows,latest)
            torch.testing.assert_close(inference_baseline,parent.forward_group(windows,latest)[...,0],rtol=0,atol=0)
        self.assertEqual(features.shape,(2,8,624))
        for number in range(8):
            clock = torch.eye(8)[[number,number]]
            with torch.no_grad():
                single, primary = encoder.single_features(windows,clock)
                torch.testing.assert_close(features[:,number],single,atol=2e-6,rtol=1e-5)
                torch.testing.assert_close(baseline[:,number],primary,atol=2e-7,rtol=1e-5)
            gradients = torch.autograd.grad(features[:,number].square().sum(),tuple(windows.values()),retain_graph=True)
            for f, gradient in zip(windows,gradients):
                bars = 240 // int(f[3:-1])
                cutoff = config['data']['trading_day_windows'][f]['previous_days'] * bars + number * (bars // 8)
                self.assertEqual(gradient[:,:,cutoff:].count_nonzero().item(),0)
        self.assertIsNone(encoder._captured_features)


if __name__ == '__main__':
    unittest.main()
