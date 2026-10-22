"""冻结组合的截面语义、成员对齐及共享推理检查。"""

from pathlib import Path
import json
import sys
import unittest

import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for version in ('v1002', 'v1101', 'v1401'):
    sys.path.insert(0, str(ROOT / 'solutions' / version))
from v1401_combine import combine_scores, align_components
from v1401_model import FrozenEnsemble
from v1401_fast import predict_day
from v1002_data import WindowBlock, regular_times, FREQUENCIES

torch.set_num_threads(1)


class FrozenEnsembleTests(unittest.TestCase):
    def test_complete_sections_are_independent_of_rows_and_positive_affine_scale(self):
        keys = pd.DataFrame([(pd.Timestamp(day), stock) for day in ('2024-01-01','2024-01-02')
                             for stock in ('A','B','C')], columns=['date','instrument'])
        scores = np.array([[1.,3.],[2.,1.],[4.,2.],[4.,8.],[5.,5.],[8.,6.]])
        actual = combine_scores(keys, scores)
        expected = np.concatenate([((part - part.mean(0)) / part.std(0)).mean(1) for part in np.split(scores, 2)])
        np.testing.assert_allclose(actual.factor, expected, rtol=1e-14, atol=1e-14)
        changed = scores.copy()
        changed[:3] = changed[:3] * [100., .1] + [200., -3.]
        np.testing.assert_allclose(combine_scores(keys, changed).factor, actual.factor, rtol=1e-14, atol=1e-14)
        order = [4, 2, 0, 5, 1, 3]
        reordered = combine_scores(keys.iloc[order], scores[order]).sort_values(['date','instrument'])
        np.testing.assert_allclose(reordered.factor, actual.factor, rtol=1e-14, atol=1e-14)

    def test_constant_components_and_missing_or_duplicate_member_keys(self):
        keys = pd.DataFrame({'date': pd.to_datetime(['2024-01-01'] * 3), 'instrument': ['A','B','C']})
        scores = np.array([[2.,1.],[2.,2.],[2.,3.]])
        actual = combine_scores(keys, scores)
        np.testing.assert_allclose(actual.factor, (scores[:,1] - 2) / np.std(scores[:,1]) / 2)
        np.testing.assert_array_equal(combine_scores(keys, np.ones((3,2))).factor, np.zeros(3))
        left, right = keys.assign(factor=scores[:,0]), keys.assign(factor=scores[:,1])
        aligned, values = align_components(left, right.iloc[::-1])
        np.testing.assert_allclose(combine_scores(aligned, values).factor, actual.factor)
        for invalid in (right.iloc[:2], pd.concat([right, right.iloc[:1]])):
            with self.assertRaises(ValueError):
                align_components(left, invalid)
        with self.assertRaises(ValueError):
            combine_scores(keys, np.full((3,2), np.nan))

    def test_shared_prediction_aggregates_after_all_stock_batches_and_matches_reference(self):
        configs = [json.loads((ROOT / f'solutions/{version}/model-config.json').read_text(encoding='utf-8'))
                   for version in ('v1002','v1101')]
        for config in configs:
            config['model'].update(hidden_dim=8, fusion_hidden_dim=16)
        configs[0]['model'].update(expert_hidden_dim=8, router_hidden_dim=8)
        model = FrozenEnsemble(configs).eval()
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        days = pd.bdate_range('2024-01-02', periods=22)
        stocks = pd.Index(['A','B','C','D','E'])
        rng = np.random.default_rng(1401)
        grids = {f: (regular_times(days, f), rng.normal(size=(len(regular_times(days, f)), 5, 72)).astype(np.float32))
                 for f in FREQUENCIES}
        keys = pd.DataFrame([(days[-1]+pd.Timedelta(minutes=m), stock) for m in (570,600,630,660,690,810,840,870)
                             for stock in stocks], columns=['date','instrument']).drop(index=[1,7,18]).sample(frac=1,random_state=5).reset_index(drop=True)
        block = WindowBlock(keys, grids, stocks, configs[0], days.to_numpy())
        windows, clocks = block.batch(np.arange(len(keys)))
        with torch.inference_mode():
            scores = model.component_scores({f: torch.from_numpy(x) for f,x in windows.items()}, torch.from_numpy(clocks)).numpy()
        expected = combine_scores(keys, scores).factor.to_numpy()
        for batch in (1,2,128):
            actual = predict_day(model, block, torch.device('cpu'), stock_batch_size=batch)
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-4)
        self.assertEqual(actual.dtype, np.float64)
        self.assertTrue(np.isfinite(actual).all())


if __name__ == '__main__':
    unittest.main()
