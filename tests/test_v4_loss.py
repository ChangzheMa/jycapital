"""真正的 IC 损失、梯度、不变性和截面权重，与既有评估器对照。"""

import sys
from pathlib import Path
import unittest
import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solutions/v4"))
sys.path.insert(0, str(ROOT / "evaluation"))
from v4_loss import pearson_ic_loss, mean_section_loss, ConstantPredictionError
from baseline_evaluation import evaluate


class LossTests(unittest.TestCase):
    def test_values_and_gradients_match_normalized_mse_and_finite_difference(self):
        y = torch.tensor([-.02, -.003, .004, .011, .05], dtype=torch.float64)
        x = torch.tensor([-.01, .005, .002, .009, .03], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda p: pearson_ic_loss(p, y, 2).loss, (x,)))
        result = pearson_ic_loss(x, y, 2)
        z = lambda v: (v-v.mean()) / v.std(correction=0)
        oracle = ((z(x)-z(y))**2).mean()/2
        torch.testing.assert_close(result.loss, oracle, atol=1e-12, rtol=0)
        first = torch.autograd.grad(result.loss, x, retain_graph=True)[0]
        second = torch.autograd.grad(oracle, x)[0]
        torch.testing.assert_close(first, second, atol=1e-10, rtol=0)

    def test_affine_and_permutation_invariance_without_epsilon_scale_bias(self):
        y = torch.tensor([1., 2., 3., 4., 5.], dtype=torch.float64)
        x = torch.tensor([2., 1., 4., 3., 7.], dtype=torch.float64)
        base = pearson_ic_loss(x, y, 2).loss
        for other in (x*.5+.001, x*1e-100):
            torch.testing.assert_close(base, pearson_ic_loss(other, y, 2).loss, atol=1e-12, rtol=0)
        indices = torch.tensor([3, 1, 4, 0, 2])
        torch.testing.assert_close(base, pearson_ic_loss(x[indices], y[indices], 2).loss)
        self.assertAlmostEqual(pearson_ic_loss(-y, y, 2).loss.item(), 2.0)
        self.assertAlmostEqual(pearson_ic_loss(y, y, 2).loss.item(), 0.0)

    def test_equal_sections_match_evaluator_and_not_global_correlation(self):
        frames, results = [], []
        for stamp, size, direction, offset in (("2024-01-02 10:00", 35, 1, 1000),
                                               ("2024-01-02 10:30", 67, -1, -1000)):
            y = np.arange(size, dtype=float)
            x = direction*y + offset
            results.append(pearson_ic_loss(torch.tensor(x), torch.tensor(y)))
            frames.append(pd.DataFrame({"date": pd.Timestamp(stamp), "instrument": [f"S{i}" for i in range(size)],
                                        "factor": x, "label": y}))
        frame = pd.concat(frames, ignore_index=True)
        protocol = {"protocol_id": "test", "label_column": "label", "min_cross_section_size": 30,
                    "max_missing_fraction_per_cross_section": .4}
        stats = evaluate(frame[["date", "instrument", "factor"]],
                         frame[["date", "instrument", "label"]], protocol)[0]
        loss, ic = mean_section_loss(results)
        self.assertAlmostEqual(loss.item(), 1-stats["ic_mean"], places=12)
        self.assertAlmostEqual(ic.item(), 0., places=12)
        self.assertGreater(abs(np.corrcoef(frame.factor, frame.label)[0, 1]-ic.item()), .1)

    def test_missing_labels_and_degenerate_cases_are_explicit(self):
        x = torch.arange(5., dtype=torch.float64)
        y = x.clone(); y[0] = float("nan")
        self.assertEqual(pearson_ic_loss(x, y, 2).rows, 4)
        self.assertEqual(pearson_ic_loss(x, y).reason, "insufficient_labels")
        self.assertEqual(pearson_ic_loss(x, torch.ones(5), 2).reason, "constant_labels")
        with self.assertRaises(ConstantPredictionError):
            pearson_ic_loss(torch.ones(5), x, 2)
        bad = x.clone(); bad[0] = float("inf")
        with self.assertRaises(FloatingPointError):
            pearson_ic_loss(bad, y, 2)
        self.assertEqual(mean_section_loss([pearson_ic_loss(x, y)]), (None, None))


if __name__ == "__main__":
    unittest.main()
