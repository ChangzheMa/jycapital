"""关键数据因果性、评估口径及权重导出检查。"""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "solutions" / "v1"), str(ROOT / "evaluation")]

try:
    import torch
except ImportError:
    torch = None

import numpy as np
import pandas as pd

from baseline_evaluation import evaluate
from v1_data import build_windows, platform_interval

CONFIG = json.loads((ROOT / "solutions/v1/config.json").read_text(encoding="utf-8"))
PROTOCOL = json.loads((ROOT / "evaluation/protocols/local-v1.json").read_text(encoding="utf-8"))


class CausalityTests(unittest.TestCase):
    def setUp(self):
        self.bars = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02 09:35", "2024-01-02 09:40", "2024-01-02 09:45"]),
            "instrument": ["A"] * 3, "open": [9., 10., 11.], "high": [10., 11., 12.],
            "low": [8., 9., 10.], "close": [9.5, 10.5, 11.5], "volume": [100., 200., 300.], "amount": [950., 2100., 3450.],
        })
        self.keys = pd.DataFrame({"date": pd.to_datetime(["2024-01-02 09:40", "2024-01-02 09:40"]), "instrument": ["A", "B"]})

    def test_future_changes_do_not_change_input(self):
        before, _ = build_windows(self.bars, self.keys, CONFIG)
        changed = self.bars.copy()
        changed.loc[2, CONFIG["data"]["fields"]] = 1e10
        after, quality = build_windows(changed, self.keys, CONFIG)
        np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(after[1], 0)
        self.assertEqual(quality["no_history_samples"], 1)
        # 恰好等于预测时刻的已结束 Bar 应进入最后一个位置。
        self.assertAlmostEqual(after[0].reshape(12, 6)[-1, 3], 0.0)
        self.assertAlmostEqual(after[0].reshape(12, 6)[-1, 4], 1 / 3, places=6)

    def test_auction_and_expired_history_are_excluded(self):
        bars = self.bars.copy()
        bars.loc[0, "date"] = pd.Timestamp("2024-01-02 09:25")
        keys = self.keys.copy()
        keys["date"] = pd.Timestamp("2024-01-02 09:30")
        x, _ = build_windows(bars, keys, CONFIG)
        np.testing.assert_array_equal(x, 0)
        keys["date"] = pd.Timestamp("2024-02-15 09:30")
        x, _ = build_windows(bars, keys, CONFIG)
        np.testing.assert_array_equal(x, 0)

    def test_missing_and_negative_values_are_finite(self):
        self.bars.loc[1, ["open", "close", "volume"]] = [np.nan, -1, -100]
        x, _ = build_windows(self.bars, self.keys, CONFIG)
        self.assertTrue(np.isfinite(x).all())

    def test_platform_end_date_and_timezone(self):
        start, stop = platform_interval("2024-01-02", "2024-01-02")
        self.assertEqual(stop, pd.Timestamp("2024-01-03"))
        start, stop = platform_interval("2024-01-02T02:00:00Z", "2024-01-02T02:30:00Z")
        self.assertEqual(start, pd.Timestamp("2024-01-02 10:00"))
        self.assertEqual(stop, pd.Timestamp("2024-01-02 10:31"))


class EvaluationTests(unittest.TestCase):
    def test_known_correlations_coverage_and_invalid_keys(self):
        protocol = copy.deepcopy(PROTOCOL)
        protocol["min_cross_section_size"] = 3
        keys = pd.DataFrame({"date": pd.to_datetime(["2024-07-01 10:00"] * 4 + ["2024-07-01 10:30"] * 4),
                             "instrument": list("ABCD") * 2, "vwap_return": [1., 2., 3., 4.] * 2})
        prediction = keys[["date", "instrument"]].copy()
        prediction["factor"] = [1., 2., 3., 4., 4., 3., 2., 1.]
        stats, sections, _ = evaluate(prediction, keys, protocol)
        self.assertAlmostEqual(stats["rank_ic_mean"], 0)
        np.testing.assert_allclose(sections["ic"], [1, -1])
        missing = prediction.iloc[:-2]
        stats, _, _ = evaluate(missing, keys, protocol)
        self.assertEqual(stats["coverage"], .75)
        self.assertEqual(stats["coverage_violations"], 1)
        with self.assertRaises(ValueError):
            evaluate(pd.concat([prediction, prediction.iloc[:1]]), keys, protocol)
        bad = prediction.copy()
        bad.loc[0, "instrument"] = "Z"
        with self.assertRaises(ValueError):
            evaluate(bad, keys, protocol)

    def test_constant_predictions_are_reported_not_zero_ic(self):
        protocol = copy.deepcopy(PROTOCOL)
        protocol["min_cross_section_size"] = 2
        keys = pd.DataFrame({"date": pd.to_datetime(["2024-07-01 10:00"] * 2), "instrument": ["A", "B"], "vwap_return": [1., 2.]})
        prediction = keys[["date", "instrument"]].assign(factor=0.)
        stats, _, _ = evaluate(prediction, keys, protocol)
        self.assertIsNone(stats["rank_ic_mean"])
        self.assertEqual(stats["rank_ic_undefined_cross_sections"], 1)


class ModelTests(unittest.TestCase):
    def test_numpy_export_matches_torch(self):
        if torch is None:
            self.skipTest("PyTorch 尚未安装")
        from v1_model import NumpyMLP, create_model, export_model
        torch.manual_seed(42)
        model = create_model(CONFIG).eval()
        x = np.random.default_rng(42).normal(size=(37, 72)).astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.json"
            export_model(model, CONFIG, path)
            result = NumpyMLP(path).predict(x)
        with torch.inference_mode():
            expected = model(torch.from_numpy(x)).squeeze(1).numpy() / 100
        np.testing.assert_allclose(result, expected, atol=1e-8, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
