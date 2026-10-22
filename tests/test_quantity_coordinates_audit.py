"""数量关系计数、训练期合并统计及价格编码的可手算检查。"""
import importlib.util
from pathlib import Path
import unittest

import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("quantity_coordinate_audit", ROOT / "scripts/audit-quantity-coordinates.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class QuantityCoordinateAuditTests(unittest.TestCase):
    def test_group_moments_match_all_finite_training_observations(self):
        values = np.arange(60, dtype=np.float64).reshape(3, 20) / 10
        values[0, 2] = np.nan
        count = np.isfinite(values).sum(axis=0)
        mean = np.nanmean(values, axis=0)
        statistics = {"fields": audit.FIELDS, "global": {"count": count.tolist(), "mean": mean.tolist(),
            "m2": np.nansum((values - mean) ** 2, axis=0).tolist()}}
        actual = audit.pooled_group_parameters(statistics, audit.GROUPS["book_volume"])
        expected = values[:, :10][np.isfinite(values[:, :10])]
        self.assertAlmostEqual(actual["center"], expected.mean())
        self.assertAlmostEqual(actual["scale"], expected.std())
        self.assertEqual(actual["observations"], len(expected))

    def test_both_quantity_directions_and_zero_ties_are_counted(self):
        raw = np.zeros((3, 10))
        raw[:, 0], raw[:, 5] = [20, 10, 0], [10, 20, 0]
        wrong = raw.copy()
        wrong[:, 0], wrong[:, 5] = [1, 1, 1], [2, 0, 2]
        counts = audit.pair_counts(raw, wrong, [(0, 5)])
        self.assertEqual(counts["direction_reversed"], 2)
        self.assertEqual(counts["raw_ties_split"], 1)
        shared = audit.common_quantity_view(raw, {"center": 2., "scale": 3.})
        counts = audit.pair_counts(raw, audit.rounded(shared), [(0, 5)])
        self.assertEqual(counts["direction_reversed"], 0)
        self.assertEqual(counts["raw_ties_split"], 0)

    def test_float32_price_anchor_preserves_small_moves_and_rejects_future_anchor(self):
        date = pd.to_datetime(["2020-06-02 09:31", "2020-06-02 09:32", "2020-06-02 09:33"])
        frame = pd.DataFrame({"date": date, "day": date.normalize(), "instrument": "S0",
            "close": [100., 100.01, 100.02], "z_close": [1., 1.00001, 1.00002],
            "anchor": 99., "anchor_day": pd.Timestamp("2020-06-01")})
        result = audit.temporal_close_counts(frame, "bar1m")
        self.assertEqual(result["nonzero_close_changes"], 2)
        self.assertEqual(result["z_fp16_erased"], 2)
        self.assertEqual(result["anchor_fp32_erased"], 0)
        self.assertEqual(result["global_anchor_fp32_erased"], 0)
        frame["anchor_day"] = pd.Timestamp("2020-06-02")
        with self.assertRaises(ValueError):
            audit.temporal_close_counts(frame, "bar1m")


if __name__ == "__main__":
    unittest.main()
