"""用逐个预测日回溯的独立实现检查历史需求范围和缺历史分母。"""
import importlib.util
from pathlib import Path
import unittest

import torch
import numpy as np

path = Path(__file__).resolve().parents[1] / "scripts/estimate-long-context.py"
spec = importlib.util.spec_from_file_location("long_context_estimate", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EstimateTests(unittest.TestCase):
    def test_matches_explicit_past_windows_with_changing_stock_membership(self):
        rng = np.random.default_rng(1701)
        for size in (1, 5, 18):
            present = rng.random((size, 7)) < .4
            for horizon in (1, 3, 20):
                expected = np.zeros_like(present)
                for day, stock in zip(*np.nonzero(present)):
                    expected[max(0, day - horizon):day, stock] = True
                np.testing.assert_array_equal(module.needed_previous_days(present, horizon), expected)

    def test_excludes_today_and_counts_short_history_without_dropping_groups(self):
        present = np.array([[1, 0], [1, 1], [0, 1], [1, 1]], bool)
        expected = np.array([[1, 1], [1, 1], [1, 1], [0, 0]], bool)
        np.testing.assert_array_equal(module.needed_previous_days(present, 2), expected)
        record = module.window_estimate(present, 2)
        self.assertEqual(record["blocks_including_zero"], 7)
        self.assertEqual(record["training_groups_with_full_calendar_history"], 3)
        self.assertEqual(record["training_groups_with_shorter_calendar_history"], 3)
        self.assertEqual(record["full_history_group_fraction"], .5)
        self.assertEqual(record["fp32_values_bytes"], 7 * 8 * 72 * 4)

    def test_rejects_invalid_horizons_and_accepts_empty_history(self):
        for value in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                module.needed_previous_days(np.zeros((2, 3)), value)
        self.assertEqual(module.needed_previous_days(np.zeros((0, 3)), 2).shape, (0, 3))


if __name__ == "__main__":
    unittest.main()
