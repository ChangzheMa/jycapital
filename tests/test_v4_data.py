"""完整分钟截面无遗漏、稳定抽样及与权重无关的退化判定。"""

import sys
from pathlib import Path
import unittest
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "solutions/v4"))
from v4_data import section_indices, structural_reason


class SectionTests(unittest.TestCase):
    def test_full_timestamps_and_variable_stock_counts_without_tail_drop(self):
        keys = pd.DataFrame({"date": pd.to_datetime(["2024-01-02 10:00"]*3+["2024-01-02 10:30"]*5),
                             "instrument": ["C", "A", "B", "E", "B", "C", "A", "D"]})
        groups = section_indices(keys, 42)
        self.assertEqual(sorted(map(len, groups)), [3, 5])
        self.assertEqual(sorted(np.concatenate(groups).tolist()), list(range(8)))
        for group in groups:
            self.assertEqual(keys.iloc[group].date.nunique(), 1)
            self.assertEqual(keys.iloc[group].instrument.tolist(), sorted(keys.iloc[group].instrument))
        for first, second in zip(groups, section_indices(keys, 42)):
            np.testing.assert_array_equal(first, second)

    def test_structural_policy_uses_finite_label_inputs_not_model_predictions(self):
        keys = pd.DataFrame({"label": [1., 2., 3., np.nan]})
        windows = {"bar1m": np.ones((4, 77, 4), np.float32)}
        clock = np.ones((4, 2), np.float32)
        windows["bar1m"][-1] = 2
        self.assertEqual(structural_reason(keys, windows, clock, "label", 2), ("identical_inputs", 3))
        windows["bar1m"][0, 0, 0] = 2
        self.assertEqual(structural_reason(keys, windows, clock, "label", 2), (None, 3))
        self.assertEqual(structural_reason(keys, windows, clock, "label", 4)[0], "insufficient_labels")
        keys["label"] = [1., 1., 1., np.nan]
        self.assertEqual(structural_reason(keys, windows, clock, "label", 2)[0], "constant_labels")


if __name__ == "__main__":
    unittest.main()
