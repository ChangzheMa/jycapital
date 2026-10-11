"""验证因果时间网格、缺失掩码、TCN 因果性和训练梯度。"""

import json
from pathlib import Path
import sys
import unittest

import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solutions/v2"))
from v2_data import WindowBlock, regular_times, FREQUENCIES
from v2_model import CausalBlock, MultiFrequencyTCN


class WindowsTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "solutions/v2/model-config.json").read_text(encoding="utf-8"))

    def make_block(self, dates, missing=None):
        keys = pd.DataFrame({"date": pd.to_datetime(dates), "instrument": "A"})
        grids = {}
        for f in FREQUENCIES:
            times = regular_times(pd.to_datetime(["2024-01-02", "2024-01-03"]), f)
            values = np.ones((len(times), 2, 72), np.float32)
            values[:, 0, 0] = np.arange(len(times))
            if missing is not None:
                values[times == pd.Timestamp(missing).value, 0] = np.nan
            grids[f] = times, values
        return WindowBlock(keys, grids, ["A", "B"], self.config)

    def test_completed_bar_no_future_and_no_other_stock(self):
        block = self.make_block(["2024-01-02 10:03"])
        before, _ = block.batch([0])
        self.assertEqual(before["bar5m"][0, 0, -1], 5)
        for times, values in block.grids.values():
            values[times > block.times[0]] = 1e9
            values[:, 1] = -1e9
        after, _ = block.batch([0])
        for f in FREQUENCIES:
            np.testing.assert_array_equal(before[f], after[f])

    def test_missing_slot_is_not_collapsed_and_zero_is_observed(self):
        block = self.make_block(["2024-01-02 09:34"], "2024-01-02 09:33")
        x, _ = block.batch([0])
        one = x["bar1m"][0]
        self.assertEqual(one[72, -2], 0)
        self.assertEqual(one[72, -1], 1)
        self.assertEqual(one[0, -4], 0)
        self.assertEqual(one[72, -4], 1)
        self.assertTrue((one[:, :-4] == 0).all())

    def test_lunch_overnight_and_frequency_alignment(self):
        block = self.make_block(["2024-01-02 13:02", "2024-01-03 09:31"])
        x, _ = block.batch([0, 1])
        self.assertGreater(x["bar1m"][0, 146, -3], x["bar1m"][0, 146, -2])
        self.assertGreater(x["bar1m"][1, 146, -2], x["bar1m"][1, 146, -1])
        self.assertEqual(x["bar30m"][0, 0, -1], 3)

    def test_no_history_is_finite_zero_input(self):
        block = self.make_block(["2024-01-02 09:20"])
        x, clock = block.batch([0])
        self.assertTrue(all((v == 0).all() for v in x.values()))
        model = MultiFrequencyTCN(self.config).eval()
        with torch.inference_mode():
            result = model({f: torch.from_numpy(v) for f, v in x.items()}, torch.from_numpy(clock))
        self.assertTrue(torch.isfinite(result).all())

    def test_causal_block_cannot_see_future(self):
        torch.manual_seed(42)
        block = CausalBlock(8, 3, 2, 0).eval()
        x = torch.randn(2, 8, 16)
        valid = torch.ones(2, 1, 16)
        before = block(x, valid)
        x[:, :, 8:] += 100
        after = block(x, valid)
        torch.testing.assert_close(before[:, :, :8], after[:, :, :8])

    def test_backward_and_checkpoint_round_trip(self):
        block = self.make_block(["2024-01-02 10:03", "2024-01-03 09:31"])
        x, clock = block.batch([0, 1])
        tensors = {f: torch.from_numpy(v) for f, v in x.items()}
        clock = torch.from_numpy(clock)
        model = MultiFrequencyTCN(self.config).eval()
        result = model(tensors, clock)
        result.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        restored = MultiFrequencyTCN(self.config).eval()
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(result, restored(tensors, clock))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
