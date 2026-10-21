"""共享股票日编码与原冻结计算的数值、因果及不完整请求对照。"""

import copy
import json
from pathlib import Path
import sys
import unittest

import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solutions/v501"))
from v501_model import MultiFrequencyGRU
from v501_fast import predict_all_clocks, predict_day
from v501_data import WindowBlock, regular_times

CONFIG = json.loads((ROOT / "solutions/v501/model-config.json").read_text(encoding="utf-8"))


class FastInferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(72)

    def test_default_model_all_clocks_match_original_and_ignore_future(self):
        model = MultiFrequencyGRU(CONFIG).eval()
        windows = {f: torch.randn(3, 75, length) for f, length in CONFIG["data"]["window_bars"].items()}
        for x in windows.values():
            x[:, 72:] = 1
            x[0] = 0
        latest = torch.eye(8)[[7, 7, 7]]
        with torch.inference_mode():
            shared = predict_all_clocks(model, windows, latest)
            expected = torch.stack([model(windows, torch.eye(8)[[i] * 3]) for i in range(8)], dim=1)
            torch.testing.assert_close(shared, expected, rtol=1e-4, atol=2e-7)
            for i in range(8):
                changed = {f: x.clone() for f, x in windows.items()}
                changed["bar1m"][:, :, 240 + i * 30:] = 1e6
                actual = predict_all_clocks(model, changed, latest)[:, i]
                torch.testing.assert_close(actual, shared[:, i], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "eval"):
            predict_all_clocks(model.train(), windows, latest)

    def test_shuffled_dynamic_pool_partial_requests_and_fp16_repairs(self):
        config = copy.deepcopy(CONFIG)
        config["model"].update(hidden_dim=8, fusion_hidden_dim=16)
        model = MultiFrequencyGRU(config).eval()
        days = pd.bdate_range(end="2024-01-31", periods=22)
        stocks = pd.Index(["A", "B", "C", "D"])
        rng = np.random.default_rng(27)
        grids = {}
        for frequency in config["data"]["window_bars"]:
            times = regular_times(days, frequency)
            raw = rng.normal(size=(len(times), len(stocks), 72)).astype(np.float32)
            raw[rng.random(raw.shape) < 0.03] = np.nan
            raw[:, 0] = np.nan
            raw[-100:, 1, 0] = 1e6
            grids[frequency] = times, raw
        minutes = (570, 600, 630, 660, 690, 810, 840, 870)
        keys = pd.DataFrame([(days[-1] + pd.Timedelta(minutes=m), s) for i, m in enumerate(minutes)
                             for s in stocks if not (s == "D" and i > 0) and not (s == "C" and i == 7)],
                            columns=["date", "instrument"]).sample(frac=1, random_state=31).reset_index(drop=True)
        block = WindowBlock(keys, grids, stocks, config, days)
        windows, clock = block.batch(np.arange(len(keys)))
        with torch.inference_mode():
            expected = model({f: torch.from_numpy(x) for f, x in windows.items()}, torch.from_numpy(clock)).numpy()
        for batch in (1, 3, 128):
            result = predict_day(model, block, torch.device("cpu"), stock_batch_size=batch)
            np.testing.assert_allclose(result, expected, rtol=1e-4, atol=2e-7)
        for timestamp in keys.date.unique():
            part = keys.loc[keys.date == timestamp].reset_index(drop=True)
            split = WindowBlock(part, grids, stocks, config, days)
            result = predict_day(model, split, torch.device("cpu"), stock_batch_size=3)
            np.testing.assert_allclose(result, expected[keys.date == timestamp], rtol=1e-4, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
