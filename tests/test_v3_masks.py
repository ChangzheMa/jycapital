"""v3 组完整标志、数值保留、因果窗口与未确认训练的保护。"""

import contextlib
import copy
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solutions/v2"))
sys.path.insert(0, str(ROOT / "solutions/v3"))
from v2_data import WindowBlock as V2WindowBlock
from v2_preprocessing import log_values as v2_log_values
from v3_data import WindowBlock, regular_times, FREQUENCIES
from v3_model import MultiFrequencyTCN
from v3_preprocessing import FIELDS, PRICE_FIELDS, log_values
from v3_schema import (INPUT_CHANNELS, NON_BOOK_VALUE_INDICES,
                       BOOK_VALUE_INDICES, BAR_CLOCK_SLICE)


CONFIG = json.loads((ROOT / "solutions/v3/model-config.json").read_text(encoding="utf-8"))


def grids_for(stocks, empty=False):
    grids = {}
    for frequency in FREQUENCIES:
        times = np.array([], np.int64) if empty else regular_times(
            pd.to_datetime(["2024-01-02", "2024-01-03"]), frequency)
        values = np.ones((len(times), len(stocks), 72), np.float32)
        grids[frequency] = times, values
    return grids


def keys_for(stocks, stamp="2024-01-03 14:00"):
    return pd.DataFrame({"date": pd.Timestamp(stamp), "instrument": stocks})


class GroupMaskTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_each_of_72_channels_breaks_only_its_group(self):
        stocks = [f"S{i:02}" for i in range(72)]
        keys = keys_for(stocks)
        grids = grids_for(stocks)
        for times, values in grids.values():
            position = np.searchsorted(times, keys.date.iloc[0].value)
            values[position, np.arange(72), np.arange(72)] = np.nan
        windows, _ = WindowBlock(keys, grids, stocks, CONFIG).batch(np.arange(72))
        for frequency in FREQUENCIES:
            x = windows[frequency]
            self.assertEqual(x.shape, (72, INPUT_CHANNELS, CONFIG["data"]["window_bars"][frequency]))
            expected = np.ones((72, 2), np.float32)
            expected[list(NON_BOOK_VALUE_INDICES), 0] = 0
            expected[list(BOOK_VALUE_INDICES), 1] = 0
            np.testing.assert_array_equal(x[:, 72:74, -1], expected)
            np.testing.assert_array_equal(x[np.arange(72), np.arange(72), -1], 0)
            self.assertTrue((np.square(x[:, BAR_CLOCK_SLICE, -1]).sum(axis=1) > 0).all())

    def test_zero_is_valid_and_partial_values_are_not_erased(self):
        stocks = ["A"]
        keys = keys_for(stocks)
        grids = grids_for(stocks)
        for times, values in grids.values():
            position = np.searchsorted(times, keys.date.iloc[0].value)
            values[position, 0, 0] = 0
            values[position, 0, 3] = 4.25
            values[position, 0, 7] = np.nan
            values[position, 0, 39] = np.nan
        windows, _ = WindowBlock(keys, grids, stocks, CONFIG).batch([0])
        for x in windows.values():
            np.testing.assert_array_equal(x[0, 72:74, -1], [0, 0])
            self.assertEqual(x[0, 0, -1], 0)
            self.assertEqual(x[0, 3, -1], 4.25)
            self.assertEqual(x[0, 6, -1], 1)
            self.assertEqual(x[0, 7, -1], 0)
        # 只含合法零值时仍是完整的一根 Bar。
        for _, values in grids.values():
            values[:] = 0
        zero_windows, _ = WindowBlock(keys, grids, stocks, CONFIG).batch([0])
        for x in zero_windows.values():
            np.testing.assert_array_equal(x[0, 72:74, -1], [1, 1])
            self.assertGreater(float(np.square(x[0, BAR_CLOCK_SLICE, -1]).sum()), 0)

    def test_original_values_and_time_encoding_match_v2(self):
        stocks = ["A", "B"]
        keys = keys_for(stocks)
        grids = grids_for(stocks)
        rng = np.random.default_rng(42)
        for _, values in grids.values():
            values[:] = rng.normal(size=values.shape)
            values[rng.random(values.shape) < .15] = np.nan
            values[-3:, 1] = np.nan
        before, clock_before = V2WindowBlock(keys, grids, stocks, CONFIG).batch([0, 1])
        after, clock_after = WindowBlock(keys, grids, stocks, CONFIG).batch([0, 1])
        np.testing.assert_array_equal(clock_before, clock_after)
        for frequency in FREQUENCIES:
            np.testing.assert_array_equal(after[frequency][:, :72], before[frequency][:, :72])
            np.testing.assert_array_equal(after[frequency][:, 74:77], before[frequency][:, 144:147])
            observed_v2 = before[frequency][:, 72:144].any(axis=1)
            observed_v3 = np.abs(after[frequency][:, BAR_CLOCK_SLICE]).sum(axis=1) > 0
            np.testing.assert_array_equal(observed_v3, observed_v2)

    def test_future_bars_do_not_change_current_windows(self):
        stocks = ["A"]
        keys = keys_for(stocks, "2024-01-02 10:03")
        grids = grids_for(stocks)
        block = WindowBlock(keys, grids, stocks, CONFIG)
        before, _ = block.batch([0])
        for times, values in grids.values():
            values[times > keys.date.iloc[0].value] = np.nan
        after, _ = block.batch([0])
        for frequency in FREQUENCIES:
            np.testing.assert_array_equal(before[frequency], after[frequency])

    def test_missing_slots_empty_grid_and_expired_history(self):
        stocks = ["A"]
        keys = keys_for(stocks, "2024-01-02 09:34")
        grids = grids_for(stocks)
        for times, values in grids.values():
            values[times == pd.Timestamp("2024-01-02 09:33").value] = np.nan
        windows, _ = WindowBlock(keys, grids, stocks, CONFIG).batch([0])
        one = windows["bar1m"][0]
        self.assertTrue((one[:, -2] == 0).all())
        np.testing.assert_array_equal(one[72:74, -1], [1, 1])
        self.assertTrue((one[:, :-4] == 0).all())
        for example_keys, example_grids in [(keys, grids_for(stocks, empty=True)),
                                            (keys_for(stocks, "2024-03-01 14:00"), grids)]:
            missing, _ = WindowBlock(example_keys, example_grids, stocks, CONFIG).batch([0])
            self.assertTrue(all((x == 0).all() for x in missing.values()))

    def test_partial_bar_survives_model_even_with_both_group_flags_zero(self):
        stocks = ["A", "B"]
        keys = keys_for(stocks)
        grids = grids_for(stocks)
        for _, values in grids.values():
            values[:, 0, [3, 7]] = np.nan
            values[:, 1] = np.nan
        windows, clock = WindowBlock(keys, grids, stocks, CONFIG).batch([0, 1])
        torch.manual_seed(42)
        model = MultiFrequencyTCN(CONFIG).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1591553)
        with torch.inference_mode():
            tensors = {frequency: torch.from_numpy(x) for frequency, x in windows.items()}
            for frequency in FREQUENCIES:
                self.assertTrue((tensors[frequency][:, 72:74] == 0).all())
                encoded = model.encoders[frequency](tensors[frequency])
                self.assertGreater(encoded[0].abs().sum().item(), 0)
                self.assertEqual(encoded[1].abs().sum().item(), 0)
            result = model(tensors, torch.from_numpy(clock))
            self.assertEqual(tuple(result.shape), (2,))
            self.assertTrue(torch.isfinite(result).all())

    def test_model_rejects_old_input_configuration(self):
        previous = json.loads((ROOT / "solutions/v2/model-config.json").read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            MultiFrequencyTCN(previous)

    def test_changed_group_or_value_policy_cannot_be_silently_ignored(self):
        for mode in ("group", "erase_values", "drop_samples"):
            altered = copy.deepcopy(CONFIG)
            if mode == "group":
                altered["data"]["masking"]["groups"]["non_book"] = ["close"]
            elif mode == "erase_values":
                altered["data"]["masking"]["value_policy"] = "zero_whole_group"
            else:
                altered["data"]["masking"]["drop_incomplete_samples"] = True
            with self.assertRaises(ValueError):
                WindowBlock(keys_for(["A"]), grids_for(["A"]), ["A"], altered)
            with self.assertRaises(ValueError):
                MultiFrequencyTCN(altered)

    def test_raw_field_rules_are_inherited_without_change(self):
        self.assertEqual(len(PRICE_FIELDS), 14)
        rows = pd.DataFrame({field: [1., 0., -1., np.inf, np.nan] for field in FIELDS})
        values = log_values(rows)
        np.testing.assert_array_equal(values, v2_log_values(rows))
        for index, field in enumerate(FIELDS):
            if field in PRICE_FIELDS:
                self.assertTrue(np.isnan(values[1:, index]).all())
            else:
                self.assertEqual(values[1, index], 0)
                self.assertTrue(np.isnan(values[2:, index]).all())

    def test_unconfirmed_training_and_reproduction_do_not_start(self):
        import v3_train
        import v3_reproduce
        import v3_notebook
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "not-created"
            arguments = ["v3_train.py", "--training-run", "unused-train", "--holdout-run", "unused-holdout", "--run-dir", str(destination)]
            with patch.object(sys, "argv", arguments), patch.object(v3_train, "train") as train, \
                    patch.object(v3_train, "snapshot_sources") as snapshot, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                v3_train.main()
            train.assert_not_called()
            snapshot.assert_not_called()
            self.assertFalse(destination.exists())
            with patch.object(v3_reproduce, "restore_sources") as restore, self.assertRaises(ValueError):
                v3_reproduce.reproduce({}, destination, ROOT / "data")
            restore.assert_not_called()
            with patch.object(v3_notebook, "_read_bundle") as read, self.assertRaises(ValueError):
                v3_notebook.reproduce(destination, ROOT / "data")
            read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
