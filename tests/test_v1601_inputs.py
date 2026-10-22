"""共同价格坐标、源数据绑定和缓存共享的定向检查。"""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "common"), str(ROOT / "solutions/v1102"), str(ROOT / "solutions/v1601")]
import price_geometry as geometry
from artifacts import sha256
from v1601_model import MultiFrequencyGRU
from v1102_model import MultiFrequencyGRU as ParentModel


def fixture():
    keys = pd.DataFrame({"date": pd.to_datetime(["2020-01-02 09:31", "2020-01-02 09:32", "2020-01-02 09:33"]),
                         "instrument": ["A", "A", "A"]})
    raw, normalized = keys.copy(), keys.copy()
    for i, field in enumerate(geometry.PRICE_FIELDS):
        raw[field] = np.array([10 + i * .01, 10 + i * .01, 11 + i * .01], np.float32)
    for i, field in enumerate(geometry.BASE_COLUMNS):
        normalized[field] = np.array([i, i + 1, i + 2], np.float32)
    return raw, normalized


class InputTests(unittest.TestCase):
    def test_only_price_second_view_changes_and_first_two_bars_are_causal(self):
        raw, normalized = fixture()
        actual = geometry.transform_frame(normalized, raw)
        keep = ["z_" + f for f in geometry.FIELDS] + ["c_" + f for f in geometry.FIELDS if f not in geometry.PRICE_FIELDS]
        pd.testing.assert_frame_equal(actual[keep], normalized[keep])
        self.assertEqual(actual.columns.tolist(), geometry.KEYS + geometry.VALUE_COLUMNS)
        self.assertTrue((actual.g_close == 0).all())
        expected = 100 * (np.log(raw.ask_price1.astype(np.float64)) - np.log(raw.close.astype(np.float64)))
        np.testing.assert_array_equal(actual.g_ask_price1, expected.astype(np.float32))
        changed = raw.copy()
        changed.loc[2, geometry.PRICE_FIELDS] = (changed.loc[2, geometry.PRICE_FIELDS].to_numpy(dtype=np.float32) * np.float32(100))
        changed_result = geometry.transform_frame(normalized, changed)
        pd.testing.assert_frame_equal(changed_result.iloc[:2], actual.iloc[:2])

    def test_invalid_close_falls_back_to_observed_price_and_keeps_missing(self):
        raw, normalized = fixture()
        raw.loc[0, "close"] = 0
        raw.loc[1, geometry.PRICE_FIELDS] = np.nan
        actual = geometry.transform_frame(normalized, raw)
        self.assertEqual(actual.g_open.iloc[0], 0)
        self.assertTrue(np.isnan(actual.g_close.iloc[0]))
        self.assertTrue(actual.loc[1, ["g_" + f for f in geometry.PRICE_FIELDS]].isna().all())
        np.testing.assert_array_equal(actual.c_volume, normalized.c_volume)
        self.assertEqual(len(actual), len(raw))

    def test_local_geometry_keeps_levels_and_same_values_after_fp16(self):
        raw, normalized = fixture()
        for level in range(1, 6):
            raw[f"ask_price{level}"] = np.float32(10 + .01 * level)
            raw[f"bid_price{level}"] = np.float32(10 - .01 * level)
        raw["close"] = np.float32(10)
        actual = geometry.transform_frame(normalized, raw)
        asks = actual[[f"g_ask_price{i}" for i in range(1, 6)]].to_numpy(dtype=np.float16)
        bids = actual[[f"g_bid_price{i}" for i in range(1, 6)]].to_numpy(dtype=np.float16)
        self.assertTrue((np.diff(asks, axis=1) > 0).all())
        self.assertTrue((np.diff(bids, axis=1) < 0).all())
        self.assertTrue((asks[:, 0] > bids[:, 0]).all())

    def test_empty_frame_and_key_mismatch(self):
        raw, normalized = fixture()
        string_keys = raw.copy()
        string_keys["instrument"] = string_keys.instrument.astype("string")
        pd.testing.assert_frame_equal(geometry.transform_frame(normalized, string_keys),
                                      geometry.transform_frame(normalized, raw))
        empty = pd.DataFrame(columns=normalized.columns)
        self.assertTrue(geometry.transform_frame(empty, raw.iloc[:0]).empty)
        with self.assertRaises(ValueError):
            geometry.transform_frame(normalized, raw.iloc[::-1])
        with self.assertRaises(ValueError):
            geometry.transform_frame(pd.concat([normalized, normalized]), pd.concat([raw, raw]))

    def test_raw_source_aligns_reordered_rows_and_rejects_missing(self):
        raw, normalized = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            for frequency in ("bar1m", "bar5m"):
                directory = data / ("cpt_jyc_2026_stock_" + frequency)
                directory.mkdir()
                raw.iloc[::-1].to_parquet(directory / "2020-01.parquet", index=False)
            reader = geometry.RawPriceSource(data)
            for frequency in ("bar1m", "bar5m"):
                pd.testing.assert_frame_equal(reader.transform(normalized, frequency), geometry.transform_frame(normalized, raw))
                self.assertEqual(reader.identity, (frequency, "2020-01"))
            missing = normalized.copy()
            missing.loc[0, "instrument"] = "MISSING"
            with self.assertRaises(ValueError):
                reader.transform(missing, "bar5m")
            reader.close()
            self.assertIsNone(reader.indexed)

    def test_raw_fingerprints_are_bound_to_frozen_preprocessing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, normalized = fixture()
            records = {}
            for frequency in ("bar1m", "bar5m", "bar30m"):
                relative = "cpt_jyc_2026_stock_" + frequency + "/2020-01.parquet"
                path = root / relative
                path.parent.mkdir()
                raw.to_parquet(path, index=False)
                records[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
            prepared = root / "prepared"
            prepared.mkdir()
            (prepared / "data-manifest.json").write_text(json.dumps({"files": records}), encoding="utf-8")
            store = SimpleNamespace(training_run=prepared, holdout_run=prepared, data_root=root,
                                    months=lambda split: ["2020-01"])
            self.assertEqual(len(geometry.verified_raw_sources(store)), 3)
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                geometry.verified_raw_sources(store)

    def test_fresh_model_and_training_objective_match_parent_configuration(self):
        current = json.loads((ROOT / "solutions/v1601/model-config.json").read_text(encoding="utf-8"))
        parent = json.loads((ROOT / "solutions/v1102/model-config.json").read_text(encoding="utf-8"))
        self.assertEqual(current["model"], parent["model"])
        left, right = copy.deepcopy(current["training"]), copy.deepcopy(parent["training"])
        left["resident_cache"].pop("format")
        right["resident_cache"].pop("format")
        self.assertEqual(left, right)
        torch.manual_seed(42)
        before = ParentModel(parent).state_dict()
        torch.manual_seed(42)
        after = MultiFrequencyGRU(current).state_dict()
        self.assertEqual(before.keys(), after.keys())
        for name in before:
            torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
