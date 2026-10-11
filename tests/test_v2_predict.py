"""验证平台逻辑源、日期接口和缺参股票历史热身的一致性。"""

import copy
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solutions/v2"))
from v2_data import regular_times
from v2_model import MultiFrequencyTCN
from v2_predict import platform_interval, predict
from v2_preprocessing import FIELDS, FREQUENCIES, moments_from_values, pack_moments
from v2_text_artifact import FORMAT, encode_state, state_digest


class FakeDai:
    def __init__(self, tables):
        self.tables, self.queries = tables, []

    def query(self, sql, filters=None):
        self.queries.append(sql)
        table = re.search(r"FROM ([A-Za-z0-9_]+)", sql).group(1)
        frame = self.tables[table].copy()
        before = re.search(r"date < '([^']+)'", sql)
        after = re.search(r"date >= '([^']+)'", sql)
        if before:
            frame = frame.loc[frame.date < pd.Timestamp(before.group(1))]
        if after:
            frame = frame.loc[frame.date >= pd.Timestamp(after.group(1))]
        excluded = re.search(r"instrument NOT IN \(([^)]+)\)", sql)
        if excluded:
            codes = re.findall(r"'([^']+)'", excluded.group(1))
            frame = frame.loc[~frame.instrument.isin(codes)]
        included = re.search(r"instrument IN \(([^)]+)\)", sql)
        if included:
            codes = re.findall(r"'([^']+)'", included.group(1))
            frame = frame.loc[frame.instrument.isin(codes)]
        if "max(date)" in sql:
            frame = pd.DataFrame({"date": [frame.date.max()]})
        elif "SELECT DISTINCT instrument" in sql:
            frame = frame[["instrument"]].drop_duplicates()
        else:
            columns = sql.split("SELECT ", 1)[1].split(" FROM", 1)[0].split(", ")
            frame = frame[columns]
        return type("Query", (), {"df": lambda _: frame})()


class PredictTest(unittest.TestCase):
    def test_interval_inclusive_date_and_minute(self):
        self.assertEqual(platform_interval("2024-01-02", "2024-01-02")[1], pd.Timestamp("2024-01-03"))
        self.assertEqual(platform_interval("2024-01-02 10:00", "2024-01-02 10:00")[1], pd.Timestamp("2024-01-02 10:01"))
        self.assertEqual(platform_interval("2024-01-02T02:00:00Z", "2024-01-02T02:00:00Z")[0], pd.Timestamp("2024-01-02 10:00"))

    def test_split_requests_match_with_dynamic_history_and_logical_sources(self):
        config = json.loads((ROOT / "solutions/v2/model-config.json").read_text(encoding="utf-8"))
        config = copy.deepcopy(config)
        config["data"]["history_days"] = 1
        config["data"]["window_bars"] = dict.fromkeys(FREQUENCIES, 4)
        config["model"].update(hidden_dim=4, fusion_hidden_dim=8, blocks=dict.fromkeys(FREQUENCIES, 1))
        preprocessing = json.loads((ROOT / "solutions/v2/config.json").read_text(encoding="utf-8"))
        days = pd.date_range("2024-01-02", periods=4)
        tables = {}
        rows = [(day + pd.Timedelta(minutes=m), stock) for day in days for m in (600, 630, 660, 690, 810, 840, 870, 900) for stock in ("A", "B", "NEW")]
        tables["cpt_jyc_2026_instruments"] = pd.DataFrame(rows, columns=["date", "instrument"])
        rng = np.random.default_rng(42)
        for frequency in FREQUENCIES:
            times = pd.to_datetime(regular_times(days, frequency))
            bars = pd.DataFrame([(t, s) for t in times for s in ("A", "B", "NEW")], columns=["date", "instrument"])
            for field in FIELDS:
                bars[field] = rng.uniform(1, 20, len(bars))
            tables["logical_" + frequency] = bars
        fixed = pack_moments(moments_from_values(np.stack([np.zeros(36), np.ones(36)] * 30)))
        statistics = {"fields": FIELDS, "fit_interval": {"start": "2020-01-01", "stop": "2024-01-01"},
                      "stocks": {"A": fixed, "B": fixed}, "global": fixed}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "normalization").mkdir()
            (directory / "preprocessing-config.json").write_text(json.dumps(preprocessing), encoding="utf-8")
            for frequency in FREQUENCIES:
                (directory / "normalization" / (frequency + ".json")).write_text(json.dumps(statistics), encoding="utf-8")
            torch.manual_seed(42)
            torch.save({"config": config, "state_dict": MultiFrequencyTCN(config).state_dict()}, directory / "best.pt")
            dai = FakeDai(tables)
            sources = {f: "logical_" + f for f in FREQUENCIES}
            whole = predict(sources, "2024-01-02", "2024-01-05", directory, dai, device="cpu")
            for frequency in FREQUENCIES:
                (directory / "normalization" / (frequency + ".json")).rename(directory / ("normalization-" + frequency + ".json"))
            suffix = predict(sources, "2024-01-04", "2024-01-05", directory, dai, device="cpu")
            checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
            text_directory = directory / "text-only"
            text_directory.mkdir()
            bundle = {"format": FORMAT, "config": config, "preprocessing": preprocessing,
                      "normalization": dict.fromkeys(FREQUENCIES, statistics),
                      "state_dict": encode_state(checkpoint["state_dict"]), "tensor_sha256": state_digest(checkpoint["state_dict"])}
            (text_directory / "v2-model.json").write_text(json.dumps(bundle), encoding="utf-8")
            text_result = predict(sources, "2024-01-04", "2024-01-05", text_directory, dai, device="cpu")
            pd.testing.assert_frame_equal(text_result, suffix, check_exact=True)
            expected = whole.loc[whole.date >= "2024-01-04"].reset_index(drop=True)
            pd.testing.assert_frame_equal(suffix[["date", "instrument"]], expected[["date", "instrument"]])
            np.testing.assert_allclose(suffix.factor, expected.factor, atol=1e-7, rtol=1e-5)
            self.assertEqual(len(whole), len(tables["cpt_jyc_2026_instruments"]))
            self.assertEqual(whole.factor.dtype, np.dtype("float64"))
            self.assertFalse(any("vwap" in q or "exposure" in q for q in dai.queries))
            self.assertTrue(any("SELECT DISTINCT instrument" in q for q in dai.queries))


if __name__ == "__main__":
    unittest.main()
