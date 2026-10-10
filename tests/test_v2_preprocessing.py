"""v2 预处理的数学、时点、分块和本地文件入口验证。"""

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solutions" / "v2"))
from v2_preprocessing import (FIELDS, FREQUENCIES, PRICE_FIELDS, Normalizer, UniverseIndex,
                              log_values, moments_from_values, pack_moments)
from v2_prepare import complete_sections, fit_file, prepare

CONFIG = json.loads((ROOT / "solutions/v2/config.json").read_text(encoding="utf-8"))


def config():
    value = copy.deepcopy(CONFIG)
    value["data"]["session"] = "regular"
    value["normalization"]["min_observations"] = 2
    value["processing"]["batch_rows"] = 3
    value["processing"]["threads"] = 1
    return value


def moments(count=100, mean=0.0, std=1.0):
    return {"count": [count], "mean": [mean], "m2": [count * std ** 2]}


def statistics(stocks=None):
    return {"fields": ["close"], "fit_interval": {"start": "2020-01-01", "stop": "2024-01-01"},
            "stocks": stocks if stocks is not None else {stock: moments() for stock in "ABC"},
            "global": moments()}


def frame(dates, stocks, logged_prices):
    return pd.DataFrame({"date": pd.to_datetime(dates), "instrument": stocks, "close": np.exp(logged_prices)})


class NormalizationTests(unittest.TestCase):
    def test_log_domains_and_zero_quantities(self):
        values = pd.DataFrame({"close": [2., 0., -1., np.inf], "volume": [0., 2., -1., np.nan]})
        result = log_values(values, ["close", "volume"])
        self.assertAlmostEqual(result[0, 0], np.log(2))
        self.assertEqual(result[0, 1], 0)
        self.assertAlmostEqual(result[1, 1], np.log1p(2))
        self.assertTrue(np.isnan(result[1:, 0]).all())
        self.assertTrue(np.isnan(result[2:, 1]).all())

    def test_other_stock_changes_only_cross_section_channel(self):
        values = frame(["2024-01-02 10:00"] * 3, list("ABC"), [1., 2., 3.])
        before, _ = Normalizer(statistics(), config()).transform(values, [True] * 3)
        changed = values.copy()
        changed.loc[2, "close"] = np.exp(8.)
        after, _ = Normalizer(statistics(), config()).transform(changed, [True] * 3)
        self.assertEqual(before.loc[0, "z_close"], after.loc[0, "z_close"])
        self.assertNotEqual(before.loc[0, "c_close"], after.loc[0, "c_close"])
        np.testing.assert_allclose(before["c_close"], [-np.sqrt(1.5), 0, np.sqrt(1.5)], atol=1e-6)

    def test_dynamic_history_and_global_fallback(self):
        values = frame(pd.date_range("2024-01-02 10:00", periods=3, freq="min"), ["D"] * 3, [1., 3., 5.])
        result, quality = Normalizer(statistics(), config()).transform(values, [True] * 3)
        np.testing.assert_allclose(result["z_close"], [1., 1., np.sqrt(1.5)], atol=1e-6)
        self.assertEqual(quality["dynamic_values"], 2)
        self.assertEqual(quality["pooled_values"], 1)
        self.assertTrue(result["c_close"].isna().all())

    def test_dynamic_fallback_is_seeded_from_known_training_history(self):
        data = statistics({"D": moments(count=1, mean=0, std=0)})
        values = frame(["2024-01-02 10:00"], ["D"], [2.])
        result, quality = Normalizer(data, config()).transform(values, [True])
        self.assertEqual(result.loc[0, "z_close"], 1.)
        self.assertEqual(quality["dynamic_values"], 1)

    def test_thirty_observation_threshold(self):
        selected = config()
        selected["normalization"]["min_observations"] = 30
        values = frame(pd.date_range("2024-01-02 10:00", periods=31, freq="min"), ["D"] * 31,
                       np.linspace(1.0, 3.0, 31))
        result, quality = Normalizer(statistics(), selected).transform(values, np.ones(31, bool))
        self.assertEqual(quality["pooled_values"], 29)
        self.assertEqual(quality["dynamic_values"], 2)
        self.assertAlmostEqual(result["z_close"].iloc[-1], np.sqrt(90 / 32), places=6)

    def test_future_values_and_chunking_do_not_change_past(self):
        dates = np.repeat(pd.date_range("2024-01-02 10:00", periods=3, freq="min"), 2)
        values = frame(dates, list("AD") * 3, [1., 2., 3., 4., 5., 6.])
        expected, _ = Normalizer(statistics(), config()).transform(values, np.ones(6, bool))
        normalizer = Normalizer(statistics(), config())
        first, _ = normalizer.transform(values.iloc[:2], [True] * 2)
        restored = Normalizer(statistics(), config(), history_dict=normalizer.history_dict())
        second, _ = restored.transform(values.iloc[2:], [True] * 4)
        pd.testing.assert_frame_equal(expected, pd.concat([first, second], ignore_index=True))
        changed = values.copy()
        changed.loc[4:, "close"] *= 1000
        actual, _ = Normalizer(statistics(), config()).transform(changed, np.ones(6, bool))
        pd.testing.assert_frame_equal(expected.iloc[:4], actual.iloc[:4])
        with self.assertRaises(ValueError):
            restored.transform(values.iloc[4:], [True] * 2)

    def test_training_period_is_explicit_and_stats_are_not_mutated(self):
        source = statistics()
        saved = copy.deepcopy(source)
        values = frame(["2020-01-02 10:00"], ["A"], [1.])
        with self.assertRaises(ValueError):
            Normalizer(source, config()).transform(values, [True])
        Normalizer(source, config(), training=True).transform(values, [True])
        self.assertEqual(source, saved)

    def test_universe_uses_no_future_snapshot(self):
        universe = pd.DataFrame({"date": pd.to_datetime(["2024-01-02 09:30"] * 2 + ["2024-01-02 10:30"] * 2),
                                 "instrument": list("ABAC")})
        index = UniverseIndex(universe)
        values = frame(["2024-01-02 09:20", "2024-01-02 10:00", "2024-01-02 10:00", "2024-01-02 10:31"],
                       list("ABCB"), [1., 2., 3., 4.])
        np.testing.assert_array_equal(index.eligible(values), [False, True, False, False])

    def test_degenerate_scale_and_nulls_are_preserved(self):
        data = statistics({stock: moments(std=0) for stock in "ABC"})
        data["global"] = moments(std=0)
        values = frame(["2024-01-02 10:00"] * 3, list("ABC"), [0., 0., 0.])
        values.loc[2, "close"] = 0
        result, _ = Normalizer(data, config()).transform(values, [True] * 3)
        np.testing.assert_array_equal(result.loc[:1, ["z_close", "c_close"]], 0)
        self.assertTrue(result.loc[2, ["z_close", "c_close"]].isna().all())

    def test_arrow_boundaries_keep_complete_sections(self):
        values = frame(["2024-01-02 10:00"] * 3 + ["2024-01-02 10:01"] * 3, list("ABC") * 2, np.arange(6.))
        reader = pa.Table.from_pandas(values).to_batches(max_chunksize=2)
        chunks = list(complete_sections(reader))
        self.assertEqual([len(chunk) for chunk in chunks], [3, 3])
        pd.testing.assert_frame_equal(pd.concat(chunks, ignore_index=True), values)


class PipelineTests(unittest.TestCase):
    def fixture(self, root):
        dates = ["2020-01-02 09:25"] * 3 + ["2020-01-02 10:00"] * 3 + ["2020-01-02 10:30"] * 3 + ["2024-01-02 10:00"] * 3
        stocks = list("ABC") * 4
        values = pd.DataFrame({"date": pd.to_datetime(dates), "instrument": stocks})
        for field in FIELDS:
            values[field] = np.arange(len(values), dtype=np.float64) + 1
        values.loc[9:, FIELDS] = 1e10
        values.loc[4, "bid_price5"] = 0
        values.loc[5, "ask_volume5"] = -1
        values.loc[3, "volume"] = 0
        for frequency in FREQUENCIES:
            directory = root / f"cpt_jyc_2026_stock_{frequency}"
            directory.mkdir(parents=True)
            values.to_parquet(directory / "2020-01.parquet", index=False)
        universe = pd.DataFrame({"date": pd.to_datetime(["2020-01-02 09:30"] * 3), "instrument": list("ABC")})
        universe.to_parquet(root / "cpt_jyc_2026_instruments.parquet", index=False)
        return values

    def test_full_training_statistics_respect_dates_sessions_and_domains(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = self.fixture(root)
            interval = {"start": "2020-01-01", "stop": "2020-02-01"}
            connection = duckdb.connect()
            result = fit_file(connection, root / "cpt_jyc_2026_stock_bar1m/2020-01.parquet", interval, config())
            self.assertEqual(result["rows"], 6)
            for stock in "ABC":
                expected = moments_from_values(log_values(raw.iloc[3:9].loc[raw.iloc[3:9]["instrument"] == stock]))
                actual = result["stocks"][stock]
                for key in expected:
                    np.testing.assert_allclose(actual[key], expected[key], atol=1e-12)
            all_sessions = config()
            all_sessions["data"]["session"] = "all"
            full = fit_file(connection, root / "cpt_jyc_2026_stock_bar1m/2020-01.parquet", interval, all_sessions)
            self.assertEqual(full["rows"], 9)
            connection.close()

    def test_four_frequency_file_pipeline_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "source"
            data.mkdir()
            self.fixture(data)
            run = root / "processed"
            protocol = {"splits": {"train": {"start": "2020-01-01", "stop": "2020-02-01"}},
                        "universe_table": "cpt_jyc_2026_instruments"}
            with contextlib.redirect_stdout(io.StringIO()):
                summary = prepare(run, data, config(), protocol)
                resumed = prepare(run, data, config(), protocol, resume=True)
            self.assertEqual(summary, resumed)
            for frequency in FREQUENCIES:
                result = pd.read_parquet(run / "data" / frequency / "2020-01.parquet")
                self.assertEqual(result.shape, (6, 74))
                self.assertFalse(result.duplicated(["date", "instrument"]).any())
                self.assertFalse(np.isinf(result.iloc[:, 2:].to_numpy()).any())
                self.assertEqual(summary["frequencies"][frequency]["rows"], 6)
            manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue(any(name.startswith("solutions/v2/") for name in manifest["source"]["files"]))
            self.assertFalse(any(name.startswith("solutions/v1/") for name in manifest["source"]["files"]))


if __name__ == "__main__":
    unittest.main()
