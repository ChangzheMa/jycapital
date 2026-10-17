"""v9 原始行情、冻结预处理、分段请求及独立文本入口的一致性。"""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import duckdb
import numpy as np
import pandas as pd

from test_v9 import ROOT, CONFIGS, PROTOCOL, small_config
from v9_data import ProcessedStore, regular_times, VALUE_COLUMNS
from v9_model import MultiFrequencyTCN
from v9_predict import predict
from v9_preprocessing import FIELDS, FREQUENCIES, Normalizer, UniverseIndex, moments_from_values, pack_moments
from v9_text_artifact import FORMAT, encode_state, state_digest


class FrameDai:
    def __init__(self, tables):
        self.connection = duckdb.connect()
        self.queries = []
        for name, frame in tables.items():
            self.connection.register(name, frame)

    def query(self, sql, filters=None):
        self.queries.append(sql)
        frame = self.connection.execute(sql).fetchdf()
        return type("Query", (), {"df": lambda _: frame})()

    def close(self):
        self.connection.close()


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.days = pd.bdate_range("2024-01-02", periods=45).delete(np.arange(10, 20))
        stocks = ("A", "B", "NEW")
        minutes = (570, 600, 630, 660, 690, 810, 840, 870)
        keys = pd.DataFrame([(d + pd.Timedelta(minutes=m), s) for d in cls.days for m in minutes for s in stocks],
                            columns=["date", "instrument"])
        cls.tables = {"cpt_jyc_2026_instruments": keys}
        rng = np.random.default_rng(98)
        for frequency in FREQUENCIES:
            times = pd.to_datetime(regular_times(cls.days, frequency))
            frame = pd.DataFrame([(t, s) for t in times for s in stocks], columns=["date", "instrument"])
            for field in FIELDS:
                frame[field] = rng.uniform(1, 20, len(frame))
            frame.loc[frame.index % 61 == 0, "ask_price1"] = 0
            frame.loc[frame.index % 79 == 0, "volume"] = np.nan
            cls.tables["logical_" + frequency] = frame.drop(index=frame.index[::113]).reset_index(drop=True)
        fixed = pack_moments(moments_from_values(np.stack([np.zeros(36), np.ones(36)] * 30)))
        cls.stats = {"fields": FIELDS, "fit_interval": {"start": "2020-01-01", "stop": "2024-01-01"},
                     "stocks": {"A": fixed, "B": fixed}, "global": fixed}
        cls.preprocessing = json.loads((ROOT / "solutions/v9/config.json").read_text(encoding="utf-8"))
        cls.config = small_config()
        torch.manual_seed(42)
        cls.state = MultiFrequencyTCN(cls.config).state_dict()
        cls.bundle = {"format": FORMAT, "config": cls.config, "preprocessing": cls.preprocessing,
                      "normalization": dict.fromkeys(FREQUENCIES, cls.stats),
                      "state_dict": encode_state(cls.state), "tensor_sha256": state_digest(cls.state)}
        cls.sources = {f: "logical_" + f for f in FREQUENCIES}

    def call_predict(self, start, end, *, tables=None, directory=None, bundle=True):
        dai = FrameDai(self.tables if tables is None else tables)
        try:
            result = predict(self.sources, start, end, model_dir=directory, dai_module=dai, device="cpu",
                             bundle=self.bundle if bundle else None)
            self.assertFalse(any("vwap" in q or "bar15m" in q for q in dai.queries))
            return result
        finally:
            dai.close()

    def test_raw_inference_matches_processed_store_across_months_and_long_holiday(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, train, holdout = (root / name for name in ("data", "train", "holdout"))
            data.mkdir()
            universe = self.tables["cpt_jyc_2026_instruments"]
            universe.to_parquet(data / "cpt_jyc_2026_instruments.parquet", index=False)
            labels = universe.copy()
            labels["vwap_return"] = np.arange(len(labels)) * 1e-6
            labels.to_parquet(data / "cpt_jyc_2026_vwap.parquet", index=False)
            index = UniverseIndex(universe)
            for frequency in FREQUENCIES:
                raw = self.tables["logical_" + frequency]
                normalizer = Normalizer(self.stats, self.preprocessing)
                frame, _ = normalizer.transform(raw, index.eligible(raw))
                for month, part in frame.groupby(frame.date.dt.to_period("M")):
                    path = holdout / "data/validation" / frequency / (str(month) + ".parquet")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    part.to_parquet(path, index=False)
            store = ProcessedStore(train, holdout, data, self.config, PROTOCOL)
            first, last = self.days[-3], self.days[-1]
            frames = []
            model = MultiFrequencyTCN(self.config).eval()
            model.load_state_dict(self.state)
            with torch.inference_mode():
                for month in pd.period_range(first, last, freq="M"):
                    block = store.load("validation", str(month))
                    chosen = np.flatnonzero((block.keys.date >= first) & (block.keys.date < last + pd.Timedelta(days=1)))
                    windows, clock = block.batch(chosen)
                    frame = block.keys.iloc[chosen][["date", "instrument"]].copy()
                    frame["factor"] = model({f: torch.from_numpy(x) for f, x in windows.items()},
                                             torch.from_numpy(clock)).numpy().astype(np.float64)
                    frames.append(frame)
            expected = pd.concat(frames, ignore_index=True)
            actual = self.call_predict(first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d"))
            pd.testing.assert_frame_equal(actual[["date", "instrument"]], expected[["date", "instrument"]])
            np.testing.assert_allclose(actual.factor, expected.factor, rtol=1e-5, atol=2e-7)
            self.assertEqual(actual.factor.dtype, np.dtype("float64"))
            self.assertTrue(np.isfinite(actual.factor).all())

    def test_split_requests_checkpoint_text_and_future_poison_agree(self):
        first, last = self.days[-3], self.days[-1]
        whole = self.call_predict(first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d"))
        stamp = last + pd.Timedelta(hours=10)
        single = self.call_predict(stamp, stamp)
        expected = whole.loc[whole.date == stamp].reset_index(drop=True)
        pd.testing.assert_frame_equal(single[["date", "instrument"]], expected[["date", "instrument"]])
        np.testing.assert_allclose(single.factor, expected.factor, rtol=1e-5, atol=2e-7)
        changed = {key: frame.copy() for key, frame in self.tables.items()}
        for frequency in FREQUENCIES:
            frame = changed["logical_" + frequency]
            frame.loc[frame.date > stamp, FIELDS] = 1e5
        poisoned = self.call_predict(last.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d"), tables=changed)
        np.testing.assert_allclose(poisoned.loc[poisoned.date == stamp, "factor"], single.factor, rtol=1e-5, atol=2e-7)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "normalization").mkdir()
            (directory / "preprocessing-config.json").write_text(json.dumps(self.preprocessing), encoding="utf-8")
            for frequency in FREQUENCIES:
                (directory / "normalization" / (frequency + ".json")).write_text(json.dumps(self.stats), encoding="utf-8")
            torch.save({"config": self.config, "state_dict": self.state}, directory / "best.pt")
            checkpoint = self.call_predict(stamp, stamp, directory=directory, bundle=False)
            pd.testing.assert_frame_equal(checkpoint, single, check_exact=True)

    def test_embedded_notebook_runtime_is_independent_and_import_has_no_queries(self):
        bundle = copy.deepcopy(self.bundle)
        names = ["schema", "preprocessing", "data", "model", "text_artifact", "predict"]
        bundle["runtime_sources"] = [f"solutions/v9/v9_{name}.py" for name in names]
        bundle["sources"] = {path: {"text": (ROOT / path).read_text(encoding="utf-8")}
                             for path in bundle["runtime_sources"]}
        template = (ROOT / "solutions/v9/v9_notebook.py").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "v9-model.json").write_text(json.dumps(bundle), encoding="utf-8")
            namespace = {"__name__": "submission_import", "__file__": str(directory / "entry.py")}
            exec(compile(template, namespace["__file__"], "exec"), namespace)
            self.assertIsNone(namespace["_RUNTIME"])
            before = sys.path[:]
            saved = {key: value for key, value in sys.modules.items() if key.startswith("v9_")}
            try:
                for key in saved:
                    del sys.modules[key]
                sys.path[:] = [p for p in sys.path if "solutions" not in p]
                _, modules = namespace["_runtime"]()
                dai = FrameDai(self.tables)
                try:
                    stamp = self.days[-1] + pd.Timedelta(hours=10)
                    actual = modules["v9_predict"].predict(self.sources, stamp, stamp, dai_module=dai,
                                                           device="cpu", bundle=bundle)
                finally:
                    dai.close()
            finally:
                sys.path[:] = before
                sys.modules.update(saved)
            expected = self.call_predict(stamp, stamp)
            pd.testing.assert_frame_equal(actual, expected, check_exact=True)
            self.assertEqual([p.name for p in directory.iterdir()], ["v9-model.json"])


if __name__ == "__main__":
    unittest.main()
