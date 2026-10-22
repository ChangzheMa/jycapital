"""盘前信息进入预测、未来隔离、fast等价及平台查询复用。"""
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

from test_v1801 import ROOT, small_config
from test_v2_predict import FakeDai
from v1801_data import regular_times
from v1801_preprocessing import FIELDS, FREQUENCIES, pack_moments, moments_from_values
from v1801_model import MultiFrequencyGRU
from v1801_predict import predict
from v1801_predict_reference import predict as reference_predict
sys.path.insert(0, str(ROOT / "solutions/v1102"))
from v1102_predict_fast import predict as parent_predict


class CalendarDai(FakeDai):
    def query(self, sql, filters=None):
        if "CAST(date AS DATE)" not in sql:
            return super().query(sql, filters)
        self.queries.append(sql)
        frame = self.tables["cpt_jyc_2026_instruments"]
        stop = pd.Timestamp(re.search(r"date < '([^']+)'", sql).group(1))
        limit = int(re.search(r"LIMIT (\d+)", sql).group(1))
        days = sorted(frame.loc[frame.date < stop, "date"].dt.normalize().unique(), reverse=True)[:limit]
        return type("Query", (), {"df": lambda _: pd.DataFrame({"date": days})})()


class PreopenPredictTests(unittest.TestCase):
    def test_preopen_effect_future_isolation_fast_reference_split_and_parent_queries(self):
        config = small_config()
        preprocessing = json.loads((ROOT / "solutions/v1801/config.json").read_text(encoding="utf-8"))
        days = pd.bdate_range("2024-01-02", periods=26)
        minutes = (570, 600, 630, 660, 690, 810, 840, 870)
        stocks = ("A", "B", "C", "NEW")
        universe = pd.DataFrame([(d + pd.Timedelta(minutes=m), s) for d in days for m in minutes
            for s in (("A", "B", "C") if m < 630 else ("A", "C", "NEW"))], columns=["date", "instrument"])
        tables = {"cpt_jyc_2026_instruments": universe}
        rng = np.random.default_rng(1801)
        for frequency in FREQUENCIES:
            times = list(pd.to_datetime(regular_times(days, frequency)))
            extra_minutes = range(556, 566) if frequency == "bar1m" else (565,)
            times += [d + pd.Timedelta(minutes=m) for d in days for m in extra_minutes]
            frame = pd.DataFrame([(t, s) for t in sorted(times) for s in stocks], columns=["date", "instrument"])
            for field in FIELDS:
                frame[field] = rng.uniform(1, 20, len(frame))
            tables["fixture_" + frequency] = frame
        fixed = pack_moments(moments_from_values(np.stack([np.zeros(36), np.ones(36)] * 40)))
        # NEW 缺少自身统计，必须沿用完整已知历史的因果回退。
        statistics = {"fields": FIELDS, "fit_interval": {"start": "2020-01-01", "stop": "2024-01-01"},
                      "stocks": {stock: fixed for stock in stocks if stock != "NEW"}, "global": fixed}
        sources = {frequency: "fixture_" + frequency for frequency in FREQUENCIES}
        first, last = (str(day.date()) for day in days[-2:])
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            parent_directory = directory / "parent"
            for dest in (directory, parent_directory):
                (dest / "normalization").mkdir(parents=True)
                for frequency in FREQUENCIES:
                    (dest / "normalization" / f"{frequency}.json").write_text(json.dumps(statistics), encoding="utf-8")
                (dest / "preprocessing-config.json").write_text(json.dumps(preprocessing), encoding="utf-8")
            torch.manual_seed(42)
            model = MultiFrequencyGRU(config)
            torch.nn.init.normal_(model.preopen_context.output.weight, std=.1)
            torch.nn.init.normal_(model.preopen_context.output.bias, std=.01)
            torch.save({"config": config, "state_dict": model.state_dict()}, directory / "best.pt")
            parent_config = copy.deepcopy(config)
            parent_config["solution_id"] = "v1102"
            parent_config.pop("preopen_context")
            parent_state = {k: v for k, v in model.state_dict().items() if not k.startswith("preopen_context.")}
            torch.save({"config": parent_config, "state_dict": parent_state}, parent_directory / "best.pt")
            dai = CalendarDai(tables)
            whole = predict(sources, first, last, directory, dai, device="cpu")
            whole_queries = list(dai.queries)
            parent_dai = CalendarDai(tables)
            parent_predict(sources, first, last, parent_directory, parent_dai, device="cpu")
            self.assertEqual(whole_queries, parent_dai.queries)
            reference = reference_predict(sources, first, last, directory, dai, device="cpu")
            split = pd.concat([predict(sources, day, day, directory, dai, device="cpu") for day in (first, last)], ignore_index=True)
            partial = predict(sources, last + " 10:30", last + " 11:30", directory, dai, device="cpu")
            for expected in (reference, split):
                pd.testing.assert_frame_equal(whole[["date", "instrument"]], expected[["date", "instrument"]])
                np.testing.assert_allclose(whole.factor, expected.factor, rtol=1e-4, atol=2e-7)
            expected = whole.loc[whole.date.between(last + " 10:30", last + " 11:30")].reset_index(drop=True)
            pd.testing.assert_frame_equal(partial[["date", "instrument"]], expected[["date", "instrument"]])
            np.testing.assert_allclose(partial.factor, expected.factor, rtol=1e-4, atol=2e-7)
            changed = {k: v.copy() for k, v in tables.items()}
            frame = changed["fixture_bar1m"]
            selected = frame.date.eq(pd.Timestamp(last + " 09:20")) & frame.instrument.eq("A")
            self.assertEqual(int(selected.sum()), 1)
            frame.loc[selected, FIELDS] = 1000
            altered = predict(sources, last, last, directory, CalendarDai(changed), device="cpu")
            expected = whole.loc[whole.date.dt.normalize().eq(days[-1])].reset_index(drop=True)
            opening = altered.date.eq(pd.Timestamp(last + " 09:30"))
            self.assertGreater(np.max(np.abs(altered.loc[opening, "factor"] - expected.loc[opening, "factor"])), 1e-9)
            before_parent = parent_predict(sources, last, last, parent_directory, CalendarDai(tables), device="cpu")
            after_parent = parent_predict(sources, last, last, parent_directory, CalendarDai(changed), device="cpu")
            np.testing.assert_array_equal(before_parent.factor, after_parent.factor)
            future = {k: v.copy() for k, v in tables.items()}
            for frequency in FREQUENCIES:
                frame = future["fixture_" + frequency]
                select = frame.date.dt.normalize().eq(days[-1]) & frame.date.ge(pd.Timestamp(last + " 09:31"))
                frame.loc[select, FIELDS] = 2000
            future_output = predict(sources, last, last, directory, CalendarDai(future), device="cpu")
            np.testing.assert_allclose(future_output.loc[opening, "factor"], expected.loc[opening, "factor"], rtol=1e-5, atol=2e-7)
            only_open = predict(sources, last + " 09:30", last + " 09:30", directory, dai, device="cpu")
            np.testing.assert_allclose(only_open.factor, expected.loc[opening, "factor"], rtol=1e-4, atol=2e-7)
        self.assertEqual(len(whole), 48)
        self.assertEqual(len(partial), 9)
        self.assertEqual(len(only_open), 3)
        self.assertEqual(whole.factor.dtype, np.float64)
        self.assertTrue(np.isfinite(whole.factor).all())
        self.assertFalse(whole.duplicated(["date", "instrument"]).any())
        self.assertFalse(any("vwap" in query or "exposure" in query for query in dai.queries))


if __name__ == "__main__":
    unittest.main()
