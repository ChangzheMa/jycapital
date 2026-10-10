"""冻结参数、跨划分因果状态及验证／测试保存入口。"""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solutions/v2"))
from v2_preprocessing import FIELDS, FREQUENCIES
from v2_prepare import prepare
from v2_apply import apply_holdout
from artifacts import sha256

class HoldoutTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        data, training = root / "data", root / "training"
        data.mkdir()
        config = json.loads((ROOT / "solutions/v2/config.json").read_text(encoding="utf-8"))
        config["normalization"]["min_observations"] = 2
        config["processing"].update(batch_rows=3, threads=1)
        protocol = {"universe_table": "cpt_jyc_2026_instruments", "splits": {
            "train": {"start": "2020-01-01", "stop": "2020-02-01"},
            "validation": {"start": "2020-02-01", "stop": "2020-03-01"},
            "test": {"start": "2020-03-01", "stop": "2020-04-01"}}}
        frames = {}
        for month, stocks, log_values in [("2020-01", "ABAB", [0., 1., 2., 3.]),
                                          ("2020-02", "ABC", [4., 5., 2.]),
                                          ("2020-03", "ABC", [6., 7., 4.])]:
            times = [f"{month}-02 10:00"] * len(stocks)
            if month == "2020-01":
                times[2:] = [f"{month}-02 10:30"] * 2
            frame = pd.DataFrame({"date": pd.to_datetime(times), "instrument": list(stocks)})
            for field in FIELDS:
                frame[field] = np.exp(log_values)
            frames[month] = frame
        for frequency in FREQUENCIES:
            target = data / f"cpt_jyc_2026_stock_{frequency}"
            target.mkdir()
            for month, frame in frames.items():
                frame.to_parquet(target / f"{month}.parquet", index=False)
        universe = pd.concat([pd.DataFrame({"date": pd.to_datetime([f"{month}-02 09:30"] * len(stocks)),
                                           "instrument": list(stocks)})
                              for month, stocks in [("2020-01", "AB"), ("2020-02", "ABC"), ("2020-03", "ABC")]])
        universe.to_parquet(data / "cpt_jyc_2026_instruments.parquet", index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(training, data, config, protocol)
        return data, training

    def test_frozen_parameters_continuous_history_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            data, training = self.fixture(directory)
            run = Path(directory) / "heldout"
            before = {frequency: sha256(training / "normalization" / f"{frequency}.json") for frequency in FREQUENCIES}
            with contextlib.redirect_stdout(io.StringIO()), patch("v2_prepare.fit_file", side_effect=AssertionError("must not refit")):
                result = apply_holdout(training, run, data)
                resumed = apply_holdout(training, run, data, resume=True)
            self.assertEqual(result, resumed)
            for frequency in FREQUENCIES:
                self.assertEqual(sha256(training / "normalization" / f"{frequency}.json"), before[frequency])
                self.assertEqual(sha256(run / "normalization" / f"{frequency}.json"), before[frequency])
                validation = pd.read_parquet(run / "data/validation" / frequency / "2020-02.parquet").set_index("instrument")
                test = pd.read_parquet(run / "data/test" / frequency / "2020-03.parquet").set_index("instrument")
                self.assertAlmostEqual(validation.loc["A", "z_close"], 3.0, places=6)
                self.assertAlmostEqual(test.loc["A", "z_close"], 5.0, places=6)
                # C 的第一个观测在验证集；第二个观测在测试集，必须继承前一个观测。
                self.assertAlmostEqual(test.loc["C", "z_close"], 1.0, places=6)
                # 训练期对数价格为 [0, 1, 2, 3]，全体均值为 1.5，方差为 1.25。
                # C 在验证期仅有一次观测，必须采用冻结的全体股票统计量。
                expected_validation = np.array([3.0, 3.0, 1.0 / np.sqrt(5.0)])
                expected_test = np.array([5.0, 5.0, 1.0])
                for actual, expected in [(validation, expected_validation), (test, expected_test)]:
                    np.testing.assert_allclose(actual.loc[list("ABC"), "z_close"], expected, atol=1e-6)
                    expected_cross_section = (expected - expected.mean()) / expected.std(ddof=0)
                    np.testing.assert_allclose(actual.loc[list("ABC"), "c_close"], expected_cross_section, atol=1e-6)
                    self.assertEqual(actual.shape, (3, 73))
                    numeric = actual.drop(columns="date")
                    self.assertTrue(all(dtype == np.dtype("float32") for dtype in numeric.dtypes))
                    self.assertTrue(np.isfinite(numeric.to_numpy()).all())
                self.assertEqual(result["splits"]["validation"]["frequencies"][frequency]["pooled_values"], 36)
                self.assertEqual(result["splits"]["test"]["frequencies"][frequency]["dynamic_values"], 36)
            params = run / "normalization/bar1m.json"
            params.write_text(params.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "参数副本已改变"):
                apply_holdout(training, run, data, resume=True)

    def test_refuses_overwriting_training_and_invalid_split_order(self):
        with tempfile.TemporaryDirectory() as directory:
            data, training = self.fixture(directory)
            with self.assertRaisesRegex(ValueError, "独立目录"):
                apply_holdout(training, training, data)
            protocol_path = training / "protocol.json"
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            protocol["splits"]["test"]["start"] = "2020-02-15"
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "训练产物已改变"):
                apply_holdout(training, Path(directory) / "invalid", data)


if __name__ == "__main__":
    unittest.main()
