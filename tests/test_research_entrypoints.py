"""临时合成 Parquet 上的预处理、留出集转换和训练主入口完整衔接。"""

import contextlib
import importlib
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

from test_research_branches import ROOT, VERSIONS, small_config


class EntrypointTests(unittest.TestCase):
    config_factory = staticmethod(small_config)

    def test_three_families_full_synthetic_entrypoints(self):
        for version in VERSIONS:
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmp:
                sys.path.insert(0, str(ROOT / "solutions" / version))
                prep = importlib.import_module(version + "_prepare")
                apply = importlib.import_module(version + "_apply")
                trainer = importlib.import_module(version + "_train")
                root = Path(tmp)
                data = root / "data"
                data.mkdir()
                config = self.config_factory(version)
                preprocessing = json.loads((ROOT / "solutions" / version / "config.json").read_text(encoding="utf-8"))
                preprocessing["processing"].update(threads=1, batch_rows=256)
                config["training"].update(epochs=1, cpu_threads=1, batch_size=8, checkpoint_batches=1)
                protocol = json.loads((ROOT / "evaluation/protocols/local-v2-ic.json").read_text(encoding="utf-8"))
                protocol["min_cross_section_size"] = 3
                protocol["splits"] = {
                    "train": {"start": "2020-01-01", "stop": "2020-02-01"},
                    "validation": {"start": "2020-02-01", "stop": "2020-03-01"},
                    "test": {"start": "2020-03-01", "stop": "2020-04-01"},
                }
                days = pd.to_datetime(["2020-01-02", "2020-02-03", "2020-03-02"])
                stocks = [f"S{i}" for i in range(6)]
                minutes = (570, 600, 630, 660, 690, 810, 840, 870)
                universe = pd.DataFrame([(d + pd.Timedelta(minutes=m), s) for d in days for m in minutes for s in stocks],
                                        columns=["date", "instrument"])
                universe.to_parquet(data / "cpt_jyc_2026_instruments.parquet", index=False)
                rng = np.random.default_rng(161)
                labels = universe.copy()
                labels["vwap_return"] = rng.normal(0, .001, len(labels))
                labels.loc[labels.date.dt.hour.eq(9), "vwap_return"] = np.nan
                labels.to_parquet(data / "cpt_jyc_2026_vwap.parquet", index=False)
                regular_times = importlib.import_module(version + "_data").regular_times
                for frequency in prep.FREQUENCIES:
                    directory = data / ("cpt_jyc_2026_stock_" + frequency)
                    directory.mkdir()
                    for day in days:
                        times = pd.to_datetime(regular_times([day], frequency))
                        frame = pd.DataFrame([(t, s) for t in times for s in stocks], columns=["date", "instrument"])
                        for field in prep.FIELDS:
                            frame[field] = rng.uniform(1, 20, len(frame))
                        frame.to_parquet(directory / (str(day.to_period("M")) + ".parquet"), index=False)
                training, holdout, run = (root / name for name in ("training", "holdout", "run"))
                config_path, protocol_path = root / "model-config.json", root / "protocol.json"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
                configure = trainer.configure

                def cpu_configure(spec):
                    configure(spec)
                    return torch.device("cpu")

                argv = [version + "_train.py", "--config", str(config_path), "--protocol", str(protocol_path),
                        "--training-run", str(training), "--holdout-run", str(holdout), "--data-root", str(data),
                        "--run-dir", str(run), "--training-confirmed", "--reproduction-only"]
                with contextlib.redirect_stdout(io.StringIO()):
                    prep.prepare(training, data, preprocessing, protocol)
                    apply.apply_holdout(training, holdout, data)
                    with patch.object(trainer, "configure", cpu_configure), \
                            patch.object(trainer, "environment", lambda: {"packages": {}}), patch.object(sys, "argv", argv):
                        trainer.main()
                manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(manifest["stage"], "complete")
                self.assertEqual(manifest["solution_id"], version)
                for split in ("validation", "test"):
                    self.assertEqual(manifest["metrics"][split]["coverage"], 1)
                    predictions = pd.read_parquet(run / "evaluation" / split / "predictions.parquet")
                    self.assertEqual(predictions.factor.dtype, np.dtype("float64"))
                    self.assertTrue(np.isfinite(predictions.factor).all())
                    self.assertFalse(predictions.duplicated(["date", "instrument"]).any())
                self.assertTrue((run / "learning-curves.png").is_file())
                self.assertIn(f"solutions/{version}/{version}_model.py", manifest["source"]["files"])
                self.assertEqual(manifest["parent_solution_id"], "v9")
                self.assertIsNone(manifest["parent_run_id"])
                self.assertFalse(any("bar15m" in item["path"]
                                     for item in trainer.read_json(run / "data-manifest.json")["files"]))


if __name__ == "__main__":
    unittest.main()
