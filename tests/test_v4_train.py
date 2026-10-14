"""真实小模型更新与恢复，及 IC 选模、12 轮早停计数。"""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solutions/v4"))
import v4_train as trainer

CONFIG = json.loads((ROOT / "solutions/v4/model-config.json").read_text(encoding="utf-8"))
PROTOCOL = json.loads((ROOT / "evaluation/protocols/local-v2-ic.json").read_text(encoding="utf-8"))


class TinyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(.2)
        self.linear = nn.Linear(77, 1)

    def forward(self, windows, clock):
        return self.linear(self.dropout(windows["bar1m"][:, :, -1])).squeeze(1)


class TinyBlock:
    def __init__(self, month, seed):
        rng = np.random.default_rng(seed)
        dates = [pd.Timestamp(month+"-02")+pd.Timedelta(minutes=m) for m in (600, 630, 660)]
        self.keys = pd.DataFrame([(t, f"S{i:02}") for t in dates for i in range(35)],
                                 columns=["date", "instrument"])
        self.values = rng.normal(size=(len(self.keys), 77, 4)).astype(np.float32)
        self.keys["vwap_return"] = self.values[:, 0, -1]*.001 + rng.normal(0, .001, len(self.keys))

    def batch(self, indices):
        return {"bar1m": self.values[indices]}, np.ones((len(indices), 2), np.float32)


class TinyStore:
    protocol = PROTOCOL

    def __init__(self):
        self.data = {"train": {"2020-01": TinyBlock("2020-01", 1), "2020-02": TinyBlock("2020-02", 2)},
                     "validation": {"2024-01": TinyBlock("2024-01", 3)},
                     "test": {"2024-07": TinyBlock("2024-07", 4)}}

    def months(self, split):
        return list(self.data[split])

    def load(self, split, month):
        return self.data[split][month]


class TrainTests(unittest.TestCase):
    def test_ic_selection_and_exact_twelve_stale_epochs(self):
        state = trainer.initial_progress()
        spec = CONFIG["training"]
        self.assertTrue(trainer.select_epoch(state, .1, 1, spec))
        for epoch in range(2, 14):
            self.assertFalse(trainer.select_epoch(state, .09, epoch, spec))
        self.assertEqual(state["stale"], 12)
        self.assertEqual(state["best_epoch"], 1)
        self.assertTrue(trainer.select_epoch(state, .100001, 14, spec))
        self.assertEqual(state["stale"], 13)
        self.assertTrue(trainer.select_epoch(state, .10002, 15, spec))
        self.assertEqual(state["stale"], 0)
        self.assertFalse(trainer.select_epoch(state, .2, 16, spec, eligible=False))
        self.assertEqual(state["best_epoch"], 15)

    def test_resume_preserves_updates_and_dropout_rng(self):
        config = copy.deepcopy(CONFIG)
        config["training"].update(epochs=2, cpu_threads=1, checkpoint_sections=1)
        store = TinyStore()
        configure = trainer.configure

        def cpu_configure(spec):
            configure(spec)
            return torch.device("cpu")

        with tempfile.TemporaryDirectory() as temporary:
            continuous, interrupted = [Path(temporary)/name for name in ("continuous", "interrupted")]
            manifests = []
            for run in (continuous, interrupted):
                run.mkdir()
                trainer.write_json(run/"data-manifest.json", {"test": True})
                trainer.write_json(run/"baseline-preservation.json", {})
                manifests.append({"status": "running"})
            with patch.object(trainer, "MultiFrequencyTCN", TinyModel), \
                    patch.object(trainer, "configure", cpu_configure), \
                    patch.object(trainer, "environment", lambda: {"packages": {}}), \
                    patch.object(trainer, "draw_history"), patch.object(trainer, "log_event"):
                trainer.train(store, continuous, config, manifests[0])
                step = trainer.training_step
                count = 0

                def stop_before_third(*args, **kwargs):
                    nonlocal count
                    count += 1
                    if count == 3:
                        raise KeyboardInterrupt
                    return step(*args, **kwargs)

                with patch.object(trainer, "training_step", stop_before_third), self.assertRaises(KeyboardInterrupt):
                    trainer.train(store, interrupted, config, manifests[1])
                partial = torch.load(interrupted/"last.pt", weights_only=True)
                self.assertEqual(partial["progress"]["optimizer_steps"], 2)
                self.assertEqual(partial["progress"]["section_cursor"], 2)
                trainer.train(store, interrupted, config, manifests[1], resume=True)
            for filename in ("best.pt", "last.pt"):
                left = torch.load(continuous/filename, weights_only=True)
                right = torch.load(interrupted/filename, weights_only=True)
                for key in left["state_dict"]:
                    self.assertTrue(torch.equal(left["state_dict"][key], right["state_dict"][key]), key)
            fields = ["train_online_ic_loss", "train_online_ic", "validation_ic_mean", "validation_rank_ic_mean"]
            first, second = [pd.read_csv(run/"history.csv") for run in (continuous, interrupted)]
            np.testing.assert_array_equal(first[fields], second[fields])
            self.assertEqual(torch.load(interrupted/"last.pt", weights_only=True)["progress"]["optimizer_steps"], 12)
            self.assertTrue((np.abs(second.validation_ic_loss + second.validation_ic_mean - 1) < 1e-8).all())
            self.assertEqual(manifests[1]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
