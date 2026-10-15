"""动态早停边界、严格新高，以及 v7 真实小模型训练的停止与断点恢复。"""

import copy
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
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(ROOT / "solutions/v7"))
from early_stopping import EARLY_STOPPING_POLICY, patience_epochs, select_epoch, should_stop
import v7_train as trainer
from test_v4_train import TinyModel, TinyStore

CONFIG = json.loads((ROOT / "solutions/v7/model-config.json").read_text(encoding="utf-8"))
PROTOCOL = json.loads((ROOT / "evaluation/protocols/local-v2-ic.json").read_text(encoding="utf-8"))


class EarlyStoppingTests(unittest.TestCase):
    def test_minimum_ceiling_and_maximum(self):
        expected = {0: 4, 1: 4, 6: 4, 7: 5, 8: 5, 9: 6, 10: 6,
                    11: 7, 17: 11, 19: 12, 20: 12, 100: 12}
        for epoch, patience in expected.items():
            with self.subTest(epoch=epoch):
                self.assertEqual(patience_epochs(epoch), patience)
        for invalid in (-1, 1.5, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                patience_epochs(invalid)

    def test_exact_stop_epoch_at_both_bounds_and_fractional_threshold(self):
        for best_epoch, expected_patience in ((1, 4), (6, 4), (7, 5), (10, 6), (20, 12)):
            state = trainer.initial_progress()
            for epoch in range(1, best_epoch + 1):
                select_epoch(state, epoch / 100, epoch)
                self.assertFalse(should_stop(state))
            for offset in range(1, expected_patience + 1):
                # 平分也属于没有提升。
                select_epoch(state, state["best"], best_epoch + offset)
                self.assertEqual(should_stop(state), offset == expected_patience)
            self.assertEqual(state["best_epoch"], best_epoch)
            self.assertEqual(state["stale"], expected_patience)

    def test_small_new_best_resets_counter_and_changes_patience(self):
        state = trainer.initial_progress()
        for epoch in range(1, 7):
            select_epoch(state, epoch / 100, epoch)
        for epoch in (7, 8, 9):
            select_epoch(state, .05, epoch)
        self.assertEqual(state["stale"], 3)
        self.assertTrue(select_epoch(state, .060000000001, 10))
        self.assertEqual(state["best_epoch"], 10)
        self.assertEqual(state["stale"], 0)
        self.assertEqual(patience_epochs(state["best_epoch"]), 6)
        for epoch in range(11, 17):
            select_epoch(state, .06, epoch)
            self.assertEqual(should_stop(state), epoch == 16)

    def test_ineligible_result_does_not_change_best_or_reset_counter(self):
        state = trainer.initial_progress()
        select_epoch(state, .1, 1)
        self.assertFalse(select_epoch(state, .9, 2, eligible=False))
        self.assertEqual((state["best"], state["best_epoch"], state["stale"]), (.1, 1, 1))
        unchanged = copy.deepcopy(state)
        for invalid in (None, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                select_epoch(state, invalid, 3)
            self.assertEqual(state, unchanged)

    def test_no_eligible_best_still_stops_after_four_epochs(self):
        state = trainer.initial_progress()
        for epoch in range(1, 5):
            select_epoch(state, .1, epoch, eligible=False)
            self.assertEqual(should_stop(state), epoch == 4)
        self.assertEqual(state["best_epoch"], 0)

    def test_training_config_rejects_old_or_different_policy(self):
        trainer.validate_training(CONFIG, PROTOCOL)
        for extra in ({"patience": 12}, {"min_delta": 1e-5}):
            config = copy.deepcopy(CONFIG)
            config["training"].update(extra)
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                trainer.validate_training(config, PROTOCOL)
        config = copy.deepcopy(CONFIG)
        config["training"]["early_stopping"]["rounding"] = "floor"
        with self.assertRaises(ValueError):
            trainer.validate_training(config, PROTOCOL)


class TrainIntegrationTests(unittest.TestCase):
    def run_comparison(self, epochs, interrupt_at_step=None):
        config = copy.deepcopy(CONFIG)
        config["training"].update(epochs=epochs, cpu_threads=1, checkpoint_sections=1)
        store = TinyStore()
        configure, predict, step = trainer.configure, trainer.predict_split, trainer.training_step

        def cpu_configure(spec):
            configure(spec)
            return torch.device("cpu")

        def controlled_validation(store, split, model, device, config, run=None):
            predictions, keys, result = predict(store, split, model, device, config, run)
            if split == "validation":
                epoch = torch.load(run / "last.pt", weights_only=True)["progress"]["epoch"]
                metric = epoch / 10 if epoch <= 7 else .65
            else:
                metric = .5
            # 仅控制验证曲线，使训练循环的停止时点可精确断言；输入及参数更新仍真实执行。
            predictions.attrs["controlled_ic"] = metric
            result[0].update(ic_mean=metric, rank_ic_mean=metric, coverage_violations=0)
            return predictions, keys, result

        def controlled_loss(predictions, keys, protocol):
            return 1 - predictions.attrs["controlled_ic"], 0

        with tempfile.TemporaryDirectory() as temporary:
            continuous, resumed = [Path(temporary) / name for name in ("continuous", "resumed")]
            manifests = []
            for run in (continuous, resumed):
                run.mkdir()
                trainer.write_json(run / "data-manifest.json", {"test": True})
                trainer.write_json(run / "baseline-preservation.json", {})
                manifests.append({"status": "running"})
            with patch.object(trainer, "MultiFrequencyTCN", TinyModel), \
                    patch.object(trainer, "configure", cpu_configure), \
                    patch.object(trainer, "environment", lambda: {"packages": {}}), \
                    patch.object(trainer, "predict_split", controlled_validation), \
                    patch.object(trainer, "ic_loss_check", controlled_loss), \
                    patch.object(trainer, "draw_history"), patch.object(trainer, "log_event"):
                trainer.train(store, continuous, config, manifests[0])
                if interrupt_at_step is not None:
                    count = 0

                    def interrupt(*args, **kwargs):
                        nonlocal count
                        count += 1
                        if count == interrupt_at_step:
                            raise KeyboardInterrupt
                        return step(*args, **kwargs)

                    with patch.object(trainer, "training_step", interrupt), self.assertRaises(KeyboardInterrupt):
                        trainer.train(store, resumed, config, manifests[1])
                    saved = torch.load(resumed / "last.pt", weights_only=True)["progress"]
                    self.assertEqual((saved["epoch"], saved["best_epoch"], saved["stale"]), (10, 7, 2))
                    trainer.train(store, resumed, config, manifests[1], resume=True)
                    for name in ("best.pt", "last.pt"):
                        first, second = [torch.load(run / name, weights_only=True) for run in (continuous, resumed)]
                        for key in first["state_dict"]:
                            self.assertTrue(torch.equal(first["state_dict"][key], second["state_dict"][key]), key)
                    self.assertEqual(manifests[1]["status"], "complete")
                    first, second = [pd.read_csv(run / "history.csv") for run in (continuous, resumed)]
                    fields = ["epoch", "best_epoch", "stale_epochs", "patience_epochs", "train_online_ic"]
                    np.testing.assert_array_equal(first[fields], second[fields])
                    for key in ("best_epoch", "epochs_completed", "patience_epochs", "stale_epochs", "stop_reason"):
                        self.assertEqual(manifests[0]["selection"][key], manifests[1]["selection"][key])
                history = pd.read_csv(continuous / "history.csv")
                checkpoint = torch.load(continuous / "last.pt", weights_only=True)
                self.assertEqual(manifests[0]["status"], "complete")
                return manifests[0]["selection"], history, checkpoint["progress"]

    def test_training_stops_at_twelve_and_resume_does_not_restart_patience(self):
        selection, history, progress = self.run_comparison(30, interrupt_at_step=55)
        self.assertEqual(selection["best_epoch"], 7)
        self.assertEqual(selection["epochs_completed"], 12)
        self.assertEqual(selection["patience_epochs"], 5)
        self.assertEqual(selection["stale_epochs"], 5)
        self.assertEqual(selection["stop_reason"], "early_stopping")
        self.assertEqual(selection["early_stopping"], EARLY_STOPPING_POLICY)
        self.assertEqual(progress["optimizer_steps"], 72)
        self.assertEqual(history.patience_epochs.tolist(), [4] * 6 + [5] * 6)
        self.assertEqual(history.stale_epochs.tolist(), [0] * 7 + [1, 2, 3, 4, 5])

    def test_max_epochs_can_end_before_early_stopping(self):
        selection, history, progress = self.run_comparison(3)
        self.assertEqual(selection["best_epoch"], 3)
        self.assertEqual(selection["epochs_completed"], 3)
        self.assertEqual(selection["patience_epochs"], 4)
        self.assertEqual(selection["stale_epochs"], 0)
        self.assertEqual(selection["stop_reason"], "max_epochs")
        self.assertEqual(progress["optimizer_steps"], 18)
        self.assertEqual(len(history), 3)


if __name__ == "__main__":
    unittest.main()
