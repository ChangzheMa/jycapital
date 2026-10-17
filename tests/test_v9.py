"""v9 组合模型、监督目标、推理入口和确定性断点恢复检查。"""

import contextlib
import copy
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ("v9",)
sys.path.insert(0, str(ROOT / "solutions/v9"))
MODELS = {"v9": importlib.import_module("v9_model")}
DATA = {"v9": importlib.import_module("v9_data")}
TRAINERS = {"v9": importlib.import_module("v9_train")}
CONFIGS = {"v9": json.loads((ROOT / "solutions/v9/model-config.json").read_text(encoding="utf-8"))}
PROTOCOL = json.loads((ROOT / "evaluation/protocols/local-v2-ic.json").read_text(encoding="utf-8"))
FREQUENCIES = tuple(CONFIGS["v9"]["data"]["window_bars"])
torch.set_num_threads(1)


def small_config(version="v9"):
    config = copy.deepcopy(CONFIGS[version])
    config["model"].update(hidden_dim=4, fusion_hidden_dim=8, blocks=dict.fromkeys(FREQUENCIES, 1))
    config["model"]["kernel_sizes"] = {f: [3] for f in FREQUENCIES}
    config["training"].update(epochs=2, cpu_threads=1, checkpoint_batches=1, batch_size=40)
    return config


def sample_inputs(config, count=4, device="cpu"):
    generator = torch.Generator().manual_seed(55)
    windows = {f: torch.randn(count, 75, length, generator=generator).to(device)
               for f, length in config["data"]["window_bars"].items()}
    for x in windows.values():
        x[0] = 0
        x[1, :, -1] = 0
    clock = F.one_hot(torch.arange(count, device=device) % 8, 8).float()
    return windows, clock


class ModelTests(unittest.TestCase):
    def test_only_input_layout_and_branch_structure_change_from_v8(self):
        baseline = json.loads((ROOT / "solutions/v8/model-config.json").read_text(encoding="utf-8"))
        config = CONFIGS["v9"]
        self.assertEqual(config["training"], baseline["training"])
        self.assertEqual(config["inference"], baseline["inference"])
        self.assertEqual(config["seed"], baseline["seed"])
        self.assertEqual(config["data"]["prediction_clock"], baseline["data"]["prediction_clock"])
        expected = copy.deepcopy(baseline["model"])
        expected["input_channels"] = 75
        expected["blocks"].pop("bar15m")
        expected["kernel_sizes"].pop("bar15m")
        self.assertEqual(config["model"], expected)
        before = (ROOT / "solutions/v8/v8_loss.py").read_text(encoding="utf-8")
        self.assertEqual(before, (ROOT / "solutions/v9/v9_loss.py").read_text(encoding="utf-8"))

    def test_full_model_dimensions_gradients_and_no_hidden_mask(self):
        config = CONFIGS["v9"]
        TRAINERS["v9"].validate_training(config, PROTOCOL)
        model = MODELS["v9"].MultiFrequencyTCN(config).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 25458401)
        self.assertEqual(tuple(model.encoders), ("bar1m", "bar5m", "bar30m"))
        self.assertEqual(model.head[0].in_features, 488)
        self.assertFalse(any(isinstance(m, nn.LayerNorm) for m in model.modules()))
        for frequency, encoder in model.encoders.items():
            self.assertEqual(encoder.project.in_channels, 75)
            self.assertEqual(encoder.pool.in_features, 160 * config["data"]["window_bars"][frequency])
            for i, block in enumerate(encoder.blocks):
                self.assertEqual(block.padding, i + 1)
                self.assertEqual(block.dropout.p, .35)
                for conv in block.convolutions:
                    self.assertEqual(conv.kernel_size, (3 + 2 * i,))
                    self.assertEqual(conv.dilation, (1,))
        output = model(*sample_inputs(config, 2))
        self.assertEqual(output.shape, (2,))
        output.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        encoder = MODELS["v9"].FrequencyEncoder({**config["model"], "hidden_dim": 1}, [], 480)
        with torch.no_grad():
            encoder.project.bias.fill_(1)
            encoder.pool.weight.fill_(1)
            encoder.pool.bias.zero_()
        self.assertGreater(encoder(torch.zeros(1, 75, 480)).item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_default_cuda_batch_and_float32_validation(self):
        trainer, config = TRAINERS["v9"], CONFIGS["v9"]
        device = trainer.configure(config)
        model = MODELS["v9"].MultiFrequencyTCN(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                      weight_decay=config["training"]["weight_decay"])
        windows, clock = sample_inputs(config, 512, device)
        target = torch.linspace(-2, 2, 512, dtype=torch.float64, device=device)
        for _ in range(2):
            result = trainer.training_step(model, optimizer, windows, clock, target, config, torch.cuda.is_bf16_supported())
            self.assertTrue(np.isfinite(result["loss"]))
        model.eval()
        with torch.inference_mode():
            prediction = model(*sample_inputs(config, 1024, device))
        self.assertEqual(prediction.dtype, torch.float32)
        self.assertTrue(torch.isfinite(prediction).all())

    def test_cli_guard_and_untrained_code_parent(self):
        trainer = TRAINERS["v9"]
        self.assertIsNone(CONFIGS["v9"]["parent_run_id"])
        preserved = trainer.parent_snapshot(ROOT, CONFIGS["v9"])
        self.assertIn("solutions/v8/v8_model.py", preserved)
        self.assertFalse(any("outputs/runs" in key for key in preserved))
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "absent"
            args = ["v9_train.py", "--training-run", "unused", "--holdout-run", "unused", "--run-dir", str(run)]
            with patch.object(sys, "argv", args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                trainer.main()
            self.assertFalse(run.exists())


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.data, self.config = DATA["v9"], CONFIGS["v9"]
        # 人工交易日历含周末和长假，20 个交易日的跨度明显超过 20 个自然日。
        self.days = pd.bdate_range("2024-01-02", periods=45).delete(np.arange(10, 20))
        self.grids = {}
        for frequency in FREQUENCIES:
            times = self.data.regular_times(self.days, frequency)
            values = np.broadcast_to(np.arange(1, len(times) + 1, dtype=np.float32)[:, None, None],
                                      (len(times), 2, 72)).copy()
            self.grids[frequency] = times, values

    def block(self, times, grids=None):
        keys = pd.DataFrame({"date": pd.to_datetime(times), "instrument": "A"})
        return self.data.WindowBlock(keys, grids or self.grids, ["A", "B"], self.config, self.days)

    def test_all_eight_cutoffs_and_exact_previous_trading_days(self):
        date = self.days[-1]
        stamps = [date + pd.Timedelta(minutes=m) for m in (570, 600, 630, 660, 690, 810, 840, 870)]
        windows, clock = self.block(stamps).batch(np.arange(8))
        np.testing.assert_array_equal(clock, np.eye(8, dtype=np.float32))
        for i, completed in enumerate((0, 30, 60, 90, 120, 150, 180, 210)):
            np.testing.assert_array_equal(windows["bar1m"][i, 0, :240 + completed],
                                          self.grids["bar1m"][1][-480:-240 + completed or None, 0, 0])
            self.assertTrue((windows["bar1m"][i, :, 240 + completed:] == 0).all())
        for frequency, daily, previous in (("bar5m", 48, 5), ("bar30m", 8, 20)):
            expected = self.grids[frequency][1][-(previous + 1) * daily:-daily, 0, 0]
            np.testing.assert_array_equal(windows[frequency][:, 0], np.tile(expected, (8, 1)))
            self.assertTrue((windows[frequency][:, 72:75] == 1).all())
        self.assertGreater((date - self.days[-21]).days, 20)

    def test_missing_slots_partial_values_and_zero_values_have_distinct_flags(self):
        grids = {f: (t.copy(), v.copy()) for f, (t, v) in self.grids.items()}
        times, values = grids["bar1m"]
        previous_day = len(times) - 480
        values[previous_day] = np.nan
        values[previous_day + 1] = np.nan
        values[previous_day + 1, 0, 0] = 0  # 唯一有效值恰好为归一化零。
        values[previous_day + 2, 0] = 0
        remove = previous_day + 3
        grids["bar1m"] = np.delete(times, remove), np.delete(values, remove, axis=0)
        windows, _ = self.block([self.days[-1] + pd.Timedelta(hours=10)], grids).batch([0])
        x = windows["bar1m"][0]
        self.assertTrue((x[:, 0] == 0).all())
        np.testing.assert_array_equal(x[72:75, 1], [0, 0, 1])
        np.testing.assert_array_equal(x[72:75, 2], [1, 1, 1])
        self.assertTrue((x[:, 3] == 0).all())
        self.assertEqual(x[0, 4], values[previous_day + 4, 0, 0])

    def test_future_poison_and_current_day_exclusion(self):
        stamp = self.days[-1] + pd.Timedelta(hours=10)
        expected, _ = self.block([stamp]).batch([0])
        grids = {f: (t.copy(), v.copy()) for f, (t, v) in self.grids.items()}
        for frequency, (times, values) in grids.items():
            cutoff = stamp.value if frequency == "bar1m" else self.days[-1].value - 1
            values[times > cutoff] = 1e8
        actual, _ = self.block([stamp], grids).batch([0])
        for frequency in FREQUENCIES:
            np.testing.assert_array_equal(actual[frequency], expected[frequency])

    def test_insufficient_history_is_left_padded_and_boundary_bar_is_included(self):
        windows, _ = self.block([self.days[0] + pd.Timedelta(hours=10)]).batch([0])
        self.assertTrue((windows["bar1m"][:, :, :240] == 0).all())
        np.testing.assert_array_equal(windows["bar1m"][0, 0, 240:270], np.arange(1, 31))
        self.assertTrue((windows["bar1m"][:, :, 270:] == 0).all())
        self.assertTrue((windows["bar5m"] == 0).all())
        self.assertTrue((windows["bar30m"] == 0).all())
        for invalid in (self.days[-1] + pd.Timedelta(hours=10, seconds=1), pd.NaT):
            with self.assertRaises(ValueError):
                self.block([invalid])


class HuberTests(unittest.TestCase):
    def setUp(self):
        self.loss = importlib.import_module("v9_loss")

    def test_loss_and_gradients_match_direct_population_zscore(self):
        y = torch.tensor([.001, -.002, .004, float("nan"), .008], dtype=torch.float64)
        p = torch.tensor([.2, -.3, 1.7, 10., -1.], dtype=torch.float64, requires_grad=True)
        result = self.loss.standardized_huber_loss(p, y, 3, .7)
        eligible = torch.isfinite(y)
        z = (y[eligible] - y[eligible].mean()) / y[eligible].std(correction=0)
        expected = F.huber_loss(p[eligible], z, delta=.7)
        torch.testing.assert_close(result.loss, expected, rtol=1e-12, atol=1e-12)
        actual_grad = torch.autograd.grad(result.loss, p, retain_graph=True)[0]
        torch.testing.assert_close(actual_grad, torch.autograd.grad(expected, p)[0], rtol=1e-12, atol=1e-12)
        self.assertEqual(actual_grad[3].item(), 0)
        transformed = self.loss.standardized_huber_loss(p, y * 4 + 2, 3, .7)
        torch.testing.assert_close(result.loss, transformed.loss, rtol=1e-12, atol=1e-12)
        self.assertGreater(self.loss.standardized_huber_loss(p + 20, y, 3, .7).loss, result.loss)

    def test_degenerate_labels_and_constant_predictions(self):
        p = torch.zeros(4, dtype=torch.float64, requires_grad=True)
        y = torch.tensor([-3., -1., 1., 3.], dtype=torch.float64)
        result = self.loss.standardized_huber_loss(p, y, 3)
        self.assertIsNone(result.ic)
        result.loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(p.grad.abs().sum().item(), 0)
        self.assertEqual(self.loss.standardized_huber_loss(p, torch.ones(4), 3).reason, "constant_labels")
        self.assertEqual(self.loss.standardized_huber_loss(p, y, 5).reason, "insufficient_labels")
        tiny = self.loss.standardized_huber_loss(p, y * 1e-12, 3)
        torch.testing.assert_close(result.loss, tiny.loss)
        for delta in (0, -1, float("nan"), float("inf")):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                self.loss.standardized_huber_loss(p, y, 3, delta)

    def test_validation_huber_uses_each_full_section_and_equal_weight(self):
        trainer = TRAINERS["v9"]
        keys = pd.DataFrame({"date": pd.to_datetime(["2024-01-02 10:00"] * 3 + ["2024-01-02 10:30"] * 5),
                             "instrument": [f"S{i}" for i in range(8)],
                             "vwap_return": [-1., 0., 1., -8., -2., 3., 5., 10.]})
        pred = keys[["date", "instrument"]].copy()
        pred["factor"] = [0., 0., 0., -2., -1., 0., 1., 2.]
        protocol = {**PROTOCOL, "min_cross_section_size": 3}
        actual, undefined = trainer.huber_loss_check(pred, keys, protocol, CONFIGS["v9"])
        losses = [self.loss.standardized_huber_loss(torch.tensor(p.factor.to_numpy()),
                  torch.tensor(keys.loc[p.index, "vwap_return"].to_numpy()), 3).loss.item()
                  for _, p in pred.groupby("date")]
        self.assertAlmostEqual(actual, sum(losses) / 2)
        self.assertEqual(undefined, 1)

    def test_random_batches_keep_targets_from_their_original_full_sections(self):
        trainer = TRAINERS["v9"]
        times = pd.to_datetime(["2020-01-02 10:00"] * 3 + ["2020-01-02 10:30"] * 4
                               + ["2020-01-02 11:00"] * 2 + ["2020-01-02 13:30"] * 3)
        keys = pd.DataFrame({"date": times, "instrument": [f"S{i}" for i in range(12)],
                             "vwap_return": [-1., 0., 1., 100., 200., 300., np.nan, 7., 9., 5., 5., 5.]})
        protocol = {**PROTOCOL, "min_cross_section_size": 3}
        targets, skipped = trainer.prepare_training_targets(keys, protocol)
        np.testing.assert_allclose(targets[:3], targets[3:6], rtol=0, atol=1e-15)
        self.assertTrue(np.isnan(targets[6:]).all())
        self.assertEqual(sorted(skipped.values()), ["constant_labels", "insufficient_labels"])
        order = trainer.shuffled_sample_indices(targets, 42, 1, "2020-01")
        np.testing.assert_array_equal(np.sort(order), np.arange(6))
        np.testing.assert_array_equal(order, trainer.shuffled_sample_indices(targets, 42, 1, "2020-01"))
        self.assertFalse(np.array_equal(order, trainer.shuffled_sample_indices(targets, 42, 2, "2020-01")))
        selected = np.array([0, 3, 4, 2])
        scores = torch.tensor([.3, -.7, 1.2, .5], dtype=torch.float64, requires_grad=True)
        target = torch.from_numpy(targets[selected])
        loss = self.loss.sample_huber_loss(scores, target)
        torch.testing.assert_close(loss, F.huber_loss(scores, target, delta=1), rtol=0, atol=0)
        wrong = self.loss.standardized_huber_loss(scores, torch.tensor(keys.vwap_return.to_numpy()[selected]), 3).loss
        self.assertGreater(abs(loss.item() - wrong.item()), 1e-6)
        torch.testing.assert_close(torch.autograd.grad(loss, scores)[0],
                                   torch.clamp(scores.detach() - target, -1, 1) / len(scores))
        self.assertTrue(torch.isfinite(self.loss.sample_huber_loss(scores[:1], target[:1])))
        tiny, reason = self.loss.standardized_targets(torch.tensor([-1e-200, 0., 1e-200], dtype=torch.float64), 3)
        self.assertIsNone(reason)
        np.testing.assert_allclose(tiny.numpy(), targets[:3], rtol=0, atol=1e-15)

    def test_v9_rejects_old_batching_and_invalid_batch_sizes(self):
        trainer = TRAINERS["v9"]
        for changes in ({"batching": "full_cross_section"}, {"sections_per_step": 1},
                        {"checkpoint_sections": 64}, {"batch_size": 0}, {"batch_size": True},
                        {"batch_size": 1.5}, {"checkpoint_batches": 0}):
            config = copy.deepcopy(CONFIGS["v9"])
            config["training"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                trainer.validate_training(config, PROTOCOL)


class SyntheticBlock:
    def __init__(self, version, month, seed):
        config = small_config(version)
        rng = np.random.default_rng(seed)
        times = [pd.Timestamp(month + "-02") + pd.Timedelta(minutes=m) for m in (600, 630, 660)]
        self.keys = pd.DataFrame([(t, f"S{i:02}") for t in times for i in range(35)], columns=["date", "instrument"])
        self.windows = {f: rng.normal(size=(len(self.keys), 75, config["data"]["window_bars"][f])).astype(np.float32) for f in FREQUENCIES}
        for values in self.windows.values():
            values[:, 72:75] = 1
        self.keys["vwap_return"] = self.windows["bar1m"][:, 0, 299] * .001 + rng.normal(0, .001, len(self.keys))
        raw = self.keys.date.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        self.clock = DATA[version].prediction_clock_values(raw)

    def batch(self, indices):
        return {f: x[indices] for f, x in self.windows.items()}, self.clock[indices]


class SyntheticStore:
    protocol = PROTOCOL

    def __init__(self, version):
        self.data = {"train": {"2020-01": SyntheticBlock(version, "2020-01", 1)},
                     "validation": {"2024-01": SyntheticBlock(version, "2024-01", 3)},
                     "test": {"2024-07": SyntheticBlock(version, "2024-07", 4)}}

    def months(self, split):
        return list(self.data[split])

    def load(self, split, month):
        return self.data[split][month]


class TrainerIntegrationTests(unittest.TestCase):
    def test_combined_model_train_evaluate_and_resume_exactly(self):
        for version in VERSIONS:
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmp:
                trainer, config = TRAINERS[version], small_config(version)
                store = SyntheticStore(version)
                configure, step = trainer.configure, trainer.training_step

                def cpu_configure(spec):
                    configure(spec)
                    return torch.device("cpu")

                runs = [Path(tmp) / name for name in ("continuous", "resumed")]
                manifests = [{"status": "running"}, {"status": "running"}]
                for run in runs:
                    run.mkdir()
                    trainer.write_json(run / "data-manifest.json", {"synthetic_test": True})
                    trainer.write_json(run / "baseline-preservation.json", {})
                with patch.object(trainer, "configure", cpu_configure), \
                        patch.object(trainer, "environment", lambda: {"packages": {}}), \
                        patch.object(trainer, "draw_history"), patch.object(trainer, "log_event"):
                    trainer.train(store, runs[0], config, manifests[0])
                    calls = 0

                    def interrupt(*args, **kwargs):
                        nonlocal calls
                        calls += 1
                        if calls == 3:
                            raise KeyboardInterrupt
                        return step(*args, **kwargs)

                    with patch.object(trainer, "training_step", interrupt), self.assertRaises(KeyboardInterrupt):
                        trainer.train(store, runs[1], config, manifests[1])
                    trainer.train(store, runs[1], config, manifests[1], resume=True)
                for name in ("best.pt", "last.pt"):
                    left, right = [torch.load(run / name, weights_only=True) for run in runs]
                    for key in left["state_dict"]:
                        torch.testing.assert_close(left["state_dict"][key], right["state_dict"][key], rtol=0, atol=0)
                for manifest in manifests:
                    self.assertEqual(manifest["status"], "complete")
                    self.assertEqual(manifest["selection"]["optimizer_steps"], 6)
                    for split in ("validation", "test"):
                        self.assertEqual(manifest["metrics"][split]["coverage"], 1)
                first, second = [pd.read_csv(run / "history.csv") for run in runs]
                columns = ["train_loss", "validation_loss", "ic_mean", "rank_ic_mean",
                           "best_epoch", "patience_epochs", "stale_epochs"]
                if version != "v9":
                    columns.append("train_online_ic")
                pd.testing.assert_frame_equal(first[columns], second[columns], check_exact=True)
                if version == "v9":
                    self.assertNotIn("train_online_ic", first.columns)
                    self.assertTrue((first.train_rows == 105).all())
                    self.assertTrue((first.batch_size == 40).all())
                    self.assertEqual(left["progress"]["sample_cursor"], 0)
                    torch.testing.assert_close(left["torch_rng"], right["torch_rng"], rtol=0, atol=0)
                    self.assertEqual(left["optimizer"]["param_groups"], right["optimizer"]["param_groups"])
                    for parameter, values in left["optimizer"]["state"].items():
                        for key, value in values.items():
                            torch.testing.assert_close(value, right["optimizer"]["state"][parameter][key], rtol=0, atol=0)
                    self.assertNotIn("validation_ic_loss", first.columns)
                    self.assertIn("validation_huber_loss", first.columns)
                    report = trainer.read_json(runs[0] / "loss-consistency.json")
                    self.assertFalse(report["ic_loss_identity_applicable"])


if __name__ == "__main__":
    unittest.main()
