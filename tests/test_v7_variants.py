"""六个 v7 派生实验的隔离性、汇总语义、损失及真实训练循环检查。"""

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
VERSIONS = ("v7a", "v7b", "v7c", "v7d", "v7e", "v7f")
for version in ("v7", *VERSIONS):
    sys.path.insert(0, str(ROOT / "solutions" / version))
MODELS = {v: importlib.import_module(v + "_model") for v in ("v7", *VERSIONS)}
DATA = {v: importlib.import_module(v + "_data") for v in ("v7", *VERSIONS)}
TRAINERS = {v: importlib.import_module(v + "_train") for v in VERSIONS}
CONFIGS = {v: json.loads((ROOT / "solutions" / v / "model-config.json").read_text(encoding="utf-8"))
           for v in ("v7", *VERSIONS)}
PROTOCOL = json.loads((ROOT / "evaluation/protocols/local-v2-ic.json").read_text(encoding="utf-8"))
FREQUENCIES = tuple(CONFIGS["v7"]["data"]["window_bars"])
torch.set_num_threads(1)


def small_config(version):
    config = copy.deepcopy(CONFIGS[version])
    config["data"]["window_bars"] = dict.fromkeys(FREQUENCIES, 4)
    config["model"].update(hidden_dim=4, fusion_hidden_dim=8, blocks=dict.fromkeys(FREQUENCIES, 1))
    if version == "v7d":
        config["model"]["kernel_sizes"] = {f: [3] for f in FREQUENCIES}
    config["training"].update(epochs=2, cpu_threads=1, checkpoint_sections=1)
    return config


def sample_inputs(config, count=35, device="cpu"):
    generator = torch.Generator().manual_seed(55)
    windows = {f: torch.randn(count, 77, length, generator=generator).to(device)
               for f, length in config["data"]["window_bars"].items()}
    for x in windows.values():
        x[:, 74:76, :] = 1
        x[0] = 0
        x[1, :, -1] = 0
    if config["solution_id"] == "v7f":
        clock = F.one_hot(torch.arange(count, device=device) % 8, 8).float()
    else:
        clock = torch.ones(count, 2, device=device)
    return windows, clock


class VariantModelTests(unittest.TestCase):
    def test_config_changes_are_limited_to_each_experiment(self):
        allowed_model = {"v7a": {"hidden_validity_mask", "pooling"},
                         "v7b": {"hidden_validity_mask", "pooling"},
                         "v7c": {"padding"}, "v7d": {"kernel_size", "kernel_sizes", "dilation"},
                         "v7e": set(), "v7f": {"prediction_clock_dim"}}
        groups = [[1], [1, 2], [3], [4], [5], [6]]
        for version, expected_groups in zip(VERSIONS, groups):
            with self.subTest(version=version):
                config = copy.deepcopy(CONFIGS[version])
                self.assertEqual(config["experiment"]["memo_groups"], expected_groups)
                self.assertEqual(config["parent_run_id"], "v7-20261016-no-layernorm-s42")
                for key in ("solution_id", "parent_run_id", "references", "experiment"):
                    config.pop(key, None)
                baseline = copy.deepcopy(CONFIGS["v7"])
                for key in ("solution_id", "parent_run_id", "references"):
                    baseline.pop(key)
                for key in allowed_model[version]:
                    config["model"].pop(key, None)
                    baseline["model"].pop(key, None)
                if version == "v7e":
                    for key in ("loss", "target_transform", "huber_delta", "label_ddof",
                                "structural_policy", "constant_prediction_policy"):
                        config["training"].pop(key, None)
                        baseline["training"].pop(key, None)
                    config["inference"]["factor_scale"] = baseline["inference"]["factor_scale"]
                if version == "v7f":
                    config["data"].pop("prediction_clock")
                    config["data"]["time_encoding"] = baseline["data"]["time_encoding"]
                self.assertEqual(config, baseline)

    def test_parameters_shapes_forward_backward_and_checkpoint_roundtrip(self):
        expected = {"v7a": 2421185, "v7b": 5185985, "v7c": 2421185,
                    "v7d": 3957185, "v7e": 2421185, "v7f": 2422721}
        for version in VERSIONS:
            with self.subTest(version=version):
                config = CONFIGS[version]
                trainer = TRAINERS[version]
                trainer.validate_training(config, PROTOCOL)
                model = MODELS[version].MultiFrequencyTCN(config).eval()
                self.assertEqual(sum(p.numel() for p in model.parameters()), expected[version])
                self.assertFalse(any(isinstance(m, nn.LayerNorm) for m in model.modules()))
                windows, clock = sample_inputs(config, count=4)
                output = model(windows, clock)
                self.assertEqual(output.shape, (4,))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                clone = MODELS[version].MultiFrequencyTCN(config).eval()
                clone.load_state_dict(model.state_dict())
                torch.testing.assert_close(clone(windows, clock), output, rtol=0, atol=0)

    def test_unchanged_weights_and_all_valid_no_mask_equivalence(self):
        torch.manual_seed(42)
        baseline = MODELS["v7"].MultiFrequencyTCN(CONFIGS["v7"]).eval()
        windows, clock = sample_inputs(CONFIGS["v7"], 4)
        for x in windows.values():
            x[:, 74:76] = 1
        for version in ("v7a", "v7c", "v7e"):
            with self.subTest(version=version):
                torch.manual_seed(42)
                model = MODELS[version].MultiFrequencyTCN(CONFIGS[version]).eval()
                for key, value in baseline.state_dict().items():
                    torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
                if version != "v7c":
                    torch.testing.assert_close(model(windows, clock), baseline(windows, clock), rtol=0, atol=0)

    def test_no_mask_pools_include_empty_slots_and_last_slot(self):
        for version in ("v7a", "v7b"):
            config = small_config(version)
            spec = {**config["model"], "hidden_dim": 1}
            cls = MODELS[version].FrequencyEncoder
            encoder = cls(spec, 0, 4) if version == "v7b" else cls(spec, 0)
            with torch.no_grad():
                encoder.project.weight.zero_()
                encoder.project.weight[0, 0, 0] = 1
                encoder.project.bias.fill_(1)
                encoder.pool.weight.fill_(1)
                encoder.pool.bias.zero_()
            x = torch.zeros(1, 77, 4)
            x[0, 0] = torch.tensor([2., 8., 4., 9.])
            captured = []
            hook = encoder.pool.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
            output = encoder(x)
            hook.remove()
            hidden = F.gelu(x[:, 0] + 1)
            expected = hidden if version == "v7b" else torch.stack([hidden[:, -1], hidden.mean(1)], dim=1)
            torch.testing.assert_close(captured[0], expected)
            self.assertGreater(output.item(), 0)
            self.assertGreater(encoder(torch.zeros_like(x)).item(), 0)
            block = MODELS[version].CausalBlock(1, 3, 1, 0)
            with torch.no_grad():
                for conv in block.convolutions:
                    conv.weight.zero_()
                    conv.bias.zero_()
            torch.testing.assert_close(block(torch.ones(1, 1, 4)), torch.ones(1, 1, 4))
        with self.assertRaises(ValueError):
            encoder(torch.zeros(1, 77, 5))

    def test_symmetric_replication_matches_explicit_endpoint_convolution(self):
        block = MODELS["v7c"].SymmetricBlock(1, 3, 2, 0).eval()
        with torch.no_grad():
            for conv in block.convolutions:
                conv.weight.fill_(1)
                conv.bias.zero_()
        x = torch.tensor([[[1., 2., 3., 4., 5.]]])
        expected = x.clone()
        for _ in range(2):
            expected = F.gelu(torch.stack([
                sum(expected[..., min(max(t + shift, 0), 4)] for shift in (-2, 0, 2))
                for t in range(5)], dim=-1))
        torch.testing.assert_close(block(x, torch.ones_like(x)), expected + x)
        for length in (1, 2, 16, 60):
            test = torch.ones(2, 1, length)
            self.assertEqual(block(test, torch.ones_like(test)).shape, test.shape)

    def test_wider_kernels_keep_dilation_one_and_causal_padding(self):
        model = MODELS["v7d"].MultiFrequencyTCN(CONFIGS["v7d"]).eval()
        for frequency, encoder in model.encoders.items():
            for i, block in enumerate(encoder.blocks):
                self.assertEqual(block.left, 2 + 2 * i)
                for conv in block.convolutions:
                    self.assertEqual(conv.kernel_size, (3 + 2 * i,))
                    self.assertEqual(conv.dilation, (1,))
        block = MODELS["v7d"].CausalBlock(1, 9, 1, 0).eval()
        first = torch.randn(1, 1, 12)
        second = first.clone()
        second[..., 7:] += 10
        torch.testing.assert_close(block(first, torch.ones_like(first))[..., :7],
                                   block(second, torch.ones_like(second))[..., :7], rtol=0, atol=0)

    def test_window_inputs_unchanged_and_one_hot_clock_exact(self):
        times = pd.to_datetime(["2024-01-03 " + t for t in DATA["v7f"].PREDICTION_TIMES])
        keys = pd.DataFrame({"date": times, "instrument": "A"})
        rng = np.random.default_rng(73)
        grids = {}
        for frequency in FREQUENCIES:
            grid_times = DATA["v7"].regular_times(pd.to_datetime(["2024-01-02", "2024-01-03"]), frequency)
            values = rng.normal(size=(len(grid_times), 1, 72)).astype(np.float32)
            values[rng.random(values.shape) < .1] = np.nan
            grids[frequency] = grid_times, values
        baseline, clock = DATA["v7"].WindowBlock(keys, grids, ["A"], CONFIGS["v7"]).batch(np.arange(8))
        for version in VERSIONS:
            with self.subTest(version=version):
                windows, actual = DATA[version].WindowBlock(keys, grids, ["A"], CONFIGS[version]).batch(np.arange(8))
                for frequency in FREQUENCIES:
                    np.testing.assert_array_equal(windows[frequency], baseline[frequency])
                np.testing.assert_array_equal(actual, np.eye(8, dtype=np.float32) if version == "v7f" else clock)
        for invalid in ("2024-01-03 10:03", "2024-01-03 10:00:01", "NaT"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                DATA["v7f"].prediction_clock_values(np.array([invalid], dtype="datetime64[ns]").astype(np.int64))

    def test_cli_requires_explicit_training_and_writes_nothing(self):
        for version in VERSIONS:
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmp:
                run = Path(tmp) / "absent"
                args = [version + "_train.py", "--training-run", "unused", "--holdout-run", "unused", "--run-dir", str(run)]
                with patch.object(sys, "argv", args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    TRAINERS[version].main()
                self.assertFalse(run.exists())


class HuberTests(unittest.TestCase):
    def setUp(self):
        self.loss = importlib.import_module("v7e_loss")

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
        trainer = TRAINERS["v7e"]
        keys = pd.DataFrame({"date": pd.to_datetime(["2024-01-02 10:00"] * 3 + ["2024-01-02 10:30"] * 5),
                             "instrument": [f"S{i}" for i in range(8)],
                             "vwap_return": [-1., 0., 1., -8., -2., 3., 5., 10.]})
        pred = keys[["date", "instrument"]].copy()
        pred["factor"] = [0., 0., 0., -2., -1., 0., 1., 2.]
        protocol = {**PROTOCOL, "min_cross_section_size": 3}
        actual, undefined = trainer.huber_loss_check(pred, keys, protocol, CONFIGS["v7e"])
        losses = [self.loss.standardized_huber_loss(torch.tensor(p.factor.to_numpy()),
                  torch.tensor(keys.loc[p.index, "vwap_return"].to_numpy()), 3).loss.item()
                  for _, p in pred.groupby("date")]
        self.assertAlmostEqual(actual, sum(losses) / 2)
        self.assertEqual(undefined, 1)


class PredictionTests(unittest.TestCase):
    def test_each_namespace_predicts_from_checkpoint_and_text_weights(self):
        from test_v3_predict import FakeDai
        prep = importlib.import_module("v7_preprocessing")
        days = pd.date_range("2024-01-02", periods=2)
        minutes = (570, 600, 630, 660, 690, 810, 840, 870)
        rows = [(day + pd.Timedelta(minutes=m), stock) for day in days for m in minutes for stock in ("A", "B")]
        tables = {"cpt_jyc_2026_instruments": pd.DataFrame(rows, columns=["date", "instrument"])}
        rng = np.random.default_rng(82)
        for frequency in FREQUENCIES:
            times = pd.to_datetime(DATA["v7"].regular_times(days, frequency))
            bars = pd.DataFrame([(t, s) for t in times for s in ("A", "B")], columns=["date", "instrument"])
            for field in prep.FIELDS:
                bars[field] = rng.uniform(1, 20, len(bars))
            tables["logical_" + frequency] = bars
        moments = prep.pack_moments(prep.moments_from_values(np.stack([np.zeros(36), np.ones(36)] * 30)))
        stats = {"fields": prep.FIELDS, "fit_interval": {"start": "2020-01-01", "stop": "2024-01-01"},
                 "stocks": {"A": moments, "B": moments}, "global": moments}
        for version in VERSIONS:
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                (directory / "normalization").mkdir()
                preprocessing = json.loads((ROOT / "solutions" / version / "config.json").read_text(encoding="utf-8"))
                (directory / "preprocessing-config.json").write_text(json.dumps(preprocessing), encoding="utf-8")
                for frequency in FREQUENCIES:
                    (directory / "normalization" / (frequency + ".json")).write_text(json.dumps(stats), encoding="utf-8")
                config = small_config(version)
                model = MODELS[version].MultiFrequencyTCN(config)
                torch.save({"config": config, "state_dict": model.state_dict()}, directory / "best.pt")
                inference = importlib.import_module(version + "_predict")
                artifact = importlib.import_module(version + "_text_artifact")
                sources = {f: "logical_" + f for f in FREQUENCIES}
                dai = FakeDai(tables)
                checkpoint_result = inference.predict(sources, "2024-01-02", "2024-01-03", directory, dai, device="cpu")
                bundle = {"format": artifact.FORMAT, "config": config, "preprocessing": preprocessing,
                          "normalization": dict.fromkeys(FREQUENCIES, stats),
                          "state_dict": artifact.encode_state(model.state_dict()),
                          "tensor_sha256": artifact.state_digest(model.state_dict())}
                text_result = inference.predict(sources, "2024-01-02", "2024-01-03", directory, dai,
                                                device="cpu", bundle=bundle)
                pd.testing.assert_frame_equal(checkpoint_result, text_result, check_exact=True)
                self.assertEqual(len(checkpoint_result), len(rows))
                self.assertEqual(checkpoint_result.factor.dtype, np.dtype("float64"))
                self.assertTrue(np.isfinite(checkpoint_result.factor).all())
                self.assertFalse(checkpoint_result.duplicated(["date", "instrument"]).any())
                self.assertFalse(any("vwap" in q for q in dai.queries))


class SyntheticBlock:
    def __init__(self, version, month, seed):
        config = small_config(version)
        rng = np.random.default_rng(seed)
        times = [pd.Timestamp(month + "-02") + pd.Timedelta(minutes=m) for m in (600, 630, 660)]
        self.keys = pd.DataFrame([(t, f"S{i:02}") for t in times for i in range(35)], columns=["date", "instrument"])
        self.windows = {f: rng.normal(size=(len(self.keys), 77, 4)).astype(np.float32) for f in FREQUENCIES}
        for values in self.windows.values():
            values[:, 74:76] = 1
        self.keys["vwap_return"] = self.windows["bar1m"][:, 0, -1] * .001 + rng.normal(0, .001, len(self.keys))
        raw = self.keys.date.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        self.clock = DATA[version].prediction_clock_values(raw) if version == "v7f" else DATA[version].clock_values(raw)

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
    def test_all_six_real_models_train_evaluate_and_resume_exactly(self):
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
                columns = ["train_loss", "validation_loss", "train_online_ic", "ic_mean", "rank_ic_mean",
                           "best_epoch", "patience_epochs", "stale_epochs"]
                pd.testing.assert_frame_equal(first[columns], second[columns], check_exact=True)
                if version == "v7e":
                    self.assertNotIn("validation_ic_loss", first.columns)
                    self.assertIn("validation_huber_loss", first.columns)
                    report = trainer.read_json(runs[0] / "loss-consistency.json")
                    self.assertFalse(report["ic_loss_identity_applicable"])


if __name__ == "__main__":
    unittest.main()
