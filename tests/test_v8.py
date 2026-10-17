"""v8 组合模型、监督目标、推理入口和确定性断点恢复检查。"""

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
VERSIONS = ("v8",)
sys.path.insert(0, str(ROOT / "solutions/v8"))
MODELS = {"v8": importlib.import_module("v8_model")}
DATA = {"v8": importlib.import_module("v8_data")}
TRAINERS = {"v8": importlib.import_module("v8_train")}
CONFIGS = {"v8": json.loads((ROOT / "solutions/v8/model-config.json").read_text(encoding="utf-8"))}
PROTOCOL = json.loads((ROOT / "evaluation/protocols/local-v2-ic.json").read_text(encoding="utf-8"))
FREQUENCIES = tuple(CONFIGS["v8"]["data"]["window_bars"])
torch.set_num_threads(1)


def small_config(version="v8"):
    config = copy.deepcopy(CONFIGS[version])
    config["data"]["window_bars"] = dict.fromkeys(FREQUENCIES, 4)
    config["model"].update(hidden_dim=4, fusion_hidden_dim=8, blocks=dict.fromkeys(FREQUENCIES, 1))
    config["model"]["kernel_sizes"] = {f: [3] for f in FREQUENCIES}
    config["training"].update(epochs=2, cpu_threads=1, checkpoint_batches=1, batch_size=40)
    return config


def sample_inputs(config, count=4, device="cpu"):
    generator = torch.Generator().manual_seed(55)
    windows = {f: torch.randn(count, 77, length, generator=generator).to(device)
               for f, length in config["data"]["window_bars"].items()}
    for x in windows.values():
        x[0] = 0
        x[1, :, -1] = 0
    clock = F.one_hot(torch.arange(count, device=device) % 8, 8).float()
    return windows, clock


class CombinedModelTests(unittest.TestCase):
    def test_config_is_exactly_the_six_memo_groups(self):
        def read(version):
            return json.loads((ROOT / "solutions" / version / "model-config.json").read_text(encoding="utf-8"))

        expected = read("v7e")
        expected["solution_id"] = "v8"
        expected["experiment"]["memo_groups"] = [1, 2, 3, 4, 5, 6]
        expected["model"].update({key: read("v7b")["model"][key]
                                  for key in ("hidden_validity_mask", "pooling")})
        expected["model"]["padding"] = read("v7c")["model"]["padding"]
        expected["model"].pop("kernel_size")
        expected["model"].update({key: read("v7d")["model"][key] for key in ("kernel_sizes", "dilation")})
        expected["model"]["prediction_clock_dim"] = read("v7f")["model"]["prediction_clock_dim"]
        expected["data"].update({key: read("v7f")["data"][key] for key in ("time_encoding", "prediction_clock")})
        self.assertEqual(CONFIGS["v8"], expected)
        for suffix in ("loss.py", "preprocessing.py", "prepare.py", "apply.py", "predict.py",
                       "text_artifact.py", "notebook.py", "reproduce.py"):
            before = (ROOT / "solutions/v7e" / ("v7e_" + suffix)).read_text(encoding="utf-8").replace("v7e", "VERSION")
            after = (ROOT / "solutions/v8" / ("v8_" + suffix)).read_text(encoding="utf-8").replace("v8", "VERSION")
            self.assertEqual(before, after, suffix)

    def test_full_model_structure_forward_backward_and_roundtrip(self):
        config = CONFIGS["v8"]
        TRAINERS["v8"].validate_training(config, PROTOCOL)
        self.assertEqual(config["experiment"]["memo_groups"], [1, 2, 3, 4, 5, 6])
        model = MODELS["v8"].MultiFrequencyTCN(config).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 6723521)
        self.assertFalse(any(isinstance(m, nn.LayerNorm) for m in model.modules()))
        self.assertEqual(model.head[0].in_features, 648)
        for frequency, encoder in model.encoders.items():
            self.assertEqual(encoder.pool.in_features, 160 * config["data"]["window_bars"][frequency])
            self.assertEqual(len(encoder.blocks), config["model"]["blocks"][frequency])
            for i, block in enumerate(encoder.blocks):
                self.assertEqual(block.padding, i + 1)
                self.assertEqual(block.dropout.p, .35)
                self.assertEqual(len(block.convolutions), 2)
                self.assertIsNot(block.convolutions[0].weight, block.convolutions[1].weight)
                for conv in block.convolutions:
                    self.assertEqual(conv.kernel_size, (3 + 2 * i,))
                    self.assertEqual(conv.dilation, (1,))
                    self.assertEqual(conv.stride, (1,))
        windows, clock = sample_inputs(config)
        output = model(windows, clock)
        self.assertEqual(output.shape, (4,))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        artifact = importlib.import_module("v8_text_artifact")
        small = MODELS["v8"].MultiFrequencyTCN(small_config()).eval()
        bundle = json.loads(json.dumps({"format": artifact.FORMAT, "state_dict": artifact.encode_state(small.state_dict())}))
        restored = artifact.decode_state(bundle)
        self.assertEqual(artifact.state_digest(small.state_dict()), artifact.state_digest(restored))
        clone = MODELS["v8"].MultiFrequencyTCN(small_config()).eval()
        clone.load_state_dict(restored)
        inputs = sample_inputs(small_config())
        torch.testing.assert_close(small(*inputs), clone(*inputs), rtol=0, atol=0)

    def test_no_hidden_mask_and_channel_time_flatten_order(self):
        spec = {**small_config()["model"], "hidden_dim": 2, "tcn_dropout": 0}
        encoder = MODELS["v8"].FrequencyEncoder(spec, [3, 5], 4).eval()
        with torch.no_grad():
            encoder.project.weight.zero_()
            encoder.project.weight[:, 0, 0] = torch.tensor([1., 2.])
            encoder.project.bias.fill_(1)
            encoder.pool.weight.fill_(1)
            encoder.pool.bias.zero_()
            for block in encoder.blocks:
                for conv in block.convolutions:
                    conv.weight.zero_()
                    conv.bias.fill_(.5)
        x = torch.zeros(1, 77, 4)
        x[0, 0] = torch.tensor([2., 8., 4., 9.])
        captured = []
        hook = encoder.pool.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
        output = encoder(x)
        hook.remove()
        expected = F.gelu(encoder.project(x)) + 2 * F.gelu(torch.tensor(.5))
        torch.testing.assert_close(captured[0], expected.flatten(start_dim=1))
        self.assertTrue((output > 0).all())
        self.assertTrue((encoder(torch.zeros_like(x)) > 0).all())
        with self.assertRaises(ValueError):
            encoder(torch.zeros(1, 77, 5))

    def test_wide_symmetric_padding_matches_endpoint_reference(self):
        for kernel in (3, 5, 7, 9):
            for length in (1, 2, 5):
                with self.subTest(kernel=kernel, length=length):
                    block = MODELS["v8"].SymmetricBlock(1, kernel, 0).eval()
                    with torch.no_grad():
                        for conv in block.convolutions:
                            conv.weight.fill_(1)
                            conv.bias.zero_()
                    x = torch.arange(1, length + 1, dtype=torch.float32).reshape(1, 1, length)
                    expected = x.clone()
                    half = kernel // 2
                    for _ in range(2):
                        expected = F.gelu(torch.stack([
                            sum(expected[..., min(max(t + shift, 0), length - 1)]
                                for shift in range(-half, half + 1)) for t in range(length)
                        ], dim=-1))
                    torch.testing.assert_close(block(x), expected + x)

    def test_window_inputs_match_parent_and_exclude_future_bars(self):
        sys.path.insert(0, str(ROOT / "solutions/v7"))
        parent = importlib.import_module("v7_data")
        parent_config = json.loads((ROOT / "solutions/v7/model-config.json").read_text(encoding="utf-8"))
        data = DATA["v8"]
        times = pd.to_datetime(["2024-01-03 " + t for t in data.PREDICTION_TIMES])
        keys = pd.DataFrame({"date": times, "instrument": "A"})
        rng = np.random.default_rng(73)
        grids = {}
        for frequency in FREQUENCIES:
            grid_times = data.regular_times(pd.to_datetime(["2024-01-02", "2024-01-03"]), frequency)
            values = rng.normal(size=(len(grid_times), 1, 72)).astype(np.float32)
            values[rng.random(values.shape) < .1] = np.nan
            grids[frequency] = grid_times, values
        baseline, _ = parent.WindowBlock(keys, grids, ["A"], parent_config).batch(np.arange(8))
        windows, clock = data.WindowBlock(keys, grids, ["A"], CONFIGS["v8"]).batch(np.arange(8))
        for frequency in FREQUENCIES:
            np.testing.assert_array_equal(windows[frequency], baseline[frequency])
        np.testing.assert_array_equal(clock, np.eye(8, dtype=np.float32))
        changed = {f: (t.copy(), v.copy()) for f, (t, v) in grids.items()}
        for t, values in changed.values():
            values[t > times[1].value] = 1e6
        after, _ = data.WindowBlock(keys.iloc[[1]], changed, ["A"], CONFIGS["v8"]).batch([0])
        for frequency in FREQUENCIES:
            np.testing.assert_array_equal(after[frequency], windows[frequency][[1]])
        for invalid in ("2024-01-03 10:03", "2024-01-03 10:00:01", "2024-01-03 10:00:00.000000001", "NaT"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                data.prediction_clock_values(np.array([invalid], dtype="datetime64[ns]").astype(np.int64))

    def test_rejects_partial_merge_configuration_and_unguarded_training(self):
        for change in ({"hidden_validity_mask": True}, {"pooling": "last_mean"},
                       {"padding": "causal"}, {"dilation": 2}, {"prediction_clock_dim": 2},
                       {"kernel_sizes": dict.fromkeys(FREQUENCIES, [3])}):
            config = copy.deepcopy(CONFIGS["v8"])
            config["model"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                TRAINERS["v8"].validate_training(config, PROTOCOL)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "absent"
            args = ["v8_train.py", "--training-run", "unused", "--holdout-run", "unused", "--run-dir", str(run)]
            with patch.object(sys, "argv", args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                TRAINERS["v8"].main()
            self.assertFalse(run.exists())

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA 检查组合模型的确定性反向传播")
    def test_cuda_default_batch_forward_backward_with_deterministic_algorithms(self):
        trainer, config = TRAINERS["v8"], CONFIGS["v8"]
        device = trainer.configure(config)
        model = MODELS["v8"].MultiFrequencyTCN(config).to(device)
        windows, clock = sample_inputs(config, count=config["training"]["batch_size"], device=device)
        targets = torch.linspace(-2, 2, len(clock), dtype=torch.float64, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                      weight_decay=config["training"]["weight_decay"])
        result = trainer.training_step(model, optimizer, windows, clock, targets, config, torch.cuda.is_bf16_supported())
        self.assertTrue(np.isfinite(result["loss"]))
        self.assertEqual(result["rows"], 512)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))


class HuberTests(unittest.TestCase):
    def setUp(self):
        self.loss = importlib.import_module("v8_loss")

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
        trainer = TRAINERS["v8"]
        keys = pd.DataFrame({"date": pd.to_datetime(["2024-01-02 10:00"] * 3 + ["2024-01-02 10:30"] * 5),
                             "instrument": [f"S{i}" for i in range(8)],
                             "vwap_return": [-1., 0., 1., -8., -2., 3., 5., 10.]})
        pred = keys[["date", "instrument"]].copy()
        pred["factor"] = [0., 0., 0., -2., -1., 0., 1., 2.]
        protocol = {**PROTOCOL, "min_cross_section_size": 3}
        actual, undefined = trainer.huber_loss_check(pred, keys, protocol, CONFIGS["v8"])
        losses = [self.loss.standardized_huber_loss(torch.tensor(p.factor.to_numpy()),
                  torch.tensor(keys.loc[p.index, "vwap_return"].to_numpy()), 3).loss.item()
                  for _, p in pred.groupby("date")]
        self.assertAlmostEqual(actual, sum(losses) / 2)
        self.assertEqual(undefined, 1)

    def test_random_batches_keep_targets_from_their_original_full_sections(self):
        trainer = TRAINERS["v8"]
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

    def test_v8_rejects_old_batching_and_invalid_batch_sizes(self):
        trainer = TRAINERS["v8"]
        for changes in ({"batching": "full_cross_section"}, {"sections_per_step": 1},
                        {"checkpoint_sections": 64}, {"batch_size": 0}, {"batch_size": True},
                        {"batch_size": 1.5}, {"checkpoint_batches": 0}):
            config = copy.deepcopy(CONFIGS["v8"])
            config["training"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                trainer.validate_training(config, PROTOCOL)


class PredictionTests(unittest.TestCase):
    def test_each_namespace_predicts_from_checkpoint_and_text_weights(self):
        from test_v3_predict import FakeDai
        prep = importlib.import_module("v8_preprocessing")
        days = pd.date_range("2024-01-02", periods=2)
        minutes = (570, 600, 630, 660, 690, 810, 840, 870)
        rows = [(day + pd.Timedelta(minutes=m), stock) for day in days for m in minutes for stock in ("A", "B")]
        tables = {"cpt_jyc_2026_instruments": pd.DataFrame(rows, columns=["date", "instrument"])}
        rng = np.random.default_rng(82)
        for frequency in FREQUENCIES:
            times = pd.to_datetime(DATA["v8"].regular_times(days, frequency))
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
        self.clock = DATA[version].prediction_clock_values(raw) if version == "v8" else DATA[version].clock_values(raw)

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
                if version != "v8":
                    columns.append("train_online_ic")
                pd.testing.assert_frame_equal(first[columns], second[columns], check_exact=True)
                if version == "v8":
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
