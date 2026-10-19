"""三个非卷积研究家族的因果性、梯度、文本张量和训练恢复检查。"""

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
VERSIONS = ("v101", "v201", "v301")
CLASSES = {"v101": "MultiFrequencyPatchTransformer", "v201": "MultiFrequencyMixer", "v301": "MultiFrequencyGRU"}
for version in VERSIONS:
    sys.path.insert(0, str(ROOT / "solutions" / version))
MODULES = {v: {name: importlib.import_module(v + "_" + name)
               for name in ("model", "schema", "data", "train", "predict", "text_artifact", "preprocessing")}
           for v in VERSIONS}
CONFIGS = {v: json.loads((ROOT / "solutions" / v / "model-config.json").read_text(encoding="utf-8")) for v in VERSIONS}
PROTOCOL = json.loads((ROOT / "evaluation/protocols/local-v2-ic.json").read_text(encoding="utf-8"))
FREQUENCIES = tuple(CONFIGS["v101"]["data"]["window_bars"])
torch.set_num_threads(1)


def small_config(version, resident=False):
    config = copy.deepcopy(CONFIGS[version]) if resident else json.loads(
        (ROOT / "solutions" / version / "model-config-monthly.json").read_text(encoding="utf-8"))
    if resident:
        config["training"]["resident_cache"].update(memory_reserve_gib=0, build_batch_rows=256, evaluation_days=1)
    spec = config["model"]
    spec.update(hidden_dim=8, fusion_hidden_dim=16)
    if version == "v101":
        spec.update(layers=1, attention_heads=2, feedforward_dim=16)
    elif version == "v201":
        spec.update(layers=1, temporal_rank=4, channel_hidden_dim=16, readout_tokens=2)
    config["training"].update(epochs=2, cpu_threads=1, batch_size=16, checkpoint_batches=1)
    return config


def create_model(version, config=None):
    return getattr(MODULES[version]["model"], CLASSES[version])(config or CONFIGS[version])


def sample_inputs(config, count=8, device="cpu"):
    generator = torch.Generator().manual_seed(55)
    windows = {f: torch.randn(count, 75, length, generator=generator).to(device)
               for f, length in config["data"]["window_bars"].items()}
    for x in windows.values():
        x[:, 72:75] = 1
        x[0] = 0
    clock = F.one_hot(torch.arange(count, device=device) % 8, 8).float()
    end = 240 + clock.argmax(dim=1) * 30
    x = windows["bar1m"]
    x.masked_fill_(torch.arange(480, device=device)[None, None, :] >= end[:, None, None], 0)
    return windows, clock


class ModelTests(unittest.TestCase):
    def test_inherit_v9_inputs_supervision_and_independent_modules(self):
        parent = json.loads((ROOT / "solutions/v9/model-config.json").read_text(encoding="utf-8"))
        for version in VERSIONS:
            with self.subTest(version=version):
                config = CONFIGS[version]
                legacy = json.loads((ROOT / "solutions" / version / "model-config-monthly.json").read_text(encoding="utf-8"))
                for key in ("data", "training", "inference", "seed"):
                    self.assertEqual(legacy[key], parent[key])
                self.assertEqual(config["training"]["batching"], "global_shuffled_samples")
                self.assertEqual(config["data"]["input_precision"], "fp16_with_fp32_overflow")
                self.assertEqual(config["parent_solution_id"], "v9")
                self.assertIsNone(config["parent_run_id"])
                self.assertIsNone(config["references"]["parent"])
                MODULES[version]["train"].validate_training(config, PROTOCOL)
                for name in ("loss", "preprocessing"):
                    actual = (ROOT / "solutions" / version / (version + "_" + name + ".py")).read_text(encoding="utf-8")
                    expected = (ROOT / "solutions/v9" / ("v9_" + name + ".py")).read_text(encoding="utf-8")
                    self.assertEqual(actual.replace(version, "v9"), expected)

    def test_default_models_full_window_gradients_and_empty_history(self):
        counts = {"v101": 656993, "v201": 552973, "v301": 433409}
        for version in VERSIONS:
            with self.subTest(version=version):
                model = create_model(version).eval()
                self.assertEqual(sum(p.numel() for p in model.parameters()), counts[version])
                self.assertFalse(any(isinstance(m, nn.Conv1d) for m in model.modules()))
                windows, clock = sample_inputs(CONFIGS[version])
                output = model(windows, clock)
                self.assertEqual(output.shape, (8,))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                for x in windows.values():
                    x.zero_()
                self.assertTrue(torch.isfinite(model(windows, clock)).all())

    def test_future_values_and_gradients_excluded_at_all_eight_cutoffs(self):
        for version in VERSIONS:
            with self.subTest(version=version):
                config = small_config(version)
                model = create_model(version, config).eval()
                windows, clock = sample_inputs(config)
                windows["bar1m"].requires_grad_()
                expected = model(windows, clock)
                gradient = torch.autograd.grad(expected.sum(), windows["bar1m"])[0]
                poisoned = {f: x.detach().clone() for f, x in windows.items()}
                for row in range(8):
                    cutoff = 240 + row * 30
                    poisoned["bar1m"][row, :, cutoff:] = float("nan")
                    self.assertEqual(torch.count_nonzero(gradient[row, :, cutoff:]), 0)
                    self.assertGreater(gradient[row, :, :cutoff].abs().sum(), 0)
                torch.testing.assert_close(model(poisoned, clock), expected, rtol=0, atol=0)

    def test_attention_ignores_entirely_future_patch_positions(self):
        config = small_config("v101")
        model = create_model("v101", config).eval()
        windows, clock = sample_inputs(config, 1)
        expected = model(windows, clock)
        encoder = model.encoders["bar1m"]
        # 09:30 只有昨日 240 根可用，CLS 加 30 个已知 patch。
        with torch.no_grad():
            encoder.position[:, 31:] += torch.randn_like(encoder.position[:, 31:]) * 100
        torch.testing.assert_close(model(windows, clock), expected, rtol=0, atol=0)

    def test_gru_readout_does_not_process_future_tail_as_observed(self):
        config = small_config("v301")
        encoder = create_model("v301", config).eval().encoders["bar1m"]
        windows, clock = sample_inputs(config)
        x = windows["bar1m"]
        lengths = 240 + clock.argmax(dim=1) * 30
        expected = encoder(x, lengths)
        poisoned = x.clone()
        for row, length in enumerate(lengths):
            poisoned[row, :, length:] = 1000
        # 直接调用编码器，绕过外层补零，验证读取的确是截点状态。
        torch.testing.assert_close(encoder(poisoned, lengths), expected, rtol=0, atol=0)

    def test_text_state_roundtrip_and_batch_partition_parity(self):
        for version in VERSIONS:
            with self.subTest(version=version):
                config = small_config(version)
                model = create_model(version, config).eval()
                artifact = MODULES[version]["text_artifact"]
                payload = json.loads(json.dumps({"format": artifact.FORMAT, "state_dict": artifact.encode_state(model.state_dict())}))
                restored = create_model(version, config).eval()
                restored.load_state_dict(artifact.decode_state(payload))
                self.assertEqual(artifact.state_digest(model.state_dict()), artifact.state_digest(restored.state_dict()))
                windows, clock = sample_inputs(config)
                expected = model(windows, clock)
                torch.testing.assert_close(restored(windows, clock), expected, rtol=0, atol=0)
                individual = torch.cat([model({f: x[i:i + 1] for f, x in windows.items()}, clock[i:i + 1]) for i in range(8)])
                torch.testing.assert_close(individual, expected, rtol=1e-5, atol=2e-7)

    def test_invalid_configs_inputs_and_training_cli_guard(self):
        for version in VERSIONS:
            with self.subTest(version=version):
                config = small_config(version)
                for values in ({"hidden_dim": 0}, {"dropout": float("nan")}, {"blocks": {}}, {"family": "tcn"}):
                    invalid = copy.deepcopy(config)
                    invalid["model"].update(values)
                    with self.assertRaises(ValueError):
                        create_model(version, invalid)
                model = create_model(version, config)
                windows, clock = sample_inputs(config)
                for invalid in (clock[:, :7], torch.zeros_like(clock), clock * .5):
                    with self.assertRaises(ValueError):
                        model(windows, invalid)
                trainer = MODULES[version]["train"]
                self.assertIn("solutions/v9/v9_model.py", trainer.parent_snapshot(ROOT, config))
                with tempfile.TemporaryDirectory() as temporary:
                    run = Path(temporary) / "absent"
                    argv = [version + "_train.py", "--training-run", "unused", "--holdout-run", "unused", "--run-dir", str(run)]
                    with patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        trainer.main()
                    self.assertFalse(run.exists())

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_cuda_optimizer_and_rng_replay_is_exact(self):
        for version in VERSIONS:
            with self.subTest(version=version):
                config, trainer = small_config(version), MODULES[version]["train"]
                device = trainer.configure(config)
                model = create_model(version, config).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
                windows, clock = sample_inputs(config, 8, device)
                targets = torch.linspace(-2, 2, 8, dtype=torch.float64, device=device)
                amp = torch.cuda.is_bf16_supported()
                trainer.training_step(model, optimizer, windows, clock, targets, config, amp)
                saved_model, saved_optimizer = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
                saved_rng = {"torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
                expected_loss = trainer.training_step(model, optimizer, windows, clock, targets, config, amp)
                expected = copy.deepcopy(model.state_dict())
                model.load_state_dict(saved_model)
                optimizer.load_state_dict(saved_optimizer)
                trainer.restore_rng(saved_rng, device)
                actual_loss = trainer.training_step(model, optimizer, windows, clock, targets, config, amp)
                self.assertEqual(actual_loss, expected_loss)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_default_cuda_training_batch_and_fp32_validation(self):
        for version in VERSIONS:
            with self.subTest(version=version):
                config, trainer = CONFIGS[version], MODULES[version]["train"]
                device = trainer.configure(config)
                model = create_model(version).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
                windows, clock = sample_inputs(config, config["training"]["batch_size"], device)
                target = torch.linspace(-2, 2, len(clock), dtype=torch.float64, device=device)
                for _ in range(2):
                    result = trainer.training_step(model, optimizer, windows, clock, target, config, torch.cuda.is_bf16_supported())
                    self.assertTrue(np.isfinite(result["loss"]))
                model.eval()
                with torch.inference_mode():
                    prediction = model(*sample_inputs(config, config["inference"]["batch_size"], device))
                self.assertEqual(prediction.dtype, torch.float32)
                self.assertTrue(torch.isfinite(prediction).all())
                del model, optimizer, windows, clock, prediction
                torch.cuda.empty_cache()


class SyntheticBlock:
    def __init__(self, version, month, seed):
        rng = np.random.default_rng(seed)
        times = [pd.Timestamp(month + "-02") + pd.Timedelta(minutes=m) for m in (600, 630, 660)]
        self.keys = pd.DataFrame([(t, f"S{i:02}") for t in times for i in range(12)], columns=["date", "instrument"])
        self.windows = {f: rng.normal(size=(len(self.keys), 75, length)).astype(np.float32)
                        for f, length in CONFIGS[version]["data"]["window_bars"].items()}
        for x in self.windows.values():
            x[:, 72:75] = 1
        self.keys["vwap_return"] = self.windows["bar1m"][:, 0, 260] * .001 + rng.normal(0, .001, len(self.keys))
        raw = self.keys.date.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        self.clock = MODULES[version]["data"].prediction_clock_values(raw)
        cutoff = 240 + self.clock.argmax(axis=1) * 30
        self.windows["bar1m"] = np.where(np.arange(480)[None, None, :] < cutoff[:, None, None], self.windows["bar1m"], 0)

    def batch(self, indices):
        return {f: x[indices] for f, x in self.windows.items()}, self.clock[indices]


class SyntheticStore:
    protocol = {**PROTOCOL, "min_cross_section_size": 3}

    def __init__(self, version):
        self.data = {split: {month: SyntheticBlock(version, month, seed)}
                     for split, month, seed in (("train", "2020-01", 1), ("validation", "2024-01", 3), ("test", "2024-07", 4))}

    def months(self, split):
        return list(self.data[split])

    def load(self, split, month):
        return self.data[split][month]


class TrainerIntegrationTests(unittest.TestCase):
    def test_train_evaluate_and_resume_exactly_for_every_family(self):
        for version in VERSIONS:
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temporary:
                trainer, config = MODULES[version]["train"], small_config(version)
                configure, step = trainer.configure, trainer.training_step
                store = SyntheticStore(version)

                def cpu_configure(spec):
                    configure(spec)
                    return torch.device("cpu")

                runs = [Path(temporary) / name for name in ("continuous", "resumed")]
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
                torch.testing.assert_close(left["torch_rng"], right["torch_rng"], rtol=0, atol=0)
                self.assertEqual(left["optimizer"]["param_groups"], right["optimizer"]["param_groups"])
                for parameter, values in left["optimizer"]["state"].items():
                    for key, value in values.items():
                        torch.testing.assert_close(value, right["optimizer"]["state"][parameter][key], rtol=0, atol=0)
                for manifest in manifests:
                    self.assertEqual(manifest["status"], "complete")
                    self.assertEqual(manifest["selection"]["optimizer_steps"], 6)
                    for split in ("validation", "test"):
                        self.assertEqual(manifest["metrics"][split]["coverage"], 1)
                first, second = [pd.read_csv(run / "history.csv") for run in runs]
                columns = ["train_loss", "validation_loss", "ic_mean", "rank_ic_mean", "best_epoch", "patience_epochs", "stale_epochs"]
                pd.testing.assert_frame_equal(first[columns], second[columns], check_exact=True)


if __name__ == "__main__":
    unittest.main()
