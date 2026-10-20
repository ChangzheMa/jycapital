"""持续候选的因果性、完整梯度、序列化和确定性训练恢复。仅 CPU 合成数据。"""

import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import pandas as pd

import test_research_branches as base

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ("v102", "v202", "v302", "v303", "v304", "v305", "v401")
CLASSES = {"v102": "MultiFrequencyPatchTransformer", "v202": "MultiFrequencyMixer",
           "v302": "MultiFrequencyGRU", "v303": "MultiFrequencyGRU", "v304": "MultiFrequencyGRU",
           "v305": "MultiFrequencyGRU", "v401": "MultiFrequencyGRU"}
PARENTS = {"v102": "v101", "v202": "v201", "v302": "v301", "v303": "v301", "v304": "v301",
           "v305": "v302", "v401": "v301"}
for version in VERSIONS:
    sys.path.insert(0, str(ROOT / "solutions" / version))
MODULES = {v: {name: importlib.import_module(v + "_" + name)
               for name in ("model", "schema", "data", "train", "predict", "text_artifact", "preprocessing")}
           for v in VERSIONS}
CONFIGS = {v: json.loads((ROOT / "solutions" / v / "model-config.json").read_text(encoding="utf-8")) for v in VERSIONS}
base.CONFIGS.update(CONFIGS)
base.MODULES.update(MODULES)


def small_config(version):
    config = copy.deepcopy(CONFIGS[version])
    config["data"].pop("input_precision", None)
    config["training"].pop("resident_cache", None)
    config["training"].update(batching="monthly_shuffled_samples", epochs=2, cpu_threads=1,
                              batch_size=16, checkpoint_batches=1)
    config["model"].update(hidden_dim=8, fusion_hidden_dim=16)
    if version == "v102":
        config["model"].update(layers=1, attention_heads=2, feedforward_dim=16)
    elif version == "v202":
        config["model"].update(layers=1, temporal_rank=4, channel_hidden_dim=16, readout_tokens=2)
    return config


def create_model(version, config=None):
    return getattr(MODULES[version]["model"], CLASSES[version])(config or CONFIGS[version])


class CandidateTests(unittest.TestCase):
    def test_huber_delta_control_preserves_network_and_changes_training_loss(self):
        parent, changed = base.CONFIGS["v301"], CONFIGS["v303"]
        for key in ("data", "model", "inference", "seed"):
            self.assertEqual(parent[key], changed[key])
        a, b = copy.deepcopy(parent["training"]), copy.deepcopy(changed["training"])
        self.assertEqual(a.pop("huber_delta"), 1.0)
        self.assertEqual(b.pop("huber_delta"), 2.0)
        self.assertEqual(a, b)

        class ConstantModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.score = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, windows, clock):
                return self.score.expand(len(clock))

        for version, expected in (("v301", 2.5), ("v303", 4.0)):
            trainer = base.MODULES[version]["train"]
            config = base.CONFIGS[version]
            model = ConstantModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0)
            result = trainer.training_step(model, optimizer, {}, torch.eye(8)[:1],
                                           torch.tensor([3.0], dtype=torch.float64), config, False)
            self.assertEqual(result["loss"], expected)

    def test_clock_gate_contains_parent_readout_as_a_limit(self):
        child_config = small_config("v304")
        parent_config = copy.deepcopy(child_config)
        parent_config["solution_id"] = "v301"
        parent_config["model"]["pooling"] = "last_available_state"
        parent = base.create_model("v301", parent_config).eval()
        child = create_model("v304", child_config).eval()
        state = child.load_state_dict(parent.state_dict(), strict=False)
        self.assertEqual(len(state.missing_keys), 6)
        self.assertFalse(state.unexpected_keys)
        with torch.no_grad():
            for encoder in child.encoders.values():
                encoder.gate.weight.zero_()
                encoder.gate.bias.fill_(-80)
        windows, clock = base.sample_inputs(child_config)
        with torch.inference_mode():
            torch.testing.assert_close(child(windows, clock), parent(windows, clock), rtol=0, atol=2e-7)

    def test_parent_protection_excludes_mutable_monitoring(self):
        for version in VERSIONS:
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = CONFIGS[version]
                trainer = MODULES[version]["train"]
                parent = root / "outputs/runs" / config["parent_run_id"]
                parent.mkdir(parents=True)
                (parent / "best.pt").write_bytes(b"frozen-test-weight")
                for name in ("config.json", "protocol.json", "data-manifest.json", "selection.json"):
                    trainer.write_json(parent / name, {})
                trainer.write_json(parent / "manifest.json", {"status": "complete", "selection": {
                    "model_sha256": trainer.sha256(parent / "best.pt")}})
                trainer.write_json(parent / "monitor-state.json", {"status": "old"})
                saved = trainer.parent_snapshot(root, config)
                trainer.write_json(parent / "monitor-state.json", {"status": "updated"})
                trainer.verify_parent(root, saved)
                (parent / "best.pt").write_bytes(b"changed-weight")
                with self.assertRaises(ValueError):
                    trainer.verify_parent(root, saved)

    def test_input_target_and_cache_functions_match_parent(self):
        import inspect
        for version in VERSIONS:
            parent = PARENTS[version]
            trainer = MODULES[version]["train"]
            trainer.validate_training(CONFIGS[version], base.PROTOCOL)
            for name in ("loss", "data", "preprocessing"):
                if version == "v401" and name == "loss":
                    continue
                left = (ROOT / "solutions" / version / (version + "_" + name + ".py")).read_text(encoding="utf-8")
                right = (ROOT / "solutions" / parent / (parent + "_" + name + ".py")).read_text(encoding="utf-8")
                self.assertEqual(left.replace(version, parent), right)
            self.assertEqual(inspect.getsource(trainer.prepare_training_targets),
                             inspect.getsource(base.MODULES[parent]["train"].prepare_training_targets))

    def test_default_model_causal_gradients_batching_and_text_roundtrip(self):
        for version in VERSIONS:
            with self.subTest(version=version):
                torch.manual_seed(17)
                model = create_model(version).eval()
                self.assertLess(sum(p.numel() for p in model.parameters()), 1_000_000)
                windows, clock = base.sample_inputs(CONFIGS[version])
                windows = {f: x.clone().requires_grad_(True) for f, x in windows.items()}
                scores = model(windows, clock)
                self.assertTrue(torch.isfinite(scores).all())
                scores.sum().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                for i in range(8):
                    cutoff = 240 + i * 30
                    self.assertEqual(windows["bar1m"].grad[i, :, cutoff:].abs().sum().item(), 0)
                    self.assertGreater(windows["bar1m"].grad[i, :, :cutoff].abs().sum().item(), 0)
                corrupted = {f: x.detach().clone() for f, x in windows.items()}
                for i in range(8):
                    corrupted["bar1m"][i, :, 240 + i * 30:] = float("nan")
                with torch.inference_mode():
                    torch.testing.assert_close(model(corrupted, clock), scores.detach(), rtol=0, atol=0)
                    divided = torch.cat([model({f: x[i:i + 2] for f, x in windows.items()}, clock[i:i + 2]) for i in range(0, 8, 2)])
                    torch.testing.assert_close(divided, scores.detach(), rtol=2e-5, atol=2e-6)
                    empty = {f: torch.zeros_like(x) for f, x in windows.items()}
                    self.assertTrue(torch.isfinite(model(empty, clock)).all())
                artifact = MODULES[version]["text_artifact"]
                restored = artifact.decode_state({"format": artifact.FORMAT, "state_dict": artifact.encode_state(model.state_dict())})
                clone = create_model(version).eval()
                clone.load_state_dict(restored, strict=True)
                self.assertEqual(artifact.state_digest(model.state_dict()), artifact.state_digest(restored))
                with torch.inference_mode():
                    torch.testing.assert_close(clone(windows, clock), scores.detach(), rtol=0, atol=0)

    def test_invalid_patch_config_and_partial_patch_boundary(self):
        for version in ("v202", "v302", "v305"):
            for size in (0, True, 7):
                config = copy.deepcopy(CONFIGS[version])
                config["model"]["patch_bars"]["bar1m"] = size
                with self.assertRaises(ValueError):
                    create_model(version, config)
        config = small_config("v302")
        config["model"]["patch_bars"]["bar1m"] = 8
        model = create_model("v302", config).eval()
        windows, clock = base.sample_inputs(config)
        altered = {f: x.clone() for f, x in windows.items()}
        for i in range(8):
            altered["bar1m"][i, :, 240 + i * 30:] = 1e6
        with torch.inference_mode():
            torch.testing.assert_close(model(windows, clock), model(altered, clock), rtol=0, atol=0)

    def test_train_evaluate_and_resume_exactly(self):
        for version in VERSIONS:
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temporary:
                trainer, config = MODULES[version]["train"], small_config(version)
                configure, step = trainer.configure, trainer.training_step
                store = base.SyntheticStore(version)

                def cpu_configure(spec):
                    configure(spec)
                    return torch.device("cpu")

                runs = [Path(temporary) / name for name in ("continuous", "resumed")]
                manifests = [{"status": "running"}, {"status": "running"}]
                for run in runs:
                    run.mkdir()
                    trainer.write_json(run / "data-manifest.json", {"synthetic": True})
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
                for parameter, values in left["optimizer"]["state"].items():
                    for key, value in values.items():
                        torch.testing.assert_close(value, right["optimizer"]["state"][parameter][key], rtol=0, atol=0)
                for manifest in manifests:
                    self.assertEqual(manifest["status"], "complete")
                    for split in ("validation", "test"):
                        self.assertEqual(manifest["metrics"][split]["coverage"], 1)
                frames = [pd.read_csv(run / "history.csv") for run in runs]
                columns = ["train_loss", "validation_loss", "ic_mean", "rank_ic_mean", "best_epoch", "patience_epochs", "stale_epochs"]
                pd.testing.assert_frame_equal(frames[0][columns], frames[1][columns], check_exact=True)


if __name__ == "__main__":
    unittest.main()
