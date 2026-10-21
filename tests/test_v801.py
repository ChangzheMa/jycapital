"""冻结表示、集合交互和完整截面恢复的定向合成检查。"""

import copy
import inspect
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
for version in ("v501", "v801"):
    sys.path.insert(0, str(ROOT / "solutions" / version))
import v801_train as trainer
import v501_train as parent
from v801_model import MultiFrequencyGRU
from v801_cache import RepresentationStore, RepresentationBlock, initialize_backbone
from v801_market_fast import predict_day
from v801_text_artifact import FORMAT, encode_state, state_digest

CONFIG = json.loads((ROOT / "solutions/v801/model-config.json").read_text(encoding="utf-8"))
CONFIGS = {"v801": CONFIG}
TEST_VERSIONS = ("v801",)
PARENTS = {"v801": "v501"}
PROTOCOL = json.loads((ROOT / "evaluation/protocols/local-v2-ic.json").read_text(encoding="utf-8"))
torch.set_num_threads(1)


def small_config(version="v801"):
    config = copy.deepcopy(CONFIG)
    config["model"].update(hidden_dim=8, fusion_hidden_dim=16)
    config["model"]["context"].update(hidden_dim=8, attention_heads=2, latent_tokens=4)
    config["training"].update(epochs=2, cpu_threads=1, checkpoint_batches=1)
    config["training"]["resident_cache"].update(memory_reserve_gib=0, build_batch_rows=256, evaluation_days=1)
    config["inference"].update(batch_size=16, stock_day_batch_size=2)
    return config


def parent_artifact(config, directory):
    with torch.random.fork_rng():
        torch.manual_seed(73)
        state = MultiFrequencyGRU(config).backbone.state_dict()
    path = Path(directory) / "backbone.json"
    config["initialization"]["state_dict_sha256"] = state_digest(state)
    path.write_text(json.dumps({"format": FORMAT, "state_dict": encode_state(state)}), encoding="utf-8")
    return path


class V801Tests(unittest.TestCase):
    def test_default_parameter_count_and_unchanged_primary_cache_transform(self):
        model = MultiFrequencyGRU(CONFIG)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2129158)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 1695749)
        self.assertEqual(inspect.getsource(trainer.prepare_training_targets), inspect.getsource(parent.prepare_training_targets))
        trainer.validate_training(CONFIG, PROTOCOL)

    def test_zero_initialization_frozen_backbone_and_complete_set_equivariance(self):
        config = small_config()
        model = MultiFrequencyGRU(config).eval()
        features, clock = trainer.preflight_inputs(config, 9, torch.device("cpu"))
        with torch.no_grad():
            expected = model.backbone.head(features).squeeze(1)
            torch.testing.assert_close(model.forward_features(features), expected, atol=0, rtol=0)
            model.context[-1].weight.normal_(std=.1)
        baseline = {key: value.clone() for key, value in model.backbone.state_dict().items()}
        permutation = torch.tensor([5, 3, 1, 8, 4, 2, 0, 7, 6])
        with torch.no_grad():
            scores = model.forward_features(features)
            torch.testing.assert_close(model.forward_features(features[permutation]), scores[permutation], atol=2e-7, rtol=1e-5)
            changed = features.clone()
            changed[1, :-8] += 2
            self.assertGreater((model.forward_features(changed)[0] - scores[0]).abs().item(), 1e-8)
        invalid = features.clone()
        invalid[1, -8:] = torch.eye(8)[3]
        with self.assertRaises(ValueError):
            model.forward_features(invalid)
        targets = torch.linspace(-2, 2, 9, dtype=torch.float64)
        targets[2] = float("nan")
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        trainer.training_step(model, optimizer, features, clock, targets, config, False)
        self.assertFalse(model.backbone.training)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.context.parameters()))
        for key, value in model.backbone.state_dict().items():
            torch.testing.assert_close(value, baseline[key], atol=0, rtol=0)

    def test_shared_encoding_matches_reference_and_chunking_does_not_change_context(self):
        from test_v701 import Block
        config = small_config()
        block = Block("2020-01", 1)
        block.config = config
        block.keys = block.keys.drop(index=[8, 15, 43]).sample(frac=1, random_state=9).reset_index(drop=True)
        block.times = block.keys.date.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        block.day_indices = np.zeros(len(block.keys), np.int64)
        block.stock_indices = block.keys.instrument.str[1:].astype(int).to_numpy()
        model = MultiFrequencyGRU(config).eval()
        with torch.no_grad():
            model.context[-1].weight.normal_(std=.1)
        left = predict_day(model, block, torch.device("cpu"), stock_batch_size=2)
        right = predict_day(model, block, torch.device("cpu"), stock_batch_size=5)
        np.testing.assert_allclose(left, right, rtol=1e-5, atol=2e-7)
        for _, keys in block.keys.groupby("date"):
            rows = keys.index.to_numpy()
            windows, clock = block.batch(rows)
            with torch.no_grad():
                actual = model({f: torch.from_numpy(x) for f, x in windows.items()}, torch.from_numpy(clock)).numpy()
            np.testing.assert_allclose(left[rows], actual, rtol=1e-5, atol=2e-7)
        original = left.copy()
        block.windows["bar1m"][:, :, 360:] += 10000
        altered = predict_day(model, block, torch.device("cpu"), stock_batch_size=3)
        early = block.keys.date.dt.hour < 12
        np.testing.assert_allclose(original[early], altered[early], rtol=1e-5, atol=2e-7)

    def test_text_initialization_rejects_different_frozen_weights(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as temporary:
            path = parent_artifact(config, temporary)
            model = MultiFrequencyGRU(config)
            initialize_backbone(model, config, path)
            self.assertEqual(state_digest(model.backbone.state_dict()), config["initialization"]["state_dict_sha256"])
            bundle = json.loads(path.read_text(encoding="utf-8"))
            next(iter(bundle["state_dict"].values()))["values"][0] += .1
            path.write_text(json.dumps(bundle), encoding="utf-8")
            with self.assertRaises(ValueError):
                initialize_backbone(model, config, path)

    def test_full_section_resume_restores_raw_ema_rng_and_optimizer(self):
        config = small_config()
        protocol = dict(PROTOCOL, min_cross_section_size=3)
        with tempfile.TemporaryDirectory() as temporary:
            artifact = parent_artifact(config, temporary)
            store = RepresentationStore.__new__(RepresentationStore)
            store.protocol, store.config, store.backbone_artifact = protocol, config, artifact
            store.loaded = {}
            rng = np.random.default_rng(13)
            for split, month in (("train", "2020-01"), ("validation", "2024-01"), ("test", "2024-07")):
                block = RepresentationBlock.__new__(RepresentationBlock)
                times = [pd.Timestamp(month + "-02") + pd.Timedelta(minutes=m) for m in (570, 600, 630, 660, 690)]
                block.keys = pd.DataFrame([(t, f"S{s}") for t in times for s in range(6)], columns=["date", "instrument"])
                block.keys["vwap_return"] = rng.normal(0, .001, len(block.keys))
                block.targets, _ = trainer.prepare_training_targets(block.keys, protocol)
                block.features = rng.normal(size=(len(block.keys), 32)).astype(np.float32)
                block.features[:, -8:] = np.eye(8)[np.repeat(np.arange(5), 6)]
                block.boundaries = np.arange(0, 31, 6)
                block.manifest = {"catalog": {"sections": 5, "skipped": {}}, "identity": {"synthetic": True}}
                store.loaded[split] = block
            runs = [Path(temporary) / name for name in ("continuous", "resumed")]
            manifests = [{"status": "running"}, {"status": "running"}]
            for run in runs:
                run.mkdir()
                trainer.write_json(run / "data-manifest.json", {})
                trainer.write_json(run / "baseline-preservation.json", {})
            configure, step = trainer.configure, trainer.training_step

            def cpu_configure(spec):
                configure(spec)
                return torch.device("cpu")

            with patch.object(trainer, "configure", cpu_configure), patch.object(trainer, "environment", lambda: {"packages": {}}), \
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
                saved = torch.load(runs[1] / "last.pt", weights_only=True)
                self.assertEqual(saved["progress"]["section_cursor"], 2)
                self.assertEqual(saved["progress"]["sample_cursor"], 12)
                trainer.train(store, runs[1], config, manifests[1], resume=True)
            left, right = [torch.load(run / "last.pt", weights_only=True) for run in runs]
            for field in ("state_dict", "ema_state_dict"):
                for key, value in left[field].items():
                    torch.testing.assert_close(value, right[field][key], atol=0, rtol=0)
            for parameter, values in left["optimizer"]["state"].items():
                for key, value in values.items():
                    torch.testing.assert_close(value, right["optimizer"]["state"][parameter][key], atol=0, rtol=0)
            torch.testing.assert_close(left["torch_rng"], right["torch_rng"], atol=0, rtol=0)
            frames = [pd.read_csv(run / "history.csv") for run in runs]
            columns = ["train_loss", "train_huber_loss", "train_ic_loss", "validation_loss", "ic_mean", "rank_ic_mean"]
            pd.testing.assert_frame_equal(frames[0][columns], frames[1][columns], check_exact=True)


if __name__ == "__main__":
    unittest.main()
