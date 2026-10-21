"""联合时序／截面模型的重算梯度、因果性与初始化检查。"""

import copy
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for version in ("v501", "v802"):
    sys.path.insert(0, str(ROOT / "solutions" / version))
import v802_train as trainer
import v501_train as parent
from v802_model import MultiFrequencyGRU
from v802_initialization import initialize_backbone
from v802_text_artifact import FORMAT, encode_state, state_digest

CONFIG = json.loads((ROOT / "solutions/v802/model-config.json").read_text(encoding="utf-8"))
CONFIGS = {"v802": CONFIG}
TEST_VERSIONS = ("v802",)
PARENTS = {"v802": "v801"}
PROTOCOL = json.loads((ROOT / "evaluation/protocols/local-v2-ic.json").read_text(encoding="utf-8"))
torch.set_num_threads(1)


def small_config(version="v802"):
    config = copy.deepcopy(CONFIG)
    config["model"].update(hidden_dim=8, fusion_hidden_dim=16, encoder_chunk_size=2)
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


class V802Tests(unittest.TestCase):
    def test_all_parameters_trainable_and_raw_target_cache_unchanged(self):
        model = MultiFrequencyGRU(CONFIG)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2129158)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertEqual(inspect.getsource(trainer.prepare_training_targets), inspect.getsource(parent.prepare_training_targets))
        trainer.validate_training(CONFIG, PROTOCOL)

    def test_checkpointing_preserves_dropout_rng_gradients_and_updates(self):
        config = small_config()
        model = MultiFrequencyGRU(config)
        reference = copy.deepcopy(model)
        reference.gradient_checkpointing = False
        windows, clock = trainer.preflight_inputs(config, 5, torch.device("cpu"))
        target = torch.tensor([-1., .3, float("nan"), 1.5, -.2], dtype=torch.float64)
        initial = state_digest(model.backbone.state_dict())
        states = []
        for network in (model, reference):
            optimizer = torch.optim.AdamW(network.parameters(), lr=.0001)
            torch.manual_seed(123)
            result = trainer.training_step(network, optimizer, windows, clock, target, config, False)
            self.assertEqual(result["rows"], 4)
            states.append(torch.get_rng_state().clone())
            self.assertTrue(network.backbone.training)
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters()))
            self.assertNotEqual(state_digest(network.backbone.state_dict()), initial)
        torch.testing.assert_close(states[0], states[1], atol=0, rtol=0)
        for left, right in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
            torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)

    def test_cross_stock_gradients_are_causal_and_eval_chunks_do_not_change_context(self):
        config = small_config()
        model = MultiFrequencyGRU(config).eval()
        windows, clock = trainer.preflight_inputs(config, 5, torch.device("cpu"))
        clock[:] = torch.eye(8)[3]
        windows = {f: value.requires_grad_(True) for f, value in windows.items()}
        with torch.no_grad():
            model.context[-1].weight.normal_(std=.1)
        scores = model(windows, clock)
        scores[0].backward()
        self.assertEqual(windows["bar1m"].grad[:, :, 330:].count_nonzero().item(), 0)
        self.assertGreater(windows["bar1m"].grad[1:, :, :330].abs().sum().item(), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        model.encoder_chunk_size = 5
        with torch.no_grad():
            torch.testing.assert_close(model(windows, clock), scores.detach(), atol=2e-7, rtol=1e-5)

    def test_initialization_matches_parent_but_remains_trainable(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as temporary:
            path = parent_artifact(config, temporary)
            model = MultiFrequencyGRU(config).eval()
            state = initialize_backbone(model, config, path)
            self.assertTrue(all(p.requires_grad for p in model.backbone.parameters()))
            self.assertEqual(state_digest(state), config["initialization"]["state_dict_sha256"])
            windows, clock = trainer.preflight_inputs(config, 5, torch.device("cpu"))
            with torch.no_grad():
                torch.testing.assert_close(model(windows, clock), model.backbone(windows, clock), atol=2e-7, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
