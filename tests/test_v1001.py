"""v1001 新计算图、股票日标签和恢复语义的少量 CPU 合成检查。"""

import copy
import math
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for version in ("v501", "v1001"):
    sys.path.insert(0, str(ROOT / "solutions" / version))
import v1001_train as trainer
import v501_train as parent
from v1001_model import MultiFrequencyGRU
from v1001_grouped import StockDayIndex, training_view, record_group_plan
from v1001_data import prediction_clock_values
from v1001_loss import standardized_targets
from v1001_targets import prepare_rank_training_targets
from v1001_temporal import attach_temporal_weights
from v1001_objective import balanced_assignment, combined_objective

CONFIG = json.loads((ROOT / "solutions/v1001/model-config.json").read_text(encoding="utf-8"))
CONFIGS = {"v1001": CONFIG}
TEST_VERSIONS = ("v1001",)
PARENTS = {"v1001": "v701"}
PROTOCOL = json.loads((ROOT / "evaluation/protocols/local-v2-ic.json").read_text(encoding="utf-8"))
torch.set_num_threads(1)


def small_config(version="v1001", resident=False):
    config = copy.deepcopy(CONFIG)
    config["model"].update(hidden_dim=8, fusion_hidden_dim=16, expert_hidden_dim=8, router_hidden_dim=8)
    config["training"].update(epochs=2, cpu_threads=1, batch_size=16, stock_day_batch_size=2,
                              checkpoint_batches=1)
    config["inference"].update(batch_size=16, stock_day_batch_size=2)
    config["training"]["resident_cache"].update(memory_reserve_gib=0, build_batch_rows=256, evaluation_days=1)
    if not resident:
        config["training"]["batching"] = "monthly_shuffled_stock_days"
        config["data"]["input_precision"] = "float32"
    return config


def inputs(config, count=3):
    generator = torch.Generator().manual_seed(55)
    windows = {f: torch.randn(count, 75, n, generator=generator) for f, n in config["data"]["window_bars"].items()}
    for value in windows.values():
        value[:, 72:] = 1
    return windows, F.one_hot(torch.full((count,), 7), 8).float()


class Block:
    def __init__(self, month, seed):
        rng = np.random.default_rng(seed)
        self.keys = pd.DataFrame([(pd.Timestamp(month + "-02") + pd.Timedelta(minutes=m), f"S{s}")
                                  for m in (570, 600, 630, 660, 690, 810, 840, 870) for s in range(6)],
                                 columns=["date", "instrument"])
        self.keys["vwap_return"] = rng.normal(0, .001, len(self.keys))
        self.keys.loc[:5, "vwap_return"] = np.nan
        if seed == 1:
            extra = self.keys.copy()
            extra['date'] += pd.Timedelta(days=20)
            self.keys = pd.concat([self.keys, extra], ignore_index=True)
        self.windows = {f: rng.normal(size=(6, 75, n)).astype(np.float32) for f, n in CONFIG["data"]["window_bars"].items()}
        for value in self.windows.values():
            value[:, 72:] = 1

    def batch(self, rows):
        keys = self.keys.iloc[rows]
        clocks = prediction_clock_values(keys.date.to_numpy(dtype="datetime64[ns]").astype(np.int64))
        stocks = keys.instrument.str[1:].astype(int).to_numpy()
        values = {f: x[stocks].copy() for f, x in self.windows.items()}
        values["bar1m"] *= np.arange(480)[None, None, :] < (240 + 30 * clocks.argmax(1))[:, None, None]
        return values, clocks


class Store:
    protocol = dict(PROTOCOL, min_cross_section_size=3)

    def __init__(self):
        self.data = {split: {month: Block(month, seed)} for split, month, seed in (
            ("train", "2020-01", 1), ("validation", "2024-01", 3), ("test", "2024-07", 4))}

    def months(self, split):
        return list(self.data[split])

    def load(self, split, month):
        return self.data[split][month]


class V1001Tests(unittest.TestCase):
    def test_data_and_primary_target_cache_identity(self):
        previous = json.loads((ROOT / "solutions/v501/model-config.json").read_text(encoding="utf-8"))
        self.assertEqual(CONFIG["data"], previous["data"])
        self.assertEqual(inspect.getsource(trainer.prepare_training_targets), inspect.getsource(parent.prepare_training_targets))
        self.assertEqual(inspect.getsource(standardized_targets), inspect.getsource(parent.standardized_targets))
        trainer.validate_training(CONFIG, PROTOCOL)
        self.assertEqual(sum(p.numel() for p in MultiFrequencyGRU(CONFIG).parameters()), 2418576)

    def test_shared_heads_match_reference_and_are_causal(self):
        config = small_config()
        model = MultiFrequencyGRU(config).eval()
        windows, latest = inputs(config)
        windows["bar1m"].requires_grad_(True)
        grouped = model.forward_group(windows, latest)
        for clock_id in range(8):
            clock = F.one_hot(torch.full((3,), clock_id), 8).float()
            with torch.no_grad():
                reference = model.forward_heads(windows, clock)
                torch.testing.assert_close(grouped[:, clock_id], reference, atol=2e-7, rtol=1e-5)
            gradient = torch.autograd.grad(grouped[:, clock_id].sum(), windows["bar1m"], retain_graph=True)[0]
            self.assertEqual(gradient[:, :, 240 + 30 * clock_id:].count_nonzero().item(), 0)
            self.assertGreater(gradient[:, :, :240 + 30 * clock_id].abs().sum().item(), 0)
        partial = F.one_hot(torch.tensor([0, 3, 5]), 8).float()
        with torch.no_grad():
            shared = model.forward_group(windows, partial)
            expected = model.forward_heads(windows, partial)
        torch.testing.assert_close(shared[torch.arange(3), partial.argmax(1)], expected, atol=2e-7, rtol=1e-5)
        target = torch.randn(3, 8, 2, dtype=torch.float64)
        step = trainer.training_step(model, torch.optim.AdamW(model.parameters()), windows, latest, target, config, False)
        self.assertEqual(step["rows"], 24)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(model.experts[0][-1].weight.grad[1].abs().sum().item(), 0)

    def test_group_mapping_partial_keys_and_unchanged_targets(self):
        block = Block("2020-01", 1)
        block.keys = block.keys.drop(index=[7, 13, 42]).sample(frac=1, random_state=4).reset_index(drop=True)
        index = StockDayIndex(block)
        output = np.full(len(block.keys), np.nan)
        expected = prediction_clock_values(block.keys.date.to_numpy(dtype="datetime64[ns]").astype(np.int64)).argmax(1)
        index.assign_predictions(output, np.arange(len(index.samples)), np.tile(np.arange(8), (len(index.samples), 1)))
        np.testing.assert_array_equal(output, expected)
        raw, skipped = trainer.prepare_training_targets(block.keys, Store.protocol)
        rank, other = prepare_rank_training_targets(block.keys, Store.protocol)
        self.assertEqual(skipped, other)
        before = raw.copy()
        view = training_view(block, raw, rank)
        targets = view.targets[np.arange(len(index.samples))]
        self.assertEqual(np.isfinite(targets[..., 0]).sum(), np.isfinite(raw).sum())
        np.testing.assert_array_equal(before, raw)
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            trainer.write_json(run / "data-manifest.json", {})
            record_group_plan(view, run, "2020-01")
            rank[np.isfinite(rank)] += .1
            with self.assertRaises(ValueError):
                record_group_plan(view, run, "2020-01")
        block.keys = pd.concat([block.keys, block.keys.iloc[:1]], ignore_index=True)
        with self.assertRaises(ValueError):
            StockDayIndex(block)

    def test_actual_training_loss_combines_heads_and_masks_missing_labels(self):
        class Constant(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scores = torch.nn.Parameter(torch.zeros(1, 8, 2))

            def forward_group_components(self, windows, clock):
                return self.scores, self.scores[:, :, None, :].expand(-1, -1, 4, -1), torch.zeros(1, 8, 4)

        model = Constant()
        targets = torch.full((1, 8, 2), float("nan"), dtype=torch.float64)
        targets[0, 1] = torch.tensor([3., 1.])
        result = trainer.training_step(model, torch.optim.SGD(model.parameters(), lr=0), {}, torch.zeros(1, 8),
                                       targets, CONFIG, False)
        self.assertAlmostEqual(result["loss"], 4.1 * 1.1 + 0.01 * math.log(4))
        self.assertEqual(result["rows"], 1)
        self.assertEqual(model.scores.grad[0, 0].count_nonzero().item(), 0)

    def test_interrupted_group_cursor_restores_raw_ema_optimizer_and_rng(self):
        config, store = small_config(), Store()
        configure, step = trainer.configure, trainer.training_step
        with tempfile.TemporaryDirectory() as temporary:
            runs = [Path(temporary) / name for name in ("continuous", "resumed")]
            manifests = [{"status": "running"}, {"status": "running"}]
            for run in runs:
                run.mkdir()
                trainer.write_json(run / "data-manifest.json", {"synthetic": True})
                trainer.write_json(run / "baseline-preservation.json", {})

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
                checkpoint = torch.load(runs[1] / "last.pt", weights_only=True)
                self.assertEqual(checkpoint["progress"]["group_cursor"], 4)
                self.assertEqual(checkpoint["progress"]["sample_cursor"], 32)
                trainer.train(store, runs[1], config, manifests[1], resume=True)
            for filename in ("best.pt", "last.pt"):
                left, right = [torch.load(run / filename, weights_only=True) for run in runs]
                for key, value in left["state_dict"].items():
                    torch.testing.assert_close(value, right["state_dict"][key], rtol=0, atol=0)
            for key, value in left["ema_state_dict"].items():
                torch.testing.assert_close(value, right["ema_state_dict"][key], rtol=0, atol=0)
            for parameter, values in left["optimizer"]["state"].items():
                for key, value in values.items():
                    torch.testing.assert_close(value, right["optimizer"]["state"][parameter][key], rtol=0, atol=0)
            torch.testing.assert_close(left["torch_rng"], right["torch_rng"], rtol=0, atol=0)
            self.assertEqual(left["ema_updates"], right["ema_updates"])
            frames = [pd.read_csv(run / "history.csv") for run in runs]
            columns = ["train_loss", "train_huber_loss", "train_rank_huber_loss", "validation_loss", "ic_mean", "rank_ic_mean", "train_expert_loss", "train_router_loss", "train_routing_entropy", "train_routing_usage", "temporal_weighting"]
            pd.testing.assert_frame_equal(frames[0][columns], frames[1][columns], check_exact=True)
            for manifest in manifests:
                self.assertEqual(manifest["status"], "complete")
                for split in ("validation", "test"):
                    self.assertEqual(manifest["metrics"][split]["coverage"], 1)


    def test_temporal_weights_use_only_training_dates_and_keep_all_samples(self):
        block = Block("2020-01", 1)
        primary, _ = trainer.prepare_training_targets(block.keys, Store.protocol)
        ranks, _ = prepare_rank_training_targets(block.keys, Store.protocol)
        view = training_view(block, primary, ranks)
        attach_temporal_weights(view, CONFIG, Store.protocol)
        before = view.sample_weights.copy()
        self.assertGreater(before.min(), CONFIG['training']['temporal_weighting']['floor'])
        self.assertGreater(before.max(), before.min())
        self.assertLess(before.max(), 1)
        other = copy.deepcopy(Store.protocol)
        other['splits']['validation'] = {'start': '2025-01-01', 'stop': '2025-02-01'}
        attach_temporal_weights(view, CONFIG, other)
        np.testing.assert_array_equal(before, view.sample_weights)
        resident = type('Resident', (), {})()
        times = block.keys.date.to_numpy(dtype='datetime64[ns]').astype('datetime64[D]').astype('datetime64[ns]').astype(np.int64)
        resident.arrays = {'days': np.unique(times)}
        resident.day = np.searchsorted(resident.arrays['days'], times)
        view.base = resident
        attach_temporal_weights(view, CONFIG, Store.protocol)
        np.testing.assert_array_equal(before, view.sample_weights)
        view.base = block
        block.keys['date'] += pd.Timedelta(days=2000)
        with self.assertRaises(ValueError):
            attach_temporal_weights(view, CONFIG, Store.protocol)

    def test_balanced_assignment_and_weighted_gradient_ignore_missing_targets(self):
        costs = (torch.ones(4, 4) - torch.eye(4)).requires_grad_()
        assignment = balanced_assignment(costs, .25, 12)
        self.assertFalse(assignment.requires_grad)
        torch.testing.assert_close(assignment.sum(0), torch.ones(4, dtype=torch.float64))
        torch.testing.assert_close(assignment.sum(1), torch.ones(4, dtype=torch.float64))
        self.assertTrue((assignment.diag() > .9).all())
        uniform = balanced_assignment(torch.ones(7, 4), .25, 12)
        torch.testing.assert_close(uniform, torch.full((7, 4), .25, dtype=torch.float64))
        prediction = torch.zeros(2, 8, 2, requires_grad=True)
        experts = torch.zeros(2, 8, 4, 2, requires_grad=True)
        logits = torch.zeros(2, 8, 4, requires_grad=True)
        targets = torch.full((2, 8, 2), float('nan'), dtype=torch.float64)
        targets[:, 0] = 1
        loss, details = combined_objective(prediction, experts, logits, targets, torch.tensor([1., .5]), CONFIG['training'])
        loss.backward()
        self.assertEqual(details['rows'], 2)
        self.assertAlmostEqual(float(prediction.grad[0, 0, 0] / prediction.grad[1, 0, 0]), 2)
        self.assertEqual(prediction.grad[:, 1:].count_nonzero().item(), 0)
        self.assertEqual(experts.grad[:, 1:].count_nonzero().item(), 0)
        self.assertEqual(logits.grad[:, 1:].count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
