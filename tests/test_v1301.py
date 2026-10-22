"""v1301 新计算图、股票日标签和恢复语义的少量 CPU 合成检查。"""

import copy
import math
import inspect
import importlib.util
from types import SimpleNamespace
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
for version in ("v501", "v1301"):
    sys.path.insert(0, str(ROOT / "solutions" / version))
import v1301_train as trainer
import v501_train as parent
from v1301_model import MultiFrequencyGRU
from v1301_grouped import StockDayIndex, training_view, record_group_plan
from v1301_data import prediction_clock_values
from v1301_loss import standardized_targets
from v1301_targets import prepare_rank_training_targets
from v1301_temporal import attach_temporal_weights
from v1301_objective import balanced_assignment, combined_objective, stabilized_correlation, full_day_objective
from v1301_calendar import CalendarDayPlan, calendar_batches
from v1301_recompute import grouped_components
from v1301_ema import WeightEMA

CONFIG = json.loads((ROOT / "solutions/v1301/model-config.json").read_text(encoding="utf-8"))
CONFIGS = {"v1301": CONFIG}
TEST_VERSIONS = ("v1301",)
PARENTS = {"v1301": "v1002"}
PROTOCOL = json.loads((ROOT / "evaluation/protocols/local-v2-ic.json").read_text(encoding="utf-8"))
torch.set_num_threads(1)


def small_config(version="v1301", resident=False):
    config = copy.deepcopy(CONFIG)
    config["model"].update(hidden_dim=8, fusion_hidden_dim=16, expert_hidden_dim=8, router_hidden_dim=8)
    config["training"].update(epochs=2, cpu_threads=1, batch_size=64, stock_day_batch_size=8, microbatch_stock_days=2,
                              checkpoint_batches=1)
    config["inference"].update(batch_size=16, stock_day_batch_size=2)
    config["training"]["resident_cache"].update(memory_reserve_gib=0, build_batch_rows=256, evaluation_days=1)
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
            third = self.keys.copy()
            third['date'] += pd.Timedelta(days=10)
            self.keys = pd.concat([self.keys, extra, third], ignore_index=True)
        self.windows = {f: rng.normal(size=(6, 75, n)).astype(np.float32) for f, n in CONFIG["data"]["window_bars"].items()}
        for value in self.windows.values():
            value[:, 72:] = 1

    def batch(self, rows):
        keys = self.keys.iloc[rows]
        clocks = prediction_clock_values(keys.date.to_numpy(dtype="datetime64[ns]").astype(np.int64))
        stocks = keys.instrument.str[1:].astype(int).to_numpy()
        values = {f: x[stocks].copy() for f, x in self.windows.items()}
        for frequency, value in values.items():
            bars = 240 // int(frequency[3:-1])
            previous = CONFIG['data']['trading_day_windows'][frequency]['previous_days']
            value *= np.arange(value.shape[-1])[None, None, :] < (previous * bars + (bars // 8) * clocks.argmax(1))[:, None, None]
        return values, clocks


class Store:
    protocol = dict(PROTOCOL, min_cross_section_size=3)

    def __init__(self):
        self.data = {split: {month: Block(month, seed)} for split, month, seed in (
            ("train", "2020-01", 1), ("validation", "2024-01", 3), ("test", "2024-07", 4))}
        self.data["train"]["global"] = self.data["train"].pop("2020-01")

    def months(self, split):
        return list(self.data[split])

    def load(self, split, month):
        return self.data[split][month]


class V1301Tests(unittest.TestCase):
    def test_data_and_primary_target_cache_identity(self):
        previous = json.loads((ROOT / "solutions/v901/model-config.json").read_text(encoding="utf-8"))
        self.assertEqual(CONFIG["data"], previous["data"])
        self.assertEqual(inspect.getsource(trainer.prepare_training_targets), inspect.getsource(parent.prepare_training_targets))
        self.assertEqual(inspect.getsource(standardized_targets), inspect.getsource(parent.standardized_targets))
        trainer.validate_training(CONFIG, PROTOCOL)
        self.assertEqual(sum(p.numel() for p in MultiFrequencyGRU(CONFIG).parameters()), 2418576)

    def test_shared_heads_match_reference_and_are_causal(self):
        config = small_config()
        model = MultiFrequencyGRU(config).eval()
        windows, latest = inputs(config)
        windows = {f: value.requires_grad_(True) for f, value in windows.items()}
        grouped = model.forward_group(windows, latest)
        for clock_id in range(8):
            clock = F.one_hot(torch.full((3,), clock_id), 8).float()
            with torch.no_grad():
                reference = model.forward_heads(windows, clock)
                torch.testing.assert_close(grouped[:, clock_id], reference, atol=2e-7, rtol=1e-5)
            gradients = torch.autograd.grad(grouped[:, clock_id].sum(), tuple(windows.values()), retain_graph=True)
            for frequency, gradient in zip(windows, gradients):
                bars = 240 // int(frequency[3:-1])
                cutoff = CONFIG["data"]["trading_day_windows"][frequency]["previous_days"] * bars + (bars // 8) * clock_id
                self.assertEqual(gradient[:, :, cutoff:].count_nonzero().item(), 0)
                self.assertGreater(gradient[:, :, :cutoff].abs().sum().item(), 0)
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
                self.assertEqual(checkpoint["progress"]["group_cursor"], 12)
                self.assertEqual(checkpoint["progress"]["sample_cursor"], 96)
                self.assertEqual(checkpoint["progress"]["day_cursor"], 2)
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
            self.assertEqual(left["ema_stock_days"], right["ema_stock_days"])
            self.assertEqual(left["ema_stock_days"], 36)
            frames = [pd.read_csv(run / "history.csv") for run in runs]
            columns = ["train_loss", "train_huber_loss", "train_rank_huber_loss", "validation_loss", "ic_mean", "rank_ic_mean", "train_expert_loss", "train_router_loss", "train_routing_entropy", "train_routing_usage", "temporal_weighting"]
            columns += ["train_" + key for key in trainer.CALENDAR_STATISTICS]
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


    def test_fresh_parameter_initialization_matches_parent(self):
        sys.path.insert(0,str(ROOT/'solutions/v1002'))
        from v1002_model import MultiFrequencyGRU as ParentModel
        config=copy.deepcopy(CONFIG)
        previous=json.loads((ROOT/'solutions/v1002/model-config.json').read_text(encoding='utf-8'))
        self.assertEqual(config['model'],previous['model'])
        torch.manual_seed(42)
        actual=MultiFrequencyGRU(config)
        torch.manual_seed(42)
        expected=ParentModel(previous)
        self.assertEqual(actual.state_dict().keys(),expected.state_dict().keys())
        for name,value in actual.state_dict().items():
            torch.testing.assert_close(value,expected.state_dict()[name],rtol=0,atol=0)

    def test_current_input_training_cache_matches_v901_identity(self):
        import v1301_resident as current
        path=ROOT/'solutions/v901/v901_resident.py'
        self.assertEqual(path.read_bytes(),(ROOT/'solutions/v1301/v1301_resident.py').read_bytes())
        spec=importlib.util.spec_from_file_location('v901_cache_reference_for_v1301',path)
        reference=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        previous=json.loads((ROOT/'solutions/v901/model-config.json').read_text(encoding='utf-8'))
        self.assertEqual(CONFIG['data'],previous['data'])
        self.assertEqual(current.cache_key(SimpleNamespace(config=CONFIG,protocol=PROTOCOL),{'synthetic':True},trainer.prepare_training_targets),
            reference.cache_key(SimpleNamespace(config=previous,protocol=PROTOCOL),{'synthetic':True},trainer.prepare_training_targets))



    def test_complete_calendar_plan_preserves_membership_weights_and_boundaries(self):
        block = Block('2020-01', 1)
        remove = block.keys.date.dt.day.eq(22) & block.keys.instrument.eq('S5')
        block.keys = block.keys.loc[~remove].sample(frac=1, random_state=31).reset_index(drop=True)
        primary, _ = trainer.prepare_training_targets(block.keys, Store.protocol)
        rank, _ = prepare_rank_training_targets(block.keys, Store.protocol)
        view = training_view(block, primary, rank)
        attach_temporal_weights(view, CONFIG, Store.protocol)
        raw = view.sample_weights.copy()
        plan = CalendarDayPlan(view, CONFIG, Store.protocol)
        order = plan.order(42, 1)
        all_groups = np.concatenate([plan.groups[i] for i in order])
        np.testing.assert_array_equal(np.sort(all_groups), np.flatnonzero(view.eligible))
        self.assertEqual(len(np.unique(all_groups)), len(all_groups))
        self.assertEqual(sorted(map(len, plan.groups)), [5, 6, 6])
        self.assertAlmostEqual(np.mean([view.sample_weights[g[0]] for g in plan.groups]), 1.)
        for offset, groups, windows, clock, targets in calendar_batches(view, plan, order, 0, 'cpu'):
            dates = block.keys.iloc[view.index.representatives[groups]].date.dt.normalize()
            self.assertEqual(dates.nunique(), 1)
            self.assertEqual(len(groups), len(plan.groups[order[offset]]))
            np.testing.assert_array_equal(raw[groups], np.repeat(raw[groups[0]], len(groups)))
            np.testing.assert_array_equal(targets.numpy(), view.targets[groups])
        resumed = list(calendar_batches(view, plan, order, 1, 'cpu'))
        self.assertEqual([item[0] for item in resumed], [1, 2])
        np.testing.assert_array_equal(primary, view.targets.primary)
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            trainer.write_json(run / 'data-manifest.json', {})
            plan.record_identity(run)
            plan.record_identity(run)
            plan.record['dates'] += 1
            with self.assertRaises(ValueError):
                plan.record_identity(run)
        limited = copy.deepcopy(CONFIG)
        limited['training']['stock_day_batch_size'] = 4
        with self.assertRaises(ValueError):
            CalendarDayPlan(view, limited, Store.protocol)

    def test_correlation_matches_clockwise_reference_and_is_not_pooled(self):
        generator = torch.Generator().manual_seed(9)
        prediction = torch.randn(9, 8, generator=generator, dtype=torch.float64, requires_grad=True)
        targets = torch.randn(9, 8, generator=generator, dtype=torch.float64)
        targets[0, 1] = float('nan')
        targets[:, 7] = float('nan')
        targets[:, 6] = 2.
        actual, std, sections = stabilized_correlation(prediction, targets, .02)
        expected = []
        for clock in range(6):
            valid = torch.isfinite(targets[:, clock])
            p, y = prediction[valid, clock], targets[valid, clock]
            p, y = p - p.mean(), y - y.mean()
            expected.append((p * y).mean() / torch.sqrt((p.square().mean() + .02 ** 2) * (y.square().mean() + .02 ** 2)))
        torch.testing.assert_close(actual, torch.stack(expected).mean())
        self.assertEqual(sections, 6)
        self.assertGreater(float(std), 0)
        self.assertTrue(torch.autograd.gradcheck(lambda p: stabilized_correlation(p, targets, .02)[0], (prediction,)))
        shifted = prediction + torch.arange(8) * 100.
        torch.testing.assert_close(actual, stabilized_correlation(shifted, targets, .02)[0], rtol=1e-12, atol=1e-12)
        actual.backward()
        self.assertEqual(prediction.grad[0, 1].item(), 0)
        self.assertEqual(prediction.grad[:, 6:].count_nonzero().item(), 0)

    def test_constant_predictions_have_finite_correlation_gradient_and_day_weight_survives(self):
        prediction = torch.zeros(4, 8, 2, dtype=torch.float64, requires_grad=True)
        experts = torch.zeros(4, 8, 4, 2, dtype=torch.float64, requires_grad=True)
        logits = torch.zeros(4, 8, 4, dtype=torch.float64, requires_grad=True)
        target = torch.arange(4, dtype=torch.float64)[:, None, None].expand(4, 8, 2).clone() - 1.5
        correlation, _, _ = stabilized_correlation(prediction[..., 0], target[..., 0], .02)
        gradient = torch.autograd.grad(correlation, prediction, retain_graph=True)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
        first, _ = full_day_objective(prediction, experts, logits, target, torch.ones(4), CONFIG['training'])
        second, _ = full_day_objective(prediction, experts, logits, target, torch.full((4,), 2.), CONFIG['training'])
        torch.testing.assert_close(second, 2 * first, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            full_day_objective(prediction, experts, logits, target, torch.arange(1., 5.), CONFIG['training'])

    def test_recomputation_matches_retained_graph_outputs_gradients_and_rng_with_dropout(self):
        config = small_config()
        windows, clock = inputs(config, count=5)
        target = torch.randn(5, 8, 2, dtype=torch.float64)
        target[0, 2] = float('nan')
        for amp in (False, True):
            with self.subTest(cpu_bfloat16=amp):
                torch.manual_seed(47)
                original = MultiFrequencyGRU(config).train()
                other = copy.deepcopy(original)
                initial_rng = torch.get_rng_state().clone()
                results = []
                for model, recompute in ((original, False), (other, True)):
                    torch.set_rng_state(initial_rng)
                    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                        pieces = grouped_components(model, windows, clock, 2, recompute)
                    loss, _ = full_day_objective(*pieces, target, None, config['training'])
                    loss.backward()
                    results.append((loss.detach(), [part.detach() for part in pieces], torch.get_rng_state().clone()))
                torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
                for left, right in zip(results[0][1], results[1][1]):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                torch.testing.assert_close(results[0][2], results[1][2], rtol=0, atol=0)
                for left, right in zip(original.parameters(), other.parameters()):
                    torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)

    def test_microbatching_keeps_full_objective_across_stock_chunks(self):
        config = small_config()
        model = MultiFrequencyGRU(config).eval()
        windows, clock = inputs(config, count=5)
        target = torch.randn(5, 8, 2, dtype=torch.float64)
        with torch.no_grad():
            full = model.forward_group_components(windows, clock)
            pieces = grouped_components(model, windows, clock, 2, False)
        for left, right in zip(full, pieces):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=2e-7)
        whole, _ = full_day_objective(*full, target, None, config['training'])
        chunked, _ = full_day_objective(*pieces, target, None, config['training'])
        torch.testing.assert_close(whole, chunked, rtol=1e-5, atol=2e-7)

    def test_ema_decay_is_scaled_by_actual_stock_days(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        ema = WeightEMA(model, .9, 0, reference_stock_days=4)
        with torch.no_grad():
            model.weight.fill_(1)
        ema.update(1, 8)
        torch.testing.assert_close(ema.model.weight, torch.full_like(model.weight, 1 - .9 ** 2))
        self.assertEqual(ema.stock_days, 8)
        restored = WeightEMA(model, .9, 0, reference_stock_days=4)
        restored.restore(ema.model.state_dict(), 1, 1, 8)
        ema.update(1, 2)
        restored.update(1, 2)
        torch.testing.assert_close(ema.model.weight, restored.model.weight, rtol=0, atol=0)
        self.assertEqual(restored.stock_days, 10)


if __name__ == "__main__":
    unittest.main()
