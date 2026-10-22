"""v1701 新计算图、股票日标签和恢复语义的少量 CPU 合成检查。"""

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
for version in ("v501", "v1701"):
    sys.path.insert(0, str(ROOT / "solutions" / version))
import v1701_train as trainer
import v501_train as parent
from v1701_model import MultiFrequencyGRU, StructuredFieldProjection
from v1701_grouped import StockDayIndex, training_view, record_group_plan
from v1701_data import prediction_clock_values
from v1701_loss import standardized_targets
from v1701_targets import prepare_rank_training_targets
from v1701_temporal import attach_temporal_weights
from v1701_objective import balanced_assignment, combined_objective

CONFIG = json.loads((ROOT / "solutions/v1701/model-config.json").read_text(encoding="utf-8"))
CONFIGS = {"v1701": CONFIG}
TEST_VERSIONS = ("v1701",)
PARENTS = {"v1701": "v1102"}
PROTOCOL = json.loads((ROOT / "evaluation/protocols/local-v2-ic.json").read_text(encoding="utf-8"))
torch.set_num_threads(1)


def small_config(version="v1701", resident=False):
    config = copy.deepcopy(CONFIG)
    config["long_context"].update(previous_days=8, day_dim=8, attention_heads=2, encoder_layers=1)
    config["model"].update(hidden_dim=8, fusion_hidden_dim=16, expert_hidden_dim=8, router_hidden_dim=8, book_level_dim=4, book_mixer_hidden_dim=4, non_book_dim=4)
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
    windows["long30m"] = torch.randn(count, 75, config["long_context"]["previous_days"] * 8, generator=generator)
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
        self.windows["long30m"] = rng.normal(size=(6, 75, 8 * 8)).astype(np.float32)
        for value in self.windows.values():
            value[:, 72:] = 1

    def batch(self, rows):
        keys = self.keys.iloc[rows]
        clocks = prediction_clock_values(keys.date.to_numpy(dtype="datetime64[ns]").astype(np.int64))
        stocks = keys.instrument.str[1:].astype(int).to_numpy()
        values = {f: x[stocks].copy() for f, x in self.windows.items()}
        for frequency, value in values.items():
            if frequency == "long30m":
                continue
            bars = 240 // int(frequency[3:-1])
            previous = CONFIG['data']['trading_day_windows'][frequency]['previous_days']
            value *= np.arange(value.shape[-1])[None, None, :] < (previous * bars + (bars // 8) * clocks.argmax(1))[:, None, None]
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


class V1701Tests(unittest.TestCase):
    def test_data_and_primary_target_cache_identity(self):
        previous = json.loads((ROOT / "solutions/v901/model-config.json").read_text(encoding="utf-8"))
        self.assertEqual(CONFIG["data"], previous["data"])
        self.assertEqual(inspect.getsource(trainer.prepare_training_targets), inspect.getsource(parent.prepare_training_targets))
        self.assertEqual(inspect.getsource(standardized_targets), inspect.getsource(parent.standardized_targets))
        trainer.validate_training(CONFIG, PROTOCOL)
        self.assertEqual(sum(p.numel() for p in MultiFrequencyGRU(CONFIG).parameters()), 2863057)

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
                if frequency == "long30m":
                    continue
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


    def test_fresh_shared_backbone_initialization_matches_field_parent(self):
        sys.path.insert(0,str(ROOT/'solutions/v1101'))
        from v1101_model import MultiFrequencyGRU as ParentModel
        config=copy.deepcopy(CONFIG)
        previous=json.loads((ROOT/'solutions/v1101/model-config.json').read_text(encoding='utf-8'))
        extra={'expert_count','expert_hidden_dim','router_hidden_dim'}
        self.assertEqual({k:v for k,v in config['model'].items() if k not in extra},previous['model'])
        torch.manual_seed(42)
        actual=MultiFrequencyGRU(config).state_dict()
        torch.manual_seed(42)
        expected=ParentModel(previous).state_dict()
        checked=0
        for name,value in expected.items():
            if name.startswith('head.3.'):
                continue
            mapped=name.replace('head.0.','shared.0.')
            torch.testing.assert_close(actual[mapped],value,rtol=0,atol=0)
            checked+=1
        self.assertEqual(checked,len(expected)-2)

    def test_current_input_training_cache_matches_v901_identity(self):
        import v1701_resident as current
        path=ROOT/'solutions/v901/v901_resident.py'
        self.assertEqual(path.read_bytes(),(ROOT/'solutions/v1701/v1701_resident.py').read_bytes())
        spec=importlib.util.spec_from_file_location('v901_cache_reference_for_v1701',path)
        reference=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        previous=json.loads((ROOT/'solutions/v901/model-config.json').read_text(encoding='utf-8'))
        self.assertEqual(CONFIG['data'],previous['data'])
        self.assertEqual(current.cache_key(SimpleNamespace(config=CONFIG,protocol=PROTOCOL),{'synthetic':True},trainer.prepare_training_targets),
            reference.cache_key(SimpleNamespace(config=previous,protocol=PROTOCOL),{'synthetic':True},trainer.prepare_training_targets))

    def test_field_mapping_and_dense_residual_preserve_original_inputs(self):
        spec=small_config()['model']
        projection=StructuredFieldProjection(spec).eval()
        from v1701_preprocessing import output_columns
        columns=output_columns()[2:]
        first=[columns[index] for index in projection.book_indices[0].tolist()]
        self.assertEqual(first, ['z_ask_price1','z_bid_price1','z_ask_volume1','z_bid_volume1',
            'z_ask_num_orders1','z_bid_num_orders1','c_ask_price1','c_bid_price1',
            'c_ask_volume1','c_bid_volume1','c_ask_num_orders1','c_bid_num_orders1'])
        self.assertEqual(set(projection.book_indices.flatten().tolist()), set(range(6,36))|set(range(42,72)))
        x=torch.randn(2,3,75)
        x[...,72:]=torch.tensor([1.,0.,1.])
        with torch.no_grad():
            projection.output_scale.zero_()
            torch.testing.assert_close(projection(x),projection.dense(x),rtol=0,atol=0)

    def test_structured_path_is_pointwise_and_keeps_incomplete_book_values(self):
        projection=StructuredFieldProjection(small_config()['model']).eval()
        x=torch.randn(2,3,75)
        x[...,72:]=torch.tensor([1.,0.,1.])
        x.requires_grad_(True)
        result=projection.structured(x)
        gradient=torch.autograd.grad(result[0,1].sum(),x)[0]
        self.assertEqual(gradient[1].count_nonzero().item(),0)
        self.assertEqual(gradient[0,[0,2]].count_nonzero().item(),0)
        self.assertTrue((gradient[0,1,:72].abs()>0).all())
        self.assertGreater(gradient[0,1,6:36].abs().sum().item(),0)


if __name__ == "__main__":
    unittest.main()
