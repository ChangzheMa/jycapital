"""v1101 新计算图、股票日标签和恢复语义的少量 CPU 合成检查。"""

import copy
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
for version in ("v501", "v1101"):
    sys.path.insert(0, str(ROOT / "solutions" / version))
import v1101_train as trainer
import v501_train as parent
from v1101_model import MultiFrequencyGRU, StructuredFieldProjection
from v1101_grouped import StockDayIndex, training_view, record_group_plan
from v1101_data import prediction_clock_values
from v1101_loss import standardized_targets
from v1101_targets import prepare_rank_training_targets

CONFIG = json.loads((ROOT / "solutions/v1101/model-config.json").read_text(encoding="utf-8"))
CONFIGS = {"v1101": CONFIG}
TEST_VERSIONS = ("v1101",)
PARENTS = {"v1101": "v901"}
PROTOCOL = json.loads((ROOT / "evaluation/protocols/local-v2-ic.json").read_text(encoding="utf-8"))
torch.set_num_threads(1)


def small_config(version="v1101", resident=False):
    config = copy.deepcopy(CONFIG)
    config["model"].update(hidden_dim=8, fusion_hidden_dim=16, book_level_dim=4, book_mixer_hidden_dim=4, non_book_dim=4)
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
            previous = CONFIG["data"]["trading_day_windows"][frequency]["previous_days"]
            cutoff = previous * bars + clocks.argmax(1) * (bars // 8)
            value *= np.arange(value.shape[2])[None, None, :] < cutoff[:, None, None]
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


class V1101Tests(unittest.TestCase):
    def test_new_input_contract_preserves_normalization_targets_and_isolates_cache(self):
        previous = json.loads((ROOT / "solutions/v501/model-config.json").read_text(encoding="utf-8"))
        self.assertEqual(CONFIG["data"]["normalization"], previous["data"]["normalization"])
        self.assertEqual(CONFIG["data"]["masking"], previous["data"]["masking"])
        self.assertEqual(CONFIG["data"]["window_bars"], {"bar1m": 480, "bar5m": 288, "bar30m": 168})
        self.assertTrue(all(v["include_today"] for v in CONFIG["data"]["trading_day_windows"].values()))
        import v1101_resident
        import resident_data
        self.assertNotEqual(v1101_resident.FORMAT, resident_data.FORMAT)
        self.assertEqual(inspect.getsource(trainer.prepare_training_targets), inspect.getsource(parent.prepare_training_targets))
        self.assertEqual(inspect.getsource(standardized_targets), inspect.getsource(parent.standardized_targets))
        trainer.validate_training(CONFIG, PROTOCOL)
        self.assertEqual(sum(p.numel() for p in MultiFrequencyGRU(CONFIG).parameters()), 2296710)

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
        self.assertGreater(model.head[-1].weight.grad[1].abs().sum().item(), 0)

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

    def test_device_windows_restore_overflow_and_mask_future_for_every_frequency(self):
        from v1101_resident import prepare_device_windows, DAILY_BARS, PREVIOUS_DAYS
        raw, repairs = {}, {}
        clock = F.one_hot(torch.tensor([3, 0]), 8).float()
        for frequency, length in CONFIG["data"]["window_bars"].items():
            raw[frequency] = torch.ones(2, length, 72, dtype=torch.float16)
            raw[frequency][0, 0, 0] = float("nan")
            bars = DAILY_BARS[frequency]
            cutoff = PREVIOUS_DAYS[frequency] * bars + 3 * (bars // 8)
            positions = torch.tensor([(cutoff - 1) * 72 + 1, cutoff * 72 + 1])
            repairs[frequency] = (positions, torch.tensor([1e8, -1e8]))
        windows = prepare_device_windows(raw, repairs, clock)
        for frequency, value in windows.items():
            bars = DAILY_BARS[frequency]
            cutoff = PREVIOUS_DAYS[frequency] * bars + 3 * (bars // 8)
            self.assertEqual(value[0, 1, cutoff - 1].item(), 1e8)
            self.assertEqual(value[0, :, cutoff:].count_nonzero().item(), 0)
            self.assertEqual(value[1, :, PREVIOUS_DAYS[frequency] * bars:].count_nonzero().item(), 0)
            self.assertEqual(value[0, 0, 0].item(), 0)
            self.assertEqual(value[0, 72, 0].item(), 0)
            self.assertEqual(value[0, 73, 0].item(), 1)
            self.assertEqual(value[0, 74, 0].item(), 1)

    def test_actual_training_loss_combines_heads_and_masks_missing_labels(self):
        class Constant(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scores = torch.nn.Parameter(torch.zeros(1, 8, 2))

            def forward_group(self, windows, clock):
                return self.scores

        model = Constant()
        targets = torch.full((1, 8, 2), float("nan"), dtype=torch.float64)
        targets[0, 1] = torch.tensor([3., 1.])
        result = trainer.training_step(model, torch.optim.SGD(model.parameters(), lr=0), {}, torch.zeros(1, 8),
                                       targets, CONFIG, False)
        self.assertAlmostEqual(result["loss"], 4.1)
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
            columns = ["train_loss", "train_huber_loss", "train_rank_huber_loss", "validation_loss", "ic_mean", "rank_ic_mean"]
            pd.testing.assert_frame_equal(frames[0][columns], frames[1][columns], check_exact=True)
            for manifest in manifests:
                self.assertEqual(manifest["status"], "complete")
                for split in ("validation", "test"):
                    self.assertEqual(manifest["metrics"][split]["coverage"], 1)


    def test_field_mapping_and_dense_residual_preserve_original_inputs(self):
        spec=small_config()['model']
        projection=StructuredFieldProjection(spec).eval()
        from v1101_preprocessing import output_columns
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

    def test_same_input_contract_preserves_parent_training_cache_identity(self):
        import v1101_resident as current
        path=ROOT/'solutions/v901/v901_resident.py'
        self.assertEqual(path.read_bytes(), (ROOT/'solutions/v1101/v1101_resident.py').read_bytes())
        spec=importlib.util.spec_from_file_location('v901_cache_reference',path)
        reference=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        previous=json.loads((ROOT/'solutions/v901/model-config.json').read_text(encoding='utf-8'))
        self.assertEqual(CONFIG['data'],previous['data'])
        left=SimpleNamespace(config=CONFIG,protocol=PROTOCOL)
        right=SimpleNamespace(config=previous,protocol=PROTOCOL)
        metadata={'synthetic':True}
        self.assertEqual(current.cache_key(left,metadata,trainer.prepare_training_targets),
                         reference.cache_key(right,metadata,trainer.prepare_training_targets))


if __name__ == "__main__":
    unittest.main()
