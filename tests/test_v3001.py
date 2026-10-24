"""v3001 新计算图、股票日标签和恢复语义的少量 CPU 合成检查。"""

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
for version in ("v501", "v3001"):
    sys.path.insert(0, str(ROOT / "solutions" / version))
import v3001_train as trainer
import v501_train as parent
from v3001_model import MultiFrequencyDeltaNet, StructuredFieldProjection
from v3001_grouped import StockDayIndex, training_view, record_group_plan
from v3001_data import prediction_clock_values
from v3001_loss import standardized_targets
from v3001_targets import prepare_rank_training_targets
from v3001_temporal import attach_temporal_weights
from v3001_objective import balanced_assignment, combined_objective

CONFIG = json.loads((ROOT / "solutions/v3001/model-config.json").read_text(encoding="utf-8"))
CONFIGS = {"v3001": CONFIG}
TEST_VERSIONS = ("v3001",)
PARENTS = {"v3001": "v2401"}
PROTOCOL = json.loads((ROOT / "evaluation/protocols/local-v2-ic.json").read_text(encoding="utf-8"))
torch.set_num_threads(1)


def small_config(version="v3001", resident=False):
    config = copy.deepcopy(CONFIG)
    config["model"].update(hidden_dim=8, fusion_hidden_dim=16, expert_hidden_dim=8, router_hidden_dim=8, book_level_dim=4, book_mixer_hidden_dim=4, non_book_dim=4)
    config["model"].update(delta_heads=2, delta_key_dim=4, delta_value_dim=4)
    config["training"].update(epochs=2, cpu_threads=1, batch_size=16, stock_day_batch_size=2,
                              checkpoint_batches=1)
    config["inference"].update(batch_size=16, stock_day_batch_size=2)
    config["training"]["resident_cache"].update(memory_reserve_gib=2.0, build_batch_rows=256, evaluation_days=1)
    config["model"]["price_memory"].update(width=8, heads=2, distance_hidden_dim=5)
    if not resident:
        config["training"]["batching"] = "monthly_shuffled_stock_days"
        config["data"]["input_precision"] = "float32"
    return config


def inputs(config, count=3):
    generator = torch.Generator().manual_seed(55)
    windows = {f: torch.randn(count, 77, n, generator=generator) for f, n in config["data"]["window_bars"].items()}
    for value in windows.values():
        value[:, 72:75] = 1
        value[:, 75] = 2. + .01 * value[:, 75]
        value[:, 76] = 1
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
        self.windows = {f: rng.normal(size=(6, 77, n)).astype(np.float32) for f, n in CONFIG["data"]["window_bars"].items()}
        for value in self.windows.values():
            value[:, 72:75] = 1
            value[:, 75] = 2. + .01 * value[:, 75]
            value[:, 76] = 1

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

    def months(self, split):
        return list(self.data[split])

    def load(self, split, month):
        return self.data[split][month]



def activate_price_memory(model):
    with torch.no_grad():
        for memory in model.price_memories.values():
            memory.price_bias[-1].weight.normal_(0,.05)
            memory.age_bias.weight.normal_(0,.05)

class V3001Tests(unittest.TestCase):
    def test_text_parameter_shards_match_notebook_and_tensor_state(self):
        from v3001_text_artifact import encode_state,decode_state,read_bundle,FORMAT
        import v3001_notebook as notebook
        model=MultiFrequencyDeltaNet(small_config()).eval();activate_price_memory(model)
        encoded=encode_state(model.state_dict());names=list(encoded)
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            shards=[]
            for part,selected in enumerate([names[::2],names[1::2]]):
                name=f'weights-{part}.json';shards.append(name)
                (directory/name).write_text(json.dumps({'state_dict':{k:encoded[k] for k in selected}}),encoding='utf-8')
            index=directory/'v3001-model.json'
            index.write_text(json.dumps({'format':FORMAT,'state_dict_shards':shards}),encoding='utf-8')
            bundle=read_bundle(index)
            with patch.object(notebook,'_bundle_directory',lambda:directory):self.assertEqual(notebook._read_bundle(),bundle)
            restored=decode_state(bundle)
            for key,value in model.state_dict().items():torch.testing.assert_close(value,restored[key],rtol=0,atol=0)
            index.write_text(json.dumps({'format':FORMAT,'state_dict_shards':[shards[0],shards[0]]}),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'重复'):read_bundle(index)

    def test_defaults_and_parent_input_contract(self):
        from price_memory_inputs import base_configuration
        original=json.loads((ROOT/'solutions/v1601/model-config.json').read_text(encoding='utf-8'))
        self.assertEqual(base_configuration(CONFIG)['data'],original['data'])
        trainer.validate_training(CONFIG,PROTOCOL)
        self.assertEqual(CONFIG['training']['resident_cache']['memory_reserve_gib'],2.)
        self.assertEqual(sum(p.numel() for p in MultiFrequencyDeltaNet(CONFIG).parameters()),3236715)
        self.assertEqual(inspect.getsource(trainer.prepare_training_targets),inspect.getsource(parent.prepare_training_targets))

    def test_encoder_independent_recurrence_and_empty_history(self):
        from v3001_schema import prepare_model_inputs
        config = small_config()
        model = MultiFrequencyDeltaNet(config).eval()
        windows, clock = inputs(config, count=2)
        windows, lengths = prepare_model_inputs(windows, clock, model.window_bars)
        with torch.no_grad():
            for frequency, encoder in model.encoders.items():
                raw = windows[frequency][:, :75].clone()
                raw[0] = 0
                raw[1, :, -40:] = 0
                actual = encoder(raw, lengths[frequency])
                expected = encoder(raw, lengths[frequency], recurrent_reference=True)
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
                self.assertEqual(actual[0].count_nonzero().item(), 0)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertGreater(actual[1].abs().sum().item(), 0)

    def test_text_roundtrip_preserves_memory_and_prediction(self):
        from v3001_text_artifact import encode_state,decode_state,FORMAT
        model=MultiFrequencyDeltaNet(small_config()).eval();activate_price_memory(model)
        restored=MultiFrequencyDeltaNet(small_config()).eval()
        restored.load_state_dict(decode_state({'format':FORMAT,'state_dict':encode_state(model.state_dict())}))
        for name,value in model.state_dict().items():torch.testing.assert_close(value,restored.state_dict()[name],rtol=0,atol=0)
        windows,clock=inputs(small_config())
        with torch.no_grad():torch.testing.assert_close(model.forward_group(windows,clock),restored.forward_group(windows,clock),rtol=0,atol=0)

    def test_bfloat16_autocast_keeps_recurrence_and_gradients_finite(self):
        config = small_config()
        model = MultiFrequencyDeltaNet(config)
        windows, clock = inputs(config, count=2)
        targets = torch.randn(2, 8, 2, dtype=torch.float64)
        result = trainer.training_step(model, torch.optim.AdamW(model.parameters()),
                                       windows, clock, targets, config, True)
        self.assertTrue(math.isfinite(result['loss']))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        model.eval()
        with torch.inference_mode():
            prediction = model.forward_group(windows, clock)
        self.assertEqual(prediction.dtype, torch.float32)
        self.assertTrue(torch.isfinite(prediction).all())

    def test_shared_heads_match_reference_and_are_causal(self):
        config = small_config()
        model = MultiFrequencyDeltaNet(config).eval()
        activate_price_memory(model)
        windows, latest = inputs(config)
        windows = {f: value.requires_grad_(True) for f, value in windows.items()}
        grouped = model.forward_group(windows, latest)
        for clock_id in range(8):
            clock = F.one_hot(torch.full((3,), clock_id), 8).float()
            with torch.no_grad():
                reference = model.forward_heads(windows, clock)
                torch.testing.assert_close(grouped[:, clock_id], reference, atol=2e-6, rtol=1e-5)
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
        torch.testing.assert_close(shared[torch.arange(3), partial.argmax(1)], expected, atol=2e-6, rtol=1e-5)
        target = torch.randn(3, 8, 2, dtype=torch.float64)
        step = trainer.training_step(model, torch.optim.AdamW(model.parameters()), windows, latest, target, config, False)
        self.assertEqual(step["rows"], 24)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(model.experts[0][-1].weight.grad[1].abs().sum().item(), 0)


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


if __name__ == "__main__":
    unittest.main()
