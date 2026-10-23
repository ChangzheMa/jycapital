"""v2501 新计算图、股票日标签和恢复语义的少量 CPU 合成检查。"""

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
for version in ("v501", "v2501"):
    sys.path.insert(0, str(ROOT / "solutions" / version))
import v2501_train as trainer
import v501_train as parent
from v2501_model import MultiFrequencyGRU, StructuredFieldProjection
from v2501_grouped import StockDayIndex, training_view, record_group_plan
from v2501_data import prediction_clock_values
from v2501_loss import standardized_targets
from v2501_targets import prepare_rank_training_targets
from v2501_temporal import attach_temporal_weights
from v2501_objective import balanced_assignment, combined_objective

CONFIG = json.loads((ROOT / "solutions/v2501/model-config.json").read_text(encoding="utf-8"))
CONFIGS = {"v2501": CONFIG}
TEST_VERSIONS = ("v2501",)
PARENTS = {"v2501": "v1601"}
PROTOCOL = json.loads((ROOT / "evaluation/protocols/local-v2-ic.json").read_text(encoding="utf-8"))
torch.set_num_threads(1)


def small_config(version="v2501", resident=False):
    config = copy.deepcopy(CONFIG)
    config["model"].update(hidden_dim=8, fusion_hidden_dim=16, expert_hidden_dim=8, router_hidden_dim=8, book_level_dim=4, book_mixer_hidden_dim=4, non_book_dim=4)
    config["training"].update(epochs=2, cpu_threads=1, batch_size=16, stock_day_batch_size=2,
                              checkpoint_batches=1)
    config["inference"].update(batch_size=16, stock_day_batch_size=2)
    config["training"]["resident_cache"].update(memory_reserve_gib=2.0, build_batch_rows=256, evaluation_days=1)
    config["model"]["future_return_head"]["hidden_dim"] = 8
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


class V2501Tests(unittest.TestCase):
    def test_total_training_loss_and_diagnostic_counts_by_hand(self):
        class Constant(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scores=torch.nn.Parameter(torch.zeros(1,8,2))
                self.future=torch.nn.Parameter(torch.zeros(1,28))
            def forward_training_group(self,windows,clock):
                return self.scores,self.scores[:,:,None,:].expand(-1,-1,4,-1),torch.zeros(1,8,4),self.future
        model=Constant()
        target=torch.full((1,8,3),float('nan'),dtype=torch.float64)
        target[...,2]=1.
        target[0,1,:2]=1.
        target[0,2,:2]=3.
        step=trainer.training_step(model,torch.optim.SGD(model.parameters(),lr=0),{},torch.zeros(1,8),target,CONFIG,False)
        primary=(2.25+.2*1.5)*1.1+.01*math.log(4)
        self.assertAlmostEqual(step['loss'],primary+.5*3.125)
        self.assertEqual(step['rows'],2)
        state=trainer.initial_progress()
        state['rows']=step['rows']
        trainer.accumulate_future_progress(state,step)
        history=trainer.future_history(state)
        self.assertEqual(history['train_future_valid_pairs'],3)
        self.assertEqual(history['train_future_valid_queries'],2)
        self.assertEqual(history['train_future_huber_loss'],3.125)
        self.assertEqual(history['train_future_zero_huber_loss'],3.125)

    def test_bfloat16_combined_objective_has_finite_gradients(self):
        config=small_config()
        model=MultiFrequencyGRU(config)
        windows,clock=inputs(config,count=2)
        target=trainer.preflight_targets(config,2,torch.device('cpu'))
        step=trainer.training_step(model,torch.optim.AdamW(model.parameters()),windows,clock,target,config,True)
        self.assertTrue(math.isfinite(step['loss']))
        self.assertEqual(step['future_pairs'],56)
        self.assertEqual(step['future_queries'],14)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_defaults_primary_targets_and_input_identity(self):
        original=json.loads((ROOT/'solutions/v1601/model-config.json').read_text(encoding='utf-8'))
        self.assertEqual(CONFIG['data'],original['data'])
        trainer.validate_training(CONFIG,PROTOCOL)
        self.assertEqual(CONFIG['training']['resident_cache']['memory_reserve_gib'],2.)
        self.assertEqual(sum(p.numel() for p in MultiFrequencyGRU(CONFIG).parameters()),2570193)
        self.assertEqual(inspect.getsource(trainer.prepare_training_targets),inspect.getsource(parent.prepare_training_targets))
        block=Block('2020-01',1)
        block.keys=block.keys.drop(index=[7,13]).sample(frac=1,random_state=5).reset_index(drop=True)
        y,_=trainer.prepare_training_targets(block.keys,Store.protocol)
        rank,_=prepare_rank_training_targets(block.keys,Store.protocol)
        view=training_view(block,y,rank)
        values=view.targets[np.arange(len(view.index.samples))]
        rows=view.index.samples
        np.testing.assert_array_equal(values[...,0],np.where(rows>=0,y[rows.clip(0)],np.nan))
        np.testing.assert_array_equal(values[...,1],np.where(rows>=0,rank[rows.clip(0)],np.nan))
        np.testing.assert_array_equal(values[...,2],rows>=0)

    def test_inference_omits_future_head_and_preserves_initial_parent(self):
        sys.path.insert(0,str(ROOT/'solutions/v1601'))
        from v1601_model import MultiFrequencyGRU as ParentModel
        config=small_config();base=copy.deepcopy(config);base['solution_id']='v1601'
        torch.manual_seed(42);previous=ParentModel(base).eval()
        torch.manual_seed(42);current=MultiFrequencyGRU(config).eval()
        for key,value in previous.state_dict().items():torch.testing.assert_close(value,current.state_dict()[key],rtol=0,atol=0)
        windows,clock=inputs(config)
        with torch.no_grad(),patch.object(current.future_return_head,'forward',side_effect=AssertionError('auxiliary head used at inference')):
            torch.testing.assert_close(current.forward_group(windows,clock),previous.forward_group(windows,clock),rtol=0,atol=0)
            torch.testing.assert_close(current(windows,clock),previous(windows,clock),rtol=0,atol=0)

    def test_future_predictions_are_causal_and_match_single_query_reference(self):
        from v2501_future import future_pairs
        from v2501_schema import prepare_model_inputs,FREQUENCIES
        config=small_config();model=MultiFrequencyGRU(config).eval()
        windows,latest=inputs(config,count=2);windows={f:x.requires_grad_(True) for f,x in windows.items()}
        values=model.forward_training_group(windows,latest)[3]
        query,target=future_pairs()
        for q in range(7):
            clock=F.one_hot(torch.full((2,),q),8).float()
            masked,lengths=prepare_model_inputs(windows,clock,config['data']['window_bars'])
            vectors=[model.encoders[f](masked[f],lengths[f]) for f in FREQUENCIES]
            shared=model.fuse_components(vectors,clock,return_features=True)[3]
            for index in torch.nonzero(query==q).flatten().tolist():
                expected=model.future_return_head.readout(model.future_return_head.projection(shared)+model.future_return_head.target_clock(target[index]-1)+model.future_return_head.lead(target[index]-q-1)).squeeze(-1)
                torch.testing.assert_close(values[:,index],expected,rtol=1e-5,atol=3e-7)
            gradient=torch.autograd.grad(values[:,query==q].sum(),tuple(windows.values()),retain_graph=True)
            for f,g in zip(windows,gradient):
                end=int(lengths[f][0]);self.assertEqual(g[:,:,end:].count_nonzero().item(),0)

    def test_text_roundtrip_includes_auxiliary_head(self):
        from v2501_text_artifact import encode_state,decode_state,FORMAT
        model=MultiFrequencyGRU(small_config()).eval();other=MultiFrequencyGRU(small_config()).eval()
        other.load_state_dict(decode_state({'format':FORMAT,'state_dict':encode_state(model.state_dict())}))
        for key,value in model.state_dict().items():torch.testing.assert_close(value,other.state_dict()[key],rtol=0,atol=0)
        windows,clock=inputs(small_config())
        with torch.no_grad():
            for a,b in zip(model.forward_training_group(windows,clock),other.forward_training_group(windows,clock)):torch.testing.assert_close(a,b,rtol=0,atol=0)

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
        target = torch.cat((torch.randn(3,8,2,dtype=torch.float64), torch.ones(3,8,1,dtype=torch.float64)),dim=-1)
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
