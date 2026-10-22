"""因果归一化、112通道语义、原目标及恢复的 CPU 合成检查。"""
import copy
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
for name in ("common", "evaluation", "solutions/v1601", "solutions/v2001"):
    sys.path.insert(0, str(ROOT / name))
import v1601_train as parent
import v2001_train as trainer
from coordinate_inputs import base_configuration, prepare_coordinate_windows
from v2001_model import MultiFrequencyGRU, CausalGroupNormalization
from v2001_data import prediction_clock_values
from v2001_objective import combined_objective

CONFIG = json.loads((ROOT / "solutions/v2001/model-config.json").read_text(encoding="utf-8"))
PROTOCOL = json.loads((ROOT / "evaluation/protocols/local-v2-ic.json").read_text(encoding="utf-8"))
CONFIGS, TEST_VERSIONS, PARENTS = {"v2001": CONFIG}, ("v2001",), {"v2001": "v1601"}
torch.set_num_threads(1)


def small_config(version="v2001", resident=False):
    config = copy.deepcopy(CONFIG)
    config["model"].update(hidden_dim=8, fusion_hidden_dim=16, expert_hidden_dim=8, router_hidden_dim=8,
        book_level_dim=4, book_mixer_hidden_dim=4, non_book_dim=4, causal_normalization_hidden_dim=4)
    config["training"].update(epochs=2, cpu_threads=1, batch_size=16, stock_day_batch_size=2, checkpoint_batches=1)
    config["inference"].update(batch_size=16, stock_day_batch_size=2)
    config["training"]["resident_cache"].update(build_batch_rows=256, evaluation_days=1)
    if not resident:
        config["training"]["batching"] = "monthly_shuffled_stock_days"
        config["data"]["input_precision"] = "float32"
    return config


def inputs(config, count=3):
    generator = torch.Generator().manual_seed(2001)
    windows = {f: torch.randn(count, 112, length, generator=generator) for f,length in config["data"]["window_bars"].items()}
    for value in windows.values(): value[:, 72:111] = 1
    return windows, F.one_hot(torch.full((count,), 7), 8).float()


def activate_normalizers(model):
    with torch.no_grad():
        for encoder in model.encoders.values():
            encoder.project.normalizer.affine.weight.normal_(0, .2)


class Block:
    def __init__(self, month, seed):
        rng = np.random.default_rng(seed)
        self.keys = pd.DataFrame([(pd.Timestamp(month + "-02") + pd.Timedelta(minutes=m), f"S{s}")
            for m in (570, 600, 630, 660, 690, 810, 840, 870) for s in range(6)], columns=["date", "instrument"])
        self.keys["vwap_return"] = rng.normal(0, .001, len(self.keys))
        self.keys.loc[:5, "vwap_return"] = np.nan
        if seed == 1:
            extra = self.keys.copy(); extra["date"] += pd.Timedelta(days=20)
            self.keys = pd.concat([self.keys, extra], ignore_index=True)
        self.windows = {f:rng.normal(size=(6,112,length)).astype(np.float32) for f,length in CONFIG["data"]["window_bars"].items()}
        for value in self.windows.values(): value[:,72:111] = 1

    def batch(self, rows):
        keys = self.keys.iloc[rows]
        clocks = prediction_clock_values(keys.date.to_numpy(dtype="datetime64[ns]").astype(np.int64))
        stocks = keys.instrument.str[1:].astype(int).to_numpy()
        values = {f:x[stocks].copy() for f,x in self.windows.items()}
        for frequency,value in values.items():
            bars = 240 // int(frequency[3:-1])
            previous = CONFIG["data"]["trading_day_windows"][frequency]["previous_days"]
            value *= np.arange(value.shape[-1])[None,None,:] < (previous*bars + (bars//8)*clocks.argmax(1))[:,None,None]
        return values, clocks


class Store:
    protocol = dict(PROTOCOL, min_cross_section_size=3)
    def __init__(self):
        self.data = {split:{month:Block(month,seed)} for split,month,seed in
                     (("train","2020-01",1),("validation","2024-01",3),("test","2024-07",4))}
    def months(self, split): return list(self.data[split])
    def load(self, split, month): return self.data[split][month]


class CoordinateModelTests(unittest.TestCase):
    def test_source_contract_targets_and_default_size(self):
        trainer.validate_training(CONFIG, PROTOCOL)
        old = json.loads((ROOT / "solutions/v1601/model-config.json").read_text(encoding="utf-8"))
        self.assertEqual(base_configuration(CONFIG)["data"], old["data"])
        for name in ("prepare_training_targets", "standardized_targets", "section_indices"):
            self.assertEqual(inspect.getsource(getattr(trainer,name)), inspect.getsource(getattr(parent,name)))
        self.assertEqual(sum(p.numel() for p in MultiFrequencyGRU(CONFIG).parameters()), 2584290)
        self.assertEqual(CONFIG["training"]["resident_cache"]["memory_reserve_gib"], 2.0)
        for name in ("v1601_data.py", "v1601_schema.py"):
            self.assertEqual((ROOT/"solutions/v2001"/name).read_bytes(), (ROOT/"solutions/v1601"/name).read_bytes())

    def test_learned_group_transform_preserves_order_and_only_uses_prefix(self):
        spec = small_config()["model"]
        torch.manual_seed(42)
        layer = CausalGroupNormalization(spec, 8).eval()
        with torch.no_grad(): layer.affine.weight.normal_(0,.2)
        x = torch.randn(2,16,112)
        x[...,72:111] = 1
        for group in layer.groups:
            x[...,list(group)] = torch.linspace(-2,2,len(group))
        output = layer(x)
        center,scale = layer.coefficients(x)
        self.assertTrue((scale >= spec["normalization_minimum_scale"]).all())
        for group in layer.groups:
            self.assertTrue((torch.diff(output[...,list(group)],dim=-1)>0).all())
        changed = x.clone(); changed[:,13:,:72] += 100
        torch.testing.assert_close(output[:,:13], layer(changed)[:,:13], atol=0, rtol=0)
        changed = x.clone(); changed[...,75+16] = 0
        self.assertTrue((layer(changed)[...,52] == 0).all())

    def test_nonzero_normalization_all_clock_causality_and_shared_reference(self):
        config = small_config()
        torch.manual_seed(42)
        model = MultiFrequencyGRU(config).eval()
        activate_normalizers(model)
        windows,latest = inputs(config)
        windows = {f:value.requires_grad_(True) for f,value in windows.items()}
        grouped = model.forward_group(windows,latest)
        for moment in range(8):
            clock = F.one_hot(torch.full((3,),moment),8).float()
            with torch.no_grad(): reference = model.forward_heads(windows,clock)
            torch.testing.assert_close(grouped[:,moment],reference,atol=3e-7,rtol=2e-5)
            gradients = torch.autograd.grad(grouped[:,moment].sum(),tuple(windows.values()),retain_graph=True)
            for frequency,gradient in zip(windows,gradients):
                bars = 240//int(frequency[3:-1])
                cutoff = config["data"]["trading_day_windows"][frequency]["previous_days"]*bars + moment*(bars//8)
                self.assertEqual(gradient[:,:,cutoff:].count_nonzero().item(),0)
                self.assertGreater(gradient[:,:,:cutoff].abs().sum().item(),0)

    def test_decoder_preserves_zero_availability_overflow_and_future_mask(self):
        parameters = {f:{"price_anchor":{"center":2.,"scale":1.}} for f in CONFIG["data"]["window_bars"]}
        raw,repairs = {},{}
        for f,length in CONFIG["data"]["window_bars"].items():
            codes = torch.full((1,length,72),float("nan"),dtype=torch.float16)
            anchors = torch.full((1,length),float("nan"))
            codes[0,0,52] = 0
            codes[0,0,0] = float("inf")
            codes[0,-1,52] = 1
            anchors[0,-1] = 3
            raw[f] = codes,anchors
            repairs[f] = torch.tensor([0]),torch.tensor([100000.])
        clock = F.one_hot(torch.zeros(1,dtype=torch.long),8).float()
        result = prepare_coordinate_windows(raw,repairs,clock,parameters)
        for f,value in result.items():
            self.assertEqual(value.shape[1],112)
            self.assertEqual(value[0,0,0],100000.)
            self.assertEqual(value[0,52,0],0)
            self.assertEqual(value[0,91,0],1)
            self.assertEqual(value[0,74,0],1)
            self.assertEqual(value[0,:,-1].count_nonzero().item(),0)

    def test_actual_objective_and_all_parameter_gradients(self):
        config = small_config()
        model = MultiFrequencyGRU(config)
        windows,clock = inputs(config)
        target = torch.randn(3,8,2,dtype=torch.float64)
        target[0,0] = float("nan")
        result = trainer.training_step(model,torch.optim.AdamW(model.parameters()),windows,clock,target,config,False)
        self.assertEqual(result["rows"],23)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(model.encoders["bar1m"].project.normalizer.affine.weight.grad.abs().sum().item(),0)

    def test_exact_group_resume_keeps_raw_ema_optimizer_and_rng(self):
        config,store = small_config(),Store()
        configure,step = trainer.configure,trainer.training_step
        with tempfile.TemporaryDirectory() as tmp:
            runs = [Path(tmp)/name for name in ("continuous","resumed")]
            manifests = [{"status":"running"},{"status":"running"}]
            for run in runs:
                run.mkdir(); trainer.write_json(run/"data-manifest.json",{"synthetic":True})
                trainer.write_json(run/"baseline-preservation.json",{})
            def cpu(spec): configure(spec); return torch.device("cpu")
            with patch.object(trainer,"configure",cpu),patch.object(trainer,"environment",lambda:{"packages":{}}), \
                    patch.object(trainer,"draw_history"),patch.object(trainer,"log_event"):
                trainer.train(store,runs[0],config,manifests[0])
                calls = 0
                def interrupt(*args,**kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 3: raise KeyboardInterrupt
                    return step(*args,**kwargs)
                with patch.object(trainer,"training_step",interrupt),self.assertRaises(KeyboardInterrupt):
                    trainer.train(store,runs[1],config,manifests[1])
                trainer.train(store,runs[1],config,manifests[1],resume=True)
            for name in ("best.pt","last.pt"):
                left,right = [torch.load(run/name,weights_only=True) for run in runs]
                for key,value in left["state_dict"].items():
                    torch.testing.assert_close(value,right["state_dict"][key],atol=0,rtol=0)
            for key,value in left["ema_state_dict"].items():
                torch.testing.assert_close(value,right["ema_state_dict"][key],atol=0,rtol=0)
            for parameter,values in left["optimizer"]["state"].items():
                for key,value in values.items():
                    torch.testing.assert_close(value,right["optimizer"]["state"][parameter][key],atol=0,rtol=0)
            torch.testing.assert_close(left["torch_rng"],right["torch_rng"],atol=0,rtol=0)
            frames = [pd.read_csv(run/"history.csv") for run in runs]
            columns = ["train_loss","validation_loss","ic_mean","rank_ic_mean","train_huber_loss","train_rank_huber_loss"]
            pd.testing.assert_frame_equal(frames[0][columns],frames[1][columns],check_exact=True)


if __name__ == "__main__":
    unittest.main()
