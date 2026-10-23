"""v2601 新计算图、股票日标签和恢复语义的少量 CPU 合成检查。"""

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
for version in ("v501", "v2601"):
    sys.path.insert(0, str(ROOT / "solutions" / version))
import v2601_train as trainer
import v501_train as parent
from v2601_model import MultiFrequencyGRU, StructuredFieldProjection
from v2601_grouped import StockDayIndex, training_view, record_group_plan
from v2601_data import prediction_clock_values
from v2601_loss import standardized_targets
from v2601_targets import prepare_rank_training_targets
from v2601_temporal import attach_temporal_weights
from v2601_objective import balanced_assignment, combined_objective

CONFIG = json.loads((ROOT / "solutions/v2601/model-config.json").read_text(encoding="utf-8"))
CONFIGS = {"v2601": CONFIG}
TEST_VERSIONS = ("v2601",)
PARENTS = {"v2601": "v1601"}
PROTOCOL = json.loads((ROOT / "evaluation/protocols/local-v2-ic.json").read_text(encoding="utf-8"))
torch.set_num_threads(1)


def small_config(version="v2601", resident=False):
    config = copy.deepcopy(CONFIG)
    config["model"].update(hidden_dim=8, fusion_hidden_dim=16, expert_hidden_dim=8, router_hidden_dim=8, book_level_dim=4, book_mixer_hidden_dim=4, non_book_dim=4)
    config["training"].update(epochs=2, cpu_threads=1, batch_size=16, stock_day_batch_size=2,
                              checkpoint_batches=1)
    config["inference"].update(batch_size=16, stock_day_batch_size=2)
    config["training"]["resident_cache"].update(memory_reserve_gib=2.0, build_batch_rows=256, evaluation_days=1)
    config["model"]["market_conditioning"].update(context_dim=8,context_tokens=3,attention_heads=2,raw_hidden_dim=16)
    config["training"]["market_conditioning"].update(context_stock_day_batch_size=2,preflight_context_stocks=6)
    if not resident:
        config["training"]["batching"] = "monthly_shuffled_stock_days"
        config["data"]["input_precision"] = "float32"
    return config


def raw_inputs(config, count=3):
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
        self.days=np.unique(np.concatenate([b.keys.date.to_numpy(dtype='datetime64[ns]').astype('datetime64[D]') for split in self.data.values() for b in split.values()]))

    def months(self, split):
        return list(self.data[split])

    def load(self, split, month):
        return self.data[split][month]



def activate_market(model):
    with torch.no_grad():
        model.market_branch.conditioner[-1].weight.normal_(0,.1)
        model.market_branch.conditioner[-1].bias.normal_(0,.1)
        model.market_branch.readout[-1].weight.normal_(0,.1)
        model.market_branch.readout[-1].bias.normal_(0,.1)


def inputs(config,count=2,context_count=4,grad=False):
    from v2601_market import extract_snapshots,MARKET_KEYS
    raw,_=raw_inputs(config,context_count)
    if grad:raw={f:x.requires_grad_(True) for f,x in raw.items()}
    snapshots=extract_snapshots(raw,config['model']['market_conditioning']['snapshot_bars'],config['data']['window_bars'])
    membership=torch.ones(context_count,8,dtype=torch.bool)
    values={f:x[:count] for f,x in raw.items()}
    values.update({MARKET_KEYS[0]:snapshots,MARKET_KEYS[1]:membership,MARKET_KEYS[2]:torch.arange(count)})
    return values,F.one_hot(torch.full((count,),7),8).float(),raw


class V2601Tests(unittest.TestCase):
    def test_text_parameter_shards_match_notebook_and_tensor_state(self):
        from v2601_text_artifact import encode_state,decode_state,read_bundle,FORMAT
        import v2601_notebook as notebook
        model=MultiFrequencyGRU(small_config()).eval();activate_market(model)
        encoded=encode_state(model.state_dict());names=list(encoded)
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            shards=[]
            for part,selected in enumerate([names[::2],names[1::2]]):
                name=f'weights-{part}.json';shards.append(name)
                (directory/name).write_text(json.dumps({'state_dict':{k:encoded[k] for k in selected}}),encoding='utf-8')
            index=directory/'v2601-model.json'
            index.write_text(json.dumps({'format':FORMAT,'state_dict_shards':shards}),encoding='utf-8')
            bundle=read_bundle(index)
            with patch.object(notebook,'_bundle_directory',lambda:directory):self.assertEqual(notebook._read_bundle(),bundle)
            restored=decode_state(bundle)
            for key,value in model.state_dict().items():torch.testing.assert_close(value,restored[key],rtol=0,atol=0)
            index.write_text(json.dumps({'format':FORMAT,'state_dict_shards':[shards[0],shards[0]]}),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'重复'):read_bundle(index)

    def test_defaults_input_identity_and_full_context_preflight(self):
        original=json.loads((ROOT/'solutions/v1601/model-config.json').read_text(encoding='utf-8'))
        self.assertEqual(CONFIG['data'],original['data'])
        trainer.validate_training(CONFIG,PROTOCOL)
        self.assertEqual(sum(p.numel() for p in MultiFrequencyGRU(CONFIG).parameters()),3372499)
        self.assertEqual(CONFIG['training']['resident_cache']['memory_reserve_gib'],2.)
        windows,clock=trainer.preflight_inputs(CONFIG,2,torch.device('cpu'))
        self.assertEqual(windows['market_snapshots'].shape,(1000,8,1350))
        self.assertEqual(windows['market_membership'].dtype,torch.bool)
        self.assertEqual(inspect.getsource(trainer.prepare_training_targets),inspect.getsource(parent.prepare_training_targets))

    def test_zero_correction_preserves_initial_parent(self):
        sys.path.insert(0,str(ROOT/'solutions/v1601'))
        from v1601_model import MultiFrequencyGRU as ParentModel
        config=small_config();base=copy.deepcopy(config);base['solution_id']='v1601'
        torch.manual_seed(42);previous=ParentModel(base).eval()
        torch.manual_seed(42);current=MultiFrequencyGRU(config).eval()
        for key,value in previous.state_dict().items():torch.testing.assert_close(value,current.state_dict()[key],rtol=0,atol=0)
        windows,clock,_=inputs(config)
        plain={f:windows[f] for f in config['data']['window_bars']}
        with torch.no_grad():torch.testing.assert_close(current.forward_group(windows,clock),previous.forward_group(plain,clock),rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'完整'):current.forward_group(plain,clock)

    def test_raw_snapshot_positions_have_manual_reference(self):
        from v2601_market import extract_snapshots
        config=small_config();windows={f:torch.arange(75*n,dtype=torch.float32).reshape(1,75,n) for f,n in config['data']['window_bars'].items()}
        actual=extract_snapshots(windows,config['model']['market_conditioning']['snapshot_bars'],config['data']['window_bars'])
        for q in range(8):
            values=[]
            for f,n in config['model']['market_conditioning']['snapshot_bars'].items():
                bars=240//int(f[3:-1]);end=config['data']['trading_day_windows'][f]['previous_days']*bars+q*(bars//8)
                for position in range(end-n,end):values.extend(windows[f][0,:,position].tolist())
            torch.testing.assert_close(actual[0,q],torch.tensor(values),rtol=0,atol=0)

    def test_market_and_main_paths_are_causal_and_reference_equivalent(self):
        config=small_config();model=MultiFrequencyGRU(config).eval();activate_market(model)
        windows,latest,raw=inputs(config,grad=True)
        windows['market_membership'][3,:2]=False
        prediction=model.forward_group(windows,latest)
        for q in range(8):
            clock=F.one_hot(torch.full((2,),q),8).float()
            with torch.no_grad():torch.testing.assert_close(prediction[:,q],model.forward_heads(windows,clock),rtol=1e-5,atol=5e-7)
            gradients=torch.autograd.grad(prediction[:,q].sum(),tuple(raw.values()),retain_graph=True)
            for f,g in zip(raw,gradients):
                bars=240//int(f[3:-1]);end=config['data']['trading_day_windows'][f]['previous_days']*bars+q*(bars//8)
                self.assertEqual(g[:,:,end:].count_nonzero().item(),0)
                self.assertGreater(g[2,:,:end].abs().sum().item(),0)
                if q<2:self.assertEqual(g[3].count_nonzero().item(),0)
        partial=F.one_hot(torch.tensor([0,3]),8).float()
        with torch.no_grad():
            group=model.forward_group(windows,partial)
            torch.testing.assert_close(group[torch.arange(2),partial.argmax(-1)],model.forward_heads(windows,partial),rtol=1e-5,atol=5e-7)

    def test_stock_permutation_target_batching_and_eval_context_cache(self):
        from v2601_market import MARKET_KEYS
        config=small_config();model=MultiFrequencyGRU(config).eval();activate_market(model)
        windows,clock,_=inputs(config)
        with torch.no_grad():
            expected=model.forward_group(windows,clock)
            perm=torch.tensor([2,0,3,1]);inverse=torch.argsort(perm)
            shuffled=dict(windows)
            shuffled[MARKET_KEYS[0]]=windows[MARKET_KEYS[0]][perm];shuffled[MARKET_KEYS[1]]=windows[MARKET_KEYS[1]][perm];shuffled[MARKET_KEYS[2]]=inverse[windows[MARKET_KEYS[2]]]
            torch.testing.assert_close(model.forward_group(shuffled,clock),expected,rtol=1e-5,atol=5e-7)
            state=model.prepare_market(windows[MARKET_KEYS[0]],windows[MARKET_KEYS[1]])
            plain={f:windows[f] for f in config['data']['window_bars']}
            cached=model.forward_group(plain,clock,state,windows[MARKET_KEYS[2]])
            torch.testing.assert_close(cached,expected,rtol=0,atol=0)
            pieces=[model.forward_group({f:x[i:i+1] for f,x in plain.items()},clock[i:i+1],state,torch.tensor([i])) for i in range(2)]
            torch.testing.assert_close(torch.cat(pieces),expected,rtol=1e-5,atol=5e-7)
        model.train()
        with self.assertRaisesRegex(ValueError,'禁止'):model.forward_group(plain,clock,state,torch.arange(2))

    def test_training_recomputes_context_and_bfloat16_gradients_are_finite(self):
        config=small_config();model=MultiFrequencyGRU(config);activate_market(model)
        windows,clock,_=inputs(config);target=torch.randn(2,8,2,dtype=torch.float64)
        optimizer=torch.optim.AdamW(model.parameters())
        with patch.object(model.market_branch,'prepare_context',wraps=model.market_branch.prepare_context) as call:
            for _ in range(2):
                result=trainer.training_step(model,optimizer,windows,clock,target,config,True,torch.ones(2))
                self.assertTrue(math.isfinite(result['loss']))
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertEqual(call.call_count,2)
        with self.assertRaisesRegex(ValueError,'同一日期'):
            trainer.training_step(model,optimizer,windows,clock,target,config,False,torch.tensor([1.,2.]))

    def test_text_roundtrip_and_all_masked_market(self):
        from v2601_text_artifact import encode_state,decode_state,FORMAT
        config=small_config();model=MultiFrequencyGRU(config).eval();activate_market(model)
        other=MultiFrequencyGRU(config).eval();other.load_state_dict(decode_state({'format':FORMAT,'state_dict':encode_state(model.state_dict())}))
        windows,clock,_=inputs(config)
        with torch.no_grad():torch.testing.assert_close(model.forward_group(windows,clock),other.forward_group(windows,clock),rtol=0,atol=0)
        windows['market_membership'].zero_()
        with torch.no_grad():
            result=model.forward_group(windows,clock)
            plain={f:windows[f] for f in config['data']['window_bars']}
            torch.testing.assert_close(result,model.individual_group_components(plain,clock)[0],rtol=0,atol=0)

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
