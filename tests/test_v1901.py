"""条件潜在收益结构的标签隔离、完整市场计算、目标和恢复检查。"""
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
for name in ("common", "evaluation", "solutions/v1901", "solutions/v1601"):
    sys.path.insert(0, str(ROOT / name))
import v1901_train as trainer
import v1601_train as parent
from v1901_model import MultiFrequencyGRU
from v1901_backbone import StockEncoder
from v1901_initialization import create_optimizer, initialize_from_parent, state_digest
from v1901_objective import full_day_objective
from v1901_grouped import training_view
from v1901_calendar import CalendarDayPlan
from v1901_temporal import attach_temporal_weights
from test_v1301 import Block, Store

CONFIG = json.loads((ROOT / "solutions/v1901/model-config.json").read_text(encoding="utf-8"))
PROTOCOL = json.loads((ROOT / "evaluation/protocols/local-v2-ic.json").read_text(encoding="utf-8"))
CONFIGS, TEST_VERSIONS, PARENTS = {"v1901": CONFIG}, ("v1901",), {"v1901": "v1601"}
torch.set_num_threads(1)


def small_config(version="v1901", resident=True):
    config = copy.deepcopy(CONFIG)
    config["model"].update(hidden_dim=8, fusion_hidden_dim=16, expert_hidden_dim=8, router_hidden_dim=8,
        book_level_dim=4, book_mixer_hidden_dim=4, non_book_dim=4, encoding_stock_batch_size=2,
        latent_factor_count=3, latent_hidden_dim=8)
    config["training"].update(epochs=2, cpu_threads=1, stock_day_batch_size=8, batch_size=64,
                               microbatch_stock_days=2, checkpoint_batches=1)
    config["inference"].update(stock_day_batch_size=8, batch_size=64)
    config["training"]["resident_cache"].update(build_batch_rows=256, evaluation_days=1)
    return config


def inputs(config, count=4):
    generator = torch.Generator().manual_seed(1901)
    windows = {f: torch.randn(count, 75, length, generator=generator)
               for f, length in config["data"]["window_bars"].items()}
    for value in windows.values():
        value[:, 72:] = 1
    return windows, F.one_hot(torch.full((count,), 7), 8).float()


def activate_prior(model):
    with torch.no_grad():
        model.prior[-1].weight.normal_(0, .15)
        model.prior[-1].bias[0] = .2


def synthetic_initialization(config, path):
    torch.manual_seed(17)
    state = StockEncoder(config).state_dict()
    config["initialization"]["state_sha256"] = state_digest(state)
    torch.save({"state_dict": state, "parent_config": copy.deepcopy(config),
                "parent_run_id": config["parent_run_id"],
                "parent_model_sha256": config["initialization"]["model_sha256"]}, path)
    return path


class ConditionalFactorTests(unittest.TestCase):
    def test_parent_inputs_cache_identity_and_explicit_two_gib(self):
        trainer.validate_training(CONFIG, PROTOCOL)
        old = json.loads((ROOT / "solutions/v1601/model-config.json").read_text(encoding="utf-8"))
        self.assertEqual(CONFIG["data"], old["data"])
        for name in ("prepare_training_targets", "standardized_targets", "section_indices", "data_manifest"):
            self.assertEqual(inspect.getsource(getattr(trainer, name)), inspect.getsource(getattr(parent, name)))
        self.assertEqual(inspect.getsource(trainer.ProcessedStore.load_range), inspect.getsource(parent.ProcessedStore.load_range))
        self.assertEqual(CONFIG["training"]["resident_cache"]["memory_reserve_gib"], 2.0)

    def test_initialization_retains_parent_and_is_bound_to_tensor_identity(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = synthetic_initialization(config, root / "parent.pt")
            model = MultiFrequencyGRU(config).eval()
            initialize_from_parent(model, config, root, root, source)
            windows, clock = inputs(config)
            with torch.no_grad():
                got = model.forward_group(windows, clock)
                expected = torch.cat([model.encoder.forward_group({f: x[i:i+2] for f,x in windows.items()}, clock[i:i+2])
                                      for i in range(0, len(clock), 2)])
            torch.testing.assert_close(got, expected, atol=0, rtol=0)
            saved = torch.load(source, weights_only=True)
            saved["state_dict"][next(iter(saved["state_dict"]))].add_(1)
            torch.save(saved, source)
            with self.assertRaisesRegex(ValueError, "指纹"):
                initialize_from_parent(model, config, root, root, source)

    def test_nonzero_prior_shared_reference_and_all_clock_future_gradients(self):
        config = small_config()
        torch.manual_seed(42)
        model = MultiFrequencyGRU(config).eval()
        activate_prior(model)
        windows, clock = inputs(config)
        windows = {f: value.requires_grad_(True) for f, value in windows.items()}
        output = model.forward_group(windows, clock)
        for moment in range(8):
            individual_clock = F.one_hot(torch.full((len(clock),), moment), 8).float()
            with torch.no_grad():
                expected = model.forward_heads(windows, individual_clock)
            torch.testing.assert_close(output[:, moment], expected, atol=3e-7, rtol=2e-5)
            gradients = torch.autograd.grad(output[:, moment].sum(), tuple(windows.values()), retain_graph=True)
            for frequency, grad in zip(windows, gradients):
                bars = 240 // int(frequency[3:-1])
                cutoff = config["data"]["trading_day_windows"][frequency]["previous_days"] * bars + moment * (bars // 8)
                self.assertEqual(grad[:, :, cutoff:].count_nonzero().item(), 0)
                self.assertGreater(grad[:, :, :cutoff].abs().sum().item(), 0)

    def test_member_masks_permutation_and_encoding_chunks(self):
        config = small_config()
        torch.manual_seed(43)
        model = MultiFrequencyGRU(config).eval()
        activate_prior(model)
        windows, clock = inputs(config)
        membership = torch.ones(4, 8, dtype=torch.bool)
        membership[3, :4] = False
        with torch.no_grad():
            original = model.forward_group(windows, clock, membership)
            changed = {f: x.clone() for f,x in windows.items()}
            for x in changed.values(): x[3, :72] += 100
            after = model.forward_group(changed, clock, membership)
            torch.testing.assert_close(original[:3, :4], after[:3, :4], atol=0, rtol=0)
            permutation = torch.tensor([2, 0, 3, 1])
            permuted = model.forward_group({f:x[permutation] for f,x in windows.items()}, clock[permutation], membership[permutation])
            torch.testing.assert_close(original[permutation], permuted, atol=3e-7, rtol=2e-5)
            model.encoding_stock_batch_size = 4
            full = model.forward_group(windows, clock, membership)
            torch.testing.assert_close(original, full, atol=3e-7, rtol=2e-5)

    def test_posterior_labels_do_not_change_prior_and_eval_never_calls_posterior(self):
        model = MultiFrequencyGRU(small_config())
        windows, clock = inputs(small_config())
        prior = model.decode_prior(model.encode_group(windows, clock))
        original = prior["prediction"].detach().clone()
        target = torch.randn(4, 8, 2, dtype=torch.float64)
        q1 = model.training_posterior(prior, target)
        q2 = model.training_posterior(prior, -target)
        torch.testing.assert_close(prior["prediction"], original, atol=0, rtol=0)
        self.assertGreater((q1["posterior_mean"] - q2["posterior_mean"]).abs().max().item(), 0)
        model.eval()
        with self.assertRaises(ValueError): model.training_posterior(prior, target)
        with patch.object(model, "training_posterior", side_effect=AssertionError("posterior called")), torch.no_grad():
            self.assertTrue(torch.isfinite(model.forward_group(windows, clock)).all())

    def test_masked_clocks_gradients_and_posterior_objective(self):
        config = small_config()
        model = MultiFrequencyGRU(config)
        windows, clock = inputs(config)
        target = torch.randn(4, 8, 2, dtype=torch.float64)
        target[0, 0] = float("nan")
        target[:, 2] = float("nan")
        result = trainer.training_step(model, create_optimizer(model, config["training"]), windows, clock, target, config, False)
        self.assertEqual(result["rows"], 27)
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))
        self.assertGreater(model.posterior[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.prior[-1].weight.grad.abs().sum().item(), 0)

    def test_analytic_posterior_expectation_and_kl(self):
        config = small_config()
        target = torch.tensor([-1., 1.], dtype=torch.float64)[:, None, None].expand(2, 8, 2)
        prior = {"prediction": torch.zeros_like(target, requires_grad=True), "individual": torch.zeros_like(target),
                 "loadings": torch.zeros(2, 8, 2, 3, dtype=torch.float64),
                 "prior_mean": torch.zeros(8, 3), "prior_std": torch.ones(8, 3),
                 "idiosyncratic_std": torch.ones(2, 8)}
        posterior = {"posterior_mean": torch.zeros(8, 3), "posterior_std": torch.ones(8, 3)}
        loss, detail = full_day_objective(prior, posterior, target, None, config["training"])
        self.assertAlmostEqual(float(detail["main_loss"]), .5)
        self.assertAlmostEqual(float(detail["rank_loss"]), .5)
        self.assertAlmostEqual(float(detail["prior_kl"]), 0)
        self.assertAlmostEqual(float(detail["posterior_nll"]), .5 * (np.log(2*np.pi) + 1))
        loss.backward()
        self.assertTrue(torch.isfinite(prior["prediction"].grad).all())

    def test_unlabelled_members_remain_context(self):
        config = small_config()
        block = Block("2020-01", 1)
        block.keys.loc[block.keys.instrument.eq("S5"), "vwap_return"] = np.nan
        primary, _ = trainer.prepare_training_targets(block.keys, Store.protocol)
        ranks, _ = trainer.prepare_rank_training_targets(block.keys, Store.protocol)
        view = training_view(block, primary, ranks)
        attach_temporal_weights(view, config, Store.protocol)
        plan = CalendarDayPlan(view, config, Store.protocol)
        self.assertGreater(int(view.eligible.sum()), int(view.supervised.sum()))
        self.assertTrue(all(len(groups) == 6 for groups in plan.groups))

    def test_interrupted_training_restores_prior_posterior_optimizer_and_rng(self):
        config, store = small_config(), Store()
        configure, step = trainer.configure, trainer.training_step
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = synthetic_initialization(config, root / "parent.pt")
            runs = [root / name for name in ("continuous", "resumed")]
            manifests = [{"status": "running"}, {"status": "running"}]
            for run in runs:
                run.mkdir()
                trainer.write_json(run / "data-manifest.json", {"synthetic": True})
                trainer.write_json(run / "baseline-preservation.json", {})
            def cpu(spec):
                configure(spec)
                return torch.device("cpu")
            with patch.object(trainer, "configure", cpu), patch.object(trainer, "environment", lambda:{"packages":{}}), \
                    patch.object(trainer, "draw_history"), patch.object(trainer, "log_event"):
                trainer.train(store, runs[0], config, manifests[0], initialization_state=source)
                calls = 0
                def interrupt(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 3: raise KeyboardInterrupt
                    return step(*args, **kwargs)
                with patch.object(trainer, "training_step", interrupt), self.assertRaises(KeyboardInterrupt):
                    trainer.train(store, runs[1], config, manifests[1], initialization_state=source)
                checkpoint = torch.load(runs[1] / "last.pt", weights_only=True)
                self.assertEqual(checkpoint["progress"]["day_cursor"], 2)
                trainer.train(store, runs[1], config, manifests[1], resume=True)
            for filename in ("best.pt", "last.pt"):
                left, right = [torch.load(run / filename, weights_only=True) for run in runs]
                for key,value in left["state_dict"].items():
                    torch.testing.assert_close(value, right["state_dict"][key], atol=0, rtol=0)
            for key,value in left["ema_state_dict"].items():
                torch.testing.assert_close(value, right["ema_state_dict"][key], atol=0, rtol=0)
            for parameter, values in left["optimizer"]["state"].items():
                for key,value in values.items():
                    torch.testing.assert_close(value, right["optimizer"]["state"][parameter][key], atol=0, rtol=0)
            torch.testing.assert_close(left["torch_rng"], right["torch_rng"], atol=0, rtol=0)
            self.assertEqual(left["ema_stock_days"], right["ema_stock_days"])
            frames = [pd.read_csv(run / "history.csv") for run in runs]
            columns = ["train_loss", "validation_loss", "ic_mean", "rank_ic_mean", "train_primary_mse_loss"] + ["train_"+key for key in trainer.CALENDAR_STATISTICS]
            pd.testing.assert_frame_equal(frames[0][columns], frames[1][columns], check_exact=True)

    def test_resume_before_first_checkpoint_keeps_parent_initialization(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            run.mkdir()
            source = synthetic_initialization(config, Path(temporary) / "parent.pt")
            with patch.object(trainer, "configure", return_value=torch.device("cpu")), \
                    patch.object(trainer, "structural_catalog", return_value={}), \
                    patch.object(trainer, "create_optimizer", side_effect=RuntimeError("stop_before_any_update")), \
                    self.assertRaisesRegex(RuntimeError, "stop_before_any_update"):
                trainer.train(Store(), run, config, {}, resume=True, initialization_state=source)
            record = trainer.read_json(run / "initialization.json")
            self.assertEqual(record["parent_state_sha256"], config["initialization"]["state_sha256"])
            self.assertFalse((run / "last.pt").exists())


if __name__ == "__main__":
    unittest.main()
