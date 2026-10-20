"""EMA 不改变原始优化路径，并完整参与验证选模和断点恢复。"""

import copy
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import pandas as pd

from test_continuous_candidates import CONFIGS, MODULES, base, create_model, small_config

WeightEMA = importlib.import_module("v501_ema").WeightEMA


class EMATests(unittest.TestCase):
    def test_shadow_weights_warmup_averaging_and_rng(self):
        model = torch.nn.Linear(2, 1)
        rng_before = torch.get_rng_state().clone()
        ema = WeightEMA(model, decay=0.5, warmup_epochs=1)
        torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
        self.assertFalse(ema.model.training)
        self.assertTrue(all(not p.requires_grad for p in ema.model.parameters()))
        with torch.no_grad():
            model.weight.fill_(2)
            model.bias.fill_(4)
        ema.update(epoch=1)
        torch.testing.assert_close(ema.model.weight, model.weight, rtol=0, atol=0)
        with torch.no_grad():
            model.weight.fill_(6)
            model.bias.fill_(8)
        ema.update(epoch=2)
        torch.testing.assert_close(ema.model.weight, torch.full_like(model.weight, 4), rtol=0, atol=0)
        torch.testing.assert_close(ema.model.bias, torch.full_like(model.bias, 6), rtol=0, atol=0)
        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 6), rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
        self.assertEqual(ema.updates, 2)
        with self.assertRaises(ValueError):
            ema.restore(ema.model.state_dict(), 1, optimizer_steps=2)
        for decay in (0, 1, float("nan"), -1):
            with self.assertRaises(ValueError):
                WeightEMA(model, decay=decay, warmup_epochs=1)

    def test_raw_optimization_matches_parent_and_checkpoint_uses_ema(self):
        for key in ("data", "model", "inference", "seed"):
            self.assertEqual(CONFIGS["v501"][key], CONFIGS["v303"][key])
        training = copy.deepcopy(CONFIGS["v501"]["training"])
        training.pop("weight_averaging")
        self.assertEqual(training, CONFIGS["v303"]["training"])
        with tempfile.TemporaryDirectory() as tmp:
            runs = {}
            for version in ("v303", "v501"):
                trainer, config = MODULES[version]["train"], small_config(version)
                config["training"]["epochs"] = 1
                if version == "v501":
                    config["training"]["weight_averaging"].update(decay=0.5, warmup_epochs=0)
                run = Path(tmp) / version
                run.mkdir()
                runs[version] = run
                trainer.write_json(run / "data-manifest.json", {"synthetic": True})
                trainer.write_json(run / "baseline-preservation.json", {})
                configure = trainer.configure

                def cpu_configure(spec):
                    configure(spec)
                    return torch.device("cpu")

                with patch.object(trainer, "configure", cpu_configure), \
                        patch.object(trainer, "environment", lambda: {"packages": {}}), \
                        patch.object(trainer, "draw_history"), patch.object(trainer, "log_event"):
                    trainer.train(base.SyntheticStore(version), run, config, {"status": "running"})
            raw, averaged = [torch.load(runs[v] / "last.pt", weights_only=True) for v in ("v303", "v501")]
            for key in raw["state_dict"]:
                torch.testing.assert_close(raw["state_dict"][key], averaged["state_dict"][key], rtol=0, atol=0)
            torch.testing.assert_close(raw["torch_rng"], averaged["torch_rng"], rtol=0, atol=0)
            for parameter, values in raw["optimizer"]["state"].items():
                for key, value in values.items():
                    torch.testing.assert_close(value, averaged["optimizer"]["state"][parameter][key], rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(averaged["state_dict"][key], averaged["ema_state_dict"][key])
                                for key in averaged["state_dict"]))
            selected = torch.load(runs["v501"] / "best.pt", weights_only=True)
            self.assertEqual(selected["weight_source"], "ema")
            self.assertEqual(selected["ema_updates"], averaged["progress"]["optimizer_steps"])
            for key in selected["state_dict"]:
                torch.testing.assert_close(selected["state_dict"][key], averaged["ema_state_dict"][key], rtol=0, atol=0)
            self.assertEqual(averaged["ema_updates"], averaged["progress"]["optimizer_steps"])
            selected_model = create_model("v501", config)
            selected_model.load_state_dict(selected["state_dict"])
            predictions, _, _ = MODULES["v501"]["train"].predict_split(
                base.SyntheticStore("v501"), "validation", selected_model, torch.device("cpu"), config)
            saved_predictions = pd.read_parquet(runs["v501"] / "evaluation/validation/predictions.parquet")
            # 与训练中的 EMA 评估使用相同参数梯度标志，严格比较独立加载后的结果。
            selected_model.requires_grad_(False)
            same_execution, _, _ = MODULES["v501"]["train"].predict_split(
                base.SyntheticStore("v501"), "validation", selected_model, torch.device("cpu"), config)
            pd.testing.assert_frame_equal(same_execution, saved_predictions, check_exact=True)
            # 常规模型构造保留 requires_grad=True；沿用仓库已有 FP32 推理容差。
            pd.testing.assert_frame_equal(predictions, saved_predictions, check_exact=False, atol=2e-7, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
