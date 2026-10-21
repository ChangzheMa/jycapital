"""混合监督与原分辨率因果卷积的关键语义；只使用 CPU 合成数据。"""

import copy
import importlib
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

from test_continuous_candidates import CONFIGS, MODULES, base, create_model
from resident_data import cache_key


class BlendTCNTests(unittest.TestCase):
    def test_blended_target_amplitudes_ties_scale_and_original_ic(self):
        loss = importlib.import_module("v402_loss")
        values = np.array([-1., 2., 2., 9.])
        raw = (values - values.mean()) / values.std()
        ranks = pd.Series(values).rank(method="average").to_numpy()
        ranked = (ranks - ranks.mean()) / ranks.std()
        mixed = 0.5 * raw + 0.5 * ranked
        expected = (mixed - mixed.mean()) / mixed.std()
        labels = torch.tensor([*values, float("nan"), float("inf")], dtype=torch.float64)
        target, reason = loss.blended_standardized_targets(labels, min_count=3)
        self.assertIsNone(reason)
        np.testing.assert_allclose(target[:4].numpy(), expected, rtol=0, atol=1e-15)
        self.assertTrue(torch.isnan(target[4:]).all())
        self.assertEqual(target[1].item(), target[2].item())
        self.assertFalse(np.allclose(target[:4].numpy(), ranked))
        self.assertFalse(np.allclose(target[:4].numpy(), raw))
        affine, _ = loss.blended_standardized_targets(labels[:4] * 100 + 700, min_count=3)
        torch.testing.assert_close(target[:4], affine, rtol=0, atol=1e-15)
        nonlinear, _ = loss.blended_standardized_targets(labels[:4].exp(), min_count=3)
        self.assertFalse(torch.allclose(target[:4], nonlinear))
        for bad, reason in ((torch.ones(4), "constant_labels"), (torch.ones(2), "insufficient_labels")):
            self.assertEqual(loss.blended_standardized_targets(bad, min_count=3)[1], reason)
        scores = target[:4].clone().requires_grad_(True)
        result = loss.blended_standardized_huber_loss(scores, labels[:4], min_count=3)
        self.assertEqual(result.loss.item(), 0)
        torch.testing.assert_close(result.ic, loss.pearson_ic_loss(scores, labels[:4], min_count=3).ic,
                                   rtol=0, atol=0)
        result.loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_new_candidates_preserve_data_loss_controls_and_cache(self):
        parent = CONFIGS["v303"]
        keys = []
        for version in ("v303", "v402", "v601"):
            config = CONFIGS[version]
            for key in ("data", "inference", "seed"):
                self.assertEqual(parent[key], config[key])
            self.assertEqual(config["training"]["huber_delta"], 2.0)
            store = SimpleNamespace(config=config, protocol=base.PROTOCOL)
            keys.append(cache_key(store, {"synthetic": True}, MODULES[version]["train"].prepare_training_targets))
        self.assertEqual(len(set(keys)), 1)
        self.assertEqual(CONFIGS["v402"]["model"], parent["model"])
        changed, baseline = copy.deepcopy(CONFIGS["v402"]["training"]), copy.deepcopy(parent["training"])
        for key in ("loss", "target_transform"):
            changed.pop(key)
            baseline.pop(key)
        self.assertEqual(changed, baseline)
        self.assertEqual(CONFIGS["v601"]["training"], parent["training"])
        self.assertEqual(sum(p.numel() for p in create_model("v402").parameters()), 433409)
        self.assertEqual(sum(p.numel() for p in create_model("v601").parameters()), 876257)

    def test_causal_convolution_prefix_and_invalid_dilations(self):
        module = MODULES["v601"]["model"]
        torch.manual_seed(19)
        block = module.CausalResidual(8, 3, 4, 0).eval()
        x = torch.randn(2, 8, 48, requires_grad=True)
        y = block(x)
        self.assertEqual(y.shape, x.shape)
        with torch.no_grad():
            future = x.detach().clone()
            future[:, :, 17:] = torch.randn_like(future[:, :, 17:]) * 100
            torch.testing.assert_close(block(future)[:, :, :17], y[:, :, :17], rtol=0, atol=0)
            torch.testing.assert_close(block(x.detach()[:, :, :17]), y[:, :, :17], rtol=2e-6, atol=2e-6)
        y[:, :, 16].sum().backward()
        self.assertEqual(x.grad[:, :, 17:].abs().sum().item(), 0)
        self.assertGreater(x.grad[:, :, :17].abs().sum().item(), 0)
        for value in ([], [0], [True], [1.5]):
            config = copy.deepcopy(CONFIGS["v601"])
            config["model"]["intraday_dilations"]["bar1m"] = value
            with self.assertRaises(ValueError):
                create_model("v601", config)


if __name__ == "__main__":
    unittest.main()
