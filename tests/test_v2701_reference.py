"""遗漏字段的有效零、独立缺失、坐标与真实梯度路径。"""
import unittest

import torch
import numpy as np
import pandas as pd

from test_v2701 import small_config, inputs, activate_price_memory
from v2701_model import MultiFrequencyGRU
from v2701_reference import reference_features
from trade_reference_inputs import reference_view, prepare_trade_reference_windows
from price_memory_inputs import prepare_price_memory_windows


class ReferenceTests(unittest.TestCase):
    def test_zero_count_and_independent_missing_fields_preserve_parent(self):
        raw = pd.DataFrame({'deal_number':[0, -1, 3, np.inf],
                            'pre_close':[10, 0, np.nan, 12], 'adjust_factor':[1, 2, 0, np.inf]})
        values = reference_view(raw)
        self.assertEqual(values[0, 0], 0.)
        self.assertEqual(values[0, 2], 0.)
        self.assertTrue(np.isnan(values[1, 0]) and np.isnan(values[1, 1]))
        self.assertAlmostEqual(float(values[2, 0]), np.log(4), places=6)
        config = small_config()
        raw_tensors, repairs = {}, {}
        for f, length in config['data']['window_bars'].items():
            parent = torch.full((1, length, 72), .25)
            parent[:, 0] = torch.nan
            anchor = torch.full((1, length), 2.)
            extra = torch.from_numpy(np.resize(values, (1, length, 3)).copy())
            raw_tensors[f] = parent, anchor, extra
            repairs[f] = torch.empty(0, dtype=torch.long), torch.empty(0)
        clock = torch.eye(8)[[0]]
        result = prepare_trade_reference_windows(raw_tensors, repairs, clock)
        previous = prepare_price_memory_windows({f:v[:2] for f,v in raw_tensors.items()}, repairs, clock)
        for f, value in result.items():
            torch.testing.assert_close(value[:, :77], previous[f], rtol=0, atol=0)
            self.assertEqual(value[0, 77, 0], 0.)
            self.assertEqual(value[0, 80, 0], 1.)
            self.assertEqual(value[0, 74, 0], 0.)
            self.assertEqual(value[0, 80, 1], 0.)
            self.assertEqual(value[0, 82, 1], 1.)
            bars = 240 // int(f[3:-1])
            self.assertEqual(value[:, 77:, -bars:].count_nonzero().item(), 0)

    def test_provided_day_reference_and_missing_pair(self):
        windows, _ = inputs(small_config(), 1)
        value = windows['bar1m']
        value[:, 75] = 2.25
        value[:, 78] = 2.
        view = reference_features(value)
        torch.testing.assert_close(view[..., 81], torch.full_like(view[..., 81], 25.))
        self.assertTrue((view[..., 82] == 1).all())
        value[:, 81, :3] = 0
        view = reference_features(value)
        self.assertEqual(view[:, :3, 81:].count_nonzero().item(), 0)

    def test_new_field_gradients_and_bfloat16_training_are_finite(self):
        config = small_config()
        model = MultiFrequencyGRU(config).eval()
        activate_price_memory(model)
        windows, clock = inputs(config, 2)
        windows = {f:x.requires_grad_(True) for f,x in windows.items()}
        prediction = model.forward_group(windows, clock)
        gradients = torch.autograd.grad(prediction[:, 3].sum(), tuple(windows.values()))
        for f, grad in zip(windows, gradients):
            bars = 240 // int(f[3:-1])
            cutoff = config['data']['trading_day_windows'][f]['previous_days'] * bars + 3 * (bars // 8)
            for channel in (77, 78, 79):
                self.assertGreater(grad[:, channel, :cutoff].abs().sum().item(), 0)
            self.assertEqual(grad[:, :, cutoff:].count_nonzero().item(), 0)
        model.train()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss = model.forward_group(windows, clock).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
