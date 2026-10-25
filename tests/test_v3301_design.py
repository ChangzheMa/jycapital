"""只验证分时读出设计的参数隔离与父模型初始化，不做正式训练。"""

from pathlib import Path
import json
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'solutions/v3301'), str(ROOT / 'solutions/v3201')]
from v3301_readout import ClockReadoutBank
import v3201_train as parent


class ClockDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = json.loads((ROOT / 'solutions/v3201/model-config.json').read_text(encoding='utf-8'))

    def test_all_clock_initial_predictions_match_parent_and_parameters_are_private(self):
        torch.manual_seed(42)
        model = parent.MultiFrequencyGRU(self.config).eval()
        bank = ClockReadoutBank(model.experts, model.router).eval()
        pointers = [p.data_ptr() for p in bank.parameters()]
        self.assertEqual(len(pointers), len(set(pointers)))
        captured = []
        handle = model.shared.register_forward_hook(lambda module, args, value: captured.append(value.detach()))
        windows, clock = parent.preflight_inputs(self.config, 2, torch.device('cpu'))
        with torch.inference_mode():
            expected = model.forward_group(windows, clock)
            features = captured[0].reshape(8, 2, -1).transpose(0, 1)
            actual = bank.forward_group(features)[0]
        handle.remove()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-7)
        for q, head in enumerate(bank.heads):
            for name, value in model.experts.state_dict().items():
                torch.testing.assert_close(head.experts.state_dict()[name], value, rtol=0, atol=0)
            with torch.inference_mode():
                torch.testing.assert_close(bank.forward_clock(features[:, q], q)[0], actual[:, q], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            bank.forward_clock(features[:, 0], 8)

    def test_one_clock_gradient_and_update_do_not_change_other_clock_models(self):
        model = parent.MultiFrequencyGRU(self.config).eval()
        bank = ClockReadoutBank(model.experts, model.router).eval()
        before = {name: value.clone() for name, value in bank.state_dict().items()}
        features = torch.randn(7, self.config['model']['fusion_hidden_dim'])
        optimizer = torch.optim.AdamW(bank.heads[3].parameters(), lr=.001)
        prediction = bank.forward_clock(features, 3)[0]
        torch.nn.functional.smooth_l1_loss(prediction, torch.ones_like(prediction)).backward()
        for q, head in enumerate(bank.heads):
            if q == 3:
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
            else:
                self.assertTrue(all(p.grad is None for p in head.parameters()))
        optimizer.step()
        self.assertTrue(any(not torch.equal(value, before[name]) for name, value in bank.state_dict().items() if name.startswith('heads.3.')))
        for name, value in bank.state_dict().items():
            if not name.startswith('heads.3.'):
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
