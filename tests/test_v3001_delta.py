"""分块delta与独立递推的数学一致性，而非只检查两次同一实现。"""
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solutions/v3001'))
from v3001_delta import delta_scan, delta_recurrent_reference

torch.set_num_threads(1)


class DeltaTests(unittest.TestCase):
    def test_double_values_gradients_and_carried_state(self):
        torch.manual_seed(3001)
        q, k = [F.normalize(torch.randn(2, 3, 43, 5, dtype=torch.float64), dim=-1) for _ in range(2)]
        v = torch.randn(2, 3, 43, 7, dtype=torch.float64)
        log_decay = -.4 * torch.rand(2, 3, 43, dtype=torch.float64)
        beta = torch.rand(2, 3, 43, dtype=torch.float64)
        initial = torch.randn(2, 3, 5, 7, dtype=torch.float64)
        left = [x.detach().clone().requires_grad_() for x in (q, k, v, log_decay, beta, initial)]
        right = [x.detach().clone().requires_grad_() for x in left]
        out, state = delta_scan(*left[:5], chunk_size=8, initial_state=left[-1])
        reference, reference_state = delta_recurrent_reference(*right[:5], initial_state=right[-1])
        torch.testing.assert_close(out, reference, atol=1e-11, rtol=1e-10)
        torch.testing.assert_close(state, reference_state, atol=1e-11, rtol=1e-10)
        weight = torch.randn_like(out)
        grads = torch.autograd.grad((out * weight).sum() + state.square().sum(), left)
        refs = torch.autograd.grad((reference * weight).sum() + reference_state.square().sum(), right)
        for actual, expected in zip(grads, refs):
            torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
        first, carry = delta_scan(*[x[:, :, :17] for x in left[:5]], chunk_size=7, initial_state=left[-1])
        second, finish = delta_scan(*[x[:, :, 17:] for x in left[:5]], chunk_size=11, initial_state=carry)
        torch.testing.assert_close(torch.cat([first, second], 2), out, atol=1e-11, rtol=1e-10)
        torch.testing.assert_close(finish, state, atol=1e-11, rtol=1e-10)

    def test_known_overwrite_erasure_and_missing_skip(self):
        k = torch.tensor([[[[1., 0.], [0., 1.], [1., 0.], [1., 0.]]]])
        v = torch.tensor([[[[2.], [3.], [7.], [99.]]]])
        log_decay = torch.zeros(1, 1, 4)
        beta = torch.tensor([[[1., 1., 1., 0.]]])
        output, state = delta_scan(k, k, v, log_decay, beta, chunk_size=3)
        torch.testing.assert_close(output.flatten(), torch.tensor([2., 3., 7., 7.]), rtol=0, atol=0)
        torch.testing.assert_close(state.flatten(), torch.tensor([7., 3.]), rtol=0, atol=0)
        log_decay[..., 2] = -1000.
        output, state = delta_scan(k, k, v, log_decay, beta, chunk_size=4)
        torch.testing.assert_close(state.flatten(), torch.tensor([7., 0.]), rtol=0, atol=0)
        self.assertTrue(torch.isfinite(output).all())

    def test_long_fp32_strong_decay_remains_finite_and_causal(self):
        torch.manual_seed(3)
        q, k = [F.normalize(torch.randn(1, 2, 480, 8), dim=-1).requires_grad_() for _ in range(2)]
        v = torch.randn(1, 2, 480, 9, requires_grad=True)
        log_decay = (-torch.rand(1, 2, 480) * 100).requires_grad_()
        beta = torch.rand(1, 2, 480, requires_grad=True)
        output, state = delta_scan(q, k, v, log_decay, beta)
        expected, expected_state = delta_recurrent_reference(q, k, v, log_decay, beta)
        torch.testing.assert_close(output, expected, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(state, expected_state, atol=3e-5, rtol=3e-5)
        gradients = torch.autograd.grad(output[:, :, 95].sum(), (q, k, v, log_decay, beta))
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertEqual(gradient[:, :, 96:].count_nonzero().item(), 0)


if __name__ == '__main__':
    unittest.main()
