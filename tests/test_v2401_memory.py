"""价格历史读取的因果与数值行为；不执行正式训练。"""
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'solutions/v2401'))
from v2401_price_memory import PriceHistoryMemory


def make_memory(length=12):
    torch.manual_seed(917)
    model = PriceHistoryMemory(8, length, {'width': 8, 'heads': 2, 'distance_hidden_dim': 5,
                               'residual_scale_initial': .1}, dropout=0.)
    # 非零价格和位置参数，防止仅验证初始退化路径。
    with torch.no_grad():
        model.price_bias[-1].weight.normal_(0, .1)
        model.price_bias[-1].bias.normal_(0, .1)
        model.age_bias.weight.normal_(0, .1)
    return model.eval()


class PriceMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_uniform_attention_has_hand_computable_relative_price_value(self):
        model = make_memory(4)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.residual_scale.fill_(1.)
            model.output.weight[0, model.width] = 1.
        raw = torch.zeros(1, 75, 4)
        raw[:, 74] = 1.
        anchor = torch.tensor([[1., 1.125, 1.25, 9.]])
        states = torch.zeros(1, 3, 8)
        out = model(states, raw, anchor, torch.ones(1, 4, dtype=torch.bool), torch.tensor([[1, 2, 3]]))
        # 只读[1]、[1,1.125]、[1,1.125,1.25]，最后9不得影响前三个查询。
        torch.testing.assert_close(out[0, :, 0], torch.tensor([0., -6.25, -12.5]))
        torch.testing.assert_close(out[..., 1:], torch.zeros_like(out[..., 1:]))

    def test_every_query_excludes_future_raw_and_anchor_gradients(self):
        model = make_memory(12)
        for end in (0, 1, 3, 5, 7, 8, 10, 12):
            raw = torch.randn(2, 75, 12)
            raw[:, 74] = 1.
            raw.requires_grad_()
            anchor = (2. + .01 * torch.randn(2, 12)).requires_grad_()
            states = torch.randn(2, 1, 8, requires_grad=True)
            available = torch.ones(2, 12, dtype=torch.bool)
            lengths = torch.full((2, 1), end, dtype=torch.int64)
            output = model(states, raw, anchor, available, lengths)
            raw_grad, anchor_grad = torch.autograd.grad(output.square().sum(), (raw, anchor))
            self.assertEqual(torch.count_nonzero(raw_grad[:, :, end:]).item(), 0)
            self.assertEqual(torch.count_nonzero(anchor_grad[:, end:]).item(), 0)
            changed_raw, changed_anchor = raw.detach().clone(), anchor.detach().clone()
            changed_raw[:, :, end:] += 17.
            changed_raw[:, 74, end:] = 1.
            changed_anchor[:, end:] += 5.
            changed = model(states.detach(), changed_raw, changed_anchor, available, lengths)
            torch.testing.assert_close(output, changed, rtol=0, atol=0)
            if end > 1:
                self.assertGreater(anchor_grad[:, :end].abs().sum().item(), 0.)

    def test_price_translation_and_query_batching_preserve_memory(self):
        model = make_memory()
        raw = torch.randn(2, 75, 12)
        raw[:, 74] = 1.
        states = torch.randn(2, 8, 8)
        anchor = 2. + .01 * torch.randn(2, 12)
        available = torch.ones(2, 12, dtype=torch.bool)
        lengths = torch.tensor([[1, 2, 3, 5, 6, 8, 10, 12]]).expand(2, -1)
        output = model(states, raw, anchor, available, lengths)
        shifted = model(states, raw, anchor + .5, available, lengths)
        torch.testing.assert_close(output, shifted, rtol=2e-5, atol=2e-5)
        reference = torch.stack([model.single(states[:, i], raw, anchor, available, lengths[:, i])
                                 for i in range(8)], dim=1)
        torch.testing.assert_close(output, reference, rtol=1e-5, atol=3e-7)
        singles = torch.cat([model(states[i:i+1], raw[i:i+1], anchor[i:i+1], available[i:i+1], lengths[i:i+1])
                             for i in range(2)])
        torch.testing.assert_close(output, singles, rtol=1e-5, atol=3e-7)

    def test_missing_centers_empty_history_and_int32_lengths(self):
        model = make_memory(4)
        raw = torch.randn(1, 75, 4)
        raw[:, 74] = torch.tensor([1., 0., 1., 1.])
        states = torch.randn(1, 3, 8)
        anchors = torch.tensor([[2., float('nan'), 2.1, float('nan')]])
        available = torch.tensor([[True, False, True, False]])
        lengths = torch.tensor([[0, 2, 4]], dtype=torch.int32)
        out = model(states, raw, anchors, available, lengths)
        self.assertTrue(torch.isfinite(out).all())
        torch.testing.assert_close(out[:, 0], states[:, 0], rtol=0, atol=0)
        # 不可用价格的具体占位值不影响任何输出。
        alt = torch.tensor([[2., 999., 2.1, -999.]])
        torch.testing.assert_close(out, model(states, raw, alt, available, lengths), rtol=0, atol=0)
        empty = raw.clone()
        empty[:, 74] = 0.
        torch.testing.assert_close(model(states, empty, anchors, available, lengths), states, rtol=0, atol=0)
        result = model(states[:0], raw[:0], anchors[:0], available[:0], lengths[:0])
        self.assertEqual(result.shape, (0, 3, 8))

    def test_cpu_bfloat16_gradients_are_finite(self):
        model = make_memory(4).train()
        raw = torch.randn(2, 75, 4)
        raw[:, 74] = 1.
        states = torch.randn(2, 2, 8)
        anchors = torch.randn(2, 4)
        available = torch.ones(2, 4, dtype=torch.bool)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            out = model(states, raw, anchors, available, torch.tensor([[2, 4], [1, 3]]))
            out.square().mean().backward()
        self.assertTrue(torch.isfinite(out).all())
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == '__main__':
    unittest.main()
