"""字段网格的独立时间递推、手算空间交互与日边界检查。"""

import copy
import unittest

import torch

from test_v3201 import small_config, inputs, trainer, activate_price_memory
from v3201_field_grid import IntradayFieldGrid, SynchronousNodeMixing


class FieldGridTests(unittest.TestCase):
    def test_packed_node_recurrence_matches_independent_values_and_gradients(self):
        torch.manual_seed(17)
        recurrent = torch.nn.GRU(4, 4, batch_first=True).double()
        tokens = torch.randn(2, 9, 6, 4, dtype=torch.float64, requires_grad=True)
        # 正式递推固定FP32；此处用FP32对照独立节点，检查维度组织和共享梯度。
        recurrent = recurrent.float()
        tokens = tokens.detach().float().requires_grad_()
        packed = IntradayFieldGrid.temporal_pass(tokens, recurrent)
        reference = torch.stack([torch.stack([recurrent(tokens[b:b+1, :, n])[0][0]
                                for n in range(6)], dim=1) for b in range(2)])
        torch.testing.assert_close(packed, reference, rtol=1e-5, atol=1e-7)
        weights = torch.randn_like(packed)
        variables = (tokens, *recurrent.parameters())
        left = torch.autograd.grad((packed * weights).sum(), variables, retain_graph=True)
        right = torch.autograd.grad((reference * weights).sum(), variables)
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
        # 初始状态不是在节点之间串联，也不沿batch传播。
        changed = tokens.detach().clone()
        changed[0, :, 0] += 7
        actual = IntradayFieldGrid.temporal_pass(changed, recurrent)
        torch.testing.assert_close(actual[1], packed.detach()[1], rtol=0, atol=0)
        torch.testing.assert_close(actual[0, :, 1:], packed.detach()[0, :, 1:], rtol=0, atol=0)

    def test_field_mapping_keeps_all_values_and_partial_missing_flags(self):
        grid = IntradayFieldGrid(small_config()['model']).eval()
        raw = torch.arange(75.).reshape(1, 1, 75).expand(2, 7, 75).clone()
        raw[..., 72:75] = 0  # 标志为0仍须保留其余有效数值。
        seen = {}
        handles = [grid.book_project.register_forward_pre_hook(lambda m, args: seen.update(book=args[0].clone())),
                   grid.non_book_project.register_forward_pre_hook(lambda m, args: seen.update(non_book=args[0].clone()))]
        tokens = grid.field_tokens(raw)
        for handle in handles:
            handle.remove()
        self.assertEqual(tokens.shape, (2, 7, 6, 8))
        for level in range(5):
            expected = torch.tensor([base + group * 5 + level for base in (6, 42) for group in range(6)] + [0, 0]).float()
            torch.testing.assert_close(seen['book'][0, 0, level], expected, rtol=0, atol=0)
        self.assertEqual(grid.book_indices.unique().numel(), 60)
        expected = torch.tensor([*range(6), *range(36, 42), 0, 0]).float()
        torch.testing.assert_close(seen['non_book'][0, 0], expected, rtol=0, atol=0)

    def test_synchronous_attention_matches_known_mean_and_has_no_time_mixing(self):
        spec = small_config()['model']['field_grid']
        mix = SynchronousNodeMixing(spec, 0).eval()
        width = spec['width']
        with torch.no_grad():
            for p in mix.parameters():
                p.zero_()
            mix.qkv.weight[2*width:].copy_(torch.eye(width))
            mix.projection.weight.copy_(torch.eye(width))
            mix.attention_scale.fill_(1)
        value = torch.randn(2, 7, 6, width, requires_grad=True)
        actual = mix(value)
        expected = value + value.mean(dim=2, keepdim=True)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=2e-7)
        gradient = torch.autograd.grad(actual[0, 3, 0, 0], value)[0]
        self.assertAlmostEqual(gradient[0, 3, 1, 0].item(), 1/6, places=6)
        self.assertEqual(gradient[0, :3].count_nonzero().item(), 0)
        self.assertEqual(gradient[0, 4:].count_nonzero().item(), 0)
        self.assertEqual(gradient[1].count_nonzero().item(), 0)

    def test_day_reset_prefix_and_whole_model_missing_history(self):
        config = small_config()
        grid = IntradayFieldGrid(config['model']).eval()
        raw = torch.randn(3, 11, 75)
        full = grid(raw)
        for stop in (1, 4, 8):
            torch.testing.assert_close(full[:, :stop], grid(raw[:, :stop]), rtol=1e-5, atol=2e-7)
        changed = raw.clone()
        changed[0] *= 13
        torch.testing.assert_close(grid(changed)[1:], full[1:], rtol=0, atol=0)
        model = trainer.MultiFrequencyGRU(config).eval()
        activate_price_memory(model)
        windows, latest = inputs(config, 3)
        for f, v in windows.items():
            v[0].zero_()
            v[1, :, 10:15] = 0
            v[2, 72:74] = 0  # 部分缺失不删除其余有效字段。
        grouped = model.forward_group(windows, latest)
        self.assertTrue(torch.isfinite(grouped).all())
        for q in range(8):
            clock = torch.nn.functional.one_hot(torch.full((3,), q), 8).float()
            torch.testing.assert_close(grouped[:, q], model.forward_heads(windows, clock), rtol=1e-5, atol=2e-7)

    def test_bfloat16_autocast_finite_loss_and_all_parameter_gradients(self):
        config = small_config()
        model = trainer.MultiFrequencyGRU(config)
        windows, clock = inputs(config, 2)
        target = torch.randn(2, 8, 2, dtype=torch.float64)
        result = trainer.training_step(model, torch.optim.AdamW(model.parameters(), lr=.0003),
                                       windows, clock, target, config, True)
        self.assertTrue(torch.isfinite(torch.tensor(result['loss'])))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(model.encoders['bar1m'].intraday.temporal_first.weight_ih_l0.grad.abs().sum().item(), 0)
        self.assertGreater(model.encoders['bar1m'].intraday.spatial.qkv.weight.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
