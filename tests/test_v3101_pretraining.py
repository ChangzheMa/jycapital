"""阶段分离、无标签采样、手算去时点统计和精确恢复的CPU检查。"""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np

from test_v3101 import trainer, small_config, inputs, Store, activate_price_memory
import v3101_pretraining as pre


def assert_nested_equal(test, left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            assert_nested_equal(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_nested_equal(test, a, b)
    else:
        test.assertEqual(left, right)


class PretrainingTests(unittest.TestCase):
    def test_loss_matches_independent_numpy_and_clock_only_is_penalized(self):
        spec = small_config()['training']['pretraining']
        left = np.random.default_rng(3).normal(size=(5, 8, 3))
        right = np.random.default_rng(4).normal(size=(5, 8, 3))
        present = np.ones((5, 8), bool)
        present[2:, 1] = False
        present[1:, 3] = False  # 单样本时点仍可做一致性，不进入方差／协方差。
        present[:, 7] = False
        invariance = ((left - right)[present] ** 2).mean()
        var, cov = [], []
        for values in (left, right):
            matrices, penalties, counts = [], [], []
            for clock in range(8):
                x = values[present[:, clock], clock]
                if len(x) < 2:
                    continue
                centered = x - x.mean(0)
                matrices.append(centered.T @ centered)
                counts.append(len(x) - 1)
                penalties.extend(np.maximum(0, 1 - np.sqrt(x.var(0, ddof=1) + 0.0001)))
            var.append(np.mean(penalties))
            matrix = sum(matrices) / sum(counts)
            cov.append(sum(matrix[i, j] ** 2 for i in range(3) for j in range(3) if i != j) / 3)
        expected = 25 * invariance + 25 * np.mean(var) + sum(cov)
        a = torch.tensor(left, requires_grad=True)
        b = torch.tensor(right, requires_grad=True)
        actual, _ = pre.representation_loss(a, b, torch.from_numpy(present), spec)
        self.assertAlmostEqual(actual.item(), expected, places=11)
        actual.backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertEqual(a.grad[~torch.from_numpy(present)].count_nonzero().item(), 0)
        clock_only = torch.arange(8.).reshape(1, 8, 1).expand(5, 8, 3) * 100
        loss, details = pre.representation_loss(clock_only, clock_only, torch.ones(5, 8, dtype=torch.bool), spec)
        self.assertAlmostEqual(loss.item(), 24.75, places=4)
        self.assertEqual(details['covariance'].item(), 0)

    def test_mask_alignment_no_mutation_and_features_remain_causal(self):
        config = small_config()
        windows, clock = inputs(config, count=2)
        saved = {f: x.clone() for f, x in windows.items()}
        keep = torch.ones(2, 21, 8, dtype=torch.bool)
        keep[0, -2, 2] = False
        masked = pre.mask_windows(windows, .2, keep=keep)
        for f, value in windows.items():
            bars = pre.DAILY_BARS[f]
            start = value.shape[-1] - 2 * bars + 2 * (bars // 8)
            self.assertEqual(masked[f][0, :, start:start + bars // 8].count_nonzero().item(), 0)
            torch.testing.assert_close(value, saved[f], rtol=0, atol=0)
            torch.testing.assert_close(masked[f][1], value[1], rtol=0, atol=0)
        model = trainer.MultiFrequencyGRU(config).eval()
        activate_price_memory(model)
        windows = {f: v.requires_grad_(True) for f, v in windows.items()}
        features = model.forward_group_features(pre.mask_windows(windows, .2, keep=keep), clock)
        for q in range(8):
            grads = torch.autograd.grad(features[:, q].sum(), tuple(windows.values()), retain_graph=True)
            for f, grad in zip(windows, grads):
                cutoff = windows[f].shape[-1] - pre.DAILY_BARS[f] + q * pre.DAILY_BARS[f] // 8
                self.assertEqual(grad[:, :, cutoff:].count_nonzero().item(), 0)

    def test_pretext_order_and_membership_ignore_label_validity(self):
        store = Store()
        block = store.load('train', '2020-01')
        before = pre.unlabelled_view(block)
        original = before.index.samples.copy()
        order = pre.shuffled_pretraining_groups(before, 42, 1, '2020-01')
        block.keys['vwap_return'] = np.nan
        after = pre.unlabelled_view(block)
        np.testing.assert_array_equal(original, after.index.samples)
        np.testing.assert_array_equal(order, pre.shuffled_pretraining_groups(after, 42, 1, '2020-01'))
        self.assertEqual(len(order), 12)
        self.assertEqual(before.targets.dtype, np.dtype(bool))

    def test_pretraining_resume_and_completed_stage_skip_are_exact(self):
        config, store = small_config(), Store()
        for block in store.data['train'].values():
            block.keys['vwap_return'] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            runs = [Path(tmp) / name for name in ('full', 'resumed')]
            for run in runs:
                run.mkdir()
                trainer.write_json(run / 'config.json', config)
                trainer.write_json(run / 'data-manifest.json', {'synthetic': True})
            load = store.load

            def train_only(split, month):
                self.assertEqual(split, 'train')
                return load(split, month)

            with patch.object(store, 'load', train_only):
                pre.run_pretraining(store, runs[0], config, {}, trainer)
                step = pre.pretraining_step
                calls = 0

                def interrupt(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 3:
                        raise KeyboardInterrupt
                    return step(*args, **kwargs)

                with patch.object(pre, 'pretraining_step', interrupt), self.assertRaises(KeyboardInterrupt):
                    pre.run_pretraining(store, runs[1], config, {}, trainer)
                state = torch.load(runs[1] / 'pretraining-last.pt', weights_only=True)
                self.assertEqual(state['progress']['group_cursor'], 4)
                pre.run_pretraining(store, runs[1], config, {}, trainer, resume=True)
            left, right = [torch.load(r / 'pretraining-last.pt', weights_only=True) for r in runs]
            for key in ('state_dict', 'projector', 'optimizer', 'torch_rng', 'cuda_rng'):
                assert_nested_equal(self, left[key], right[key])
            for a, b in zip(left['progress']['history'], right['progress']['history']):
                self.assertEqual({k: v for k, v in a.items() if k != 'seconds'},
                                 {k: v for k, v in b.items() if k != 'seconds'})
            receipt = pre.verified_pretraining(runs[1], config)
            with patch.object(store, 'load', side_effect=AssertionError('已完成预训练不得重新加载行情')):
                self.assertEqual(pre.run_pretraining(store, runs[1], config, {}, trainer, resume=True), receipt)
            with (runs[1] / 'pretrained.pt').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaises(ValueError):
                pre.verified_pretraining(runs[1], config)

    def test_bfloat16_pretext_updates_encoder_only_and_projector_is_training_only(self):
        config = small_config()
        model = trainer.MultiFrequencyGRU(config)
        projector = pre.make_projector(config)
        heads = {k: v.clone() for k, v in model.state_dict().items() if k.startswith(('experts.', 'router.'))}
        original = model.encoders['bar1m'].project.dense.weight.detach().clone()
        optimizer = torch.optim.AdamW([*model.parameters(), *projector.parameters()], lr=.0001)
        windows, clock = inputs(config, 3)
        result = pre.pretraining_step(model, projector, optimizer, windows, clock,
                                     torch.ones(3, 8, dtype=torch.bool), config, True)
        self.assertTrue(np.isfinite(result['loss']))
        self.assertFalse(torch.equal(original, model.encoders['bar1m'].project.dense.weight))
        for key, value in heads.items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        self.assertFalse(any('projector' in k for k in model.state_dict()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in projector.parameters()))


if __name__ == '__main__':
    unittest.main()
