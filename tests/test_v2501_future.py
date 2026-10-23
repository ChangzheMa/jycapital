"""后续收益监督的可手算损失、成员资格和因果索引检查。"""
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'solutions/v2501'))
from v2501_future import future_pairs, future_objective, FutureReturnHead


class FutureLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.spec = {'huber_delta': 2.}

    def targets(self):
        value = torch.full((1, 8, 3), float('nan'), dtype=torch.float64)
        value[..., 2] = 1.
        value[0, 1, :2] = 1.
        value[0, 2, :2] = 3.
        return value

    def test_strict_triangle_and_query_equal_weighting_by_hand(self):
        query, target = future_pairs()
        self.assertEqual(query.numel(), 28)
        self.assertTrue((target > query).all())
        prediction = torch.zeros(1, 28, requires_grad=True)
        loss, result = future_objective(prediction, self.targets(), None, self.spec)
        # q0有两项Huber(.5,4)，均值2.25；q1有一项4；查询等权均值3.125。
        self.assertAlmostEqual(float(loss.detach()), 3.125)
        self.assertEqual(result['future_queries'], 2)
        self.assertEqual(result['future_pairs'], 3)
        self.assertEqual(result['future_lead_pair_counts'].tolist(), [2, 1, 0, 0, 0, 0, 0])
        loss.backward()
        expected = torch.zeros_like(prediction)
        expected[0, 0] = -.25
        expected[0, 1] = -.5
        expected[0, 7] = -1.
        torch.testing.assert_close(prediction.grad, expected, rtol=0, atol=0)

    def test_current_membership_and_date_weights(self):
        targets = self.targets()
        targets[0, 0, 2] = 0.
        loss, detail = future_objective(torch.zeros(1, 28), targets, None, self.spec)
        self.assertEqual(float(loss), 4.)
        self.assertEqual(detail['future_pairs'], 1)
        second = self.targets()
        second[..., :2] = float('nan')
        second[0, 1, :2] = 2.
        values = torch.cat((self.targets(), second), dim=0)
        loss, detail = future_objective(torch.zeros(2, 28), values, torch.tensor([1., 3.]), self.spec)
        self.assertAlmostEqual(float(loss), (2.25 + 4. + 3.*2.) / 5.)
        self.assertEqual(detail['future_queries'], 3)

    def test_missing_future_targets_have_zero_gradient_and_finite_zero_loss(self):
        targets = self.targets()
        targets[..., :2] = float('nan')
        targets[0, 0, :2] = 1.
        prediction = torch.randn(1, 28, requires_grad=True)
        loss, detail = future_objective(prediction, targets, None, self.spec)
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(detail['future_queries'], 0)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction), rtol=0, atol=0)

    def test_only_same_group_target_is_used(self):
        first = self.targets()
        second = self.targets()
        second[0, 1:3, :2] *= -1.
        targets = torch.cat((first, second), dim=0)
        prediction = torch.zeros(2, 28, requires_grad=True)
        loss, _ = future_objective(prediction, targets, None, self.spec)
        loss.backward()
        torch.testing.assert_close(prediction.grad[0], -prediction.grad[1], rtol=0, atol=0)
        bad = targets.clone()
        bad[0, 1, 2] = 0.
        with self.assertRaisesRegex(ValueError, '非成员'):
            future_objective(prediction, bad, None, self.spec)

    def test_head_reads_query_state_without_target_state(self):
        torch.manual_seed(31)
        head = FutureReturnHead(6, 8, 0.).eval()
        states = torch.randn(2, 8, 6, requires_grad=True)
        values = head(states)
        query, target = future_pairs()
        for index in (0, 6, 7, 17, 27):
            gradient = torch.autograd.grad(values[:, index].sum(), states, retain_graph=True)[0]
            other = torch.arange(8) != query[index]
            self.assertEqual(gradient[:, other].count_nonzero().item(), 0)
            self.assertGreater(gradient[:, query[index]].abs().sum().item(), 0.)
        self.assertEqual(gradient[:, 7].count_nonzero().item(), 0)


if __name__ == '__main__':
    unittest.main()
