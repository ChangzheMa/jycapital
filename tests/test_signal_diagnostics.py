"""检查换手代理的权重、同分、股票变化与跨日边界。"""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from signal_diagnostics import intraday_diagnostics, tail_weights


class SignalDiagnosticsTests(unittest.TestCase):
    def test_tie_allocation_is_permutation_invariant(self):
        scores = np.array([0, 0, 1, 2, 3, 3], float)
        weights = tail_weights(scores, 0.2)
        np.testing.assert_allclose(weights, [-0.25, -0.25, 0, 0, 0.25, 0.25])
        permutation = [4, 1, 3, 0, 5, 2]
        np.testing.assert_allclose(tail_weights(scores[permutation], 0.2), weights[permutation])
        self.assertAlmostEqual(weights.sum(), 0)
        self.assertAlmostEqual(np.abs(weights).sum(), 1)

    def test_constant_signal_is_not_a_valid_full_gross_portfolio(self):
        np.testing.assert_array_equal(tail_weights(np.ones(10)), np.zeros(10))
        frame = pd.DataFrame([(t, f'S{i}', 1.) for t in pd.to_datetime(['2024-01-02 10:00','2024-01-02 10:30'])
                              for i in range(10)], columns=['date','instrument','factor'])
        summary, _ = intraday_diagnostics(frame, min_common=2)
        self.assertEqual(summary['undefined_portfolio_transitions'], 1)
        self.assertIsNone(summary['one_way_turnover_mean'])

    def test_reversal_turnover_and_no_overnight_transition(self):
        stamps = pd.to_datetime(['2024-01-02 10:00','2024-01-02 10:30','2024-01-03 10:00','2024-01-03 10:30'])
        frame = pd.DataFrame([(t, f'S{i}', float(i if n in [0,2,3] else -i))
                              for n,t in enumerate(stamps) for i in range(10)], columns=['date','instrument','factor'])
        summary, transitions = intraday_diagnostics(frame, min_common=2)
        self.assertEqual(summary['transitions'], 2)
        np.testing.assert_allclose(transitions.one_way_turnover, [1,0])
        np.testing.assert_allclose(transitions.rank_autocorrelation, [-1,1])
        np.testing.assert_allclose(transitions.long_retention, [0,1])

    def test_changed_stock_uses_union_for_turnover(self):
        frame = pd.DataFrame([(pd.Timestamp('2024-01-02 10:00'), f'A{i}', float(i)) for i in range(10)] +
                             [(pd.Timestamp('2024-01-02 10:30'), f'B{i}', float(i)) for i in range(10)],
                             columns=['date','instrument','factor'])
        summary, _ = intraday_diagnostics(frame, min_common=2)
        self.assertEqual(summary['one_way_turnover_mean'], 1)
        self.assertEqual(summary['minimum_common_stocks'], 0)
        self.assertIsNone(summary['rank_autocorrelation_mean'])


if __name__ == '__main__':
    unittest.main()
