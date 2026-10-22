"""未来数量目标与已知基准的因果对齐检查。"""
import importlib.util
from pathlib import Path
import unittest

import torch
import numpy as np

spec = importlib.util.spec_from_file_location("quantity_supervision", Path(__file__).resolve().parents[1] / "scripts/audit-quantity-supervision.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class QuantitySupervisionTest(unittest.TestCase):
    def test_target_and_persistence_boundary(self):
        current = np.broadcast_to(np.arange(8)[None, :, None], (2, 8, 22)).astype(float)
        previous = current + 100
        aligned = audit.align_targets(current, previous)
        np.testing.assert_array_equal(aligned['target'], current)
        np.testing.assert_array_equal(aligned['last_closed_bar'][:, 0], previous[:, -1])
        np.testing.assert_array_equal(aligned['last_closed_bar'][:, 1:], current[:, :-1])
        self.assertEqual(audit.TARGET_CLOSES[4], '13:30')
        changed = current.copy(); changed[:, 4:] += 999
        new = audit.align_targets(changed, previous)
        np.testing.assert_array_equal(new['last_closed_bar'][:, :5], aligned['last_closed_bar'][:, :5])

    def test_sparse_overflow_and_common_coverage(self):
        values = np.zeros((2, 8, 72), np.float16)
        values[1, 0, 4] = np.inf
        restored = audit.decode_block(values, np.array([0, 0, 1]), np.array([4]), np.array([100000.]), 1)
        self.assertEqual(restored[0, 4], 100000.)
        y = np.arange(64, dtype=float)
        last, old = y.copy(), y + 1
        last[0], old[1] = np.nan, np.nan
        r = audit.statistics(y, last, old)
        self.assertEqual(r['matched'], 62)
        self.assertEqual(r['target_finite'], 64)
        self.assertEqual(r['last_closed_bar_squared_error_sum'], 0)
        self.assertEqual(r['previous_day_same_slot_squared_error_sum'], 62)
        self.assertAlmostEqual(r['last_closed_bar_correlation'], 1)


if __name__ == '__main__':
    unittest.main()
