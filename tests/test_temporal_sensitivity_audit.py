"""时间敏感度归一化、缺失和边界的可手算检查。"""
import importlib.util
from pathlib import Path
import unittest

import torch
import numpy as np

spec = importlib.util.spec_from_file_location("temporal_audit", Path(__file__).resolve().parents[1] / "scripts/audit-temporal-sensitivity.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class TemporalProfileTest(unittest.TestCase):
    def test_exact_age_and_missing_mass(self):
        gradient = np.zeros((72, 480), np.float32)
        finite = np.ones_like(gradient, bool)
        gradient[0, 269], gradient[0, 264], gradient[0, 239] = 2, -1, 1
        gradient[1, 0] = 100
        finite[1, 0] = False
        r = audit.time_profile(gradient, finite, 270, 240)
        self.assertEqual(r["total_absolute_gradient"], 4)
        self.assertEqual(r["by_trading_minute_age"]["0_5"], .5)
        self.assertEqual(r["by_trading_minute_age"]["5_30"], .25)
        self.assertEqual(r["by_trading_minute_age"]["30_60"], .25)
        self.assertEqual(r["by_day_offset"], {"0": .75, "1": .25})
        self.assertEqual(r["bars_covering_50_percent"], 1)
        self.assertEqual(r["bars_covering_90_percent"], 31)

    def test_opening_and_future_boundary(self):
        gradient = np.zeros((72, 168), np.float32)
        finite = np.ones_like(gradient, bool)
        gradient[0, 159] = 1
        r = audit.time_profile(gradient, finite, 160, 8)
        self.assertEqual(r["by_day_offset"]["0"], 0)
        self.assertEqual(r["by_day_offset"]["1"], 1)
        gradient[0, 160] = .1
        with self.assertRaises(ValueError):
            audit.time_profile(gradient, finite, 160, 8)
        gradient[:] = 0
        self.assertIsNone(audit.time_profile(gradient, finite, 160, 8))


if __name__ == "__main__":
    unittest.main()
