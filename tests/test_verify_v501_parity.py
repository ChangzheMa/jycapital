"""GRU 不能仅放宽冻结预测误差；跨后端容差必须有独立同后端验证。"""

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_v501_parity", ROOT / "scripts/verify-v501-submission.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class NumericParityTests(unittest.TestCase):
    def test_original_tolerance_retained_without_cpu_reference(self):
        with self.assertRaises(AssertionError):
            verifier.check_numeric_parity([2e-6], [0.])

    def test_failed_same_backend_parity_cannot_be_overridden(self):
        with self.assertRaises(AssertionError):
            verifier.check_numeric_parity([2e-6], [0.], [0.])

    def test_small_backend_difference_retains_original_failure_evidence(self):
        result = verifier.check_numeric_parity([2e-6], [0.], [2e-6])
        self.assertFalse(result["original_tolerance_passed"])
        self.assertEqual(result["original_mismatched_rows"], 1)
        self.assertEqual(result["same_backend_max_abs_difference"], 0)

    def test_large_backend_difference_still_fails(self):
        with self.assertRaises(AssertionError):
            verifier.check_numeric_parity([1e-4], [0.], [1e-4])

    def test_nonfinite_or_wrong_shape_rejected(self):
        for package, frozen, cpu in [([float("nan")], [float("nan")], None), ([0., 0.], [0.], None),
                                      ([0.], [0.], [float("nan")])]:
            with self.assertRaises(AssertionError):
                verifier.check_numeric_parity(package, frozen, cpu)


if __name__ == "__main__":
    unittest.main()
