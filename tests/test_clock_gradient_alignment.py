"""同一时点噪声基准与跨时点余弦的可手算检查。"""
import importlib.util
from pathlib import Path
import unittest

import torch

spec=importlib.util.spec_from_file_location('clock_gradient_audit',Path(__file__).resolve().parents[1]/'scripts/analyze-clock-gradient-alignment.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class ClockGradientTests(unittest.TestCase):
    def test_orthogonal_tasks_identical_halves(self):
        values=torch.eye(8).repeat_interleave(2,dim=0)
        profile=audit.cross_half_profile(audit.normalized_gram(values))
        self.assertEqual(profile['same_clock_cross_half_mean'],1)
        self.assertEqual(profile['different_clock_cross_half_mean'],0)
        self.assertEqual(profile['different_clock_negative_fraction'],0)

    def test_scale_invariance_and_zero_rejection(self):
        values=torch.arange(1,49,dtype=torch.float32).reshape(16,3)
        original=audit.normalized_gram(values)
        scaled=audit.normalized_gram(values*torch.arange(1,17)[:,None])
        torch.testing.assert_close(original,scaled,rtol=1e-12,atol=1e-12)
        with self.assertRaises(ValueError):audit.normalized_gram(torch.zeros(16,3))


if __name__=='__main__':unittest.main()
