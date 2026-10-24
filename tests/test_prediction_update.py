"""用可手算关系检查预测差分解，避免将分母变化误作新增信息。"""
import importlib.util
from pathlib import Path
import unittest

import torch
import numpy as np

path=Path(__file__).resolve().parents[1]/'scripts/analyze-prediction-update.py'
spec=importlib.util.spec_from_file_location('prediction_update',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class UpdateTests(unittest.TestCase):
    def test_known_orthogonal_signal_and_dilution(self):
        parent=np.array([-1.,0.,1.]);extra=np.array([1.,-2.,1.]);target=parent+extra
        row=module.decompose_section(parent,target,target)
        for key,expected in {'parent_ic':.5,'child_ic':1.,'ic_delta':.5,
                             'orthogonal_ic_contribution':.75,'parent_direction_ic_change':-.25,
                             'orthogonal_child_variance_share':.75,'prediction_correlation':.5}.items():
            self.assertAlmostEqual(row[key],expected,places=14)

    def test_positive_affine_changes_no_ic_or_new_direction(self):
        parent=np.array([-2.,-1.,0.,1.,2.]);target=np.array([1.,-1.,2.,0.,4.])
        row=module.decompose_section(parent,3*parent+7,target)
        for key in ('ic_delta','orthogonal_ic_contribution','parent_direction_ic_change','orthogonal_child_variance_share'):
            self.assertAlmostEqual(row[key],0.,places=14)
        self.assertAlmostEqual(row['intercept_update'],7.)

    def test_constant_day_difference_interval_and_contribution_sum(self):
        report=module.moving_block_interval(np.full(23,.03))
        np.testing.assert_allclose(report['percentile_95'],[.03,.03],atol=1e-14,rtol=0)
        p=np.array([-1.,0.,1.]);y=np.array([0.,2.,1.])
        terms,pvar,yvar=module.normalized_terms(p,y)
        self.assertAlmostEqual(terms.sum(),.5)
        self.assertAlmostEqual(pvar.sum(),1.)
        self.assertAlmostEqual(yvar.sum(),1.)


if __name__ == '__main__':
    unittest.main()
