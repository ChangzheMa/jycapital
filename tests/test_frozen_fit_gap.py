"""抽样拟合诊断的有限性、标签仿射不变性和截面加权口径。"""
import importlib.util
from pathlib import Path
import unittest

import torch
import numpy as np

spec=importlib.util.spec_from_file_location('fit_gap_audit',Path(__file__).resolve().parents[1]/'scripts/analyze-frozen-fit-gap.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class FrozenFitGapTests(unittest.TestCase):
    def test_label_standardization_does_not_change_sample_ic(self):
        x=np.array([1.,3.,2.,5.,4.,np.nan]);y=np.array([3.,8.,2.,6.,9.,0.])
        left=audit.section_metrics(x,y,3);right=audit.section_metrics(x,7*y+100,3)
        self.assertEqual(left['valid_rows'],5)
        self.assertAlmostEqual(left['ic'],right['ic'],places=14)
        self.assertAlmostEqual(left['rank_ic'],right['rank_ic'],places=14)
        self.assertIsNone(audit.section_metrics(np.ones(5),np.arange(5),3)['ic'])
        self.assertIsNone(audit.section_metrics(x,y,6)['ic'])

    def test_means_weight_sections_and_resampling_is_reproducible(self):
        rows=[{'day':'2020-01-02','clock':'10:00','ic':.9,'rank_ic':.8,'valid_rows':10},
              {'day':'2020-01-03','clock':'10:00','ic':0.,'rank_ic':0.,'valid_rows':10},
              {'day':'2020-01-03','clock':'10:30','ic':0.,'rank_ic':0.,'valid_rows':10}]
        result=audit.aggregate(rows,42)
        self.assertAlmostEqual(result['ic_mean'],.3)
        self.assertEqual(result['defined_sections'],3)
        self.assertEqual(result,audit.aggregate(rows,42))
        lo,hi=result['ic_day_resampling_95pct_interval']
        self.assertGreaterEqual(lo,0);self.assertLessEqual(hi,.9)


if __name__=='__main__':unittest.main()
