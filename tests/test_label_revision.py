"""两份提供标签的相似度采用同一截面共同有限样本。"""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

path=Path(__file__).resolve().parents[1]/'scripts/analyze-label-revision.py'
spec=importlib.util.spec_from_file_location('label_revision',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class LabelRevisionTests(unittest.TestCase):
    def test_opposite_and_positive_affine_labels_have_known_correlations(self):
        for values,expected in (([1.,0.,-1.],-1.),([-1.,2.,5.],1.)):
            frame=pd.DataFrame({'date':pd.to_datetime(['2024-01-02']*3),'old':[-1.,0.,1.],'new':values})
            result=module.section_similarity(frame,3)[0]
            self.assertAlmostEqual(result['pearson'],expected);self.assertAlmostEqual(result['rank'],expected)

    def test_common_finite_mask_minimum_and_constant_are_explicit(self):
        frame=pd.DataFrame({'date':pd.to_datetime(['2024-01-02']*4),'old':[-1.,0.,1.,np.nan],'new':[3.,3.,3.,100.]})
        result=module.section_similarity(frame,3)[0]
        self.assertIsNone(result['pearson']);self.assertIsNone(result['rank'])
        self.assertEqual(module.section_similarity(frame,4),[])


if __name__=='__main__':unittest.main()
