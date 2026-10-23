"""预检可识别无关增益舍入，同时严格拒绝真实树状态变化。"""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
for directory in ('common', 'evaluation', 'solutions/v2801'):
    sys.path.insert(0, str(ROOT / directory))
from v2801_preflight import compare_resume_models


class ResumePreflightTests(unittest.TestCase):
    def setUp(self):
        self.document = {'name':'gbtree', 'model':{'trees':[{
            'left_children':[1,-1,-1], 'right_children':[2,-1,-1],
            'split_indices':[0,0,0], 'split_conditions':[.125,.25,-.375],
            'default_left':[True,False,False], 'sum_hessian':[2048.,1024.,1024.],
            'loss_changes':[.35052836,0.,0.]}], 'iteration_indptr':[0,1]}}

    def test_gain_rounding_is_separate_from_exact_prediction_state(self):
        changed = copy.deepcopy(self.document)
        changed['model']['trees'][0]['loss_changes'][0] = .35052833
        result = compare_resume_models(self.document,changed)
        self.assertFalse(result['native_tree_document_exact'])
        self.assertTrue(result['prediction_structure_and_other_state_exact'])
        self.assertAlmostEqual(result['nonpredictive_gain_max_abs_difference'],3e-8)
        self.assertEqual(self.document['model']['trees'][0]['loss_changes'][0],.35052836)

    def test_threshold_leaf_topology_missing_direction_and_other_state_are_strict(self):
        cases = [('split_conditions',0,.12500001),('split_conditions',1,.25000001),
                 ('left_children',0,2),('default_left',0,False),('sum_hessian',0,2047.)]
        for field,index,value in cases:
            with self.subTest(field=field,index=index):
                changed = copy.deepcopy(self.document)
                changed['model']['trees'][0][field][index] = value
                with self.assertRaises(ValueError):compare_resume_models(self.document,changed)

    def test_large_nonfinite_gain_and_count_changes_are_rejected(self):
        for value in (.36,float('nan'),float('inf')):
            changed = copy.deepcopy(self.document)
            changed['model']['trees'][0]['loss_changes'][0] = value
            with self.assertRaises((ValueError,AssertionError)):compare_resume_models(self.document,changed)
        changed = copy.deepcopy(self.document)
        changed['model']['trees'].clear()
        with self.assertRaises(ValueError):compare_resume_models(self.document,changed)


if __name__ == '__main__':
    unittest.main()
