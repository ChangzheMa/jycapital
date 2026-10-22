"""用可手算的量化例子核验诊断，并确保排除模型网格外的行情。"""

import importlib.util
from pathlib import Path
import unittest

import torch
import numpy as np
import pandas as pd

path=Path(__file__).resolve().parents[1]/'scripts/audit-input-precision.py'
spec=importlib.util.spec_from_file_location('input_precision_audit',path)
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class PrecisionAuditTests(unittest.TestCase):
    def test_previous_regular_day_anchor_preserves_small_changes(self):
        frame=pd.DataFrame({'date':pd.to_datetime(['2020-06-01 09:16','2020-06-01 09:31','2020-06-01 15:00',
            '2020-06-01 15:30','2020-06-02 09:31','2020-06-02 09:32','2020-06-02 09:33']), 'instrument':['A']*7})
        for column in audit.output_columns()[2:]:
            frame[column]=np.array([10000.,64.,64.,10000.,64.+1/1024,64.+2/1024,64.+3/1024],np.float32)
        result,days=audit.inspect_frame(frame,'bar1m')
        self.assertEqual(days,['2020-06-02'])
        for row in result.values():
            self.assertEqual(row['finite_pairs'],2)
            self.assertEqual(row['nonzero_changes'],2)
            self.assertEqual(row['absolute_erased'],2)
            self.assertEqual(row['delta_erased'],0)
            self.assertEqual(row['delta_change_error_sum'],0)
            self.assertEqual(row['delta_value_error_sum'],0)

    def test_overflow_and_nan_policy_matches_existing_codec(self):
        values=np.array([70000.,-70000.,np.nan,1.],np.float32)
        actual=audit.rounded(values)
        np.testing.assert_array_equal(actual[[0,1,3]],values[[0,1,3]])
        self.assertTrue(np.isnan(actual[2]))


if __name__=='__main__':
    unittest.main()
