"""检验几何统计分母、共同变换与严格前日锚点。"""
import importlib.util
from pathlib import Path
import unittest

import torch
import numpy as np
import pandas as pd

path = Path(__file__).resolve().parents[1] / "scripts/audit-price-geometry.py"
spec = importlib.util.spec_from_file_location("price_geometry_audit", path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class PriceGeometryTests(unittest.TestCase):
    def test_counts_separate_inversions_ties_missing_and_raw_errors(self):
        raw = np.ones((5, len(audit.FIELDS)))
        a, b = [audit.FIELDS.index(k) for k in ("bid_price1", "ask_price1")]
        raw[:, a], raw[:, b] = [10, 10, 10, 0, 11], [11, 11, 10, 11, 10]
        transformed = raw.copy()
        transformed[:, a], transformed[:, b] = [1, 1, 1, 1, 2], [0, 1, 2, 2, 1]
        result = audit.relation_counts(raw, transformed, [("bid_price1", "ask_price1")])
        self.assertEqual(result, {"valid_pairs": 4, "raw_order_violations": 1, "strict_raw_pairs": 2,
                                  "strict_reversed": 1, "strict_erased": 1, "raw_ties": 1,
                                  "raw_ties_split": 1, "sign_agreement_pairs": 1})

    def test_anchor_uses_only_prior_observed_regular_day(self):
        frame = pd.DataFrame({"instrument": ["A"] * 7,
                              "date": pd.to_datetime(["2020-06-01 09:16", "2020-06-01 15:00", "2020-06-01 15:30",
                                                       "2020-06-02 09:31", "2020-06-02 15:00", "2020-06-04 09:31", "2020-06-04 15:00"]),
                              "close": [999, 10, 888, 0, np.nan, 12, 13]})
        anchors = audit.prior_close_anchors(frame)
        self.assertTrue(np.isnan(anchors.anchor.iloc[0]))
        self.assertEqual(anchors.anchor.tolist()[1:], [10, 10])
        self.assertTrue((anchors.anchor_day.iloc[1:] < anchors.day.iloc[1:]).all())
        changed = frame.copy()
        changed.loc[changed.date.dt.day.eq(4), "close"] = 999
        pd.testing.assert_frame_equal(audit.prior_close_anchors(changed), anchors)

    def test_shared_positive_price_center_preserves_order_and_ties(self):
        raw = np.array([[10, 10.01, 10.02], [100, 100, 101], [0, -1, np.nan]])
        result = audit.shared_center(raw, np.array([9.9, 101, 10]))
        for values in (result, audit.rounded(result)):
            self.assertTrue((np.diff(values[0]) > 0).all())
            self.assertEqual(values[1, 0], values[1, 1])
            self.assertGreater(values[1, 2], values[1, 1])
            self.assertTrue(np.isnan(values[2]).all())


if __name__ == "__main__":
    unittest.main()
