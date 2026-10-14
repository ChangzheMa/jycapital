"""只允许选模字段差异，不能放宽数据、日期和评估算法。"""

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from protocol_compatibility import validate_protocol_compatibility


class ProtocolTests(unittest.TestCase):
    def test_strict_default_and_selection_only_allowance(self):
        old = json.loads((ROOT / "evaluation/protocols/local-v1.json").read_text(encoding="utf-8"))
        new = json.loads((ROOT / "evaluation/protocols/local-v2-ic.json").read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            validate_protocol_compatibility(old, new)
        differences = validate_protocol_compatibility(old, new, allow_selection_metric_change=True)
        self.assertEqual(set(differences), {"protocol_id", "selection_metric"})
        for key in ("splits", "label_table", "universe_table", "aggregation", "min_cross_section_size"):
            changed = copy.deepcopy(new)
            changed[key] = "changed"
            with self.assertRaises(ValueError):
                validate_protocol_compatibility(old, changed, allow_selection_metric_change=True)
        changed = copy.deepcopy(new); changed["new_contract_field"] = None
        with self.assertRaises(ValueError):
            validate_protocol_compatibility(old, changed, allow_selection_metric_change=True)


if __name__ == "__main__":
    unittest.main()
