"""紧凑数值数组保留浮点位模式、字符串内容和附件大小约束。"""

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from text_json import compact_numeric_json, write_compact_numeric_json


class TextJsonTest(unittest.TestCase):
    def test_numeric_bits_and_source_strings_survive_roundtrip(self):
        numbers = [0.0, -0.0, 5e-324, -5e-324, 1.7976931348623157e308, 1.0000000000000002]
        original = {"nested": {"values": numbers, "shape": [2, 3]},
                    "source": "中文  两个空格\n\tjson='[  1, 2 ]'", "names": ["alpha", "beta"]}
        text = compact_numeric_json(original)
        restored = json.loads(text)
        self.assertEqual(restored, original)
        for before, after in zip(numbers, restored["nested"]["values"]):
            self.assertEqual(struct.pack(">d", before), struct.pack(">d", after))
        self.assertIn('  "nested": {\n    "values": [', text)
        self.assertLess(len(text.encode("utf-8")),
                        len(json.dumps(original, ensure_ascii=False, indent=2).encode("utf-8")))

    def test_nonfinite_numbers_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                compact_numeric_json({"values": [value]})

    def test_size_limit_fails_before_replacing_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_bytes(b"original")
            with self.assertRaisesRegex(ValueError, "超过"):
                write_compact_numeric_json(path, {"values": [0.123456789]}, max_bytes=5)
            self.assertEqual(path.read_bytes(), b"original")
            size = write_compact_numeric_json(path, {"values": [0.123456789]}, max_bytes=100)
            self.assertEqual(size, path.stat().st_size)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"values": [0.123456789]})


if __name__ == "__main__":
    unittest.main()
