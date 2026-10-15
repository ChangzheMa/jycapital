"""紧凑数值数组保留浮点位模式、字符串内容和附件大小约束。"""

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from text_json import Float32Values, compact_numeric_json, write_compact_numeric_json


class TextJsonTest(unittest.TestCase):
    def test_float32_values_preserve_edge_cases_and_random_bit_patterns(self):
        import random
        bits = [0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
                0x7f7fffff, 0xff7fffff, 0x3f800001, 0x3f7fffff]
        rng = random.Random(42)
        bits.extend(rng.getrandbits(32) for _ in range(10_000))
        finite = [value for value in bits if value & 0x7f800000 != 0x7f800000]
        numbers = [struct.unpack(">f", struct.pack(">I", value))[0] for value in finite]
        value = {"weights": Float32Values(numbers), "statistics": [1.0000000000000002]}
        text = compact_numeric_json(value)
        restored = json.loads(text)
        self.assertEqual([struct.pack(">f", value) for value in restored["weights"]],
                         [struct.pack(">I", value) for value in finite])
        self.assertEqual(struct.pack(">d", restored["statistics"][0]),
                         struct.pack(">d", value["statistics"][0]))
        self.assertLess(len(text), len(compact_numeric_json({"weights": numbers, "statistics": value["statistics"]})))

    def test_float32_nonfinite_numbers_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                compact_numeric_json(Float32Values([value]))

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
