"""文本权重保留位级精度，训练源码恢复不运行训练。"""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solutions/v2"))
from v2_text_artifact import FORMAT, encode_state, decode_state, state_digest
from v2_reproduce import restore_sources


class TextArtifactTest(unittest.TestCase):
    def test_json_roundtrip_preserves_float_bits_integers_and_shapes(self):
        state = {
            "float": torch.tensor([0.0, -0.0, 1e-40, -1e-30, 3.4028234e38], dtype=torch.float32),
            "double": torch.tensor([[1.0000000000000002, 1e-300]], dtype=torch.float64),
            "integer": torch.tensor([2 ** 60 + 1, -2 ** 60], dtype=torch.int64),
            "scalar": torch.tensor(True),
        }
        bundle = {"format": FORMAT, "state_dict": encode_state(state), "tensor_sha256": state_digest(state)}
        restored = decode_state(json.loads(json.dumps(bundle, allow_nan=False)))
        for name in state:
            self.assertEqual(state[name].dtype, restored[name].dtype)
            self.assertEqual(state[name].shape, restored[name].shape)
            self.assertEqual(state[name].numpy().tobytes(), restored[name].numpy().tobytes())

    def test_changed_text_weight_is_rejected(self):
        state = {"weight": torch.tensor([0.25])}
        bundle = {"format": FORMAT, "state_dict": encode_state(state), "tensor_sha256": state_digest(state)}
        bundle["state_dict"]["weight"]["values"][0] = 0.5
        with self.assertRaisesRegex(ValueError, "哈希"):
            decode_state(bundle)

    def test_non_finite_weights_are_not_exported(self):
        with self.assertRaises(ValueError):
            encode_state({"weight": torch.tensor([float("nan")])})

    def test_training_source_bytes_restore_without_execution(self):
        text = 'raise RuntimeError("must not run while restoring")\r\n'
        relative = "solutions/v2/train.py"
        bundle = {"training_sources": [relative], "sources": {relative: {
            "text": text, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}}}
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "new"
            project = restore_sources(bundle, workspace)
            self.assertEqual((project / relative).read_bytes(), text.encode("utf-8"))
            with self.assertRaises(FileExistsError):
                restore_sources(bundle, workspace)

    def test_restore_rejects_path_outside_workspace(self):
        relative = "../../outside.py"
        bundle = {"training_sources": [relative], "sources": {relative: {"text": "", "sha256": ""}}}
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                restore_sources(bundle, Path(temporary) / "new")


if __name__ == "__main__":
    unittest.main()
