"""独立复现的父版文本权重载荷；不启动正式训练。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_v1901 import ROOT, small_config, synthetic_initialization
from v1901_reproduce import reproduce
from v1901_text_artifact import FORMAT, encode_state, state_digest


class ReproductionPayloadTests(unittest.TestCase):
    def test_separate_initialization_text_restores_exact_weights_and_training_argument(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = synthetic_initialization(config, root / "source.pt")
            saved = torch.load(source, weights_only=True)
            text = {"format": FORMAT, "state_dict": encode_state(saved["state_dict"]),
                    **{key: saved[key] for key in ("parent_config", "parent_run_id", "parent_model_sha256")}}
            text = json.loads(json.dumps(text, allow_nan=False))
            relative = "solutions/v1901/v1901_text_artifact.py"
            bundle = {"config": config, "training_sources": [relative],
                      "sources": {relative: {"text": (ROOT / relative).read_text(encoding="utf-8")}}}
            data, workspace = root / "data", root / "reproduced"
            data.mkdir()
            with patch("v1901_reproduce.subprocess.run") as execute:
                run = reproduce(bundle, workspace, data, training_confirmed=True, initialization_bundle=text)
            self.assertEqual(run, (workspace / "run").resolve())
            self.assertEqual(execute.call_count, 3)
            arguments = execute.call_args_list[-1].args[0]
            self.assertIn("--initialization-state", arguments)
            payload = torch.load(Path(arguments[arguments.index("--initialization-state") + 1]), weights_only=True)
            self.assertEqual(state_digest(payload["state_dict"]), config["initialization"]["state_sha256"])
            self.assertEqual(payload["parent_config"]["data"], config["data"])
            for key, value in saved["state_dict"].items():
                torch.testing.assert_close(value, payload["state_dict"][key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
