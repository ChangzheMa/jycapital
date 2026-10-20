"""排名监督的语义、原标签评估及共享缓存复用；仅 CPU 合成数据。"""

import copy
import importlib
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from test_continuous_candidates import CONFIGS, MODULES, base
from resident_data import cache_key, CLOCK_MINUTES, MINUTE_NS

loss = importlib.import_module("v401_loss")
target_module = importlib.import_module("v401_targets")


class RankTargetTests(unittest.TestCase):
    def test_ties_invalid_labels_monotone_invariance_and_raw_ic(self):
        labels = torch.tensor([2., 2., -1., 9., float("nan"), float("inf")], dtype=torch.float64)
        targets, reason = loss.rank_standardized_targets(labels, min_count=3)
        self.assertIsNone(reason)
        expected = torch.tensor([0., 0., -math.sqrt(2), math.sqrt(2)], dtype=torch.float64)
        torch.testing.assert_close(targets[:4], expected, rtol=0, atol=1e-15)
        self.assertTrue(torch.isnan(targets[4:]).all())
        transformed, _ = loss.rank_standardized_targets(torch.exp(labels[:4]), min_count=3)
        torch.testing.assert_close(targets[:4], transformed, rtol=0, atol=0)
        self.assertEqual(loss.rank_standardized_targets(labels, min_count=5)[1], "insufficient_labels")
        self.assertEqual(loss.rank_standardized_targets(torch.ones(4), min_count=3)[1], "constant_labels")
        score = targets[:4].clone().requires_grad_(True)
        result = loss.rank_standardized_huber_loss(score, labels[:4], min_count=3)
        self.assertEqual(result.loss.item(), 0)
        original_ic = loss.pearson_ic_loss(score.detach(), labels[:4], min_count=3).ic
        torch.testing.assert_close(result.ic, original_ic, rtol=0, atol=0)
        self.assertLess(result.ic.item(), 0.99)
        result.loss.backward()
        self.assertTrue(torch.isfinite(score.grad).all())
        constant = loss.rank_standardized_huber_loss(torch.zeros(4), labels[:4], min_count=3)
        self.assertIsNone(constant.ic)
        self.assertTrue(torch.isfinite(constant.loss))

    def test_model_control_and_base_cache_identity(self):
        for key in ("data", "model", "inference", "seed"):
            self.assertEqual(CONFIGS["v401"][key], base.CONFIGS["v301"][key])
        a, b = copy.deepcopy(CONFIGS["v401"]["training"]), copy.deepcopy(base.CONFIGS["v301"]["training"])
        for key in ("loss", "target_transform"):
            a.pop(key)
            b.pop(key)
        self.assertEqual(a, b)
        keys = []
        for version in ("v301", "v401"):
            store = SimpleNamespace(config=base.CONFIGS[version], protocol=base.PROTOCOL)
            keys.append(cache_key(store, {"synthetic": True}, base.MODULES[version]["train"].prepare_training_targets))
        self.assertEqual(*keys)

    def test_overlay_alignment_boundary_resume_and_shared_array_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = copy.deepcopy(base.PROTOCOL)
            protocol["min_cross_section_size"] = 3
            protocol["splits"]["train"] = {"start": "2020-01-02", "stop": "2020-01-02 10:10"}
            keys = pd.DataFrame(
                [(pd.Timestamp("2020-01-02") + pd.Timedelta(minutes=m), s)
                 for m in (570, 600) for s in ("A", "B", "C", "D")],
                columns=["date", "instrument"],
            )
            labels = keys.copy()
            labels[protocol["label_column"]] = [2., 2., -1., 9.] * 2
            label_path = root / (protocol["label_table"] + ".parquet")
            labels.iloc[::-1].to_parquet(label_path, index=False)
            eligible_keys = labels.copy()
            eligible_keys.loc[eligible_keys.date.dt.hour.eq(10), protocol["label_column"]] = np.nan
            baseline, skipped = MODULES["v401"]["train"].prepare_training_targets(eligible_keys, protocol)
            original_baseline = baseline.copy()
            block = SimpleNamespace(
                arrays={"days": np.array([pd.Timestamp("2020-01-02").value]), "targets": baseline},
                day=np.zeros(8, dtype=np.int32), stock=np.tile(np.arange(4), 2),
                clock=np.repeat(np.arange(2), 4),
                manifest={"stocks": ["A", "B", "C", "D"], "cache_key": "synthetic"},
                targets=baseline,
            )
            block.training_targets = lambda: (block.targets, skipped)
            store = SimpleNamespace(block=block, original=SimpleNamespace(
                universe=keys.iloc[::-1], protocol=protocol, data_root=root,
            ))
            (root / "data-manifest.json").write_text("{}\n", encoding="utf-8")
            record = target_module.attach_rank_targets(store, root)
            self.assertEqual(record["eligible_rows"], 4)
            np.testing.assert_array_equal(block.arrays["targets"], original_baseline)
            self.assertIsNot(block.targets, block.arrays["targets"])
            expected, _ = target_module.prepare_rank_training_targets(eligible_keys, protocol)
            np.testing.assert_array_equal(block.targets, expected)
            self.assertEqual(target_module.attach_rank_targets(store, root), record)
            block.stock = block.stock[::-1]
            with self.assertRaisesRegex(ValueError, "行顺序"):
                target_module.attach_rank_targets(store, root)
            block.stock = block.stock[::-1]
            labels.loc[0, protocol["label_column"]] = 100.
            labels.to_parquet(label_path, index=False)
            with self.assertRaisesRegex(ValueError, "断点恢复"):
                target_module.attach_rank_targets(store, root)


if __name__ == "__main__":
    unittest.main()
