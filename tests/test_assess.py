"""候选晋级必须看验证主指标并保持完整截面覆盖。"""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

# 与模型测试同进程运行时，Windows CUDA DLL 须先于 NumPy/OpenMP 初始化。
try:
    import torch
except ImportError:
    pass
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("experiment_assess", ROOT / "scripts/assess.py")
assessor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assessor)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class AssessmentTest(unittest.TestCase):
    def fixture(self, root, validation, test, coverage=1.0):
        experiments = root / "experiments"
        experiments.mkdir()
        run_paths = []
        for name, val, tst, cov in (("parent", 0.06, 0.07, 1.0), ("new", validation, test, coverage)):
            run = root / "outputs/runs" / name
            run_paths.append(run)
            selection = {"best_epoch": 1}
            write(run / "manifest.json", {"status": "complete", "solution_id": "v2", "parent_run_id": "parent",
                  "source": {"files": {"evaluation/baseline_evaluation.py": "same"}}, "selection": selection})
            write(run / "protocol.json", {"protocol_id": "test", "selection_metric": "rank_ic_mean", "universe_table": "pool", "label_table": "labels"})
            write(run / "config.json", {"seed": 42})
            write(run / "data-manifest.json", {"mode": "local", "files": [{"path": "pool.parquet", "sha256": "pool"}, {"path": "labels.parquet", "sha256": "labels"}]})
            write(run / "environment.json", {"parameters": 1, "device_name": "CPU"})
            write(run / "selection.json", selection)
            pd.DataFrame({"seconds": [1.0]}).to_csv(run / "history.csv", index=False)
            for split, metric in (("validation", val), ("test", tst)):
                directory = run / "evaluation" / split
                metrics = dict.fromkeys(assessor.METRICS, 0.0)
                metrics.update(rank_ic_mean=metric, coverage=cov, min_cross_section_coverage=cov, coverage_violations=0, evaluated_rows=100)
                write(directory / "metrics.json", metrics)
                pd.DataFrame({"month": ["2024-01"], "ic_mean": [metric], "rank_ic_mean": [metric]}).to_csv(directory / "monthly.csv", index=False)
                pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "expected": [100], "coverage": [cov]}).to_parquet(directory / "cross-sections.parquet", index=False)
        pd.DataFrame({"run_id": ["parent"], "run_path": ["outputs/runs/parent"]}).to_csv(experiments / "runs.csv", index=False)
        pd.DataFrame({"platform_score": [None]}).to_csv(experiments / "submissions.csv", index=False)
        return run_paths[-1], experiments

    def check_case(self, validation, test, coverage, expected):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            run, experiments = self.fixture(root, validation, test, coverage)
            with patch.object(assessor, "ROOT", root):
                result = assessor.assess(run, experiments)
            self.assertEqual(result["candidate"], expected)
            self.assertEqual(result["references"][0]["roles"], ["local_best", "parent"])
            self.assertIsNone(result["platform_best"])

    def test_validation_improvement_qualifies_and_reports_test_decline(self):
        self.check_case(0.060001, 0.05, 1.0, True)

    def test_test_only_improvement_does_not_qualify(self):
        self.check_case(0.059, 0.09, 1.0, False)

    def test_coverage_decline_does_not_qualify(self):
        self.check_case(0.08, 0.09, 0.99, False)

    def test_equal_validation_does_not_qualify(self):
        self.check_case(0.06, 0.07, 1.0, False)

    def test_record_preserves_existing_metric_precision_and_column_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            run, experiments = self.fixture(root, 0.08, 0.09)
            header = "run_id,solution_id,protocol_id,seed,best_epoch,validation_ic,validation_rank_ic,test_ic,test_rank_ic,test_coverage,model_sha256,run_path"
            original = "parent,v1,test,42,8,0.07044462729051033,0.06133269188586244,0.08354889956132129,0.06831586201868786,1.0,parent-sha,outputs/runs/parent"
            index = experiments / "runs.csv"
            index.write_text(header + "\n" + original + "\n", encoding="utf-8-sig")
            with patch.object(assessor, "ROOT", root):
                result = assessor.assess(run, experiments)
                result["selection"]["model_sha256"] = "new-sha"
                assessor.record(result)
            lines = index.read_text(encoding="utf-8-sig").splitlines()
            self.assertEqual(lines[:2], [header, original])
            self.assertEqual(len(lines), 3)
            self.assertTrue(index.read_bytes().startswith(b"\xef\xbb\xbf"))


if __name__ == "__main__":
    unittest.main()
