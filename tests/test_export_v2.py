"""平台测试导出保留本地判定，且不能绕过冻结材料校验。"""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_v2", ROOT / "scripts/export-v2.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class ExportV2Test(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "outputs/runs/v2-test"
        self.run.mkdir(parents=True)
        self.destination = self.root / "submissions/v2-test"
        write, digest = exporter.write_json, exporter.sha256
        for name in ("config.json", "protocol.json"):
            write(self.run / name, {})
        torch.save({"config": {}, "state_dict": {"weight": torch.tensor([0.125, -0.5])}}, self.run / "best.pt")
        selection = {"model_sha256": digest(self.run / "best.pt"), "best_epoch": 4}
        write(self.run / "selection.json", selection)
        training = self.root / "preprocessing"
        statistics = {}
        for frequency in ("bar1m", "bar5m", "bar15m", "bar30m"):
            path = self.run / "normalization" / (frequency + ".json")
            write(path, {})
            statistics[frequency] = {"statistics_sha256": digest(path)}
        write(training / "summary.json", {"frequencies": statistics})
        write(training / "manifest.json", {"summary_sha256": digest(training / "summary.json")})
        write(self.run / "data-manifest.json", {"preprocessing": {"train": {
            "path": str(training), "manifest_sha256": digest(training / "manifest.json")}}})
        core = ["v2_preprocessing.py", "v2_prepare.py", "v2_apply.py", "v2_data.py", "v2_model.py", "v2_train.py",
                "config.json", "requirements.txt", "requirements-training.txt"]
        files = {}
        for relative in ["solutions/v2/" + name for name in core] + ["common/artifacts.py", "evaluation/baseline_evaluation.py"]:
            write(self.run / "source" / relative, {})
            write(self.root / relative, {})
            files[relative] = digest(self.run / "source" / relative)
        for name in ("v2_predict.py", "v2_reproduce.py", "v2_text_artifact.py", "requirements-inference.txt"):
            write(self.root / "solutions/v2" / name, {})
        (self.root / "solutions/v2/v2_notebook.py").write_bytes((ROOT / "solutions/v2/v2_notebook.py").read_bytes())
        self.manifest = {"status": "complete", "solution_id": "v2", "selection": selection,
                         "source": {"files": files, "git_commit": "source-commit", "git_dirty": True}}
        write(self.run / "manifest.json", self.manifest)
        self.comparison = {"candidate": False, "decision": "no_local_improvement", "criterion": {}, "metrics": {},
                           "config_sha256": digest(self.run / "config.json"),
                           "data_manifest_sha256": digest(self.run / "data-manifest.json")}
        write(self.run / "comparison.json", self.comparison)
        patcher = patch.object(exporter, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_default_rejects_non_candidate_without_creating_package(self):
        with self.assertRaisesRegex(ValueError, "默认只导出"):
            exporter.export(self.run)
        self.assertFalse(self.destination.exists())

    def test_explicit_platform_test_keeps_assessment_and_records_reason(self):
        original = (self.run / "comparison.json").read_bytes()
        reason = "用户要求验证平台运行及评分"
        package = exporter.export(self.run, platform_test_reason=reason)
        manifest = exporter.read_json(exporter.manifest_path(package))
        self.assertEqual(manifest["export_purpose"], "user_requested_platform_test")
        self.assertEqual(manifest["export_reason"], reason)
        self.assertFalse(manifest["local_candidate"])
        self.assertEqual((self.run / "comparison.json").read_bytes(), original)
        self.assertFalse(exporter.read_json(package / "v2-model.json")["metadata"]["candidate"])
        self.assertEqual({p.name for p in package.iterdir()}, {"v2-submission.ipynb", "v2-model.json"})
        self.assertIn("未达到本地改善条件", (package / "v2-submission.ipynb").read_text(encoding="utf-8"))
        for name, digest in manifest["files"].items():
            self.assertEqual(exporter.sha256(package / name), digest)

    def test_platform_test_still_rejects_changed_weights(self):
        (self.run / "best.pt").write_bytes(b"changed weights")
        with self.assertRaisesRegex(ValueError, "所选权重已改变"):
            exporter.export(self.run, platform_test_reason="用户要求平台测试")
        self.assertFalse(self.destination.exists())

    def test_platform_test_still_requires_completed_run(self):
        self.manifest["status"] = "running"
        exporter.write_json(self.run / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "只能导出已完成"):
            exporter.export(self.run, platform_test_reason="用户要求平台测试")

    def test_blank_reason_is_not_authorization(self):
        with self.assertRaisesRegex(ValueError, "须注明"):
            exporter.export(self.run, platform_test_reason="  ")


if __name__ == "__main__":
    unittest.main()
