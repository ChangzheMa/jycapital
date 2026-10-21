"""v303 文本导出使用冻结源码，保持本地判定并恢复完整依赖闭包。"""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_v303", ROOT / "scripts/export-v303.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class ExportV303Test(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "outputs/runs/v303-test"
        self.run.mkdir(parents=True)
        write, digest = exporter.write_json, exporter.sha256
        config = exporter.read_json(ROOT / "solutions/v303/model-config.json")
        protocol = exporter.read_json(ROOT / "evaluation/protocols/local-v2-ic.json")
        write(self.run / "config.json", config)
        write(self.run / "protocol.json", protocol)
        torch.save({"config": config, "state_dict": {"weight": torch.tensor([0.125, -0.5])}},
                   self.run / "best.pt")
        self.selection = {"model_sha256": digest(self.run / "best.pt"), "best_epoch": 6}
        write(self.run / "selection.json", self.selection)
        training = self.root / "preprocessing"
        statistics = {}
        for frequency in exporter.FREQUENCIES:
            path = self.run / "normalization" / (frequency + ".json")
            write(path, {})
            statistics[frequency] = {"statistics_sha256": digest(path)}
        write(training / "summary.json", {"frequencies": statistics})
        write(training / "manifest.json", {"summary_sha256": digest(training / "summary.json")})
        write(self.run / "data-manifest.json", {"preprocessing": {"train": {
            "path": str(training), "manifest_sha256": digest(training / "manifest.json")}}})
        sources = list((ROOT / "solutions/v303").glob("*.py"))
        sources += list((ROOT / "solutions/v303").glob("*.json"))
        sources += list((ROOT / "solutions/v303").glob("*.txt"))
        sources += [ROOT / name for name in ("common/artifacts.py", "common/protocol_compatibility.py", "common/early_stopping.py",
        "common/resident_data.py", "common/evaluation_cache.py", "common/evaluation_upgrade.py", "common/training_resources.py",
                                             "evaluation/baseline_evaluation.py", "evaluation/protocols/local-v1.json")]
        files = {}
        for source in sources:
            relative = source.relative_to(ROOT)
            target = self.run / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            files[relative.as_posix()] = digest(target)
        self.manifest = {"status": "complete", "solution_id": "v303", "selection": self.selection,
                         "source": {"files": files, "git_commit": "frozen-source", "git_dirty": True}}
        write(self.run / "manifest.json", self.manifest)
        write(self.run / "comparison.json", {
            "candidate": False, "decision": "no_local_improvement", "criterion": {}, "metrics": {},
            "config_sha256": digest(self.run / "config.json"),
            "data_manifest_sha256": digest(self.run / "data-manifest.json")})
        patcher = patch.object(exporter, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_default_rejects_non_candidate_without_creating_package(self):
        with self.assertRaisesRegex(ValueError, "默认只导出"):
            exporter.export(self.run)
        self.assertFalse((self.root / "submissions/v303-test").exists())

    def test_candidate_exports_bit_exact_float32_text_with_signed_zero(self):
        comparison = exporter.read_json(self.run / "comparison.json")
        comparison.update(candidate=True, decision="local_improvement_candidate")
        exporter.write_json(self.run / "comparison.json", comparison)
        weights = torch.tensor([-0.0, 0.123456789, 1.401298464324817e-45,
                                3.4028234663852886e38], dtype=torch.float32)
        torch.save({"config": exporter.read_json(self.run / "config.json"),
                    "state_dict": {"weight": weights}}, self.run / "best.pt")
        self.manifest["selection"]["model_sha256"] = exporter.sha256(self.run / "best.pt")
        exporter.write_json(self.run / "manifest.json", self.manifest)
        package = exporter.export(self.run)
        bundle = exporter.read_json(package / "v303-model.json")
        notebook = exporter.read_json(package / "v303-submission.ipynb")
        description = "".join(notebook["cells"][0]["source"])
        self.assertIn("delta=2", description)
        self.assertEqual(bundle["config"]["training"]["huber_delta"], 2.0)
        restored = exporter.decode_state(bundle)
        self.assertEqual(exporter.state_digest(restored), exporter.state_digest({"weight": weights}))
        self.assertEqual(bundle["state_dict"]["weight"]["values"][1], 0.123456791)
        self.assertIn("-0.0", (package / "v303-model.json").read_text(encoding="utf-8"))
        report = exporter.read_json(exporter.manifest_path(package))
        self.assertEqual(report["weight_encoding"]["roundtrip"], "bit_exact")

    def test_authorized_export_uses_frozen_runtime_and_preserves_decision(self):
        comparison = (self.run / "comparison.json").read_bytes()
        current = self.root / "solutions/v303"
        current.mkdir(parents=True)
        (current / "v303_notebook.py").write_text("raise RuntimeError('changed adapter')\n")
        (current / "v303_schema.py").write_text("raise RuntimeError('changed model core')\n")
        package = exporter.export(self.run, platform_test_reason="用户要求平台比较")
        bundle = exporter.read_json(package / "v303-model.json")
        self.assertFalse(bundle["metadata"]["candidate"])
        self.assertEqual(bundle["metadata"]["decision"], "no_local_improvement")
        self.assertEqual((self.run / "comparison.json").read_bytes(), comparison)
        self.assertEqual({path.name for path in package.iterdir()}, {"v303-model.json", "v303-submission.ipynb"})
        notebook = exporter.read_json(package / "v303-submission.ipynb")
        namespace = {"__name__": "submission_import", "__file__": str(package / "v303-submission.ipynb")}
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                exec("".join(cell["source"]), namespace)
        _, modules = namespace["_runtime"]()
        self.assertEqual(set(modules), {"v303_schema", "v303_preprocessing", "v303_data",
                                       "v303_model", "v303_text_artifact", "v303_predict"})
        restored = exporter.decode_state(bundle)
        self.assertTrue(torch.equal(restored["weight"], torch.tensor([0.125, -0.5])))
        self.assertEqual(restored["weight"].dtype, torch.float32)
        with self.assertRaisesRegex(ValueError, "明确确认"):
            namespace["reproduce"]("unused", "unused")

    def test_fast_variant_preserves_frozen_model_and_records_runtime_override(self):
        current = self.root / "solutions/v303"
        current.mkdir(parents=True)
        for name in ("v303_fast.py", "v303_predict_fast.py"):
            shutil.copyfile(ROOT / "solutions/v303" / name, current / name)
        (current / "v303_model.py").write_text("raise RuntimeError('must use frozen model')\n")
        package = exporter.export(self.run, platform_test_reason="用户要求推理加速", fast_inference=True)
        bundle = exporter.read_json(package / "v303-model.json")
        self.assertEqual(package.name, "v303-test-fast")
        self.assertEqual(bundle["config"], exporter.read_json(self.run / "config.json"))
        self.assertEqual(bundle["sources"]["solutions/v303/v303_model.py"]["sha256"],
                         self.manifest["source"]["files"]["solutions/v303/v303_model.py"])
        self.assertEqual(bundle["inference_execution"]["method"], "shared_stock_day_encoding")
        self.assertNotIn("solutions/v303/v303_fast.py", bundle["training_sources"])
        notebook = exporter.read_json(package / "v303-submission.ipynb")
        namespace = {"__name__": "submission_import", "__file__": str(package / "v303-submission.ipynb")}
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                exec("".join(cell["source"]), namespace)
        _, modules = namespace["_runtime"]()
        self.assertIn("v303_fast", modules)
        self.assertIs(modules["v303_predict"].predict_day, modules["v303_fast"].predict_day)
        report = exporter.read_json(exporter.manifest_path(package))
        self.assertEqual(report["runtime_overrides"]["solutions/v303/v303_predict.py"]["source_path"],
                         "solutions/v303/v303_predict_fast.py")

    def test_training_restore_can_import_huber_trainer_without_repository(self):
        package = exporter.export(self.run, platform_test_reason="用户要求平台比较")
        bundle = exporter.read_json(package / "v303-model.json")
        spec = importlib.util.spec_from_file_location("v303_reproduce_check", ROOT / "solutions/v303/v303_reproduce.py")
        reproduction = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reproduction)
        project = reproduction.restore_sources(bundle, self.root / "reproduced")
        code = ("import torch,sys,json; from pathlib import Path; "
                "sys.path.insert(0,str(Path('solutions/v303').resolve())); "
                "from v303_prepare import validate_config; "
                "validate_config(json.loads(Path('solutions/v303/config.json').read_text(encoding='utf-8'))); "
                "json.loads(Path('evaluation/protocols/local-v1.json').read_text(encoding='utf-8')); "
                "from v303_train import validate_training; "
                "validate_training(json.loads(Path('solutions/v303/model-config.json').read_text(encoding='utf-8')),"
                "json.loads(Path('evaluation/protocols/local-v2-ic.json').read_text(encoding='utf-8'))); "
                "print('Huber trainer import passed')")
        result = subprocess.run([sys.executable, "-c", code], cwd=project,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Huber trainer import passed", result.stdout)

    def test_changed_selected_weights_rejected(self):
        (self.run / "best.pt").write_bytes(b"changed weights")
        with self.assertRaisesRegex(ValueError, "所选权重已改变"):
            exporter.export(self.run, platform_test_reason="用户要求平台比较")

    def test_attachment_limit_rejects_export_before_creating_package(self):
        with patch.object(exporter, "MAX_ATTACHMENT_BYTES", 10):
            with self.assertRaisesRegex(ValueError, "超过"):
                exporter.export(self.run, platform_test_reason="用户要求平台比较")
        self.assertFalse((self.root / "submissions/v303-test").exists())

    def test_changed_frozen_loss_dependency_rejected(self):
        (self.run / "source/solutions/v303/v303_loss.py").write_text("changed loss\n")
        with self.assertRaisesRegex(ValueError, "冻结源码已改变"):
            exporter.export(self.run, platform_test_reason="用户要求平台比较")

    def test_running_or_blank_authorization_rejected(self):
        with self.assertRaisesRegex(ValueError, "须注明"):
            exporter.export(self.run, platform_test_reason="  ")
        self.manifest["status"] = "running"
        exporter.write_json(self.run / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "只能导出已完成"):
            exporter.export(self.run, platform_test_reason="用户要求平台比较")


if __name__ == "__main__":
    unittest.main()
