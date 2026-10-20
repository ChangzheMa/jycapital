"""候选发布门槛、重复运行保护和容量阈值。"""

import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from research_queue import atomic_json, completion, publish, safe_path, snapshot, source_files


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        atomic_json(self.root / "experiments/research-policy.json", {
            "legacy_runs": [], "construction_pause_when_unevaluated_gt": 3,
            "construction_resume_when_unevaluated_lt": 3})
        atomic_json(self.root / "evaluation/protocols/local-v2-ic.json", {})

    def candidate(self, version, priority=10):
        directory = self.root / "solutions" / version
        directory.mkdir(parents=True)
        (directory / (version + "_train.py")).write_text("# test\n", encoding="utf-8")
        atomic_json(directory / "model-config.json", {})
        atomic_json(directory / "experiment.json", {
            "solution_id": version, "status": "draft", "run_id": version + "-s42", "formal_attempts": 1,
            "priority": priority, "entrypoint": f"solutions/{version}/{version}_train.py",
            "config": f"solutions/{version}/model-config.json"})
        return directory

    def ready(self, version):
        report = f"outputs/checks/{version}.json"
        atomic_json(self.root / report, {"solution_id": version, "status": "passed",
                                       "source_files": source_files(self.root, version)})
        publish(self.root, version, report)

    def test_ready_fingerprint_and_existing_run(self):
        directory = self.candidate("v202")
        self.assertIsNone(snapshot(self.root)["next"])
        self.ready("v202")
        self.assertEqual(snapshot(self.root)["next"]["solution_id"], "v202")
        (directory / "v202_train.py").write_text("# changed\n", encoding="utf-8")
        self.assertIsNone(snapshot(self.root)["next"])
        self.ready("v202")
        atomic_json(self.root / "outputs/runs/v202-s42/manifest.json", {"status": "running", "stage": "training"})
        self.assertIsNone(snapshot(self.root)["next"])
        with self.assertRaises(ValueError):
            self.ready("v202")

    def test_capacity_hysteresis_and_incomplete_completion(self):
        for version in ("v102", "v202", "v302"):
            self.candidate(version)
        result = snapshot(self.root)
        self.assertEqual(result["unevaluated_count"], 3)
        self.assertFalse(result["pause_construction"])
        self.assertFalse(result["resume_construction"])
        self.candidate("v401")
        self.assertTrue(snapshot(self.root)["pause_construction"])
        atomic_json(self.root / "outputs/runs/v102-s42/manifest.json", {"status": "complete", "stage": "complete"})
        self.assertEqual(completion(self.root, "v102-s42")[0], "needs_attention")
        self.assertEqual(snapshot(self.root)["unevaluated_count"], 4)

    def test_priority_retired_and_path_escape(self):
        for version, priority in (("v102", 30), ("v202", 10)):
            self.candidate(version, priority)
            self.ready(version)
        self.assertEqual(snapshot(self.root)["next"]["solution_id"], "v202")
        path = self.root / "solutions/v202/experiment.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        spec.update(status="retired", retirement_reason="明确撤销")
        atomic_json(path, spec)
        result = snapshot(self.root)
        self.assertTrue(result["resume_construction"])
        self.assertEqual(result["next"]["solution_id"], "v102")
        with self.assertRaises(ValueError):
            safe_path(self.root, "../outside")

    def test_capacity_remembers_a_peak_between_hourly_checks(self):
        for version in ("v102", "v202", "v302", "v303"):
            self.candidate(version)
        self.assertTrue(snapshot(self.root)["pause_construction"])
        for version, expected_count, expected_paused in (("v303", 3, True), ("v302", 2, False)):
            path = self.root / "solutions" / version / "experiment.json"
            spec = json.loads(path.read_text(encoding="utf-8"))
            spec.update(status="retired", retirement_reason="测试中明确撤销")
            atomic_json(path, spec)
            # 每次调用重新读取文件，不能依赖同一 Python 对象保存暂停状态。
            result = snapshot(self.root)
            self.assertEqual(result["unevaluated_count"], expected_count)
            self.assertEqual(result["pause_construction"], expected_paused)

    def test_only_imported_common_dependencies_invalidate_readiness(self):
        directory = self.candidate("v202")
        common = self.root / "common"
        common.mkdir()
        (common / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
        (directory / "v202_train.py").write_text("from helper import VALUE\n", encoding="utf-8")
        self.ready("v202")
        (common / "unrelated.py").write_text("VALUE = 2\n", encoding="utf-8")
        self.assertIsNotNone(snapshot(self.root)["next"])
        (common / "helper.py").write_text("VALUE = 3\n", encoding="utf-8")
        self.assertIsNone(snapshot(self.root)["next"])

    def test_hourly_watcher_hysteresis_and_new_completion(self):
        spec = importlib.util.spec_from_file_location("capacity_watcher", ROOT / "scripts/watch-research-capacity.py")
        watcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(watcher)

        def state(count, complete=False):
            return {"pause_construction": count > 3, "resume_construction": count < 3,
                    "candidates": [{"run_id": "v301-s42", "run_status": "complete" if complete else "running"}]}

        first = watcher.observe({}, state(4))
        self.assertTrue(first["construction_paused"])
        second = watcher.observe(first, state(3))
        self.assertTrue(second["construction_paused"])
        self.assertFalse(second["notify"])
        third = watcher.observe(second, state(3, complete=True))
        self.assertTrue(third["construction_paused"])
        self.assertTrue(third["notify"])
        fourth = watcher.observe(third, state(2, complete=True))
        self.assertFalse(fourth["construction_paused"])
        self.assertTrue(fourth["capacity_recovered"])


if __name__ == "__main__":
    unittest.main()
