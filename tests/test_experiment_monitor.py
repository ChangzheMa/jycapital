"""正常批次不唤醒，小时、结束与异常通知去重。"""

import importlib.util
import os
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_monitor", ROOT / ".agents/skills/jycapital-experiment/scripts/monitor-run.py")
MONITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MONITOR)


class MonitorTest(unittest.TestCase):
    def test_normal_progress_does_not_wake_until_hour(self):
        policy = MONITOR.NoticePolicy(0)
        for minute in range(60):
            self.assertIsNone(policy.check(minute * 60, "running", True, minute,
                                           [{"event": "trained_month", "at": str(minute)}]))
        self.assertEqual(policy.check(3600, "running", True, 60, []), "hourly")
        self.assertIsNone(policy.check(3610, "running", True, 61, []))

    def test_phase_end_deduplicates_and_resets_hour(self):
        policy = MONITOR.NoticePolicy(0)
        event = {"event": "model_frozen", "at": "frozen"}
        self.assertEqual(policy.check(100, "running", True, 1, [event]), "phase_finished")
        self.assertIsNone(policy.check(200, "running", True, 1, [event]))
        self.assertIsNone(policy.check(3600, "running", True, 1, []))
        self.assertEqual(policy.check(3700, "running", True, 1, []), "hourly")

    def test_terminal_exit_and_stall(self):
        policy = MONITOR.NoticePolicy(0, stall_seconds=900)
        self.assertIsNone(policy.check(0, "running", True, 1, []))
        self.assertEqual(policy.check(901, "running", True, 1, []), "possibly_stalled")
        self.assertIsNone(policy.check(902, "running", True, 1, []))
        self.assertEqual(policy.check(903, "running", False, 1, []), "process_exited")
        self.assertEqual(policy.check(904, "complete", False, 2, []), "complete")
        self.assertEqual(policy.check(905, "failed", True, 2, []), "error")

    def test_validation_prediction_inside_training_is_normal_progress(self):
        policy = MONITOR.NoticePolicy(0)
        events = [{"event": "training_started", "at": "start"},
                  {"event": "prediction_completed", "at": "validation", "split": "validation"}]
        self.assertIsNone(policy.check(100, "running", True, 1, events))
        self.assertEqual(policy.check(200, "running", True, 2, [{"event": "model_frozen", "at": "freeze"}]), "phase_finished")
        self.assertEqual(policy.check(300, "running", True, 3, [{"event": "prediction_completed", "at": "test"}]), "phase_finished")

    def test_process_watch_does_not_signal_running_process(self):
        watch = MONITOR.ProcessWatch(os.getpid())
        try:
            self.assertEqual(watch.status(), (True, None))
        finally:
            watch.close()


if __name__ == "__main__":
    unittest.main()
