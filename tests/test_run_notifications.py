"""通知目标、长度、接受状态和资源关闭，不把接口接受当作空闲恢复。"""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import run_notifications


class FakeRpc:
    instances = []
    error = False

    def __init__(self, binary, thread_id):
        self.thread_id, self.closed, self.calls = thread_id, False, []
        self.instances.append(self)

    def call(self, method, params):
        self.calls.append((method, params))
        return {"isError": self.error, "content": []}

    def close(self):
        self.closed = True


class NotificationTests(unittest.TestCase):
    def notice(self):
        return {"kind": "complete", "run_id": "v4-test", "status": "complete",
                "at": "2026-10-14T00:00:00+08:00", "state_file": "outputs/runs/v4-test/monitor-state.json",
                "event_id": "complete:test"}

    def test_existing_target_and_accepted_state_only(self):
        FakeRpc.error = False
        with patch.object(run_notifications, "RpcProxy", FakeRpc):
            result = run_notifications.send_notice("codex", "existing-thread", self.notice())
        instance = FakeRpc.instances[-1]
        self.assertTrue(instance.closed)
        method, params = instance.calls[0]
        self.assertEqual(method, "mcpServer/tool/call")
        self.assertEqual(params["threadId"], "existing-thread")
        self.assertEqual(params["arguments"]["threadId"], "existing-thread")
        self.assertEqual(params["tool"], "send_message_to_thread")
        self.assertLessEqual(len(params["arguments"]["prompt"].encode("utf-8")), 1000)
        self.assertIn("accepted_at", result)
        self.assertNotIn("automatic_wake_verified", result)

    def test_rejected_notice_closes_client(self):
        FakeRpc.error = True
        with patch.object(run_notifications, "RpcProxy", FakeRpc), self.assertRaises(RuntimeError):
            run_notifications.send_notice("codex", "existing-thread", self.notice())
        self.assertTrue(FakeRpc.instances[-1].closed)
        FakeRpc.error = False


if __name__ == "__main__":
    unittest.main()
