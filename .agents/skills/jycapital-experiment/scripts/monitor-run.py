"""静默监控已有训练／预测进程，仅在整点间隔、阶段结束和异常时通知。"""

import argparse
import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import time


class ProcessWatch:
    def __init__(self, pid):
        self.pid, self.handle = pid, None
        if os.name == "nt":
            from ctypes import wintypes
            self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            self.kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            self.kernel.OpenProcess.restype = wintypes.HANDLE
            self.kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
            self.kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            self.handle = self.kernel.OpenProcess(0x1000, False, pid)
            if not self.handle:
                raise ProcessLookupError(pid)

    def status(self):
        if os.name == "nt":
            if not self.handle:
                raise RuntimeError("进程监控句柄已关闭")
            code = ctypes.c_ulong()
            if not self.kernel.GetExitCodeProcess(self.handle, ctypes.byref(code)):
                return False, None
            return code.value == 259, None if code.value == 259 else code.value
        try:
            os.kill(self.pid, 0)
            return True, None
        except ProcessLookupError:
            return False, None
        except PermissionError:
            return True, None

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class NoticePolicy:
    terminal = {"complete", "failed", "interrupted", "cancelled"}
    phases = {"model_frozen", "training_finished", "training_completed", "prediction_finished", "prediction_completed"}

    def __init__(self, now, report_seconds=3600, stall_seconds=None):
        self.last_notice = self.last_progress = now
        self.report_seconds, self.stall_seconds = report_seconds, stall_seconds
        self.progress_stamp, self.stall_sent, self.phase_seen = None, False, set()
        self.training_active = False

    def check(self, now, status, alive, stamp, events):
        if stamp != self.progress_stamp:
            self.progress_stamp, self.last_progress, self.stall_sent = stamp, now, False
        if status in self.terminal:
            return "complete" if status == "complete" else "error"
        if not alive:
            return "process_exited"
        ended = []
        for event in events:
            name = event.get("event")
            if name == "training_started":
                self.training_active = True
            if name in {"model_frozen", "training_finished", "training_completed"}:
                self.training_active = False
            if name in {"prediction_finished", "prediction_completed"} and self.training_active:
                continue
            key = (name, event.get("at"))
            if name in self.phases and key not in self.phase_seen:
                self.phase_seen.add(key)
                ended.append(name)
        if ended:
            self.last_notice = now
            return "phase_finished"
        if self.stall_seconds and now - self.last_progress >= self.stall_seconds and not self.stall_sent:
            self.stall_sent, self.last_notice = True, now
            return "possibly_stalled"
        if now - self.last_notice >= self.report_seconds:
            self.last_notice = now
            return "hourly"
        return None


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_state(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def monitor(run, pid, interval=10, report_seconds=3600, stall_seconds=None):
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock, state_path = run / "monitor.lock", run / "monitor-state.json"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        raise RuntimeError("该运行已有监控锁；先检查 monitor-state.json 中的进程，确认旧监控结束后再清理锁文件")
    os.close(descriptor)
    watcher = None
    try:
        watcher = ProcessWatch(pid)
        policy = NoticePolicy(time.monotonic(), report_seconds, stall_seconds)
        offset, pending, latest, epoch = 0, "", {}, None
        while True:
            manifest = read_json(run / "manifest.json")
            events_path = run / "events.jsonl"
            events = []
            if events_path.exists():
                with events_path.open("r", encoding="utf-8") as stream:
                    stream.seek(offset)
                    pending += stream.read()
                    offset = stream.tell()
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    events.append(event)
                    latest = event
                    epoch = event.get("epoch", epoch)
            alive, exit_code = watcher.status()
            paths = [events_path, run / "manifest.json", run / "last.pt"]
            stamp = tuple(path.stat().st_mtime_ns if path.exists() else None for path in paths)
            status = manifest.get("status", "unknown")
            notice = policy.check(time.monotonic(), status, alive, stamp, events)
            state = {"checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                     "run_id": run.name, "task_pid": pid, "monitor_pid": os.getpid(),
                     "task_alive": alive, "exit_code": exit_code, "status": status,
                     "epoch": epoch, "last_event": latest, "notification": notice,
                     "report_seconds": report_seconds, "stall_seconds": stall_seconds}
            write_state(state_path, state)
            if notice:
                notification = {"kind": notice, "run_id": run.name, "status": status,
                                "at": state["checked_at"], "state_file": str(state_path),
                                "epoch": epoch, "last_event": latest.get("event"), "exit_code": exit_code}
                with (run / "monitor-notifications.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(notification, ensure_ascii=False) + "\n")
                print(json.dumps(notification, ensure_ascii=False), flush=True)
            if status in policy.terminal or not alive:
                break
            time.sleep(interval)
    finally:
        if watcher:
            watcher.close()
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--report-seconds", type=float, default=3600)
    parser.add_argument("--stall-seconds", type=float)
    args = parser.parse_args()
    if args.interval <= 0 or args.report_seconds <= 0 or args.stall_seconds is not None and args.stall_seconds <= 0:
        parser.error("所有时间间隔必须大于零")
    monitor(args.run_dir, args.pid, args.interval, args.report_seconds, args.stall_seconds)
