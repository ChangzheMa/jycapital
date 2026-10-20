"""研究 session 的每小时容量监测；不训练、不修改候选、不通知其他 session。"""

import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from artifacts import now, read_json
from research_queue import atomic_json, snapshot
from run_notifications import send_notice


def observe(previous, current):
    paused = bool(previous.get("construction_paused", False))
    if current["pause_construction"]:
        paused = True
    elif paused and current["resume_construction"]:
        paused = False
    completed = sorted(row["run_id"] for row in current["candidates"] if row["run_status"] == "complete")
    new_completed = sorted(set(completed) - set(previous.get("completed", completed)))
    recovered = bool(previous.get("construction_paused", False)) and not paused
    return {"construction_paused": paused, "completed": completed, "new_completed": new_completed,
            "capacity_recovered": recovered,
            "notify": bool(recovered or new_completed or (not paused and current["resume_construction"]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--codex-binary", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=3600)
    parser.add_argument("--once", action="store_true", help="只读取状态，不发送通知")
    args = parser.parse_args()
    if args.thread_id != os.environ.get("CODEX_THREAD_ID"):
        parser.error("只允许监测并通知启动者自己的当前 session")
    if args.interval < 3600:
        parser.error("本轮容量检查间隔至少为一小时")
    control = ROOT / "outputs/run-control/continuous-research" / args.thread_id
    control.mkdir(parents=True, exist_ok=True)
    current = snapshot(ROOT)
    if args.once:
        print(json.dumps(observe({}, current), ensure_ascii=False, indent=2))
        return
    # 持有操作系统文件锁到进程退出，避免并发启动和 PID 复用造成重复通知。
    lock_handle = (control / "watcher.lock").open("a+b")
    if lock_handle.tell() == 0:
        lock_handle.write(b"0")
        lock_handle.flush()
    lock_handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    enabled = control / "enabled.json"
    atomic_json(enabled, {"enabled": True, "thread_id": args.thread_id, "started_at": now()})
    previous = observe({}, current)
    state = {"thread_id": args.thread_id, "pid": os.getpid(), "checked_at": now(),
             **previous, "queue": current, "notification_delivery": "not_yet_sent",
             "current_thread_idle_wake_verified": False,
             "next_check_at": (datetime.now().astimezone() + timedelta(seconds=args.interval)).isoformat()}
    atomic_json(control / "state.json", state)
    while True:
        deadline = time.monotonic() + args.interval
        while time.monotonic() < deadline:
            if not read_json(enabled).get("enabled", False):
                atomic_json(control / "stopped.json", {"at": now(), "reason": "disabled_by_user_or_owner"})
                return
            time.sleep(min(5, max(0, deadline - time.monotonic())))
        try:
            current = snapshot(ROOT)
            observed = observe(previous, current)
            state.update(checked_at=now(), queue=current, **observed,
                         next_check_at=(datetime.now().astimezone() + timedelta(seconds=args.interval)).isoformat())
            if observed["notify"]:
                notice = {"kind": "research_capacity_hourly", "event_id": "research-capacity:" + now(),
                          "unevaluated": current["unevaluated_count"], "paused": observed["construction_paused"],
                          "new_completed": observed["new_completed"],
                          "state_file": str((control / "state.json").relative_to(ROOT)),
                          "action": "先分析完成结果；待评测少于3才构造，不启动训练。"}
                atomic_json(control / "state.json", state)
                delivery = send_notice(args.codex_binary, args.thread_id, notice)
                with (control / "deliveries.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(delivery, ensure_ascii=False) + "\n")
                state["notification_delivery"] = delivery
            previous = observed
            atomic_json(control / "state.json", state)
        except Exception as error:
            state.update(last_error={"at": now(), "type": type(error).__name__, "message": str(error)})
            atomic_json(control / "state.json", state)


if __name__ == "__main__":
    main()
