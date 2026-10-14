"""独立后台监控通知与空闲验收，仅向已授权的当前任务发送事件。"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from codex_rpc import RpcProxy

ROOT = Path(__file__).resolve().parents[1]
MONITOR = ROOT / ".agents/skills/jycapital-experiment/scripts/monitor-run.py"


def now():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def send_notice(binary, thread_id, notice):
    message = json.dumps(notice, ensure_ascii=False, separators=(",", ":"))
    prompt = ("进益资本当前已授权任务的后台事件：" + message +
              "。读取记录核验当前状态；沿用现有运行，不重复启动。"
              "若训练与测试已完成，继续已授权的比较、结果记录及最终方案说明。")
    if len(prompt.encode("utf-8")) > 1000:
        raise ValueError("通知超过宿主接口长度")
    rpc = RpcProxy(binary, thread_id)
    try:
        returned = rpc.call("mcpServer/tool/call", {
            "threadId": thread_id, "server": "codex_tui", "tool": "send_message_to_thread",
            "arguments": {"threadId": thread_id, "prompt": prompt},
        })
        if returned.get("isError"):
            raise RuntimeError("宿主未接受通知")
        return {"accepted_at": now(), "thread_id": thread_id, "event_id": notice["event_id"]}
    finally:
        rpc.close()


def idle_probe(args):
    directory = args.probe_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    state = {"event_id": "idle-probe-" + uuid.uuid4().hex[:12], "thread_id": args.thread_id,
             "status": "waiting_for_idle", "armed_at": now(), "automatic_wake_verified": False,
             "execution_permissions_verified": False}
    path = directory / "result.json"
    write_json(path, state)
    rpc = RpcProxy(args.codex_binary, args.thread_id)
    try:
        deadline, idle_since = time.monotonic() + 900, None
        while time.monotonic() < deadline:
            status = rpc.thread()["status"]["type"]
            if status == "idle":
                if idle_since is None:
                    idle_since = time.monotonic()
                    state["idle_observed_at"] = now()
                if time.monotonic() - idle_since >= 8:
                    break
            else:
                idle_since = None
            time.sleep(1)
        else:
            raise TimeoutError("未观察到对话空闲")
        state.update(status="sending", idle_before_dispatch=True, dispatched_at=now(),
                     idle_seconds=time.monotonic() - idle_since)
        write_json(path, state)
        prompt = ("当前已授权任务的独立通知发送器空闲验收事件。"
                  f"event_id={state['event_id']}，记录目录={directory.as_posix()}。"
                  "实际执行命令读取 result.json 并写 ack.json，包含 event_id、thread_id（CODEX_THREAD_ID）、"
                  "received_at（带时区时间）、command_exit_code=0 和 sandbox_mode（本回合实际模式）。"
                  "命令成功后才写确认；如权限报错，明确报告未通过。只验证通知与执行权限，"
                  "不启动新训练、不重复发送验收；之后继续当前已授权任务。")
        returned = rpc.call("mcpServer/tool/call", {
            "threadId": args.thread_id, "server": "codex_tui", "tool": "send_message_to_thread",
            "arguments": {"threadId": args.thread_id, "prompt": prompt},
        })
        state.update(status="waiting_for_ack", accepted=not returned.get("isError", False),
                     accepted_at=now())
        write_json(path, state)
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if (directory / "ack.json").exists():
                ack = json.loads((directory / "ack.json").read_text(encoding="utf-8"))
                if (ack.get("event_id") == state["event_id"] and ack.get("thread_id") == args.thread_id
                        and ack.get("command_exit_code") == 0):
                    state.update(status="passed", receiver_ack=ack, automatic_wake_verified=True,
                                 execution_permissions_verified=True)
                    write_json(path, state)
                    return
            time.sleep(1)
        raise TimeoutError("没有收到自动恢复回合确认")
    except Exception as error:
        state.update(status="failed", error=str(error))
        write_json(path, state)
        raise
    finally:
        rpc.close()


def monitor(args):
    run = args.run_dir.resolve()
    command = [sys.executable, "-u", str(MONITOR), "--run-dir", str(run), "--pid", str(args.pid),
               "--interval", "10", "--report-seconds", "3600", "--stall-seconds", "7200"]
    sent = set()
    delivery_path = run / "notification-deliveries.jsonl"
    if delivery_path.exists():
        for line in delivery_path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item.get("accepted_at"):
                sent.add(item["event_id"])
    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, encoding="utf-8",
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        for line in process.stdout:
            original = json.loads(line)
            notice = {key: original[key] for key in ("kind", "run_id", "status", "at", "state_file")}
            notice["event_id"] = original["kind"] + ":" + original["at"]
            if notice["event_id"] in sent:
                continue
            result = {"event_id": notice["event_id"], "created_at": now()}
            for attempt in range(3):
                try:
                    result.update(send_notice(args.codex_binary, args.thread_id, notice))
                    sent.add(notice["event_id"])
                    break
                except Exception as error:
                    result["error"] = str(error)
                    if attempt < 2:
                        time.sleep(2 ** attempt)
            with delivery_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        if process.wait() != 0:
            raise RuntimeError("监控进程异常退出")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex-binary", type=Path, required=True)
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--probe-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--pid", type=int)
    args = parser.parse_args()
    if args.probe_dir:
        idle_probe(args)
    elif args.run_dir and args.pid:
        monitor(args)
    else:
        parser.error("必须指定空闲验收目录，或现有运行目录与 PID")


if __name__ == "__main__":
    main()
