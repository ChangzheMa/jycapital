"""查看／发布持续研究队列、保存完成结果；本入口不会启动训练。"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from research_queue import atomic_json, publish, record_result, snapshot


def apply_data_refresh_gate(result, policy):
    """用户要求新数据期间，不向现有控制器发放新的旧数据训练任务。"""
    refresh = policy.get("data_refresh", {})
    if refresh.get("block_new_training", False):
        result = dict(result)
        result["next"] = None
        result["data_refresh_gate"] = {
            "blocked": True, "status": refresh.get("status"),
            "reason": "用户要求后续新训练使用更新后的数据；等待下载、独立预处理及调度路径绑定。已有运行和已完成结果保持。",
            "instructions": "docs/data-refresh.md",
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("status")
    sub.add_parser("next")
    command = sub.add_parser("publish")
    command.add_argument("--solution-id", required=True)
    command.add_argument("--check-report", required=True)
    command = sub.add_parser("record")
    command.add_argument("--run-dir", required=True)
    command = sub.add_parser("wait")
    command.add_argument("--seconds", type=int, default=3600)
    args = parser.parse_args()
    if args.action == "publish":
        result = publish(ROOT, args.solution_id, args.check_report)
        result = {key: result[key] for key in ("solution_id", "run_id", "status", "published_at")}
    elif args.action == "record":
        result = record_result(ROOT, args.run_dir)
    else:
        if args.action == "wait":
            if args.seconds < 1:
                parser.error("等待时长必须为正数")
            time.sleep(args.seconds)
        result = snapshot(ROOT)
        policy = json.loads((ROOT / "experiments/research-policy.json").read_text(encoding="utf-8"))
        result = apply_data_refresh_gate(result, policy)
        atomic_json(ROOT / "outputs/run-control/continuous-research/queue-status.json", result)
        if args.action == "next":
            result = result["next"]
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
