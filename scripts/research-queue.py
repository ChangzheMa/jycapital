"""查看／发布持续研究队列、保存完成结果；本入口不会启动训练。"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from research_queue import atomic_json, publish, record_result, snapshot


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
        atomic_json(ROOT / "outputs/run-control/continuous-research/queue-status.json", result)
        if args.action == "next":
            result = result["next"]
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
