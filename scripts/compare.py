"""只在评估协议、评估器和评估数据一致时比较运行。"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from artifacts import read_json, sha256
from protocol_compatibility import validate_protocol_compatibility

import pandas as pd


def compare(runs, split, *, allow_selection_metric_change=False):
    rows, reference, reference_protocol = [], None, None
    for value in runs:
        run = Path(value)
        manifest = read_json(run / "manifest.json")
        protocol = read_json(run / "protocol.json")
        data = read_json(run / "data-manifest.json")
        if manifest["status"] != "complete":
            raise ValueError(f"运行未完成: {run.name}")
        if data["mode"] != "local":
            raise ValueError("平台数据需要先提供可验证的数据快照标识")
        tables = {protocol["universe_table"], protocol["label_table"]}
        fingerprint = {
            "protocol": sha256(run / "protocol.json"),
            "evaluator": manifest["source"]["files"]["evaluation/baseline_evaluation.py"],
            "evaluation_data": {item["path"]: item["sha256"] for item in data["files"] if Path(item["path"]).stem in tables},
        }
        if reference is not None:
            validate_protocol_compatibility(reference_protocol, protocol,
                                             allow_selection_metric_change=allow_selection_metric_change)
            allowed = allow_selection_metric_change and all(
                fingerprint[key] == reference[key] for key in ("evaluator", "evaluation_data"))
            if fingerprint != reference and not allowed:
                raise ValueError("评估协议、评估器或标签/股票池快照不同，不能直接比较")
        reference = fingerprint
        reference_protocol = protocol
        metrics = read_json(run / "evaluation" / split / "metrics.json")
        rows.append({"run_id": run.name, "solution_id": manifest["solution_id"], "split": split,
                     "best_epoch": manifest["selection"]["best_epoch"],
                     **{key: metrics[key] for key in ["ic_mean", "rank_ic_mean", "ic_ir", "rank_ic_ir", "mse", "coverage", "min_cross_section_coverage", "evaluated_rows"]}})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--output")
    parser.add_argument("--allow-selection-metric-change", action="store_true")
    args = parser.parse_args()
    result = compare(args.runs, args.split, allow_selection_metric_change=args.allow_selection_metric_change)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
