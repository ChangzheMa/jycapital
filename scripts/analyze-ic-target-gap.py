"""用冻结逐截面结果量化 IC 目标差距，不拟合模型或改动评估口径。"""
import argparse
import csv
import math
from pathlib import Path
import sys

import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for directory in ("common", "scripts"):
    sys.path.insert(0, str(ROOT / directory))
from artifacts import now, read_json, sha256, write_json
from compare import compare


def analyze_split(run, split, target):
    directory = run / "evaluation" / split
    metrics = read_json(directory / "metrics.json")
    sections = pd.read_parquet(directory / "cross-sections.parquet")
    if sections.date.duplicated().any():
        raise ValueError("逐截面日期重复")
    if len(sections) != metrics["cross_sections"]:
        raise ValueError("逐截面记录数与冻结指标不一致")
    for metric in ("ic", "rank_ic"):
        if not np.isfinite(sections[metric]).all():
            raise ValueError("此目标差距诊断要求全部截面指标有定义")
        if not np.isclose(sections[metric].mean(), metrics[metric + "_mean"], atol=1e-12, rtol=0):
            raise ValueError("逐截面均值与冻结指标不一致")
    sections = sections.assign(clock=sections.date.dt.strftime("%H:%M"),
                               month=sections.date.dt.strftime("%Y-%m"))
    grouped = sections.groupby("clock").agg(
        sections=("ic", "size"), ic_mean=("ic", "mean"), rank_ic_mean=("rank_ic", "mean"))
    reconstructed = float((grouped.ic_mean * grouped.sections).sum() / len(sections))
    if not np.isclose(reconstructed, metrics["ic_mean"], atol=1e-12, rtol=0):
        raise ValueError("时点分解未还原整体指标")
    clocks = []
    for clock, row in grouped.iterrows():
        count, mean = int(row.sections), float(row.ic_mean)
        weight = count / len(sections)
        others = metrics["ic_mean"] - weight * mean
        clocks.append({
            "clock": clock, "sections": count, "weight": weight,
            "ic_mean": mean, "rank_ic_mean": float(row.rank_ic_mean),
            "gap_to_target": target - mean,
            "overall_if_only_this_clock_reaches_target": others + weight * target,
            "overall_if_only_this_clock_is_perfect": others + weight,
            "required_this_clock_ic_if_others_fixed": (target - others) / weight,
            "required_other_clocks_ic_if_this_clock_fixed": (target - weight * mean) / (1 - weight),
        })
    monthly = sections.groupby("month").agg(
        sections=("ic", "size"), ic_mean=("ic", "mean"), rank_ic_mean=("rank_ic", "mean"))
    return {
        "split": split, "ic_mean": metrics["ic_mean"], "rank_ic_mean": metrics["rank_ic_mean"],
        "target": target, "absolute_gap": target - metrics["ic_mean"],
        "target_multiple_of_current": target / metrics["ic_mean"],
        "coverage": metrics["coverage"], "cross_sections": len(sections),
        "clock_weighted_mean_reconstructs_frozen_metric": True,
        "any_single_clock_perfect_ceiling": max(r["overall_if_only_this_clock_is_perfect"] for r in clocks),
        "all_single_clock_perfect_scenarios_below_target": all(
            r["overall_if_only_this_clock_is_perfect"] < target for r in clocks),
        "fraction_sections_at_or_above_target": float((sections.ic >= target).mean()),
        "fraction_negative_sections": float((sections.ic < 0).mean()),
        "clocks": clocks,
        "monthly": [{"month": month, "sections": int(row.sections),
                     "ic_mean": float(row.ic_mean), "rank_ic_mean": float(row.rank_ic_mean)}
                    for month, row in monthly.iterrows()],
        "source_hashes": {name: sha256(directory / name)
                          for name in ("metrics.json", "cross-sections.parquet")},
    }


def compare_clocks(run, reference, split):
    paths = [path / "evaluation" / split / "cross-sections.parquet" for path in (run, reference)]
    frames = [pd.read_parquet(path).set_index("date").sort_index() for path in paths]
    if not frames[0].index.equals(frames[1].index):
        raise ValueError("时点比较的截面日期不一致")
    delta = frames[0][["ic", "rank_ic"]] - frames[1][["ic", "rank_ic"]]
    if not np.isfinite(delta.to_numpy()).all():
        raise ValueError("时点比较存在未定义指标")
    clock = delta.groupby(delta.index.strftime("%H:%M")).mean()
    counts = delta.groupby(delta.index.strftime("%H:%M")).size()
    monthly = delta.groupby([delta.index.strftime("%Y-%m"), delta.index.strftime("%H:%M")]).mean()
    monthly.index.names = ["month", "clock"]
    positive = (monthly > 0).groupby("clock").sum()
    months = monthly.groupby("clock").size()
    reconstructed = clock.mul(counts, axis=0).sum() / len(delta)
    if not np.allclose(reconstructed, delta.mean(), atol=1e-12, rtol=0):
        raise ValueError("时点差值未还原整体差值")
    return {
        "source_hashes": [sha256(path) for path in paths],
        "weighted_clock_deltas_reconstruct_overall_delta": True,
        "clocks": [{"clock": key, "sections": int(counts[key]),
                    "ic_delta": float(row.ic), "rank_ic_delta": float(row.rank_ic),
                    "ic_improved_months": int(positive.loc[key, "ic"]),
                    "rank_ic_improved_months": int(positive.loc[key, "rank_ic"]),
                    "months": int(months[key])} for key, row in clock.iterrows()],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="v1601-price-geometry-s42")
    parser.add_argument("--target", type=float, default=0.25)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments/ic-target-gap-20261023.json")
    args = parser.parse_args()
    if not math.isfinite(args.target) or not 0 < args.target <= 1:
        raise ValueError("IC 目标必须在 (0, 1] 范围内")
    run = ROOT / "outputs/runs" / args.run
    references = [ROOT / "outputs/runs" / name for name in (
        "v1601-price-geometry-s42", "v1102-structured-temporal-experts-s42",
        "v1401-frozen-equal-ensemble-s42") if name != run.name]
    if read_json(run / "manifest.json")["status"] != "complete":
        raise ValueError("只分析已完成运行")
    report = {
        "computed_at": now(), "script_sha256": sha256(__file__), "run_id": run.name,
        "target_ic_mean": args.target, "protocol_id": read_json(run / "protocol.json")["protocol_id"],
        "method": "逐截面等权均值按预测时点分解；仅替换一个时点的假设 IC，其余时点保持冻结结果。",
        "formal_metrics_modified": False, "model_training_performed": False,
        "raw_market_data_read": False, "label_table_read": False, "exposure_read": False,
        "results": [], "references": [],
        "limitations": [
            "单时点完美预测上界只约束其余时点完全不变的假设，不是整体任务的可预测性上限。",
            "0.25 是用户指定目标，不能由分解或论文证明可达。",
            "不使用测试结果选择 checkpoint，不改变已有候选晋级规则。",
            "训练损失来自不同 batch 中的原始训练权重，不能当作训练集正式 IC 或与 EMA 验证 IC 直接比较。",
        ],
    }
    for split in ("validation", "test"):
        compare([run, *references], split, allow_selection_metric_change=True)
        report["results"].append(analyze_split(run, split, args.target))
        for reference in references:
            metrics = read_json(reference / "evaluation" / split / "metrics.json")
            report["references"].append({"run_id": reference.name, "split": split,
                                         "ic_mean": metrics["ic_mean"], "rank_ic_mean": metrics["rank_ic_mean"],
                                         "clock_effects": compare_clocks(run, reference, split)})
    with (run / "history.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    report["completed_training_history"] = [
        {"epoch": int(row["epoch"]), "train_loss": float(row["train_loss"]),
         "train_huber_loss": float(row["train_huber_loss"]),
         "validation_ic_mean": float(row["validation_ic_mean"])} for row in rows]
    report["history_sha256"] = sha256(run / "history.csv")
    write_json(args.output, report)
    for row in report["results"]:
        opening = next(clock for clock in row["clocks"] if clock["clock"] == "09:30")
        print({key: row[key] for key in ("split", "ic_mean", "absolute_gap", "target_multiple_of_current",
                                        "any_single_clock_perfect_ceiling")})
        print({"opening": opening})


if __name__ == "__main__":
    main()
