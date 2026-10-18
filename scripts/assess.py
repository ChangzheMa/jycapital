"""比较一次已完成实验与历史参照，记录验证主指标及覆盖率晋级结论。"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from artifacts import read_json, write_json, sha256, experiment_filename
from protocol_compatibility import validate_protocol_compatibility
from compare import compare
import pandas as pd


METRICS = ("ic_mean", "rank_ic_mean", "ic_ir", "rank_ic_ir", "mse", "mae", "coverage",
           "min_cross_section_coverage", "evaluated_rows", "ic_undefined_cross_sections",
           "rank_ic_undefined_cross_sections")


def resources(run):
    env = read_json(run / "environment.json")
    selection = read_json(run / "selection.json")
    parameters = env.get("parameters")
    if parameters is None and (run / "events.jsonl").exists():
        for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("event") == "training_started":
                parameters = event.get("parameters")
                break
    return {"parameters": parameters, "device": env["device_name"],
            "epoch_seconds_including_validation": float(pd.read_csv(run / "history.csv").seconds.sum()),
            "recorded_training_seconds": selection.get("training_seconds", selection.get("training_seconds_this_session")),
            "checkpoint_bytes": (run / "best.pt").stat().st_size if (run / "best.pt").exists() else None,
            "timing_note": "耗时包含的数据读取与精度路径不同，不能将时间比解释为纯网络计算比"}


def assess(run, experiments=ROOT / "experiments", *, allow_selection_metric_change=False):
    run, experiments = Path(run).resolve(), Path(experiments)
    manifest = read_json(run / "manifest.json")
    protocol = read_json(run / "protocol.json")
    if manifest["status"] != "complete":
        raise ValueError("必须先完成训练和验证／测试")
    index = pd.read_csv(experiments / "runs.csv")
    comparable, excluded = [], []
    for row in index.to_dict("records"):
        if row["run_id"] == run.name:
            continue
        path = ROOT / row["run_path"]
        try:
            compare([run, path], "validation", allow_selection_metric_change=allow_selection_metric_change)
            compare([run, path], "test", allow_selection_metric_change=allow_selection_metric_change)
            comparable.append(path)
        except (ValueError, FileNotFoundError) as error:
            excluded.append({"run_id": row["run_id"], "reason": str(error)})
    if not comparable:
        raise ValueError("没有可核验的历史参照；不能跳过已有基线作晋级判断")
    metric = protocol["selection_metric"]
    best = max(comparable, key=lambda p: read_json(p / "evaluation/validation/metrics.json")[metric])
    roles = {best.name: {"path": best, "roles": ["local_best"]}}
    parent_id = manifest.get("parent_run_id")
    if parent_id:
        parent = next((p for p in comparable if p.name == parent_id), None)
        if parent is None:
            raise ValueError("父版本的可比性未通过")
        roles.setdefault(parent_id, {"path": parent, "roles": []})["roles"].append("parent")
    # 平台最优必须由有评分口径的可信记录选定，不能将运行确认当作分数。
    submissions = pd.read_csv(experiments / "submissions.csv")
    if submissions.platform_score.notna().any():
        raise ValueError("已有平台分：请先按评分口径确定平台最优参照，再扩展比较记录")
    current = {s: read_json(run / "evaluation" / s / "metrics.json") for s in ("validation", "test")}
    comparisons = []
    coverage_ok, defined_sections_ok = True, True
    for name, reference in roles.items():
        entry = {"run_id": name, "roles": reference["roles"], "resources": resources(reference["path"]),
                 "protocol_sha256": sha256(reference["path"] / "protocol.json"),
                 "protocol_differences": validate_protocol_compatibility(
                     protocol, read_json(reference["path"] / "protocol.json"),
                     allow_selection_metric_change=allow_selection_metric_change), "splits": {}}
        for split in ("validation", "test"):
            before = read_json(reference["path"] / "evaluation" / split / "metrics.json")
            after = current[split]
            monthly = pd.read_csv(run / "evaluation" / split / "monthly.csv").merge(
                pd.read_csv(reference["path"] / "evaluation" / split / "monthly.csv"),
                on="month", suffixes=("_new", "_reference"), validate="one_to_one")
            for key in ("ic_mean", "rank_ic_mean"):
                monthly[key + "_delta"] = monthly[key + "_new"] - monthly[key + "_reference"]
            sections = pd.read_parquet(run / "evaluation" / split / "cross-sections.parquet").merge(
                pd.read_parquet(reference["path"] / "evaluation" / split / "cross-sections.parquet"),
                on="date", suffixes=("_new", "_reference"), validate="one_to_one", how="outer")
            if sections[["expected_new", "expected_reference"]].isna().any().any() or not (sections.expected_new == sections.expected_reference).all():
                raise ValueError("评估截面或股票池分母不同")
            reduced = int((sections.coverage_new < sections.coverage_reference).sum())
            if name == best.name:
                coverage_ok &= reduced == 0
                defined_sections_ok &= after["ic_undefined_cross_sections"] <= before["ic_undefined_cross_sections"]
            entry["splits"][split] = {
                "reference": {k: before[k] for k in METRICS},
                "new_minus_reference": {k: after[k] - before[k] if after[k] is not None and before[k] is not None else None for k in METRICS},
                "coverage_reduced_cross_sections": reduced,
                "rank_ic_improved_months": int((monthly.rank_ic_mean_delta > 0).sum()),
                "ic_improved_months": int((monthly.ic_mean_delta > 0).sum()),
                "selection_metric_improved_months": int((monthly[metric + "_delta"] > 0).sum()),
                "months": len(monthly), "monthly": monthly.to_dict("records"),
            }
        comparisons.append(entry)
    best_metric = read_json(best / "evaluation/validation/metrics.json")[metric]
    coverage_ok &= all(v["coverage_violations"] == 0 for v in current.values())
    candidate = bool(current["validation"][metric] > best_metric and coverage_ok and
                     (defined_sections_ok or metric != "ic_mean"))
    result = {
        "solution_id": manifest["solution_id"], "run_id": run.name, "status": "complete",
        "protocol_id": protocol["protocol_id"], "protocol_sha256": sha256(run / "protocol.json"),
        "config_sha256": sha256(run / "config.json"), "source": manifest["source"],
        "data_manifest_sha256": sha256(run / "data-manifest.json"),
        "assessment_script_sha256": sha256(__file__),
        "selection": manifest["selection"], "metrics": current,
        "references": comparisons, "excluded_runs": excluded, "platform_best": None,
        "platform_status": "暂无可信平台成绩", "candidate": candidate,
        "decision": "local_improvement_candidate" if candidate else "no_local_improvement",
        "criterion": {"metric": "validation." + metric, "new": current["validation"][metric],
                      "local_best": best_metric, "coverage_not_reduced": coverage_ok,
                      "undefined_ic_sections_not_increased": defined_sections_ok},
        "resources": resources(run),
        "exported": False, "run_path": run.relative_to(ROOT).as_posix(),
        "limitations": ["仅一个种子，未进行统计显著性或消融验证", "研究变量以各运行的冻结配置为准，多项改动的效果只针对整体方案，不能分别归因", "测试期为持续研究中已观察区间，未用于本次 checkpoint 选择", "本地指标不包含平台 BARRA 处理及官方评分"],
    }
    checks = run / "implementation-checks.json"
    if checks.exists():
        saved = read_json(checks)
        result["implementation_checks"] = {k: saved[k] for k in (
            "tests", "checked_source_sha256", "evaluation_key_counts_match_parent",
            "finite_label_counts_match_parent", "raw_inference_parity") if k in saved}
    for filename, key in (("final-checks.json", "final_checks"),
                          ("loss-consistency.json", "loss_consistency")):
        if (run / filename).exists():
            result[key] = read_json(run / filename)
    diagnostic = run / "precision-diagnostic.json"
    if diagnostic.exists():
        saved = read_json(diagnostic)
        result["precision_diagnostic"] = {k: saved[k] for k in ("sample", "rows", "fields")}
        result["precision_diagnostic"]["limitation"] = "仅一个训练日的处理后行情抽样；未进行 FP32 训练对照，不能据此解释本次指标差距"
    write_json(run / "comparison.json", result)
    return result


def record(result):
    config = read_json(ROOT / result["run_path"] / "config.json")
    experiment = ROOT / "experiments" / experiment_filename(
        result["solution_id"], config.get("model", {}).get("family"))
    write_json(experiment, result)
    path = ROOT / "experiments/runs.csv"
    index = pd.read_csv(path, dtype=str, keep_default_na=False)
    selection, metrics = result["selection"], result["metrics"]
    row = {"run_id": result["run_id"], "solution_id": result["solution_id"], "protocol_id": result["protocol_id"],
           "seed": config["seed"], "best_epoch": selection["best_epoch"],
           "validation_ic": metrics["validation"]["ic_mean"], "validation_rank_ic": metrics["validation"]["rank_ic_mean"],
           "test_ic": metrics["test"]["ic_mean"], "test_rank_ic": metrics["test"]["rank_ic_mean"],
           "test_coverage": metrics["test"]["coverage"], "model_sha256": selection["model_sha256"], "run_path": result["run_path"]}
    index = pd.concat([index.loc[index.run_id != row["run_id"]], pd.DataFrame([row])], ignore_index=True)
    index.to_csv(path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--allow-selection-metric-change", action="store_true")
    args = parser.parse_args()
    result = assess(args.run, allow_selection_metric_change=args.allow_selection_metric_change)
    if args.record:
        record(result)
    print(result["decision"], result["criterion"])
