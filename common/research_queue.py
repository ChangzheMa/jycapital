"""研究候选的发布、只读发现和结果交接；不启动训练进程。"""

from __future__ import annotations

import ast
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import tempfile

from artifacts import now, read_json, sha256


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def capacity_lock(root):
    """串行更新容量滞回状态，锁不涉及训练进程或模型文件。"""
    path = Path(root) / "outputs/run-control/continuous-research/capacity.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def safe_path(root, value):
    root = Path(root).resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("路径必须位于仓库内: " + str(value))
    return path


def source_files(root, solution_id):
    root = Path(root).resolve()
    directory = safe_path(root, "solutions/" + solution_id)
    files = [p for p in directory.rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".txt")
             and p.name != "experiment.json" and "__pycache__" not in p.parts]
    # 仅冻结实际导入的公共模块；其他 session 新增无关工具不应使 ready 候选失效。
    pending, visited = [p for p in files if p.suffix == ".py"], set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
        for module in modules:
            relative = Path(*module.split(".")).with_suffix(".py")
            for folder in (root / "common", root / "evaluation"):
                dependency = folder / relative
                if dependency.is_file() and dependency not in visited:
                    files.append(dependency)
                    pending.append(dependency)
    files.append(root / "evaluation/protocols/local-v2-ic.json")
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(set(files))}


def validate_spec(root, path):
    spec = read_json(path)
    version = path.parent.name
    if not re.fullmatch(r"v[0-9]+[a-z]?", version) or spec.get("solution_id") != version:
        raise ValueError("方案目录与标识不一致")
    if spec.get("status") not in ("draft", "ready", "retired"):
        raise ValueError("候选状态必须为 draft、ready 或 retired")
    run_id = spec.get("run_id", "")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", run_id):
        raise ValueError("非法 run_id")
    if spec.get("formal_attempts") != 1:
        raise ValueError("每个候选必须明确为一次正式训练")
    if isinstance(spec.get("priority"), bool) or not isinstance(spec.get("priority"), int):
        raise ValueError("priority 必须为整数")
    entry = safe_path(root, spec["entrypoint"])
    if entry.parent != path.parent.resolve() or not entry.is_file():
        raise ValueError("训练入口必须位于本方案目录")
    if not safe_path(root, spec["config"]).is_file():
        raise ValueError("候选配置不存在")
    if spec["status"] == "retired" and not spec.get("retirement_reason"):
        raise ValueError("撤销候选必须保留原因")
    return spec


def completion(root, run_id):
    run = safe_path(root, "outputs/runs/" + run_id)
    if not run.exists():
        return "not_started", None
    if not (run / "manifest.json").exists():
        return "needs_attention", "运行目录存在但缺少 manifest"
    try:
        manifest = read_json(run / "manifest.json")
    except (OSError, ValueError):
        return "updating", "manifest 正在写入或无法读取"
    if manifest.get("status") == "complete" and manifest.get("stage", "complete") == "complete":
        required = ["selection.json", "best.pt", "history.csv"]
        required += [f"evaluation/{split}/{name}" for split in ("validation", "test")
                     for name in ("metrics.json", "predictions.parquet", "cross-sections.parquet", "monthly.csv")]
        missing = [name for name in required if not (run / name).is_file()]
        if missing:
            return "needs_attention", "完成状态缺少产物: " + ", ".join(missing)
        return "complete", None
    if manifest.get("status") == "running":
        return "running", manifest.get("stage")
    return "needs_attention", manifest.get("status")


def readiness_error(root, spec):
    record = spec.get("readiness", {})
    if not record.get("report") or not record.get("source_files"):
        return "缺少实现检查与代码指纹"
    try:
        report = read_json(safe_path(root, record["report"]))
        if report.get("status") != "passed" or report.get("solution_id") != spec["solution_id"]:
            return "实现检查未通过或不属于此版本"
        if report.get("source_files") != record["source_files"]:
            return "检查报告与发布指纹不同"
        if source_files(root, spec["solution_id"]) != record["source_files"]:
            return "发布后源码或依赖发生变化；需要重新验证"
    except (OSError, ValueError) as error:
        return str(error)
    return None


def _snapshot_unlocked(root):
    root = Path(root).resolve()
    policy = read_json(root / "experiments/research-policy.json")
    candidates, seen = [], set()
    for path in sorted((root / "solutions").glob("*/experiment.json")):
        spec = validate_spec(root, path)
        if spec["run_id"] in seen:
            raise ValueError("重复候选 run_id: " + spec["run_id"])
        seen.add(spec["run_id"])
        status, detail = completion(root, spec["run_id"])
        error = readiness_error(root, spec) if spec["status"] == "ready" and status == "not_started" else None
        candidates.append({"solution_id": spec["solution_id"], "run_id": spec["run_id"],
                           "candidate_status": spec["status"], "run_status": status,
                           "detail": detail, "readiness_error": error, "priority": spec["priority"],
                           "manifest": path.relative_to(root).as_posix(), "entrypoint": spec["entrypoint"],
                           "config": spec["config"], "legacy": False})
    for spec in policy["legacy_runs"]:
        if spec["run_id"] in seen:
            continue
        status, detail = completion(root, spec["run_id"])
        candidates.append({**spec, "candidate_status": "legacy", "run_status": status,
                           "detail": detail, "priority": 0, "legacy": True})
    candidates.sort(key=lambda row: (row["priority"], int(re.search(r"\d+", row["solution_id"])[0]), row["run_id"]))
    unfinished = [r for r in candidates if r["candidate_status"] != "retired" and r["run_status"] != "complete"]
    active = [r for r in candidates if r["run_status"] in ("running", "updating")]
    ready = [r for r in candidates if r["candidate_status"] == "ready" and
             r["run_status"] == "not_started" and not r["readiness_error"]]
    return {"at": now(), "unevaluated_count": len(unfinished), "active_count": len(active),
            "pause_construction": len(unfinished) > policy["construction_pause_when_unevaluated_gt"],
            "resume_construction": len(unfinished) < policy["construction_resume_when_unevaluated_lt"],
            "next": ready[0] if ready and not active else None, "candidates": candidates}


def snapshot(root):
    root = Path(root).resolve()
    with capacity_lock(root):
        result = _snapshot_unlocked(root)
        path = root / "outputs/run-control/continuous-research/capacity.json"
        previous = read_json(path) if path.exists() else {}
        paused = bool(previous.get("construction_paused", False))
        if result["pause_construction"]:
            paused = True
        elif result["resume_construction"]:
            paused = False
        atomic_json(path, {"at": result["at"], "unevaluated_count": result["unevaluated_count"],
                           "construction_paused": paused})
        result["pause_construction"] = paused
        return result


def publish(root, solution_id, report_path):
    root = Path(root).resolve()
    path = safe_path(root, f"solutions/{solution_id}/experiment.json")
    spec = validate_spec(root, path)
    status, _ = completion(root, spec["run_id"])
    if status != "not_started" or spec["status"] == "retired":
        raise ValueError("只能发布尚未启动且未撤销的候选")
    report_path = safe_path(root, str(report_path))
    report = read_json(report_path)
    fingerprints = source_files(root, solution_id)
    if report.get("status") != "passed" or report.get("solution_id") != solution_id or report.get("source_files") != fingerprints:
        raise ValueError("实现检查报告未通过或源码已改变")
    spec.update(status="ready", published_at=now(), readiness={
        "report": report_path.relative_to(root).as_posix(), "source_files": fingerprints})
    atomic_json(path, spec)
    return spec


def record_result(root, run):
    root = Path(root).resolve()
    run = safe_path(root, str(run))
    if run.parent != root / "outputs/runs":
        raise ValueError("结果必须来自 outputs/runs 的直接子目录")
    status, detail = completion(root, run.name)
    if status != "complete":
        raise ValueError("双集评估尚未完成: " + str(detail))
    manifest, selection = read_json(run / "manifest.json"), read_json(run / "selection.json")
    weight_file = "v1-model.json" if manifest["solution_id"] == "v1" else "best.pt"
    if sha256(run / weight_file) != selection["model_sha256"]:
        raise ValueError("冻结权重与 selection 哈希不符")
    import numpy as np
    import pandas as pd
    metrics, checks = {}, {}
    for split in ("validation", "test"):
        folder = run / "evaluation" / split
        metrics[split] = read_json(folder / "metrics.json")
        frame = pd.read_parquet(folder / "predictions.parquet")
        sections = pd.read_parquet(folder / "cross-sections.parquet")
        check = {"finite_factors": bool(np.isfinite(frame.factor).all()),
                 "unique_keys": not bool(frame.duplicated(["date", "instrument"]).any()),
                 "prediction_count_matches": len(frame) == metrics[split]["predicted_rows"],
                 "ic_matches_sections": bool(np.isclose(sections.ic.mean(), metrics[split]["ic_mean"], atol=1e-12, rtol=0)),
                 "rank_ic_matches_sections": bool(np.isclose(sections.rank_ic.mean(), metrics[split]["rank_ic_mean"], atol=1e-12, rtol=0))}
        if not all(check.values()):
            raise ValueError("预测产物检查失败: " + split)
        checks[split] = check
    env = read_json(run / "environment.json")
    history = pd.read_csv(run / "history.csv")
    result = {"run_id": run.name, "solution_id": manifest["solution_id"], "status": "complete",
              "checked_at": now(), "parent_run_id": manifest.get("parent_run_id"),
              "parent_solution_id": manifest.get("parent_solution_id"), "selection": selection,
              "metrics": metrics, "checks": checks, "parameters": env.get("parameters"),
              "device": env.get("device_name"), "epoch_seconds_including_validation": float(history.seconds.sum()),
              "fingerprints": {"config": sha256(run / "config.json"), "protocol": sha256(run / "protocol.json"),
                               "data_manifest": sha256(run / "data-manifest.json"), "model": selection["model_sha256"]},
              "source": manifest["source"], "run_path": run.relative_to(root).as_posix(),
              "limitations": ["单种子结果；测试期已被持续研究观察。", "本地指标不等同于平台官方综合分。"]}
    target = root / "outputs/checks/continuous-research" / (run.name + ".json")
    atomic_json(target, result)
    return {"run_id": run.name, "record": target.relative_to(root).as_posix(), "checks": checks}
