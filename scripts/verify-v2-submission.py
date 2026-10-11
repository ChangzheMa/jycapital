"""核验 v2 冻结提交包，并将原始行情入口与本地评测预测逐键比较。"""

import argparse
import importlib
from pathlib import Path
import re
import sys
import time

import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "common"))
from artifacts import now, read_json, sha256, write_json


class RecordingDai:
    def __init__(self, module):
        self.module, self.tables, self.calls = module, set(), 0

    def query(self, sql, *args, **kwargs):
        tables = set(re.findall(r"\bFROM\s+([A-Za-z_][A-Za-z0-9_]*)", sql, flags=re.IGNORECASE))
        if any("vwap" in t or "exposure" in t for t in tables):
            raise AssertionError("推理读取了标签或风险暴露表")
        self.tables.update(tables)
        self.calls += 1
        return self.module.query(sql, *args, **kwargs)


def verify(run, package, start, end, split="test"):
    run, package = Path(run).resolve(), Path(package).resolve()
    manifest = read_json(package / "submission-manifest.json")
    for name, digest in manifest["files"].items():
        if sha256(package / name) != digest:
            raise ValueError("提交文件已改变: " + name)
    if manifest["run_id"] != run.name or sha256(package / "best.pt") != read_json(run / "selection.json")["model_sha256"]:
        raise ValueError("提交包与运行／所选权重不匹配")
    notebooks = list(package.glob("*.ipynb"))
    if len(notebooks) != 1:
        raise ValueError("必须恰好包含一个推理 Notebook")
    notebook = read_json(notebooks[0])
    cells = notebook["cells"]
    if len(cells) != 2 or any(c.get("outputs") or c.get("execution_count") is not None for c in cells):
        raise ValueError("Notebook 单元格或输出不符合提交约定")
    if 'if __name__ == "__main__":' not in "".join(cells[-1]["source"]):
        raise ValueError("调试单元格缺少主程序保护")
    import dai
    recorder = RecordingDai(dai)
    sys.path.insert(0, str(package))
    # 独立命令中也显式清理同名模块，防止误用工作区实现。
    for name in ("v2_predict", "v2_data", "v2_model", "v2_preprocessing"):
        sys.modules.pop(name, None)
    previous_dai = sys.modules.get("dai")
    sys.modules["dai"] = recorder
    namespace = {"__name__": "submission_import_check"}
    try:
        for cell in cells:
            exec(compile("".join(cell["source"]), str(notebooks[0]), "exec"), namespace)
        if recorder.calls:
            raise AssertionError("导入 Notebook 自动查询了行情")
        sources = {f: "cpt_jyc_2026_stock_" + f for f in ("bar1m", "bar5m", "bar15m", "bar30m")}
        begin = time.perf_counter()
        debug_executed = start == "2024-12-30" and end == "2024-12-31"
        if debug_executed:
            namespace.update(__name__="__main__", display=lambda value: None)
            exec(compile("".join(cells[-1]["source"]), str(notebooks[0]), "exec"), namespace)
            result = namespace["result"]
        else:
            result = namespace["main"](sources, start, end)
        seconds = time.perf_counter() - begin
        predictor = importlib.import_module("v2_predict")
        lo, hi = predictor.platform_interval(start, end)
    finally:
        sys.modules["dai"] = previous_dai
    if list(result.columns) != ["date", "instrument", "factor"] or result.factor.dtype != np.dtype("float64"):
        raise AssertionError("输出列或 factor 类型错误")
    if result.empty or result.duplicated(["date", "instrument"]).any() or result[["date", "instrument"]].isna().any().any():
        raise AssertionError("输出为空或键非法")
    if not np.isfinite(result.factor).all() or not ((result.date >= lo) & (result.date < hi)).all():
        raise AssertionError("数值非有限或时间范围错误")
    reference = pd.read_parquet(run / "evaluation" / split / "predictions.parquet",
                                filters=[("date", ">=", lo), ("date", "<", hi)])
    merged = result.merge(reference, on=["date", "instrument"], how="outer", suffixes=("_package", "_reference"), validate="one_to_one")
    if merged[["factor_package", "factor_reference"]].isna().any().any():
        raise AssertionError("提交入口与评测输出键集合不同")
    difference = np.abs(merged.factor_package - merged.factor_reference)
    np.testing.assert_allclose(merged.factor_package, merged.factor_reference, atol=2e-7, rtol=1e-4)
    report = {"checked_at": now(), "run_id": run.name, "package": package.relative_to(ROOT).as_posix(),
              "environment": "local_dai_shim" if getattr(dai, "IS_LOCAL_SHIM", False) else "platform_dai",
              "start": str(lo), "stop_exclusive": str(hi), "reference_split": split,
              "rows": len(result), "coverage_vs_reference": 1.0, "max_abs_difference": float(difference.max()),
              "p99_abs_difference": float(difference.quantile(0.99)), "atol": 2e-7, "rtol": 1e-4,
              "seconds": round(seconds, 3), "device": "cuda" if torch.cuda.is_available() else "cpu",
              "torch": torch.__version__, "queried_tables": sorted(recorder.tables),
              "notebook_import_queried_data": False, "debug_cell_executed": debug_executed,
              "model_sha256": sha256(package / "best.pt"), "passed": True,
              "scope": "已在所列本地环境与完整请求区间验证，不代表平台运行或官方评分通过"}
    write_json(package / "submission-validation.json", report)
    write_json(run / "submission-validation.json", report)
    manifest.update(validation_status="local_passed", verification_script_sha256=sha256(__file__),
                    files={p.name: sha256(p) for p in sorted(package.iterdir()) if p.is_file() and p.name != "submission-manifest.json"})
    write_json(package / "submission-manifest.json", manifest)
    return report


def record(run, package, report):
    index_path = ROOT / "experiments/submissions.csv"
    index = pd.read_csv(index_path, dtype=str, keep_default_na=False)
    row = {"submission_id": run.name, "run_id": run.name, "status": "local_verified_not_submitted",
           "directory": package.relative_to(ROOT).as_posix(), "manifest_sha256": sha256(package / "submission-manifest.json"),
           "platform_submission_id": None, "platform_score": None}
    index = pd.concat([index.loc[index.submission_id != run.name], pd.DataFrame([row])], ignore_index=True)
    index.to_csv(index_path, index=False, encoding="utf-8-sig")
    summary_path = ROOT / "experiments/v2-tcn.json"
    result = read_json(summary_path) if summary_path.is_file() else read_json(run / "comparison.json")
    if result["run_id"] != run.name:
        result = read_json(run / "comparison.json")
    manifest = read_json(package / "submission-manifest.json")
    result.update(exported=True, submission_directory=row["directory"], submission_validation=report,
                  submission_manifest_sha256=row["manifest_sha256"],
                  export_purpose=manifest["export_purpose"], export_reason=manifest["export_reason"])
    write_json(summary_path, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--start", default="2024-12-30")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    report = verify(args.run_dir, args.package, args.start, args.end, args.split)
    if args.record:
        record(args.run_dir.resolve(), args.package.resolve(), report)
    print(report)
