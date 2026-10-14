"""核验 v4 冻结提交包，并将原始行情入口与本地评测预测逐键比较。"""

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(ROOT / "solutions/v4"))
from artifacts import now, read_json, sha256, write_json
from v4_text_artifact import decode_state


def manifest_path(package):
    return ROOT / "experiments" / (Path(package).name + "-submission.json")


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
    manifest = read_json(manifest_path(package))
    if set(p.name for p in package.iterdir()) != {"v4-submission.ipynb", "v4-model.json"}:
        raise ValueError("提交目录必须只含 Notebook 和 JSON 两个文本文件")
    for name, digest in manifest["files"].items():
        (package / name).read_text(encoding="utf-8")
        if sha256(package / name) != digest:
            raise ValueError("提交文件已改变: " + name)
    if manifest["run_id"] != run.name or manifest["model_sha256"] != read_json(run / "selection.json")["model_sha256"]:
        raise ValueError("提交包与运行／所选权重不匹配")
    bundle = read_json(package / "v4-model.json")
    for relative, entry in bundle["sources"].items():
        digest = hashlib.sha256(entry["text"].encode("utf-8")).hexdigest()
        if digest != entry["sha256"] or digest != manifest["embedded_source_hashes"][relative]:
            raise ValueError("内嵌源码已改变: " + relative)
    if sha256(run / "best.pt") != manifest["model_sha256"]:
        raise ValueError("原始 checkpoint 已改变")
    original = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    restored = decode_state(bundle)
    if original["config"] != bundle["config"] or original["state_dict"].keys() != restored.keys():
        raise ValueError("文本权重配置或张量集合不同")
    for name, tensor in original["state_dict"].items():
        if tensor.dtype != restored[name].dtype or not torch.equal(tensor, restored[name]):
            raise ValueError("文本权重与原 checkpoint 不同: " + name)
    if bundle["preprocessing"] != read_json(run / "source/solutions/v4/config.json"):
        raise ValueError("预处理配置不同")
    for frequency, statistics in bundle["normalization"].items():
        if statistics != read_json(run / "normalization" / (frequency + ".json")):
            raise ValueError("冻结归一化统计量不同")
    notebooks = list(package.glob("*.ipynb"))
    if len(notebooks) != 1:
        raise ValueError("必须恰好包含一个推理 Notebook")
    notebook = read_json(notebooks[0])
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    if len(cells) != 2 or any(c.get("outputs") or c.get("execution_count") is not None for c in cells):
        raise ValueError("Notebook 单元格或输出不符合提交约定")
    if 'if __name__ == "__main__":' not in "".join(cells[-1]["source"]):
        raise ValueError("调试单元格缺少主程序保护")
    import dai
    recorder = RecordingDai(dai)
    previous_dai = sys.modules.get("dai")
    previous_path = sys.path[:]
    previous_modules = {name: module for name, module in sys.modules.items() if name.startswith("v4_")}
    for name in previous_modules:
        del sys.modules[name]
    def repository_code(path):
        resolved = Path(path or ".").resolve()
        return resolved.is_relative_to(ROOT) and not any(
            resolved.is_relative_to(ROOT / name) for name in (".venv", ".venv-training"))
    sys.path[:] = [path for path in sys.path if not repository_code(path)]
    sys.modules["dai"] = recorder
    isolated = tempfile.TemporaryDirectory()
    isolated_path = Path(isolated.name)
    for name in manifest["files"]:
        shutil.copyfile(package / name, isolated_path / name)
    namespace = {"__name__": "submission_import_check", "__file__": str(isolated_path / notebooks[0].name)}
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
        _, modules = namespace["_runtime"]()
        predictor = modules["v4_predict"]
        lo, hi = predictor.platform_interval(start, end)
        if set(p.name for p in isolated_path.iterdir()) != set(manifest["files"]):
            raise AssertionError("推理生成了额外附件")
        helpers = namespace["_load_modules"](bundle, ["solutions/v4/v4_reproduce.py"])
        reproduced = helpers["v4_reproduce"].restore_sources(bundle, isolated_path / "reproduced")
        for relative in bundle["training_sources"]:
            if sha256(reproduced / relative) != bundle["sources"][relative]["sha256"]:
                raise AssertionError("训练源码恢复不一致: " + relative)
        check = ("import torch,sys,json; from pathlib import Path; "
                 "sys.path.insert(0,str(Path('solutions/v4').resolve())); "
                 "from v4_prepare import validate_config; "
                 "validate_config(json.loads(Path('solutions/v4/config.json').read_text(encoding='utf-8'))); "
                 "json.loads(Path('evaluation/protocols/local-v1.json').read_text(encoding='utf-8')); "
                 "from v4_train import validate_training; "
                 "validate_training(json.loads(Path('solutions/v4/model-config.json').read_text(encoding='utf-8')),"
                 "json.loads(Path('evaluation/protocols/local-v2-ic.json').read_text(encoding='utf-8'))) ")
        subprocess.run([sys.executable, "-c", check], cwd=reproduced, check=True,
                       capture_output=True, text=True, timeout=30,
                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    finally:
        sys.modules["dai"] = previous_dai
        sys.path[:] = previous_path
        sys.modules.update(previous_modules)
        isolated.cleanup()
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
              "model_sha256": manifest["model_sha256"], "text_tensor_sha256": bundle["tensor_sha256"],
              "exact_weight_roundtrip": True, "weight_tensors": len(restored),
              "training_sources_restored": len(bundle["training_sources"]),
              "restored_ic_trainer_import_passed": True,
              "restored_preprocessing_config_validated": True,
              "isolated_two_file_inference": True, "repository_code_on_import_path": False,
              "utf8_text_only": True, "passed": True,
              "scope": "已在所列本地环境与完整请求区间验证，不代表平台运行或官方评分通过"}
    write_json(run / "submission-validation.json", report)
    manifest.update(validation_status="local_passed", verification_script_sha256=sha256(__file__), validation=report)
    write_json(manifest_path(package), manifest)
    write_json(run / "text-submission-manifest.json", manifest)
    return report


def record(run, package, report):
    index_path = ROOT / "experiments/submissions.csv"
    index = pd.read_csv(index_path, dtype=str, keep_default_na=False)
    row = {"submission_id": run.name, "run_id": run.name, "status": "local_verified_not_submitted",
           "directory": package.relative_to(ROOT).as_posix(), "manifest_sha256": sha256(manifest_path(package)),
           "platform_submission_id": None, "platform_score": None,
           "manifest_path": manifest_path(package).relative_to(ROOT).as_posix()}
    index = pd.concat([index.loc[index.submission_id != run.name], pd.DataFrame([row])], ignore_index=True)
    index["manifest_path"] = index["manifest_path"].fillna("")
    missing = index.manifest_path == ""
    index.loc[missing, "manifest_path"] = index.loc[missing, "directory"] + "/submission-manifest.json"
    index.to_csv(index_path, index=False, encoding="utf-8-sig")
    summary_path = ROOT / "experiments/v4-tcn.json"
    result = read_json(summary_path) if summary_path.is_file() else read_json(run / "comparison.json")
    if result["run_id"] != run.name:
        result = read_json(run / "comparison.json")
    manifest = read_json(manifest_path(package))
    result.update(exported=True, submission_directory=row["directory"], submission_validation=report,
                  submission_manifest_sha256=row["manifest_sha256"],
                  export_purpose=manifest["export_purpose"], export_reason=manifest["export_reason"],
                  submission_manifest_path=row["manifest_path"])
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
