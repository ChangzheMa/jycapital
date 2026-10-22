"""导出仅含 Notebook 和 JSON 的 v1601 材料，保留完整权重及训练复现源码。"""

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(ROOT / "solutions/v1601"))
from artifacts import now, read_json, sha256, write_json
from text_json import Float32Values, compact_numeric_json, write_compact_numeric_json
from v1601_text_artifact import FORMAT, encode_state, decode_state, state_digest

FREQUENCIES = ("bar1m", "bar5m", "bar30m")
MAX_ATTACHMENT_BYTES = 50_000_000


def manifest_path(package):
    return ROOT / "experiments" / (Path(package).name + "-submission.json")


def source_entry(path):
    return {"text": Path(path).read_bytes().decode("utf-8"), "sha256": sha256(path)}


def notebook(description, entry):
    debug = '''if __name__ == "__main__":
    result = main(
        {f: "cpt_jyc_2026_stock_" + f for f in ["bar1m", "bar5m", "bar30m"]},
        "2024-12-30", "2024-12-31",
    )
    import numpy as np
    import pandas as pd
    assert list(result.columns) == ["date", "instrument", "factor"]
    assert not result.empty and result["factor"].dtype == np.dtype("float64")
    assert pd.api.types.is_datetime64_any_dtype(result["date"])
    assert result["instrument"].map(lambda value: isinstance(value, str)).all()
    assert not result[["date", "instrument"]].isna().any().any()
    assert not result.duplicated(["date", "instrument"]).any()
    assert np.isfinite(result["factor"]).all()
    print("输出行数：", len(result))
    print(result.dtypes)
    display(result.head())
'''
    cells = [{"cell_type": "markdown", "id": "v1601-usage", "metadata": {}, "source": description.splitlines(keepends=True)}]
    cells += [{"cell_type": "code", "execution_count": None, "id": "v1601-" + name, "metadata": {},
               "outputs": [], "source": code.splitlines(keepends=True)} for name, code in (("inference", entry), ("debug", debug))]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}


def export(run, destination=None, platform_test_reason=None, *, fast_inference=True):
    if not fast_inference:
        raise ValueError("v1601 平台材料固定使用 fast 执行路径")
    run = Path(run).resolve()
    manifest = read_json(run / "manifest.json")
    if manifest["status"] != "complete" or manifest["solution_id"] != "v1601":
        raise ValueError("只能导出已完成训练和验证／测试的 v1601 运行")
    comparison = read_json(run / "comparison.json")
    if platform_test_reason is not None:
        platform_test_reason = platform_test_reason.strip()
        if not platform_test_reason:
            raise ValueError("平台测试导出须注明用户要求及测试目的")
    if not comparison["candidate"] and platform_test_reason is None:
        raise ValueError("默认只导出改善候选；用户明确要求平台测试时须提供 --platform-test-reason")
    purpose = "user_requested_platform_test" if platform_test_reason else "local_improvement"
    reason = platform_test_reason or "本地验证主指标提高且覆盖未降低"
    if sha256(run / "best.pt") != manifest["selection"]["model_sha256"]:
        raise ValueError("所选权重已改变")
    if comparison["config_sha256"] != sha256(run / "config.json") or comparison["data_manifest_sha256"] != sha256(run / "data-manifest.json"):
        raise ValueError("配置或输入记录已改变")
    reference = read_json(run / "data-manifest.json")["preprocessing"]["train"]
    training = Path(reference["path"])
    if sha256(training / "manifest.json") != reference["manifest_sha256"]:
        raise ValueError("训练预处理来源记录已改变")
    if sha256(training / "summary.json") != read_json(training / "manifest.json")["summary_sha256"]:
        raise ValueError("训练预处理摘要已改变")
    statistics = read_json(training / "summary.json")["frequencies"]
    for frequency in FREQUENCIES:
        if sha256(run / "normalization" / (frequency + ".json")) != statistics[frequency]["statistics_sha256"]:
            raise ValueError("冻结归一化参数已改变")
    package_name = run.name + "-fast" if fast_inference else run.name
    destination = Path(destination).resolve() if destination else ROOT / "submissions" / package_name
    if destination.exists():
        raise FileExistsError(destination)

    frozen = manifest["source"]["files"]
    training_files = sorted(relative for relative in frozen
        if relative.startswith("solutions/v1601/") and relative != "solutions/v1601/model-config.json")
    training_files += ["common/artifacts.py", "common/protocol_compatibility.py", "common/early_stopping.py",
        "common/training_resources.py", "common/price_geometry.py", "common/geometry_resident.py",
        "common/geometry_evaluation_cache.py", "evaluation/baseline_evaluation.py", "evaluation/protocols/local-v1.json"]
    sources = {}
    for relative in training_files:
        if sha256(run / "source" / relative) != frozen[relative]:
            raise ValueError("冻结源码已改变: " + relative)
        sources[relative] = source_entry(run / "source" / relative)
    for relative, path in (("solutions/v1601/model-config.json", run / "config.json"),
                           ("evaluation/protocols/local-v2-ic.json", run / "protocol.json")):
        sources[relative] = source_entry(path)
        training_files.append(relative)
    adapters = ("v1601_predict.py", "v1601_text_artifact.py", "v1601_reproduce.py",
                "v1601_notebook.py", "requirements-inference.txt")
    for name in adapters:
        relative = "solutions/v1601/" + name
        path = run / "source" / relative
        if sha256(path) != frozen[relative]:
            raise ValueError("冻结源码已改变: " + relative)
        sources[relative] = source_entry(path)

    runtime_overrides = {}
    inference_execution = {"method": "original_per_sample"}
    if fast_inference:
        for embedded, actual in (("v1601_predict.py", "v1601_predict_fast.py"), ("v1601_fast.py", "v1601_fast.py")):
            relative = "solutions/v1601/" + embedded
            actual_path = run / "source/solutions/v1601" / actual
            if sha256(actual_path) != frozen["solutions/v1601/" + actual]:
                raise ValueError("冻结 fast 源码已改变: " + actual)
            sources[relative] = source_entry(actual_path)
            runtime_overrides[relative] = {"source_path": actual_path.relative_to(ROOT).as_posix(),
                                           "sha256": sources[relative]["sha256"],
                                           "frozen_source_sha256": frozen.get(relative)}
        inference_execution = {"method": "shared_stock_day_encoding", "stock_batch_size": 128,
                               "model_precision": "float32", "grid_stocks": "requested_instruments_only",
                               "normalization_cross_sections": "unchanged_full_source_and_original_pool"}

    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    config = read_json(run / "config.json")
    if checkpoint["config"] != config:
        raise ValueError("权重配置与运行配置不一致")
    if checkpoint.get("weight_source") != "ema" or manifest["selection"].get("evaluation_weights") != "ema":
        raise ValueError("v1601 所选权重必须是验证选中的 EMA 参数")
    if (checkpoint.get("epoch") != manifest["selection"]["best_epoch"]
            or type(checkpoint.get("ema_updates")) is not int or checkpoint["ema_updates"] < 1):
        raise ValueError("EMA checkpoint 的选中轮次或更新次数不完整")
    bundle = {
        "format": FORMAT, "config": config, "preprocessing": read_json(run / "source/solutions/v1601/config.json"),
        "normalization": {f: read_json(run / "normalization" / (f + ".json")) for f in FREQUENCIES},
        "state_dict": encode_state(checkpoint["state_dict"]), "tensor_sha256": state_digest(checkpoint["state_dict"]),
        "sources": sources, "training_sources": training_files,
        "runtime_sources": ["common/price_geometry.py"] + ["solutions/v1601/" + name for name in (
            "v1601_schema.py", "v1601_preprocessing.py", "v1601_data.py", "v1601_model.py", "v1601_text_artifact.py", "v1601_predict.py")],
        "requirements": sources["solutions/v1601/requirements-inference.txt"]["text"].splitlines(),
        "metadata": {"run_id": run.name, "selection": read_json(run / "selection.json"),
                     "protocol": read_json(run / "protocol.json"), "candidate": comparison["candidate"],
                     "decision": comparison["decision"], "criterion": comparison["criterion"], "metrics": comparison["metrics"],
                     "export_purpose": purpose, "export_reason": reason,
                     "training_trajectory": "fresh_structured_fields_shared_price_geometry_stock_day",
                     "weight_averaging": config["training"]["weight_averaging"],
                     "selected_weight_source": checkpoint["weight_source"],
                     "selected_ema_updates": checkpoint["ema_updates"],
                     "reproduction_scope": "恢复冻结源码后从头全局训练当前配置；本运行没有继承其他模型权重"},
    }
    if fast_inference:
        bundle["runtime_sources"].insert(-1, "solutions/v1601/v1601_fast.py")
        bundle["inference_execution"] = inference_execution
        bundle["metadata"]["runtime_overrides"] = runtime_overrides
    for item in bundle["state_dict"].values():
        if item["dtype"] == "float32":
            item["values"] = Float32Values(item["values"])
    # 经由实际 JSON 文本和标准解析器恢复，验证九位有效数字的位精确往返。
    restored = decode_state(json.loads(compact_numeric_json({
        "format": FORMAT, "state_dict": bundle["state_dict"],
    })))
    for name, original in checkpoint["state_dict"].items():
        if original.dtype != restored[name].dtype or not torch.equal(original.cpu(), restored[name]):
            raise ValueError("文本转换改变权重: " + name)
    if state_digest(restored) != bundle["tensor_sha256"]:
        raise ValueError("文本转换改变权重位模式")
    assessment = "达到本地改善条件" if comparison["candidate"] else "未达到本地改善条件"
    description = f"""# v1601 共同价格坐标模型 · fast 平台材料

上传本 Notebook 和同目录的 `v1601-model.json`，共两个 UTF-8 文本文件，保留文件名。
运行 `{run.name}`，固定使用验证选中的第 {checkpoint['epoch']} 轮 EMA 权重；本地判定：**{assessment}**。导出理由：{reason}。

入口为 `main(datasources, start_date, end_date)`，`datasources` 提供 `bar1m`、`bar5m`、`bar30m` 的实际行情表名。返回 `date、instrument、factor`，覆盖请求股票池。日期字符串的结束日包含全天，时间戳的结束分钟也包含在内。导入不查询行情；下方受保护的调试单元格可用于短区间试运行。

本版保留原 z 与数量 c，在同一已闭合 Bar 内使用共同价格中心表达价格字段，再由结构编码 GRU 和行情条件分支生成因子。使用股票日共享编码和当前因果前缀；模型参数为 2,518,992。输入精度遵循冻结配置，模型计算为 FP32，输出为标准化目标单位的分数。

JSON 包含完整数值权重、归一化统计、配置、依赖及训练／推理源码。FP32 权重用九位有效数字保存，实际文本恢复已逐张量核对；统计量保留原精度。推理读取授权行情与股票池，训练标签直接使用已提供的 vwap_return，模型不依赖 exposure。

推理依赖：{', '.join(bundle['requirements'])}，以及平台 `dai`。有 CUDA 时使用 GPU，否则使用 CPU。正式本地指标及选择依据保存在 JSON 的 metadata；本地验证范围与平台运行状态分别记录。

需要从头复现训练时，安装 JSON 所附训练依赖，提供授权原始 Parquet，并显式调用 `reproduce("reproduced-v1601", "/path/to/data", training_confirmed=True)`；该工作目录必须不存在。正常导入、评测入口和调试单元格均不启动训练。
"""
    write_compact_numeric_json(destination / "v1601-model.json", bundle,
                               max_bytes=MAX_ATTACHMENT_BYTES)
    write_json(destination / "v1601-submission.ipynb", notebook(
        description, sources["solutions/v1601/v1601_notebook.py"]["text"]))
    report = {
        "solution_id": "v1601", "run_id": run.name, "exported_at": now(), "layout": "two-text-files",
        "json_layout": "two_space_objects_inline_numeric_arrays_float32_9g",
        "weight_encoding": {"float32_significant_digits": 9,
                            "roundtrip": "bit_exact", "normalization_precision": "unchanged"},
        "max_attachment_bytes": MAX_ATTACHMENT_BYTES,
        "source_git_commit": manifest["source"]["git_commit"], "source_git_dirty": manifest["source"]["git_dirty"],
        "source_snapshot_hashes": frozen, "embedded_source_hashes": {p: v["sha256"] for p, v in sources.items()},
        "model_sha256": manifest["selection"]["model_sha256"], "tensor_sha256": bundle["tensor_sha256"],
        "selected_weight_source": checkpoint["weight_source"], "selected_ema_updates": checkpoint["ema_updates"],
        "weight_averaging": config["training"]["weight_averaging"],
        "comparison_sha256": sha256(run / "comparison.json"), "local_candidate": comparison["candidate"],
        "local_decision": comparison["decision"], "export_purpose": purpose, "export_reason": reason,
        "export_script_sha256": sha256(__file__),
        "notebook_template_sha256": sources["solutions/v1601/v1601_notebook.py"]["sha256"],
        "inference_execution": inference_execution, "runtime_overrides": runtime_overrides,
        "platform_status": "not_submitted", "validation_status": "pending",
        "attachment_bytes": {p.name: p.stat().st_size for p in sorted(destination.iterdir()) if p.is_file()},
        "files": {p.name: sha256(p) for p in sorted(destination.iterdir()) if p.is_file()},
    }
    write_json(manifest_path(destination), report)
    write_json(run / "text-submission-manifest.json", report)
    return destination


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--platform-test-reason", help="用户明确要求平台测试的理由；不改变本地晋级判定")
    parser.add_argument("--fast-inference", action="store_true", default=True,
                        help="固定采用股票日共享编码的 fast 路径")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(export(args.run_dir, args.destination, args.platform_test_reason, fast_inference=args.fast_inference))
