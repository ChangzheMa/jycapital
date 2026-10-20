"""导出仅含 Notebook 和 JSON 的 v101 材料，保留完整权重及训练复现源码。"""

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(ROOT / "solutions/v101"))
from artifacts import now, read_json, sha256, write_json
from text_json import Float32Values, compact_numeric_json, write_compact_numeric_json
from v101_text_artifact import FORMAT, encode_state, decode_state, state_digest

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
    cells = [{"cell_type": "markdown", "id": "v101-usage", "metadata": {}, "source": description.splitlines(keepends=True)}]
    cells += [{"cell_type": "code", "execution_count": None, "id": "v101-" + name, "metadata": {},
               "outputs": [], "source": code.splitlines(keepends=True)} for name, code in (("inference", entry), ("debug", debug))]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}


def export(run, destination=None, platform_test_reason=None):
    run = Path(run).resolve()
    manifest = read_json(run / "manifest.json")
    if manifest["status"] != "complete" or manifest["solution_id"] != "v101":
        raise ValueError("只能导出已完成训练和验证／测试的 v101 运行")
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
    destination = Path(destination).resolve() if destination else ROOT / "submissions" / run.name
    if destination.exists():
        raise FileExistsError(destination)

    frozen = manifest["source"]["files"]
    core = ["v101_schema.py", "v101_preprocessing.py", "v101_prepare.py", "v101_apply.py", "v101_data.py",
            "v101_loss.py", "v101_model.py", "v101_train.py",
            "config.json", "requirements.txt", "requirements-training.txt"]
    training_files = ["solutions/v101/" + name for name in core] + [
        "common/artifacts.py", "common/protocol_compatibility.py", "common/early_stopping.py",
        "common/resident_data.py", "common/training_migration.py", "common/training_resources.py",
        "evaluation/baseline_evaluation.py",
        "evaluation/protocols/local-v1.json"]
    sources = {}
    for relative in training_files:
        if sha256(run / "source" / relative) != frozen[relative]:
            raise ValueError("冻结源码已改变: " + relative)
        sources[relative] = source_entry(run / "source" / relative)
    for relative, path in (("solutions/v101/model-config.json", run / "config.json"),
                           ("evaluation/protocols/local-v2-ic.json", run / "protocol.json")):
        sources[relative] = source_entry(path)
        training_files.append(relative)
    adapters = ("v101_predict.py", "v101_text_artifact.py", "v101_reproduce.py",
                "v101_notebook.py", "requirements-inference.txt")
    for name in adapters:
        relative = "solutions/v101/" + name
        path = run / "source" / relative
        if sha256(path) != frozen[relative]:
            raise ValueError("冻结源码已改变: " + relative)
        sources[relative] = source_entry(path)

    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    config = read_json(run / "config.json")
    if checkpoint["config"] != config:
        raise ValueError("权重配置与运行配置不一致")
    bundle = {
        "format": FORMAT, "config": config, "preprocessing": read_json(run / "source/solutions/v101/config.json"),
        "normalization": {f: read_json(run / "normalization" / (f + ".json")) for f in FREQUENCIES},
        "state_dict": encode_state(checkpoint["state_dict"]), "tensor_sha256": state_digest(checkpoint["state_dict"]),
        "sources": sources, "training_sources": training_files,
        "runtime_sources": ["solutions/v101/" + name for name in (
            "v101_schema.py", "v101_preprocessing.py", "v101_data.py", "v101_model.py", "v101_text_artifact.py", "v101_predict.py")],
        "requirements": sources["solutions/v101/requirements-inference.txt"]["text"].splitlines(),
        "metadata": {"run_id": run.name, "selection": read_json(run / "selection.json"),
                     "protocol": read_json(run / "protocol.json"), "candidate": comparison["candidate"],
                     "decision": comparison["decision"], "criterion": comparison["criterion"], "metrics": comparison["metrics"],
                     "export_purpose": purpose, "export_reason": reason,
                     "training_trajectory": manifest.get("migration"),
                     "reproduction_scope": "恢复源码后从头训练当前配置；不重放月内 FP32 到全局 FP16 的历史迁移轨迹"},
    }
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
    description = f"""# v101 多频 Patch Transformer 平台提交

上传本 Notebook 和同目录的 `v101-model.json`，共两个 UTF-8 文本文件。
运行 `{run.name}`，按验证集选定第 {manifest['selection']['best_epoch']} 轮权重。本地判定：**{assessment}**。导出理由：{reason}。
JSON 包含完整数值权重、三频统计量、配置、依赖及训练／推理源码。FP32 权重以九位有效数字保存，加载后按原 dtype 恢复，并已校验逐张量字节完全一致；统计量保留原精度。
入口为 `main(datasources, start_date, end_date)`。`datasources` 提供 `bar1m`、`bar5m`、`bar30m` 的实际行情表名。导入不查询行情；推理仅使用授权股票池和三频行情，不读取标签或训练。日期字符串的结束日包含全天，时间戳的结束分钟也包含在内。
训练目标为完整截面有限标签的 z-score（总体标准差，`ddof=0`），使用 Huber `delta=1` 和 batch 512；标准化仅用于监督目标。按验证 IC 选模；动态早停耐心为最佳 epoch 的 60% 向上取整、最少 4 轮、最多 12 轮，严格创新高即重置。
本运行的训练轨迹见 `metadata.selection` 与 `metadata.training_trajectory`。迁移运行继承旧月内 FP32 训练状态，再以全局 FP16 输入继续训练；推理保留冻结配置指定的 FP16 舍入及 FP32 极值恢复，模型计算为 FP32。
`factor_scale=1`，直接输出原始模型分数，单位为标准化目标单位，未校准为预测收益。公共评估器对原始收益标签计算的 MSE／MAE 不代表标准化目标上的拟合误差；IC／Rank IC 仍使用官方原始标签。
三路窗口为 480／240／160 根，patch 长度为 8／4／4；每路使用两层、四头、宽度 96 的 Pre-LayerNorm Transformer 和 CLS 汇总。参数量为 656,993，Dropout 为 `{config['model']['dropout']}`，推理时关闭；使用 math SDPA 后端。
推理依赖：{', '.join(bundle['requirements'])}，以及平台 `dai`。有 CUDA 时使用 GPU，否则使用 CPU；平台环境与成绩须实际验证。
训练复现：先安装 JSON 所附 `requirements-training.txt` 对应依赖，提供授权原始 Parquet，再显式调用 `reproduce("reproduced-v101", "/path/to/data", training_confirmed=True)`。
复现工作目录必须不存在；该调用会生成预处理数据并从头训练当前配置，不重放月内 FP32 到全局 FP16 的历史迁移轨迹，不保证重新得到这份权重。冻结权重可直接用于推理。仅执行下方调试单元格不会训练。
本地评估摘要与原权重 SHA-256 见 JSON 的 `metadata`；逐文件哈希及本地入口验证记录位于仓库 `experiments/{destination.name}-submission.json`，无需上传。
"""
    write_compact_numeric_json(destination / "v101-model.json", bundle,
                               max_bytes=MAX_ATTACHMENT_BYTES)
    write_json(destination / "v101-submission.ipynb", notebook(
        description, sources["solutions/v101/v101_notebook.py"]["text"]))
    report = {
        "solution_id": "v101", "run_id": run.name, "exported_at": now(), "layout": "two-text-files",
        "json_layout": "two_space_objects_inline_numeric_arrays_float32_9g",
        "weight_encoding": {"float32_significant_digits": 9,
                            "roundtrip": "bit_exact", "normalization_precision": "unchanged"},
        "max_attachment_bytes": MAX_ATTACHMENT_BYTES,
        "source_git_commit": manifest["source"]["git_commit"], "source_git_dirty": manifest["source"]["git_dirty"],
        "source_snapshot_hashes": frozen, "embedded_source_hashes": {p: v["sha256"] for p, v in sources.items()},
        "model_sha256": manifest["selection"]["model_sha256"], "tensor_sha256": bundle["tensor_sha256"],
        "comparison_sha256": sha256(run / "comparison.json"), "local_candidate": comparison["candidate"],
        "local_decision": comparison["decision"], "export_purpose": purpose, "export_reason": reason,
        "export_script_sha256": sha256(__file__),
        "notebook_template_sha256": sources["solutions/v101/v101_notebook.py"]["sha256"],
        "platform_status": "not_submitted", "validation_status": "pending",
        "attachment_bytes": {p.name: p.stat().st_size for p in sorted(destination.iterdir()) if p.is_file()},
        "files": {p.name: sha256(p) for p in sorted(destination.iterdir()) if p.is_file()},
    }
    write_json(manifest_path(destination), report)
    write_json(run / "text-submission-manifest.json", report)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--platform-test-reason", help="用户明确要求平台测试的理由；不改变本地晋级判定")
    args = parser.parse_args()
    print(export(args.run_dir, args.destination, args.platform_test_reason))
