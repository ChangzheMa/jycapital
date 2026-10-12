"""导出仅含 Notebook 和 JSON 的 v2 材料，保留完整权重及训练复现源码。"""

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(ROOT / "solutions/v2"))
from artifacts import now, read_json, sha256, write_json
from v2_text_artifact import FORMAT, encode_state, decode_state, state_digest

FREQUENCIES = ("bar1m", "bar5m", "bar15m", "bar30m")


def manifest_path(package):
    return ROOT / "experiments" / (Path(package).name + "-submission.json")


def source_entry(path):
    return {"text": Path(path).read_bytes().decode("utf-8"), "sha256": sha256(path)}


def notebook(description):
    entry = (ROOT / "solutions/v2/v2_notebook.py").read_text(encoding="utf-8")
    debug = '''if __name__ == "__main__":
    result = main(
        {f: "cpt_jyc_2026_stock_" + f for f in ["bar1m", "bar5m", "bar15m", "bar30m"]},
        "2024-12-30", "2024-12-31",
    )
    import numpy as np
    assert list(result.columns) == ["date", "instrument", "factor"]
    assert not result.empty and result["factor"].dtype == np.dtype("float64")
    assert not result.duplicated(["date", "instrument"]).any()
    assert np.isfinite(result["factor"]).all()
    print("输出行数：", len(result))
    print(result.dtypes)
    display(result.head())
'''
    cells = [{"cell_type": "markdown", "id": "v2-usage", "metadata": {}, "source": description.splitlines(keepends=True)}]
    cells += [{"cell_type": "code", "execution_count": None, "id": "v2-" + name, "metadata": {},
               "outputs": [], "source": code.splitlines(keepends=True)} for name, code in (("inference", entry), ("debug", debug))]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}


def export(run, destination=None, platform_test_reason=None):
    run = Path(run).resolve()
    manifest = read_json(run / "manifest.json")
    if manifest["status"] != "complete" or manifest["solution_id"] != "v2":
        raise ValueError("只能导出已完成训练和验证／测试的 v2 运行")
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
    core = ["v2_preprocessing.py", "v2_prepare.py", "v2_apply.py", "v2_data.py", "v2_model.py", "v2_train.py",
            "config.json", "requirements.txt", "requirements-training.txt"]
    training_files = ["solutions/v2/" + name for name in core] + ["common/artifacts.py", "evaluation/baseline_evaluation.py"]
    sources = {}
    for relative in training_files:
        if sha256(run / "source" / relative) != frozen[relative]:
            raise ValueError("冻结源码已改变: " + relative)
        sources[relative] = source_entry(run / "source" / relative)
    for name in ("v2_model.py", "v2_data.py", "v2_preprocessing.py"):
        if sha256(ROOT / "solutions/v2" / name) != frozen["solutions/v2/" + name]:
            raise ValueError("当前推理数值核心与训练快照不同")
    for relative, path in (("solutions/v2/model-config.json", run / "config.json"),
                           ("evaluation/protocols/local-v1.json", run / "protocol.json")):
        sources[relative] = source_entry(path)
        training_files.append(relative)
    adapters = ("v2_predict.py", "v2_text_artifact.py", "v2_reproduce.py")
    for name in adapters:
        sources["solutions/v2/" + name] = source_entry(ROOT / "solutions/v2" / name)

    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    config = read_json(run / "config.json")
    if checkpoint["config"] != config:
        raise ValueError("权重配置与运行配置不一致")
    bundle = {
        "format": FORMAT, "config": config, "preprocessing": read_json(run / "source/solutions/v2/config.json"),
        "normalization": {f: read_json(run / "normalization" / (f + ".json")) for f in FREQUENCIES},
        "state_dict": encode_state(checkpoint["state_dict"]), "tensor_sha256": state_digest(checkpoint["state_dict"]),
        "sources": sources, "training_sources": training_files,
        "runtime_sources": ["solutions/v2/" + name for name in (
            "v2_preprocessing.py", "v2_data.py", "v2_model.py", "v2_text_artifact.py", "v2_predict.py")],
        "requirements": (ROOT / "solutions/v2/requirements-inference.txt").read_text(encoding="utf-8").splitlines(),
        "metadata": {"run_id": run.name, "selection": read_json(run / "selection.json"),
                     "protocol": read_json(run / "protocol.json"), "candidate": comparison["candidate"],
                     "decision": comparison["decision"], "criterion": comparison["criterion"], "metrics": comparison["metrics"],
                     "export_purpose": purpose, "export_reason": reason},
    }
    restored = decode_state(bundle)
    for name, original in checkpoint["state_dict"].items():
        if original.dtype != restored[name].dtype or not torch.equal(original.cpu(), restored[name]):
            raise ValueError("文本转换改变权重: " + name)
    assessment = "达到本地改善条件" if comparison["candidate"] else "未达到本地改善条件"
    description = f"""# v2 四频 TCN 平台测试

上传本 Notebook 和同目录的 `v2-model.json`，共两个 UTF-8 文本文件。
运行 `{run.name}`，按验证集选定第 {manifest['selection']['best_epoch']} 轮权重。本地判定：**{assessment}**。导出理由：{reason}。
JSON 包含完整数值权重、四频统计量、配置、依赖及训练／推理源码；加载后按原 dtype 恢复，不截断小数精度。
入口为 `main(datasources, start_date, end_date)`。导入不查询行情；推理仅使用授权股票池和四频行情，不读取标签或训练。
推理依赖：{', '.join(bundle['requirements'])}，以及平台 `dai`。有 CUDA 时使用 GPU，否则使用 CPU；平台环境与成绩须实际验证。
训练复现：先安装 JSON 所附 `requirements-training.txt` 对应依赖，提供授权原始 Parquet，再显式调用 `reproduce("reproduced-v2", "/path/to/data")`。
复现工作目录必须不存在；该调用会生成预处理数据并运行正式训练。仅执行下方调试单元格不会训练。
本地评估摘要与原权重 SHA-256 见 JSON 的 `metadata`；逐文件哈希及本地入口验证记录位于仓库 `experiments/{destination.name}-submission.json`，无需上传。
"""
    destination.mkdir(parents=True)
    write_json(destination / "v2-model.json", bundle)
    write_json(destination / "v2-submission.ipynb", notebook(description))
    report = {
        "solution_id": "v2", "run_id": run.name, "exported_at": now(), "layout": "two-text-files",
        "source_git_commit": manifest["source"]["git_commit"], "source_git_dirty": manifest["source"]["git_dirty"],
        "source_snapshot_hashes": frozen, "embedded_source_hashes": {p: v["sha256"] for p, v in sources.items()},
        "model_sha256": manifest["selection"]["model_sha256"], "tensor_sha256": bundle["tensor_sha256"],
        "comparison_sha256": sha256(run / "comparison.json"), "local_candidate": comparison["candidate"],
        "local_decision": comparison["decision"], "export_purpose": purpose, "export_reason": reason,
        "export_script_sha256": sha256(__file__), "notebook_template_sha256": sha256(ROOT / "solutions/v2/v2_notebook.py"),
        "platform_status": "not_submitted", "validation_status": "pending",
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
