"""导出 v2 改善候选，或用户明确要求的平台测试材料。"""

import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from artifacts import now, read_json, sha256, write_json


def notebook():
    entry = "def main(datasources, start_date, end_date):\n    from v2_predict import main as predict_main\n    return predict_main(datasources, start_date, end_date)\n"
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
    return {"cells": [
        {"cell_type": "code", "execution_count": None, "id": "v2-" + name, "metadata": {}, "outputs": [], "source": code.splitlines(keepends=True)}
        for name, code in (("inference", entry), ("debug", debug))
    ], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
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
    export_purpose = "user_requested_platform_test" if platform_test_reason else "local_improvement"
    export_reason = platform_test_reason or "本地验证主指标提高且覆盖未降低"
    if sha256(run / "best.pt") != manifest["selection"]["model_sha256"]:
        raise ValueError("所选权重已改变")
    if comparison["config_sha256"] != sha256(run / "config.json") or comparison["data_manifest_sha256"] != sha256(run / "data-manifest.json"):
        raise ValueError("配置或输入记录已改变")
    reference = read_json(run / "data-manifest.json")["preprocessing"]["train"]
    training = Path(reference["path"])
    if sha256(training / "manifest.json") != reference["manifest_sha256"]:
        raise ValueError("训练预处理来源记录已改变")
    training_manifest = read_json(training / "manifest.json")
    if sha256(training / "summary.json") != training_manifest["summary_sha256"]:
        raise ValueError("训练预处理摘要已改变")
    statistics = read_json(training / "summary.json")["frequencies"]
    for frequency in ("bar1m", "bar5m", "bar15m", "bar30m"):
        if sha256(run / "normalization" / (frequency + ".json")) != statistics[frequency]["statistics_sha256"]:
            raise ValueError("冻结归一化参数已改变")
    destination = Path(destination).resolve() if destination else ROOT / "submissions" / run.name
    if destination.exists():
        raise FileExistsError(destination)
    frozen = manifest["source"]["files"]
    core = ["v2_preprocessing.py", "v2_prepare.py", "v2_apply.py", "v2_data.py", "v2_model.py", "v2_train.py"]
    sources = ["solutions/v2/" + name for name in core] + ["common/artifacts.py", "evaluation/baseline_evaluation.py"]
    for relative in sources + ["solutions/v2/config.json", "solutions/v2/requirements.txt"]:
        if sha256(run / "source" / relative) != frozen[relative]:
            raise ValueError("冻结源码已改变: " + relative)
    # 推理接口在训练后独立冻结；模型、切窗、归一化仍逐字使用训练时源码。
    for name in ("v2_model.py", "v2_data.py", "v2_preprocessing.py"):
        relative = "solutions/v2/" + name
        if sha256(ROOT / relative) != frozen[relative]:
            raise ValueError("当前推理适配器所依赖的数值实现与训练快照不同")
    inference = run / "inference-source"
    inference.mkdir(exist_ok=True)
    adapter_hashes = {}
    for name in ("v2_predict.py", "v2_reproduce.py", "requirements-inference.txt"):
        source, target = ROOT / "solutions/v2" / name, inference / name
        if target.exists() and sha256(target) != sha256(source):
            raise ValueError("已冻结的推理接口不同，须先核查已有导出来源")
        shutil.copyfile(source, target)
        adapter_hashes[name] = sha256(target)
    write_json(inference / "manifest.json", {"frozen_at": now(), "files": adapter_hashes, "trained_core_source": {r: frozen[r] for r in sources}})
    destination.mkdir(parents=True)
    for relative in sources:
        shutil.copyfile(run / "source" / relative, destination / Path(relative).name)
    for name in ("v2_predict.py", "v2_reproduce.py"):
        shutil.copyfile(inference / name, destination / name)
    for source, name in ((run / "best.pt", "best.pt"), (run / "config.json", "model-config.json"),
                         (run / "protocol.json", "protocol.json"), (run / "selection.json", "selection.json"),
                         (run / "source/solutions/v2/config.json", "preprocessing-config.json"),
                         (run / "source/solutions/v2/requirements.txt", "requirements-preprocessing.txt"),
                         (inference / "requirements-inference.txt", "requirements.txt")):
        shutil.copyfile(source, destination / name)
    for path in sorted((run / "normalization").glob("bar*.json")):
        shutil.copyfile(path, destination / ("normalization-" + path.name))
    if len(list(destination.glob("normalization-bar*.json"))) != 4:
        raise ValueError("四频归一化参数不完整")
    (destination / "requirements-training.txt").write_text("-r requirements.txt\n-r requirements-preprocessing.txt\nmatplotlib>=3.8,<4\n", encoding="utf-8")
    write_json(destination / "v2-submission.ipynb", notebook())
    write_json(destination / "local-evaluation.json", {
        "run_id": run.name, "candidate": comparison["candidate"], "decision": comparison["decision"],
        "criterion": comparison["criterion"], "metrics": comparison["metrics"],
        "selection": read_json(run / "selection.json"),
        "export_purpose": export_purpose, "export_reason": export_reason,
    })
    assessment = "达到本地改善条件" if comparison["candidate"] else "未达到本地改善条件"
    readme = f"""# v2 四频 TCN 平台材料

来源运行：`{run.name}`。本地判定：**{assessment}**。导出用途：`{export_purpose}`；理由：{export_reason}。本地结论与指标保存在 `local-evaluation.json`，不因导出而改变；这不是官方平台成绩。

将本目录所有文件放在平台同一目录，选择唯一的 `v2-submission.ipynb`。不要上传本地 `dai.py` 或行情。推理使用平台 DAI 和 `datasources` 中的四种频率映射，不联网、不训练、不读取收益标签。

`best.pt` 为冻结权重与模型配置；四个 `normalization-bar*.json` 和 `preprocessing-config.json` 为冻结预处理参数。`v2_model.py`、`v2_data.py`、`v2_preprocessing.py` 与正式训练时源码一致。`v2_predict.py` 是独立冻结的原始行情推理接口，验证结果以 `submission-validation.json` 为准。Notebook 末尾有带主程序保护的两日调试单元格；作为模块导入不执行行情查询，文件中不保存执行输出。

每只股票使用 1／5／15／30 分钟的 60／24／16／16 个常规交易时间槽，72 个 z/c 值、72 个缺失掩码以及时间编码，四分支因果 TCN 输出一个因子。缺参股票按截至该时点的可用历史更新统计量。推理开始前只为缺参股票查询训练期之后的历史，分日处理完整截面以限制内存。

推理依赖见 `requirements.txt`，支持 CUDA 时使用 GPU，否则使用 CPU；实际平台依赖和性能须在平台验证。训练附加依赖见 `requirements-training.txt`，复现需先准备授权原始 Parquet 数据。

```bash
python v2_reproduce.py --workspace reproduced-v2 --data-root /path/to/authorized/data
```

该命令要求一个尚不存在的工作目录，按原仓库结构还原所附源码，重新拟合 2020—2023 年统计量、生成验证／测试输入并进行一次正式训练。产生大量本地中间数据，需要充足磁盘与训练资源。跨硬件或依赖版本不保证逐位相同。

`model-config.json`、`protocol.json`、`selection.json` 记录 seed、超参数、日期和权重选择依据。`submission-manifest.json` 包含全部文件哈希；`submission-validation.json` 记录实际完成的本地验证范围。比赛平台由用户提交，平台运行和官方评分另行确认。
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")
    write_json(destination / "submission-manifest.json", {
        "solution_id": "v2", "run_id": run.name, "exported_at": now(), "layout": "flat-directory",
        "source_git_commit": manifest["source"]["git_commit"], "source_git_dirty": manifest["source"]["git_dirty"],
        "source_snapshot_hashes": frozen, "inference_source_hashes": adapter_hashes,
        "model_sha256": manifest["selection"]["model_sha256"], "comparison_sha256": sha256(run / "comparison.json"),
        "local_candidate": comparison["candidate"], "local_decision": comparison["decision"],
        "export_purpose": export_purpose, "export_reason": export_reason,
        "export_script_sha256": sha256(__file__), "platform_status": "not_submitted",
        "validation_status": "pending", "files": {p.name: sha256(p) for p in sorted(destination.iterdir()) if p.is_file()},
    })
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--platform-test-reason", help="仅在用户明确要求平台测试时填写；保留原本地判定及完整性校验")
    args = parser.parse_args()
    print(export(args.run_dir, args.destination, args.platform_test_reason))
