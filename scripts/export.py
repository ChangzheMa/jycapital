"""从完成的运行快照导出可由 Git 管理的平铺提交目录。"""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from artifacts import now, read_json, sha256, write_json


def build_notebook():
    source = (
        "def main(datasources, start_date, end_date):\n"
        "    from v1_predict import main as predict_main\n"
        "    return predict_main(datasources, start_date, end_date)\n"
    )
    debug_source = '''if __name__ == "__main__":
    import numpy as np

    factor_data = main(
        datasources={"bar5m": "cpt_jyc_2026_stock_bar5m"},
        start_date="2024-12-30",
        end_date="2024-12-31",
    )

    assert list(factor_data.columns) == ["date", "instrument", "factor"]
    assert not factor_data.empty
    assert factor_data["factor"].dtype == np.dtype("float64")
    assert not factor_data.duplicated(["date", "instrument"]).any()
    assert np.isfinite(factor_data["factor"]).all()

    print("输出行数：", len(factor_data))
    print(factor_data.dtypes)
    display(factor_data.head())
'''
    return {
        "cells": [
            {"cell_type": "code", "execution_count": None, "id": "v1-inference", "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)},
            {"cell_type": "code", "execution_count": None, "id": "v1-debug", "metadata": {}, "outputs": [], "source": debug_source.splitlines(keepends=True)},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def export(run, destination=None):
    run = Path(run).resolve()
    destination = Path(destination).resolve() if destination is not None else ROOT / "submissions" / run.name
    manifest = read_json(run / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("只能导出已完成训练和评估的运行")
    if sha256(run / "v1-model.json") != manifest["selection"]["model_sha256"]:
        raise ValueError("所选模型已被修改")
    for relative, digest in manifest["source"]["files"].items():
        if sha256(run / "source" / relative) != digest:
            raise ValueError(f"源码快照已被修改: {relative}")
    destination.mkdir(parents=True, exist_ok=False)
    sources = [
        "solutions/v1/v1_data.py", "solutions/v1/v1_model.py",
        "solutions/v1/v1_train.py", "solutions/v1/v1_predict.py",
        "solutions/v1/requirements.txt", "solutions/v1/requirements-training.txt",
        "evaluation/baseline_evaluation.py", "common/artifacts.py",
    ]
    for relative in sources:
        shutil.copy2(run / "source" / relative, destination / Path(relative).name)
    for name in ["v1-model.json", "config.json", "protocol.json", "selection.json"]:
        shutil.copy2(run / name, destination / name)
    # 模型逻辑来自冻结源码；末尾调试单元格只在交互运行时执行。
    write_json(destination / "v1-submission.ipynb", build_notebook())
    readme = f"""# v1 平台提交材料

来源运行：`{run.name}`。生成日期：{now()}。

本目录作为 `submissions/` 下的完整提交材料随 Git 推送。拉取仓库后，将本目录文件放在平台同一目录，选择唯一的 `v1-submission.ipynb` 作为推理 Notebook，并附带全部 `.py`、模型、配置和依赖文件。平台附件选择方式以界面为准。

## 文件用途

- `v1-submission.ipynb`：`main(datasources, start_date, end_date)` 入口。
- `v1_predict.py`：读取逻辑行情源 `datasources["bar5m"]` 并生成三列输出。
- `v1_data.py`：v1 专属因果切窗与归一化。
- `v1_model.py`、`v1-model.json`：网络定义、完整权重及冻结配置；NumPy 推理。
- `v1_train.py`：训练复现入口；公榜推理不调用训练。
- `artifacts.py`、`baseline_evaluation.py`：训练记录与本地评估支持。
- `config.json`、`protocol.json`、`selection.json`：随机种子、超参数、日期划分和权重选择依据。
- `requirements.txt`：推理依赖；DAI 使用平台内置版本。
- `requirements-training.txt`：训练附加依赖；完整本地环境记录保存在来源运行目录。
- `submission-manifest.json`：运行来源与提交文件哈希。

## 平台运行

在平台已提供 NumPy、Pandas、DAI 的环境中，从本目录调用 Notebook 的 `main`。推理不需要 PyTorch、不联网、不读取标签。不得上传本地 `dai.py` 或行情文件。

Notebook 末尾包含带 `if __name__ == "__main__":` 保护的调试单元格。交互运行时对 2024-12-30 至 2024-12-31 执行推理，检查输出列、`float64` 类型、非空、有限值和键唯一性，显示行数、类型及前五行。作为模块导入时不执行该调试调用。提交的 Notebook 不保存执行输出。

模型使用 5 分钟原始 OHLC、volume、amount 的最近 12 条常规交易 Bar。历史不足时前补归一化后的零；完全无行情历史时也通过同一网络得到有限输出。价格按窗口最后有效 close 归一化，成交量和金额分别按窗口有效均值归一化。没有人工技术指标、外部权重或推理期拟合。

训练复现命令（需已具备所需依赖和授权数据）：

```bash
python v1_train.py --run-dir reproduced-v1 --config config.json --protocol protocol.json --bar-table cpt_jyc_2026_stock_bar5m
```

训练 2020—2023 年，验证 2024 年上半年，测试 2024 年下半年。标签直接使用 `vwap_return`；本地 IC/Rank IC 不包含平台 BARRA 处理，不等同于官方分数。

本提交目录由本地训练与评估产生；平台运行、附件识别和正式提交结果待用户实际提交确认。
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")
    files = {p.name: sha256(p) for p in sorted(destination.iterdir()) if p.is_file()}
    write_json(destination / "submission-manifest.json", {
        "solution_id": "v1", "run_id": run.name, "exported_at": now(),
        "layout": "flat-directory",
        "source_git_commit": manifest["source"]["git_commit"],
        "source_git_dirty": manifest["source"]["git_dirty"],
        "source_snapshot_hashes": manifest["source"]["files"],
        "config_sha256": manifest["config_sha256"], "protocol_sha256": manifest["protocol_sha256"],
        "data_manifest_sha256": sha256(run / "data-manifest.json"),
        "export_script_sha256": sha256(__file__), "files": files,
        "platform_status": "not_submitted",
    })
    print(json.dumps({"directory": str(destination), "manifest_sha256": sha256(destination / "submission-manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--destination", help="默认为仓库根目录的 submissions/<run_id>")
    args = parser.parse_args()
    export(args.run_dir, args.destination)
