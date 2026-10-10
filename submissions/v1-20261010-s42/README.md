# v1 平台提交材料

来源运行：`v1-20261010-s42`。生成日期：2026-10-10T03:38:43+08:00。

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

本提交目录由本地训练与评估产生。用户于 2026-10-10（UTC+08:00）确认 v1 在平台可运行；正式模型提交是否通过及官方评分尚未确认。
