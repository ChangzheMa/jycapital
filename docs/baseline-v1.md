# v1：原始量价序列 MLP 基线

[返回目录](README.md) · [通用开发与评估协议](development.md)

## 方法

代码位于 [`solutions/v1/`](../solutions/v1/)，完整参数见 [config.json](../solutions/v1/config.json)。

| 项目 | 设置 |
| --- | --- |
| 输入 | `bar5m` 的 `open`、`high`、`low`、`close`、`volume`、`amount` |
| 窗口 | 预测时点已结束的最近 12 条常规交易 Bar，最多回看 20 个自然日 |
| 网络 | 展平 72 维 → Linear(64) → ReLU → Dropout(0.1) → Linear(32) → ReLU → Dropout(0.1) → Linear(1) |
| 参数数 | 6,785 |
| 标签 | 官方表 `vwap_return`，不自行重算 |
| 训练目标 | 标签仅在训练损失中裁剪至 `[-0.1, 0.1]` 并乘 100，Huber loss，delta=1 |
| 优化器 | AdamW，学习率 0.001，weight decay 0.0001 |
| 批量/种子 | 8,192 / 42 |
| 停止与选择 | 最多 12 轮，验证 Rank IC 连续 3 轮未提升至少 0.00001 则早停；保留最佳验证权重 |
| 输出 | 原始收益尺度的有限 `float64`，列为 `date`、`instrument`、`factor` |

网络使用原始量价窗口，价格和成交量之间的关系由网络学习。模型不读取风险暴露，不输入人工技术指标，不使用预训练权重。数据清洗、归一化、切窗均绑定 `v1_data.py`，后续方案可独立修改。

## 数据处理

- 只接收已结束、时间不晚于预测时点的 Bar；按实际 Bar 序列切窗，午休和非交易日不会生成虚假 Bar。
- 输入排除 09:25 等竞价 Bar，保留上午 `(09:30, 11:30]` 和下午 `(13:00, 15:00]` 的常规 Bar；输出时点仍来自股票池原始记录。
- 非有限值、非正价格及负成交量/金额视为缺失，不向未来填充。
- OHLC 统一除以窗口最后一个有效 close 再减一；成交量和金额分别除以各自窗口内的有效均值再减一。
- 所有归一化统计只依赖当前历史窗口，无需在验证/测试期间拟合参数。变换后裁剪至 `[-10, 10]`，缺失及不足历史填 0。
- 完全无历史的键也使用同一网络的零输入预测，不按标签或预测目标筛选股票；另行记录无历史样本数。这能提供有限输出，但不表示这些样本包含有效预测信息。
- 只用有限标签训练。评估保持原始标签，不套用训练损失中的裁剪。

## 运行

从仓库根目录执行。现有本地行情通过根目录 `dai.py` 读取：

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-local.txt
.venv/Scripts/python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/python.exe -m pip install matplotlib==3.10.7 nbformat==5.10.4
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -u solutions/v1/v1_train.py --run-dir outputs/runs/v1-new-run
```

CUDA 安装命令依据 [PyTorch 官方版本安装说明](https://pytorch.org/get-started/previous-versions/)，获取日期 2026-10-10（UTC+08:00）。无 CUDA 时可安装对应 CPU 构建，代码会使用 CPU。PyTorch 仅用于训练和权重一致性检查；平台推理由 NumPy 执行。

支持 `--prepare-only` 单独准备训练/验证缓存，随后对同一目录运行不带此参数的命令开始训练。继续前校验源码、配置、数据源及数据文件哈希；开始训练后禁止复用该目录重新训练。

本次实际训练使用 `.venv-training/Scripts/python.exe`。该项目环境复用系统已安装的 PyTorch 2.11.0 CUDA 13.0 构建，并在环境内安装所需数据处理和记录依赖；具体生效版本以运行目录的 `environment.json` 为准。上面的独立安装方式用于新环境复现。

本地研究划分：2020—2023 年训练，2024 年上半年验证，2024 年下半年测试。测试数据在权重选择完成后才进入运行缓存和评估；文件指纹登记不用于模型选择。

## 本次本地结果

运行标识：`v1-20261010-s42`，完成日期 2026-10-10（UTC+08:00）。种子 42，在 RTX 4080 上训练；第 11 轮早停，按验证 Rank IC 选中第 8 轮。仅进行了这一个正式训练运行，未做多种子汇总。

| 划分 | 有效标签样本 | IC | Rank IC | IC_IR（不年化） | 最低截面覆盖率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 训练 | 7,742,958 | 0.084225 | 0.080082 | 0.778969 | 100% |
| 验证 | 935,224 | 0.070445 | 0.061333 | 0.566824 | 100% |
| 测试 | 998,539 | 0.083549 | 0.068316 | 0.591439 | 100% |

股票池总样本数分别为 7,760,000、936,000、1,000,000，全部有有限预测。训练期首个截面缺乏更早历史，因子为常数，该截面相关系数记为空；验证/测试没有未定义相关系数的截面。测试集原始收益尺度 MSE 为 `2.2055884e-5`，零预测参照为 `2.2081764e-5`；相关性为正不等于已验证交易收益。

权重导出后，32,768 个样本的 PyTorch/NumPy 预测最大绝对差为 `6.985e-10`。真实提交 Notebook 的跨月、年末和分钟边界检查通过，未导入 PyTorch、未读取标签。另从原始行情执行完整测试半年推理，100 万条结果与冻结评估预测完全一致，本地 CPU 耗时 13.11 秒；该耗时不代表平台机型。

可入库的汇总证据：[v1-baseline.json](../experiments/v1-baseline.json)、[运行索引](../experiments/runs.csv)、[提交索引](../experiments/submissions.csv)。详细日志、逐轮指标、曲线、预测、权重和源代码快照保存在 `outputs/runs/v1-20261010-s42/`。

最终模型 SHA-256：`06c374f22569cf902fc8a38d05db3859229c145eaea98dcbba01ef4be573220b`。本地完整训练与评估、导出包验证已完成；平台运行与正式提交尚未执行，由用户上传后确认。

## 提交

可直接使用根目录的 [v1 提交目录](../submissions/v1-20261010-s42/README.md)，全部脚本、Notebook、冻结权重和配置随 Git 推送到远程。从远程仓库拉取后，将该目录文件放在平台同一目录，选择唯一的 `v1-submission.ipynb`，并附脚本、模型、配置和依赖声明。公榜入口只加载权重执行推理。

后续使用 [导出和验证命令](development.md#运行和提交追溯) 生成新的平铺提交目录，不生成 ZIP。

依据：[比赛介绍](https://bigquant.com/square/competition/45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca)、[新手入门](https://bigquant.com/wiki/doc/oSGtQiGtq0)，获取日期 2026-10-10（UTC+08:00）。平台实际附件协议、官方评分与正式提交结果不由本地验证替代。
