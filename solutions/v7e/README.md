# v7e：截面标准化标签的 Huber 回归

[实验总览与边界设置](../../docs/v7-ablation-experiments.md) · [配置](model-config.json) · [损失](v7e_loss.py) · [训练器](v7e_train.py)

父运行为 `v7-20261016-no-layernorm-s42`，对应备忘录第 5 组。网络结构完全沿用 v7，共 2,421,185 个参数；每个完整截面的有限标签按总体标准差做 z-score，原始预测直接拟合该目标。固定 Huber `delta=1`，输出 `factor_scale=1`，按验证 IC 选模。

训练先按完整预测时间戳构造标准化目标，再沿用 v2/v3 的月内随机样本 batch：每轮打乱月份，月内以 seed、epoch、月份确定样本排列，batch 为 512，保留不足 512 的尾批。随机 batch 可以混合本月不同日期、时刻和股票，不在 batch 内重新标准化。训练 Huber 按样本取均值，轮次日志按实际样本数加权；验证／测试 Huber 仍按完整截面计算并等权汇总。随机 batch 上的相关性不记作训练截面 IC，验证／测试 IC 独立报告。

每 64 个 batch 及月末保存 checkpoint，断点保存样本游标、优化器、随机状态及早停计数，恢复时重建相同排列。旧的逐截面更新配置不能续跑此版本。相对 v7，本实验同时改变监督目标、损失和 batch 组织，后续比较需按这一整体设置解释。

标签退化、常数预测、损失日志及输出尺度说明统一见实验总览。

## 冻结结果与平台材料

运行 `v7e-zscore-huber-s42` 已完成训练、验证和测试，seed=42；按验证 IC 选择第 7 轮权重，第 12 轮触发动态早停。结果来自本地冻结产物，归档于 2026-10-19（UTC+08:00），详见[评估与比较](../../experiments/v7e-tcn.json)。

| 版本 | 验证 IC | 验证 Rank IC | 测试 IC | 测试 Rank IC |
| --- | ---: | ---: | ---: | ---: |
| v7（此前本地验证最优及父版本） | 0.073347 | 0.069186 | 0.079857 | 0.073054 |
| v7e | 0.076757 | 0.072320 | 0.079773 | 0.074215 |
| v7e − v7 | +0.003409 | +0.003135 | −0.000084 | +0.001161 |

两套评估的股票池覆盖和最低截面覆盖均为 100%，IC／Rank IC 无未定义截面。验证 IC 超过可比历史最优，达到本地改善条件；验证和测试各有 3／6 个月 IC 提高，验证 IC IR／Rank IC IR 分别下降 0.003807／0.015561，测试分别下降 0.032183／0.051776，不能解释为全面改善。参数量与 v7 相同；本次记录的训练耗时约 3.68 小时（RTX 4080），数据读取和精度路径会影响耗时，不能据此单独判断网络速度。

最佳验证与测试 Huber 分别为 0.360116 和 0.343903。输出仍为标准化目标单位，原始收益尺度上的 MSE／MAE 不代表 Huber 目标拟合误差。结果仅来自一个种子，测试期已在持续研究中观察过。已有平台公榜成绩的本地 run／权重绑定未确认，v7e 尚无平台成绩。

平台附件位于 [`submissions/v7e-zscore-huber-s42/`](../../submissions/v7e-zscore-huber-s42/)，上传同目录的 `v7e-submission.ipynb` 和 `v7e-model.json`，选择 Notebook 作为入口。参数 JSON 为 44,707,826 字节，低于单文件 50 MB 上限；包含完整精度权重、配置、归一化统计量和冻结训练／推理源码。平台状态与上传说明见[提交索引](../../submissions/README.md)，逐文件哈希和验证范围见[提交清单](../../experiments/v7e-zscore-huber-s42-submission.json)。

本地已核验全部 72 个张量精确恢复、18 个训练复现文件恢复及训练器导入，并在隔离目录仅靠两个附件执行 2024-06-27—28（验证期）和 2024-12-30—31（测试期）的完整区间推理。每个区间 16,000 行，输出类型、有限性、键唯一性和预测键覆盖通过，与冻结评估预测在既定容差内一致。Notebook 导入不查询行情，调试单元格可执行；这些检查不代表平台运行或官方评分通过。

## 复现与导出入口

以下训练命令仅适用于新运行目录；现有冻结运行直接使用对应权重。

```powershell
.venv-training/Scripts/python.exe -u solutions/v7e/v7e_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v7e-new-s42 --training-confirmed
```

[导出器](../../scripts/export-v7e.py)与[验证器](../../scripts/verify-v7e-submission.py)只使用所选运行的冻结材料，默认拒绝覆盖已有提交目录：

```powershell
.venv-training/Scripts/python.exe scripts/export-v7e.py --run-dir outputs/runs/v7e-zscore-huber-s42
.venv-training/Scripts/python.exe scripts/verify-v7e-submission.py --run-dir outputs/runs/v7e-zscore-huber-s42 --package submissions/v7e-zscore-huber-s42 --record
```
