# 平台提交材料

[文档目录](../docs/README.md) · [导出与验证](../docs/development.md#比较与导出工具) · [提交状态索引](../experiments/submissions.csv)

每个子目录对应一份冻结材料。只上传选定目录内的文件，保留文件名，选择其中唯一的推理 Notebook；不混入其他运行或本地 `dai.py`，不生成 ZIP。平台提交由用户执行。

| 运行／材料版本 | 上传材料 | 本地判定与验证清单 |
| --- | --- | --- |
| [v1-20261010-s42](v1-20261010-s42/README.md) | 目录中的 Notebook 及配套脚本、权重和配置，见目录说明 | 历史基线；[清单](v1-20261010-s42/submission-manifest.json) |
| [v2-20261011-s42](v2-20261011-s42/v2-submission.ipynb) | `v2-submission.ipynb`、`v2-model.json` | 未晋级，按用户要求导出；[清单](../experiments/v2-20261011-s42-submission.json) |
| [v4-20261014-s42](v4-20261014-s42/v4-submission.ipynb) | `v4-submission.ipynb`、`v4-model.json` | 未晋级，按用户要求导出；[清单](../experiments/v4-20261014-s42-submission.json) |
| [v5-20261015-s42](v5-20261015-s42/v5-submission.ipynb) | `v5-submission.ipynb`、`v5-model.json` | 本地改善候选；[清单](../experiments/v5-20261015-s42-submission.json) |
| [v6-20261015-s42](v6-20261015-s42/v6-submission.ipynb) | `v6-submission.ipynb`、`v6-model.json` | 本地改善候选；[清单](../experiments/v6-20261015-s42-submission.json) |
| [v7-20261016-no-layernorm-s42](v7-20261016-no-layernorm-s42/v7-submission.ipynb) | `v7-submission.ipynb`、`v7-model.json` | 本地改善候选；[清单](../experiments/v7-20261016-no-layernorm-s42-submission.json) |
| [v7e-zscore-huber-s42](v7e-zscore-huber-s42/v7e-submission.ipynb) | `v7e-submission.ipynb`、`v7e-model.json` | 本地改善候选；截面标准化 Huber，输出 `factor_scale=1`；[清单](../experiments/v7e-zscore-huber-s42-submission.json) |
| [v101-global-fp16-migrated-s42](v101-global-fp16-migrated-s42/v101-submission.ipynb) | `v101-submission.ipynb`、`v101-model.json` | 未晋级，按用户要求导出平台测试；Patch Transformer，第 13 轮权重；[清单](../experiments/v101-global-fp16-migrated-s42-submission.json) |
| [v201-global-fp16-s42](v201-global-fp16-s42/v201-submission.ipynb) | `v201-submission.ipynb`、`v201-model.json` | 本地改善候选；低秩 Mixer，第 5 轮权重，已通过独立 CPU 推理验证；[清单](../experiments/v201-global-fp16-s42-submission.json) |
| [v301-global-fp16-s42](v301-global-fp16-s42/v301-submission.ipynb) | `v301-submission.ipynb`、`v301-model.json` | 本地改善候选；分层 GRU，第 7 轮权重，已通过独立 CPU 与冻结 checkpoint 对照；[清单](../experiments/v301-global-fp16-s42-submission.json) |
| [v303-huber-delta2-s42](v303-huber-delta2-s42/v303-submission.ipynb) | `v303-submission.ipynb`、`v303-model.json` | 本地改善候选；Huber delta=2 的分层 GRU，第 7 轮权重，已通过独立 CPU 与冻结 checkpoint 对照；[清单](../experiments/v303-huber-delta2-s42-submission.json) |
| [v303-huber-delta2-s42-fast](v303-huber-delta2-s42-fast/v303-submission.ipynb) | 同一加速目录的 `v303-submission.ipynb`、`v303-model.json` | **v303 后续评测推荐材料**；同一冻结权重，12 日 CPU 对照约快 2.94 倍且预测一致；[清单](../experiments/v303-huber-delta2-s42-fast-submission.json)、[说明](../docs/v303-inference-fast.md) |
| [v501-ema-gru-s42-fast](v501-ema-gru-s42-fast/v501-submission.ipynb) | `v501-submission.ipynb`、`v501-model.json` | 先前本地验证最优、可作对照；第 8 轮 EMA，默认 fast；[清单](../experiments/v501-ema-gru-s42-fast-submission.json)、[结果](../solutions/v501/README.md#冻结结果) |
| [v901-current-multifrequency-s42-fast](v901-current-multifrequency-s42-fast/v901-submission.ipynb) | `v901-submission.ipynb`、`v901-model.json` | 前一本地最优、保留为 IR 对照；第 5 轮 EMA，三频当天因果输入、默认 fast；[清单](../experiments/v901-current-multifrequency-s42-fast-submission.json)、[结果](../solutions/v901/README.md#冻结结果与平台材料) |
| [v1001-temporal-experts-s42-fast](v1001-temporal-experts-s42-fast/v1001-submission.ipynb) | `v1001-submission.ipynb`、`v1001-model.json` | 前一本地最优、换手代理较低的对照；第 4 轮 EMA，默认 fast；[清单](../experiments/v1001-temporal-experts-s42-fast-submission.json)、[结果](../solutions/v1001/README.md#冻结结果与平台材料) |
| [v1002-current-temporal-experts-s42-fast](v1002-current-temporal-experts-s42-fast/v1002-submission.ipynb) | `v1002-submission.ipynb`、`v1002-model.json` | 单模型及 IR 对照；第 4 轮 EMA，当天多频与时间加权分支组合，默认 fast；[清单](../experiments/v1002-current-temporal-experts-s42-fast-submission.json)、[结果](../solutions/v1002/README.md#冻结结果与平台材料) |
| [v1101-structured-fields-s42-fast](v1101-structured-fields-s42-fast/v1101-submission.ipynb) | `v1101-submission.ipynb`、`v1101-model.json` | **按用户要求导出平台对照**；原第 5 轮 EMA，字段结构与当天多频、默认 fast；验证 IC 未晋级判定保持；[清单](../experiments/v1101-structured-fields-s42-fast-submission.json)、[说明](../solutions/v1101/README.md#用户指定的平台对照材料) |
| [v1401-frozen-equal-ensemble-s42-fast](v1401-frozen-equal-ensemble-s42-fast/v1401-submission.ipynb) | **同目录三个文件**：`v1401-submission.ipynb`、`v1002-model.json`、`v1101-model.json` | 原本地总体 IC 最优，保留为固定组合对照；共享预处理、双主干 fast，新增训练为零；[清单](../experiments/v1401-frozen-equal-ensemble-s42-fast-submission.json)、[结果与代价](../solutions/v1401/README.md) |
| [v1601-price-geometry-s42-fast](v1601-price-geometry-s42-fast/v1601-submission.ipynb) | `v1601-submission.ipynb`、`v1601-model.json` | 前一本地最优、可作对照；第 4 轮 EMA，共同价格坐标、单主干 fast；测试 Rank IC IR 和换手代理有代价；[清单](../experiments/v1601-price-geometry-s42-fast-submission.json)、[冻结结果](../solutions/v1601/README.md#冻结结果与平台材料) |
| [v2401-price-memory-s42-fast](v2401-price-memory-s42-fast/v2401-submission.ipynb) | `v2401-submission.ipynb`、`v2401-model.json` | **当前本地优先评测候选**；第4轮EMA，完整父输入与因果价格历史读取，默认fast；[清单](../experiments/v2401-price-memory-s42-fast-submission.json)、[结果与代价](../solutions/v2401/README.md#冻结结果与平台材料) |

v2／v4—v7／v7e／v101／v201／v301／v303 的两个文本文件需置于同一平台目录。JSON 内嵌冻结权重、归一化统计量、配置和训练／推理源码；Notebook 提供评测入口、依赖及复现说明。附件大小、数值精度及独立推理的验证范围以各清单为准。

v2401 fast 的 JSON 为 **45,113,148 字节**，Notebook 为 **7,214 字节**。实际文本权重精确恢复、隔离两文件八时点合成推理与独立训练源码导入通过，21行 fast／逐时点参考最大差为 \(5.96\times10^{-8}\)。未重复长行情或性能基准，平台运行及评分待用户评测。

v1601 fast 的 JSON 为 **43,403,545 字节**，Notebook 为 **7,155 字节**。实际文本权重精确恢复、隔离两文件推理和训练源码导入均通过；21 行合成 fast／参考最大差为 \(5.96\times10^{-8}\)，其中实际执行了共同价格变换。按简化约定未重复长行情或速度基准；尚未提交平台，不能把本地改善当作官方评分结果。

v101 使用 `bar1m`、`bar5m`、`bar30m` 三个行情源，JSON 为 18,195,183 字节；输出为标准化目标单位的模型分数。它来自月内 FP32 到全局 FP16 的迁移训练，复现入口从头训练当前配置，不重放历史迁移轨迹。冻结指标与对照结论见 [v101 说明](../solutions/v101/README.md#冻结结果与平台材料)。

v201 使用同样的三个行情源，JSON 为 16,942,197 字节，Notebook 为 7,943 字节；从头全局训练，输出为标准化目标单位。独立 CPU 验证覆盖 2024-01-02 至 2024-01-05 和 2024-12-30 至 2024-12-31，共 48,000 行，权重精确恢复、键和覆盖一致，预测通过预先设定的绝对与相对误差检查。冻结比较、局限及代理诊断见 [v201 说明](../solutions/v201/README.md#冻结结果与平台材料)。

v301 的 JSON 为 15,274,281 字节，Notebook 为 8,005 字节。相同两段区间共 48,000 行的 CPU 平台入口输出，与原 checkpoint 和冻结预处理数据在相同 CPU 后端及逐日批量边界下的输出完全一致。它与原 CUDA 冻结预测的最大差约为 4.77×10⁻⁶；两类比较分别记录，不把跨执行路径的浮点差异解释为权重文本损失。数值核验方法、容差及模型局限见 [v301 说明](../solutions/v301/README.md#冻结结果与平台材料)。

v303 的 JSON 为 15,272,894 字节，Notebook 为 8,007 字节。相同两段区间共 48,000 行通过独立 CPU 核验：原始行情入口与原 checkpoint 的同后端对照完全一致，与原 CUDA 冻结预测的最大差约 5.87×10⁻⁶。它以较大的 Huber delta 提高双集 IC 和 IC IR，但 Rank IC 略低、换手代理增加，提交理由和完整验证见 [v303 说明](../solutions/v303/README.md#冻结结果与平台材料)。

## 平台状态与完整性

2026-10-22 11:35:35（UTC+08:00）用户通知后查询得到 12 条有公榜分的记录。[本次观测](../experiments/platform-score-observations-20261022.json)中，按附件名识别的 v1101／v901／v1001／v1002 总分分别为 0.44308／0.44000／0.43436／0.42308；当前账号这批记录以 v1101 最高，私榜均为空。参数附件大小与对应材料相符，但 Notebook 大小不同，仍未确认本地 run／权重映射，不回填提交索引或冒充比赛名次。此次查询由用户通知触发，不恢复自动定期查分。

v1401 的两个参数 JSON 分别为 41,994,269／40,277,558 字节，Notebook 为 27,908 字节。实际权重精确恢复、隔离三文件的合成窗口和完整合成 main 检查通过；共享行情查询，两个网络分别计算，再在完整截面组合。未进行长行情性能基准或实际平台验证，具体检查与数值容差见清单。

v1002 fast 的 JSON 为 41,994,269 字节、Notebook 为 9,403 字节。实际权重精确恢复、隔离两文件的 21 行合成窗口对照和独立复现源码导入检查通过；没有重复长行情或性能基准，也未验证平台运行。四个分支共享三频时序编码，推理不使用训练标签或时间损失权重。

v1101 fast 按用户 2026-10-22 的明确要求导出，JSON 为 40,277,558 字节、Notebook 为 9,198 字节。权重精确恢复、隔离两文件的 21 行合成窗口对照与独立复现源码导入检查通过；未重跑长行情、未测具体速度倍数，也未验证平台运行。此次导出不改变原本地未晋级判定。

v1001 fast 的 JSON 为 42,023,924 字节、Notebook 为 9,326 字节。完成实际文本权重位精确恢复、隔离两文件的 21 行合成窗口 fast 对照和独立复现源码导入；没有重复长行情区间或速度基准，也未验证平台运行。四个预测分支共享一个时序主干，推理不使用训练日期权重、标签辅助分配或历史误差记忆。

v901 fast 的 JSON 为 30,498,868 字节、Notebook 为 8,917 字节。按用户的简化转换约定完成冻结源码逻辑审查、实际文本权重位精确恢复、隔离两文件的 21 行合成窗口推理和独立训练源码导入；没有重复长行情区间验证或性能基准，也未验证平台运行。上传时将这两份文件放在同一目录，选择其中唯一 Notebook。

用户于 2026-10-21（UTC+08:00）确认，v101 对应提交 `9a994f25-1181-4630-b27e-755522151391` 运行超时；接口负值不作为模型效果评分。该反馈确认执行问题，未额外确认上传权重与本地 run 的精确对应关系。用户同时要求停止主动查询平台成绩，后续平台更新由用户同步，本地研究和候选材料准备继续进行。

保留的历史接口快照为 2026-10-21 01:47:08（UTC+08:00），包含当前账号可见的 10 条提交：当时按文件名对应 v2、v4、v5、v6、v7、v7e 的提交已有公榜分，v101、v201、v301、v303 为空，私榜全部为空。该快照中 v7e 对应提交的公榜分最高，为 0.36264；本地 run／权重绑定仍待确认，未据文件名或大小回填本地提交索引。原始分数、分项指标和提交 ID 见[历史成绩记录](../experiments/platform-scores.json)，不将其作为当前实时状态。

材料随 Git 分发；`.gitattributes` 保持提交文件原始字节。重新打包时须更新清单并核验冻结权重和推理一致性，具体格式与验证要求见开发约定。行情、离线预测答案、缓存和凭据不进入提交材料。

长期未出分的处理依据、获取日期与状态限制统一见[来源记录](../docs/bigquant-2026-end-to-end/sources.md)和[未决事项](../docs/bigquant-2026-end-to-end/open-questions.md)。
