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
| [v901-current-multifrequency-s42-fast](v901-current-multifrequency-s42-fast/v901-submission.ipynb) | `v901-submission.ipynb`、`v901-model.json` | **当前优先候选**；第 5 轮 EMA，三频当天因果输入、默认 fast；双集 IC／Rank IC 提高，验证 IC IR 略低、换手代理略高；[清单](../experiments/v901-current-multifrequency-s42-fast-submission.json)、[结果](../solutions/v901/README.md#冻结结果与平台材料) |

v2／v4—v7／v7e／v101／v201／v301／v303 的两个文本文件需置于同一平台目录。JSON 内嵌冻结权重、归一化统计量、配置和训练／推理源码；Notebook 提供评测入口、依赖及复现说明。附件大小、数值精度及独立推理的验证范围以各清单为准。

v101 使用 `bar1m`、`bar5m`、`bar30m` 三个行情源，JSON 为 18,195,183 字节；输出为标准化目标单位的模型分数。它来自月内 FP32 到全局 FP16 的迁移训练，复现入口从头训练当前配置，不重放历史迁移轨迹。冻结指标与对照结论见 [v101 说明](../solutions/v101/README.md#冻结结果与平台材料)。

v201 使用同样的三个行情源，JSON 为 16,942,197 字节，Notebook 为 7,943 字节；从头全局训练，输出为标准化目标单位。独立 CPU 验证覆盖 2024-01-02 至 2024-01-05 和 2024-12-30 至 2024-12-31，共 48,000 行，权重精确恢复、键和覆盖一致，预测通过预先设定的绝对与相对误差检查。冻结比较、局限及代理诊断见 [v201 说明](../solutions/v201/README.md#冻结结果与平台材料)。

v301 的 JSON 为 15,274,281 字节，Notebook 为 8,005 字节。相同两段区间共 48,000 行的 CPU 平台入口输出，与原 checkpoint 和冻结预处理数据在相同 CPU 后端及逐日批量边界下的输出完全一致。它与原 CUDA 冻结预测的最大差约为 4.77×10⁻⁶；两类比较分别记录，不把跨执行路径的浮点差异解释为权重文本损失。数值核验方法、容差及模型局限见 [v301 说明](../solutions/v301/README.md#冻结结果与平台材料)。

v303 的 JSON 为 15,272,894 字节，Notebook 为 8,007 字节。相同两段区间共 48,000 行通过独立 CPU 核验：原始行情入口与原 checkpoint 的同后端对照完全一致，与原 CUDA 冻结预测的最大差约 5.87×10⁻⁶。它以较大的 Huber delta 提高双集 IC 和 IC IR，但 Rank IC 略低、换手代理增加，提交理由和完整验证见 [v303 说明](../solutions/v303/README.md#冻结结果与平台材料)。

## 平台状态与完整性

v901 fast 的 JSON 为 30,498,868 字节、Notebook 为 8,917 字节。按用户的简化转换约定完成冻结源码逻辑审查、实际文本权重位精确恢复、隔离两文件的 21 行合成窗口推理和独立训练源码导入；没有重复长行情区间验证或性能基准，也未验证平台运行。上传时将这两份文件放在同一目录，选择其中唯一 Notebook。

用户于 2026-10-21（UTC+08:00）确认，v101 对应提交 `9a994f25-1181-4630-b27e-755522151391` 运行超时；接口负值不作为模型效果评分。该反馈确认执行问题，未额外确认上传权重与本地 run 的精确对应关系。用户同时要求停止主动查询平台成绩，后续平台更新由用户同步，本地研究和候选材料准备继续进行。

保留的历史接口快照为 2026-10-21 01:47:08（UTC+08:00），包含当前账号可见的 10 条提交：当时按文件名对应 v2、v4、v5、v6、v7、v7e 的提交已有公榜分，v101、v201、v301、v303 为空，私榜全部为空。该快照中 v7e 对应提交的公榜分最高，为 0.36264；本地 run／权重绑定仍待确认，未据文件名或大小回填本地提交索引。原始分数、分项指标和提交 ID 见[历史成绩记录](../experiments/platform-scores.json)，不将其作为当前实时状态。

材料随 Git 分发；`.gitattributes` 保持提交文件原始字节。重新打包时须更新清单并核验冻结权重和推理一致性，具体格式与验证要求见开发约定。行情、离线预测答案、缓存和凭据不进入提交材料。

长期未出分的处理依据、获取日期与状态限制统一见[来源记录](../docs/bigquant-2026-end-to-end/sources.md)和[未决事项](../docs/bigquant-2026-end-to-end/open-questions.md)。
