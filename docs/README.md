# 进益资本杯比赛资料

本仓库保存 BigQuant 2026 年进益资本杯端到端赛道的规则、数据说明、模型方案和冻结结果。比赛 ID：`45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca`。

## 冻结模型结果

本地研究采用 2020—2023 年训练、2024 年上半年验证、下半年测试；股票池和最低截面覆盖率均为 100%。各模型保持原先选出的权重，详细指标及来源见对应结果文件。

| 方案 | 参数量 | 验证 IC | 验证 Rank IC | 测试 IC | 测试 Rank IC | 结果 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| [v1](baseline-v1.md) | 6,785 | 0.070445 | 0.061333 | 0.083549 | 0.068316 | 历史验证 IC 最高 |
| [v2](../solutions/v2/README.md) | 428,673 | 0.034507 | 0.030664 | 0.032298 | 0.033902 | 未超过 v1 |
| [v3](../solutions/v3/README.md) | 1,591,553 | 0.044650 | 0.044401 | 0.037489 | 0.042015 | 超过 v2，未超过 v1 |
| [v4](../solutions/v4/README.md) | 1,591,553 | 0.069469 | 0.068725 | 0.073400 | 0.070554 | 超过 v3，按验证 IC 未晋级；输出未校准 |

v1—v3 按验证 Rank IC 选模，v4 按验证 IC 选模。测试期已在持续研究中观察过，本地指标不等同于平台官方成绩。

Mock 已通过平台测试提交；v1 平台可运行由用户于 2026-10-10（UTC+08:00）确认。v2／v4 的平台材料已按用户要求导出并通过本地验证，正式平台成绩尚未确认。附件和上传说明见 [平台提交目录](../submissions/README.md)，状态及证据见 [提交索引](../experiments/submissions.csv)。

## 文档索引

| 主题 | 文档 |
| --- | --- |
| 比赛要求 | [规则与赛程](bigquant-2026-end-to-end/rules.md)、[数据与字段](bigquant-2026-end-to-end/data.md) |
| 来源及未决问题 | [官方来源](bigquant-2026-end-to-end/sources.md)、[待确认事项](bigquant-2026-end-to-end/open-questions.md) |
| 本地使用与数据质量 | [本地环境](bigquant-2026-end-to-end/local-download.md)、[数据一致性结论](bigquant-2026-end-to-end/data-validation.md)、[验证摘要](bigquant-2026-end-to-end/data-validation-summary.json) |
| 结构化比赛资料 | [配置摘录](bigquant-2026-end-to-end/competition-config.json)、[字段 JSON](bigquant-2026-end-to-end/data-schemas.json)、[字段 CSV](bigquant-2026-end-to-end/data-dictionary.csv)，共 104 条字段记录 |
| 实验与评估 | [开发约定](development.md)、[实验技能](../.agents/skills/jycapital-experiment/SKILL.md)、[查分技能](../.agents/skills/jycapital-scores/SKILL.md) |
| 模型与损失 | [v2 交互图解](v2-model.html)、[字段可用性及组掩码](v3-field-validity.md)、[Pearson IC 损失设计](v4-ic-loss-plan.md) |
| 缺失数据结论 | [完整样本过滤代价](v2-complete-samples.md)、[逐字段无效值统计](raw-field-validity.md)、[盘口零价格含义](book-price-zero.md) |
| 冻结实验结果 | [v1](../experiments/v1-baseline.json)、[v2](../experiments/v2-tcn.json)、[v3](../experiments/v3-tcn.json)、[v4](../experiments/v4-tcn.json)、[运行索引](../experiments/runs.csv) |
| 预处理结果 | [训练](../experiments/v2-preprocessing.json)、[验证／测试](../experiments/v2-holdout-preprocessing.json) |
| 提交清单 | [v2](../experiments/v2-20261011-s42-submission.json)、[v4](../experiments/v4-20261014-s42-submission.json) |
| 接口示例与依赖 | [Mock Notebook](../examples/mock-submission.ipynb)、[本地依赖](../requirements-local.txt) |

## 输入与输出约定

使用授权原始分钟量价及盘口序列预测未来 30 分钟 VWAP 收益。允许必要清洗、归一化和切窗；人工衍生特征、未来信息及外部预训练权重的边界见比赛规则。

推理入口为 `main(datasources, start_date, end_date)`，返回列顺序为 `date、instrument、factor` 的 DataFrame：保留完整分钟时间，因子为有限 `float64`，键唯一。正式行情表通过 `datasources` 取得。

本地原始五档 `stock` 表及辅助表共 243 个文件、723,258,293 行，与 2026-09-30 平台基准的逻辑内容一致。行情、凭据及本地 `dai.py` 不进入 Git。

比赛时间、资源限制及官方口径差异统一保留在规则和待确认事项中；来源日期见来源页。
