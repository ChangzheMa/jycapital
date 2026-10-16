# 进益资本杯比赛资料

本仓库保存 BigQuant 2026 年进益资本杯端到端赛道的规则、数据说明、模型方案和冻结结果。比赛 ID：`45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca`。

## 冻结模型结果

本地研究采用 2020—2023 年训练、2024 年上半年验证、下半年测试；股票池和最低截面覆盖率均为 100%。各模型保持原先选出的权重，详细指标及来源见对应结果文件。

| 方案 | 参数量 | 验证 IC | 验证 Rank IC | 测试 IC | 测试 Rank IC | 结果 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| [v1](baseline-v1.md) | 6,785 | 0.070445 | 0.061333 | 0.083549 | 0.068316 | 历史基线，测试 IC 高于 v4—v7 |
| [v2](../solutions/v2/README.md) | 428,673 | 0.034507 | 0.030664 | 0.032298 | 0.033902 | 未超过 v1 |
| [v3](../solutions/v3/README.md) | 1,591,553 | 0.044650 | 0.044401 | 0.037489 | 0.042015 | 超过 v2，未超过 v1 |
| [v4](../solutions/v4/README.md) | 1,591,553 | 0.069469 | 0.068725 | 0.073400 | 0.070554 | 超过 v3，按验证 IC 未晋级；输出未校准 |
| [v5](../solutions/v5/README.md) | 2,429,505 | 0.070462 | 0.069256 | 0.073423 | 0.071142 | 验证 IC 略高于 v1；单种子本地改善候选 |
| [v6](../solutions/v6/README.md) | 2,429,505 | 0.071842 | 0.067675 | 0.074766 | 0.069818 | 残差块 Dropout 0.35；单种子本地改善候选 |
| [v7](../solutions/v7/README.md) | 2,421,185 | 0.073347 | 0.069186 | 0.079857 | 0.073054 | 删除全部 LayerNorm；达到本地改善条件，属于单种子改善候选 |

v1—v3 按验证 Rank IC 选模，v4—v7 按验证 IC 选模。测试期已在持续研究中观察过，本地指标不等同于平台官方成绩。

附件、上传方式及平台状态统一见 [平台提交目录](../submissions/README.md)；Mock 接口确认依据见 [来源记录](bigquant-2026-end-to-end/sources.md)。

## 文档索引

| 主题 | 文档 |
| --- | --- |
| 比赛要求 | [规则与赛程](bigquant-2026-end-to-end/rules.md)、[数据与字段](bigquant-2026-end-to-end/data.md) |
| 来源及未决问题 | [官方来源](bigquant-2026-end-to-end/sources.md)、[待确认事项](bigquant-2026-end-to-end/open-questions.md) |
| 本地使用与数据质量 | [本地环境](bigquant-2026-end-to-end/local-download.md)、[数据一致性结论](bigquant-2026-end-to-end/data-validation.md)、[验证摘要](bigquant-2026-end-to-end/data-validation-summary.json) |
| 结构化比赛资料 | [配置摘录](bigquant-2026-end-to-end/competition-config.json)、[字段 JSON](bigquant-2026-end-to-end/data-schemas.json)、[字段 CSV](bigquant-2026-end-to-end/data-dictionary.csv)，共 104 条字段记录 |
| 实验与评估 | [开发约定](development.md)、[实验技能](../.agents/skills/jycapital-experiment/SKILL.md)、[查分技能](../.agents/skills/jycapital-scores/SKILL.md) |
| 模型与损失 | [v6 冻结模型逐层图解](v6-model.html)、[v2 交互图解](v2-model.html)、[字段可用性及组掩码](v3-field-validity.md)、[Pearson IC 损失设计](v4-ic-loss-plan.md) |
| 扩容方案 | [v5：四频 TCN 隐藏通道扩至 160](../solutions/v5/README.md) |
| 正则化方案 | [v6：TCN 残差块 Dropout 提高至 0.35](../solutions/v6/README.md) |
| 当前开发版本 | [v7：删除全部 LayerNorm](../solutions/v7/README.md)、[修改备忘录](v7-discussion.md)；动态早停已生效，全窗口平均及边界复制仍待实施 |
| 缺失数据结论 | [完整样本过滤代价](v2-complete-samples.md)、[逐字段无效值统计](raw-field-validity.md)、[盘口零价格含义](book-price-zero.md) |
| 冻结实验结果 | [v1](../experiments/v1-baseline.json)、[v2](../experiments/v2-tcn.json)、[v3](../experiments/v3-tcn.json)、[v4](../experiments/v4-tcn.json)、[v5](../experiments/v5-tcn.json)、[v6](../experiments/v6-tcn.json)、[v7](../experiments/v7-tcn.json)、[运行索引](../experiments/runs.csv) |
| 平台成绩 | [已查询成绩与版本对应关系](../experiments/platform-scores.json)，公榜／私榜分列，保留来源与查询时间 |
| 预处理结果 | [训练](../experiments/v2-preprocessing.json)、[验证／测试](../experiments/v2-holdout-preprocessing.json) |
| 提交清单 | [v2](../experiments/v2-20261011-s42-submission.json)、[v4](../experiments/v4-20261014-s42-submission.json)、[v5](../experiments/v5-20261015-s42-submission.json)、[v6](../experiments/v6-20261015-s42-submission.json)、[v7](../experiments/v7-20261016-no-layernorm-s42-submission.json) |
| 接口示例与依赖 | [Mock Notebook](../examples/mock-submission.ipynb)、[本地依赖](../requirements-local.txt) |

## 输入与输出约定

使用授权原始分钟量价及盘口序列预测未来 30 分钟 VWAP 收益。允许必要清洗、归一化和切窗；人工衍生特征、未来信息及外部预训练权重的边界见比赛规则。

推理入口为 `main(datasources, start_date, end_date)`，返回列顺序为 `date、instrument、factor` 的 DataFrame：保留完整分钟时间，因子为有限 `float64`，键唯一。正式行情表通过 `datasources` 取得。

本地数据范围及一致性依据见 [数据验证](bigquant-2026-end-to-end/data-validation.md)；行情、凭据及本地 `dai.py` 不进入 Git。
