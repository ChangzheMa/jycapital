# v701 完成后的分布适应研究筛选

核验日期：2026-10-22（UTC+08:00）。本文延续[前一轮文献及组合方向](research-directions-20261021.md)，结合 v701 的正式结果，审查时间分布变化与多模式预测的适用条件。论文效果、本地观察与待验证假设分别记录；不把会议级别或发表时间作为效果保证。

## 本地证据

[v701](../solutions/v701/README.md#冻结结果)的验证 IC 0.086159 低于 v501 的 0.087199，测试 IC 0.091387 高于 v501 的 0.090330，双集 Rank IC 提高；IC IR 均略低。验证和测试各 3/6 个月 IC 提高，测试增益主要来自 11 月。股票日共享后的平均单轮训练及验证约 10.7 分钟，说明可以继续使用这条计算路径探索组合方案，但不能把月份差异直接归因于概念漂移。

[分预测时点比较](../experiments/v701-clock-comparison.json)仅汇总已有冻结截面，不新增推理。验证 10:00 的 IC 差值为 −0.006042，测试同一时点则为 +0.006152；09:30 分别为 +0.000899 和 −0.006529。差异并未形成验证、测试一致的固定弱时点，因此当前证据不支持按已观察测试表现手工挑时点或替换输出。

v801 的表示缓存残差头虽使 IR／换手代理改善，但验证和测试 IC／Rank IC 下降。v802 已完成联合适应：验证 IC 0.086299、测试 IC 0.089058，双集 IC／Rank IC 均低于 v801、v501，训练及验证约 145 分钟；不继续扩大当前集合模块。v901 随后完成，验证 IC 0.087905、测试 IC 0.093509，成为新的本地最优；当天多频、适中宽度与历史状态共享的组合取得正向证据，但未分别归因。

## 新近论文与机制核验

| 研究与来源 | 已核验范围与质量 | 对本项目的含义 |
| --- | --- | --- |
| **ShifTS: Tackling Time-Series Forecasting Generalization via Mitigating Concept Drift**，ICLR 2026；[正式记录](https://proceedings.iclr.cc/paper_files/paper/2026/hash/96d328a1f6d8396d8c8a62f2beee252a-Abstract-Conference.html)、[论文](https://proceedings.iclr.cc/paper_files/paper/2026/file/96d328a1f6d8396d8c8a62f2beee252a-Paper-Conference.pdf) | 正式发表，核验方法、实验与附录；区分边际统计变化和条件关系变化，评估汇率、流感与 ETT 等六个数据集，有不同主干及模块消融。该评估范围不含本项目的分钟股票截面收益任务 | 提醒将归一化与预测关系变化分开分析；不能根据通用序列误差改善，断言本地 IC 提高，也不能仅由月份成绩波动诊断漂移 |
| **Meta-learning for return prediction in shifting market regimes / FinPFN**，Journal of Financial Markets，2026 年 6 月；[出版社记录与摘要](https://www.sciencedirect.com/science/article/pii/S1386418125000825) | 本次仅核验出版社记录、摘要和研究范围，未独立核验全文协议、代码或复现；方法条件包含近期特征—收益对 | 不能默认平台推理阶段能够取得相同的收益反馈；在信息可用性和复现证据明确前，不作为实现依据 |
| **Adaptive mixture of experts framework for multi horizon stock market forecasting using heterogeneous deep learning models**，Discover Computing，2026-09-25；[出版社页面](https://link.springer.com/article/10.1007/s10791-026-10613-z) | 本次核验正式记录及摘要；五只股票、1／5／10 日预测，报告价格误差与统计比较。股票范围和指标与大股票池分钟收益排序存在明显差异 | 新近发表及显著性报告不足以支持本项目效果预期；不据价格 RMSE 改善部署多个重型时序网络 |
| **DoubleAdapt**，KDD 2023；[正式 DOI](https://doi.org/10.1145/3580305.3599315)、[作者正文 v3](https://arxiv.org/html/2306.09862v3) | 正式发表，正文明确模型和数据适配器在增量任务间更新，新的真实标签到达后推动后续任务。该使用场景属于持续在线适应 | 与本项目冻结权重的评测入口不同；本轮不移植在线更新。若借鉴元学习，训练内任务必须全部限制于 2020—2023，并另行证明冻结推理方式可用 |
| **TRA: Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport**，KDD 2021；[作者机构记录](https://www.microsoft.com/en-us/research/publication/learning-multiple-stock-trading-patterns-with-temporal-routing-adaptor-and-optimal-transport/)、[作者正文 v2](https://arxiv.org/html/2106.12950v2)、[官方代码](https://github.com/microsoft/qlib/tree/main/examples/benchmarks/TRA) | 时间顺序划分，有路由输入与最优传输消融；表格报告五种子的 MSE／MAE 离散程度，不能据此声称 IC 也有完整不确定性报告。完整路由使用历史预测误差，测试需按时间推进；官方旧论文复现入口标注不维护 | 原始完整方案不直接适配当前提交。仅表示输入的消融 IC 为 0.053，结合误差记忆为 0.059，不能删除误差输入后照搬完整效果。模型内部的可学习多分支仍可作为假设，但须解决路由坍缩，并保持推理与历史标签解耦 |

补充搜索发现 DDG-DA（AAAI 2022）的分布预测／重加权路线，以及基于需求预测、单只股票价格或外部新闻的研究；本轮不据这些任务的总分选择具体时间权重、金融专家数量或模型规模。没有找到能直接证明固定近期加权、解冻主干或增加专家一定提高本项目验证 IC 的证据。

## 下一步决策

v802 已提供联合适应的负结果，不能单独归因于解冻。v1001 在 v901 完成前已经冻结，以当时已完成的 v701 股票日共享主干为基础，不扩大当前集合模块。v901 已改善并准备 fast 平台材料，成为后续新方案的比较参照；不在 v1001 运行中修改其输入、结构或父版本。

下一项组合将以下两个假设落实到 [v1001](../solutions/v1001/README.md)：约 242 万参数，单个股票日共享主干、四个纯行情路由分支，训练标签只用于辅助分配；时间权重采用 730 日半衰期、0.25 下限。只评价整体组合，不把论文结果或月份波动当作有效性证明。

- **训练期分布适应**：在全部训练日期仍可用的前提下，研究时间加权或训练内时间段约束。权重只能由训练期信息确定；不能用已观察测试月份来拟合权重，也不能把单调近期加权等同于已证明的未来分布估计。
- **纯行情条件的多模式预测**：共享一个已有效的时序主干，用小型模型分支和可学习路由表达不同关系。训练辅助分配可使用训练标签，推理路由只看当时行情表示，不引入历史标签／预测误差存储。需要先验证不会坍缩到单一分支，且不破坏股票日复用；不能自称复现了 TRA 的完整效果。

v1001 仍按原验证 IC 选模、完整评估双集，保留 EMA、动态早停和因果 fast 路径；实际 ready 状态以候选清单为准。仅检索或讨论方向不构成已发布候选，未实现的方向不占据训练队列。研究 session 不启动正式训练，训练 session 继续串行调度。
