# 收益分布辅助监督的研究依据

检索及核验日期：2026-10-22（UTC+08:00）。以已完成的 [v1002](../solutions/v1002/README.md#冻结结果与平台材料)为本地最优参照，探索分布监督能否改善最终点预测的 IC。概率拟合、校准和收益排序是不同目标，本项目仍按原验证 IC 选择权重。

## 文献质量与适用性

| 来源 | 已核验范围与局限 | 设计含义 |
| --- | --- | --- |
| **Temporal multivariate mixture density networks for enhanced portfolio optimization**，Intelligent Systems with Applications，2026 年 9 月；[出版社记录及摘要](https://www.sciencedirect.com/science/article/pii/S2667305326000748) | 核验发表记录和摘要：比较 Gaussian、Student-t 及两成分 Student-t 混合，研究 9／26 资产组合的均值与协方差预测。数据来自不能公开共享的 Bloomberg；本轮全文访问受限，未独立核验全部训练划分、代码和复现 | 提供重尾混合分布的金融应用线索，不把组合 Sharpe 的结论当作分钟截面 IC 的证据，不采用其协方差网络或外部数据 |
| **Embracing Heteroscedasticity for Probabilistic Time Series Forecasting**，2026 年 3 月预印本；[作者正文 v1](https://arxiv.org/html/2603.24254v1) | 核验方法、实验范围和训练说明：使用 Gaussian 位置／尺度模型，评估 ETT、Exchange、ILI 等数据的 CRPS 与 NMAE；明确最大 50 轮、早停耐心 10，本轮未核验其代码中的早停监控变量及独立复现 | 位置与尺度联合学习可能改变表示，但通用序列误差改善不保证股票 IC 改善；不复制 VAE 或将预测尺度当作已校准风险 |
| **Fisher8: Stabilizing Neural Heteroscedastic Regression via Output-Layer Fisher Geometry**，UAI 2026；[正式记录与摘要](https://proceedings.mlr.press/v337/vemuganti26a.html) | 本次核验正式发表与摘要，讨论均值和不确定性共同训练的不稳定性及输出层几何修正；未完成全文算法适配或独立复现 | 不能将针对其设定的修正直接套到 Student-t 混合；先保留稳定的点损失并约束尺度，不自称复现 Fisher8 |
| **On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks**，ICLR 2022；[会议目录](https://iclr.cc/Downloads/2022)、[作者正文 v2](https://arxiv.org/html/2203.09168v2) | 正式发表，正文给出合成分析和多类回归实验：逆方差加权可能使高误差区域长期得不到充分学习，并提出 beta-NLL。研究主要为 Gaussian 回归，不是本项目金融评价 | 保留不依赖预测尺度的 Huber 主项，以免完全依赖似然的自适应降权；正尺度下限用于数值和优化稳定，但不保证消除全部风险 |

没有核验到可以直接证明该方案提高本项目 IC 的新结果。以上文献用于说明方法可行性和优化限制，不作为平台效果承诺。

## 本地依据

v1002 已将验证／测试 IC 提高到 0.091592／0.097627；其时间权重、双头 Huber 与标签辅助分配组合有本地正向证据。v306 增大 Huber delta 未改善，因此本轮保持主项 delta=2。[v1002 冻结摘要](../experiments/v1002-gru.json)中，第 4 轮训练原始权重的平均路由熵为 1.375907，接近四分支均匀分布的 1.386294；这不是所选 EMA 在留出集上的诊断，也不能解释为明确市场模式分工。

本地暂未证明收益残差的条件分布需要 Student-t 混合，也未证明不确定性辅助目标会改善排序。这是一个新的训练监督假设，使用相同原始标签和完整覆盖独立检验，避免在看到结果后换选模标准。

## v1201 的冻结设计

开启 v120x 收益分布辅助监督系列。以 v1002 的单个共享时序主干、三频当天输入和时间权重为基础，每个分支增加一个原始尺度输出。推理仍使用收益均值及原行情路由，不使用预测尺度调整 factor。

设分支输出为收益位置 \(\mu_k\)、排名分数 \(r_k\) 和原始尺度 \(a_k\)，路由权重为 \(\pi_k\)。固定自由度 \(\nu=4\)，尺度为

\[
s_k=0.2+\operatorname{softplus}(a_k).
\]

初始尺度取 \(\sqrt{(\nu-2)/\nu}\)，使单分支初始 Student-t 方差为 1，与完整截面标准化目标的单位方差对应。尺度是标准化监督目标的参数，不是原始收益波动率。预测分布和平台分数分别为

\[
p(z\mid x)=\sum_k\pi_k\,t_\nu(z;\mu_k,s_k),
\]

\[
\widehat y=\sum_k\pi_k\mu_k.
\]

保留原 Huber、排名、分支分配和路由损失，新增权重 0.2 的混合负对数似然；训练分支的分配代价同时增加权重 0.1 的单分支负对数似然。分配保持停止梯度，似然使用 log-sum-exp 和 Float64 计算。原 Huber 主项不会因尺度变大而自动减小，正下限防止尺度趋近零；这些是工程约束，不是完整优化稳定性证明。

记录似然、尺度、混合标准差和贴近尺度下限的比例，判断是否只改善了分布目标而没有提高 IC。尺度头对训练共享表示及路由产生梯度；部署仍只输出混合均值。参数只在分支末层少量增加，保留 fast 共享编码和现有输入缓存，不引入外部权重或额外行情字段。

每版一次正式训练，沿用验证 IC、动态早停和完整双集评估。训练损失定义变化后不与旧版总损失直接比较，不因更低似然或更好校准而替代 IC 晋级规则。

## 冻结实验反馈

[v1201 已完成](../solutions/v1201/README.md#冻结结果)：验证／测试 IC 为 0.090019／0.097396，均低于父 v1002；双集 Rank IC 为 0.082366／0.082965，退化更明显。八轮训练及验证约 102.57 分钟，完整覆盖。训练密度损失下降未转化为排序改善，暂不继续本系列的小权重扫描，不默认导出。该组合阴性结果不等于所有分布学习无效。
