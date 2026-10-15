# Pearson IC 损失的定义与验证口径

[文档目录](README.md) · [v4 实现与结果](../solutions/v4/README.md) · [损失实现](../solutions/v4/v4_loss.py) · [评估器](../evaluation/baseline_evaluation.py)

纯 Pearson IC 损失使同一份预测上的优化目标与截面 IC 严格对应。网络结构、生效配置、冻结权重及实测结果由 v4 方案和结果摘要保存。

## 标签与目标

标签使用原始有限的 `vwap_return`，沿用标签时间边界；计算损失时转为 FP64，不裁剪收益、不排名、不高斯化。未来收益仅作为监督标签。

在预测时刻 t，以标签有限且边界合法的股票组成截面。令模型分数为 s、标签为 y：

\[
u_i=s_i-\bar{s},\qquad v_i=y_i-\bar{y}
\]

\[
IC_t=\frac{\sum_i u_iv_i}{\sqrt{\sum_i u_i^2}\sqrt{\sum_i v_i^2}}
\]

\[
L_t=1-IC_t,\qquad L=\frac{1}{T}\sum_t L_t=1-\operatorname{MeanIC}
\]

各有效截面等权，不按股票数量加权。预测、均值和范数均保留计算图；损失不混合 Huber／MSE、不取绝对 IC 或 IC 平方。AdamW 的解耦 weight decay 不计入该 loss 数值。

预测和标签分别按截面标准化，标准差采用 `ddof=0` 时：

\[
\frac{1}{2n}\sum_i(z_{s,i}-z_{y,i})^2=1-IC_t
\]

该恒等式的值和梯度可用于独立验收。

## 完整截面

一次优化器更新使用一个完整预测时间戳的股票截面，股票数来自实际股票池；不按交易日混组，不截取固定条数，也不丢弃尾部截面。多个截面应分别计算 IC 后等权平均，不能先拼成一次全局相关系数。

分块前向必须保留计算图，拼接同一截面的全部分数后再计算损失。分别计算半截面 IC 再平均不等价于完整截面目标。月份、截面顺序及 Dropout RNG 由种子控制，断点恢复需保存优化器、游标和随机状态。

## 数值与退化规则

相关统计在 autocast 外使用 FP64。用分别归一化后的中心化向量计算相关系数，避免任意 epsilon 改写目标。

| 情况 | 处理 |
| --- | --- |
| 少于 30 个有效标签，或标签恒定 | IC 无定义，记录原因；不记为 0 |
| 有效标签股票的完整输入全部相同 | 可按训练前固定、与权重无关的清单跳过优化器更新；保留全部预测键 |
| 非结构性常数预测 | 恢复同一 RNG 后重试 FP32 前向；仍为常数则报错，不静默排除 |
| 非零但很小的方差 | 使用稳定归一化；不靠阈值删截面来提高平均 IC |
| 非有限预测、损失或梯度 | 报错，不补零继续训练 |
| 验证／测试常数预测 | 按原评估器报告未定义截面，禁止借此减少分母获得晋级 |

结构性退化清单不能由预测误差或已观察 IC 决定。所有截面均应保留输出覆盖证据。

## 选模与一致性

v4 的 [local-v2-ic](../evaluation/protocols/local-v2-ic.json) 与 [local-v1](../evaluation/protocols/local-v1.json) 仅在协议标识和选择指标上不同。显式兼容检查只允许这两项差异；日期、标签、股票池、聚合、指标算法和数据指纹仍需一致。旧模型保持原先冻结的权重选择依据。

每轮完整验证以 FP32 生成固定预测，独立核验 `validation_ic_loss + validation_ic_mean = 1`，容差为 `1e-8`。严格最高验证 IC 保存为最佳权重；冻结后评估测试。历史 v4 早停按其[配置](../solutions/v4/model-config.json)复现，v7 及以后采用[统一早停策略](development.md#后续训练的统一早停策略)。

在线训练 IC 取各截面更新前的预测，不等于轮末模型重评；训练与验证属于不同集合，仍可出现不同趋势。纯 IC 对正比例缩放和平移不敏感，不校准收益幅度，固定 `factor = raw_score / 100` 只能解释为因子打分。

## 验证依据

[损失测试](../tests/test_v4_loss.py)覆盖有限差分、标准化 MSE 等价梯度、缩放／平移／置换不变性、截面等权及退化情况；[训练测试](../tests/test_v4_train.py)覆盖 IC 选模、早停和确定性恢复。冻结运行的实际误差及输出尺度见 [v4 结果摘要](../experiments/v4-tcn.json)。

数学恒等式为直接推导。基础算子参考 [PyTorch 2.11 corrcoef](https://docs.pytorch.org/docs/2.11/generated/torch.corrcoef.html) 和 [cosine_similarity](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.cosine_similarity.html)，查阅日期为 2026-10-14（UTC+08:00）。
