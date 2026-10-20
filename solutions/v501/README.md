# v501：用 EMA 参数做验证选模的 GRU

[文档目录](../../docs/README.md) · [持续研究](../../docs/continuous-research.md) · [候选清单](experiment.json) · [父版本 v303](../v303/README.md)

v50x 开启训练方法系列，以 `v303-huber-delta2-s42` 为父版本，从头初始化。保持网络、输入、标准化收益 Huber delta=2、全局采样和原始梯度更新方法，用独立的参数指数滑动平均（EMA）副本做验证和 checkpoint 选择，检验参数平滑能否提高泛化稳定性。

## 参数平均与选模

每个优化步骤先更新原始参数 \(\theta_t\)，再更新平均参数 \(\bar{\theta}_t\)。第一轮训练采用逐步复制，使平均参数与原始参数一致；从第二轮起使用：

\[
\bar{\theta}_t=\beta\bar{\theta}_{t-1}+(1-\beta)\theta_t,\qquad \beta=0.9999.
\]

平均参数使用 FP32，不参与反向传播，优化器只维护原始模型的状态。每轮仅评估一次 EMA 模型，按其原始收益标签上的验证 IC 选 checkpoint，并使用原动态早停规则。第一轮结束时 EMA 与原模型一致，之后选模和早停依据 EMA 预测，实际完成轮数可能与 v303 不同。

训练损失来自原始模型，验证损失来自 EMA 模型；运行记录明确保存这一区别。`best.pt` 保存被选中的 EMA 参数，测试和平台推理均加载这份固定参数，不在推理期间继续更新平均值。

## 控制、成本与恢复

推理网络仍为 **433,409** 参数，输入、batch 512、学习率、优化器、Huber delta=2、精度和种子与 v303 相同。训练增加一份约 1.7 MB 的 FP32 权重副本及与参数量成正比的逐步平均运算，不增加递推长度或第二次每轮验证。复用现有行情、目标和评估缓存，实际开销以训练记录为准。

`last.pt` 同时保存原始参数、EMA 参数、EMA 更新次数、优化器、随机状态和采样进度。EMA 更新次数必须与已完成优化步骤一致；缺少平均参数时拒绝恢复，不能以当前原始参数重新初始化 EMA。平均过程不消耗随机数，原始参数与平均副本不共享可变存储。

## 执行与检查

唯一正式运行为 `v501-ema-gru-s42`，由训练 session 在 ready 后串行执行：

```powershell
.venv-training/Scripts/python.exe -u solutions/v501/v501_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v501-ema-gru-s42 --evaluation-cache-mode stream --training-confirmed
```

CPU 合成检查已通过，覆盖暖启动复制、平均公式、随机状态不变、原始优化路径的合成数据对照、真实选中 EMA 权重、平均状态的确定性恢复，以及月内和全局缓存模式的完整训练入口。独立加载所选 checkpoint 后，在相同参数梯度标志下的验证预测逐项一致；常规模型构造下的推理差异使用仓库已有 FP32 容差检查。

未来信息隔离、输出键与覆盖、文本权重恢复保持原检查。该候选尚无正式行情结果；参数平均是研究假设，不代表已经改善平台分数。
