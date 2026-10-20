# v304：时钟门控的日内／跨日 GRU 读出

[文档目录](../../docs/README.md) · [持续研究](../../docs/continuous-research.md) · [候选清单](experiment.json) · [父版本 v301](../v301/README.md)

v30x 系列的读出改进，以 `v301-global-fp16-s42` 为父版本，从头初始化。研究假设：只读取跨日 GRU 状态可能压缩掉一部分近期信息；让预测时钟参与选择日内状态和跨日状态，有机会改善日内表现。

## 门控结构

保留原日内／跨日单向 GRU 和全部输入配置。每个分支同时取最后已开始交易日的日内状态与跨日状态，将两者和八维预测时钟拼接，经 `Linear(200,96) → Sigmoid` 产生逐通道门，再加权合并两级状态。门偏置初始化为零，融合头仍接收三个 96 维分支向量与时钟。

总参数 **491,297**，较 v301 增加 57,888。原先已经计算了两级状态，因此没有增加递推长度或新的长窗口大矩阵。门控饱和选择跨日状态时可恢复父版本的读出行为。

09:30 读取昨日的两级状态；其他时点只读取已发生槽位。未来 Bar 在进入网络前清零，全空历史仍产生有限输出。该方案不根据测试期月份或时段设置手工切换规则。

## 训练与验证

保留截面标准化 Huber delta=1、全局随机 batch 512、原学习率与优化器、FP16 输入缓存、FP32 GRU 状态、验证 IC 选模和动态早停。新增门控属于整体读出修改，不能将其效果单独归因于时钟或直连通路中的某一项。

唯一正式运行为 `v304-clock-gated-gru-s42`，由训练 session 在 ready 后串行执行：

```powershell
.venv-training/Scripts/python.exe -u solutions/v304/v304_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v304-clock-gated-gru-s42 --evaluation-cache-mode stream --training-confirmed
```

实现检查覆盖门控退化到父版本读出的对照、全部参数梯度、未来值与梯度隔离、八个预测时点、空历史、分批推理、文本张量往返、完整训练入口和确定性恢复。预期训练成本接近 v301；实际默认批量的 GPU 开销由训练 session 在模型间隙预检。
