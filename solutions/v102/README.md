# v102：时钟条件与近期状态共同读出的 Patch Transformer

[文档目录](../../docs/README.md) · [研究约定](../../docs/continuous-research.md) · [候选清单](experiment.json) · [父版本 v101](../v101/README.md)

v10x 系列的第二版，以冻结运行 `v101-global-fp16-migrated-s42` 为对照，从头全局训练。研究假设：单独读取 CLS 可能弱化最近信息及不同时刻的需求；用预测时钟和最后可见 patch 共同查询历史，有机会改善日内信号的不均衡。

## 读出变化

保留三路窗口、8／4／4 的 patch 长度、96 维编码、两层四头 Pre-LayerNorm Transformer 及 math 注意力后端。编码结束后，对全部 token 归一化，取最后一个已开始 patch；它与八维预测时钟共同产生查询向量，对可见 token 做点积注意力，再将汇总向量与最后 patch 拼接投影回 96 维。三路融合预测头保持不变。

未来 Bar 在投影前清零，全未来 patch 不能作为编码器或读出注意力的 key/value。CLS 始终可见，历史全空仍输出有限值。不根据已经观察过的测试月份或时点手工选择模型。

总参数 **770,465**，v101 为 656,993。新增读出约增加 17.3% 参数，不引入长窗口大展平层。

## 训练控制与解释边界

沿用截面标准化 Huber、全局随机 batch 512、FP16 输入缓存、FP32 验证、固定本地协议和动态早停。使用已验证的分块评估缓存。与历史 v101 比较同时包含“从头全局训练”相对“月内训练后迁移”的轨迹差异，因此先解释为整体候选效果，不能单独归因于读出。

唯一正式运行标识为 `v102-clock-readout-s42`，由训练 session 在 ready 后执行：

```powershell
.venv-training/Scripts/python.exe -u solutions/v102/v102_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v102-clock-readout-s42 --evaluation-cache-mode stream --training-confirmed
```

CPU 合成检查覆盖全部时刻的未来输入及梯度隔离、空历史、分批推理、文本权重往返、完整训练入口和确定性断点恢复。GPU 默认批量预检由训练 session 在模型间隙执行。

2026-10-20 的单 CPU 线程、batch 8 合成训练步中，本版中位耗时约 0.043 秒，v101 约 0.041 秒；该测量不能替代 GPU 吞吐。正式效果待本地验证／测试，不提前生成平台提交包。
