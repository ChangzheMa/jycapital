# v202：学习时间压缩的 Patch Mixer

[文档目录](../../docs/README.md) · [研究约定](../../docs/continuous-research.md) · [候选清单](experiment.json) · [父版本 v201](../v201/README.md)

v20x 系列的第二版。父版本是已完成的 `v201-global-fp16-s42`，本版从头初始化，不继承父权重。研究假设：逐 Bar 的通道 MLP 占据大量计算；先联合投影短片段，可以在保留长历史覆盖的同时减少计算，并通过共享表示抑制噪声。

## 变化与控制

保持三频固定交易日输入、75 通道、截面标准化 Huber、全局随机 batch 512、FP16 输入缓存、验证 IC 选模及统一动态早停。评估仍使用原始官方标签，不改变覆盖和时间划分。

| 分支 | 原始 Bar 数 | 每个 patch 的 Bar 数 | Mixer token 数 |
| --- | ---: | ---: | ---: |
| 1 分钟 | 480 | 8 | 60 |
| 5 分钟 | 240 | 4 | 60 |
| 30 分钟 | 160 | 4 | 40 |

patch 内按时间和字段顺序展平，经学习的线性投影进入 96 维；不手工计算价量统计特征。两层低秩时间／通道 Mixer、32 维时间瓶颈、4 个分解读出槽位和融合头沿用 v201。所有未来 Bar 在投影前清零，全未来 patch 的隐藏状态继续清零；历史缺失位置保留字段和缺失标志。

总参数 **550,093**，父版本为 552,973。主要变化是表示的时间分辨率；不能把潜在效果提升单独解释为训练加速。

## 本地执行与验证

唯一正式运行标识为 `v202-patch-mixer-s42`。由训练 session 在候选状态为 ready 后串行执行：

```powershell
.venv-training/Scripts/python.exe -u solutions/v202/v202_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v202-patch-mixer-s42 --evaluation-cache-mode stream --training-confirmed
```

复用既有 resident cache；源码中的目标构造与父版本相同。验证缓存采用保留内存的分块模式。父版本保护仅覆盖冻结源码、权重、配置和评估产物，不锁定其他 session 可追加的监控／分析记录。

CPU 合成数据验证覆盖未来输入及梯度隔离、空历史、批量一致性、全部参数梯度、文本权重精确恢复、完整原始数据预处理到训练／验证／测试入口，以及中断恢复后的逐参数和优化器状态一致。GPU 默认批量预检由训练 session 在模型间隙执行。

2026-10-20 的单 CPU 线程、batch 8 合成训练步中，本版中位耗时约 0.027 秒，v201 约 0.111 秒；仅为设计阶段测量，不代表 GPU 完整训练加速或正式效果。正式本地成绩以对应 run 的双集产物为准；达到平台候选条件后再导出提交材料。
