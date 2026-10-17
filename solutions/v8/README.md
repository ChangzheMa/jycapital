# v8：整合 v7 备忘录六组修改

[文档目录](../../docs/README.md) · [修改备忘录](../../docs/v7-discussion.md) · [模型配置](model-config.json) · [模型代码](v8_model.py) · [训练入口](v8_train.py)

父版本为冻结运行 `v7-20261016-no-layernorm-s42`。根据用户于 2026-10-18（UTC+08:00）的要求整合备忘录全部六组修改，同次确认将第 6 组预测时刻 one-hot 纳入 v8。当前交付为待训练代码，可继续讨论调整；尚无正式训练、验证／测试成绩或平台材料。

v8 的研究变量限定为备忘录六组修改，没有混入后续讨论的长窗口、删除 15 分钟分支或移除 Bar 时间通道；这些变更单独实现于 [v9](../v9/README.md)。代码中的额外整合工作仅为版本命名、配置／形状校验及无用辅助函数清理。配置范围和沿用模块由测试核对；原始 Parquet → 预处理 → 留出集转换 → 训练／验证／测试主入口已在临时合成数据上贯通，正式行情效果仍待训练评估。

## 组合方案

| 备忘录组 | v8 实现 |
| --- | --- |
| 1 取消模型内有效性掩码 | 输入投影、26 次卷积、残差相加及分支输出均不按原始 Bar 有效性清零；输入缺失值处理和组完整标志保留 |
| 2 展平投影 | 每路残差输出按通道、时间顺序展平，独立 Linear → GELU 投影为 160 维，替代 last + mean |
| 3 对称边界复制 | 每次时序卷积使用 `F.pad(..., mode="replicate")`，左右各复制 `(kernel - 1) // 2` 个端点 |
| 4 连续宽卷积 | dilation=1；1 分钟分支核宽 3／5／7／9，其余三路 3／5／7；每块两次卷积权重独立 |
| 5 截面标准化标签 + Huber | 完整时间截面的有限合法标签先做总体 z-score，再按月打乱样本；原始预测直接拟合，`delta=1` |
| 6 预测时刻 one-hot | 固定八时刻，融合输入为 648 维；历史 Bar 的 sin/cos、log age 和 77 通道布局保留 |

四路窗口长度分别为 60／24／16／16，隐藏宽度均为 160。展平投影输入维数为 9600／3840／2560／2560，输出各 160 维；拼接八维预测时钟后经过 `Linear(648,256) → GELU → Dropout(0.1) → Linear(256,1)`。全模型共 **6,723,521** 个参数，约为父版本的 **2.78 倍**，不含 LayerNorm；残差块 Dropout 为 0.35。

卷积可在历史窗口内部双向计算，整个窗口仅包含截至预测时点已结束的 Bar。中间隐藏位置不代表该历史时刻可立即生成的预测。取消隐藏掩码后，全空窗口可以产生非零分支表示；输入阶段仍保留缺失槽位并将非有限数值填零。

研究假设是通过保留全部隐藏状态、学习全窗口组合、扩大连续卷积核、改变监督目标及时间表示共同改善泛化。该版本同时改变多个变量，后续只能据正式结果判断整体方案；各组增量作用须结合 [v7a—v7f 独立实验](../../docs/v7-ablation-experiments.md)分析。展平投影和宽卷积增加参数、优化器状态与计算开销，实际速度、显存和效果需正式运行测量。

## 训练与输出约定

- 从头初始化，seed=42，AdamW，学习率 `3e-4`，weight decay `1e-4`，梯度范数上限 1。
- 完整截面标签按 `ddof=0` 标准化，不裁剪、不乘固定倍数。有限标签少于 30 或标签恒定的截面不参与优化，仍保留预测键；很小的非零标准差使用等价缩放公式，不加 epsilon 改写目标。
- 月份及月内样本按种子和 epoch 确定性打乱，batch=512，保留尾批；batch 内不重新标准化。训练 Huber 按样本均值，验证／测试 Huber 按截面等权。常数预测仍可参与 Huber 更新，IC 无定义单独记录。
- 按 `local-v2-ic` 的验证 IC 选择权重，沿用[统一动态早停](../../docs/development.md#后续训练的统一早停策略)，上限 100 轮。IC／Rank IC 对原始官方标签计算，测试使用冻结最佳验证权重。
- CUDA 支持时使用 BF16 前向，损失统计为 FP64，验证／测试为 FP32；每 64 个 batch 及月末保存样本游标、优化器、随机状态和早停计数。
- `factor_scale=1`，输出保留标准化目标单位，不对预测做截面标准化，也不解释为收益预测。公共评估器的原始标签差值 MSE／MAE 与标准化目标 Huber 分开解释。
- 预测时刻按 UTC+08:00 固定为 `09:30、10:00、10:30、11:00、11:30、13:30、14:00、14:30`。未知时刻、非整分钟或 NaT 报错。

模型、切窗、损失、预处理、训练、推理、文本张量和复现模块均位于本目录，不导入 v7a—v7f 代码。训练快照按 `solution_id=v8` 保存实际源码；[比较工具](../../scripts/compare.py)和[候选评估工具](../../scripts/assess.py)接受该方案标识。当前只准备训练和推理代码，平台导出及附件分片在后续材料阶段处理。

## 后续运行

复用冻结输入预处理，无需因本版改动重新生成：`outputs/preprocessing/v2-20261010` 和 `outputs/preprocessing/v2-holdout-20261011`。时间划分和数据边界见[开发约定](../../docs/development.md)。以下命令仅供后续明确开始训练时使用，本次未执行：

```powershell
.venv-training/Scripts/python.exe -u solutions/v8/v8_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v8-combined-s42 --training-confirmed
```

运行目录必须是新目录。中断恢复添加 `--resume`，配置、源码和数据必须与原运行一致。正式训练前复核本地验证最优、已有可信平台最优及父运行；当前配置以冻结 v7 为本地和父参照，未绑定可信平台最优。

## 实现检查

[test_v8.py](../../tests/test_v8.py)覆盖六组组合后的参数和维度、全部复制边界、隐藏掩码取消和展平顺序、输入与父版本一致及无未来 Bar、八时刻编码、Huber 数值和梯度、目标先标准化后打乱、默认 batch 的 CUDA 确定性反向传播、合成数据的训练／评估及精确断点恢复、checkpoint 与文本张量推理一致性。合成检查使用临时目录，不构成正式研究训练。

```powershell
.venv-training/Scripts/python.exe -W ignore::DeprecationWarning -c "import torch, unittest; result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests', pattern='test_v8.py')); raise SystemExit(not result.wasSuccessful())"
```
