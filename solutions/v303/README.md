# v303：提高 Huber 二次区间的 GRU 对照

[文档目录](../../docs/README.md) · [持续研究](../../docs/continuous-research.md) · [候选清单](experiment.json) · [父版本 v301](../v301/README.md)

v30x 系列的损失敏感性对照，以 `v301-global-fp16-s42` 为父版本，从头初始化。唯一研究变量是将截面标准化目标上的 Huber `delta` 从 1 改为 2，增加较大误差的梯度权重，检验其对 Pearson IC、月份稳定性和平台价值的影响。

## 控制与假设

网络和参数量保持 **433,409**，三频窗口、75 通道、标签标准化、全局随机 batch 512、优化器、学习率、精度和动态早停均与 v301 相同。原标签不裁剪，不重新定义评估标签。训练和验证 Huber 均使用 delta=2，模型仍按验证 IC 选 checkpoint。

较大的 delta 扩大二次损失区间，对尾部误差施加更大梯度；它可能改善较大收益样本的拟合，也可能增加噪声敏感性。现有长窗口模型在部分月份相对 v7e 的 IC 退化，仅是需要继续检验的研究线索，不能据此宣称此修改必然有效。

不同 delta 的 Huber 损失绝对值不能直接跨版本排名。统一比较验证／测试 IC、Rank IC、IR、逐月和日内分布、覆盖以及换手代理；测试期已经被持续研究观察。

## 执行与成本

保持原目标构造函数和输入精度，复用现有 resident cache 与评估缓存，不重建同一份输入数据。计算图和递推长度不变，预期成本接近 v301 的实际运行；不会为这项损失对照扩大模型。

唯一正式运行为 `v303-huber-delta2-s42`，由训练 session 在 ready 后串行执行：

```powershell
.venv-training/Scripts/python.exe -u solutions/v303/v303_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v303-huber-delta2-s42 --evaluation-cache-mode stream --training-confirmed
```

实现检查覆盖损失实际使用 delta=2、网络及其他训练设置与父版本一致、未来值和梯度隔离、全空历史、文本权重恢复、完整训练入口及确定性断点恢复。正式行情结果以运行产物为准，平台材料在本地评估后决定。
