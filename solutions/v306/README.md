# v306：Huber delta=4 的 GRU 对照

[文档目录](../../docs/README.md) · [持续研究](../../docs/continuous-research.md) · [候选清单](experiment.json) · [父版本 v303](../v303/README.md)

以 `v303-huber-delta2-s42` 为父版本，从头初始化。v303 将 Huber delta 从 1 提高到 2 后，双集 IC 与 IC IR 改善，但 Rank IC 和换手代理存在代价。本版只将 delta 从 2 提高到 4，检验继续增加较大误差的梯度权重是否仍有收益。

网络保持 **433,409** 个参数。输入窗口、75 通道、完整截面收益标准化、全局随机 batch 512、优化器、学习率、精度、种子和动态早停均沿用 v303。原始收益标签不裁剪，正式验证／测试协议和 checkpoint 选取指标不变。复用原行情、训练目标及评估缓存。

较大的 delta 使更多误差处于二次损失区间，可能加强收益幅度信息，也可能增加尾部噪声和换手；不能从 v303 的改善推断本版必然更好。比较验证／测试 IC、Rank IC、IR、逐月稳定性和换手代理；不同 delta 的 Huber 损失绝对值不直接跨版本排名。

唯一正式运行为 `v306-huber-delta4-s42`。候选 ready 后由训练 session 串行执行：

```powershell
.venv-training/Scripts/python.exe -u solutions/v306/v306_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v306-huber-delta4-s42 --evaluation-cache-mode stream --training-confirmed
```

CPU 合成检查已通过，覆盖 delta 的实际损失值、其余配置与父版本一致性、未来值及梯度隔离、文本权重精确恢复、完整入口和确定性断点恢复。单步计算图与 v303 相同，实际训练轮数仍由验证 IC 和早停决定。
