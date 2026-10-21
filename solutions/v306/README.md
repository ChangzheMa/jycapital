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

## 冻结结果

`v306-huber-delta4-s42` 已完成 12 轮，按验证 IC 选择第 7 轮。累计训练及逐轮验证 11,620.511 秒，约 3.23 小时，与父版本的 3.16 小时接近。双集覆盖率和最低截面覆盖率均为 100%，IC／Rank IC 未定义截面均为 0。

| 指标 | v303 验证 | v306 验证 | v303 测试 | v306 测试 |
| --- | ---: | ---: | ---: | ---: |
| IC | 0.085625 | 0.082907 | 0.090160 | 0.087217 |
| Rank IC | 0.083970 | 0.078575 | 0.084293 | 0.078165 |
| IC IR | 0.785680 | 0.792288 | 0.763490 | 0.778620 |
| Rank IC IR | 0.881964 | 0.895593 | 0.808206 | 0.842843 |
| 日内单边换手代理 | 0.633187 | 0.638864 | 0.639189 | 0.631623 |

父版本与本地验证最优均为 v303。验证／测试 IC 分别下降 0.002718／0.002942，Rank IC 下降 0.005395／0.006128；IC 改善月份仅 1／6 和 2／6，Rank IC 改善月份均为 1／6。双集 IR 提高，换手代理在验证期变差、测试期改善，不能把稳定性收益扩展为全面改善。代理不等于平台 BARRA、中性化及交易成本后的指标。

**未达到验证 IC 晋级条件，不导出平台材料。** 本次结果不支持继续增大 Huber delta，后续组合方案优先保留 delta=2；不据此展开细密的 delta 搜索。损失绝对值随 delta 改变，不直接跨版本排名。单种子且测试期已持续观察，不能宣称统计显著性；平台成绩尚未可靠绑定本地 run 与权重。

依据：[完整比较](../../experiments/v306-gru.json)、[信号诊断](../../experiments/v306-signal-diagnostics.json)。模型 SHA-256 为 `fe4baa616b6dae95d394a6beffc0b0449885833f84c4bab4c45daebb777b0a9b`。
