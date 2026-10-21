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

实现检查覆盖损失实际使用 delta=2、网络及其他训练设置与父版本一致、未来值和梯度隔离、全空历史、文本权重恢复、完整训练入口及确定性断点恢复。正式训练和双集评估已完成，冻结结果如下。

## 冻结结果与平台材料

`v303-huber-delta2-s42` 于 2026-10-21（UTC+08:00）完成 12 个 epoch，按验证 IC 选择第 7 个 epoch。验证与测试均为完整股票池覆盖，最低截面覆盖率为 100%，没有未定义 IC 或 Rank IC 截面。完整指标、逐月对比和权重指纹见[比较摘要](../../experiments/v303-gru.json)。

| 指标 | v303 验证 | v301 验证 | v303 测试 | v301 测试 |
| --- | ---: | ---: | ---: | ---: |
| IC | 0.085625 | 0.082544 | 0.090160 | 0.086791 |
| Rank IC | 0.083970 | 0.084265 | 0.084293 | 0.085596 |
| IC IR | 0.785680 | 0.739916 | 0.763490 | 0.715680 |
| Rank IC IR | 0.881964 | 0.858989 | 0.808206 | 0.785475 |
| 单边换手代理 | 0.633187 | 0.623516 | 0.639189 | 0.628674 |

相对父版本 v301，验证 IC 提高 0.003081、测试 IC 提高 0.003369，验证和测试各 6／6 个月的 IC 均改善。原网络及训练规模没有扩大：433,409 参数，RTX 4080 上 12 轮训练连同逐轮验证累计约 3.16 小时，父版本约 3.20 小时；不含预处理和训练后的最终评估。

代价是双集 Rank IC 略低、换手代理有所增加，不能宣称所有指标同时改善。信号持续性、逐月及日内代理结果见[诊断记录](../../experiments/v303-signal-diagnostics.json)，方法和平台口径的区别见[说明](../../docs/signal-diagnostics.md)。本版达到本地改善条件，建议作为下一份平台评测候选；官方总分仍须平台反馈，当前结论为单种子、已经观察过的测试期结果。

原始平台材料从冻结源码、配置、统计量和第 7 轮权重导出：[v303-submission.ipynb](../../submissions/v303-huber-delta2-s42/v303-submission.ipynb) 与 [v303-model.json](../../submissions/v303-huber-delta2-s42/v303-model.json)，分别为 8,007 和 15,272,894 字节。以下保留原版验证记录；后续评测推荐采用[共享编码加速材料](../../docs/v303-inference-fast.md)，使用相同冻结模型和权重。

独立 CPU 核验覆盖 2024-01-02 至 2024-01-05、2024-12-30 至 2024-12-31，共 **48,000 行**。34 个权重张量经九位有效数字文本逐位恢复，22 份训练源码恢复及训练入口导入通过。平台原始行情入口与原 checkpoint 加冻结预处理数据，在同一 CPU 后端及逐日批量边界下的输出完全一致；输出键、覆盖率、类型和有限性均通过检查，Notebook 调试单元格也已本地执行。

沿用 [v301 的数值核验方法](../v301/README.md#冻结结果与平台材料)：同后端保持 `atol=2e-7、rtol=1e-4`；同后端验证通过后，再以 `atol=1e-5、rtol=1e-4` 单独检查原 CUDA 冻结预测。两段跨后端最大绝对差为 4.35×10⁻⁶ 和 5.87×10⁻⁶，原较严容差的未通过计数保留在[清单](../../experiments/v303-huber-delta2-s42-submission.json)。核验区间内固定两端 10% 的组合权重没有变化，IC 差异的绝对值小于 4.6×10⁻⁸；详见[数值影响记录](../../experiments/v303-numeric-validation.json)。这些局部核验不代替平台运行及官方评分。

## 推理加速材料

2026-10-21 增加 `v303-huber-delta2-s42-fast` 材料版本，通过股票日编码及切窗复用提速，原训练 run、参数、完整历史和 FP32 计算保持不变。12 个交易日、96,000 行的本地 CPU 4 线程对照由 204.1 秒降到 69.4 秒，约加速 2.94 倍，预测逐行一致；另有验证／测试共 48,000 行独立 checkpoint 对照通过。完整范围与局限见[加速说明](../../docs/v303-inference-fast.md)。

上传同一加速目录内的 [Notebook](../../submissions/v303-huber-delta2-s42-fast/v303-submission.ipynb) 和 [模型 JSON](../../submissions/v303-huber-delta2-s42-fast/v303-model.json)；[新版清单](../../experiments/v303-huber-delta2-s42-fast-submission.json)记录推理适配器覆盖及验证结果。原版保留追溯；加速版尚待实际平台运行验证。
