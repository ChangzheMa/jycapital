# v7 六组独立对比实验

[文档目录](README.md) · [修改备忘录](v7-discussion.md) · [训练与评估约定](development.md)

六个实验均以已删除 LayerNorm 的冻结运行 `v7-20261016-no-layernorm-s42` 为父版本，代码各自位于独立目录，使用各自的模块、配置、训练与推理入口。当前交付为代码及检查，不含正式训练结果；基线 v7、历史权重和提交材料保持原样。

## 版本与变量

| 版本 | 备忘录修改组 | 本次实验变量 | 参数量 | 建议 run_id |
| --- | --- | --- | ---: | --- |
| [v7a](../solutions/v7a/README.md) | 1 | 取消隐藏有效性掩码；保留池化，mean 用全窗口、last 取窗口末端 | 2,421,185 | `v7a-no-mask-s42` |
| [v7b](../solutions/v7b/README.md) | 1 + 2 | 取消掩码，并用展平投影替代 last + mean | 5,185,985 | `v7b-no-mask-flatten-s42` |
| [v7c](../solutions/v7c/README.md) | 3 | 左右对称边界复制；保留核宽 3 和原膨胀率 | 2,421,185 | `v7c-replicate-padding-s42` |
| [v7d](../solutions/v7d/README.md) | 4 | 膨胀率固定 1，核宽逐块 3／5／7／9；保留左侧补零 | 3,957,185 | `v7d-dense-kernels-s42` |
| [v7e](../solutions/v7e/README.md) | 5 | 截面 z-score label，原始预测直接用 Huber 拟合 | 2,421,185 | `v7e-zscore-huber-s42` |
| [v7f](../solutions/v7f/README.md) | 6 | 仅预测时刻改为八维 one-hot，融合层输入由 642 改为 648 | 2,422,721 | `v7f-one-hot-clock-s42` |

各版本都直接继承基线 v7；仅 v7b 按用户指定组合修改 1 + 2，其余各改自己的一组。v7b 相对 v7 的差异包含两个变量，后续需额外比较 v7b 与 v7a，观察取消掩码条件下展平的增量效果。one-hot 仅获准进入本轮对比，尚未据结果决定纳入主方案。自适应池化、注意力池化及近期保留等后续讨论未纳入本轮。

## 统一训练设置

全部版本从头初始化，seed=42；沿用四频窗口 60／24／16／16、77 通道输入、隐藏宽度 160、残差 Dropout 0.35、预测头 Dropout 0.1、AdamW、学习率 `3e-4`、weight decay `1e-4`、梯度范数上限 1 和确定性设置。CUDA BF16 前向、FP64 损失统计、FP32 验证／测试。每次更新使用完整时间截面，截面等权。

协议为 [local-v2-ic](../evaluation/protocols/local-v2-ic.json)，按验证 IC 选模，使用统一动态早停，上限 100 轮。冻结最佳验证权重后计算测试结果，包括没有改善的版本。时间划分、覆盖率、指标解释与恢复规则统一见[开发约定](development.md)。

所有版本复用已有训练和留出集预处理目录：`outputs/preprocessing/v2-20261010`、`outputs/preprocessing/v2-holdout-20261011`。不需要为这些改动重新生成输入预处理；原始 Bar 缺失值填零、两个组完整标志及 Bar 时间编码均保留。

## 本轮明确的边界设置

- **v7e**：Huber `delta=1`，单位是标准化目标单位；总体标准差 `ddof=0`，不裁剪标签、不加固定收益倍数、不归一化预测。有限标签少于 30 或标签恒定时跳过该截面的训练更新，仍保留预测键；非零但很小的标准差用缩放后的等价公式计算，不加 epsilon 改写目标。标准化统计只构造监督目标。
- **v7e**：常数预测允许参与 Huber 更新，IC 无定义单独计数；不再按输入完全相同跳过截面。损失日志使用 `train_huber_loss`、`validation_huber_loss`，IC 独立报告，不要求 `loss = 1 - IC`。
- **v7e**：`factor_scale=1`，输出保留标准化目标单位。公共评估器仍对原始官方标签报告 IC／Rank IC 及原始标签差值 MSE／MAE；后两者不能解释为标准化目标上的拟合误差，也不能据此比较收益校准优劣。其余版本沿用 `factor_scale=100`。
- **v7f**：预测时刻固定为 UTC+08:00 的 `09:30、10:00、10:30、11:00、11:30、13:30、14:00、14:30`。未知时刻、非整分钟和 NaT 报错。历史 Bar 仍使用 sin/cos 和 log age。

这些值是本轮固定设置，不代表经过调参得到的最优设置。

## 6.1sol 执行顺序与完成条件

本轮由新 session 中的 6.1sol 执行训练与监控，全部完成后交回 astra 分析。预算为六个固定方案各一次正式训练，顺序为 **v7a → v7b → v7c → v7d → v7e → v7f**；同一时刻只运行一个实验。每版完成训练、验证和测试并核对产物后，自动开始下一版，不逐版再次索要启动许可。

1. 读取仓库 `AGENTS.md`、[开发约定](development.md)、[实验技能](../.agents/skills/jycapital-experiment/SKILL.md)、[监控约定](../.agents/skills/jycapital-experiment/references/monitoring.md)和对应版本 README。确认当前代码与配置，沿用现有 `.venv-training`、CUDA 环境及冻结预处理；核对父运行和数据目录存在。先检查是否已有同名运行或训练进程，避免重复启动。
2. 每版使用其 README 的命令与表中 run_id，添加 `--training-confirmed`。由后台进程运行，Windows 使用 `Start-Process` 时加 `-WindowStyle Hidden`；stdout、stderr、PID 和调度记录放在 `outputs/run-control/<run_id>/`。新实验的 `outputs/runs/<run_id>/` 由训练器创建，不为存放日志提前创建该目录。
3. 接入独立后台监控。正常情况下每小时汇总，用户询问或任务完成／异常时及时检查。数据指纹校验和月份加载可能长时间没有训练日志，应按阶段判断停滞，不能仅因日志静默就重启。只有实际验收过的通知路径才可声称结束对话后能自动继续；否则使用技能规定的工具等待方式。
4. 一版完成后，确认 `manifest.json` 的 `status` 和 `stage` 均为 `complete`，验证与测试指标和预测产物齐全，`best.pt` 哈希与 `selection.json` 一致。记录覆盖率、未定义 IC／Rank IC 截面和异常；指标没有改善也继续下一版，不因成绩较低追加训练或省略测试。
5. 若同名运行已完成，核对其源码、配置和产物后直接使用，不能重训覆盖。可恢复中断只在原配置、原源码和原 run 下使用 `--resume`，并沿用优化器、随机状态及早停计数。失败保留证据；如果必须改变研究变量或新开正式尝试，报告原因，不通过换 run_id 绕过本轮次数预算。

六版固定结构、seed、损失、早停和评估划分均以已提交配置为准。正常运行不调参，不训练基线 v7，不追加池化方案或其他组合；仅当前请求明确的训练及两套评估属于执行阶段。

训练器会在独立运行目录保存生效配置、协议、源码快照、数据指纹、环境、`history.csv`、`events.jsonl`、`last.pt`、`best.pt`、`selection.json`、训练曲线，以及验证／测试的预测、逐截面指标、月度指标和汇总。`manifest.json` 达到 `complete` 才代表训练和两套评估完成；中断或失败保留原记录并如实报告。

## 交回 astra 的结果

每完成一版，在本地 `outputs/checks/v7-ablation-results.json` 更新一条结果，保存从实际产物读取的以下字段；未完成项标明状态和原因，不填造指标。

- `solution_id`、run_id、运行目录、完成状态、父 run_id、实际源码／配置／最佳权重指纹。
- 最佳 epoch、完成 epoch 数、停止原因、实际早停耐心、参数量、设备与运行耗时。
- 验证／测试的 IC、Rank IC、对应 IR、MSE、MAE、全局／最低截面覆盖率、有效标签数和未定义 IC／Rank IC 截面数。
- `history.csv`、训练曲线、逐截面／月度指标及预测文件位置；v7e 另保留最佳验证与测试 Huber 损失和输出单位说明。
- 失败、中断、恢复或修复是否发生，以及对可比性的潜在影响；详细过程保留在忽略的运行日志中。

全部完成后给出六版状态和原始指标汇总，指出结果清单路径，然后停止等待用户切换至 astra。6.1sol 阶段不作最终归因或晋级判定，不执行 `assess.py --record`，不改冻结结果索引，不导出／提交平台材料，不自行追加 Git 提交或推送；比较分析和长期结果归档留给 astra。

后续分析至少比较各版与冻结 v7，并比较 v7b 与 v7a；历史最优和已有平台参照按[比较规则](../.agents/skills/jycapital-experiment/references/comparison.md)补充。现有 [compare.py](../scripts/compare.py) 可在数据指纹与评估协议一致时比较这些新 solution_id。结果未产生前不填写冻结成绩或晋级结论。

## 实现检查

[测试](../tests/test_v7_variants.py)覆盖配置变量隔离、参数量、掩码及汇总语义、对称复制边界、核宽及因果性、八时刻编码和输入一致性、Huber 数值与梯度、退化标签、合成数据上的训练／验证／测试与确定性断点恢复，以及 checkpoint／文本权重推理一致性。测试运行使用临时目录，不构成正式研究实验。

```powershell
.venv-training/Scripts/python.exe -W ignore::DeprecationWarning -c "import torch, unittest; result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests', pattern='test_v7_variants.py')); raise SystemExit(not result.wasSuccessful())"
```
