# v9：三路固定交易日窗口与 75 通道输入

[文档目录](../../docs/README.md) · [代码父版本 v8](../v8/README.md) · [模型配置](model-config.json) · [切窗](v9_data.py) · [网络结构](v9_model.py) · [训练入口](v9_train.py)

依据用户于 2026-10-18（UTC+08:00）确认的方案，在 v8 六组备忘录整合版上调整输入窗口及对应模型结构。运行 `v9-trading-day-windows-s42` 已完成训练及双集评估：验证 IC 0.067835、测试 IC 0.077534，未达到改善条件。完整指标见[冻结汇总](../../experiments/completed-models-20261020.json)，合成实现检查与正式效果分开记录。

## 相对 v8 的变化

| 项目 | v8 | v9 |
| --- | --- | --- |
| 分支 | 1／5／15／30 分钟 | 1／5／30 分钟，删除 15 分钟分支 |
| 窗口 | 60／24／16／16 根，截止预测时刻滚动截取 | 480／240／160 根，按完整交易日固定位置 |
| 每根 Bar 输入 | 72 个数值＋2 个组完整标志＋sin/cos＋log age，共 77 维 | 72 个数值＋2 个组完整标志＋`bar_has_value`，共 75 维 |
| 分支输出 | 各 160 维 | 各 160 维 |
| 融合输入 | 4×160＋8＝648 维 | 3×160＋8＝488 维 |
| 总参数量 | 6,723,521 | **25,458,401** |

数值变换、字段顺序、Huber 目标、训练 batch、优化器、种子、精度、验证 IC 选模和动态早停沿用 v8。保留无 LayerNorm、无隐藏有效性掩码、对称复制填充、dilation=1、逐块扩大卷积核及展平投影。多个输入因素同时调整，后续结果只能首先归因于整体方案。

## 固定交易日窗口

每个完整交易日按 09:30—11:30、13:00—15:00 的 240 分钟建立规则槽位，Bar 时间为结束时刻。输入按日期、日内时间从早到晚排列。

| 分支 | 窗口范围 | 根数 | 展平投影 |
| --- | --- | ---: | --- |
| 1 分钟 | 上一交易日全天＋今天全天；今天预测点之后的所有通道补 0 | 2×240＝480 | `Linear(76800,160)` → GELU |
| 5 分钟 | 今天之前最近 5 个完整交易日，不含今天 | 5×48＝240 | `Linear(38400,160)` → GELU |
| 30 分钟 | 今天之前最近 20 个完整交易日，不含今天 | 20×8＝160 | `Linear(25600,160)` → GELU |

例如在 10:30 预测，1 分钟分支包含上一交易日 240 个槽位、今天已结束的 60 个槽位及未来 180 个全零槽位；10:30 恰好结束的 Bar 可以进入输入。5／30 分钟分支当天所有预测时刻使用相同的历史日期范围。

交易日历来自授权股票池的全市场日期。周末、节假日按真实交易日期跳过，不以固定自然日或工作日近似；删除原有 `history_days=20` 的读取和有效性限制。个股停牌、单根 Bar 缺失不改变网格位置。历史数据起点不足时左侧补零，已知交易日上没有数值时也补零，保留全部股票池预测键。

训练从冻结预处理文件读取，平台推理查询历史交易日期并顺序处理原始行情；两者共用 [WindowBlock](v9_data.py)。推理按频率加载所需的历史交易日，保留缺参股票的因果归一化历史热身；不查询 15 分钟源或标签表。

## 输入与网络

75 个通道顺序固定如下：

| 索引 | 含义 |
| --- | --- |
| 0—35 | 36 个原始字段对应的 z 值 |
| 36—71 | 同顺序的 c 值 |
| 72 | 非盘口组的全部 z/c 数值是否有限且时间合法 |
| 73 | 盘口组的全部 z/c 数值是否有限且时间合法 |
| 74 | `bar_has_value`：该槽位是否至少有一个有限且时间合法的数值 |

归一化后的合法零值仍是有效观测；组内部分缺失时保留其余数值。两个组标志均为 0 时，`bar_has_value` 仍可为 1。未来位置全部 75 通道均为 0，填充不写回归一化统计量。这个标志只作为输入，不恢复隐藏层强制清零；全空输入经过网络后允许产生非零隐藏表示。

Bar 的 sin/cos 和 log age 均移除。固定位置表达交易日序号及日内槽位，模型不再直接获得周末／长假对应的自然时间跨度。预测时刻仍使用 UTC+08:00 的八维 one-hot，顺序为 `09:30、10:00、10:30、11:00、11:30、13:30、14:00、14:30`；未知时刻、非整分钟及 NaT 报错。

1 分钟分支保留 4 个残差块，核宽 3／5／7／9；5／30 分钟分支各保留 3 个残差块，核宽 3／5／7。每块两次独立卷积，隐藏宽度 160，Dropout 0.35。三路展平投影后拼接预测时钟，预测头为 `Linear(488,256) → GELU → Dropout(0.1) → Linear(256,1)`。

长窗口使展平投影参数和激活增大。默认 batch 512 已通过本机 CUDA 合成前向、反向和 AdamW 更新，FP32 验证 batch 1024 也通过检查；这些检查不代表完整行情训练的耗时、显存上限或效果。

## 训练、复现与追溯

训练约定见 [v8](../v8/README.md#训练与输出约定)及[开发约定](../../docs/development.md)。`parent_solution_id=v8` 表达代码来源；设计时没有指定父权重，冻结运行的 `parent_run_id` 保持为空，没有继承 v8 权重。实际方案比较使用已完成 v8 作为代码父版本对照；新实验的历史最优以当时已完成记录重新核对，不沿用配置中的旧参照冒充当前最优。

可复用已有四频预处理目录，v9 只加载、核验和记录其中 1／5／30 分钟分区及统计量，无需为这次切窗修改重新拟合。需要独立复现时，本目录的 `v9_prepare.py`、`v9_apply.py` 生成三频数据；模型推理及训练不导入其他 `solutions/` 版本代码。

以下是历史运行入口；新实验须换用独立 run_id，不重跑覆盖已经完成的目录：

```powershell
.venv-training/Scripts/python.exe -u solutions/v9/v9_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v9-trading-day-windows-s42 --training-confirmed
```

独立复现预处理入口如下。它们要求完整的授权原始数据：

```powershell
.venv-training/Scripts/python.exe solutions/v9/v9_prepare.py --run-dir outputs/preprocessing/v9-train --protocol evaluation/protocols/local-v2-ic.json
.venv-training/Scripts/python.exe solutions/v9/v9_apply.py --training-run outputs/preprocessing/v9-train --run-dir outputs/preprocessing/v9-holdout
```

新运行保存实际配置、源码、协议、数据指纹、归一化统计量和 checkpoint。断点恢复必须使用原配置、原源码和原数据；冻结最佳验证权重后评估测试。公共比较工具按标签／股票池快照核对可比性，允许版本使用不同输入方案。

已保存冻结训练权重，但因未改善而没有导出平台提交包。若另行要求导出，须按 50 MB 单文件上限处理文本参数，不能直接沿用 v7 的两文件导出器。

## 实现检查

- [v8 范围与行为测试](../../tests/test_v8.py)：验证 v8 配置恰为备忘录六组整合，预处理及损失等实现与对应来源一致。
- [v9 模型与切窗测试](../../tests/test_v9.py)：三路参数和形状、75 通道、全部八个预测截点、长假／缺 Bar／不足历史、未来修改不影响输入、Huber 和确定性断点恢复。
- [v9 推理链路测试](../../tests/test_v9_pipeline.py)：原始行情与冻结预处理一致、跨月和长假、整段与分段请求一致、动态历史热身、checkpoint／文本张量及独立内嵌源码推理。
- [完整入口测试](../../tests/test_v8_v9_entrypoints.py)：临时合成 Parquet 上实际执行 v8／v9 的 prepare、apply 和训练主入口，检查验证／测试、源码快照及完整运行产物。不会读取正式行情或写入正式实验索引。

```powershell
.venv-training/Scripts/python.exe -W ignore::DeprecationWarning -c "import torch, unittest; suite=unittest.TestSuite(unittest.defaultTestLoader.discover('tests', pattern=p) for p in ('test_v8.py', 'test_v9*.py', 'test_v8_v9_entrypoints.py')); r=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not r.wasSuccessful())"
```
