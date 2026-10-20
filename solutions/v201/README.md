# v201：低秩时间／通道 Mixer

[文档目录](../../docs/README.md) · [三个研究家族](../../docs/research-branches.md) · [代码父版本 v9](../v9/README.md) · [模型](v201_model.py) · [模型配置](model-config.json)

v20x MLP 混合器研究家族的首版，以 v9 为代码父版本。研究假设是以时间／通道分解、共享权重及低秩约束，学习长窗口中的有效组合并控制容量。全局运行 `v201-global-fp16-s42` 已完成训练、验证和测试，达到本地改善条件；冻结结果及平台材料见下文。

## 主干

每路保留 480／240／160 个原始槽位，先将 75 通道线性投影为 96 维。两层 Mixer 交替执行：

1. 时间混合：LayerNorm 后转置，所有隐藏通道共享 `Linear(T,32) → GELU → Dropout(0.1) → Linear(32,T)`，再做残差相加。
2. 通道混合：LayerNorm 后，所有时间位置共享 `Linear(96,192) → GELU → Dropout(0.1) → Linear(192,96)`，再做残差相加。

时间混合包含非线性，32 表示瓶颈宽度，不表示整个非线性映射严格只有秩 32。未来隐藏位置在每次混合后清零，历史缺失位置保留，不依据有效值比例删除样本。

汇总时先做 LayerNorm，各隐藏通道共享 `Linear(T,4)`，再将 `96×4` 展平并经 `Linear(384,96) → GELU` 得到分支表示。三路拼接八维预测时钟，预测头为 `Linear(296,256) → GELU → Dropout(0.1) → Linear(256,1)`。

总参数 **552,973**。主干没有卷积或注意力；固定的可学习时间权重表达位置，所有历史槽位可以在时间混合阶段交互。该方案的适应性与容量取舍、共同监督及检查入口见[研究家族说明](../../docs/research-branches.md)。

## 全量随机训练模式

当前默认配置采用[全量 FP16 常驻缓存与全局 shuffle](../../docs/resident-training.md)：基础行情按股票／交易日去重，CPU 缓存与传输使用 FP16，有限极值通过稀疏 FP32 记录恢复，双缓冲预取下一批。label、损失、batch 大小和选模规则保持原定义；验证／推理采用相同输入舍入，验证数据按有限交易日块加载。

原月内 FP32 配置保存在 [model-config-monthly.json](model-config-monthly.json)。全局模式的运行状态以各 run 冻结记录为准；既有月内运行只代表其冻结代码和配置，不能直接切换模式续跑。缓存容量估算、构建命令、内存预算和完整验证范围见上面的统一说明。

验证与测试默认使用[评估缓存与异步双缓冲](../../docs/resident-training.md#v201v301-验证与测试加速)，按内存预算选择全量常驻或分块加载，保留原 FP32 模型计算和完整选模口径。新 run 自动启用；旧全局 run 停止后用 `--resume --upgrade-evaluation` 显式升级恢复，详见[恢复说明](../../docs/resident-training.md#已有全局-run-的恢复)。

## 后续使用

目录内包含独立预处理、切窗、训练、原始行情推理、文本张量和复现模块，不导入其他方案。沿用 v9 的窗口、字段和监督定义，采样及存储精度按当前配置；低秩瓶颈、LayerNorm、Dropout 和分解汇总属于本主干方案。`parent_solution_id=v9`，父 run 为空。

后续新建独立运行可使用以下入口；已完成的冻结运行不能重训覆盖：

```powershell
.venv-training/Scripts/python.exe -u solutions/v201/v201_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v201-global-fp16-new-s42 --evaluation-cache-mode stream --training-confirmed
```

原始行情入口为 [v201_predict.py](v201_predict.py)，与训练共用 [WindowBlock](v201_data.py)。独立预处理使用 `v201_prepare.py`、`v201_apply.py`；协议、恢复、评估和文本附件约定见[开发约定](../../docs/development.md)。

## 冻结结果与平台材料

依据 2026-10-20（UTC+08:00）的实际产物，seed=42，第 5 轮按验证 IC 选中，第 9 轮动态早停。训练从头使用全局随机样本与 FP16 输入；验证执行路径升级和资源暂停保留原模型、配置、优化器、随机状态及早停计数。累计逐轮耗时约 4.59 小时，包含验证；最后一次恢复会话的 1.01 小时不是全部训练成本。

| 参照角色／方案 | 验证 IC | 验证 Rank IC | 测试 IC | 测试 Rank IC |
| --- | ---: | ---: | ---: | ---: |
| v201 | 0.077352 | 0.078214 | 0.081427 | 0.080190 |
| 此前本地验证最优 v7e | 0.076757 | 0.072320 | 0.079773 | 0.074215 |
| 轻量非卷积对照 v101 | 0.076741 | 0.079479 | 0.082303 | 0.083960 |
| 代码父版本 v9 | 0.067835 | 0.067583 | 0.077534 | 0.075617 |

相对 v7e，验证／测试 IC 分别提高 0.000595／0.001654，两个区间各有 4／6 个月改善；全局及最低截面覆盖均为 100%，IC 未定义截面为零，因此属于单种子本地改善候选。测试 IC IR 从 0.720469 降至 0.681233，9 月和 10 月 IC 退化，不能称为全面改善。v9 是代码父版本对照，没有继承其权重。

[换手与持续性诊断](../../docs/signal-diagnostics.md)中，v201 相对 v7e 的日内换手代理下降、秩自相关提高，为平台测试提供补充依据；该代理不等同于平台 BARRA 后的官方换手。已有最高公榜提交按文件名对应 v7e，但尚缺可信本地 run／权重绑定。

冻结比较见 [v201-mixer.json](../../experiments/v201-mixer.json)。平台材料在 [submissions/v201-global-fp16-s42](../../submissions/v201-global-fp16-s42)，只上传其中的 `v201-submission.ipynb` 和 `v201-model.json`。权重为完整精度文本，训练及推理使用冻结源码；独立入口验证的环境、日期范围、误差和状态以[提交清单](../../experiments/v201-global-fp16-s42-submission.json)为准。平台上传及官方评分另行记录。
