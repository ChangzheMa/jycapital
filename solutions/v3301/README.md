# v3301：共享冻结编码器与八个独立分时预测器

[研究判断](../../docs/clock-specialization-research-20261025.md) · [机器可读设计](design.json) · [独立读出原型](v3301_readout.py)

用户于2026-10-25要求补齐代码，正式训练仍由训练session控制。实现已完成并发布ready；本session只做CPU合成验证和父权重只读核对，没有启动正式训练或构建真实特征缓存。持续研究和定时检查保持暂停。

## 固定设计

结构及初始化父版为新快照最优`v3201-field-grid-s42`，第4轮EMA，权重SHA256为`a30086445b8e2b88e392c03a0bb208e4e0b2e2e8164d8096e00263b697363cff`。只复用这个已完成模型，不重跑父训练。

冻结三频编码器、价格历史读取、三频融合及384维共享层。收益专家和路由器复制为八套独立参数，对应09:30、10:00、10:30、11:00、11:30、13:30、14:00、14:30。初始预测应与父版一致，每套后续仅由自身时点的标签更新。

参数量：冻结共享部分2,129,356；每时点223,052；八时点可训练参数1,784,416；最终模型共 **3,913,772**。只执行当前时点的读出，历史编码保持股票日共享fast。参数JSON可能需要分片，每片仍不得超过50,000,000字节；实际文件数量在正式导出时验证。

## 数据与训练计划

- 直接使用新快照提供的`vwap_return`及原标准化／排名转换，不重算公式，不使用exposure。
- 特征为父模型收益读出之前的384维FP32表示，不混入标签或预测答案。按冻结父权重、输入来源、归一化和样本键建立独立缓存身份，预计三划分约13.87GiB，键及目标另计。
- 一个正式候选运行内串行处理八个时点，每个时点按日期混合采样512条，AdamW学习率0.0001、权重衰减0.0001、梯度裁剪1。沿用主Huber、排名辅助、专家和路由目标；专家分配只在该时点内进行。
- 每个时点使用独立优化器、EMA和早停。EMA以64条有效时点样本为参考单位，步衰减为\(0.9999^{n/64}\)，其中\(n\)为本步有效条数，首轮为warmup；避免批量从64改成512后仍沿用相同每步记忆长度。
- 动态早停仍为最佳轮数×0.6向上取整，最少4、最多12，严格验证IC改善，最高100轮。父版读出作为预先登记的初始候选；没有改善则保留该时点父参数。
- 各时点分别按验证IC选定读出，所有选择冻结后统一评测测试集。汇总记录`selected_epochs_by_clock`及父版保留标志，不伪造一个不存在的共同最佳epoch；分析和导出工具需要显式支持这个结构。
- 主机内存预留2GiB。恢复记录须包含当前时点、缓存身份、游标、优化器、EMA、RNG及已完成时点；不能因某时点中断重新训练已经冻结的其他时点。

## 入口与恢复

| 文件 | 作用 |
| --- | --- |
| [v3301_train.py](v3301_train.py)、[v3301_run.py](v3301_run.py) | 参数校验、父权重与新数据绑定、特征准备、八时点串行运行和冻结后测试 |
| [v3301_cache.py](v3301_cache.py) | 仅用冻结父表示的FP32缓存；初始化与分块提取均可恢复，已完成块核对指纹 |
| [v3301_fit.py](v3301_fit.py) | 独立AdamW、按样本量EMA、动态早停及每时点验证IC选择 |
| [v3301_model.py](v3301_model.py) | 共享冻结编码器和八个独立读出的fast／参考推理 |
| [v3301_verify.py](v3301_verify.py) | 完成后重放八个时点的选模，并复核组合权重和完整双集指标 |

训练只在显式传入`--training-confirmed`后执行。以下命令供**训练session**使用，本session未执行：

```powershell
.venv-training/Scripts/python.exe scripts/training-data-paths.py --solution-id v3301
.venv-training/Scripts/python.exe scripts/check-candidate-device.py --solution-id v3301 --device cuda
.venv-training/Scripts/python.exe -u solutions/v3301/v3301_train.py --parent-run outputs/runs/v3201-field-grid-s42 --data-root data/redownload/jycapital-raw-20261024 --training-run outputs/preprocessing/data-20261024/train --holdout-run outputs/preprocessing/data-20261024/holdout --run-dir outputs/runs/v3301-clock-specialists-s42 --evaluation-cache-mode stream --device cuda --training-confirmed
```

采用路径解析器返回的当前绑定，不传旧全局`--resident-cache`。本版要求父模型与当前输入清单完全一致，不能在冻结父版仍绑定旧来源时悄悄换标签或行情。

恢复同一运行追加`--resume`。特征目录为预处理根目录下的`clock-feature-cache/<identity>`，未完成目录使用`.building`后缀。键、目标与已完成特征块改变时拒绝恢复；不会跳过校验或从零覆盖已有完整缓存。留出监督目标始终从原始标签键生成，不使用底层评估缓存的零占位向量。

活动时点的最佳读出、原始参数、优化器、EMA、RNG和游标一起原子保存到`last.pt`；各时点完成后冻结到`clocks/<clock_id>/best.pt`。因此在验证选模写入中断、时点切换或最终测试中断时，可以恢复正确阶段，不重训已完成时点。EMA用线性插值更新，相同参数不会仅因舍入漂移改变。

**调度注意：**本版没有单一最佳epoch，`selection.best_epoch`为`null`，真实选择在`selected_epochs_by_clock`及`clock_selections`中；`epochs_completed`单位为`clock_epoch`。训练session应使用本版完成检查器，不按单一epoch重放。通用`assess.py`已经适配结果文件名和空单一epoch字段。

```powershell
.venv-training/Scripts/python.exe solutions/v3301/v3301_verify.py --run-dir outputs/runs/v3301-clock-specialists-s42 --output outputs/run-control/v3301-clock-specialists-s42/artifact-check-specialists.json
.venv-training/Scripts/python.exe scripts/assess.py outputs/runs/v3301-clock-specialists-s42 --record --allow-selection-metric-change
```

## 实现验证与导出

13项CPU模型检查通过，覆盖父模型参数和特征初始化、分时目标与原目标等价、分时梯度隔离、缓存初始化／分块恢复及损坏拒绝、活动时点／时点边界／验证写入恢复、父读出回退、完整合成Parquet入口、冻结测试恢复、复合选模重放、文本分片恢复，以及动态股票池和部分请求的fast一致性。比较工具另有13项回归检查通过，包括复合选择的CSV／JSON记录。完整范围和源码指纹见[实现检查报告](../../experiments/readiness/v3301.json)。

真实v3201冻结参数、来源清单和原始缓存键已只读核对，见[兼容性记录](../../experiments/v3301-input-cache-compatibility.json)；没有加载真实行情数组或执行真实训练。默认宽度CPU预检通过；默认GPU批量仍由训练session在合适时机检查。实现通过不代表收益IC改善。

正式晋级后，使用[导出入口](../../scripts/export-v3301.py)和[简化核验](../../scripts/verify-v3301-submission.py)。平台材料为一个Notebook、主JSON及数值参数分片，每份均限制在50,000,000字节以内；不带特征缓存、标签或离线答案。运行时仅加载这些文本附件。显式训练复现入口会先恢复父源码目录并重训父模型，再执行八时点流程；正常导入和平台推理不启动训练。
