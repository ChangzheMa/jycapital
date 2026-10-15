# 方案版本、运行与评估

[文档目录](README.md) · [实验技能](../.agents/skills/jycapital-experiment/SKILL.md) · [提交材料](../submissions/README.md)

## 运行和提交追溯

`solutions/<version>/` 保存模型、预处理、训练及推理；公共评估位于 `evaluation/`，运行记录位于 `common/`，比较和导出工具位于 `scripts/`。

精确实验版本由 run_id、生效配置、协议、数据指纹和实际源码快照共同确定。存在未提交源码时如实记录 `git_dirty=true`，不能只用 HEAD 代替实际执行代码。每次独立实验使用新的运行目录，保留原权重和结果。

| 位置 | 保存内容 |
| --- | --- |
| `outputs/runs/<run_id>/`，被 Git 忽略 | 环境、源码快照、输入指纹、checkpoint、逐轮记录、曲线和完整预测 |
| `experiments/` | 结论摘要、逐月比较、运行／提交索引和提交验证清单 |
| `submissions/<run_id>/` | 随 Git 分发的冻结推理及训练复现材料 |

常用运行文件为 `manifest.json`、`config.json`、`protocol.json`、`data-manifest.json`、`source/`、`selection.json`、`best.pt`、`last.pt` 和 `evaluation/`。各版的恢复入口及独有产物见方案说明。

## 本地研究协议

以下为 UTC+08:00、左闭右开区间，属于本地研究约定，不是官方公榜／私榜边界。

| 划分 | 开始（含） | 结束（不含） | 用途 |
| --- | --- | --- | --- |
| 训练 | 2020-01-01 | 2024-01-01 | 拟合模型 |
| 验证 | 2024-01-01 | 2024-07-01 | 选择权重与早停 |
| 测试 | 2024-07-01 | 2025-01-01 | 权重冻结后的评估 |

输入可使用前一划分已经发生的历史，不读取未来 Bar。标签窗口不得跨划分边界或隔夜。测试不参与 checkpoint 选择；持续查看测试结果会削弱其独立性。

[local-v1](../evaluation/protocols/local-v1.json) 按验证 Rank IC 选模；[local-v2-ic](../evaluation/protocols/local-v2-ic.json) 按验证 IC 选模。比较两者时显式启用 `--allow-selection-metric-change`，只允许协议标识和选择指标差异。

## 统一指标

[评估器](../evaluation/baseline_evaluation.py)使用原始股票池的完整 `(date, instrument)` 键及官方 `vwap_return`。标签仅按有限性和边界规则过滤，不裁剪评估标签。

- IC 为逐截面 Pearson 相关系数；Rank IC 为双方取平均秩后的 Pearson 相关系数。
- 每个截面等权，至少 30 个有效样本。常数因子或标签的相关系数为空，单独报告，不能记为 0。
- 两种 IR 均为均值除以样本标准差（`ddof=1`），不年化。
- MSE／MAE 按原始收益尺度计算，并保留零预测 MSE 参照；纯 IC 分数未校准收益幅度。
- 覆盖率以原股票池为分母，报告全局覆盖、最低截面覆盖及缺失超过 40% 的截面数。

本地指标不包含平台 BARRA、交易成本或压力场景评分。完整口径来源为各运行协议和指标文件；平台说明见 [官方来源](bigquant-2026-end-to-end/sources.md)。

## 比较与导出工具

| 用途 | 入口 |
| --- | --- |
| 指纹核对与指标表 | [compare.py](../scripts/compare.py)，`--split validation` 或 `--split test` |
| 历史参照、逐月差值及晋级记录 | [assess.py](../scripts/assess.py)，`--record` 写结果及运行索引 |
| v1 导出／验证 | [export.py](../scripts/export.py)、[verify-submission.py](../scripts/verify-submission.py) |
| v2 导出／验证 | [export-v2.py](../scripts/export-v2.py)、[verify-v2-submission.py](../scripts/verify-v2-submission.py) |
| v4 导出／验证 | [export-v4.py](../scripts/export-v4.py)、[verify-v4-submission.py](../scripts/verify-v4-submission.py) |
| v5 导出／验证 | [export-v5.py](../scripts/export-v5.py)、[verify-v5-submission.py](../scripts/verify-v5-submission.py) |

晋级按当前协议的验证主指标，要求覆盖不降低；v4／v5 还要求未定义 IC 截面不增加。用户要求导出未晋级版本时，使用 `--platform-test-reason` 保留原判定与导出理由。平台最优须有可信分数及运行对应关系。

导出取冻结源码、配置和权重，默认拒绝覆盖已有目录。附件为 UTF-8 文本，权重保留完整精度；JSON 可拆分，每个模型参数文件不超过 50,000,000 字节。v4 使用 [紧凑数值数组布局](../common/text_json.py)，不截断浮点值。所有分片须完整加载并通过推理一致性核验。

v5 在该布局中显式标记 FP32 权重数组，以九位有效数字保存，包含负零处理；普通数值和归一化统计量沿用原精度。导出前经由实际 JSON 文本恢复并逐张量校验 dtype、形状和字节；验证程序进一步检查隔离推理、训练源码恢复及附件大小。多个验证区间保存在提交清单的 `validations` 中。

```powershell
.venv-training/Scripts/python.exe scripts/export-v5.py --run-dir outputs/runs/v5-20261015-s42
.venv-training/Scripts/python.exe scripts/verify-v5-submission.py --run-dir outputs/runs/v5-20261015-s42 --package submissions/v5-20261015-s42 --record
```

[提交索引](../experiments/submissions.csv) 的 `manifest_path` 指向验证清单。`.gitattributes` 保持提交文件原始字节；校验工作区及 Git 中的附件哈希。行情、预测答案、凭据、本地 DAI 和训练环境不进入提交目录。平台上传由用户执行。

## 环境与后台监控

本地查询使用 `.venv`，训练使用 `.venv-training`；依赖和实际环境以版本清单与运行记录为准。

[run_notifications.py](../common/run_notifications.py) 接入已有进程，参数为 `--codex-binary`、`--thread-id`、`--run-dir`、`--pid`。普通进度留在本地，每小时、冻结、完成或异常时通知原任务。

`--probe-dir` 验证空闲唤醒和执行权限：接收回合实际执行命令并写确认，事件／任务标识匹配且退出码为 0 才通过；探测不启动训练。Windows 后台隐藏启动，日志保存在忽略目录。详细要求见 [后台监控约定](../.agents/skills/jycapital-experiment/references/monitoring.md)。
