# 方案版本、运行与评估

[文档目录](README.md) · [实验技能](../.agents/skills/jycapital-experiment/SKILL.md) · [平台材料](../submissions/README.md)

## 运行和提交追溯

`solutions/<version>/` 保存模型及数据处理，`evaluation/` 提供公共评估，`common/` 提供运行工具，`scripts/` 提供比较和导出入口。

| 位置 | 内容 |
| --- | --- |
| `outputs/runs/<run_id>/`，被 Git 忽略 | 生效配置、协议、数据指纹、实际源码快照、环境、checkpoint、训练记录和完整预测 |
| `experiments/` | 冻结结论、运行／提交索引、比较及验证清单 |
| `submissions/<run_id>/` | 随 Git 分发的冻结平台材料 |

实验由 run_id、配置、协议、数据指纹和实际源码共同确定。有未提交源码时记录 `git_dirty=true`；仅保存 HEAD 不足以复现。独立实验使用新目录，保留历史权重与结果。

## 本地研究协议

以下为 UTC+08:00 的左闭右开区间，不是官方公榜／私榜边界。

| 划分 | 开始（含） | 结束（不含） | 用途 |
| --- | --- | --- | --- |
| 训练 | 2020-01-01 | 2024-01-01 | 拟合模型 |
| 验证 | 2024-01-01 | 2024-07-01 | 选择权重与早停 |
| 测试 | 2024-07-01 | 2025-01-01 | 冻结后的评估 |

输入可使用前一划分已发生的历史，不读取未来 Bar；标签窗口不得跨划分边界或隔夜。每个完成训练的版本都评估验证和测试；测试不选择 checkpoint，持续研究已观察的测试期不称为最终盲测。单种子改善不构成统计显著性证据。

[local-v1](../evaluation/protocols/local-v1.json) 按验证 Rank IC 选模；[local-v2-ic](../evaluation/protocols/local-v2-ic.json) 按验证 IC 选模。比较时的 `--allow-selection-metric-change` 仅允许协议标识和选择指标差异，历史权重不重选。

## 后续训练的统一早停策略

v7 及后续新版本使用 [early_stopping.py](../common/early_stopping.py)。最佳 epoch 从 1 开始计数，耐心轮数为：

\[
x=\min\left(12,\max\left(4,\left\lceil0.6e_{\text{best}}\right\rceil\right)\right)
\]

合格验证主指标严格创新高时，更新最佳权重与轮次并清零无提升计数，不设 `min_delta` 门槛。平分、下降或不合格轮次计为无提升；连续达到耐心即停止，`epochs` 仍是总上限。尚无合格结果时耐心为 4；始终无合格权重不能记为完成。

逐轮与选模记录保存策略、实际耐心和无提升计数；断点恢复沿用最佳轮次及计数。源码快照与独立复现材料须包含公共早停模块。历史冻结版本保留原规则。

## 统一指标

[评估器](../evaluation/baseline_evaluation.py)使用原股票池的完整 `(date, instrument)` 键及官方 `vwap_return`；标签仅按有限性和边界过滤，不裁剪评估标签。

按用户 2026-10-22 的约定，已提供的 `vwap_return` 不另行重算或反推公式，训练时可在其基础上标准化、排名等加工。已删除的 `exposure` 表不用于训练、评估、推理或诊断，具体约束见[持续研究约定](continuous-research.md#效果效率与平台额度)。

- IC 为逐截面 Pearson 相关系数；Rank IC 为双方取平均秩后的 Pearson 相关系数。
- 截面等权，至少 30 个有效样本；常数因子或标签的相关系数为空，单独报告，不记为 0。
- IR 为有效截面均值除以样本标准差（`ddof=1`），不年化。
- MSE／MAE 按原始收益尺度计算，保留零预测 MSE 参照；纯 IC 分数未校准收益幅度。
- 覆盖率以原股票池为分母，报告全局覆盖、最低截面覆盖及缺失超过 40% 的截面数。

本地指标不包含平台 BARRA、交易成本和压力场景评分。损失定义见 [Pearson IC](v4-ic-loss-plan.md)，官方评分口径见 [比赛规则](bigquant-2026-end-to-end/rules.md#6-评分)。

## 训练入口

本地查询使用 `.venv`，训练使用 `.venv-training`；依赖见各版本的 `requirements*.txt`，实际环境保存在运行记录中。预处理入口为对应版本的 `prepare`／`apply` 脚本。v2—v7 的现有输入可复用冻结目录 `outputs/preprocessing/v2-20261010/` 与 `outputs/preprocessing/v2-holdout-20261011/`；来源见 [训练摘要](../experiments/v2-preprocessing.json)和[验证／测试摘要](../experiments/v2-holdout-preprocessing.json)。

仅在用户要求训练时运行，例如：

```powershell
.venv-training/Scripts/python.exe solutions/v7/v7_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v7-new-s42 --training-confirmed
```

v3 及以后使用 `--training-confirmed`；v2 不接受该参数。中断运行可加 `--resume`，已完成运行不能续跑；源码、配置或数据改变时创建新运行。v1 的不同入口见 [基线说明](baseline-v1.md#使用与提交)。

限定例外：v201／v301 已有全局 run 可通过[验证缓存升级入口](resident-training.md#已有全局-run-的恢复)在原目录恢复。该入口只接受已登记的验证执行路径补丁，保留原源码及 checkpoint 状态，继续检查配置、协议和其他源码；不用于改变模型或训练方案。

## 比较与导出工具

| 用途 | 入口 |
| --- | --- |
| 指纹核对与指标表 | [compare.py](../scripts/compare.py)，`--split validation` 或 `--split test` |
| 三类参照、逐月差值及晋级记录 | [assess.py](../scripts/assess.py)，`--record` 写结果及运行索引 |
| v1 导出／验证 | [export.py](../scripts/export.py)、[verify-submission.py](../scripts/verify-submission.py) |
| v2 导出／验证 | [export-v2.py](../scripts/export-v2.py)、[verify-v2-submission.py](../scripts/verify-v2-submission.py) |
| v4 导出／验证 | [export-v4.py](../scripts/export-v4.py)、[verify-v4-submission.py](../scripts/verify-v4-submission.py) |
| v5 导出／验证 | [export-v5.py](../scripts/export-v5.py)、[verify-v5-submission.py](../scripts/verify-v5-submission.py) |
| v6 导出／验证 | [export-v6.py](../scripts/export-v6.py)、[verify-v6-submission.py](../scripts/verify-v6-submission.py) |
| v7 导出／验证 | [export-v7.py](../scripts/export-v7.py)、[verify-v7-submission.py](../scripts/verify-v7-submission.py) |
| v7e 导出／验证 | [export-v7e.py](../scripts/export-v7e.py)、[verify-v7e-submission.py](../scripts/verify-v7e-submission.py) |
| v101 导出／验证 | [export-v101.py](../scripts/export-v101.py)、[verify-v101-submission.py](../scripts/verify-v101-submission.py) |
| v201 导出／验证 | [export-v201.py](../scripts/export-v201.py)、[verify-v201-submission.py](../scripts/verify-v201-submission.py) |
| v301 导出／验证 | [export-v301.py](../scripts/export-v301.py)、[verify-v301-submission.py](../scripts/verify-v301-submission.py)；GRU 数值核验说明见[版本文档](../solutions/v301/README.md#冻结结果与平台材料) |
| v303 导出／验证 | [export-v303.py](../scripts/export-v303.py)、[verify-v303-submission.py](../scripts/verify-v303-submission.py)；默认导出 fast 共享编码材料，详见[加速说明](v303-inference-fast.md)；验证 CLI 默认 `--device cpu` |
| v303 推理测量 | [profile-v303-submission.py](../scripts/profile-v303-submission.py)，CPU 上测量独立提交包的完整请求及各阶段耗时 |
| v501 导出／诊断 | [export-v501.py](../scripts/export-v501.py)默认 fast 并核对 EMA 来源；[verify-v501-submission.py](../scripts/verify-v501-submission.py)、[profile-v501-submission.py](../scripts/profile-v501-submission.py)保留为需要时的诊断工具，后续同构转换按下述简化约定 |
| v901 导出／简化核验 | [export-v901.py](../scripts/export-v901.py)从冻结快照导出当天多频 fast 材料；[verify-v901-submission.py](../scripts/verify-v901-submission.py)检查权重精确恢复、隔离合成推理及复现依赖，不重复长行情或性能基准 |
| v1001 导出／简化核验 | [export-v1001.py](../scripts/export-v1001.py)、[verify-v1001-submission.py](../scripts/verify-v1001-submission.py)；保留训练期时间权重和分支辅助目标的复现源码，平台仅使用共享编码与行情路由 |
| v1002 导出／简化核验 | [export-v1002.py](../scripts/export-v1002.py)、[verify-v1002-submission.py](../scripts/verify-v1002-submission.py)；三频当天输入、共享历史状态与多分支收益头，默认 fast |
| v1101 用户指定导出／简化核验 | [export-v1101.py](../scripts/export-v1101.py)、[verify-v1101-submission.py](../scripts/verify-v1101-submission.py)；使用 `--platform-test-reason` 记录明确请求，保留原未晋级判定，默认 fast |
| 信号换手与持续性代理 | [diagnose-signals.py](../scripts/diagnose-signals.py)，定义及局限见[诊断说明](signal-diagnostics.md) |

晋级、历史参照及平台分回填统一按[比较规则](../.agents/skills/jycapital-experiment/references/comparison.md)。用户要求导出未晋级版本时，使用 `--platform-test-reason` 保留原判定和导出理由。

导出取冻结源码、配置、统计量和所选权重，默认拒绝覆盖已有目录。全部附件为 UTF-8 文本，模型参数可分片，每个参数文件不超过 **50,000,000 字节**。v4 使用[紧凑数值数组布局](../common/text_json.py)；v5—v7、v7e、v101、v201、v301 和 v303 的 FP32 权重用九位有效数字保存，并校验 dtype、形状和字节精确恢复，其他数值保持原精度。

依据用户 2026-10-21 的要求，后续所有平台材料统一使用 **fast 推理路径**：按股票日复用切窗与可共享历史编码，保留完整截面归一化、原输出键和因果性。新模型应在设计时兼顾该路径，按实际网络适配，不能直接将单向 GRU 的前缀状态复用套用到非因果注意力等其他结构。历史导出工具及原包保留追溯，新平台导出须先满足此约定。v303 默认 fast，`--reference-inference` 仅供本地回归对照。

用户进一步要求简化 fast 转换检查。相同计算结构和输入语义的后续模型，优先复用已有路径的验证证据，检查权重恢复、输入／归一化、因果前缀、读出和请求键的逻辑等价，并从重复编码与取窗次数判断预期效率；仅有必要时补少量针对性测试，不逐版重复长区间推理及性能基准。新结构存在未解决的实现疑点时，再针对该疑点扩展检查。记录明确区分逻辑审查与实际运行范围，未实测不填写具体加速倍数；这不取消正式模型的本地验证和测试。

附件加载、权重精度、训练依赖闭包和输出契约仍需检查。Notebook 导入不查询行情，推理不读取标签、不生成额外权重附件；运行时不添加哈希／附件匹配检查，哈希仅用于本地追溯。

[提交索引](../experiments/submissions.csv) 的 `manifest_path` 指向清单，`.gitattributes` 保持附件原始字节；检查从 Git 取出的字节与哈希。上传方式和平台状态仅在[平台材料索引](../submissions/README.md)维护，平台上传由用户执行。

## 环境与后台监控

[run_notifications.py](../common/run_notifications.py) 通过 `--codex-binary`、`--thread-id`、`--run-dir`、`--pid` 接入已有进程。普通进度留在本地，每小时、用户询问、完成或异常时检查。后台启动、停滞判断及空闲唤醒验收按[监控约定](../.agents/skills/jycapital-experiment/references/monitoring.md)执行。
