# 方案版本、运行与评估

[返回目录](README.md) · [v1 基线](baseline-v1.md)

## 代码组织

`solutions/v1/` 保存第一个完整方案，包含模型、清洗、归一化、切窗、训练和推理。后续方案可新增 `solutions/v2/`，自由修改内部实现。公共评估位于 `evaluation/`，运行记录工具位于 `common/`，导出和比较命令位于 `scripts/`。

一个方案可以运行多次。精确版本由方案标识、Git commit、实际源码快照哈希、完整配置和运行标识共同确定。未提交的新增代码也进入源码快照；`git_dirty=true` 如实记录，不能仅凭 Git commit 恢复本次运行。

运行目录位于被 Git 忽略的 `outputs/runs/`，保存详细训练记录、预测、缓存和训练 checkpoint。正式提交材料导出到根目录 `submissions/<run_id>/`，其中的脚本、Notebook、配置和冻结推理权重一并进入 Git。行情、凭据和本地 DAI 不进入 Git；完整运行记录应在授权存储范围内另行备份。

## 通用时间划分

协议文件：[local-v1.json](../evaluation/protocols/local-v1.json)。以下均为 UTC+08:00，内部采用左闭右开区间。

| 划分 | 开始（含） | 结束（不含） | 用途 |
| --- | --- | --- | --- |
| 训练 | 2020-01-01 | 2024-01-01 | 拟合模型 |
| 验证 | 2024-01-01 | 2024-07-01 | 选择 checkpoint 和停止训练 |
| 测试 | 2024-07-01 | 2025-01-01 | 权重冻结后的独立评估 |

用户授权依据本地现有数据决定划分；本划分是项目研究约定，不是官方公榜/私榜边界。输入窗口允许使用预测时点之前的历史，包括前一划分的历史行情；不能使用预测时点之后的行情。训练标签的未来窗口不得跨训练边界或隔夜。

后续模型沿用该划分。测试集不用于选超参数、早停或选择 checkpoint；长期反复查看测试结果并据此开发，也会削弱其独立性，届时应另设新的最终保留区间。

## 统一评估口径

评估器：[baseline_evaluation.py](../evaluation/baseline_evaluation.py)。

- 样本集合采用 `cpt_jyc_2026_instruments` 的完整 `(date, instrument)` 键，不能仅以模型有结果的样本定义覆盖率。
- 标签直接使用 `cpt_jyc_2026_vwap.vwap_return`。只剔除空值、非有限值及跨日期/划分边界的标签；不猜测生成公式，不裁剪评估标签。
- IC 是每个分钟截面的 Pearson 相关系数；Rank IC 是双方各自在截面内取平均秩后的 Pearson 相关系数。
- 每个截面等权汇总。最少 30 个有效样本；常数因子或常数标签的相关系数为空，单独报告数量，不记成 0。
- IC_IR、Rank IC_IR 是有效截面均值除以样本标准差（`ddof=1`），不年化。
- MSE/MAE 在有限标签且有预测的样本上按原始收益尺度计算，同时给出零预测 MSE 作为参照。
- 覆盖率以原始股票池为分母，报告全局覆盖率、最低截面覆盖率和缺失超过 40% 的截面数量。
- 保留逐截面指标、逐月汇总及预测文件，便于定位整体指标变化。

本地指标不包含平台的 BARRA 风格剔除、交易成本、多空夏普或压力场景算法，因此不构成官方成绩。平台说明依据：[新手入门](https://bigquant.com/wiki/doc/oSGtQiGtq0)，获取日期 2026-10-10（UTC+08:00）。

比较运行：

```powershell
.venv/Scripts/python.exe scripts/compare.py outputs/runs/v1-20261010-s42 --split validation
```

命令支持多个运行目录，并检查协议、评估实现及标签/股票池文件指纹是否一致。模型输入和数据处理方式可以不同。若预测覆盖不同，还应结合覆盖率判断，不能只比较模型各自剩余样本上的相关系数。

[assess.py](../scripts/assess.py) 在运行完成后，从历史索引选出可比的本地最优，合并父版本角色，保存验证／测试差值、逐月对比和逐截面覆盖检查；`--record` 同步更新实验摘要和运行索引。默认要求验证主指标严格提高且覆盖不下降，不因测试分单独提高而晋级。已有平台分时须先核对评分口径与对应模型，不能将“平台可运行”视为平台最优成绩。

## 运行和提交追溯

每次运行保留：

| 文件或目录 | 内容 |
| --- | --- |
| `manifest.json` | 运行状态、源码来源、配置/协议哈希、所选模型 |
| `source/` | 本次实际执行的源码、配置与协议快照 |
| `local-only/dai.py` | 对应本地兼容层，不进入提交包 |
| `data-manifest.json` | 实际输入 Parquet 文件大小和 SHA-256 |
| `config.json`、`protocol.json`、`datasources.json` | 完整实验配置、协议和实际表映射 |
| `environment.json`、`requirements-lock.txt` | Python、包版本、设备及确定性设置 |
| `events.jsonl`、`history.csv` | 数据处理记录和每轮损失/验证指标 |
| `learning-curves.png` | 训练损失与验证指标曲线 |
| `best.pt`、`last.pt` | 最佳权重、最后一轮模型及优化器/RNG 状态 |
| `v1-model.json` | 用于 NumPy 推理的完整冻结权重 |
| `selection.json` | 仅按验证集选权重的依据和模型哈希 |
| `evaluation/` | 各划分指标、逐月/逐截面结果与预测 |
| `inference-parity.json` | PyTorch 与 NumPy 推理一致性结果 |
| `submission-validation.json` | 导出包的本地入口、键、数值及读表验证 |

`last.pt` 保存检查与后续开发所需状态。v1 不提供断点续训入口；v2 每完成一个训练月份保存状态，可用原命令加 `--resume` 续跑，要求源码、配置和输入指纹一致。重新开始独立实验应使用新的运行目录。跨平台/硬件不承诺逐位一致。

导出器从运行源码快照取文件，不混用当前工作区代码。每版提交尽量少文件，全部采用 UTF-8 文本，合并保存唯一推理 Notebook、训练/推理源码、完整数值权重、配置、协议、随机种子和依赖声明；不包含行情、预测答案或本地 DAI。默认导出到根目录 `submissions/<run_id>/`，不生成压缩包。本地二进制 checkpoint 留在 `outputs/`。

```powershell
.venv/Scripts/python.exe scripts/export.py --run-dir outputs/runs/v1-20261010-s42
.venv/Scripts/python.exe scripts/verify-submission.py --run-dir outputs/runs/v1-20261010-s42 --package submissions/v1-20261010-s42
```

现有目录已冻结，导出器默认拒绝覆盖；用户要求重新打包时，保留 Git 历史或本地备份并重新验证。后续训练使用新的运行标识。推送 `submissions/` 后即可从远程仓库拉取材料并上传平台。`.gitattributes` 对该目录禁用换行转换，保持模型和文件清单中的 SHA-256 在不同操作系统上一致。[提交索引](../experiments/submissions.csv) 记录目录路径、`manifest_path` 及清单哈希。v1 历史清单仍在提交目录内，v2 清单与验证报告合并放在 `experiments/`，无需上传。

提交目录的本地验证与平台运行、正式提交分开记录。平台的附件识别方式和依赖环境仍需实际上传验证，用户负责提交。

v2 使用 [export-v2.py](../scripts/export-v2.py) 和 [verify-v2-submission.py](../scripts/verify-v2-submission.py)，默认仅导出本地改善候选。用户明确要求对未改善版本进行平台测试时，可指定 `--platform-test-reason "用户要求及测试目的"`；导出清单记录用途、理由和原本地判定，不绕过完整性检查。现有 [v2 平台 Notebook](../submissions/v2-20261011-s42/v2-submission.ipynb) 属于这一情形，不能据此将 v2 标为本地最优。

v4 使用 [export-v4.py](../scripts/export-v4.py) 和 [verify-v4-submission.py](../scripts/verify-v4-submission.py)，从运行快照取全部模型与推理源码。其纯 IC 训练依赖 `v4_schema.py`、`v4_loss.py`、`common/protocol_compatibility.py` 和新协议；导出恢复这些依赖，并验证隔离推理、完整精度权重和训练入口导入。现有 [v4 平台 Notebook](../submissions/v4-20261014-s42/v4-submission.ipynb) 为用户明确要求的未晋级版本导出，不改变本地验证 IC 判定。

v4 的 [local-v2-ic](../evaluation/protocols/local-v2-ic.json) 仅改变协议标识和选择指标。与旧运行比较时须显式传入 `--allow-selection-metric-change`；默认检查仍严格要求原协议相同，日期、标签、股票池、聚合规则和评估器的其他差异均拒绝。

v4 参数 JSON 通过 [text_json.py](../common/text_json.py) 保留两空格对象缩进、紧凑写入数值数组；不缩短浮点有效数字。导出和验证均按单文件 50,000,000 字节上限检查，超限不生成附件。文字、完整数值与标准 JSON 解析结果保持一致。

## 后台通知入口

[run_notifications.py](../common/run_notifications.py) 通过本地 Codex daemon 的 WebSocket RPC 将独立监控通知送回指定的已有任务。调用时提供 `--codex-binary`、`--thread-id`、`--run-dir` 和已有训练进程的 `--pid`；不固化本机路径或任务标识。脚本接入 [监控程序](../.agents/skills/jycapital-experiment/scripts/monitor-run.py)，普通进度只更新本地状态，每小时、冻结、完成或异常时发送通知。

使用 `--probe-dir` 代替运行参数可验收空闲唤醒与执行权限：接收回合须实际执行命令并写确认，只有事件／任务标识匹配且命令退出码为 0 才标记通过。探测不启动训练；接口接受消息不等于执行成功。路径和权限设置变化后，应按 [后台监控约定](../.agents/skills/jycapital-experiment/references/monitoring.md) 验收。后台进程在 Windows 下隐藏启动，通知、确认和日志保留在忽略目录中。
