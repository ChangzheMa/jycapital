# 仓库约定与工具

定位实现、训练或整理材料时阅读。目录、协议和工具的统一说明见 [开发文档](../../../../docs/development.md)。

## 目录与数据

模型及其数据处理位于 `solutions/<version>/`，评估位于 `evaluation/`；`experiments/` 保存摘要和索引，`submissions/<run_id>/` 保存冻结平台材料。行情在 `data/`，详细运行在 `outputs/`，本地 `dai.py`、凭据和环境均被 Git 忽略。

本地研究训练 2020—2023 年，验证 2024 年上半年，测试下半年；UTC+08:00、区间左闭右开。标签直接使用官方 `vwap_return`，精确公式及官方评测边界见 [未决事项](../../../../docs/bigquant-2026-end-to-end/open-questions.md)。

输入只能使用预测时刻已知的授权原始量价序列；必要清洗、归一化和切窗遵循比赛规则。权重由验证主指标选择，冻结后评估测试，不覆盖旧运行或重选旧模型权重。

## 工具与环境

- `snapshot_sources(solution_id=...)` 捕获实际方案及依赖，保留生效配置、数据指纹、Git 来源和未提交状态。
- `compare.py` 检查协议、评估器及标签／股票池指纹；`assess.py` 选择本地验证最优和父版本，保存逐月差值、覆盖与晋级结论。
- `local-v1` 使用 Rank IC；`local-v2-ic` 使用 IC。显式 `--allow-selection-metric-change` 仅允许协议标识和选模指标差异。
- 导出与验证工具按版本使用：v1 为 `export.py`／`verify-submission.py`，v2／v4 使用对应版本后缀。检查数值核心、导入路径和复现依赖，不混用不同版本的权重与代码。
- 查询和训练环境分别为 `.venv`、`.venv-training`；先核实环境及设备，实际依赖写入运行记录。训练恢复规则见各版 README。

## 提交材料

默认导出改善候选；用户明确要求的平台测试保留本地判定与用途。材料位于独立 `submissions/<run_id>/`，不生成 ZIP；重新打包保留 Git 历史或本地备份。

唯一推理 Notebook 提供 `main(datasources, start_date, end_date)`，返回 `date、instrument、factor`，保留完整分钟，因子有限、键唯一。行情表通过 `datasources` 读取，平台使用真实 DAI；调试单元格使用主程序保护，提交时不保存输出。

附件为 UTF-8 文本，保留完整权重、预处理状态、配置、依赖和训练复现代码。JSON 可分片，不强求两个文件，大小遵循 [技能约定](../SKILL.md#已确定的工作约定)。逐张量及逐预测核验完整精度，不提交二进制权重或在推理时生成中间权重附件。

验证提交目录独立加载、时间边界、完整输出、推理后端及源码恢复。清单路径从提交索引的 `manifest_path` 读取；更新附件时同步更新文件哈希和索引。`.gitattributes` 保持提交文件字节，校验从 Git 取出的附件。

## Git 与结果边界

默认远程为 `git@github.com:ChangzheMa/jycapital.git`，主分支 `master`。按用户授权提交源码、配置、结论及平台材料；不纳入行情、预测、凭据、环境和过程性记录。

平台可运行、正式提交状态和官方分数分别记录来源及日期。平台上传由用户执行，查询成绩不自动训练或发布。
