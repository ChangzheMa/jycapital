# 仓库贡献指南

## 项目语境与技能

本仓库及子目录默认属于 BigQuant 2026 进益资本杯端到端项目，无需用户重复项目名。

实验设计、模型或数据处理迭代、训练、评估、版本比较、成绩回填和模型提交材料整理，必须读取并应用仓库内的 [jycapital-experiment](.agents/skills/jycapital-experiment/SKILL.md)。只执行当前请求涉及的阶段；加载技能不自动授权额外训练、Git 推送或平台提交。普通文档、无关代码修改及纯 Git 操作按实际任务处理。

查询平台成绩时读取 [jycapital-scores](.agents/skills/jycapital-scores/SKILL.md)，沿用已接入接口和认证配置；只查看则报告结果，回填须确认 submission、run 和权重对应关系。技能以仓库文件为准，优先于全局副本；用户切换项目时遵从指定。

## 目录与入口

先读 [文档目录](docs/README.md)。通用开发、训练、评估及导出规则集中在 [development.md](docs/development.md)，官方材料在 `docs/bigquant-2026-end-to-end/`，新增文档同步更新索引。

- `solutions/`：版本化模型、数据处理、训练与推理。
- `evaluation/`、`common/`、`scripts/`：公共评估、运行及工具。
- `experiments/`：冻结结论、运行／提交索引和验证清单。
- `submissions/<run_id>/`：随 Git 分发的冻结平台材料；上传与状态见 [提交索引](submissions/README.md)。
- `examples/`：Notebook 示例；本地环境见[使用说明](docs/bigquant-2026-end-to-end/local-download.md)。
- `data/`、`outputs/` 及根目录 `dai.py`：被忽略的行情、运行产物和本地兼容层，不分发。

## 编写与验证

说明使用中文，官方标识符保持原样。Markdown 使用含义明确的标题、相对链接和围栏代码块；文档文件名采用小写连字符。Markdown／JSON 为 UTF-8，JSON 两空格缩进；CSV 保留 UTF-8 BOM 和列顺序。Python 使用四空格缩进及 `snake_case`。

数学公式行内用 `\(...\)`，块级用 `\[...\]`；分段条件用正文说明，避免 `cases` 环境。

文档只保留有效结论、必要使用说明、来源和未决事项，不保留排错过程、堆栈、执行时间线、临时检查次数或中间记录。重复规则链接到统一说明；历史结果注明版本，不能冒充当前行为。

修改文档后检查相对链接、代码围栏及 JSON 解析。字段字典的每个 `(table, field)` 唯一且 CSV／JSON 一致，基线为 **104 条**；只有官方来源变化才更新字段及计数。修改 Notebook 后检查输出 `date、instrument、factor` 的类型、有限性及键唯一性；本地运行、平台运行和正式模型验证分别标注。

## 来源与安全

记录官方来源、获取日期及时区，区分正文、接口配置和整理者判断；冲突保留在 [open-questions.md](docs/bigquant-2026-end-to-end/open-questions.md)。用户确认的结果注明依据，不扩展为未经验证的能力。

禁止提交密码、Cookie、令牌、账号资料、受限行情、本地兼容层和离线预测答案。提交材料的文本格式、精度与大小遵循开发约定，`.gitattributes` 保持附件原始字节以便校验。

## Git 与拉取请求

主分支为 `master`，`origin` 为 `git@github.com:ChangzheMa/jycapital.git`。提交只保留有长期意义的修改，说明简短明确，如 `docs: 明确 VWAP 标签定义`。PR 说明目的、来源、验证和未决事项；仅解释界面行为时附截图。
