# 仓库贡献指南

## 项目结构与文件组织

本仓库整理 BigQuant 2026 年进益资本杯端到端赛道资料。先阅读 `docs/README.md`，比赛材料位于 `docs/bigquant-2026-end-to-end/`：

- `rules.md`、`data.md`：比赛要求与数据说明。
- `open-questions.md`、`sources.md`：待确认差异与来源记录。
- `competition-config.json`：官方比赛配置的相关字段摘录。
- `data-schemas.json`、`data-dictionary.csv`：结构化字段字典。

示例 Notebook 位于 `examples/`，依赖清单位于 `requirements-local.txt`。根目录 `dai.py` 为仅限本地使用且被 Git 忽略的兼容层，不随仓库分发；实际行情只存于被忽略的 `data/`。相关资料应集中存放，新增文件时同步更新文档索引。

当前 `mock-submission.ipynb` 已在本地和平台正常运行，测试 submission 已通过；有效返回列为 `date`、`instrument`、`factor`。

## 构建、检查与开发命令

目前没有应用服务或构建系统。在仓库根目录使用 PowerShell 和项目 Python 环境检查文件及本地兼容层：

```powershell
rg --files docs
python -m json.tool docs/bigquant-2026-end-to-end/competition-config.json > $null
python -m json.tool docs/bigquant-2026-end-to-end/data-schemas.json > $null
.venv/Scripts/python.exe -m pip install -r requirements-local.txt
```

以上命令列出文档、检查 JSON 语法并安装本地依赖。应检查命令退出码，确认验证成功。本地运行使用 `examples/mock-submission.ipynb`；`dai.py` 不随 Notebook 上传云端。

## 编写风格与命名约定

说明文档使用中文，官方标识符保持原样。使用含义明确的 Markdown 标题、相对文件链接和围栏代码块；文件名采用小写字母，单词间用连字符分隔。Markdown 和 JSON 使用 UTF-8，JSON 使用两空格缩进；CSV 保留 UTF-8 BOM 和既有列顺序。Python 示例使用四空格缩进和 `snake_case` 命名。目前未配置格式化或静态检查工具。

## 验证要求

修改文档后，检查相对链接、代码围栏、JSON 解析及 CSV 与 JSON 字段字典的一致性。每个 `(table, field)` 组合必须唯一，并与 JSON 字典对应。现有基线为 104 条字段记录；确认官方来源变化后，再同步更新字段和计数。修改 Notebook 后检查输出列、类型、有限值及 `(date, instrument)` 唯一性；本地结果、平台运行与正式模型验证的结论应分别标注。

## 来源记录与安全

记录官方来源链接、获取日期和时区偏移，明确区分正文规则、接口配置与整理者判断。存在冲突时，保留在 `open-questions.md`，待官方证据明确后更新。禁止提交密码、Cookie、令牌、账号个人资料或受限行情数据。

文档只保留有效结论、必要使用说明、来源和未决事项，不保存排错经过、堆栈、执行时间线、临时检查次数或中间版本记录。用户确认的运行和提交结果应注明依据，不扩展为模型效果或其他未验证能力。

## 提交与拉取请求约定

主分支使用 `master`，远程 `origin` 为 `git@github.com:ChangzheMa/jycapital.git`。提交说明应简短、明确，建议采用 `docs: 明确 VWAP 标签定义` 这样的格式。拉取请求应说明修改目的、更新来源、已执行的验证和待确认事项，并关联相关问题。仅在需要说明界面行为时附截图。
