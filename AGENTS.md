# 仓库贡献指南

## 项目结构与文件组织

本仓库整理 BigQuant 2026 年进益资本杯端到端赛道资料。先阅读 `docs/README.md`，比赛材料位于 `docs/bigquant-2026-end-to-end/`：

- `rules.md`、`data.md`：比赛要求与数据说明。
- `open-questions.md`、`sources.md`：待确认差异与来源记录。
- `competition-config.json`：官方比赛配置的相关字段摘录。
- `data-schemas.json`、`data-dictionary.csv`：结构化字段字典。

目前没有应用代码、测试目录或构建资源。相关资料应集中存放，新增文件时同步更新文档索引。

## 构建、检查与开发命令

目前未配置构建系统、依赖清单或本地应用启动入口。在仓库根目录使用 PowerShell 和 Python 检查文件：

```powershell
rg --files docs
python -m json.tool docs/bigquant-2026-end-to-end/competition-config.json > $null
python -m json.tool docs/bigquant-2026-end-to-end/data-schemas.json > $null
```

第一条命令列出文档，后两条检查 JSON 语法。应检查命令退出码，确认验证成功。

## 编写风格与命名约定

说明文档使用中文，官方标识符保持原样。使用含义明确的 Markdown 标题、相对文件链接和围栏代码块；文件名采用小写字母，单词间用连字符分隔。Markdown 和 JSON 使用 UTF-8，JSON 使用两空格缩进；CSV 保留 UTF-8 BOM 和既有列顺序。Python 示例使用四空格缩进和 `snake_case` 命名。目前未配置格式化或静态检查工具。

## 验证要求

目前没有自动化测试框架或覆盖率要求。修改文档后，检查相对链接、代码围栏、JSON 解析及 CSV 与 JSON 字段字典的一致性。每个 `(table, field)` 组合必须唯一，并与 JSON 字典对应。现有基线为 104 条字段记录；确认官方来源变化后，再同步更新字段和计数。验证记录应区分元数据检查、实际数据查询和模型运行。

## 来源记录与安全

记录官方来源链接、获取日期和时区偏移，明确区分正文规则、接口配置与整理者判断。存在冲突时，保留在 `open-questions.md`，待官方证据明确后更新。禁止提交密码、Cookie、令牌、账号个人资料或受限行情数据。

## 提交与拉取请求约定

主分支使用 `master`，远程 `origin` 为 `git@github.com:ChangzheMa/jycapital.git`。提交说明应简短、明确，建议采用 `docs: 明确 VWAP 标签定义` 这样的格式。拉取请求应说明修改目的、更新来源、已执行的验证和待确认事项，并关联相关问题。仅在需要说明界面行为时附截图。
