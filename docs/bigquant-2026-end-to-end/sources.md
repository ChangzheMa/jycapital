# 来源、访问结果与记录方法

核查日期：**2026-09-22（UTC+08:00）**。[返回目录](../README.md)

## 1. 官方页面

| 来源 | 地址 | 本次结果与用途 |
| --- | --- | --- |
| 比赛主页 | [进益资本杯 · 端到端赛道](https://bigquant.com/square/competition/45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca) | HTTP 200，读取介绍、评分、赛程和正文要求 |
| 数据标签页 | [比赛数据](https://bigquant.com/square/competition/45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca?activeKey=data) | 直接 HTTP 读取成功，得到数据说明；浏览抓取工具单独打开此查询参数链接曾报错 |
| 规则标签页 | [比赛规则](https://bigquant.com/square/competition/45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca?activeKey=rule) | 从主页 HTML 内 `id=rule` 的完整规则区读取；未将标签 URL 的工具报错当作正文缺失 |
| 数据空间 | [进益资本](https://jinyicapital.bigquant.com/) | 空间入口，目录名称由比赛数据正文确认 |
| 原文链接的股票列表 | [股票列表详情](https://jinyicapital.bigquant.com/data/datasources/cpt_jyc_2026_instruments) | 未登录 HTML 仅显示通用页面；登录后从官方详情接口获取 schema |
| 原文链接的风险暴露 | [风险暴露详情](https://jinyicapital.bigquant.com/data/datasources/cpt_jyc_2026_exposure) | 同上 |
| 原文链接的 1 分钟表 | [1 分钟表详情](https://jinyicapital.bigquant.com/data/datasources/cpt_jyc_2026_stock_bar1m) | 未登录 HTML 为通用页面；登录后详情接口返回 404 |

2026-09-22 22:04（UTC+08:00）的公开页面抓取中，两条用户提供的 URL 均返回 **87,308 字节**，响应体 SHA-256 相同：

```text
2c16c782579d0df2db6007325533d281eea9c37ac57a39674f53c4809f511264
```

HTML 同时包含 `overview`、`data` 和 `rule` 内容。哈希用于记录当次响应，不保证今后的导航栏、脚本版本或正文不变；本目录没有附整页转载。

## 2. 登录后的官方接口核查

用户授权登录后继续只读核查比赛、当前账号报名状态、数据元信息及公开模板。登录成功，当前比赛报名记录状态为 `approved_join_space`。本目录不记录该账号的手机号、密码、用户 ID、令牌或 Cookie。

### 比赛详情

请求：[官方比赛详情列表接口](https://bigquant.com/bigapis/alphathon/v1/competitions)。

```text
GET /bigapis/alphathon/v1/competitions
constraints = {"id": "45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca"}
```

响应 HTTP 200、业务 `code=0`，仅返回目标比赛。比赛记录的 `updated_at` 为 **2026-09-22 11:11:59.606690+08:00**，这是记录级更新时间，不代表每条规则都有独立版本时间。提取来源包括：

- `data.items[0]`：比赛名、日期、总奖金、空间 ID。
- `summary`：报名、提交、候选选择、合队截止配置。
- `data.competition`：介绍、数据、规则正文，以及人数、模板链接等配置。

正文分别用于 [rules.md](rules.md) 和 [data.md](data.md)；配置仅保留与任务相关的字段，见 [competition-config.json](competition-config.json)，其中 `checked_at` 记录本次读取时间。

### 数据源详情

这是官网前端使用的详情接口形式，访问相应表名，不做表内容查询：

```text
GET https://jinyicapital.bigquant.com/bigapis/data/v1/spacedatasources/spaces/
    db3bf8c9-0b87-4b89-a17a-d42cb0b7567b/datasources/{table_name}
```

上面为便于阅读而换行，实际请求 URL 为连续字符串。

| 表 | HTTP/业务结果 | 已读取内容 |
| --- | --- | --- |
| `cpt_jyc_2026_stock_bar1m` | 404 / `NOT_FOUND` | 未取得元数据 |
| `cpt_jyc_2026_stock_bar5m` | 404 / `NOT_FOUND` | 未取得元数据 |
| `cpt_jyc_2026_stock_bar15m` | 404 / `NOT_FOUND` | 未取得元数据 |
| `cpt_jyc_2026_stock_bar30m` | 404 / `NOT_FOUND` | 未取得元数据 |
| `cpt_jyc_2026_instruments` | 200 / `code=0` | `metadata.schema`、`docs.schema`、中文表名 |
| `cpt_jyc_2026_exposure` | 200 / `code=0` | 同上 |
| `cpt_jyc_2026_vwap` | 200 / `code=0` | 同上 |

逐表完整 URL 和辅助表的读取时间均保存在 [data-schemas.json](data-schemas.json)。此外，1 分钟表全局接口 `GET https://bigquant.com/bigapis/data/v1/datasources/cpt_jyc_2026_stock_bar1m` 也返回 404。比赛空间的表列表接口返回 HTTP 200 但条目数为 0；这与直接获取三张辅助表元数据的结果一并保留，未据此推断表不存在。

### 配置指向的代码分享

入口：[代码分享 7c55a1c9…](https://bigquant.com/codesharev3/7c55a1c9-7591-43d1-b934-1ae943513ce5)。

- 元数据接口：`GET https://bigquant.com/bigapis/codeshare/v1/v2shares/7c55a1c9-7591-43d1-b934-1ae943513ce5`，返回 HTTP 200、业务成功。
- 官方前端读取内容使用 `POST /bigapis/codeshare/v1/v2shares/download`，请求体仅为对应 `share_id`；本次调用成功并返回 Notebook JSON。
- 名称为“模型评估v2_3”，2025-08-19 创建和更新；公开分享，共 5 个代码单元。
- 本次仅静态阅读代码，未运行、克隆、提交或修改线上作品。
- 原先尝试 `/shares/{id}` 路径返回 404，随后按官网前端实际使用的 `/v2shares/{id}` 路径成功；模板不可据前一个错误判断不存在。
- 对旧模板与当前正文的差异只作事实记录，见 [open-questions.md](open-questions.md)。

## 3. 补充官方入口

以下为数据页或比赛配置提供的入口，不代表本次已获取其所有内容：

| 入口 | 用途 | 本次覆盖 |
| --- | --- | --- |
| [DAI 使用文档](https://bigquant.com/wiki/doc/PLSbc1SbZX) | 查询和数据格式转换 | 记录官方引用链接，未逐项验证 |
| [DAI 函数文档](https://bigquant.com/wiki/doc/Rceb2JQBdS) | 时序、截面、时间等算子 | 记录官方引用链接，未逐项验证 |
| [配置中的讨论文档](https://bigquant.com/wiki/doc/zkrliPcm32) | 官方配置的讨论入口 | 记录路径，未将讨论内容作为已确认规则 |

## 4. 文档生成与可核查范围

- 人工整理赛程、提交、评分、约束和冲突，每节附相关来源。
- 从比赛正文数据字段表提取 1 分钟行情的 42 个字段；未擅自向其他频率表复制 schema。
- 从三张辅助表的 `metadata.schema` 与 `docs.schema` 提取字段类型、定义和主键标记，排除创建者和账号资料。
- [data-dictionary.csv](data-dictionary.csv) 含 **104 条字段记录**：1 分钟行情 42、股票列表 4、风险暴露 49、VWAP 9；后三者均含一个未提供业务说明的 `__PARTITION__`。
- JSON 使用 UTF-8，CSV 使用带 BOM 的 UTF-8 便于 Windows 表格工具打开。
- 实际数据查询、数据分布统计、模型训练、推理、提交、支付或报名变更均不属于本次核查结果。
- 记录“尚未确认”表示本次材料不足，不等于官方永远未提供；需要后续公告、授权环境查询或正式答复补齐。

## 5. 后续更新方式

重新读取目标比赛后，对照 `updated_at`、正文赛程、`summary` 的截止时间和数据表 schema。若出现差异，在 [open-questions.md](open-questions.md) 记录新来源与确认状态，再更新说明文档和结构化字典。不得仅根据配置中的较晚日期、旧模板或猜测的字段公式覆盖正文规则。
