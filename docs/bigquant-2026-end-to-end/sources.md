# 来源与核查范围

公开页面与比赛配置核查日期：**2026-09-29（UTC+08:00）**。[返回目录](../README.md)

## 1. 当前官方来源

| 来源 | 获取日期（UTC+08:00） | 已核对内容 |
| --- | --- | --- |
| [比赛主页](https://bigquant.com/square/competition/45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca) | 2026-09-29 | 资格、端到端限制、模板名称、提交、评分、赛程、答辩和奖励 |
| [比赛数据](https://bigquant.com/square/competition/45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca?activeKey=data) | 2026-09-29 | 通过主页 HTML 的 `id=data` 区读取数据表、42 个行情字段、本地下载入口及查询示例 |
| [比赛规则](https://bigquant.com/square/competition/45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca?activeKey=rule) | 2026-09-29 | 通过主页 HTML 的 `id=rule` 区读取完整通用规则 |
| [新手入门](https://bigquant.com/wiki/doc/oSGtQiGtq0) | 2026-09-29 | 数据页指定的申请指南；核对评测入口、唯一 Notebook、下载审批、历史上下文、提交和资源 |
| [比赛详情接口](https://bigquant.com/bigapis/alphathon/v1/competitions) | 2026-09-29 | HTTP 200、业务 `code=0`，返回目标比赛，摘录截止时间、人数、模板和奖励配置 |
| [赛道讨论入口](https://bigquant.com/wiki/doc/zkrliPcm32) | 2026-09-29 | 页面链接回比赛，未提供可用于裁定规则冲突的补充内容 |

主页 HTTP 200，响应体为 **88,129 字节**，同时包含 `overview`、`data`、`rule` 三个区域。响应体 SHA-256：

```text
277223532822f886cbdd85ddf8a7609794a15ed1df04b72737931f35d532e279
```

《新手入门》由比赛数据页直接引用，页面显示更新于 `2026-09-29 06:30`；该显示值未注明时区，不能自行按 UTC+08:00 解读。本文的获取日期使用 UTC+08:00。数据页原始链接带邀请参数，正文引用使用相同文档的无参数地址。

## 2. 比赛配置

接口查询限定当前比赛：

```text
GET https://bigquant.com/bigapis/alphathon/v1/competitions
constraints = {"id": "45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca"}
```

记录级 `updated_at` 为 **2026-09-29T14:33:21.656666+08:00**，不代表各条规则都有独立版本时间。相关字段保存于 [competition-config.json](competition-config.json)：

- `data.items[0]`：比赛名称、起止时间、总奖金和空间 ID。
- `summary`：报名、提交、候选选择和合队截止配置。
- `data.competition`：队伍人数、模板链接、提交配置、奖励等字段。

接口值与正文的差异见 [open-questions.md](open-questions.md)。本次接口读取不使用登录凭据，也不复核个人报名或数据访问权限。

## 3. 数据表与字典来源

当前[数据页](https://bigquant.com/square/competition/45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca?activeKey=data)公布四张 `stock_bar*` 开发表、三张辅助表及四张 `e2e_bar*` 本地下载表。当前核对的是公开表名、链接及 1 分钟行情字段表；没有重新请求表详情或查询数据内容。

辅助表字段与开发行情表的详情接口结果来自 **2026-09-22（UTC+08:00）** 的核查，逐表获取时间保存在 [data-schemas.json](data-schemas.json)。接口形式如下，实际 URL 不换行：

```text
GET https://jinyicapital.bigquant.com/bigapis/data/v1/spacedatasources/spaces/
    db3bf8c9-0b87-4b89-a17a-d42cb0b7567b/datasources/{table_name}
```

| 表组 | 元数据核查结果（2026-09-22） | 字典来源 |
| --- | --- | --- |
| `cpt_jyc_2026_stock_bar1m` | HTTP 404 / `NOT_FOUND` | 赛事页字段表，42 列；已与当前页面核对 |
| `cpt_jyc_2026_stock_bar5m`、`stock_bar15m`、`stock_bar30m` | HTTP 404 / `NOT_FOUND` | 未取得 schema，不复制 1 分钟字段 |
| `cpt_jyc_2026_instruments` | HTTP 200 / `code=0` | `metadata.schema` 与 `docs.schema`，4 列 |
| `cpt_jyc_2026_exposure` | HTTP 200 / `code=0` | 同上，49 列 |
| `cpt_jyc_2026_vwap` | HTTP 200 / `code=0` | 同上，9 列 |

四张本地下载表的 schema 本次未核查，JSON 仅记录当前官方入口和空字段列表。字典仍有 **104 条字段记录**，CSV 使用带 BOM 的 UTF-8，列顺序保持一致。字段定义不代表各字段均获准作为端到端输入。

## 4. 模板来源与范围

当前介绍和《新手入门》列出的端到端模板为 `Transformer_modelsave_train.ipynb` 和 `Transformer_modelsave_predict.ipynb`，获取路径为“比赛 → 代码”。已核对模板名称、用途及参考提交组合，未取得两个文件的实际内容。

当前配置中的[代码分享](https://bigquant.com/codesharev3/7c55a1c9-7591-43d1-b934-1ae943513ce5)对应以下读取接口：

```text
GET https://bigquant.com/bigapis/codeshare/v1/v2shares/7c55a1c9-7591-43d1-b934-1ae943513ce5
POST https://bigquant.com/bigapis/codeshare/v1/v2shares/download
请求体：{"share_id": "7c55a1c9-7591-43d1-b934-1ae943513ce5"}
```

该分享最近一次内容核查为 **2026-09-22**，名称“模型评估v2_3”，创建和更新于 2025-08-19，共 5 个代码单元，包含旧表和日频逻辑。未把其实现作为当前赛道规则；本次没有运行或克隆该 Notebook。

## 5. 验证边界

- **页面与配置检查**：核对当前官方正文、指南、配置及字段表，区分规则、配置与整理建议。
- **表元数据检查**：三张辅助表 schema 沿用注明日期的官方结果；四张下载表 schema 未取得。
- **实际数据查询**：本次未执行；SDK 和 Studio 的最近一次查询结果见 [local-download.md](local-download.md)，未成功取得比赛数据。
- **模型运行**：未训练、推理或提交模型；文档中的代码仅作接口说明。
- **本地文档检查**：检查相对链接、代码围栏、JSON 解析，以及 CSV/JSON 字段唯一性和一致性。

正文还提供 [DAI 使用文档](https://bigquant.com/wiki/doc/PLSbc1SbZX)和[DAI 函数文档](https://bigquant.com/wiki/doc/Rceb2JQBdS)作为参考；本次没有逐项验证函数实现。本目录不保存凭据、账号个人资料、学生卡、限制协议签署材料或受限行情数据。
