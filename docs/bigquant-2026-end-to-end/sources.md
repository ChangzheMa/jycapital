# 来源与结论范围

整理日期：**2026-10-10（UTC+08:00）**。[返回目录](../README.md)

## 已确认结论

| 事项 | 结论 | 依据日期（UTC+08:00） | 依据 |
| --- | --- | --- | --- |
| Mock 本地运行 | 正常 | 2026-10-10 | 用户确认 |
| Mock 平台运行 | 正常 | 2026-10-10 | 用户确认 |
| v1 平台运行 | 可运行 | 2026-10-10 | 用户确认；正式模型提交是否通过及官方评分尚未确认 |
| 测试 submission | 已通过 | 2026-10-10 | 用户确认 |
| 有效输出格式 | Pandas DataFrame，列顺序为 `date`、`instrument`、`factor` | 2026-10-10 | 当前已通过提交的 Mock |
| 本地与平台数据一致性 | 七张表全量逻辑内容一致，243 个文件、723,258,293 行 | 2026-09-30 | [结构化验证摘要](data-validation-summary.json)，平台证据来自用户回传的查询摘要 |
| 本地查询口径 | 直接读取原始五档 `stock` 表及辅助表，保留字段类型与数值 | 2026-10-10 | 本地兼容层与现有 Parquet |
| 字段字典 | 104 条记录，CSV 与 JSON 对应 | 2026-09-22／29 | 官方数据页与辅助表元数据 |

Mock 的运行与提交通过结论不代表模型训练、权重复现或预测能力已获验证。数据一致性结论对应所记录的平台查询基准，不保证源表后续不发生修订。

## 官方来源

| 来源 | 获取日期（UTC+08:00） | 适用内容 |
| --- | --- | --- |
| [比赛介绍](https://bigquant.com/square/competition/45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca) | 2026-09-29；输出与赛程 2026-10-09 | 资格、方法限制、模型材料、评分、赛程与奖励；页面仍使用 `score` 字段名 |
| [比赛数据](https://bigquant.com/square/competition/45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca?activeKey=data) | 2026-09-29／10-09 | 表名、逻辑数据源、1 分钟行情字段、读取方式 |
| [比赛规则](https://bigquant.com/square/competition/45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca?activeKey=rule) | 2026-09-29／10-09 | 通用研究、数据与提交要求 |
| [新手入门](https://bigquant.com/wiki/doc/oSGtQiGtq0) | 2026-10-09 | 唯一 Notebook、`main` 入口、资源及本地训练授权 |
| [端到端模型数据下载指南](https://bigquant.com/wiki/doc/cay6LBmvc1) | 2026-10-09 | `e2e` 与 `stock` 的字段对应、单位缩放和缺失值 |
| [比赛配置接口](https://bigquant.com/bigapis/alphathon/v1/competitions) | 2026-10-09 | [配置摘录](competition-config.json)，以比赛 ID 限定查询 |
| [股票列表元数据](https://jinyicapital.bigquant.com/bigapis/data/v1/spacedatasources/spaces/db3bf8c9-0b87-4b89-a17a-d42cb0b7567b/datasources/cpt_jyc_2026_instruments) | 2026-09-22 | 股票列表字段定义 |
| [风险暴露元数据](https://jinyicapital.bigquant.com/bigapis/data/v1/spacedatasources/spaces/db3bf8c9-0b87-4b89-a17a-d42cb0b7567b/datasources/cpt_jyc_2026_exposure) | 2026-09-22 | 风险暴露字段定义 |
| [VWAP 元数据](https://jinyicapital.bigquant.com/bigapis/data/v1/spacedatasources/spaces/db3bf8c9-0b87-4b89-a17a-d42cb0b7567b/datasources/cpt_jyc_2026_vwap) | 2026-09-22 | 标签字段文字定义，不能替代精确生成公式 |
| [DAI 使用文档](https://bigquant.com/wiki/doc/PLSbc1SbZX)、[DAI 函数文档](https://bigquant.com/wiki/doc/Rceb2JQBdS) | 2026-09-29 | 平台数据查询 API |
| [DuckDB Parquet 文档](https://duckdb.org/docs/current/data/parquet/overview)、[Python API](https://duckdb.org/docs/current/clients/python/reference/) | 2026-10-10 | 本地兼容层所用的 Parquet 查询与结果转换能力 |

## 来源适用原则

- 正文规则、接口配置、用户提供的模板与实际提交结论分别标注；差异保留在 [open-questions.md](open-questions.md)。
- 实际接口采用已通过测试提交的 `factor`；页面字段名差异不改变这一结论。
- [data-schemas.json](data-schemas.json) 和 [data-dictionary.csv](data-dictionary.csv) 保留页面／元数据口径；实际文件类型以 [data-validation-summary.json](data-validation-summary.json) 为准。
- 用户提供的 Transformer 推理模板说明了完整时间连接、三列输出及训练／推理分离；实际正式模型的附件、导入名与权重协议仍须以提交包为准。
- 配置中的[代码分享](https://bigquant.com/codesharev3/7c55a1c9-7591-43d1-b934-1ae943513ce5)的内容依据日期为 2026-09-22，属于旧日频评估代码，不能作为当前赛道评测算法。
- [通用模型本地化训练指南](https://bigquant.com/wiki/doc/MvWIurzUct)包含 BigAlpha 表名和未来一日收益示例，不适用于确定本赛事的标签或训练边界。
