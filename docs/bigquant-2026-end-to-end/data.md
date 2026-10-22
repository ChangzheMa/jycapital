# 比赛数据与字段说明

页面与元数据的获取日期见 [sources.md](sources.md)。[返回目录](../README.md) · [规则](rules.md) · [待确认事项](open-questions.md)

本地已有四张五档分钟表和三张辅助表；运行方式和文件路径见 [local-download.md](local-download.md)。[data-dictionary.csv](data-dictionary.csv) 与 [data-schemas.json](data-schemas.json) 保存 104 条页面／元数据字段定义，实际文件类型以 [data-validation-summary.json](data-validation-summary.json) 为准。

本地覆盖、实际字段类型及原始异常见 [数据一致性结论](data-validation.md)；内容一致不代表每个数值都可直接作为有效输入。

## 1. 范围与访问方式

| 项目 | 官方页面公布内容 |
| --- | --- |
| 股票池 | 以中证 1000 指数历史成分股为基础 |
| 数据年份 | 数据页总体说明为 2020—2024 年；未给出各表精确首末时间或本地下载数据的独立范围 |
| 行情频率 | 1、5、15、30 分钟 K 线及盘口快照 |
| 预测窗口 | 未来 30 分钟；不预测隔夜收益 |
| 采样 | 每隔 30 分钟，每日约 8 个截面，跳过集合竞价等特殊时段 |
| 空间 | [进益资本空间](https://jinyicapital.bigquant.com/) |
| 数据目录 | 数据平台 `/进益资本杯2026` |
| 授权流程 | 报名审核通过后加入该空间并获得数据权限 |
| 本地训练数据 | 通过身份核验、签署限制协议后申请下载，下载数据仅限本人或本队参赛使用 |
| 数据限制 | 仅可使用比赛指定且获授权的数据源 |

来源：[比赛数据页](https://bigquant.com/square/competition/45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca?activeKey=data)、[介绍中的数据要求](https://bigquant.com/square/competition/45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca)。

官方训练、公榜验证和私榜评测区间互不重叠，精确分界未公布；项目的本地研究划分见 [评估约定](../development.md)，不代表官方分界。《新手入门》说明验证集在正式评测区间前额外提供一年历史上下文，供滚动窗口、滞后项和序列模型使用；最终输出必须限制在 `start_date` 至 `end_date` 内，公榜仍只加载已训练权重推理。

## 2. 数据表、逻辑名与本地下载

比赛空间 ID：`db3bf8c9-0b87-4b89-a17a-d42cb0b7567b`，来自比赛详情配置。

### 2.1 开发数据表

| 表名 | 用途 | 本地实际字段数 |
| --- | --- | ---: |
| `cpt_jyc_2026_stock_bar1m` | 1 分钟行情与五档盘口 | 42 |
| `cpt_jyc_2026_stock_bar5m` | 5 分钟行情与五档盘口 | 42 |
| `cpt_jyc_2026_stock_bar15m` | 15 分钟行情与五档盘口 | 42 |
| `cpt_jyc_2026_stock_bar30m` | 30 分钟行情与五档盘口 | 42 |
| `cpt_jyc_2026_instruments` | 采样时刻及股票池 | 3 |
| `cpt_jyc_2026_exposure` | 风险／风格暴露 | 48 |
| `cpt_jyc_2026_vwap` | 未来 30 分钟 VWAP 相关标签 | 8 |

辅助表元数据中的 `__PARTITION__` 不在实际导出文件中，因此元数据字段数分别为 4／49／9，实际文件为 3／48／8。字段定义与查询口径分开保留。

来源：[比赛数据页](https://bigquant.com/square/competition/45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca?activeKey=data)及[实际数据摘要](data-validation-summary.json)。

### 2.2 评测数据源映射

评测入口为 `main(datasources, start_date, end_date)`。`datasources` 将 `bar1m`、`bar5m`、`bar15m`、`bar30m` 映射到当前评测数据表，必须通过逻辑名取得行情表，例如 `datasources["bar1m"]`，不能在评测 SQL 中硬编码开发表或下载表名称。股票池可使用 `cpt_jyc_2026_instruments`。

来源：[新手入门“三、实现评测入口”](https://bigquant.com/wiki/doc/oSGtQiGtq0)。

### 2.3 本地训练下载数据

| 页面列出的表 | 用途 |
| --- | --- |
| [`cpt_jyc_2026_e2e_bar1m`](https://jinyicapital.bigquant.com/data/datasources/cpt_jyc_2026_e2e_bar1m) | 端到端本地下载，1 分钟 |
| [`cpt_jyc_2026_e2e_bar5m`](https://jinyicapital.bigquant.com/data/datasources/cpt_jyc_2026_e2e_bar5m) | 端到端本地下载，5 分钟 |
| [`cpt_jyc_2026_e2e_bar15m`](https://jinyicapital.bigquant.com/data/datasources/cpt_jyc_2026_e2e_bar15m) | 端到端本地下载，15 分钟 |
| [`cpt_jyc_2026_e2e_bar30m`](https://jinyicapital.bigquant.com/data/datasources/cpt_jyc_2026_e2e_bar30m) | 端到端本地下载，30 分钟 |

申请与使用限制见[本地训练数据申请](rules.md#51-本地训练数据申请)。下载版数据与公榜训练集的关系仍待官方材料确认。

[官方端到端数据下载指南](https://bigquant.com/wiki/doc/cay6LBmvc1)明确了压缩版与云端表的对应关系：

| 维度 | 压缩下载版 `e2e_bar*` | 云端及当前本地 `stock_bar*` |
| --- | --- | --- |
| 字段与盘口 | 28 列，三档 | 42 列，五档 |
| 股票代码 | 仅 `instrument_id` | 同时有 `instrument` 和 `instrument_id` |
| OHLC／盘口价格 | 整数，单位“分” | 浮点，单位“元” |
| 成交金额 | 整数，单位“分” | `double`，单位“元” |
| OHLC 缺失 | `-1` | 缺失值 |
| 盘口价格缺失 | `0` | `0` |
| `pre_close` | 无 | 有 |
| `date` | Bar 结束时刻 | Bar 结束时刻 |

使用压缩版时，OHLC 的 `-1` 先转为空值，价格和成交金额再除以 100；两端模型输入统一为前三档及共同字段。**当前项目实际存放的是 `stock` 版本，不应再次缩放。** 全量一致性结论覆盖 `stock` 及三张辅助表，不包含压缩版全量内容比较。

## 3. 分钟行情字段：赛事页公布的 1 分钟表

页面共列出 **42 个字段**。下表中 `1 … 5` 表示五个独立字段，CSV 已逐个展开，类型保留页面给出的 NumPy/Pandas 口径。2026-09-30 的实际数据验证已确认四张 `stock_bar*` 均有相同的 42 列；其中 `deal_number`、`volume` 实际为 `int64`，`amount` 为 `double`，与下表的页面类型有差异。实际类型见 [验证摘要](data-validation-summary.json)，不直接覆盖历史页面字典。

| 字段 | 页面类型 | 含义 |
| --- | --- | --- |
| `date` | `np.datetime64` | Bar 的结束时刻 |
| `instrument` | `pd.StringDtype` | 股票标识 |
| `instrument_id` | `np.int16` | 股票标识的数值 ID |
| `adjust_factor` | `np.float32` | 复权因子 |
| `pre_close` | `np.float32` | 前收盘价，具体前一日/前一 Bar 口径未展开 |
| `high` | `np.float32` | Bar 最高价 |
| `open` | `np.float32` | Bar 开盘价 |
| `low` | `np.float32` | Bar 最低价 |
| `close` | `np.float32` | Bar 收盘价 |
| `deal_number` | `np.int32` | 成交笔数 |
| `volume` | `np.int32` | 成交量，单位未在该字段表中明确 |
| `amount` | `np.float32` | 页面类型；云端实际为 `double`，单位“元” |
| `ask_price1` … `ask_price5` | `np.float32` | 卖方 1—5 档报价 |
| `bid_price1` … `bid_price5` | `np.float32` | 买方 1—5 档报价 |
| `ask_volume1` … `ask_volume5` | `np.int32` | 卖方各档委托量 |
| `bid_volume1` … `bid_volume5` | `np.int32` | 买方各档委托量 |
| `ask_num_orders1` … `ask_num_orders5` | `np.int32` | 卖方各档委托笔数 |
| `bid_num_orders1` … `bid_num_orders5` | `np.int32` | 买方各档委托笔数 |

页面示例中股票标识采用类似 `000010.SZ` 的格式，`date` 保留分钟。原文提到可用 `instrument_id` 节省内存，但未解释 ID 的跨表稳定性及映射规则。输出仍应按比赛要求提供 `instrument`。

盘口快照具体取样时刻、高频聚合规则、零值含义、复权方式和成交量单位均需明确。不能由字段名推断 5/15/30 分钟表的盘口是区间平均或末值。

来源：[比赛数据页“数据格式”“读取示例”](https://bigquant.com/square/competition/45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca?activeKey=data)。

## 4. 股票列表（元数据口径）

`cpt_jyc_2026_instruments` 的 schema 有 4 列，其中 3 列有业务字段说明。

| 字段 | 元数据类型 | 官方字段说明及标记 |
| --- | --- | --- |
| `date` | `timestamp[ns]` | 日期；`primary=true` |
| `instrument` | `string` | 证券代码；`primary=true` |
| `name` | `string` | 证券简称 |
| `__PARTITION__` | `int64` | 系统 schema 中存在，未提供业务说明 |

输出按完整 `date`、`instrument` 对齐股票池。历史成分调整、停牌／ST／上市／退市的业务处理仍需明确；不得用未来成分列表回填历史。

来源：[股票列表官方详情](https://jinyicapital.bigquant.com/data/datasources/cpt_jyc_2026_instruments)，元数据读取日期与接口见 [sources.md](sources.md)。

## 5. 风险暴露（元数据口径）

**当前研究限制（用户于 2026-10-22 确认）：**该表已删除，训练、评估、推理和诊断均不使用，也不恢复。以下只保留历史官方元数据，不表示当前本地可用或获准使用。

`cpt_jyc_2026_exposure` 的 schema 有 **49 列**：日期与股票、10 个风格暴露、32 个行业哑变量、4 个辅助业务字段和 1 个系统分区字段。完整名字及说明见 CSV。

| 字段/分组 | 类型 | 字段含义 |
| --- | --- | --- |
| `date`、`instrument` | `timestamp[ns]`、`string` | 日期、证券代码；元数据均标记为主键字段 |
| `SIZE` | `double` | 市值风格 |
| `BETA` | `double` | 贝塔 |
| `MOMENTUM` | `double` | 传统动量 |
| `RESVOL` | `double` | 残差波动率 |
| `SIZENL` | `double` | 非线性市值 |
| `BTOP` | `double` | 账面市值比 |
| `LIQUIDTY` | `double` | 流动性；保留官方拼写 |
| `EARNYILD` | `double` | 盈利能力；保留官方拼写 |
| `GROWTH` | `double` | 成长 |
| `LEVERAGE` | `double` | 杠杆 |
| 32 个行业列 | `int8` | 0/1 行业归属哑变量 |
| `industry_level1_code` | `string` | 一级行业代码 |
| `float_market_cap` | `double` | 流通市值 |
| `weights` | `double` | 截面归一化市值权重 |
| `ret` | `double` | 当期收益率，具体收益区间未说明 |
| `__PARTITION__` | `int64` | 系统分区字段，未提供业务说明 |

32 个行业列为：

```text
AGRIFOREST MINING CHEM IRONSTEEL NONFERMETAL ELECTRONICS AUTO HOUSEAPP
FOODBEVER TEXTILE LIGHTINDUS HEALTH UTILITIES TRANSPORTATION REALESTATE
COMMETRADE LEISERVICE BANK NONBANKFINAN CONGLOMERATES CONMAT BUILDDECO
ELECEQP MACHIEQUIP AERODEF COMPUTER MEDIA TELECOM COAL PETRO ENVP BEAUTY
```

正文说明平台会对输出做 BARRA 风格剔除，但未给出回归权重、截距、行业约束、去极值参数、缺失处理及暴露的可用时间。该表含风格统计量和 `ret`，**被列为授权表不等于所有字段都获准作为端到端模型输入**；用途和时点仍需确认。

来源：[风险暴露官方详情](https://jinyicapital.bigquant.com/data/datasources/cpt_jyc_2026_exposure)及对应元数据接口；预处理规则见 [比赛介绍](https://bigquant.com/square/competition/45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca)。

## 6. VWAP 标签：需要特别核实的口径

**当前实现约定（用户于 2026-10-22 确认）：**直接读取已提供的 `vwap_return`，可继续做标准化、排名等加工；不另行计算、替换或反推其具体公式。下面的文字差异作为历史来源问题保留，不据此重算标签。

`cpt_jyc_2026_vwap` 的官方中文名称为“未来30分VWAP收益率”，2026-09-22 读取的 schema 有 9 列。元数据说明如下：

| 字段 | 元数据类型 | 字段定义 |
| --- | --- | --- |
| `date` | `timestamp[ns]` | 采样时刻，即未来 30 分钟窗口起点 |
| `instrument` | `string` | 股票或指数标识 |
| `vwap_return` | `float` | 窗口终点价格相对于未来 30 分钟 VWAP 的收益率 |
| `vwap` | `float` | 未来 30 分钟的成交量加权均价 |
| `end_price` | `float` | 未来窗口结束时最新有效成交价 |
| `volume` | `int64` | 未来窗口成交量 |
| `amount` | `double` | 未来窗口成交额，单位为元 |
| `num_trades` | `int32` | 未来窗口成交笔数 |
| `__PARTITION__` | `int64` | 系统分区字段，未提供业务说明 |

来源：[VWAP 表元数据接口](https://jinyicapital.bigquant.com/bigapis/data/v1/spacedatasources/spaces/db3bf8c9-0b87-4b89-a17a-d42cb0b7567b/datasources/cpt_jyc_2026_vwap)，访问方法见 [sources.md](sources.md)。

需要区分三件事：

1. **正文概述**：未来 30 分钟 VWAP 收益；开头又出现残差收益率目标。
2. **表字段说明**：`vwap_return` 使用窗口终点价格与未来窗口 VWAP 比较，文字上并未以当前价为分母。
3. **尚未确认**：精确代数公式、是否为简单／对数收益、是否再做风险残差化、窗口端点包含规则及无成交处理。

因此不应直接把标签写成 `future_vwap/current_price - 1`，也不应未经核实就把 `end_price/vwap - 1` 当作官方确认公式。后者仅是依据字段描述可能作出的解释。

`vwap`、`end_price`、`amount`、`volume`、`num_trades` 描述的都是未来窗口信息，在预测时点不能作为已知输入。训练阶段使用标签，与推理阶段读取未来字段，是不同的数据使用行为。

## 7. 时间对齐与覆盖

| 数据或输出 | 时间语义 |
| --- | --- |
| 行情 `date` | Bar 结束时间 |
| VWAP 表 `date` | 预测窗口开始时间 |
| 模型输出 `date` | 与股票池的完整分钟采样时刻对齐 |
| 股票列表 `date` | 含完整分钟的采样时刻 |
| 风险暴露 `date` | 日频日期；实际可获得时间待确认 |

本地股票池的日内时刻为 `09:30`、`10:00`、`10:30`、`11:00`、`11:30`、`13:30`、`14:00`、`14:30`；实际推理读取股票池记录，不将此列表硬编码为所有评测区间的日历。

时间对齐要求：

- 对齐前先确定“信息已可获得”的时点，不能仅因时间字符串相等就拼接所有字段。
- 获取官方采样点列表，并核实午休是否跨越、最后一个窗口如何截断、集合竞价是否排除。
- 保留完整分钟时间，通用示例中的 `date::DATE` 会丢失时分秒。
- 高频序列可向过去扩展以构建输入，但不能读取预测时点之后的 Bar。
- 对归一化统计量和数据填充记录拟合区间及方向，避免使用验证/私榜未来信息。

上述是由字段语义和禁止未来信息规则得到的实现建议，并非官方公布的完整对齐算法。来源：[比赛数据页](https://bigquant.com/square/competition/45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca?activeKey=data)、[比赛规则](https://bigquant.com/square/competition/45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca?activeKey=rule)。

## 8. 读取方式与开发资源

本地四张 `cpt_jyc_2026_stock_bar*` 按自然月存储，三张辅助表各保存一个文件。路径与本地 DAI 支持范围见 [local-download.md](local-download.md)，一致性结论见 [data-validation.md](data-validation.md)。

按月读取应同时设置 SQL 和 `filters`，两者使用相同的下月 1 日上界。例如 2020 年 11 月采用 `date >= '2020-11-01' AND date < '2020-12-01'`，并传入 `filters={"date": ["2020-11-01", "2020-12-01"]}`。不要把 `filters` 上界减一天：日期字符串可能按当天零点解释，从而漏掉月末日内行情。严格的 SQL `<` 条件负责排除下月起点。该边界同时适用于本地兼容层的查询。

平台通过 `dai.query(SQL, filters={"date": [start, end]}).df()` 读取数据，也支持 Arrow／Polars 和绑定 DataFrame。正式推理的行情表名来自 `datasources`，返回完整评测区间的结果；本地兼容层的支持范围见本地环境文档。

开发计算资源与最终评测配额分别管理，见 [运行环境](rules.md#52-线上运行与资源)。来源：[比赛数据页](https://bigquant.com/square/competition/45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca?activeKey=data)、[DAI 使用文档](https://bigquant.com/wiki/doc/PLSbc1SbZX)、[DAI 函数文档](https://bigquant.com/wiki/doc/Rceb2JQBdS)。

## 9. 待明确事项

官方数据边界、标签公式、信息可获得时间、供应商聚合、覆盖豁免及完整评分口径见 [未决事项](open-questions.md)。
