# 比赛数据与字段说明

核查日期：2026-09-22（UTC+08:00）。[返回目录](../README.md) · [规则](rules.md) · [待确认事项](open-questions.md)

本文件区分赛事页介绍与登录后读取的表元数据。**已读取字段和说明，未运行 DAI 查询、统计实际数据或下载比赛行情。** 完整字段清单见 [data-dictionary.csv](data-dictionary.csv) 和 [data-schemas.json](data-schemas.json)。

## 1. 范围与访问方式

| 项目 | 官方页面公布内容 |
| --- | --- |
| 股票池 | 以中证 1000 指数历史成分股为基础 |
| 数据年份 | 2020—2024 年；未给出每张表精确首末时间 |
| 行情频率 | 1、5、15、30 分钟 K 线及盘口快照 |
| 预测窗口 | 未来 30 分钟；不预测隔夜收益 |
| 采样 | 每隔 30 分钟，每日约 8 个截面，跳过集合竞价等特殊时段 |
| 空间 | [进益资本空间](https://jinyicapital.bigquant.com/) |
| 数据目录 | 数据平台 `/进益资本杯2026` |
| 授权流程 | 报名审核通过后加入该空间并获得数据权限 |
| 数据限制 | 仅可使用比赛指定且获授权的数据源 |

来源：[比赛数据页](https://bigquant.com/square/competition/45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca?activeKey=data)、[介绍中的数据要求](https://bigquant.com/square/competition/45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca)。

训练集、公榜验证集与私榜数据互不重叠，但未公布精确分界。本次没有把 2020—2024 年自行拆成训练/验证/测试，也没有假定这些年份全部可以本地训练使用。

## 2. 数据表清单及本次核查结果

比赛空间 ID：`db3bf8c9-0b87-4b89-a17a-d42cb0b7567b`，来自比赛详情配置。

| 表名 | 公布用途 | 登录后详情接口结果 |
| --- | --- | --- |
| `cpt_jyc_2026_stock_bar1m` | 1 分钟行情与盘口 | HTTP 404 / `NOT_FOUND`；只能引用赛事页字段表 |
| `cpt_jyc_2026_stock_bar5m` | 5 分钟行情与盘口 | HTTP 404 / `NOT_FOUND` |
| `cpt_jyc_2026_stock_bar15m` | 15 分钟行情与盘口 | HTTP 404 / `NOT_FOUND` |
| `cpt_jyc_2026_stock_bar30m` | 30 分钟行情与盘口 | HTTP 404 / `NOT_FOUND` |
| `cpt_jyc_2026_instruments` | 股票列表 | HTTP 200，读到 schema 与字段说明 |
| `cpt_jyc_2026_exposure` | 风险/风格暴露 | HTTP 200，读到 schema 与字段说明 |
| `cpt_jyc_2026_vwap` | 未来 30 分钟 VWAP 相关标签 | HTTP 200，读到 schema 与字段说明 |

当前授权账号报名记录的状态为 `approved_join_space`。元数据可读不等于已验证实际行情查询权限；404 也不能区分表未发布、表名调整、空间配置或权限屏蔽等原因。另检查了 1 分钟表的全局详情接口，同样为 404；空间表列表返回空列表，不能据此覆盖三张辅助表的直接读取结果。

原文提供了 [1 分钟行情详情](https://jinyicapital.bigquant.com/data/datasources/cpt_jyc_2026_stock_bar1m)、[股票列表详情](https://jinyicapital.bigquant.com/data/datasources/cpt_jyc_2026_instruments)、[风险暴露详情](https://jinyicapital.bigquant.com/data/datasources/cpt_jyc_2026_exposure) 的完整链接。VWAP 在正文中只有表名，本次通过该表名成功读取官方详情接口，接口路径见 [来源记录](sources.md)。

## 3. 分钟行情字段：赛事页公布的 1 分钟表

页面共列出 **42 个字段**。下表中 `1 … 5` 表示五个独立字段，CSV 已逐个展开。类型是页面给出的 NumPy/Pandas 类型，**不是从实际行情表读取后确认的 dtype**；其他三个频率表也未验证具有相同 schema。

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
| `amount` | `np.float32` | 成交额，单位未在该字段表中明确 |
| `ask_price1` … `ask_price5` | `np.float32` | 卖方 1—5 档报价 |
| `bid_price1` … `bid_price5` | `np.float32` | 买方 1—5 档报价 |
| `ask_volume1` … `ask_volume5` | `np.int32` | 卖方各档委托量 |
| `bid_volume1` … `bid_volume5` | `np.int32` | 买方各档委托量 |
| `ask_num_orders1` … `ask_num_orders5` | `np.int32` | 卖方各档委托笔数 |
| `bid_num_orders1` … `bid_num_orders5` | `np.int32` | 买方各档委托笔数 |

页面示例中股票标识采用类似 `000010.SZ` 的格式，`date` 保留分钟。原文提到可用 `instrument_id` 节省内存，但未解释 ID 的跨表稳定性及映射规则。输出仍应按比赛要求提供 `instrument`。

盘口快照具体取样时刻、高频聚合规则、零值含义、复权方式和量额单位均应结合实际表进一步确认。不能由字段名推断 5/15/30 分钟表的盘口是区间平均或末值。

来源：[比赛数据页“数据格式”“读取示例”](https://bigquant.com/square/competition/45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca?activeKey=data)。

## 4. 股票列表：登录后读取的元数据

`cpt_jyc_2026_instruments` 的 schema 有 4 列，其中 3 列有业务字段说明。

| 字段 | 元数据类型 | 官方字段说明及标记 |
| --- | --- | --- |
| `date` | `timestamp[ns]` | 日期；`primary=true` |
| `instrument` | `string` | 证券代码；`primary=true` |
| `name` | `string` | 证券简称 |
| `__PARTITION__` | `int64` | 系统 schema 中存在，未提供业务说明 |

表说明未明确历史成分有效区间、调整日生效方式、停牌/ST/上市与退市处理，也没有公布每个截面的评估股票集合。实际取样时应使用相应历史时点股票池，不能用未来成分列表回填历史；这是根据禁止未来信息规则提出的实现要求。

来源：[股票列表官方详情](https://jinyicapital.bigquant.com/data/datasources/cpt_jyc_2026_instruments)，本次读取其对应元数据接口，详见 [sources.md](sources.md)。

## 5. 风险暴露：登录后读取的元数据

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

`cpt_jyc_2026_vwap` 的官方中文名称为“未来30分VWAP收益率”，schema 有 9 列。元数据说明如下：

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
3. **尚未确认**：精确代数公式、是否为简单/对数收益、是否再做风险残差化、窗口端点包含规则及无成交处理。本次未读取实际行来验证公式。

因此不应直接把标签写成 `future_vwap/current_price - 1`，也不应未经核实就把 `end_price/vwap - 1` 当作官方确认公式。后者仅是依据字段描述可能作出的解释。

`vwap`、`end_price`、`amount`、`volume`、`num_trades` 描述的都是未来窗口信息，在预测时点不能作为已知输入。训练阶段使用标签，与推理阶段读取未来字段，是不同的数据使用行为。

## 7. 时间对齐与覆盖

| 数据或输出 | 本次查到的时间语义 |
| --- | --- |
| 行情 `date` | Bar 结束时间 |
| VWAP 表 `date` | 预测窗口开始时间 |
| 模型输出 `datetime` | 约定采样时刻，精确到分钟 |
| 风险暴露/股票列表 `date` | 仅描述为日期，可获得时间待确认 |

整理者据此建议：

- 对齐前先确定“信息已可获得”的时点，不能仅因时间字符串相等就拼接所有字段。
- 获取官方采样点列表，并核实午休是否跨越、最后一个窗口如何截断、集合竞价是否排除。
- 保留完整分钟时间，通用示例中的 `date::DATE` 会丢失时分秒。
- 高频序列可向过去扩展以构建输入，但不能读取预测时点之后的 Bar。
- 对归一化统计量和数据填充记录拟合区间及方向，避免使用验证/私榜未来信息。

上述是由字段语义和禁止未来信息规则得到的实现建议，并非官方公布的完整对齐算法。来源：[比赛数据页](https://bigquant.com/square/competition/45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca?activeKey=data)、[比赛规则](https://bigquant.com/square/competition/45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca?activeKey=rule)。

## 8. 读取方式与开发资源

数据页使用 `dai.query(SQL, filters={"date": [start, end]})`，并通过 `.df()` 得到 Pandas DataFrame。下面为按页面接口形式改写的最小读取示例，**未在本次核查中执行**；表可用性和权限需要在获授权的比赛环境验证。

```python
import dai

preview = dai.query(
    """
    SELECT date, instrument, close, volume
    FROM cpt_jyc_2026_stock_bar1m
    ORDER BY date, instrument
    LIMIT 10
    """,
    filters={"date": ["2020-01-02 09:31:00", "2020-01-02 09:35:00"]},
    compression=True,
).df()
```

页面还说明：

- 可通过 `.pl()`、`.arrow()` 转为 Polars 或 Arrow；Polars 并非默认内置依赖。
- `compression=True` 可压缩内存，页面特别提到 `instrument` 转为 category；绑定回 DAI SQL 时可能需转回 string。
- 可以用 `bind_relations` 绑定本地 DataFrame；DAI 支持 UDF。
- `m_` 前缀时序算子及 `c_` 前缀截面算子在页面有示例，但 `c_` 的文字解释存在疑似复制错误，使用前应查函数文档。
- 页面建议 4C/16G 或更高开发规格，参赛宽币可用于升级。GPU 评测规格仍待公告。
- 原文的按日聚合因子、UDF 因子案例仅说明数据引擎能力，不能据此认定可把这些因子作为本赛道输入。

来源：[比赛数据页“读取示例”“因子计算示例”“Tips”“计算资源”](https://bigquant.com/square/competition/45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca?activeKey=data)。扩展阅读：[官方 DAI 使用文档](https://bigquant.com/wiki/doc/PLSbc1SbZX)、[官方 DAI 函数文档](https://bigquant.com/wiki/doc/Rceb2JQBdS)；本次仅记录这些官方参考入口，未逐项验证函数实现。

## 9. 本次未取得的信息

尚未取得实际数据行数和大小、各表精确时间范围、训练/验证/私榜边界、官方采样点清单、行情复权和量额单位、盘口聚合算法、历史股票池变更规则、标签精确生成程序、完整评测代码以及正式端到端模板。具体影响和确认事项见 [open-questions.md](open-questions.md)。
