# 本地环境与数据使用

[文档目录](../README.md) · [数据说明](data.md) · [来源](sources.md)

本地查询直接读取已有 Parquet；平台使用真实 DAI。本地兼容层不包含官方评估模块，正式平台评分在平台执行。

## 环境

使用 Python 3.11，本地依赖见 [requirements-local.txt](../../requirements-local.txt)：

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-local.txt
```

Notebook 内核须安装对应依赖。根目录 dai.py 被 Git 忽略，重新克隆时另行保留或配置，不上传平台。Mock 支持从根目录或 examples 启动；本地兼容层只在 dai 无法导入时回退。

## 数据位置

| 表 | 路径 | 字段数 |
| --- | --- | ---: |
| cpt_jyc_2026_stock_bar1m | data/cpt_jyc_2026_stock_bar1m/*.parquet | 42 |
| cpt_jyc_2026_stock_bar5m | data/cpt_jyc_2026_stock_bar5m/*.parquet | 42 |
| cpt_jyc_2026_stock_bar15m | data/cpt_jyc_2026_stock_bar15m/*.parquet | 42 |
| cpt_jyc_2026_stock_bar30m | data/cpt_jyc_2026_stock_bar30m/*.parquet | 42 |
| cpt_jyc_2026_instruments | data/cpt_jyc_2026_instruments.parquet | 3 |
| cpt_jyc_2026_vwap | data/cpt_jyc_2026_vwap.parquet | 8 |

实际文件为原始五档 stock 口径，保留字段类型及空值，不再除以 100，不需股票 ID 映射。数据目录可用 `JYCAPITAL_DATA_DIR` 覆盖。全量来源和质量见 [数据一致性结论](data-validation.md)。

exposure已按用户要求删除并禁用，上述一致性文档中的七表结果仅对应2026-09-30历史快照。2026-10-24重新下载入口及新训练数据切换见[数据更新说明](../data-refresh.md)：使用独立新目录，保留原始`vwap_return`，已完成模型不重跑；训练路径须显式切换，不能只修改DAI环境变量。

## 本地 DAI 支持范围

- DuckDB 单条 SELECT、CTE、聚合、连接，以及 Pandas／Arrow 的 bind_relations。
- query(...).df()、.arrow()；.pl() 需要可选 Polars 依赖。
- filters 的日期范围为闭区间，纯日期表示零点，None 为开放端点，时区端点转换为 UTC+08:00；严格月末边界同时使用 SQL 的半开区间。
- compression=True 只把 instrument 转为 category，不降低数值精度。
- 不支持平台专有 m_*、c_*、PRUNE JOIN、udf_list、DaiUDF；缺表或不支持的参数报错。

## 平台与授权

上传材料见 [平台目录](../../submissions/README.md)，推理接口及数据授权要求见[比赛规则](rules.md)。
