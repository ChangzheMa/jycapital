# 本地运行与数据申请

更新日期：**2026-10-10（UTC+08:00）**。[返回目录](../README.md) · [数据说明](data.md) · [来源](sources.md)

## 运行结论

用户已确认 [mock-submission.ipynb](../../examples/mock-submission.ipynb) 在本地和平台均可正常运行，测试 submission 已通过。两端使用同一 `main(datasources, start_date, end_date)` 入口，返回 `date`、`instrument`、`factor` 三列 DataFrame。

本地查询依赖现有 Parquet，不需要联网、凭据或 SDK 数据权限；平台查询使用真实 DAI。本地兼容层不包含 `bigmodule` 或 `M.jyc_eval`，官方评测在平台执行。

## 本地环境

本地兼容层位于项目根目录 `dai.py`，受 `/dai.py` 忽略规则保护，不随 Git 仓库分发。重新克隆仓库时需另行保留或配置该本地文件。

项目使用 Python 3.11，本地依赖见 [requirements-local.txt](../../requirements-local.txt)，其中 DuckDB 固定为 `1.5.6`。从仓库根目录安装：

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-local.txt
```

Notebook 应使用安装了这些依赖的 Python 内核。其他内核可执行 `%pip install -r E:/src/jycapital/requirements-local.txt`。

Mock 支持从项目根目录或 `examples/` 启动。只有 `dai` 本身无法导入时，Notebook 才向上定位本地兼容层；平台模块的内部依赖错误不会触发本地回退。

## 本地数据位置

| 表 | 路径 | 实际字段数 |
| --- | --- | ---: |
| `cpt_jyc_2026_stock_bar1m` | `data/cpt_jyc_2026_stock_bar1m/*.parquet` | 42 |
| `cpt_jyc_2026_stock_bar5m` | `data/cpt_jyc_2026_stock_bar5m/*.parquet` | 42 |
| `cpt_jyc_2026_stock_bar15m` | `data/cpt_jyc_2026_stock_bar15m/*.parquet` | 42 |
| `cpt_jyc_2026_stock_bar30m` | `data/cpt_jyc_2026_stock_bar30m/*.parquet` | 42 |
| `cpt_jyc_2026_instruments` | `data/cpt_jyc_2026_instruments.parquet` | 3 |
| `cpt_jyc_2026_vwap` | `data/cpt_jyc_2026_vwap.parquet` | 8 |
| `cpt_jyc_2026_exposure` | `data/cpt_jyc_2026_exposure.parquet` | 48 |

这些文件已采用云端 `stock` 和辅助表口径，保留五档盘口、`pre_close`、实际字段类型及原始空值；不再除以 100，不需要股票 ID 映射。目录可通过环境变量 `JYCAPITAL_DATA_DIR` 覆盖。总量与数据质量结论见 [data-validation.md](data-validation.md)。

## 本地 DAI 支持范围

- 支持 DuckDB 的单条 `SELECT`、CTE、聚合及连接，使用平台表名查询本地文件。
- 支持 `query(...).df()`、`.arrow()`；`.pl()` 需要可选依赖 Polars。
- `filters={"date": [lo, hi]}` 为本地闭区间约定。纯日期字符串表示当天零点，`None` 表示开放端点，带时区端点转换为 UTC+08:00。SQL 内的更严格条件继续生效。
- `bind_relations` 支持 Pandas 或 Arrow 对象，日期过滤不修改绑定对象。
- `compression=True` 仅将 DataFrame 的 `instrument` 转为 category，不降低数值精度；`full_db_scan` 作为兼容参数接收。
- `m_*`、`c_*`、`PRUNE JOIN`、`udf_list` 和 `DaiUDF` 不支持；缺表、缺文件和不支持的参数明确报错。

## 平台运行与提交

上传 Mock 时只需 [mock-submission.ipynb](../../examples/mock-submission.ipynb)。**不上传本地 `dai.py`**，避免遮蔽平台真实模块。平台可直接运行：

```python
from bigmodule import M

datasources = {"bar1m": "cpt_jyc_2026_stock_bar1m"}
factor_data = main(datasources, "2022-01-01 00:00:00", "2022-03-01 23:59:59")
result = M.jyc_eval._latest(factor_data=factor_data, show=True)
```

测试 submission 通过验证了当前 Mock 的运行与提交接口。正式模型的训练、权重加载及预测表现需独立验证，材料要求见 [rules.md](rules.md)。

## 数据下载授权

端到端本地训练数据须完成身份核验并签署限制协议，审核通过后由工作人员提供下载链接和密码。下载数据仅限本人或本队参赛使用，不得外传或用于其他用途。

官方安排为 2026-10-08 起继续处理申请。申请方式见[新手入门](https://bigquant.com/wiki/doc/oSGtQiGtq0)，压缩下载版格式见[端到端模型数据下载指南](https://bigquant.com/wiki/doc/cay6LBmvc1)。

凭据及受限材料仅存本地忽略目录。SDK 直连访问权限与当前本地离线查询、平台运行和测试提交的通过结论分别管理。
