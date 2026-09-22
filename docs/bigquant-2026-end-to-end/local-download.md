# 本地配置与辅助表下载记录

核查日期：**2026-09-22（UTC+08:00）**。[返回目录](../README.md) · [数据说明](data.md) · [来源记录](sources.md)

## 本地配置

用户授权将 AK/SK 保存到仓库根目录下的 `.bigquant/config.json`，JSON 使用 UTF-8 和两空格缩进，字段为 `ak`、`sk`。实际凭据只保存在本地，本文不包含其内容。密码未写入配置。

[`.gitignore`](../../.gitignore) 中的 `/.bigquant/` 忽略项目配置目录，`/data/` 忽略下载目录。已检查配置文件未被 Git 跟踪，且忽略规则生效。

本次下载环境位于 `.venv/bqsdk-download/`，Python 为 3.11，使用 `bigquant==0.1.11`、`bigquant-core==0.1.14`。版本来自本次查阅的[官方 SDK 使用文档](https://bigquant.com/wiki/doc/vac4qwmQr4)，安装源为 `https://pypi.bigquant.com/simple/`。原有 `.venv/` 中的 `bigquant==6.0.2` 未替换。

本次从项目 JSON 读取凭据后，调用 `bigquant.init(ak=..., sk=...)` 初始化。该版本的 `init_from_config(path)` 对此配置返回 `Config file must contain ak/sk or token`，因此实际使用显式初始化接口；接口定义见[官方 SDK API 手册](https://bigquant.com/wiki/doc/064mz8zFJD)。

## 实际查询结果

2026-09-22 23:54（UTC+08:00），通过 SDK 对三张辅助表分别执行全表聚合查询，尝试取得 `COUNT(*)`、`MIN(date)` 和 `MAX(date)`，作为完整下载前的核对依据。

| 表名 | SDK 直连结果 | 已保存的数据行 |
| --- | --- | --- |
| `cpt_jyc_2026_instruments` | `FlightUnavailableError`：请先申请 SDK 使用权限 | 0 |
| `cpt_jyc_2026_vwap` | 同上 | 0 |
| `cpt_jyc_2026_exposure` | 同上 | 0 |

补充检查：

- 股票列表的 `DataSource.read_bdb()`，日期过滤为 `2020-01-02`，同样返回 SDK 使用权限错误。
- 使用 `dai.query(..., use_studio=True, space_id="db3bf8c9-0b87-4b89-a17a-d42cb0b7567b")` 在比赛空间尝试读取股票列表的 1 行，返回 `Failed to start AIStudio within 60 seconds`。该失败不能证明比赛空间的数据查询权限不足。
- SDK 的数据源读取、空间参数及默认免费计算规格说明见[官方 DAI 数据管理 API](https://bigquant.com/wiki/doc/obft7eKPjB)，本次访问日期为 2026-09-22（UTC+08:00）。

本地 `data/bigquant-2026-end-to-end/coverage.json` 保存了三张表直连查询的时间和错误结果。它是失败记录，不是成功取得的行数或日期范围。当前没有生成数据 Parquet 文件，也没有下载完成的表。

## 后续条件与保存方式

需要先为这组密钥对应的账号开通 SDK 使用权限。[官方 SDK 使用文档](https://bigquant.com/wiki/doc/vac4qwmQr4) 提供了申请权限的联系入口。本次未代用户联系平台或申请权限。

权限开通后，计划将三张完整辅助表保存到 `data/bigquant-2026-end-to-end/`，使用原表名作为 Parquet 文件名，并核对实际行数、日期范围与字段。当前尚未取得实际数据，不能把比赛页面的 2020—2024 年直接当作三张表各自已经验证的覆盖范围。

本记录区分此前成功的元数据核查、本次失败的实际查询，以及尚未执行的下载和模型运行；未改变结构化字段字典中的元数据记录。
