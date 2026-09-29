# 本地配置、数据申请与访问结果

官方申请流程核查日期：**2026-09-29（UTC+08:00）**；历史 SDK 查询日期：**2026-09-22（UTC+08:00）**；平台下载数据比对日期：**2026-09-30（UTC+08:00）**。[返回目录](../README.md) · [数据说明](data.md) · [来源记录](sources.md)

## 端到端本地训练数据申请

官方[新手入门](https://bigquant.com/wiki/doc/oSGtQiGtq0)要求先向比赛小助手提交本人学生卡照片和平台用户 ID，签署并回传限制协议，审核通过后领取数据下载链接。小助手二维码在该指南中。本地训练数据只限本人或本队参赛使用，不得外传或用于其他用途。

申请自 **2026-09-30** 开放，当天集中处理一天；**10 月 1—7 日暂停处理**，未完成申请从 **10 月 8 日起**继续处理。本次仅核对公开申请说明，未提交身份材料、签署协议或发起申请。

当前[比赛数据页](https://bigquant.com/square/competition/45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca?activeKey=data)列出 `cpt_jyc_2026_e2e_bar1m`、`cpt_jyc_2026_e2e_bar5m`、`cpt_jyc_2026_e2e_bar15m`、`cpt_jyc_2026_e2e_bar30m` 作为本地下载数据，详见 [数据说明](data.md)。该流程提供经审核的下载链接，与下面记录的辅助表 SDK 直连查询权限分别处理；本次未确认 SDK 权限是否已经开通。

## 本地配置

用户授权将 AK/SK 保存到仓库根目录下的 `.bigquant/config.json`，JSON 使用 UTF-8 和两空格缩进，字段为 `ak`、`sk`。实际凭据只保存在本地，本文不包含其内容。密码未写入配置。

[`.gitignore`](../../.gitignore) 中的 `/.bigquant/` 忽略项目配置目录，`/data/` 忽略下载目录。已检查配置文件未被 Git 跟踪，且忽略规则生效。

本次下载环境位于 `.venv/bqsdk-download/`，Python 为 3.11，使用 `bigquant==0.1.11`、`bigquant-core==0.1.14`。版本来自本次查阅的[官方 SDK 使用文档](https://bigquant.com/wiki/doc/vac4qwmQr4)，安装源为 `https://pypi.bigquant.com/simple/`。原有 `.venv/` 中的 `bigquant==6.0.2` 未替换。

本次从项目 JSON 读取凭据后，调用 `bigquant.init(ak=..., sk=...)` 初始化。该版本的 `init_from_config(path)` 对此配置返回 `Config file must contain ak/sk or token`，因此实际使用显式初始化接口；接口定义见[官方 SDK API 手册](https://bigquant.com/wiki/doc/064mz8zFJD)。

## 历史 SDK 查询结果（2026-09-22）

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

当次三张表直连查询均未取得数据，未生成 Parquet 文件；这些历史错误不能用于判断后来平台会话中的读取结果。

## 平台导出与本地验证（2026-09-29—30）

根据 2026-09-22 的查询错误，SDK 直连方式需要先为密钥对应的账号开通 SDK 使用权限。[官方 SDK 使用文档](https://bigquant.com/wiki/doc/vac4qwmQr4)提供了联系入口，本地训练下载则按上述比赛专用流程申请。

用户已在平台读取数据、导出 Parquet 并手动下载。四张五档 `cpt_jyc_2026_stock_bar*` 按月保存于 `data/bigquant-2026-stock-export/`，三张辅助表分别保存于 `data/bigquant-2026-end-to-end-export/` 下的同名 Parquet 文件。

2026-09-30 已完成 243 个文件、723,258,293 行的本地全量检查与用户回传的平台内容摘要比较，七张表全部一致。股票列表覆盖至 2026-09-22，风险暴露覆盖至 2026-09-21，VWAP 覆盖至 2024-12-31；各表实际覆盖范围与缺失值见 [下载数据验证](data-validation.md)，校验基准见 [结构化摘要](data-validation-summary.json)。

本次成功路径为平台会话导出与手动下载，不据此推断本地 SDK 权限已经开通。记录区分历史元数据与 SDK 查询、平台实际数据和本地比对，未运行模型。
