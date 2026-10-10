# v2：四频量价与五档盘口预处理

[文档目录](../../docs/README.md) · [开发与评估协议](../../docs/development.md) · [配置](config.json)

父方案为 `v1`，父运行为 `v1-20261010-s42`。当前已完成训练、验证、测试数据预处理；后续采用共享模型逐股票预测自身。窗口长度、模型结构、模型阶段的时段筛选、缺失填充与跨频率组合尚未确定。

## 数据约定

- 频率：`bar1m`、`bar5m`、`bar15m`、`bar30m`，分别处理，不重采样或切窗。
- 每种频率使用 36 个原始字段：OHLC、`volume`、`amount`，以及五档买卖价格、委托量和委托笔数；具体顺序见配置。
- 保留原始表全部时段。价格使用 `log(x)`，数量、金额和笔数使用 `log1p(x)`；非有限值、非正价格和负数量作为缺失，正常零数量保留。
- 按股票、频率、字段，在完整训练区间拟合均值和总体标准差（`ddof=0`）。每根 Bar 只计入一次，各字段独立忽略缺失；各月统计按有效计数合并。
- 个股归一化结果为 `z = (u - mean_train) / scale_train`。再在同一频率、同一 Bar 结束时刻，对当时股票池中的 `z` 做截面标准化，得到 `c`。
- 股票池采用不晚于该时点的最后一份完整快照；没有已知快照时不向未来查找。池外原始行情行仍保留，股票池仅决定截面统计的参与者。
- 每行顺序为 `date`、`instrument`、36 个 `z_<field>`、36 个 `c_<field>`，共 74 列。72 个数值通道使用 `float32`，缺失保留为 Parquet null，不裁剪或提前填零。

股票、频率、字段的训练期统计量至少需要 30 根有效 Bar，尺度须大于 `1e-8`。不足时依次使用该股票截至当前已结束 Bar 的扩展历史统计量、训练期同频同字段的全体股票统计量；仍不足则保留缺失。全体股票训练字段恒定时使用尺度 1，仍减去训练均值。

截面至少需要两个有效池内股票；数量不足时 `c` 缺失，恒定截面使用尺度 1。缺失与退化尺度不造成原始行丢失。

## 冻结与时间边界

沿用 [local-v1](../../evaluation/protocols/local-v1.json) 的半开区间，均为 UTC+08:00：

| 划分 | 开始（含） | 结束（不含） |
| --- | --- | --- |
| 训练 | 2020-01-01 | 2024-01-01 |
| 验证 | 2024-01-01 | 2024-07-01 |
| 测试 | 2024-07-01 | 2025-01-01 |

验证与测试直接复用冻结的训练参数，校验 SHA-256，不重新拟合长期统计量。缺参股票的动态历史从训练期已有观测初始化，并连续穿过月份和验证／测试边界；续跑恢复已保存状态。每次处理保留完整时间截面，不因 Arrow 分块改变截面统计。

数据处理不读取标签。后续模型切窗只能使用预测时点之前已结束的 Bar；归一化配置、参数、字段顺序和窗口约定须与模型绑定。

## 运行与产物

使用项目本地环境及 [依赖清单](requirements.txt)。本地入口读取 `data/cpt_jyc_2026_stock_bar*/*.parquet`，数据根目录可由 `--data-root` 或 `JYCAPITAL_DATA_DIR` 指定。以下目录名用于新的运行：

```powershell
.venv/Scripts/python.exe solutions/v2/v2_prepare.py --run-dir outputs/preprocessing/v2-train-new
.venv/Scripts/python.exe solutions/v2/v2_apply.py --training-run outputs/preprocessing/v2-train-new --run-dir outputs/preprocessing/v2-holdout-new
```

中断后使用相同命令加 `--resume`。配置、源码或来源数据变化时须新建运行目录；已有产物保留其实际运行时的源码快照和指纹。

| 产物 | 位置与用途 |
| --- | --- |
| 冻结参数 | `normalization/bar*.json`，包含拟合区间、字段顺序、逐股票及全体统计量 |
| 训练序列 | `data/bar*/YYYY-MM.parquet` |
| 验证／测试序列 | `data/validation/bar*/YYYY-MM.parquet`、`data/test/bar*/YYYY-MM.parquet` |
| 配置与追溯 | `config.json`、`protocol.json`、`manifest.json`、`source/`、`data-manifest.json` |
| 本地处理记录 | `records/`、`summary.json`，包含行数、质量计数、动态历史状态和文件哈希 |

行情、处理结果、统计量和详细运行记录位于被 Git 忽略的 `outputs/`。正式数据处理入口分别为 [训练期拟合与转换](v2_prepare.py)、[冻结参数应用](v2_apply.py)，两者共用 [归一化实现](v2_preprocessing.py)。后续平台推理需通过 `datasources` 读取授权数据，不能直接把本地入口当作平台评测入口。

## 已有结果与验证

| 划分 | 行数 | 月度分区 | Parquet 大小（十进制 GB） |
| --- | ---: | ---: | ---: |
| 训练 | 551,954,507 | 192 | 127.30 |
| 验证 | 71,253,965 | 24 | 16.61 |
| 测试 | 75,674,909 | 24 | 17.63 |

训练运行是 `outputs/preprocessing/v2-20261010/`，验证／测试运行是 `outputs/preprocessing/v2-holdout-20261011/`。各分区行数与原始文件一致，验证／测试参数副本与训练期原文件相同。各频率统计及来源哈希见 [训练摘要](../../experiments/v2-preprocessing.json)、[验证／测试摘要](../../experiments/v2-holdout-preprocessing.json)。

独立数据核验已覆盖全部分区元数据，并从原始行情重算训练期 576 个、验证／测试期 192 个完整截面，后者包含新股票回退情形。结果符合 `float32` 舍入精度；详细报告保存在各运行目录的 `verification.json`。这些结论仅涉及本地预处理，不代表模型效果或平台运行已经验证。

回归测试覆盖对数定义域、未来信息隔离、股票池时点、分块一致性、冻结参数、30 根阈值、跨划分历史连续性与续跑：

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p "test_v2*.py" -v
```
