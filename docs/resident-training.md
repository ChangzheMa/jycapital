# 三频全量 FP16 缓存与全局随机训练

[文档目录](README.md) · [研究家族](research-branches.md) · [开发约定](development.md)

v101、v201、v301 的当前默认配置使用全量 FP16 常驻训练数据和全局样本 shuffle。模型主干、标签定义、Huber、batch 512、验证 IC 选模及动态早停保持原方案；采样顺序与输入舍入是本次训练方案变量。原月内 FP32 模式保存在各目录的 `model-config-monthly.json`，可用于新的对照运行。

全局模式及原月内模式的实际运行状态、源码和成绩以各 run 的冻结记录为准；不能将月内模式的结果当成从头全局训练的结果。本文说明当前实现与使用约定。

## 数据与内存布局

[resident_data.py](../common/resident_data.py) 从已有三频预处理文件构建缓存，不重新拟合 z/c，也不改变截面统计时使用的股票池。按全部训练样本需要的历史取股票／交易日并集，同一份历史只保存一次，数组采用股票、交易日、日内 Bar、72 个数值通道的布局。零号块统一表示不足历史或缺少该股票的数据。

2026-10-20（UTC+08:00）依据本地股票池索引的只读容量估算如下；没有实际分配对应的大数组。

| 项目 | 规模 |
| --- | ---: |
| 训练期 | 2020—2023 年，970 个交易日 |
| 股票池预测键 | 7,760,000 |
| 1 分钟 FP16 数值 | 31.25 GiB |
| 5 分钟 FP16 数值 | 6.26 GiB |
| 30 分钟 FP16 数值 | 1.06 GiB |
| 三频数值合计 | **38.56 GiB** |
| 数值、计划元数据及一个全局排列，未计极值例外 | 约 **38.84 GiB** |

启动前检查可用物理内存，默认在缓存之外预留 8 GiB，供样本键、训练环境、批次缓冲和验证使用。构建和加载均保留内存预算检查，磁盘不足时也提前报错。预算不是完整运行峰值的保证；默认配置不会在内存不足时自动退回按月训练或静默使用磁盘随机读取。

缓存文件位于被忽略的 `outputs/preprocessing/resident-cache/<cache_key>/`，默认与训练预处理目录同级；也可显式指定路径。构建使用分批读取和磁盘数组，不同时保留完整 FP32 与 FP16 副本。加载直接读入最终内存并同步校验文件哈希，训练期间不再解压或读取行情分区。

缓存键绑定数据清单、协议、输入约定、标签转换和构建器源码。三版使用相同来源与输入约定时可复用同一份磁盘缓存；每个训练进程会加载自己的内存副本，当前实现不提供跨训练进程的共享内存。不要因磁盘缓存共用就按单份内存预算并发启动三版。

## 精度与因果性

`data.input_precision=fp16_with_fp32_overflow`：普通数值先舍入为 FP16；原本有限但超出 FP16 范围的值另存稀疏 FP32 例外，构造模型输入时精确写回。NaN／Inf 的原有无效语义保持；三个有效性通道根据恢复后的 72 个数值及预测截点重新生成。构建记录各频率的有限值数量、FP32 例外数量和最大绝对舍入误差。

验证、测试和原始行情推理共用各版本的 `WindowBlock`，执行相同的 FP16 舍入和 FP32 极值保留，避免训练／推理输入精度不一致。旧 checkpoint 配置没有 `input_precision` 时继续使用原 FP32 路径。

缓存可以保存一个交易日的完整行情，但每个样本仍仅暴露预测截点之前的值。1 分钟未来位置清零，5／30 分钟继续排除今天。历史不足、长假、个股缺 Bar、部分字段缺失和合法零值均保持既有含义；不通过删除缺失样本降低覆盖要求。

标签继续在原始完整时间截面内构造 FP64 z-score，再筛选合格标签形成全局样本索引。label 不量化，随机 batch 不重新标准化。FP16 舍入及全局采样可能改变优化结果，不能据合成一致性测试宣称对 IC 没有影响。

## 全局采样、双缓冲与恢复

每轮使用种子和 epoch 对全部合格样本进行一次随机排列，随后依次取 batch；一个 batch 可以跨月份。每个合格样本每轮一次，保留尾批。只打乱索引，行情数组保持适合切窗的排列。输入、预测时钟和 label 始终由相同样本索引读取。

一个后台线程填充两个复用的 CPU 缓冲区；CUDA 环境下，这两个缓冲区使用 pinned FP16 存储。独立 CUDA stream 提前搬运下一批，在 GPU 恢复 FP32、修复极值并生成 75 通道输入。事件保证 CPU 不覆盖仍被传输使用的缓冲区，输出张量登记计算流的生命周期。CPU 环境使用同一数值准备逻辑，无 CUDA stream。

主干继续采用既有 BF16 autocast；v301 的 GRU 状态递推继续 FP32。验证、测试和标准推理的模型计算保持 FP32。全量数据不整体 pin 到内存，也不交给多个 DataLoader worker 隐式复制。

checkpoint 保存模型、优化器、CPU／CUDA 随机状态、epoch、全局样本游标和动态早停计数。预取本身不消耗训练随机状态；恢复时从已提交的样本游标重新生成剩余批次。`progress.json` 提供轻量进度，`resident-cache.json` 记录缓存来源、占用及本次构建／加载耗时。

验证和测试保留最多 5 个交易日一块的顺序及尾批。v101 沿用原网格加载；v201、v301 采用下面的评估缓存。指标仍由完整预测键按原协议聚合，不按数据块改变截面或月度评估口径。

## v201／v301 验证与测试加速

[evaluation_cache.py](../common/evaluation_cache.py)将评估所需行情也按股票／交易日去重，保存 FP16 数值和 FP32 极值例外。历史可以来自上一划分，但每个样本仍仅使用其预测截点以前的数据；原始 FP64 标签、标签边界及全部股票池预测键保持不变，包括无有效标签的预测键。

首次验证时构建缓存，后续 epoch 和相同来源的另一模型复用磁盘缓存。路径为 `outputs/preprocessing/evaluation-cache/<cache_key>/`，键绑定数据清单、协议、划分、输入约定及实现；完整缓存通过校验后才加载。首次构建及每次进程启动的缓存校验／加载有一次性成本，日志分别记录 `evaluation_cache_ready` 和 `cached_evaluation_finished`。

`--evaluation-cache-mode auto` 为默认值：在训练数据已常驻的情况下，按当前可用物理内存、既定预留空间及双缓冲占用，选择评估数值全量常驻或按块加载。可显式选择 `resident` 或 `stream`；强制常驻仍检查预算，内存不足直接报错。分块模式每次只读当前／下一块需要的数据到 RAM，batch 不直接对磁盘做随机切窗；不会每轮重新读取 Parquet、重建 FP32 网格和逐 batch 舍入。

验证 batch 复用训练的 pinned FP16 双缓冲和独立 CUDA 传输流，在 GPU 恢复 FP32、有效性通道和因果遮罩。保持原分块边界、batch 1024、尾批、连续 NCL 布局和 FP32 模型计算；没有改变验证频率、样本、IC 选模或动态早停，也没有启用 BF16 验证。验证和测试只同时保留一个划分的常驻数据。

完整验证期数值缓存的只读估算约 4.73 GiB，包含所需的跨划分历史，不包含标签键、索引、例外值和缓冲。自动模式可能选择分块，这仍使用相同的快速批次路径；不要求把全部训练和验证数据强行同时放入内存。

### 已有全局 run 的恢复

新建 v201／v301 全局 run 自动使用优化。旧进程已经导入旧函数，修改磁盘源码不会让它自动切换。旧 run 必须先停止其训练进程，再在原启动参数基础上使用 `--resume --upgrade-evaluation`，例如：

```powershell
.venv-training/Scripts/python.exe -u solutions/v201/v201_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v201-global-fp16-s42 --resume --upgrade-evaluation --training-confirmed
```

若原命令指定配置、协议或训练缓存路径，恢复时继续使用原参数。v301 已有旧全局 run 时采用同样方式；尚未开始的 v301 用普通训练命令即可。升级后的同一源码再次恢复只需 `--resume`。

[evaluation_upgrade.py](../common/evaluation_upgrade.py)只接受本次登记的 v201／v301 旧训练入口和新入口组合，且其他已有源码必须保持一致。原源码和清单归档至 `source-upgrades/evaluation-cache-v1/`，当前源码另行快照并记录哈希。模型、优化器、随机状态、样本游标、历史最佳权重及早停计数从原 checkpoint 恢复；升级本身不改写这些文件。普通 `--resume` 仍严格拒绝源码变化，不提供通用跳过校验的开关。

从最后保存的 checkpoint 继续；停止前尚未保存的少量 batch 会重新计算。如果在验证中停止，该轮验证会完整重算。已完成 run 不能用此入口重新训练，月内转全局仍属于独立迁移流程。执行重启前还应与管理该 run 的调度／监控进程协调，避免同一 run 被重复启动。

[评估缓存测试](../tests/test_evaluation_cache.py)覆盖 CPU／CUDA 下两种加载模式的新旧输入、预测及指标逐值一致，跨划分历史、全缺失标签、缓存复用与损坏拒绝、内存预算和显式升级恢复。正式训练速度的结论须以实际 run 的完整计时为准。

2026-10-20（UTC+08:00）在 RTX 4080 上，以 2024-01-02 至 2024-01-08 的 5 个交易日、40,000 个真实样本进行单次对照，采用默认模型规模和 batch 1024：

| 模型 | 原验证路径 | 缓存分块模式 | 缓存常驻模式 |
| --- | ---: | ---: | ---: |
| v201 | 47.95 秒 | 9.00 秒 | 8.71 秒 |
| v301 | 41.13 秒 | 4.91 秒 | 6.06 秒 |

两种优化模式的预测和全部指标与原路径逐值一致。v201 使用当时运行的最佳权重，v301 使用固定种子初始化权重，仅核验实现及耗时，不代表训练效果。测速期间另有 v201 正式训练共享资源，不能用这个样本断言常驻／分块哪种恒定更快，也不能直接外推完整验证期的加速倍数。该五日缓存首次构建 19.36 秒，每次进程加载／校验约 0.23—0.26 秒，未包含在表中的优化耗时内；表中分块模式包含取块和构造 batch 的时间。

## 使用入口

训练依赖新增 `psutil`，已列入各版 `requirements-training.txt`。使用现有 `.venv-training`，不因本改造重新安装 PyTorch。

只估算容量，不读取大行情分区、不创建缓存、不训练：

```powershell
.venv-training/Scripts/python.exe scripts/prepare-resident-cache.py --solution v101 --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --estimate-only
```

需要预先构建缓存时，去掉 `--estimate-only`；该命令会校验来源、读取全部所需预处理数据并写入较大的缓存，但仍不运行模型训练。正式训练入口也会在缓存不存在时构建，并在已有完整缓存时复用。失败或来源不匹配的显式缓存目录不会被覆盖，应使用新的目录处理。

未来明确开始正式训练时使用新 run，例如：

```powershell
.venv-training/Scripts/python.exe -u solutions/v101/v101_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v101-global-fp16-s42 --training-confirmed
.venv-training/Scripts/python.exe -u solutions/v201/v201_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v201-global-fp16-s42 --training-confirmed
.venv-training/Scripts/python.exe -u solutions/v301/v301_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v301-global-fp16-s42 --training-confirmed
```

这是三个独立命令，不构成自动并行训练授权。运行及监控继续遵循[实验技能](../.agents/skills/jycapital-experiment/SKILL.md)。显式复用某目录时添加 `--resident-cache <目录>`；缓存来源必须匹配。

月内 FP32 对照增加 `--config solutions/<版本>/model-config-monthly.json`，并使用新的 run 目录。已有月内训练 run 不能通过切换配置直接续跑成全局训练；旧运行如需恢复，应使用其原配置和冻结源码。

## 从旧月内运行显式迁移 v101

[v101 训练器](../solutions/v101/v101_train.py)提供 `--migrate-from`，将已中断的月内 FP32 运行迁移到独立的全局 FP16 run。模型、种子、协议、窗口、标签、损失、AdamW、batch 大小、动态早停和累计 epoch 上限必须保持一致；迁移只改变输入舍入与样本组织，来源 run 保持只读，不作为已完成父运行登记。

```powershell
.venv-training/Scripts/python.exe -u solutions/v101/v101_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v101-global-fp16-migrated-s42 --migrate-from outputs/runs/v101-patch-transformer-s42 --training-confirmed
```

迁移从来源 `last.pt` 继承模型、完整优化器和随机状态。根据旧月份排列、月内排列及游标，按规范 `(date, instrument)` 键重建本轮已处理与剩余集合，验证互斥、完整及样本数／更新数吻合后，仅打乱剩余集合。下一累计 epoch 才使用完整训练期的全局排列；不在新全局排列中简单跳过旧样本数量。

来源 `best.pt` 在新 FP16 输入规则下重新计算验证结果，初始化新 run 的最佳权重和选模基线；这不替代从 `last.pt` 恢复优化。最佳 epoch 和已有无提升计数保留，后续验证沿用统一动态早停，100 轮仍为累计上限。

新 run 保存 `migration.json`、`migration-remaining-indices.npy`、`migration-baseline.json` 及基线完整验证产物，`legacy-history.csv` 保存来源训练记录。迁移轮的日志分别记录旧阶段与新阶段的样本、损失、耗时和更新数；历史曲线与结果需注明月内 FP32 后迁移到全局 FP16，不能描述为从头全局训练。迁移 run 中断后只用其原配置、原源码和 `--resume` 恢复，不重复使用 `--migrate-from`。

## 验证范围与性能边界

[全量缓存测试](../tests/test_resident_data.py)覆盖三版模型的原始输入／缓存输入一致性、八个预测截点、长假和股票池变化、缺失及 FP32 极值恢复、全局索引一次遍历、CPU／CUDA 双缓冲、缓存来源与文件损坏拒绝、资源预算、有限块验证，以及 CPU／CUDA 全局训练的精确中断恢复。另复用[推理测试](../tests/test_research_pipeline.py)验证 checkpoint、独立文本模块和整段／分段请求一致，[入口测试](../tests/test_research_entrypoints.py)检查完整预处理到训练评估链路。

[迁移测试](../tests/test_resident_migration.py)核对旧游标的精确键集合、权重／优化器继承、迁移轮衔接、FP16 验证基线及 CPU／CUDA 中断恢复；来源运行文件哈希保持不变。

合成实现检查不写正式运行或成绩索引；评估缓存另有上述真实五日抽样对照。正式运行的完整缓存构建／加载耗时、训练与验证端到端速度、内存峰值和模型效果以各 run 的记录为准；不能把容量估算或抽样测速解释为完整训练的加速倍数。

```powershell
.venv-training/Scripts/python.exe -W ignore::DeprecationWarning -m unittest discover -s tests -p test_resident_data.py -v
```
