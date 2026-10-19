# 三频全量 FP16 缓存与全局随机训练

[文档目录](README.md) · [研究家族](research-branches.md) · [开发约定](development.md)

v101、v201、v301 的当前默认配置使用全量 FP16 常驻训练数据和全局样本 shuffle。模型主干、标签定义、Huber、batch 512、验证 IC 选模及动态早停保持原方案；采样顺序与输入舍入是本次训练方案变量。原月内 FP32 模式保存在各目录的 `model-config-monthly.json`，可用于新的对照运行。

本次交付实现与合成验证，尚未构建正式行情全量缓存，也未启动本次全局模式的正式训练。原月内模式已有运行的状态和成绩以冻结记录为准；不能把这些结果当成本模式的结果。

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

验证和测试按最多 5 个交易日一块加载，保留各频率所需的历史回看，并仅预取下一块；不同时构建完整留出集常驻副本。指标仍由完整预测键按原协议聚合，不按数据块改变截面或月度评估口径。

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

## 验证范围与性能边界

[全量缓存测试](../tests/test_resident_data.py)覆盖三版模型的原始输入／缓存输入一致性、八个预测截点、长假和股票池变化、缺失及 FP32 极值恢复、全局索引一次遍历、CPU／CUDA 双缓冲、缓存来源与文件损坏拒绝、资源预算、有限块验证，以及 CPU／CUDA 全局训练的精确中断恢复。另复用[推理测试](../tests/test_research_pipeline.py)验证 checkpoint、独立文本模块和整段／分段请求一致，[入口测试](../tests/test_research_entrypoints.py)检查完整预处理到训练评估链路。

实现检查使用临时合成数据，不写正式运行或成绩索引。正式缓存完整构建耗时、加载耗时、训练与验证端到端速度、内存峰值和 FP16 对模型效果的影响尚待正式验证；不能把数据容量估算解释为已实现的加速倍数。

```powershell
.venv-training/Scripts/python.exe -W ignore::DeprecationWarning -m unittest discover -s tests -p test_resident_data.py -v
```
