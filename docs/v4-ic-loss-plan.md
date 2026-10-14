# v4 技术方案：完整截面训练与纯 Pearson IC 损失

[文档目录](README.md) · [v3 方案](../solutions/v3/README.md) · [v3 冻结结果](../experiments/v3-tcn.json) · [实验工作流](../.agents/skills/jycapital-experiment/SKILL.md)

## 1. 交付范围与版本边界

本文件定义 v4 的实现与验收契约，独立实现位于 [solutions/v4](../solutions/v4/README.md)，实际训练状态和结果以该方案说明及运行记录为准。实施目标是完成“完整截面 batch + 纯 Pearson IC loss + 验证 IC 选模和早停”，进行一次种子为 42 的正式训练，冻结最佳权重后完成验证／测试评估，并生成 v4 方案说明和历史比较。

研究假设：将反向传播目标、验证主指标和早停指标统一为截面 Pearson IC，消除同一组预测上回归损失与 IC 的目标差异。该设计不保证 IC 一定超过历史版本，也不保证优化器每一步都改善指标。

必须保留的父版本证据：

| 项目 | 冻结值 |
| --- | --- |
| 父运行 | `v3-20261014-s42` |
| v3 代码与结果入库提交 | `7b5fb49cf1f78872d3e781a1fa47552abb385598` |
| 原始训练源码快照 | `outputs/runs/v3-20261014-s42/source/`；与上述整理提交分别记录 |
| 冻结权重 | `outputs/runs/v3-20261014-s42/best.pt` |
| 权重 SHA-256 | `517dd470cbc0c706954040e5eb204388ed56e9bf0ba3c92294363b5e90237ad0` |
| 已选轮次／停止轮次 | 9／21，按验证 Rank IC 选择 |
| 验证 IC／Rank IC | `0.044650190204144874`／`0.04440126656626242` |
| 测试 IC／Rank IC | `0.03748916875555763`／`0.042015468025291755` |

实施前后核验 v3 的权重、配置、协议、预测、指标及源码快照，确保没有被覆盖。不得重选 v3 的 epoch，也不得修改旧运行的选模依据。新运行使用尚不存在的 `outputs/runs/v4-<日期>-s42/`；从随机初始化开始，不加载 v3 权重微调。中断恢复只能恢复这次 v4 运行自身的 checkpoint。

`outputs/` 中的权重和详细证据不随 Git 分发。只有源码仓库时，需取得这些现存的冻结产物后再做完整预测核验；不能用重新训练 v3 代替原始基线。缺少产物时明确比较范围，不补造证据。

## 2. 固定部分与实验变量

网络和输入沿用扩容后的 v3，研究变量限定为损失、训练样本组织、选模策略及其必要的记录／协议适配。

| 项目 | v4 设定 |
| --- | --- |
| 输入 | 72 个 z/c 数值、2 个组完整标志、3 个时间通道，共 77 通道 |
| 四频窗口 | 1／5／15／30 分钟分别 60／24／16／16 根 Bar |
| 隐藏通道／融合隐藏层 | 128／256 |
| TCN 残差块 | 1 分钟 4 块，其他频率各 3 块；卷积核 3 |
| 参数量 | 1,591,553 |
| Dropout | 0.1 |
| 数值与掩码 | 保留有效数值，具体无效值补零；组完整标志不乘到整组数据上，不删除不完整样本 |
| 归一化 | 复用冻结 v2 训练统计和处理结果，沿用已有因果窗口与历史回退 |
| 优化器 | AdamW，学习率 `3e-4`，weight decay `1e-4`，梯度裁剪范数 1 |
| 训练上限／早停 | 最多 100 个 epoch；连续 12 个 epoch 未达到改善阈值后早停 |
| 随机种子 | 42；一次正式运行 |
| 网络计算精度 | 沿用 v3 的受支持 CUDA BF16 autocast；评估前向 FP32 |
| 损失统计精度 | autocast 外使用 FP64 完成中心化、范数、相关系数和截面汇总，保留计算图 |

训练仍用 2020—2023 年，验证用 2024 年上半年，测试用下半年，边界与 [local-v1.json](../evaluation/protocols/local-v1.json) 相同。历史输入可跨划分边界，但只能使用预测时刻已知的信息；标签窗口不得跨训练边界或隔夜。

预处理来源为 `outputs/preprocessing/v2-20261010/` 和 `outputs/preprocessing/v2-holdout-20261011/`。修改损失不需要重新拟合统计量或重写这两份数据。`v4_schema.py` 应识别 v4 方案标识，同时保持全部输入布局与组完整性约束。

## 3. 标签与纯 Pearson IC 损失

训练标签使用原始有限的 `vwap_return`，保留数据源的 FP64 数值，不先转成 FP32 再升精度。取消 v3 训练标签的 ±10% 裁剪、Huber 参数及收益缩放训练目标，不进行排名、分位数变换或高斯化。标签在损失内部进行截面中心化／归一化；这些未来收益只用于监督，不进入模型输入或推理入口。

对预测时刻 t，以同一股票池中满足原标签边界规则、且标签有限的股票构成集合 S_t。所有 S_t 样本均参与该截面统计，不能因输入组不完整而排除。

设模型分数为 s，标签为 y，中心化后为 u、v：

\[
u_i=s_i-\bar{s},\qquad v_i=y_i-\bar{y}
\]

\[
IC_t=\frac{\sum_i u_iv_i}{\sqrt{\sum_i u_i^2}\sqrt{\sum_i v_i^2}}
\]

\[
L_t=1-IC_t,\qquad L=\frac{1}{T}\sum_t L_t=1-\operatorname{MeanIC}
\]

对有效且非退化的截面，这是可直接求导的精确目标。也可用双边截面标准化后的 MSE 验证：标准差采用 `ddof=0` 时，`0.5 * mean((z_score - z_label) ** 2)` 与 `1 - IC` 的数值和梯度等价。

建议新增 `v4_loss.py`。下面仅表示已经完成有效性检查的**一个完整截面**的核心计算：

```python
def pearson_ic_loss(scores, labels):
    scores = scores.to(torch.float64)
    labels = labels.to(torch.float64)
    u = scores - scores.mean()
    v = labels - labels.mean()
    ic = ((u / torch.linalg.vector_norm(u)) *
          (v / torch.linalg.vector_norm(v))).sum()
    return 1.0 - ic, ic
```

不得对预测、预测均值或范数调用 `detach()`；不得改成绝对 IC、IC 平方或混合 Huber／MSE。AdamW 的解耦 weight decay 仍可保留，但不将其写入这里的 loss 数值。不得仅把日志改写成 `1 - IC` 而实际继续反向传播旧损失。

### 数值和退化情况

- 与评估器一致，少于 30 个有效标签或标签恒定的截面不构成有效 IC；单独统计原因，不计为 IC=0。
- 预检必须覆盖训练起始边界、全无历史和停牌等情况。对所有有效标签股票的完整模型输入完全相同、模型无法区分的截面，可建立**与权重无关、训练前固定**的结构性退化清单，从训练损失中排除；保留全部预测键，评估仍执行原协议并报告未定义截面。不能根据预测误差或已观察到的 IC 决定清单。
- 非结构性截面的预测方差为零时，不得静默丢弃。先按配置约定重算 FP32 前向以检查 BF16 量化问题，记录回退；仍无方差则将其作为训练异常处理。不得引入随机噪声制造相关性。
- 非零但很小的方差优先通过 FP64 统计和稳定归一化处理。不要任意加较大 epsilon 或用阈值丢弃低方差截面；分母保护若实际生效，必须记录此时不再严格等于原始 Pearson IC。
- NaN／Inf 的模型分数、损失或梯度应使本次运行报错并保存状态，不补零继续训练。
- 验证和测试的常数预测照原评估器报告。选模与候选判定不得通过增加模型导致的未定义截面、减少有效评估截面来获得更高平均 IC。

实现时将损失值、IC、有效截面数、各类退化计数分别返回；没有可用截面的 batch 不执行优化器更新，也不伪造一个零损失。

## 4. 完整截面 batch 与确定性恢复

当前 [v3_train.py](../solutions/v3/v3_train.py) 在月份内随机打乱样本，再按 512 条切分；v4 改为按完整 `date` 时间戳分组，不能只按交易日分组。

默认每个 optimizer step 使用 **1 个完整预测时刻截面**，股票数量由该时刻实际股票池决定，不能固定截取 512 条或假设永远恰好 1,000 条。该时刻全部股票可前向计算，损失只对原标签规则允许的有限标签取值。

月份及月份内时间截面的顺序使用可复现的种子打乱；同一截面内部按 instrument 保持稳定顺序。每轮遍历每个截面一次，不使用 `drop_last` 丢弃尾部截面。若支持一次合并多个截面，必须逐截面求相关系数后等权平均，不能对拼接后的所有股票计算一次全局相关系数。

显存不足时，可使用激活重算或保持计算图的分块前向，最终拼接同一截面的完整分数后计算一次 loss。不能分别计算两个半截面的相关系数再平均，也不能对各块独立 backward 来冒充完整截面梯度。普通分块前向仍保留全部反向激活，不应未经测量就宣称解决显存问题；激活重算须保持 Dropout RNG 一致。

仍复用按月加载与预取的窗口数据。保存 epoch、月份顺序／游标、截面顺序／游标、优化器更新数、模型与优化器状态、CPU／CUDA RNG，以及恢复所需的种子。恢复后不得重跑已经更新过的截面。数据、配置、训练源码变化时拒绝续跑。

完整截面使单步样本数通常约为旧 512 batch 的两倍，单轮更新步数相应减少；实际值按数据统计。这是本方案的实验变量，需在资源报告中披露，不能宣称仅改变了损失。

## 5. 新协议与历史兼容

新增 `evaluation/protocols/local-v2-ic.json`。从现有 `local-v1.json` 复制，只改以下两个字段：

```json
{
  "protocol_id": "local-v2-ic",
  "selection_metric": "ic_mean"
}
```

以上是字段差异，不是完整协议文件。日期划分、股票池、标签、标签边界、IC／Rank IC 算法、截面权重和覆盖规则均沿用原值。保留原 `local-v1.json` 和全部历史运行的 `protocol.json`。

当前有两处阻碍直接复用：

| 位置 | 现有行为 | v4 必须完成的适配 |
| --- | --- | --- |
| `v3_train.py:data_manifest()` 的 v4 对应实现 | 要求预处理协议与模型协议字典完全相等 | 允许经过明确检查的选模字段差异，继续验证所有数据文件、统计量及原协议指纹 |
| [compare.py](../scripts/compare.py) | 协议文件哈希不同就拒绝比较 | 新增显式的“仅选模协议不同”兼容路径，并记录实际差异 |

兼容检查采用严格白名单：只有 `protocol_id` 和 `selection_metric` 可以不同；移除这两项后，完整协议对象必须相同。其他任何差异都拒绝使用此兼容路径。比较还必须核对评估器源码指纹、标签／股票池数据指纹、样本键、有效标签数和逐截面覆盖。

建议将兼容逻辑放到 `common/protocol_compatibility.py`，由训练数据验证和比较工具共同调用；默认比较仍严格拒绝不同协议，仅在调用者显式启用时允许上述差异。报告同时保存两个原始协议哈希和差异字段，不修改旧元数据、不伪造一致哈希。

本次保持 [baseline_evaluation.py](../evaluation/baseline_evaluation.py) 的指标算法和源码不变；新损失放在 v4 自有模块中。历史文件已经包含 IC，无需为了改变主指标重训或重选历史权重。如果后续发现必须修改评估算法，应另建评估版本并基于冻结预测／权重重评到新目录，不能覆盖已有评估结果。

预处理仍可按原数据协议复现，v4 训练使用新选模协议；复现脚本必须正确传递两者并执行兼容检查。

## 6. 配置、选模和日志

在复制并隔离 v3 方案后，v4 的训练配置应表达以下设定；其他数据与网络配置完整保留：

```json
{
  "solution_id": "v4",
  "parent_run_id": "v3-20261014-s42",
  "seed": 42,
  "training": {
    "batching": "full_cross_section",
    "sections_per_step": 1,
    "loss": "pearson_ic",
    "loss_reduction": "equal_cross_sections",
    "loss_precision": "float64",
    "target_transform": "identity",
    "epochs": 100,
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "gradient_clip": 1.0,
    "patience": 12,
    "min_delta": 0.00001,
    "cpu_threads": 4,
    "precision": "bfloat16_autocast_on_supported_cuda",
    "deterministic": true
  },
  "inference": {
    "batch_size": 1024,
    "precision": "float32",
    "factor_scale": 100.0
  }
}
```

此片段不是完整可运行配置。移除旧的训练 `batch_size=512`、`huber_delta`、`target_clip`、`target_scale`，使其不能被误读为有效训练参数；截面数量与前向分块大小使用不同字段。

为保持固定输出尺度，v4 的训练评估入口与正式推理入口统一输出 `factor = raw_score / inference.factor_scale`，默认仍除以 100。该正比例变换不影响 IC，仅是输出约定，不再表示模型在拟合百分数收益。原始收益尺度的 MSE／MAE继续保存，但因纯 IC 训练不校准均值和幅度，不能把这些数值变化解释为同等意义的收益回归改进。不得在测试集上拟合额外校准器。

选择指标唯一来源是新协议的 `selection_metric=ic_mean`；不要再在其他代码里写死 Rank IC。每个 epoch 后执行完整验证：

1. 固定模型，关闭 Dropout，使用 FP32 推理获得同一份预测。
2. 计算验证 IC、Rank IC、IR、MSE／MAE、覆盖、有效截面数及未定义截面数。
3. 对相同有效截面计算 `validation_ic_loss`，核验其与 `1 - validation_ic_mean` 的误差不超过 `1e-8`；输出检查或非结构性常数预测检查失败的 epoch 不得晋级。
4. 严格最高的验证 IC 保存为 `best.pt`；相同分数保留更早的权重。
5. 只有相对 `patience_best` 的改善大于 `1e-5` 才重置早停计数，否则增加一次；连续 12 轮未达到改善阈值或完成 100 轮后停止。
6. 冻结权重及 SHA-256 后，再读取测试标签做正式测试评估。测试结果不参与选择 epoch、调整参数或增加种子。

保留 `train_online_ic_loss` 与 `train_online_ic`：它们是这一轮在各截面更新前计算的等权均值，应满足互补关系。它们不是轮末固定权重在训练集上的完整重评结果。验证曲线必须来自同一份轮末预测，使 `validation_ic_loss + validation_ic_mean = 1`，无需额外全量训练集重评来制造形式上的一致。

新日志应分别保存实际的 `validation_ic_mean` 和 `validation_rank_ic_mean`。现有 v3 日志将主指标写进 `rank_ic_mean`，`selection.json` 和完成事件也有 `best_validation_rank_ic`／`validation_rank_ic` 等固定名称，复制到 v4 时必须全部适配。建议使用 `selection_metric`、`best_validation_metric`、`best_epoch` 等通用选择字段，同时保留两个指标的真实值。

记录每轮截面数、有效样本数、退化原因计数、优化器步数、停止原因和耗时；保留 `seconds` 等资源统计依赖的字段。详细记录在 `outputs/`，入库文档只保留方案、结论、必要验证与局限。

## 7. 实现文件与依赖闭包

| 文件／模块 | 工作内容 |
| --- | --- |
| `solutions/v4/v4_schema.py`、`v4_model.py` | 独立 v4 命名空间与方案校验；保持 77 通道与 1,591,553 个参数 |
| `solutions/v4/v4_data.py` | 完整时间截面索引、确定性采样和有效标签掩码；沿用原因果切窗 |
| `solutions/v4/v4_loss.py` | 可求导的纯 Pearson IC、数值稳定及退化分类 |
| `solutions/v4/v4_train.py` | 截面训练、IC 选模早停、正确日志、可恢复状态、冻结后测试 |
| `solutions/v4/v4_predict.py` | 新配置和权重读取、统一 factor_scale、无标签依赖的推理 |
| `solutions/v4/v4_preprocessing.py`、`v4_prepare.py`、`v4_apply.py` | 独立复现入口；保持现有数据变换与统计格式的兼容性 |
| `solutions/v4/v4_text_artifact.py`、`v4_notebook.py`、`v4_reproduce.py` | 新版本标识、运行模块、显式协议路径和完整复现链；不得误读 v3 权重 |
| `solutions/v4/model-config.json`、`config.json`、依赖清单、`README.md` | 完整生效配置和最终 v4 方案说明 |
| `evaluation/protocols/local-v2-ic.json` | 新选模协议，保留原协议 |
| `common/protocol_compatibility.py`、`scripts/compare.py` | 显式的选模协议兼容检查，保留严格默认行为 |
| `scripts/assess.py` | 按当前协议的 IC 选参照、晋级；正确统计 IC 与 Rank IC 的月度改善，记录常数截面限制 |
| `tests/test_v4_loss.py`、`test_v4_data.py`、`test_v4_train.py`、`test_v4_predict.py` | 对应的数学、样本组织、选择恢复和推理回归验证 |

运行时不得从可变的 `solutions/v3/` 导入模型或数据模块。快照调用必须捕获 `solution_id="v4"`、所有实际依赖以及新协议；使用 `common/artifacts.py` 保存 Git 来源与实际源码哈希，保留 `git_dirty` 的真实状态。

`scripts/assess.py` 虽然部分逻辑已读取协议主指标，仍有月度统计的 Rank IC 固定命名。历史 v1—v3 数据和默认比较行为必须保持兼容。CSV 维持原有列顺序和 UTF-8 BOM，新增 v4 行时不重写历史浮点文本或旧 run。

## 8. 训练前必须通过的验收

| 验收 | 要证明的行为 |
| --- | --- |
| 损失与评估器一致 | 同一预测及标签，逐截面 IC 与原评估器一致，平均 loss 与 `1 - mean_ic` 误差不超过 `1e-8` |
| 数学及梯度 | FP64 有限差分检查；与双边标准化 MSE／2 的值和梯度一致；正相关与负相关的符号正确 |
| 不变性 | 正比例缩放、平移、股票顺序置换不改变 IC loss；缩放不能靠 epsilon 获得虚假改善 |
| 截面权重 | 不同股票数量的截面仍等权；多个截面平均不等于全样本相关，测试须能区分错误实现 |
| 有效样本 | NaN 标签、边界标签、少于 30 个样本、常数标签及预测、近常数预测均按明确规则处理并报告 |
| 分组完整性 | 无混合时间、无遗漏／重复／尾部丢弃，完整截面与原股票池键一致；分块前向后拼接仍得到完整损失 |
| 因果与掩码 | 沿用 v3 的未来数据隔离、窗口缺失、两个组标志、部分有效 Bar 和全部缺失输入检查 |
| 选模和早停 | 用可控指标序列证明按 IC 选择；Rank IC 反向变化不影响选模；连续 12 个 epoch 的计数及 min_delta 正确 |
| 恢复 | 中断恢复后截面顺序、Dropout RNG、优化器更新和结果与连续执行一致，不重复更新 |
| 协议兼容 | 仅协议 ID／选模指标不同可通过显式兼容；日期、标签、股票池、算法、聚合或数据指纹改变必须拒绝 |
| 推理一致性 | 离线处理数据与原始数据推理在代表区间一致；整段与拆分请求一致；无未来标签查询 |
| 资源与产物保护 | 最大实际截面的 GPU 前向／反向可运行；确认 v3 冻结产物及历史索引未改动 |

损失测试须调用真正参与反向传播的函数，对照既有评估器，不能只测试日志赋值恒等式。常数预测等例外单独断言状态，不能用 `nanmean` 隐藏失败。

## 9. 正式运行、监控与比较

使用项目已有 Python 和 CUDA 环境，先核实依赖、GPU 和显存，再运行一个正式 v4 实验。以下接口已在独立 v4 实现中提供；每次新实验必须使用新的运行目录：

```powershell
.venv-training/Scripts/python.exe -u solutions/v4/v4_train.py --config solutions/v4/model-config.json --protocol evaluation/protocols/local-v2-ic.json --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v4-YYYYMMDD-s42 --training-confirmed
```

将命令中的 `YYYYMMDD` 替换为实际运行日期，确认新目录不存在后再启动。按 [后台监控约定](../.agents/skills/jycapital-experiment/references/monitoring.md) 设置独立进程和通知发送器，验证主对话空闲后能自动开始新回合并写入事件确认。正常只在每小时、询问、阶段结束或异常时唤醒。不得依赖当前对话中的临时工具单元、既有测试 PID 或忽略目录中的一次性脚本来保证长期通知；需要的新通知组件应形成可复用实现并重新验收当前发送路径。

完成后以**验证 IC**选取可比历史最优，排除新运行本身。当前参照如下，启动正式训练前仍须从完整索引核对：

| 角色 | 运行 | 验证 IC | 测试 IC |
| --- | --- | ---: | ---: |
| 本地验证 IC 最优 | v1-20261010-s42 | 0.07044462729051033 | 0.08354889956132129 |
| 父版本 | v3-20261014-s42 | 0.044650190204144874 | 0.03748916875555763 |
| 历史补充参照 | v2-20261011-s42 | 0.03450688496488348 | 0.03229787109940824 |
| 平台最优 | 暂无可信平台成绩 | — | — |

比较对象为这些已经冻结的模型。旧模型按原 Rank IC 策略选择，v4 按 IC 选择；必须披露策略差异，不追溯性地为旧模型更换 checkpoint。报告验证／测试 IC、Rank IC、两种 IR、覆盖率、有效样本、未定义截面、逐月差值、参数和资源成本，以及输出尺度下的 MSE／MAE及其解释限制。

本方案的改善判据是验证 IC 严格超过可比历史最优，完整股票池及逐截面覆盖不下降，不增加非结构性未定义 IC 截面，并且实现和输出检查全部通过。只超过 v3 不等于超过本地最优。测试和辅助指标退化必须披露，无改善也完成测试和结果归档；不自动增加其他种子、再试一轮参数或重新训练 v3。

## 10. 最终交付与执行提示

实施模型应交付：

- 完整 v4 源码、配置、新协议、必要通用工具适配及有意义的回归检查。
- 独立运行目录中的源码快照、数据指纹、环境、训练记录、曲线、`best.pt`／`last.pt`、选择依据、完整验证／测试预测与截面／月度指标。
- `experiments/v4-tcn.json`、新增运行索引行、`solutions/v4/README.md` 与文档索引；旧结果保持原值。
- 对父版本、本地 IC 最优及可信平台最优的比较结论，明确改善与否、输出尺度限制和单种子证据范围。

如果后续执行任务还要求整理平台候选，仅在改善判定通过后按实验工作流适配 v4 文本导出与验证；v2 导出脚本不能直接套用。是否推送该阶段实现遵从执行任务的 Git 授权，比赛平台提交不自动进行。

本文件保留实现和验收契约；已完成的单次正式运行、冻结结果、比较与输出尺度限制见 [最终 v4 方案](../solutions/v4/README.md) 和 [结果摘要](../experiments/v4-tcn.json)。后续开发不得覆盖该运行或 v3 冻结结果。

## 参考依据

- 本仓库 [v3 训练实现](../solutions/v3/v3_train.py)、[评估器](../evaluation/baseline_evaluation.py)、[原协议](../evaluation/protocols/local-v1.json)及 [v3 结果](../experiments/v3-tcn.json)。
- [PyTorch 2.11：Pearson 相关系数](https://docs.pytorch.org/docs/2.11/generated/torch.corrcoef.html)。
- [PyTorch 2.11：余弦相似度及 epsilon 分母保护](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.cosine_similarity.html)。

公式、双边标准化 MSE 恒等式及配置取舍为本方案的数学推导和工程设计；PyTorch 文档用于核对基础算子定义。外部文档查阅日期：2026-10-14（UTC+08:00）。
