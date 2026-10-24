# 原始数据重新下载与后续训练切换

[文档目录](README.md) · [本地数据使用](bigquant-2026-end-to-end/local-download.md) · [训练session提示词](training-session-prompt.md)

用户于2026-10-24要求重新下载可能更新的原始训练数据，后续训练使用更新数据，已完成模型不重跑；用户确认在**BigQuant Notebook**执行下载。新快照已交付至`data/redownload/jycapital-raw-20261024/`。v2901已沿原数据完成，v3001已按新原始目录和独立预处理目录启动。新数据只用于此前尚未启动的训练，已开始运行及恢复均保留各自冻结来源。

**当前数据准备结果：**2026-10-24 16:08（UTC+08:00）独立训练及留出预处理完成，[验证记录](../experiments/data-preprocessing-20261024.json)核对180个月度Parquet的schema、行数、大小和划分时间范围，并确认三频留出统计与训练统计逐字节一致。共149,814,943,197字节，约139.53GiB；旧目录未修改。v3001新路径解析已通过，v2901重新绑定被明确拒绝。训练session于16:37启用新绑定，v3001于16:38启动；启动参数及运行数据清单均已核对为新路径。完整来源清单验收尚未确认通过，当前`controller_activation_verified`仍为`false`，不将路径核对等同于完整验收。

## 下载入口

将[独立Notebook](../examples/redownload-raw-data.ipynb)上传至有赛事数据权限的BigQuant环境，按顺序运行。Notebook已内嵌全部实现，无需上传本仓库的其他文件或本地`dai.py`。对应可维护脚本为[redownload-raw-data.py](../scripts/redownload-raw-data.py)。

默认设置：

```python
OUTPUT_DIR = Path("jycapital-raw-20261024")
START_DATE = "2020-01-01"
STOP_DATE = "2025-01-01"  # 不包含：覆盖2020—2024完整年度
DOWNLOAD_FREQUENCIES = ("bar1m", "bar5m", "bar15m", "bar30m")
RESUME = False
```

1. 首次运行保持`RESUME=False`，输出目录必须不存在。重新获取另一份云端数据应换新目录，不能将续传当作刷新已有文件。
2. 中断后使用同一目录、同一日期和频率，将`RESUME=True`后重跑。已完成文件校验后跳过，未完成分片重做；发布文件与总清单更新之间中断也能恢复。
3. 行情按日读取、按月写Parquet；股票池和标签按月读取、各写一个Parquet。四频默认共242个Parquet文件，另有下载记录与总清单。当前模型只需要1／5／30分钟；如只下载训练所需频率，可在首次运行前移除`bar15m`。
4. 完成后从平台文件浏览器下载新目录。Notebook另有可选分卷单元格：将`MAKE_ARCHIVES=True`，按已有Parquet文件分组生成约4GiB一卷的ZIP并显示下载链接。单个文件不拆分，单卷可能超过目标大小；全部卷解压至同一位置，得到完整目录。Parquet已压缩，ZIP使用存储模式；打包会额外占用一份空间，默认关闭。

下载六张表：四张`cpt_jyc_2026_stock_bar*`、`cpt_jyc_2026_instruments`、`cpt_jyc_2026_vwap`。使用五档stock口径，保留原始数值、类型和缺失值，只去掉`__PARTITION__`系统列；不除以100、不映射股票代码、不重算`vwap_return`，不访问exposure。下载的VWAP辅助字段仅作原始资料保存，不因此成为允许的模型输入。

下载入口验证每次查询区间、键唯一性、实际行数与源表日期计数；保留schema、字节数、文件SHA和获取时间。完整目录以`download-manifest.json`中的`status="complete"`为准。查询分批进行，未取得DAI全表统一快照令牌，因此不能宣称长时间下载具有云端事务快照一致性。

原文件及缓存不被覆盖或删除。下载到本地建议使用独立目录，例如`outputs/raw-data/jycapital-raw-20261024/`，也可以放到空间更充足的其他磁盘。不要直接覆盖现有`data/`；旧运行仍绑定其原始数据与缓存。

## 新旧内容比较

新目录到位后，用[比较入口](../scripts/compare-raw-snapshots.py)核验实际变化，例如：

```powershell
.venv-training/Scripts/python.exe scripts/compare-raw-snapshots.py --new-root outputs/raw-data/jycapital-raw-20261024 --output outputs/data-refresh/20261024/comparison.json
```

默认旧目录为当前仓库`data/`，可以用`--old-root`指定其他旧快照；中断后追加`--resume`。源清单或实现发生变化时使用新报告路径，不混合不同来源的比较记录。本次已完成真实全量比较，范围、修订数量及局限见[修订报告](raw-data-revision-20261024.md)。

入口先核对新文件与下载清单，再按月组织、按日对齐`date、instrument`，精确比较同类型业务字段。忽略行序、列序、Parquet压缩及元数据差异；字符串字典编码不当作业务类型变化，NaN位模式及正负零按逻辑值一致处理，NULL与NaN分别比较。新增／删除键、字段增删和真实类型变化单独记录，不靠隐式类型转换掩盖变化；不输出原始行值。

报告按原训练／验证／测试区间区分：基础行情输入变化、成交笔数／前收盘价／复权因子变化、原始`vwap_return`或标签键变化、股票池成员键变化。VWAP表中其他未来辅助字段变化不会直接标成主标签变化，股票名称变化也不等同于成员变化。报告的变化标志仅提示需检查的组件，不能据此跳过训练入口的缓存身份校验。

默认单线程，DuckDB工作内存限额512MB、临时文件限额1GB，可通过`--memory-limit`和`--max-temp-size`调整；这不是整个Python进程的峰值保证。比较过程不加载全市场输入数组，不启动训练，不改写原始数据、预处理或模型。`covers_required_protocol`必须为真，才能将报告用于整个固定协议的数据接入；小范围比较不能代替完整检查。

[六项比较检查](../tests/test_raw_snapshot_comparison.py)覆盖不同编码／行列顺序的相同内容、细小数值变化、增删键、NULL／NaN／正负零、类型变化、异常键、标签辅助字段区分、完整CLI续传和文件保持。这些合成检查验证比较实现；真实数据的差异结论另以完整比较记录为依据。

## 训练交接规则

`experiments/research-policy.json`的`data_refresh`记录本次用户要求。等待新数据期间，[队列CLI](../scripts/research-queue.py)的`next`返回空，并在`status`提供`data_refresh_gate`原因；已完成记录、在训进程、恢复状态和模型源码均保持原样。当前控制器通过该CLI取得下一候选，本次没有停止或修改训练进程。

控制器还会从`status.candidates`回退选择任务，因此未启动ready候选的CLI视图也标记为`waiting_for_data`，并保留`source_candidate_status="ready"`。这只影响新任务发放，不修改`experiment.json`中的正式发布状态；在训、完成及原运行恢复视图保持。解除数据等待后视图自动恢复。

新文件到本地后，由训练session按以下顺序接入：

- 核对下载清单完整性和实际文件，先比较新旧内容。Parquet字节变化可能只是压缩或元数据不同，不能单凭文件SHA不同声称业务内容修订；重点明确行情、标签、股票池分别是否变化。
- 已完成模型保留原权重与原结果，不重训；已经开始的运行沿其冻结数据完成，不在中途换数据。尚未开始的候选使用新数据后才启动。
- 根据实际变化准备预处理与目标缓存。行情或拟合统计变化时，新建训练和留出预处理目录，统计量只在2020—2023训练范围拟合；不覆盖旧预处理或绕过输入指纹校验。不要盲目复制所有旧缓存，先核对磁盘和可复用部分。
- 调度启动参数必须显式绑定新的`--data-root`、`--training-run`、`--holdout-run`，资源预检也读取新目录。不能仅修改本地DAI的环境变量，因为训练入口使用显式路径。新运行的数据清单保留新来源与获取时间。
- 完成预处理并确认控制器实际使用新路径后，训练session在`data_refresh`记录实际路径、清单指纹和绑定证据，再将`block_new_training`改为`false`并更新状态。不要只解除入口等待而继续沿用旧控制器路径。

已确认行情和标签都有内容修订，见[全量结果](raw-data-revision-20261024.md)。直接命令模板如下；本次使用后面的冻结执行入口完成相同数据处理阶段：

```powershell
.venv-training/Scripts/python.exe solutions/v3001/v3001_prepare.py --run-dir outputs/preprocessing/data-20261024/train --data-root <新原始数据目录> --protocol evaluation/protocols/local-v2-ic.json
.venv-training/Scripts/python.exe solutions/v3001/v3001_apply.py --training-run outputs/preprocessing/data-20261024/train --run-dir outputs/preprocessing/data-20261024/holdout --data-root <新原始数据目录>
```

实际接入优先使用[冻结执行入口](../scripts/prepare-refreshed-data.py)，将预处理源码复制到独立执行目录，避免研究工作区后续改动影响数据准备的续跑：

```powershell
.venv-training/Scripts/python.exe scripts/prepare-refreshed-data.py --new-root data/redownload/jycapital-raw-20261024 --comparison outputs/data-refresh/20261024/comparison.json --workspace outputs/data-refresh/20261024 --preprocessing-root outputs/preprocessing/data-20261024
```

该入口要求完整比较通过，随后串行执行训练统计拟合、训练数据转换及留出集转换，**不调用模型训练**。默认2线程、1GB DuckDB工作内存和32,768行处理批量，只调整执行资源，保持原始数据处理定义。旧数据、旧预处理和现有模型运行均不改写；中断时用相同参数追加`--resume`。归一化只在固定训练区间拟合，留出集复用其逐字节相同的统计量。

完成后生成`outputs/data-refresh/20261024/data-binding.json`，其中包含新原始目录、训练及留出预处理目录、源码和数据证据对应的指纹。状态`prepared_for_scheduler`只表示数据准备完成，**不表示控制器已切换**；仍由训练session验证未来启动及资源预检使用这些路径后解除队列等待。已有运行及恢复继续使用其原命令和原数据。

调度器可调用[新运行路径解析器](../scripts/training-data-paths.py)：

```powershell
.venv-training/Scripts/python.exe scripts/training-data-paths.py --solution-id v3001
```

它只接受尚无运行目录、也没有正式启动记录的ready候选，核对新数据清单、比较证据、预处理配置和摘要，返回显式训练参数以及同评估数据组的历史run列表。已启动v2901会被此入口拒绝，应继续使用自己的原恢复命令。新运行省略旧全局`--resident-cache`参数，由对应加载器生成新缓存键；控制器的资源估算和结果核验也需使用新路径及评估组。该解析器不启动训练、不改变队列等待状态。

具体交接保存在`outputs/run-control/continuous-research/data-refresh-handoff.json`，v3001已核对的启动数据参数在`outputs/data-refresh/20261024/v3001-launch-data.json`。调度session应在新运行控制目录保存所选绑定的冻结副本；已完成或已经启动的运行不套用这份新绑定。新数据的常驻／评估数值缓存尚未提前构建，由下一次串行新训练的加载器按新身份创建，避免与旧运行竞争全量常驻内存。

数据准备阶段产生的`data-binding.json`保持冻结。调度器的实际切换记录另存，并更新`research-policy.json`的激活状态，不覆盖已归档的准备证据；准备记录中的`controller_activation_verified=false`描述的是当时尚未验证调度切换的状态。

验证／测试日期划分不随下载改变。若标签或股票池的实际内容更新，新旧指标需按数据快照分组，不能把跨快照差值直接解释为模型改进，也不能为比较而自动重跑已完成模型；数据更新与网络改动分别记录。

## 已验证范围与来源

[七项合成检查](../tests/test_raw_download.py)覆盖月末日内行、Float32／Float64／大整数／null原值保留、原标签、分卷恢复、下载中断续传、损坏文件拒绝、源计数差异、独立Notebook完整执行、新任务出口及控制器候选列表回退。用户随后在平台完成实际下载，交付文件的全量核验结果见[修订报告](raw-data-revision-20261024.md)。

接口依据为[官方DAI文档](https://fudan.bigquant.com/wiki/doc/PLSbc1SbZX)中的`query(..., filters=..., compression=False).arrow()`，2026-10-24（UTC+08:00）核验；原BigQuant文档入口当次访问超时，使用同平台文档镜像。stock/e2e差异沿用[已有官方来源整理](bigquant-2026-end-to-end/data.md#23-本地训练下载数据)，本轮不更新104条历史字段字典或宣称获得新的源表schema。
