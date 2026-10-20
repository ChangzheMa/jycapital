# v401：截面排名监督的 GRU

[文档目录](../../docs/README.md) · [持续研究](../../docs/continuous-research.md) · [候选清单](experiment.json) · [父版本 v301](../v301/README.md)

v40x 开启监督目标系列。以 `v301-global-fp16-s42` 为父版本，从头初始化；保持 v301 网络与 **433,409** 个参数，只将训练目标由标准化收益改为标准化截面平均秩。研究假设是削弱收益幅度极值的影响，检验排序监督能否提高稳定性及平台价值。排名会丢弃收益幅度信息，也可能降低原始收益的 Pearson IC，效果须由评估决定。

## 目标与评估

每个完整预测时间截面中，只对有限收益标签计算平均秩；相同收益使用相同平均秩。随后减去秩均值、除以总体标准差，不加 epsilon。无效位置保留 NaN；标签数不足或全相同的截面按原结构规则跳过。完成截面变换后，才进行全局随机采样，不能在随机样本 batch 内重新排名。

训练使用标准化排名目标上的 Huber delta=1。验证 Huber 采用相同目标，但 checkpoint 仍按**原始收益标签的验证 Pearson IC**选择。验证／测试的 IC、Rank IC、覆盖、股票池和时间划分保持原协议；排名目标的损失数值不与父版本的收益目标损失直接排名。推理只需原始行情，输出模型分数，不依赖当期真实收益或目标排名。

## 缓存与效率

复用 v301 的 FP16 行情输入缓存及评估缓存。基础缓存构建函数保持一致，不为更换目标复制数十 GiB 的行情数组；每次启动仅从原标签构造约 62 MB 的 FP64 目标向量，并替换当前进程使用的目标引用。共享缓存和其中原目标均不改写。

新的目标向量通过日期、时刻及股票索引与缓存逐行对齐，保留原训练时间边界和可用样本集合。运行目录的 `target-overlay.json` 记录基础缓存标识、数据来源及目标指纹；断点恢复时重新构造并核验。该文件用于本地复现与恢复，不参与平台推理。网络、batch 512、优化器、学习率、精度、早停和种子均沿用 v301；主要训练成本预期接近父版本。

## 执行与验证

唯一正式运行为 `v401-rank-target-gru-s42`。候选 ready 后由训练 session 串行执行：

```powershell
.venv-training/Scripts/python.exe -u solutions/v401/v401_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v401-rank-target-gru-s42 --evaluation-cache-mode stream --training-confirmed
```

CPU 检查覆盖平均秩并列、单调变换不变性、无效和常数标签、原收益 IC、缓存行顺序、时间边界、目标指纹恢复、共享数组不变、未来信息隔离及确定性训练恢复；完整入口还覆盖全局缓存模式。正式 GPU 默认批量检查由训练 session 在模型间隙执行。当前无正式行情成绩，不能把实现检查作为模型改善证据。
