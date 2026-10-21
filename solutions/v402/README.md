# v402：收益与排名的混合监督

[文档目录](../../docs/README.md) · [持续研究](../../docs/continuous-research.md) · [候选清单](experiment.json) · [父版本 v303](../v303/README.md)

属于 v40x 监督目标系列。v401 的排名监督提高了 Rank IC，但相对 v303 丢失部分 Pearson IC、换手代理更高。v402 以当前本地最优 v303 为对照，只改变监督目标；独立目标向量实现复用 v401 的方法。保持 **433,409** 个参数、Huber delta=2、全局随机 batch 512、种子 42、优化器、精度和验证 IC 早停，从头初始化。

## 完整截面目标

对每个完整截面内的有限收益分别构造收益 z-score \(z_i\) 和平均秩 z-score \(r_i\)，并列收益使用相同平均秩。先等权混合，再按该完整截面的总体标准差重新标准化：

\[
u_i = 0.5 z_i + 0.5 r_i
\]

\[
t_i = \frac{u_i-\operatorname{mean}(u)}{\operatorname{std}_{ddof=0}(u)}
\]

固定等权，不扫描权重。再次标准化使目标保持单位方差，避免仅因混合后尺度缩小而改变 Huber delta 的相对意义。保留无效标签和原有不足／常数截面的处理；完成完整截面变换后才全局打乱，不能在随机样本 batch 内重新排名或标准化。

研究假设是兼顾收益幅度与相对排序。该目标不等于两项 Huber 损失的简单平均，也不保证同时超过两种端点目标。验证 Huber 使用相同混合目标；checkpoint 仍只依据原始收益的验证 IC，正式 IC／Rank IC、样本集合和时间区间均沿用原协议。

## 数据与效率

输入、网络、推理和共享 FP16 行情缓存与 v303 一致。启动时读取原标签，创建约 62 MB 的 FP64 目标向量；核对日期、时刻、股票、有效标签及时间边界，不修改共享缓存中的原目标。`target-overlay.json` 记录等权混合及再次标准化规则、来源和结果指纹，恢复时重建核验。

除目标准备外，单步训练计算图不变；实际训练时间取决于验证早停。推理只执行单个 GRU，不需要标签、排名或双模型融合。

## 执行与验证

唯一正式运行为 `v402-blended-target-gru-s42`，由训练 session 在 ready 后串行执行：

```powershell
.venv-training/Scripts/python.exe -u solutions/v402/v402_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v402-blended-target-gru-s42 --evaluation-cache-mode stream --training-confirmed
```

实现检查覆盖并列值、无效／常数标签、幅度信息保留、正仿射变换不变性、原收益 IC、缓存身份、目标对齐与恢复、共享目标不变、未来信息隔离、文本权重往返及确定性训练恢复；完整合成入口覆盖月度和全局两种模式。默认 GPU 批量预检由训练 session 在模型间隙执行。尚无正式行情评估结果，不能将实现检查解释为分数改善。
