# v302：短片段压缩的分层 GRU

[文档目录](../../docs/README.md) · [研究约定](../../docs/continuous-research.md) · [候选清单](experiment.json) · [父版本 v301](../v301/README.md)

v30x 系列的第二版。以 `v301-global-fp16-s42` 为父版本对照，从头初始化。研究假设：对原始片段进行可学习投影后再递推，可以缩短 GRU 链、改善训练效率，同时保留日内与跨日的层次结构。必须在父运行完整结束后执行本版。

## 变化与控制

三频原始窗口、字段、监督、全局 batch 512、输入精度、验证协议和早停均沿用 v301。日内 GRU 前增加无重叠学习投影：

| 分支 | 每日原始 Bar 数 | 每个 patch 的 Bar 数 | 每日递推步数 |
| --- | ---: | ---: | ---: |
| 1 分钟 | 240 | 5 | 48 |
| 5 分钟 | 48 | 2 | 24 |
| 30 分钟 | 8 | 1 | 8 |

跨日仍分别覆盖 2／5／20 个交易日。每个 patch 联合投影全部 75 个输入通道，隐藏宽度 96，日内和跨日 GRU 均保持 FP32 状态递推，无 LayerNorm。总参数 **469,409**，父版本为 433,409。

预测时点之后的 Bar 在投影前清零。读取当天最后一个已开始 patch；09:30 不把当天未来补零作为有效状态，跨日读取昨日状态。一般合法 patch 尺寸下的部分 patch 也保留已发生 Bar，未来值不进入输出或梯度。历史缺失仍保留固定槽位，不过滤股票池键。

## 本地执行与验证

唯一正式运行标识为 `v302-patch-gru-s42`，由训练 session 在 ready 后串行执行：

```powershell
.venv-training/Scripts/python.exe -u solutions/v302/v302_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v302-patch-gru-s42 --evaluation-cache-mode stream --training-confirmed
```

复用原训练与验证缓存，参数和完整输入形状的 GPU 预检由训练 session 在模型间隙执行。CPU 合成数据覆盖未来隔离、部分 patch、空历史、全部参数梯度、分批推理、文本权重恢复、完整训练入口和确定性断点恢复。

2026-10-20 的单 CPU 线程、batch 8 合成训练步中，本版中位耗时约 0.031 秒，v301 约 0.226 秒；GPU 实际吞吐和正式指标仍待本地训练测量。若压缩损伤局部信息，不能只因更快就视为效果改善。平台材料在本地评估后按候选结论生成。
