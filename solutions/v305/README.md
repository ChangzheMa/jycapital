# v305：保留更细时间分辨率的 Patch GRU

[文档目录](../../docs/README.md) · [持续研究](../../docs/continuous-research.md) · [候选清单](experiment.json) · [父版本 v302](../v302/README.md)

以 `v302-patch-gru-s42` 为父版本，从头初始化。v302 相比 v301 节省约 42% 的逐轮训练与验证耗时，但验证 IC 和双集 Rank IC 退化。本版仅减小时间压缩率，检验能否恢复细节，同时保留部分效率优势。

| 分支 | v302 每片段 Bar 数 | v305 每片段 Bar 数 | v305 每日日内递推步数 |
| --- | ---: | ---: | ---: |
| 1 分钟 | 5 | 2 | 120 |
| 5 分钟 | 2 | 1 | 48 |
| 30 分钟 | 1 | 1 | 8 |

总参数 **440,609**。其余网络、75 通道输入、固定交易日窗口、标准化收益 Huber delta=1、全局随机 batch 512、优化器、精度和动态早停均沿用 v302。保留投影前未来清零及最后可用片段读出，原始标签和正式评估协议不变。

v301 保留完整递推，v302 使用较强压缩，本版提供较弱压缩的对照。每轮成本预期介于两者附近，须由实际 GPU 测量确认，不能按递推步数直接推算总训练时间。复用现有行情与目标缓存。

唯一正式运行为 `v305-fine-patch-gru-s42`。候选 ready 后由训练 session 串行执行：

```powershell
.venv-training/Scripts/python.exe -u solutions/v305/v305_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v305-fine-patch-gru-s42 --evaluation-cache-mode stream --training-confirmed
```

实现检查覆盖参数梯度、未来值及梯度隔离、非法片段配置、空历史、分批一致性、文本权重精确恢复、完整训练入口和确定性断点恢复。比较时同时使用父版本 v302 和本地最优 v301；训练后完成双集评估，再决定平台材料。
