# v7f：预测时刻八维 one-hot

[实验总览与边界设置](../../docs/v7-ablation-experiments.md) · [配置](model-config.json) · [时间编码](v7f_data.py) · [模型](v7f_model.py)

父运行为 `v7-20261016-no-layernorm-s42`，对应备忘录第 6 组候选。仅将最终拼接的预测时刻由 sin/cos 改为八维 one-hot，融合层输入由 642 改为 648；历史 Bar 时间编码、77 通道输入、掩码、卷积、池化和纯 IC 损失保留，共 2,422,721 个参数。

固定八时刻顺序和未知时刻处理见实验总览。代码已准备，尚未正式训练，是否纳入主方案由结果决定。

```powershell
.venv-training/Scripts/python.exe -u solutions/v7f/v7f_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v7f-one-hot-clock-s42 --training-confirmed
```
