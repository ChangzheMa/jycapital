# v7b：取消掩码并展平分支特征

[实验总览与统一设置](../../docs/v7-ablation-experiments.md) · [配置](model-config.json) · [模型](v7b_model.py)

父运行为 `v7-20261016-no-layernorm-s42`，按用户指定采用备忘录第 1 + 2 组。取消隐藏有效性清零，四个分支分别将 9,600／3,840／2,560／2,560 维展平特征投影为 160 维。保留原卷积及纯 IC 损失，共 5,185,985 个参数。

后续同时与 v7 和 v7a 比较：相对 v7 含两个变量，相对 v7a 用于观察展平的增量效果。代码已准备，尚未正式训练；本版输入窗口长度须与配置一致。

```powershell
.venv-training/Scripts/python.exe -u solutions/v7b/v7b_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v7b-no-mask-flatten-s42 --training-confirmed
```
