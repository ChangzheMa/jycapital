# v7a：取消隐藏有效性掩码

[实验总览与统一设置](../../docs/v7-ablation-experiments.md) · [配置](model-config.json) · [模型](v7a_model.py)

父运行为 `v7-20261016-no-layernorm-s42`，对应备忘录第 1 组。独立移除输入投影后、残差块内及分支输出的有效性清零；mean 使用全部时间位置，last 取窗口末端。保留 `Linear(320,160)` 汇总投影、原卷积及纯 IC 损失，共 2,421,185 个参数。

代码已准备，尚未正式训练。训练、恢复、产物及后续分析要求统一见实验总览。

```powershell
.venv-training/Scripts/python.exe -u solutions/v7a/v7a_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v7a-no-mask-s42 --training-confirmed
```
