# v7c：左右对称边界复制

[实验总览与统一设置](../../docs/v7-ablation-experiments.md) · [配置](model-config.json) · [模型](v7c_model.py)

父运行为 `v7-20261016-no-layernorm-s42`，对应备忘录第 3 组。仅将全部 26 次时序卷积的左侧补零改为左右对称边界复制，每侧填充 `(k - 1) * dilation // 2`。核宽仍为 3，膨胀率仍按块为 1／2／4／8；掩码、池化和纯 IC 损失保留，共 2,421,185 个参数。

输入仍截至预测时刻，窗口内部允许双向计算。代码已准备，尚未正式训练。

```powershell
.venv-training/Scripts/python.exe -u solutions/v7c/v7c_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v7c-replicate-padding-s42 --training-confirmed
```
