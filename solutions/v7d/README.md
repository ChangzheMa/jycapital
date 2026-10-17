# v7d：连续位置的逐块增宽卷积

[实验总览与统一设置](../../docs/v7-ablation-experiments.md) · [配置](model-config.json) · [模型](v7d_model.py)

父运行为 `v7-20261016-no-layernorm-s42`，对应备忘录第 4 组。膨胀率全部为 1；1 分钟分支各块核宽为 3／5／7／9，其余分支为 3／5／7。每块两次卷积宽度相同、权重独立，左侧仍补零 `k - 1` 个位置。掩码、池化和纯 IC 损失保留，共 3,957,185 个参数。

代码已准备，尚未正式训练；不包含第 3 组的对称复制填充。

```powershell
.venv-training/Scripts/python.exe -u solutions/v7d/v7d_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v7d-dense-kernels-s42 --training-confirmed
```
