# v7e：截面标准化标签的 Huber 回归

[实验总览与边界设置](../../docs/v7-ablation-experiments.md) · [配置](model-config.json) · [损失](v7e_loss.py) · [训练器](v7e_train.py)

父运行为 `v7-20261016-no-layernorm-s42`，对应备忘录第 5 组。网络结构完全沿用 v7，共 2,421,185 个参数；每个完整截面的有限标签按总体标准差做 z-score，原始预测直接拟合该目标。固定 Huber `delta=1`，输出 `factor_scale=1`，按验证 IC 选模。

标签退化、常数预测、损失日志及输出尺度说明统一见实验总览。代码已准备，尚未正式训练。

```powershell
.venv-training/Scripts/python.exe -u solutions/v7e/v7e_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v7e-zscore-huber-s42 --training-confirmed
```
