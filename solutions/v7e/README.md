# v7e：截面标准化标签的 Huber 回归

[实验总览与边界设置](../../docs/v7-ablation-experiments.md) · [配置](model-config.json) · [损失](v7e_loss.py) · [训练器](v7e_train.py)

父运行为 `v7-20261016-no-layernorm-s42`，对应备忘录第 5 组。网络结构完全沿用 v7，共 2,421,185 个参数；每个完整截面的有限标签按总体标准差做 z-score，原始预测直接拟合该目标。固定 Huber `delta=1`，输出 `factor_scale=1`，按验证 IC 选模。

训练先按完整预测时间戳构造标准化目标，再沿用 v2/v3 的月内随机样本 batch：每轮打乱月份，月内以 seed、epoch、月份确定样本排列，batch 为 512，保留不足 512 的尾批。随机 batch 可以混合本月不同日期、时刻和股票，不在 batch 内重新标准化。训练 Huber 按样本取均值，轮次日志按实际样本数加权；验证／测试 Huber 仍按完整截面计算并等权汇总。随机 batch 上的相关性不记作训练截面 IC，验证／测试 IC 独立报告。

每 64 个 batch 及月末保存 checkpoint，断点保存样本游标、优化器、随机状态及早停计数，恢复时重建相同排列。旧的逐截面更新配置不能续跑此版本。相对 v7，本实验同时改变监督目标、损失和 batch 组织，后续比较需按这一整体设置解释。

标签退化、常数预测、损失日志及输出尺度说明统一见实验总览。代码已准备，尚未正式训练。

```powershell
.venv-training/Scripts/python.exe -u solutions/v7e/v7e_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v7e-zscore-huber-s42 --training-confirmed
```
