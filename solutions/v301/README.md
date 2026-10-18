# v301：日内／跨日分层 GRU

[文档目录](../../docs/README.md) · [三个研究家族](../../docs/research-branches.md) · [代码父版本 v9](../v9/README.md) · [模型](v301_model.py) · [模型配置](model-config.json)

v30x 循环状态研究家族的首版，以 v9 为代码父版本。研究假设是先建模日内过程，再积累跨日背景，两级状态共同解释未来收益分数。当前交付代码与合成实现验证，尚未正式训练，没有模型成绩或平台材料。

## 主干

| 分支 | 每日槽位 | 输入交易日数 | 范围 |
| --- | ---: | ---: | --- |
| 1 分钟 | 240 | 2 | 昨日＋今日截至预测点 |
| 5 分钟 | 48 | 5 | 此前 5 个完整交易日 |
| 30 分钟 | 8 | 20 | 此前 20 个完整交易日 |

每路先以 `Linear(75,96) → GELU → Dropout(0.1)` 投影每根 Bar，随后使用一层单向日内 GRU。日内 GRU 在每个交易日从零状态开始，其权重在该频率的所有交易日间共享。读取各日最后一个已发生槽位的状态，再经 Dropout 输入一层单向跨日 GRU；跨日层在每个独立预测窗口从零状态开始，不跨 batch 或请求保留状态。

1 分钟分支在 09:30 读取昨日跨日状态，此后读取包含今日已发生 Bar 的状态。实现可以批量计算补零尾部，但这些尾部状态不进入被选中的输出，也不会反向影响之前的单向状态。历史缺失槽位仍保留时间位置和输入标志；模型可学习缺失输入下的状态更新。未来槽位不作为已观察状态。

三路输出各 96 维，拼接八维预测时钟，经 `Linear(296,256) → GELU → Dropout(0.1) → Linear(256,1)` 输出。总参数 **433,409**；没有卷积、注意力或 LayerNorm。

GRU 状态递推显式使用 FP32，输入投影和预测头保留训练器的 BF16 autocast；评估整体为 FP32。这属于主干的精度适配，不据此宣称与其他网络具有相同吞吐。共同监督、因果性及检查入口见[研究家族说明](../../docs/research-branches.md)。

## 后续使用

目录内包含独立预处理、切窗、训练、原始行情推理、文本张量和复现模块，不导入其他方案。沿用 v9 的输入与训练配置，`parent_solution_id=v9`，父 run 为空。

以下仅为未来获准正式训练后的命令，本轮未执行：

```powershell
.venv-training/Scripts/python.exe -u solutions/v301/v301_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v301-hierarchical-gru-s42 --training-confirmed
```

原始行情入口为 [v301_predict.py](v301_predict.py)，与训练共用 [WindowBlock](v301_data.py)。独立预处理使用 `v301_prepare.py`、`v301_apply.py`；协议、恢复、评估和文本附件约定见[开发约定](../../docs/development.md)。正式导出及平台运行尚未验证。
