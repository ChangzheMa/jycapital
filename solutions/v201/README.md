# v201：低秩时间／通道 Mixer

[文档目录](../../docs/README.md) · [三个研究家族](../../docs/research-branches.md) · [代码父版本 v9](../v9/README.md) · [模型](v201_model.py) · [模型配置](model-config.json)

v20x MLP 混合器研究家族的首版，以 v9 为代码父版本。研究假设是以时间／通道分解、共享权重及低秩约束，学习长窗口中的有效组合并控制容量。当前交付代码与合成实现验证，尚未正式训练，没有模型成绩或平台材料。

## 主干

每路保留 480／240／160 个原始槽位，先将 75 通道线性投影为 96 维。两层 Mixer 交替执行：

1. 时间混合：LayerNorm 后转置，所有隐藏通道共享 `Linear(T,32) → GELU → Dropout(0.1) → Linear(32,T)`，再做残差相加。
2. 通道混合：LayerNorm 后，所有时间位置共享 `Linear(96,192) → GELU → Dropout(0.1) → Linear(192,96)`，再做残差相加。

时间混合包含非线性，32 表示瓶颈宽度，不表示整个非线性映射严格只有秩 32。未来隐藏位置在每次混合后清零，历史缺失位置保留，不依据有效值比例删除样本。

汇总时先做 LayerNorm，各隐藏通道共享 `Linear(T,4)`，再将 `96×4` 展平并经 `Linear(384,96) → GELU` 得到分支表示。三路拼接八维预测时钟，预测头为 `Linear(296,256) → GELU → Dropout(0.1) → Linear(256,1)`。

总参数 **552,973**。主干没有卷积或注意力；固定的可学习时间权重表达位置，所有历史槽位可以在时间混合阶段交互。该方案的适应性与容量取舍、共同监督及检查入口见[研究家族说明](../../docs/research-branches.md)。

## 后续使用

目录内包含独立预处理、切窗、训练、原始行情推理、文本张量和复现模块，不导入其他方案。沿用 v9 的输入与训练配置；低秩瓶颈、LayerNorm、Dropout 和分解汇总属于本主干方案。`parent_solution_id=v9`，父 run 为空。

以下仅为未来获准正式训练后的命令，本轮未执行：

```powershell
.venv-training/Scripts/python.exe -u solutions/v201/v201_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v201-factorized-mixer-s42 --training-confirmed
```

原始行情入口为 [v201_predict.py](v201_predict.py)，与训练共用 [WindowBlock](v201_data.py)。独立预处理使用 `v201_prepare.py`、`v201_apply.py`；协议、恢复、评估和文本附件约定见[开发约定](../../docs/development.md)。正式导出及平台运行尚未验证。
