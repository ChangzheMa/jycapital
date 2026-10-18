# v101：多频 Patch Transformer

[文档目录](../../docs/README.md) · [三个研究家族](../../docs/research-branches.md) · [代码父版本 v9](../v9/README.md) · [模型](v101_model.py) · [模型配置](model-config.json)

v10x 注意力研究家族的首版，以 v9 为代码父版本。研究假设是让不同历史片段在编码阶段按内容交互，改善长窗口利用方式。当前交付代码与合成实现验证，尚未正式训练，没有模型成绩或平台材料。

## 主干

| 分支 | 原始窗口 | patch 长度 | patch 数量 |
| --- | ---: | ---: | ---: |
| 1 分钟 | 480 | 8 | 60 |
| 5 分钟 | 240 | 4 | 60 |
| 30 分钟 | 160 | 4 | 40 |

每个 patch 按时间、通道顺序展开全部 75 通道，经 Linear 投影到 96 维。每路加入独立的可学习 CLS 和位置编码，使用两层 Pre-LayerNorm Transformer，4 个注意力头，前馈宽度 192，Dropout 0.1。位置参数按固定交易日槽位学习；所有已知历史可双向交互。每路取归一化后的 CLS，三路拼接八维预测时钟，经 `Linear(296,256) → GELU → Dropout(0.1) → Linear(256,1)` 输出。

总参数 **656,993**。没有时序卷积或全窗口大展平投影。注意力使用 PyTorch math SDPA 后端，确保既有确定性训练要求下无需依赖 FlashAttention 反向路径。

预测截点之后的数值先清零；部分可用 patch 仍保留已知 Bar，全未来 patch 不作为 key/value。历史缺失 patch 保留位置及缺失标志，CLS 始终可用，避免全空历史产生全掩码注意力。因果性、训练控制、比较局限和检查入口见[研究家族说明](../../docs/research-branches.md)。

## 后续使用

目录内包含独立预处理、切窗、训练、原始行情推理、文本张量和复现模块，不导入其他方案。沿用 v9 的输入与训练配置；LayerNorm、Dropout、容量和汇总方式作为本主干的一部分适配。`parent_solution_id=v9`，父 run 为空。

以下仅为未来获准正式训练后的命令，本轮未执行：

```powershell
.venv-training/Scripts/python.exe -u solutions/v101/v101_train.py --training-run outputs/preprocessing/v2-20261010 --holdout-run outputs/preprocessing/v2-holdout-20261011 --run-dir outputs/runs/v101-patch-transformer-s42 --training-confirmed
```

原始行情入口为 [v101_predict.py](v101_predict.py)，与训练共用 [WindowBlock](v101_data.py)。独立预处理使用 `v101_prepare.py`、`v101_apply.py`；协议、恢复、评估和文本附件约定见[开发约定](../../docs/development.md)。正式导出及平台运行尚未验证。
