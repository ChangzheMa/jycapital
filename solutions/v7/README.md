# v7 待修改版本

[文档目录](../../docs/README.md) · [修改讨论与确认记录](../../docs/v7-discussion.md) · [父版本 v6](../v6/README.md) · [模型配置](model-config.json)

v7 是从 v6 建立的独立版本，当前状态为**待修改、未训练**。模型结构、数据处理、损失、训练和推理设置与 v6 相同；已完成文件名、模块导入、配置标识、源码路径和文本模型格式的 v7 命名适配。后续讨论确认的改动在 v7 上实施，v6 保留为历史基线。

## 基线与当前设置

父运行及当前本地验证最优参照为 `v6-20261015-s42`。该运行的权重和评估结果属于 v6，未作为 v7 的训练产物复制；v7 尚无独立权重、评估成绩或平台提交材料。

| 项目 | 当前设置，与 v6 相同 |
| --- | --- |
| 输入 | 四频窗口 60／24／16／16 根 Bar，每根 77 通道 |
| 编码器 | 隐藏通道 160；1 分钟分支 4 个残差块，其余各 3 个 |
| Dropout | TCN 残差块 0.35；融合预测层 0.10 |
| 池化与预测头 | 最后有效时点与有效时点均值拼接；四频向量加时钟后 642 → 256 → 1 |
| 参数量 | 2,429,505 |
| 训练目标 | 完整截面的纯 Pearson IC，按验证 IC 选模 |
| 数据划分 | 2020—2023 年训练、2024 年上半年验证、下半年测试 |
| 推理输出 | `date、instrument、factor`，因子为网络分数除以 100 |

全无效 Bar 的隐藏值清零及有效时点池化仍沿用 v6。相关讨论尚未形成确定的模型修改要求，以[讨论文档](../../docs/v7-discussion.md)中的已确认条目为准。

## 代码入口

| 用途 | 文件 |
| --- | --- |
| 输入布局与切窗 | [v7_schema.py](v7_schema.py)、[v7_data.py](v7_data.py) |
| 网络与损失 | [v7_model.py](v7_model.py)、[v7_loss.py](v7_loss.py) |
| 预处理与归一化 | [v7_prepare.py](v7_prepare.py)、[v7_apply.py](v7_apply.py)、[v7_preprocessing.py](v7_preprocessing.py) |
| 训练与评估 | [v7_train.py](v7_train.py) |
| 推理 | [v7_predict.py](v7_predict.py) |
| 文本模型与 Notebook 适配 | [v7_text_artifact.py](v7_text_artifact.py)、[v7_notebook.py](v7_notebook.py)、[v7_reproduce.py](v7_reproduce.py) |

版本标识为 `solution_id=v7`；文本模型适配器使用 `jycapital-v7-text-v1` 格式及 `v7-model.json` 文件名。训练源码快照和复现入口均指向 `solutions/v7/`。版本初始化不创建训练运行；后续实施和实验按用户明确指定的范围执行。
