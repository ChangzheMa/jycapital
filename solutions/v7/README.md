# v7 待修改版本

[文档目录](../../docs/README.md) · [修改备忘录](../../docs/v7-discussion.md) · [父版本 v6](../v6/README.md) · [模型配置](model-config.json)

v7 尚未训练，没有独立权重、评估成绩或提交材料。父运行是 `v6-20261015-s42`；当前模型、输入处理、损失和推理沿用 v6，共 **2,429,505** 个参数。已生效的训练变化是[统一动态早停](../../docs/development.md#后续训练的统一早停策略)，其余训练设置沿用 v6。

隐藏层清零、有效时点池化和卷积左侧补零仍是当前实现；已确认的模型修改及 LayerNorm 等未决问题统一保留在备忘录，尚未实施。

## 代码入口

| 用途 | 文件 |
| --- | --- |
| 输入布局与切窗 | [v7_schema.py](v7_schema.py)、[v7_data.py](v7_data.py) |
| 网络与损失 | [v7_model.py](v7_model.py)、[v7_loss.py](v7_loss.py) |
| 预处理 | [v7_prepare.py](v7_prepare.py)、[v7_apply.py](v7_apply.py)、[v7_preprocessing.py](v7_preprocessing.py) |
| 训练与推理 | [v7_train.py](v7_train.py)、[v7_predict.py](v7_predict.py) |
| 文本模型与复现 | [v7_text_artifact.py](v7_text_artifact.py)、[v7_notebook.py](v7_notebook.py)、[v7_reproduce.py](v7_reproduce.py) |

版本标识为 `v7`，文本格式为 `jycapital-v7-text-v1`，模型附件名为 `v7-model.json`。训练复现材料须包含 [early_stopping.py](../../common/early_stopping.py)；运行和源码追溯见 [开发约定](../../docs/development.md)。
