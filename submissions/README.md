# 平台提交材料

[文档目录](../docs/README.md) · [导出与验证](../docs/development.md#比较与导出工具) · [提交状态索引](../experiments/submissions.csv)

每个子目录对应一份冻结材料。只上传选定目录内的文件，保留文件名，选择其中唯一的推理 Notebook；不混入其他运行或本地 `dai.py`，不生成 ZIP。平台提交由用户执行。

| 运行 | 上传材料 | 本地判定与验证清单 |
| --- | --- | --- |
| [v1-20261010-s42](v1-20261010-s42/README.md) | 目录中的 Notebook 及配套脚本、权重和配置，见目录说明 | 历史基线；[清单](v1-20261010-s42/submission-manifest.json) |
| [v2-20261011-s42](v2-20261011-s42/v2-submission.ipynb) | `v2-submission.ipynb`、`v2-model.json` | 未晋级，按用户要求导出；[清单](../experiments/v2-20261011-s42-submission.json) |
| [v4-20261014-s42](v4-20261014-s42/v4-submission.ipynb) | `v4-submission.ipynb`、`v4-model.json` | 未晋级，按用户要求导出；[清单](../experiments/v4-20261014-s42-submission.json) |
| [v5-20261015-s42](v5-20261015-s42/v5-submission.ipynb) | `v5-submission.ipynb`、`v5-model.json` | 本地改善候选；[清单](../experiments/v5-20261015-s42-submission.json) |
| [v6-20261015-s42](v6-20261015-s42/v6-submission.ipynb) | `v6-submission.ipynb`、`v6-model.json` | 本地改善候选；[清单](../experiments/v6-20261015-s42-submission.json) |
| [v7-20261016-no-layernorm-s42](v7-20261016-no-layernorm-s42/v7-submission.ipynb) | `v7-submission.ipynb`、`v7-model.json` | 本地改善候选；[清单](../experiments/v7-20261016-no-layernorm-s42-submission.json) |

v2／v4—v7 的两个文本文件需置于同一平台目录。JSON 内嵌冻结权重、归一化统计量、配置和训练／推理源码；Notebook 提供评测入口、依赖及复现说明。附件大小、数值精度及独立推理的验证范围以各清单为准。

## 平台状态与完整性

平台于 2026-10-16（UTC+08:00）查询返回 6 条公榜成绩，私榜分数均为空；文件名显示 v1、v2、v4、v5、v6 及 Mock，本地 run／权重绑定待确认。原始分数、分项指标与对应依据见 [成绩记录](../experiments/platform-scores.json)。v7 已通过本地独立推理验证，尚未提交平台。本地验证、平台可运行和官方成绩分别记录；提交状态以索引及清单为准。

材料随 Git 分发；`.gitattributes` 保持提交文件原始字节。重新打包时须更新清单并核验冻结权重和推理一致性，具体格式与验证要求见开发约定。行情、离线预测答案、缓存和凭据不进入提交材料。
