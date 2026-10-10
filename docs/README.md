# 进益资本杯比赛资料

更新日期：**2026-10-10（UTC+08:00）**。本仓库整理 BigQuant 2026 年第二届“进益资本”杯量化因子挑战赛端到端赛道资料，比赛 ID 为 `45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca`。

**当前结论：`mock-submission.ipynb` 已在本地和平台正常运行，测试 submission 已通过。** 该结论由用户确认；有效返回格式为 Pandas DataFrame，列顺序为 `["date", "instrument", "factor"]`。Mock 用于验证运行与提交接口，分数不代表模型预测能力。

**v1 基线已完成本地训练、评估与提交目录导出，用户已确认可在平台运行。** 采用 2020—2023 年训练、2024 年上半年验证、下半年测试；测试集本地 IC 为 0.083549、Rank IC 为 0.068316，预测覆盖率 100%。平台运行结论依据用户于 2026-10-10（UTC+08:00）的确认；正式模型提交结果尚未确认。详细方法与结论见 [v1 基线](baseline-v1.md)。

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [实验工作流技能](../.agents/skills/jycapital-experiment/SKILL.md) | 本仓库中按实验、训练、评估、比较和提交请求自动使用，无需重复项目名或技能名 |
| [方案版本与评估协议](development.md) | 通用训练/验证/测试划分、统一指标、运行追溯和提交导出 |
| [v1 深度学习基线](baseline-v1.md) | 版本绑定的数据处理、小型 MLP、训练与提交使用说明 |
| [平台提交目录](../submissions/README.md) | 随 Git 推送的 Notebook、训练/推理脚本、配置和冻结模型权重 |
| [v1 结果摘要](../experiments/v1-baseline.json) | 本地指标、源代码/模型哈希和导出验证证据 |
| [运行索引](../experiments/runs.csv)、[提交索引](../experiments/submissions.csv) | 方案运行与待上传提交包的对应关系 |
| [规则与赛程](bigquant-2026-end-to-end/rules.md) | 参赛要求、有效提交格式、评分、赛程及奖励 |
| [数据说明](bigquant-2026-end-to-end/data.md) | 数据表、字段、下载版与云端口径、标签和时间对齐 |
| [冲突与待确认事项](bigquant-2026-end-to-end/open-questions.md) | 尚未解决的规则差异与正式模型开发问题 |
| [来源与结论范围](bigquant-2026-end-to-end/sources.md) | 官方来源、获取日期及用户确认的运行结论 |
| [比赛配置摘录](bigquant-2026-end-to-end/competition-config.json) | 2026-10-09 官方配置快照，不替代正文规则 |
| [结构化数据字典](bigquant-2026-end-to-end/data-schemas.json) | 页面及元数据口径的 104 条字段定义 |
| [数据字典 CSV](bigquant-2026-end-to-end/data-dictionary.csv) | 与 JSON 对应的 104 条字段记录 |
| [本地运行与数据申请](bigquant-2026-end-to-end/local-download.md) | 本地数据位置、DAI 兼容层、环境配置及下载授权 |
| [数据一致性结论](bigquant-2026-end-to-end/data-validation.md) | 七张表的覆盖范围、一致性结论及原始数值异常 |
| [数据验证摘要](bigquant-2026-end-to-end/data-validation-summary.json) | 行数、字段类型、缺失计数和已匹配的内容哈希 |
| [提交格式 Mock](../examples/mock-submission.ipynb) | 已通过本地运行、平台运行和测试提交的 Notebook |
| [本地依赖](../requirements-local.txt) | 本地 Parquet 查询环境的依赖 |
| [Git 忽略规则](../.gitignore)、[文件字节规则](../.gitattributes) | 本地文件忽略范围及提交材料的跨平台哈希一致性 |

## 当前可用接口

```python
def main(datasources, start_date, end_date):
    # 返回评测区间内的结果。
    return result[["date", "instrument", "factor"]]
```

- `date` 保留完整分钟时间，`instrument` 使用平台股票代码，`factor` 为有限的 `float64` 数值；`(date, instrument)` 不重复。
- Mock 根据股票池记录生成确定性分数，按完整时间和股票代码对齐，清理非法值后返回三列 DataFrame。
- 本地通过被 Git 忽略的根目录 `dai.py` 读取现有 Parquet；平台使用真实 `dai`。本地兼容层不随仓库分发，安装和使用说明见[本地运行](bigquant-2026-end-to-end/local-download.md)。
- 测试 Mock 只需上传一个 Notebook；不上传本地 `dai.py`、凭据或行情文件。正式模型需按[提交规范](bigquant-2026-end-to-end/rules.md)提供训练脚本、依赖、配置、随机种子和权重。

## 参赛要点

- **任务**：使用授权原始分钟量价序列预测未来 30 分钟 VWAP 收益，按 30 分钟采样，不预测隔夜；允许必要清洗、归一化和切窗，禁止人工衍生特征输入及外部预训练权重。
- **数据**：本地已有四张五档分钟表及三张辅助表，共 243 个文件、723,258,293 行。七张表与 2026-09-30 平台基准的全量逻辑内容一致；原始数据异常仍需按业务规则处理。
- **赛程**：按正文的公榜提交截止 **10 月 29 日 23:59:59**、私榜候选选择截止 **10 月 30 日 23:59:59** 安排；接口的较晚时间仍与正文冲突。时间均为 UTC+08:00。
- **资源与资格**：按 3 小时运行限制及无外网环境准备；进入决赛的队伍须至少有一名清华大学在读学生。
- **剩余问题**：标签精确公式、训练与评测日期边界、正式模型附件协议及完整评测算法仍需明确。测试 submission 通过不替代这些要求。

来源及证据日期见 [sources.md](bigquant-2026-end-to-end/sources.md)。仓库仅保存代码示例、结论、必要来源及结构化字典，不保存凭据、账号个人资料或实际行情数据。
