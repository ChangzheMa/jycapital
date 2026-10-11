# 仓库约定与工具限制

在定位实现、运行训练或整理交付材料时阅读。以下是已建立的项目约定；执行时检查仓库实际内容，避免把旧路径或旧实现当作当前状态。

## 资料和产物位置

| 位置 | 用途 |
| --- | --- |
| `AGENTS.md`、`docs/README.md` | 仓库约定与资料入口 |
| `docs/bigquant-2026-end-to-end/rules.md`、`data.md`、`open-questions.md` | 输入边界、数据语义及尚未确认的官方要求 |
| `docs/development.md`、`evaluation/protocols/` | 本地协议、日期划分和记录方式 |
| `solutions/<version>/` | 方案自有模型、数据处理、训练、推理和配置 |
| `evaluation/`、`common/`、`scripts/` | 公共指标、运行记录、比较和导出工具 |
| `experiments/runs.csv`、`submissions.csv`、结果 JSON | 可入库的运行/提交索引和汇总证据 |
| `outputs/runs/<run_id>/` | 忽略的日志、预测、缓存、环境、源码快照和训练 checkpoint |
| `submissions/<run_id>/` | 进入 Git 的平铺提交材料与冻结推理权重 |
| `data/`、根目录 `dai.py` | 忽略的授权本地行情与本地查询兼容层 |

当前本地协议为 `local-v1`：2020-01-01 至 2024-01-01 前训练，2024 年上半年验证，下半年测试；均为 UTC+08:00，区间左闭右开。标签使用官方表 `vwap_return`，精确生成公式仍需参考未决事项，不擅自重算标签。

默认每个版本都在选定 checkpoint 后评估测试集，与该协议的训练后评估次序一致。不要为了表达工作流习惯而修改既有协议文件，造成历史比较指纹变化。

比赛使用授权原始量价序列，允许必要清洗、归一化与切窗；人工技术指标、盘口衍生特征或其他输入边界按仓库规则执行，有实际歧义时先讨论。时间点 t 的输入不能使用 t 之后的信息。使用未来标签训练和在推理时读取标签是不同操作。

## 新版本前必须检查的工具约束

已有工具可以复用，但并未全部泛化：

- `common/artifacts.py` 的 `snapshot_sources()` 曾固定读取 `solutions/v1`。新版本要检查并改为捕获实际方案及其依赖，否则记录的源码与实际训练不同。
- `scripts/export.py` 的脚本清单、模型名、Notebook 名和文案曾固定为 v1。按目标运行的版本、模型格式和依赖闭包适配，不能把 v2 权重配上 v1 推理。
- `scripts/verify-submission.py` 曾固定检查特定日期、1,000 只股票和调试区间。根据当前协议及 Notebook 调试参数校验预期键，不把示例数据量当成永恒平台规则。
- `scripts/compare.py` 目前主要检查指纹并列出指标；版本参照选择、差值、逐月稳定性、候选决策和平台分需另外完成。
- 版本脚本有路径推断/导入逻辑。进入 `submissions/` 后要验证使用目录内冻结附件，避免意外导入工作区同名模块；训练复现依赖也要完整。
- 文件变更后同步更新提交清单中的对应哈希及 `experiments/submissions.csv` 的清单哈希。不要把旧训练快照的 Git commit/hash 改写成后来整理文档时的值。

应按新实验需要修正这些限制，并运行相应检查。创建/使用本技能不意味着这些适配已经实现。

## 环境与运行

在仓库根目录选择验证过的 Python 环境。项目已有 `.venv` 本地查询环境和 `.venv-training` 训练环境，但先检查它们是否存在、实际依赖与 GPU 是否正常。训练记录保存生效版本，不能只记录意图安装的 requirements。

Windows 的当前训练实现已对 PyTorch/NumPy 导入顺序作处理；复用现有可工作的启动方式，遇到具体依赖故障再定位，不要把历史上的某次环境故障扩散为所有平台必须遵守的规则。

典型命令形态如下；运行前将版本、脚本及 run_id 换成实际值：

```powershell
.venv-training/Scripts/python.exe -u solutions/v2/v2_train.py --run-dir outputs/runs/<run_id>
.venv/Scripts/python.exe scripts/compare.py outputs/runs/<candidate> outputs/runs/<local-best> --split validation
.venv/Scripts/python.exe scripts/compare.py outputs/runs/<candidate> outputs/runs/<local-best> --split test
.venv/Scripts/python.exe scripts/export.py --run-dir outputs/runs/<candidate>
.venv/Scripts/python.exe scripts/verify-submission.py --run-dir outputs/runs/<candidate> --package submissions/<candidate>
```

这些是接口形态示例，不表示仓库已经存在 v2 文件或占位运行。先确认脚本支持当前方案；模型冻结后再导出，不为导出重新训练一遍。

## 提交材料

默认只导出改善候选；用户明确要求的平台测试保留原本地判定。直接使用 `submissions/<run_id>/`，不生成 ZIP。用户要求重新打包时保留旧材料的 Git 历史或本地备份，再更新目录、清单与索引。

提交目录应包含唯一推理 Notebook、完整推理依赖、训练复现代码、完整模型权重/必要预处理状态、配置、随机种子和依赖声明。Notebook 的 `main(datasources, start_date, end_date)` 返回三列 DataFrame：`date` 保留完整分钟，`instrument` 为平台股票代码，`factor` 为有限 `float64`，键 `(date, instrument)` 唯一。

推理使用平台 `dai`；根目录本地 `dai.py` 不上传。通过 `datasources` 获取实际行情表；调试单元格可以显式传入开发表，但应有 `if __name__ == "__main__":` 保护且无保存输出。

平台附件全部采用 UTF-8 文本，尽量合并为唯一 Notebook 与一个参数 JSON。JSON 可同时保存数值权重、归一化统计、配置及完整训练／推理源码；Notebook 内提供使用说明、依赖和显式训练复现入口。权重保留原始 dtype 与数值精度，导出后逐张量及逐预测核验。不提交 `.pt` 等二进制附件，也不在推理时还原二进制权重文件。

清单及验证记录放在 `experiments/<run_id>-submission.json`，提交索引中的 `manifest_path` 记录位置，减少平台目录文件数。历史 v1 材料仍可使用目录内清单；读取工具按索引定位，不硬编码位置。用户未要求重做的历史提交不自动重新打包。

验证导入路径、真实入口、短区间调试、时间边界、完整评估区间输出和实际使用的推理后端；后端转换时比较预测一致性。对未改动且已验证的部分，复用证据，不无理由重复训练或扩展检查。

`.gitattributes` 保持 `submissions/` 的文件原始字节，并允许已有 CRLF 作为合法换行，避免 Git 自动换行破坏 SHA-256。提交前校验逐文件哈希，并检查从 Git 检出后的材料；只校验工作区不能证明远程取回后仍一致。

## Git 与结果边界

仓库默认远程为 `git@github.com:ChangzheMa/jycapital.git`，主分支为 `master`；执行前核对，遵从用户当次指定。完整工作流中推送代码、配置、比较摘要、必要文档和改善候选的提交目录；原始行情、预测文件、凭据、本地兼容层、环境和训练缓存继续忽略。

如果没有改善，可以在完整流程授权内提交/推送本次研究代码与结果摘要，但不增加可提交目录。不能把未完成/失败的实验包装成可运行提交，也不能因失败就抹去已有研究记录。

平台运行状态、正式提交状态和官方分数独立保存。用户确认平台可运行时注明来源和日期，不扩展为官方成绩；平台实际提交由用户进行。
