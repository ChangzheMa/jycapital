---
name: jycapital-scores
description: "查询 BigQuant 进益资本比赛的提交状态、公榜/私榜分数和分项指标；用户说查平台分、查看提交成绩、刷新成绩或回填平台分时使用。支持 Cookie 与账号登录，不提交作品、不启动训练。"
---

# 查询进益资本平台成绩

使用用户指定的 BigQuant 接口查询当前登录账号可见的提交。默认比赛 ID 为 `45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca`，用户指定其他比赛时使用其 ID。接口依据：用户于 2026-10-10（UTC+08:00）提供的请求及同日真实查询验证。

## 查询

运行本技能目录内的 `scripts/query-scores.py`，只依赖 Python 标准库。从仓库使用：

```powershell
.venv/Scripts/python.exe .agents/skills/jycapital-scores/scripts/query-scores.py
```

全局调用时从实际安装的技能目录解析脚本路径，不假定当前目录是仓库。需要了解认证、接口字段或分页行为时读 [接口与认证](references/api.md)。

- 已配置的环境变量或本地认证文件可直接使用，不重复索要登录信息。支持 `BIGQUANT_COOKIE`，或 `BIGQUANT_USERNAME` 与 `BIGQUANT_PASSWORD`；也支持 `BIGQUANT_AUTH_FILE`、`--auth-file` 或 `--auth-stdin`。
- 仓库内默认向上查找 `.bigquant/platform-auth.json`。该目录受 Git 忽略；现有 `.bigquant/config.json` 的 AK/SK 不用于网站登录。技能不内置账号、密码或 Cookie，也不自动持久化本次登录结果。
- 用户在当前会话提供的凭据可经标准输入交给脚本。不要把原始 curl、登录响应、Cookie、账号资料或密码写入技能、日志、提交材料或 Git；只保存经过脚本筛选的成绩。
- 有 Cookie 时先查询；明确认证失效且有账号密码时最多登录一次，再重试查询。没有 Cookie 时使用已提供的账号密码登录。认证仍失败就报告，不循环试密码。缺少凭据时说明缺少哪一种配置，不把缺分当成 0。
- 默认分页查全当前账号可见的提交。`--submission-id` 过滤特定提交，`--competition-id` 切换比赛。不要根据列表顺序推断名次或认定最新条目就是当前模型。

## 解释查询结果

报告查询时间（含时区）、平台提交 ID、文件名、原始状态（接口提供时）、公榜/私榜原始总分及分项指标。公私榜分列；排名仅在接口返回排名字段时展示。`selected_for_private` 只是是否选择私榜，不代表已有私榜分数。

当前列表返回 `public_score`、`public_score_data`、`private_score`、`private_score_data`；总分可能是字符串。脚本保留原值，并另给有限数值和 `available / missing / non_numeric` 状态。空值不是 0；数值 0 是有效分数。字段缺失或非数值分数不能解释成正式提交失败。当前已验证的响应没有任务状态或名次字段，不能据此编造。

分项指标保留平台字段名；`ic_mean` 不自动改称本地 `rank_ic_mean`，也不重新计算一个自创总分。接口返回了分数可以报告“平台已返回成绩”；平台可运行、正式提交状态及官方评分仍是不同证据。

## 对应本地版本与回填

只请求查看时展示结果。请求回填或更新平台最优时，在仓库读取 `experiments/submissions.csv`、`experiments/runs.csv`、对应结果 JSON 和提交清单，并按 [实验工作流的成绩回填约定](../jycapital-experiment/references/comparison.md#平台成绩回填) 更新。

优先使用已有的 `platform_submission_id` 绑定。文件名、文件大小、描述和提交时间仅作为候选线索；接口文件映射中的不透明 ID 不能当成内容哈希。没有可靠的 run/权重对应关系时，先报告平台结果及候选版本，再澄清对应关系，不把 Mock 成绩或最新提交自动归到 v1。

回填保存原始指标、榜单、平台提交 ID、查询时间和 `source=bigquant_submissions_api`；只在同一榜单/评分口径下更新最优。账号个人资料不入库。查询及回填不自动启动训练、导出、Git 推送、提交作品或改变私榜选择。
