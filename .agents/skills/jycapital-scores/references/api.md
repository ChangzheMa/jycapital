# 平台成绩接口与认证

依据：用户于 2026-10-10（UTC+08:00）提供的 curl，以及同日登录和提交列表的实际 JSON 响应。该接口用于账号范围内的查询，不代表完整排行榜接口。

## 认证配置

脚本使用 Python 3.10+ 标准库。认证优先级为显式 `--auth-stdin` / `--auth-file`、`BIGQUANT_AUTH_FILE`、进程环境变量、当前目录及其父目录中的 `.bigquant/platform-auth.json`。各来源不混合，避免将不同账号的 Cookie 和密码组合使用。

认证 JSON 支持 `cookie`、`username`、`password` 三个字符串字段。Cookie 和账号密码可同时提供；只有 Cookie 失效时才需要账号密码。文件内容由用户在本地安全配置，技能中不保留真实值。`BIGQUANT_COOKIE` 对应完整 Cookie 字符串，脚本仅发送其中的 `bigjwt`；其他浏览器追踪 Cookie 不需要。

```powershell
# 使用已有环境变量，或自动发现被忽略的 .bigquant/platform-auth.json。
.venv/Scripts/python.exe .agents/skills/jycapital-scores/scripts/query-scores.py

# 使用用户已配置的其他认证文件；文件路径可含空格。
.venv/Scripts/python.exe .agents/skills/jycapital-scores/scripts/query-scores.py --auth-file '本地认证文件.json'

# 成绩 JSON 仅包含经筛选的提交字段，可保存到忽略的输出目录。
.venv/Scripts/python.exe .agents/skills/jycapital-scores/scripts/query-scores.py --output outputs/platform-scores/latest.json
```

`--auth-stdin` 从标准输入读取同一格式的认证 JSON；不提供直接携带密码的命令行参数。登录响应和原始提交响应不写盘，错误信息不回显响应正文或凭据。缺少认证返回非零退出码。

## 登录

`POST https://bigquant.com/bigapis/auth/v1/auth/login`

- `Content-Type: application/json;charset=UTF-8`。
- JSON 结构：`{"payload":{"username":"本地提供","password":"本地提供"},"auth_name":"UserPass"}`。
- 已验证的响应为 `{code, data: {token, user, ...}}`，并设置 `bigjwt` Cookie。脚本优先读取该 Cookie，也兼容 `data.token`；不输出 `data.user`。
- 账号密码仅发往上述固定 HTTPS 登录地址。拒绝 HTTP 重定向，保持 TLS 校验；不会把认证信息跟随重定向发送到其他站点。
- 每次命令最多登录一次。仅 HTTP 或业务认证状态 401/403 触发已有 Cookie 的登录回退；其他业务错误、429、网络错误不自动重试或切换账号。

## 提交列表

`GET https://bigquant.com/bigapis/alphathon/v1/submissions`

| 参数 | 脚本值 |
| --- | --- |
| `competition_id` | 默认进益资本比赛 ID，可显式覆盖 |
| `order_by` | `-created_at` |
| `constraints` | `{}`，通过 URL 编码传输 |
| `page` | 从 1 开始逐页递增 |
| `size` | 默认 100，可用 `--page-size` 调整，最多 1,000 |

响应结构为 `{code, reason, message, data: {page, size, total, count, items}}`，其中 **`total` 是总页数，`count` 是当前页条数**。脚本逐页取到 `total`，核对 `count == len(items)` 并去重校验，完成后再计算提交总数；遇到重复页、总页数变化、缺页、页数上限或未知响应结构会明确失败，不将截断列表报告为全部记录。默认最多 100 页，可用 `--max-pages` 调整；单次网络请求超时默认 30 秒。

每条记录的已验证字段：

| 字段 | 含义与处理 |
| --- | --- |
| `id` | 平台提交 ID，用于本地绑定 |
| `created_at`、`updated_at` | 平台返回的时间原值 |
| `competition_id` | 必须与请求比赛一致 |
| `data.files` | 文件 ID 到 `{name, size}` 的映射；输出仅保留文件名/大小 |
| `data.description` | 用户文本，仅在内存中处理；查询输出不保留，避免夹带个人资料 |
| `public_score`、`private_score` | 原始总分；保留字符串/空值，另给数值解析结果 |
| `public_score_data`、`private_score_data` | 分项指标对象 |
| `selected_for_private` | 是否选择为私榜候选，查询不修改 |

已验证的分项字段为 `ic_ir`、`ic_mean`、`turnover`、`sharpe_ratio`、`stress_stability`。脚本另支持这些字段对应的 IC/Rank IC、排名等数值字段，但不推断其含义。新的未识别指标只报告字段名，需核实后更新白名单，避免静默遗漏。

`user_id`、登录用户资料、描述、内部文件 ID、链接、错误堆栈和未知嵌套内容均不输出。总分与分项均不乘 100、不自创排序方向；公榜总分如何计算以平台规则为准。接口未提供榜单版本/评估日期时标记未知，不用提交时间代替评估区间。

## 输出与筛选

输出 JSON 包含 `source`、接口地址、比赛 ID、带 UTC+08:00 偏移的 `queried_at`、获取页数、`total_pages`、`total`、筛选后的数量和 `submissions`。输出的 `total` 是完整获取后计算的提交总数，接口原始总页数保存在 `total_pages`。`--submission-id` 在完整获取后筛选；找不到指定 ID 时返回非零退出码，不输出一个看似成功的空成绩。

`--output` 写入筛选后的成绩 JSON；未指定时打印到标准输出。脚本不修改实验索引，回填由技能按已确认的本地版本映射执行。
