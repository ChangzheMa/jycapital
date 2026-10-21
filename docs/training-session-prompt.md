# 训练监控 session 提示词

[持续研究约定](continuous-research.md) · [文档目录](README.md)

将下面整段发给当前已经调度 v201／v301 的 session：

```text
你是 E:\src\jycapital 的唯一正式训练与监控 session。另一个研究 session 持续在 solutions/<version>/ 构造模型；本轮目标是尽可能提高平台分数，持续到我叫停。请从只运行 v201／v301 的固定队列，升级为持续扫描新候选的串行调度。

先读取 AGENTS.md、仓库 .agents/skills/jycapital-experiment/SKILL.md、docs/development.md、docs/continuous-research.md、experiments/research-policy.json。用户已经授权持续调度后续 ready 候选，每个候选一次正式训练；无需逐版再次确认。运行目录和模型系列沿用 solutions/、outputs/runs/ 的既有约定。

用户于 2026-10-21 要求研究端加大组合改进力度，并允许适当扩展模型和训练时间。后续方案可能约 100 万—500 万参数、预计 6—12 小时；这是设计参考，不是自动终止阈值。你仍按每版冻结配置执行 GPU 预检、训练及早停，不自行缩小模型或延长轮数；新方案的文献筛选和构造由研究 session 负责。既有 ready／在训版本保持原配置。

用户同日进一步要求停止主动查平台分：平台反馈由用户同步，两个 session 均继续围绕本地训练、评估和材料准备工作；例行检查不调用平台查分接口。研究 session 的例行检查间隔已改为 3 小时，以下训练 session 自身的调度与报告频率保持原约定。

1. 接管当前 v201／v301 的真实状态和已有控制器，保留在训进程、checkpoint、优化器、随机状态与早停计数。先防止旧控制器与新扫描控制器重复拉起任务。已经完成的 v201 不重训；v301 若仍在运行则等待其完成训练、验证及测试。不要在本任务之外再启动第二个 GPU 训练。

2. 在每版完成后、启动下一版前扫描 solutions/*/experiment.json。使用 .venv-training/Scripts/python.exe scripts/research-queue.py status 查看队列，使用 next 获取下一个可执行候选。只启动 status=ready、代码指纹与实现检查相符、没有完成或正在运行的同一 run_id 的候选；draft 不启动，retired 不启动。按 priority 升序、版本序排序。没有 ready 候选时保持后台监控，每小时扫描；后续新增 ready 候选自动接入，不在 v301 结束后退出整个任务。

3. 使用候选清单指定的训练入口、配置和 run_id；复用 .venv-training、现有训练／留出集预处理、共享 resident cache 及已验证的 evaluation cache。先做可用内存与显存检查，默认使用保留内存的流式验证缓存模式；不要为抢速度引发换页或同时构建多个大缓存。GPU 批量预检放在正式训练间隙，运行 .venv-training/Scripts/python.exe scripts/check-candidate-device.py --solution-id <version> --device cuda，检查默认训练 batch 的前向、反向及默认推理 batch；预检不拟合正式行情。

   新输入契约不能强行复用旧缓存。若候选声明独立缓存（例如 v901 的当天 5／30 分钟输入），沿用候选自己的加载入口和格式；其入口可把旧格式路径参数转到独立新键目录，但不得改写旧目录。交接及资源统计读取该 run 的 resident-cache.json 与事件记录，不把控制器的全局 CACHE 常量当作所有模型的实际缓存。初次构建时间必须与训练时间分别记录。

4. 运行前保存实际源码、配置、协议、数据指纹和环境；仓库有未提交内容时如实记录。研究 session 会继续新增目录，已运行版本使用自己的冻结快照。不要因为其他版本或无关工具新增而把现有运行偷偷切换到新代码。若恢复检查发生冲突，保留原运行并报告，由冻结源码恢复或使用已登记的执行路径升级，不通过忽略指纹继续。

5. 采用候选的训练目标与配置，验证／测试保持固定的 local-v2-ic 评估协议，checkpoint 按验证 IC 选取，使用仓库动态早停。每个训练完成的候选都完成验证和测试，包括未改善版本；测试不选 epoch。保存 predictions、cross-sections、monthly、metrics、history、selection、best/last checkpoint 及资源记录。核对输出有限、键唯一、覆盖率、未定义截面和权重哈希。manifest 的 status、stage 及双集产物都完整才算完成。
   联合损失按该版配置与源码核验，不能套用旧版的损失恒等式。例如 v1001 还包含分支分配及路由交叉熵，并在有效样本上按训练日期权重归一化；交接保留 stock-day-training.json 中的时间权重指纹，以及逐轮分支损失、路由概率和熵。合成 GPU 预检使用单位权重，日期权重的入口验证见该版 readiness；正式训练始终传入真实训练日期权重。

6. 正常进度留在后台日志，仅每小时、用户询问、版本完成或异常时报告。一个候选完整结束后自动扫描并继续下一候选。原运行中断只能沿原配置和原状态恢复；改变模型、损失、数据或另开种子属于新的尝试，不能擅自改参数或重试覆盖失败。无法恢复的候选标记 needs_attention，记录原因；允许调度其他独立且 ready 的候选，不让失败版本无限重试。

7. 每个候选完整完成后，把实际结果保存在 outputs/checks/continuous-research/<run_id>.json，包含 run_id、solution_id、状态、父版本、源码/配置/权重指纹、最佳与完成轮次、参数量、训练/验证/测试耗时、资源、验证与测试全部指标、逐月产物、异常与局限。可使用 scripts/research-queue.py record --run-dir outputs/runs/<run_id> 生成基础交接记录。分析 session 会据此决定后续结构和平台候选；你不修改研究方案，不自动导出、提交平台或 Git 推送。

8. 研究 session 会在未评测候选超过 3 个时停止构造，并每 3 小时例行检查，少于 3 个才继续；完成及异常通知仍及时处理。你继续串行消化已有队列。平台每天 5 次仅是平台测试额度，不限制本地训练数量。平台提交由我执行。本 session 持续扫描和训练直到我明确叫停，不以某个模型得分不错或队列暂时为空作为完成条件。

请先核对当前进程与 v201／v301 状态，报告接管方式和下一次扫描安排，然后持续执行。后台通知只对本 session 使用；不能把“消息已发送”当成已通过空闲自动唤醒验收。
```
