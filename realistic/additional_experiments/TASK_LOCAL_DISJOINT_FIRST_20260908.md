# 新任务独立选头：随机对照优先不重叠

协议修正：优先使用与 Selected 不重叠的随机头；只有该层未选头不足以匹配头数时才允许重叠。版本为 `task_local_disjoint_first_20260908_v3`。之前全池允许重叠的 v2 已停止，原始包和部分结果保留，不与本版本合并。

## 随机对照

对某层的 H 个头及 n 个 Selected：若 H−n≥n，从 H−n 个未选头中均匀无放回抽取 n 个；否则包含全部 H−n 个未选头，再从 Selected 中均匀无放回抽取 2n−H 个。每层实际重叠恰为 `max(0,2n−H)`。例如 H=8、n=5 时重叠2个；H=8、n=4时完全不重叠。三个随机组之间可有重叠，不强制不同组合。Broad seeds=7000–7002，Targeted=6000–6002。

Selected 始终使用该新任务 discovery 的全局 Top-K，不为对照设置每层半数上限。逐层保存总头数、所选数、未选数、是否需要补足及实际重叠，并逐条件检查。主实验 Native 的 selected-excluded 控制在可行层原样适用；容量不足的层采用协议规定的最低重叠扩展。

## 任务与选择

- 600个已核验的新任务输入、2400条自然输出；保留原始 prompt、token IDs、gold 和端点。两个任务为 kth 和 city/flower category；两个模型为 Qwen3-8B 和 Gemma4-E4B。
- 每任务30 seeds，每 seed 10输入；discovery 1234–1253，confirmation 1254–1263。这个划分已用于旧分析，不是新独立确认队列。
- Broad 每个模型×任务×模式，按 `B=M×exp(H)/J` 先 seed 内平均再 seed 等权，全局排序冻结。2026-09-09 执行核对更正：Non-thinking 使用完整原文记录 span attention；Native-thinking 使用最终答案前全部已注册生成记录的末尾 token attention。原冻结副本将两者均写成完整原文 span，与实际执行有差异，冻结副本保留原样，证据见 `runs/task_local_disjoint_first_20260908_v3/category_broad_scope_audit_20260909.json`。
- Targeted 每个模型×任务，在本任务 Native discovery 的已注册下一记录事件查询处，使用目标原文完整记录的 raw attention mass；同样 seed 等权、全局排序。v3 从 discovery 重新采集和冻结，不使用旧计数任务的头名单。
- 干预位置来自各自任务的自然 trace。Broad 为其最终答案前 query；Targeted 为该输入注册序列的下一记录事件，Qwen rank-before 使用 post-marker，其余按冻结语义路由使用 P0。
- Qwen Broad K=1/2/4/8/16/32/64/128，Targeted=32/64/80/96/112/128；Gemma两类 K=1/2/4/6/8。K 网格是已冻结的新任务扫描网格，不声称所有历史主实验采用相同 K。

## 干预与评分

Broad 仅消融一次 prefill 的最终答案 query，pre-O head slices置零，最多64 tokens；评分为最终任务答案准确率（kth 城市及分数均正确，category数量正确）。Targeted 在注册 query 及后续 decode 持续消融，最多256 tokens；评分为首条有效语义记录的实体是否为下一注册记录。错误首记录不因后面纠正而改判；正确首记录不因最终答案错误而改判。

Targeted 的解析边界与主实验保持一致：仅在没有可识别语义记录、且原始目标 token IDs 从续写 offset=0 起完整匹配时使用 exact-target-prefix 兜底。原始目标 token IDs 由冻结自然生成的注册实体字符位置定位；重分词不一致时使用稳定的原始 token 前缀解码恢复边界，不替换原始 IDs。额外识别 flower-score audit 记录，避免把未知花名后面的正确实体误当首记录。保存兜底是否触发，审计重算。

## 分析与交付

每有效 confirmation 端点比较 Clean、Selected、Random三组；所有K共用同一Clean和端点。全部有效输入为主分析，Clean-correct为补充。主效应 Random−Selected，另报Clean−Selected。10个 seed 配对bootstrap 20,000次、seed20260907、逐点95%CI；精确双侧seed sign-flip，加每种assay×population内跨模型/任务/模式/K的Holm校正。Broad本次重点报告准确率；主实验还包含计数偏移指标，不能声称数值终点逐项相同。

所有干预重新运行，不拼接v2的对照或Selected输出。先双模型discovery，再双模型小规模检查，随后全量6739个输入×K点。完整重评分、头集合、逐层最少重叠、端点及统计审计通过后，更新两份 `NiaH_Additional-tasks_report.html`。



Anonymous release note: This technical protocol is retained for package construction. Deployment/account records are excluded; it is not the byte-identical historical protocol artifact.
