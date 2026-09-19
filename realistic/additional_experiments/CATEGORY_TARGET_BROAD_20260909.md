# Non-thinking category：目标类别 Broad

范围：数 city 的题只对 city 记录计算 Broad，数 flower 的题只对 flower 记录计算 Broad。本版本为 `category_target_broad_20260909_v1`，只改变 Non-thinking category 的选头指标范围。原v3和全部Native、Targeted、kth输出保留。

## 冻结设置

- 原始category输入、问题、gold、token IDs和最终回答query保持原样。每模型300输入，city / flower各150。
- discovery seeds 1234–1253，每seed十题，共200题；confirmation seeds 1254–1263，共100题。沿用已有划分，这次变体是在观察原结果后提出，按探索性后续实验解释。
- 每题按 `record.category == case.target_category` 筛选原文完整记录spans，核对is_target与gold；M为这些记录的attention总质量，H为这些记录间归一化分布的熵，J为这些记录的数量，取1/3/5/7/9。分数B=M×exp(H)/J。其他类别不进入M、H或J。
- 在同一category任务内合并两类问题：先seed内平均、再seed等权。每模型重新采集discovery attention、重新全局排序及冻结Top-K，不使用旧Broad头名单。
- Qwen K=1/2/4/8/16/32/64/128；Gemma K=1/2/4/6/8。每层随机头数匹配Selected；优先不重叠，仅容量不足时最低必要重叠，三组random seeds=7000–7002。
- 在最终回答query的一次prefill处置零pre-O切片，最多继续64 tokens，保持SDPA、模型修订与原生成配置。Clean、Selected和Random全部重新生成，按题目要求的类别数量精确评分。
- 小规模验证固定confirmation首个seed中city/flower各自目标数量1与9的四题，检查最小与最大K；双模型通过评分、完整discovery重算、对照几何及Clean基线一致性核验后，运行完整1300个case×K点。

## 统计和输出

准确率与Random−Selected、Clean−Selected按seed等权，配对bootstrap 20,000次，seed=20260907，逐点95%区间。主分析合并两类问题；city、flower分组单列描述性准确率和效应。Clean-correct为补充。精确双侧seed sign-flip检验，主分析和Clean-correct各自在本变体双模型全部13个K比较中做Holm校正；不得将其与旧版本52个Broad比较的校正p直接当作同一比较族。

与旧“全部记录”版本使用同一confirmation输入和K，逐样本核对Clean准确率一致；额外报告新旧Δ的seed配对差值及描述性区间。峰值K只作事后探索性汇总，完整曲线保留。

运行目录 `additional_experiments/runs/category_target_broad_20260909_v1`。`package`保存哈希冻结代码、plans、基线表和协议；`discovery`保存所有记录的attention mass以及实际目标类别mask，便于独立重算；`banks`保存完整排序和对照；`canary/full`保存生成与评分；`analysis`保存逐样本表、汇总和核验。大文件不提交git。

本地准备：运行 `python -m unittest test_category_target_broad`（在additional_experiments目录），再运行 `python additional_experiments/deployment/prepare_category_target_broad.py`。部署使用独立版本目录；不覆盖旧结果。完成后逐样本重评分、核验两份HTML及全部曲线来源。

## 完成与复核说明（2026-09-09）

全量完成并已交付10图HTML，本地重算discovery分数时检测到浮点末位差异；使用显式 `--score-comparison ulp8` 仅允许非负有限binary64分数最多8个可表示数间隔的误差，零值仍须精确一致。实际297600个分数中847个不逐位相同，最大4个间隔、绝对差1.11e-16。两模型重新计算的完整头顺序、所有Top-K及冻结对照精确一致，6500条输出重评分和四份分析表均一致。此项为完成后的跨平台数值复核政策，冻结协议、代码和GPU输出保留原样；不能将其描述为所有分数跨平台逐位一致。


Anonymous release note: This technical protocol is retained for package construction. Deployment/account records are excluded; it is not the byte-identical historical protocol artifact.
