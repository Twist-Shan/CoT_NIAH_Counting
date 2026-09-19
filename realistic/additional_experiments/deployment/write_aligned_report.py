import json,csv
from pathlib import Path
b=Path(__file__).resolve().parents[1]
rows=[]
for version,model in [('v2','Qwen3-8B'),('v3','Gemma4-E4B')]:
 a=json.loads((b/f'runs/aligned_transfer_20260906_{version}/analysis/audit.json').read_text(encoding='utf-8'))
 rows.extend(dict(r,version=version) for r in a['summary'] if r['model']==model)
out=b/'ALIGNED_TRANSFER_RESULTS_20260907.md'
intro='''# kth 与混合 topic：正文取点语义下的检索干预

状态：全量运行、结构审计、评分复核和已发现的 backend bug 修复已完成；以下结论包含明确的历史 trace 来源限制。Qwen 使用 v2，Gemma 使用修复后的 v3。旧 Gemma v2 结果保留用于诊断，不进入主表。

## 结论与实验目的

实验检验正文的 Broad retrieval 与 Targeted retrieval 干预能否迁移到 kth 和 city/flower 筛选计数任务。两模型、两任务的 Native Targeted 消融均降低下一注册记录的正确率，且效应超过同层同数量随机头对照。Non-thinking Broad 在 kth 中有正向效应；混合 topic 中 Qwen 有正向效应，Gemma 的区间跨零。Native Broad 在当前最终回答单 query 干预下效应接近零。

这些结果适合加入 additional experiments，作为局部检索干预的迁移证据。结论限于冻结自然 trace 上的取点与短续写；尚未证明完整推理过程中的必要性、唯一性或整题准确率收益，也没有测试 count representation。

## 定义、数据与对照

每任务每模型300输入，每输入两种模式。kth 的 k=1..10 在30个 seeds 内均匀覆盖；混合 topic 每段10条记录，city数量1/3/5/7/9，分别询问city或flower。沿用原 Excerpt/audit needle 格式。完整实际 prompt 和原生成保存在冻结捕获的 prompt.json/generation.json；本次通过 plans 中 source 路径及 SHA256 引用，未改写提示词或重新选择自然输出。

Seeds 1234–1253用于discovery，1254–1263用于confirmation。每模型完成1200 attention文件、400 causal文件；文件包含不可用端点的原因，不能把文件总数当作有效干预样本数。主表每行100个候选confirmation输入，n为有效端点数量，缺失=100−n。缺失按端点规则保留，无按正确率筛样本。

Broad：Non-thinking针对原文needle spans选头；Native针对正文parser注册序列的item-end tokens选头。均只在最终答案query消融，最多生成64 tokens。Broad头数Qwen Non-thinking128、Native32，Gemma均6；用discovery选头，每层最多一半heads以构造不重叠随机对照。Qwen128为先前头数探索后的工作设定，带有事后选择限制。

Targeted：仅Native，取注册序列中可用转移的较早中位位置；Qwen使用可用的post-marker，否则上一项P0，Gemma使用P0。从当前query持续消融后续decode，保留此前完整前缀，最多128 tokens。目标是下一注册记录；使用正文first semantic record规则并扩展city/flower audit句。复用冻结Qwen128/Gemma6头集合。部分端点来自复述序列，不能全部解释为第一次检索。

每端点运行clean、selected和3组与selected不重叠、同层同数量的random。Broad准确率为短答案正确率，Targeted准确率为首semantic记录命中率，两类指标不可直接比较。先在每seed内平均，再对10个seeds等权平均。额外失败率Δ=random准确率−selected准确率（百分点）；正数代表选定头消融损伤更大。区间为20000次seed bootstrap的95%百分位区间，随机种子20260906，未作多重比较校正。说明性例子：random正确率80%、selected50%，则Δ=30个百分点。

## 主结果

准确率单位为%；Δ和区间单位为百分点。每行统计单位为10个confirmation seeds。

| 模型 | 任务 | 模式 | 干预 | n/100 | Clean | Selected | Random | Δ [95% CI] |
|---|---|---|---|---:|---:|---:|---:|---:|
'''
table=[]
for r in rows:
 table.append(f"| {r['model']} | {r['task']} | {r['mode']} | {r['assay']} | {r['n']}/100 | {r['clean']:.2f} | {r['selected']:.2f} | {r['random']:.2f} | {r['extra_failure_pp']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}] |")
tail='''

## 审计、修复与解释范围

Gemma旧实现逐个保存并修改root/text attention配置，root setter会先改变text配置，导致恢复时留下eager backend。修复先保存所有配置，再修改并在finally恢复。v3在隔离目录重新运行全部Gemma attention/causal；保留全部原始输入和v2证据。两个canary任务通过8 attention/4 causal及clean/hook门槛后才进入全量。两版各233个代码文件与各自contract哈希一致。

v2审计验证4800 source hashes、16个discovery排序及5550次canary/full续写结构；v3验证2400 source hashes、8个排序及2915次续写结构。两模型最终采用的全部344个可用confirmation Targeted端点、1720个arms已用冻结parser重评，与保存评分完全一致。随机对照已核查head唯一性、不重叠、同层数量、hook调用；Broad随机集合与selection逐项一致。Targeted随机集合未额外独立重放其RNG序列。

Qwen有64条Targeted clean token后缀差异，Gemma v3有7条Targeted及4条kth Non-thinking Broad差异。将原始Targeted后缀按相同128 tokens解码并用同一规则评分后，344例中仅Qwen category_count_seed1261_city7_targetcity改变首记录和正确性：clean给iris补上score，首记录被评为iris，目标Tallinn随后出现。该错误完整保留。Qwen此样本两模式各3次重复，包括attention读取之后，均与保存clean逐token一致；未复现状态污染。其余token措辞差异的具体数值原因未查明，不将其统一归因于浮点误差。

Gemma四例kth Non-thinking在SDPA下各3次均复现v3 clean，在eager下各3次均复现旧自然输出。旧kth采集程序每条生成后读取attention，因此历史自然trace受到旧backend恢复bug影响。本次保留这些冻结trace，在修复后的模型上进行条件干预；结果不代表整套自然生成已经按修复后的SDPA重采。旧kth自然准确率和旧因果表不能作为纯SDPA基准。若论文需要全流程一致的SDPA结论，需要另行重采自然trace并重新注册端点；本报告不以现有条件干预替代该实验。

Targeted达到128-token上限较常见，评分只读取第一条semantic记录，不代表完整trace完成。各arm逐组截断计数保存在audit.json；Broad均无截断。尤其Gemma mixed-topic selected常快速结束，短续写中未出现目标也计为失败；本实验不区分局部检索损伤与更广泛的生成损伤。Native Broad的零效应只说明此最终query干预未改变答案，不能推出Broad retrieval在此前推理中没有作用。

本设计与正文对齐的是注册端点、取点位置和干预语义。固定20/10 discovery-confirmation划分与正文部分Broad跨seed留一设计不同；Targeted头来源、部分复述端点和冻结历史trace也构成解释边界。没有新增类别判断分解实验。

## 复现与证据

- 冻结协议与诊断记录：[ALIGNED_TRANSFER.md](ALIGNED_TRANSFER.md)。
- Qwen最终统计：[v2 audit](runs/aligned_transfer_20260906_v2/analysis/audit.json)、[summary.csv](runs/aligned_transfer_20260906_v2/analysis/summary.csv)（该文件亦含已弃用Gemma v2行，须按model筛选）。
- Gemma最终统计：[v3 audit](runs/aligned_transfer_20260906_v3/analysis/audit.json)、[summary.csv](runs/aligned_transfer_20260906_v3/analysis/summary.csv)。
- 合并最终表：[aligned_final_summary.csv](aligned_final_summary.csv)。数值保留完整精度，报告显示两位小数。
- 严格原始后缀评分：[original_target_score_audit.json](runs/aligned_transfer_20260906_v3/analysis/original_target_score_audit.json)。
- Qwen重复诊断：[clean_gpu_qwen_diagnostic.json](runs/aligned_transfer_20260906_v2/analysis/clean_gpu_qwen_diagnostic.json)。
- Gemma SDPA/eager诊断：[SDPA](runs/aligned_transfer_20260906_v3/analysis/clean_gpu_kth_diagnostic.json)、[eager](runs/aligned_transfer_20260906_v3/analysis/clean_gpu_kth_eager_diagnostic.json)。
- 本地审计脚本：deployment/audit_aligned_results.py（v2）、deployment/audit_aligned_gemma_v3_results.py（v3）；远端CPU重评分脚本deployment/audit_original_target_scores.py。
- v2归档SHA256：1f4ad52fd9f283cbb7f84a0921485e8e3cd2d9cb9b8a36ad5d224b8b6f3fba2f。
- v3归档SHA256：42494c8c72704c0ef25768539a51ead71e345eb022412046763b111d5e4b3f6b。

原始audit的CLEAN_MISMATCH_REVIEW标签保留，表示token不完全一致；本报告记录后续评分复核与backend诊断，不将其改写成逐token一致。所有运行和诊断已结束，实例保持不变。
'''
out.write_text(intro+'\n'.join(table)+tail,encoding='utf-8')
with (b/'aligned_final_summary.csv').open('w',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
assert len(rows)==12
print(out)
