"""Summarize the frozen retrieval audit without changing primary scores."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'work'
AUDIT = WORK / 'v58_retrieval_generation_audit_20260908'
LONG = WORK / 'v58_retrieval_sustained_cap64_20260908'
OLD = WORK / 'v58_final/analysis'


def load_json(p):
    return json.loads(p.read_text(encoding='utf-8'))


def canon(s):
    return ';'.join(f'L{l}H{h}' for l, h in sorted(json.loads(s)))


def md_table(frame, digits=1):
    lines = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
             '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        values = ['—' if pd.isna(v) else f'{v:.{digits}f}' if isinstance(v, float) else str(v) for v in row]
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def main():
    for folder in [AUDIT, LONG]:
        manifest = load_json(folder / 'manifest.json')
        assert manifest['status'] == 'complete'
        for name, h in manifest['files'].items():
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == h, name
    validation = load_json(AUDIT / 'validation.json')
    hooks = load_json(AUDIT / 'hook_validation.json')
    assert hooks['status'] == 'complete'
    rows, termination, original_rows, matched_three, logits_rows = [], [], [], [], []
    for mode in ['nonthinking', 'thinking']:
        frame = pd.read_csv(LONG / mode / 'generation_summary.csv').fillna({'heads': ''})
        original = pd.read_csv(AUDIT / mode / 'generation_summary.csv').fillna({'heads': ''})
        sites = load_json(OLD / 'v58_alignment_supplement_20260905' / mode / 'frozen_sites.json')
        baseline = frame.loc[frame.arm.eq('clean')].iloc[0]
        for k in [1, 2, 4]:
            selected = frame.loc[frame.arm.eq('selected') & frame.top_k.eq(k)].iloc[0]
            control = frame.loc[frame.arm.eq('control') & frame.top_k.eq(k)]
            rows.append({'Mode': mode, 'K': k, 'Controls': len(control),
                         'Clean answer (%)': 100 * baseline.ar_accuracy,
                         'Selected answer (%)': 100 * selected.ar_accuracy,
                         'Control answer (%)': 100 * control.ar_accuracy.mean(),
                         'Selected trace (%)': 100 * selected.trace_exact,
                         'Control trace (%)': 100 * control.trace_exact.mean()})
            termination.append({'Mode': mode, 'K': k,
                                'Selected EOS (%)': 100 * selected.eos_reached,
                                'Control EOS (%)': 100 * control.eos_reached.mean(),
                                'Selected correct+EOS (%)': 100 * selected.answer_and_eos_correct,
                                'Control correct+EOS (%)': 100 * control.answer_and_eos_correct.mean()})
            hs = {canon(json.dumps(x)) for x in sites['controls'][str(k)]}
            sub = control.loc[control.heads.isin(hs)]
            assert len(sub) == len(hs)
            matched_three.append({'mode': mode, 'K': k, 'control_sets': len(sub),
                                  'selected_answer': selected.ar_accuracy, 'control_answer': sub.ar_accuracy.mean(),
                                  'selected_trace': selected.trace_exact, 'control_trace': sub.trace_exact.mean()})
            orig = original.loc[original.protocol.eq('original') & original.top_k.eq(k)]
            orig_s = orig.loc[orig.arm.eq('selected')].iloc[0]
            orig_c = orig.loc[orig.arm.eq('control') & orig.heads.isin(hs)]
            original_rows.append({'mode': mode, 'K': k, 'selected_answer': orig_s.ar_accuracy,
                                  'control_answer': orig_c.ar_accuracy.mean(), 'selected_trace': orig_s.trace_exact,
                                  'control_trace': orig_c.trace_exact.mean()})
    summary = pd.DataFrame(rows)
    summary.to_csv(AUDIT / 'sustained_results_all_controls.csv', index=False)
    pd.DataFrame(termination).to_csv(AUDIT / 'sustained_termination.csv', index=False)
    pd.DataFrame(matched_three).to_csv(AUDIT / 'sustained_results_original_control_subset.csv', index=False)
    pd.DataFrame(original_rows).to_csv(AUDIT / 'original_protocol_control_subset.csv', index=False)

    local = pd.read_csv(AUDIT / 'nonthinking/original_query_summary.csv')
    for k in [1, 2, 4]:
        s = local.loc[local.arm.eq('selected') & local.top_k.eq(k)].iloc[0]
        c = local.loc[local.arm.eq('control') & local.top_k.eq(k)]
        logits_rows.append({'K': k, 'Selected restricted count (%)': 100 * s.count_restricted_accuracy,
                            'Control restricted count (%)': 100 * c.count_restricted_accuracy.mean()})
    pd.DataFrame(logits_rows).to_csv(AUDIT / 'original_query_count_diagnostic.csv', index=False)

    current = pd.read_csv(OLD / 'v58_alignment_supplement_20260905/nonthinking/ablation.csv')
    older = pd.read_csv(OLD / 'v58_confirmation_broad_topk_hp500/free_running_detail.csv')
    current['canonical_heads'] = current.heads.map(canon)
    older['canonical_heads'] = older.heads.fillna('').map(lambda s: ';'.join(sorted(s.split(';'))))
    pairs = current.merge(older, on=['prompt_sha256', 'canonical_heads'], suffixes=('_current', '_old'))
    assert len(pairs) == 168 and pairs.generated_tokens_current.eq(pairs.generated_tokens_old).all()
    old_v20 = pd.read_csv(WORK / 'v22_free_running_topk/free_running_summary.csv')
    old_v20 = old_v20.loc[old_v20.comparison_mode.eq('nonthinking')]
    document = '''# Synthetic retrieval 消融审计（2026-09-08）

## 当前结论

原始实验可以精确复现；未发现绘图汇总、头索引、样本对齐或消融执行错误。
但结果不能支持“v58 中按 retrieval score 选出的头，在持续干预下比同层 controls 更关键”的一般性结论。
持续干预和原来的 query-local 干预测量不同范围的功能，不能直接沿用原来的机制解释。

本次没有修改原始数据、正文、appendix 或论文图。新增结果供确认后再决定如何写入论文。

## 复跑范围与评分

- GPU：用户提供的 `user@compute-host`，NVIDIA A10；没有重新训练。
- 同一 v58、step 10000 checkpoint；checkpoint 文件 SHA256 与历史 manifest 一致。
- 固定 200 个 discovery 输入和 100 个 confirmation 输入，count 1–10 各 10 个；不按正确性筛选。
- 重新计算 discovery 排名只用于核查；所有干预沿用原冻结 Top-1/2/4 头。
- 每个模式枚举全部同层、同数量且与当前 selected 集合不重叠的 controls：K=1/2/4 分别为 7/15/1 组。
- Non-thinking 从输入的 `<Ans>` query 起，Thinking 从第一个实际生成的 trace `<Sep>` query 起，所有后续 query 都持续消融，直到 EOS 或生成预算耗尽。每次完整前缀重算时，之前已干预的位置也保持置零。
- 原协议分别只屏蔽最初的 answer query，或每个实际生成的 trace separator query；两种协议均完整保存。
- 答案准确率沿用原解析规则：读取第一个 `<Ans>` 后紧接的数字。它不要求后续到达 EOS。终止行为另列，不能把“答对数字”写成“正确且完整结束”。
- 最长生成预算为 64 tokens；只延续原短预算中没有 EOS 的轨迹，已结束轨迹逐 token 保留。持续干预在全部延续步保持开启。
- 每条 control 曲线是对 head sets 的等权平均；这些 control sets 不是独立训练种子。没有新增显著性检验。

## 复现与实现验证

两个模式各 1,100 条历史记录，共 2,200 条，预测、答案评分、Thinking trace 评分及首个 EOS 之前的生成文本全部一致；两个模式的 discovery 分数最大绝对误差均为 0。
Non-thinking 的旧、新生成函数在相同输入和 head set 下也完全一致。batch size 1 与 8 的首 token 一致，数字 logits 最大差为 2.43e-5。

对五种条件、每个条件十个输入（每个 count 一个）直接检查了真实 forward：所有被声明的 head-output slices 在指定 query 范围确实置零，其他 slices 不变，prompt 范围未被误伤，运行后 hooks 已移除。详见 `hook_validation.json`。
本地与服务器旧代码的 SHA256 差异仅来自 CRLF/LF 换行；统一换行后，生成函数所在文件、评分文件和消融 hook 文件的哈希一致。

## 持续干预结果

下表所有数字均为百分比。Control 是全部可用 head sets 的平均；每组使用同样的 100 个输入。完整 trace 只对 Thinking 定义。

'''
    document += md_table(summary) + '\n\n'
    document += '''Non-thinking 的 selected Top-1/2/4 答案准确率与原 query-local 协议相同。
Thinking 持续干预后，Top-4 selected 的答案/trace 准确率为 72%/46%，唯一同层 complement 为 22%/15%；这个 control 的损伤更大。
Top-2 selected 的 trace 准确率为 62%，全部 controls 平均为 63.7%，差别很小。不能将持续干预结果表述为普遍的 retrieval-bank 特异性。

### 终止行为

'''
    document += md_table(pd.DataFrame(termination)) + '\n\n'
    document += '''将预算延长到 64 tokens 后，没有任何输入的原规则答案准确率发生变化。
Non-thinking 仍有 821/2700 个“输入 × 条件”轨迹触及预算；Thinking 的 2700 条轨迹都在预算内结束。
Non-thinking selected Top-4 有 7% 输入读出正确数字，但“正确数字且到达 EOS”为 0%，这部分限制必须随结果保留。

## 为什么 matched control 会更差

原始输出中，包含 L1H2 的 Non-thinking controls 经常生成 `<Ans> <Ans> <3> <EOS>` 这类重复 answer marker 的轨迹。
原解析器在第一个 `<Ans>` 后遇到另一个 `<Ans>`，因此记为无答案。之前“模型完全不输出数字”的解释过强，已更正。

原 query-local 协议只干预第一个 `<Ans>` 的位置，因此第二个 `<Ans>` 的 query 可以恢复正常 head output。
仅在原保存文本中跳过重复 `<Ans>` 后，当前 Top-4 control 的计数准确率为 20%；这是未持续干预时的恢复，不能替代原指标。
持续消融后，重复 query 仍被干预；该恢复消失。单独 L1H2 control 的原位置数字总概率降到约 0.0415%，说明跨 token 类型的竞争显著变化。

同时，L1H2 也会降低数字候选之间的区分能力：在最初 query、强制只比较 1–10 的 logits 时，准确率由 clean 21% 降到 12%。因此不能把 L1H2 定义成“只控制格式”的纯语法头。
数字候选诊断用相同 query 和所有输入，不依赖第二次生成，也不按是否输出合法数字筛选。它不是自由生成准确率。

'''
    document += md_table(pd.DataFrame(logits_rows)) + '\n\n'
    document += '''K=4 的数字候选诊断为 selected 7%、control 12%，但 K=1/2 并没有出现 selected 更大的损伤，且 K=4 只有一个 complement。这只支持有限范围的数值信息损伤，无法建立统一的特异性结论。

## 为什么旧版本看起来更符合预期

1. 更早的 `work/v22_free_running_topk` 实际比较 v22 Thinking 与 v20 Non-thinking。v20 Non-thinking clean 为 35%，Top-1 selected 为 22.5%，三个 controls 为 25.4%–31.7%；这个版本确实呈现 selected 更大的损伤。它使用 width 256、4 heads、count 1–30，和当前 width 512、8 heads、count 1–10 的 v58 不是同一实验设置。
2. 旧 v58 的 500-input 结果里，Top-4 selected 为 11.4%、control 为 0%；异常并非当前绘图新引入。在与当前结果交叉核对的 24 个共同输入、168 个相同输入/head-set 组合上，保存的生成文本完全一致。
3. 旧 v58 Non-thinking 由 20 个 discovery 输入选出 L1H1/H0/H3/H4；当前由 200 个输入选出 L1H3/H0/H4/H6。
4. 旧 v58 K=2 枚举 15 个 controls，其中 5 组含 L1H2；当前图只取 3 组，其中 2 组含 L1H2。原图又使用 control mean，而较早报告显示 median。这些差异会改变曲线外观。本次保留原三组子集，并另报完整枚举结果，不删除导致格式错误的 controls。

## 与正文主实验的对齐范围

这次按用户最新要求实施“持续到结束”，属于明确的新协议。
目前 `Paper_Draft/main.tex` 的主实验说明还存在不同约定：Non-thinking 的图 B 报告的是归一化输出位移，当前 synthetic 图报的是答案准确率；Thinking 主实验图 B 的现有图注写到首次识别城市为止。
因此不能直接宣称现有所有实验的干预时窗和指标已完全统一。后续需要分别明确干预起点、终点与 estimand，再写入正文/appendix。

## 文件与复现

- `validation.json`、`hook_validation.json`：复现及真实干预位置验证。
- `sustained_results_all_controls.csv`：上表结果；`sustained_results_original_control_subset.csv`：保持原三组 control 的比较。
- `nonthinking/original_query_logits.csv`：逐输入的原 query 数字 logits。
- `../v58_retrieval_sustained_cap64_20260908/`：最终持续干预逐输入记录、EOS 指标与 manifest。
- `protocol.json`：运行前冻结的方案、完整 head sets、源数据与脚本哈希。
- 新增运行脚本：`scripts/audit_v58_retrieval_generation.py`、`scripts/extend_v58_sustained_generation.py`、`scripts/verify_v58_sustained_hooks.py`。
- 本报告由 `scripts/summarize_v58_retrieval_audit.py` 从保存结果生成。

这轮只核查并复跑 retrieval Top-K 头消融。现有 successor/factorial 和 value-patching 图仍对应其原协议；论文图重排暂缓。
'''
    (AUDIT / 'AUDIT_REPORT.md').write_text(document, encoding='utf-8')
    print('Report written:', AUDIT / 'AUDIT_REPORT.md')


if __name__ == '__main__':
    main()
