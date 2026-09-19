"""Descriptive summaries for the query-stage control audit, without new tests."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work/v58_control_stage_audit_20260908'
ALIGN = ROOT / 'work/v58_final/analysis/v58_alignment_supplement_20260905'
SCOPE_NAMES = {
    'clean': 'Clean', 'all': '全程', 'original_query': '仅最初 Ans query',
    'trace_phase': '仅 trace 阶段', 'output_phase': '仅输出阶段',
    'retrieval': 'trace Sep query', 'trace_other': 'trace 其他 query',
    'bridge': '结束 trace 到 Ans', 'answer_marker': '每个 Ans query',
    'post_answer': '答案后其他 query',
}


def table(frame):
    lines = ['| ' + ' | '.join(map(str, frame.columns)) + ' |', '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        vals = ['—' if pd.isna(v) else f'{v:.1f}' if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines)


def decode_events(frame):
    clean = frame.loc[frame.scope.eq('clean')].set_index('key').generated_tokens.to_dict()
    events = []
    for r in frame.loc[~frame.scope.eq('clean')].itertuples():
        tokens = r.generated_tokens.split()
        base = clean[r.key].split()
        stop = tokens.index('<Ans>') + 1 if r.mode == 'nonthinking' else tokens.index('<Think>') + 1
        first = r.first_changed_token
        before, after, kind = '', '', 'unchanged'
        if pd.notna(first):
            j = stop + int(first)
            before = base[j] if j < len(base) else '[ended]'
            after = tokens[j] if j < len(tokens) else '[ended]'
            if before == '<Sep>' and after == '</Think>':
                kind = 'close_instead_of_next_item'
            elif before == '</Think>' and after == '<Sep>':
                kind = 'next_item_instead_of_close'
            elif before.startswith('<CH_') and after.startswith('<CH_'):
                kind = 'changed_character'
            elif after == '<Ans>' and r.first_changed_query_stage == 'answer_marker':
                kind = 'duplicate_answer_marker'
            else:
                kind = 'other'
        events.append({'mode': r.mode, 'arm': r.arm, 'top_k': r.top_k, 'repeat': r.repeat,
                       'heads': r.heads, 'scope': r.scope, 'key': r.key,
                       'first_changed_query_stage': r.first_changed_query_stage,
                       'first_clean_token': before, 'first_ablated_token': after, 'first_change_kind': kind,
                       'true_count': r.count, 'answer_count': r.ar_pred_count,
                       'trace_marker_count': r.trace_generated_marker_count,
                       'answer_matches_trace_length': float(pd.notna(r.ar_pred_count) and r.ar_pred_count == r.trace_generated_marker_count) if r.mode == 'thinking' else np.nan,
                       'trace_too_short': float(r.trace_generated_marker_count < r.count) if r.mode == 'thinking' else np.nan,
                       'trace_too_long': float(r.trace_generated_marker_count > r.count) if r.mode == 'thinking' else np.nan,
                       'trace_length_correct': float(r.trace_generated_marker_count == r.count) if r.mode == 'thinking' else np.nan,
                       'trace_format_valid': r.trace_format_valid,
                       'ar_accuracy': r.ar_accuracy, 'eos_reached': r.eos_reached})
    return pd.DataFrame(events)


def main():
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'complete'
    for name, h in manifest['files'].items():
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == h, name
    combined, events = [], []
    for mode in ['nonthinking', 'thinking']:
        summary = pd.read_csv(OUT / mode / 'generation_summary.csv').fillna({'heads': ''})
        summary.insert(0, 'mode', mode)
        combined.append(summary)
        detail = pd.read_csv(OUT / mode / 'generation_trials.csv').fillna({'heads': '', 'first_changed_query_stage': ''})
        assert not detail.duplicated(['key', 'heads', 'scope']).any()
        assert set(detail.groupby(['heads', 'scope']).size()) == {100}
        events.append(decode_events(detail))
    summary = pd.concat(combined, ignore_index=True)
    events = pd.concat(events, ignore_index=True)
    events.to_csv(OUT / 'first_divergence_trials.csv', index=False)
    summary.to_csv(OUT / 'all_stage_results.csv', index=False)
    mean = summary.groupby(['mode', 'arm', 'top_k', 'scope'], dropna=False)[
        ['ar_accuracy', 'trace_exact', 'trace_marker_count_accuracy', 'eos_reached', 'answer_and_eos_correct']].mean().reset_index()
    mean.to_csv(OUT / 'stage_results_control_means.csv', index=False)
    counts = events.groupby(['mode', 'heads', 'scope', 'first_change_kind'], dropna=False).size().rename('inputs').reset_index()
    counts.to_csv(OUT / 'first_divergence_counts.csv', index=False)
    stop_metrics = events.groupby(['mode', 'heads', 'scope'], dropna=False)[
        ['answer_matches_trace_length', 'trace_too_short', 'trace_too_long', 'trace_length_correct', 'trace_format_valid']].mean().reset_index()
    stop_metrics.to_csv(OUT / 'trace_length_diagnostics.csv', index=False)

    t = summary.loc[summary['mode'].eq('thinking')]
    nt = summary.loc[summary['mode'].eq('nonthinking')]
    t_scopes = ['all', 'trace_phase', 'output_phase', 'retrieval', 'trace_other', 'bridge', 'answer_marker', 'post_answer']
    bank_rows = []
    for scope in t_scopes:
        row = {'干预范围': SCOPE_NAMES[scope]}
        for arm, name in [('selected', 'Selected'), ('control', 'Control')]:
            f = t.loc[t.arm.eq(arm) & t.top_k.eq(4) & t.scope.eq(scope)].iloc[0]
            row[name + ' answer (%)'] = 100 * f.ar_accuracy
            row[name + ' trace (%)'] = 100 * f.trace_exact
        bank_rows.append(row)
    nt_rows = []
    for label, hs in [('L1H2', 'L1H2'), ('Top-4 control', 'L1H1;L1H2;L1H5;L1H7'), ('Top-4 selected', 'L1H0;L1H3;L1H4;L1H6')]:
        for scope in ['all', 'original_query', 'answer_marker', 'post_answer']:
            f = nt.loc[nt.heads.eq(hs) & nt.scope.eq(scope)].iloc[0]
            nt_rows.append({'Heads': label, '干预范围': SCOPE_NAMES[scope], 'Answer (%)': 100 * f.ar_accuracy,
                            'EOS (%)': 100 * f.eos_reached, 'Correct + EOS (%)': 100 * f.answer_and_eos_correct})
    single = t.loc[t.top_k.eq(1)].pivot(index='heads', columns='scope', values='ar_accuracy').mul(100)
    single = single.reindex(columns=t_scopes).reset_index()
    single.to_csv(OUT / 'thinking_single_head_answers.csv', index=False)
    single_trace = t.loc[t.top_k.eq(1)].pivot(index='heads', columns='scope', values='trace_exact').mul(100)
    single_trace.reindex(columns=t_scopes).to_csv(OUT / 'thinking_single_head_traces.csv')
    report = '''# Control 损伤来源：分阶段消融（2026-09-08）

本报告比较同一个 head set 在不同 query 阶段的干预结果。头、输入和评分均固定；没有根据本轮结果重新选头或删除 controls。
本轮是在此前异常现象之后开展的诊断，结论限于 v58 的一个训练种子和同一批 100 个输入。

## 目标指标与筛选口径

根据最新讨论，targeted 的目标应是对应 query 上的 next-marker 身份是否正确，broad 的目标是最终 count 是否正确。Targeted 的 marker 已正确时，后续答案、trace 终止或 EOS 出错不应扣除这一步的 marker 分数；broad 的 count 已正确时，后续 EOS 出错也不应扣除 count 分数。

本报告中的完整 trace、trace 长度和 EOS 均用于损伤诊断，不等同于固定 query 的 next-marker 正确率。本轮尚未新增固定前缀的候选限制评分。

不按各干预条件的输出是否成功分别删除输入。若目标 query 未出现或应输出目标的位置没有合法 token，应单独记录缺失／无效输出；仅在有效输出中的正确率必须同时报告有效输出率。若需要预先筛选 clean 下有效的输入，应冻结同一输入集合并用于全部条件。

为进一步分离 token 身份选择与继续／停止或格式错误，可在固定、对齐的前缀上，分别限制到合法 marker 或数字候选来计算正确率。这是候选之间的内容识别诊断，不代表自由生成成功率。任何新增评分均应与原始输出并存，保留原记录。

## 设置与范围

- 同一 step-10000 checkpoint；两种模式各 100 个输入，count 1–10 各十个。
- 全部冻结 head sets：每种模式 3 个 selected banks，以及 K=1/2/4 的 7/15/1 组同层 controls。
- 贪心生成至 EOS 或 64 个新 token，不提供 gold trace，不筛选成功样本。
- 每个条件在指定阶段持续置零对应 head 的全部输出；每次完整前缀重算时，先前干预过的位置仍被置零。其他阶段保持原计算。
- Thinking：从第一个生成的 trace Sep 开始；trace 内的 Sep queries 与其余 queries 分开；随后是 trace close 到首次 Ans 之间的过渡；最后区分 Ans-token queries 与答案后的其他 queries。
- Non-thinking：区分原始 Ans query、所有 Ans-token queries（含重复 Ans）、答案后的其他 queries。Prompt 内 Sep 均不属于干预范围。
- Trace 其他 query 包括 character-token query：其下一 token 可以是下一个 Sep 或 trace close。因此这里同时涉及写入下一项的结构与停止 trace 的决定，尚不指定某个头内部究竟如何计算。
- 答案准确率沿用原解析规则，不要求 EOS；EOS 和正确且到达 EOS 单独记录。

## 验证

完整 head banks、checkpoint 哈希和输入列表保存在运行前的 protocol.json。
新实现逐条复现两个模式所有 26 个非 clean head sets 的全程干预轨迹，以及 clean 轨迹。
Query 分组构成完整、不重叠且不会回溯改变的划分。Top-4 selected/control 在每个阶段均检查真实 hook 输入，确认只修改指定切片，运行结束后 hooks 已移除。
所有输出阶段消融均保持之前生成的 trace 与 clean 一致；Non-thinking 的答案后消融保持第一个数字 token 与 clean 一致。
原始实验、正文、appendix 和论文图未修改。

## Thinking：Top-4 banks

Clean answer 为 95%，完整 trace 为 87%。Selected 为 L4H5/H0/H1/H7；唯一同层 complement 为 L4H2/H3/H4/H6。
下表每格对应同样 100 个输入；不同阶段干预的效应不能相加。

'''
    report += table(pd.DataFrame(bank_rows)) + '\n\n'
    report += '## Thinking：各单头的答案准确率\n\n'
    report += table(single.rename(columns=SCOPE_NAMES)) + '\n\n'
    report += '完整单头 trace 结果保存在 `thinking_single_head_traces.csv`。\n\n'
    report += '## Non-thinking：关键头与 banks\n\nClean answer 为 21%，EOS 为 100%。\n\n'
    report += table(pd.DataFrame(nt_rows)) + '\n\n'
    report += '''## 错误轨迹诊断

`first_divergence_counts.csv` 比较干预轨迹与同输入的 clean 轨迹，从首次不同的输出 token 判断变化表现。`close_instead_of_next_item` 和 `next_item_instead_of_close` 分别表示相对于 clean 更早关闭 trace、或原本应关闭时继续下一项；这些事件标签以 clean 为参照，不假定 clean 总是正确。
`trace_length_diagnostics.csv` 另外使用真实 count，记录 trace 偏短、偏长、长度正确，以及最终答案是否等于生成的 trace 长度；缺失答案按不匹配计，不筛选合法输出。

Thinking 的 L4H2 在 trace 其他 query 上消融时，32 个输入首次偏离 clean：26 个将关闭 trace 改为继续，6 个将继续改为关闭；其余 68 个输入轨迹不变。仅在输出阶段消融时，100 个输入的完整轨迹均与 clean 相同。仅在 retrieval query 消融时，96 个输入轨迹不变，其余 4 个首次改变的是 character token。该定位支持将 next-marker 身份与 trace 继续／停止分开评价；完整 trace 下降不能全部解释为检索身份错误。

Non-thinking 的 L1H2 在仅最初 Ans query 干预时，答案正确率已降为 0%，但 EOS 仍为 100%；在答案后阶段干预时，答案准确率保持 clean 的 21%。Top-4 selected 在答案后阶段干预则保持 21% 答案准确率、EOS 降为 51%。因此 count 内容、目标 token 是否有效输出、最终停止也应分别记录。L1H2 的低合法数字输出率需结合此前固定 query 的候选数字诊断判断，不能仅凭自由生成轨迹认定为纯格式问题。

这些指标用于定位损伤的行为表现，不能直接证明一个头实现了纯语法功能或独立计数器。

## 数据文件

- `all_stage_results.csv`：每个 head set 和阶段的结果。
- `stage_results_control_means.csv`：全部匹配 controls 的等权平均。
- `first_divergence_trials.csv`、`first_divergence_counts.csv`：首次轨迹偏离的逐输入记录和汇总。
- `trace_length_diagnostics.csv`：长度、格式及最终答案的一致性。
- 两个模式目录中的 `generation_trials.csv`：全部原始输出；`generation_summary.csv`：原始汇总。
- `protocol.json`、`validation.json`、`manifest.json`：冻结设置、实现验证和文件 SHA256。

复现脚本为 `scripts/run_v58_control_stage_audit.py`；本报告由 `scripts/summarize_v58_control_stages.py` 生成。没有新增显著性检验。
'''
    (OUT / 'STAGE_REPORT.md').write_text(report, encoding='utf-8')
    print('Wrote', OUT / 'STAGE_REPORT.md')


if __name__ == '__main__':
    main()
