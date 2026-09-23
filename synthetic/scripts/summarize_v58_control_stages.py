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
    'clean': 'Clean', 'all': 'All stages', 'original_query': 'Initial Ans query only',
    'trace_phase': 'Trace stage only', 'output_phase': 'Output stage only',
    'retrieval': 'trace Sep query', 'trace_other': 'Other trace queries',
    'bridge': 'End of trace to Ans', 'answer_marker': 'Every Ans query',
    'post_answer': 'Other queries after answer',
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
        row = {'Intervention scope': SCOPE_NAMES[scope]}
        for arm, name in [('selected', 'Selected'), ('control', 'Control')]:
            f = t.loc[t.arm.eq(arm) & t.top_k.eq(4) & t.scope.eq(scope)].iloc[0]
            row[name + ' answer (%)'] = 100 * f.ar_accuracy
            row[name + ' trace (%)'] = 100 * f.trace_exact
        bank_rows.append(row)
    nt_rows = []
    for label, hs in [('L1H2', 'L1H2'), ('Top-4 control', 'L1H1;L1H2;L1H5;L1H7'), ('Top-4 selected', 'L1H0;L1H3;L1H4;L1H6')]:
        for scope in ['all', 'original_query', 'answer_marker', 'post_answer']:
            f = nt.loc[nt.heads.eq(hs) & nt.scope.eq(scope)].iloc[0]
            nt_rows.append({'Heads': label, 'Intervention scope': SCOPE_NAMES[scope], 'Answer (%)': 100 * f.ar_accuracy,
                            'EOS (%)': 100 * f.eos_reached, 'Correct + EOS (%)': 100 * f.answer_and_eos_correct})
    single = t.loc[t.top_k.eq(1)].pivot(index='heads', columns='scope', values='ar_accuracy').mul(100)
    single = single.reindex(columns=t_scopes).reset_index()
    single.to_csv(OUT / 'thinking_single_head_answers.csv', index=False)
    single_trace = t.loc[t.top_k.eq(1)].pivot(index='heads', columns='scope', values='trace_exact').mul(100)
    single_trace.reindex(columns=t_scopes).to_csv(OUT / 'thinking_single_head_traces.csv')


if __name__ == '__main__':
    main()
