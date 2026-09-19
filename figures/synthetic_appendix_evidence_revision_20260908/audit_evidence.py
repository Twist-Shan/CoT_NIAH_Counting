"""Recalculate donor matching from archived trials; no model execution."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[1]
DATA = WORK / 'synthetic/work/v58_final/analysis'
PRIOR = WORK / 'figures/synthetic_appendix_paper_checked_20260908'
SOURCES = {}

def read(path, **kwargs):
    SOURCES[str(path.relative_to(WORK))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path, **kwargs)

def main():
    (OUT / 'plot_data').mkdir(exist_ok=True)
    summaries, pairs = [], []
    for mode in ['nonthinking', 'thinking']:
        d = read(DATA / f'v58_alignment_supplement_20260905/{mode}/trials.csv')
        c = d[d.family.eq('source') & d.arm.eq('clean')].copy()
        assert len(c) == 100 and not c.duplicated(['block', 'count']).any()
        assert c.predicted_count.between(1, 10).all(), 'Non-count outputs cannot be compared by the count-only archive.'
        p = d[d.family.eq('answer') & d.arm.eq('adjacent_donor')].copy()
        assert len(p) == 720 and not p.duplicated(['layer', 'block', 'count', 'offset']).any()
        p['donor_true_count'] = p['count'] + p.offset
        lookup = c[['block', 'count', 'predicted_count', 'key']]
        p = p.merge(lookup.rename(columns={'count': 'donor_true_count', 'predicted_count': 'donor_prediction', 'key': 'donor_key'}),
                    on=['block', 'donor_true_count'], how='left', validate='many_to_one')
        p = p.merge(lookup.rename(columns={'predicted_count': 'receiver_prediction', 'key': 'receiver_key'}),
                    on=['block', 'count'], how='left', validate='many_to_one')
        assert p[['donor_prediction', 'receiver_prediction']].notna().all().all()
        assert p.key.eq(p.receiver_key).all()
        p['patched_match_truth'] = p.predicted_count.eq(p.donor_true_count)
        p['baseline_match_truth'] = p.receiver_prediction.eq(p.donor_true_count)
        p['patched_match_prediction'] = p.predicted_count.eq(p.donor_prediction)
        p['baseline_match_prediction'] = p.receiver_prediction.eq(p.donor_prediction)
        metrics = ['patched_match_truth', 'baseline_match_truth', 'patched_match_prediction', 'baseline_match_prediction']
        for layer, group in p.groupby('layer'):
            assert len(group) == 180
            summaries.append(dict(mode=mode, layer=int(layer), pairs=len(group),
                                  **{m: float(group[m].mean()) for m in metrics}))
        assert p[p.layer.eq(4)].patched_match_prediction.all()
        pairs.append(p[['mode', 'layer', 'block', 'count', 'offset', 'receiver_key', 'donor_key',
                        'donor_true_count', 'donor_prediction', 'receiver_prediction', 'predicted_count'] + metrics])
    summary = pd.DataFrame(summaries)
    pd.concat(pairs, ignore_index=True).to_csv(OUT / 'plot_data/08_donor_paired_trials.csv', index=False)
    summary.to_csv(OUT / 'plot_data/08_donor_metrics.csv', index=False)
    old = read(PRIOR / 'plot_data/08_answer_state_donor.csv')
    check = summary.merge(old, on=['mode', 'layer'], validate='one_to_one')
    assert np.allclose(check.patched_match_truth, check.adoption)

    # Retain the archived prompt-bootstrap intervals of the paired difference.
    intervals = read(PRIOR / 'plot_data/07_donor_continuation.csv')
    raw_ci = read(DATA / 'v58_unified_legacy_20260905/continuation/item_span_w2/rollout_contrasts.csv')
    raw_ci = raw_ci[raw_ci.split.eq('confirmation') & raw_ci.layer.eq(1) & raw_ci.subset.eq('all') &
                    raw_ci.contrast.eq('full_donor_patch - full_norm_orthogonal_mean')]
    merged = intervals.merge(raw_ci, on='metric', suffixes=('_old', '_raw'), validate='one_to_one')
    assert len(merged) == 4
    for col in ['pairs', 'prompts', 'treatment_mean', 'control_mean', 'effect', 'ci_low', 'ci_high']:
        assert np.allclose(merged[col + '_old'], merged[col + '_raw'])
    intervals.to_csv(OUT / 'plot_data/07_donor_continuation.csv', index=False)

    dyn = []
    for mode in ['nonthinking', 'thinking']:
        b = read(DATA / f'v58_unified_legacy_20260905/{mode}/dynamics_behavior_trials.csv',
                 usecols=['step', 'prompt_sha256', 'ar_accuracy', 'trace_exact'])
        assert b.groupby('step').size().eq(100).all()
        assert b.groupby('step').prompt_sha256.apply(set).apply(lambda x: x == set(b.prompt_sha256)).all()
        s = b.groupby('step', as_index=False).agg(inputs=('ar_accuracy', 'size'), free_count_accuracy=('ar_accuracy', 'mean'), trace_exact=('trace_exact', 'mean'))
        s['mode'] = mode
        dyn.append(s)
    behavior = pd.concat(dyn, ignore_index=True)
    ncc = read(PRIOR / 'plot_data/10_ncc_dynamics.csv')
    final_ncc = ncc[ncc.endpoint.str.contains('answer_query')]
    dynamics = behavior.merge(final_ncc[['mode', 'step', 'layer', 'ncc_correct']], on=['mode', 'step'], validate='one_to_one')
    dynamics.to_csv(OUT / 'plot_data/10_decoding_vs_generation.csv', index=False)
    step400 = dynamics[dynamics['mode'].eq('thinking') & dynamics.step.eq(400)].iloc[0]
    assert np.isclose(step400.ncc_correct, 1) and np.isclose(step400.free_count_accuracy, .12)
    transport = read(PRIOR / 'plot_data/03_value_transport.csv')
    v4 = transport[transport.top_k.eq(4) & transport.condition.isin(['value_selected', 'value_control'])]
    roles = read(PRIOR / 'plot_data/09_attention_roles.csv')
    successor = roles[roles['mode'].eq('thinking') & roles.layer.eq(2) & roles['head'].eq(3) & roles.step.isin([0, 100, 200, 400, 1500, 10000])]
    verdict = dict(donor_metrics=summary.to_dict('records'), continuation_intervals=intervals.to_dict('records'),
                   thinking_step400=step400.to_dict(), value_top4=v4.to_dict('records'),
                   successor_L2H3=successor[['step', 'successor', 'targeted']].to_dict('records'),
                   source_sha256=SOURCES, model_runs=False,
                   metric_definition='Patched or clean receiver prediction matches donor ground-truth count or clean prediction on identical directed pairs.',
                   invalid_donor_handling='All clean donor predictions are valid count tokens; no invalid-token identities need reconstruction.',
                   uncertainty='Existing 95% percentile bootstrap of prompt-averaged paired differences, 10,000 resamples, seed 20260905; no new significance tests.')
    (OUT / 'evidence_audit.json').write_text(json.dumps(verdict, indent=2), encoding='utf-8')
    print(summary.to_string(index=False))
    print('\nThinking step 400:', step400.to_dict())
    print('\nValue Top-4:', v4[['condition', 'margin']].to_dict('records'))
    print('\nDynamics:', dynamics[dynamics['mode'].eq('thinking')].to_string(index=False))
    print('\nL2H3:', successor[['step', 'successor', 'targeted']].to_string(index=False))

if __name__ == '__main__':
    main()
