"""Normalize causal effects per prompt and clarify model-specific causal plots.

Immutable experiment inputs; no inference, refitting, or resampling of prompts.
Run with python -s from the workspace root.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, PercentFormatter
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
REPORT = ROOT / 'realistic/reports/v4_non-thinking_causal'
STYLE = ROOT / 'figures/nonthinking_appendix_style_20260911/build_figures.py'
spec = importlib.util.spec_from_file_location('appendix_style', STYLE)
style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)
style.OUT = OUT
MODELS, COLORS = style.MODELS, style.COLORS
INPUTS, AUDIT, SEED_ROWS = {}, {}, []


def read(path):
    INPUTS[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)


def estimate(values, label, draws=10000, test=False):
    values = np.asarray(values, dtype=float)
    assert values.ndim == 1 and len(values) >= 2 and np.isfinite(values).all(), label
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    low, high = np.quantile(boot, [.025, .975])
    result = dict(mean=float(values.mean()), ci95_low=float(low), ci95_high=float(high),
                  seeds=len(values), bootstrap_draws=draws, bootstrap_rng_seed=seed)
    if test:
        assert len(values) <= 10
        signs = 2 * ((np.arange(2**len(values))[:, None] >> np.arange(len(values))) & 1) - 1
        null = (signs * values).mean(axis=1)
        result['exact_seed_signflip_p_two_sided'] = float(
            np.mean(np.abs(null) >= abs(values.mean()) - 1e-12))
    return result


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty(len(order))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(pvalues[order] * np.arange(len(order), 0, -1)))
    return adjusted


def normalize_removal():
    base = REPORT / 'v4_4_extension/layerwise_subspace/answer_query_removal'
    raw = read(base / 'layerwise_answer_query_removal_paired_examples.csv')
    old = read(base / 'layerwise_answer_query_removal_statistics.csv')
    assert len(raw) == 2070 and not raw.duplicated(['model_label', 'seed', 'gold_count', 'layer']).any()
    assert set(raw.seed) == set(range(1254, 1264)) and set(raw.gold_count) == set(range(2, 11))
    assert np.isfinite(raw.absolute_error_specificity).all()
    assert np.allclose(raw.absolute_error_specificity,
                       raw.candidate_absolute_error_damage_from_clean - raw.control_absolute_error_damage_from_clean)
    raw['normalized_effect'] = raw.absolute_error_specificity / raw.gold_count
    raw.to_csv(OUT / 'removal_normalized_paired_examples.csv', index=False)
    rows = []
    for pop in ['all', 'clean_correct']:
        frame = raw if pop == 'all' else raw[raw.clean_correct.eq(1)]
        for (model, layer), group in frame.groupby(['model_label', 'layer'], sort=True):
            seed_effects = group.groupby('seed').normalized_effect.mean()
            assert len(seed_effects) == 10
            original = old[(old.model_label == model) & (old.population == pop)
                           & (old.layer == layer) & (old.endpoint == 'absolute_error_specificity')]
            assert len(original) == 1
            assert np.isclose(group.groupby('seed').absolute_error_specificity.mean().mean(),
                              original.mean_effect.iloc[0], atol=1e-12)
            if pop == 'all':
                assert len(group) == 90 and group.groupby('seed').size().eq(9).all()
            label = f'removal:{pop}:{model}:{layer}'
            rows.append(dict(model=model, population=pop, layer=int(layer), examples=len(group),
                             **estimate(seed_effects, label, 50000, test=True)))
            SEED_ROWS.extend(dict(analysis='removal', population=pop, model=model, layer=int(layer),
                                  metric='normalized_error_specificity', seed=int(seed), value=float(value))
                             for seed, value in seed_effects.items())
    summary = pd.DataFrame(rows)
    for _, group in summary.groupby(['model', 'population']):
        summary.loc[group.index, 'holm_p_within_model_population'] = holm(group.exact_seed_signflip_p_two_sided)
    summary.to_csv(OUT / 'removal_normalized_summary.csv', index=False)
    AUDIT['removal'] = {'normalization': '(aligned absolute error - orthogonal absolute error) / gold N, per prompt',
                        'invalid_output_error': 'The original raw error=10 convention is retained, then divided by N.',
                        'aggregation': 'Mean within seed, then equal mean across seeds.',
                        'original_aggregate_means_reproduced': True,
                        'holm_family': 'Layers within model/population; two-sided exact seed sign flip.',
                        'significant_layers_one_based': {f'{m}/{p}': (g[g.holm_p_within_model_population < .05].layer + 1).tolist()
                            for (m, p), g in summary.groupby(['model', 'population'])}}
    return summary[summary.population == 'all']


def normalize_serial():
    paths = ['v4_4_5_followup/exp19/Qwen3-8B/paired_serial_effects.csv',
             'v4_4_5_top6_followup/serial_mediation/Gemma4-E4B/paired_serial_effects.csv']
    metrics = ['source_repair', 'retrieval_mediation', 'late_mediation', 'joint_interaction', 'remaining_repair']
    old = read(ROOT / 'figures/nonthinking_appendix_20260911/causal/serial_plot.csv')
    summaries, normalized = [], []
    for model, path in zip(MODELS, paths):
        raw = read(REPORT / path)
        assert len(raw) == 100 and not raw.duplicated(['seed', 'gold_count']).any()
        assert set(raw.seed) == set(range(1254, 1264)) and set(raw.gold_count) == set(range(1, 11))
        assert raw.groupby('seed').size().eq(10).all()
        assert np.isfinite(raw[metrics]).all().all()
        detail = read((REPORT / path).with_name('detail_flat.csv'))
        assert len(detail) == 1100 and not detail.duplicated(['seed', 'gold_count', 'arm']).any()
        assert np.isfinite(detail.expected_count).all()
        detail['relative_error'] = (detail.expected_count - detail.gold_count).abs() / detail.gold_count
        e = detail.pivot(index=['seed', 'gold_count'], columns='arm', values='relative_error')
        contrasts = {
            'source_repair': e.O - e.S,
            'retrieval_mediation': e.S_Raligned - e.S_Rorth,
            'late_mediation': e.S_Taligned - e.S_Torth,
            'joint_interaction': (e.S_Raligned_Taligned - e.S_Rorth_Taligned)
                                 - (e.S_Raligned_Torth - e.S_Rorth_Torth),
            'remaining_repair': e.O - e.S_Raligned_Taligned,
        }
        derived = raw[['model_label', 'seed', 'gold_count']].copy()
        for metric in metrics:
            original = old[(old.model == model) & (old.metric == metric)]
            assert len(original) == 1 and np.isclose(raw[metric].mean(), original['mean'].iloc[0], atol=1e-12)
            derived[metric] = raw[metric] / raw.gold_count
            assert np.allclose(derived.set_index(['seed', 'gold_count'])[metric].sort_index(),
                               contrasts[metric].sort_index(), atol=1e-12)
            seed_effects = derived.groupby('seed')[metric].mean()
            summaries.append(dict(model=model, metric=metric, examples=100,
                                  **estimate(seed_effects, f'serial:{model}:{metric}', test=True)))
            SEED_ROWS.extend(dict(analysis='serial', population='all', model=model,
                                  metric=metric, seed=int(seed), value=float(value))
                             for seed, value in seed_effects.items())
        normalized.append(derived)
    pd.concat(normalized).to_csv(OUT / 'serial_normalized_paired_examples.csv', index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / 'serial_normalized_summary.csv', index=False)
    AUDIT['serial'] = {'base_error': 'abs(E - gold N) / gold N, evaluated before seed averaging',
                       'original_aggregate_means_reproduced': True,
                       'all_contrasts_reconstructed_from_raw_condition_expected_counts': True,
                       'not_a_mediated_fraction': True,
                       'normalization_does_not_clip_negative_or_above_one_effects': True}
    return summary


def two_panels(models=False, height=2.05):
    fig = plt.figure(figsize=(6.5, height))
    axes = [fig.add_axes([x, .235, .365, .555]) for x in [.10, .61]]
    if models:
        fig.legend(style.model_handles(), style.NAMES, loc='upper center', bbox_to_anchor=(.54, 1.006),
                   ncol=2, frameon=False, fontsize=9, handlelength=1.5, borderaxespad=.1)
    return fig, axes


def error_points(ax, x, rows, color, marker='o', label=None, ls='none'):
    means, low, high = [np.array([float(r[key]) for r in rows]) for key in ['mean', 'ci95_low', 'ci95_high']]
    assert np.isfinite([means, low, high]).all() and (low <= means).all() and (means <= high).all()
    ax.errorbar(x, means, yerr=[means-low, high-means], color=color, marker=marker, ms=4,
                linestyle=ls, lw=1.2, capsize=2.5, label=label)


def save(fig, name):
    style.save(fig, name)
    qa = style.AUDITS[name]
    assert not qa['outside'] and not qa['text_overlaps'], (name, qa)


def ablation():
    raw = read(ROOT / 'figures/nonthinking_appendix_20260910/retrieval/ablation_summary.csv')
    k6 = read(REPORT / 'v4_4_causal_v2/full_span_topk_k6_extension/provenance/detail.csv.gz')
    assert len(k6) == 400 and k6.top_n.eq(6).all()
    assert set(k6.seed) == set(range(1316, 1336)) and set(k6.gold_count) == set(range(1, 6))
    assert k6.groupby('condition').size().to_dict() == {'layer_matched_random': 300, 'ranked': 100}
    fig, axes = two_panels()
    for j, (ax, model, name, color) in enumerate(zip(axes, MODELS, style.NAMES, COLORS)):
        style.panel(ax, f'{"AB"[j]}. {name}: head ablation', 'Effect', 'Number of ablated heads', True)
        expected = [1, 2, 4, 8, 16, 32] if j == 0 else [1, 2, 4, 6, 8, 16, 32]
        for condition, ls, marker, label in [('ranked', '-', 'o', 'Ranked'), ('layer_matched_random', '--', 'D', 'Random')]:
            rows = raw[(raw.model == model) & (raw.metric == 'relative_count_shift')
                       & (raw.condition == condition)].sort_values('k')
            assert list(rows.k) == expected and rows.seeds.eq(20).all() and rows.prompts.eq(100).all()
            ax.plot(rows.k, rows['mean'], color=color, ls=ls, marker=marker, ms=3, label=label)
            ax.fill_between(rows.k, rows.ci95_low, rows.ci95_high, color=color,
                            alpha=.12 if condition == 'ranked' else .06, lw=0)
        ax.set_xscale('log', base=2)
        ax.set_xticks(expected, expected)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set(xlim=(.83, 38), ylim=(-.015, .80), yticks=[0, .2, .4, .6])
        style.legend(ax, loc='upper left', ncol=2, columnspacing=.6)
    save(fig, 'nonthinking_head_ablation')
    AUDIT['gemma_k6'] = {'intervention_rows': 400, 'ranked_rows': 100, 'random_rows': 300,
                         'seeds': list(range(1316, 1336)), 'counts': list(range(1, 6)),
                         'same_frozen_prompts_and_controls': True,
                         'provenance': 'Later dose extension rerun on the original cohort; not an independent replication.',
                         'plotted_values_and_intervals_unchanged': True}


def answer_function(removal):
    correct = read(REPORT / 'v4_4_causal_v2/correct_patching_aggregate.csv')
    correct = correct[correct.family == 'answer_patching']
    assert len(correct) == 12
    fig, axes = two_panels(models=True)
    ax = axes[0]
    style.panel(ax, 'A. Correct-input transfer', 'Source-count match', r'Source count $-$ target count', True)
    for j, (model, color) in enumerate(zip(MODELS, COLORS)):
        frame = correct[correct.model_label == model].copy()
        frame['shift'] = frame.k * np.where(frame.target_direction == 'increase', 1, -1)
        frame = frame.sort_values('shift').rename(columns={'average_patching_acc': 'mean'})
        error_points(ax, np.arange(6) + (j-.5)*.2, frame.to_dict('records'), color, ls='-')
    ax.set(xticks=range(6), xticklabels=['$-5$', '$-3$', '$-1$', '+1', '+3', '+5'],
           ylim=(.80, 1.03), yticks=[.8, .9, 1])
    ax = axes[1]
    style.panel(ax, 'B. Answer-subspace removal', 'Effect', 'Intervention layer', True)
    assert removal.ci95_low.min() > -.025 and removal.ci95_high.max() < .30
    for model, color in zip(MODELS, COLORS):
        rows = removal[removal.model == model].sort_values('layer')
        style.curve(ax, rows.to_dict('records'), color)
        ax.plot(rows.layer, rows['mean'], 'o', ms=2.5, color=color)
    ax.set(xlim=(-.5, 41.5), xticks=[0, 10, 20, 30, 40], ylim=(-.025, .30), yticks=[0, .1, .2, .3])
    ax.axhline(0, color=style.GRAY, lw=.7, zorder=0)
    save(fig, 'nonthinking_answer_function')


def serial(summary):
    fig, axes = two_panels(models=True)
    specs = [(['source_repair', 'retrieval_mediation', 'late_mediation'],
              ['Span repair', 'Retrieval\nremoval', 'Answer\nremoval'], 'A. Sequential interventions'),
             (['joint_interaction', 'remaining_repair'],
              ['Interaction', 'Remaining repair'], 'B. Joint intervention')]
    for ax, (metrics, labels, title) in zip(axes, specs):
        style.panel(ax, title, 'Effect', percent=True)
        for j, (model, color) in enumerate(zip(MODELS, COLORS)):
            rows = [summary[(summary.model == model) & (summary.metric == metric)].iloc[0].to_dict() for metric in metrics]
            error_points(ax, np.arange(len(metrics)) + (j-.5)*.20, rows, color)
        ax.set(xticks=range(len(metrics)), xticklabels=labels, xlim=(-.55, len(metrics)-.45),
               ylim=(-.30, .68), yticks=[-.2, 0, .2, .4, .6])
        ax.axhline(0, color=style.GRAY, lw=.7, zorder=0)
    save(fig, 'nonthinking_serial_mediation')


def qwen():
    summary = read(REPORT / 'v4_4_4/read_write/metric_summary.csv.gz')
    seeds = read(REPORT / 'v4_4_4/read_write/seed_metrics.csv.gz')
    fig, axes = two_panels()
    plot_rows = []
    specs = [('A. Source-state transfer', ['Full state', 'Attention\nweights', 'Values'],
              ['read_full_behavior_transport', 'read_routing_behavior_transport', 'read_value_behavior_transport']),
             ('B. Cancel the OV component', ['Attention weights', 'Values'],
              ['read_routing_ov_mediation_specificity', 'read_value_ov_mediation_specificity'])]
    for j, (ax, (title, labels, metrics)) in enumerate(zip(axes, specs)):
        style.panel(ax, title, 'Answer transfer' if j == 0 else 'Transfer reduction', percent=True)
        rows = []
        for metric in metrics:
            sub = summary[(summary.metric == metric) & (summary.stratum == 'all')]
            vals = seeds[(seeds.metric == metric) & (seeds.stratum == 'all')]
            assert len(sub) == 1 and len(vals) == 20 and set(vals.seed) == set(range(1274, 1294))
            row = sub.iloc[0].to_dict()
            assert np.isclose(row['mean'], vals.value.mean(), atol=1e-12)
            rows.append(row); plot_rows.append(row)
        error_points(ax, np.arange(len(metrics)), rows, COLORS[0])
        ax.set(xticks=range(len(metrics)), xticklabels=labels, xlim=(-.5, len(metrics)-.5))
        if j == 0:
            ax.set(ylim=(0, .155), yticks=[0, .05, .10, .15])
        else:
            ax.set(ylim=(0, .016), yticks=[0, .005, .01, .015])
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
        ax.axhline(0, color=style.GRAY, lw=.7, zorder=0)
    pd.DataFrame(plot_rows).to_csv(OUT / 'qwen_ov_plotted_summary.csv', index=False)
    save(fig, 'nonthinking_qwen_routing')
    AUDIT['qwen_ov'] = {'effect': 'T_E=(expected_count_patched-expected_count_baseline)/(N_source-N_target)',
                        'cancellation': 'T_E(component patch + orthogonal control) - T_E(component patch + natural-axis block)',
                        'saved_seed_means_reproduced': True, 'saved_intervals_unchanged': True,
                        'figure_change': 'Replace abstract propagation curve with direct OV cancellation contrast.',
                        'different_panel_scales': 'Explicit 0-15% vs 0-1.5%; cancellations are small, not full mediation.'}


def gemma():
    base = REPORT / 'v4_4_4/gemma/residual/k2'
    seeds = read(base / 'residual_seed_metrics.csv.gz')
    original = read(base / 'residual_endpoint_summary.csv.gz')
    metrics = ['source_donor_transport', 'exact_residual_mediation', 'count_axis_mediation']
    rows = []
    for metric in metrics:
        for role in ['candidate_core', 'matched_control']:
            sub = seeds[(seeds.endpoint == metric) & (seeds.set_role == role)]
            count = 1 if role == 'candidate_core' else 3
            assert sub.groupby('seed').size().eq(count).all() and set(sub.seed) == set(range(1466, 1486))
            values = sub.groupby('seed').value.mean()
            if role == 'candidate_core':
                saved = original[(original.endpoint == metric) & (original.set_role == role)]
                assert len(saved) == 1 and np.isclose(saved['mean'].iloc[0], values.mean(), atol=1e-12)
                row = saved.iloc[0].to_dict()
                row['bootstrap_draws'] = 10000
            else:
                row = dict(endpoint=metric, set_role=role, **estimate(values, f'gemma:{metric}:{role}'))
            rows.append(row)
            SEED_ROWS.extend(dict(analysis='gemma_residual', population=role, model=MODELS[1],
                                  metric=metric, seed=int(seed), value=float(value)) for seed, value in values.items())
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / 'gemma_residual_plotted_summary.csv', index=False)
    fig, axes = two_panels()
    handles = [Line2D([], [], color=c, marker=m, ls='', markersize=4) for c, m in [(COLORS[1], 'o'), (style.GRAY, 'D')]]
    fig.legend(handles, ['Candidate heads', 'Matched heads'], loc='upper center', bbox_to_anchor=(.54, 1.006),
               ncol=2, frameon=False, fontsize=9, handlelength=1., borderaxespad=.1)
    for j, (ax, wanted, labels, title) in enumerate(zip(axes, [metrics[:1], metrics[1:]],
                [['Source-state patch'], ['Full residual\nchange', 'Count-aligned\ncomponent']],
                ['A. Source-state transfer', 'B. Cancel the residual change'])):
        style.panel(ax, title, 'Answer transfer' if j == 0 else 'Transfer reduction', percent=True)
        for k, (role, color, marker) in enumerate([('candidate_core', COLORS[1], 'o'), ('matched_control', style.GRAY, 'D')]):
            plotted = [summary[(summary.endpoint == metric) & (summary.set_role == role)].iloc[0].to_dict() for metric in wanted]
            error_points(ax, np.arange(len(wanted)) + (k-.5)*.18, plotted, color, marker)
        ax.set(xticks=range(len(wanted)), xticklabels=labels, xlim=(-.5, len(wanted)-.5),
               ylim=(-.015, .13), yticks=[0, .04, .08, .12])
        ax.axhline(0, color=style.GRAY, lw=.7, zorder=0)
    save(fig, 'nonthinking_gemma_residual')
    AUDIT['gemma_residual'] = {'effect': 'Same expected-count source-directed transfer T_E as Qwen.',
                              'cancellation': 'Transfer under norm-matched orthogonal cancellation minus aligned cancellation.',
                              'head_control': 'Mean of three frozen layer-matched banks within seed, then seed bootstrap.',
                              'candidate_saved_means_and_intervals_unchanged': True,
                              'localized_ov_test_passed': False,
                              'positive_mechanism': 'Candidate-head source patch affects count; cancellation at downstream residual L38 reduces its transfer.'}


def main():
    started = time.perf_counter()
    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix', 'font.size': 10,
        'axes.titlesize': 11, 'axes.titleweight': 'normal', 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
        'legend.frameon': False, 'axes.linewidth': .7, 'lines.linewidth': 1.5,
        'lines.markersize': 4, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'text.color': style.INK, 'axes.labelcolor': style.INK, 'axes.titlecolor': style.INK,
        'xtick.color': style.INK, 'ytick.color': style.INK, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    removal = normalize_removal()
    serial_summary = normalize_serial()
    ablation()
    answer_function(removal)
    serial(serial_summary)
    qwen()
    gemma()
    pd.DataFrame(SEED_ROWS).to_csv(OUT / 'derived_seed_effects.csv', index=False)
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in INPUTS.items())
    result = dict(status='PASS', input_sha256=INPUTS, analysis=AUDIT, figures=style.AUDITS,
                  model_inference=False, refitting=False, indexing='Source zero-based; displayed layers one-based.',
                  palette=dict(zip(MODELS, COLORS)), font_points=dict(title=11, axis=10, tick=9, legend=9),
                  elapsed_seconds=time.perf_counter()-started)
    (OUT / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='PASS', figures=list(style.AUDITS), seconds=result['elapsed_seconds'],
                         removal=AUDIT['removal'], serial=serial_summary.to_dict('records'))))


if __name__ == '__main__':
    main()
