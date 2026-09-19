"""Reproduce the Non-thinking causal appendix figures from retained audit tables.

Run: python figures/nonthinking_appendix_20260910/causal/build_causal_figures.py
Inputs remain immutable. The script validates matched cells and clustered means.
"""
from pathlib import Path
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

START = time.perf_counter()
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
REPO = ROOT / 'realistic'
REPORT = REPO / 'reports/v4_non-thinking_causal'
DATA = ROOT / 'figures/non-thinking/data'
MODELS = ['Qwen3-8B', 'Gemma4-E4B']
NAMES = ['Qwen', 'Gemma']
COLORS = ['#168DCA', '#E87824']
SOURCES = {}
SEED = 20260910
audit = {'status': 'RUNNING', 'bootstrap_repetitions': 10000,
         'bootstrap_seed': SEED, 'new_uncertainty_unit': 'equal-weight seed means',
         'dimensions_inches': [6.5, 3.8], 'minimum_font_points': 8}

def record(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path

def rows(path):
    record(path)
    op = gzip.open if path.suffix == '.gz' else open
    with op(path, 'rt', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def dump_rows(name, data):
    with (OUT / name).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)

def summarize(values, rng):
    values = np.asarray(values, dtype=float)
    assert values.ndim == 1 and np.isfinite(values).all()
    means = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
    lo, hi = np.quantile(means, [.025, .975])
    return float(values.mean()), float(lo), float(hi)

def grouped_layer_summary(raw, value_key, group_key, seeds):
    grouped = defaultdict(list)
    for r in raw:
        grouped[(r['model'], int(r['layer']), r[group_key], int(r['seed']))].append(float(r[value_key]))
    out = []
    rng = np.random.default_rng(SEED)
    for model, layer, arm in sorted({key[:3] for key in grouped}):
        actual = sorted(key[3] for key in grouped if key[:3] == (model, layer, arm))
        assert actual == seeds
        vals = [np.mean(grouped[(model, layer, arm, seed)]) for seed in seeds]
        mean, lo, hi = summarize(vals, rng)
        out.append(dict(model=model, layer=layer, arm=arm, mean=mean, ci95_low=lo,
                        ci95_high=hi, seeds=len(seeds)))
    return out

# Endpoint and ordinary effects are recovered by exact paired subtraction from
# the historical full-span effect, never estimated from the published image.
full = rows(DATA / 'restoration_confirmation.csv')
span_root = REPORT / 'v4_4_5_followup/span_restoration'
delta = rows(span_root / 'full_minus_endpoint.csv')
specificity = rows(span_root / 'needle_minus_ordinary_specificity.csv')
key = lambda r: (r['model_label'], int(r['seed']), int(r['gold_count']), int(r['patch_layer']))
delta = {key(r): r for r in delta}
specificity = {key(r): r for r in specificity}
restore = []
for r in full:
    k = key(r)
    assert k in delta and k in specificity and 1254 <= k[1] <= 1263
    f = float(r['expected_absolute_error_reduction'])
    e = f - float(delta[k]['expected_absolute_error_reduction_full_minus_endpoint'])
    o = f - float(specificity[k]['expected_absolute_error_reduction_specificity'])
    for arm, value in [('Full span', f), ('Endpoint', e), ('Ordinary', o)]:
        restore.append(dict(model=k[0], seed=k[1], layer=k[3], count=k[2], arm=arm,
                            effect=value / k[2]))
assert len(restore) == 7800 * 3
restore_summary = grouped_layer_summary(restore, 'effect', 'arm', list(range(1254, 1264)))
dump_rows('restore_control_plot.csv', restore_summary)
audit['restore_cells'] = len(restore)
audit['restoration_derivation'] = 'endpoint = full - (full minus endpoint); ordinary = full - (needle minus ordinary), followed by per-input division by N'

# Keep the three raw arms: the same-count source control is scaled by the
# displacement of its matched different-count pair, not its own zero offset.
answer = []
for model, short, layers in zip(MODELS, ['qwen', 'gemma'], [36, 42]):
    root = REPO / f'exports/realistic_20260803_v4_4_causal_v2_{short}/run' / model / 'numeric/causal_v2'
    paths = list((root / 'answer_patching').glob('screen_*/detail.csv.gz'))
    assert len(paths) == 1
    raw = [r for r in rows(paths[0]) if r['patch_protocol'] == 'single_layer' and r['site'] == 'answer_query']
    assert len(raw) == 5 * 18 * layers * 3
    assert {int(r['seed']) for r in raw} == set(range(1254, 1259))
    for r in raw:
        denominator = int(r['donor_count']) - int(r['receiver_count'])
        strict = float(r['strict_normalized_transport'])
        assert denominator != 0
        if r['transport_numeric_valid'].lower() == 'true':
            assert abs(strict - float(r['generated_count_shift']) / denominator) < 1e-10
        else:
            assert strict == 0
        answer.append(dict(model=model, seed=int(r['seed']), layer=int(r['start_layer']),
                           arm={'donor_transport': 'Different count', 'self_patch': 'Self patch',
                                'same_count_seed': 'Same count'}[r['condition']], effect=strict))
answer_summary = grouped_layer_summary(answer, 'effect', 'arm', list(range(1254, 1259)))
dump_rows('answer_control_plot.csv', answer_summary)
main_answer = rows(DATA / 'answer_patching_plot.csv')
for r in main_answer:
    arms = {z['arm']: z['mean'] for z in answer_summary
            if z['model'] == r['model'] and z['layer'] == int(r['layer'])}
    adjusted = arms['Different count'] - (arms['Self patch'] + arms['Same count']) / 2
    assert abs(adjusted - float(r['mean'])) < 1e-12
audit['answer_control_adjustment_matches_main'] = True
audit['answer_rows'] = len(answer)

# Existing clean-correct transfer CIs cluster repeated patch conditions by seed.
correct = [r for r in rows(REPORT / 'v4_4_causal_v2/correct_patching_aggregate.csv')
           if r['family'] == 'answer_patching']
assert len(correct) == 12 and all(int(r['seed_clusters']) == 5 for r in correct)
pooled = [r for r in rows(REPORT / 'v4_4_causal_v2/correct_patching_pooled.csv')
          if r['family'] == 'answer_patching']
for r in pooled:
    arm = [z for z in correct if z['model_label'] == r['model_label']]
    assert sum(int(z['patching_acc_successes']) for z in arm) == int(r['patching_acc_successes'])
    assert sum(int(z['patching_acc_denominator']) for z in arm) == int(r['patching_acc_denominator'])
audit['correct_transfer'] = pooled

removal = [r for r in rows(REPORT / 'v4_4_extension/layerwise_subspace/answer_query_removal/layerwise_answer_query_removal_statistics.csv')
           if r['population'] == 'all' and r['endpoint'] == 'absolute_error_specificity']
assert len(removal) == 23 and all(int(r['examples']) == 90 and int(r['seeds']) == 10 for r in removal)
audit['removal_significant_layers'] = {m: [int(r['layer']) for r in removal if r['model_label'] == m and r['significant_holm_0_05'] == 'True'] for m in MODELS}

serial = []
for model, sub in zip(MODELS, ['v4_4_5_followup/exp19', 'v4_4_5_top6_followup/serial_mediation']):
    folder = REPORT / sub / model
    payload = json.loads(record(folder / 'serial_summary.json').read_text())
    assert payload['schema_version'] == 'realistic_niah_v4_4_5_serial_mediation_summary_v2'
    assert payload['inference']['independent_unit'] == 'seed'
    paired = rows(folder / 'paired_serial_effects.csv')
    assert len(paired) == 100 and {int(r['seed']) for r in paired} == set(range(1254, 1264))
    for metric in ['source_repair', 'retrieval_mediation', 'late_mediation', 'joint_interaction', 'remaining_repair']:
        v = payload['models'][model][metric]
        assert v['units'] == 10 and v['paired_seed_count_units'] == 100
        seed_means = [np.mean([float(r[metric]) for r in paired if int(r['seed']) == s]) for s in range(1254, 1264)]
        assert abs(np.mean(seed_means) - v['mean']) < 1e-12
        serial.append(dict(model=model, metric=metric, mean=v['mean'], ci95_low=v['ci95_low'], ci95_high=v['ci95_high'],
                           seeds=10, p_two_sided=v['exact_sign_flip_p_two_sided']))
dump_rows('serial_plot.csv', serial)

plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 8.5,
                     'mathtext.fontset': 'stix', 'axes.labelsize': 8.5,
                     'axes.titlesize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'legend.fontsize': 8, 'axes.linewidth': .7, 'pdf.fonttype': 42,
                     'svg.fonttype': 'none', 'legend.frameon': False})

def axes_grid():
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 3.8))
    fig.subplots_adjust(left=.087, right=.985, bottom=.115, top=.925, wspace=.29, hspace=.71)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#E7E8EE', linewidth=.6)
        ax.set_axisbelow(True)
        ax.axhline(0, color='#8190A5', linewidth=.6)
        ax.tick_params(length=2.5, pad=2)
    return fig, axes

def title(ax, label, text):
    ax.set_title(f'{label}  {text}', loc='left', weight='bold', pad=6)

def curve(ax, data, color, style, label):
    data = sorted(data, key=lambda x: x['layer'])
    x = [r['layer'] for r in data]
    ax.plot(x, [r['mean'] for r in data], color=color, ls=style, lw=1.45, label=label)
    ax.fill_between(x, [r['ci95_low'] for r in data], [r['ci95_high'] for r in data], color=color, alpha=.12, lw=0)

def save(fig, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [(t.get_text(), t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text()]
    outside = [s for s, b in texts if b.x0 < -.5 or b.y0 < -.5 or b.x1 > fig.bbox.width+.5 or b.y1 > fig.bbox.height+.5]
    assert not outside, outside
    # Check all text-to-text collisions, allowing only sub-pixel edge contact.
    overlaps = []
    for i, (s, a) in enumerate(texts):
        for t, b in texts[i+1:]:
            if min(a.x1, b.x1) - max(a.x0, b.x0) > 1 and min(a.y1, b.y1) - max(a.y0, b.y0) > 1:
                overlaps.append([s, t])
    assert not overlaps, overlaps
    audit[stem + '_text_qa'] = {'outside': outside, 'overlaps': overlaps}
    for suffix in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / (stem + '.' + suffix), dpi=260)
    plt.close(fig)

fig, axs = axes_grid()
for i, (model, name, color, n) in enumerate(zip(MODELS, NAMES, COLORS, [36, 42])):
    ax = axs[0, i]
    for arm, style, c in [('Full span', '-', color), ('Endpoint', '--', color), ('Ordinary', ':', '#7D8490')]:
        curve(ax, [r for r in restore_summary if r['model'] == model and r['arm'] == arm], c, style, arm)
    title(ax, 'AB'[i], name + ': restoration sites')
    ax.set(xlim=(-.5, n-.5), xticks=list(range(0,n,10)), ylim=(-.055, .7), yticks=[0, .2, .4, .6], xlabel='Intervention layer', ylabel='Restoration effect')
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(loc='upper right', ncol=1, handlelength=2, labelspacing=.2)
    ax = axs[1, i]
    for arm, style, c in [('Different count', '-', color), ('Self patch', ':', '#7D8490'), ('Same count', '--', color)]:
        curve(ax, [r for r in answer_summary if r['model'] == model and r['arm'] == arm], c, style, arm)
    title(ax, 'CD'[i], name + ': answer-state controls')
    ax.set(xlim=(-.5, n-.5), xticks=list(range(0,n,10)), ylim=(-.075, .94), yticks=[0, .4, .8], xlabel='Intervention layer', ylabel='Signed count transfer')
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(loc='upper left', handlelength=2, labelspacing=.2)
save(fig, 'nonthinking_patching_controls')

fig, axs = axes_grid()
ax = axs[0, 0]
offsets = [-5, -3, -1, 1, 3, 5]
for i, (model, color) in enumerate(zip(MODELS, COLORS)):
    z = sorted([r for r in correct if r['model_label'] == model], key=lambda r: int(r['k'])*(1 if r['target_direction']=='increase' else -1))
    x = np.arange(6) + (-.1 if i == 0 else .1)
    y = np.array([float(r['average_patching_acc']) for r in z])
    ax.errorbar(x, y, yerr=[y-np.array([float(r['ci95_low']) for r in z]), np.array([float(r['ci95_high']) for r in z])-y], color=color, fmt='o-', ms=3, lw=1.1, capsize=2, label=NAMES[i])
ax.set(xticks=range(6), xticklabels=['−5', '−3', '−1', '+1', '+3', '+5'], ylim=(.80, 1.03), yticks=[.8, .9, 1.0],
       xlabel='Source count − target count', ylabel='Source-count match')
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
title(ax, 'A', 'Transfer between correct inputs')
ax.legend(loc='lower left', ncol=2, handlelength=1.4, columnspacing=.9)

ax = axs[0, 1]
for model, name, color in zip(MODELS, NAMES, COLORS):
    z = [dict(layer=int(r['layer']), mean=float(r['mean_effect']), ci95_low=float(r['bootstrap_95ci_low']), ci95_high=float(r['bootstrap_95ci_high'])) for r in removal if r['model_label'] == model]
    curve(ax, z, color, '-', name)
    ax.plot([r['layer'] for r in z], [r['mean'] for r in z], 'o', ms=2.5, color=color)
ax.set(xlim=(-.5, 41.5), xticks=[0,10,20,30,40], ylim=(-.2, 1.62), yticks=[0, .5, 1, 1.5], xlabel='Intervention layer', ylabel='Excess error (counts)')
title(ax, 'B', 'Answer-subspace removal')
ax.legend(loc='upper left', handlelength=1.5, labelspacing=.2)

for ax, metrics, labels, letter, heading in [
    (axs[1, 0], ['source_repair','retrieval_mediation','late_mediation'], ['Span repair','Retrieval','Answer'], 'C', 'Sequential interventions'),
    (axs[1, 1], ['joint_interaction','remaining_repair'], ['Interaction','Remaining repair'], 'D', 'Interaction and residual effect')]:
    for i, (model, color) in enumerate(zip(MODELS, COLORS)):
        z = [next(r for r in serial if r['model'] == model and r['metric'] == metric) for metric in metrics]
        y = np.array([r['mean'] for r in z]); x = np.arange(len(metrics)) + (-.10 if i == 0 else .10)
        ax.errorbar(x, y, yerr=[y-np.array([r['ci95_low'] for r in z]), np.array([r['ci95_high'] for r in z])-y], color=color, fmt='o', ms=3.5, lw=1.2, capsize=2.5)
    ax.set(xticks=range(len(metrics)), xticklabels=labels, xlim=(-.55, len(metrics)-.45), ylabel='Effect (counts)')
    if letter == 'C': ax.set(ylim=(-.12, 3.4),yticks=[0,1,2,3])
    else: ax.set(ylim=(-.72, 1.98),yticks=[-.5,0,.5,1,1.5])
    title(ax, letter, heading)
save(fig, 'nonthinking_answer_and_mediation')

audit['source_sha256'] = SOURCES
audit['elapsed_seconds'] = time.perf_counter() - START
audit['status'] = 'PASS'
(OUT / 'causal_figure_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
print(json.dumps({'status': audit['status'], 'elapsed_seconds': audit['elapsed_seconds'], 'figures': ['nonthinking_patching_controls','nonthinking_answer_and_mediation']}, indent=2))
