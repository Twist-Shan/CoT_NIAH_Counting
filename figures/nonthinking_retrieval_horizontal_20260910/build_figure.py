"""Horizontal retrieval figure, with frozen Top-3/Top-6 profile comparisons.

Run from any directory: python build_figure.py
Reads immutable discovery captures; no inference or manuscript modification.
"""
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import csv
import gzip
import hashlib
import json
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

START = time.perf_counter()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--compact', action='store_true', help='Export a shorter layout into compact/, preserving the original figures.')
parser.add_argument('--output-directory', type=Path, help='Stage figures in a separate directory.')
args = parser.parse_args()
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OUT = args.output_directory or (BASE / 'compact' if args.compact else BASE)
OUT.mkdir(parents=True, exist_ok=True)
REPO = ROOT / 'realistic'
REPORT = REPO / 'reports/v4_non-thinking_causal'
CAPTURES = REPO / 'exports/run_20260731_v4_numeric_presentation_v3'
MODELS = ['Qwen3-8B', 'Gemma4-E4B']
NAMES = ['Qwen', 'Gemma']
COLORS = ['#168DCA', '#E87824']
GRAY = '#7E8791'
INPUTS = {}


def record_input(path):
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_csv(path):
    record_input(path)
    opener = gzip.open(path, 'rt', encoding='utf-8-sig') if path.suffix == '.gz' else path.open(encoding='utf-8-sig', newline='')
    with opener as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


atlas = [r for r in read_csv(REPORT / 'v4_4/realistic_niah_v4_head_atlas.csv') if r['variant'] == 'v4.4']
old_profiles = [r for r in read_csv(REPORT / 'v4_4/realistic_niah_v4_head_phenotypes.csv') if r['variant'] == 'v4.4']
membership = read_csv(REPORT / 'v4_4_causal_v2/full_span_topk/full_span_topk_membership.csv')
selected = {}
profiles = {}
audit = {}
seed_rows = []
mean_rows = []
scatter_rows = []

for model, k in zip(MODELS, [32, 6]):
    selected[model] = sorted([r for r in membership if r['model_label'] == model and int(r['top_n']) == 32 and int(r['rank']) <= k], key=lambda r: int(r['rank']))
    assert len(selected[model]) == k
    heads = selected[model][:6]
    attention = CAPTURES / model / 'numeric/attention'
    index_path = record_input(attention / 'capture/attention_capture_index.jsonl')
    index = [json.loads(line) for line in index_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    index = sorted([r for r in index if r['design_variant'] == 'v4.4' and r['split'] == 'discovery' and int(r['count']) == 10], key=lambda r: int(r['seed']))
    assert len(index) == 20 and [int(r['seed']) for r in index] == list(range(1234, 1254))
    occurrence = read_csv(attention / 'analysis_span_sum_v3/tables/occurrence_attention.csv.gz')
    spans = defaultdict(list)
    for row in occurrence:
        if row['design_variant'] == 'v4.4' and row['split'] == 'discovery' and int(row['count']) == 10 and row['pooling'] == 'span_end':
            spans[row['stimulus_id']].append(row)
    per_head = defaultdict(list)
    for record in index:
        span = sorted(spans[record['stimulus_id']], key=lambda r: int(r['occurrence_index']))
        assert len(span) == 10 and [int(r['occurrence_index']) for r in span] == list(range(1, 11))
        assert {int(r['seed']) for r in span} == {int(record['seed'])}
        path = record_input(attention / 'capture' / record['raw_attention_shard_path'])
        with np.load(path, allow_pickle=False) as raw:
            layer_cache = {}
            for h in heads:
                layer, head = int(h['layer']), int(h['head'])
                if layer not in layer_cache:
                    layer_cache[layer] = np.asarray(raw[f'layer_{layer:03d}'], dtype=np.float32)
                row = layer_cache[layer][head]
                key_start = int(raw['key_starts'][layer])
                starts = np.array([int(r['span_start']) for r in span]) - key_start
                ends = np.array([int(r['span_end']) for r in span]) - key_start
                assert np.all(starts >= 0) and np.all(ends <= len(row)) and np.all(ends > starts)
                # Match the established float32 span reduction, then normalize in float64.
                mass = np.array([row[a:b].sum() for a, b in zip(starts, ends)], dtype=float)
                assert np.isfinite(mass).all() and (mass >= 0).all() and mass.sum() > 0
                share = mass / mass.sum()
                per_head[h['head_label']].append(share)
                for i, value in enumerate(share, 1):
                    seed_rows.append(dict(model=model, rank=int(h['rank']), layer=layer, head=head, head_label=h['head_label'], seed=int(record['seed']), record_index=i, attention_share=float(value)))
    profiles[model] = np.array([np.mean(per_head[h['head_label']], axis=0) for h in heads])
    assert profiles[model].shape == (6, 10) and np.allclose(profiles[model].sum(axis=1), 1, atol=1e-12)
    old_differences = {}
    newly_extracted = []
    for h, values in zip(heads, profiles[model]):
        old = [r for r in old_profiles if r['model'] == model and r['layer'] == h['layer'] and r['head'] == h['head']]
        assert len(old) <= 1
        if old:
            assert int(old[0]['samples']) == 20
            difference = float(np.max(np.abs(values - np.array(json.loads(old[0]['span_sum_profile'])))))
            old_differences[h['head_label']] = difference
            assert difference < 1e-7, (model, h['head_label'], difference)
        else:
            newly_extracted.append(h['head_label'])
        for i, value in enumerate(values, 1):
            mean_rows.append(dict(model=model, rank=int(h['rank']), layer=int(h['layer']), head=int(h['head']), head_label=h['head_label'], record_index=i, mean_attention_share=float(value), seeds=20))
    keys = {(r['layer'], r['head']) for r in selected[model]}
    rows = [r for r in atlas if r['model'] == model and r['pooling'] == 'span_sum']
    bank = [r for r in rows if (r['layer'], r['head']) in keys]
    assert len(bank) == k and all(int(r['seeds']) == 20 and int(r['examples']) == 180 for r in rows)
    for r in rows:
        scatter_rows.append(dict(model=model, layer=int(r['layer']), head=int(r['head']), retrieval_score=float(r['pool_primary']), selected=(r['layer'], r['head']) in keys))
    audit[model] = dict(profile_heads=[h['head_label'] for h in heads], newly_extracted_profiles=newly_extracted,
        existing_profile_max_abs_differences=old_differences, frozen_bank_count=k, atlas_head_count=len(rows),
        bank_layer_counts=dict(Counter(int(r['layer']) for r in bank)),
        bank_mean_effective_coverage=float(np.mean([float(r['pool_coverage']) for r in bank])),
        top6_profile_max_share=float(profiles[model].max()))

write_csv('profile_seed_values.csv', seed_rows)
write_csv('profile_means.csv', mean_rows)
write_csv('layer_scores.csv', scatter_rows)

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
    'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 10.5, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#8D99A5', 'axes.linewidth': .65, 'text.color': '#202020',
    'axes.labelcolor': '#202020', 'xtick.color': '#202020', 'ytick.color': '#202020',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

STYLES = ['-', '--', ':', '-.', (0, (5, 1, 1, 1)), (0, (1, 1))]
MARKERS = ['o', 's', '^', 'D', 'v', 'P']
WIDTH, HEIGHT = 6.5, 2.10 if args.compact else 2.65
LEFTS = [.48, 2.035, 3.59, 5.145]
PANEL_WIDTH = 1.125
BOTTOM, PANEL_HEIGHT = (.72, .98) if args.compact else (.82, 1.32)
HEADING_Y = (HEIGHT - .05) / HEIGHT if args.compact else .952
LEGEND_Y = .41 / HEIGHT if args.compact else .17
RIGHT_LEGEND_Y = .045 / HEIGHT
if args.compact:
    plt.rcParams['axes.labelpad'] = 2
layout_audit = {}


def frame():
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    axes = [fig.add_axes([x / WIDTH, BOTTOM / HEIGHT, PANEL_WIDTH / WIDTH, PANEL_HEIGHT / HEIGHT]) for x in LEFTS]
    fig.text((LEFTS[0] + LEFTS[1] + PANEL_WIDTH) / (2 * WIDTH), HEADING_Y, 'A  Attention across needles', ha='center', va='top', weight='bold', fontsize=10.5)
    fig.text((LEFTS[2] + LEFTS[3] + PANEL_WIDTH) / (2 * WIDTH), HEADING_Y, 'B  Retrieval by layer', ha='center', va='top', weight='bold', fontsize=10.5)
    for i, ax in enumerate(axes):
        ax.set_title(NAMES[i % 2], loc='center', fontsize=9, fontweight='normal', pad=4 if args.compact else 7)
        ax.tick_params(length=3, width=.6, pad=2)
        ax.grid(axis='y', color='#E7E8EE', lw=.55)
        ax.set_axisbelow(True)
    return fig, axes


def plot_layers(fig, axes):
    for j, (model, color, k) in enumerate(zip(MODELS, COLORS, [32, 6])):
        ax = axes[j + 2]
        rows = [r for r in scatter_rows if r['model'] == model]
        bank = [r for r in rows if r['selected']]
        offset = lambda rr: [r['layer'] + 1 + (r['head'] / (32 if j == 0 else 8) - .5) * .55 for r in rr]
        ax.scatter(offset(rows), [r['retrieval_score'] for r in rows], s=4.3, color=color, alpha=.18, edgecolor='none')
        ax.scatter(offset(bank), [r['retrieval_score'] for r in bank], s=12, color=color, edgecolor='white', linewidth=.22, zorder=4)
        ax.set(xlim=(0, 36.9 if j == 0 else 43), ylim=(-.015, .61), xlabel='Layer')
        ax.set_xticks([1, 11, 21, 31] if j == 0 else [6, 18, 30, 42])
        ax.set_yticks([0, .2, .4, .6])
        if j == 0:
            ax.set_ylabel('Retrieval score', labelpad=3)
        # Shared semantic key; the selected bank sizes are explicit per panel.
        ax.text(.04, .94, f'Top-{k}', transform=ax.transAxes, va='top', fontsize=8.5, color=color)
    handles = [Line2D([], [], marker='o', ls='', ms=3.5, color=GRAY, label='Selected heads'),
        Line2D([], [], marker='o', ls='', ms=2.7, color=GRAY, alpha=.25, label='Other heads')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=((LEFTS[2] + LEFTS[3] + PANEL_WIDTH) / (2 * WIDTH), RIGHT_LEGEND_Y),
        ncol=2, frameon=False, handlelength=1.5, handletextpad=.45, columnspacing=1.0, labelspacing=.3, borderaxespad=0)


def export(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.width + .5 or box.y1 > fig.bbox.height + .5:
                outside.append(text.get_text())
    assert not outside, outside
    layout_audit[name] = {'figure_size_inches': [WIDTH, HEIGHT], 'compact': args.compact, 'panel_height_inches': PANEL_HEIGHT, 'equal_panel_width_inches': PANEL_WIDTH,
        'equal_panel_gap_inches': LEFTS[1] - LEFTS[0] - PANEL_WIDTH, 'text_outside_canvas': outside}
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, facecolor='white')
    plt.close(fig)


for head_count in [3, 6]:
    fig, axes = frame()
    for j, (model, color) in enumerate(zip(MODELS, COLORS)):
        ax = axes[j]
        ax.axhline(.1, color=GRAY, lw=.75, alpha=.7)
        for q, (h, values) in enumerate(zip(selected[model][:head_count], profiles[model][:head_count])):
            ax.plot(range(1, 11), values, color=color, lw=1.15, ls=STYLES[q], marker=MARKERS[q],
                ms=2.7, markerfacecolor=color if q < 3 else 'white', markeredgewidth=.65, label=f"L{int(h['layer'])+1}H{int(h['head'])+1}")
        ax.set(xlim=(.8, 10.2), ylim=(0, .25), xlabel='Needle index ($N=10$)')
        ax.set_xticks([1, 4, 7, 10])
        ax.set_yticks([0, .1, .2])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        if j == 0:
            ax.set_ylabel('Share of needle attention', labelpad=3)
        handles, labels = ax.get_legend_handles_labels()
        # Put successive ranks across columns, not down columns.
        order = list(range(0, head_count, 2)) + list(range(1, head_count, 2)) if head_count == 6 else list(range(head_count))
        fig.legend([handles[q] for q in order], [labels[q] for q in order], loc='upper center',
            bbox_to_anchor=((LEFTS[j] + PANEL_WIDTH / 2) / WIDTH, LEGEND_Y), ncol=2 if head_count == 6 else 1,
            frameon=False, handlelength=1.45, handletextpad=.35, columnspacing=.65, labelspacing=.1, borderaxespad=0)
    plot_layers(fig, axes)
    assert max(p.max() for p in profiles.values()) <= .25
    export(fig, f'retrieval_horizontal_top{head_count}')

fig, axes = frame()
for j, (model, color) in enumerate(zip(MODELS, COLORS)):
    ax = axes[j]
    cmap = LinearSegmentedColormap.from_list(f'{NAMES[j]}_attention', ['#FFFFFF', color])
    img = ax.imshow(profiles[model], aspect='auto', cmap=cmap, vmin=0, vmax=.25, interpolation='nearest')
    ax.grid(False)
    ax.set_xticks([0, 3, 6, 9], [1, 4, 7, 10])
    ax.set_yticks(range(6), [f"{int(h['layer'])+1},{int(h['head'])+1}" for h in selected[model][:6]], fontsize=8)
    ax.set_xlabel('Needle index ($N=10$)')
    if j == 0:
        ax.set_ylabel('Head (layer, index)', labelpad=3)
    ax.tick_params(axis='y', length=0)
    bar_ax = fig.add_axes([LEFTS[j] / WIDTH, .285 / HEIGHT, PANEL_WIDTH / WIDTH, .055 / HEIGHT])
    bar = fig.colorbar(img, cax=bar_ax, orientation='horizontal', ticks=[0, .1, .2, .25], format=PercentFormatter(1, decimals=0))
    bar.ax.tick_params(length=2, pad=1, labelsize=8)
    bar.set_label('Share of needle attention', fontsize=8.5, labelpad=1)
    bar.outline.set_linewidth(.35)
plot_layers(fig, axes)
export(fig, 'retrieval_horizontal_top6_heatmap')

manifest = dict(inputs_sha256=INPUTS, models=audit,
    profile_scope={'variant': 'v4.4', 'count': 10, 'seeds': list(range(1234, 1254)), 'split': 'discovery',
        'normalization': 'Sum answer-query attention over each needle span, normalize over the 10 spans within each prompt, then average the 20 prompts.',
        'selection': 'First 3 or 6 entries in the original frozen full-span ranking; no reranking or selection by profile flatness.'},
    atlas_scope={'variant': 'v4.4', 'counts': list(range(2, 11)), 'seeds': list(range(1234, 1254)), 'pooling': 'span_sum', 'endpoint_overlay': False,
        'gemma': 'Only full/global-attention layers are shown, as in the original figure.'},
    interpretation='Equal attention share is 10% for N=10. The profiles show broad but nonuniform attention. Mean profiles are not a per-prompt uniformity statistic.',
    palette=dict(zip(MODELS, COLORS)), layout=layout_audit,
    typography={'family': 'Times New Roman', 'math': 'STIX', 'axis_label_pt': 8.5, 'tick_pt': 8, 'legend_pt': 8,
        'group_title_pt': 10.5, 'model_subtitle_pt': 9, 'model_subtitle_alignment': 'center',
        'group_labels': {'A': 'Attention across needles', 'B': 'Retrieval by layer'},
        'heatmap_head_label_pt': 8, 'colorbar_tick_pt': 8, 'colorbar_label_pt': 8.5,
        'reference': 'Non-thinking main figure at 6.5-inch paper width: axis 8.45 pt, tick 7.8 pt, model legend 8.45 pt, title 10.595 pt.'},
    python_packages={'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
    elapsed_seconds=time.perf_counter() - START)
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': 'PASS', 'elapsed_seconds': manifest['elapsed_seconds'], 'models': audit, 'figures': list(layout_audit)}, indent=2))
