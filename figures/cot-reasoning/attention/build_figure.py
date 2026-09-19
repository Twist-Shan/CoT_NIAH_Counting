"""Render four CoT attention plots from saved discovery measurements.

Layout and typography follow nonthinking_retrieval_horizontal_20260910.
No model execution, new sample selection, or causal-bank changes are made.
Run from the workspace: python -s figures/cot-reasoning/attention/build_figure.py
"""
from pathlib import Path
import sys as _font_sys
_font_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figure_style import paper_font
FONT_FAMILY = paper_font()
from itertools import combinations
import csv
import argparse
import hashlib
import json
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

START = time.perf_counter()
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SOURCE = ROOT / 'realistic/work/v5_native_p0_head_atlas_20260820'
parser = argparse.ArgumentParser()
parser.add_argument('--allow-missing-queries', action='store_true',
                    help='Export a clearly marked pending preview if Gemma has not been recaptured.')
parser.add_argument('--measurement-review', type=Path,
                    help='Audited decision for a corrected measurement that differs from the archived GPU cache.')
ARGS = parser.parse_args()
MODELS = [('qwen', 'Qwen3-8B', 'Qwen', '#168DCA', 36, 32),
          ('gemma', 'Gemma4-E4B', 'Gemma', '#E87824', 42, 8)]
INPUTS = {}
REPO = ROOT / 'realistic'


def record(path):
    INPUTS[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def write_csv(name, rows):
    with (OUT / 'data' / (DATA_PREFIX + name)).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


(OUT / 'data').mkdir(exist_ok=True)
record(ROOT / 'figures/nonthinking_retrieval_horizontal_20260910/build_figure.py')
record(ROOT / 'runs/paper_figures/figures/nonthinking_broad_retrieval.pdf')
record(ROOT / 'figures/aurora_attention_pca_concept/build_main_figure_v4.py')
examples, scores, events, model_audits = [], [], [], {}
matrices, rankings, displayed = {}, {}, {}
selected_banks = {}
numerical_reviews = {}
reference_atlas = REPO / 'reports/v4_non-thinking_causal/v4_4/realistic_niah_v4_head_atlas.csv'
with record(reference_atlas).open(encoding='utf-8-sig', newline='') as stream:
    gemma_display_layers = {int(r['layer']) for r in csv.DictReader(stream)
        if r['variant'] == 'v4.4' and r['model'] == 'Gemma4-E4B' and r['pooling'] == 'span_sum'}
assert gemma_display_layers == {5, 11, 17, 23, 29, 35, 41}
for short, model, name, color, nlayers, nheads in MODELS:
    bank_size = 128 if short == 'qwen' else 6
    selection_path = REPO / f'configs/realistic_niah_v5_{short}_shared_k{bank_size}_targeted_selection_frozen.json'
    selection = json.loads(record(selection_path).read_text(encoding='utf-8'))
    selection_info = selection['development_selection']
    bank = [tuple(head) for head in selection_info['primary_bank_heads']]
    assert len(bank) == len(set(bank)) == bank_size
    selected_banks[model] = set(bank)
    display_layers = set(range(nlayers)) if short == 'qwen' else gemma_display_layers
    assert all(layer in display_layers for layer, head in bank)
    if short == 'qwen':
        ranking_path = REPO / selection_info['source_ranking']
        assert hashlib.sha256(ranking_path.read_bytes()).hexdigest() == selection_info['source_ranking_sha256']
        with record(ranking_path).open(encoding='utf-8', newline='') as stream:
            raw_rows = list(csv.DictReader(stream))
        assert {r['fold'] for r in raw_rows} == {'0'}
        ranking = sorted([dict(layer=int(r['layer']), head=int(r['head']),
            score=float(r['discovery_selection_value']), rank=int(r['discovery_rank']),
            n_seeds=int(r['n_seeds'])) for r in raw_rows], key=lambda r: r['rank'])
        assert [(r['layer'], r['head']) for r in ranking[:bank_size]] == bank
        example = json.loads(record(ROOT / 'figures/aurora_attention_pca_concept/seed1255_native_transitions.json').read_text(encoding='utf-8'))
        seed_count = 15
    else:
        source = json.loads(record(SOURCE / 'p0_head_atlas_gemma.json').read_text(encoding='utf-8'))
        ranking = sorted(source['rankings']['same_unit_rank_before_city']['rows'], key=lambda r: r['rank'])
        assert {(r['layer'], r['head']) for r in ranking[:bank_size]} == set(bank)
        capture_path = OUT / 'data/gemma_seed1240_marker_attention.json'
        if capture_path.exists():
            example = json.loads(record(capture_path).read_text(encoding='utf-8'))
            assert example['queries_complete']
            if example['replay_requires_review']:
                if ARGS.measurement_review is None:
                    raise RuntimeError('The GPU numerical replay differs; an explicit, data-bound measurement review is required before publishing.')
                review_path = ARGS.measurement_review.resolve()
                review = json.loads(record(review_path).read_text(encoding='utf-8'))
                assert review['decision'] in ['accept_new_cpu_measurement_for_illustration', 'accept_new_gpu_measurement_for_illustration']
                assert review['capture_sha256'] == hashlib.sha256(capture_path.read_bytes()).hexdigest()
                assert review['input_sha256'] == example['input_sha256']
                assert review['gpu_exact_replay_claimed'] is False
                assert review['requires_methods_disclosure'] is True
                audit_path = (ROOT / review['measurement_audit_path']).resolve()
                assert hashlib.sha256(audit_path.read_bytes()).hexdigest() == review['measurement_audit_sha256']
                audit = json.loads(record(audit_path).read_text(encoding='utf-8'))
                assert audit['capture_sha256'] == review['capture_sha256']
                assert audit['measurement_invariants'] == 'PASS'
                assert audit['original_query_positions_and_record_spans_match'] is True
                assert audit['backend_restored_for_all_queries'] is True
                if review['decision'] == 'accept_new_gpu_measurement_for_illustration':
                    assert example['execution']['device_types'] == ['cuda']
                    assert audit['all_rows_from_one_gpu_run'] is True
                    assert audit['independent_gpu_repeat_reproduced_exactly'] is True
                else:
                    assert example['execution']['device_map'] == 'cpu'
                    assert example['execution']['device_types'] == ['cpu']
                    assert audit['all_rows_from_one_cpu_run'] is True
                    assert audit['independent_cpu_pilot_reproduced_exactly'] is True
                numerical_reviews[model] = review
        else:
            example = next(e for e in source['examples'] if e['grammar'] == 'adjacent_rank_after_city'
                           and (e['layer'], e['head']) == bank[0])
        seed_count = 19
    assert len(ranking) == nlayers * nheads
    assert [row['rank'] for row in ranking] == list(range(1, len(ranking) + 1))
    assert {(row['layer'], row['head']) for row in ranking} == {
        (layer, head) for layer in range(nlayers) for head in range(nheads)}
    assert all(row['n_seeds'] == seed_count and np.isfinite(row['score']) and 0 <= row['score'] <= 1 for row in ranking)
    assert all(a['score'] >= b['score'] for a, b in zip(ranking, ranking[1:]))
    chosen = next(r for r in ranking if (r['layer'], r['head']) == bank[0])
    assert example['gold_count'] == 10
    assert (example['layer'], example['head']) == bank[0]
    assert example['seed'] == (1255 if short == 'qwen' else 1240)
    matrix = np.full((10, 10), np.nan)
    observed_markers = []
    for event in example['events']:
        step = event['from_occurrence']
        assert step not in observed_markers and 0 <= step < 10
        observed_markers.append(step)
        records = sorted(event['records'], key=lambda row: row['source_index'])
        assert [row['source_index'] for row in records] == list(range(1, 11))
        assert event['to_occurrence'] == event['from_occurrence'] + 1
        assert sum(bool(row['is_target']) for row in records) == 1
        target = next(row for row in records if row['is_target'])
        assert target['source_index'] == event['to_occurrence']
        if 'target_city' in event:
            assert target['city'] == event['target_city']
        mass = np.array([row['mass'] for row in records], dtype=float)
        assert np.isfinite(mass).all() and (mass >= 0).all() and mass.sum() > 0
        assert np.isclose(event['attention_total_mass'], 1, atol=.01)
        assert np.isclose(mass.sum() + event['non_needle_context_mass'], event['attention_total_mass'], atol=.01)
        share = mass / mass.sum()
        assert np.isclose(share.sum(), 1, atol=1e-12)
        matrix[step] = share
        for row, value in zip(records, share):
            examples.append(dict(model=model, seed=example['seed'], layer=chosen['layer'] + 1,
                head=chosen['head'] + 1, row=step + 1, from_needle=event['from_occurrence'],
                target_needle=event['to_occurrence'], needle=row['source_index'], city=row['city'],
                raw_attention_mass=row['mass'], needle_attention_share=float(value), is_target=row['is_target']))
        events.append(dict(model=model, seed=example['seed'], from_needle=event['from_occurrence'],
            target_needle=event['to_occurrence'], query_output_token=event['query_output_token_index'],
            query_full_sequence_token=event['query_full_sequence_token'], query_token=event['query_token_text'],
            needle_mass=float(mass.sum()), total_attention_mass=event['attention_total_mass'],
            target_share=float(share[event['to_occurrence'] - 1]),
            largest_share_needle=int(np.argmax(share)) + 1))
    missing_markers = sorted(set(range(10)) - set(observed_markers))
    if missing_markers and not ARGS.allow_missing_queries:
        raise RuntimeError(f'{model} is missing markers {missing_markers}; run the prepared capture before publishing.')
    matrices[model] = matrix
    rankings[model] = ranking
    displayed[model] = chosen
    for row in ranking:
        scores.append(dict(model=model, layer=row['layer'] + 1, head=row['head'] + 1,
            discovery_rank=row['rank'], mean_targeted_score=row['score'], seeds=row['n_seeds'],
            displayed_global_attention=row['layer'] in display_layers,
            highlighted_ablation_bank=(row['layer'], row['head']) in set(bank),
            illustrated_head=(row['layer'], row['head']) == bank[0]))
    model_audits[model] = dict(archived_ranking_rows=len(ranking), heads_displayed=len(display_layers) * nheads,
        displayed_layers_one_based=sorted(l + 1 for l in display_layers),
        display_scope='All Qwen layers; Gemma full/global-attention layers, matching Non-thinking. Local-layer archive rows are omitted without score-based filtering.',
        supporting_discovery_seeds=seed_count,
        ablation_bank_size=bank_size, ablation_bank_one_based=[[l+1,h+1] for l,h in bank],
        ranking_grammar=selection_info['head_ranking_source_grammar'],
        ranking_query_site=selection_info['head_ranking_source_anchor'],
        example_seed=example['seed'], example_request=example['request_id'],
        illustrated_head_one_based=[chosen['layer'] + 1, chosen['head'] + 1],
        illustrated_head_discovery_score=chosen['score'], example_target_indices=list(range(1,11)),
        observed_markers=sorted(observed_markers), missing_markers=missing_markers,
        example_target_argmax_count=sum(row['largest_share_needle'] == row['target_needle'] for row in events if row['model'] == model))
    if 'execution' in example:
        model_audits[model]['execution'] = example['execution']
        model_audits[model]['gpu_replay_tolerances_passed'] = not example['replay_requires_review']
        model_audits[model]['maximum_gpu_replay_mass_difference'] = example['maximum_replay_mass_difference']
        model_audits[model]['maximum_gpu_replay_share_difference'] = example.get('maximum_replay_share_difference')

PENDING = any(a['missing_markers'] for a in model_audits.values())
DATA_PREFIX = 'pending_' if PENDING else ''
STEM = 'cot_reasoning_attention_pending' if PENDING else 'cot_reasoning_attention'
write_csv('attention_examples.csv', examples)
write_csv('attention_events.csv', events)
write_csv('head_scores.csv', scores)

plt.rcParams.update({'font.family': 'serif', 'font.serif': [FONT_FAMILY], 'mathtext.fontset': 'stix',
    'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 10.5, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#8D99A5', 'axes.linewidth': .65, 'text.color': '#202020',
    'axes.labelcolor': '#202020', 'xtick.color': '#202020', 'ytick.color': '#202020',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.labelpad': 2})
WIDTH, HEIGHT = 6.5, 2.10
LEFTS = [.48, 2.035, 3.59, 5.145]
PANEL_WIDTH, BOTTOM, PANEL_HEIGHT = 1.125, .72, .98
fig = plt.figure(figsize=(WIDTH, HEIGHT))
axes = [fig.add_axes([x / WIDTH, BOTTOM / HEIGHT, PANEL_WIDTH / WIDTH, PANEL_HEIGHT / HEIGHT]) for x in LEFTS]
for start, title in [(0, 'A  Attention from CoT markers'), (2, 'B  Retrieval by layer')]:
    fig.text((LEFTS[start] + LEFTS[start + 1] + PANEL_WIDTH) / (2 * WIDTH), (HEIGHT - .05) / HEIGHT,
        title, ha='center', va='top', weight='bold', fontsize=10.5)

for j, (_, model, name, color, nlayers, nheads) in enumerate(MODELS):
    ax = axes[j]
    chosen = displayed[model]
    ax.set_title(f"{name} · L{chosen['layer'] + 1}H{chosen['head'] + 1}", fontsize=9, pad=4)
    cmap = LinearSegmentedColormap.from_list(name + '_attention', ['#FFFFFF', color])
    cmap.set_bad('#ECECEC')
    matrix = matrices[model]
    mesh = ax.pcolormesh(np.arange(11), np.arange(len(matrix) + 1), matrix, cmap=cmap,
        vmin=0, vmax=1, shading='flat', rasterized=False)
    ax.set(xlim=(0, 10), ylim=(10, 0), xlabel='Needle')
    ax.set_xticks([.5, 3.5, 6.5, 9.5], [1, 4, 7, 10])
    targets = model_audits[model]['example_target_indices']
    ax.set_yticks(np.arange(10) + .5, range(10))
    if j == 0:
        ax.set_ylabel('Marker', labelpad=3)
    ax.tick_params(length=2, width=.6, pad=2)
    observed = model_audits[model]['observed_markers']
    ax.plot(np.asarray(observed) + .5, np.asarray(observed) + .5, 'o', ms=2.2,
        mfc='none', mec='#202020', mew=.6, linestyle='none')
    for marker in model_audits[model]['missing_markers']:
        ax.text(5, marker + .5, 'not captured', ha='center', va='center', fontsize=7, color='#666666')
    bar_ax = fig.add_axes([LEFTS[j] / WIDTH, .285 / HEIGHT, PANEL_WIDTH / WIDTH, .055 / HEIGHT])
    bar = fig.colorbar(mesh, cax=bar_ax, orientation='horizontal', ticks=[0, .5, 1],
        format=PercentFormatter(1, decimals=0))
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor('face')
    bar.ax.tick_params(length=2, pad=1, labelsize=8)
    bar.set_label('Share of needle attention', fontsize=8.5, labelpad=1)
    bar.outline.set_linewidth(.35)

    ax = axes[j + 2]
    rows = [r for r in rankings[model] if r['layer'] + 1 in model_audits[model]['displayed_layers_one_based']]
    offset = lambda rr: [row['layer'] + 1 + (row['head'] / nheads - .5) * .55 for row in rr]
    ax.scatter(offset(rows), [row['score'] for row in rows], s=4.3, color=color, alpha=.18, edgecolor='none')
    highlighted = [row for row in rows if (row['layer'], row['head']) in selected_banks[model]]
    ax.scatter(offset(highlighted), [row['score'] for row in highlighted], s=12, color=color,
        edgecolor='white', linewidth=.22, zorder=4)
    ax.set_title(name, fontsize=9, pad=4)
    ax.set(xlim=(0, nlayers + 1), ylim=(-.015, .61), xlabel='Layer', yticks=[0, .2, .4, .6])
    ax.set_xticks([1, 11, 21, 31] if j == 0 else [6, 18, 30, 42])
    if j == 0:
        ax.set_ylabel('Targeted score $T_h$', labelpad=3)
    ax.tick_params(length=3, width=.6, pad=2)
    ax.grid(axis='y', color='#E7E8EE', lw=.55)
    ax.set_axisbelow(True)
    ax.text(.04, .94, f'Top-{len(highlighted)}', transform=ax.transAxes, va='top', fontsize=8.5, color=color)

handles = [Line2D([], [], marker='o', ls='', ms=3.5, color='#7E8791', label='Ablated Top-$K$'),
           Line2D([], [], marker='o', ls='', ms=2.7, color='#7E8791', alpha=.25, label='Other heads')]
fig.legend(handles=handles, loc='lower center',
    bbox_to_anchor=((LEFTS[2] + LEFTS[3] + PANEL_WIDTH) / (2 * WIDTH), .045 / HEIGHT),
    ncol=2, frameon=False, handlelength=1.5, handletextpad=.45, columnspacing=1.0, borderaxespad=0)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
texts = [(t.get_text(), t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text)
         if t.get_visible() and t.get_text()]
outside = [s for s, b in texts if b.x0 < -.5 or b.y0 < -.5 or b.x1 > fig.bbox.width + .5 or b.y1 > fig.bbox.height + .5]
overlaps = [(s, t) for (s, a), (t, b) in combinations(texts, 2)
            if min(a.x1, b.x1) - max(a.x0, b.x0) > 1 and min(a.y1, b.y1) - max(a.y0, b.y0) > 1]
assert not outside, outside
assert not overlaps, overlaps
final = ROOT / 'output/pdf' / (STEM + '.pdf')
fig.savefig(final, metadata={'Title': 'Targeted attention during CoT enumeration', 'CreationDate': None})
fig.savefig(OUT / (STEM + '.svg'))
fig.savefig(OUT / (STEM + '.png'), dpi=300)
plt.close(fig)

manifest = {'status': 'PENDING_GEMMA_CAPTURE' if PENDING else ('PASS_WITH_DISCLOSED_GPU_REPLAY_DIFFERENCE' if numerical_reviews else 'PASS'),
    'numerical_reviews': numerical_reviews, 'sources_sha256': INPUTS, 'models': model_audits,
    'query_site': 'Main-figure Marker 0 initialization, Marker 1-9 item endpoints.',
    'heatmap_normalization': 'Each saved single-head query: sum over each full prompt-record span, then divide by total mass over all ten record spans.',
    'score_aggregation': 'Seed-first means at the registered ranking sites: Qwen post_marker (15 supporting discovery seeds), Gemma P0 (19 supporting discovery seeds).',
    'head_emphasis': 'Exact frozen ablation membership: Qwen 128, Gemma 6. No reselection.',
    'trace_selection': 'Qwen reuses the main-paper seed1255 ten-query capture. Gemma retains the previously selected seed1240 and head L30H5.',
    'scope': 'Single-trace illustrations, not cross-seed means. Pending previews visibly mask absent measurements and cannot replace the manuscript asset.',
    'one_based_display_indices': True, 'plot_axes': 4, 'size_inches': [WIDTH, HEIGHT],
    'heatmap_axes': {'x': 'Needle 1-10 (full prompt-record span)', 'y': 'Marker 0-9 (same meaning as main Fig. 1)'},
    'layout': {'outside_text': outside, 'text_overlaps': overlaps, 'panel_width_inches': PANEL_WIDTH,
        'panel_height_inches': PANEL_HEIGHT, 'equal_horizontal_gap_inches': LEFTS[1] - LEFTS[0] - PANEL_WIDTH},
    'typography': {'font': 'Times New Roman', 'math': 'STIX', 'group_titles_pt': 10.5,
        'model_titles_pt': 9, 'axis_labels_pt': 8.5, 'ticks_and_legend_pt': 8},
    'pdf_sha256': hashlib.sha256(final.read_bytes()).hexdigest(),
    'packages': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
    'elapsed_seconds': time.perf_counter() - START}
(OUT / ('manifest_pending.json' if PENDING else 'manifest.json')).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'status': manifest['status'], 'pdf': str(final), 'head_scores': len(scores),
    'example_cells': len(examples), 'source_files': len(INPUTS), 'seconds': manifest['elapsed_seconds']}))
