"""Within-cell Enumeration representation figures, aligned with the Thinking appendix.

Run with python -s <this file>. Renders the September 17 own-state CPU analysis with discovery NCC selection
and discovery-fitted PCA. No model inference. See representation_manifest.json.
"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PAPER = ROOT / 'runs/paper_figures/figures/enumeration_appendix'
ARCHIVE = ROOT / 'realistic/outputs/enumeration_replay_midlayer_20260917/pca3_randomized_v2'
METRICS = ARCHIVE
MODELS = [('Qwen3-8B', 'Qwen', '#168DCA', 36), ('Gemma4-E4B', 'Gemma', '#E87824', 42)]
MODES = ['enumeration_index', 'enumeration_bullet']
INPUTS, FIGURES = {}, {}


def read(path):
    raw = path.read_bytes()
    INPUTS[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
    return raw.decode('utf-8-sig')


def module(name, path):
    read(path)
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def csv_write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, name, rows, extra, ref=None):
    style.save(fig, name, rows, extra, ref)
    FIGURES[name] = style.FIGURES[name]
    shutil.copy2(OUT / (name + '.pdf'), PAPER / (name + '.pdf'))


def readouts():
    fig, axes = style.four_panels(height=3.85)
    plotted, chosen_rows = [], []
    for i, endpoint in enumerate(['running_index', 'final_count']):
        path = METRICS / f'{endpoint}_candidate_metrics.csv'
        candidates = list(csv.DictReader(read(path).splitlines()))
        saved_hash = selection['source_sha256'][path.name]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == saved_hash
        for j, mode in enumerate(MODES):
            ax = axes[i][j]
            title = f'{"ABCD"[2*i+j]}. {"Index" if j == 0 else "Bullet"}: '
            title += 'running index' if i == 0 else 'final count'
            style.panel(ax, title, 'Balanced accuracy', 'Layer')
            for model, short, color, depth in MODELS:
                rr = sorted([r for r in candidates if r['model_label'] == model and r['prompt_mode'] == mode], key=lambda r: int(r['layer']))
                assert [int(r['layer']) for r in rr] == list(range(depth))
                chosen, = [r for r in selection['selected'][endpoint] if r['model_label'] == model and r['prompt_mode'] == mode]
                # Verification of the already-frozen rule, without choosing a new layer.
                expected = max(rr, key=lambda r: (round(float(r['discovery_oof_ncc_balanced_accuracy']), 12), round(float(r['discovery_oof_logistic_balanced_accuracy']), 12), -int(r['layer'])))
                assert int(chosen['layer']) == int(expected['layer'])
                for metric in ['discovery_oof_ncc_balanced_accuracy', 'discovery_oof_logistic_balanced_accuracy']:
                    assert abs(float(chosen[metric]) - float(expected[metric])) < 1e-12
                for method, ls in [('ncc', '-'), ('logistic', '--')]:
                    values = [float(r[f'discovery_oof_{method}_balanced_accuracy']) for r in rr]
                    assert all(0 <= v <= 1 for v in values)
                    ax.plot([int(r['layer']) + 1 for r in rr], values, color=color, ls=ls)
                    plotted.extend({'format': mode, 'model': model, 'endpoint': endpoint, 'split': 'discovery_oof', 'method': method, 'layer_display_one_based': int(r['layer']) + 1, 'balanced_accuracy': value, 'states': int(r['discovery_oof_rows'])} for r, value in zip(rr, values))
                layer, value = int(chosen['layer']) + 1, float(chosen['discovery_oof_ncc_balanced_accuracy'])
                ax.scatter(layer, value, s=27, color=color, edgecolor='white', linewidth=.6, zorder=4)
                ax.annotate(f'L{layer}', (layer, value), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9, color=color)
                chosen_rows.append(dict(chosen, layer_display_one_based=layer))
            ax.set(xlim=(.5, 42.5), xticks=[1, 10, 20, 30, 42], ylim=(0, 1.15), yticks=[0, .5, 1])
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.axhline(.1, color=style.GRAY, ls=':', lw=.9)
    fig.legend([Line2D([], [], color=m[2]) for m in MODELS] + [Line2D([], [], color=style.INK, ls=s) for s in ['-', '--']], ['Qwen', 'Gemma', 'Nearest centroid', 'Logistic'], loc='upper center', bbox_to_anchor=(.535, .995), ncol=4, handlelength=1.7, handletextpad=.45, columnspacing=1.4)
    assert len(plotted) == 624
    csv_write(OUT / 'data/readout_discovery_curves.csv', plotted)
    (OUT / 'data/selected_layers.json').write_text(json.dumps(chosen_rows, indent=2), encoding='utf-8')
    save(fig, 'enumeration_representations', plotted, {'displayed_split': 'discovery_oof', 'confirmation_curves_plotted': False, 'selected_layers': chosen_rows})


def pca():
    coordinates = list(csv.DictReader(read(ARCHIVE / 'data/enumeration_pca_coordinates.csv').splitlines()))
    metadata = json.loads(read(ARCHIVE / 'pca_manifest.json'))
    # The source CSV retains the full within-cell population after the randomized-SVD alignment.
    assert len(coordinates) == sum(p['confirmation_display_rows'] for f in metadata['figures'] for p in f['panels'])
    ref = style.reference()
    INPUTS.update(style.INPUTS)
    all_rows = []
    for mode in MODES:
        fig = plt.figure(figsize=(6.5, 5.25))
        panels, shown = [], []
        source_figure, = [r for r in metadata['figures'] if r['name'] == 'enumeration_pca_' + mode.split('_')[-1]]
        for i, endpoint in enumerate(['running', 'final']):
            for j, (model, short, _, _) in enumerate(MODELS):
                group = [dict(r, display_condition='enumeration') for r in coordinates if r['format'] == mode and r['model'] == model and r['endpoint'] == endpoint]
                meta, = [p for p in source_figure['panels'] if p['cell'] == mode + '|' + model and p['endpoint'] == endpoint]
                chosen, = [r for r in selection['selected']['running_index' if i == 0 else 'final_count'] if r['prompt_mode'] == mode and r['model_label'] == model]
                layer = int(chosen['layer']) + 1
                assert {int(r['layer_display_one_based']) for r in group} == {layer} == {meta['layer_display_one_based']}
                assert len(group) == meta['confirmation_display_rows']
                ax = fig.add_axes([[.03, .53][j], [.562, .106][i], .44, .320])
                # Same display renderer and row-specific camera as Thinking; no changes to PCA coordinates.
                camera = style.PCA_CAMERAS[(model, 'running_index' if i == 0 else 'final_count')]
                panel = style.pca_panel(ax, group, meta['discovery_explained_variance_ratio'], f'{"ABCD"[2*i+j]}. {short} L{layer}', [('enumeration', 'Enumeration', 'o', '-', 25)], ref, camera)
                panel.update(endpoint=endpoint, model=model, layer_display_one_based=layer)
                panels.append(panel)
                shown.extend(group)
                all_rows.extend(group)
        for y, title in zip([.993, .537], ['Running index: item-end states', 'Final count: answer-query states']):
            fig.text(.5, y, title, fontsize=11, ha='center', va='top')
        cmap = ListedColormap([ref.COLORS[k] for k in range(1, 11)])
        bounds = np.arange(.5, 11, 1)
        cax = fig.add_axes([.31, .035, .38, .017])
        cb = fig.colorbar(ScalarMappable(norm=BoundaryNorm(bounds, cmap.N), cmap=cmap), cax=cax, orientation='horizontal', boundaries=bounds, ticks=[1, 10], drawedges=True)
        cb.dividers.set_color('white'); cb.dividers.set_linewidth(.65)
        cb.outline.set_edgecolor(ref.INK); cb.outline.set_linewidth(.6)
        cb.ax.tick_params(labelsize=9, length=2, pad=2, width=.6)
        fig.text(.5, .001, 'Running index / final count', fontsize=9, ha='center', va='bottom')
        style.place_domain_axis_labels(fig)
        save(fig, 'enumeration_pca_' + mode.split('_')[-1], shown, {'panels': panels, 'coordinates_refit': True, 'fit_scope': 'discovery only; each cell own available states; randomized SVD seed 0 and sklearn native signs', 'points_subsampled': False, 'axis_signs_modified': False, 'layout': 'rows running/final; columns Qwen/Gemma'}, ref)
    csv_write(OUT / 'data/pca_coordinates.csv', all_rows)
    assert sorted(tuple(r[k] for k in coordinates[0]) for r in all_rows) == sorted(tuple(r[k] for k in coordinates[0]) for r in coordinates)


if __name__ == '__main__':
    PAPER.mkdir(parents=True, exist_ok=True)
    (OUT / 'data').mkdir(exist_ok=True)
    selection = json.loads(read(ARCHIVE / 'selection.json'))
    style = module('thinking_appendix_renderer', ROOT / 'figures/thinking_appendix_style_20260913/build_figures.py')
    style.OUT = OUT
    style.setup()
    readouts()
    pca()
    INPUTS.update(style.INPUTS)
    read(Path(__file__))
    (OUT / 'representation_manifest.json').write_text(json.dumps({'source_sha256': INPUTS, 'figures': FIGURES,
        'model_inference': False, 'analysis_population': 'each cell own available original states',
        'analysis_manifest': str(ARCHIVE.relative_to(ROOT) / 'analysis_manifest.json'),
        'font_profile': {'family': 'Times New Roman', 'titles_pt': 11, 'axes_pt': 10, 'ticks_legend_pt': 9, 'width_inches': 6.5}}, indent=2), encoding='utf-8')
    print('PASS: within-cell readouts and PCA rendered.')
