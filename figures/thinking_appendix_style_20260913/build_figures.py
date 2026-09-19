"""Match the seven Thinking appendix figures to the current Non-thinking style.

Run with python -s <this file> from the workspace.
Reads frozen coordinates and summary tables only. No inference, fitting,
selection, resampling, or point filtering is performed by this renderer.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextPath
from matplotlib.ticker import PercentFormatter, FixedLocator, FixedFormatter, MaxNLocator
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
REP = ROOT / 'figures/paper_ncc_unification_20260913/data'
MECH = ROOT / 'figures/cot_appendix_20260912/mechanisms/data'
REFERENCE = ROOT / 'figures/nonthinking_appendix_revision_20260913/build_geometry.py'
MODELS = [('Qwen3-8B', 'Qwen', '#168DCA', 36), ('Gemma4-E4B', 'Gemma', '#E87824', 42)]
INK, GRAY, GRID = '#161923', '#8190A5', '#E7E8EE'
INPUTS, FIGURES = {}, {}
# Display rotations only. Both models share one camera within each row;
# all conditions in a panel rotate together.
PCA_CAMERAS = {
    ('Qwen3-8B', 'running_index'): {'yaw_deg': -165., 'pitch_deg': -25.},
    ('Gemma4-E4B', 'running_index'): {'yaw_deg': -165., 'pitch_deg': -25.},
    ('Qwen3-8B', 'final_count'): {'yaw_deg': 30., 'pitch_deg': -20.},
    ('Gemma4-E4B', 'final_count'): {'yaw_deg': 30., 'pitch_deg': -20.},
}
# The three-domain running-state panels use a slightly more rightward view.
DOMAIN_PCA_CAMERAS = {
    'running_index': {'yaw_deg': -150., 'pitch_deg': -25.},
    'final_count': {'yaw_deg': 30., 'pitch_deg': -20.},
}


def read(path):
    raw = path.read_bytes()
    INPUTS[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
    return raw.decode('utf-8-sig')


def rows(path):
    return list(csv.DictReader(read(path).split('\n')))


def setup():
    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
        'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'normal',
        'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 9, 'legend.frameon': False, 'axes.linewidth': .7,
        'axes.edgecolor': 'black', 'lines.linewidth': 1.5, 'lines.markersize': 4,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'text.color': INK, 'axes.labelcolor': INK, 'axes.titlecolor': INK,
        'xtick.color': INK, 'ytick.color': INK, 'figure.facecolor': 'white',
        'savefig.facecolor': 'white'})


def panel(ax, title, ylabel=None, xlabel=None):
    ax.set_title(title, loc='left', fontsize=11, fontweight='normal', pad=7)
    if ylabel: ax.set_ylabel(ylabel, labelpad=4)
    if xlabel: ax.set_xlabel(xlabel, labelpad=4)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color=GRID, linewidth=.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.7, pad=3)


def four_panels(height=4.65):
    fig = plt.figure(figsize=(6.5, height))
    axes = [[fig.add_axes([x, y, .365, .27]) for x in [.10, .61]] for y in [.60, .145]]
    return fig, axes


def stats(r):
    return tuple(float(r[k]) for k in ['mean', 'ci95_low', 'ci95_high'])


def errorbar(ax, x, row, color):
    m, lo, hi = stats(row)
    ax.errorbar(x, m, yerr=[[m-lo], [hi-m]], color=color, marker='o',
                ms=4, capsize=2.5, lw=1, zorder=4)


def save(fig, name, data, extra=None, pca_ref=None):
    assert fig.get_figwidth() == 6.5
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text()]
    outside = [t.get_text() for t, b in texts if b.x0 < -.5 or b.y0 < -.5
               or b.x1 > fig.bbox.width+.5 or b.y1 > fig.bbox.height+.5]
    overlaps = [[a.get_text(), b.get_text()] for i, (a, ab) in enumerate(texts)
                for b, bb in texts[i+1:] if min(ab.x1, bb.x1)-max(ab.x0, bb.x0) > .8
                and min(ab.y1, bb.y1)-max(ab.y0, bb.y0) > .8]
    assert not outside and not overlaps, (name, outside, overlaps)
    checks = {'text_outside': outside, 'text_overlaps': overlaps}
    if pca_ref: checks.update(pca_ref.verify(fig))
    artifacts = {}
    for ext in ['pdf', 'svg', 'png']:
        path = OUT / f'{name}.{ext}'
        fig.savefig(path, dpi=240, **({'metadata': {'CreationDate': None}} if ext == 'pdf' else {}))
        artifacts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    FIGURES[name] = {'canvas_inches': list(fig.get_size_inches()), 'checks': checks,
        'source_rows': len(data), 'plotted_records_sha256': hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()).hexdigest(),
        'font_sizes': sorted({t.get_fontsize() for t, _ in texts}),
        'fonts': sorted({t.get_fontfamily()[0] for t, _ in texts}),
        'artifacts_sha256': artifacts, **(extra or {})}
    plt.close(fig)
    print(name, 'PASS', flush=True)


def head_scores():
    data = rows(ROOT / 'figures/cot-reasoning/attention/data/head_scores.csv')
    assert Counter(r['model'] for r in data) == {'Qwen3-8B': 1152, 'Gemma4-E4B': 336}
    fig = plt.figure(figsize=(6.5, 6.8))
    rank_labels, shown, audits = [], [], []
    for i, (model, short, color, depth) in enumerate(MODELS):
        nheads, k = [(32, 128), (8, 6)][i]
        layers = list(range(1, 37)) if i == 0 else [6, 12, 18, 24, 30, 36, 42]
        group = [r for r in data if r['model'] == model and r['displayed_global_attention'] == 'True']
        mapping = {(int(r['layer']), int(r['head'])): r for r in group}
        assert set(mapping) == {(l, h) for l in layers for h in range(1, nheads+1)}
        chosen = json.loads(read(ROOT / f'realistic/configs/realistic_niah_v5_{short.lower()}_shared_k{k}_targeted_selection_frozen.json'))['development_selection']
        bank = {(l+1, h+1) for l, h in chosen['primary_bank_heads']}
        selected = [r for r in group if r['highlighted_ablation_bank'] == 'True']
        assert {(int(r['layer']), int(r['head'])) for r in selected} == bank
        assert {int(r['discovery_rank']) for r in selected} == set(range(1, k+1))
        matrix = np.array([[float(mapping[l, h]['mean_targeted_score']) for h in range(1, nheads+1)] for l in layers])
        assert np.isfinite(matrix).all() and 0 <= matrix.min() <= matrix.max() <= .6
        rect = [.095, .355, .775, .59] if i == 0 else [.095, .078, .775, .175]
        ax = fig.add_axes(rect)
        mesh = ax.pcolormesh(np.arange(nheads+1)+.5, np.arange(len(layers)+1)+.5,
            matrix, cmap=LinearSegmentedColormap.from_list(short, ['#F7FAFC', color]),
            norm=Normalize(0, .6), edgecolors='white', linewidth=.18, rasterized=False)
        ax.set(xlim=(.5, nheads+.5), ylim=(len(layers)+.5, .5))
        xt = [1, 5, 9, 13, 17, 21, 25, 29, 32] if i == 0 else list(range(1, 9))
        yt = list(range(1, 37, 3))+[36] if i == 0 else list(range(1, 8))
        ax.set_xticks(xt); ax.set_yticks(yt, [layers[j-1] for j in yt])
        ax.set_xlabel('Head index', labelpad=4)
        ax.set_ylabel('Layer' if i == 0 else 'Global layer', labelpad=5)
        ax.set_title(f'{"AB"[i]}. {short}: '+('all heads' if i == 0 else 'global-attention heads'),
                     loc='left', fontsize=11, fontweight='normal', pad=8)
        ax.tick_params(length=0, pad=4); ax.spines[:].set_visible(False)
        for r in selected:
            x, y = int(r['head']), layers.index(int(r['layer']))+1
            ax.add_patch(Rectangle((x-.5, y-.5), 1, 1, fill=False, edgecolor='#273743', linewidth=.65))
            t = ax.text(x, y, r['discovery_rank'], fontsize=7.5, fontweight='bold', color=INK,
                        ha='left', va='baseline')
            rank_labels.append((t, ax, x, y))
        cax = fig.add_axes([.902, rect[1], .022, rect[3]])
        cb = fig.colorbar(mesh, cax=cax, ticks=[0, .2, .4, .6])
        cb.solids.set_rasterized(False)
        cb.set_label(r'Targeted retrieval score $T_h$', fontsize=10, labelpad=4)
        cb.ax.tick_params(labelsize=9, length=2, width=.5, pad=2)
        cb.outline.set_linewidth(.5); cb.outline.set_edgecolor(GRAY)
        shown.extend(group)
        audits.append({'model': model, 'heads': len(group), 'selected': k, 'layers': layers})
    errors = []
    for t, ax, x, y in rank_labels:
        ink = TextPath((0, 0), t.get_text(), size=t.get_fontsize(), prop=t.get_fontproperties()).get_extents()
        center = ax.transData.transform([x, y])
        ink_center = np.array([(ink.x0+ink.x1)/2, (ink.y0+ink.y1)/2])
        baseline = center-ink_center*fig.dpi/72
        t.set_position(ax.transData.inverted().transform(baseline))
        corners = ax.transData.transform([[x-.5, y-.5], [x+.5, y+.5]])
        ink_box = np.array([[ink.x0, ink.y0], [ink.x1, ink.y1]])*fig.dpi/72+baseline
        assert np.all(ink_box.min(0) > corners.min(0)) and np.all(ink_box.max(0) < corners.max(0))
        errors.append(float(np.linalg.norm(ax.transData.transform(t.get_position())+ink_center*fig.dpi/72-center)*72/fig.dpi))
    save(fig, 'cot_full_head_scores', shown, {'models': audits, 'rank_labels': len(rank_labels),
        'rank_ink_center_max_error_pt': max(errors), 'rank_ink_inside_cells': True, 'raw_color_scale': [0, .6]})


def reference():
    read(REFERENCE)
    spec = importlib.util.spec_from_file_location('nonthinking_pca_style', REFERENCE)
    ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
    read(ref.STYLE_FILE); read(ref.GRID_FILE)
    return ref


def pca_panel(ax, data, evr, title, specs, ref, camera=None):
    """Non-thinking's display transform, generalized to unequal cohort sizes."""
    width = ax.get_position().width*ax.figure.get_figwidth()*72
    height = ax.get_position().height*ax.figure.get_figheight()*72
    ax.set(xlim=(0, width), ylim=(0, height), aspect='equal'); ax.set_axis_off()
    ax.text(0, 1.025, title, transform=ax.transAxes, ha='left', va='bottom', fontsize=11)
    ax.text(1, 1.025, f'PC1-3: {100*sum(evr):.1f}%', transform=ax.transAxes,
            ha='right', va='bottom', color=ref.GRAY, fontsize=9)
    xyz = np.array([[float(r[k]) for k in ['pc1', 'pc2', 'pc3']] for r in data])
    labels = np.array([int(r['count_or_index']) for r in data])
    assert np.isfinite(xyz).all() and set(labels) == set(range(1, 11))
    conditions = [s[0] for s in specs]
    masks = [np.array([r['display_condition'] == c for r in data]) for c in conditions]
    assert sum(m.sum() for m in masks) == len(data)
    assert all((m & (labels == k)).any() for m in masks for k in range(1, 11))
    means = np.array([xyz[m & (labels == k)].mean(0) for m in masks for k in range(1, 11)])
    center = xyz.mean(0); rms = max(float(np.sqrt(np.mean(np.sum((xyz-center)**2, axis=1)))), 1e-8)
    camera = camera if camera is not None else ref.CAMERAS['domain']
    rotated = ref.rotation((xyz-center)/rms, camera)
    mean_rotated = ref.rotation((means-center)/rms, camera)
    depth_scale = max(float(np.std(rotated[:, 2])), 1e-8)
    def project(r): return r[:, :2]/np.clip(1+.09*r[:, 2:3]/depth_scale, .78, 1.22)
    point_xy, mean_xy = project(rotated), project(mean_rotated)
    joined = np.vstack([point_xy, mean_xy]); lo, hi = joined.min(0), joined.max(0)
    fit = min((width-24)/max(hi[0]-lo[0], 1e-8), (height-22)/max(hi[1]-lo[1], 1e-8))
    shift = np.array([width, height])/2-(lo+hi)*fit/2
    point_xy, mean_xy = point_xy*fit+shift, mean_xy*fit+shift
    assert np.all(point_xy > 0) and np.all(point_xy < [width, height])
    directions = ref.rotation(np.eye(3), camera)[:, :2]
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    origin = np.array([width*.50, height*.45])
    for d in directions:
        normal = np.array([-d[1], d[0]])
        for k, (alpha, lw) in zip(range(-3, 4), ref.grid_styles()):
            anchor = origin+k*min(width, height)*.125*normal
            segment = np.vstack([anchor-400*d, anchor+400*d])
            ax.plot(segment[:, 0], segment[:, 1], color=ref.GRID, alpha=alpha, lw=lw, zorder=-2, clip_on=True)
    for j, d in enumerate(directions):
        end = origin+min(width*.36, height*.43)*d
        ax.annotate('', xy=end, xytext=origin, arrowprops={'arrowstyle': '-|>', 'color': ref.GRAY,
            'lw': .7, 'mutation_scale': 8, 'alpha': .8, 'shrinkA': 0, 'shrinkB': 0}, zorder=-1)
        text = ax.text(*end, f'PC{j+1}', color=ref.GRAY, fontsize=10, ha='center', va='center', zorder=5)
        text._axis_end, text._axis_direction = end, d
    ax.scatter(*origin, s=9, color=ref.GRAY, alpha=.8, zorder=-1)
    near = 1-(rotated[:, 2]-rotated[:, 2].min())/max(np.ptp(rotated[:, 2]), 1e-8)
    ax._point_boxes = []
    for n, (_, _, marker, line, _) in enumerate(specs):
        mxy = mean_xy[n*10:(n+1)*10]
        ax.plot(mxy[:, 0], mxy[:, 1], color=ref.GRAY, lw=.9, ls=line, alpha=.78, zorder=1)
    marker_map = {c: marker for c, _, marker, _, _ in specs}
    for idx in np.argsort(near):
        size = 7+4*near[idx]
        ax.scatter(*point_xy[idx], s=size, color=ref.COLORS[int(labels[idx])],
            marker=marker_map[data[idx]['display_condition']], alpha=.44+.40*near[idx],
            edgecolor='white', lw=.3, zorder=2)
        ax._point_boxes.append((point_xy[idx], size))
    for n in reversed(range(len(specs))):
        _, _, marker, _, size = specs[n]; mxy = mean_xy[n*10:(n+1)*10]
        ax.scatter(mxy[:, 0], mxy[:, 1], s=size, marker=marker,
            facecolors=[ref.MEAN_COLORS[k] for k in range(1, 11)], edgecolors=ref.INK,
            linewidths=.55, zorder=3+n*.01)
        ax._point_boxes.extend((p, size) for p in mxy)
    return {'title': title, 'points': len(data), 'condition_counts': dict(zip(conditions, [int(m.sum()) for m in masks])),
        'evr': evr, 'camera': camera, 'condition_alignment': 'one shared display transform per panel',
        'coordinate_sha256': hashlib.sha256(xyz.tobytes()).hexdigest()}


def place_domain_axis_labels(fig):
    """Keep labels off all three domains without moving the plotted states."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        if not hasattr(ax, '_point_boxes'):
            continue
        centers = ax.transData.transform(np.array([p for p, _ in ax._point_boxes]))
        radii = np.sqrt([size for _, size in ax._point_boxes])*fig.dpi/144+1
        for label in ax.texts:
            if not hasattr(label, '_axis_end'):
                continue
            end = ax.transData.transform(label._axis_end)
            candidates = []
            for distance in [5, 9, 14, 20, 27, 34, 42, 50, 60]:
                for angle in np.arange(0, 2*np.pi, np.pi/8):
                    direction = np.array([np.cos(angle), np.sin(angle)])
                    label.set_position(ax.transData.inverted().transform(end+direction*distance*fig.dpi/72))
                    label.set_ha('left' if direction[0] > .2 else 'right' if direction[0] < -.2 else 'center')
                    label.set_va('bottom' if direction[1] > .2 else 'top' if direction[1] < -.2 else 'center')
                    box = label.get_window_extent(renderer)
                    outside = (box.x0 < ax.bbox.x0+1 or box.x1 > ax.bbox.x1-1
                               or box.y0 < ax.bbox.y0+1 or box.y1 > ax.bbox.y1-1)
                    point_hits = ((centers[:, 0] > box.x0-radii) & (centers[:, 0] < box.x1+radii)
                                  & (centers[:, 1] > box.y0-radii) & (centers[:, 1] < box.y1+radii))
                    text_hits = sum(box.overlaps(other.get_window_extent(renderer))
                                    for other in ax.texts if other is not label)
                    cost = 10000*outside+1000*(int(point_hits.sum())+text_hits)+distance
                    candidates.append((cost, label.get_position(), label.get_ha(), label.get_va()))
            cost, position, ha, va = min(candidates, key=lambda candidate: candidate[0])
            assert cost < 1000, ('No collision-free axis-label position', label.get_text())
            label.set_position(position)
            label.set_ha(ha)
            label.set_va(va)


def pca_figures(only=None):
    ref = reference(); meta = json.loads(read(REP/'metadata.json'))
    for name, filename, endpoints, headings in [
        ('cot_count_pca', 'canonical_selected_pca.csv', ['running_index', 'final_count'],
         ['Running index: item-end states', 'Final count: answer-query states']),
        ('cot_domain_pca', 'domain_selected_pca.csv', ['running_index', 'answer_token'],
         ['Needle domain: running-index states', 'Needle domain: final-count states'])]:
        if only is not None and name not in only:
            continue
        source = rows(REP/filename)
        domain = name == 'cot_domain_pca'
        data = [r for r in source if r['split'] == 'confirmation']
        fig = plt.figure(figsize=(6.5, 5.25)); panels = []
        specs = ref.CONDITIONS['domain'] if domain else [('canonical', 'Canonical', 'o', '-', 25)]
        if domain:
            assert {r['domain'] for r in data} == {s[0] for s in specs} == {'city', 'flower', 'animal'}
        for row, endpoint in enumerate(endpoints):
            for col, (model, short, _, _) in enumerate(MODELS):
                group = [dict(r, display_condition=r['domain'] if domain else 'canonical') for r in data if r['model'] == model and r['endpoint'] == endpoint]
                setting = meta['domain' if domain else 'canonical']['models'][model][endpoint]
                layer = setting['selected_layer_display_one_based']
                assert {int(r['layer_display_one_based']) for r in group} == {layer}
                if not domain: assert len(group) == setting['states_by_split']['confirmation']
                evr = setting['pca3_explained_variance_ratio' if domain else 'evr']
                ax = fig.add_axes([[.03, .53][col], [.562, .106][row], .44, .320])
                camera_row = 'running_index' if row == 0 else 'final_count'
                camera = DOMAIN_PCA_CAMERAS[camera_row] if domain else PCA_CAMERAS[(model, camera_row)]
                panels.append(pca_panel(ax, group, evr, f'{"ABCD"[2*row+col]}. {short} L{layer}', specs, ref, camera))
        assert panels[0]['camera'] == panels[1]['camera']
        assert panels[2]['camera'] == panels[3]['camera']
        for row, heading in enumerate(headings):
            y = [.993, .537][row]
            fig.text(.5, y, heading, fontsize=11, ha='center', va='top')
            if domain:
                fig.legend([Line2D([], [], color=ref.GRAY, lw=.9, ls=s[3], marker=s[2], ms=4) for s in specs],
                    [s[1] for s in specs], loc='upper center', bbox_to_anchor=(.5, y-.035), ncol=len(specs),
                    frameon=False, fontsize=9, handlelength=1.6, handletextpad=.4,
                    borderaxespad=0, borderpad=.05, columnspacing=2)
        cmap = ListedColormap([ref.COLORS[k] for k in range(1, 11)]); bounds = np.arange(.5, 11, 1)
        cax = fig.add_axes([.31, .035, .38, .017])
        cb = fig.colorbar(ScalarMappable(norm=BoundaryNorm(bounds, cmap.N), cmap=cmap), cax=cax,
            orientation='horizontal', boundaries=bounds, ticks=[1, 10], drawedges=True)
        cb.dividers.set_color('white'); cb.dividers.set_linewidth(.65)
        cb.outline.set_edgecolor(ref.INK); cb.outline.set_linewidth(.6)
        cb.ax.tick_params(labelsize=9, length=2, pad=2, width=.6)
        fig.text(.5, .001, 'Running index / final count', fontsize=9, ha='center', va='bottom')
        if domain:
            place_domain_axis_labels(fig)
        else:
            ref.place_labels(fig)
        save(fig, name, data, {'panels': panels, 'coordinates_refit': False, 'subsampled': False}, ref)


def readouts():
    data = rows(REP/'canonical_layerwise_readouts.csv')
    fig = plt.figure(figsize=(6.5, 2.55))
    axes = [fig.add_axes([x, .215, .400, .535]) for x in [.095, .580]]
    plotted, selected_layers = [], []
    for j, (ax, endpoint) in enumerate(zip(axes, ['running_index', 'final_count'])):
        label = ['Running index', 'Final count'][j]
        panel(ax, f'{"AB"[j]}. {label}', 'Balanced accuracy' if j == 0 else None, 'Layer')
        for model, _, color, depth in MODELS:
            rr = sorted([r for r in data if r['model'] == model and r['endpoint'] == endpoint], key=lambda r: int(r['layer_display_one_based']))
            assert [int(r['layer_display_one_based']) for r in rr] == list(range(1, depth+1))
            chosen, = [r for r in rr if r['is_selected_layer'] == '1']
            for method, ls in [('ncc', '-'), ('logistic', '--')]:
                values = [float(r[f'discovery_cv_{method}']) for r in rr]
                assert all(0 <= value <= 1 for value in values)
                ax.plot([int(r['layer_display_one_based']) for r in rr], values, color=color, ls=ls)
                plotted.extend({'model': model, 'endpoint': endpoint, 'split': 'discovery_cv',
                    'layer_display_one_based': int(r['layer_display_one_based']),
                    'method': method, 'balanced_accuracy': value} for r, value in zip(rr, values))
            layer, score = int(chosen['layer_display_one_based']), float(chosen['discovery_cv_ncc'])
            ax.scatter(layer, score, s=27, color=color, edgecolor='white', linewidth=.6, zorder=4)
            ax.annotate(f'L{layer}', (layer, score), xytext=(0, 8), textcoords='offset points',
                        ha='center', fontsize=8, color=color)
            selected_layers.append({'model': model, 'endpoint': endpoint, 'layer': layer,
                                    'discovery_ncc': score})
        ax.set(xlim=(.5, 42.5), xticks=[1, 10, 20, 30, 42], ylim=(0, 1.12), yticks=[0, .2, .4, .6, .8, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.axhline(.1, color=GRAY, ls=':', lw=.9)
    fig.legend([Line2D([], [], color=m[2]) for m in MODELS]+[Line2D([], [], color=INK, ls=s) for s in ['-', '--']],
        ['Qwen', 'Gemma', 'Nearest centroid', 'Logistic'], loc='upper center', bbox_to_anchor=(.535, .995),
        ncol=4, handlelength=1.7, handletextpad=.45, columnspacing=1.4)
    assert len(plotted) == 312
    save(fig, 'cot_count_readouts', plotted, {'source_table_rows': len(data),
        'panel_count': 2, 'displayed_split': 'discovery_cv', 'confirmation_curves_plotted': False,
        'selected_layers': selected_layers, 'reference_canvas': 'Non-thinking: 6.5 x 2.55 inches'})


def dose():
    data = rows(ROOT/'figures/cot_completion_20260912/cot_current_bank_dose.csv')
    fig, axs = four_panels()
    for i, (model, short, color, _) in enumerate(MODELS):
        for j, measure in enumerate(['next_item', 'final_count']):
            ax = axs[i][j]
            panel(ax, f'{"ABCD"[2*i+j]}. {short}: '+['next item', 'final count'][j], 'Ablation effect', r'Number of ablated heads, $K$')
            ks = None
            for cond, ls, marker in [('selected_bank', '-', 'o'), ('layer_matched_random', '--', 'D')]:
                rr = sorted([r for r in data if r['model'] == model and r['outcome'] == measure and r['condition'] == cond], key=lambda r: int(r['k']))
                ks = [int(r['k']) for r in rr]
                assert ks == ([1, 2, 4, 8, 16, 32, 64, 128] if i == 0 else [1, 2, 4, 6])
                val = np.array([stats(r) for r in rr])
                ax.plot(ks, val[:, 0], color=color, ls=ls, marker=marker, ms=3.5)
                ax.fill_between(ks, val[:, 1], val[:, 2], color=color, alpha=.12, lw=0)
            ax.set_xscale('log', base=2); ax.set_xlim(ks[0]/1.12, ks[-1]*1.12)
            ax.xaxis.set_major_locator(FixedLocator(ks)); ax.xaxis.set_major_formatter(FixedFormatter([str(k) for k in ks])); ax.minorticks_off()
            ax.set(ylim=(min(-.04, min(float(r['ci95_low']) for r in data)-.04), 1.04), yticks=[0, .5, 1])
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0)); ax.axhline(0, color=GRAY, lw=.7)
    fig.legend([Line2D([], [], color=INK, marker='o'), Line2D([], [], color=INK, ls='--', marker='D')],
        ['Frozen Top-$K$ prefix', 'Layer-matched random'], loc='upper center', bbox_to_anchor=(.535, .998), ncol=2)
    save(fig, 'cot_current_bank_dose', data)


def progress():
    # Use the complete fixed-layer rerun; fail if its audited grid is unavailable.
    from types import SimpleNamespace
    source = ROOT / 'figures/thinking_update_concision_20260915/build_scope_figure.py'
    spec = importlib.util.spec_from_file_location('cot_update_scope_figure', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    read(source)
    module.render(SimpleNamespace(read=read, MODELS=MODELS, panel=panel, INK=INK, save=save))


def answer_controls():
    blank = rows(MECH/'answer_blanking.csv')
    fig = plt.figure(figsize=(6.5, 2.5))
    axs = [fig.add_axes([x, .24, .365, .59]) for x in [.10, .61]]
    for col, (model, short, color, _) in enumerate(MODELS):
        ax = axs[col]
        panel(ax, f'{"AB"[col]}. {short}', 'Exact-count accuracy')
        for x, cond in enumerate(['clean', 'prompt_records_blank', 'trace_all_blank']):
            r, = [r for r in blank if r['model'] == model and r['condition'] == cond]
            assert int(r['prompts']) == 100 and int(r['seeds']) == 10
            errorbar(ax, x, r, color)
        ax.set(xticks=[0, 1, 2], xticklabels=['Original', 'Prompt\nblank', 'Trace\nblank'],
               xlim=(-.4, 2.4), ylim=(-.04, 1.06), yticks=[0, .5, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    save(fig, 'cot_answer_readout_controls', blank,
         extra={'conditions': ['clean', 'prompt_records_blank', 'trace_all_blank'],
                'metric': 'Exact-count accuracy', 'prompts_per_model': 100,
                'seed_clusters_per_model': 10, 'statistics_recomputed': False})


def main():
    start = time.perf_counter(); setup()
    head_scores(); pca_figures(); readouts(); dose(); progress(); answer_controls()
    assert len(FIGURES) == 7
    assert all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == value for path, value in INPUTS.items())
    manifest = {'status': 'PASS', 'figures': FIGURES, 'source_sha256': INPUTS,
        'model_inference': False, 'refitting': False, 'layer_selection_changed': False,
        'statistics_recomputed': False, 'data_source': 'Current NCC-selected coordinates and frozen corrected causal tables',
        'font': 'Times New Roman / STIX', 'font_points': {'title': 11, 'axis': 10, 'ticks': 9, 'legend': 9, 'dense_head_rank': 7.5},
        'reference': 'Current Non-thinking appendix: 6.5 inch width; regular panel titles; black spines; common colors and PCA rendering.',
        'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'elapsed_seconds': time.perf_counter()-start}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'figures': len(FIGURES), 'seconds': manifest['elapsed_seconds']}))


if __name__ == '__main__':
    main()
