"""Plot saved, shared-basis PCA coordinates; no model inference or PCA refitting.

Run from the project root with python -s <this file>.
Display indices are one-based. Input metadata retain their zero-based indices.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
REPO = ROOT / 'realistic'
CUE_SOURCE = REPO / 'reports/v4_non-thinking_causal/v4_4_2/realistic_niah_v4_4_2_mode_geometry_attention_report.html'
DOMAIN_SOURCE = REPO / 'work/domain_transfer_geometry/analysis/report_payload.json'
STYLE_SOURCE = ROOT / 'figures/aurora_attention_pca_concept/assets/v4_source_v3_settings.json'
GRID_SOURCE = ROOT / 'figures/aurora_attention_pca_concept/assets/v4_source_v3.drawio'
COLORBAR_SOURCE = ROOT / 'figures/synthetic_appendix_label_revision_20260908/build_geometry.py'
MODELS = [('Qwen3-8B', 'Qwen', '#168DCA', 8, 24),
          ('Gemma4-E4B', 'Gemma', '#E87824', 9, 38)]
STYLE = json.loads(STYLE_SOURCE.read_text(encoding='utf-8'))
COUNT_COLORS = {int(k): v for k, v in STYLE['palette']['count_colors'].items()}
CENTROID_COLORS = {int(k): v for k, v in STYLE['palette']['centroid_colors'].items()}
INK, GRAY = '#30312E', '#64665F'
REFERENCE_SOURCE = REPO / 'reports/NiaH_Geometry_Comparison.html'
AXIS_COLOR = GRAY
GRID_COLOR = '#E9E8E2'
CAMERA = STYLE['representation']['camera']
DISPLAY_CAMERAS = {'cue': {'yaw_deg': math.degrees(-.72), 'pitch_deg': 52},
                   'domain': {'yaw_deg': math.degrees(-.72), 'pitch_deg': math.degrees(.46)}}
AXIS_SIGNS = np.ones(3)


def main_grid_styles():
    """Read the seven-line fade profile from the main figure's actual grid."""
    cells = {c.get('id'): c for c in ET.parse(GRID_SOURCE).getroot().iter('mxCell')}
    styles = []
    for cell_id in range(236, 243):
        attrs = dict(s.split('=', 1) for s in cells[f'v2_pca_{cell_id}'].get('style').split(';') if '=' in s)
        assert attrs['strokeColor'] == GRID_COLOR
        styles.append({'alpha': float(attrs['opacity']) / 100,
                       'linewidth_pt': float(attrs['strokeWidth']) * 6.5 * 72 / STYLE['canvas'][0]})
    return styles


GRID_STYLES = main_grid_styles()


def export_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_data():
    html = CUE_SOURCE.read_text(encoding='utf-8')
    marker = 'const PROMPT_GEOM='
    cue, _ = json.JSONDecoder().raw_decode(html[html.index(marker) + len(marker):])
    # The report's rowCoords function defines these raw shared-PCA offsets.
    assert "start=condition==='cue_present'?4:10" in html
    domain = json.loads(DOMAIN_SOURCE.read_text(encoding='utf-8'))
    assert domain['design']['city_discovery_seeds'] == list(range(1234, 1254))
    assert domain['design']['confirmation_seeds'] == list(range(1254, 1264))
    return cue, domain


def rotate(xyz, camera=None):
    """The reference HTML's rotate3: yaw followed by pitch, no axis flips."""
    camera = camera or CAMERA
    yaw, pitch = map(math.radians, [camera['yaw_deg'], camera['pitch_deg']])
    x, y, z = (np.asarray(xyz) * AXIS_SIGNS).T
    x1 = math.cos(yaw) * x + math.sin(yaw) * z
    z1 = -math.sin(yaw) * x + math.cos(yaw) * z
    y1 = math.cos(pitch) * y - math.sin(pitch) * z1
    return np.column_stack([x1, y1, math.sin(pitch) * y + math.cos(pitch) * z1])


def draw_panel(ax, series, evr, title, camera=None):
    camera = camera or CAMERA
    ax.text(0, 1.015, title, transform=ax.transAxes, fontsize=11,
            fontweight='normal', ha='left', va='bottom')
    ax.text(1, 1.015, f'PC1-3: {100 * sum(evr[:3]):.1f}%', transform=ax.transAxes,
            fontsize=9, color=GRAY, ha='right', va='bottom')
    all_xyz = np.array([[r['pc1'], r['pc2'], r['pc3']]
                        for group, *_ in series for r in group])
    all_rotated = rotate(all_xyz, camera)
    # Reproduce drawEndpointDomain3D's frame fitting. All conditions share
    # one camera and one pair of screen scales; source PC coordinates stay
    # untouched. The canvas fit is for display, not a distance statistic.
    axis_length = max(float(abs(all_xyz).max()), 1e-6) * .70
    ends = rotate(np.eye(3) * axis_length, camera)
    limits = np.vstack([all_rotated, ends, np.zeros((1, 3))])[:, :2]
    lower, upper = limits.min(axis=0), limits.max(axis=0)
    span = np.maximum(upper - lower, 1e-6)
    lower, upper = lower - .11 * span, upper + .11 * span
    def to_canvas(rotated):
        return .035 + .93 * (np.asarray(rotated)[..., :2] - lower) / (upper - lower)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect='auto')
    ax.set_axis_off()
    origin = to_canvas(np.zeros(3))
    # The main figure uses three projected line families through the PC origin,
    # each with seven evenly spaced lines, fading toward the panel edges.
    # Orient the same background with this row's retained camera; do not move
    # or rescale the plotted point coordinates to fit the background.
    for direction in to_canvas(ends) - origin:
        for k, style in zip(range(-3, 4), GRID_STYLES):
            if abs(direction[0]) < 1e-6:
                x = origin[0] + k / 8
                xy = np.array([[x, 0], [x, 1]])
            else:
                slope = direction[1] / direction[0]
                intercept = origin[1] + k / 6
                xy = np.array([[0, intercept - slope * origin[0]],
                               [1, intercept + slope * (1 - origin[0])]])
            ax.plot(xy[:, 0], xy[:, 1], color=GRID_COLOR,
                    linewidth=style['linewidth_pt'], alpha=style['alpha'],
                    zorder=-1, clip_on=True)
    for j, end in enumerate(to_canvas(ends)):
        ax.annotate('', xy=end, xytext=origin,
                    arrowprops={'arrowstyle': '-|>', 'color': AXIS_COLOR,
                                'linewidth': .65, 'mutation_scale': 7,
                                'shrinkA': 0, 'shrinkB': 0, 'alpha': .85}, zorder=0)
        label = ax.text(*(end + [.014, .026]), f'PC{j + 1}', fontsize=10,
                        color=AXIS_COLOR, ha='left', va='bottom', zorder=5)
        label._axis_endpoint = end
        label._axis_direction = end - origin
    ax._pca_markers = []
    zmin, zmax = all_rotated[:, 2].min(), all_rotated[:, 2].max()
    zspan = max(zmax - zmin, 1e-6)
    individuals, centroids = [], []
    for group, marker, filled, style in series:
        xyz = np.array([[r['pc1'], r['pc2'], r['pc3']] for r in group])
        projected = rotate(xyz, camera)
        xy = to_canvas(projected)
        labels = np.array([r['count_or_index'] for r in group])
        means = np.array([xy[labels == label].mean(axis=0) for label in range(1, 11)])
        mean_size = 24 if marker == 'o' else 35
        for point, label, depth in zip(xy, labels, projected[:, 2]):
            d = (depth - zmin) / zspan
            individuals.append((depth, point, int(label), marker, 6 + 7 * d, .34 + .52 * d))
        ax._pca_markers.extend((point, mean_size) for point in means)
        ax.plot(means[:, 0], means[:, 1], color=GRAY, lw=.85,
                linestyle=style, alpha=.70, zorder=1)
        centroids.append((means, marker, mean_size))
    for depth, point, label, marker, size, alpha in sorted(individuals, key=lambda item: item[0]):
        ax._pca_markers.append((point, size))
        ax.scatter(*point, s=size, marker=marker, facecolor=COUNT_COLORS[label],
                   edgecolor='white', linewidth=.4, alpha=alpha, zorder=2)
    for means, marker, size in reversed(centroids):
        ax.scatter(means[:, 0], means[:, 1], s=size, marker=marker,
                   facecolors=[CENTROID_COLORS[label] for label in range(1, 11)],
                   edgecolors=INK, linewidths=.55, zorder=3)
    return {'joint_screen_bounds': [lower.tolist(), upper.tolist()],
            'axis_length': axis_length,
            'three_pc_explained_variance': float(sum(evr[:3])), 'camera': camera}


def place_axis_labels(fig):
    """Move labels only, keeping them outside point markers and inside panels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        if not hasattr(ax, '_pca_markers'):
            continue
        points = [(ax.transData.transform(p), math.sqrt(size) * fig.dpi / 144)
                  for p, size in ax._pca_markers]
        for label in ax.texts:
            if not hasattr(label, '_axis_endpoint'):
                continue
            end = ax.transData.transform(label._axis_endpoint)
            candidates = []
            for distance in [5, 9, 14]:
                for dx, dy in [(1, 1), (-1, -1), (1, -1), (-1, 1), (0, 1), (0, -1)]:
                    pos = end + np.array([dx, dy]) * distance * fig.dpi / 72
                    label.set_position(ax.transData.inverted().transform(pos))
                    label.set_ha('left' if dx > 0 else 'right' if dx < 0 else 'center')
                    label.set_va('bottom' if dy > 0 else 'top')
                    box = label.get_window_extent(renderer)
                    collisions = sum(box.x0 - r - 1 < p[0] < box.x1 + r + 1
                                     and box.y0 - r - 1 < p[1] < box.y1 + r + 1
                                     for p, r in points)
                    outside = (box.x0 < ax.bbox.x0 + 2 or box.x1 > ax.bbox.x1 - 2
                               or box.y0 < ax.bbox.y0 + 2 or box.y1 > ax.bbox.y1 - 2)
                    outward = np.dot([dx, dy], label._axis_direction)
                    score = 1000 * outside + 100 * collisions + distance - .1 * outward
                    candidates.append((score, label.get_position(), label.get_ha(), label.get_va()))
            _, position, ha, va = min(candidates, key=lambda v: v[0])
            label.set_position(position)
            label.set_ha(ha)
            label.set_va(va)


def main():
    plt.rcParams.update({
        'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
        'font.size': 10, 'axes.labelsize': 10, 'xtick.labelsize': 9,
        'ytick.labelsize': 9, 'legend.fontsize': 9, 'axes.linewidth': .7,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
        'ytick.color': INK, 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    })
    cue, domain = source_data()
    fig = plt.figure(figsize=(6.5, 5.3))
    plotted, ncc_rows, control_stats, panel_meta = [], [], [], []
    for col, (model, short, color, cue_layer, domain_layer) in enumerate(MODELS):
        left = [.035, .535][col]
        cue_key = f'{model}|prompt_counter|{cue_layer}'
        raw = cue['datasets'][cue_key]
        assert len(raw['rows']) == 100
        assert {(int(r[0]), int(r[1])) for r in raw['rows']} == {
            (seed, i) for seed in range(1234, 1244) for i in range(1, 11)}
        groups = []
        for condition, offset in [('cue_present', 4), ('cue_absent', 10)]:
            groups.append([{'analysis': 'cue', 'model': model, 'site': 'needle_end',
                            'layer_source_zero_based': cue_layer,
                            'layer_display_one_based': cue_layer + 1,
                            'condition': condition, 'seed': int(r[0]),
                            'count_or_index': int(r[1]), 'pc1': float(r[offset]),
                            'pc2': float(r[offset + 1]), 'pc3': float(r[offset + 2])}
                           for r in raw['rows']])
        plotted.extend(groups[0] + groups[1])
        ax = fig.add_axes([left, .565, .430, .300])
        display = draw_panel(ax, [(groups[0], 'o', True, '-'), (groups[1], '^', False, '--')],
                             raw['evr_raw'], f'{"AB"[col]}. {short} L{cue_layer + 1}', DISPLAY_CAMERAS['cue'])
        stats = cue['statistics'][cue_key]
        control_stats.append({'model': model, 'layer_display_one_based': cue_layer + 1,
                              **{k: stats[k] for k in ['centroid_cka', 'r2_present', 'r2_absent']}})
        panel_meta.append({'panel': 'AB'[col], 'model': model,
                           'basis': 'Unwhitened PCA6 jointly fit on both cue conditions; raw, not condition-centered',
                           'coordinates': 'PC1, PC2, PC3', 'fit_rows': 200,
                           'plotted_rows': 200, 'evr': raw['evr_raw'][:3], 'display': display})

        mode = domain['models'][model]['non_thinking']
        vis = mode['visualization']
        assert mode['selected_layer'] == vis['layer'] == domain_layer
        assert vis['basis'] == 'StandardScaler + PCA3 fitted on city discovery rows only'
        assert vis['fit_rows'] == 200
        groups = []
        for condition in ['city', 'flower']:
            points = [p for p in vis['points'] if p['entity_domain'] == condition]
            assert {(p['seed'], p['gold_count']) for p in points} == {
                (seed, n) for seed in range(1254, 1264) for n in range(1, 11)}
            assert all(p['analysis_split'] == 'confirmation' for p in points)
            groups.append([{'analysis': 'domain', 'model': model, 'site': 'answer_query',
                            'layer_source_zero_based': domain_layer,
                            'layer_display_one_based': domain_layer + 1,
                            'condition': condition, 'seed': p['seed'],
                            'count_or_index': p['gold_count'], 'pc1': p['x'],
                            'pc2': p['y'], 'pc3': p['z']} for p in points])
        plotted.extend(groups[0] + groups[1])
        ax = fig.add_axes([left, .125, .430, .300])
        display = draw_panel(ax, [(groups[0], 'o', True, '-'), (groups[1], '^', False, '--')],
                             vis['explained_variance_ratio'], f'{"CD"[col]}. {short} L{domain_layer + 1}', DISPLAY_CAMERAS['domain'])
        panel_meta.append({'panel': 'CD'[col], 'model': model,
                           'basis': vis['basis'] + '; unwhitened',
                           'coordinates': 'PC1, PC2, PC3', 'fit_rows': 200,
                           'plotted_rows': 200, 'evr': vis['explained_variance_ratio'][:3], 'display': display})
        scores = mode['metrics']['count_by_evaluation_domain']
        for topic in ['city', 'flower', 'animal']:
            ncc_rows.append({'model': model, 'layer_display_one_based': domain_layer + 1,
                             'training_topic': 'city', 'evaluation_topic': topic,
                             'prompts': 100, 'seeds': 10,
                             'ncc_accuracy': scores[topic]['ncc_balanced_accuracy'],
                             'logistic_accuracy': scores[topic]['logistic_balanced_accuracy']})
    fig.text(.50, .990, 'Opening cue: needle-end states', ha='center', va='top', fontsize=11)
    fig.text(.50, .550, 'Needle domain: answer-query states', ha='center', va='top', fontsize=11)
    handles = [Line2D([], [], color=GRAY, marker='o', ms=4, lw=.8, mfc=GRAY),
               Line2D([], [], color=GRAY, marker='^', ms=4, lw=.8, ls='--', mfc=GRAY)]
    for labels, y in [(['Cue present', 'Cue absent'], .951), (['City', 'Flower'], .511)]:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.50, y),
                   ncol=2, frameon=False, handlelength=1.6, handletextpad=.4,
                   columnspacing=2, borderaxespad=0)
    # Match the synthetic appendix's compact discrete color key: ten equal
    # bins with white separators, endpoint labels and a centered descriptor.
    colors = ListedColormap([COUNT_COLORS[k] for k in range(1, 11)])
    boundaries = np.arange(.5, 11, 1)
    norm = BoundaryNorm(boundaries, colors.N)
    count_ax = fig.add_axes([.31, .04, .38, .017])
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=colors), cax=count_ax,
                           orientation='horizontal', boundaries=boundaries,
                           ticks=[1, 10], spacing='uniform', drawedges=True)
    colorbar.dividers.set_color('white')
    colorbar.dividers.set_linewidth(.65)
    colorbar.outline.set_linewidth(.6)
    colorbar.outline.set_edgecolor(INK)
    colorbar.ax.tick_params(labelsize=9, length=2, pad=2, width=.6)
    fig.text(.50, .005, 'Running index / final count',
             fontsize=9, ha='center', va='bottom')
    place_axis_labels(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text()]
    outside = [t.get_text() for t, b in texts if b.x0 < -.5 or b.y0 < -.5
               or b.x1 > fig.bbox.width + .5 or b.y1 > fig.bbox.height + .5]
    overlaps = []
    for i, (a, ab) in enumerate(texts):
        for b, bb in texts[i + 1:]:
            if min(ab.x1, bb.x1) - max(ab.x0, bb.x0) > .8 and min(ab.y1, bb.y1) - max(ab.y0, bb.y0) > .8:
                overlaps.append([a.get_text(), b.get_text()])
    text_marker_overlaps = []
    for ax in fig.axes:
        if not hasattr(ax, '_pca_markers'):
            continue
        for label in ax.texts:
            if label.get_text() not in ['PC1', 'PC2', 'PC3']:
                continue
            box = label.get_window_extent(renderer)
            for point, size in ax._pca_markers:
                x, y = ax.transData.transform(point)
                radius = math.sqrt(size) * fig.dpi / 144
                if box.x0 - radius < x < box.x1 + radius and box.y0 - radius < y < box.y1 + radius:
                    text_marker_overlaps.append({'axis_label': label.get_text(), 'panel': ax.texts[0].get_text()})
                    break
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(OUT / f'nonthinking_cue_domain_pca.{ext}', dpi=300)
    plt.close(fig)
    export_csv('pca_plot_data.csv', plotted)
    export_csv('domain_classifier_scores.csv', ncc_rows)
    export_csv('cue_control_statistics.csv', control_stats)
    manifest = {'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [CUE_SOURCE, DOMAIN_SOURCE, STYLE_SOURCE, REFERENCE_SOURCE,
                                           GRID_SOURCE, COLORBAR_SOURCE]},
                'model_inference': False, 'pca_refitted': False, 'display_indexing': 'one-based',
                'panels': panel_meta, 'individual_states_plotted': len(plotted),
                'canvas_inches': [6.5, 5.3], 'font': 'Times New Roman',
                'style_reference': 'Main-figure open white panels, neutral gray PC axes, light reference grid, count/centroid palettes; appendix Times New Roman typography. HTML is a reference only for rotation, point depth and centroid-path rendering.',
                'presentation': {'background': '#FFFFFF', 'panel_border': False,
                                 'axis_color': AXIS_COLOR, 'grid_color': GRID_COLOR,
                                 'grid': 'three PC-aligned families, seven lines per family, fading profile from main-figure drawio',
                                 'grid_styles': GRID_STYLES},
                'color_key': {'style': 'synthetic appendix discrete horizontal colorbar',
                              'bins': 10, 'ticks': [1, 10], 'label': 'Running index / final count',
                              'colors': [COUNT_COLORS[k] for k in range(1, 11)]},
                'count_colors': COUNT_COLORS, 'centroid_colors': CENTROID_COLORS,
                'display_cameras': DISPLAY_CAMERAS,
                'camera_selection': 'User-directed row sharing: identical camera for both models in each row, with the cue row tilted farther downward; not used in any statistical analysis',
                'axis_display_signs': AXIS_SIGNS.tolist(), 'projection': 'orthographic rotation followed by reference-HTML frame fitting; common screen transform across conditions within each panel',
                'font_points': {'title': 11, 'axis': 10, 'tick_legend': 9},
                'text_overlaps': overlaps, 'text_outside': outside,
                'axis_label_marker_overlaps': text_marker_overlaps}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'individual_states': len(plotted), 'text_overlaps': overlaps,
                      'text_outside': outside, 'axis_label_marker_overlaps': text_marker_overlaps}))
    assert not outside and not overlaps and not text_marker_overlaps


if __name__ == '__main__':
    main()
