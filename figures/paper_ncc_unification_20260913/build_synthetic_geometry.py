r"""\gpt: running-index titles, equal 3D frames, and a discrete count color key."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[1]
PREVIOUS = WORK / 'figures/synthetic_appendix_evidence_revision_20260908'
PAPER = WORK / 'runs/paper_figures/figures/synthetic_appendix/03_count_geometry_3d.pdf'
sys.path.insert(0, str(PREVIOUS))
import build_figures as original
from audit_style import inspect_figure
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

PANEL_LAYERS = [2, 4, 4, 2]
selection = json.loads((OUT / 'selection.json').read_text())['selected']
assert PANEL_LAYERS == [selection['nonthinking']['nonthinking_prompt_occurrence'],
                        selection['thinking']['thinking_item_end'],
                        selection['nonthinking']['nonthinking_answer_query'],
                        selection['thinking']['thinking_answer_query']]
PANEL_AUDITS = []


def place_pc_labels(fig):
    r"""\gpt: use one physical gap from each projected axis to its PC label."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72
    axes_info = []
    tick_extent = 0.
    for ax in fig.axes[:4]:
        plot_center = ax.bbox.get_points().mean(axis=0)
        for axis, label in zip([ax.xaxis, ax.yaxis, ax.zaxis], ['PC1', 'PC2', 'PC3']):
            ends = axis.line.get_transform().transform(axis.line.get_xydata())
            midpoint = ends.mean(axis=0)
            direction = ends[-1] - ends[0]
            normal = np.array([-direction[1], direction[0]])
            normal /= np.linalg.norm(normal)
            if np.dot(normal, midpoint - plot_center) < 0:
                normal *= -1
            lo, hi = axis.get_view_interval()
            for tick in axis.get_major_ticks():
                if lo <= tick.get_loc() <= hi and tick.label1.get_visible():
                    bbox = tick.label1.get_window_extent(renderer)
                    corners = np.array([[x, y] for x in [bbox.x0, bbox.x1]
                                        for y in [bbox.y0, bbox.y1]])
                    tick_extent = max(tick_extent, np.max((corners - midpoint) @ normal))
            axes_info.append((label, midpoint, normal))
    gap = tick_extent + 4 * px_per_pt
    label_audits = []
    for label, midpoint, normal in axes_info:
        item = fig.text(.5, .5, label, fontsize=10, ha='center', va='center')
        bbox = item.get_window_extent(renderer)
        support = .5 * (abs(normal[0]) * bbox.width + abs(normal[1]) * bbox.height)
        position = midpoint + normal * (gap + support)
        item.set_position(fig.transFigure.inverted().transform(position))
        label_audits.append({'label': label, 'axis_gap_pt': gap / px_per_pt})
    return label_audits


def save_relabelled(fig, _stem, _title):
    # \gpt: enlarge equal frames and reclaim unused space around the four panels.
    fig.set_size_inches(6.5, 5.65)
    replacements = {
        'A. Non-thinking: occurrence index': 'A. Non-thinking: running index (L2)',
        'B. Thinking: occurrence index': 'B. Thinking: running index (L4)',
        'C. Non-thinking: final count': 'C. Non-thinking: final count (L4)',
        'D. Thinking: final count': 'D. Thinking: final count (L2)',
    }
    found = set()
    for item in fig.texts:
        text = item.get_text()
        if text in replacements:
            item.set_text(replacements[text])
            found.add(text)
    assert found == set(replacements)
    colors = ListedColormap(original.CMAP(np.linspace(0, 1, 10)), name='aurora_counts')
    boundaries = np.arange(.5, 11, 1)
    norm = BoundaryNorm(boundaries, colors.N)
    frame_sizes = []
    line_style = dict(width_pt=.6, axis_color='#161923', grid_color='#DCE1E9')
    line_audits = []
    view_audits = []
    for panel_index, ax in enumerate(fig.axes[:4]):
        column, row = panel_index % 2, panel_index // 2
        ax.set_position([.012 + .5 * column, .565 if row == 0 else .11, .435, .40])
        for label in [ax.xaxis.label, ax.yaxis.label, ax.zaxis.label]:
            label.set_text('')
        for item in list(ax.texts):
            if item.get_text() == 'PC3':
                item.remove()
        # \gpt: fit the cube around the complete cloud, rather than around zero.
        # Keep equal coordinate units, identical frames, and a small safety margin.
        xyz = np.column_stack(ax.collections[0]._offsets3d).astype(float)
        centers = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
        radius = float(1.05 * np.ptp(xyz, axis=0).max() / 2)
        for center, setter, ticker in zip(centers,
                                         [ax.set_xlim, ax.set_ylim, ax.set_zlim],
                                         [ax.set_xticks, ax.set_yticks, ax.set_zticks]):
            setter(center - radius, center + radius, view_margin=0)
            ticks = MaxNLocator(nbins=3).tick_values(center - radius, center + radius)
            ticker(ticks[(ticks > center - .88 * radius) & (ticks < center + .88 * radius)])
        assert np.all(np.abs(xyz - centers) < radius)
        ax.set_box_aspect((1, 1, 1), zoom=1.12)
        view_audits.append({'center': centers.tolist(), 'radius': radius,
                            'zoom': 1.12, 'all_points_inside': True})
        # Explicitly match every 3D line artist; rcParams alone misses panes/ticks.
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.line.set_linewidth(line_style['width_pt'])
            axis.line.set_color(line_style['axis_color'])
            axis.line.set_alpha(1)
            axis._axinfo['axisline'].update(linewidth=line_style['width_pt'],
                                            color=line_style['axis_color'])
            axis._axinfo['grid'].update(linewidth=line_style['width_pt'],
                                        color=line_style['grid_color'], linestyle='-')
            axis._axinfo['tick']['linewidth'] = {True: line_style['width_pt'], False: line_style['width_pt']}
            axis.pane.set_fill(False)
            axis.pane.set_alpha(1)
            axis.pane.set_linewidth(line_style['width_pt'])
            axis.pane.set_edgecolor(line_style['grid_color'])
            axis.gridlines.set_alpha(1)
            axis.line.set_solid_capstyle('butt')
            axis.gridlines.set_capstyle('butt')
            axis.pane.set_joinstyle('miter')
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                tick.tick1line.set_markeredgewidth(line_style['width_pt'])
                tick.tick2line.set_markeredgewidth(line_style['width_pt'])
            line_audits.append({
                'axis_pt': axis.line.get_linewidth(), 'pane_pt': axis.pane.get_linewidth(),
                'grid_pt': axis._axinfo['grid']['linewidth'],
                'major_tick_pt': axis._axinfo['tick']['linewidth'][True],
            })
        for collection in ax.collections:
            if collection.get_array() is not None:
                collection.set_cmap(colors)
                collection.set_norm(norm)
        frame_sizes.append(ax.get_position().bounds[2:])
    assert np.allclose(frame_sizes, frame_sizes[0])
    for item in fig.texts:
        if item.get_text() in replacements.values() or item.get_text().startswith(('C. ', 'D. ')):
            index = ord(item.get_text()[0]) - ord('A')
            item.set_position((.025 + .5 * (index % 2), .969 if index < 2 else .505))
    cax = fig.axes[-1]
    cax.set_position([.31, .04, .38, .017])
    cax.clear()
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=colors), cax=cax,
                           orientation='horizontal', boundaries=boundaries,
                           ticks=[1, 10], spacing='uniform', drawedges=True)
    colorbar.dividers.set_color('white')
    colorbar.dividers.set_linewidth(.65)
    colorbar.ax.tick_params(labelsize=9, length=2, pad=2)
    for item in fig.texts:
        if item.get_text() == 'Occurrence index / final count':
            item.set_text('Running index / final count')
            item.set_position((.5, .005))
    label_audits = place_pc_labels(fig)
    fig.canvas.draw()
    for item in fig.texts:
        if item.get_text() in ['PC1', 'PC2', 'PC3']:
            assert not item.get_window_extent(fig.canvas.get_renderer()).overlaps(cax.bbox)
    audit = inspect_figure(fig, '03_count_geometry_3d')
    (OUT / 'qa').mkdir(exist_ok=True)
    fig.savefig(OUT / 'qa/compact_layout.png', dpi=180, facecolor='white')
    assert not audit['text_outside_canvas'] and not audit['text_overlap_candidates'], audit
    for extension in ['pdf', 'png', 'svg']:
        fig.savefig(OUT / f'03_count_geometry_3d.{extension}', dpi=450, facecolor='white')
    original.plt.close(fig)
    backup = OUT / '03_count_geometry_3d.before.pdf'
    if PAPER.exists() and not backup.exists():
        shutil.copy2(PAPER, backup)
    shutil.copy2(OUT / '03_count_geometry_3d.pdf', PAPER)
    metadata = {
        'changed_titles': replacements, 'camera_unchanged': True,
        'panel_layers': PANEL_LAYERS, 'panels': PANEL_AUDITS,
        'same_evaluation_states': True, 'selection_rule': 'discovery NCC, logistic tie-break, then earlier block',
        'equal_panel_frames': frame_sizes, 'cubic_axes': True,
        'equal_coordinate_units_within_panel': True,
        'fitted_views': view_audits,
        'pc_label_distances': label_audits,
        'colorbar_bins': 10, 'colorbar_ticks': [1, 10],
        'coordinate_line_style': line_style, 'coordinate_line_audit': line_audits,
        'source_sha256': original.SOURCES, 'layout_audit': audit,
        'paper_pdf_sha256': hashlib.sha256(PAPER.read_bytes()).hexdigest(),
    }
    (OUT / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def geometry():
    """Use archived PCA coordinates at each endpoint's NCC-selected block."""
    l2 = original.read(original.OLD / 'plot_data/11_pca_common_l2_points.csv')
    l4 = original.read(original.OLD / 'plot_data/12_pca_common_l4_points.csv')
    fig = original.plt.figure(figsize=(6.5, 5.8))
    specs = [('nonthinking', False, 'A. Non-thinking: occurrence index'),
             ('thinking', False, 'B. Thinking: occurrence index'),
             ('nonthinking', True, 'C. Non-thinking: final count'),
             ('thinking', True, 'D. Thinking: final count')]
    frames = []
    for i, (mode, answer, title) in enumerate(specs):
        layer = PANEL_LAYERS[i]
        data = l4 if layer == 4 else l2
        f = data[data['mode'].eq(mode) & data.endpoint.str.contains('answer_query').eq(answer)].copy()
        old = l2[l2['mode'].eq(mode) & l2.endpoint.str.contains('answer_query').eq(answer)]
        keys = ['prompt_sha256', 'occurrence', 'total_count', 'position']
        pd.testing.assert_frame_equal(f[keys].reset_index(drop=True), old[keys].reset_index(drop=True))
        if i not in [1, 2]:
            pd.testing.assert_frame_equal(f, old)
        assert len(f) == (100 if answer else 550) and f.layer.eq(layer).all()
        assert f.prompt_sha256.nunique() == 100
        frames.append(f)
        x, y = (.01 if i % 2 == 0 else .515), (.555 if i < 2 else .12)
        ax = fig.add_axes([x, y, .42, .38], projection='3d', computed_zorder=False)
        original.draw_pca(ax, f, original.CAMERA)
        ax.tick_params(labelsize=9.5, pad=0)
        ax.set_xlabel('PC1', labelpad=0, fontsize=10)
        ax.set_ylabel('PC2', labelpad=0, fontsize=10)
        ax.set_zlabel('')
        ax.text2D(1.15, .54, 'PC3', transform=ax.transAxes, fontsize=10)
        fig.text(x+.01, y+.395, title, fontsize=11,
                 color=original.NT if mode == 'nonthinking' else original.T)
        variance = 100*sum(f[f'pc{k}_variance_ratio'].iloc[0] for k in [1, 2, 3])
        PANEL_AUDITS.append(dict(panel=chr(65+i), mode=mode, layer=layer,
                                endpoint='final_count' if answer else 'running_index',
                                points=len(f), explained_variance_percent=variance))
    original.export(pd.concat(frames), '03_count_geometry_points.csv')
    original.export(pd.DataFrame(PANEL_AUDITS), '03_count_geometry_variance.csv')
    cax = fig.add_axes([.31, .079, .38, .017])
    fig.colorbar(ScalarMappable(norm=Normalize(1, 10), cmap=original.CMAP),
                 cax=cax, orientation='horizontal')
    fig.text(.5, .012, 'Occurrence index / final count', ha='center', fontsize=9.5)
    save_relabelled(fig, '03_count_geometry_3d', 'Count geometry')


def main():
    (OUT / 'plot_data').mkdir(exist_ok=True)
    original.OUT = OUT
    original.save = save_relabelled
    original.plt.rcParams.update({
        'font.family': 'Times New Roman', 'font.size': 10, 'mathtext.fontset': 'stix',
        'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 9, 'legend.frameon': False, 'axes.linewidth': .7,
        'lines.linewidth': 1.5, 'lines.markersize': 4, 'pdf.fonttype': 42,
        'svg.fonttype': 'none', 'text.color': original.INK,
        'axes.labelcolor': original.INK, 'axes.titlecolor': original.INK,
    })
    geometry()
    print('Installed NCC-selected PCA layers: L2, L4, L4, L2.')


if __name__ == '__main__':
    main()
