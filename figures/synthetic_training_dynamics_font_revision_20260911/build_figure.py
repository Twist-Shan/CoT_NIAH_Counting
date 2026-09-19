"""Match synthetic main-figure typography to the preceding retrieval figure.

Run with python.
The figure is authored at its 6.5-inch manuscript width, with 10.5-pt panel
titles, 8.5-pt axis labels, and 8-pt ticks and legends. All artwork is vector.
"""
from pathlib import Path
from itertools import combinations
import hashlib
import json
import sys

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[1]
sys.path.insert(0, str(OUT.parent / 'synthetic_training_dynamics_three-panel'))
import plot_synthetic_section as source
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


def inspect_layout(fig):
    renderer = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for ax in fig.axes:
        texts.extend([ax.xaxis.label, ax.yaxis.label])
        texts.extend(ax.get_xticklabels() + ax.get_yticklabels())
        if ax.get_legend() is not None:
            texts.extend(ax.get_legend().get_texts())
    entries = []
    outside = []
    for text in texts:
        if not text.get_visible() or not text.get_text().strip():
            continue
        bounds = text.get_window_extent(renderer)
        entry = {'text': text.get_text(), 'font_pt': text.get_fontsize(),
                 'bbox_px': bounds.extents.tolist()}
        entries.append(entry)
        if (bounds.x0 < -.5 or bounds.y0 < -.5 or
                bounds.x1 > fig.bbox.width + .5 or bounds.y1 > fig.bbox.height + .5):
            outside.append(entry)
    overlaps = []
    for first, second in combinations(entries, 2):
        x1, y1, x2, y2 = first['bbox_px']
        x3, y3, x4, y4 = second['bbox_px']
        if min(x2, x4) - max(x1, x3) > 1 and min(y2, y4) - max(y1, y3) > 1:
            overlaps.append([first['text'], second['text']])
    result = {'texts': entries, 'outside_canvas': outside, 'overlaps': overlaps}
    (OUT / 'layout_audit.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    assert not outside, outside
    assert not overlaps, overlaps
    assert min(entry['font_pt'] for entry in entries) >= 8
    return result


def main():
    attention, behavior, sources = source.load()
    previous = OUT.parent / 'synthetic_training_dynamics_three-panel'
    pd.testing.assert_frame_equal(
        behavior.sort_values(['mode', 'step']).reset_index(drop=True),
        pd.read_csv(previous / 'synthetic_section_accuracy.csv')
          .sort_values(['mode', 'step']).reset_index(drop=True))
    previous_heads = pd.read_csv(previous / 'synthetic_section_head_scores.csv')
    for mode, role in [('nonthinking', 'broad'), ('thinking', 'targeted')]:
        matrix = source.matrix(attention, mode, role)
        expected = previous_heads[previous_heads['mode'].eq(mode) & previous_heads.role.eq(role)]
        expected = expected.pivot(index=['layer', 'head'], columns='step', values='score')
        np.testing.assert_allclose(matrix.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-14)

    plt.rcParams.update({
        'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
        'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 10.5,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.linewidth': .6, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'savefig.facecolor': 'white',
    })
    width, height = 6.5, 2.26
    fig = plt.figure(figsize=(width, height))
    positions = [.43, 2.38, 4.55]
    panel_widths = [1.65, 1.4, 1.4]
    axes = [fig.add_axes([left / width, .43 / height, panel_width / width, 1.55 / height])
            for left, panel_width in zip(positions, panel_widths)]
    headings = [('A  Answer accuracy', .01),
                ('B  Non-thinking: broad retrieval', 1.96),
                ('C  Thinking: targeted retrieval', 4.13)]
    for title, left in headings:
        fig.text(left / width, 2.20 / height, title, fontsize=10.5, weight='bold',
                 va='top', ha='left', linespacing=1.05)
    for mode, label in [('nonthinking', 'Non-thinking'), ('thinking', 'Thinking')]:
        frame = behavior[behavior['mode'].eq(mode)]
        axes[0].plot(frame.step, 100 * frame.ar_accuracy, 'o-',
                     color=source.AURORA_LINES[mode], label=label, lw=1.2, ms=3.2)
    axes[0].set(ylabel='Free-running accuracy (%)', ylim=(0, 103), yticks=range(0, 101, 20))
    axes[0].yaxis.labelpad = 3
    axes[0].grid(color='#e5e7eb', lw=.5)
    axes[0].legend(loc='upper left', frameon=False, handlelength=1.2,
                   handletextpad=.35, borderaxespad=.35, labelspacing=.3)
    for i, mode, role, vmax, label in [
        (1, 'nonthinking', 'broad', .06, r'Broad score $B_h$'),
        (2, 'thinking', 'targeted', 1., r'Targeted needle mass $T_h$'),
    ]:
        matrix = source.matrix(attention, mode, role)
        mesh = axes[i].pcolormesh(
            matrix.columns, np.arange(32), matrix.to_numpy(), shading='nearest',
            cmap=source.AURORA_CMAP, vmin=0, vmax=vmax, rasterized=False,
            edgecolors='face', linewidth=.03, antialiased=False)
        axes[i].set_ylim(31.5, -.5)
        axes[i].set_yticks([3.5, 11.5, 19.5, 27.5], ['L1', 'L2', 'L3', 'L4'])
        axes[i].tick_params(axis='y', length=0, pad=3)
        for y in [7.5, 15.5, 23.5]:
            axes[i].axhline(y, color='white', alpha=.5, lw=.45)
        color_ax = fig.add_axes([(positions[i] + panel_widths[i] + .06) / width, .43 / height,
                                 .055 / width, 1.55 / height])
        bar = fig.colorbar(mesh, cax=color_ax, ticks=[0, vmax / 2, vmax])
        bar.solids.set_rasterized(False)
        bar.solids.set_edgecolor('face')
        bar.ax.tick_params(labelsize=8, length=2, pad=2)
        bar.set_label(label, fontsize=8.5, labelpad=3)
    for i, ax in enumerate(axes):
        ax.set(xlim=(0, 10000), xlabel='Training steps')
        ax.xaxis.labelpad = 3
        ax.set_xticks([0, 2500, 5000, 7500, 10000])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '0' if x == 0 else f'{x / 1000:g}k'))
        ax.tick_params(axis='x', length=3, pad=2)
        if i == 0:
            ax.tick_params(axis='y', length=3, pad=2)
        ax.axvline(1500, color='#666' if i == 0 else '#e0f2fe',
                   ls=(0, (3, 2)), lw=.7)
    fig.canvas.draw()
    audit = inspect_layout(fig)
    for extension in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'synthetic_section.{extension}', dpi=300)
    plt.close(fig)
    manifest = {
        'size_inches': [width, height], 'reference': 'nonthinking_broad_retrieval.pdf',
        'fonts_pt_at_manuscript_width': {'titles': 10.5, 'axis_labels': 8.5, 'ticks_and_legends': 8},
        'title_layout': 'Single-line titles on a shared baseline, close to the plot tops',
        'panel_widths_inches': panel_widths,
        'title_left_offsets_from_plot_inches': [-.42, -.42, -.42],
        'gap_between_A_and_B_plot_boxes_inches': positions[1] - positions[0] - panel_widths[0],
        'source_sha256': {str(path.relative_to(WORK)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
        'data_comparison': '22 accuracy and 6464 head-score values match previous exports',
        'rasterized': False, 'layout_overlap_count': len(audit['overlaps']),
        'final_accuracy': behavior[behavior.step.eq(10000)].to_dict('records'),
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
