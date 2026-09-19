"""Combine attention roles and count decoding on linear training-step axes."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[1]
PREVIOUS = WORK / 'figures/synthetic_appendix_evidence_revision_20260908'
PAPER = WORK / 'runs/paper_figures/figures/synthetic_appendix/08_training_dynamics.pdf'
sys.path.insert(0, str(PREVIOUS))
import build_figures as original
from audit_style import inspect_figure


def save_combined(fig, _stem, _title):
    # Preserve the physical sizes of all four heatmaps and their text.
    original_height, joint_height = fig.get_figheight(), 5.35
    ratio = original_height / joint_height
    offset = 1 - ratio
    fig.set_figheight(joint_height)
    for ax in fig.axes:
        x, y, w, h = ax.get_position().bounds
        ax.set_position([x, offset + ratio*y, w, ratio*h])
    for item in fig.texts:
        x, y = item.get_position()
        item.set_position((x, offset + ratio*y))
    fig.axes[3].set_title('D. Thinking\nSuccessor-like', loc='left', fontsize=11, pad=7, linespacing=1.2)
    fig.axes[5].set_xlabel('Targeted / successor-like score', fontsize=9, labelpad=2)

    data = original.read(PREVIOUS / 'plot_data/10_ncc_dynamics.csv')
    original.export(data, 'joint_dynamics_ncc.csv')
    layers = {}
    for i, title in enumerate(['E. Running index', 'F. Final count']):
        ax = fig.add_axes([.095 + .50*i, .16, .375, .245])
        for mode, color, label in [('nonthinking', original.NT, 'Non-thinking'),
                                    ('thinking', original.T, 'Thinking')]:
            subset = data[data['mode'].eq(mode) & data.endpoint.str.contains('answer_query').eq(i==1)].sort_values('step')
            assert len(subset) == 11 and subset.layer.nunique() == 1
            layer = int(subset.layer.iloc[0])
            layers[f'{i}_{mode}'] = layer
            ax.plot(subset.step, 100*subset.ncc_correct, 'o-', color=color, ms=3, clip_on=False,
                    label=f'{label} (L{layer})')
        ax.axhline(10, color=original.GRAY, ls=':', lw=.8)
        ax.axvline(1500, color=original.GRAY, ls=':', lw=.9)
        ax.set(xscale='linear', xlim=(0,10000), ylim=(-3,105),
               xticks=[0,5000,10000], xticklabels=['0','5k','10k'],
               yticks=[0,25,50,75,100], xlabel='Training step')
        original.panel(ax, title, 'Balanced accuracy (%)')
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(.2825+.5*i,.035),
                   ncol=2, fontsize=9, handlelength=1.2, columnspacing=.9,
                   handletextpad=.35)
    fig.canvas.draw()
    audit = inspect_figure(fig, '08_training_dynamics')
    assert not audit['text_outside_canvas'] and not audit['text_overlap_candidates'], audit
    assert all(ax.get_xscale() == 'linear' and ax.get_xlim() == (0,10000)
               for ax in [*fig.axes[:4], *fig.axes[6:]])
    for extension in ['pdf', 'png', 'svg']:
        fig.savefig(OUT / f'08_training_dynamics.{extension}', dpi=450, facecolor='white')
    original.plt.close(fig)
    shutil.copy2(OUT / '08_training_dynamics.pdf', PAPER)
    (OUT / 'joint_dynamics_manifest.json').write_text(json.dumps({
        'change': 'Four attention heatmaps above two NCC curves, with linear training steps in every panel.',
        'data_unchanged': True, 'source_sha256': original.SOURCES,
        'xscale': 'linear', 'xlim': [0,10000], 'selected_layers': layers,
        'layout_audit': audit,
        'paper_pdf_sha256': hashlib.sha256(PAPER.read_bytes()).hexdigest(),
    }, indent=2), encoding='utf-8')


def main():
    (OUT / 'plot_data').mkdir(exist_ok=True)
    original.OUT = OUT
    original.save = save_combined
    original.plt.rcParams.update({
        'font.family': 'Times New Roman', 'font.size': 10, 'mathtext.fontset': 'stix',
        'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 9, 'legend.frameon': False, 'axes.linewidth': .7,
        'lines.linewidth': 1.5, 'lines.markersize': 4, 'pdf.fonttype': 42,
        'svg.fonttype': 'none', 'text.color': original.INK,
        'axes.labelcolor': original.INK, 'axes.titlecolor': original.INK,
    })
    original.attention_dynamics()
    print('Installed the combined training dynamics figure with linear time axes.')


if __name__ == '__main__':
    main()
