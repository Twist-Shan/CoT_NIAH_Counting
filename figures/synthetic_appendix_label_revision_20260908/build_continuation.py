r"""\gpt: retain the continuation exact-match panel as a single paper figure."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[1]
PREVIOUS = WORK / 'figures/synthetic_appendix_evidence_revision_20260908'
PAPER = WORK / 'runs/paper_figures/figures/synthetic_appendix/06_progress_continuation.pdf'
sys.path.insert(0, str(PREVIOUS))
import build_figures as original
from audit_style import inspect_figure


def save_single_panel(fig, _stem, _title):
    # Retain the original data and bar artists; omit the second panel in this export.
    ax = fig.axes[0]
    fig.axes[1].remove()
    for item in list(fig.texts):
        item.remove()
    for legend in list(fig.legends):
        legend.remove()
    fig.set_size_inches(4.5, 2.6)
    ax.set_position([.13, .28, .85, .56])
    ax.set_title('Continuation exact match', loc='left', fontsize=11, pad=8)
    fig.legend(*ax.get_legend_handles_labels(), loc='lower center',
               bbox_to_anchor=(.55, .005), ncol=2, fontsize=9,
               handlelength=1.2, columnspacing=1.2)
    fig.canvas.draw()
    audit = inspect_figure(fig, '06_progress_continuation')
    assert not audit['text_outside_canvas'] and not audit['text_overlap_candidates'], audit
    for extension in ['pdf', 'png', 'svg']:
        fig.savefig(OUT / f'06_progress_continuation.{extension}', dpi=450, facecolor='white')
    original.plt.close(fig)
    backup = OUT / '06_progress_continuation.before_A_only.pdf'
    if not backup.exists():
        shutil.copy2(PAPER, backup)
    shutil.copy2(OUT / '06_progress_continuation.pdf', PAPER)
    (OUT / 'continuation_manifest.json').write_text(json.dumps({
        'change': 'Retain the original exact-match panel only; use a compact single-panel layout.',
        'data_unchanged': True,
        'latex_width_fraction': .70,
        'source_sha256': original.SOURCES,
        'layout_audit': audit,
        'paper_pdf_sha256': hashlib.sha256(PAPER.read_bytes()).hexdigest(),
    }, indent=2), encoding='utf-8')


def main():
    (OUT / 'plot_data').mkdir(exist_ok=True)
    original.OUT = OUT
    original.save = save_single_panel
    original.plt.rcParams.update({
        'font.family': 'Times New Roman', 'font.size': 10, 'mathtext.fontset': 'stix',
        'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 9, 'legend.frameon': False, 'axes.linewidth': .7,
        'lines.linewidth': 1.5, 'lines.markersize': 4, 'pdf.fonttype': 42,
        'svg.fonttype': 'none', 'text.color': original.INK,
        'axes.labelcolor': original.INK, 'axes.titlecolor': original.INK,
    })
    original.continuation()
    print('Installed continuation exact-match figure with panel A only.')


if __name__ == '__main__':
    main()
