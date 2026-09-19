"""Align the dynamics figure with the descriptive 'successor-like' terminology."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[1]
PREVIOUS = WORK / 'figures/synthetic_appendix_evidence_revision_20260908'
PAPER = WORK / 'runs/paper_figures/figures/synthetic_appendix/08_attention_dynamics.pdf'
sys.path.insert(0, str(PREVIOUS))
import build_figures as original
from audit_style import inspect_figure


def save_relabelled(fig, _stem, _title):
    assert fig.axes[3].get_title(loc='left') == 'D. Thinking\nSuccessor'
    fig.axes[3].set_title('D. Thinking\nSuccessor-like', loc='left', fontsize=11, pad=7, linespacing=1.2)
    fig.axes[-1].set_xlabel('Targeted / successor-like score', fontsize=9, labelpad=2)
    fig.canvas.draw()
    audit = inspect_figure(fig, '08_attention_dynamics')
    assert not audit['text_outside_canvas'] and not audit['text_overlap_candidates'], audit
    for extension in ['pdf', 'png', 'svg']:
        fig.savefig(OUT / f'08_attention_dynamics.{extension}', dpi=450, facecolor='white')
    original.plt.close(fig)
    backup = OUT / '08_attention_dynamics.before_successor_like.pdf'
    if not backup.exists():
        shutil.copy2(PAPER, backup)
    shutil.copy2(OUT / '08_attention_dynamics.pdf', PAPER)
    (OUT / 'attention_dynamics_manifest.json').write_text(json.dumps({
        'change': 'Label the local attention metric successor-like; no changes to data, scales or palette.',
        'data_unchanged': True,
        'source_sha256': original.SOURCES,
        'layout_audit': audit,
        'paper_pdf_sha256': hashlib.sha256(PAPER.read_bytes()).hexdigest(),
    }, indent=2), encoding='utf-8')


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
    original.attention_dynamics()
    print('Updated the dynamics labels; data and figure styling are unchanged.')


if __name__ == '__main__':
    main()
