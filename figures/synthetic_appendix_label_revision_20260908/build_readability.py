"""Use 'Running index' in the cross-layer NCC figure; retain the existing data."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[1]
PREVIOUS = WORK / 'figures/synthetic_appendix_evidence_revision_20260908'
PAPER = WORK / 'runs/paper_figures/figures/synthetic_appendix/04_count_readability.pdf'
sys.path.insert(0, str(PREVIOUS))
import build_figures as original
from audit_style import inspect_figure


def save_relabelled(fig, _stem, _title):
    assert fig.axes[0].get_title(loc='left') == 'A. Occurrence index'
    fig.axes[0].set_title('A. Running index', loc='left', fontsize=11, pad=8)
    fig.canvas.draw()
    audit = inspect_figure(fig, '04_count_readability')
    assert not audit['text_outside_canvas'] and not audit['text_overlap_candidates'], audit
    for extension in ['pdf','png','svg']:
        fig.savefig(OUT / f'04_count_readability.{extension}', dpi=450, facecolor='white')
    original.plt.close(fig)
    shutil.copy2(OUT / '04_count_readability.pdf', PAPER)
    (OUT / 'readability_manifest.json').write_text(json.dumps(dict(
        change='Rename the occurrence-index panel to Running index.', data_unchanged=True,
        source_sha256=original.SOURCES, layout_audit=audit,
        paper_pdf_sha256=hashlib.sha256(PAPER.read_bytes()).hexdigest()),indent=2),encoding='utf-8')


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
    original.readability()
    print('Updated the cross-layer decoding title to Running index.')


if __name__ == '__main__':
    main()
