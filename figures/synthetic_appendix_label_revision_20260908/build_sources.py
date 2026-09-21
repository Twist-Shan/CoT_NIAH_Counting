r"""\gpt: center the next-needle question beneath the source-position boxes."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[1]
PREVIOUS = WORK / 'figures/synthetic_appendix_evidence_revision_20260908'
PAPER = WORK / 'runs/paper_figures/figures/synthetic_appendix/05_progress_sources.pdf'
sys.path.insert(0, str(PREVIOUS))
import build_figures as original
from audit_style import inspect_figure


def save_centered(fig, _stem, _title):
    diagram = fig.axes[0]
    texts = {item.get_text(): item for item in diagram.texts}
    question = texts['Next needle: ?']
    question.set_position((.49, .14))
    question.set_ha('center')
    question.set_va('center')
    question.set_transform(diagram.transAxes)
    footer = texts['Token positions are unchanged']
    footer.set_position((.49, -.075))
    footer.set_transform(diagram.transAxes)
    # \gpt: these are intervention sites within one run, not patch-provider prompts.
    from matplotlib.text import Text
    for label in fig.findobj(Text):
        if label.get_text() == 'A. Source positions':
            label.set_text('A. Ablation positions')
        elif label.get_text() == 'Source ablation':
            label.set_text('Prompt/trace ablation')
    fig.canvas.draw()
    audit = inspect_figure(fig, '05_progress_sources')
    assert not audit['text_outside_canvas'] and not audit['text_overlap_candidates'], audit
    for extension in ['pdf', 'png', 'svg']:
        fig.savefig(OUT / f'05_progress_sources.{extension}', dpi=450, facecolor='white')
    original.plt.close(fig)
    backup = OUT / '05_progress_sources.before_centering.pdf'
    if not backup.exists():
        shutil.copy2(PAPER, backup)
    temporary = PAPER.with_suffix('.terms.tmp.pdf')
    shutil.copy2(OUT / '05_progress_sources.pdf', temporary)
    os.replace(temporary, PAPER)
    (OUT / 'sources_manifest.json').write_text(json.dumps({
        'change': 'Center the question and footer on the source boxes; raise the question.',
        'diagram_center_x': .49,
        'question_y': .14,
        'data_unchanged': True,
        'terminology_revision': 'Ablation positions; Prompt/trace ablation (2026-09-10)',
        'source_sha256': original.SOURCES,
        'layout_audit': audit,
        'paper_pdf_sha256': hashlib.sha256(PAPER.read_bytes()).hexdigest(),
    }, indent=2), encoding='utf-8')


def main():
    (OUT / 'plot_data').mkdir(exist_ok=True)
    original.OUT = OUT
    original.save = save_centered
    original.plt.rcParams.update({
        'font.family': 'Times New Roman', 'font.size': 10, 'mathtext.fontset': 'stix',
        'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 9, 'legend.frameon': False, 'axes.linewidth': .7,
        'lines.linewidth': 1.5, 'lines.markersize': 4, 'pdf.fonttype': 42,
        'svg.fonttype': 'none', 'text.color': original.INK,
        'axes.labelcolor': original.INK, 'axes.titlecolor': original.INK,
    })
    original.sources()
    print('Centered the next-needle question and installed the paper PDF.')


if __name__ == '__main__':
    main()
