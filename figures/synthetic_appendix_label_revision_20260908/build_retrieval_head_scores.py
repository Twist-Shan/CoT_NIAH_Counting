"""Final-checkpoint layer-by-head maps from the frozen evaluation scores."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[1]
PREVIOUS = WORK / 'figures/synthetic_appendix_evidence_revision_20260908'
SOURCE = PREVIOUS / 'plot_data/09_attention_roles.csv'
PAPER = WORK / 'runs/paper_figures/figures/synthetic_appendix/02_retrieval_head_scores.pdf'
sys.path.insert(0, str(PREVIOUS))
from audit_style import inspect_figure

INK = '#161923'
CMAP = LinearSegmentedColormap.from_list('aurora_warm',
    ['#161923', '#40204F', '#963878', '#E65D91', '#F7A35C', '#F9EDAD'])


def main():
    (OUT / 'plot_data').mkdir(exist_ok=True)
    plt.rcParams.update({
        'font.family': 'Times New Roman', 'font.size': 10, 'mathtext.fontset': 'stix',
        'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'axes.linewidth': .7, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'text.color': INK, 'axes.labelcolor': INK, 'axes.titlecolor': INK,
    })
    all_scores = pd.read_csv(SOURCE)
    final = all_scores[all_scores.step.eq(10000)]
    broad_max = float(all_scores.broad.max())
    fig = plt.figure(figsize=(6.5, 2.8))
    tables, summaries = [], []
    for i, (mode, score, title, label, vmax, ticks) in enumerate([
        ('nonthinking', 'broad', 'A. Non-thinking: broad', r'Broad score $B_h$', broad_max, [0,.025,.05]),
        ('thinking', 'targeted', 'B. Thinking: targeted', r'Targeted score $T_h$', 1., [0,.5,1]),
    ]):
        rows = final[final['mode'].eq(mode)][['mode','step','layer','head',score]].copy()
        assert len(rows) == 32 and not rows.duplicated(['layer','head']).any()
        matrix = rows.pivot(index='layer', columns='head', values=score).reindex(index=range(1,5),columns=range(8))
        assert matrix.shape == (4,8) and np.isfinite(matrix.to_numpy()).all()
        assert np.all((matrix.to_numpy() >= 0) & (matrix.to_numpy() <= vmax))
        x = .08 + .50*i
        ax = fig.add_axes([x, .37, .38, .38*6.5/2/2.8])
        im = ax.imshow(matrix, cmap=CMAP, norm=Normalize(0,vmax), origin='upper',
                       interpolation='nearest', aspect='equal')
        ax.set(xticks=range(8), xticklabels=[f'H{h+1}' for h in range(8)],
               yticks=range(4), yticklabels=[f'L{l}' for l in range(1,5)],
               xlabel='Head', ylabel='Layer')
        ax.set_title(title, loc='left', fontsize=11, pad=8)
        ax.tick_params(length=2.5, pad=3)
        # Every visible cell is a physical head; no sorting or score normalization.
        ax.set_xticks(np.arange(.5,7.5), minor=True)
        ax.set_yticks(np.arange(.5,3.5), minor=True)
        ax.grid(which='minor', color='white', linewidth=.4, alpha=.35)
        ax.tick_params(which='minor', length=0)
        cax = fig.add_axes([x+.045, .15, .29, .027])
        cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=ticks)
        cb.ax.tick_params(labelsize=9, length=2, pad=2)
        cb.set_label(label, fontsize=10, labelpad=3)
        tables.append(rows.rename(columns={score:'score'}).assign(metric=score))
        summaries.append(dict(panel=chr(65+i), mode=mode, metric=score,
                              layer_means=rows.groupby('layer')[score].mean().to_dict(),
                              top_heads=rows.nlargest(4,score)[['layer','head',score]].to_dict('records'),
                              color_limits=[0,vmax]))
    fig.canvas.draw()
    audit = inspect_figure(fig, '02_retrieval_head_scores')
    assert not audit['text_outside_canvas'] and not audit['text_overlap_candidates'], audit
    for extension in ['pdf','png','svg']:
        fig.savefig(OUT / f'02_retrieval_head_scores.{extension}', dpi=450, facecolor='white')
    plt.close(fig)
    pd.concat(tables, ignore_index=True).to_csv(OUT / 'plot_data/02_retrieval_head_scores.csv', index=False)
    shutil.copy2(OUT / '02_retrieval_head_scores.pdf', PAPER)
    (OUT / 'retrieval_head_scores_manifest.json').write_text(json.dumps(dict(
        source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        checkpoint=10000, evaluation_inputs=100, aggregation='equal input weights',
        raw_scores=True, color_scales_match_dynamics=True, summaries=summaries,
        layout_audit=audit, paper_sha256=hashlib.sha256(PAPER.read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    print(json.dumps(dict(figure=str(PAPER), panels=summaries, layout_audit=audit),indent=2))


if __name__ == '__main__':
    main()
