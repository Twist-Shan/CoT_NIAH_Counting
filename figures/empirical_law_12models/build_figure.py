"""Render the appendix's 12-group grid with the current Section 3 hypotheses.

Run with Python, numpy, pandas, scipy, and matplotlib. No inference or refitting.
"""
from pathlib import Path
from itertools import combinations
import hashlib
import json
import runpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
TABLES = ROOT / 'realistic/outputs/anvil_realistic_niah_v3_1_20260819_formal/analysis/v3_2_inverse_n_candidate_extension/tables'
SUMMARY = ROOT / 'figures/empirical_law_1x4/observed_count_panel_quantiles.csv'
ANALYSIS = ROOT / 'figures/empirical_section3_refit'
ORDER = [
    ('Qwen3-4B', 'Qwen3-4B'),
    ('Qwen3-8B', 'Qwen3-8B'),
    ('Qwen3-14B', 'Qwen3-14B'),
    ('Qwen3-32B', 'Qwen3-32B'),
    ('Gemma4-E4B', 'Gemma-4-E4B'),
    ('Gemma4-12B', 'Gemma-4-12B'),
    ('Gemma4-26B-A4B', 'Gemma-4-26B-A4B'),
    ('Gemma4-31B', 'Gemma-4-31B'),
    ('Nemotron-Nano-v2-9B', 'Nemotron Nano 9B-v2'),
    ('Nemotron-3-Nano-4B', 'Nemotron-3-Nano-4B'),
    ('GLM-4/Z1-9B', 'GLM-4/Z1-9B$^{\u2020}$'),
    ('Ministral-3-8B pair', 'Ministral-3-8B pair$^{\u2020}$'),
]
MODES = [('direct', 'Non-thinking', '-', 'o'),
         ('native_thinking', 'Thinking', '--', '^')]


def main():
    inputs = [TABLES / 'cell_outcomes.csv.gz', ANALYSIS / 'full_fit_parameters.json',
              ANALYSIS / 'analysis_manifest.json', SUMMARY]
    cells = pd.read_csv(inputs[0])
    cells = cells[cells.prompt_mode.isin([m[0] for m in MODES])].copy()
    raw_fits = json.loads(inputs[1].read_text(encoding='utf-8'))
    selected = {'direct': 'gaussian_linear', 'native_thinking': 'interaction_free_length'}
    fits = {(r['model'], r['mode']): r for r in raw_fits
            if r['scope'] == 'original_short' and r['form'] == selected.get(r['mode'])}
    assert len(fits) == 24
    predict = runpy.run_path(str(ANALYSIS / 'analyze.py'))['predict']
    counts, lengths = sorted(cells.N.unique()), sorted(cells.L.unique())
    assert len(cells) == 2688 and cells.n_total.eq(30).all()
    assert set(cells.comparison_slot) == {s for s, _ in ORDER}
    assert len(counts) == 14 and len(lengths) == 8
    assert not cells.duplicated(['comparison_slot', 'prompt_mode', 'N', 'L']).any()
    assert np.allclose(cells.parsed_exact_accuracy, cells.n_correct / cells.n_total)
    records = []
    for slot, _ in ORDER:
        for mode, _, _, _ in MODES:
            for length in lengths:
                observed = cells[cells.comparison_slot.eq(slot) & cells.prompt_mode.eq(mode) & cells.L.eq(length)].set_index('N').loc[counts]
                prediction = predict(pd.DataFrame({'N': counts, 'L': length}), fits[(slot, mode)])
                for n, p in zip(counts, prediction):
                    records.append(dict(comparison_slot=slot, mode=mode, N=int(n), L=int(length),
                                        n_total=int(observed.loc[n, 'n_total']),
                                        n_correct=int(observed.loc[n, 'n_correct']),
                                        observed=float(observed.loc[n, 'parsed_exact_accuracy']), fitted=float(p)))
    plotted = pd.DataFrame(records)
    # The observed summaries, not the fitted overlays, reproduce main panels A/B.
    summary = pd.read_csv(SUMMARY).set_index(['mode', 'N', 'L'])
    for key, group in plotted.groupby(['mode', 'N', 'L']):
        actual = np.quantile(group.observed, [.25, .5, .75])
        expected = summary.loc[key, ['observed_q25', 'observed_median', 'observed_q75']].to_numpy(float)
        assert np.allclose(actual, expected, atol=1e-12, rtol=0), key

    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
        'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 11,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
        'axes.linewidth': .65, 'text.color': '#202020', 'axes.labelcolor': '#202020',
        'xtick.color': '#202020', 'ytick.color': '#202020',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    })
    width, height = 6.5, 6.5
    fig = plt.figure(figsize=(width, height), facecolor='white')
    colors = plt.colormaps['plasma_r'](np.linspace(.12, .94, len(lengths)))
    for i, (slot, title) in enumerate(ORDER):
        row, col = divmod(i, 3)
        ax = fig.add_axes([(.55 + 2.00 * col) / width, (.87 + 1.40 * (3-row)) / height,
                           1.78 / width, 1.08 / height])
        for mode, _, ls, marker in MODES:
            for length, color in zip(lengths, colors):
                g = plotted[plotted.comparison_slot.eq(slot) & plotted['mode'].eq(mode) & plotted.L.eq(length)].sort_values('N')
                dense = pd.DataFrame({'N': np.geomspace(1, 20, 400), 'L': length})
                ax.plot(dense.N, predict(dense, fits[(slot, mode)]), color=color, lw=1.0, ls=ls, alpha=.95)
                ax.scatter(g.N, g.observed, s=5.3 if mode == 'direct' else 6.5,
                           marker=marker, facecolor='white', edgecolor=color, linewidth=.42,
                           alpha=.78, zorder=3)
        ax.set_title(f'{chr(65+i)}  {title}', loc='left', fontsize=11, fontweight='normal', pad=6)
        ax.set_xscale('log', base=2)
        ax.set_xlim(.93, 21.5)
        ax.set_xticks([1, 2, 3, 5, 10, 20], labels=['1', '2', '3', '5', '10', '20'])
        ax.set_ylim(-.035, 1.04)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.tick_params(length=3, width=.65, pad=2, labelleft=(col == 0), labelbottom=(row == 3))
        ax.minorticks_off()
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#8D99A5')
        ax.grid(axis='y', color='#E7E8EE', linewidth=.55)
        ax.set_axisbelow(True)
        if row == 3:
            ax.set_xlabel('Target count $N$', labelpad=4)
    fig.text(.015, .56, 'Exact accuracy', rotation=90, va='center', ha='center', fontsize=10)
    length_handles = [Line2D([], [], color=c, lw=1.3, label=f'{l//1000}k') for l, c in zip(lengths, colors)]
    fig.legend(handles=length_handles,
               loc='lower center', bbox_to_anchor=(.58, .033), ncol=8, frameon=False,
               handlelength=1.2, handletextpad=.4, columnspacing=1.0,
               borderaxespad=0, labelspacing=.3)
    fig.text(.16, .050, '$L$ (tokens):', fontsize=9, va='center')
    mode_handles = [Line2D([], [], color='#444444', lw=1.1, ls=ls, marker=marker,
                          markerfacecolor='white', markersize=3.3, markeredgewidth=.6, label=name)
                    for _, name, ls, marker in MODES]
    fig.legend(handles=mode_handles, loc='lower center', bbox_to_anchor=(.50, -.002),
               ncol=2, frameon=False, handlelength=2.1, handletextpad=.5, columnspacing=2,
               borderaxespad=0)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [dict(text=t.get_text(), font_pt=t.get_fontsize(), bbox=t.get_window_extent(renderer).extents.tolist())
             for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    outside = [t for t in texts if t['bbox'][0] < -.5 or t['bbox'][1] < -.5 or
               t['bbox'][2] > fig.bbox.width+.5 or t['bbox'][3] > fig.bbox.height+.5]
    overlaps = []
    for a, b in combinations(texts, 2):
        aa, bb = a['bbox'], b['bbox']
        if min(aa[2], bb[2])-max(aa[0], bb[0]) > 1 and min(aa[3], bb[3])-max(aa[1], bb[1]) > 1:
            overlaps.append([a['text'], b['text']])
    (OUT/'layout_audit.json').write_text(json.dumps(dict(outside=outside, overlaps=overlaps, texts=texts), indent=2), encoding='utf-8')
    # Save a preview even if the layout needs another iteration.
    fig.savefig(OUT/'empirical_accuracy_12models.png', dpi=240, facecolor='white')
    assert not outside and not overlaps, dict(outside=outside, overlaps=overlaps)
    for suffix in ['pdf', 'svg']:
        fig.savefig(OUT/f'empirical_accuracy_12models.{suffix}', facecolor='white')
    plt.close(fig)
    plotted.to_csv(OUT/'plotted_cells.csv', index=False)
    target = ROOT/'runs/paper_figures/figures/empirical_appendix/empirical_accuracy_12models.pdf'
    target.write_bytes((OUT/'empirical_accuracy_12models.pdf').read_bytes())
    manifest = dict(rows=4, columns=3, comparison_groups=12, modes=2, lengths=lengths,
                    counts=counts, conditions=len(plotted), requests=int(plotted.n_total.sum()),
                    curves_per_panel=16, observed_points_per_panel=224, inference_runs=0, refits=0,
                    xscale='log2', yscale='linear', size_inches=[width,height],
                    fonts_pt=dict(title=11, axis_label=10, ticks_legend=9),
                    font_family='Times New Roman', math_font='STIX',
                    fit_forms=selected, fresh_analysis_manifest=str(inputs[2].relative_to(ROOT)),
                    main_figure_quantiles_verified=True, statistical_uncertainty_bands=False,
                    input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (OUT/'build_manifest.json').write_text(json.dumps(manifest, indent=2, default=int), encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='input_sha256'}, default=int, indent=2))


if __name__ == '__main__':
    main()
