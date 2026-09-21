"""Display fixed-layer scope controls separately for forward/backward transfer."""
from pathlib import Path
import importlib.util
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / 'realistic/work/cot_update_ncc_l21_20260915/analysis/summary.json'


def render(style):
    payload = json.loads(style.read(SOURCE))
    assert payload['status'] == 'COMPLETE_PAIRED_AUDIT' and payload['condition_rows'] == 1080
    assert not payload['primary_only']
    scopes = ['endpoint', 'four_token_tail', 'item_span']
    directions = ['forward', 'backward']
    records = [r for r in payload['summaries'] if r['direction'] in directions]
    assert len(records) == 12
    fig = plt.figure(figsize=(6.5, 4.25))
    panel_records = []
    for row, direction in enumerate(directions):
        for col, (model, short, color, _) in enumerate(style.MODELS):
            ax = fig.add_axes([.10 + col * .51, [.60, .13][row], .365, .275])
            group = [r for r in records if r['model'] == model and r['direction'] == direction]
            layer, = {r['layer'] for r in group}
            label = 'ABCD'[row * 2 + col]
            style.panel(ax, f'{label}. {short} L{layer}: {direction}', 'Source-successor adoption')
            for x, scope in enumerate(scopes):
                r, = [r for r in group if r['scope'] == scope]
                assert r['n'] == 30
                pooled, = [p for p in payload['summaries'] if p['model'] == model
                           and p['scope'] == scope and p['direction'] == 'both']
                paired = [p for p in records if p['model'] == model and p['scope'] == scope]
                assert sum(p['target_adoption'] for p in paired) == pooled['target_adoption']
                assert sum(p['self_adoption'] for p in paired) == pooled['self_adoption']
                for key, dx, filled in [('adoption', -.055, True), ('self', .055, False)]:
                    mean = r[key]['estimate']
                    lo, hi = r[key]['ci95']
                    assert len(r[key]['seed_values']) == 10
                    ax.errorbar(x + dx, mean, yerr=[[mean - lo], [hi - mean]], fmt='o',
                                color=color, mfc=color if filled else 'white', mec=color,
                                ms=4.2, capsize=2.5, lw=1, zorder=4)
                ax.text(x - .055, r['adoption']['ci95'][1] + .045,
                        f"{r['target_adoption']}/30", fontsize=9,
                        ha='center', va='bottom', color=color)
                panel_records.append({'panel': label, 'model': model, 'layer': layer,
                                      'direction': direction, 'scope': scope,
                                      'target': r['target_adoption'], 'self': r['self_adoption'],
                                      'n': 30, 'ci95': r['adoption']['ci95']})
            ax.set(xticks=[0, 1, 2], xticklabels=['Endpoint', 'Four-token\ntail', 'Item span'],
                   xlim=(-.45, 2.45), ylim=(-.04, 1.19), yticks=[0, .5, 1])
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    fig.legend([Line2D([], [], marker='o', color=style.INK, ls='None'),
                Line2D([], [], marker='o', color=style.INK, mfc='white', ls='None')],
               ['Source patch', 'Self patch'], loc='upper center', bbox_to_anchor=(.535, 1),
               ncol=2, columnspacing=2)
    style.save(fig, 'cot_progress_controls', records,
               extra={'layer_choice': 'no-index discovery NCC; Gemma candidates L1--L22',
                      'directions': 'forward and backward displayed separately',
                      'directed_pairs_per_panel': 30, 'seed_clusters_per_panel': 10,
                      'width_comparison_holds_layer_fixed': True,
                      'intervals': 'Archived direction-specific seed-bootstrap intervals; no refit or resampling.',
                      'panels': panel_records})


def main():
    reference = ROOT / 'figures/thinking_appendix_style_20260913/build_figures.py'
    spec = importlib.util.spec_from_file_location('thinking_appendix_style', reference)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.OUT = OUT
    style.setup()
    render(style)
    (OUT / 'figure_manifest.json').write_text(json.dumps(
        {'inputs': style.INPUTS, 'figures': style.FIGURES}, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
