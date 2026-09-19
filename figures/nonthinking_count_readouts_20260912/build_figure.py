"""Place running-index and final-count classification scans side by side.

Panel A is refit on 20 discovery seeds by refit_running_index_20_seeds.py;
panel B retains the historical 20-seed results. No fitting is performed here.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DATA = ROOT / 'realistic/reports/v4_non-thinking_causal/v4_4_extension/classification'
MODELS = [('Qwen3-8B', 'qwen', '#168DCA', 36), ('Gemma4-E4B', 'gemma', '#E87824', 42)]
METHODS = [('nearest_centroid', '-'), ('logistic_l2', '--')]
PANELS = [
    ('prompt', 'running_index', 'A. Running index', 200, 20),
    ('all', 'final_count', 'B. Final count', 200, 20),
]
INK, GRAY, GRID = '#161923', '#8190A5', '#E7E8EE'


def main():
    plt.rcParams.update({
        'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
        'font.size': 10, 'axes.labelsize': 10, 'xtick.labelsize': 9,
        'ytick.labelsize': 9, 'legend.fontsize': 9, 'axes.linewidth': .7,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
        'ytick.color': INK, 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    })
    fig = plt.figure(figsize=(6.5, 2.55))
    axes = [fig.add_axes([x, .215, .400, .535]) for x in [.095, .580]]
    inputs, plotted, peaks = {}, [], []
    for ax, (source, target, title, expected_rows, expected_seeds) in zip(axes, PANELS):
        for model, short, color, layers in MODELS:
            path = (OUT / f'classification_prompt_20_{short}/answer_classifier_metrics.csv'
                    if source == 'prompt' else DATA / f'classification_all_{short}/answer_classifier_metrics.csv')
            inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open(encoding='utf-8-sig') as handle:
                rows = list(csv.DictReader(handle))
            for method, style in METHODS:
                selected = sorted((r for r in rows if r['algorithm'] == method), key=lambda r: int(r['layer']))
                assert [int(r['layer']) for r in selected] == list(range(layers))
                assert all(int(r['rows']) == expected_rows and int(r['seeds']) == expected_seeds for r in selected)
                assert all(int(r['count_class_count']) == 10 and int(r['pca_components']) == 32 for r in selected)
                y = [float(r['accuracy']) for r in selected]
                assert all(0 <= value <= 1 for value in y)
                ax.plot([int(r['layer']) + 1 for r in selected], y, color=color,
                        linestyle=style, linewidth=1.5)
                plotted.extend({'target': target, 'model': model,
                                'layer_source_zero_based': int(r['layer']),
                                'layer_display_one_based': int(r['layer']) + 1,
                                'method': method, 'accuracy': float(r['accuracy']),
                                'states': expected_rows, 'seeds': expected_seeds}
                               for r in selected)
                peak = max(selected, key=lambda r: float(r['accuracy']))
                peaks.append({'target': target, 'model': model, 'method': method,
                              'layer_one_based': int(peak['layer']) + 1,
                              'accuracy': float(peak['accuracy'])})
        ax.axhline(.1, color=GRAY, linestyle=':', linewidth=.9, zorder=0)
        ax.set(xlim=(.5, 42.5), xticks=[1, 10, 20, 30, 42],
               ylim=(0, .88), yticks=[0, .2, .4, .6, .8], xlabel='Layer')
        ax.set_title(title, loc='left', fontsize=11, fontweight='normal', pad=7)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color=GRID, linewidth=.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=.7, pad=3)
    axes[0].set_ylabel('Classification accuracy')
    fig.legend(
        [Line2D([], [], color=m[2], lw=1.5) for m in MODELS]
        + [Line2D([], [], color=INK, lw=1.5, linestyle=s) for _, s in METHODS],
        ['Qwen', 'Gemma', 'Nearest centroid', 'Logistic'],
        loc='upper center', bbox_to_anchor=(.535, .995), ncol=4,
        frameon=False, handlelength=1.7, handletextpad=.45, columnspacing=1.4,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text()]
    outside = [t.get_text() for t, b in texts if b.x0 < -.5 or b.y0 < -.5
               or b.x1 > fig.bbox.width + .5 or b.y1 > fig.bbox.height + .5]
    overlaps = []
    for i, (a, ab) in enumerate(texts):
        for b, bb in texts[i + 1:]:
            if min(ab.x1, bb.x1) - max(ab.x0, bb.x0) > .8 and min(ab.y1, bb.y1) - max(ab.y0, bb.y0) > .8:
                overlaps.append([a.get_text(), b.get_text()])
    assert not outside and not overlaps, (outside, overlaps)
    assert len(plotted) == (36 + 42) * 2 * 2
    for model, layer, accuracy in [('Qwen3-8B', 30, .545), ('Gemma4-E4B', 38, .55)]:
        value = next(r['accuracy'] for r in plotted if r['target'] == 'final_count'
                     and r['model'] == model and r['layer_display_one_based'] == layer
                     and r['method'] == 'nearest_centroid')
        assert abs(value - accuracy) < 1e-10
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(OUT / f'nonthinking_count_readouts.{ext}', dpi=300)
    plt.close(fig)
    with (OUT / 'plot_data.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plotted[0]))
        writer.writeheader()
        writer.writerows(plotted)
    manifest = {
        'source_sha256': inputs, 'observations': len(plotted),
        'display_layer_indexing': 'one-based',
        'measurements_changed': {'running_index': 'refit on 20 discovery seeds', 'final_count': False},
        'refit_audit': 'running_index_20_audit.json',
        'canvas_inches': [6.5, 2.55], 'axis_limits': {'layer': [.5, 42.5], 'accuracy': [0, .88]},
        'font': 'Times New Roman', 'font_points': {'title': 11, 'axis': 10, 'tick': 9, 'legend': 9},
        'text_overlaps': overlaps, 'text_outside': outside, 'descriptive_peaks': peaks,
        'protocols': {
            'running_index': {'states': 200, 'prompts': 20, 'seed_range': [1234, 1253],
                              'prompt_final_count': 10, 'labels': 'running index 1--10',
                              'preprocessing': 'unwhitened PCA32, then coordinate standardization',
                              'nearest_centroid_shrinkage': None},
            'final_count': {'states': 200, 'prompts': 200, 'seed_range': [1234, 1253],
                            'labels': 'final count 1--10',
                            'preprocessing': 'hidden-coordinate standardization, then unwhitened PCA32',
                            'nearest_centroid_shrinkage': .1},
            'shared': {'folds': 5, 'split_unit': 'seed', 'fit_on_training_fold_only': True,
                       'seed_to_fold_mapping': 'identical, frozen from historical answer-query predictions',
                       'logistic_C': 1, 'chance': .1},
        },
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'observations': len(plotted), 'text_overlaps': overlaps, 'text_outside': outside,
                      'output': str(OUT / 'nonthinking_count_readouts.pdf')}))


if __name__ == '__main__':
    main()
