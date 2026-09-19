"""Separate mixed appendix figures into adjacent experiment-specific rows.

Reuses the frozen-data builders and typography from the existing appendix.
No refitting or numerical changes. Run from the workspace root with
python -s figures/nonthinking_appendix_layout_20260913/build_figures.py
"""
from pathlib import Path
import hashlib
import importlib.util
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / 'figures/nonthinking_appendix_style_20260911/build_figures.py'
spec = importlib.util.spec_from_file_location('frozen_appendix_figures', SOURCE)
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
source.OUT = OUT
original_save = source.save
AUDIT = {}


def data_hash(axes):
    digest = hashlib.sha256()
    for ax in axes:
        for line in ax.lines:
            for array in [line.get_xdata(orig=False), line.get_ydata(orig=False)]:
                values = np.asarray(array, dtype=float)
                digest.update(str(values.shape).encode())
                digest.update(values.tobytes())
        for collection in ax.collections:
            for path in collection.get_paths():
                values = np.asarray(path.vertices, dtype=float)
                digest.update(str(values.shape).encode())
                digest.update(values.tobytes())
    return digest.hexdigest()


def export_row(builder, row, name, legend_kind):
    def save_row(fig, old_name, ranks=None):
        assert ranks is None and len(fig.axes) == 4
        all_axes = list(fig.axes)
        axes = all_axes[row*2:row*2+2]
        before = data_hash(axes)
        for ax in all_axes:
            if ax not in axes:
                ax.remove()
        for legend in list(fig.legends):
            legend.remove()
        fig.set_size_inches(6.5, 2.05)
        for col, ax in enumerate(axes):
            ax.set_position([[.10, .61][col], .23, .365, .57])
            title = ax.get_title(loc='left')
            assert title[1:3] == '. '
            ax.set_title('AB'[col] + title[1:], loc='left', fontsize=11,
                         fontweight='normal', pad=7)
        if legend_kind == 'models':
            fig.legend(source.model_handles(), source.NAMES, loc='upper center',
                       bbox_to_anchor=(.54, 1.012), ncol=2, frameon=False,
                       fontsize=9, borderaxespad=.1, handlelength=1.5)
        elif legend_kind == 'ov':
            handles, labels = axes[1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center',
                       bbox_to_anchor=(.793, 1.012), ncol=2, frameon=False,
                       fontsize=9, borderaxespad=.1, columnspacing=.7,
                       handlelength=1.5, handletextpad=.4)
        original_save(fig, name)
        assert data_hash(axes) == before
        qa = source.AUDITS[name]
        assert not qa['outside'] and not qa['text_overlaps'], qa
        AUDIT[name] = {'source_figure': old_name,
                       'source_panels': ['AB', 'CD'][row],
                       'display_panels': 'AB', 'plotted_data_unchanged': True,
                       'plotted_data_sha256': before, 'layout': qa}
    source.save = save_row
    builder()


def main():
    plt.rcParams.update({
        'font.family': 'Times New Roman', 'mathtext.fontset': 'stix', 'font.size': 10,
        'axes.titlesize': 11, 'axes.titleweight': 'normal', 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
        'legend.frameon': False, 'axes.linewidth': .7, 'lines.linewidth': 1.5,
        'lines.markersize': 4, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'text.color': source.INK,
        'axes.labelcolor': source.INK, 'axes.titlecolor': source.INK,
        'xtick.color': source.INK, 'ytick.color': source.INK,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    })
    for builder, row, name, legend in [
        (source.retrieval, 0, 'nonthinking_head_ablation', None),
        (source.patching, 1, 'nonthinking_answer_patching', None),
        (source.answer_mediation, 0, 'nonthinking_answer_function', 'models'),
        (source.answer_mediation, 1, 'nonthinking_serial_mediation', 'models'),
        (source.retrieval, 1, 'nonthinking_qwen_routing', 'ov'),
    ]:
        export_row(builder, row, name, legend)
    manifest = {'status': 'PASS', 'figures': AUDIT,
                'source_builder_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                'input_sha256': source.INPUTS, 'model_inference': False, 'refitting': False,
                'indexing': 'Source data zero-based; displayed layers one-based.',
                'palette': dict(zip(source.MODELS, source.COLORS)),
                'font_points': {'title': 11, 'axis': 10, 'ticks': 9, 'legend': 9}}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'figures': list(AUDIT)}))


if __name__ == '__main__':
    main()
