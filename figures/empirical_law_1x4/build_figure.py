"""Plot observed counting accuracy without imposing an empirical-law family.

Run from any directory with Python, numpy, pandas, scipy and matplotlib installed.
The left panels summarize model-level accuracies by their median and IQR.
The right panels compare both modes at six counts over 25k--100k.
Qwen uses the complete YaRN-off rerun; Gemma retains the archived evaluation.
Pointwise Wilson intervals are exported and shown in the appendix figure.
Only the left display lines use local Gaussian-kernel averages.
Raw medians, IQRs and seed means remain unchanged in their numerical exports.
"""
from pathlib import Path
from itertools import combinations
import argparse
import hashlib
import json
import time
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'figures'))
from figure_style import paper_font
FONT_FAMILY = paper_font()
TITLE_PT = 10.5 if FONT_FAMILY == 'Times New Roman' else 9.5
OUT = Path(__file__).resolve().parent
RESEARCH = ROOT / 'realistic'
SHORT = RESEARCH / 'outputs/anvil_realistic_niah_v3_1_20260819_formal/analysis/v3_2_inverse_n_candidate_extension/tables'
SAVED = RESEARCH / 'reports/assets/niah_empirical_paper'
REQUESTS = RESEARCH / 'outputs/anvil_realistic_niah_v3_3_long_context_20260906_holdout/analysis/v3_3_regression_scan/input'
QWEN_RERUN = RESEARCH / 'outputs/qwen3_32b_yarn_off_anvil_20260913/final_results'
LONG_CONFIG = RESEARCH / 'configs/realistic_niah_v3_3_long_context.json'
MODES = ('direct', 'native_thinking')
MODELS = ('Qwen3-32B', 'Gemma4-31B')
INPUTS = set()
MODE_COLORS = {'direct': '#B52F6B', 'native_thinking': '#007EAB'}
DISPLAY_COUNTS = (3, 5, 8, 10, 15, 20)
APPENDIX_COUNTS = (1, 5, 10, 20)
LONG_LENGTHS = (25000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000)
SMOOTHING_BANDWIDTH = {'log2_count': .30, 'length_fraction': .20,
                       'length_min_k': 1.0, 'length_max_k': 8.0}


def read(path):
    INPUTS.add(path)
    return pd.read_csv(path)


def input_label(path):
    path = path.resolve()
    for label, base in [('short', SHORT), ('saved', SAVED), ('requests', REQUESTS), ('qwen', QWEN_RERUN)]:
        if path.is_relative_to(base.resolve()):
            return label + '/' + path.relative_to(base.resolve()).as_posix()
    if path == LONG_CONFIG.resolve():
        return 'long_context_config.json'
    return path.relative_to(ROOT).as_posix()


def observed_curve(x, y, *, log_x=False):
    """Smooth summaries with positive local weights in the displayed x-coordinate.

    This local-constant kernel smoother stays within the observed value range
    without clipping. Bandwidth depends only on axis type, never on model/mode.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    assert np.all(np.diff(x) > 0) and np.isfinite(y).all()
    assert np.all((0 <= y) & (y <= 1))
    z = np.log2(x) if log_x else x
    dense = np.unique(np.r_[np.linspace(z[0], z[-1], 500), z])
    if log_x:
        bandwidth = SMOOTHING_BANDWIDTH['log2_count']
    else:
        # Dense short lengths need narrower neighborhoods than the long grid.
        # The rule is identical for all curves and depends only on length.
        bandwidth = np.clip(SMOOTHING_BANDWIDTH['length_fraction'] * dense,
                            SMOOTHING_BANDWIDTH['length_min_k'],
                            SMOOTHING_BANDWIDTH['length_max_k'])[:, None]
    log_weights = -.5 * ((dense[:, None] - z[None, :]) / bandwidth)**2
    # Rescaling each row keeps the normalized average numerically stable.
    weights = np.exp(log_weights - log_weights.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    values = weights @ y
    assert np.allclose(weights.sum(axis=1), 1, atol=1e-12)
    assert values.min() >= y.min()-1e-12 and values.max() <= y.max()+1e-12
    return 2**dense if log_x else dense, values


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#8D99A5')
    ax.grid(axis='y', color='#E7E8EE', linewidth=.55)
    ax.set_ylim(-.025, 1.025)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.tick_params(length=3, width=.6, pad=2)
    ax.minorticks_off()
    ax.set_axisbelow(True)


def inspect_layout(fig, filename='layout_audit.json'):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    entries = []
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        entries.append(dict(text=artist.get_text(), font_pt=artist.get_fontsize(),
                            bbox_px=artist.get_window_extent(renderer).extents.tolist()))
    outside = [e for e in entries if e['bbox_px'][0] < -.5 or e['bbox_px'][1] < -.5
               or e['bbox_px'][2] > fig.bbox.width + .5
               or e['bbox_px'][3] > fig.bbox.height + .5]
    overlaps = []
    for first, second in combinations(entries, 2):
        a, b = first['bbox_px'], second['bbox_px']
        if min(a[2], b[2])-max(a[0], b[0]) > 1 and min(a[3], b[3])-max(a[1], b[1]) > 1:
            overlaps.append([first['text'], second['text']])
    audit = dict(texts=entries, outside_canvas=outside, overlaps=overlaps)
    (OUT / filename).write_text(json.dumps(audit, indent=2), encoding='utf-8')
    assert not outside, outside
    assert not overlaps, overlaps
    assert min(e['font_pt'] for e in entries) >= 8
    return audit


def wilson_interval(accuracy, n):
    """Pointwise 95% score intervals for a mean of n binary seed outcomes."""
    p, n = np.asarray(accuracy, float), np.asarray(n, float)
    z = 1.959963984540054
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n))/denominator
    radius = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
    return center-radius, center+radius


def draw_observations(ax, rows, color, *, linestyle, linewidth=1.1, show_intervals=True,
                      marker='o'):
    """Connect observed means and pointwise limits; no fitted/smoothed values."""
    x = rows.L.to_numpy()/1000
    if show_intervals:
        ax.fill_between(x, rows.wilson_lo.to_numpy(), rows.wilson_hi.to_numpy(),
                        color=color, alpha=.075, linewidth=0, zorder=1)
    ax.plot(x, rows.observed.to_numpy(), color=color, lw=linewidth, ls=linestyle,
            marker=marker, ms=1.9, mfc='white', mec=color, mew=.4, zorder=3)


def load_qwen_rerun():
    """Verify all raw requests against the final summary and effective RoPE config."""
    config_path = QWEN_RERUN / 'config.json'
    source = QWEN_RERUN / 'requests.jsonl'
    INPUTS.update([config_path, source])
    config = json.loads(config_path.read_text(encoding='utf-8'))
    rows = []
    with source.open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            outcome = row['evaluation']
            assert bool(outcome['exact_count']) == bool(
                outcome['parsed_exact_count'] and not outcome['truncated'])
            rows.append(dict(
                model_label='Qwen3-32B', prompt_mode=row['prompt_mode'],
                N=row['num_needles'], L=row['target_passage_tokens'], seed=row['seed'],
                exact_count=int(outcome['exact_count']), truncated=int(outcome['truncated']),
                parse_success=int(outcome['parse_status'] == 'ok'),
                output_tokens=row['output_tokens'], request_id=row['request_key'],
                batch='qwen_yarn_off'))
    frame = pd.DataFrame(rows)
    assert len(frame) == config['expected_requests'] == 14280
    assert tuple(sorted(frame.L.unique())) == tuple(config['passage_lengths'])
    assert not frame.duplicated(['prompt_mode', 'N', 'L', 'seed']).any()
    grouped = frame.assign(parse_failed=1-frame.parse_success).groupby(
        ['L', 'N', 'prompt_mode'], as_index=False).agg(
            completed_requests=('exact_count', 'size'), exact_correct=('exact_count', 'sum'),
            truncations=('truncated', 'sum'), parse_failures=('parse_failed', 'sum'),
            output_tokens=('output_tokens', 'sum')).rename(columns={'prompt_mode': 'mode'})
    official = read(QWEN_RERUN / 'cell_summary.csv').sort_values(['L', 'N', 'mode']).reset_index(drop=True)
    pd.testing.assert_frame_equal(grouped[official.columns], official, check_dtype=False)
    runtime = []
    for path in sorted(QWEN_RERUN.glob('worker-*/run_manifest.json')):
        INPUTS.add(path)
        manifest = json.loads(path.read_text(encoding='utf-8'))
        assert not manifest['yarn_enabled'] and manifest['original_rope_frequencies_preserved']
        assert manifest['effective_model_config']['rope_parameters']['rope_type'] == 'default'
        effective = dict(manifest['effective_model_config'])
        effective['_name_or_path'] = config['model_id']
        runtime.append(dict(worker=manifest['worker'], yarn_enabled=manifest['yarn_enabled'],
                            effective_model_config=effective))
    assert len(runtime) == 4
    scientific_keys = ('model_label', 'model_id', 'model_revision', 'expected_requests',
                       'passage_lengths', 'needle_counts', 'seeds', 'prompt_modes',
                       'engine', 'decoding', 'additional_engine_overrides', 'scoring_policy')
    return frame, dict(config={key: config[key] for key in scientific_keys}, runtime_manifests=runtime,
                       raw_requests_match_official_summary=True)


def full_range_figure(observations, *, publish):
    """Keep both modes and every observed length visible in separate count panels."""
    width, height = 6.5, 4.35
    fig = plt.figure(figsize=(width, height), facecolor='white')
    fallback_margin = .08 if FONT_FAMILY != 'Times New Roman' else 0.
    xs = [.51+fallback_margin, 2.00, 3.63, 5.12]
    for row, model in enumerate(MODELS):
        y = 2.70 if row == 0 else .85
        model_title = 'Qwen3-32B' if model == 'Qwen3-32B' else 'Gemma-4-31B'
        fig.text(.51/width, (y+1.43)/height, model_title, fontsize=10.5,
                 fontweight='bold', ha='left', va='center')
        for col, n in enumerate(APPENDIX_COUNTS):
            ax = fig.add_axes([xs[col]/width, y/height, (1.22-(fallback_margin if col == 0 else 0))/width, 1.1/height])
            for mode in MODES:
                o = observations.loc[observations.model.eq(model)
                    & observations['mode'].eq(mode) & observations.N.eq(n)].sort_values('L')
                assert len(o) == 17
                # Qwen is one consistent rerun; Gemma retains two evaluation batches.
                segments = (o,) if model == 'Qwen3-32B' else (
                    o.loc[o.L.le(20000)], o.loc[o.L.ge(25000)])
                for segment in segments:
                    draw_observations(ax, segment, MODE_COLORS[mode],
                        linestyle='-' if mode == 'direct' else (0, (3, 1.6)), linewidth=1,
                        marker='o' if mode == 'direct' else '^')
            if model == 'Gemma4-31B':
                ax.axvspan(20, 25, color='#F1F2F4', zorder=0)
            ax.set_xlim(0, 102)
            ax.set_xticks([0, 25, 50, 75, 100])
            style(ax)
            ax.set_title(f'{chr(65+4*row+col)}  $N={n}$', fontsize=10.5,
                         fontweight='bold', loc='left', pad=4)
            if row == 1:
                ax.set_xlabel('Length $L$ (k tokens)', labelpad=3)
            if col == 0:
                ax.set_ylabel('Exact accuracy', labelpad=4)
            else:
                ax.tick_params(labelleft=False)
    handles = [Line2D([], [], color=MODE_COLORS[mode], lw=1.1,
                     ls='-' if mode == 'direct' else (0, (3, 1.6)),
                     label='Non-thinking' if mode == 'direct' else 'Thinking')
               for mode in MODES]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .035),
               ncol=2, frameon=False, fontsize=8, handlelength=2.2, columnspacing=2)
    audit = inspect_layout(fig, 'full_range_layout_audit.json')
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(OUT / f'empirical_accuracy_full_range.{ext}', dpi=360, facecolor='white')
    plt.close(fig)
    if publish:
        target = ROOT / 'runs/paper_figures/figures/empirical_appendix/empirical_accuracy_full_range.pdf'
        target.write_bytes((OUT / 'empirical_accuracy_full_range.pdf').read_bytes())
    return audit


def reference_result_check(group_accuracy, *, verify=False):
    """Report historical outcome agreement separately from protocol validity."""
    gains = group_accuracy.thinking_gain
    checks = {
        'all_thinking_gains_positive': bool(gains.gt(0).all()),
        'minimum_gain_matches_27_9_pp': round(100*float(gains.min()), 1) == 27.9,
        'maximum_gain_matches_63_1_pp': round(100*float(gains.max()), 1) == 63.1,
    }
    result = {'verification_requested': verify, 'checks': checks,
              'minimum_gain_pp': 100*float(gains.min()),
              'maximum_gain_pp': 100*float(gains.max()),
              'matches_archived_outcome_summary': all(checks.values())}
    if verify and not all(checks.values()):
        raise ValueError('Archived-result verification failed: ' + json.dumps(result))
    return result


def main(*, publish=False, verify_reference_results=False):
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    cells = read(SHORT / 'cell_outcomes.csv.gz')
    config_path = LONG_CONFIG
    config = json.loads(config_path.read_text(encoding='utf-8'))
    INPUTS.update([config_path, RESEARCH / 'src/realistic_niah_v3_3_long_context/spec.py',
                   RESEARCH / 'src/realistic_niah_v3_3_long_context/runner.py'])
    assert tuple(config['target_passage_tokens']) == LONG_LENGTHS
    extension = next(m['context_extension'] for m in config['models'] if m['label'] == 'Qwen3-32B')
    assert extension['method'] == 'YaRN' and extension['factor'] == 4
    frames = []
    for version, expected_lengths in [('v3_1', tuple(sorted(cells.L.unique()))), ('v3_3', LONG_LENGTHS)]:
        frame = read(REQUESTS / f'{version}_two_model_request_level.csv.gz')
        assert tuple(sorted(frame.L.unique())) == expected_lengths
        assert frame.seed.isin(config['seeds']).all()
        frame['batch'] = version
        frames.append(frame)
    requests = pd.concat(frames, ignore_index=True)
    assert len(requests) == 28560 and requests.request_id.is_unique
    assert requests.exact_count.isin([0, 1]).all()
    keys = ['model_label', 'prompt_mode', 'N', 'L']
    assert not requests.duplicated(keys + ['seed']).any()
    observations = requests.groupby(keys).agg(
        observed=('exact_count', 'mean'), n_requests=('exact_count', 'size'),
        n_seeds=('seed', 'nunique')).reset_index().rename(
            columns={'model_label': 'model', 'prompt_mode': 'mode'})
    assert observations.n_seeds.eq(30).all()
    archived = read(SAVED / 'figure2_observed_cells.csv')
    check = observations.merge(archived, on=['model', 'mode', 'N', 'L'],
                               suffixes=('', '_archived'), validate='one_to_one')
    assert len(check) == 952 and np.allclose(check.observed, check.observed_archived, atol=1e-12)
    counts = sorted(cells.N.unique())
    lengths = sorted(cells.L.unique())
    slots = sorted(cells.comparison_slot.unique())
    assert len(slots) == 12 and len(counts) == 14 and len(lengths) == 8
    short_modes = cells.loc[cells.prompt_mode.isin(MODES)]
    assert len(short_modes) == 2688 and short_modes.n_total.eq(30).all()
    group_accuracy = short_modes.groupby(['comparison_slot', 'prompt_mode'])['parsed_exact_accuracy'].mean().unstack()
    group_accuracy['thinking_gain'] = group_accuracy.native_thinking - group_accuracy.direct
    reference_check = reference_result_check(group_accuracy, verify=verify_reference_results)
    assert len(observations) == 952 and observations.n_requests.eq(30).all()
    assert not observations.duplicated(['model', 'mode', 'N', 'L']).any()
    assert observations.observed.between(0, 1).all()
    assert np.allclose(short_modes.parsed_exact_accuracy,
                       short_modes.n_correct/short_modes.n_total, atol=1e-12)
    assert np.allclose(observations.observed*30,
                       np.rint(observations.observed*30), atol=1e-12)
    # Independently check the overlapping 1k--20k observations in the two exports.
    overlap = observations.loc[observations.L.le(20000)].merge(
        short_modes, left_on=['model', 'mode', 'N', 'L'],
        right_on=['comparison_slot', 'prompt_mode', 'N', 'L'], validate='one_to_one')
    assert len(overlap) == 448
    assert np.allclose(overlap.observed, overlap.parsed_exact_accuracy, atol=1e-12)
    # Preserve the original 12-group benchmark in A,B. Replace Qwen only in the
    # separate length analyses; never splice the new run into the archived fits.
    qwen, qwen_evidence = load_qwen_rerun()
    gemma = requests.loc[requests.model_label.eq('Gemma4-31B')].copy()
    assert not gemma.loc[gemma.truncated.eq(1), 'exact_count'].any()
    requests = pd.concat([qwen, gemma], ignore_index=True)
    assert len(requests) == 28560 and not requests.duplicated(keys + ['seed']).any()
    observations = requests.groupby(keys).agg(
        observed=('exact_count', 'mean'), n_requests=('exact_count', 'size'),
        n_seeds=('seed', 'nunique'), truncations=('truncated', 'sum'),
        batch=('batch', 'first')).reset_index().rename(
            columns={'model_label': 'model', 'prompt_mode': 'mode'})
    assert len(observations) == 952 and observations.n_requests.eq(30).all()
    assert observations.n_seeds.eq(30).all()
    intervals = observations.copy()
    intervals['wilson_lo'], intervals['wilson_hi'] = wilson_interval(
        intervals.observed, intervals.n_requests)
    assert intervals.wilson_lo.ge(-1e-12).all() and intervals.wilson_hi.le(1+1e-12).all()
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': [FONT_FAMILY], 'mathtext.fontset': 'stix',
        'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 10.5,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.linewidth': .65, 'text.color': '#202020', 'axes.labelcolor': '#202020',
        'xtick.color': '#202020', 'ytick.color': '#202020',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    })
    width, height = 6.5, 1.95
    fig = plt.figure(figsize=(width, height), facecolor='white')
    # Give the twelve long-context curves more room, keeping one horizontal row.
    fallback_margin = .08 if FONT_FAMILY != 'Times New Roman' else 0.
    positions = [.51+fallback_margin, 1.94, 3.56, 5.04]
    panel_widths = [1.20-fallback_margin, 1.20, 1.31, 1.31]
    axes = [fig.add_axes([x/width, .76/height, w/width, 1.0/height])
            for x, w in zip(positions, panel_widths)]
    # Restore the original yellow-orange-purple passage-length gradient.
    length_colors = plt.colormaps['plasma_r'](np.linspace(.12, .94, 8))
    count_colors = {
        mode: LinearSegmentedColormap.from_list(mode, colors)(np.linspace(0, 1, len(DISPLAY_COUNTS)))
        for mode, colors in {'direct': ['#E394B9', '#B52F6B', '#72143E'],
                             'native_thinking': ['#7DC6E0', '#007EAB', '#004768']}.items()
    }
    records, count_curves = [], []
    for index, (ax, mode) in enumerate(zip(axes[:2], MODES)):
        for length, color in zip(lengths, length_colors):
            observed = cells.loc[cells.prompt_mode.eq(mode) & cells.L.eq(length)]
            observed = observed.pivot(index='comparison_slot', columns='N', values='parsed_exact_accuracy').reindex(index=slots, columns=counts).to_numpy()
            assert observed.shape == (12, 14) and np.isfinite(observed).all()
            oq = np.quantile(observed, [.25, .5, .75], axis=0)
            grid, median = observed_curve(counts, oq[1], log_x=True)
            # Keep dispersion at the measured counts: no interpolated IQR band.
            ax.errorbar(counts, oq[1], yerr=[oq[1]-oq[0], oq[2]-oq[1]], fmt='none',
                        ecolor=color, elinewidth=.3, capsize=0, alpha=.45, zorder=2)
            ax.plot(grid, median, color=color, lw=1.15, zorder=3)
            ax.plot(counts, oq[1], ls='none', marker='o', ms=1.45, mfc='white',
                    mec=color, mew=.32, alpha=.85, zorder=4)
            records.extend(dict(mode=mode, N=int(n), L=int(length), observed_q25=oq[0, k],
                                observed_median=oq[1, k], observed_q75=oq[2, k])
                           for k, n in enumerate(counts))
            count_curves.extend(dict(mode=mode, N=float(n), L=int(length),
                                     display_median=float(mid))
                                for n, mid in zip(grid, median))
        ax.set_xscale('log', base=2)
        ax.set_xlim(.94, 21)
        ax.set_xticks([1, 2, 3, 5, 10, 20], labels=['1', '2', '3', '5', '10', '20'])
        style(ax)
        title = 'Non-thinking' if mode == 'direct' else 'Thinking'
        ax.set_title(f'{chr(65+index)}  {title}', loc='left',
                 fontsize=TITLE_PT, fontweight='bold', pad=4)
    axes[0].set_ylabel('Exact accuracy', labelpad=4)
    axes[1].tick_params(labelleft=False)
    legend = [Line2D([], [], color=color, lw=1.2, label=f'{length//1000}k')
              for length, color in zip(lengths, length_colors)]
    length_legend = fig.legend(handles=legend, loc='center left',
               bbox_to_anchor=(.65/width, .20/height), ncol=8, fontsize=8,
               frameon=False, handlelength=.9, handletextpad=.3,
               columnspacing=.45, borderaxespad=0, borderpad=.15)
    length_records = []
    for index, (ax, model) in enumerate(zip(axes[2:], MODELS)):
        for mode in MODES:
            for n, color in zip(DISPLAY_COUNTS, count_colors[mode]):
                o = intervals.loc[intervals.model.eq(model) & intervals['mode'].eq(mode)
                                 & intervals.N.eq(n) & intervals.L.ge(25000)].sort_values('L')
                assert tuple(o.L) == LONG_LENGTHS
                assert o.batch.eq('qwen_yarn_off' if model == 'Qwen3-32B' else 'v3_3').all()
                draw_observations(ax, o, color,
                    linestyle='-' if mode == 'direct' else (0, (3, 1.6)),
                    marker=None, linewidth=1.05, show_intervals=False)
                length_records.extend(dict(model=model, mode=mode, N=int(n), L=int(row.L),
                    display_mean=float(row.observed), wilson_lo=float(row.wilson_lo),
                    wilson_hi=float(row.wilson_hi), batch=row.batch) for row in o.itertuples())
        ax.set_xlim(24, 102)
        ax.set_xticks([25, 50, 75, 100])
        style(ax)
        title = 'Qwen3-32B' if model == 'Qwen3-32B' else 'Gemma-4-31B'
        ax.set_title(f'{chr(67+index)}  {title}', loc='left',
                     fontsize=TITLE_PT, fontweight='bold', pad=4)
    axes[3].tick_params(labelleft=False)
    # Shared labels reduce repeated text while preserving both horizontal axes.
    for first, last, label in [(0, 1, 'Target count $N$'),
                                (2, 3, 'Length $L$ (k tokens)')]:
        center = (positions[first] + positions[last] + panel_widths[last])/2
        fig.text(center/width, .49/height, label, fontsize=8.5, ha='center', va='center')
    # A compact legend matrix gives each N one heading for the two mode colors.
    legend_x = np.linspace(4.59, 6.28, len(DISPLAY_COUNTS))
    fig.text(4.34/width, .34/height, '$N$', fontsize=8, ha='right', va='center')
    for x, n in zip(legend_x, DISPLAY_COUNTS):
        fig.text(x/width, .34/height, str(n), fontsize=8, ha='center', va='center')
    for mode, legend_y in zip(MODES, [.20, .075]):
        label = 'Non-thinking' if mode == 'direct' else 'Thinking'
        fig.text(3.56/width, legend_y/height, label,
                 fontsize=8, ha='left', va='center')
        for x, color in zip(legend_x, count_colors[mode]):
            fig.add_artist(Line2D([(x-.09)/width, (x+.09)/width],
                                 [legend_y/height]*2, transform=fig.transFigure,
                                 color=color, lw=1.05,
                                 ls='-' if mode == 'direct' else (0, (3, 1.6))))
    fig.canvas.draw()
    first_label = length_legend.get_texts()[0].get_window_extent(fig.canvas.get_renderer())
    legend_center_y = (first_label.y0 + first_label.y1) / (2 * fig.bbox.height)
    fig.text(.07/width, legend_center_y, r'$L$ (tokens):', fontsize=8,
             va='center', ha='left', color='#202020')
    audit = inspect_layout(fig)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(OUT / f'empirical_accuracy_1x4.{ext}', dpi=360, facecolor='white')
    plt.close(fig)
    if publish:
        (ROOT / 'runs/paper_figures/figures/empirical_accuracy_1x4.pdf').write_bytes((OUT / 'empirical_accuracy_1x4.pdf').read_bytes())
    pd.DataFrame(records).to_csv(OUT / 'observed_count_panel_quantiles.csv', index=False)
    pd.DataFrame(count_curves).to_csv(OUT / 'observed_count_panel_curves.csv', index=False)
    pd.DataFrame(length_records).to_csv(OUT / 'observed_selected_count_curves.csv', index=False)
    observations.to_csv(OUT / 'long_context_observations.csv', index=False)
    intervals.to_csv(OUT / 'observed_accuracy_intervals.csv', index=False)
    group_accuracy.to_csv(OUT / 'cross_model_mean_accuracy.csv')
    mean_accuracy = observations.groupby(['model', 'mode', 'L'])['observed'].mean().reset_index()
    mean_accuracy.to_csv(OUT / 'observed_length_means_all_counts.csv', index=False)
    full_range_audit = full_range_figure(intervals, publish=publish)
    manifest = dict(
        historical_result_check=reference_check,
        empirical_law_coefficients_used=False, regression_refits=0,
        panel_order=['Non-thinking', 'Thinking', *MODELS],
        left_statistic='Median of 12 model-level accuracies, each over 30 seeds',
        right_statistic='Mean of 30 binary exactness outcomes at fixed model, mode, N and L; truncated/unparseable outputs are errors',
        short_cells=2688, short_requests=80640, long_cells=952, long_requests=28560,
        archived_short_export_overlap_cells_verified=448,
        left_source='Original 12-group short-context benchmark, preserved without mixing in the Qwen rerun',
        right_sources={'Qwen3-32B': 'Complete YaRN-off rerun',
                       'Gemma4-31B': 'Archived request-level outcomes'},
        smoothing=dict(method='Local-constant Gaussian-kernel weighted mean',
                       coordinates={'left': 'log2(N)', 'right': None},
                       bandwidth={'left': SMOOTHING_BANDWIDTH['log2_count'], 'right': None},
                       common_parameters_across_models_and_modes=True,
                       raw_statistics_preserved=True, display_values_smoothed={'left': True, 'right': False},
                       global_monotonicity_imposed=False,
                       probability_clipping=False, extrapolate=False),
        intervals={'left': 'Observed IQR error bars across 12 groups at each measured count',
                   'right': 'Not drawn; pointwise Wilson intervals remain in numerical exports and the four-count appendix figure'},
        right_interval_shading=False,
        right_xscale='linear', right_display_counts=DISPLAY_COUNTS,
        right_display_lengths=LONG_LENGTHS, right_modes=MODES,
        right_curves_per_panel=2*len(DISPLAY_COUNTS), right_markers_per_panel=0,
        right_observed_values_per_panel=18*len(DISPLAY_COUNTS),
        right_lines='Straight segments connect observed means; no smoothing, fit, or monotonicity constraint',
        scope='Main C,D compare both modes at six illustrative counts over 25k--100k; the appendix retains N=1,5,10,20 over all 17 lengths',
        count_selection='User-selected N=3,5,8,10,15,20 after inspection; illustrative, not confirmatory. N=1 and its mode reversal remain in the full-range appendix.',
        configuration_evidence={'qwen_yarn_off': qwen_evidence,
                                'gemma_source': 'Archived registration, runner and request outcomes',
                                'gemma_original_runtime_manifests_locally_verified': False},
        full_range_figure={'counts': APPENDIX_COUNTS, 'models': MODELS, 'modes': MODES,
                          'lengths_per_curve': 17,
                          'joined_across_20k_25k_boundary': {'Qwen3-32B': True, 'Gemma4-31B': False},
                          'layout_overlap_count': len(full_range_audit['overlaps'])},
        size_inches=[width, height], panel_widths_inches=panel_widths,
        main_legend='Shared N headings above two rows of mode-color swatches',
        main_xlabels='One shared label per pair of panels',
        font_family=FONT_FAMILY, math_font='STIX',
        panel_title_weight='bold',
        fonts_pt_at_manuscript_width={'title': TITLE_PT, 'axis_label': 8.5, 'tick_legend_annotation': 8},
        mode_linestyles={'direct': 'solid', 'native_thinking': 'dashed'},
        length_colors=[to_hex(c) for c in length_colors],
        count_colors={mode: [to_hex(c) for c in colors] for mode, colors in count_colors.items()},
        layout_overlap_count=len(audit['overlaps']),
        input_sha256={input_label(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(INPUTS)},
        elapsed_seconds=time.perf_counter()-start)
    (OUT / 'build_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'input_sha256'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--short-tables', type=Path, required=True)
    parser.add_argument('--saved-cells-dir', type=Path, required=True)
    parser.add_argument('--request-tables', type=Path, required=True)
    parser.add_argument('--qwen-results', type=Path, required=True)
    parser.add_argument('--long-config', type=Path, default=LONG_CONFIG)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--verify-reference-results', action='store_true',
                        help='Require the original paper outcome summary; omit for fresh replication')
    args = parser.parse_args()
    SHORT, SAVED, REQUESTS, QWEN_RERUN = args.short_tables, args.saved_cells_dir, args.request_tables, args.qwen_results
    LONG_CONFIG, OUT = args.long_config, args.output_dir
    main(verify_reference_results=args.verify_reference_results)
