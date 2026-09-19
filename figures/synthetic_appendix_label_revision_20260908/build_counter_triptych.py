r"""\gpt: combine continuation, answer sources, and source-prediction agreement."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import os
import shutil

import build_counter_panels as previous

OUT, WORK, PAPER = previous.OUT, previous.WORK, previous.PAPER
np, pd, plt, original = previous.np, previous.pd, previous.plt, previous.original


def draw(donor, sources, answer):
    fig = plt.figure(figsize=(6.5, 2.85))
    axs = [fig.add_axes([left, .40, .24, .46]) for left in [.082, .412, .742]]
    left, middle, right = axs
    means = 100 * donor.donor_match.to_numpy()
    lo, hi = 100 * donor.ci_low.to_numpy(), 100 * donor.ci_high.to_numpy()
    x = np.arange(1, 5)
    left.bar(x, means, width=.58, color=original.T, label='Source state')
    left.errorbar(x, means, yerr=np.vstack([means-lo, hi-means]), fmt='none',
                  ecolor=original.INK, capsize=2.5, elinewidth=1, capthick=1)
    for xx, value, upper in zip(x, means, hi):
        left.text(xx, upper+2.5, f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    original.panel(left, 'A. Source continuation', 'Exact match (%)')
    left.set(xticks=x, xlabel='Subsequent markers', xlim=(.4, 4.6))
    fig.legend(*left.get_legend_handles_labels(), loc='center', bbox_to_anchor=(.202, .12),
               fontsize=9, handlelength=1.2, handletextpad=.4, borderpad=.1)

    thinking = sources.loc[sources['mode'].eq('thinking')]
    x, width = np.arange(2), .30
    for arms, offset, color, label in [
        (['records', 'trace'], -width/2, original.T, 'Prompt/trace ablation'),
        (['ordinary_records_budget', 'ordinary_trace_budget'], width/2, original.GRAY, 'Control ablation'),
    ]:
        vals = [100*thinking.loc[thinking.arm.eq(arm), 'accuracy'].iloc[0] for arm in arms]
        bars = middle.bar(x+offset, vals, width=width, color=color, label=label)
        middle.bar_label(bars, fmt='%.0f', padding=2, fontsize=9)
    clean = 100*thinking.loc[thinking.arm.eq('clean'), 'accuracy'].iloc[0]
    middle.axhline(clean, color=original.INK, ls=':', lw=1, label=f'Clean ({clean:.0f}%)')
    original.panel(middle, 'B. Final-answer sources', 'Answer accuracy (%)')
    middle.set(xticks=x, xticklabels=['Prompt\ntargets', 'Generated\ntrace'])
    fig.legend(*middle.get_legend_handles_labels(), loc='center', bbox_to_anchor=(.532, .12),
               ncol=1, fontsize=9, handlelength=1.2, labelspacing=.25,
               handletextpad=.4, borderpad=.1)

    for mode, color, marker in [('nonthinking', original.NT, 's'), ('thinking', original.T, 'o')]:
        data = answer.loc[answer['mode'].eq(mode)].sort_values('layer')
        assert data.pairs.eq(180).all()
        right.plot(data.layer, 100*data.patched_match_prediction, marker+'-', color=color,
                   mfc='white' if mode == 'thinking' else color, ms=4.5)
        right.plot(data.layer, 100*data.baseline_match_prediction, ':', color=color, lw=1.4)
    original.panel(right, 'C. Source prediction', 'Match (%)')
    right.set(xticks=[1, 2, 3, 4], xlabel='Patching layer', xlim=(.85, 4.15))
    Line2D = original.Line2D
    fig.legend([Line2D([], [], color=original.NT, marker='s'),
                Line2D([], [], color=original.T, marker='o', mfc='white'),
                Line2D([], [], color=original.INK, ls='-'),
                Line2D([], [], color=original.INK, ls=':')],
               ['Non-thinking', 'Thinking', 'After patching', 'Before patching'],
               loc='center', bbox_to_anchor=(.862, .12), ncol=1, fontsize=9,
               handlelength=1.5, labelspacing=.25, handletextpad=.4, borderpad=.1)
    for ax in axs:
        ax.set(ylim=(-4, 110), yticks=[0, 25, 50, 75, 100])
    previous.save(fig, '06_counter_interventions')


def main():
    (OUT / 'plot_data').mkdir(exist_ok=True)
    plt.rcParams.update({
        'font.family': 'Times New Roman', 'font.size': 10, 'mathtext.fontset': 'stix',
        'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 9, 'legend.frameon': False, 'axes.linewidth': .7,
        'lines.linewidth': 1.5, 'lines.markersize': 4, 'pdf.fonttype': 42,
        'svg.fonttype': 'none', 'text.color': original.INK,
        'axes.labelcolor': original.INK, 'axes.titlecolor': original.INK,
    })
    donor = previous.donor_estimates()
    sources = previous.read(previous.PREVIOUS / 'plot_data/08_answer_sources_all_modes.csv')
    answer = previous.read(previous.PREVIOUS / 'plot_data/08_answer_state_donor.csv')
    # The plotted readout metric includes both patched and clean target baselines.
    plotted = answer[['mode', 'layer', 'pairs', 'patched_match_prediction', 'baseline_match_prediction']]
    plotted.to_csv(OUT / 'plot_data/06_answer_prediction_transplants.csv', index=False)
    draw(donor, sources, answer)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    destination = PAPER / '06_counter_interventions.pdf'
    if destination.exists():
        shutil.copy2(destination, OUT / f'06_counter_interventions.before_{stamp}.pdf')
    temporary = destination.with_suffix('.terms.tmp.pdf')
    shutil.copy2(OUT / '06_counter_interventions.pdf', temporary)
    os.replace(temporary, destination)
    (OUT / 'counter_triptych_manifest.json').write_text(json.dumps({
        'change': 'One row of three panels: source continuation, final-answer prompt and trace ablation, source-prediction agreement.',
        'original_point_estimates_unchanged': True,
        'continuation_uncertainty_unchanged': True,
        'answer_transplant_metric': 'Agreement with the clean source prediction, with clean target baselines.',
        'source_sha256': previous.SOURCES,
        'layout_audits': previous.AUDITS,
        'paper_pdf_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
    }, indent=2), encoding='utf-8')
    print(f'Installed {destination}')


if __name__ == '__main__':
    main()
