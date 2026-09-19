r"""\gpt: group donor continuation with answer sources; retain full-state readout separately."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import shutil
import sys

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[1]
PREVIOUS = WORK / 'figures/synthetic_appendix_evidence_revision_20260908'
PAPER = WORK / 'runs/paper_figures/figures/synthetic_appendix'
sys.path.insert(0, str(PREVIOUS))
import build_figures as original
from audit_style import inspect_figure

np, pd, plt = original.np, original.pd, original.plt
SOURCES = {}
AUDITS = []


def read(path):
    SOURCES[str(path.relative_to(WORK))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)


def donor_estimates():
    # Use the frozen legacy cohort that generated the current paper figure.
    # The larger, later native-continuation run is a separate experiment.
    trials = read(WORK / 'synthetic/work/v58_final/analysis/'
                  'v58_unified_legacy_20260905/continuation/item_span_w2/rollout_trials.csv')
    archived = read(PREVIOUS / 'plot_data/07_donor_continuation.csv')
    trials = trials.loc[trials.split.eq('confirmation') & trials.layer.eq(1)
                        & trials.condition.eq('full_donor_patch')]
    metrics = ['donor_marker_adoption', 'donor_prefix_h2', 'donor_prefix_h3', 'donor_prefix_h4']
    rows, input_rows = [], []
    for h, metric in enumerate(metrics, 1):
        eligible = trials.loc[trials[metric].notna()]
        if h == 1:
            eligible = eligible.loc[eligible.successor_identity_distinct]
        values = eligible.groupby('prompt_sha256', sort=True)[metric].mean()
        old = archived.loc[archived.metric.eq(metric)].iloc[0]
        assert len(values) == int(old.prompts) == 8
        assert len(eligible) == int(old.pairs)
        assert np.isclose(values.mean(), old.treatment_mean)
        rng = np.random.default_rng(20260908 + h)
        draws = rng.choice(values.to_numpy(), size=(10000, len(values)), replace=True).mean(axis=1)
        low, high = np.percentile(draws, [2.5, 97.5])
        rows.append(dict(h=h, metric=metric, layer=1, pairs=len(eligible), inputs=len(values),
                         donor_match=values.mean(), ci_low=low, ci_high=high,
                         bootstrap_seed=20260908+h, bootstrap_resamples=10000))
        input_rows.extend(dict(h=h, prompt_sha256=key, donor_match=value)
                          for key, value in values.items())
    pd.DataFrame(input_rows).to_csv(OUT / 'plot_data/06_donor_continuation_by_input.csv', index=False)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'plot_data/06_donor_continuation_absolute.csv', index=False)
    return result


def save(fig, stem):
    fig.canvas.draw()
    audit = inspect_figure(fig, stem)
    assert not audit['text_outside_canvas'] and not audit['text_overlap_candidates'], audit
    AUDITS.append(audit)
    for extension in ['pdf', 'png', 'svg']:
        fig.savefig(OUT / f'{stem}.{extension}', dpi=450, facecolor='white')
    plt.close(fig)


def continuation_and_sources(donor, sources):
    fig = plt.figure(figsize=(6.5, 2.85))
    left = fig.add_axes([.078, .30, .37, .54])
    right = fig.add_axes([.61, .30, .37, .54])
    x = np.arange(1, 5)
    means = 100 * donor.donor_match.to_numpy()
    low, high = 100 * donor.ci_low.to_numpy(), 100 * donor.ci_high.to_numpy()
    left.bar(x, means, width=.57, color=original.T, label='Donor state')
    left.errorbar(x, means, yerr=np.vstack([means-low, high-means]), fmt='none',
                  ecolor=original.INK, capsize=3, elinewidth=1.0, capthick=1.0)
    for xx, mean, upper in zip(x, means, high):
        left.text(xx, upper+2.5, f'{mean:.1f}', ha='center', va='bottom', fontsize=9)
    original.panel(left, 'A. Donor continuation', 'Exact match (%)')
    left.set(xticks=x, xlabel='Number of subsequent markers', xlim=(.4, 4.6),
             ylim=(0, 110), yticks=[0, 25, 50, 75, 100])
    fig.legend(*left.get_legend_handles_labels(), loc='center',
               bbox_to_anchor=(.263, .078), fontsize=9,
               handlelength=1.3, handletextpad=.4, borderpad=.1)

    thinking = sources.loc[sources['mode'].eq('thinking')]
    x, width = np.arange(2), .30
    for arms, offset, color, label in [
        (['records', 'trace'], -width/2, original.T, 'Source ablation'),
        (['ordinary_records_budget', 'ordinary_trace_budget'], width/2, original.GRAY, 'Control ablation'),
    ]:
        vals = [100 * thinking.loc[thinking.arm.eq(arm), 'accuracy'].iloc[0] for arm in arms]
        bars = right.bar(x+offset, vals, width=width, color=color, label=label)
        right.bar_label(bars, fmt='%.0f', padding=2, fontsize=9)
    clean = 100 * thinking.loc[thinking.arm.eq('clean'), 'accuracy'].iloc[0]
    right.axhline(clean, color=original.INK, ls=':', lw=1, label=f'Clean ({clean:.0f}%)')
    original.panel(right, 'B. Final-answer sources', 'Answer accuracy (%)')
    right.set(xticks=x, xticklabels=['Prompt\ntargets', 'Generated\ntrace'],
              ylim=(0, 110), yticks=[0, 25, 50, 75, 100])
    handles, labels = right.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.795, .002),
               ncol=1, fontsize=9, handlelength=1.3, labelspacing=.18,
               handletextpad=.4, borderpad=.1)
    save(fig, '06_progress_continuation')


def answer_transplants(data):
    fig = plt.figure(figsize=(6.5, 2.6))
    axes = [fig.add_axes([x, .28, .37, .56]) for x in [.078, .61]]
    for ax, metric, title in zip(axes, ['truth', 'prediction'],
                                 ['A. Donor ground truth', 'B. Donor prediction']):
        for mode, color, marker in [('nonthinking', original.NT, 's'), ('thinking', original.T, 'o')]:
            group = data.loc[data['mode'].eq(mode)].sort_values('layer')
            assert group.pairs.eq(180).all()
            ax.plot(group.layer, 100*group['patched_match_'+metric], marker+'-', color=color,
                    mfc='white' if mode == 'thinking' else color, ms=4.5)
            ax.plot(group.layer, 100*group['baseline_match_'+metric], ':', color=color, lw=1.4)
        original.panel(ax, title, 'Match (%)')
        ax.set(xticks=[1, 2, 3, 4], xlabel='Patching layer', xlim=(.85, 4.15),
               ylim=(-4, 110), yticks=[0, 25, 50, 75, 100])
    Line2D = original.Line2D
    fig.legend([Line2D([], [], color=original.NT, marker='s'),
                Line2D([], [], color=original.T, marker='o', mfc='white'),
                Line2D([], [], color=original.INK, ls='-'),
                Line2D([], [], color=original.INK, ls=':')],
               ['Non-thinking', 'Thinking', 'After patching', 'Before patching'],
               loc='lower center', bbox_to_anchor=(.53, .002), ncol=4, fontsize=9,
               handlelength=1.5, columnspacing=1.3, handletextpad=.45)
    save(fig, '07_answer_readout')


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
    donor = donor_estimates()
    sources = read(PREVIOUS / 'plot_data/08_answer_sources_all_modes.csv')
    answer = read(PREVIOUS / 'plot_data/08_answer_state_donor.csv')
    sources.to_csv(OUT / 'plot_data/06_final_answer_sources.csv', index=False)
    answer.to_csv(OUT / 'plot_data/07_answer_transplants.csv', index=False)
    continuation_and_sources(donor, sources)
    answer_transplants(answer)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for stem in ['06_progress_continuation', '07_answer_readout']:
        shutil.copy2(PAPER / f'{stem}.pdf', OUT / f'{stem}.before_grouping_{stamp}.pdf')
        shutil.copy2(OUT / f'{stem}.pdf', PAPER / f'{stem}.pdf')
    (OUT / 'counter_panels_manifest.json').write_text(json.dumps({
        'change': 'Remove continuation random controls; group continuation and answer sources; keep answer transplants in two panels.',
        'original_point_estimates_unchanged': True,
        'continuation_interval': '95% percentile interval for the input-averaged donor-match rate; 10,000 resamples of 8 inputs per endpoint.',
        'source_sha256': SOURCES,
        'layout_audits': AUDITS,
        'paper_pdf_sha256': {stem: hashlib.sha256((PAPER / f'{stem}.pdf').read_bytes()).hexdigest()
                             for stem in ['06_progress_continuation', '07_answer_readout']},
    }, indent=2), encoding='utf-8')
    print(donor.to_string(index=False))
    print('Installed both revised paper figures.')


if __name__ == '__main__':
    main()
