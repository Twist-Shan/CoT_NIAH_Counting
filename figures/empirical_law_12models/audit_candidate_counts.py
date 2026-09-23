"""Audit candidate table counts against request records; do not edit manuscript."""
from pathlib import Path
from collections import Counter
import json
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BASE = ROOT / 'realistic/outputs/realistic_niah_v3_1_20260819_formal/20260819_formal'
COUNTS = [5, 8, 9, 12, 15]
MODES = ['direct', 'native_thinking']
KEYS = ['comparison_slot', 'L', 'N', 'prompt_mode']

def main():
    d = pd.read_csv(OUT / 'candidate_n_audit_requests.csv')
    t = d[d.N.isin(COUNTS)].copy()
    assert not t.duplicated(KEYS + ['seed']).any()
    g = t.groupby(KEYS).agg(n_total=('seed', 'size'), n_correct=('exact_count', 'sum'),
                            n_parseable=('parse_success', 'sum'), n_truncated=('truncated', 'sum')).reset_index()
    assert len(g) == 240 and g.n_total.eq(30).all()
    plotted = pd.read_csv(OUT / 'plotted_cells.csv').rename(columns={'mode': 'prompt_mode'})
    matched = g.merge(plotted, on=KEYS, suffixes=('_audit', '_plot'), validate='one_to_one')
    assert len(matched) == 240 and matched.n_correct_audit.eq(matched.n_correct_plot).all()
    verified = 0
    for source, b in t.groupby('source_file'):
        path = BASE / source.replace('\\', '/')
        wanted = {(int(r.N), int(r.L), r.prompt_mode, int(r.seed)): r for r in b.itertuples()}
        found = set()
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                r = json.loads(line)
                key = (r['num_needles'], r['target_passage_tokens'], r['prompt_mode'], r['seed'])
                if key not in wanted:
                    continue
                assert key not in found
                found.add(key)
                ref, e = wanted[key], r['evaluation']
                assert bool(e['exact_count']) == bool(ref.exact_count)
                assert bool(e['truncated']) == bool(ref.truncated)
                assert e['predicted_count'] == ref.predicted_count or (e['predicted_count'] is None and pd.isna(ref.predicted_count))
        assert found == set(wanted), source
        verified += len(found)
        print(f'Raw records verified: {verified}/7200', flush=True)
    g['accuracy_percent'] = 100 * g.n_correct / g.n_total
    g.to_csv(OUT / 'candidate_counts_5_8_9_12_15.csv', index=False)
    full = d.groupby(KEYS).exact_count.sum()
    all_n = sorted(int(n) for n in d.N.unique())
    flags, distributions = [], []
    for row in g.itertuples():
        n = row.N
        left, right = all_n[all_n.index(n)-1], all_n[all_n.index(n)+1]
        a, c, z = [int(full.loc[row.comparison_slot, row.L, k, row.prompt_mode]) for k in [left,n,right]]
        if c-max(a,z) >= 5 or min(a,z)-c >= 5:
            flags.append(dict(model=row.comparison_slot,L=row.L,mode=row.prompt_mode,N=n,
                              neighbor_N=[left,n,right],correct=[a,c,z],total_each=30))
        b = d[(d.comparison_slot == row.comparison_slot) & (d.L == row.L) & (d.prompt_mode == row.prompt_mode)]
        for k in [left,n,right]:
            q = b[b.N == k]
            distributions.append(dict(model=row.comparison_slot,L=row.L,mode=row.prompt_mode,target_N=n,true_N=k,
                                      predictions=dict(Counter(str(v) for v in q.predicted_count))))
    (OUT/'candidate_counts_diagnostics.json').write_text(json.dumps(dict(
        candidate_counts=COUNTS,conditions=240,raw_requests_verified=verified,
        descriptive_screen='Candidate accuracy at least 5/30 above both sampled neighbors, or below both. Not a significance test; neighbors can be unequally spaced.',
        local_flags=flags,prediction_distributions=distributions),indent=2)+'\n',encoding='utf-8')
    md = ['# Candidate counts: 5, 8, 9, 12, 15', '',
          'All 12 comparison groups, L=1k/20k, both modes; 30 paired seeds per condition. 7,200 raw records checked. No manuscript edits.', '',
          'Values below are Non-thinking / CoT-reasoning parsed exact accuracy (%). All requests remain in the denominator. GLM and Ministral use separate checkpoints.', '',
          'These are descriptive checks, not confirmatory significance tests. Data-dependent choice of a count cannot establish absence of bias. Full observed count curves should remain available.', '']
    for n in COUNTS:
        md += [f'## N={n}', '', '| Model | 1k | 20k |', '|---|---|---|']
        for slot,b in g[g.N == n].groupby('comparison_slot'):
            vals = []
            for length in [1000,20000]:
                q = b[b.L == length].set_index('prompt_mode')
                vals.append(' / '.join(f'{q.loc[m,"accuracy_percent"]:.1f}' for m in MODES))
            md.append('| '+slot+' | '+' | '.join(vals)+' |')
        md.append('')
    md += ['## Local departures', '', 'All counts below are correct requests out of 30; rows flagged by the descriptive screen above.', '']
    for f in flags:
        md.append(f'- {f["model"]}, L={f["L"]}, {f["mode"]}: N={f["neighbor_N"]}, correct={f["correct"]}.')
    md += ['', '## Parsing and output budget', '', '| Model | N | L | Mode | Unparseable / 30 | Truncated / 30 |', '|---|---|---|---|---|---|']
    for r in g[(g.n_parseable < 30) | (g.n_truncated > 0)].itertuples():
        md.append(f'| {r.comparison_slot} | {r.N} | {r.L} | {r.prompt_mode} | {30-r.n_parseable} | {r.n_truncated} |')
    md += ['', 'Unparseable and truncated categories can overlap; do not add their counts.', '',
           'Recommendation: N=5 is the most interpretable moderate-load illustration among these candidates, but is not an unbiased overall model ranking. N=8 has Qwen local peaks; N=9 has pronounced output-digit-related troughs; N=12 and N=15 retain model-specific irregularities. Use full observed curves or an explicitly defined multi-count summary for overall claims.']
    (OUT/'candidate_counts_5_8_9_12_15.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('Complete: 240 cells, 7200 raw requests, all matched plotted observations.',flush=True)

if __name__ == '__main__':
    main()
