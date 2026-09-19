"""Verify display-only changes against the frozen sources, without re-estimation."""
from pathlib import Path
import csv
import hashlib
import json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OLD = ROOT / 'figures/enumeration_bullet_only_20260917'
load = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = load(OUT / 'manifest.json')
checks = {}
for path, expected in manifest['source_sha256'].items():
    assert sha(ROOT / path) == expected, path
checks['source_hashes'] = len(manifest['source_sha256'])
for name, rec in manifest['figures'].items():
    assert not rec['checks']['text_outside'] and not rec['checks']['text_overlaps'], name
    for file, expected in rec['artifacts_sha256'].items():
        assert sha(OUT / file) == expected, file
    assert rec['source_rows'] == len(load(OUT / 'data' / f'{name}.json')), name

for name in ['enumeration_representations', 'enumeration_retrieve', 'enumeration_readout', 'enumeration_pca_bullet']:
    actual = load(OUT / 'data' / f'{name}.json')
    assert actual == load(OLD / 'data' / f'{name}.json'), name
    checks[name + '_unchanged_rows'] = len(actual)
assert sha(OUT / 'enumeration_pca_bullet.pdf') == sha(OLD / 'enumeration_pca_bullet.pdf')

native = list(csv.DictReader((ROOT / 'figures/cot-reasoning/attention/data/head_scores.csv').read_text(encoding='utf-8-sig').splitlines()))
bullet = load(OLD / 'data/enumeration_head_scores.json')
for name, source in [('cot_full_head_scores', native), ('enumeration_head_scores', bullet)]:
    expected = []
    for model, bank_size in [('Qwen3-8B', 128), ('Gemma4-E4B', 6)]:
        group = [r for r in source if r['model'] == model and (name != 'cot_full_head_scores' or r['displayed_global_attention'] == 'True')]
        if name == 'cot_full_head_scores':
            expected.extend(dict(model=model, mode='thinking', layer=int(r['layer']), head=int(r['head']), score=float(r['mean_targeted_score']), selected=r['highlighted_ablation_bank'] == 'True', rank=int(r['discovery_rank'])) for r in group)
        else:
            assert all(float(a['score']) >= float(b['score']) for a, b in zip(group, group[1:])), model
            expected.extend(dict(model=model, mode='enumeration_bullet', layer=int(r['layer']) + 1, head=int(r['head']) + 1, score=float(r['score']), selected=i < bank_size, rank=i + 1) for i, r in enumerate(group))
    actual = load(OUT / 'data' / f'{name}.json')
    assert actual == expected, name
    for model, size in [('Qwen3-8B', 128), ('Gemma4-E4B', 6)]:
        assert sum(r['selected'] for r in actual if r['model'] == model) == size
    checks[name] = dict(scores_and_frozen_membership_unchanged=True, rows=len(actual))

doses = list(csv.DictReader((ROOT / 'figures/cot_completion_20260912/cot_current_bank_dose.csv').read_text(encoding='utf-8-sig').splitlines()))
actual = load(OUT / 'data/cot_current_bank_dose.json')
assert len(actual) == len(doses) == 48
for a, r in zip(actual, doses):
    assert a['model'] == r['model'] and a['k'] == int(r['k'])
    assert a['metric'] == {'next_item': 'next_city_failure', 'final_count': 'final_exact_count_failure'}[r['outcome']]
    assert a['condition'] == ('selected_minus_clean' if r['condition'] == 'selected_bank' else 'random_minus_clean')
    assert (a['estimate'], *a['ci95']) == tuple(float(r[k]) for k in ['mean', 'ci95_low', 'ci95_high'])
checks['thinking_dose_unchanged_rows'] = len(actual)

old_update = load(OLD / 'data/enumeration_update.json')
new_update = load(OUT / 'data/enumeration_update.json')
continuation = lambda rows: [r for r in rows if r['metric'] == 'conditional_next_step']
assert continuation(new_update) == continuation(old_update)
assert len(continuation(new_update)) == 16
audit_path = ROOT / 'realistic/outputs/enumeration_replay_midlayer_20260917/midlayer_audit/combined_update_audit.json'
audit = load(audit_path)
assert audit['status'] == 'PASS'
for a in [r for r in new_update if r['metric'] == 'target_successor_adoption']:
    c, = [c for c in audit['cells'] if c['model'] == a['model'] and c['mode'] == 'enumeration_bullet']
    g, = [g for g in c['groups'] if (g['scope'], g['direction'], g['donor_k']) == (a['scope'], a['direction'], 'all')]
    h = g['conditions'][a['condition']]['continuation']['1']
    assert a['layer'] == c['layer_one_based']
    assert (a['numerator'], a['denominator']) == (h['successes'], h['conditional_eligible'])
    assert a['estimate'] == h['conditional']['estimate'] and a['ci95'] == h['conditional']['ci95']
    assert h['conditional_eligible'] == h['total_trials'] == 30
assert len(new_update) == 40
checks['update'] = dict(raw_adoption_rows=24, unchanged_continuation_rows=16,
    display_change='Archived Target and self adoption replace the paired difference in scope panels; no refitting or bootstrap rerun.')

a, b = [manifest['figures'][name] for name in ['cot_full_head_scores', 'enumeration_head_scores']]
assert a['canvas_inches'] == b['canvas_inches'] == [6.5, 3.5]
assert a['axis_order'] == b['axis_order']
assert a['color_range'] == b['color_range'] == [0, 1]
a, b = [manifest['figures'][name] for name in ['cot_current_bank_dose', 'enumeration_retrieve']]
assert a['canvas_inches'] == b['canvas_inches'] == [6.5, 4.25]
assert a['model_rows'] == b['model_rows'] and a['outcome_columns'] == b['outcome_columns']
checks['paired_display_profiles'] = 'PASS'
result = dict(status='PASS', checks=checks, statistics_recomputed=False, inference=False)
(OUT / 'data_audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
