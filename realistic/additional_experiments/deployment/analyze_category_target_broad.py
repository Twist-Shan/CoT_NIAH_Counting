"""Audit all record masks, rankings and generated outputs; summarize by seed."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import struct
import time


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def csv_rows(p):
    with p.open(encoding='utf-8', newline='') as f: return list(csv.DictReader(f))
def write_json(p, obj): p.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def score_distance(saved, actual):
    """Distance between nonnegative finite binary64 scores, in representable steps."""
    assert math.isfinite(saved) and math.isfinite(actual) and min(saved, actual) >= 0
    if saved == actual: return 0
    assert saved > 0 and actual > 0, 'Zero/nonzero score disagreement'
    bits=lambda value:struct.unpack('>Q',struct.pack('>d',value))[0]
    return abs(bits(saved)-bits(actual))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--stage', choices=['canary', 'full'], default='full')
    ap.add_argument('--score-comparison', choices=['exact','ulp8'], default='exact')
    ap.add_argument('--analysis-output', type=Path, help='Write a separate full recheck without replacing delivered analysis')
    args = ap.parse_args()
    plain = args.root.resolve(); root = Path('\\\\?\\'+str(plain)) if os.name == 'nt' else plain
    started = time.perf_counter(); cfg = read(root/'protocol.json'); protocol_hash = sha(root/'protocol.json')
    sys.path[:0] = [str(root), str(root/'src')]
    from category_target_broad import target_indices, score_record_masses, rank_rows, canary_cases
    from local_selection import select_heads, random_control, control_audit
    from task_scoring import score_generation
    checks = Counter(); hashes = {}; detail = []; geometry_rows = []
    numeric = dict(mode=args.score_comparison,allowed_ulp_distance=8 if args.score_comparison=='ulp8' else 0,
                   scores_compared=0,nonidentical_scores=0,max_ulp_distance=0,max_absolute_difference=0.)
    assert cfg['record_scope'] == 'question_target_category'
    for rel, h in cfg['files'].items(): assert sha(root/rel) == h, rel; checks['frozen_files'] += 1
    baseline = {(r['model'], r['case_id'], int(r['k'])):r for r in csv_rows(root/'baseline_per_case.csv')}
    for model in cfg['models']:
        plans = read(root/'plans'/f'{model}.json'); plan_map = {p['case_id']:p for p in plans}
        assert len(plans) == len(plan_map) == 300
        assert Counter((p['split'],p['case']['target_category']) for p in plans) == {
            ('discovery','city'):100, ('discovery','flower'):100, ('confirmation','city'):50, ('confirmation','flower'):50}
        for stage in ['discovery', args.stage]:
            complete = read(root/stage/model/'complete.json')
            assert complete['status'] == 'PASS' and complete['protocol_sha256'] == protocol_hash
        bankpath = root/'banks'/f'{model}.json'; bank = read(bankpath)
        assert bank['record_scope'] == cfg['record_scope'] and bank['protocol_sha256'] == protocol_hash
        observations = []; recomputed_observations = []
        for rel, h in bank['source_hashes'].items():
            assert sha(root/rel) == h
            row = read(root/rel); p = plan_map[row['case_id']]
            assert row['split'] == p['split'] == 'discovery' and row['seed'] in cfg['discovery_seeds']
            assert row['protocol_sha256'] == protocol_hash and row['prefix_length'] == p['broad_prefix']
            assert all(x == 'sdpa' for x in row['backend'])
            geo = row['geometry']; expected = target_indices(p['case'])
            assert geo['selected_record_indices'] == expected
            assert geo['target_category'] == p['case']['target_category'] and geo['target_record_count'] == len(expected)
            assert len(geo['all_record_token_spans']) == 10
            assert len(row['heads']) == len(row['all_record_masses']) == sum(cfg['widths'][model])
            recomputed_heads = []
            for score, (layer, head, masses) in zip(row['heads'],row['all_record_masses']):
                value = score_record_masses(p['case'], masses)
                assert score[:2] == [layer, head]
                distance = score_distance(score[2],value)
                assert distance <= numeric['allowed_ulp_distance'], (model,row['case_id'],layer,head,score[2],value,distance)
                numeric['scores_compared'] += 1
                numeric['nonidentical_scores'] += int(distance > 0)
                numeric['max_ulp_distance'] = max(numeric['max_ulp_distance'],distance)
                numeric['max_absolute_difference'] = max(numeric['max_absolute_difference'],abs(score[2]-value))
                recomputed_heads.append([layer,head,value])
                assert 0 <= layer < len(cfg['widths'][model]) and 0 <= head < cfg['widths'][model][layer]
            observations.append(row); checks['discovery_rows_rescored'] += 1
            recomputed_observations.append(dict(row,heads=recomputed_heads))
            hashes[rel.replace('\\','/')] = h
        assert len(observations) == 200 and rank_rows(observations) == bank['ranking']
        assert [r[:2] for r in rank_rows(recomputed_observations)] == [r[:2] for r in bank['ranking']], 'Recomputed complete head order changed'
        checks['recomputed_head_orders_identical'] += 1
        checks['complete_rankings_reproduced'] += 1
        assert bank['sizes'] == cfg['sizes'][model]
        for k in bank['sizes']:
            c = bank['conditions'][str(k)]; heads = select_heads(bank['ranking'], k)
            assert c['selected'] == heads
            assert c['random'] == [random_control(heads,cfg['widths'][model],s) for s in cfg['random_seeds']]
            assert c['control_audit'] == control_audit(heads,cfg['widths'][model],c['random'])
            for repeat, layers in enumerate(c['control_audit']):
                for layer in layers: geometry_rows.append(dict(model=model,k=k,repeat=repeat,**layer))
            checks['frozen_doses'] += 1
        panel = canary_cases(plans) if args.stage == 'canary' else [p for p in plans if p['split']=='confirmation']
        sizes = sorted({bank['sizes'][0],bank['sizes'][-1]}) if args.stage == 'canary' else bank['sizes']
        for p in panel:
            assert p['seed'] in cfg['confirmation_seeds']
            dest = root/args.stage/model/'rows'/p['case_id']; saved = read(dest/'clean.json')
            assert saved['protocol_sha256'] == protocol_hash and saved['prefix_length'] == p['broad_prefix']
            assert {int(f.stem[1:]) for f in dest.glob('K*.json')} == set(sizes)
            for k in sizes:
                path = dest/f'K{k}.json'; x = read(path); c = bank['conditions'][str(k)]
                assert x['protocol_sha256'] == protocol_hash and x['bank_sha256'] == sha(bankpath)
                assert x['case_id'] == p['case_id'] and x['seed'] == p['seed'] and x['target_category'] == p['case']['target_category']
                assert x['prefix_length'] == p['broad_prefix'] and x['max_new_tokens'] == cfg['max_new_tokens'] == 64
                assert all(v == 'sdpa' for v in x['backend']) and x['arms'][0] == saved['arm']
                assert [a['name'] for a in x['arms']] == ['clean','selected','random_0','random_1','random_2']
                assert [a['heads'] for a in x['arms']] == [[],c['selected']]+c['random']
                values = []
                for arm in x['arms']:
                    gen = arm['generation']
                    assert score_generation(gen,p['case'],mode='nonthinking',assay='broad') == arm['score']
                    assert len(gen['generated_token_ids']) <= 64
                    values.append(int(arm['score']['correct'])); checks['arms_rescored'] += 1
                old = baseline[model,p['case_id'],k]
                assert values[0] == int(old['clean']), ('Clean baseline drift',model,p['case_id'],k)
                rand = sum(values[2:])/3
                detail.append(dict(model=model,task='category',mode='nonthinking',assay='broad',
                    k=k,case_id=p['case_id'],seed=p['seed'],target_category=p['case']['target_category'],
                    clean=values[0],selected=values[1],random=rand,delta=rand-values[1],clean_drop=values[0]-values[1],
                    all_records_selected=float(old['selected']),all_records_random=float(old['random']),
                    delta_gain_vs_all_records=rand-values[1]-float(old['delta'])))
                hashes[str(path.relative_to(root)).replace('\\','/')] = sha(path); checks['points'] += 1
    expected_points = 16 if args.stage == 'canary' else cfg['expected_full_points']
    assert checks['points'] == expected_points and checks['discovery_rows_rescored'] == 400
    audit = dict(status='PASS',stage=args.stage,checks=dict(checks),protocol_sha256=protocol_hash,
                 file_hashes=hashes,elapsed_seconds=time.perf_counter()-started,python=sys.version,analyzer_sha256=sha(Path(__file__)),numeric_comparison=numeric)
    if args.stage == 'canary':
        write_json(root/'canary_audit.json',audit)
        print(json.dumps(dict(status='PASS',checks=dict(checks))),flush=True); return
    import numpy as np
    groups = defaultdict(list)
    for row in detail: groups[row['model'],row['k']].append(row)
    summary, tests = [], []
    for (model,k), records in sorted(groups.items()):
        for population in ['all_examples','city_questions','flower_questions','clean_correct']:
            chosen = [r for r in records if population == 'all_examples' or
                      (population == 'city_questions' and r['target_category'] == 'city') or
                      (population == 'flower_questions' and r['target_category'] == 'flower') or
                      (population == 'clean_correct' and r['clean'] == 1)]
            if not chosen: continue
            for metric in ['clean','selected','random','delta','clean_drop','delta_gain_vs_all_records']:
                byseed = defaultdict(list)
                for r in chosen: byseed[r['seed']].append(r[metric])
                if population != 'clean_correct': assert sorted(byseed) == cfg['confirmation_seeds']
                v = np.array([np.mean(byseed[s]) for s in sorted(byseed)])
                idx = np.random.default_rng(cfg['stats']['seed']).integers(0,len(v),(cfg['stats']['bootstrap'],len(v)))
                lo, hi = np.quantile(v[idx].mean(axis=1),[.025,.975])
                out = dict(model=model,k=k,population=population,metric=metric,n=len(chosen),seeds=len(v),
                           mean=float(v.mean()),lower=float(lo),upper=float(hi))
                summary.append(out)
                if metric == 'delta' and population in ['all_examples','clean_correct']:
                    signs = np.array(list(itertools.product([-1,1],repeat=len(v))))
                    pvalue = float(np.mean(np.abs((signs*v).mean(axis=1)) >= abs(v.mean())-1e-12))
                    tests.append(dict(out,p_exact=pvalue,p_holm=None))
    for population in ['all_examples','clean_correct']:
        family = sorted([r for r in tests if r['population']==population], key=lambda r:r['p_exact'])
        running = 0.
        for i,r in enumerate(family):
            running = max(running,min(1.,(len(family)-i)*r['p_exact'])); r['p_holm'] = running
    analysis = args.analysis_output.resolve() if args.analysis_output else root/'analysis'; analysis.mkdir(parents=True,exist_ok=True)
    for name,rows in [('summary.csv',summary),('hypothesis_tests.csv',tests),('per_case.csv',detail),('random_control_geometry.csv',geometry_rows)]:
        with (analysis/name).open('w',encoding='utf-8',newline='') as f:
            w = csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    audit.update(elapsed_seconds=time.perf_counter()-started,numpy=np.__version__)
    write_json(analysis/'audit.json',audit)
    print(json.dumps(dict(status='PASS',checks=dict(checks),elapsed_seconds=audit['elapsed_seconds'])),flush=True)


if __name__ == '__main__': main()
