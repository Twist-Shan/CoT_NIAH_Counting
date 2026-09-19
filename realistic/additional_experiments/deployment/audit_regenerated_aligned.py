"""Offline provenance, coverage, controls and seed-equal causal summary."""
import json, hashlib, math, csv
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

B = Path(__file__).resolve().parents[1]
ROOT = B/'runs/aligned_gemma_kth_sdpa_20260907_v1/downloaded'
OUT = ROOT.parent/'analysis'
OUT.mkdir(exist_ok=True)
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
checks=Counter(); mismatches=[]; rows=[]; unavailable=Counter(); trunc=Counter()
manifest=read(ROOT/'plans/manifest.json')
for name,h in manifest['files'].items():
    assert sha(ROOT/'plans'/name)==h
    checks['plan_hashes']+=1
for model in ['Gemma4-E4B']:
    for task in ['kth']:
        plans=read(ROOT/'plans'/f'{task}_{model}.json')
        assert len(plans)==600
        sources={}
        for p in plans:
            rel=p['source'].split('/additional_experiments/')[1].split('/')
            d=B/'runs'/rel[0]/'downloaded'/Path(*rel[1:])
            for name,h in p['source_hashes'].items():
                assert sha(d/name)==h,(d,name)
                checks['source_hashes']+=1
            prompt=read(d/'prompt.json'); gen=read(d/'generation.json')
            sources[p['case_id'],p['mode']]=prompt['input_ids']+gen['generated_token_ids']
        for stage in ['canary','full']:
            folder=ROOT/stage/model/task
            assert read(ROOT/stage/model/'complete.json')['status']=='PASS'
            contract=read(ROOT/stage/model/'contract.json')
            assert contract['manifest']==sha(ROOT/'plans/manifest.json')
            if stage=='full': assert contract==read(ROOT/'canary'/model/'contract.json')
            index={(p['case_id'],p['mode']):p for p in plans}
            attention=[read(f) for f in (folder/'attention').glob('*.json')]
            causal=[read(f) for f in (folder/'causal').glob('*.json')]
            assert len(attention)==(600 if stage=='full' else 4)
            assert len(causal)==(200 if stage=='full' else 2)
            if stage=='full':
                assert {(r['case_id'],r['mode']) for r in attention}==set(index)
                assert {(r['case_id'],r['mode']) for r in causal}=={k for k,p in index.items() if p['split']=='confirmation'}
            selection=read(folder/'selection.json'); widths={}
            for r in attention:
                p=index[r['case_id'],r['mode']]
                assert r['seed']==p['seed'] and r['split']==p['split']
                assert bool(r['heads'])==bool(p['broad_prefix'])
                for l,h,s in r['heads']:
                    assert math.isfinite(s) and s>=0
                    widths[l]=max(widths.get(l,0),h+1)
            for mode in ['nonthinking','native_thinking']:
                groups=defaultdict(lambda:defaultdict(list))
                for r in attention:
                    if r['mode']==mode and r['split']=='discovery':
                        for l,h,s in r['heads']: groups[l,h][r['seed']].append(s)
                scores={h:sum(sum(v)/len(v) for v in seeds.values())/len(seeds) for h,seeds in groups.items()}
                selected=[]; used=Counter(); k=manifest['broad_K'][model][mode]
                for l,h in sorted(scores,key=lambda h:(-scores[h],h)):
                    if used[l]<widths[l]//2: selected.append([l,h]); used[l]+=1
                    if len(selected)==k: break
                assert selected==selection[mode]['heads']
                checks['discovery_ranking_reproduced']+=1
            for r in causal:
                p=index[r['case_id'],r['mode']]; original=sources[r['case_id'],r['mode']]
                assert r['seed']==p['seed']
                assert set(r['assays'])==({'broad'} if r['mode']=='nonthinking' else {'broad','targeted'})
                for assay,a in r['assays'].items():
                    length=p['broad_prefix'] if assay=='broad' else (p['target'] or {}).get('prefix_length')
                    key=(model,task,r['mode'],assay)
                    if not length:
                        assert a['status']=='unavailable' and a['reason']==p['unavailable'][assay]
                        if stage=='full': unavailable[key]+=1
                        continue
                    assert a['status']=='PASS' and a['prefix_length']==length and 0<length<len(original)
                    arms={v['name']:v for v in a['arms']}
                    assert set(arms)=={'clean','selected','random_0','random_1','random_2'}
                    bank=selection[r['mode']]['heads'] if assay=='broad' else read(ROOT/'plans/targeted_banks.json')['banks'][model]
                    assert arms['selected']['heads']==bank and arms['clean']['heads']==[]
                    clean=arms['clean']['generation']['generated_token_ids']
                    exact=clean==original[length:][:len(clean)]
                    checks['clean_comparisons']+=1
                    if not exact: mismatches.append(dict(stage=stage,key=key,case_id=r['case_id'],assay=assay))
                    for name,arm in arms.items():
                        heads=arm['heads']; g=arm['generation']
                        assert len(heads)==len(set(map(tuple,heads)))
                        assert len(g['generated_token_ids'])<=(64 if assay=='broad' else 128)
                        if name.startswith('random'):
                            assert not set(map(tuple,heads)) & set(map(tuple,bank))
                            assert Counter(l for l,h in heads)==Counter(l for l,h in bank)
                            if assay=='broad': assert heads==selection[r['mode']]['controls'][int(name[-1])]
                        if name!='clean':
                            hooks=g.get('hook_calls',g.get('intervention_hook_applications',{}))
                            assert set(map(int,hooks))=={l for l,h in heads} and all(v>0 for v in hooks.values())
                        if stage=='full': trunc[key+(name,)]+=bool(g.get('generation_truncated'))
                        checks['arms']+=1
                    if stage=='full':
                        rows.append(dict(model=model,task=task,mode=r['mode'],assay=assay,seed=r['seed'],case_id=r['case_id'],clean=int(arms['clean']['score']['correct']),selected=int(arms['selected']['score']['correct']),random=sum(int(arms[f'random_{i}']['score']['correct']) for i in range(3))/3))
groups=defaultdict(list)
for r in rows: groups[r['model'],r['task'],r['mode'],r['assay']].append(r)
summary=[]
for key,rs in sorted(groups.items()):
    seeds=sorted({r['seed'] for r in rs}); assert seeds==list(range(1254,1264))
    means=np.array([[np.mean([r[v] for r in rs if r['seed']==s]) for v in ['clean','selected','random']] for s in seeds])*100
    diff=means[:,2]-means[:,1]
    rng=np.random.default_rng(20260906); ci=np.percentile(diff[rng.integers(0,10,(20000,10))].mean(axis=1),[2.5,97.5])
    summary.append(dict(zip(['model','task','mode','assay'],key),n=len(rs),seeds=10,clean=means[:,0].mean(),selected=means[:,1].mean(),random=means[:,2].mean(),extra_failure_pp=diff.mean(),ci_low=ci[0],ci_high=ci[1],unavailable=unavailable[key]))
for name,data in [('per_case.csv',rows),('summary.csv',summary)]:
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
result=dict(status='PASS' if not mismatches else 'CLEAN_MISMATCH_REVIEW',checks=dict(checks),clean_mismatches=mismatches,truncated={'|'.join(k):v for k,v in trunc.items()},summary=summary)
(OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
