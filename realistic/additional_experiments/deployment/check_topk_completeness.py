"""Independent local completeness and aggregate checks from archived raw arms."""
import csv,hashlib,json,tarfile,os
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
B=Path(__file__).resolve().parents[1];R=B/'runs/topk_completion_20260907_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(b):return hashlib.sha256(b).hexdigest()
def local(remote):
    rest=remote.split('/additional_experiments/')[1];version,rel=rest.split('/',1)
    p=(B/'runs'/version/'downloaded'/rel).resolve()
    return Path('\\\\?\\'+str(p)) if os.name=='nt' else p
with tarfile.open(R/'results.tgz') as archive:
    files={m.name.removeprefix('./'):archive.extractfile(m).read() for m in archive if m.isfile()}
def get(name):return json.loads(files[name])
cfg=get('protocol.json');audit=read(R/'analysis/audit.json');checks=Counter();panels=[];reasons=Counter();groups=defaultdict(lambda:defaultdict(list));expected_files=set();natural=set()
assert sha(files['protocol.json'])==audit['protocol_sha256']
for name,h in audit['file_hashes'].items():assert sha(files[name])==h;checks['result_hashes']+=1
for name,h in cfg['files'].items():
    if name in files:assert sha(files[name])==h;checks['archived_frozen_hashes']+=1
for model in cfg['sources']:
    for stage in ['canary','full']:assert get(f'{stage}/{model}/complete.json')['status']=='PASS'
    widths=get(f'full/{model}/environment.json')['widths']
    for task in ['kth','category']:
        plans=get(f'plans/{task}_{model}.json');banks=get(f'banks/{task}_{model}.json')
        assert len(plans)==600 and len({(p['case_id'],p['mode']) for p in plans})==600
        assert Counter(p['seed'] for p in plans)=={s:20 for s in range(1234,1264)}
        for p in plans:
            assert p['split']==('discovery' if p['seed']<1254 else 'confirmation')
            for name,h in p['source_hashes'].items():assert sha((local(p['source']+'/placeholder').parent/name).read_bytes())==h;checks['natural_hashes']+=1
            natural.add((model,task,p['mode'],p['case_id']))
        for key,bank in banks.items():
            mode,assay=key.rsplit('_',1);panel=[p for p in plans if p['split']=='confirmation' and p['mode']==mode];assert len(panel)==100
            eligible=0;counts=Counter();seen_by_k=defaultdict(set)
            for p in panel:
                prefix=p['broad_prefix'] if assay=='broad' else (p['target'] or {}).get('prefix_length')
                folder=f"full/{model}/{task}/{key}/{p['case_id']}"
                if not prefix:
                    u=get(folder+'/unavailable.json');assert u['reason']==p['unavailable'].get(assay)
                    reasons[model,task,key,str(u['reason'])]+=1;continue
                eligible+=1;clean=None
                for k in bank['sizes']:
                    name=folder+f'/K{k}.json';expected_files.add(name);x=get(name)
                    assert (x['case_id'],x['mode'],x['assay'],x['seed'],x['k'],x['prefix_length'])==(p['case_id'],mode,assay,p['seed'],k,prefix)
                    assert all(v=='sdpa' for v in x['backend'])
                    arms=x['arms'];assert [v['name'] for v in arms]==['clean','selected','random_0','random_1','random_2']
                    selected=set(map(tuple,bank['heads'][:k]));assert arms[1]['heads']==bank['heads'][:k] and len(selected)==k
                    for arm in arms[2:]:
                        hs=set(map(tuple,arm['heads']));assert len(hs)==k and not hs&selected
                        assert Counter(l for l,h in hs)==Counter(l for l,h in selected)
                    for arm in arms:
                        assert all(0<=l<len(widths) and 0<=h<widths[l] for l,h in arm['heads'])
                        counts['truncated_'+arm['name']]+=bool(arm['generation'].get('generation_truncated'))
                        counts['unparsed_'+arm['name']]+=arm['score'].get('prediction','present') is None
                    if clean is None:clean=arms[0]['generation']
                    else:assert clean==arms[0]['generation']
                    vals=[int(a['score']['correct']) for a in arms];rr=sum(vals[2:])/3
                    for metric,v in zip(['clean','selected','random','delta'],[vals[0],vals[1],rr,rr-vals[1]]):groups[model,task,mode,assay,k,metric][p['seed']].append(v)
                    seen_by_k[k].add(p['case_id']);counts['points']+=1
            assert len({frozenset(x) for x in seen_by_k.values()})==1
            assert counts['points']==cfg['expected_points'][model+'_'+task][key]
            panels.append(dict(model=model,task=task,assay=key,candidates=100,eligible=eligible,unavailable=100-eligible,sizes=bank['sizes'],**counts))
actual={n for n in files if n.startswith('full/') and n.rsplit('/',1)[-1].startswith('K') and n.endswith('.json')}
assert actual==expected_files and len(actual)==6739 and len(natural)==2400
with (R/'analysis/summary.csv').open(encoding='utf-8') as f:summary=list(csv.DictReader(f))
assert len(summary)==len(groups)==296
for row in summary:
    key=(row['model'],row['task'],row['mode'],row['assay'],int(row['k']),row['metric']);g=groups[key]
    assert sorted(g)==list(range(1254,1264))
    v=np.array([sum(g[s])/len(g[s]) for s in sorted(g)])
    idx=np.random.default_rng(20260907).integers(0,10,size=(20000,10))
    lo,hi=np.quantile(v[idx].mean(axis=1),[.025,.975])
    assert np.allclose([float(row[k]) for k in ['mean','lower','upper']],[v.mean(),lo,hi],atol=1e-12,rtol=0)
    assert int(row['n'])==sum(map(len,g.values()));checks['summary_rows_recomputed']+=1
checks.update(natural_outputs=len(natural),curves=len(panels),dose_points=len(actual),unavailable=sum(x['unavailable'] for x in panels))
result=dict(status='PASS',checks=dict(checks),panels=panels,unavailable_reasons=[dict(model=k[0],task=k[1],assay=k[2],reason=k[3],n=v) for k,v in reasons.items()],scope='Local recheck of archive, sources, coverage, arms, dose pairing and aggregates; raw text rescoring evidence in existing audit.json')
(R/'analysis/completeness_recheck.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
