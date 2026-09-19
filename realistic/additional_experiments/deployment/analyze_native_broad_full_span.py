"""Audit full record spans, discovery ranks, controls, raw answers, and paired effects."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import struct
import sys
import time

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def write(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def distance(a,b):
    assert math.isfinite(a) and math.isfinite(b) and min(a,b)>=0
    if a==b:return 0
    assert a>0 and b>0
    bits=lambda x:struct.unpack('>Q',struct.pack('>d',x))[0]
    return abs(bits(a)-bits(b))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--stage',choices=['geometry','canary','full'],default='full')
    ap.add_argument('--score-comparison',choices=['exact','ulp8'],default='exact')
    ap.add_argument('--diagnose-numeric',action='store_true',help='Complete remaining checks in an isolated output; any score violation keeps status NUMERIC_HOLD.')
    ap.add_argument('--analysis-output',type=Path);a=ap.parse_args()
    if a.diagnose_numeric and (a.stage!='full' or a.analysis_output is None):
        ap.error('--diagnose-numeric requires full stage and an explicit isolated --analysis-output')
    plain=a.root.resolve();root=Path('\\\\?\\'+str(plain)) if os.name=='nt' else plain
    if a.diagnose_numeric and a.analysis_output.resolve()==plain/'analysis':
        ap.error('Numeric diagnosis must not overwrite the original analysis')
    start=time.perf_counter();cfg=read(root/'protocol.json');ph=sha(root/'protocol.json')
    sys.path[:0]=[str(root),str(root/'src')]
    from native_broad_full_span import score_masses,canary_cases
    from category_target_broad import rank_rows
    from local_selection import select_heads,random_control,control_audit
    from task_scoring import score_generation
    def source(remote):
        if os.name!='nt':return Path(remote)
        run,rest=remote.split('/additional_experiments/',1)[1].split('/',1)
        return Path('\\\\?\\'+str(plain.parent.parent/run/'downloaded'/rest))
    checks=Counter();hashes={};detail=[];coverage=[];geometry_stats=[];controls=[]
    numeric=dict(mode=a.score_comparison,allowed_ulp_distance=8 if a.score_comparison=='ulp8' else 0,
                 scores_compared=0,nonidentical_scores=0,max_ulp_distance=0,max_absolute_difference=0.)
    violations=[]
    assert cfg['record_scope']=='registered_generated_full_record_spans'
    for rel,h in cfg['files'].items():assert sha(root/rel)==h,rel;checks['frozen_files']+=1
    baseline={(r['model'],r['task'],r['case_id'],int(r['k'])):r for r in rows(root/'baseline_per_case.csv')}
    expected_full=0
    for model in cfg['models']:
        assert read(root/'geometry'/model/'complete.json')['protocol_sha256']==ph
        if a.stage!='geometry':
            for stage in ['discovery',a.stage]:
                complete=read(root/stage/model/'complete.json')
                assert complete['status']=='PASS' and complete['protocol_sha256']==ph
        for task in cfg['tasks']:
            plans=read(root/'plans'/f'{task}_{model}.json');pmap={p['case_id']:p for p in plans};geometry={};geohashes={}
            assert len(plans)==len(pmap)==300 and all(p['mode']=='native_thinking' for p in plans)
            assert Counter(p['split'] for p in plans)=={'discovery':200,'confirmation':100}
            for p in plans:
                path=root/'geometry'/model/task/f"{p['case_id']}.json";x=read(path);g=x['geometry'];geometry[p['case_id']]=g
                assert x['protocol_sha256']==ph and x['case_id']==p['case_id'] and x['source_hashes']==p['source_hashes']
                for name,h in p['source_hashes'].items():assert sha(source(p['source'])/name)==h
                checks['geometry_inputs']+=1
                assert g['record_scope']==cfg['record_scope'] and g['record_count']==len(g['records'])
                assert g['available']==bool(g['records'])
                if g['available']:
                    prompt=read(source(p['source'])/'prompt.json');generation=read(source(p['source'])/'generation.json')
                    raw=generation['completion_text_raw'];n=len(prompt['input_ids'])
                    assert g['prompt_token_count']==n and g['prefix_length']==p['broad_prefix']
                    assert len({r['occurrence'] for r in g['records']})==g['record_count']
                    for r in g['records']:
                        sa,sb=r['char_start'],r['char_end'];ta,tb=r['token_start'],r['token_end']
                        assert 0<=sa<sb<=len(raw) and n<=ta<tb<=p['broad_prefix']
                        assert r['text']==raw[sa:sb] and r['token_ids']==generation['generated_token_ids'][ta-n:tb-n]
                        for end,key in [(sa,'prefix_start_sha256'),(sb,'prefix_end_sha256')]:
                            assert hashlib.sha256(raw[:end].encode()).hexdigest()==r[key]
                        checks['literal_record_spans']+=1
                geohashes[p['case_id']]=sha(path);hashes[str(path.relative_to(root)).replace('\\','/')]=sha(path)
                geometry_stats.append(dict(model=model,task=task,case_id=p['case_id'],seed=p['seed'],split=p['split'],
                    baseline_available=bool(p['broad_prefix']),available=g['available'],reason=g['reason'],
                    records=g['record_count'],excluded_records=len(g['excluded_records'])))
            for split in ['discovery','confirmation']:
                assert sorted({p['seed'] for p in plans if p['split']==split and geometry[p['case_id']]['available']})==cfg[split+'_seeds']
            expected_full += sum(p['split']=='confirmation' and geometry[p['case_id']]['available'] for p in plans)*len(cfg['sizes'][model])
            if a.stage=='geometry':continue
            bankpath=root/'banks'/f'{task}_{model}.json';bank=read(bankpath)
            assert bank['protocol_sha256']==ph and bank['record_scope']==cfg['record_scope'] and bank['sizes']==cfg['sizes'][model]
            observed=[];recomputed=[]
            assert len(bank['source_hashes'])==200
            for rel,h in bank['source_hashes'].items():
                path=root/rel;assert sha(path)==h;x=read(path);p=pmap[x['case_id']];g=geometry[p['case_id']]
                assert x['split']==p['split']=='discovery' and p['seed'] in cfg['discovery_seeds']
                assert x['protocol_sha256']==ph and x['geometry_sha256']==geohashes[p['case_id']]
                assert x['available']==g['available'] and x['prefix_length']==p['broad_prefix'] and all(v=='sdpa' for v in x['backend'])
                checks['discovery_rows']+=1;hashes[rel.replace('\\','/')]=h
                if not g['available']:
                    assert x['heads']==x['record_masses']==[];continue
                assert len(x['heads'])==len(x['record_masses'])==sum(cfg['widths'][model])
                new=[]
                for saved,(l,h,masses) in zip(x['heads'],x['record_masses']):
                    assert saved[:2]==[l,h] and len(masses)==g['record_count']
                    v=score_masses(masses);d=distance(saved[2],v)
                    if d>numeric['allowed_ulp_distance']:
                        violations.append(dict(model=model,task=task,case_id=p['case_id'],layer=l,head=h,
                            ulp_distance=d,saved=saved[2],recomputed=v))
                        assert a.diagnose_numeric,(model,task,p['case_id'],l,h,d)
                    numeric['scores_compared']+=1;numeric['nonidentical_scores']+=int(d>0)
                    numeric['max_ulp_distance']=max(numeric['max_ulp_distance'],d)
                    numeric['max_absolute_difference']=max(numeric['max_absolute_difference'],abs(saved[2]-v))
                    new.append([l,h,v])
                observed.append(x);recomputed.append(dict(x,heads=new));checks['discovery_rows_rescored']+=1
            assert rank_rows(observed)==bank['ranking']
            assert [r[:2] for r in rank_rows(recomputed)]==[r[:2] for r in bank['ranking']]
            checks['complete_rankings_reproduced']+=1
            for k in bank['sizes']:
                c=bank['conditions'][str(k)];selected=select_heads(bank['ranking'],k)
                assert c['selected']==selected and c['random']==[random_control(selected,cfg['widths'][model],s) for s in cfg['random_seeds']]
                assert c['control_audit']==control_audit(selected,cfg['widths'][model],c['random'])
                for repeat,layers in enumerate(c['control_audit']):
                    for r in layers:controls.append(dict(model=model,task=task,k=k,repeat=repeat,**r))
                checks['frozen_doses']+=1
            panel=canary_cases(plans,geometry) if a.stage=='canary' else [p for p in plans if p['split']=='confirmation']
            sizes=sorted({bank['sizes'][0],bank['sizes'][-1]}) if a.stage=='canary' else bank['sizes']
            for p in panel:
                g=geometry[p['case_id']];dest=root/a.stage/model/task/p['case_id']
                coverage.append(dict(model=model,task=task,case_id=p['case_id'],seed=p['seed'],available=g['available'],reason=g['reason']))
                if not g['available']:
                    assert read(dest/'unavailable.json')['reason']==g['reason'] and not list(dest.glob('K*.json'));continue
                saved=read(dest/'clean.json');assert saved['protocol_sha256']==ph and saved['prefix_length']==p['broad_prefix']
                assert {int(f.stem[1:]) for f in dest.glob('K*.json')}==set(sizes)
                for k in sizes:
                    path=dest/f'K{k}.json';x=read(path);c=bank['conditions'][str(k)]
                    assert x['protocol_sha256']==ph and x['bank_sha256']==sha(bankpath) and x['geometry_sha256']==geohashes[p['case_id']]
                    assert x['prefix_length']==p['broad_prefix'] and x['max_new_tokens']==64 and all(v=='sdpa' for v in x['backend'])
                    assert x['case_id']==p['case_id'] and x['seed']==p['seed'] and x['k']==k
                    assert [r['name'] for r in x['arms']]==['clean','selected','random_0','random_1','random_2']
                    assert [r['heads'] for r in x['arms']]==[[],c['selected']]+c['random'] and x['arms'][0]==saved['arm']
                    values=[]
                    for arm in x['arms']:
                        assert score_generation(arm['generation'],p['case'],mode='native_thinking',assay='broad')==arm['score']
                        assert len(arm['generation']['generated_token_ids'])<=64
                        values.append(int(arm['score']['correct']));checks['arms_rescored']+=1
                    old=baseline[model,task,p['case_id'],k];assert values[0]==int(old['clean']),('Clean baseline drift',model,task,p['case_id'])
                    rand=sum(values[2:])/3
                    detail.append(dict(model=model,task=task,mode='native_thinking',assay='broad',k=k,case_id=p['case_id'],seed=p['seed'],
                        clean=values[0],selected=values[1],random=rand,delta=rand-values[1],clean_drop=values[0]-values[1],
                        old_end_token_delta=float(old['delta']),delta_gain_vs_end_token=rand-values[1]-float(old['delta'])))
                    hashes[str(path.relative_to(root)).replace('\\','/')]=sha(path);checks['points']+=1
    assert checks['geometry_inputs']==1200 and expected_full<=cfg['expected_full_points_upper_bound']
    audit=dict(status='NUMERIC_HOLD' if violations else 'PASS',stage=a.stage,checks=dict(checks),expected_full_points=expected_full,file_hashes=hashes,
        numeric_comparison=numeric,protocol_sha256=ph,python=sys.version,analyzer_sha256=sha(Path(__file__)),elapsed_seconds=time.perf_counter()-start)
    if a.diagnose_numeric:
        audit.update(diagnostic_only=True,numeric_violations=violations)
    if a.stage=='geometry':
        audit['panels']=[dict(model=m,task=t,split=s,available=sum(r['available'] for r in geometry_stats if (r['model'],r['task'],r['split'])==(m,t,s)),
            candidates=sum((r['model'],r['task'],r['split'])==(m,t,s) for r in geometry_stats)) for m in cfg['models'] for t in cfg['tasks'] for s in ['discovery','confirmation']]
        write(root/'geometry_audit.json',audit)
    elif a.stage=='canary':
        assert checks['points']==16;write(root/'canary_audit.json',audit)
    else:
        assert checks['points']==expected_full and checks['complete_rankings_reproduced']==4 and checks['frozen_doses']==26
        import numpy as np
        groups=defaultdict(list)
        for r in detail:groups[r['model'],r['task'],r['k']].append(r)
        summary=[];tests=[]
        for (model,task,k),group in sorted(groups.items()):
            for population in ['all_examples','clean_correct']:
                chosen=group if population=='all_examples' else [r for r in group if r['clean']]
                if not chosen:continue
                for metric in ['clean','selected','random','delta','clean_drop','old_end_token_delta','delta_gain_vs_end_token']:
                    byseed=defaultdict(list)
                    for r in chosen:byseed[r['seed']].append(r[metric])
                    v=np.array([np.mean(byseed[s]) for s in sorted(byseed)])
                    idx=np.random.default_rng(cfg['stats']['seed']).integers(0,len(v),(cfg['stats']['bootstrap'],len(v)))
                    lo,hi=np.quantile(v[idx].mean(axis=1),[.025,.975])
                    r=dict(model=model,task=task,k=k,population=population,metric=metric,n=len(chosen),seeds=len(v),mean=float(v.mean()),lower=float(lo),upper=float(hi));summary.append(r)
                    if metric=='delta':
                        signs=np.array(list(itertools.product([-1,1],repeat=len(v))))
                        p=float(np.mean(np.abs((signs*v).mean(axis=1))>=abs(v.mean())-1e-12))
                        tests.append(dict(r,p_exact=p,p_holm=None))
        for pop in ['all_examples','clean_correct']:
            family=sorted([r for r in tests if r['population']==pop],key=lambda r:r['p_exact']);running=0.
            for i,r in enumerate(family):running=max(running,min(1.,(len(family)-i)*r['p_exact']));r['p_holm']=running
        out=a.analysis_output.resolve() if a.analysis_output else root/'analysis';out.mkdir(parents=True,exist_ok=True)
        for name,rs in [('summary.csv',summary),('hypothesis_tests.csv',tests),('per_case.csv',detail),('coverage.csv',coverage),('geometry.csv',geometry_stats),('random_control_geometry.csv',controls)]:
            with (out/name).open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
        audit['elapsed_seconds']=time.perf_counter()-start;write(out/'audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ['file_hashes']}),flush=True)


if __name__=='__main__':main()
