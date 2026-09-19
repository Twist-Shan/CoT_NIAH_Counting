"""Verify and summarize the task-local, disjoint-first campaign."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import itertools
from types import FunctionType

import numpy as np


def read(p): return json.loads(p.read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def sequential_sum(values, start=0):
    """Reproduce the frozen GPU Python 3.10 float-sum arithmetic exactly."""
    for value in values:
        start += value
    return start


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, required=True)
    ap.add_argument('--discovery-sum', choices=['builtin', 'python310-sequential'], default='builtin')
    ap.add_argument('--analysis-output', type=Path, help='Write a separate recheck without replacing delivered analysis')
    args=ap.parse_args()
    root=args.root.resolve(); started=time.perf_counter(); cfg=read(root/"protocol.json")
    sys.path[:0]=[str(root),str(root/"src")]
    from local_selection import rank_discovery, select_heads, random_control, control_audit
    runtime_rank_discovery = rank_discovery
    if args.discovery_sum == 'python310-sequential':
        # Keep the hash-locked source unchanged; only reproduce its original
        # interpreter's sum semantics. Full rankings and scores still compare
        # exactly, and the current interpreter must agree on every head's order.
        rank_discovery = FunctionType(rank_discovery.__code__,
            {**rank_discovery.__globals__, 'sum': sequential_sum})
    from run import summarize_generation
    from realistic_niah_v5.causal import _first_generated_city_record
    checks=Counter(); detail=[]; coverage=[]; diagnostics=[]; file_hashes={}; control_rows=[]
    assert cfg['control_policy']=='disjoint_first_minimum_overlap'
    for name,h in cfg["files"].items(): assert sha(root/name)==h; checks["frozen_files"]+=1
    assert cfg["layer_cap"] is False and cfg["old_results_reused"] is False
    for model in cfg["models"]:
        widths=cfg["widths"][model]
        for stage in ["discovery","canary","full"]:
            assert read(root/stage/model/"complete.json")["status"]=="PASS"
        for task in cfg["tasks"]:
            plans=read(root/"plans"/f"{task}_{model}.json")
            assert len(plans)==600 and len({(p['case_id'],p['mode']) for p in plans})==600
            for mode,assay in [("nonthinking","broad"),("native_thinking","broad"),("native_thinking","targeted")]:
                bankpath=root/"banks"/f"{task}_{model}_{mode}_{assay}.json";bank=read(bankpath)
                if assay=="targeted":
                    observations=[]
                    for rel,h in bank["source_hashes"].items():
                        assert sha(root/rel)==h
                        row=read(root/rel)
                        if row['heads']:observations.append(row)
                    assert rank_discovery(observations)==bank['ranking']
                    assert [r[:2] for r in runtime_rank_discovery(observations)] == [r[:2] for r in bank['ranking']]
                    checks['targeted_rankings_reproduced']+=1
                for k in bank['sizes']:
                    selected=select_heads(bank['ranking'],k);conditions=bank['conditions'][str(k)]
                    assert selected==conditions['selected']
                    assert [random_control(selected,widths,(7000 if assay=='broad' else 6000)+i) for i in range(3)]==conditions['random']
                    geometry=control_audit(selected,widths,conditions['random'])
                    assert geometry==conditions['control_audit']
                    for repeat,layers in enumerate(geometry):
                        for layer in layers:control_rows.append(dict(model=model,task=task,mode=mode,assay=assay,k=k,repeat=repeat,**layer))
                    checks['frozen_doses']+=1
                eligible=set()
                for p in plans:
                    if p['mode']!=mode or p['split']!='confirmation':continue
                    assert p['seed'] in cfg['confirmation_seeds']
                    d=root/'full'/model/task/(mode+'_'+assay)/p['case_id']
                    length=p['broad_prefix'] if assay=='broad' else (p['target'] or {}).get('prefix_length')
                    coverage.append(dict(model=model,task=task,mode=mode,assay=assay,seed=p['seed'],case_id=p['case_id'],available=bool(length),reason='' if length else p['unavailable'].get(assay)))
                    if not length:
                        assert read(d/'unavailable.json')['reason']==p['unavailable'].get(assay)
                        assert not list(d.glob('K*.json'));checks['unavailable']+=1;continue
                    eligible.add(p['case_id'])
                    assert {int(f.stem[1:]) for f in d.glob('K*.json')}==set(bank['sizes'])
                    clean=read(d/'clean.json')['arm']
                    for k in bank['sizes']:
                        path=d/f'K{k}.json';x=read(path);arms=x['arms'];values=[]
                        assert x['bank_sha256']==sha(bankpath) and x['protocol_sha256']==sha(root/'protocol.json')
                        assert x['prefix_length']==length and all(v=='sdpa' for v in x['backend'])
                        assert arms[0]==clean
                        assert [a['name'] for a in arms]==['clean','selected','random_0','random_1','random_2']
                        selected=bank['conditions'][str(k)]['selected'];randoms=bank['conditions'][str(k)]['random']
                        assert [a['heads'] for a in arms]==[[],selected]+randoms
                        assert x['overlap_counts']==[len(set(map(tuple,selected)) & set(map(tuple,r))) for r in randoms]
                        assert x['control_audit']==bank['conditions'][str(k)]['control_audit']
                        for a in arms:
                            g=a['generation']
                            if assay=='broad':
                                result=summarize_generation(g,p['case'],mode=mode,prefixed=True)
                                pred,correct=result['prediction'],result['correct']
                            else:
                                raw=g['completion_text_raw'];pred,pos,_=_first_generated_city_record(raw,[r['city'] for r in p['case']['records']]);hits=[] if pred is None else [(pos,pred)]
                                for m in re.finditer(r'(?:city|flower) score audit,\s+([^\n,.]+?) received a score',raw,re.I):hits.append((m.start(1),m[1]))
                                pred=min(hits)[1] if hits else None
                                correct=pred is not None and pred.casefold()==p['target']['target_city'].casefold()
                                geometry=x['target_token_geometry'];target_ids=geometry['target_token_ids']
                                exact_prefix=bool(target_ids and geometry['target_token_offset']==0 and g['generated_token_ids'][:len(target_ids)]==target_ids)
                                assert a['score']['exact_target_prefix']==exact_prefix
                                assert a['score']['token_prefix_fallback_used']==(pred is None and exact_prefix)
                                if pred is None and exact_prefix:correct=True;checks['token_prefix_fallbacks']+=1
                            assert pred==a['score']['prediction'] and bool(correct)==bool(a['score']['correct'])
                            assert 'reused_from' not in a
                            values.append(int(correct));checks['arms_rescored']+=1
                            diagnostics.append(dict(model=model,task=task,mode=mode,assay=assay,k=k,case_id=p['case_id'],condition=a['name'],unparsed=int(pred is None),truncated=int(g['generation_truncated'])))
                        rand=sum(values[2:])/3
                        detail.append(dict(model=model,task=task,mode=mode,assay=assay,k=k,case_id=p['case_id'],seed=p['seed'],clean=values[0],selected=values[1],random=rand,delta=rand-values[1],clean_drop=values[0]-values[1]))
                        file_hashes[str(path.relative_to(root))]=sha(path);checks['points']+=1
                assert len([p for p in plans if p['mode']==mode and p['split']=='confirmation'])==100
    assert checks['points']==cfg['expected_full_points']==6739
    groups=defaultdict(list)
    for row in detail:groups[row['model'],row['task'],row['mode'],row['assay'],row['k']].append(row)
    summary=[];secondary=[];tests=[]
    for key,rows in sorted(groups.items()):
        for metric in ['clean','selected','random','delta','clean_drop']:
            seeds=defaultdict(list)
            for r in rows:seeds[r['seed']].append(r[metric])
            assert sorted(seeds)==cfg['confirmation_seeds']
            v=np.array([np.mean(seeds[s]) for s in sorted(seeds)])
            idx=np.random.default_rng(cfg['stats']['seed']).integers(0,len(v),(cfg['stats']['bootstrap'],len(v)))
            lo,hi=np.quantile(v[idx].mean(axis=1),[.025,.975])
            summary.append(dict(zip(['model','task','mode','assay','k'],key),metric=metric,n=len(rows),seeds=len(v),mean=float(v.mean()),lower=float(lo),upper=float(hi)))
        for population in ['all_examples','clean_correct']:
            chosen=rows if population=='all_examples' else [r for r in rows if r['clean']==1]
            seeds=defaultdict(list)
            for r in chosen:seeds[r['seed']].append(r['delta'])
            if not seeds:
                tests.append(dict(zip(['model','task','mode','assay','k'],key),population=population,n=0,seeds=0,mean=None,lower=None,upper=None,p_exact=None,p_holm=None,interpretation='no clean-correct support'))
                continue
            v=np.array([np.mean(seeds[s]) for s in sorted(seeds)])
            signs=np.array(list(itertools.product([-1,1],repeat=len(v))))
            pvalue=float(np.mean(np.abs((signs*v).mean(axis=1))>=abs(v.mean())-1e-12))
            idx=np.random.default_rng(cfg['stats']['seed']).integers(0,len(v),(cfg['stats']['bootstrap'],len(v)))
            lo,hi=np.quantile(v[idx].mean(axis=1),[.025,.975])
            tests.append(dict(zip(['model','task','mode','assay','k'],key),population=population,n=len(chosen),seeds=len(v),mean=float(v.mean()),lower=float(lo),upper=float(hi),p_exact=pvalue,p_holm=None,interpretation='seed-paired' if len(v)==10 else 'descriptive fewer than 10 supported seeds'))
            if population=='clean_correct':
                for metric in ['clean','selected','random','delta','clean_drop']:
                    byseed=defaultdict(list)
                    for row in chosen:byseed[row['seed']].append(row[metric])
                    values=np.array([np.mean(byseed[s]) for s in sorted(byseed)])
                    low,high=np.quantile(values[idx].mean(axis=1),[.025,.975])
                    secondary.append(dict(zip(['model','task','mode','assay','k'],key),metric=metric,n=len(chosen),seeds=len(values),mean=float(values.mean()),lower=float(low),upper=float(high)))
    for assay in ['broad','targeted']:
        for population in ['all_examples','clean_correct']:
            family=[r for r in tests if r['assay']==assay and r['population']==population and r['p_exact'] is not None]
            family.sort(key=lambda r:r['p_exact']);running=0.
            for i,r in enumerate(family):
                running=max(running,min(1.,(len(family)-i)*r['p_exact']));r['p_holm']=running
    out=args.analysis_output.resolve() if args.analysis_output else root/'analysis';out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('summary.csv',summary),('clean_correct_summary.csv',secondary),('hypothesis_tests.csv',tests),('per_case.csv',detail),('coverage.csv',coverage),('diagnostics.csv',diagnostics),('random_control_geometry.csv',control_rows)]:
        with (out/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    audit=dict(status='PASS',checks=dict(checks),elapsed_seconds=time.perf_counter()-started,protocol_sha256=sha(root/'protocol.json'),file_hashes=file_hashes,
        runtime=dict(python=sys.version,numpy=np.__version__,discovery_sum=args.discovery_sum),analyzer_sha256=sha(Path(__file__)))
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='PASS',checks=dict(checks),elapsed_seconds=audit['elapsed_seconds'])))


if __name__=='__main__':main()
