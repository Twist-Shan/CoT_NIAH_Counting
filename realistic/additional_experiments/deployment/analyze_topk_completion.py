"""Audit completed frozen sweep and calculate seed-paired dose curves on GPU host (CPU only)."""
import argparse,csv,hashlib,json,random,re,sys
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
    cfg=read(r/'protocol.json');checks=Counter();values=defaultdict(lambda:defaultdict(list));detail=[];coverage=[];hashes={}
    for name,h in cfg['files'].items():assert sha(r/name)==h;checks['frozen_files']+=1
    for name,h in {**cfg['source_hashes'],**cfg['prior_results']}.items():assert sha(Path(name))==h;checks['prior_files']+=1
    sys.path[:0]=[str(r),str(r/'src'),str(r/'helpers')]
    from run import summarize_generation
    from realistic_niah_v5.causal import _first_generated_city_record
    from run_aligned_transfer import controls
    for model in cfg['sources']:
        assert read(r/'canary'/model/'complete.json')['status']=='PASS'
        assert read(r/'full'/model/'complete.json')['status']=='PASS'
        widths=read(r/'full'/model/'environment.json')['widths']
        for task in ['kth','category']:
            plans=read(r/'plans'/f'{task}_{model}.json');banks=read(r/'banks'/f'{task}_{model}.json')
            assert len(plans)==600
            for p in plans:
                for name,h in p['source_hashes'].items():assert sha(Path(p['source'])/name)==h;checks['natural_source_files']+=1
                if p['split']!='confirmation':continue
                for assay in ['broad','targeted']:
                    if assay=='targeted' and p['mode']=='nonthinking':continue
                    key=p['mode']+'_'+assay;bank=banks[key];d=r/'full'/model/task/key/p['case_id']
                    length=p['broad_prefix'] if assay=='broad' else (p['target'] or {}).get('prefix_length')
                    coverage.append(dict(model=model,task=task,mode=p['mode'],assay=assay,case_id=p['case_id'],seed=p['seed'],available=bool(length)))
                    if not length:
                        assert read(d/'unavailable.json')['reason']==p['unavailable'].get(assay)
                        assert not list(d.glob('K*.json'));checks['unavailable']+=1;continue
                    assert {int(f.stem[1:]) for f in d.glob('K*.json')}==set(bank['sizes'])
                    clean=None
                    for k in bank['sizes']:
                        f=d/f'K{k}.json';x=read(f);hashes[str(f.relative_to(r))]=sha(f)
                        assert x['prefix_length']==length and x['seed']==p['seed'] and x['k']==k
                        assert x['protocol_sha256']==sha(r/'protocol.json') and all(v=='sdpa' for v in x['backend'])
                        arms=x['arms'];assert [z['name'] for z in arms]==['clean','selected','random_0','random_1','random_2']
                        heads=bank['heads'][:k];assert len(set(map(tuple,heads)))==k
                        expected=[[],heads]+[controls(heads,widths,(7000 if assay=='broad' else 6000)+i) for i in range(3)]
                        assert [z['heads'] for z in arms]==[[list(h) for h in hh] for hh in expected]
                        selected=set(map(tuple,heads))
                        for hh in expected[2:]:
                            assert len(set(map(tuple,hh)))==k and not selected.intersection(map(tuple,hh))
                            assert Counter(h[0] for h in hh)==Counter(h[0] for h in heads)
                        thisclean=arms[0]['generation']
                        if clean is None:clean=thisclean
                        else:assert thisclean==clean # Shared cached clean, not comparison with an old natural suffix.
                        for arm in arms:
                            gen=arm['generation']
                            if assay=='broad':correct=summarize_generation(gen,p['case'],mode=p['mode'],prefixed=True)['correct']
                            else:
                                raw=gen['completion_text_raw'];pred,start,_=_first_generated_city_record(raw,[v['city'] for v in p['case']['records']]);hits=[] if pred is None else [(start,pred)]
                                for m in re.finditer(r'(?:city|flower) score audit,\s+([^\n,.]+?) received a score',raw,re.I):hits.append((m.start(1),m[1]))
                                prediction=min(hits)[1] if hits else None
                                correct=prediction is not None and prediction.casefold()==p['target']['target_city'].casefold()
                                assert prediction==arm['score']['prediction']
                            assert bool(correct)==bool(arm['score']['correct']);checks['arms_rescored']+=1
                            if 'reused_from' in arm:
                                old=read(Path(arm['reused_from']))['assays'][assay]['arms']
                                orig=next(v for v in old if v['name']==arm['name'])
                                assert all(arm[n]==orig[n] for n in ['heads','score','generation']);checks['reused_arms']+=1
                            checks['truncated_arms']+=bool(gen.get('generation_truncated'))
                        cleanval=int(arms[0]['score']['correct']);sel=int(arms[1]['score']['correct']);rand=np.mean([int(z['score']['correct']) for z in arms[2:]])
                        row=dict(model=model,task=task,mode=p['mode'],assay=assay,k=k,case_id=p['case_id'],seed=p['seed'],clean=cleanval,selected=sel,random=rand,delta=rand-sel)
                        detail.append(row)
                        for metric in ['clean','selected','random','delta']:values[(model,task,p['mode'],assay,k,metric)][p['seed']].append(row[metric])
                        checks['points']+=1
    expected=sum(sum(v.values()) for v in cfg['expected_points'].values());assert checks['points']==expected
    summary=[]
    for key,seeds in sorted(values.items()):
        assert sorted(seeds)==list(range(1254,1264))
        v=np.array([np.mean(seeds[s]) for s in sorted(seeds)])
        # Identical bootstrap seed makes comparisons across K use the same resampled seed indices.
        bootstrap=np.random.default_rng(20260907).choice(v,(20000,len(v)),replace=True).mean(axis=1)
        lo,hi=np.percentile(bootstrap,[2.5,97.5])
        summary.append(dict(zip(['model','task','mode','assay','k','metric'],key),n=sum(map(len,seeds.values())),seeds=len(v),mean=float(v.mean()),lower=float(lo),upper=float(hi)))
    out=r/'analysis';out.mkdir(exist_ok=True)
    for name,rows in [('summary.csv',summary),('per_case.csv',detail),('coverage.csv',coverage)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'audit.json').write_text(json.dumps(dict(status='PASS',checks=dict(checks),protocol_sha256=sha(r/'protocol.json'),file_hashes=hashes),indent=2))
    print(json.dumps(dict(status='PASS',checks=dict(checks),summary_rows=len(summary))))
if __name__=='__main__':main()
