"""Audit head-count sweep and summarize paired seed-level effects."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze_kth_retrieval import estimate

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--base',type=Path,required=True)
    a=ap.parse_args(); start=time.perf_counter()
    r=a.root/'downloaded/results'; old=a.base/'downloaded/full/Qwen3-8B'
    contract=read(r/'contract.json')
    assert read(r/'complete.json')['status']=='PASS'
    assert contract==read(a.root/'contract.json')
    assert contract['selection_sha256']==sha(old/'frozen_selection.json')
    assert contract['base_contract_sha256']==sha(old/'contract.json')
    assert contract['frozen_manifest_sha256']==sha(a.base/'frozen/manifest.json')
    assert contract['script_sha256']==sha(Path(__file__).with_name('qwen_broad_sweep.py'))
    files=list(r.glob('kth_needle_seed*.json'))
    assert len(files)==100 and {p.stem for p in files}==set(contract['cases'])
    for n,b in contract['banks'].items():
        selected=set(map(tuple,b['heads'])); assert len(selected)==int(n)
        for bank in b['random_heads']:
            assert len(set(map(tuple,bank)))==int(n)
            assert not selected.intersection(map(tuple,bank))
            assert Counter(x[0] for x in bank)==Counter(x[0] for x in selected)
    for k in (1,10):
        cid=f'kth_needle_seed1254_level{k}'
        clean=read(r/f'clean_check_{cid}.json')
        assert clean['generated_token_ids']==read(old/'causal/nonthinking'/f'{cid}.json')['assays']['broad']['arms'][0]['generation']['generated_token_ids']
    values=defaultdict(lambda:defaultdict(list)); byk=defaultdict(list); truncated=Counter(); hashes={}
    for path in files:
        d=read(path); hashes[path.name]=sha(path)
        assert d['case_id']==path.stem and set(d['sizes'])=={'64','128','256'}
        original=read(old/'causal/nonthinking'/path.name)
        assert (d['seed'],d['k'])==(original['seed'],original['k'])
        panels={'32':original['assays']['broad']['arms'],**d['sizes']}
        for n,arms in panels.items():
            assert [x['name'] for x in arms]==['clean','selected','random_0','random_1','random_2']
            assert arms[0]==panels['32'][0]
            if n!='32':
                b=contract['banks'][n]
                assert [x['heads'] for x in arms]==[[],b['heads']]+b['random_heads']
            c={x['name']:int(x['score']['correct']) for x in arms}
            rand=sum(c[f'random_{i}'] for i in range(3))/3
            row={'clean':c['clean'],'selected':c['selected'],'random':rand,'clean_minus_selected':c['clean']-c['selected'],'random_minus_selected':rand-c['selected']}
            for metric,v in row.items(): values[(int(n),metric)][d['seed']].append(v)
            byk[(int(n),d['k'])].append(c['selected'])
            truncated[n]+=sum(x['generation']['generation_truncated'] for x in arms[1:])
    rows=[{'heads':n,'metric':metric,**estimate([sum(v)/len(v) for v in seeds.values()])} for (n,metric),seeds in sorted(values.items())]
    output=a.root/'analysis'; output.mkdir(exist_ok=True)
    with (output/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (output/'audit.json').write_text(json.dumps({'status':'PASS','cases':100,'new_intervention_arms':1200,'clean_checks':2,'truncated_by_size':dict(truncated),'file_hashes':hashes,'elapsed_seconds':time.perf_counter()-start,'script_sha256':sha(Path(__file__))},indent=2),encoding='utf-8')
    (output/'selected_accuracy_by_k.json').write_text(json.dumps([{'heads':n,'k':k,'correct':sum(v),'cases':len(v)} for (n,k),v in sorted(byk.items())],indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))

if __name__=='__main__': main()
