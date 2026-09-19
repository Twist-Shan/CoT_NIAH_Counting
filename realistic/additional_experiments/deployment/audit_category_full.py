import json, hashlib, csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1] / 'runs/category_full_20260906'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
cases = {c['case_id']: c for c in map(json.loads, (ROOT/'frozen/cases.jsonl').open(encoding='utf-8'))}
prompts = {(p['case_id'],p['mode']):p for p in map(json.loads,(ROOT/'frozen/user_prompts.jsonl').open(encoding='utf-8'))}
out=ROOT/'analysis'; out.mkdir(exist_ok=True)
groups=defaultdict(list); rows=[]; hashes=0
for model in ('Qwen3-8B','Gemma4-E4B'):
    base=ROOT/'downloaded/outputs'/model
    assert read(base/'complete.json')['captures']==600
    assert read(base/'contract.json')['manifest_sha256']==digest(ROOT/'frozen/manifest.json')
    for mode in ('nonthinking','native_thinking'):
        cap=base/'captures'/mode
        assert {p.name for p in cap.iterdir()}==set(cases)
        for cid,c in cases.items():
            d=cap/cid
            for name,h in read(d/'complete.json')['files'].items():
                assert digest(d/name)==h, str(d/name)
                hashes+=1
            p=read(d/'prompt.json'); g=read(d/'generation.json')
            assert p['user_text']==prompts[cid,mode]['user_text']
            assert g['correct']==(g['prediction']==c['gold'])
            r=dict(model=model,mode=mode,case_id=cid,seed=c['seed'],category=c['target_category'],count=c['level'],correct=g['correct'],parse_failed=not g['parse_ok'],truncated=g['generation_truncated'])
            rows.append(r)
            for category,count in [('all','all'),(r['category'],'all'),('all',r['count']),(r['category'],r['count'])]: groups[model,mode,category,count].append(r)
with (out/'per_case.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
summary=[]
for (model,mode,category,count),rs in groups.items():
    summary.append(dict(model=model,mode=mode,category=category,count=count,n=len(rs),correct=sum(r['correct'] for r in rs),accuracy=sum(r['correct'] for r in rs)/len(rs),parse_failed=sum(r['parse_failed'] for r in rs),truncated=sum(r['truncated'] for r in rs)))
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(out/'audit.json').write_text(json.dumps(dict(status='PASS',captures=len(rows),file_hashes=hashes,exact_user_prompts=len(rows),archive_sha256=digest(ROOT/'results.tgz')),indent=2),encoding='utf-8')
print(json.dumps([r for r in summary if r['category']=='all' and r['count']=='all'],indent=2))
