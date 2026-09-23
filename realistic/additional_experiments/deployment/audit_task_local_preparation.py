"""Independently check the v3 freeze and its disjoint-first control geometry."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

B = Path(__file__).resolve().parents[1]
VERSION = "task_local_disjoint_first_20260908_v3"

started=time.perf_counter(); root=B/'runs'/VERSION/'package'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
cfg=read(root/'protocol.json'); old=B/'runs/task_local_overlap_20260908_v2/package'
assert cfg['control_policy']=='disjoint_first_minimum_overlap'
for name,h in cfg['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
counts=Counter(); fallback=[]
for path in (root/'banks').glob('*.json'):
    bank=read(path); before=read(old/'banks'/path.name)
    assert bank['ranking']==before['ranking'];counts['unchanged_rankings']+=1
    for k,c in bank['conditions'].items():
        assert c['selected']==before['conditions'][k]['selected'];counts['unchanged_selected_doses']+=1
        selected=set(map(tuple,c['selected'])); selected_counts=Counter(l for l,h in selected)
        for repeat,control in enumerate(c['random']):
            assert len(control)==len(selected)==len(set(map(tuple,control)))
            assert Counter(l for l,h in control)==selected_counts
            for layer,n in selected_counts.items():
                width=cfg['widths'][bank['model']][layer]
                actual=sum((l,h) in selected for l,h in control if l==layer)
                minimum=max(0,2*n-width);assert actual==minimum
                counts['layer_control_checks']+=1
                if minimum and repeat==0:
                    fallback.append(dict(model=bank['model'],task=bank['task'],mode=bank['mode'],k=int(k),layer=layer,width=width,selected=n,overlap=minimum))
            counts['random_banks_checked']+=1
for path in (root/'plans').glob('*.json'):
    assert read(path)==read(old/'plans'/path.name);counts['unchanged_plan_files']+=1
result=dict(status='PASS',version=VERSION,frozen_files=len(cfg['files']),checks=dict(counts),broad_required_overlap=fallback,elapsed_seconds=time.perf_counter()-started)
(root.parent/'preparation_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
