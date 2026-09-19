"""Check every available new bank, including actual capacity-driven overlap."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

R=Path('outputs/external/lambda_nfs_CoT-Native-thinking-v5_additional_experiments_task_local_disjoint_first_20260908_v3')
OLD=R.parent/'task_local_overlap_20260908_v2'
sys.path[:0]=[str(R),str(R/'src')]
from local_selection import rank_discovery, control_audit
read=lambda p:json.loads(p.read_text())
started=time.perf_counter();cfg=read(R/'protocol.json');fallback=[];counts=Counter()
for path in (R/'banks').glob('*.json'):
    b=read(path)
    if b['assay']=='targeted':
        observations=[]
        for rel,h in b['source_hashes'].items():
            f=R/rel;assert hashlib.sha256(f.read_bytes()).hexdigest()==h
            row=read(f)
            if row['heads']:observations.append(row)
        assert rank_discovery(observations)==b['ranking'];counts['target_rankings_reproduced']+=1
        old=read(OLD/'banks'/path.name)
        if [r[:2] for r in old['ranking']]==[r[:2] for r in b['ranking']]:
            counts['target_rank_orders_unchanged_from_v2']+=1
        else:counts['target_rank_orders_changed_after_fresh_discovery']+=1
    for k,c in b['conditions'].items():
        geometry=control_audit(c['selected'],cfg['widths'][b['model']],c['random'])
        assert geometry==c['control_audit'];counts['doses']+=1
        if int(k)==max(b['sizes']):
            saturated=[r for r in geometry[0] if r['fallback_required']]
            fallback.append(dict(model=b['model'],task=b['task'],mode=b['mode'],assay=b['assay'],k=int(k),overlap_per_repeat=sum(r['actual_overlap'] for r in saturated),layers=saturated))
    counts['banks']+=1
result=dict(status='PASS' if counts['banks']==12 else 'PARTIAL',counts=dict(counts),maximum_dose_overlap=fallback,elapsed_seconds=time.perf_counter()-started)
(R/'runtime_bank_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
