"""Read-only compact status for full-span Native Broad."""
import hashlib
import json
from pathlib import Path
import time

root=Path('outputs/external/lambda_nfs_CoT-Native-thinking-v5_additional_experiments_native_broad_full_span_20260909_v1')
read=lambda p:json.loads(p.read_text())
if not (root/'protocol.json').exists():
    print(json.dumps(dict(status='NOT_DEPLOYED')));raise SystemExit()
cfg=read(root/'protocol.json');launch=read(root/'launch.json') if (root/'launch.json').exists() else {}
pid=launch.get('supervisor_pid');proc=Path('/proc')/str(pid)/'stat'
alive=proc.exists() and not proc.read_text().split(') ',1)[1].startswith('Z ')
stages={};points=0;geo=read(root/'geometry_audit.json') if (root/'geometry_audit.json').exists() else None
for stage in ['geometry','discovery','canary','full']:
    stages[stage]={}
    for model in cfg['models']:
        for task in cfg['tasks']:
            out=root/stage/model/task
            count=len(list(out.glob('*.json'))) if stage in ['geometry','discovery'] else len(list(out.glob('*/K*.json')))
            if stage=='geometry':expected=300
            elif stage=='discovery':expected=200
            elif stage=='canary':expected=4
            elif geo:
                expected=next(r['available'] for r in geo['panels'] if (r['model'],r['task'],r['split'])==(model,task,'confirmation'))*len(cfg['sizes'][model])
            else:expected=cfg['baseline_available'][model][task]*len(cfg['sizes'][model])
            stages[stage][model+'/'+task]=dict(completed=count,expected=expected,complete=(root/stage/model/'complete.json').exists())
            if stage=='full':points+=count
tail=(root/'run.log').read_text(errors='replace')[-2500:] if (root/'run.log').exists() else ''
complete=(root/'analysis/audit.json').exists() and (root/'results.tgz').exists() and 'NATIVE_BROAD_FULL_SPAN_COMPLETE' in tail
result=dict(status='COMPLETE' if complete else 'RUNNING' if alive else 'STOPPED',supervisor_pid=pid,alive=alive,
    full_points=points,expected_full_points=geo['expected_full_points'] if geo else cfg['expected_full_points_upper_bound'],
    denominator_finalized=bool(geo),stages=stages,updated_unix=time.time(),tail=tail)
for stage in ['geometry','canary']:
    p=root/(stage+'_audit.json')
    if p.exists():a=read(p);result[stage+'_audit']={k:v for k,v in a.items() if k in ['status','checks','expected_full_points','panels']}
if complete:result['archive_sha256']=hashlib.sha256((root/'results.tgz').read_bytes()).hexdigest()
print(json.dumps(result))
