"""Read-only, compact status for the target-category Broad worker."""
import hashlib
import json
from pathlib import Path
import time

root=Path('outputs/external/lambda_nfs_CoT-Native-thinking-v5_additional_experiments_category_target_broad_20260909_v1')
read=lambda p:json.loads(p.read_text())
if not root.exists():
    print(json.dumps(dict(status='NOT_DEPLOYED')));raise SystemExit()
cfg=read(root/'protocol.json');launch=read(root/'launch.json') if (root/'launch.json').exists() else {}
pid=launch.get('supervisor_pid');proc=Path('/proc')/str(pid)/'stat'
alive=proc.exists() and not proc.read_text().split(') ',1)[1].startswith('Z ')
stages={};points=0
for stage in ['discovery','canary','full']:
    stages[stage]={}
    for model in cfg['models']:
        out=root/stage/model
        count=len(list((out/'rows').glob('*.json'))) if stage=='discovery' else len(list((out/'rows').glob('*/K*.json')))
        expected=200 if stage=='discovery' else 8 if stage=='canary' else 100*len(cfg['sizes'][model])
        stages[stage][model]=dict(completed=count,expected=expected,complete=(out/'complete.json').exists())
        if stage=='full':points+=count
tail=(root/'run.log').read_text(errors='replace')[-2200:] if (root/'run.log').exists() else ''
complete=(root/'analysis/audit.json').exists() and (root/'results.tgz').exists() and 'CATEGORY_TARGET_BROAD_COMPLETE' in tail
result=dict(status='COMPLETE' if complete else 'RUNNING' if alive else 'STOPPED',supervisor_pid=pid,alive=alive,
            full_points=points,expected_full_points=1300,stages=stages,updated_unix=time.time(),tail=tail)
if (root/'canary_audit.json').exists():
    canary=read(root/'canary_audit.json')
    result['canary_audit']=dict(status=canary['status'],checks=canary['checks'])
if complete:result['archive_sha256']=hashlib.sha256((root/'results.tgz').read_bytes()).hexdigest()
print(json.dumps(result))
