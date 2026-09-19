"""Compact read-only status for the frozen campaign (run on the GPU host)."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

root=Path('outputs/external/lambda_nfs_CoT-Native-thinking-v5_additional_experiments_task_local_disjoint_first_20260908_v3')
launch=json.loads((root/'launch.json').read_text())
stat=Path('/proc')/str(launch['supervisor_pid'])/'stat'
alive=stat.exists() and not stat.read_text().split(') ',1)[1].startswith('Z ')
with (root/'run.log').open('rb') as f:
    f.seek(max(0,(root/'run.log').stat().st_size-4000)); tail=f.read().decode('utf-8',errors='replace')
archive=root/'results.tgz'
complete='TASK_LOCAL_COMPLETE' in tail and archive.is_file()
stages={}
for stage in ['discovery','canary','full']:
    stages[stage]={}
    for model in ['Qwen3-8B','Gemma4-E4B']:
        folder=root/stage/model
        counts=Counter()
        pattern='*/*.json' if stage=='discovery' else '*/*/*/K*.json'
        for p in folder.glob(pattern): counts[p.relative_to(folder).parts[0]]+=1
        stages[stage][model]={'counts':dict(counts),'complete':(folder/'complete.json').is_file()}
status='COMPLETE' if complete else ('RUNNING' if alive else 'STOPPED')
result=dict(status=status,supervisor_pid=launch['supervisor_pid'],elapsed_seconds=time.time()-launch['started_unix'],stages=stages,tail=tail[-1800:])
if complete:
    result['archive_sha256']=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(archive.read_bytes()).hexdigest()
    result['archive_bytes']=archive.stat().st_size
print(json.dumps(result))
