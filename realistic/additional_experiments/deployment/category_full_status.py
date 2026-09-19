import json
from pathlib import Path
import time
root=Path('outputs/external/lambda_nfs_CoT-Native-thinking-v5_additional_experiments_category_full_20260906')
status={'checked_unix':time.time(),'models':{},'processes':[]}
for model in ('Qwen3-8B','Gemma4-E4B'):
    r=root/'outputs'/model
    files=list(r.glob('captures/*/*/complete.json'))
    status['models'][model]={'captures':len(files),'nonthinking':sum(p.parents[1].name=='nonthinking' for p in files),'native_thinking':sum(p.parents[1].name=='native_thinking' for p in files),'complete':(r/'complete.json').exists(),'seconds_since_last_result':time.time()-max(p.stat().st_mtime for p in files) if files else None}
for p in Path('/proc').iterdir():
    if not p.name.isdigit(): continue
    try: cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except OSError: continue
    if str(root) in cmd and ('category_full.py' in cmd or 'launch_category_full.sh' in cmd): status['processes'].append({'pid':int(p.name),'command':cmd})
tail=(root/'run.log').read_text(errors='replace')[-1800:] if (root/'run.log').exists() else ''
status.update(archive=(root/'results.tgz').exists(),traceback='Traceback' in tail,log_tail=tail)
print(json.dumps(status,indent=2))
