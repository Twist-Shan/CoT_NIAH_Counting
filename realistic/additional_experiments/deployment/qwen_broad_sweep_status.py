import json
from pathlib import Path
import time
root=Path('outputs/external/lambda_nfs_CoT-Native-thinking-v5_additional_experiments_qwen_broad_sweep_20260906')
files=list((root/'results').glob('kth_needle_seed*.json'))
tail=(root/'run.log').read_text(errors='replace')[-2500:] if (root/'run.log').exists() else ''
processes=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit(): continue
    try: cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except (OSError,PermissionError): continue
    if str(root) in cmd and ('qwen_broad_sweep.py' in cmd or 'launch_qwen_broad_sweep.sh' in cmd):
        processes.append({'pid':int(p.name),'command':cmd})
print(json.dumps({'checked_unix':time.time(),'cases':len(files),'expected_cases':100,'intervention_arms':len(files)*12,'complete':(root/'results/complete.json').exists(),'archive':(root/'results.tgz').exists(),'clean_checks':len(list((root/'results').glob('clean_check_*.json'))),'seconds_since_last_result':time.time()-max(p.stat().st_mtime for p in files) if files else None,'processes':processes,'traceback':'Traceback' in tail,'log_tail':tail},indent=2))
