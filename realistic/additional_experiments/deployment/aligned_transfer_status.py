import json,time
from pathlib import Path
root=Path('outputs/external/lambda_nfs_CoT-Native-thinking-v5_additional_experiments_aligned_transfer_20260906_v2')
result={'checked_unix':time.time(),'plans_frozen':(root/'plans/manifest.json').exists(),'stages':{},'processes':[]}
for stage in ['canary','full']:
    result['stages'][stage]={}
    for model in ['Qwen3-8B','Gemma4-E4B']:
        d=root/stage/model; files=list(d.glob('*/attention/*.json'))+list(d.glob('*/causal/*.json'))
        result['stages'][stage][model]={'attention':len(list(d.glob('*/attention/*.json'))),'causal':len(list(d.glob('*/causal/*.json'))),'complete':(d/'complete.json').exists(),'seconds_since_result':time.time()-max(p.stat().st_mtime for p in files) if files else None}
for p in Path('/proc').iterdir():
    if p.name.isdigit():
        try:
            cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            if str(root) in cmd and ('run_aligned_transfer.py' in cmd or 'launch.sh' in cmd): result['processes'].append({'pid':int(p.name),'command':cmd})
        except (OSError,UnicodeDecodeError): pass
log=(root/'run.log').read_text(errors='replace') if (root/'run.log').exists() else ''
result.update(traceback='Traceback (most recent call last)' in log,log_tail=log[-2000:],archive=(root/'results.tgz').exists())
print(json.dumps(result))
