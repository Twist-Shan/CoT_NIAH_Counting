import json,time
from pathlib import Path
r=Path('outputs/external/lambda_nfs_CoT-Native-thinking-v5_additional_experiments_gemma_kth_sdpa_20260907_v1')
out=r/'full/Gemma4-E4B'; counts={}; bad=[];times=[]
for mode in ['nonthinking','native_thinking']:
 fs=list((out/'captures'/mode).glob('*/complete.json'));counts[mode]=len(fs)
 for f in fs:
  x=json.loads(f.read_text());times.append(f.stat().st_mtime)
  if any(v!='sdpa' for k in ['backend_before','backend_after'] for v in x[k]):bad.append(str(f))
log=(r/'run.log').read_text();procs=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  if str(r) in cmd and ('regenerate_gemma_kth.py' in cmd or 'launch.sh' in cmd):procs.append(dict(pid=int(p.name),command=cmd))
 except (OSError,UnicodeError):pass
print(json.dumps(dict(checked_unix=time.time(),counts=counts,backend_violations=bad,seconds_since_result=time.time()-max(times) if times else None,complete=(out/'complete.json').exists(),archive=(r/'natural.tgz').exists(),processes=procs,traceback='Traceback' in log,log_tail=log[-1000:])))
