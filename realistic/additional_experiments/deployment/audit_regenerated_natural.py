import json,hashlib
from pathlib import Path
b=Path(__file__).resolve().parents[1];r=b/'runs/gemma_kth_sdpa_20260907_v1/downloaded';old=b/'runs/kth_uniform_20260906_v2/downloaded'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cases=[json.loads(x) for x in (r/'frozen/cases.jsonl').open(encoding='utf-8')];summary={};checks=0
for mode in ['nonthinking','native_thinking']:
 fs=list((r/'full/Gemma4-E4B/captures'/mode).glob('*/complete.json'));assert len(fs)==300
 assert {f.parent.name for f in fs}=={c['case_id'] for c in cases}
 correct=truncated=0
 for f in fs:
  x=read(f)
  for n,h in x['files'].items():assert sha(f.parent/n)==h;checks+=1
  assert all(v=='sdpa' for k in ['backend_before','backend_after'] for v in x[k])
  original=old/'full/Gemma4-E4B/captures'/mode/f.parent.name/'prompt.json'
  assert sha(original)==x['original_prompt_sha256']
  assert read(original)==read(f.parent/'prompt.json')
  g=read(f.parent/'generation.json');correct+=bool(g['correct']);truncated+=bool(g['generation_truncated'])
 summary[mode]=dict(n=300,correct=correct,accuracy=correct/300,truncated=truncated)
out=dict(status='PASS',capture_hashes=checks,identical_original_prompts=600,backend_checked_captures=600,summary=summary)
(r.parent/'natural_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out))
