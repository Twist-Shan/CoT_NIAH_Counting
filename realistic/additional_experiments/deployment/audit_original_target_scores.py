import sys,json,re
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer
b=Path('outputs/external/lambda_nfs_CoT-Native-thinking-v5_additional_experiments')
sys.path[:0]=[str(b/'aligned_transfer_20260906_v3/src'),str(b/'aligned_transfer_20260906_v3/helpers')]
from realistic_niah_v5.causal import _first_generated_city_record
def read(p):return json.loads(p.read_text())
def score(raw,p):
 pred,start,kind=_first_generated_city_record(raw,[r['city'] for r in p['case']['records']]); hits=[] if pred is None else [(start,pred)]
 for m in re.finditer(r'(?:city|flower) score audit,\s+([^\n,.]+?) received a score',raw,re.I): hits.append((m.start(1),m[1]))
 prediction=min(hits)[1] if hits else None
 return dict(prediction=prediction,correct=prediction is not None and prediction.casefold()==p['target']['target_city'].casefold())
out=[];checks=Counter()
for model,version,hub in [('Qwen3-8B','v2','models--Qwen--Qwen3-8B'),('Gemma4-E4B','v3','models--google--gemma-4-E4B-it')]:
 root=b/f'aligned_transfer_20260906_{version}'
 cache=Path('outputs/external/lambda_nfs_CoT-Native-thinking-v5_hf_cache')/hub/'snapshots'
 tok=AutoTokenizer.from_pretrained(next(cache.iterdir()),local_files_only=True)
 for task in ['kth','category']:
  for p in read(root/'plans'/f'{task}_{model}.json'):
   if p['split']!='confirmation' or p['mode']!='native_thinking' or not p['target']:continue
   f=root/'full'/model/task/'causal'/f"native_thinking_{p['case_id']}.json"
   a=read(f)['assays']['targeted']
   if a['status']!='PASS':continue
   d=Path(p['source']); original=read(d/'prompt.json')['input_ids']+read(d/'generation.json')['generated_token_ids']
   raw=tok.decode(original[a['prefix_length']:a['prefix_length']+128],skip_special_tokens=False)
   origscore=score(raw,p)
   for arm in a['arms']:
    rescored=score(arm['generation']['completion_text_raw'],p)
    assert all(rescored[k]==arm['score'][k] for k in rescored),(model,p['case_id'],arm['name'])
    checks['arms_rescored']+=1
   clean=a['arms'][0]
   out.append(dict(model=model,task=task,case_id=p['case_id'],original=origscore,clean=score(clean['generation']['completion_text_raw'],p),original_text=raw))
result=dict(checks=dict(checks),rows=out)
(b/'aligned_original_target_score_audit.json').write_text(json.dumps(result))
print(json.dumps(dict(checks=dict(checks),cases=len(out),prediction_changes=sum(x['original']['prediction']!=x['clean']['prediction'] for x in out),correctness_changes=sum(x['original']['correct']!=x['clean']['correct'] for x in out))))
