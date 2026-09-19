import sys,json,hashlib,time
from pathlib import Path
R=Path(__file__).resolve().parent
sys.path[:0]=[str(R/'src'),str(R/'helpers')]
import torch
from realistic_niah_v4.modeling import load_registered_model,generate_answer_completion,query_attention_rows,_text_config
from realistic_niah_v4.spec import resolve_model_spec
from protocol import encode_ids
from run import summarize_generation
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False));t.replace(p)
old=R.parent/'kth_uniform_20260906_v2'
out=R/'full/Gemma4-E4B'
contract=dict(code={str(p.relative_to(R)):sha(p) for p in R.rglob('*.py')},frozen={p.name:sha(p) for p in (R/'frozen').iterdir() if p.is_file()},backend='sdpa',seed=20260906)
if (out/'contract.json').exists():assert read(out/'contract.json')==contract
else:write(out/'contract.json',contract)
torch.manual_seed(20260906)
model,tok,adapter=load_registered_model(resolve_model_spec('Gemma4-E4B'),cache_dir='outputs/external/lambda_nfs_CoT-Native-thinking-v5_hf_cache')
def backend():
 values=[getattr(c,'_attn_implementation',None) for c in [model.config,_text_config(model)] if c is not None]
 assert values and all(x=='sdpa' for x in values),values
 return values
backend()
cases=[json.loads(x) for x in (R/'frozen/cases.jsonl').open()]
for mode in ['nonthinking','native_thinking']:
 for c in cases:
  d=out/'captures'/mode/c['case_id']; source=old/'full/Gemma4-E4B/captures'/mode/c['case_id']
  if (d/'complete.json').exists():
   for n,h in read(d/'complete.json')['files'].items():assert sha(d/n)==h
   continue
  for n,h in read(source/'complete.json')['files'].items():
   if n in ['prompt.json','generation.json']:assert sha(source/n)==h
  p=read(source/'prompt.json'); enc=encode_ids(p['input_ids']); before=backend()
  if c==cases[0]:
   query_attention_rows(model,adapter,enc);assert backend()==before
  gen=generate_answer_completion(model,tok,enc,max_new_tokens=4096 if mode=='native_thinking' else 64)
  gen.update(summarize_generation(gen,c,mode=mode,prefixed=mode=='nonthinking'));after=backend()
  write(d/'prompt.json',p);write(d/'generation.json',gen)
  write(d/'complete.json',dict(status='PASS',files={n:sha(d/n) for n in ['prompt.json','generation.json']},backend_before=before,backend_after=after,original_prompt_sha256=sha(source/'prompt.json')))
  print('capture',mode,c['case_id'],gen['correct'],flush=True)
assert len(list((out/'captures').glob('*/*/complete.json')))==600
write(out/'complete.json',dict(status='PASS',captures=600,backend=backend()))
print('NATURAL_COMPLETE',flush=True)
