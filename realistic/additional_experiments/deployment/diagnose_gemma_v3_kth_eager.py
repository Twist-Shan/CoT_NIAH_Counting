import sys,json
from pathlib import Path
base=Path('outputs/external/lambda_nfs_CoT-Native-thinking-v5_additional_experiments'); root=base/'aligned_transfer_20260906_v3'
sys.path[:0]=[str(root/'src'),str(root/'helpers')]
from realistic_niah_v4.modeling import load_registered_model,generate_answer_completion,query_attention_rows
from realistic_niah_v4.spec import resolve_model_spec
from protocol import encode_ids
from realistic_niah_v4.modeling import _temporary_attention_backend
def read(p): return json.loads(p.read_text())
model,tok,adapter=load_registered_model(resolve_model_spec('Gemma4-E4B'),cache_dir='outputs/external/lambda_nfs_CoT-Native-thinking-v5_hf_cache')
out=[]
for cid in ['kth_needle_seed1257_level8','kth_needle_seed1258_level9','kth_needle_seed1260_level10','kth_needle_seed1261_level10']:
 p=next(p for p in read(root/'plans/kth_Gemma4-E4B.json') if p['case_id']==cid and p['mode']=='nonthinking')
 d=Path(p['source']); prompt=read(d/'prompt.json'); original=read(d/'generation.json'); ids=prompt['input_ids']+original['generated_token_ids']; length=p['broad_prefix']; enc=encode_ids(ids[:length]); gens=[]
 for i in range(3):
  if i==2: query_attention_rows(model,adapter,enc)
  with _temporary_attention_backend(model, "eager"):
   gens.append(generate_answer_completion(model,tok,enc,max_new_tokens=64))
 saved=read(root/'full/Gemma4-E4B/kth/causal'/f'nonthinking_{cid}.json')['assays']['broad']['arms'][0]['generation']
 out.append(dict(case_id=cid,prefix_length=length,prompt_length=len(prompt['input_ids']),fresh=gens,saved=saved,original=original))
 (base/'aligned_gemma_v3_kth_eager_diagnostic.json').write_text(json.dumps(out))
 print('completed',cid,flush=True)
