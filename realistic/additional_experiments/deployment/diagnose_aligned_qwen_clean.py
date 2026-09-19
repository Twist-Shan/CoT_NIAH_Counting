import sys,json
from pathlib import Path
base=Path('outputs/external/lambda_nfs_CoT-Native-thinking-v5_additional_experiments'); root=base/'aligned_transfer_20260906_v2'
sys.path[:0]=[str(root/'src'),str(root/'helpers')]
from realistic_niah_v4.modeling import load_registered_model,generate_answer_completion,query_attention_rows
from realistic_niah_v4.spec import resolve_model_spec
from protocol import encode_ids
if '--fixed' in sys.argv:
 from contextlib import contextmanager
 import realistic_niah_v4.modeling as modeling
 @contextmanager
 def restored_backend(model,backend):
  configs=[model.config]; text=modeling._text_config(model)
  if text is not None and text is not model.config: configs.append(text)
  saved=[(c,c._attn_implementation) for c in configs if hasattr(c,'_attn_implementation')]
  try:
   for c,v in saved: c._attn_implementation=backend
   yield
  finally:
   for c,v in saved: c._attn_implementation=v
 modeling._temporary_attention_backend=restored_backend
def read(p):return json.loads(p.read_text())
model,tok,adapter=load_registered_model(resolve_model_spec('Qwen3-8B'),cache_dir='outputs/external/lambda_nfs_CoT-Native-thinking-v5_hf_cache')
out=[]
for cid,mode,assay in [('category_count_seed1261_city7_targetcity','nonthinking','broad'),('category_count_seed1261_city7_targetcity','native_thinking','targeted')]:
 p=next(p for p in read(root/'plans/category_Qwen3-8B.json') if p['case_id']==cid and p['mode']==mode)
 d=Path(p['source']); prompt=read(d/'prompt.json'); original=read(d/'generation.json'); ids=prompt['input_ids']+original['generated_token_ids']; length=p['broad_prefix'] if assay=='broad' else p['target']['prefix_length']
 enc=encode_ids(ids[:length]); gens=[]
 for i in range(3):
  if i==2: query_attention_rows(model,adapter,enc)
  gens.append(generate_answer_completion(model,tok,enc,max_new_tokens=64 if assay=='broad' else 128))
 saved=read(root/'full/Qwen3-8B/category/causal'/f'{mode}_{cid}.json')['assays'][assay]['arms'][0]['generation']
 out.append(dict(case_id=cid,mode=mode,assay=assay,prefix_length=length,prompt_length=len(prompt['input_ids']),fresh=gens,saved=saved,original=original))
 print('completed',mode,flush=True)
(base/('aligned_qwen_clean_diagnostic_fixed.json' if '--fixed' in sys.argv else 'aligned_qwen_clean_diagnostic.json')).write_text(json.dumps(out))
