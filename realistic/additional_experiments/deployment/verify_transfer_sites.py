"""Independently decode every proposed intervention prefix from original IDs."""
import json, argparse, hashlib
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sites(x):
    if isinstance(x,dict):
        if 'prefix_length' in x and ('char_start' in x or 'char_end' in x): yield x
        for v in x.values(): yield from sites(v)
    elif isinstance(x,list):
        for v in x: yield from sites(v)

p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--base',type=Path,required=True); p.add_argument('--cache',required=True); a=p.parse_args()
summary={}; failures=[]
for model,mid,rev in [('Qwen3-8B','Qwen/Qwen3-8B','b968826d9c46dd6066d109eabc6255188de91218'),('Gemma4-E4B','google/gemma-4-E4B-it','ee0ef6023621cff504d758262d4e04895a5af4a2')]:
    tok=AutoTokenizer.from_pretrained(mid,revision=rev,cache_dir=a.cache,local_files_only=True)
    for task,folder,sub in [('kth','kth_uniform_20260906_v2','full'),('category','category_full_20260906','outputs')]:
        counts=Counter()
        for r in map(json.loads,(a.root/(task+'_'+model+'.jsonl')).open(encoding='utf-8')):
            counts['compiled']+=1
            if r['mode']!='native_thinking': continue
            d=a.base/folder/sub/model/'captures/native_thinking'/r['case_id']; g=read(d/'generation.json'); p=read(d/'prompt.json')
            raw=g['completion_text_raw']; ids=g['generated_token_ids']; n=len(p['input_ids']); unique={}
            for s in sites(r):
                j=s['prefix_length']-n; end=s.get('char_end',s.get('char_start')); key=(j,end,s['alignment'])
                if key in unique: continue
                unique[key]=True; counts['unique_sites_checked']+=1
                decoded=tok.decode(ids[:j],skip_special_tokens=False,clean_up_tokenization_spaces=False)
                valid=decoded==raw[:end]
                if s['alignment']=='leading_whitespace_only': valid=raw[:end].startswith(decoded) and raw[len(decoded):end].isspace()
                if not (valid and s['query_position']==s['prefix_length']-1 and 0<=j<=len(ids)):
                    failures.append(dict(model=model,task=task,case_id=r['case_id'],site=s,decoded_tail=decoded[-30:]))
            counts['answer_query_available']+='query_position' in r['answer_query']
            if 'unavailable_reason' in r['answer_query']: counts['answer_'+r['answer_query']['unavailable_reason']]+=1
            counts['broad_record_and_answer_available']+=bool(r['trace_record_sites']) and 'query_position' in r['answer_query']
            if task=='kth':
                counts['target_available']+='query_position' in r['target_query']
                if 'unavailable_reason' in r['target_query']: counts['target_'+r['target_query']['unavailable_reason']]+=1
                if r['split']=='confirmation':
                    counts['confirmation_broad_available']+=bool(r['trace_record_sites']) and 'query_position' in r['answer_query']
                    counts['confirmation_target_available']+='query_position' in r['target_query']
        summary[task+'_'+model]=dict(counts)
result=dict(status='PASS' if not failures else 'FAIL',summary=summary,failures=failures,registry_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.root.glob('*.jsonl')})
(a.root/'prefix_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False),flush=True)
