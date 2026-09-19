import json
from pathlib import Path
from collections import Counter
b=Path(__file__).resolve().parents[1]; root=b/'runs/aligned_transfer_20260906_v3/downloaded'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
a=read(root.parent/'analysis/audit.json'); out=[]
for m in a['clean_mismatches']:
    model,task,mode,assay=m['key']; cid=m['case_id']
    p=next(p for p in read(root/'plans'/f'{task}_{model}.json') if p['case_id']==cid and p['mode']==mode)
    rel=p['source'].split('/additional_experiments/')[1].split('/')
    d=b/'runs'/rel[0]/'downloaded'/Path(*rel[1:]); prompt=read(d/'prompt.json'); orig=read(d/'generation.json')
    r=read(root/'full'/model/task/'causal'/f'{mode}_{cid}.json')['assays'][assay]
    clean=r['arms'][0]; ids=clean['generation']['generated_token_ids']; expected=(prompt['input_ids']+orig['generated_token_ids'])[r['prefix_length']:]
    div=next((i for i,(x,y) in enumerate(zip(ids,expected)) if x!=y),min(len(ids),len(expected)))
    out.append(dict(key=m['key'],case_id=cid,first_divergent_token=div,clean_score=clean['score'],clean_text=clean['generation']['completion_text_raw'],original_tail=orig['completion_text_raw'][p['target']['query_char_end']:] if assay=='targeted' else orig['completion_text_raw'][-300:]))
(root.parent/'analysis/clean_review.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(Counter(tuple(r['key']) for r in out))
for key in sorted({tuple(r['key']) for r in out}):
    rs=[r for r in out if tuple(r['key'])==key]; print(key,'min divergence',min(r['first_divergent_token'] for r in rs)); print(json.dumps(rs[0],ensure_ascii=False)[:2000])
