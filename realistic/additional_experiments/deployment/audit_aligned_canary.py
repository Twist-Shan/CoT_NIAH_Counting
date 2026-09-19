import json,sys
from pathlib import Path
root=Path(sys.argv[1])
def read(p): return json.loads(p.read_text())
checks=[]
for model in ['Qwen3-8B','Gemma4-E4B']:
    if not (root/'canary'/model/'complete.json').exists(): raise RuntimeError('Canary not complete: '+model)
    for task in ['kth','category']:
        plans={(p['case_id'],p['mode']):p for p in read(root/'plans'/(task+'_'+model+'.json'))}
        for f in (root/'canary'/model/task/'causal').glob('*.json'):
            r=read(f); p=plans[r['case_id'],r['mode']]; d=Path(p['source']); prompt=read(d/'prompt.json'); g=read(d/'generation.json'); original=prompt['input_ids']+g['generated_token_ids']
            for name,assay in r['assays'].items():
                if assay['status']!='PASS': continue
                clean=next(x for x in assay['arms'] if x['name']=='clean')['generation']['generated_token_ids']; expected=original[assay['prefix_length']:][:len(clean)]
                checks.append(dict(model=model,task=task,case_id=p['case_id'],assay=name,clean_exact=clean==expected))
                for arm in assay['arms']:
                    if arm['name']!='clean': assert arm['generation'].get('hook_calls') or arm['generation'].get('intervention_hook_applications')
result=dict(status='PASS' if all(c['clean_exact'] for c in checks) else 'FAIL',checks=checks)
assert all(any(c['model']==model and c['task']==task and c['assay']=='targeted' for c in checks) for model in ['Qwen3-8B','Gemma4-E4B'] for task in ['kth','category']),'Targeted canary coverage missing'
(root/'canary_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
assert result['status']=='PASS','Clean continuation mismatch'
