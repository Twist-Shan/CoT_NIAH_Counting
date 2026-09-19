"""Read only: report generation settings from the referenced main experiments."""
import json
from pathlib import Path

base=Path('runs/v5_native_mechanism_reboot_20260818')
for model,folder in [('Qwen3-8B','head_behavior_shared_k128_confirmation_prompt_balanced_frozenregistry_v2'),
                     ('Gemma4-E4B','head_behavior_shared_k6_confirmation_prompt_balanced_frozenregistry_v3')]:
    root=base/model/folder
    print(model, 'exists',root.is_dir())
    for f in root.glob('*.json'):
        if not any(w in f.name for w in ['plan','manifest','config']):continue
        x=json.loads(f.read_text());values={}
        def walk(y,path=''):
            if isinstance(y,dict):
                for k,v in y.items():
                    if any(w in k for w in ['max_new_tokens','ablation_steps','anchor_role','random','scope']) and isinstance(v,(str,int,float,bool)):
                        values[path+k]=v
                    if isinstance(v,dict):walk(v,path+k+'.')
        walk(x)
        print(json.dumps(dict(file=str(f),values=values)))
