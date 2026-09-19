"""CPU-only verification of every saved Targeted target-token window."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

R=Path('outputs/external/lambda_nfs_CoT-Native-thinking-v5_additional_experiments_task_local_disjoint_first_20260908_v3')
sys.path[:0]=[str(R),str(R/'src')]
os.environ['HF_HUB_OFFLINE']='1'
from task_scoring import target_token_geometry
from realistic_niah_v4.spec import resolve_model_spec
from transformers import AutoTokenizer

started=time.perf_counter();cfg=json.loads((R/'protocol.json').read_text());counts=Counter();examples=[]
for model in cfg['models']:
    spec=resolve_model_spec(model)
    tok=AutoTokenizer.from_pretrained(spec.model_id,revision=spec.revision,cache_dir=cfg['cache'],local_files_only=True)
    for task in cfg['tasks']:
        for p in json.loads((R/'plans'/f'{task}_{model}.json').read_text()):
            if not p['target']:continue
            source=Path(p['source'])
            for name,h in p['source_hashes'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==h
            prompt=json.loads((source/'prompt.json').read_text());g=json.loads((source/'generation.json').read_text())
            meta=target_token_geometry(tok,g['completion_text_raw'],g['generated_token_ids'],p['target']['target_city_start'],p['target']['target_city'],p['target']['prefix_length']-len(prompt['input_ids']))
            counts[model+'/'+task+'/'+p['split']]+=1
            if meta['target_token_offset']==0:counts['offset_zero']+=1
        print(json.dumps(dict(stage='target_token_geometry',model=model,task=task,counts=dict(counts))),flush=True)
result=dict(status='PASS',counts=dict(counts),elapsed_seconds=time.perf_counter()-started)
(R/'target_token_geometry_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
