#!/usr/bin/env python3
"""Measure ordered current-bank prefixes on the frozen common ten traces."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts.run_realistic_niah_v5 import _model
from realistic_niah_v5.causal import run_retrieval_head_behavior_trial


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--device-map',default='auto')
    p.add_argument('--torch-dtype',default='bfloat16')
    p.add_argument('--attention-backend',default='sdpa')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    plan=json.loads((args.input/'plan.json').read_text())
    assert plan['model']==args.model and plan['max_new_tokens']==512
    rows={r['request_id']:r for r in (json.loads(s) for s in (args.input/'generations.jsonl').read_text().split('\n') if s.strip())}
    assert len(rows)==10
    tasks=plan['trials']
    if args.smoke:
        # Replay clean and the full selected bank for the first frozen row.
        tasks=[t for t in tasks if t['seed']==1254 and (t['k']==0 or (t['k']==max(plan['ks']) and t['condition']=='selected_bank'))]
    args.output.mkdir(parents=True,exist_ok=False)
    model,tokenizer,adapter=_model(args)
    started=time.monotonic()
    for i,task in enumerate(tasks):
        begin=time.monotonic()
        result=run_retrieval_head_behavior_trial(model,tokenizer,adapter,rows[task['request_id']],
            heads=task['heads'],condition=task['condition'],anchor_equivalence_id=task['anchors'],
            max_new_tokens=plan['max_new_tokens'],decode_head_ablation_steps=plan['decode_head_ablation_steps'])
        assert result['seed']==task['seed'] and result['gold_count']==task['gold_count']
        result.update({'dose_k':task['k'],'repeat':task['repeat'],'seconds':time.monotonic()-begin,
                       'plan_sha256':hashlib.sha256((args.input/'plan.json').read_bytes()).hexdigest()})
        with (args.output/f'trial_{i:04d}.jsonl').open('x') as f:
            f.write(json.dumps(result)+'\n')
        print(f'dose {i+1}/{len(tasks)} seed={task["seed"]} K={task["k"]} {task["condition"]} seconds={result["seconds"]:.2f}',flush=True)
    (args.output/'complete.json').write_text(json.dumps({'status':'COMPLETE','trials':len(tasks),'seconds':time.monotonic()-started},indent=2))


if __name__=='__main__':
    main()
