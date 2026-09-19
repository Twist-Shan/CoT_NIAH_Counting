#!/usr/bin/env python3
"""Free continuation with a frozen clean-carrier intervention schedule."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'src')]
from scripts.run_realistic_niah_v5 import _model
from realistic_niah_v5.causal import mechanism_continuations, run_retrieval_head_behavior_trial
from realistic_niah_v5.causal_sites import compile_causal_site_plan
from realistic_niah_v5.count_stream import build_answer_source_registry
from realistic_niah_v5.free_carrier_recovery import match_vector_norms, scheduled_carrier_clamp
from realistic_niah_v5.integrated_bridge import _capture_states_with_query_head_ablation
from realistic_niah_v5.terminal_token_state import _grammar_timed_geometry_positions, _matched_state_donor_positions


def read(path):
    return [json.loads(s) for s in path.read_text().split('\n') if s.strip()]


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
    dose=json.loads((args.input/'plan.json').read_text())
    assert dose['model']==args.model
    tasks=[t for t in dose['trials'] if t['condition']=='selected_bank' and t['k']==max(dose['ks'])]
    rows={r['request_id']:r for r in read(args.input/'generations.jsonl')}
    assert len(tasks)==len(rows)==10
    if args.smoke:tasks=tasks[:1]
    source_layer=19 if args.model=='Qwen3-8B' else 16
    arms=[('clean',False,0,None),('clean_restore',False,0,'clean'),
          ('local_lesion',True,0,None),('local_restore',True,0,'clean'),('local_matched',True,0,'matched'),
          ('persistent_lesion',True,-1,None),('persistent_restore',True,-1,'clean'),('persistent_matched',True,-1,'matched')]
    args.output.mkdir(parents=True,exist_ok=False)
    protocol={'arms':arms,'source_layer':source_layer,'max_new_tokens':512,
              'donor':'Archived clean trace carrier; oracle state intervention, no forced tokens after query.',
              'schedule':'Frozen absolute clean-trace carrier positions; no outcome-based realignment.',
              'input_plan_sha256':hashlib.sha256((args.input/'plan.json').read_bytes()).hexdigest()}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    model,tokenizer,adapter=_model(args)
    started=time.monotonic()
    count=0
    for task in tasks:
        row=rows[task['request_id']]
        encoding,registry=build_answer_source_registry(row,tokenizer,answer_site_id='answer_query_v3')
        events=compile_causal_site_plan(row,tokenizer)['events']
        assert len(events)==len(registry.trace_items)==task['gold_count']
        geometries,grammar_audit=_grammar_timed_geometry_positions(registry,events[-1])
        carrier=tuple(geometries['marker_core'] if grammar_audit['grammar_timing_stratum']=='rank_after_city' else geometries['grammar_terminal_update'])
        matched=tuple(_matched_state_donor_positions(registry,carrier))
        specifications,_=mechanism_continuations(row,tokenizer,mechanism='retrieval_anchor_localization')
        by_id={s['anchor_equivalence_id']:s for s in specifications}
        branch=max(int(by_id[a]['query_output_token_index']) for a in task['anchors'])
        prefix_length=int(encoding.prompt_token_count)+branch+1
        assert min(carrier)>=prefix_length and len(carrier)==len(matched)
        assert not set(carrier)&set(matched)
        layers=tuple(range(source_layer,int(adapter.num_layers)-1))
        positions=tuple(sorted(set(carrier)|set(matched)))
        states,_=_capture_states_with_query_head_ablation(model,adapter,encoding,
            capture_positions=positions,capture_layers=layers,heads=[],hook_positions=[prefix_length-1])
        index={pos:i for i,pos in enumerate(positions)}
        clean_states={l:states[l][[index[pos] for pos in carrier]].clone() for l in layers}
        controls={l:match_vector_norms(states[l][[index[pos] for pos in matched]],clean_states[l]) for l in layers}
        for name,lesion,decode_steps,restore in arms:
            begin=time.monotonic()
            replacements=clean_states if restore=='clean' else controls
            context=(scheduled_carrier_clamp(adapter,prefix_length=prefix_length,positions=carrier,replacements=replacements)
                     if restore else nullcontext({}))
            with context as audit:
                result=run_retrieval_head_behavior_trial(model,tokenizer,adapter,row,
                    heads=task['heads'] if lesion else [],condition=name,anchor_equivalence_id=task['anchors'],
                    max_new_tokens=512,decode_head_ablation_steps=decode_steps)
            generated=result['generated_token_ids']
            comparisons=[]
            for pos in carrier:
                offset=pos-prefix_length
                actual=generated[offset] if offset<len(generated) else None
                expected=int(encoding.input_ids[pos])
                comparisons.append({'position':pos,'clean_token_id':expected,'generated_token_id':actual,'same_token':actual==expected})
            result.update({'carrier_audit':audit,'carrier_positions':list(carrier),'matched_positions':list(matched),
                           'source_layer':source_layer,'patch_layers':list(layers),'grammar_audit':grammar_audit,
                           'carrier_generated_token_comparison':comparisons,'forced_trace_tokens_after_query':False,
                           'oracle_clean_state_donor':True,'seconds':time.monotonic()-begin})
            with (args.output/f'seed{task["seed"]}_{name}.jsonl').open('x') as f:f.write(json.dumps(result)+'\n')
            count+=1
            print(f'recovery {count}/{8*len(tasks)} seed={task["seed"]} arm={name} seconds={result["seconds"]:.2f}',flush=True)
    (args.output/'complete.json').write_text(json.dumps({'status':'COMPLETE','trials':count,'seconds':time.monotonic()-started},indent=2))


if __name__=='__main__':
    main()
