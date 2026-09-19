"""Compare original and lean evaluators in one runtime without relaxing gates."""
import argparse
import json
from pathlib import Path
import torch
from realistic_niah_v4.modeling import load_registered_model, capture_post_block_states
from realistic_niah_v4.prompts import render_v4_prompt
from realistic_niah_v4.spec import V4Config,resolve_model_spec
from realistic_niah_v4.stimuli import load_stimuli
from realistic_niah_v4_4_5.restoration import build_corruption_plan,corrupt_encoding,segment_positions,residual_patch_hook
from scripts.run_realistic_niah_v4_4_5_span_restoration import evaluate as original
from scripts.run_realistic_niah_v4_4_5_reverse_patch import evaluate as lean,compare


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--cache',required=True);args=p.parse_args()
    root=Path(args.root);out=root/'diagnostic_backend';out.mkdir(exist_ok=True)
    base=root.parent/'nonthinking_v445_8gpu_20260813'
    row=next(r for r in load_stimuli(base/'dataset/canonical_run_20260731_v4_numeric_presentation_v3_stimuli.jsonl') if r['design_variant']=='v4.4' and r['seed']==1234 and r['gold_count']==2)
    model,tokenizer,adapter=load_registered_model(resolve_model_spec('Gemma4-E4B'),cache_dir=args.cache,torch_dtype='bfloat16',attention_backend='sdpa')
    def backend():
        return {'root':model.config._attn_implementation,
                'text':model.config.text_config._attn_implementation,
                'first_attention':adapter.attentions[0].config._attn_implementation}
    print('BACKEND_INITIAL',json.dumps(backend()),flush=True)
    enc=render_v4_prompt(row,tokenizer=tokenizer,model_spec=resolve_model_spec('Gemma4-E4B'),config=V4Config.from_json(root/'code/configs/realistic_niah_v4.json'),answer_format='numeric')
    plan=build_corruption_plan(enc);corrupt,_=corrupt_encoding(enc,plan,condition='needle_corrupt')
    config=json.loads((root/'code/configs/realistic_niah_v4_4_5_span_restoration_canonical.json').read_text())
    results={}
    for name,e in [('clean',enc),('needle_corrupt',corrupt)]:
        a=lean(model,tokenizer,adapter,e)
        backend_before=backend()
        b,_=original(model,tokenizer,adapter,e,retrieval_heads=config['retrieval_heads']['Gemma4-E4B'],answer_layers=range(adapter.num_layers),max_new_tokens=8,state_path=out/(name+'.pt'),cache_logit_tolerance=.75,cache_probability_tv_tolerance=.10,audit_cache_equivalence=False,reuse_prefill_for_generation=True)
        backend_after=backend()
        print('BACKEND_TRANSITION',name,json.dumps({'before':backend_before,'after':backend_after}),flush=True)
        c=lean(model,tokenizer,adapter,e)
        _,states=capture_post_block_states(model,adapter,e,segment_positions(plan,condition='needle'),layers=[0,17,41])
        d=lean(model,tokenizer,adapter,e)
        result={'backend_before':backend_before,'backend_after':backend_after,'lean_before':a,'original':b,'lean_after':c,'lean_after_capture':d,'original_vs_lean':compare(a,b),'repeat':compare(a,c),'capture_effect':compare(a,d),'self_patch':{}}
        for layer in [0,17,41]:
            with residual_patch_hook(adapter,e,layer=layer,positions=segment_positions(plan,condition='needle'),replacement=states[layer]) as applications:
                patched=lean(model,tokenizer,adapter,e)
            result['self_patch'][layer]={**compare(d,patched),'hook_applications':applications['count']}
        results[name]=result
        (out/'evaluator_comparison.json').write_text(json.dumps(results,indent=2))
        print(name,json.dumps({k:v for k,v in result.items() if k in ['original_vs_lean','repeat','capture_effect','self_patch']}),flush=True)

if __name__=='__main__':main()
