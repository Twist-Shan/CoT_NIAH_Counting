"""Aligned clean/corrupted reverse-patch pilot; never modifies historical runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import torch
import transformers

from realistic_niah_v4.modeling import capture_post_block_states, load_registered_model, _text_config
from realistic_niah_v4.prompts import render_v4_prompt
from realistic_niah_v4.spec import V4Config, resolve_model_spec
from realistic_niah_v4.stimuli import load_stimuli
from realistic_niah_v4_4_3.interventions import capture_query_bundle, candidate_sequence_metrics
from realistic_niah_v4_4_5.restoration import (build_corruption_plan, corrupt_encoding,
    segment_positions, residual_patch_hook, generate_answer_completion_from_prefill)
from realistic_niah_v4_4_5.reverse_patch import audit_alignment, damage_metrics
from scripts.run_realistic_niah_v4_4_5_span_restoration import strict_fields, take_states, append_jsonl


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def compare(a, b):
    pa = [float(x) for x in a['candidate_probabilities'].split(',')]
    pb = [float(x) for x in b['candidate_probabilities'].split(',')]
    return {'expected_count_delta': abs(a['expected_count']-b['expected_count']),
            'probability_tv': sum(abs(x-y) for x,y in zip(pa,pb))/2,
            'strict_agrees': a['strict_prediction'] == b['strict_prediction']}


@torch.inference_mode()
def evaluate(model, tokenizer, adapter, encoding):
    torch.cuda.synchronize()
    start = time.perf_counter()
    bundle = capture_query_bundle(model, adapter, encoding, layers=[0],
        capture_attention=False, capture_values=False, audit_cache_equivalence=False,
        retain_prefill_output=True)
    strict = generate_answer_completion_from_prefill(model, tokenizer, encoding,
        bundle.reusable_prefill_output, max_new_tokens=8)
    metrics = candidate_sequence_metrics(bundle.candidate_log_scores, encoding)
    torch.cuda.synchronize()
    return {**metrics, **strict_fields(strict, encoding.count),
            'elapsed_seconds': time.perf_counter()-start}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('model','stimuli','stimuli-config','experiment-config','historical-root','output-dir','cache-dir'):
        parser.add_argument('--'+flag, required=True)
    parser.add_argument('--seeds', default='1234,1235')
    parser.add_argument('--counts', default='2,10')
    parser.add_argument('--layers', required=True)
    parser.add_argument('--effective-text-backend',choices=('sdpa','eager'),default='sdpa',
                        help='Explicit effective backend; use eager only for audited historical Gemma compatibility.')
    args = parser.parse_args()
    seeds, counts, layers = [tuple(int(x) for x in s.split(',')) for s in
                              (args.seeds,args.counts,args.layers)]
    if any(len(v) != len(set(v)) or not v for v in (seeds,counts,layers)):
        raise ValueError('Empty or duplicate selections')
    out = Path(args.output_dir)/args.model
    out.mkdir(parents=True, exist_ok=True)
    if (out/'detail.jsonl').exists() or (out/'run_provenance.json').exists():
        raise RuntimeError('Output already used; choose a new directory')
    experiment = json.loads(Path(args.experiment_config).read_text())
    if sha(args.stimuli) != experiment['stimulus_sha256']:
        raise RuntimeError('Canonical stimulus hash mismatch')
    if not set(seeds).issubset(experiment['discovery_seeds']):
        raise ValueError('This exploratory pilot must use discovery seeds only')
    rows = {(int(r['seed']),int(r['gold_count'])):r for r in load_stimuli(args.stimuli)
            if r.get('design_variant') == 'v4.4' and int(r['seed']) in seeds and int(r['gold_count']) in counts}
    if set(rows) != {(s,c) for s in seeds for c in counts}:
        raise RuntimeError('Missing canonical stimuli')
    history_path = Path(args.historical_root)/args.model/'detail.jsonl'
    history = {}
    for line in history_path.open():
        r = json.loads(line)
        if r['seed'] in seeds and r['gold_count'] in counts:
            key = (r['seed'],r['gold_count'],r['condition'],r['patch_layer'])
            if key in history: raise RuntimeError('Duplicate historical key')
            history[key] = r
    spec = resolve_model_spec(args.model)
    provenance = {'schema_version':'v445_reverse_patch_pilot_v2', 'args':vars(args),
        'seeds':seeds, 'counts':counts, 'layers':layers, 'exploratory':True,
        'model_id':spec.model_id,'model_revision':spec.revision,
        'stimuli_sha256':sha(args.stimuli),'runner_sha256':sha(__file__),
        'experiment_config_sha256':sha(args.experiment_config),
        'torch':torch.__version__,'transformers':transformers.__version__,
        'python':platform.python_version(),'command':sys.argv,
        'dtype':'bfloat16','loader_attention_backend':'sdpa',
        'effective_text_attention_backend':args.effective_text_backend,
        'patch_timing':'zero_based_post_block',
        'scoring':'original candidate sequence sum including termination; 1..10',
        'generation':'greedy from cloned retained full prefill; max_new_tokens=8',
        'omitted_observation':'attention reconstruction and broad-head state dumps',
        'self_gate':{'max_expected_count_delta':1e-5,'max_probability_tv':1e-6,'strict_agreement':True},
        'history_gate':{'max_expected_count_delta':0.05,'max_probability_tv':0.01,'strict_agreement':True}}
    write_json(out/'run_provenance.json',provenance)
    started=time.perf_counter()
    model,tokenizer,adapter=load_registered_model(spec,cache_dir=args.cache_dir,
        device_map='auto',torch_dtype='bfloat16',attention_backend='sdpa')
    text_config=_text_config(model)
    if text_config is None: raise RuntimeError('No text config found')
    text_config._attn_implementation=args.effective_text_backend
    effective=[attention.config._attn_implementation for attention in adapter.attentions]
    if any(value!=args.effective_text_backend for value in effective):
        raise RuntimeError(f'Effective attention backend mismatch: {effective}')
    write_json(out/'effective_backend.json',{'root':model.config._attn_implementation,
               'text':text_config._attn_implementation,'attention_layers':effective})
    if min(layers)<0 or max(layers)>=adapter.num_layers: raise ValueError('Layer out of bounds')
    if any(str(p.device) != 'cuda:0' for p in model.parameters()):
        raise RuntimeError('Pilot requires model entirely on the one GPU')
    print(json.dumps({'event':'model_loaded','seconds':time.perf_counter()-started,
                      'gpu':torch.cuda.get_device_name(0)}),flush=True)
    config=V4Config.from_json(args.stimuli_config)
    row_count=0
    for seed in seeds:
      for count in counts:
        enc=render_v4_prompt(rows[(seed,count)],tokenizer=tokenizer,model_spec=spec,
                            config=config,answer_format='numeric')
        plan=build_corruption_plan(enc)
        needle,_=corrupt_encoding(enc,plan,condition='needle_corrupt')
        ordinary,_=corrupt_encoding(enc,plan,condition='ordinary_corrupt')
        for condition,e in [('needle',needle),('ordinary',ordinary)]:
            audit=audit_alignment(enc,e,plan,condition)
            append_jsonl(out/'alignment.jsonl',{'seed':seed,'gold_count':count,**audit})
        positions={'needle_full':segment_positions(plan,condition='needle'),
                   'needle_endpoint':segment_positions(plan,condition='needle',endpoint_only=True),
                   'ordinary_full':segment_positions(plan,condition='ordinary')}
        capture_positions=tuple(sorted(set(positions['needle_full']+positions['ordinary_full'])))
        base={}; banks={}
        for name,e in [('clean',enc),('needle_corrupt',needle),('ordinary_corrupt',ordinary)]:
            start=time.perf_counter()
            _,banks[name]=capture_post_block_states(model,adapter,e,capture_positions,layers=layers)
            if any(not torch.isfinite(v).all() for v in banks[name].values()):
                raise RuntimeError('Nonfinite captured state')
            capture_seconds=time.perf_counter()-start
            base[name]=evaluate(model,tokenizer,adapter,e)
            old=history[(seed,count,name,-1)]
            if old['sequence_length']!=enc.sequence_length or old['token_budget']!=plan.token_budget:
                raise RuntimeError('Historical length/budget mismatch')
            match=compare(base[name],old)
            append_jsonl(out/'history_alignment.jsonl',{'seed':seed,'gold_count':count,
                         'condition':name,'layer':-1,**match})
            if match['expected_count_delta']>0.05 or match['probability_tv']>0.01 or not match['strict_agrees']:
                raise RuntimeError(f'Historical baseline gate failed: {name} {match}')
            append_jsonl(out/'detail.jsonl',{'seed':seed,'gold_count':count,'condition':name,
                         'patch_layer':-1,'capture_seconds':capture_seconds,**base[name]})
            row_count+=1
        # Self-patch is evaluated first at each layer, before reverse conditions.
        for layer in layers:
          conditions=[('self_needle_full','clean',enc,'needle_full'),
                      ('reverse_needle_full','needle_corrupt',enc,'needle_full'),
                      ('reverse_needle_endpoint','needle_corrupt',enc,'needle_endpoint'),
                      ('reverse_ordinary_full','ordinary_corrupt',enc,'ordinary_full'),
                      ('restore_needle_full','clean',needle,'needle_full')]
          for name,donor,recipient,kind in conditions:
            patch=take_states(banks[donor][layer],capture_positions,positions[kind])
            with residual_patch_hook(adapter,recipient,layer=layer,positions=positions[kind],
                                     replacement=patch) as applications:
                result=evaluate(model,tokenizer,adapter,recipient)
            if applications['count'] != 1:
                raise RuntimeError(f'Expected one full-prefill patch: {applications}')
            if name.startswith('self'):
                match=compare(result,base['clean'])
                if match['expected_count_delta']>1e-5 or match['probability_tv']>1e-6 or not match['strict_agrees']:
                    raise RuntimeError(f'Self-patch gate failed {layer}: {match}')
                append_jsonl(out/'self_patch_audit.jsonl',{'seed':seed,'gold_count':count,'layer':layer,**match})
            if name=='restore_needle_full':
                match=compare(result,history[(seed,count,name,layer)])
                append_jsonl(out/'history_alignment.jsonl',{'seed':seed,'gold_count':count,
                              'condition':name,'layer':layer,**match})
                if match['expected_count_delta']>0.05 or match['probability_tv']>0.01 or not match['strict_agrees']:
                    raise RuntimeError(f'Historical restoration gate failed {layer}: {match}')
            effects={}
            if name.startswith('reverse'):
                corrupt_base=base['ordinary_corrupt' if kind=='ordinary_full' else 'needle_corrupt']
                effects=damage_metrics(base['clean']['expected_count'],corrupt_base['expected_count'],
                                       result['expected_count'],count)
                effects['strict_error_increase']=result['strict_absolute_error']-base['clean']['strict_absolute_error']
            row={'seed':seed,'gold_count':count,'condition':name,'patch_layer':layer,
                 'donor_condition':donor,'recipient_condition':'needle_corrupt' if name.startswith('restore') else 'clean',
                 'patch_positions_sha256':hashlib.sha256(json.dumps(positions[kind]).encode()).hexdigest(),
                 'patch_token_count':len(positions[kind]),'hook_applications':applications['count'],
                 **result,**effects}
            append_jsonl(out/'detail.jsonl',row); row_count+=1
            print(json.dumps({k:row[k] for k in ('seed','gold_count','condition','patch_layer','expected_count','strict_prediction','elapsed_seconds')}),flush=True)
        del banks
    expected=len(seeds)*len(counts)*(3+len(layers)*5)
    if row_count!=expected: raise RuntimeError('Incomplete pilot grid')
    write_json(out/'complete.json',{'status':'PASS','rows':row_count,'expected_rows':expected,
               'elapsed_seconds':time.perf_counter()-started,'exploratory':True})


if __name__=='__main__':
    main()
