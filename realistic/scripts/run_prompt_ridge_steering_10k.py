"""Run or benchmark discovery-fitted, cached 10k prompt-span ridge steering."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
import transformers

from realistic_niah_v4.behavior import parse_numeric_completion
from realistic_niah_v4.modeling import (load_registered_model, _text_config,
    capture_post_block_states, generate_answer_completion)
from realistic_niah_v4.prompts import render_v4_prompt
from realistic_niah_v4.spec import V4Config, resolve_model_spec
from realistic_niah_v4.stimuli import load_stimuli
from realistic_niah_v4.prompt_ridge_steering import fit_direction, additive_span_hook
from realistic_niah_v4_4_3.interventions import capture_query_bundle, candidate_sequence_metrics
from realistic_niah_v4_4_5.restoration import generate_answer_completion_from_prefill

STIMULUS_SHA = 'da4dd86142eb8a07f9a7e53497efd3375184c8e68367d4db994370fcb331f090'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def append(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(value,allow_nan=False)+'\n')
        f.flush()
        os.fsync(f.fileno())


def ints(text):
    return [int(x) for x in text.split(',')]


@torch.inference_mode()
def evaluate(model,tokenizer,adapter,encoding):
    torch.cuda.synchronize()
    start=time.perf_counter()
    bundle=capture_query_bundle(model,adapter,encoding,layers=[0],capture_attention=False,
        capture_values=False,audit_cache_equivalence=False,retain_prefill_output=True)
    result=generate_answer_completion_from_prefill(model,tokenizer,encoding,
        bundle.reusable_prefill_output,max_new_tokens=8)
    metrics=candidate_sequence_metrics(bundle.candidate_log_scores,encoding)
    parsed=parse_numeric_completion(result['completion_text'])
    torch.cuda.synchronize()
    return {**metrics,**result,**parsed,'elapsed_seconds':time.perf_counter()-start,
        'strict_correct':parsed['parsed_count']==encoding.count and not result['generation_truncated']}


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','stimuli','config','cache-dir','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--layers',required=True)
    p.add_argument('--pca-components',type=int,choices=(16,32),default=16)
    p.add_argument('--mode',choices=('benchmark','formal','n3_full'),default='benchmark')
    p.add_argument('--supplementary-stimuli')
    p.add_argument('--seeds',default='1254,1255,1256,1257,1258,1259,1260,1261,1262,1263')
    p.add_argument('--betas',default='-2,-1,-0.5,0.5,1,2')
    p.add_argument('--backend',choices=('sdpa','eager'),default='sdpa')
    args=p.parse_args()
    started=time.perf_counter()
    out=Path(args.output)/args.model
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'worker.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=V4Config.from_json(args.config)
    if sha(args.stimuli)!=STIMULUS_SHA: raise RuntimeError('Canonical stimulus hash mismatch')
    spec=resolve_model_spec(args.model)
    layers=list(range(36 if args.model=='Qwen3-8B' else 42)) if args.layers=='all' else ints(args.layers)
    seeds=[1234] if args.mode=='benchmark' else ints(args.seeds)
    counts=[2,10] if args.mode=='benchmark' else ([3] if args.mode=='n3_full' else list(range(1,11)))
    betas=[-1.,1.] if args.mode in ('benchmark','n3_full') else [float(x) for x in args.betas.split(',')]
    if args.mode=='n3_full' and (args.layers!='all' or seeds!=list(range(1254,1264)) or not args.supplementary_stimuli):
        raise ValueError('N3 protocol requires all layers, original ten confirmation seeds and supplementary stimuli')
    if len(set(layers))!=len(layers) or len(set(seeds))!=len(seeds) or len(set(betas))!=len(betas):
        raise ValueError('Duplicate layer/seed/dose')
    if args.mode=='formal' and not set(seeds)<=set(range(1254,1264)):
        raise ValueError('Formal evaluation uses only held-out confirmation seeds')
    contract=dict(model=args.model,model_id=spec.model_id,revision=spec.revision,
        layers=layers,seeds=seeds,counts=counts,betas=betas,mode=args.mode,pca_components=args.pca_components,
        dtype='bfloat16',backend=args.backend,stimuli_sha256=STIMULUS_SHA,config_sha256=sha(args.config),
        runner_sha256=sha(__file__),module_sha256=sha(Path(__file__).parents[1]/'src/realistic_niah_v4/prompt_ridge_steering.py'),
        torch=torch.__version__,transformers=transformers.__version__,python=platform.python_version(),
        train_seeds=list(range(1234,1254)),train_count=10,position_mode='one_span_at_a_time',
        beta_unit='discovery_all_span_projection_population_std',zero_based_post_block=True)
    if args.mode=='n3_full':
        contract.update(position_mode='last_active_span',supplementary_sha256=sha(args.supplementary_stimuli),
            selection_policy='first_10_clean_correct_in_ascending_seed_order; original_1254_to_1263_then_new_1264_to_1283',
            selection_target=10)
    cp=out/'contract.json'
    if cp.exists() and json.loads(cp.read_text())!=contract:
        raise RuntimeError('Resume contract changed; use a new output directory')
    write_json(cp,contract)
    append(out/'events.jsonl',dict(event='start',time=time.time(),pid=os.getpid(),argv=sys.argv))
    model,tokenizer,adapter=load_registered_model(spec,cache_dir=args.cache_dir,
        device_map='auto',torch_dtype='bfloat16',attention_backend='sdpa')
    _text_config(model)._attn_implementation=args.backend
    backends=[a.config._attn_implementation for a in adapter.attentions]
    if any(x!=args.backend for x in backends): raise RuntimeError('Backend was not propagated')
    if any(str(p.device)!='cuda:0' for p in model.parameters()): raise RuntimeError('Model offloaded from assigned GPU')
    if not layers or min(layers)<0 or max(layers)>=adapter.num_layers: raise ValueError('Invalid layers')
    write_json(out/'runtime.json',dict(gpu=torch.cuda.get_device_name(0),backend_layers=backends,
        load_seconds=time.perf_counter()-started,hostname=platform.node(),num_layers=adapter.num_layers))
    stimuli={}
    for row in load_stimuli(args.stimuli):
        if row.get('design_variant')!='v4.4': continue
        key=(int(row['seed']),int(row['gold_count']))
        if key in stimuli: raise RuntimeError('Duplicate stimulus')
        stimuli[key]=row
    if args.mode=='n3_full':
        extra_rows=load_stimuli(args.supplementary_stimuli)
        for row in extra_rows:
            key=(int(row['seed']),int(row['gold_count']))
            if key in stimuli or key[0] not in range(1264,1284) or key[1]!=3 or row['design_variant']!='v4.4' or row['split']!='confirmation':
                raise RuntimeError('Foreign, duplicate, or training row in supplementary data')
            stimuli[key]=row
    def encoding(seed,count):
        return render_v4_prompt(stimuli[seed,count],tokenizer=tokenizer,model_spec=spec,
            config=config,answer_format='numeric')

    selected_baselines={}
    if args.mode=='n3_full':
        screening=out/'selection'/'screening.jsonl'
        locked=out/'selection'/'cohort.json'
        screened={}
        if screening.exists():
            raw=screening.read_bytes()
            if not raw.endswith(b'\n'): raise RuntimeError('Incomplete screening journal')
            for line in raw.decode().splitlines():
                row=json.loads(line)
                if row['seed'] in screened: raise RuntimeError('Duplicate screened seed')
                screened[row['seed']]=row
        candidates=list(range(1254,1264))+sorted(s for s,c in stimuli if s>=1264 and c==3)
        chosen=[]
        for seed in candidates:
            enc=encoding(seed,3)
            token_hash=hashlib.sha256(np.asarray(enc.input_ids,dtype='<i8').tobytes()).hexdigest()
            if seed not in screened:
                result=evaluate(model,tokenizer,adapter,enc)
                row=dict(seed=seed,sequence_length=enc.sequence_length,input_sha256=token_hash,
                    last_span=[enc.needle_spans[-1].start,enc.needle_spans[-1].end],**result)
                append(screening,row); screened[seed]=row
                print(json.dumps(dict(event='clean_screen',model=args.model,seed=seed,
                    correct=row['strict_correct'],prediction=row['parsed_count'])),flush=True)
            row=screened[seed]
            if row['input_sha256']!=token_hash: raise RuntimeError('Screening input changed')
            if row['strict_correct']:
                chosen.append(seed); selected_baselines[seed]=row
            if len(chosen)==10: break
        if len(chosen)!=10: raise RuntimeError('Insufficient clean-correct held-out seeds; no steering launched')
        cohort=dict(model=args.model,seeds=chosen,count=3,selection='clean_correct_only',
            excluded_seeds=[s for s,r in screened.items() if not r['strict_correct']],
            screened_seeds=sorted(screened),screening_sha256=sha(screening),
            train_seeds=list(range(1234,1254)))
        if locked.exists() and json.loads(locked.read_text())!=cohort:
            raise RuntimeError('Frozen cohort changed')
        write_json(locked,cohort)
        seeds=chosen
        # Execute an independent, standard generation to validate reusable-cache readout.
        first=encoding(seeds[0],3)
        standard=generate_answer_completion(model,tokenizer,first,max_new_tokens=8)
        if standard['generated_token_ids']!=selected_baselines[seeds[0]]['generated_token_ids']:
            raise RuntimeError('Selected clean baseline differs from standard generation')
        write_json(out/'selection/cache_equivalence.json',dict(status='PASS',seed=seeds[0],
            cached_tokens=selected_baselines[seeds[0]]['generated_token_ids'],standard_tokens=standard['generated_token_ids']))
        print(json.dumps(dict(event='cohort_frozen',model=args.model,seeds=seeds)),flush=True)

    # One forward captures all requested layers. Only selected span states are stored.
    probes={}
    missing=[]
    for layer in layers:
        file=out/'probes'/f'L{layer:02d}.npz'
        if file.exists():
            audit_path=file.with_suffix('.json')
            if not audit_path.exists() or json.loads(audit_path.read_text())['pca_components'] != args.pca_components:
                raise RuntimeError('Cached probe PCA dimension mismatch; use a new output directory')
            with np.load(file,allow_pickle=False) as data: probes[layer]={k:data[k] for k in data.files}
        else: missing.append(layer)
    if missing:
        train_started=time.perf_counter()
        xs={l:[] for l in missing}; spans={l:[] for l in missing};ys=[];train_rows=[]
        for seed in range(1234,1254):
            enc=encoding(seed,10)
            positions=tuple(p for s in enc.needle_spans for p in range(s.start,s.end))
            ends=[positions.index(s.end-1) for s in enc.needle_spans]
            if len(ends)!=10 or list(positions)!=sorted(set(positions)):
                raise RuntimeError('Unexpected discovery span layout')
            _,bank=capture_post_block_states(model,adapter,enc,positions,layers=missing)
            for layer in missing:
                xs[layer].append(bank[layer][ends].numpy())
                spans[layer].append(bank[layer].numpy())
            ys.extend(range(1,11))
            train_rows.append(dict(seed=seed,count=10,sequence_length=enc.sequence_length,
                input_sha256=hashlib.sha256(np.asarray(enc.input_ids,dtype='<i8').tobytes()).hexdigest()))
            print(json.dumps(dict(event='training_capture',model=args.model,seed=seed,
                elapsed_seconds=time.perf_counter()-train_started)),flush=True)
        for layer in missing:
            probes[layer],audit=fit_direction(np.concatenate(xs[layer]),ys,np.concatenate(spans[layer]),
                pca_components=args.pca_components,
                random_seed=20260910+layer+(0 if args.model=='Qwen3-8B' else 1000))
            file=out/'probes'/f'L{layer:02d}.npz'; file.parent.mkdir(parents=True,exist_ok=True)
            temporary=file.with_name(file.name+'.tmp.npz')
            np.savez(temporary,**probes[layer]); temporary.replace(file)
            write_json(file.with_suffix('.json'),dict(**audit,training_inputs=train_rows,
                training_seconds=time.perf_counter()-train_started))
            print(json.dumps(dict(event='probe_fitted',model=args.model,layer=layer,
                sigma=audit['sigma'],training_r2=audit['training_r2'])),flush=True)
        del xs,spans,bank
    expected=0
    times=[]
    for seed in seeds:
        for count in counts:
            enc=encoding(seed,count)
            journal=out/'prompts'/f'seed{seed}_N{count}.jsonl'
            existing={}
            if journal.exists():
                raw=journal.read_bytes()
                if raw and not raw.endswith(b'\n'): raise RuntimeError('Incomplete journal tail; preserve and repair explicitly')
                for line in raw.decode('utf-8').splitlines():
                    row=json.loads(line); key=(row['layer'],row['span'],row['beta'],row['condition'])
                    if key in existing: raise RuntimeError('Duplicate cell in journal')
                    existing[key]=row
            def record(key, result, audit=None):
                layer,span,beta,condition=key
                row={**result,'model':args.model,'seed':seed,'count':count,'layer':layer,
                    'span':span,'beta':beta,'condition':condition,'sequence_length':enc.sequence_length}
                if audit is not None: row['intervention_audit']=audit
                append(journal,row);existing[key]=row
                times.append(result['elapsed_seconds'])
                print(json.dumps(dict(event='cell',model=args.model,seed=seed,count=count,
                    layer=layer,span=span,beta=beta,condition=condition,
                    elapsed_seconds=result['elapsed_seconds'],expected_count=result['expected_count'],
                    parsed_count=result['parsed_count'])),flush=True)
                return row
            basekey=(-1,-1,0.,'clean')
            base=existing.get(basekey)
            if base is None:
                base=record(basekey,selected_baselines[seed] if args.mode=='n3_full' else evaluate(model,tokenizer,adapter,enc))
            if args.mode=='benchmark':
                standard=generate_answer_completion(model,tokenizer,enc,max_new_tokens=8)
                if standard['generated_token_ids']!=base['generated_token_ids']:
                    raise RuntimeError('Cached evaluator differs from standard greedy generation')
                write_json(out/f'cache_equivalence_N{count}.json',dict(status='PASS',
                    cached_tokens=base['generated_token_ids'],standard_tokens=standard['generated_token_ids']))
            targets=list(enumerate(enc.needle_spans))
            if args.mode=='benchmark': targets=[targets[0],targets[-1]]
            elif args.mode=='n3_full': targets=[targets[-1]]
            expected+=1+len(layers)*(1+len(targets)*len(betas)*2)
            for layer in layers:
                no_key=(layer,-1,0.,'noop')
                if no_key not in existing:
                    s=enc.needle_spans[-1] if args.mode=='n3_full' else enc.needle_spans[0]
                    with additive_span_hook(adapter,enc,layer=layer,positions=range(s.start,s.end),
                        probe=probes[layer],beta=0.,condition='noop') as audit:
                        result=evaluate(model,tokenizer,adapter,enc)
                    if result['generated_token_ids']!=base['generated_token_ids'] or abs(result['expected_count']-base['expected_count'])>1e-5:
                        raise RuntimeError('No-op changed output')
                    record(no_key,result,audit)
                for ordinal,s in targets:
                    for beta in betas:
                        for condition in ('ridge','random'):
                            key=(layer,ordinal,beta,condition)
                            if key in existing: continue
                            with additive_span_hook(adapter,enc,layer=layer,positions=range(s.start,s.end),
                                probe=probes[layer],beta=beta,condition=condition) as audit:
                                result=evaluate(model,tokenizer,adapter,enc)
                            record(key,result,audit)
            allowed={basekey}|{(l,-1,0.,'noop') for l in layers}|{
                (l,i,b,c) for l in layers for i,s in targets for b in betas for c in ('ridge','random')}
            if set(existing)!=allowed: raise RuntimeError('Missing or unexpected result cells')
            write_json(journal.with_suffix('.complete.json'),dict(rows=len(existing),sha256=sha(journal)))
    write_json(out/'complete.json',dict(status='PASS',mode=args.mode,expected_cells=expected,
        run_seconds=time.perf_counter()-started,new_cells=len(times),
        mean_cell_seconds=float(np.mean(times)) if times else None,
        median_cell_seconds=float(np.median(times)) if times else None,
        max_gpu_memory_gb=torch.cuda.max_memory_allocated()/1024**3))
    print((out/'complete.json').read_text(),flush=True)


if __name__=='__main__':
    main()
