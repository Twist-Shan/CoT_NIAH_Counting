"""Resume the canonical full-cohort reverse-patch experiment one cell at a time."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import torch
import transformers

from realistic_niah_v4.modeling import load_registered_model, capture_post_block_states, _text_config
from realistic_niah_v4.prompts import render_v4_prompt
from realistic_niah_v4.spec import V4Config, resolve_model_spec
from realistic_niah_v4.stimuli import load_stimuli
from realistic_niah_v4_4_5.restoration import build_corruption_plan, corrupt_encoding, segment_positions, residual_patch_hook
from realistic_niah_v4_4_5.reverse_patch import audit_alignment, damage_metrics
from realistic_niah_v4_4_5.reverse_full import (BASELINES, PATCH_CONDITIONS, expected_cells,
    atomic_json, frozen_contract, load_journal, journal_digest, history_gate, self_gate)
from scripts.run_realistic_niah_v4_4_5_reverse_patch import evaluate, compare, sha
from scripts.run_realistic_niah_v4_4_5_span_restoration import take_states, append_jsonl


def ints(text):
    return [int(x) for x in text.split(',')]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model','stimuli','stimuli-config','full-config','historical-root','output-dir','cache-dir','source-manifest'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--seeds', required=True)
    parser.add_argument('--counts')
    parser.add_argument('--layers')
    args = parser.parse_args()
    design = json.loads(Path(args.full_config).read_text())
    seeds = ints(args.seeds)
    counts = ints(args.counts) if args.counts else design['counts']
    layers = ints(args.layers) if args.layers else list(range(design['num_layers'][args.model]))
    if any(not x or len(x)!=len(set(x)) for x in (seeds,counts,layers)):
        raise ValueError('Empty or duplicate selections')
    if not set(seeds)<=set(design['seeds']) or not set(counts)<=set(design['counts']):
        raise ValueError('Selections outside canonical cohort')
    if min(layers)<0 or max(layers)>=design['num_layers'][args.model]:
        raise ValueError('Invalid layer grid')
    if sha(args.stimuli)!=design['stimulus_sha256']:
        raise RuntimeError('Canonical stimulus hash mismatch')
    validation_layers = [l for l in layers if l in design.get('validation_layers', {}).get(args.model, layers)]
    out = Path(args.output_dir)/args.model
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out/'worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    backend = design['effective_text_backend'][args.model]
    spec = resolve_model_spec(args.model)
    contract = {'schema_version':'v445_reverse_full_run_v1','model':args.model,
        'model_id':spec.model_id,'model_revision':spec.revision,
        'design':design,'counts':counts,'layers':layers,'effective_text_backend':backend,
        'stimuli_sha256':sha(args.stimuli),'stimuli_config_sha256':sha(args.stimuli_config),
        'source_manifest_sha256':sha(args.source_manifest),'runner_sha256':sha(__file__),
        'torch':torch.__version__,'transformers':transformers.__version__,
        'python':platform.python_version(),'dtype':'bfloat16',
        'patch_timing':'zero_based_post_block; one full prefill, no decode patch',
        'self_gate':{'expected_delta':1e-5,'tv':1e-6,'strict_agreement':True},
        'history_gate':{'expected_delta':0.05,'tv':0.01,'strict_agreement':True}}
    frozen_contract(out/'run_contract.json',contract)
    append_jsonl(out/'attempts.jsonl',{'time':time.time(),'command':sys.argv,'seeds':seeds})
    all_cells = expected_cells(layers, validation_layers)
    pending = []
    for seed in seeds:
        for count in counts:
            p=out/'prompts'/f'seed_{seed}_count_{count}.jsonl'
            rows=load_journal(p,seed=seed,count=count,layers=layers,validation_layers=validation_layers)
            marker=p.with_suffix('.complete.json')
            if set(rows)==all_cells and marker.exists():
                if json.loads(marker.read_text())['journal_sha256']!=journal_digest(p):
                    raise RuntimeError('Completed journal changed')
                continue
            pending.append((seed,count))
    if not pending:
        print(json.dumps({'event':'selected_grid_already_complete','seeds':seeds}),flush=True)
        return
    stimuli = {}
    for row in load_stimuli(args.stimuli):
        key=(int(row['seed']),int(row['gold_count']))
        if row.get('design_variant')=='v4.4' and key in set(pending):
            if key in stimuli: raise RuntimeError('Duplicate stimulus')
            stimuli[key]=row
    if set(stimuli)!=set(pending): raise RuntimeError('Missing stimuli')
    history={}
    for line in (Path(args.historical_root)/args.model/'detail.jsonl').open():
        row=json.loads(line)
        if (row['seed'],row['gold_count']) in set(pending):
            key=(row['seed'],row['gold_count'],row['condition'],row['patch_layer'])
            if key in history: raise RuntimeError('Duplicate historical row')
            history[key]=row
    start=time.perf_counter()
    model,tokenizer,adapter=load_registered_model(spec,cache_dir=args.cache_dir,
        torch_dtype='bfloat16',attention_backend='sdpa',device_map='auto')
    text=_text_config(model)
    text._attn_implementation=backend
    if any(a.config._attn_implementation!=backend for a in adapter.attentions):
        raise RuntimeError('Effective backend mismatch')
    if any(str(p.device)!='cuda:0' for p in model.parameters()):
        raise RuntimeError('Model must fit on the assigned GPU')
    append_jsonl(out/'backend_audit.jsonl',{'time':time.time(),'root':model.config._attn_implementation,
        'text':text._attn_implementation,'attention_layers':[a.config._attn_implementation for a in adapter.attentions],
        'gpu':torch.cuda.get_device_name(0),'load_seconds':time.perf_counter()-start})
    config=V4Config.from_json(args.stimuli_config)
    for seed,count in pending:
        prompt_start=time.perf_counter()
        journal=out/'prompts'/f'seed_{seed}_count_{count}.jsonl'
        cells=load_journal(journal,seed=seed,count=count,layers=layers,validation_layers=validation_layers)
        enc=render_v4_prompt(stimuli[seed,count],tokenizer=tokenizer,model_spec=spec,config=config,answer_format='numeric')
        plan=build_corruption_plan(enc)
        needle,_=corrupt_encoding(enc,plan,condition='needle_corrupt')
        ordinary,_=corrupt_encoding(enc,plan,condition='ordinary_corrupt')
        align=[audit_alignment(enc,e,plan,name) for name,e in [('needle',needle),('ordinary',ordinary)]]
        metadata={'seed':seed,'gold_count':count,'alignment':align,'layers':layers,
            'split':'discovery' if seed in design['discovery_seeds'] else 'confirmation',
            'effective_text_backend':backend}
        frozen_contract(journal.with_suffix('.meta.json'),metadata)
        positions={'needle_full':segment_positions(plan,condition='needle'),
            'needle_endpoint':segment_positions(plan,condition='needle',endpoint_only=True),
            'ordinary_full':segment_positions(plan,condition='ordinary')}
        capture_positions=tuple(sorted(set(positions['needle_full']+positions['ordinary_full'])))
        banks={}; base={}; legacy_bank=None

        def historical_check(name,layer,result):
            nonlocal legacy_bank
            old=history[seed,count,name,layer]
            direct=compare(result,old)
            audit={'primary':direct,'status':'DIRECT_PASS' if history_gate(direct) else 'MISMATCH',
                   'historical_expected_count':old['expected_count'],'historical_strict_prediction':old['strict_prediction']}
            if history_gate(direct): return audit
            # This diagnostic never replaces the fixed-backend primary row.
            if args.model=='Gemma4-E4B' and name in ('clean','restore_needle_full'):
                try:
                    text._attn_implementation='sdpa'
                    if name=='clean':
                        reconstruction=evaluate(model,tokenizer,adapter,enc)
                    elif legacy_bank is None:
                        _,legacy_bank=capture_post_block_states(model,adapter,enc,capture_positions,layers=layers)
                finally:
                    text._attn_implementation=backend
                if name=='restore_needle_full':
                    patch=take_states(legacy_bank[layer],capture_positions,positions['needle_full'])
                    with residual_patch_hook(adapter,needle,layer=layer,positions=positions['needle_full'],replacement=patch) as hook:
                        reconstruction=evaluate(model,tokenizer,adapter,needle)
                    if hook['count']!=1: raise RuntimeError('Legacy diagnostic hook mismatch')
                legacy=compare(reconstruction,old)
                audit['legacy_sdpa_reconstruction']=legacy
                if history_gate(legacy):
                    audit['status']='LEGACY_REPRODUCED_PRIMARY_RETAINED'
                    return audit
            atomic_json(out/'failure.json',{'seed':seed,'gold_count':count,'condition':name,
                'layer':layer,'reason':'unexplained_historical_mismatch','audit':audit,'primary_result':result})
            raise RuntimeError(f'Unexplained historical mismatch {seed}/{count}/{name}/{layer}: {audit}')

        for name,e in [('clean',enc),('needle_corrupt',needle),('ordinary_corrupt',ordinary)]:
            capture_start=time.perf_counter()
            _,banks[name]=capture_post_block_states(model,adapter,e,capture_positions,layers=layers)
            if any(not torch.isfinite(v).all() for v in banks[name].values()):
                raise RuntimeError('Nonfinite donor bank')
            capture_seconds=time.perf_counter()-capture_start
            base[name]=evaluate(model,tokenizer,adapter,e)
            old=history[seed,count,name,-1]
            if old['sequence_length']!=enc.sequence_length or old['token_budget']!=plan.token_budget:
                raise RuntimeError('Historical token length/budget mismatch')
            audit=historical_check(name,-1,base[name])
            if (name,-1) in cells:
                if not self_gate(compare(base[name],cells[name,-1])):
                    raise RuntimeError('Resumed baseline differs from checkpoint')
                continue
            row={'model':args.model,'seed':seed,'gold_count':count,'split':metadata['split'],
                'condition':name,'patch_layer':-1,'history_check':audit,'capture_seconds':capture_seconds,
                'cell_audit':'PASS',**base[name]}
            append_jsonl(journal,row);cells[name,-1]=row
        for layer in layers:
            definitions=[('self_needle_full','clean',enc,'needle_full'),
                ('reverse_needle_full','needle_corrupt',enc,'needle_full'),
                ('reverse_needle_endpoint','needle_corrupt',enc,'needle_endpoint'),
                ('reverse_ordinary_full','ordinary_corrupt',enc,'ordinary_full'),
                ('restore_needle_full','clean',needle,'needle_full')]
            for name,donor,recipient,kind in definitions:
                if (name,layer) not in all_cells or (name,layer) in cells: continue
                patch=take_states(banks[donor][layer],capture_positions,positions[kind])
                with residual_patch_hook(adapter,recipient,layer=layer,positions=positions[kind],replacement=patch) as hook:
                    result=evaluate(model,tokenizer,adapter,recipient)
                if hook['count']!=1: raise RuntimeError('Expected one prefill hook')
                audit={}
                if name.startswith('self'):
                    match=compare(result,base['clean'])
                    if not self_gate(match):
                        atomic_json(out/'failure.json',{'reason':'self_patch_failure','seed':seed,'gold_count':count,'layer':layer,'comparison':match})
                        raise RuntimeError(f'Self-patch failed {match}')
                    audit['self_check']=match
                if name=='restore_needle_full':audit['history_check']=historical_check(name,layer,result)
                effects={}
                if name.startswith('reverse'):
                    corrupt_base=base['ordinary_corrupt' if kind=='ordinary_full' else 'needle_corrupt']
                    effects=damage_metrics(base['clean']['expected_count'],corrupt_base['expected_count'],result['expected_count'],count)
                    effects['strict_error_increase']=result['strict_absolute_error']-base['clean']['strict_absolute_error']
                row={'model':args.model,'seed':seed,'gold_count':count,'split':metadata['split'],
                    'condition':name,'patch_layer':layer,'donor_condition':donor,
                    'recipient_condition':'needle_corrupt' if name=='restore_needle_full' else 'clean',
                    'patch_token_count':len(positions[kind]),'hook_applications':hook['count'],
                    'cell_audit':'PASS',**result,**audit,**effects}
                append_jsonl(journal,row);cells[name,layer]=row
                print(json.dumps({k:row[k] for k in ('seed','gold_count','condition','patch_layer','expected_count','elapsed_seconds')}),flush=True)
        if set(cells)!=all_cells:raise RuntimeError('Incomplete prompt grid')
        atomic_json(journal.with_suffix('.complete.json'),{'status':'PASS','seed':seed,'gold_count':count,
            'rows':len(cells),'journal_sha256':journal_digest(journal),'elapsed_seconds_this_attempt':time.perf_counter()-prompt_start})
        atomic_json(out/'progress.json',{'last_completed_seed':seed,'last_completed_count':count,
            'complete_prompts':len(list((out/'prompts').glob('*.complete.json'))),'time':time.time()})
        del banks,legacy_bank
    print(json.dumps({'event':'selected_grid_complete','seeds':seeds,'counts':counts,'layers':layers}),flush=True)


if __name__=='__main__':main()
