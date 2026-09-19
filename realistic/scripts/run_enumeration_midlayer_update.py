"""Freeze L19 before new confirmation and run the unchanged aligned Update kernel."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def rows(path):
    with Path(path).open() as f: return [json.loads(line) for line in f if line.strip()]
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False)+'\n');temp.replace(path)


def choose(ledger,cfg,used):
    lookup={r['seed']:r for r in ledger if r['gold_count']==10}
    if len(lookup)!=sum(r['gold_count']==10 for r in ledger): raise ValueError('Duplicate N10 input')
    candidates=list(range(cfg['candidate_seed_start'],cfg['candidate_seed_stop_exclusive']))
    if any(s not in lookup for s in candidates): raise ValueError('Candidate missing from original registry')
    selected=[s for s in candidates if s not in set(used) and lookup[s]['update_eligible']][:cfg['confirmation_quota']]
    if len(selected)!=cfg['confirmation_quota']: raise ValueError('Insufficient fixed candidates')
    if set(selected)&set(used): raise ValueError('Confirmation overlaps previous selection/interventions')
    return selected


def jobs(cfg,seeds):
    result=[]
    for phase in ['smoke','formal']:
        for k in cfg['donor_k']:
            for direction in cfg['directions']:
                for scope in cfg['scopes']:
                    used=seeds[:1] if phase=='smoke' else seeds
                    result.append(dict(id=f'{phase}/k{k}_{direction}_{scope}',phase=phase,k=k,
                        j=k-1 if direction=='forward' else k+1,direction=direction,scope=scope,seeds=used,expected_rows=3*len(used)))
    return result


def freeze(root,stage,protocol):
    cfg=read(protocol)
    assert cfg['layer_one_based']==19 and cfg['model']=='Qwen3-8B' and cfg['mode']=='enumeration_index'
    assert cfg['donor_k']==[4,6,8] and cfg['directions']==['forward','backward']
    assert cfg['scopes']==['endpoint','four_token_tail','item_span']
    assert cfg['conditions']==['receiver_self','native_donor','donor_to_receiver']
    assert cfg['max_new_tokens']==96 and cfg['prefill_chunk_size']==512
    assert cfg['correctness_filter'] is False
    primary=root/'fresh_n10_update_v1'; previous=read(primary/'manifest.json')
    cell=next(c for c in previous['cells'] if (c['model'],c['mode'])==(cfg['model'],cfg['mode']))
    registry=Path(cell['registry']);registered=read(registry/'manifest.json');ledger=read(registry/'ledger.json')
    assert sha(registry/'ledger.json')==registered['ledger_sha256']
    assert sha(registry/'adapted_generations.jsonl')==registered['adapted_generations_sha256']
    selection=read(primary/'selection_manifest.json')
    chosen=next(c for c in selection['cells'] if (c['model'],c['mode'])==(cfg['model'],cfg['mode']))
    used=set(chosen['confirmation_seeds'])
    old_layer=root/'fresh_qwen_index_n10_layer_diagnostic_v2/manifest.json';used.update(read(old_layer)['selected_seeds'])
    baseline=read(root/'fresh_v1/manifest.json');used.update(baseline['discovery_seeds']);used.update(baseline['read_seeds'])
    for prior in ['fresh_causal_v1/cohorts.json','fresh_native_update_v1/manifest.json']:
        obj=read(root/prior)
        if 'models' in obj: used.update(obj['models'][cfg['model']]['confirmation_seeds'])
        for c in obj.get('cells',[]):
            if c['model']==cfg['model'] and c['mode']==cfg['mode']:used.update(c['selected_seeds'])
    selected=choose(ledger,cfg,used)
    stage.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,stage/Path(__file__).name);shutil.copy2(protocol,stage/'protocol.json')
    (stage/'inputs').mkdir()
    source_rows=[r for r in rows(registry/'adapted_generations.jsonl') if r['seed'] in selected and r['gold_count']==10]
    assert len(source_rows)==10
    with (stage/'inputs/adapted_generations.jsonl').open('x') as f:
        for row in source_rows:f.write(json.dumps(row,ensure_ascii=True)+'\n')
    selected_ledger=[r for r in ledger if r['seed'] in selected and r['gold_count']==10]
    write(stage/'inputs/ledger.json',selected_ledger)
    for row in selected_ledger:
        src=registry/row['geometry_file'];assert sha(src)==row['geometry_sha256'];dst=stage/'inputs'/row['geometry_file'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    all_jobs=jobs(cfg,selected);write(stage/'jobs.json',all_jobs)
    assert sum(j['expected_rows'] for j in all_jobs if j['phase']=='formal')==cfg['formal_rows']==540
    assert sum(j['expected_rows'] for j in all_jobs if j['phase']=='smoke')==cfg['smoke_rows']==54
    source_paths=[root/'fresh_causal_v1/code_manifest.json',root/'fresh_v1/manifest.json',root/'fresh_causal_v1/cohorts.json',
        root/'fresh_native_update_v1/manifest.json',primary/'manifest.json',primary/'selection_manifest.json',old_layer,
        registry/'manifest.json',registry/'ledger.json',registry/'adapted_generations.jsonl']
    runtime=primary/'jobs/formal'/cfg['model']/cfg['mode']/'runtime.json';source_paths.append(runtime)
    manifest=dict(status='FROZEN_BEFORE_NEW_CONFIRMATION',utc=datetime.now(timezone.utc).isoformat(),root=str(root),
        source_code=str(root/'fresh_causal_v1/code'),registry=str(registry),source_runtime=str(runtime),
        cache_dir=str(root.parents[1]/'cache/huggingface'),selected_seeds=selected,previously_used_seeds=sorted(used),
        original_ncc_layer_one_based=chosen['layer_one_based'],new_layer_one_based=19,selection_after_exploration=True,
        selected_using_confirmation_outcomes=False,baseline_generation_reused=True,new_interventions=True,
        source_sha256={str(p):sha(p) for p in source_paths},
        frozen_sha256={p.relative_to(stage).as_posix():sha(p) for p in stage.rglob('*') if p.is_file()})
    write(stage/'manifest.json',manifest);verify(stage)
    print(json.dumps(dict(status=manifest['status'],seeds=selected,manifest_sha256=sha(stage/'manifest.json'),formal_rows=540,smoke_rows=54)),flush=True)


def verify(stage):
    manifest=read(stage/'manifest.json')
    for p,h in manifest['source_sha256'].items():assert sha(p)==h,p
    for p,h in manifest['frozen_sha256'].items():assert sha(stage/p)==h,p
    code=read(Path(manifest['root'])/'fresh_causal_v1/code_manifest.json')
    for p,h in {**code['original_code_sha256'],**code['additive_code_sha256']}.items():assert sha(Path(manifest['source_code'])/p)==h,p
    return manifest


def execute(stage,m,cfg,job,model,tokenizer,adapter,kernel):
    from scripts.enumeration_fresh_geometry import json_sha
    tick=time.monotonic();folder=stage/'jobs'/job['id'];folder.mkdir(parents=True,exist_ok=False)
    layer=cfg['layer_one_based']-1;scope='item_span' if job['scope']=='item_span' else 'fixed_suffix';width=4 if job['scope']=='four_token_tail' else 1
    command=[str(kernel.__file__),'--model',cfg['model'],'--cache-dir',m['cache_dir'],'--device-map','auto','--torch-dtype','bfloat16',
        '--attention-backend','sdpa','--prefill-chunk-size',str(cfg['prefill_chunk_size']),'--generations',str(stage/'inputs/adapted_generations.jsonl'),
        '--cohort-mode','indexed_positive_control','--gold-count','10','--receiver-occurrence',str(job['j']),'--donor-occurrence',str(job['k']),
        '--layers',str(layer),'--conditions',*cfg['conditions'],'--generation-conditions',*cfg['conditions'],'--max-new-tokens','96',
        '--tail-offset','0','--patch-scope',scope,'--patch-width',str(width),'--seeds',*map(str,job['seeds']),'--output',str(folder)]
    write(folder/'command.json',command)
    original_load,original_generate,original_prefill=kernel._experiment_model,kernel.generate_answer_completion_from_prefill,kernel._chunked_prefill_with_span_replacement
    raw,hooks=[],[]
    try:
        kernel._experiment_model=lambda _: (model,tokenizer,adapter)
        with (folder/'raw_generations.jsonl').open('x') as rf,(folder/'prefill_hooks.jsonl').open('x') as hf:
            def generation(*a,**kw):
                result=original_generate(*a,**kw);encoding=a[2]
                record=dict(seed=int(encoding.seed),condition=cfg['conditions'][len(raw)%3],query_position=encoding.query_position,**result)
                raw.append(record);rf.write(json.dumps(record,ensure_ascii=True)+'\n');rf.flush();return result
            def prefill(*a,**kw):
                result=original_prefill(*a,**kw);encoding=a[2]
                record=dict(seed=int(encoding.seed),layer=kw['layer'],site=kw['site'],width=int(kw['states'].shape[0]),
                    applications=int(result[1]),delta_norm=float(result[2]),input_ids_sha256=json_sha(list(encoding.input_ids)))
                assert record['applications']==1 and math.isfinite(record['delta_norm'])
                hooks.append(record);hf.write(json.dumps(record)+'\n');hf.flush();return result
            kernel.generate_answer_completion_from_prefill=generation;kernel._chunked_prefill_with_span_replacement=prefill
            saved=sys.argv
            try:sys.argv=command;kernel.main()
            finally:sys.argv=saved
        trials=rows(folder/'trials.jsonl')
        assert len(trials)==len(raw)==job['expected_rows'] and len(hooks)==2*len(trials)
        assert {(r['seed'],r['condition']) for r in trials}=={(s,c) for s in job['seeds'] for c in cfg['conditions']}
        for i,(t,r) in enumerate(zip(trials,raw)):
            for key in ['seed','condition','completion_text','generated_token_count','generation_truncated']:assert t[key]==r[key],key
            assert t['layer']==layer and t['patch_applications']==1 and t['generated_token_count']>0
            assert math.isfinite(t['donor_vs_receiver_sum_logodds'])
            for h in hooks[2*i:2*i+2]:
                assert h['seed']==t['seed'] and h['layer']==layer and h['site']==t['shared_commit_position'] and h['width']==t['patch_width']
                assert h['delta_norm']>0 if t['condition']=='donor_to_receiver' else h['delta_norm']==0
        result=dict(status='COMPLETE',completed_rows=len(trials),expected_rows=job['expected_rows'],seconds=time.monotonic()-tick,
            behavioral_effect_required=False,sha256={p.name:sha(p) for p in folder.iterdir() if p.is_file()})
        write(folder/'status.json',result);return result
    finally:
        kernel._experiment_model=original_load;kernel.generate_answer_completion_from_prefill=original_generate;kernel._chunked_prefill_with_span_replacement=original_prefill


def run(stage):
    tick=time.monotonic();m=verify(stage);cfg=read(stage/'protocol.json');state=dict(status='RUNNING',phase='LOAD_MODEL',pid=os.getpid(),completed_rows=0,jobs=[])
    def save():state['seconds']=time.monotonic()-tick;write(stage/'status.json',state)
    save();sys.path[:0]=[m['source_code']+'/src',m['source_code']]
    try:
        import torch
        from realistic_niah_v4.modeling import load_registered_model
        from realistic_niah_v4.spec import resolve_model_spec
        from realistic_niah_v6.kernel import install_v6_kernel_adapters,install_v6_specialized_geometry
        install_v6_kernel_adapters();install_v6_specialized_geometry(cfg['mode'])
        from scripts import run_realistic_niah_v5_natural_aligned_progress_transplant as kernel
        model,tokenizer,adapter=load_registered_model(resolve_model_spec(cfg['model']),cache_dir=m['cache_dir'],device_map='auto',torch_dtype='bfloat16',attention_backend='sdpa');model.eval()
        runtime=dict(model_revision=resolve_model_spec(cfg['model']).revision,model_source_sha256=sha(sys.modules[type(model).__module__].__file__),
            dtype=str(next(model.parameters()).dtype),backend='sdpa',layers=adapter.num_layers,torch=torch.__version__,transformers=importlib.metadata.version('transformers'),
            gpu=torch.cuda.get_device_name(),python=sys.version,selected_seeds=m['selected_seeds'],layer_one_based=cfg['layer_one_based'],
            selection='Fixed middle layer after exploration; independently evaluated confirmation inputs',manifest_sha256=sha(stage/'manifest.json'))
        previous=read(m['source_runtime'])
        for key in ['model_revision','model_source_sha256','dtype','backend','layers','torch','transformers']:assert runtime[key]==previous[key],key
        write(stage/'runtime.json',runtime)
        for job in read(stage/'jobs.json'):
            state['phase']=job['id'];save();result=execute(stage,m,cfg,job,model,tokenizer,adapter,kernel)
            state['jobs'].append(dict(job,**{k:result[k] for k in ['status','completed_rows','seconds']}));state['completed_rows']+=result['completed_rows'];save()
        state.update(status='COMPLETE',phase='MIDLAYER_GPU_COMPLETE_ANALYSIS_PENDING')
    except BaseException as e:state.update(status='FAILED',error=repr(e));raise
    finally:save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);p.add_argument('--stage',type=Path,required=True);p.add_argument('--root',type=Path);p.add_argument('--protocol',type=Path)
    a=p.parse_args();freeze(a.root,a.stage,a.protocol) if a.action=='freeze' else run(a.stage)
