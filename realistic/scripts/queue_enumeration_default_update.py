"""Queue N10 discovery selection and Update after each GPU's Retrieve work."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import shutil
import sys
import time
from prepare_enumeration_default_seed import ROOT, OLD, MOUNT, MODE, MODELS, sha, read, write
from launch_enumeration_default_seed import CODE, CACHE, run

def rows(path):
    with Path(path).open() as f:return [json.loads(line) for line in f]

def save_rows(path,values):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x') as f:
        for r in values:f.write(json.dumps(r,ensure_ascii=True)+'\n')

def freeze_reserve():
    bundle=ROOT/'update_reserve_bundle'
    bundle.mkdir(exist_ok=False);(bundle/'code').symlink_to(CODE,target_is_directory=True)
    cfg=read(ROOT/'fresh_v1/protocol.json')
    ds=cfg['discovery_reserve'];cs=cfg['confirmation_reserve']
    assert ds==list(range(1264,1324)) and cs==list(range(1324,1384))
    assert not set(ds)&set(cs) and not set(ds+cs)&set(range(1234,1264))
    write(bundle/'protocol.json',cfg)
    sys.path[:0]=[str(CODE/'src'),str(CODE)]
    from dataset_generation.dynamic_niah import TokenizerAdapter
    from realistic_niah_v4.spec import V4Config
    from realistic_niah_v4.stimuli import ControlledFreezeSpec,build_controlled_family
    from scripts.build_realistic_niah_v6_replacement_seed_pool import _anchor_signature
    v4=V4Config(seeds=tuple([1234,1254]+ds+cs),discovery_seeds=tuple([1234]+ds),confirmation_seeds=tuple([1254]+cs))
    v4.validate()
    tok=TokenizerAdapter(v4.canonical_tokenizer,revision=v4.canonical_tokenizer_revision,cache_dir=str(CACHE))
    assert tok.backend=='huggingface',tok.load_error
    data=MOUNT/'code/Realistic_CoT_NiaH_Count_v6_20260828/data'
    spec=ControlledFreezeSpec(config=v4,haystack_dir=str(data/'haystacks/paul_graham'),
        entities_path=str(data/'entities/cities.csv'),fact_templates_path=str(data/'templates/niah_fact_single_template.txt'),tokenizer_cache_dir=str(CACHE))
    anchors={(r['seed'],r['gold_count']):r for r in rows(ROOT/'fresh_v1/stimuli.jsonl') if r['seed'] in (1234,1254)}
    frozen=[]
    for seed in [1234,1254]+ds+cs:
        family,_=build_controlled_family(variant='v4.4',seed=seed,tokenizer=tok,freeze_spec=spec)
        if seed in (1234,1254):
            for r in family:assert _anchor_signature(r)==_anchor_signature(anchors[seed,r['gold_count']])
        else:
            r=next(r for r in family if r['gold_count']==10)
            r['split']='discovery' if seed in ds else 'confirmation'
            r['alignment_roles']=['update_discovery_reserve' if seed in ds else 'update_confirmation_reserve']
            frozen.append(r)
    assert len(frozen)==120
    save_rows(bundle/'stimuli.jsonl',frozen)
    manifest=read(ROOT/'fresh_v1/manifest.json')
    manifest.update(status='FROZEN_UPDATE_RESERVE_ONLY',rows_per_cell=120,baseline_cells=1,baseline_rows=120,
        discovery_seeds=ds,confirmation_candidates=cs,read_seeds=[],stimuli_sha256=sha(bundle/'stimuli.jsonl'),
        protocol_sha256=sha(bundle/'protocol.json'),anchor_seeds_exact_match=[1234,1254])
    write(bundle/'manifest.json',manifest)

def build_combined_bundle(stage,model):
    original=ROOT/'fresh_v1'
    if model==MODELS[0]:return original,ROOT/'fresh_causal_v1/registries'/model/MODE
    reserve=rows(ROOT/'update_reserve/generations.jsonl')
    ds=[r for r in reserve if r['split']=='discovery'];cs=[r for r in reserve if r['split']=='confirmation']
    assert [r['seed'] for r in ds]==list(range(1264,1324))
    assert [r['seed'] for r in cs]==list(range(1324,1384))
    dest=stage/'reserve_baselines'/model/MODE
    save_rows(dest/'generations.jsonl',ds)
    write(dest/'status.json',dict(status='COMPLETE',completed=60,generations_sha256=sha(dest/'generations.jsonl')))
    bundle=stage/'bundle';bundle.mkdir();(bundle/'code').symlink_to(CODE,target_is_directory=True)
    shutil.copy2(original/'protocol.json',bundle/'protocol.json')
    stimuli=rows(original/'stimuli.jsonl')+[r for r in rows(ROOT/'update_reserve_bundle/stimuli.jsonl') if r['split']=='confirmation']
    save_rows(bundle/'stimuli.jsonl',stimuli)
    values=rows(original/'baseline'/model/MODE/'generations.jsonl')+cs
    manifest=read(original/'manifest.json')
    manifest.update(rows_per_cell=360,baseline_cells=1,baseline_rows=360,
        confirmation_candidates=list(range(1254,1264))+list(range(1324,1384)),
        stimuli_sha256=sha(bundle/'stimuli.jsonl'),protocol_sha256=sha(bundle/'protocol.json'))
    write(bundle/'manifest.json',manifest)
    baseline=bundle/'baseline'/model/MODE
    save_rows(baseline/'generations.jsonl',values)
    write(baseline/'status.json',dict(status='COMPLETE',completed=360,generations_sha256=sha(baseline/'generations.jsonl')))
    registry=stage/'registry'
    run(f'update_registry/{model}',[sys.executable,CODE/'scripts/prepare_enumeration_fresh_causal_registry.py',
        '--bundle',bundle,'--model',model,'--mode',MODE,'--cache-dir',CACHE,'--output',registry])
    return bundle,registry

def freeze_stage(stage,model,bundle,registry):
    ledger=read(registry/'ledger.json')
    cfg=read(bundle/'protocol.json')
    confirm=list(range(1254,1264))+(cfg['confirmation_reserve'] if model==MODELS[1] else [])
    eligible={r['seed'] for r in ledger if r['gold_count']==10 and r['update_eligible']}
    selected=[s for s in confirm if s in eligible][:10]
    assert len(selected)==10,'Fixed confirmation reserve insufficient; do not expand'
    default=read(ROOT/'fresh_v1/manifest.json')
    source_runtime=OLD/'fresh_n10_update_v1/cells'/model/MODE/'capture_runtime.json'
    assert source_runtime.exists()
    info=dict(model=model,mode=MODE,registry=str(registry),confirmation_seeds=selected,
              selected_seeds=selected,source_runtime=str(source_runtime))
    manifest=dict(status='FROZEN_BEFORE_N10_DISCOVERY',cache_dir=str(CACHE),cells=[info],
        initial_discovery_seeds=list(range(1234,1254)),all_original_confirmation_candidates=confirm,
        reserve_models=[MODELS[1]] if model==MODELS[1] else [],reserve_seeds=cfg['discovery_reserve'],
        code_sha256=default['code_sha256'],source_sha256={str(registry/'adapted_generations.jsonl'):sha(registry/'adapted_generations.jsonl'),
            str(source_runtime):sha(source_runtime)})
    write(stage/'manifest.json',manifest)
    cohort=dict(baseline_manifest_sha256=sha(bundle/'manifest.json'),
        registries={f'{model}/{MODE}':dict(manifest_sha256=sha(registry/'manifest.json'))},
        models={model:dict(confirmation_seeds=selected,actual_confirmation_n=10)})
    write(stage/'cohorts.json',cohort)

def worker(model,gpu):
    stage=ROOT/'update'/model;stage.mkdir(parents=True,exist_ok=False)
    status=stage/'queue_status.json';state=dict(status='RUNNING',phase='WAIT_RETRIEVE_GPU',gpu=gpu,pid=os.getpid())
    start=time.monotonic()
    def save():state['seconds']=time.monotonic()-start;write(status,state)
    save()
    try:
        while True:
            finished=ROOT/'execution/retrieve/formal'/model/'status.json'
            if finished.exists() and read(finished)['status']=='COMPLETE':break
            for p in (ROOT/'execution/retrieve').glob(f'*/{model}/status.json'):
                if read(p)['status']=='FAILED':raise RuntimeError(f'Retrieve failed: {p}')
            if time.monotonic()-start>86400:raise TimeoutError('Retrieve wait exceeded one day')
            time.sleep(15)
        if model==MODELS[1]:
            state['phase']='RESERVE_BASELINES';save()
            for smoke in (True,False):
                out=ROOT/('update_reserve_smoke' if smoke else 'update_reserve')
                cmd=[sys.executable,CODE/'scripts/run_enumeration_fresh_baseline.py','--bundle',ROOT/'update_reserve_bundle',
                    '--model',model,'--mode',MODE,'--cache-dir',CACHE,'--output',out]+(['--limit','1'] if smoke else [])
                run(f'update_reserve/{"smoke" if smoke else "formal"}',cmd,gpu)
                assert read(out/'status.json')['status']=='COMPLETE'
        bundle,registry=build_combined_bundle(stage,model)
        freeze_stage(stage,model,bundle,registry)
        for phase in ('prepare','capture','analyze'):
            state['phase']=phase;save()
            run(f'update/{phase}/{model}',[sys.executable,CODE/'scripts/run_enumeration_n10_discovery.py',
                '--stage',stage,'--model',model,'--mode',MODE,'--phase',phase],gpu)
        cell=stage/'cells'/model/MODE
        selection=read(cell/'selection.json');cohort=read(cell/'cohort.json')
        record=dict(model=model,mode=MODE,gold_count=10,discovery_states=200,
            discovery_seeds=cohort['discovery_seeds'],confirmation_seeds=cohort['confirmation_seeds'],
            layer_one_based=selection['selected']['layer_one_based'],selection_path=str(cell/'selection.json'),selection_sha256=sha(cell/'selection.json'))
        write(stage/'selection_manifest.json',dict(status='FROZEN_N10_DISCOVERY_LAYERS',confirmation_used_for_selection=False,cells=[record]))
        run(f'update/confirm/{model}',[sys.executable,CODE/'scripts/run_enumeration_n10_discovery.py',
            '--stage',stage,'--model',model,'--mode',MODE,'--phase','confirm'],gpu)
        for smoke in (True,False):
            phase='smoke' if smoke else 'formal';out=stage/phase;state['phase']=phase;save()
            cmd=[sys.executable,CODE/'scripts/run_enumeration_fresh_update.py','--bundle',bundle,
                '--registry',registry,'--cohorts',stage/'cohorts.json','--selection-manifest',stage/'selection_manifest.json',
                '--cache-dir',CACHE,'--output',out]+(['--smoke'] if smoke else [])
            run(f'update/{phase}/{model}',cmd,gpu)
            assert read(out/'status.json')['status']=='COMPLETE'
        state.update(status='COMPLETE',phase='UPDATE_COMPLETE')
    except BaseException as e:state.update(status='FAILED',error=repr(e));raise
    finally:save()

def main():
    freeze_reserve()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,model,gpu) for model,gpu in zip(MODELS,(2,3))]
        for future in futures:future.result()

if __name__=='__main__':main()
