"""Freeze and queue four Native-style N=10 discovery selections and Update grids."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=True)+'\n',encoding='utf-8');temp.replace(path)


def freeze(args):
    tick=time.monotonic();root=args.root.resolve();stage=args.output.resolve()
    cfg=read(args.protocol);baseline=root/'fresh_v1';causal=root/'fresh_causal_v1';primary=root/'fresh_native_update_v1'
    base=read(baseline/'manifest.json');old=read(primary/'manifest.json')
    history=read(args.history_audit);review=read(args.history_review)
    reserve=list(range(cfg['reserve_first_seed'],cfg['reserve_first_seed']+cfg['reserve_count']))
    assert history['status']=='PASS_UNPACKED_INVENTORY' and not history['conflicts'] and not history['errors']
    assert review['status']=='PASS' and review['history_audit_sha256']==sha(args.history_audit)
    assert history['candidate_seeds']==review['candidate_seeds']==reserve
    assert not set(reserve)&set(base['discovery_seeds']+base['confirmation_candidates'])
    assert read(primary/'gpu_pipeline_status.json')['status']=='COMPLETE'
    old_diagnostic=read(root/'fresh_qwen_index_layer_diagnostic_v1/gpu_pipeline_status.json')
    assert old_diagnostic['status']=='SUPERSEDED' and not old_diagnostic['jobs']
    assert cfg['gold_count']==10 and cfg['discovery_quota']==20 and cfg['confirmation_quota']==10
    assert cfg['update']['formal_rows']==2160 and cfg['update']['smoke_rows']==216
    original_code=read(causal/'code_manifest.json')
    source_code=causal/'code'
    for relative,digest in {**original_code['original_code_sha256'],**original_code['additive_code_sha256']}.items():
        assert sha(source_code/relative)==digest,relative
    probe='src/realistic_niah_v5/trace_stratified_geometry.py'
    assert sha(source_code/probe)==cfg['native_probe_source_sha256']
    stage.mkdir(parents=True,exist_ok=False)
    shutil.copy2(args.protocol,stage/'protocol.json')
    shutil.copy2(__file__,stage/Path(__file__).name)
    code=stage/'code'
    shutil.copytree(source_code,code,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
    extensions=('scripts/run_enumeration_n10_discovery.py','scripts/run_enumeration_fresh_update.py',
                'src/realistic_niah_v6/update_n10.py','tests/test_enumeration_update_n10.py')
    for relative in extensions:
        target=code/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(args.extensions/relative,target)
    # The historical baseline manifest excludes the newly introduced Update wrapper.
    for relative,digest in base['code_sha256'].items():
        assert sha(code/relative)==digest,relative
    source_paths=[baseline/'manifest.json',baseline/'protocol.json',baseline/'stimuli.jsonl',
                  causal/'code_manifest.json',causal/'cohorts.json',primary/'manifest.json',primary/'assembly_audit.json',
                  args.history_audit,args.history_review]
    cells=[]
    for model in cfg['models']:
        for mode in cfg['modes']:
            previous=next(c for c in old['cells'] if c['model']==model and c['mode']==mode)
            registry=Path(previous['registry']);reg=read(registry/'manifest.json')
            assert sha(registry/'manifest.json')==previous['registry_sha256']
            assert sha(registry/'adapted_generations.jsonl')==reg['adapted_generations_sha256']
            assert sha(registry/'ledger.json')==reg['ledger_sha256']
            runtime=primary/'jobs/formal'/model/mode/'runtime.json'
            assert runtime.exists()
            source_paths.extend([registry/'manifest.json',registry/'ledger.json',registry/'adapted_generations.jsonl',runtime])
            cohort=deepcopy(read(causal/'cohorts.json'))
            cohort.update(status='N10_SELECTION_AMENDMENT_REUSED_CONFIRMATION',primary_selected_seeds=previous['selected_seeds'],
                          chronology=cfg['chronology'],selection_used_final_correctness=False)
            cohort['models'][model].update(confirmation_seeds=previous['selected_seeds'],actual_confirmation_n=10)
            path=stage/'cohorts'/model/f'{mode}.json';write(path,cohort)
            ledger=read(registry/'ledger.json')
            d=[r for r in ledger if r['gold_count']==10 and r['seed'] in base['discovery_seeds']]
            assert len(d)==20
            cells.append(dict(model=model,mode=mode,registry=str(registry),registry_sha256=sha(registry/'manifest.json'),
                source_runtime=str(runtime),confirmation_seeds=previous['selected_seeds'],selected_seeds=previous['selected_seeds'],
                original_discovery_format_eligible=sum(r['format_eligible'] for r in d),cohort=str(path),cohort_sha256=sha(path)))
    # Freeze reserve inputs using the same controlled stimulus generator and two old anchors.
    sys.path[:0]=[str(code/'src'),str(code)]
    from dataset_generation.dynamic_niah import TokenizerAdapter
    from realistic_niah_v4.spec import V4Config
    from realistic_niah_v4.stimuli import ControlledFreezeSpec,build_controlled_family
    from scripts.build_realistic_niah_v6_replacement_seed_pool import _anchor_signature
    v4=V4Config(seeds=tuple([1234,1254]+reserve),discovery_seeds=tuple([1234]+reserve),confirmation_seeds=(1254,))
    v4.validate()
    tokenizer=TokenizerAdapter(v4.canonical_tokenizer,revision=v4.canonical_tokenizer_revision,cache_dir=str(args.cache_dir))
    assert tokenizer.backend=='huggingface',tokenizer.load_error
    data=args.data_root
    spec=ControlledFreezeSpec(config=v4,haystack_dir=str(data/'haystacks/paul_graham'),entities_path=str(data/'entities/cities.csv'),
        fact_templates_path=str(data/'templates/niah_fact_single_template.txt'),tokenizer_cache_dir=str(args.cache_dir))
    for path,digest in base['data_sha256'].items():assert sha(path)==digest,path
    assert sha(args.source_stimuli)==base['source_stimuli_sha256']
    anchors={}
    with args.source_stimuli.open() as handle:
        for line in handle:
            row=json.loads(line)
            if row['seed'] in (1234,1254) and row['design_variant']=='v4.4':anchors[row['seed'],row['gold_count']]=row
    for seed in (1234,1254):
        family,_=build_controlled_family(variant='v4.4',seed=seed,tokenizer=tokenizer,freeze_spec=spec)
        for row in family:assert _anchor_signature(row)==_anchor_signature(anchors[seed,row['gold_count']])
    reserve_bundle=stage/'reserve_bundle';reserve_bundle.mkdir()
    shutil.copy2(baseline/'protocol.json',reserve_bundle/'protocol.json')
    families=[]
    with (reserve_bundle/'stimuli.jsonl').open('x',encoding='utf-8') as handle:
        for seed in reserve:
            family,meta=build_controlled_family(variant='v4.4',seed=seed,tokenizer=tokenizer,freeze_spec=spec)
            row=next(r for r in family if r['gold_count']==10)
            row.update(split='discovery',alignment_roles=['n10_discovery_reserve'])
            handle.write(json.dumps(row,ensure_ascii=True)+'\n');families.append(meta)
    reserve_manifest=dict(schema='enumeration_n10_discovery_reserve_v1',status='FROZEN_BEFORE_GENERATION',
        discovery_seeds=reserve,confirmation_candidates=[],rows_per_cell=len(reserve),
        stimuli_sha256=sha(reserve_bundle/'stimuli.jsonl'),protocol_sha256=sha(reserve_bundle/'protocol.json'),
        code_sha256=base['code_sha256'],families=families,anchor_seeds_exact_match=[1234,1254],
        history_audit_sha256=sha(args.history_audit),history_review_sha256=sha(args.history_review))
    write(reserve_bundle/'manifest.json',reserve_manifest)
    jobs=[]
    def job(identity,command,status,expected_rows=None):
        jobs.append(dict(id=identity,command=command,status_path=str(status),expected_rows=expected_rows))
    for model in cfg['reserve_models']:
        for mode in cfg['modes']:
            out=stage/'reserve_baselines'/model/mode
            job(f'reserve/{model}/{mode}',[sys.executable,str(code/'scripts/run_enumeration_fresh_baseline.py'),
                '--bundle',str(reserve_bundle),'--model',model,'--mode',mode,'--cache-dir',str(args.cache_dir),'--output',str(out)],out/'status.json',60)
    for phase in ('prepare','capture','analyze','confirm'):
        for cell in cells:
            out=stage/'cells'/cell['model']/cell['mode']
            job(f'{phase}/{cell["model"]}/{cell["mode"]}',[sys.executable,str(code/'scripts/run_enumeration_n10_discovery.py'),
                '--stage',str(stage),'--model',cell['model'],'--mode',cell['mode'],'--phase',phase],out/f'{phase}_status.json')
    for phase in ('smoke','formal'):
        for cell in cells:
            out=stage/'jobs'/phase/cell['model']/cell['mode']
            command=[sys.executable,str(code/'scripts/run_enumeration_fresh_update.py'),'--bundle',str(baseline),
                '--registry',cell['registry'],'--cohorts',cell['cohort'],'--cache-dir',str(args.cache_dir),'--output',str(out),
                '--selection-manifest',str(stage/'selection_manifest.json')]
            if phase=='smoke':command.append('--smoke')
            job(f'{phase}/{cell["model"]}/{cell["mode"]}',command,out/'status.json',54 if phase=='smoke' else 540)
    write(stage/'jobs.json',jobs)
    code_hashes={p.relative_to(code).as_posix():sha(p) for p in sorted(code.rglob('*')) if p.is_file() and p.suffix in ('.py','.json')}
    contract_paths=[stage/'protocol.json',stage/'jobs.json',stage/Path(__file__).name,
                    reserve_bundle/'manifest.json',reserve_bundle/'protocol.json',reserve_bundle/'stimuli.jsonl']+[Path(c['cohort']) for c in cells]
    manifest=dict(schema='enumeration_n10_update_manifest_v1',status='FROZEN_BEFORE_N10_DISCOVERY',created_utc=datetime.now(timezone.utc).isoformat(),
        root=str(root),code=str(code),cache_dir=str(args.cache_dir),initial_discovery_seeds=base['discovery_seeds'],
        all_original_confirmation_candidates=base['confirmation_candidates'],reserve_seeds=reserve,reserve_models=cfg['reserve_models'],cells=cells,
        prerequisite=str(root/'fresh_retrieve_gpu_v2/gpu_pipeline_status.json'),prerequisite_complete_phase='RETRIEVE_GPU_COMPLETE_ANALYSIS_PENDING',
        source_sha256={str(p):sha(p) for p in source_paths},code_sha256=code_hashes,
        contract_sha256={str(p.relative_to(stage)):sha(p) for p in contract_paths},
        expected_reserve_rows=120,expected_discovery_states=800,expected_confirmation_states=400,
        expected_formal_rows=2160,expected_smoke_rows=216,command=sys.argv,seconds=time.monotonic()-tick)
    write(stage/'manifest.json',manifest);verify(stage)
    print(json.dumps({'status':manifest['status'],'manifest_sha256':sha(stage/'manifest.json'),'jobs':len(jobs),
                      'cells':[{k:c[k] for k in ('model','mode','original_discovery_format_eligible')} for c in cells]}),flush=True)


def verify(stage):
    m=read(stage/'manifest.json')
    for relative,digest in m['contract_sha256'].items():assert sha(stage/relative)==digest,relative
    for relative,digest in m['code_sha256'].items():assert sha(Path(m['code'])/relative)==digest,relative
    for path,digest in m['source_sha256'].items():assert sha(path)==digest,path
    return m


def freeze_selection(stage,manifest):
    assert not (stage/'selection_manifest.json').exists()
    cells=[]
    for info in manifest['cells']:
        folder=stage/'cells'/info['model']/info['mode']
        cohort=read(folder/'cohort.json');selection=read(folder/'selection.json')
        assert selection['status']=='PASS' and selection['confirmation_used'] is False
        assert selection['discovery_seeds']==cohort['discovery_seeds'] and selection['gold_count']==10
        assert sha(folder/'discovery_states.npz')==selection['states_sha256']
        assert sha(folder/'discovery_layer_metrics.csv')==selection['metrics_sha256']
        cells.append(dict(model=info['model'],mode=info['mode'],gold_count=10,discovery_states=200,
            discovery_seeds=cohort['discovery_seeds'],confirmation_seeds=info['confirmation_seeds'],
            layer_one_based=selection['selected']['layer_one_based'],selection_path=str(folder/'selection.json'),
            selection_sha256=sha(folder/'selection.json'),cohort_sha256=sha(folder/'cohort.json')))
    write(stage/'selection_manifest.json',dict(status='FROZEN_N10_DISCOVERY_LAYERS',confirmation_used_for_selection=False,
        confirmation_is_reused=True,cells=cells,created_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(stage/'protocol.json')))
    return cells


def assemble(stage,manifest):
    tick=time.monotonic();sources={};total=0
    for cell in manifest['cells']:
        source=stage/'jobs/formal'/cell['model']/cell['mode']
        assert read(source/'status.json')['status']=='COMPLETE'
        for folder in sorted(source.glob('k*')):
            audit=read(folder/'technical_audit.json');assert audit['status']=='PASS' and audit['trials']==30
            target=stage/'primary'/cell['model']/cell['mode']/folder.name
            target.mkdir(parents=True,exist_ok=False)
            for name,key in (('trials.jsonl','trials_sha256'),('raw_generations.jsonl','raw_sha256'),('prefill_hooks.jsonl','hooks_sha256')):
                assert sha(folder/name)==audit[key]
                shutil.copy2(folder/name,target/name);sources[str(target/name)]=sha(target/name)
            total+=30
    assert total==2160
    write(stage/'assembly_audit.json',dict(status='PASS',scope='Complete formal grid and exact output copies; statistical audit pending',
        primary_rows=total,source_files_sha256=sources,selection_manifest_sha256=sha(stage/'selection_manifest.json'),seconds=time.monotonic()-tick))


def run(stage):
    manifest=verify(stage);tick=time.monotonic();path=stage/'gpu_pipeline_status.json'
    with path.open('x',encoding='utf-8') as handle:json.dump({'status':'STARTING','pid':os.getpid()},handle)
    state=dict(status='RUNNING',phase='CPU_TESTS',pid=os.getpid(),jobs=[],command=sys.argv)
    def save():
        state['seconds']=time.monotonic()-tick;write(path,state)
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',
             OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    save()
    try:
        command=[sys.executable,'-m','pytest','tests/test_enumeration_update_n10.py','-q','-p','no:cacheprovider']
        with (stage/'cpu_validation.log').open('x') as log:
            result=subprocess.run(command,cwd=manifest['code'],env=env,stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,'N10 contract CPU tests failed'
        write(stage/'cpu_validation.json',dict(status='PASS',command=command,manifest_sha256=sha(stage/'manifest.json'),
              log_sha256=sha(stage/'cpu_validation.log'),gpu_validation='PENDING'))
        state.update(phase='WAIT_PRIMARY_RETRIEVE_COMPLETE');save()
        while True:
            previous=read(manifest['prerequisite'])
            if previous['status'] in ('FAILED','SUPERSEDED'):raise RuntimeError('Primary prerequisite needs resolution')
            if previous['status']=='COMPLETE':
                assert previous['phase']==manifest['prerequisite_complete_phase'];break
            if time.monotonic()-tick>48*3600:raise TimeoutError('Primary prerequisite did not complete in 48 hours')
            time.sleep(20);save()
        verify(stage)
        for job in read(stage/'jobs.json'):
            if job['id'].startswith('confirm/') and not (stage/'selection_manifest.json').exists():
                state['phase']='FREEZE_FOUR_N10_LAYERS';save()
                state['selected_cells']=freeze_selection(stage,manifest);save()
            record=dict(**job,status='RUNNING');state['jobs'].append(record);state['phase']=job['id'];save()
            log_path=stage/'logs'/job['id']/'process.log';log_path.parent.mkdir(parents=True,exist_ok=False)
            start=time.monotonic()
            with log_path.open('x') as log:
                process=subprocess.Popen(job['command'],env=env,stdout=log,stderr=subprocess.STDOUT)
                record['pid']=process.pid;save()
                while process.poll() is None:
                    status_path=Path(job['status_path'])
                    if status_path.exists():
                        record['worker_status']=read(status_path)
                    save();time.sleep(10)
                record['returncode']=process.returncode
            record['seconds']=time.monotonic()-start
            if record['returncode']:
                record['status']='FAILED';raise RuntimeError(f'Failed {job["id"]}; inspect {log_path}')
            completed=read(job['status_path']);assert completed['status']=='COMPLETE'
            if job['expected_rows'] is not None:
                actual=completed.get('completed_rows',completed.get('completed'))
                assert actual==job['expected_rows'],(job['id'],actual)
                record['rows']=actual
            record.update(status='COMPLETE',worker_status=completed);save()
        state['phase']='ASSEMBLE_N10_PRIMARY';save();assemble(stage,manifest)
        state.update(status='COMPLETE',phase='N10_UPDATE_GPU_COMPLETE_ANALYSIS_PENDING')
    except BaseException as error:
        state.update(status='FAILED',error=repr(error));raise
    finally:save()


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    freeze_parser=sub.add_parser('freeze')
    for name in ('root','protocol','extensions','history-audit','history-review','cache-dir','data-root','source-stimuli','output'):
        freeze_parser.add_argument('--'+name,type=Path,required=True)
    sub.add_parser('run').add_argument('--stage',type=Path,required=True)
    args=parser.parse_args()
    freeze(args) if args.action=='freeze' else run(args.stage)


if __name__=='__main__':main()
