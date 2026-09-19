"""Run canonical Enumeration checks and four GPU queues with fail-stop logging."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from prepare_enumeration_default_seed import ROOT, OLD, MOUNT, MODE, MODELS, sha, read, write

CODE=ROOT/'code'
CACHE=MOUNT/'cache/huggingface'
ENV=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',
         OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
         PYTHONPATH=f'{CODE}:{CODE}/src:{CODE}/scripts')

def run(name,cmd,gpu=None):
    folder=ROOT/'execution'/name
    if (folder/'status.json').exists():
        previous=read(folder/'status.json')
        assert previous['status']=='COMPLETE' and previous['command']==list(map(str,cmd))
        return
    folder.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();state=dict(status='RUNNING',command=list(map(str,cmd)),gpu=gpu)
    write(folder/'status.json',state)
    env=dict(ENV)
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    try:
        with (folder/'process.log').open('x') as log:
            proc=subprocess.Popen(state['command'],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
            state['pid']=proc.pid;write(folder/'status.json',state)
            state['returncode']=proc.wait()
        if state['returncode']:raise RuntimeError(f'{name} failed; see {folder}/process.log')
        state['status']='COMPLETE'
    except BaseException as e:state.update(status='FAILED',error=repr(e));raise
    finally:state['seconds']=time.monotonic()-start;write(folder/'status.json',state)

def stage(kind,registry,oldstage,scriptnames):
    dest=ROOT/kind;dest.mkdir()
    (dest/'code/scripts').mkdir(parents=True)
    for name in scriptnames:
        source=ROOT/'tools'/name if (ROOT/'tools'/name).exists() else CODE/'scripts'/name
        shutil.copy2(source,dest/'code/scripts'/name)
    shutil.copy2(registry/'protocol.json',dest/'protocol.json')
    write(dest/'manifest.json',dict(status='FROZEN_BEFORE_GPU',registry_root=str(registry),
        registry_manifest_sha256=sha(registry/'manifest.json'),protocol_sha256=sha(dest/'protocol.json'),
        entrypoints_sha256={f'scripts/{n}':sha(dest/'code/scripts'/n) for n in scriptnames}))
    return dest

def freeze():
    # Protocols preserve numerical choices; only modes/cohort provenance change.
    for name,origin in [('retrieve',OLD/'fresh_retrieve_registry_v1/protocol.json'),
                        ('answer',OLD/'fresh_answer_patch_registry_v3_native_eligibility/protocol.json')]:
        cfg=read(origin);cfg['modes']=[MODE];cfg['models']=MODELS
        cfg['cohort_chronology']='Default historical inputs; no new independent confirmation claim.'
        if name=='answer':
            assert not cfg.get('clean_regeneration_required',False)
        write(ROOT/f'{name}_protocol.json',cfg)
    # The historical bundle had only N10 discovery rows; the canonical bundle
    # retains all N. Explicitly keep the same N10-only retrieval protocol.
    tools=ROOT/'tools';tools.mkdir(exist_ok=True)
    source=(CODE/'scripts/prepare_enumeration_fresh_retrieve.py').read_text()
    old='if seed not in baseline["discovery_seeds"] + baseline["read_seeds"]:'
    new='if seed not in baseline["discovery_seeds"] + baseline["read_seeds"] or (seed in baseline["discovery_seeds"] and n != 10):'
    assert source.count(old)==1
    compiler=tools/'prepare_enumeration_fresh_retrieve.py';compiler.write_text(source.replace(old,new))
    run('retrieve_registry_v2',[sys.executable,compiler,'prepare',
        '--root',ROOT,'--output',ROOT/'retrieve_registry_v2','--cache-dir',CACHE,'--protocol',ROOT/'retrieve_protocol.json'])
    run('answer_registry',[sys.executable,CODE/'scripts/prepare_enumeration_fresh_answer_patch.py',
        '--root',ROOT,'--output',ROOT/'answer_registry','--protocol',ROOT/'answer_protocol.json'])
    assert read(ROOT/'answer_registry/manifest.json')['status']=='FROZEN_BEFORE_ANSWER_PATCH_GPU'
    stage('retrieve',ROOT/'retrieve_registry_v2',None,['prepare_enumeration_fresh_retrieve.py','run_enumeration_fresh_retrieve.py'])
    stage('answer',ROOT/'answer_registry',None,['run_enumeration_fresh_answer_patch.py'])

def worker(model,gpu,kind):
    registry=ROOT/'fresh_causal_v1/registries'/model/MODE
    if kind=='read_answer':
        for smoke in (True,False):
            phase='smoke' if smoke else 'formal'
            out=ROOT/'read'/phase/model/MODE
            cmd=[sys.executable,CODE/'scripts/run_enumeration_fresh_read.py','--bundle',ROOT/'fresh_v1',
                '--registry',registry,'--cache-dir',CACHE,'--output',out]+(['--smoke'] if smoke else [])
            run(f'read/{phase}/{model}',cmd,gpu)
            assert read(out/'status.json')['status']=='COMPLETE'
        for smoke in (True,False):
            phase='smoke' if smoke else 'formal';out=ROOT/'answer/jobs'/phase/model/MODE
            cmd=[sys.executable,ROOT/'answer/code/scripts/run_enumeration_fresh_answer_patch.py',
                '--stage',ROOT/'answer','--cache-dir',CACHE,'--model',model,'--mode',MODE,'--output',out]+(['--smoke'] if smoke else [])
            run(f'answer/{phase}/{model}',cmd,gpu)
            assert read(out/'status.json')['status']=='COMPLETE'
    else:
        for phase,action,smoke in [('localize_smoke','localize',True),('localize','localize',False),
                                  ('banks','banks',False),('behavior_smoke','behavior',True),('formal','behavior',False)]:
            out=ROOT/'retrieve'/('banks' if action=='banks' else f'jobs/{phase}')/model/MODE
            cmd=[sys.executable,ROOT/'retrieve/code/scripts/run_enumeration_fresh_retrieve.py',action,
                 '--stage',ROOT/'retrieve','--cache-dir',CACHE,'--model',model,'--mode',MODE,'--output',out]+(['--smoke'] if smoke else [])
            run(f'retrieve/{phase}/{model}',cmd,gpu)
            assert read(out/'status.json')['status']=='COMPLETE'

def main():
    started=time.monotonic();state=dict(status='RUNNING',phase='WAIT_SETUP',pid=os.getpid(),workers=[])
    state_path=ROOT/'pipeline_v2_status.json'
    if state_path.exists():raise FileExistsError(state_path)
    def save():state['seconds']=time.monotonic()-started;write(state_path,state)
    save()
    try:
        while True:
            setup=read(ROOT/'setup_status.json')
            if setup['status']=='FAILED':raise RuntimeError(setup)
            if setup['status']=='COMPLETE':break
            if time.monotonic()-started>1800:raise TimeoutError('Setup exceeded 30 minutes')
            time.sleep(10)
        state['phase']='CPU_VALIDATION';save()
        run('cpu_tests',[sys.executable,'-m','pytest','tests/test_enumeration_fresh_geometry.py',
            'tests/test_enumeration_update_n10.py','-q','-p','no:cacheprovider'])
        freeze()
        state['phase']='GPU_READ_RETRIEVE_ANSWER';save()
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[(model,gpu,kind,pool.submit(worker,model,gpu,kind)) for model,gpu,kind in
                [(MODELS[0],0,'read_answer'),(MODELS[1],1,'read_answer'),(MODELS[0],2,'retrieve'),(MODELS[1],3,'retrieve')]]
            errors=[]
            for model,gpu,kind,future in futures:
                try:future.result();result='COMPLETE'
                except BaseException as e:result='FAILED';errors.append(repr(e))
                state['workers'].append(dict(model=model,gpu=gpu,kind=kind,status=result));save()
            if errors:raise RuntimeError(errors)
        state.update(status='COMPLETE',phase='READ_RETRIEVE_ANSWER_COMPLETE_UPDATE_PENDING')
    except BaseException as e:state.update(status='FAILED',error=repr(e));raise
    finally:save()

if __name__=='__main__':main()
