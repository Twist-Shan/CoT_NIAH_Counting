"""Finish the CPU prerequisites and dispatch the frozen N=10 queue exactly once."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','mount','extensions'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();root=args.root;stage=root/'fresh_n10_update_v1';started=time.monotonic()
    status_path=root/'n10_setup_status.json'
    with status_path.open('x') as handle:json.dump({'status':'STARTING','pid':os.getpid()},handle)
    state=dict(status='RUNNING',phase='WAIT_SEED_HISTORY',pid=os.getpid(),jobs=[],command=sys.argv)
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',
             OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    def save():
        state['seconds']=time.monotonic()-started;write(status_path,state)
    def wait_file(path,limit=7200):
        tick=time.monotonic()
        while not path.exists():
            if time.monotonic()-tick>limit:raise TimeoutError(str(path))
            time.sleep(10);save()
    def run(identity,command):
        log_path=root/f'n10_setup_{identity}.log';record=dict(id=identity,status='RUNNING',command=command)
        state['jobs'].append(record);state['phase']=identity.upper();save();tick=time.monotonic()
        with log_path.open('x') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            record['pid']=process.pid;save()
            while process.poll() is None:time.sleep(10);save()
            record['returncode']=process.returncode
        record.update(seconds=time.monotonic()-tick,status='COMPLETE' if process.returncode==0 else 'FAILED');save()
        if process.returncode:raise RuntimeError(f'{identity} failed: {log_path}')
    save()
    try:
        history=root/'n10_seed_history_20260916/audit.json';wait_file(history)
        h=read(history);assert h['status']=='PASS_UNPACKED_INVENTORY' and not h['conflicts'] and not h['errors']
        state['history_files']=len(h['files']);state['history_sha256']=hashlib.sha256(history.read_bytes()).hexdigest();save()
        review=root/'n10_seed_history_review.json'
        if not review.exists():
            old_launch=root/'n10_seed_history_review_launch.json'
            if old_launch.exists():wait_file(review)
            else:run('history_review',[sys.executable,str(root/'analysis_tools/review_enumeration_seed_history.py'),
                                      '--audit',str(history),'--output',str(review)])
        assert read(review)['status']=='PASS'
        run('freeze',[sys.executable,str(args.extensions/'scripts/launch_enumeration_n10_update.py'),'freeze',
            '--root',str(root),'--protocol',str(args.extensions/'configs/enumeration_n10_update_20260916.json'),
            '--extensions',str(args.extensions),'--history-audit',str(history),'--history-review',str(review),
            '--cache-dir',str(args.mount/'cache/huggingface'),
            '--data-root',str(args.mount/'code/Realistic_CoT_NiaH_Count_v6_20260828/data'),
            '--source-stimuli',str(args.mount/'runs/v6_enumeration_replication_20260828/source_stimuli/stimuli.jsonl'),
            '--output',str(stage)])
        state['phase']='LAUNCH_FROZEN_QUEUE';save()
        command=[sys.executable,str(stage/'launch_enumeration_n10_update.py'),'run','--stage',str(stage)]
        with (stage/'gpu_pipeline.log').open('x') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record=dict(pid=process.pid,command=command,manifest_sha256=hashlib.sha256((stage/'manifest.json').read_bytes()).hexdigest())
        write(stage/'launch_record.json',record);state['n10_queue']=record;save()
        tick=time.monotonic()
        while True:
            if process.poll() is not None and process.returncode!=0:raise RuntimeError('N10 queue failed at startup')
            path=stage/'gpu_pipeline_status.json'
            if path.exists():
                observed=read(path)
                if observed['status']=='FAILED':raise RuntimeError(observed)
                if observed['status']=='RUNNING' and observed['phase']!='CPU_TESTS':break
            if time.monotonic()-tick>180:raise TimeoutError('N10 queue startup verification timed out')
            time.sleep(2)
        state.update(status='COMPLETE',phase='FROZEN_N10_QUEUE_LAUNCHED',observed_queue=observed)
    except BaseException as error:
        state.update(status='FAILED',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
