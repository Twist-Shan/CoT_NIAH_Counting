"""Freeze the scoped Native Broad full-span rerun before inspecting new outcomes."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time

B=Path(__file__).resolve().parents[1];OLD=B/'runs/task_local_disjoint_first_20260908_v3'
VERSION='native_broad_full_span_20260909_v1';RUN=B/'runs'/VERSION
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')

def main():
    start=time.perf_counter();root=RUN/'package';root.mkdir(parents=True,exist_ok=False)
    old=read(OLD/'package/protocol.json');assert read(OLD/'downloaded/analysis/audit.json')['status']=='PASS'
    for rel,h in old['files'].items():
        if rel.startswith('src/') or rel in ['run.py','protocol.py','diagnostics.py','local_selection.py','task_scoring.py']:
            p=OLD/'package'/rel;assert sha(p)==h
            d=root/rel;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
    main_files=['src/realistic_niah_v5/encoding.py','src/realistic_niah_v5/capture.py','src/realistic_niah_v5/parsing.py','src/realistic_niah_v3/first_list_cutoff.py']
    for rel in main_files:assert sha(root/rel)==sha(B.parent/rel),('Main source differs',rel)
    for name in ['run_native_broad_full_span.py','analyze_native_broad_full_span.py','native_broad_full_span_status.py']:
        shutil.copyfile(B/'deployment'/name,root/name)
    for name in ['native_broad_full_span.py','category_target_broad.py']:shutil.copyfile(B/name,root/name)
    shutil.copyfile(B/'NATIVE_BROAD_FULL_SPAN_20260909.md',root/'PROTOCOL.md')
    (root/'plans').mkdir();available={}
    for model in old['models']:
        available[model]={}
        for task in old['tasks']:
            p=OLD/'package/plans'/f'{task}_{model}.json';assert sha(p)==old['files'][f'plans/{task}_{model}.json']
            plans=[p for p in read(p) if p['mode']=='native_thinking'];assert len(plans)==300
            write(root/'plans'/f'{task}_{model}.json',plans)
            available[model][task]=sum(p['split']=='confirmation' and bool(p['broad_prefix']) for p in plans)
    baseline=OLD/'downloaded/analysis/per_case.csv'
    with baseline.open(encoding='utf-8',newline='') as f:rs=[r for r in csv.DictReader(f) if r['mode']=='native_thinking' and r['assay']=='broad']
    assert len(rs)==2247
    with (root/'baseline_per_case.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    py='outputs/external/lambda_nfs_CoT-Native-thinking-v5_venv_v6_20260828_bin_python'
    lines=['#!/bin/bash','set -euo pipefail','cd "$(dirname "$0")"','exec 9>worker.lock','flock -n 9 || exit 1','export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1']
    for stage in ['geometry','discovery','canary','full']:
        for model in old['models']:lines.append(f'{py} run_native_broad_full_span.py --root . --model {model} --stage {stage}')
        if stage in ['geometry','canary']:lines.append(f'{py} analyze_native_broad_full_span.py --root . --stage {stage}')
    lines += [f'{py} analyze_native_broad_full_span.py --root .',
              'tar -czf results.tgz -C . geometry geometry_audit.json discovery canary canary_audit.json full banks analysis protocol.json',
              'echo NATIVE_BROAD_FULL_SPAN_COMPLETE']
    (root/'launch.sh').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    cfg=dict(version=VERSION,record_scope='registered_generated_full_record_spans',models=old['models'],tasks=old['tasks'],modes=['native_thinking'],
        cache=old['cache'],widths=old['widths'],generation_seed=20260908,discovery_seeds=old['discovery_seeds'],confirmation_seeds=old['confirmation_seeds'],
        sizes={'Qwen3-8B':[1,2,4,8,16,32,64,128],'Gemma4-E4B':[1,2,4,6,8]},baseline_available=available,
        expected_full_points_upper_bound=2247,geometry_inputs=1200,discovery_inputs=800,canary_points=16,
        max_new_tokens=64,control_policy=old['control_policy'],random_seeds=[7000,7001,7002],
        main_geometry='parse_trace_record -> item_end char sites -> align_trace_sites -> _visible_item_spans at original answer prefix',
        main_metric='realistic_niah_v5.capture._broad_span_metrics; epsilon=1e-12',
        main_source_hashes={rel:sha(root/rel) for rel in main_files},
        selection='per model/task discovery-only global Top-K; case then seed equal',
        eligibility='old answer query available and at least one literal registered full span before that query; no outcome filtering',
        stats=dict(seed=20260907,bootstrap=20000,correction='Holm across both models/tasks/all 26 K separately for all_examples and clean_correct',
            k_selection='post-hoc maximum all_examples delta with smallest-K tie; exploratory',comparison='same case and K, seed paired delta change vs endpoint-token version'),
        local_numeric_validation='ulp8 at most; exact zero and complete head-order equality; GPU exact',
        baseline_protocol_sha256=sha(OLD/'package/protocol.json'),baseline_per_case_source_sha256=sha(baseline),experiments_reused=False,natural_inputs_reused=True)
    cfg['files']={str(p.relative_to(root)).replace('\\','/'):sha(p) for p in root.rglob('*') if p.is_file()};write(root/'protocol.json',cfg)
    for p in root.rglob('*.py'):compile(p.read_text(encoding='utf-8'),str(p),'exec')
    with tarfile.open(RUN/'package.tgz','w:gz') as t:
        for p in sorted(root.rglob('*')):
            if p.is_file():t.add(p,arcname=str(p.relative_to(root)).replace('\\','/'))
    result=dict(status='PREPARED',version=VERSION,files=len(cfg['files']),archive_sha256=sha(RUN/'package.tgz'),protocol_sha256=sha(root/'protocol.json'),baseline_available=available,elapsed_seconds=time.perf_counter()-start)
    write(RUN/'preparation.json',result);print(json.dumps(result))

if __name__=='__main__':main()
