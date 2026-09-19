"""Create an isolated, hash-frozen matched-category Broad experiment package."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time

B = Path(__file__).resolve().parents[1]
VERSION = 'category_target_broad_20260909_v1'
OLD = B/'runs/task_local_disjoint_first_20260908_v3'
RUN = B/'runs'/VERSION
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,x): p.write_text(json.dumps(x,indent=2),encoding='utf-8')


def main():
    started = time.perf_counter(); root = RUN/'package'; root.mkdir(parents=True,exist_ok=False)
    old = read(OLD/'package/protocol.json')
    assert read(OLD/'downloaded/analysis/audit.json')['status']=='PASS'
    # Snapshot only verified dependencies, including the working Gemma backend restoration.
    for rel,h in old['files'].items():
        if rel.startswith('src/') or rel in ['run.py','protocol.py','diagnostics.py','local_selection.py','task_scoring.py']:
            source=OLD/'package'/rel; assert sha(source)==h
            dest=root/rel; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,dest)
    for name in ['run_category_target_broad.py','analyze_category_target_broad.py']:
        shutil.copyfile(B/'deployment'/name,root/name)
    shutil.copyfile(B/'category_target_broad.py',root/'category_target_broad.py')
    shutil.copyfile(B/'CATEGORY_TARGET_BROAD_20260909.md',root/'PROTOCOL.md')
    (root/'plans').mkdir();(root/'baseline_banks').mkdir()
    for model in old['models']:
        source=OLD/'package/plans'/f'category_{model}.json'
        assert sha(source)==old['files'][f'plans/category_{model}.json']
        plans=[p for p in read(source) if p['mode']=='nonthinking']
        assert len(plans)==300 and all(p['broad_prefix'] for p in plans)
        write(root/'plans'/f'{model}.json',plans)
        shutil.copyfile(OLD/'package/banks'/f'category_{model}_nonthinking_broad.json',root/'baseline_banks'/f'{model}.json')
    baseline=OLD/'downloaded/analysis/per_case.csv'
    with baseline.open(encoding='utf-8',newline='') as f:
        rows=[r for r in csv.DictReader(f) if r['task']=='category' and r['mode']=='nonthinking' and r['assay']=='broad']
    assert len(rows)==1300
    with (root/'baseline_per_case.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    py='outputs/external/lambda_nfs_CoT-Native-thinking-v5_venv_v6_20260828_bin_python'
    lines=['#!/bin/bash','set -euo pipefail','cd "$(dirname "$0")"','exec 9>worker.lock','flock -n 9 || exit 1',
           'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1']
    for stage in ['discovery','canary','full']:
        for model in old['models']:lines.append(f'{py} run_category_target_broad.py --root . --model {model} --stage {stage}')
        if stage=='canary':lines.append(f'{py} analyze_category_target_broad.py --root . --stage canary')
    lines += [f'{py} analyze_category_target_broad.py --root . --stage full',
              'tar -czf results.tgz -C . discovery canary full banks analysis canary_audit.json protocol.json',
              'echo CATEGORY_TARGET_BROAD_COMPLETE']
    (root/'launch.sh').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    cfg=dict(version=VERSION,record_scope='question_target_category',tasks=['category'],modes=['nonthinking'],
        models=old['models'],widths=old['widths'],cache=old['cache'],generation_seed=20260908,
        discovery_seeds=old['discovery_seeds'],confirmation_seeds=old['confirmation_seeds'],
        sizes={'Qwen3-8B':[1,2,4,8,16,32,64,128],'Gemma4-E4B':[1,2,4,6,8]},
        discovery_cases_per_model=200,confirmation_cases_per_model=100,expected_full_points=1300,
        metric='M*exp(H)/J; all terms use only records matching the question target_category',
        pooling='case then seed equal; city and flower questions share one category ranking per model',
        selection='global Top-K without per-layer cap; discovery only',
        max_new_tokens=64,ablation_scope='answer_query_one_prefill_pre_O',
        control_policy=old['control_policy'],random_seeds=[7000,7001,7002],
        scoring='exact requested category count, same saved prompts and final-answer query as baseline',
        stats=dict(seed=20260907,bootstrap=20000,unit='seed',
                   correction='exact two-sided seed sign-flip; Holm over both models and all 13 K comparisons separately for all_examples and clean_correct',
                   subgroups='city_questions and flower_questions are descriptive; pointwise CIs',
                   selection='peak K is post-hoc exploratory; same previously used confirmation seeds'),
        baseline_protocol_sha256=sha(OLD/'package/protocol.json'),baseline_per_case_source_sha256=sha(baseline),
        experiments_reused=False,natural_inputs_reused=True)
    cfg['files']={str(p.relative_to(root)).replace('\\','/'):sha(p) for p in root.rglob('*') if p.is_file()}
    write(root/'protocol.json',cfg)
    for p in root.rglob('*.py'):compile(p.read_text(encoding='utf-8'),str(p),'exec')
    archive=RUN/'package.tgz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(root.rglob('*')):
            if p.is_file():tar.add(p,arcname=str(p.relative_to(root)).replace('\\','/'))
    result=dict(status='PREPARED',version=VERSION,files=len(cfg['files']),archive_sha256=sha(archive),
                protocol_sha256=sha(root/'protocol.json'),elapsed_seconds=time.perf_counter()-started)
    write(RUN/'preparation.json',result);print(json.dumps(result))


if __name__=='__main__':main()
