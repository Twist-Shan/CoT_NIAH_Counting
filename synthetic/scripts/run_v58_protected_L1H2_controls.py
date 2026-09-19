"""Add protected-L1H2 controls, preserving all earlier ablation results."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
for d in (ROOT/'src',ROOT/'scripts'):
    sys.path.insert(0,str(d))
import pandas as pd
import torch
import audit_v58_retrieval_generation as audit
import run_v58_top1to8_aligned as original
import v58_cached_decode as cached
from v58_protected_control_policy import protected_control_plan


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=50)
    args=p.parse_args()
    root=args.output
    if root.exists():
        raise FileExistsError(root)
    dest=root/'nonthinking'
    (dest/'arms').mkdir(parents=True)
    prior=args.run_dir/'analysis/v58_top1to8_final_query_20260908'
    old=json.loads((prior/'protocol.json').read_text())
    align=args.run_dir/'analysis/v58_alignment_supplement_20260905'
    frozen=json.loads((align/'nonthinking/frozen_sites.json').read_text())
    ranking=[r[:2] for r in frozen['ranking']['broad']]
    plans={str(k):protected_control_plan(ranking[:k]) for k in range(1,9)}
    arms=[]
    for k,s in plans.items():
        arms.append(dict(arm='selected',top_k=int(k),repeat=0,heads=s['selected'],overlap=0))
        arms.extend(dict(arm='control',top_k=int(k),repeat=j,**c) for j,c in enumerate(s['controls']))
    assert len(arms)==352
    source=Path(old['examples_source'])
    assert audit.digest(source)==old['source_sha256'][str(source.resolve())]
    protocol={'status':'frozen_before_protected_control_inference','created_unix':time.time(),
              'mode':'nonthinking','excluded_from_controls':[[1,2]],'selection_ranking_unchanged':True,
              'reason':'User-requested diagnostic extension after L1H2 removal caused answer-marker repetition. Discovery mean replacement preserves the original answer readout; all original controls remain archived.',
              'interpretation':'Exploratory sensitivity analysis with L1H2 retained. This exclusion was chosen after inspecting the original controls; it is not a preregistered validation.',
              'control_rule':'Exhaustive same-layer count matching, disjoint first, minimum necessary overlap after excluding L1H2. No further control exclusion.',
              'scopes':{'nonthinking':['sustained']},'decoding':'Full-vocabulary greedy until EOS or 64 new tokens; same pre-O zeroing and original count parser; invalid outputs retained as failures.',
              'plans':{'nonthinking':plans},'arms':arms,'checkpoint_sha256':old['checkpoint_sha256']['nonthinking'],
              'inputs':100,'support_keys':old['support_keys'],
              'sources':{str(s):audit.digest(s) for s in [Path(__file__),ROOT/'scripts/v58_protected_control_policy.py',
                         Path(cached.__file__),Path(original.__file__),prior/'protocol.json',prior/'prefixes.jsonl',source]}}
    audit.write_json(root/'protocol.json',protocol)
    shutil.copyfile(prior/'prefixes.jsonl',root/'prefixes.jsonl')
    records=[json.loads(s) for s in (prior/'prefixes.jsonl').read_text().splitlines()]
    records=[r for r in records if r['mode']=='nonthinking']
    examples={e.prompt_sha256:e for e in [audit.base.example_from_dict(json.loads(s)) for s in source.read_text().splitlines() if s.strip()]}
    assert len(records)==100 and set(r['key'] for r in records)==set(old['support_keys'])
    checkpoint=audit.checkpoint_path(args.run_dir,'nonthinking',10000)
    assert audit.digest(checkpoint)==protocol['checkpoint_sha256']
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    cfg,vocab,_,_,model=audit.base.load_v20_checkpoint_model(args.run_dir,'rope','nonthinking',step=10000,device='cuda')
    model.eval()
    small=[next(r for r in records if r['count']==n) for n in range(1,11)]
    checks=[]
    for k in [4,8]:
        heads=plans[str(k)]['controls'][0]['heads']
        full,hook=original.generate(model,cfg,vocab,small,examples,heads,'sustained',10,64,verify=True)
        fast,_=cached.generate(model,cfg,vocab,small,examples,heads,'sustained',10,64)
        assert full.generated_tokens.tolist()==fast.generated_tokens.tolist()
        checks.append(dict(k=k,full_vs_cached_exact=True,**hook))
    clean=pd.read_csv(prior/'nonthinking/clean.csv')
    clean_map=clean.set_index('key')
    shutil.copyfile(prior/'nonthinking/clean.csv',dest/'clean.csv')
    # Reuse exact head sets only; the actual intervention does not depend on K labels.
    old_map={audit.canonical(a['heads']):prior/'nonthinking/arms'/f"sustained_{a['arm']}_k{a['top_k']}_r{a['repeat']}.csv"
             for a in old['arms'] if a['mode']=='nonthinking'}
    reused=[]
    for i,a in enumerate(arms):
        heads=audit.canonical(a['heads'])
        if heads in old_map:
            source_path=old_map[heads]
            frame=pd.read_csv(source_path)
            reused.append(dict(new_arm=a,source=str(source_path),sha256=audit.digest(source_path)))
        else:
            frame,_=cached.generate(model,cfg,vocab,records,examples,a['heads'],'sustained',args.batch_size,64)
        for c in ['arm','top_k','repeat','overlap']:
            frame[c]=a[c]
        frame['heads']=heads
        frame['clean_accuracy']=frame.key.map(clean_map.ar_accuracy)
        frame['clean_pred_count']=frame.key.map(clean_map.ar_pred_count)
        frame['absolute_count_shift']=(pd.to_numeric(frame.ar_pred_count)-pd.to_numeric(frame.clean_pred_count)).abs()
        frame['normalized_count_shift']=frame.absolute_count_shift/frame['count']
        frame.to_csv(dest/'arms'/f"sustained_{a['arm']}_k{a['top_k']}_r{a['repeat']}.csv",index=False)
        if i%20==0 or a['arm']=='selected':
            print(i+1,len(arms),a['arm'],a['top_k'],a['repeat'],'accuracy',frame.ar_accuracy.mean(),'valid',frame.ar_answered.mean(),flush=True)
    audit.write_json(root/'validation.json',{'checkpoint_matches':True,'exact_original_control_reuse':reused,
        'cached_vs_full_checks':checks,'protected_L1H2_never_ablated_in_controls':True,
        'all_control_counts':[plans[str(k)]['actual_unique_controls'] for k in range(1,9)]})
    audit.write_json(root/'manifest.json',{'status':'complete','rows':(len(arms)+1)*100,'reused_conditions':len(reused),
        'new_condition_trajectories':(len(arms)-len(reused))*100,
        'files':{str(s.relative_to(root)):audit.digest(s) for s in root.rglob('*') if s.is_file()}})
    print('PROTECTED L1H2 SWEEP COMPLETE',flush=True)


if __name__=='__main__':
    main()
