"""Independently verify the fixed-layer confirmation and merge its display cell."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path,PurePosixPath
import sys
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from realistic_niah_v6.update_audit import verify_trial,matched_trials,summarize,jsonl
from realistic_niah_v6.read_audit import json_sha
from realistic_niah_v5.trace_stratified_geometry import _fit_projection_and_predict
from scripts.run_enumeration_midlayer_update import choose,jobs
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main(source,discovery_source,prior,output):
    tick=time.monotonic();inventory=read(source/'backup_manifest.json')
    assert inventory['schema']=='enumeration_replay_midlayer_backup_v1'
    for name,record in inventory['files'].items():assert sha(source/name)==record['sha256'],name
    for name,record in inventory['external_dependencies'].items():assert sha(discovery_source/name)==record['sha256'],name
    def relocate(path):
        name=PurePosixPath(path).relative_to(PurePosixPath(inventory['source_root']))
        return source/name if (source/name).is_file() else discovery_source/name
    stage=source/'fresh_midlayer_update_20260917_v1';m=read(stage/'manifest.json');cfg=read(stage/'protocol.json');state=read(stage/'status.json')
    for name,h in m['source_sha256'].items():assert sha(relocate(name))==h,name
    for name,h in m['frozen_sha256'].items():assert sha(stage/name)==h,name
    seeds=m['selected_seeds'];ledger=read(relocate(m['registry']+'/ledger.json'))
    selected=read(source/'fresh_n10_update_v1/selection_manifest.json')
    original_cell=next(c for c in selected['cells'] if c['model']=='Qwen3-8B' and c['mode']=='enumeration_index')
    baseline=read(source/'fresh_v1/manifest.json')
    used=set(original_cell['confirmation_seeds'])|set(baseline['discovery_seeds'])|set(baseline['read_seeds'])
    used.update(read(source/'fresh_qwen_index_n10_layer_diagnostic_v2/manifest.json')['selected_seeds'])
    used.update(read(source/'fresh_causal_v1/cohorts.json')['models']['Qwen3-8B']['confirmation_seeds'])
    for c in read(source/'fresh_native_update_v1/manifest.json')['cells']:
        if c['model']=='Qwen3-8B' and c['mode']=='enumeration_index':used.update(c['selected_seeds'])
    assert sorted(used)==m['previously_used_seeds']
    assert choose(ledger,cfg,used)==seeds and len(seeds)==10
    assert m['new_layer_one_based']==19 and m['original_ncc_layer_one_based']==30 and m['selected_using_confirmation_outcomes'] is False
    assert state['status']=='COMPLETE' and state['completed_rows']==594
    expected_jobs=jobs(cfg,seeds);assert read(stage/'jobs.json')==expected_jobs
    assert [j['id'] for j in state['jobs']]==[j['id'] for j in expected_jobs]
    runtime=read(stage/'runtime.json');previous=read(relocate(m['source_runtime']))
    for key in ['model_revision','model_source_sha256','dtype','backend','layers','torch','transformers']:assert runtime[key]==previous[key],key
    assert runtime['manifest_sha256']==sha(stage/'manifest.json') and runtime['selected_seeds']==seeds and runtime['layer_one_based']==19
    inputs={r['seed']:r for r in jsonl(stage/'inputs/adapted_generations.jsonl')};assert set(inputs)==set(seeds)
    original={r['seed']:r for r in jsonl(relocate(m['registry']+'/adapted_generations.jsonl')) if r['gold_count']==10}
    selected_ledger=read(stage/'inputs/ledger.json')
    frozen_geometry={}
    for r in selected_ledger:
        assert inputs[r['seed']]==original[r['seed']]
        g=stage/'inputs'/r['geometry_file'];assert sha(g)==r['geometry_sha256'] and json_sha(inputs[r['seed']])==read(g)['adapted_row_sha256']
        for item in read(g)['update']['trials']:
            frozen_geometry[r['seed'],item['donor_occurrence'],item['direction'],item['scope']]=item
    parser_path=source/'fresh_causal_v1/code/src/realistic_niah_v5/same_site_progress_transplant.py'
    frozen=read(source/'fresh_causal_v1/code_manifest.json');assert sha(parser_path)==frozen['original_code_sha256']['src/realistic_niah_v5/same_site_progress_transplant.py']
    spec=importlib.util.spec_from_file_location('_frozen_midlayer_parser',parser_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    by_phase={'smoke':[],'formal':[]};widths=Counter()
    for job,done in zip(expected_jobs,state['jobs']):
        folder=stage/'jobs'/job['id'];status=read(folder/'status.json')
        assert status['status']==done['status']=='COMPLETE' and status['completed_rows']==done['completed_rows']==job['expected_rows']
        for name,h in status['sha256'].items():assert sha(folder/name)==h,(job['id'],name)
        command=read(folder/'command.json')
        for flag,value in [('--layers','18'),('--donor-occurrence',str(job['k'])),('--receiver-occurrence',str(job['j'])),('--max-new-tokens','96')]:assert command[command.index(flag)+1]==value
        trials,raw,hooks=[jsonl(folder/name) for name in ['trials.jsonl','raw_generations.jsonl','prefill_hooks.jsonl']]
        matched=matched_trials(trials,job['seeds']);assert len(raw)==len(trials)==job['expected_rows'] and len(hooks)==2*len(trials)
        geometry={r['seed']:r for r in jsonl(folder/'geometry_audit.jsonl')};assert set(geometry)==set(job['seeds'])
        receiver_hash={}
        for i,(trial,generation) in enumerate(zip(trials,raw)):
            seed=trial['seed'];g=geometry[seed];pair=hooks[2*i:2*i+2]
            assert trial['request_id']==inputs[seed]['request_id']
            assert g['aligned_absolute_site']==trial['shared_commit_position'] and g['effective_patch_width']==trial['patch_width']
            assert g['endpoint_aligned'] is True and g['hidden_state_resampling'] is False
            role='donor' if trial['condition']=='native_donor' else 'receiver';assert trial['shared_commit_token_id']==g[f'{role}_commit_token_id']
            expected_geometry=frozen_geometry[seed,job['k'],job['direction'],job['scope']]
            expected_prefix=expected_geometry['aligned_prefix_sha256' if expected_geometry['aligned_prompt_role']==role else 'unaltered_prefix_sha256']
            assert pair[0]['input_ids_sha256']==pair[1]['input_ids_sha256']==expected_prefix
            assert trial['shared_commit_position']==expected_geometry['shared_absolute_endpoint']
            assert trial['patch_width']==expected_geometry['effective_patch_width']
            for field in ['receiver_item_coverage','donor_item_coverage','equal_length_complete_item']:assert trial[field]==g[field]
            verify_trial(trial,generation,pair,cities=[r['city'] for r in inputs[seed]['gold_records']],parser=module.generated_bullet_city_ordinals,
                layer=18,k=job['k'],direction=job['direction'],scope=job['scope'],max_tokens=96)
            receiver_hash[seed,trial['condition']]=pair[0]['input_ids_sha256']
            by_phase[job['phase']].append(dict(trial,audit_scope=job['scope'],audit_direction=job['direction']))
            if job['phase']=='formal':widths[job['scope'],trial['patch_width']]+=1
        for seed in job['seeds']:
            assert receiver_hash[seed,'receiver_self']==receiver_hash[seed,'donor_to_receiver']
            for field in ['shared_commit_position','patch_width']:assert len({matched[seed,c][field] for c in cfg['conditions']})==1
            g=geometry[seed]
            assert g['deletion_avoids_prompt_records'] and g['deletion_avoids_special_tokens'] and g['deletion_before_answer']
    assert len(by_phase['smoke'])==54 and len(by_phase['formal'])==540
    # Refit the original discovery decoder on CPU and verify all new predictions.
    rs=source/'fresh_midlayer_readout_20260917_v1';rm=read(rs/'manifest.json');rr=read(rs/'readout.json')
    assert read(rs/'status.json')['status']=='COMPLETE' and rr['status']=='PASS' and rr['first_trace_repeat_exact']
    for p,h in rm['source_sha256'].items():assert sha(relocate(p))==h,p
    for p,h in rr['files_sha256'].items():assert sha(rs/p)==h,p
    assert rr['manifest_sha256']==sha(rs/'manifest.json') and rr['used_for_layer_selection'] is False
    dpath=relocate(rm['discovery']+'/discovery_states.npz');dmeta=pd.read_csv(relocate(rm['discovery']+'/discovery_metadata.csv'))
    with np.load(dpath) as d:dx=d['states'][:,18]
    with np.load(rs/'confirmation_states.npz') as c:cx=c['states'][:,0];assert c['layer_indices'].tolist()==[18]
    cmeta=pd.read_csv(rs/'confirmation_metadata.csv');assert len(cmeta)==100 and set(cmeta['seed'])==set(seeds) and not set(cmeta['seed'])&set(dmeta['seed'])
    logistic,ncc,_,components=_fit_projection_and_predict(dx,dmeta['occurrence'].to_numpy(dtype=int),cx,np.arange(1,11),pca_dim=16,random_state=0,pca_whiten=True)
    assert ncc.tolist()==[r['ncc'] for r in rr['predictions']] and logistic.tolist()==[r['logistic'] for r in rr['predictions']]
    assert float(np.mean(ncc==cmeta['occurrence'].to_numpy()))==rr['confirmation_ncc']
    metrics=pd.read_csv(relocate(rm['discovery']+'/discovery_layer_metrics.csv'));drow=metrics[metrics['layer_one_based'].eq(19)].iloc[0]
    assert rr['discovery_ncc']==float(drow['discovery_oof_ncc_balanced_accuracy'])
    result=dict(model='Qwen3-8B',mode='enumeration_index',audit='PASS',layer_one_based=19,selected_seeds=seeds,
        formal_rows=540,smoke_rows=54,confirmation_inputs_used_in_prior_selection=False,layer_choice_after_exploration=True,
        patch_width_rows={f'{s}/{w}':n for (s,w),n in sorted(widths.items())},readout=rr,
        groups=summarize(by_phase['formal'],bootstrap=dict(repetitions=cfg['bootstrap_repetitions'],seed=cfg['bootstrap_seed'])))
    report=dict(status='PASS',schema='enumeration_midlayer_confirmation_audit_v1',cells=[result],
        source_inventory_sha256=sha(source/'backup_manifest.json'),manifest_sha256=sha(stage/'manifest.json'),
        formal_rows=540,smoke_rows=54,readout_independent_cpu_replay=True,source_seed_clusters=10,
        limits=['L19 was fixed after the exploratory layer scan; it is not the NCC-maximizing layer.',
            'Confirmation uses ten previously generated but unused source inputs; all three controls and all failed/truncated continuations retained.',
            'Intervals are pointwise seed-bootstrap intervals conditional on the fixed layer and analysis.'],seconds=time.monotonic()-tick)
    output.mkdir(parents=True,exist_ok=False);write(output/'audit.json',report)
    with (output/'parsed_trials.jsonl').open('x') as f:
        for row in by_phase['formal']:f.write(json.dumps(row,ensure_ascii=True)+'\n')
    old=read(prior);assert old['status']=='PASS' and old['schema']=='enumeration_n10_update_audit_v1'
    merged=dict(status='PASS',schema='enumeration_midlayer_combined_audit_v1',cells=deepcopy(old['cells']),
        formal_rows=2160,smoke_rows=216,bootstrap=old['bootstrap'],
        limits=report['limits']+old['limits'][2:])
    merged['historical_ncc_qwen_index']=next(c for c in old['cells'] if c['model']=='Qwen3-8B' and c['mode']=='enumeration_index')
    merged['cells']=[result if c['model']=='Qwen3-8B' and c['mode']=='enumeration_index' else c for c in merged['cells']]
    merged['source_audits']=dict(original_n10=dict(path=str(prior),sha256=sha(prior)),midlayer=dict(path=str(output/'audit.json'),sha256=sha(output/'audit.json')))
    merged['chronology']='Qwen Index displayed at fixed middle L19 on independent confirmation; original NCC-selected L30 retained separately. Other three cells unchanged.'
    write(output/'combined_update_audit.json',merged)
    table=[]
    for g in result['groups']:
        if g['donor_k']!='all':continue
        values={c:g['conditions'][c]['continuation']['1'] for c in cfg['conditions']};second=g['conditions']['donor_to_receiver']['continuation']['2']
        table.append(dict(scope=g['scope'],direction=g['direction'],counts={c:f"{v['successes']}/{v['horizon_eligible']}" for c,v in values.items()},
            second=f"{second['successes']}/{second['conditional_eligible']}",paired=g['paired_target_minus_self_adoption']))
    write(output/'summary.json',dict(status='PASS',discovery_ncc=rr['discovery_ncc'],confirmation_ncc=rr['confirmation_ncc'],groups=table))
    print(json.dumps(dict(status='PASS',formal_rows=540,smoke_rows=54,discovery_ncc=rr['discovery_ncc'],confirmation_ncc=rr['confirmation_ncc'],groups=table,seconds=report['seconds'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['source','discovery-source','prior','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();main(a.source,a.discovery_source,a.prior,a.output)
