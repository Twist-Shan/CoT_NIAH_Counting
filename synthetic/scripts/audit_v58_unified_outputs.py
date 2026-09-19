"""Fail closed on mismatched realized samples, repeats or endpoint state keys."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd


def audit(root):
    a=root/'v58_alignment_supplement_20260905'
    l=root/'v58_unified_legacy_20260905'
    e=root/'v58_unified_additional_20260905'
    registry=pd.read_csv(a/'input_registry.csv')
    keys=set(registry.loc[registry.split.eq('confirmation'),'key'])
    assert len(keys)==100
    records=[]
    def check(frame, groups, family, key='key', expected=keys):
        for labels,part in frame.groupby(groups,dropna=False):
            assert len(part)==len(expected) and set(part[key])==expected, (family,labels,len(part))
            records.append({'family':family,'condition':str(labels),'rows':len(part),'unique_inputs':part[key].nunique()})
    for mode in ['nonthinking','thinking']:
        check(pd.read_csv(a/mode/'ablation.csv'),['arm','top_k','repeat'],mode+'/ablation')
        check(pd.read_csv(l/mode/'dynamics_behavior_trials.csv'),['step'],mode+'/dynamics_behavior',key='prompt_sha256')
        check(pd.read_csv(l/mode/'dynamics_causal_trials.csv'),['step','condition','repeat'],mode+'/dynamics_causal',key='prompt_sha256')
        probe=pd.read_csv(l/mode/'geometry/frozen_probe_trials.csv')
        for (endpoint,layer,condition,k,repeat),part in probe.groupby(['endpoint','layer','condition','top_k','repeat']):
            assert set(part.prompt_sha256)==keys
            expected_rows=550 if endpoint.endswith(('occurrence','item_end')) else 100
            assert len(part)==expected_rows
            assert not part.duplicated(['prompt_sha256','occurrence']).any()
        records.append({'family':mode+'/geometry_probe','condition':'all endpoints/layers/arms','rows':len(probe),'unique_inputs':100})
        gd=pd.read_csv(e/mode/'geometry_dynamics_trials.csv')
        for (endpoint,layer),part in gd.loc[gd.step.eq(10000)].groupby(['endpoint','layer']):
            reference=probe.loc[probe.endpoint.eq(endpoint)&probe.layer.eq(layer)&probe.condition.eq('clean')]
            x=part.set_index(['prompt_sha256','occurrence']).ncc_correct.sort_index()
            y=reference.set_index(['prompt_sha256','occurrence']).ncc_correct.sort_index()
            assert x.equals(y), (mode,endpoint,'final dynamic probe mismatch')
        for family,filename in [('count_direction','count_vector_trials.csv')]:
            frame=pd.read_csv(e/mode/filename)
            for (layer,arm,offset),part in frame.groupby(['layer','arm','offset']):
                expected=set(registry.loc[registry.split.eq('confirmation') & registry['count'].add(offset).between(1,10),'key'])
                assert len(part)==90 and set(part.key)==expected
            records.append({'family':mode+'/'+family,'condition':'4 layers/5 arms/2 offsets','rows':len(frame),'unique_inputs':100})
    for phase in ['discovery','confirmation']:
        for kind,nt,th in [('running','nonthinking_prompt_occurrence','thinking_item_end'),('answer','nonthinking_answer_query','thinking_answer_query')]:
            x=pd.read_csv(l/'nonthinking/geometry'/f'{nt}_metadata.csv')
            y=pd.read_csv(l/'thinking/geometry'/f'{th}_metadata.csv')
            x=x.loc[x.split.eq(phase),['prompt_sha256','occurrence']].sort_values(['prompt_sha256','occurrence'])
            y=y.loc[y.split.eq(phase),['prompt_sha256','occurrence']].sort_values(['prompt_sha256','occurrence'])
            assert x.to_records(index=False).tolist()==y.to_records(index=False).tolist(), (phase,kind)
    check(pd.read_csv(l/'thinking/factorial_trials.csv'),['arm'],'thinking/factorial',key='prompt_sha256')
    check(pd.read_csv(l/'thinking/transport_trials.csv'),['condition','top_k','repeat'],'thinking/transport')
    check(pd.read_csv(e/'thinking/source_next_trials.csv'),['arm','ordinary_control'],'thinking/source_next')
    pairs=pd.read_csv(l/'continuation/frozen_pairs.csv')
    ck=set(registry.loc[registry.split.eq('confirmation') & registry['count'].eq(10),'key'])
    assert set(pairs.loc[pairs.split.eq('confirmation'),'prompt_sha256'])==ck
    assert len(pairs.loc[pairs.split.eq('confirmation')])==60
    for scope in ['item_end_w1','item_span_w2']:
        path=l/'continuation'/scope/'rollout_trials.csv'
        if path.exists():
            frame=pd.read_csv(path)
            for condition,part in frame.groupby('condition'):
                assert len(part)==60 and set(part.prompt_sha256)==ck
            pivot=frame.pivot(index='pair_id',columns='condition',values='continuation_tokens')
            assert (pivot.clean==pivot.self_patch).all()
    for folder in [a,l,e]:
        assert json.loads((folder/'manifest.json').read_text())['status']=='complete'
    result={'status':'passed','input_registry_sha256':hashlib.sha256((a/'input_registry.csv').read_bytes()).hexdigest(),
        'discovery_prompts':200,'confirmation_prompts':100,'count10_confirmation_prompts':10,
        'paired_endpoint_state_keys_equal':True,'checks':records}
    (l/'unified_sample_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True)
    audit(p.parse_args().analysis)
