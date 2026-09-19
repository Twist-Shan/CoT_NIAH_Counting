#!/usr/bin/env python3
"""Strict paired analysis of CoT supplement outputs; no incomplete-case drops."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import numpy as np

SEED = 20260912
DRAWS = 10000


def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').split('\n') if s.strip()]


def keyed(rows, keys):
    result={tuple(r[k] for k in keys):r for r in rows}
    if len(result)!=len(rows):raise ValueError('Duplicate trial keys')
    return result


def interval(values):
    values=np.asarray(values,dtype=float)
    if len(values)!=10 or not np.isfinite(values).all():raise ValueError('Expected ten finite seed-level contrasts')
    rng=np.random.default_rng(SEED)
    boot=values[rng.integers(0,len(values),(DRAWS,len(values)))].mean(axis=1)
    return {'estimate':float(values.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
            'seed_values':values.tolist(),'bootstrap_seed':SEED,'bootstrap_draws':DRAWS}


def load_blank(root,variant,bank):
    folder=root/'backend_replay'/variant/f'blank_{bank}'
    status=json.loads((folder/'process_status.json').read_text())
    assert status['status']=='COMPLETE'
    files=sorted((folder/'results/shards').glob('*.jsonl'))
    assert len(files)==100
    rows=[r for f in files for r in read(f)]
    assert len(rows)==500 and {r['status'] for r in rows}=={'ok'}
    index=keyed(rows,['seed','gold_count','condition'])
    conditions=['clean','prompt_all_blank','prompt_records_blank','trace_all_blank','prompt_and_trace_blank']
    assert set(index)=={(s,n,c) for s in range(1254,1264) for n in range(1,11) for c in conditions}
    events=read(folder/'backend_events.jsonl')
    attention=[e for e in events if e['event']=='position_attention_outputs']
    generation=[e for e in events if e['event']=='generate_answer_completion']
    assert len(attention)==len(generation)==500
    expected='sdpa' if variant=='fixed' else 'eager'
    assert all(e['after']['text']==expected for e in attention+generation)
    return rows,index


def blank(root,bank):
    old,left=load_blank(root,'legacy',bank)
    new,right=load_blank(root,'fixed',bank)
    geometry=['request_id','query_full_sequence_token','token_blank_registry_sha256','blank_positions_sha256','bank_heads']
    assert set(left)==set(right)
    changes=[]
    for k in sorted(left):
        a,b=left[k],right[k]
        assert all(a[f]==b[f] for f in geometry),(k,'geometry changed')
        fields=[f for f in ['prediction','exact_count','completion_text_raw','generated_token_count','generation_truncated'] if a[f]!=b[f]]
        if fields:changes.append({'key':k,'fields':fields,'legacy_prediction':a['prediction'],'fixed_prediction':b['prediction']})
    summaries=[]
    for c in sorted({r['condition'] for r in old}):
        a=[r for r in old if r['condition']==c];b=[r for r in new if r['condition']==c]
        values=[np.mean([float(right[(s,n,c)]['exact_count'])-float(left[(s,n,c)]['exact_count']) for n in range(1,11)]) for s in range(1254,1264)]
        summaries.append({'condition':c,'n':100,'legacy_exact':sum(bool(r['exact_count']) for r in a),'fixed_exact':sum(bool(r['exact_count']) for r in b),
                          'legacy_truncated':sum(bool(r['generation_truncated']) for r in a),'fixed_truncated':sum(bool(r['generation_truncated']) for r in b),
                          'fixed_minus_legacy_accuracy':interval(values),
                          'max_gold_token_logp_difference':max(abs(left[k]['gold_first_answer_token_log_probability']-right[k]['gold_first_answer_token_log_probability']) for k in left if k[-1]==c)})
    archive=root.parent.parent/'runs/v5_token_level_ablation_20260821/Gemma4-E4B'/f'answer_{bank}bank_top32_confirmation_all20_v1/shards'
    archived=[r for f in sorted(archive.glob('*.jsonl')) for r in read(f)]
    assert len(archived)==500
    historical=keyed(archived,['seed','gold_count','condition'])
    assert set(historical)==set(left)
    historical_changes=[]
    for k,a in historical.items():
        b=left[k]
        assert a['token_blank_registry_sha256']==b['token_blank_registry_sha256'],(k,'historical registry')
        fields=[f for f in ['prediction','exact_count','completion_text_raw','generated_token_count'] if a[f]!=b[f]]
        if fields:historical_changes.append({'key':k,'fields':fields})
    return {'status':'COMPLETE_PAIRED_AUDIT','bank':bank,'paired_trials':500,'geometry_unchanged':True,
            'legacy_backend_leak_reproduced':True,'fixed_backend_restored':True,'changed_trials':len(changes),'changes':changes,'summaries':summaries,
            'legacy_vs_archived_changed_trials':len(historical_changes),'legacy_vs_archived_changes':historical_changes}


def progress(root):
    result={}
    indices={}
    for variant in ['legacy','fixed']:
        allrows=[]
        for direction in ['forward','backward']:
            for k in [4,6,8]:
                folder=root/'backend_replay'/variant/f'progress_{direction}_k{k}'
                assert json.loads((folder/'process_status.json').read_text())['status']=='COMPLETE'
                rows=read(folder/'results/trials.jsonl')
                assert len(rows)==30
                assert Counter(r['seed'] for r in rows)=={s:3 for s in range(1276,1286)}
                expected_receiver=k-1 if direction=='forward' else k+1
                assert {(r['seed'],r['condition']) for r in rows}=={(s,c) for s in range(1276,1286) for c in ['receiver_self','native_donor','donor_to_receiver']}
                for row in rows:
                    assert row['layer']==16 and row['gold_count']==10 and row['patch_scope']=='item_span'
                    assert row['receiver_occurrence_j']==expected_receiver and row['donor_occurrence_k']==k
                    assert row['patch_applications']==1
                    assert all(np.isfinite(row[f]) for f in ['donor_vs_receiver_sum_logodds','donor_vs_receiver_attention_log_ratio'])
                for r in rows:r.update(direction=direction,target_k=k)
                allrows+=rows
        idx=keyed(allrows,['direction','target_k','seed','condition'])
        assert len(idx)==180
        indices[variant]=idx
        summaries=[]
        for direction in ['forward','backward']:
            for condition in ['receiver_self','donor_to_receiver']:
                rows=[r for r in allrows if r['direction']==direction and r['condition']==condition]
                assert len(rows)==30
                steps=[]
                for hop in range(1,5):
                    eligible=success=0
                    for r in rows:
                        k=r['target_k'];cities=r['generated_known_city_ordinals_any_surface']
                        valid=k+hop<=10 and cities[:hop-1]==list(range(k+1,k+hop))
                        eligible+=valid;success+=valid and len(cities)>=hop and cities[hop-1]==k+hop
                    steps.append({'hop':hop,'eligible':eligible,'success':success})
                summaries.append({'direction':direction,'condition':condition,'n':30,
                                  'adoption':sum(r['greedy_donor_successor_adoption'] for r in rows),
                                  'truncated':sum(r['generation_truncated'] for r in rows),'steps':steps})
        result[variant]=summaries
    changes=[]
    for k,a in indices['legacy'].items():
        b=indices['fixed'][k]
        for f in ['shared_commit_position','patch_width','targeted_bank_sha256']:
            assert a[f]==b[f],(k,f)
        fields=[f for f in ['completion_text','generated_token_count','first_generated_known_city_ordinal','greedy_donor_successor_adoption','donor_successor_argmax'] if a.get(f)!=b.get(f)]
        if fields:changes.append({'key':k,'fields':fields})
    return {'status':'COMPLETE_PAIRED_AUDIT','paired_trials':180,'summaries':result,'changed_trials':len(changes),'changes':changes}


def count_ok(row):
    matches=re.findall(r'(?i)(?:^|\b)total\s*:\s*([0-9]+)\b',row['completion_text'])
    return bool(matches and int(matches[-1])==int(row['gold_count']))


def generation(root,model,task):
    folder=root/'generation/runs'/model/f'{task}_full'
    assert json.loads((folder/'process_status.json').read_text())['status']=='COMPLETE'
    rows=[r for f in sorted((folder/'results').glob('*.jsonl')) for r in read(f)]
    expected=(330 if model=='Qwen3-8B' else 170) if task=='dose' else 80
    assert len(rows)==expected,(len(rows),expected)
    assert {r['seed'] for r in rows}==set(range(1254,1264))
    for r in rows:r['final_count_correct']=count_ok(r)
    summaries=[]
    if task=='dose':
        idx=keyed(rows,['seed','dose_k','condition','repeat'])
        plan=json.loads((root/'dose_inputs'/model/'plan.json').read_text())
        for k in plan['ks']:
            for outcome in ['final_count_correct','correct_next_needle']:
                selected=[];random=[];clean=[]
                for s in range(1254,1264):
                    selected.append(float(idx[(s,k,'selected_bank',0)][outcome]))
                    rs=[r for key,r in idx.items() if key[:3]==(s,k,'layer_matched_random')]
                    assert len(rs)==3
                    random.append(float(np.mean([r[outcome] for r in rs])))
                    clean.append(float(idx[(s,0,'clean',0)][outcome]))
                summaries.append({'k':k,'outcome':outcome,'selected_mean':float(np.mean(selected)),'random_mean':float(np.mean(random)),
                                  'clean_mean':float(np.mean(clean)),'selected_minus_random':interval(np.array(selected)-random)})
        historical=keyed(read(root/'dose_inputs'/model/'historical.jsonl'),['seed','condition','repeat'])
        full=[r for r in rows if r['dose_k'] in [0,max(plan['ks'])]]
        assert len(full)==len(historical)==50
        replay=[]
        for r in full:
            old=historical[(r['seed'],r['condition'],r['repeat'])]
            assert {tuple(h) for h in r['heads']}=={tuple(h) for h in old['heads']}
            assert r['intervention_anchor_equivalence_ids']==old['intervention_anchor_equivalence_ids']
            fields=[f for f in ['generated_token_ids','correct_next_needle','generation_truncated'] if r[f]!=old[f]]
            if fields:replay.append({'seed':r['seed'],'condition':r['condition'],'repeat':r['repeat'],'fields':fields,
                                     'old_final_correct':count_ok(old),'new_final_correct':count_ok(r)})
        return {'status':'COMPLETE_AUDIT','model':model,'trials':len(rows),'summaries':summaries,
                'full_k_historical_replay':{'compared':50,'changed_trials':len(replay),'changes':replay}}
    else:
        idx=keyed(rows,['seed','condition'])
        arms=sorted({r['condition'] for r in rows})
        assert len(arms)==8
        for arm in arms:
            values=[idx[(s,arm)] for s in range(1254,1264)]
            summaries.append({'arm':arm,'n':10,'final_count_correct':sum(r['final_count_correct'] for r in values),
                              'next_city_correct':sum(r['correct_next_needle'] for r in values),'truncated':sum(r['generation_truncated'] for r in values),
                              'incomplete_carrier_schedule':sum(any(v for v in r['carrier_audit'].get('missing_positions',{}).values()) for r in values)})
        contrasts=[]
        for scope in ['local','persistent']:
            for outcome in ['final_count_correct','correct_next_needle']:
                for comparator in ['lesion','matched']:
                    values=[float(idx[(s,scope+'_restore')][outcome])-float(idx[(s,scope+'_'+comparator)][outcome]) for s in range(1254,1264)]
                    contrasts.append({'scope':scope,'outcome':outcome,'comparison':'restore_minus_'+comparator,**interval(values)})
        return {'status':'COMPLETE_AUDIT','model':model,'trials':len(rows),'summaries':summaries,'contrasts':contrasts}
    return {'status':'COMPLETE_AUDIT','model':model,'trials':len(rows),'summaries':summaries}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['blank_trace','blank_prompt','progress','dose','recovery'],required=True)
    p.add_argument('--model',choices=['Qwen3-8B','Gemma4-E4B'])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.stage.startswith('blank_'):report=blank(args.root,args.stage.split('_')[1])
    elif args.stage=='progress':report=progress(args.root)
    else:report=generation(args.root,args.model,args.stage)
    report['analysis_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['changes','summaries']},indent=2))


if __name__=='__main__':main()
