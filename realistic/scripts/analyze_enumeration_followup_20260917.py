"""Audit the fixed layer diagnostic and Native-eligibility answer-patch amendment."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from realistic_niah_v6.read_audit import read,require,sha,json_sha
from realistic_niah_v6.update_audit import jsonl,verify_trial,CONDITIONS
from realistic_niah_v6.fresh_behavior_audit import unique_grid,verify_generation,seed_mean
from realistic_niah_v6.aligned_reporting import continuation_summary
from analyze_enumeration_fresh_read import frozen_parser


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    start=time.monotonic();out=a.output;out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'work/ef17/src';inputs=ROOT/'work/eu2/src';old=ROOT/'work/ea2/src'
    n10=ROOT/'work/en10u/src';read_inputs=ROOT/'outputs/enumeration_alignment_20260916/read_audit_v1/source'
    bases=[source,n10,ROOT/'work/en10d/src',old,inputs,read_inputs]
    remote=PurePosixPath(read(source/'backup_manifest.json')['source_root'])
    verified={}
    def resolve(path):
        relative=PurePosixPath(path).relative_to(remote)
        found=[b/relative for b in bases if (b/relative).is_file()]
        require(bool(found),f'Missing archived input: {relative}');return found[0]
    def check(path,expected):
        path=Path(path);require(sha(path)==expected,f'Hash mismatch: {path}');verified[str(path)]=expected
    for name,rec in read(source/'backup_manifest.json')['files'].items():
        check(source/name,rec['sha256'])
    # Layer diagnostic: every layer, direction, seed, arm, raw output and hook.
    stage=source/'fresh_qwen_index_n10_layer_diagnostic_v2';man=read(stage/'manifest.json');cfg=read(stage/'protocol.json')
    for path,h in man['input_sha256'].items():check(resolve(path),h)
    for f,key in [('protocol.json','protocol_sha256'),('jobs.json','jobs_sha256'),('launch_enumeration_qwen_layer_diagnostic.py','entrypoint_sha256')]:check(stage/f,man[key])
    require(read(stage/'gpu_pipeline_status.json')['status']=='COMPLETE','Layer queue incomplete')
    original_runtime=read(resolve(man['source_runtime']));runtime=read(stage/'runtime.json')
    for key in ['model_revision','model_source_sha256','dtype','backend','layers','torch','transformers']:
        require(runtime[key]==original_runtime[key],f'Layer numerical environment changed: {key}')
    parserpath=inputs/'fresh_causal_v1/code/src/realistic_niah_v5/same_site_progress_transplant.py'
    parser=module(parserpath,'_layer_parser').generated_bullet_city_ordinals
    registered=jsonl(resolve(man['registry']+'/adapted_generations.jsonl'))
    source_rows={r['seed']:r for r in registered if r['gold_count']==10}
    rows=[];smoke_count=0
    for job in read(stage/'jobs.json'):
        folder=stage/'jobs'/job['phase']/job['direction'];status=read(folder/'status.json')
        require(status['status']=='COMPLETE','Layer job incomplete')
        for f,h in status['files_sha256'].items():check(folder/f,h)
        trial,raw,hooks=[jsonl(folder/f) for f in ['trials.jsonl','raw_generations.jsonl','prefill_hooks.jsonl']]
        require(len(trial)==len(raw)==job['expected_rows'] and len(hooks)==2*len(trial),'Raw/hook coverage mismatch')
        grid=unique_grid(trial,['seed','layer','condition'],{(s,l-1,c) for s in job['seeds'] for l in job['layers_one_based'] for c in CONDITIONS})
        geometries={g['seed']:g for g in jsonl(folder/'geometry_audit.jsonl')}
        hashes={}
        for i,(r,generation) in enumerate(zip(trial,raw)):
            cities=[g['city'] for g in source_rows[r['seed']]['gold_records']]
            require(r['request_id']==source_rows[r['seed']]['request_id'] and generation['layer']==r['layer'],'Wrong source/layer')
            g=geometries[r['seed']]
            require(g['aligned_absolute_site']==r['shared_commit_position'] and g['effective_patch_width']==r['patch_width'] and
                    g['endpoint_aligned'] and not g['hidden_state_resampling'],'Geometry mismatch')
            require(g['deletion_avoids_prompt_records'] and g['deletion_avoids_special_tokens'] and g['deletion_before_answer'],'Unsafe alignment')
            verify_trial(r,generation,hooks[2*i:2*i+2],cities=cities,parser=parser,layer=r['layer'],k=6,direction=job['direction'],scope='item_span',max_tokens=96)
            hashes[r['seed'],r['layer'],r['condition']]=hooks[2*i]['input_ids_sha256']
            if job['phase']=='formal':rows.append(dict(r,direction=job['direction'],source='new'))
        for s in job['seeds']:
            for l in job['layers_one_based']:
                require(hashes[s,l-1,'receiver_self']==hashes[s,l-1,'donor_to_receiver'],'Unpaired self/Target input')
        if job['phase']=='smoke':smoke_count+=len(trial)
    # L30 already received full raw/hook/geometry audit; verify its exact source hashes again.
    primary_audit=ROOT/'outputs/enumeration_alignment_20260916/n10_update_audit_v2/audit.json'
    check(primary_audit,'be1fe9af2001044c103bbc7df214bfc0d6d68a316b11b6b4b8a8587a685e4e70')
    require(read(primary_audit)['status']=='PASS','Original N10 audit failed')
    for reuse in man['reused']:
        rows.extend(dict(r,direction=reuse['direction'],source='reused_n10_primary') for r in jsonl(resolve(reuse['path']+'/trials.jsonl')))
    unique_grid(rows,['seed','layer','condition','direction'],{(s,l-1,c,d) for s in man['selected_seeds'] for l in cfg['layers_one_based'] for c in CONDITIONS for d in cfg['directions']})
    layer_stats=[]
    for l in cfg['layers_one_based']:
        for direction in cfg['directions']:
            group=[r for r in rows if r['layer']==l-1 and r['direction']==direction]
            arms={c:[r for r in group if r['condition']==c] for c in CONDITIONS}
            controls={r['seed']:r for r in arms['receiver_self']};target=arms['donor_to_receiver']
            result=dict(layer_one_based=l,direction=direction,
                conditions={c:seed_mean([(r['seed'],int(r['greedy_donor_successor_adoption'])) for r in rr]) for c,rr in arms.items()},
                paired_adoption=seed_mean([(r['seed'],int(r['greedy_donor_successor_adoption'])-int(controls[r['seed']]['greedy_donor_successor_adoption'])) for r in target]),
                continuation={str(h):continuation_summary(target,h,draws=10000,random_seed=20260915) for h in range(1,5)},
                truncated=sum(r['generation_truncated'] for r in group))
            layer_stats.append(result)
            print('layer',l,direction,{c:int(v['observation_sum']) for c,v in result['conditions'].items()},flush=True)
    save(out/'layer_audit.json',dict(status='PASS',new_formal_rows=360,reused_rows=60,smoke_rows=smoke_count,
        scope='Exploratory fixed seven-layer panel after N10 outcomes, k=6 item-span; original ten inputs reused; L30 retained as NCC primary.',
        results=layer_stats,selection_manifest_sha256=man['primary_selection_manifest_sha256']))
    # Answer eligibility: reconstruct Native selector using every original candidate.
    stage=source/'fresh_answer_patch_gpu_v3_native_eligibility';regroot=source/'fresh_answer_patch_registry_v3_native_eligibility'
    man=read(stage/'manifest.json');reg=read(regroot/'manifest.json');cfg=read(stage/'protocol.json')
    require(not reg['clean_regeneration_required'],'Extra replay correctness filter still enabled')
    for mapping in ['subset_sha256','reuse_evidence_sha256']:
        for path,h in man[mapping].items():check(resolve(path),h)
    for f,key in [('protocol.json','protocol_sha256'),('jobs.json','jobs_sha256')]:check(stage/f,man[key])
    check(regroot/'manifest.json',man['registry_manifest_sha256'])
    for f,h in man['entrypoints_sha256'].items():check(stage/'code'/f,h)
    compiler=module(stage/'code/scripts/prepare_enumeration_fresh_answer_patch.py','_new_answer_compiler')
    worker=module(stage/'code/scripts/run_enumeration_fresh_answer_patch.py','_new_answer_worker')
    parse_total=frozen_parser(inputs);indexed={};prepared={}
    for cell in reg['cells']:
        model,mode=cell['model'],cell['mode'];folder=regroot/model/mode
        for f,key in [('eligibility.json','eligibility_sha256'),('pairs.json','pairs_sha256')]:check(folder/f,cell[key])
        source_reg=inputs/'fresh_causal_v1/registries'/model/mode
        check(source_reg/'manifest.json',cell['source_registry_sha256']);check(source_reg/'adapted_generations.jsonl',cell['source_generations_sha256'])
        originals={r['request_id']:r for r in jsonl(source_reg/'adapted_generations.jsonl')}
        ledger={r['request_id']:r for r in read(source_reg/'ledger.json') if 'unfiltered_read' in r['roles']}
        pop=read(folder/'eligibility.json');require(len(pop)==100 and {e['request_id'] for e in pop}==set(ledger),'Candidate population changed')
        by_seed=indexed.setdefault(mode,{})[model]={}
        for e in pop:
            original=originals[e['request_id']];entry=ledger[e['request_id']]
            gpath=resolve(e['geometry_file']);check(gpath,e['geometry_sha256']);g=read(gpath)
            require(json_sha(original)==e['adapted_row_sha256']==g['adapted_row_sha256'],'Input identity changed')
            reason=compiler.eligibility(entry,original,require_clean=False)
            require(e['exclusion']==reason and e['eligible']==(reason is None),'Native eligibility mismatch')
            if e['eligible']:
                require(e['read_geometry']==g['read'],'Legal query geometry changed')
                by_seed.setdefault(e['seed'],{})[e['gold_count']]=e
        prepared[model,mode]={e['request_id']:e for e in pop}
    for mode,by_model in indexed.items():
        panels,counts,shortfalls=compiler.select_pairs(by_model,reg['source_seeds'],cfg['models'])
        require(not shortfalls and counts==reg['modes'][mode]['common_counts_by_seed'],'Wrong common candidate counts')
        for model,pairs in panels.items():
            expected=[dict(pair,mode=mode,layers=list(range(cfg['num_layers'][model]))) for pair in pairs]
            require(expected==read(regroot/model/mode/'pairs.json'),'Native pair selector differs')
    allrows=[];cells=[];new_formal=0;new_smoke=0;reused=0
    for cell in reg['cells']:
        model,mode=cell['model'],cell['mode'];pairs=read(regroot/model/mode/'pairs.json');pairmap={r['pair_id']:r for r in pairs};depth=cfg['num_layers'][model]
        oldpairs={r['pair_id']:r for r in read(old/'fresh_answer_patch_registry_v2'/model/mode/'pairs.json')}
        kept=set(pairmap)&set(oldpairs);added=set(pairmap)-set(oldpairs)
        require(all(pairmap[k]==oldpairs[k] for k in kept),'Reused pair geometry changed')
        oldjob=old/'fresh_answer_patch_gpu_v2/jobs/formal'/model/mode
        sources=[('reused',oldjob,[p for p in pairs if p['pair_id'] in kept])]
        if added:
            newpairs=[p for p in pairs if p['pair_id'] in added]
            sources.extend([('smoke',stage/'jobs/smoke'/model/mode,newpairs[:2]),('formal',stage/'jobs/formal'/model/mode,newpairs)])
        formal=[]
        for phase,job,chosen in sources:
            status=read(job/'status.json');runtime=read(job/'runtime.json');check(job/'trials.jsonl',status['trials_sha256'])
            require(status['status']=='COMPLETE' and runtime['layers']==list(range(depth)) and runtime['max_new_tokens']==16,'Answer runtime/grid changed')
            expected_stage=old/'fresh_answer_patch_gpu_v2' if phase=='reused' else stage
            require(runtime['stage_manifest_sha256']==sha(expected_stage/'manifest.json'),'Answer runtime manifest mismatch')
            oldruntime=read(oldjob/'runtime.json')
            for key in ['model_revision','model_source_sha256','dtype','backend','torch','transformers']:
                require(runtime[key]==oldruntime[key],f'Answer numerical setting changed: {key}')
            trial=jsonl(job/'trials.jsonl');require(len(trial)==status['completed_rows']==status['expected_rows'],'Answer saved rows incomplete')
            chosen_ids={p['pair_id'] for p in chosen}
            trial=[r for r in trial if r['pair_id'] in chosen_ids]
            grid=unique_grid(trial,['pair_id','layer','condition'],{(p['pair_id'],l,c) for p in chosen for l in range(depth) for c in cfg['conditions']})
            replays=read(job/'clean_query_replays.json') if phase!='reused' else None
            for pair in chosen:
                receiver,donor=[prepared[model,mode][pair[k]] for k in ['receiver_request_id','donor_request_id']]
                query=receiver['read_geometry']['query_position']
                ref=None if replays is None else replays[pair['receiver_request_id']]
                if replays is not None:
                    for rid,entry in [(pair['receiver_request_id'],receiver),(pair['donor_request_id'],donor)]:
                        replay=replays[rid];verify_generation(replay['generated'],16)
                        require(replay['query_position']==entry['read_geometry']['query_position'] and replay['gold_count']==entry['gold_count'],'Replay query identity mismatch')
                        require(parse_total(replay['generated']['full_answer_text'])==replay['prediction'] and replay['exact_count']==(replay['prediction']==replay['gold_count']),'Replay parse mismatch')
                for l in range(depth):
                    panel=[grid[pair['pair_id'],l,c] for c in cfg['conditions']]
                    worker.validate_panel(panel,[r['hook_audit'] for r in panel],pair,l,query,clean_reference=ref)
                    for r in panel:
                        expected_hash=sha(old/'fresh_answer_patch_registry_v2'/model/mode/'pairs.json') if phase=='reused' else cell['pairs_sha256']
                        require(r['pairs_sha256']==expected_hash and r['alignment_pair_id']==pair['alignment_pair_id'],'Row registry provenance mismatch')
                        require((r['model_label'],r['mode'],r['seed'],r['gold_count'],r['donor_count'])==(model,mode,pair['seed'],pair['receiver_count'],pair['donor_count']),'Wrong answer pair identity')
                        require(r['request_id']==pair['receiver_request_id'] and r['donor_request_id']==pair['donor_request_id'],'Wrong source request')
                        require(r['receiver_query_position']==query and r['donor_query_position']==donor['read_geometry']['query_position'],'Wrong query position')
                        if replays is not None:
                            require(r['receiver_clean_replay']==ref and r['donor_clean_replay']==replays[pair['donor_request_id']],'Replay evidence differs')
                        raw=r['hook_audit']['raw_generation'];verify_generation(raw,16);prediction=parse_total(raw['full_answer_text'])
                        require(prediction==r['prediction'] and (prediction==pair['receiver_count'])==r['exact_count'],'Answer prediction mismatch')
                        if phase!='smoke':formal.append(dict(model=model,mode=mode,seed=r['seed'],pair_id=r['pair_id'],layer_one_based=l+1,condition=r['condition'],
                            donor_adoption=prediction==pair['donor_count'],receiver_preserved=prediction==pair['receiver_count'],
                            invalid_count=prediction not in range(1,11),truncated=raw['generation_truncated'],provenance=phase,source_file=str(job/'trials.jsonl')))
            if phase=='reused':reused+=len(trial)
            elif phase=='smoke':new_smoke+=len(trial)
            else:new_formal+=len(trial)
        unique_grid(formal,['pair_id','layer_one_based','condition'],{(p['pair_id'],l+1,c) for p in pairs for l in range(depth) for c in cfg['conditions']})
        stats=[]
        for l in range(1,depth+1):
            group=[r for r in formal if r['layer_one_based']==l];arms={}
            for cond in cfg['conditions']:
                rr=[r for r in group if r['condition']==cond];require(Counter(r['seed'] for r in rr)==Counter({s:4 for s in reg['source_seeds']}),'Unequal seed support')
                arms[cond]={m:seed_mean([(r['seed'],int(r[m])) for r in rr]) for m in ['donor_adoption','receiver_preserved']}
            controls={r['pair_id']:r for r in group if r['condition']=='self_patch'}
            diff=seed_mean([(r['seed'],int(r['donor_adoption'])-int(controls[r['pair_id']]['donor_adoption'])) for r in group if r['condition']=='full_donor_patch'])
            stats.append(dict(layer_one_based=l,conditions=arms,donor_minus_self_adoption=diff))
        cells.append(dict(model=model,mode=mode,audit='PASS',directed_pairs=40,source_seed_count=10,layerwise=stats,final_layer=stats[-1],
            reused_pairs=len(kept),new_pairs=len(added),eligible_inputs=cell['eligible_inputs']))
        allrows.extend(formal)
        print('answer',model,mode,'target/self',[(c,stats[-1]['conditions'][c]['donor_adoption']['observation_sum']) for c in cfg['conditions']],flush=True)
    require((new_formal,new_smoke,reused,len(allrows))==(624,312,11856,12480),'Assembled coverage mismatch')
    save(out/'answer_audit.json',dict(status='PASS',cells=cells,new_formal_rows=new_formal,new_smoke_rows=new_smoke,reused_formal_rows=reused,assembled_rows=len(allrows),
        eligibility='Original correct answer, one-to-one trace, legal query; Native common low/high pair selector. No replay correctness exclusion.',
        self_guard='New rows require zero state delta and exact unmodified replay token match; wrong replay counts retained. Reused rows retain original exact/zero controls and hash provenance.',
        chronology='Amendment requested after v2 results; unchanged pairs reused, only changed pairs newly run; original confirmation reused.'))
    (out/'answer_parsed_outcomes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in allrows),encoding='utf-8')
    save(out/'audit.json',dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-start,verified_files=verified,
        source_inventory_sha256=sha(source/'backup_manifest.json'),script_sha256=sha(__file__),
        limitations=['Saved raw outputs/hook evidence audited on CPU; tokenizer decoding not repeated.',
                    'Layer diagnostic is exploratory on reused inputs; pointwise seed bootstrap is not independent confirmation.']))


if __name__=='__main__':main()
