"""Audit the default-seed replication using frozen inputs and existing validators.

Retrieve and Answer validators are adapted from the previous audited runners;
only layout, deployment metadata and all-new execution differ. No outcome filter.
"""
from pathlib import Path
from collections import Counter
import argparse, json, sys, time, importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from realistic_niah_v6.read_audit import read, require, sha, json_sha, audit_cell as audit_read
from realistic_niah_v6.update_audit import jsonl, verify_trial, matched_trials, summarize, CONDITIONS, SCOPES
from realistic_niah_v6.update_n10 import rank_layers, validate_selection
from realistic_niah_v6.fresh_behavior_audit import unique_grid,verify_generation,seed_mean,load_frozen_city_scorer
from analyze_enumeration_fresh_read import frozen_parser

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
load_module=module
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
def check(path,h):require(sha(path)==h,f'Hash mismatch: {path}')
def resolve(path):return Path(path)

def retrieve(root,out):
    stage,registry=root/'retrieve',root/'retrieve_registry_v2'
    manifest,reg,cfg=read(stage/'manifest.json'),read(registry/'manifest.json'),read(stage/'protocol.json')
    for file,expected in [(stage/'protocol.json',manifest['protocol_sha256']),(registry/'manifest.json',manifest['registry_manifest_sha256']),
                          (root/'fresh_causal_v1/code_manifest.json',reg['causal_code_manifest_sha256'])]:check(file,expected)
    for name,h in manifest['entrypoints_sha256'].items():check(stage/'code'/name,h)
    class Args:pass
    a=Args();a.inputs=root
    compiler = load_module(stage / "code/scripts/prepare_enumeration_fresh_retrieve.py", "_retrieve_compiler")
    worker = load_module(stage / "code/scripts/run_enumeration_fresh_retrieve.py", "_retrieve_worker")
    parse_total = frozen_parser(a.inputs)
    scorer_path = a.inputs / "fresh_causal_v1/code/src/realistic_niah_v5/causal.py"
    score_city = load_frozen_city_scorer(scorer_path)
    require(not reg["uses_final_correctness_for_selection"] and not reg["uses_intervention_outcomes"], "Outcome-selected Retrieve population")
    require(not set(reg["discovery_seeds"]) & set(reg["confirmation_seeds"]), "Discovery/confirmation overlap")
    cells, all_derived = [], []
    for mode in cfg["modes"]:
        queries_by_model = {model: read(registry / model / mode / "queries.json") for model in cfg["models"]}
        selected, missing = compiler.choose_confirmation(queries_by_model, reg["confirmation_seeds"])
        require(not missing, "Missing confirmation source seeds")
        for model in cfg["models"]:
            require(selected[model] == read(registry / model / mode / "confirmation.json"), "Highest-common-count selector mismatch")
    for cell in reg["cells"]:
        model, mode = cell["model"], cell["mode"]
        folder = registry / model / mode
        source = a.inputs / "fresh_causal_v1/registries" / model / mode
        for file, expected in ((source / "manifest.json", cell["source_registry_sha256"]),
            (source / "adapted_generations.jsonl", cell["source_generations_sha256"]),
            (folder / "queries.json", cell["queries_sha256"]), (folder / "confirmation.json", cell["confirmation_sha256"])):
            require(sha(file) == expected, "Query/input hash mismatch")
        queries = read(folder / "queries.json")
        by_id = {q["query_id"]: q for q in queries}
        require(len(by_id) == len(queries), "Duplicate query ID")
        inputs = {r["request_id"]: r for r in jsonl(source / "adapted_generations.jsonl")}
        for q in queries:
            row = inputs[q["request_id"]]
            require(json_sha(row) == q["row_sha256"], "Query source row mismatch")
            require(q["frozen_anchor_role"] == cfg["anchor_roles"][model][mode], "Anchor role mismatch")
            count = q["query_output_token_index"] + 1
            ids = row["input_ids"] + row["output_token_ids"][:count]
            mask = row["attention_mask"] + [1] * count
            require(q["query_position"] == len(ids) - 1 and json_sha(ids) == q["input_ids_sha256"] and json_sha(mask) == q["attention_mask_sha256"], "Literal query input reconstruction mismatch")
            span = next(s for s in row["prompt_record_spans"] if s["city"] == q["target_city"])
            require([span["start"], span["end"]] == q["source_record_span"], "Retrieval target span mismatch")
        local = stage / "jobs/localize" / model / mode
        observations, ranking = jsonl(local / "observations.jsonl"), read(local / "ranking.json")
        loc_grid = unique_grid(observations, ["query_id"], {(q["query_id"],) for q in queries if q["split"] == "discovery"})
        for observation in loc_grid.values():
            query = by_id[observation["query_id"]]
            require(observation["query_position"] == query["query_position"] and observation["source_record_span"] == query["source_record_span"], "Localization position mismatch")
        recomputed = compiler.rank_heads(observations)
        require([(r["layer"], r["head"]) for r in recomputed] == [(r["layer"], r["head"]) for r in ranking], "Discovery head order mismatch")
        max_error = max(abs(x["score"] - y["score"]) for x, y in zip(recomputed, ranking))
        require(max_error <= 1e-12, "Discovery score mismatch")
        bank_folder = stage / "banks" / model / mode
        bank, plan = read(bank_folder / "banks.json"), read(bank_folder / "plan.json")
        require(sha(bank_folder / "banks.json") == plan["banks_sha256"] and sha(local / "observations.jsonl") == plan["discovery_observations_sha256"], "Bank evidence mismatch")
        require(plan["stage_manifest_sha256"] == sha(stage / "manifest.json") and not plan["uses_confirmation_outcomes"], "Bank provenance mismatch")
        require(compiler.freeze_banks(ranking, cfg["dose_grid"][model], reg["confirmation_seeds"], salt=f'{cfg["random_seed"]}/{model}/{mode}', repeats=cfg["random_repeats"]) == bank, "Frozen random bank not reproducible")
        confirmation = read(folder / "confirmation.json")
        expected_tasks = []
        for q in confirmation:
            base = {"query_id": q["query_id"], "request_id": q["request_id"], "seed": q["seed"]}
            expected_tasks.append({**base, "k": 0, "condition": "clean", "repeat": 0, "heads": []})
            for k in bank["ks"]:
                expected_tasks.append({**base, "k": k, "condition": "selected_bank", "repeat": 0, "heads": bank["selected_heads"][:k]})
                expected_tasks.extend({**base, "k": k, "condition": bank["random_control_by_k"][str(k)], "repeat": repeat, "heads": heads}
                    for repeat, heads in enumerate(bank["random_banks"][str(q["seed"])][str(k)]))
        require(plan["tasks"] == expected_tasks, "Behavior task grid not generated from frozen banks")
        coverage = {}
        for phase in ("behavior_smoke", "formal"):
            job = stage / "jobs" / phase / model / mode
            status, runtime = read(job / "status.json"), read(job / "runtime.json")
            require(status["status"] == "COMPLETE" and sha(job / "trials.jsonl") == status["trials_sha256"], "Behavior job incomplete or changed")
            require(runtime["stage_manifest_sha256"] == sha(stage / "manifest.json") and runtime["backend"] == cfg["attention_backend"], "Behavior runtime mismatch")
            tasks = plan["smoke_tasks"] if phase == "behavior_smoke" else expected_tasks
            trials = jsonl(job / "trials.jsonl")
            grid = unique_grid(trials, ["query_id", "dose_k", "condition", "repeat"],
                {(t["query_id"], t["k"], t["condition"], t["repeat"]) for t in tasks})
            require(len(grid) == status["completed_rows"] == status["expected_rows"], "Behavior row count mismatch")
            for task in tasks:
                row = grid[task["query_id"], task["k"], task["condition"], task["repeat"]]
                query = by_id[task["query_id"]]
                worker.validate_trial(row, task, query, row["prefill_zeroed_head_l2_by_layer"])
                require(row["status"] == "ok" and row["trial_complete"] and row["model_label"] == model and row["mode"] == mode, "Behavior row identity/status mismatch")
                require(row["plan_sha256"] == sha(bank_folder / "plan.json") and row["request_id"] == query["request_id"], "Behavior query/plan hash mismatch")
                require(row["free_generation_max_new_tokens"] == cfg["max_new_tokens"] and row["head_ablation_tensor_site"] == "attention_output_projection_input_pre_o", "Intervention budget/site mismatch")
                verify_generation(row, cfg["max_new_tokens"])
                ids, target = row["generated_token_ids"], query["target_token_ids"]
                offset = query["target_output_token_start"] - query["query_output_token_index"] - 1
                require(offset >= 0, "Target city precedes query")
                at_offset = ids[offset:offset + len(target)] == target
                prefix = offset == 0 and at_offset
                require(row["generated_target_city_exact_at_registered_path_offset"] == at_offset and row["generated_exact_target_city_token_prefix"] == prefix, "Target token-prefix audit mismatch")
                scored = score_city(row["completion_text"], expected_city=query["target_city"],
                    gold_cities=[r["city"] for r in inputs[query["request_id"]]["prompt_record_spans"]], exact_target_prefix=prefix)
                require(all(row[k] == v for k, v in scored.items()), "Frozen semantic-city parser mismatch")
                if phase == "formal":
                    prediction = parse_total(row["completion_text"])
                    all_derived.append({"model": model, "mode": mode, "query_id": query["query_id"], "seed": query["seed"],
                        "gold_count": query["gold_count"], "dose_k": task["k"], "condition": task["condition"], "repeat": task["repeat"],
                        "next_city_failure": int(not scored["correct_next_needle"]), "final_exact_count_failure": int(prediction != query["gold_count"]),
                        "prediction": prediction, "invalid_count": prediction not in range(1, 11), "truncated": row["generation_truncated"]})
            coverage[phase] = {"rows": len(trials), "truncated": sum(r["generation_truncated"] for r in trials)}
        derived = [r for r in all_derived if r["model"] == model and r["mode"] == mode]
        clean = {r["seed"]: r for r in derived if r["condition"] == "clean"}
        require(set(clean) == set(reg["confirmation_seeds"]), "Clean source-seed coverage")
        stats = []
        for k in bank["ks"]:
            selected = {r["seed"]: r for r in derived if r["dose_k"] == k and r["condition"] == "selected_bank"}
            control = bank["random_control_by_k"][str(k)]
            random = {s: [r for r in derived if r["seed"] == s and r["dose_k"] == k and r["condition"] == control] for s in clean}
            require(set(selected) == set(clean) and all(len(v) == cfg["random_repeats"] for v in random.values()), "Paired effect coverage")
            metrics = {}
            for metric in cfg["primary_metrics"]:
                terms = {name: [] for name in ["clean", "selected", "random", *cfg["contrasts"]]}
                for seed, baseline in clean.items():
                    c, s, r = baseline[metric], selected[seed][metric], float(np.mean([t[metric] for t in random[seed]]))
                    for name, value in {"clean": c, "selected": s, "random": r, "selected_minus_clean": s-c,
                                        "random_minus_clean": r-c, "selected_minus_random": s-r}.items():
                        terms[name].append((seed, value))
                metrics[metric] = {name: seed_mean(values) for name, values in terms.items()}
            stats.append({"dose_k": k, "random_control": control, "metrics": metrics})
        cells.append({"model": model, "mode": mode, "audit": "PASS", "coverage": coverage, "discovery_queries": len(observations),
            "confirmation_counts": [q["gold_count"] for q in confirmation], "source_seed_count": len(clean),
            "ranking_max_absolute_score_error": max_error, "random_capacity_fallbacks": bank["layer_matching_infeasible"], "dose_response": stats})
        print(json.dumps({"model": model, "mode": mode, "audit": "PASS", "coverage": coverage,
            "largest_dose": {metric: stats[-1]["metrics"][metric]["selected_minus_clean"]["estimate"] for metric in cfg["primary_metrics"]}}), flush=True)
    save(out/'retrieve_audit.json',dict(status='PASS',cells=cells))
    save(out/'retrieve_outcomes.json',all_derived)

def answer(root,out):
    stage,regroot=root/'answer',root/'answer_registry';inputs=root
    man,reg,cfg=read(stage/'manifest.json'),read(regroot/'manifest.json'),read(stage/'protocol.json')
    require(reg['clean_regeneration_required'] is False and not reg['intervention_outcomes_accessed'],'Unexpected outcome eligibility')
    check(regroot/'manifest.json',man['registry_manifest_sha256']);check(stage/'protocol.json',man['protocol_sha256'])
    for f,h in man['entrypoints_sha256'].items():check(stage/'code'/f,h)
    compiler=module(root/'code/scripts/prepare_enumeration_fresh_answer_patch.py','_new_answer_compiler')
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
        kept=set();added=set(pairmap)
        sources=[('smoke',stage/'jobs/smoke'/model/mode,pairs[:2]),('formal',stage/'jobs/formal'/model/mode,pairs)]
        formal=[]
        for phase,job,chosen in sources:
            status=read(job/'status.json');runtime=read(job/'runtime.json');check(job/'trials.jsonl',status['trials_sha256'])
            require(status['status']=='COMPLETE' and runtime['layers']==list(range(depth)) and runtime['max_new_tokens']==16,'Answer runtime/grid changed')
            expected_stage=old/'fresh_answer_patch_gpu_v2' if phase=='reused' else stage
            require(runtime['stage_manifest_sha256']==sha(expected_stage/'manifest.json'),'Answer runtime manifest mismatch')
            reference=read(root/'update'/model/'cells'/model/mode/'capture_runtime.json')
            for key in ['model_revision','model_source_sha256','dtype','backend','torch','transformers']:
                require(runtime[key]==reference[key],f'Answer numerical setting changed: {key}')
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
    require(len(allrows)==6240 and new_formal==6240 and new_smoke==312,'Coverage mismatch')
    save(out/'answer_audit.json',dict(status='PASS',cells=cells))
    save(out/'answer_outcomes.json',allrows)

def update(root,out):
    import pandas as pd
    from realistic_niah_v6.update_n10 import select_discovery
    from realistic_niah_v5.trace_stratified_geometry import grouped_discovery_cv_metrics, _fit_projection_and_predict
    parser=module(root/'code/src/realistic_niah_v5/same_site_progress_transplant.py','_default_update_parser').generated_bullet_city_ordinals
    reports=[]
    for model in ['Qwen3-8B','Gemma4-E4B']:
        mode='enumeration_bullet';stage=root/'update'/model;cell=stage/'cells'/model/mode
        cohort=read(cell/'cohort.json');manifest=read(stage/'manifest.json');selection=read(stage/'selection_manifest.json')
        require(not cohort['selection_used_final_correctness'] and not cohort['intervention_outcomes_accessed'],'Outcome-selected cohort')
        for name,key in [('selected_generations.jsonl','inputs_sha256'),('geometry.json','geometry_sha256'),('candidate_ledger.json','ledger_sha256')]:check(cell/name,cohort[key])
        candidates=manifest['initial_discovery_seeds']+(manifest['reserve_seeds'] if model in manifest['reserve_models'] else [])
        require(select_discovery(read(cell/'candidate_ledger.json'),candidates,manifest['all_original_confirmation_candidates'])==cohort['discovery_seeds'],'Discovery selection mismatch')
        reg=Path(manifest['cells'][0]['registry']);ledger=read(reg/'ledger.json');lookup={r['seed']:r for r in ledger if r['gold_count']==10}
        chosen=[s for s in manifest['all_original_confirmation_candidates'] if lookup[s]['update_eligible']][:10]
        require(chosen==cohort['confirmation_seeds'],'Confirmation selection mismatch')
        layer=validate_selection(selection,model,mode,chosen)
        saved=read(cell/'selection.json');check(cell/'discovery_states.npz',saved['states_sha256']);check(cell/'discovery_metadata.csv',saved['metadata_sha256'])
        states=np.load(cell/'discovery_states.npz')['states'];metadata=pd.read_csv(cell/'discovery_metadata.csv')
        metrics=pd.read_csv(cell/'discovery_layer_metrics.csv').to_dict('records');recomputed=[]
        require(set(metadata['seed'])==set(cohort['discovery_seeds']) and len(metadata)==200,'Wrong discovery tensor population')
        for row in metrics:
            value=grouped_discovery_cv_metrics(states[:,int(row['layer_zero_based'])],metadata,np.arange(1,11),pca_dim=16,random_state=0,folds=5,pca_whiten=True)
            for key in ['discovery_oof_ncc_balanced_accuracy','discovery_oof_logistic_balanced_accuracy']:
                require(abs(value[key]-row[key])<1e-10,'Discovery CV replay mismatch')
            recomputed.append(dict(row,**value))
        require(rank_layers(recomputed,model)[0]['layer_one_based']==layer+1,'Layer selection mismatch')
        conf=read(cell/'confirmation_readout.json');check(stage/'selection_manifest.json',conf['selection_manifest_sha256'])
        truth=pd.read_csv(cell/'confirmation_metadata.csv');cstates=np.load(cell/'confirmation_states.npz')['states']
        logistic,ncc,_,_= _fit_projection_and_predict(states[:,layer],metadata['occurrence'].to_numpy(dtype=int),cstates[:,0],np.arange(1,11),pca_dim=16,random_state=0,pca_whiten=True)
        require(set(truth['seed'])==set(chosen) and len(truth)==100,'Wrong confirmation tensor population')
        require(abs(float((ncc==truth['occurrence']).mean())-conf['ncc_balanced_accuracy'])<1e-10,'Confirmation readout mismatch')
        inputs={r['seed']:r for r in jsonl(reg/'adapted_generations.jsonl') if r['gold_count']==10}
        rows=[];widths=set()
        for phase in ['smoke','formal']:
            source=stage/phase;runtime=read(source/'runtime.json');status=read(source/'status.json');reference=read(cell/'capture_runtime.json')
            require(status['status']=='COMPLETE' and status['completed_jobs']==18,'Incomplete Update')
            check(reg/'manifest.json',runtime['registry_sha256']);check(stage/'cohorts.json',runtime['cohorts_sha256'])
            check(root/'code/scripts/run_enumeration_fresh_update.py',runtime['runner_sha256'])
            for key in ['model_revision','model_source_sha256','dtype','backend','torch','transformers']:
                require(runtime[key]==reference[key],'Update runtime mismatch '+key)
            require(runtime['patch_layer_zero_based']==layer and runtime['attention_readout'] is False and runtime['all_three_conditions_generate'],'Wrong update execution')
            seeds=chosen[:1] if phase=='smoke' else chosen
            for k in [4,6,8]:
                for direction in ['forward','backward']:
                    for scope in SCOPES:
                        folder=source/f'k{k}_{direction}_{scope}';audit=read(folder/'technical_audit.json')
                        require(audit['status']=='PASS' and audit['backend_after']=='sdpa','Update technical failure')
                        for f,key in [('trials.jsonl','trials_sha256'),('raw_generations.jsonl','raw_sha256'),('prefill_hooks.jsonl','hooks_sha256')]:check(folder/f,audit[key])
                        trials,raw,hooks=[jsonl(folder/f) for f in ['trials.jsonl','raw_generations.jsonl','prefill_hooks.jsonl']]
                        matched=matched_trials(trials,seeds);require(len(raw)==len(trials) and len(hooks)==2*len(trials),'Raw/hook count')
                        geometry={r['seed']:r for r in jsonl(folder/'geometry_audit.jsonl')};hashes={}
                        for i,(trial,generation) in enumerate(zip(trials,raw)):
                            g=geometry[trial['seed']];cities=[r['city'] for r in inputs[trial['seed']]['gold_records']]
                            require(trial['request_id']==inputs[trial['seed']]['request_id'],'Wrong request')
                            require(g['aligned_absolute_site']==trial['shared_commit_position'] and g['effective_patch_width']==trial['patch_width'] and g['endpoint_aligned'] and not g['hidden_state_resampling'],'Geometry mismatch')
                            require(g['deletion_avoids_prompt_records'] and g['deletion_avoids_special_tokens'] and g['deletion_before_answer'],'Protected prefix deleted')
                            role='donor' if trial['condition']=='native_donor' else 'receiver'
                            require(g[role+'_commit_token_id']==trial['shared_commit_token_id'],'Wrong commit token')
                            verify_trial(trial,generation,hooks[2*i:2*i+2],cities=cities,parser=parser,layer=layer,k=k,direction=direction,scope=scope,max_tokens=96)
                            hashes[trial['seed'],trial['condition']]=hooks[2*i]['input_ids_sha256']
                            if phase=='formal':rows.append(dict(trial,audit_scope=scope,audit_direction=direction))
                            if scope=='item_span':widths.add(trial['patch_width'])
                        for seed in seeds:require(hashes[seed,'receiver_self']==hashes[seed,'donor_to_receiver'],'Unpaired input')
        require(len(rows)==540,'Wrong Update total')
        reports.append(dict(model=model,mode=mode,audit='PASS',formal_rows=len(rows),layer_one_based=layer+1,cohort=cohort,discovery=saved['selected'],confirmation=conf,span_widths=sorted(widths),groups=summarize(rows,bootstrap=read(root/'fresh_v1/protocol.json')['bootstrap'])))
        print('UPDATE PASS',model,flush=True)
    save(out/'update_audit.json',dict(status='PASS',cells=reports))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic()
    a.output.mkdir(parents=True,exist_ok=False)
    inv=read(a.root/'backup_manifest.json')
    for name,record in inv['files'].items():
        f=a.root/name;require(f.stat().st_size==record['bytes'],'File size mismatch');check(f,record['sha256'])
    print('ALL INVENTORY HASHES PASS',len(inv['files']),flush=True)
    cfg=read(a.root/'fresh_v1/protocol.json');parse_total=frozen_parser(a.root)
    cells=[]
    for m in cfg['models']:
        c=audit_read(a.root,m,'enumeration_bullet',parse_total,cfg,result_folder=a.root/'read/formal'/m/'enumeration_bullet');cells.append(c);print('READ PASS',m,flush=True)
    save(a.output/'read_audit.json',dict(status='PASS',cells=cells))
    retrieve(a.root,a.output);answer(a.root,a.output);update(a.root,a.output)
    save(a.output/'audit.json',dict(status='PASS',seconds=time.monotonic()-t,inventory_sha256=sha(a.root/'backup_manifest.json'),verified_files=len(inv['files']),script_sha256=sha(__file__),limitations=['Saved token IDs, raw text, geometry and hooks audited; GPU rerun/tokenizer decode not repeated.','Default confirmation samples are reused, not fresh independent confirmation.','Pointwise intervals use ten seed clusters and condition on frozen choices.']))
if __name__=='__main__':main()
