"""Fresh Native-aligned layer selection and continuation confirmation; no training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

import run_v58_commit_query as base
from run_v58_aligned_item_query import aligned_donor, capture

SCOPES = {"item_end_w1": 1, "item_span_w2": 2}
METRICS = ["donor_marker_adoption", "receiver_marker_retention", "donor_continuation_adoption",
           "receiver_continuation_retention", "donor_prefix_h2", "donor_prefix_h3", "donor_prefix_h4",
           "donor_route_adoption", "final_correct", "generated_marker_count"]


def choose_layer(frame):
    """Native 95%-of-peak earliest-layer rule, with identical markers undefined."""
    distinct=frame.loc[frame.successor_identity_distinct]
    if distinct.empty:
        return {"selected_layer":None,"status":"no identifiable discovery transitions","layer_summaries":[]}
    pivot=distinct.pivot(index=["prompt_sha256","pair_id","layer","offset"],columns="condition",values="donor_vs_receiver_marker_logodds")
    pivot["effect"]=pivot.full_donor_patch-pivot.clean
    summaries=[]
    for layer,group in pivot.reset_index().groupby("layer"):
        means=group.groupby("prompt_sha256").effect.mean()
        f=group.loc[group.offset.eq(1),"effect"]
        b=group.loc[group.offset.eq(-1),"effect"]
        summaries.append({"layer":int(layer),"pairs":len(group),"prompts":len(means),
                          "forward_pairs":len(f),"backward_pairs":len(b),
                          "forward_median":float(f.median()) if len(f) else None,
                          "backward_median":float(b.median()) if len(b) else None,
                          "median_prompt_mean":float(means.median()),
                          "eligible":bool(layer<4 and len(f)>0 and len(b)>0 and f.median()>0 and b.median()>0)})
    eligible=[r for r in summaries if r["eligible"]]
    if not eligible:
        return {"selected_layer":None,"status":"no bidirectionally positive layer","layer_summaries":summaries}
    peak=max(r["median_prompt_mean"] for r in eligible)
    selected=min(r["layer"] for r in eligible if r["median_prompt_mean"]>=.95*peak)
    return {"selected_layer":selected,"status":"selected","peak":peak,"threshold":.95*peak,"layer_summaries":summaries}


def continuation_scores(generated, donor, receiver):
    """Input-defined identifiability; generation errors remain zero, never drop."""
    limit=min(4,len(donor),len(receiver))
    q=next((h for h in range(1,limit+1) if donor[:h]!=receiver[:h]),None)
    result={"distinguishing_horizon":q or 0,
            "donor_continuation_adoption":float(generated[:q]==donor[:q]) if q else np.nan,
            "receiver_continuation_retention":float(generated[:q]==receiver[:q]) if q else np.nan}
    for h in [2,3,4]:
        eligible=len(donor)>=h and len(receiver)>=h and donor[:h]!=receiver[:h]
        result[f"donor_prefix_h{h}"]=float(generated[:h]==donor[:h]) if eligible else np.nan
    return result


def parse_markers(tokens):
    result=[]
    for i in range(0,len(tokens)-1,2):
        if tokens[i]!="<Sep>" or not tokens[i+1].startswith("<CH_"):
            break
        result.append(tokens[i+1])
    return result


@torch.inference_mode()
def score_discovery(model,vocab,cases,layer,batch_size):
    rows=[]
    for start in range(0,len(cases),batch_size):
        batch=cases[start:start+batch_size]
        seqs=[c["prefix"]+[vocab.token_to_id["<Sep>"]] for c in batch]
        with base.patch_batch(model,layer,batch,next(model.parameters()).device):
            output=base.forward(model,vocab,seqs)  # No attention or free generation.
        for i,c in enumerate(batch):
            logits=output.logits[i,len(seqs[i])-1].float()
            rows.append({**c["meta"],"layer":layer,"condition":c["condition"],
                         "donor_vs_receiver_marker_logodds":float(logits[c["donor_id"]]-logits[c["receiver_id"]]),
                         **c["norms"]})
        del output
    return rows


def prepare_pairs(chosen,discovery_keys,vocab):
    pairs=[]
    for e in chosen:
        item=base.render_v20(e,vocab,"thinking")
        split="discovery" if e.prompt_sha256 in discovery_keys else "confirmation"
        for donor_k in ([6] if split=="discovery" else [4,6,8]):
            for offset in [1,-1]:
                receiver_k=donor_k-offset
                meta=base.pair_metadata(e,item,receiver_k,offset,split)
                adjusted=base.render_v20(aligned_donor(e,offset),vocab,"thinking")
                assert adjusted.spans.trace_marker_positions[donor_k-1]==meta["commit_position"]
                meta["donor_commit_position"]=meta["commit_position"]
                meta["donor_prompt_length"]=len(e.seq_tokens)-2*offset
                meta["donor_future_markers"]=json.dumps(list(e.needle_markers[donor_k:]))
                meta["receiver_future_markers"]=json.dumps(list(e.needle_markers[receiver_k:]))
                pairs.append({"meta":meta,"item":item})
    return pairs


def make_cases(pairs,states,bases,vocab,scope,layer,condition):
    span=SCOPES[scope]
    cases=[]
    for pair in pairs:
        m,item=pair["meta"],pair["item"]
        r=states[(m["prompt_sha256"],0)][layer][m["receiver_k"]-1].reshape(2,-1)[-span:].flatten()
        d=states[(m["prompt_sha256"],m["offset"])][layer][m["donor_k"]-1].reshape(2,-1)[-span:].flatten()
        vectors,norms=base.make_condition_states(r,d,bases[(scope,layer)],m["pair_id"],layer)
        cases.append({"meta":m,"condition":condition,"vector":vectors[condition],"patch_span":span,
                      "prefix":item.input_ids[:m["commit_position"]+1],"positions":item.prompt_needle_positions,"norms":norms,
                      "delta_norm":float((vectors[condition]-r).norm()),
                      "receiver_id":vocab.token_to_id[m["receiver_next_marker"]],"donor_id":vocab.token_to_id[m["donor_next_marker"]]})
    return cases


def descriptive_audit(rollouts):
    rows=[]
    for condition,group in rollouts.groupby("condition"):
        for metric in METRICS:
            sub=group.loc[group[metric].notna()]
            if metric in ["donor_marker_adoption","receiver_marker_retention"]:
                sub=sub.loc[sub.successor_identity_distinct]
            rows.append({"condition":condition,"metric":metric,"pairs":len(sub),"prompts":sub.prompt_sha256.nunique(),
                         "sum":float(sub[metric].sum()),"pair_mean":float(sub[metric].mean()) if len(sub) else None,
                         "prompt_mean":float(sub.groupby("prompt_sha256")[metric].mean().mean()) if len(sub) else None})
        first=group.loc[group.successor_identity_distinct & group.donor_marker_adoption.eq(1)]
        for h in [2,3,4]:
            sub=first.loc[first[f"donor_prefix_h{h}"].notna()]
            rows.append({"condition":condition,"metric":f"conditional_h{h}_given_first_transfer", "pairs":len(sub),
                         "prompts":sub.prompt_sha256.nunique(),"sum":float(sub[f"donor_prefix_h{h}"].sum()),
                         "pair_mean":float(sub[f"donor_prefix_h{h}"].mean()) if len(sub) else None,
                         "prompt_mean":None})
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--panel-registry",type=Path)
    parser.add_argument("--frozen-sites",type=Path)
    args=parser.parse_args()
    out=args.output
    if (out/"plan.json").exists():
        raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    cfg,vocab,_,selection,reporting=base._load_bundle(args.run_dir,device="cuda")
    assert cfg.version=="v58" and cfg.trace_format=="separator" and cfg.n_layer==4
    source=args.run_dir/"analysis/behavior_confirmation_v58/examples.jsonl"
    examples=[base.example_from_dict(json.loads(s)) for s in source.read_text().splitlines() if s.strip()]
    previous=args.run_dir/"analysis/v58_commit_query_20260905/frozen_pairs.csv"
    old_hashes=set(pd.read_csv(previous).prompt_sha256)
    old_hashes |= set(pd.read_csv(args.run_dir/"analysis/v58_free_running_sufficiency/progress_rollout_confirmation.csv").prompt_sha256)
    old_keys={(e.set_id,e.corpus_start) for e in examples if e.prompt_sha256 in old_hashes}
    old_keys |= {(e.set_id,e.corpus_start) for e in selection+reporting}
    eligible=sorted((e for e in examples if e.count==10 and e.prompt_sha256 not in old_hashes and (e.set_id,e.corpus_start) not in old_keys),key=lambda e:e.prompt_sha256)
    if args.panel_registry:
        registry=pd.read_csv(args.panel_registry)
        registry=registry.loc[registry['count'].eq(10)].copy()
        registry['split_order']=registry.split.map({'discovery':0,'confirmation':1})
        registry=registry.sort_values(['split_order','block'])
        lookup={e.prompt_sha256:e for e in examples}
        chosen=[lookup[k] for k in registry.key]
        assert registry.groupby('split').size().to_dict()=={'confirmation':10,'discovery':20}
    else:
        if len(eligible)<80:
            raise ValueError(f"Need 80 fresh mechanistic prompts, found {len(eligible)}")
        chosen=eligible[:80]
    if args.frozen_sites:
        sites=json.loads(args.frozen_sites.read_text())
        base.BANK=[tuple(row[:2]) for row in sites['ranking']['targeted'][:4]]
    discovery_keys={e.prompt_sha256 for e in chosen[:20]}
    confirm_keys={e.prompt_sha256 for e in chosen[20:]}
    assert len(discovery_keys)==20 and len(confirm_keys)==len(chosen)-20 and not discovery_keys&confirm_keys
    assert not ({(e.set_id,e.corpus_start) for e in chosen}&old_keys)
    pairs=prepare_pairs(chosen,discovery_keys,vocab)
    pd.DataFrame([p["meta"] for p in pairs]).to_csv(out/"frozen_pairs.csv",index=False)
    plan={"question":"Native-aligned donor-directed continuation sufficiency", "scopes":SCOPES,
          "selection_rule":"distinct-marker discovery pairs; positive forward/backward medians; earliest >=95% peak median prompt-mean log-odds shift; final block excluded",
          "discovery_prompts":20,"confirmation_prompts":len(confirm_keys),"available_fresh_mechanistic_prompts":len(eligible),
          "panel_registry_sha256":base.digest(args.panel_registry) if args.panel_registry else None,
          "discovery_donor_k":[6],"confirmation_donor_k":[4,6,8],"receiver_k":"donor_k - offset", "offsets":[1,-1],
          "primary_metric":"donor_marker_adoption on distinct-successor pairs", "additional_metric":"shortest distinguishing future prefix through <=4 markers",
          "conditions":base.CONDITIONS,"bank":base.BANK,"bootstrap_unit":"prompt","bootstrap_resamples":10000,
          "new_mechanistic_data_not_new_behavior_test":True,"old_prompt_overlap":0,"old_canonical_key_overlap":0,
          "sample_plan_sha256":base.digest(out/"frozen_pairs.csv"),"source_examples_sha256":base.digest(source),
          "script_sha256":base.digest(Path(__file__)),"base_helper_sha256":base.digest(Path(base.__file__)),
          "created_unix":time.time()}
    base.write_json(out/"plan.json",plan)
    print(f"FROZEN: 20 discovery + {len(confirm_keys)} confirmation prompts ({len(pairs)} pairs)",flush=True)
    _,_,_,_,model=base.load_v20_checkpoint_model(args.run_dir,"rope","thinking",step=cfg.train_steps,device="cuda")
    def capture_examples(group):
        return capture(model,vocab,[(e.prompt_sha256,offset,e if offset==0 else aligned_donor(e,offset)) for e in group for offset in [0,1,-1]],args.batch_size)
    states=capture_examples(chosen[:20])  # Confirmation states not captured yet.
    bases={}
    for scope,span in SCOPES.items():
        for layer in base.LAYERS:
            cloud=torch.stack([states[(e.prompt_sha256,0)][layer][:9].reshape(9,2,-1)[:,-span:].flatten(1) for e in chosen[:20]])
            centroids=cloud.mean(0)
            bases[(scope,layer)]=torch.linalg.svd(centroids-centroids.mean(0),full_matrices=False).Vh[:3]
    np.savez(out/"frozen_bases.npz",**{f"{s}_L{l}":v.numpy() for (s,l),v in bases.items()})
    discovery_pairs=[p for p in pairs if p["meta"]["split"]=="discovery"]
    selections={}
    validation={}
    for scope in SCOPES:
        root=out/scope; root.mkdir()
        rows=[]
        for layer in base.LAYERS:
            for condition in ["clean","self_patch","full_donor_patch"]:
                cases=make_cases(discovery_pairs,states,bases,vocab,scope,layer,condition)
                rows.extend(score_discovery(model,vocab,cases,layer,args.batch_size))
        frame=pd.DataFrame(rows)
        frame.to_csv(root/"discovery_trials.csv",index=False)
        pivot=frame.pivot(index=["pair_id","layer"],columns="condition",values="donor_vs_receiver_marker_logodds")
        err=float((pivot.clean-pivot.self_patch).abs().max())
        assert err<1e-4,err
        null=float((pivot.xs(4,level="layer").full_donor_patch-pivot.xs(4,level="layer").clean).abs().max())
        assert null<1e-4,null
        validation[scope]={"discovery_self_max_error":err,"L4_null_max_error":null}
        selections[scope]=choose_layer(frame)
        print(scope,json.dumps(selections[scope]),flush=True)
    selection_record={"scopes":selections,"selection_completed_unix":time.time(),"confirmation_inference_started":False,
                      "plan_sha256":base.digest(out/"plan.json")}
    base.write_json(out/"selected_layers.json",selection_record)
    selected_hash=base.digest(out/"selected_layers.json")
    print("LAYERS FROZEN; starting fresh confirmation",flush=True)
    states.update(capture_examples(chosen[20:]))
    confirm_pairs=[p for p in pairs if p["meta"]["split"]=="confirmation"]
    for scope,selected in selections.items():
        layer=selected["selected_layer"]
        if layer is None:
            continue
        root=out/scope
        local,rollouts,routes=[],[],[]
        for condition in base.CONDITIONS:
            cases=make_cases(confirm_pairs,states,bases,vocab,scope,layer,condition)
            local.extend(base.evaluate_local(model,vocab,cases,layer,args.batch_size))
            detail,route=base.evaluate_rollout(model,vocab,cases,layer,args.batch_size,28)
            for row in detail:
                generated=parse_markers(row["continuation_tokens"].split())
                donor=json.loads(row["donor_future_markers"]); receiver=json.loads(row["receiver_future_markers"])
                row.update(continuation_scores(generated,donor,receiver))
            rollouts.extend(detail); routes.extend(route)
            pd.DataFrame(rollouts).to_csv(root/"rollout_trials.csv",index=False)
            pd.DataFrame(routes).to_csv(root/"rollout_routing.csv",index=False)
            print(f"confirmation {scope} L{layer} {condition} completed",flush=True)
        local_frame=pd.DataFrame(local); roll_frame=pd.DataFrame(rollouts)
        local_frame.to_csv(root/"local_trials.csv",index=False)
        base.summarize(local_frame,["routing_y","qk_margin_mean","donor_vs_receiver_marker_logodds"],root,"local")
        base.summarize(roll_frame,METRICS,root,"rollout")
        descriptive_audit(roll_frame).to_csv(root/"continuation_audit.csv",index=False)
        pivot=roll_frame.pivot(index="pair_id",columns="condition",values="continuation_tokens")
        same=bool((pivot.clean==pivot.self_patch).all())
        assert same
        assert len(roll_frame)==len(confirm_pairs)*len(base.CONDITIONS)
        validation[scope].update({"clean_self_rollouts_equal":same,"rollout_rows":len(roll_frame),
                                  "hit_token_cap_rows":int(roll_frame.hit_token_cap.sum()),
                                  "identifiable_first_marker_pairs":int(roll_frame.loc[roll_frame.condition.eq("clean"),"successor_identity_distinct"].sum()),
                                  "identifiable_continuation_pairs":int(roll_frame.loc[roll_frame.condition.eq("clean"),"donor_continuation_adoption"].notna().sum())})
    assert base.digest(out/"selected_layers.json")==selected_hash
    checkpoint=base.checkpoint_source(args.run_dir,cfg.train_steps)
    manifest={"status":"complete","source_run":str(args.run_dir),"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,
              "checkpoint":str(checkpoint),"checkpoint_sha256":base.digest(checkpoint),"checkpoint_step":cfg.train_steps,
              "selection_sha256":selected_hash,"validation":validation,"completed_unix":time.time(),
              "files":{str(p.relative_to(out)):base.digest(p) for p in out.rglob("*") if p.is_file()}}
    base.write_json(out/"manifest.json",manifest)
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=="__main__":
    main()
