"""Position-aligned two-token item-scope sensitivity, using the frozen first assay pairs.

Specified after inspecting the single-token assay; not a new independent confirmation.
No training or receiver trace edits. See docs/v58_commit_query_protocol.md.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

import run_v58_commit_query as base


def aligned_donor(example, offset):
    """Shorten/extend only non-needle filler so donor item k+offset is at receiver k."""
    tokens = list(example.seq_tokens)
    needle_positions = set(example.needle_positions)
    filler = [i for i in range(len(tokens)) if i not in needle_positions]
    if len(filler) < 2:
        raise ValueError("Need two non-needle filler tokens")
    if offset == 1:
        removed = set(filler[-2:])
        new_tokens = [t for i,t in enumerate(tokens) if i not in removed]
        new_positions = tuple(p-sum(i<p for i in removed) for p in example.needle_positions)
    elif offset == -1:
        new_tokens = tokens + [tokens[filler[-1]]] * 2
        new_positions = example.needle_positions
    else:
        raise ValueError(offset)
    assert tuple(new_tokens[i] for i in new_positions) == example.needle_markers
    assert len(new_tokens) == len(tokens)-2*offset
    # Retain original prompt identity as the clustering key; these are intervention inputs.
    return replace(example, seq_tokens=new_tokens, needle_positions=new_positions)


def capture(model, vocab, examples, batch_size):
    states = {}
    for start in range(0,len(examples),batch_size):
        batch=examples[start:start+batch_size]
        items=[base.render_v20(e,vocab,"thinking") for _,_,e in batch]
        seqs=[x.input_ids[:x.spans.trace_marker_positions[-1]+1] for x in items]
        output=base.forward(model,vocab,seqs,hidden=True)
        for i,((key,offset,_),item) in enumerate(zip(batch,items)):
            indices=[[p-1,p] for p in item.spans.trace_marker_positions]
            states[(key,offset)]={l:output.hidden_states[l][i,indices].float().flatten(1).cpu() for l in base.LAYERS}
        del output
    return states


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--original-results",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--batch-size",type=int,default=16)
    args=parser.parse_args()
    out=args.output
    if (out/"plan.json").exists():
        raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    cfg,vocab,_,_,_=base._load_bundle(args.run_dir,device="cuda")
    frozen=pd.read_csv(args.original_results/"frozen_pairs.csv")
    source=args.run_dir/"analysis/behavior_confirmation_v58/examples.jsonl"
    examples={e.prompt_sha256:e for e in (base.example_from_dict(json.loads(s)) for s in source.read_text().splitlines() if s.strip())}
    keys=list(frozen.prompt_sha256.drop_duplicates())
    discovery=set(frozen.loc[frozen.split.eq("discovery"),"prompt_sha256"])
    shifted={(key,offset):(examples[key] if offset==0 else aligned_donor(examples[key],offset)) for key in keys for offset in [0,-1,1]}
    pairs=[]
    for meta in frozen.to_dict("records"):
        key,offset=meta["prompt_sha256"],meta["offset"]
        item=base.render_v20(examples[key],vocab,"thinking")
        donor=base.render_v20(shifted[(key,offset)],vocab,"thinking")
        assert donor.spans.trace_marker_positions[meta["donor_k"]-1]==meta["commit_position"]
        assert len(shifted[(key,offset)].seq_tokens) in [254,258]
        meta.update({"donor_commit_position":meta["commit_position"],"donor_prompt_length":len(shifted[(key,offset)].seq_tokens),"patch_span":2})
        pairs.append({"meta":meta,"item":item})
    pd.DataFrame([p["meta"] for p in pairs]).to_csv(out/"frozen_pairs.csv",index=False)
    plan={"scope":"position-aligned two-token item", "analysis_status":"post-primary sensitivity on same prompts; not independent confirmation",
          "primary_layer":1,"local_layers":base.LAYERS,"rollout_layers":[1],"conditions":base.CONDITIONS,
          "discovery_prompts":len(discovery),"confirmation_prompts":len(keys)-len(discovery),"basis_rank":3,
          "receiver_prompt_length":256,"donor_prompt_lengths":[254,258],"bank":base.BANK,"pairs_per_prompt":6,
          "sample_plan_sha256":base.digest(out/"frozen_pairs.csv"),"script_sha256":base.digest(Path(__file__)),
          "helper_sha256":base.digest(Path(base.__file__)),"original_plan_sha256":base.digest(args.original_results/"plan.json"),"created_unix":time.time()}
    base.write_json(out/"plan.json",plan)
    _,_,_,_,model=base.load_v20_checkpoint_model(args.run_dir,"rope","thinking",step=cfg.train_steps,device="cuda")
    states=capture(model,vocab,[(key,offset,e) for (key,offset),e in shifted.items()],args.batch_size)
    bases={}
    for layer in base.LAYERS:
        cloud=torch.stack([states[(key,0)][layer][:9] for key in keys if key in discovery])
        means=cloud.mean(0)
        bases[layer]=torch.linalg.svd(means-means.mean(0),full_matrices=False).Vh[:3]
    np.savez(out/"frozen_bases.npz",**{f"L{k}":v.numpy() for k,v in bases.items()})
    def cases_for(layer,condition,confirmation_only=False):
        cases=[]
        for pair in pairs:
            m,item=pair["meta"],pair["item"]
            if confirmation_only and m["split"]!="confirmation":
                continue
            r=states[(m["prompt_sha256"],0)][layer][m["receiver_k"]-1]
            d=states[(m["prompt_sha256"],m["offset"])][layer][m["donor_k"]-1]
            vectors,norms=base.make_condition_states(r,d,bases[layer],m["pair_id"],layer)
            cases.append({"meta":m,"vector":vectors[condition],"condition":condition,"patch_span":2,
                          "prefix":item.input_ids[:m["commit_position"]+1],"positions":item.prompt_needle_positions,"norms":norms,
                          "delta_norm":float((vectors[condition]-r).norm()),"receiver_id":vocab.token_to_id[m["receiver_next_marker"]],
                          "donor_id":vocab.token_to_id[m["donor_next_marker"]]})
        return cases
    local=[]
    for layer in base.LAYERS:
        for condition in base.CONDITIONS:
            local.extend(base.evaluate_local(model,vocab,cases_for(layer,condition),layer,args.batch_size))
            print(f"aligned local L{layer} {condition}",flush=True)
        pd.DataFrame(local).to_csv(out/"local_trials.csv",index=False)
    base.summarize(pd.DataFrame(local),["routing_y","top2_routing_y","qk_margin_mean","donor_pair_share","donor_vs_receiver_marker_logodds"],out,"local")
    rollouts,routes=[],[]
    for condition in base.CONDITIONS:
        detail,route=base.evaluate_rollout(model,vocab,cases_for(1,condition,True),1,args.batch_size,28)
        rollouts.extend(detail); routes.extend(route)
        pd.DataFrame(rollouts).to_csv(out/"rollout_trials.csv",index=False)
        pd.DataFrame(routes).to_csv(out/"rollout_routing.csv",index=False)
    base.summarize(pd.DataFrame(rollouts),["donor_marker_adoption","receiver_marker_retention","donor_route_adoption",
                    "receiver_route_retention","donor_first_three_routes_exact","final_correct","generated_marker_count"],out,"rollout")
    # Shared finalizer validates all row counts, self controls, architectural nulls and hashes.
    base.finalize_saved_results(args.run_dir,out)
    manifest=json.loads((out/"manifest.json").read_text())
    manifest["bookkeeping_recovery"]="None: completed normally; shared finalizer performs validation and provenance only."
    manifest["inference_script_sha256"]=plan["script_sha256"]
    base.write_json(out/"manifest.json",manifest)


if __name__=="__main__":
    main()
