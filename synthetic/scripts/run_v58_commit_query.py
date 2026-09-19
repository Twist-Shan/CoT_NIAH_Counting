#!/usr/bin/env python
"""Frozen full commit -> next query assay; see docs/v58_commit_query_protocol.md."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from compare_v22_modes_ncc import _load_bundle
from synthetic_counting_v20.data import example_from_dict, render_v20
from synthetic_counting_v20.training import load_v20_checkpoint_model
from synthetic_counting_v20.v10_port_analysis import _residual_patch

CONDITIONS = ["clean", "self_patch", "full_donor_patch", "count_subspace_transplant",
              "norm_matched_orthogonal_patch", *[f"full_norm_orthogonal_r{i}" for i in range(3)]]
PRIMARY_LAYER = 1
LAYERS = [1, 2, 3, 4]
BANK = [(4, 5), (4, 0), (4, 1), (4, 4)]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def checkpoint_source(run_dir, step):
    """Resolve the same numeric-checkpoint/dense-shard fallback as the loader."""
    root = Path(run_dir) / "checkpoints/rope/thinking"
    numeric = root / f"step_{step:06d}" / "checkpoint.pt"
    if numeric.is_file():
        return numeric
    index = pd.read_csv(root / "snapshot_index.csv")
    rows = index.loc[index.step.astype(int).eq(step)]
    if rows.empty:
        raise FileNotFoundError(f"No snapshot indexed at step {step}")
    shard = root / str(rows.iloc[-1]["shard"])
    if not shard.is_file():
        raise FileNotFoundError(shard)
    return shard


def finalize_saved_results(run_dir, out):
    """Recover bookkeeping only; never recompute, change or select outcomes."""
    if (out / "manifest.json").exists():
        raise FileExistsError(out / "manifest.json")
    plan = json.loads((out / "plan.json").read_text())
    assert digest(out / "frozen_pairs.csv") == plan["sample_plan_sha256"]
    frame = pd.read_csv(out / "local_trials.csv")
    roll = pd.read_csv(out / "rollout_trials.csv")
    expected = (plan["discovery_prompts"] + plan["confirmation_prompts"]) * 6
    assert len(frame) == expected * len(LAYERS) * len(CONDITIONS)
    assert len(roll) == plan["confirmation_prompts"] * 6 * len(CONDITIONS)
    for table in ["local_summary", "local_contrasts", "rollout_summary", "rollout_contrasts", "rollout_routing"]:
        assert (out / f"{table}.csv").is_file()
    pivot = frame.pivot(index=["pair_id", "layer"], columns="condition", values="routing_y")
    late = pivot.xs(4, level="layer")
    qk = frame.loc[frame.layer.eq(3)].set_index(["pair_id", "condition"])[[f"L{l}H{h}_qk_margin" for l,h in BANK]]
    continuations = roll.pivot(index="pair_id", columns="condition", values="continuation_tokens")
    validation = {
        "self_patch_max_abs_attention_error": float((pivot.clean-pivot.self_patch).abs().max()),
        "L4_structural_null_error": float(late.sub(late.clean, axis=0).abs().max().max()),
        "L3_max_per_head_QK_margin_change": max(float((qk.xs(c,level="condition")-qk.xs("clean",level="condition")).abs().max().max()) for c in CONDITIONS),
        "no_nonfinite_local_metrics": bool(np.isfinite(frame[["routing_y", "top2_routing_y", "qk_margin_mean", "donor_pair_share", "donor_vs_receiver_marker_logodds"]].values).all()),
        "clean_self_rollouts_equal": bool((continuations.clean == continuations.self_patch).all()),
        "rollout_token_cap_rows": int(roll.hit_token_cap.sum()),
        "discovery_confirmation_prompt_overlap": len(set(frame.loc[frame.split.eq("discovery"),"prompt_sha256"]) & set(frame.loc[frame.split.eq("confirmation"),"prompt_sha256"])),
    }
    assert validation["self_patch_max_abs_attention_error"] < 2e-5
    assert validation["L4_structural_null_error"] < 2e-5
    assert validation["L3_max_per_head_QK_margin_change"] < 2e-4
    assert validation["no_nonfinite_local_metrics"] and validation["clean_self_rollouts_equal"]
    cfg = json.loads((run_dir / "config.json").read_text())
    checkpoint = checkpoint_source(run_dir, cfg["train_steps"])
    manifest = {"status": "complete", "source_run": str(run_dir),
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "not queried",
                "torch": torch.__version__, "validation": validation,
                "checkpoint": str(checkpoint), "checkpoint_step": cfg["train_steps"], "checkpoint_sha256": digest(checkpoint),
                "inference_script_sha256": plan["script_sha256"], "finalizer_script_sha256": digest(Path(__file__)),
                "bookkeeping_recovery": "Inference and CSV summaries completed. Recovered manifest after numeric checkpoint provenance path failed; actual loader used indexed dense snapshot. No inference repeated or outcomes changed.",
                "files": {f.name:digest(f) for f in out.iterdir() if f.is_file()}}
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def pad(sequences, vocab, device):
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), vocab.pad_id, device=device, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, values in enumerate(sequences):
        ids[i, :len(values)] = torch.as_tensor(values, device=device)
        mask[i, :len(values)] = 1
    return ids, mask


@torch.inference_mode()
def forward(model, vocab, sequences, *, attention=False, hidden=False):
    ids, mask = pad(sequences, vocab, next(model.parameters()).device)
    return model(ids, attention_mask=mask, output_attentions=attention, output_hidden_states=hidden)


def pair_metadata(example, rendered, k, offset, split):
    d = k + offset
    markers = example.needle_markers
    key = f"{example.prompt_sha256}:{k}:{offset}"
    return {
        "prompt_sha256": example.prompt_sha256, "pair_id": hashlib.sha256(key.encode()).hexdigest(),
        "split": split, "gold_count": int(example.count), "receiver_k": k, "donor_k": d,
        "offset": offset, "commit_position": rendered.spans.trace_marker_positions[k-1],
        "donor_commit_position": rendered.spans.trace_marker_positions[d-1],
        "receiver_successor": k+1, "donor_successor": d+1,
        "receiver_source_position": rendered.prompt_needle_positions[k],
        "donor_source_position": rendered.prompt_needle_positions[d],
        "same_commit_marker": markers[k-1] == markers[d-1],
        "successor_identity_distinct": markers[k] != markers[d],
        "receiver_next_marker": markers[k], "donor_next_marker": markers[d],
    }


def make_condition_states(receiver, donor, basis, pair_id, layer):
    receiver, donor, basis = (x.float().cpu() for x in (receiver, donor, basis))
    delta = donor - receiver
    projected = basis.T @ (basis @ delta)
    seed = int(hashlib.sha256(f"{pair_id}:{layer}:controls".encode()).hexdigest()[:12], 16)
    gen = torch.Generator().manual_seed(seed)
    orthogonals = []
    for _ in range(3):
        v = torch.randn(receiver.shape, generator=gen)
        v -= basis.T @ (basis @ v)
        v /= v.norm().clamp_min(1e-12)
        orthogonals.append(v)
    result = {
        "clean": receiver, "self_patch": receiver,
        "full_donor_patch": donor,
        "count_subspace_transplant": receiver + projected,
        "norm_matched_orthogonal_patch": receiver + orthogonals[0] * projected.norm(),
        **{f"full_norm_orthogonal_r{i}": receiver + v * delta.norm() for i, v in enumerate(orthogonals)},
    }
    for name, value in result.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite patch: {name}")
    assert torch.allclose(basis @ (result["norm_matched_orthogonal_patch"]-receiver), torch.zeros(basis.shape[0]), atol=2e-5)
    return result, {"full_delta_norm": delta.norm().item(), "count_delta_norm": projected.norm().item()}


@contextlib.contextmanager
def patch_batch(model, layer, cases, device):
    # All batches are homogeneous in condition; clean is the no-hook reference.
    if cases[0]["condition"] == "clean":
        assert all(c["condition"] == "clean" for c in cases)
        yield
    else:
        positions = [c["meta"]["commit_position"] for c in cases]
        vectors = torch.stack([c["vector"] for c in cases]).to(device)
        span = cases[0].get("patch_span", 1)
        assert all(c.get("patch_span", 1) == span for c in cases)
        vectors = vectors.reshape(len(cases), span, -1)
        with contextlib.ExitStack() as stack:
            for j in range(span):
                stack.enter_context(_residual_patch(model, layer, [p-span+1+j for p in positions], vectors[:,j]))
            yield


def routing_metrics(output, row, query, case):
    r = case["meta"]["receiver_source_position"]
    d = case["meta"]["donor_source_position"]
    positions = case["positions"]
    per_head = []
    for layer, head in BANK:
        a = output.attentions[layer-1][row, head, query].float()
        ad, ar = a[d].item(), a[r].item()
        per_head.append((ad, ar, float(torch.log(a[d].clamp_min(1e-38))-torch.log(a[r].clamp_min(1e-38)))))
    bank_positions = sum(output.attentions[l-1][row,h,query,list(positions)].float() for l,h in BANK)
    pairs = np.asarray(per_head)
    metrics = {
        "routing_y": float(np.sum(pairs[:, 0]-pairs[:, 1])),
        "top2_routing_y": float(np.sum(pairs[:2, 0]-pairs[:2, 1])),
        "donor_mass": float(pairs[:, 0].sum()), "receiver_mass": float(pairs[:, 1].sum()),
        "qk_margin_mean": float(pairs[:, 2].mean()),
        "donor_pair_share": float(np.mean(pairs[:,0]/np.maximum(pairs[:,0]+pairs[:,1], 1e-38))),
        "bank_argmax_occurrence": int(bank_positions.argmax().item()+1),
        "prompt_needle_mass": float(bank_positions.sum()),
    }
    for i,(layer,head) in enumerate(BANK):
        metrics[f"L{layer}H{head}_qk_margin"] = float(pairs[i,2])
    return metrics


def cases_for(pairs, states, bases, vocab, layer, condition):
    result = []
    for pair in pairs:
        meta, item = pair["meta"], pair["item"]
        state = states[meta["prompt_sha256"]][layer]
        vectors, norms = make_condition_states(state[meta["receiver_k"]-1], state[meta["donor_k"]-1], bases[layer], meta["pair_id"], layer)
        prefix = item.input_ids[:meta["commit_position"]+1]
        result.append({"meta":meta, "vector":vectors[condition], "condition":condition,
                       "prefix":prefix, "positions":item.prompt_needle_positions, "norms":norms,
                       "delta_norm":float((vectors[condition]-state[meta["receiver_k"]-1]).norm()),
                       "receiver_id":vocab.token_to_id[meta["receiver_next_marker"]],
                       "donor_id":vocab.token_to_id[meta["donor_next_marker"]]})
    return result


@torch.inference_mode()
def evaluate_local(model, vocab, cases, layer, batch_size):
    rows = []
    device = next(model.parameters()).device
    sep = vocab.token_to_id["<Sep>"]
    for start in range(0, len(cases), batch_size):
        batch = cases[start:start+batch_size]
        seqs = [c["prefix"]+[sep] for c in batch]
        with patch_batch(model, layer, batch, device):
            output = forward(model, vocab, seqs, attention=True)
        for i,case in enumerate(batch):
            q = len(seqs[i])-1
            logit = output.logits[i,q].float()
            rows.append({**case["meta"], "layer":layer, "condition":case["condition"], **case["norms"],
                         "patch_delta_norm":case["delta_norm"], **routing_metrics(output,i,q,case),
                         "donor_vs_receiver_marker_logodds":float(logit[case["donor_id"]]-logit[case["receiver_id"]]),
                         "forced_marker_id":int(logit.argmax())})
        del output
    return rows


@torch.inference_mode()
def evaluate_rollout(model, vocab, cases, layer, batch_size, max_tokens):
    rows, routing_rows = [], []
    device = next(model.parameters()).device
    sep = vocab.token_to_id["<Sep>"]
    ans = vocab.token_to_id["<Ans>"]
    for start in range(0,len(cases),batch_size):
        batch=cases[start:start+batch_size]
        seqs=[list(c["prefix"]) for c in batch]
        continuations=[[] for _ in batch]
        active=[True]*len(batch)
        routes=[[] for _ in batch]
        for step in range(max_tokens):
            if not any(active):
                break
            want_a=any(active[i] and seqs[i][-1]==sep for i in range(len(batch)))
            with patch_batch(model,layer,batch,device):
                output=forward(model,vocab,seqs,attention=want_a)
            for i,case in enumerate(batch):
                if not active[i]:
                    continue
                q=len(seqs[i])-1
                if seqs[i][-1]==sep and want_a:
                    metric=routing_metrics(output,i,q,case)
                    ordinal=len(routes[i])
                    metric.update({**case["meta"],"layer":layer,"condition":case["condition"],
                                   "continuation_query_index":ordinal+1,
                                   "expected_receiver_occurrence":case["meta"]["receiver_successor"]+ordinal,
                                   "expected_donor_occurrence":case["meta"]["donor_successor"]+ordinal})
                    routes[i].append(metric)
                    routing_rows.append(metric)
                token=int(output.logits[i,q].argmax())
                continuations[i].append(token)
                seqs[i].append(token)
                active[i]=token!=vocab.eos_id
            del output
        for i,case in enumerate(batch):
            ids=continuations[i]
            marker_ids=[]
            cursor=0
            while cursor+1<len(ids) and ids[cursor]==sep and vocab.id_to_token[ids[cursor+1]] in vocab.character_tokens:
                marker_ids.append(ids[cursor+1]); cursor+=2
            final_count=None
            if ans in ids and ids.index(ans)+1<len(ids):
                token=vocab.id_to_token[ids[ids.index(ans)+1]]
                if token in vocab.numbers:
                    final_count=vocab.decode_number_tokens([token])
            first=marker_ids[0] if marker_ids else None
            obs=routes[i]
            first_route=obs[0]["bank_argmax_occurrence"] if obs else 0
            expected_remaining=10-case["meta"]["donor_k"]
            first_three_n=min(3,expected_remaining)
            route_three=(len(obs)>=first_three_n and all(obs[j]["bank_argmax_occurrence"]==case["meta"]["donor_successor"]+j for j in range(first_three_n)))
            rows.append({**case["meta"],"layer":layer,"condition":case["condition"],
                         "first_generated_marker":vocab.id_to_token[first] if first is not None else "",
                         "donor_marker_adoption":float(first==case["donor_id"]),
                         "receiver_marker_retention":float(first==case["receiver_id"]),
                         "first_route":first_route,
                         "donor_route_adoption":float(first_route==case["meta"]["donor_successor"]),
                         "receiver_route_retention":float(first_route==case["meta"]["receiver_successor"]),
                         "donor_first_three_routes_exact":float(route_three),
                         "generated_marker_count":case["meta"]["receiver_k"]+len(marker_ids),
                         "predicted_shifted_total":10-case["meta"]["offset"],
                         "generated_final_count":final_count,"final_correct":float(final_count==10),
                         "answer_matches_generated_markers":float(final_count==case["meta"]["receiver_k"]+len(marker_ids)),
                         "hit_token_cap":active[i],
                         "continuation_tokens":" ".join(vocab.decode(ids))})
        print(f"rollout L{layer} {cases[0]['condition']} {min(start+batch_size,len(cases))}/{len(cases)}",flush=True)
    return rows,routing_rows


def summarize(frame, metrics, output, prefix):
    summaries=[]
    comparisons=[("full_donor_patch","self_patch"),("full_donor_patch","norm_matched_orthogonal_patch"),
                 ("count_subspace_transplant","norm_matched_orthogonal_patch"),
                 ("full_donor_patch","full_norm_orthogonal_mean"),("self_patch","clean")]
    orth=frame.loc[frame.condition.str.startswith("full_norm_orthogonal")]
    if not orth.empty:
        means=orth.groupby(["prompt_sha256","pair_id","split","layer","offset","same_commit_marker","successor_identity_distinct"],as_index=False)[metrics].mean()
        means["condition"]="full_norm_orthogonal_mean"
        frame=pd.concat([frame,means],ignore_index=True)
    for (split,layer),base in frame.groupby(["split","layer"]):
        subsets={"all":base,"forward":base.loc[base.offset==1],"backward":base.loc[base.offset==-1],
                 "same_commit_marker":base.loc[base.same_commit_marker],
                 "distinct_successor":base.loc[base.successor_identity_distinct]}
        for subset,sub in subsets.items():
            for metric in metrics:
                # Identical successor tokens cannot identify an ordinal transfer.
                if metric in ("donor_marker_adoption","receiver_marker_retention","donor_vs_receiver_marker_logodds"):
                    data=sub.loc[sub.successor_identity_distinct]
                else:
                    data=sub
                if data.empty:
                    continue
                pivot=data.pivot(index=["prompt_sha256","pair_id"],columns="condition",values=metric)
                for treatment,control in comparisons:
                    paired=pivot[[treatment,control]].dropna()
                    if paired.empty:
                        continue
                    delta=paired[treatment]-paired[control]
                    per_prompt=delta.groupby(level=0).mean().to_numpy()
                    rng=np.random.default_rng(20260905)
                    boot=per_prompt[rng.integers(0,len(per_prompt),(10000,len(per_prompt)))].mean(axis=1)
                    low,high=np.quantile(boot,[.025,.975])
                    summaries.append({"split":split,"layer":int(layer),"subset":subset,"metric":metric,
                                      "contrast":f"{treatment} - {control}","pairs":len(paired),"prompts":len(per_prompt),
                                      "treatment_mean":float(paired[treatment].groupby(level=0).mean().mean()),
                                      "control_mean":float(paired[control].groupby(level=0).mean().mean()),
                                      "effect":float(per_prompt.mean()),"ci_low":float(low),"ci_high":float(high)})
    result=pd.DataFrame(summaries)
    result.to_csv(output/f"{prefix}_contrasts.csv",index=False)
    frame.groupby(["split","layer","condition"])[metrics].mean().reset_index().to_csv(output/f"{prefix}_summary.csv",index=False)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--discovery-prompts",type=int,default=20)
    p.add_argument("--confirmation-prompts",type=int,default=60)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--device",default="cuda")
    p.add_argument("--skip-rollout",action="store_true")
    p.add_argument("--finalize-existing", action="store_true")
    args=p.parse_args()
    if args.finalize_existing:
        finalize_saved_results(args.run_dir, args.output)
        return
    started=time.time()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/"manifest.json").exists() or (out/"plan.json").exists():
        raise FileExistsError(f"Use a fresh output directory: {out}")
    torch.set_num_threads(4)
    cfg,vocab,train,selection,reporting=_load_bundle(args.run_dir,device=args.device)
    assert cfg.version=="v58" and cfg.trace_format=="separator" and cfg.n_layer==4
    source=args.run_dir/"analysis/behavior_confirmation_v58/examples.jsonl"
    examples=[example_from_dict(json.loads(line)) for line in source.read_text().splitlines() if line.strip()]
    old=args.run_dir/"analysis/v58_free_running_sufficiency/progress_rollout_confirmation.csv"
    old_hashes=set(pd.read_csv(old).prompt_sha256)
    historical_keys={(e.set_id,e.corpus_start) for e in selection+reporting}
    eligible=[e for e in examples if e.count==10 and e.prompt_sha256 not in old_hashes and (e.set_id,e.corpus_start) not in historical_keys]
    eligible=sorted(eligible,key=lambda e:e.prompt_sha256)
    n=args.discovery_prompts+args.confirmation_prompts
    if len(eligible)<n:
        raise ValueError(f"Only {len(eligible)} eligible prompts, need {n}")
    chosen=eligible[:n]
    assert len(set(e.prompt_sha256 for e in chosen))==n
    historical_layers=json.loads((args.run_dir/"analysis/v58_free_running_sufficiency/selected_layers.json").read_text())
    assert historical_layers["thinking_progress"]==PRIMARY_LAYER
    ranks=pd.read_csv(args.run_dir/"analysis/phase_transition/tables/fixed_head_rankings.csv")
    top=ranks.loc[ranks.role=="targeted_retrieval"].sort_values("rank").head(4)
    assert list(zip(top.layer,top['head']))==BANK
    pairs=[]
    for i,example in enumerate(chosen):
        item=render_v20(example,vocab,"thinking")
        split="discovery" if i<args.discovery_prompts else "confirmation"
        for k in [4,6,8]:
            for offset in [-1,1]:
                pairs.append({"meta":pair_metadata(example,item,k,offset,split),"item":item})
    pd.DataFrame([q["meta"] for q in pairs]).to_csv(out/"frozen_pairs.csv",index=False)
    plan={"primary_layer":PRIMARY_LAYER,"layers":LAYERS,"bank":BANK,"conditions":CONDITIONS,
          "discovery_prompts":args.discovery_prompts,"confirmation_prompts":args.confirmation_prompts,
          "pairs_per_prompt":6,"basis_rank":3,"sample_plan_sha256":digest(out/"frozen_pairs.csv"),
          "script_sha256":digest(Path(__file__)),"examples_sha256":digest(source),
          "primary_endpoint":"sum_bank A(donor successor)-A(receiver successor)",
          "layer_policy":"historical discovery progress L1; L2 sensitivity; L3/L4 structural controls",
          "split_policy":"canonical-disjoint external behavior suite; exclude historical geometry/progress; sorted hashes",
          "prefix_policy":"gold no-index prefix through completed item; forced Sep for local assay; free continuation for rollout",
          "rollout_layers":[PRIMARY_LAYER],"bootstrap_unit":"prompt (six paired repeats averaged within prompt)",
          "created_unix":time.time()}
    write_json(out/"plan.json",plan)
    print(f"Frozen {n} prompts, {len(pairs)} pairs; loading final checkpoint",flush=True)
    _,loaded_vocab,_,_,model=load_v20_checkpoint_model(args.run_dir,"rope","thinking",step=cfg.train_steps,device=args.device)
    assert loaded_vocab.fingerprint==vocab.fingerprint
    model.eval()
    states={}
    for start in range(0,n,args.batch_size):
        batch=chosen[start:start+args.batch_size]
        items=[render_v20(e,vocab,"thinking") for e in batch]
        seqs=[item.input_ids[:item.spans.trace_marker_positions[-1]+1] for item in items]
        output=forward(model,vocab,seqs,hidden=True)
        for j,(e,item) in enumerate(zip(batch,items)):
            states[e.prompt_sha256]={layer:output.hidden_states[layer][j,list(item.spans.trace_marker_positions)].float().cpu() for layer in LAYERS}
        del output
    bases={}
    for layer in LAYERS:
        discovery=torch.stack([states[e.prompt_sha256][layer][:9] for e in chosen[:args.discovery_prompts]])
        centroids=discovery.mean(0)
        centered=centroids-centroids.mean(0)
        _,_,vh=torch.linalg.svd(centered,full_matrices=False)
        bases[layer]=vh[:3]
    np.savez(out/"frozen_bases.npz",**{f"L{k}":v.numpy() for k,v in bases.items()})
    local=[]
    for layer in LAYERS:
        for condition in CONDITIONS:
            cases=cases_for(pairs,states,bases,vocab,layer,condition)
            local.extend(evaluate_local(model,vocab,cases,layer,args.batch_size))
            print(f"local L{layer} {condition}: {len(cases)} pairs",flush=True)
        pd.DataFrame(local).to_csv(out/"local_trials.csv",index=False)
    frame=pd.DataFrame(local)
    metrics=["routing_y","top2_routing_y","qk_margin_mean","donor_pair_share","donor_vs_receiver_marker_logodds"]
    contrasts=summarize(frame,metrics,out,"local")
    pivot=frame.pivot(index=["pair_id","layer"],columns="condition",values="routing_y")
    self_error=float((pivot.clean-pivot.self_patch).abs().max())
    late=frame.loc[frame.layer==4].pivot(index="pair_id",columns="condition",values="routing_y")
    late_error=float(late.sub(late.clean,axis=0).abs().max().max())
    qkcols=[f"L{l}H{h}_qk_margin" for l,h in BANK]
    qk=frame.loc[frame.layer==3].set_index(["pair_id","condition"])[qkcols]
    qk_late=max(float((qk.xs(c,level="condition")-qk.xs("clean",level="condition")).abs().max().max()) for c in CONDITIONS)
    validation={"self_patch_max_abs_attention_error":self_error,"L4_structural_null_error":late_error,
                "L3_max_per_head_QK_margin_change":qk_late,"no_nonfinite_local_metrics":bool(np.isfinite(frame[metrics].values).all()),
                "discovery_confirmation_prompt_overlap":0,"historical_progress_prompt_overlap":0}
    if self_error>2e-5 or late_error>2e-5 or qk_late>2e-4 or not validation["no_nonfinite_local_metrics"]:
        write_json(out/"validation_failed.json",validation)
        raise RuntimeError(f"Assay sanity failed: {validation}")
    if not args.skip_rollout:
        rollouts,routes=[],[]
        confirm_pairs=[q for q in pairs if q["meta"]["split"]=="confirmation"]
        for condition in CONDITIONS:
            cases=cases_for(confirm_pairs,states,bases,vocab,PRIMARY_LAYER,condition)
            detail,route=evaluate_rollout(model,vocab,cases,PRIMARY_LAYER,args.batch_size,2*cfg.count_max_threshold+8)
            rollouts.extend(detail); routes.extend(route)
            pd.DataFrame(rollouts).to_csv(out/"rollout_trials.csv",index=False)
            pd.DataFrame(routes).to_csv(out/"rollout_routing.csv",index=False)
        rollout=pd.DataFrame(rollouts)
        summarize(rollout,["donor_marker_adoption","receiver_marker_retention","donor_route_adoption",
                           "receiver_route_retention","donor_first_three_routes_exact","final_correct","generated_marker_count"],out,"rollout")
        same=rollout.pivot(index="pair_id",columns="condition",values="continuation_tokens")
        validation["clean_self_rollouts_equal"]=bool((same.clean==same.self_patch).all())
        validation["rollout_token_cap_rows"]=int(rollout.hit_token_cap.sum())
        if not validation["clean_self_rollouts_equal"]:
            raise RuntimeError("Self-patch rollout mismatch")
    checkpoint=checkpoint_source(args.run_dir, cfg.train_steps)
    manifest={"status":"complete","source_run":str(args.run_dir),"gpu":torch.cuda.get_device_name() if args.device=="cuda" else "cpu",
              "torch":torch.__version__,"elapsed_seconds":time.time()-started,"validation":validation,
              "files":{f.name:digest(f) for f in out.iterdir() if f.is_file()},
              "checkpoint":str(checkpoint),"checkpoint_sha256":digest(checkpoint)}
    write_json(out/"manifest.json",manifest)
    print(contrasts.loc[(contrasts.split=="confirmation")&(contrasts.layer==1)&(contrasts.subset=="all")].to_string(index=False),flush=True)
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=="__main__":
    main()
