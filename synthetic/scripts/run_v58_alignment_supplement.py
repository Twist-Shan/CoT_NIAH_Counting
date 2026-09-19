"""Frozen two-mode body-mechanism supplement; no training or trace changes."""
from __future__ import annotations

import argparse
import contextlib
from collections import Counter
import copy
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

import run_v58_commit_query as base
from v58_alignment_core import Engine, changed_source, controls_for, fit_space, make_record, rank_discovery
from run_v22_free_running_sufficiency import _generate_to_answer
from run_v22_free_running_topk import _local_attention_edit, _parse_generation


@torch.inference_mode()
def free_running_rows(model, cfg, vocab, examples, *, mode, heads, batch_size):
    """Mask registered trace queries only; never include the prompt's <Sep>."""
    rows = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start+batch_size]
        items = [base.render_v20(e, vocab, mode) for e in batch]
        stops = [i.spans.ans_pos+1 if mode == "nonthinking" else i.spans.think_pos+1 for i in items]
        assert len(set(stops)) == 1
        generated = torch.tensor([i.input_ids[:s] for i, s in zip(items, stops)], device=cfg.device)
        done = torch.zeros(len(batch), dtype=torch.bool, device=cfg.device)
        for _ in range(4 if mode == "nonthinking" else cfg.max_render_len-stops[0]+2):
            if mode == "nonthinking":
                positions = [[i.spans.ans_pos] for i in items]
            else:
                positions = [[p for p in torch.nonzero(row.eq(vocab.token_to_id["<Sep>"])).flatten().tolist()
                              if p >= stop] for row, stop in zip(generated, stops)]
            ctx = _local_attention_edit(model, heads, positions) if heads else contextlib.nullcontext()
            with ctx:
                next_ids = model(input_ids=generated).logits[:, -1].argmax(-1)
            next_ids = torch.where(done, torch.full_like(next_ids, vocab.eos_id), next_ids)
            generated = torch.cat([generated, next_ids[:, None]], dim=1)
            done |= next_ids.eq(vocab.eos_id)
            if bool(done.all()):
                break
        for e, seq in zip(batch, generated.cpu().tolist()):
            rows.append({"mode": mode, "count": e.count, "prompt_sha256": e.prompt_sha256,
                         **_parse_generation(vocab.decode(seq), vocab, e, mode)})
    return pd.DataFrame(rows)


def freeze_examples(run, old_selection):
    source = run / "analysis/behavior_confirmation_v58/examples.jsonl"
    examples = [base.example_from_dict(json.loads(s)) for s in source.read_text().splitlines() if s.strip()]
    old = set()
    for name in ["v58_commit_query_20260905", "v58_native_continuation_20260905"]:
        old |= set(pd.read_csv(run / "analysis" / name / "frozen_pairs.csv").prompt_sha256)
    old |= set(pd.read_csv(run / "analysis/v58_free_running_sufficiency/progress_rollout_confirmation.csv").prompt_sha256)
    excluded = {(e.set_id, e.corpus_start) for e in examples if e.prompt_sha256 in old}
    excluded |= {(e.set_id, e.corpus_start) for e in old_selection}
    chosen, registry, seen = [], [], set(excluded)
    for n in range(1, 11):
        bucket = sorted((e for e in examples if e.count == n and e.prompt_sha256 not in old), key=lambda e: e.prompt_sha256)
        retained = []
        for e in bucket:
            key = (e.set_id, e.corpus_start)
            if key in seen:
                continue
            retained.append(e); seen.add(key)
            if len(retained) == 30:
                break
        if len(retained) != 30:
            raise ValueError(f"count {n}: need 30 canonical-distinct prompts, have {len(retained)}")
        for i, e in enumerate(retained):
            split, block = ("discovery", i) if i < 20 else ("confirmation", i-20)
            chosen.append((e, split, block))
            registry.append({"key": e.prompt_sha256, "count": n, "split": split, "block": block,
                             "set_id": e.set_id, "corpus_start": e.corpus_start, "source_seed": e.seed})
    return chosen, pd.DataFrame(registry), source


def fit_all_spaces(records, captured, bank):
    spaces = {}
    labels = [r["count"] for r in records]
    for l in range(5):
        spaces[("answer", l)] = fit_space(np.stack([c["hidden"][l][r["q"]].numpy() for r, c in zip(records, captured)]), labels)
        cloud, ys = [], []
        for r, c in zip(records, captured):
            cloud.extend(c["hidden"][l][r["needles"]].numpy())
            ys.extend(range(1, len(r["needles"])+1))
        spaces[("running", l)] = fit_space(np.stack(cloud), ys)
    for l in sorted({l for l, h in bank}):
        spaces[("retrieval", l)] = fit_space(np.stack([c["writes"][l].numpy() for c in captured]), labels)
    return spaces


def record_clean_match(generated, reference):
    return bool(generated == reference)


def run_mode(args, cfg, vocab, chosen, mode, output):
    started = time.time()
    _, _, _, _, model = base.load_v20_checkpoint_model(args.run_dir, "rope", mode, step=cfg.train_steps, device=args.device)
    model.eval()
    records = [make_record(e, vocab, mode, split, block) for e, split, block in chosen]
    discovery = [r for r in records if r["split"] == "discovery"]
    confirmation = [r for r in records if r["split"] == "confirmation"]
    ranking = rank_discovery(model, vocab, discovery, args.batch_size)
    role = "broad" if mode == "nonthinking" else "targeted"
    role_bank = [(l, h) for l, h, score in ranking[role][:4]]
    # Answer-side broad instrumentation is separate from Thinking trace-targeted bank.
    bank = [(l, h) for l, h, score in ranking["broad"][:4]]
    ret_layer = min(sorted({l for l, h in bank}), key=lambda l: (-sum(ll == l for ll, h in bank), l))
    plan = {"mode": mode, "ranking": ranking, "role_bank": role_bank, "answer_broad_bank": bank,
            "ret_layer": ret_layer, "late_layer": 4, "terminal_source_layer": 2, "relay_reset_layer": 3,
            "discovery_prompts": 200, "confirmation_prompts": 100,
            "controls": {str(k): controls_for(role_bank, k) for k in [1, 2, 4]},
            "selection_uses_confirmation": False, "created_unix": time.time()}
    base.write_json(output / "frozen_sites.json", plan)
    print(mode, "FROZEN", json.dumps(plan), flush=True)
    engine = Engine(model, vocab, bank, args.batch_size)
    dc = engine.run(discovery, capture=True)
    engine.spaces = fit_all_spaces(discovery, dc, bank)
    torch.save(engine.spaces, output / "frozen_spaces.pt")
    cc = engine.run(confirmation, capture=True)
    caches = {r["key"]: c for r, c in zip(discovery+confirmation, dc+cc)}
    rows = []
    def evaluate(family, arm, subset, actions=None, **tags):
        t0 = time.time()
        values = engine.run(subset, actions)
        for row in values:
            row.update(family=family, arm=arm, **tags)
        rows.extend(values)
        pd.DataFrame(rows).to_csv(output / "trials.csv", index=False)
        print(f"{mode} {family}/{arm} n={len(subset)} {tags} seconds={time.time()-t0:.2f}", flush=True)
        return values
    def patch(r, l, ps, donor=None, donor_ps=None):
        donor = donor or r
        return (l, ps, caches[donor["key"]]["hidden"][l][donor_ps if donor_ps is not None else ps])
    def clean_actions(rs, l, ps_fn):
        return [{"patch": [patch(r, l, ps_fn(r))]} for r in rs]
    def subset_index(rs):
        return {(r["block"], r["count"]): r for r in rs}
    ci, di = subset_index(confirmation), subset_index(discovery)

    # 1. Source input -> state -> answer, all depths including embedding.
    evaluate("source", "clean", confirmation)
    for kind in ["needles", "ordinary"]:
        evaluate("source", f"corrupt_{kind}", confirmation, [{"seq": changed_source(r, kind)} for r in confirmation])
    for l in range(5):
        for kind in ["needles", "ordinary"]:
            actions = []
            for r in confirmation:
                ps = r["needles"] if kind == "needles" else r["matched_ordinary"]
                actions.append({"seq": changed_source(r, "needles"), "patch": [patch(r, l, ps)]})
            values = evaluate("source", f"restore_{kind}", confirmation, actions, layer=l)
            if kind == "needles" and l == 0:
                err = max(abs(x["margin"]-caches[r["key"]]["margin"]) for x, r in zip(values, confirmation))
                assert err < 1e-4, ("Embedding restoration failed", err)

    # 2. Complete answer-state sufficiency and matched direction necessity.
    for l in range(1, 5):
        values = evaluate("answer", "self", confirmation, clean_actions(confirmation, l, lambda r: [r["q"]]), layer=l)
        err = max(abs(x["margin"]-caches[r["key"]]["margin"]) for x, r in zip(values, confirmation))
        assert err < 1e-4, ("Self patch failed", err)
        for direction in [-1, 1]:
            subset = [r for r in confirmation if 1 <= r["count"]+direction <= 10]
            for kind in ["adjacent_donor", "same_count_context"]:
                actions = []
                for r in subset:
                    donor = ci[(r["block"], r["count"]+direction)] if kind == "adjacent_donor" else di[(r["block"], r["count"])]
                    actions.append({"patch": [patch(r, l, [r["q"]], donor, [donor["q"]])]})
                evaluate("answer", kind, subset, actions, layer=l, offset=direction)
        for kind in ["aligned", "orthogonal"]:
            evaluate("answer", f"remove_{kind}", confirmation, [{"late_layer": l, "late_kind": kind} for r in confirmation], layer=l)

    # 3. Same-forward factorial mediation: source-state layer 0 and L1 separately.
    if mode == "nonthinking":
        for source_layer in [0, 1]:
            for restore, ret_kind, late_kind in itertools.product([False, True], ["none", "aligned", "orthogonal"], ["none", "aligned", "orthogonal"]):
                actions = []
                for r in confirmation:
                    a = {"seq": changed_source(r, "needles")}
                    if restore:
                        a["patch"] = [patch(r, source_layer, r["needles"])]
                    if ret_kind != "none":
                        a.update(ret_layer=ret_layer, ret_kind=ret_kind)
                    if late_kind != "none":
                        a.update(late_layer=4, late_kind=late_kind)
                    actions.append(a)
                evaluate("serial", "factorial", confirmation, actions, source_layer=source_layer,
                         source_restored=restore, retrieval=ret_kind, late=late_kind)
        for kind in ["none", "aligned", "orthogonal"]:
            evaluate("retrieval_natural", kind, confirmation,
                     [{} if kind == "none" else {"ret_layer": ret_layer, "ret_kind": kind} for r in confirmation], layer=ret_layer)

    # 4. Free-generated trace at answer time; no accuracy filtering.
    prefixes, reached = _generate_to_answer(model, cfg, vocab, [r["example"] for r in confirmation], mode=mode, device=args.device)
    generated_records, generated_registry = [], []
    for r, seq, ok in zip(confirmation, prefixes, reached):
        if not ok:
            raise RuntimeError("Prefix failed to reach answer query: retain registry and revise missing-output accounting before continuing")
        g = dict(r, seq=seq, q=len(seq)-1)
        if mode == "thinking":
            g["markers"] = [p+1 for p in range(r["think"]+1, len(seq)-1) if seq[p] == vocab.token_to_id["<Sep>"] and vocab.id_to_token[seq[p+1]].startswith("<CH_")]
        generated_records.append(g)
        generated_registry.append({"key": r["key"], "count": r["count"], "block": r["block"], "reached_answer": ok,
                                   "gold_prefix_exact": record_clean_match(seq, r["seq"]), "generated_prefix_ids": json.dumps(seq)})
    pd.DataFrame(generated_registry).to_csv(output / "generated_prefixes.csv", index=False)
    for kind in ["clean", "records", "trace", "ordinary_records_budget", "ordinary_trace_budget", "all_context"]:
        actions = []
        for r in generated_records:
            trace = list(range(r["think"], r["q"])) if r["think"] is not None else []
            ps = {"clean": [], "records": r["needles"], "trace": trace,
                  "ordinary_records_budget": r["matched_ordinary"],
                  "ordinary_trace_budget": r["ordinary"][-len(trace):] if trace else [], "all_context": list(range(1, r["q"]))}[kind]
            actions.append({"blank": ps})
        evaluate("answer_source", kind, generated_records, actions)

    # 5. Re-ranked dose response with actual, not duplicated, random-bank counts.
    ablation = []
    arms = [("clean", 0, 0, [])]
    for k in [1, 2, 4]:
        arms.append(("selected", k, 0, role_bank[:k]))
        arms.extend(("random", k, j, hs) for j, hs in enumerate(controls_for(role_bank, k)))
    for arm, k, repeat, heads in arms:
        detail = free_running_rows(model, cfg, vocab, [r["example"] for r in confirmation], mode=mode, heads=heads, batch_size=args.batch_size)
        for row, r in zip(detail.to_dict("records") if isinstance(detail, pd.DataFrame) else detail, confirmation):
            row.update(key=r["key"], block=r["block"], arm=arm, top_k=k, repeat=repeat, heads=json.dumps(heads))
            ablation.append(row)
        pd.DataFrame(ablation).to_csv(output / "ablation.csv", index=False)
        print(mode, "free ablation", arm, k, repeat, flush=True)

    if mode == "thinking":
        # 6. Terminal bridge at original two-token item scopes; no invented index.
        def terminal_positions(r, scope):
            p = r["markers"][-1]
            return {"item": [p-1, p], "marker": [p], "separator": [p-1]}[scope]
        def damaged_trace(r):
            seq = list(r["seq"])
            ps = [p for m in r["markers"] for p in [m-1, m]]
            for p, d in zip(ps, r["ordinary"]):
                seq[p] = seq[d]
            return seq
        evaluate("terminal_bridge", "clean", confirmation)
        evaluate("terminal_bridge", "damaged", confirmation, [{"seq": damaged_trace(r)} for r in confirmation])
        for scope, control in itertools.product(["item", "marker", "separator"], [False, True]):
            actions = []
            for r in confirmation:
                ps = terminal_positions(r, scope)
                src = r["ordinary"][:len(ps)] if control else ps
                actions.append({"seq": damaged_trace(r), "patch": [patch(r, l, ps, donor_ps=src) for l in [2, 3, 4]]})
            evaluate("terminal_bridge", "ordinary_restore" if control else "semantic_restore", confirmation, actions, scope=scope)

        # 7. Terminal-source by downstream-reset factorial on the same pairs.
        for direction in [-1, 1]:
            subset = [r for r in confirmation if 1 <= r["count"]+direction <= 10]
            for donor_patch, reset in itertools.product([False, True], ["none", "suffix", "query"]):
                actions = []
                for r in subset:
                    donor = ci[(r["block"], r["count"]+direction)] if donor_patch else r
                    ps, ds = terminal_positions(r, "item"), terminal_positions(donor, "item")
                    patches = [patch(r, 2, ps, donor, ds)]
                    if reset != "none":
                        rp = list(range(r["markers"][-1]+1, r["q"]+1)) if reset == "suffix" else [r["q"]]
                        patches.append(patch(r, 3, rp))
                    actions.append({"patch": patches})
                evaluate("terminal_relay", "donor" if donor_patch else "self", subset, actions, offset=direction, reset=reset)

        # 8. Fixed-visible-token read -> later item-state feasibility, including null.
        subset = [r for r in confirmation if r["count"] >= 2]
        carrier_rows = []
        for k in [2, 4]:
            conditions = [("selected", 0, role_bank[:k])]+[("random", j, hs) for j, hs in enumerate(controls_for(role_bank, k))]
            for arm, repeat, heads in conditions:
                actions = [{"mask": heads, "mask_queries": [r["markers"][-1]-1]} for r in subset]
                captures = engine.run(subset, actions, capture=True)
                for r, c in zip(subset, captures):
                    for l in range(1, 5):
                        ps = [r["markers"][-1]]
                        rms = float((c["hidden"][l][ps]-caches[r["key"]]["hidden"][l][ps]).square().mean().sqrt())
                        carrier_rows.append({"key": r["key"], "block": r["block"], "count": r["count"], "top_k": k,
                                             "arm": arm, "repeat": repeat, "layer": l, "item_end_rms": rms})
                print(mode, "read_to_item_state", arm, k, repeat, flush=True)
        pd.DataFrame(carrier_rows).to_csv(output / "read_to_item_state.csv", index=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "trials.csv", index=False)
    coverage = frame.groupby(["family", "arm"], dropna=False).agg(rows=("key", "size"), prompts=("key", "nunique"), blocks=("block", "nunique")).reset_index()
    coverage.to_csv(output / "coverage.csv", index=False)
    checkpoint_root = args.run_dir / "checkpoints/rope" / mode
    index = pd.read_csv(checkpoint_root / "snapshot_index.csv")
    cp = checkpoint_root / str(index.loc[index.step.eq(cfg.train_steps)].iloc[-1]["shard"])
    manifest = {"status": "complete", "mode": mode, "seconds": time.time()-started,
                "checkpoint": str(cp), "checkpoint_sha256": base.digest(cp), "rows": len(frame),
                "generated_prefix_failures": int(np.sum(np.logical_not(reached))),
                "files": {p.name: base.digest(p) for p in output.iterdir() if p.is_file()}}
    base.write_json(output / "manifest.json", manifest)
    del engine, model, caches, dc, cc
    torch.cuda.empty_cache()
    print(mode, "COMPLETE", manifest["seconds"], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    started = time.time()
    cfg, vocab, _, old_selection, _ = base._load_bundle(args.run_dir, device=args.device)
    assert cfg.version == "v58" and cfg.n_layer == 4 and cfg.n_head == 8
    chosen, registry, source = freeze_examples(args.run_dir, old_selection)
    registry.to_csv(args.output / "input_registry.csv", index=False)
    plan = {"version": "v58", "checkpoint_step": cfg.train_steps, "modes": ["nonthinking", "thinking"],
            "counts": list(range(1, 11)), "discovery_prompts": 200, "confirmation_prompts": 100,
            "statistical_blocks": "20 discovery / 10 confirmation balanced prompt blocks; not model training seeds",
            "old_progress_excluded": True, "source_is_previously_behavior_evaluated": True,
            "input_registry_sha256": base.digest(args.output / "input_registry.csv"), "source_sha256": base.digest(source),
            "script_sha256": base.digest(Path(__file__)), "core_sha256": base.digest(Path(__file__).with_name("v58_alignment_core.py")),
            "prefix_failure_policy": "stop and preserve outputs; never silently select successful prefixes",
            "created_unix": started, "torch": torch.__version__}
    base.write_json(args.output / "plan.json", plan)
    print("FROZEN COMMON PANEL", json.dumps(plan), flush=True)
    for mode in plan["modes"]:
        output = args.output / mode
        output.mkdir()
        run_mode(args, cfg, vocab, chosen, mode, output)
    a = pd.read_csv(args.output / "nonthinking/trials.csv")
    b = pd.read_csv(args.output / "thinking/trials.csv")
    common = ["source", "answer", "answer_source"]
    fields = ["family", "arm", "key", "count", "block", "layer", "offset"]
    canonical = lambda f: sorted(map(tuple, f.loc[f.family.isin(common), fields].fillna(-999).astype(str).values))
    assert canonical(a) == canonical(b), "Realized common trial multiset mismatch"
    base.write_json(args.output / "manifest.json", {"status": "complete", "seconds": time.time()-started,
                    "common_realized_trial_multisets_equal": True, "common_families": common,
                    "files": {str(p.relative_to(args.output)): base.digest(p) for p in args.output.rglob("*") if p.is_file()}})
    print("ALL COMPLETE; REALIZED MULTISETS VERIFIED", flush=True)


if __name__ == "__main__":
    main()
