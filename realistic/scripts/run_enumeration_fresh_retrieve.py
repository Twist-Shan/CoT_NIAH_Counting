"""Measure fresh Retrieve localization and frozen persistent head ablations."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_enumeration_fresh_retrieve import read, write, sha, json_sha, rank_heads, freeze_banks


def validate_trial(trial, task, query, norms):
    if trial["seed"] != query["seed"] or trial["gold_count"] != query["gold_count"]:
        raise ValueError("Trial source differs from frozen query")
    if trial["intervention_full_sequence_token_indices"] != [query["query_position"]]:
        raise ValueError("Head intervention missed the registered query")
    if trial["heads"] != task["heads"] or trial["head_ablation_decode_steps_requested"] != -1:
        raise ValueError("Head bank or persistent policy changed")
    if not 0 < len(trial["generated_token_ids"]) <= 512:
        raise ValueError("Missing generation or incorrect token budget")
    layers = {str(layer) for layer, _ in task["heads"]}
    expected = {layer: 1 for layer in layers}
    if trial["head_ablation_prefill_layer_applications"] != expected or set(norms) != layers:
        raise ValueError("Incomplete head hook coverage")
    if task["heads"]:
        if trial["head_ablation_selected_post_zero_max_abs"] != 0:
            raise ValueError("Selected head slices were not zero")
        expected_decode = len(trial["generated_token_ids"]) - 1
        if trial["head_ablation_decode_layer_applications"] != {layer: expected_decode for layer in layers}:
            raise ValueError("Head mask did not persist through cached decoding")
        if any(not math.isfinite(x) or x < 0 for x in norms.values()):
            raise ValueError("Nonfinite ablation magnitude")


@contextmanager
def observe_prefill_norms(adapter, heads, query):
    """Observe the vectors subsequently zeroed by the unchanged Native hook."""
    import torch
    grouped, norms, handles = {}, {}, []
    for layer, head in heads:
        grouped.setdefault(layer, []).append(head)
    for layer, selected in grouped.items():
        def observe(_module, inputs, *, layer=layer, selected=tuple(selected)):
            value = inputs[0]
            if value.shape[1] != query + 1:
                return
            if str(layer) in norms:
                raise RuntimeError("Repeated full prefill in head-ablation observer")
            width = adapter.head_dims[layer]
            vectors = torch.cat([value[:, query, h * width:(h + 1) * width].detach().float().reshape(-1) for h in selected])
            if not torch.isfinite(vectors).all():
                raise RuntimeError("Nonfinite selected head vector before zeroing")
            norms[str(layer)] = float(torch.linalg.vector_norm(vectors))
        handles.append(adapter.output_projections[layer].register_forward_pre_hook(observe))
    try:
        yield norms
    finally:
        for handle in handles:
            handle.remove()


def freeze_cell_banks(args, cfg, cell, queries, folder):
    source = args.stage / "jobs/localize" / args.model / args.mode
    status = read(source / "status.json")
    assert status["status"] == "COMPLETE"
    assert sha(source / "observations.jsonl") == status["observations_sha256"]
    assert sha(source / "ranking.json") == status["ranking_sha256"]
    with (source / "observations.jsonl").open(encoding="utf-8") as handle:
        observations = list(map(json.loads, handle))
    expected = {q["query_id"] for q in queries if q["split"] == "discovery"}
    assert len(observations) == len(expected) and {r["query_id"] for r in observations} == expected
    ranking = rank_heads(observations)
    assert ranking == read(source / "ranking.json") and len(ranking) == cfg["expected_heads"][args.model]
    confirmation = read(folder / "confirmation.json")
    bank = freeze_banks(ranking, cfg["dose_grid"][args.model], sorted(q["seed"] for q in confirmation),
        salt=f'{cfg["random_seed"]}/{args.model}/{args.mode}', repeats=cfg["random_repeats"])
    tasks = []
    for q in confirmation:
        base = {"query_id": q["query_id"], "request_id": q["request_id"], "seed": q["seed"]}
        tasks.append({**base, "k": 0, "condition": "clean", "repeat": 0, "heads": []})
        for k in bank["ks"]:
            tasks.append({**base, "k": k, "condition": "selected_bank", "repeat": 0, "heads": bank["selected_heads"][:k]})
            for repeat, heads in enumerate(bank["random_banks"][str(q["seed"])][str(k)]):
                tasks.append({**base, "k": k, "condition": bank["random_control_by_k"][str(k)], "repeat": repeat, "heads": heads})
    # A single source tests full selected and each actual random-control type.
    first, kmax = confirmation[0]["seed"], max(bank["ks"])
    control_ks = {kind: max(int(k) for k, c in bank["random_control_by_k"].items() if c == kind)
                  for kind in set(bank["random_control_by_k"].values())}
    smoke = [t for t in tasks if t["seed"] == first and (t["condition"] == "clean" or
        (t["condition"] == "selected_bank" and t["k"] == kmax) or
        (t["condition"] in control_ks and t["k"] == control_ks[t["condition"]] and t["repeat"] == 0))]
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "banks.json", bank)
    write(args.output / "plan.json", {"status": "FROZEN_BEFORE_BEHAVIOR", "model": args.model, "mode": args.mode,
        "tasks": tasks, "smoke_tasks": smoke, "banks_sha256": sha(args.output / "banks.json"),
        "discovery_observations_sha256": status["observations_sha256"], "ranking_sha256": status["ranking_sha256"],
        "confirmation_sha256": cell["confirmation_sha256"], "protocol_sha256": sha(args.stage / "protocol.json"),
        "stage_manifest_sha256": sha(args.stage / "manifest.json"), "uses_confirmation_outcomes": False})
    write(args.output / "status.json", {"status": "COMPLETE", "expected_formal_rows": len(tasks),
        "expected_smoke_rows": len(smoke), "plan_sha256": sha(args.output / "plan.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("localize", "banks", "behavior"))
    for name in ("stage", "cache-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    tick = time.monotonic()
    deployment = read(args.stage / "manifest.json")
    for name, expected in deployment["entrypoints_sha256"].items():
        assert sha(args.stage / "code" / name) == expected, name
    assert sha(args.stage / "protocol.json") == deployment["protocol_sha256"]
    cfg = read(args.stage / "protocol.json")
    registry_root = Path(deployment["registry_root"])
    assert sha(registry_root / "manifest.json") == deployment["registry_manifest_sha256"]
    registry = read(registry_root / "manifest.json")
    causal = Path(registry["causal_stage"])
    assert sha(causal / "code_manifest.json") == registry["causal_code_manifest_sha256"]
    frozen = read(causal / "code_manifest.json")
    for name, expected in {**frozen["original_code_sha256"], **frozen["additive_code_sha256"]}.items():
        assert sha(causal / "code" / name) == expected, name
    cell = next(c for c in registry["cells"] if c["model"] == args.model and c["mode"] == args.mode)
    folder = registry_root / args.model / args.mode
    for name in ("queries", "eligibility", "confirmation"):
        assert sha(folder / f"{name}.json") == cell[f"{name}_sha256"]
    source = Path(cell["source_registry"])
    assert sha(source / "manifest.json") == cell["source_registry_sha256"]
    assert sha(source / "adapted_generations.jsonl") == cell["source_generations_sha256"]
    queries = read(folder / "queries.json")
    if args.action == "banks":
        freeze_cell_banks(args, cfg, cell, queries, folder)
        return
    with (source / "adapted_generations.jsonl").open(encoding="utf-8") as handle:
        rows = {r["request_id"]: r for r in map(json.loads, handle)}
    by_id = {q["query_id"]: q for q in queries}
    if args.action == "localize":
        tasks = [q for q in queries if q["split"] == "discovery"]
        if args.smoke:
            tasks = tasks[:1]
    else:
        bank_root = args.stage / "banks" / args.model / args.mode
        plan = read(bank_root / "plan.json")
        assert sha(bank_root / "plan.json") == read(bank_root / "status.json")["plan_sha256"]
        assert plan["stage_manifest_sha256"] == sha(args.stage / "manifest.json")
        assert plan["confirmation_sha256"] == cell["confirmation_sha256"]
        assert sha(bank_root / "banks.json") == plan["banks_sha256"]
        tasks = plan["smoke_tasks" if args.smoke else "tasks"]
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"status": "RUNNING", "action": args.action, "smoke": args.smoke, "completed_rows": 0, "expected_rows": len(tasks)}
    write(args.output / "status.json", state)
    sys.path[:0] = [str(causal / "code/src"), str(causal / "code")]
    import torch
    from realistic_niah_v4 import modeling
    from realistic_niah_v4.spec import resolve_model_spec
    from realistic_niah_v6.kernel import install_v6_kernel_adapters, install_v6_specialized_geometry
    install_v6_kernel_adapters()
    install_v6_specialized_geometry(args.mode)
    from realistic_niah_v5 import causal as kernel
    try:
        model, tokenizer, adapter = modeling.load_registered_model(resolve_model_spec(args.model), cache_dir=args.cache_dir,
            device_map="auto", torch_dtype=cfg["dtype"], attention_backend=cfg["attention_backend"])
        model.eval()
        active = list(range(adapter.num_layers)) if args.model == "Qwen3-8B" else [i for i, kind in enumerate(adapter.layer_types) if kind == "full_attention"]
        assert sum(adapter.num_heads[i] for i in active) == cfg["expected_heads"][args.model]
        text_cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        write(args.output / "runtime.json", {"torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
            "python": sys.version, "gpu": torch.cuda.get_device_name(), "model_revision": resolve_model_spec(args.model).revision,
            "model_source_sha256": sha(sys.modules[type(model).__module__].__file__), "active_layers": active,
            "layer_types": list(adapter.layer_types), "num_heads": list(adapter.num_heads), "command": sys.argv,
            "dtype": str(next(model.parameters()).dtype), "backend": cfg["attention_backend"], "model_load_seconds": time.monotonic() - tick,
            "stage_manifest_sha256": sha(args.stage / "manifest.json"), "scientific_effect_required": False})
        records = []
        name = "observations" if args.action == "localize" else "trials"
        with (args.output / f"{name}.jsonl").open("x", encoding="utf-8") as output:
            for task in tasks:
                start = time.monotonic()
                query = by_id[task["query_id"]]
                row = rows[query["request_id"]]
                assert json_sha(row) == query["row_sha256"]
                enc = kernel.build_native_causal_encoding(row, tokenizer, query_output_token_index=query["query_output_token_index"],
                    sequence_output_token_end=query["query_output_token_index"] + 1, selected_site=query)
                assert enc.query_position == query["query_position"]
                assert json_sha(list(enc.input_ids)) == query["input_ids_sha256"]
                assert json_sha(list(enc.attention_mask)) == query["attention_mask_sha256"]
                if args.action == "localize":
                    attention, starts, logits = modeling.position_attention_outputs(model, adapter, enc, enc.query_position)
                    assert torch.isfinite(logits).all()
                    left, right = query["source_record_span"]
                    heads = []
                    for layer in active:
                        weights = attention[layer]
                        assert torch.isfinite(weights).all() and (weights >= 0).all()
                        assert starts[layer] <= left < right <= starts[layer] + weights.shape[-1]
                        masses = weights[:, left - starts[layer]:right - starts[layer]].sum(-1)
                        assert torch.isfinite(masses).all() and (masses >= 0).all() and (masses <= 1.01).all()
                        heads.extend({"layer": layer, "head": head, "mass": float(value)} for head, value in enumerate(masses))
                    record = {"query_id": query["query_id"], "seed": query["seed"], "heads": heads,
                        "query_position": enc.query_position, "source_record_span": query["source_record_span"], "key_starts": starts}
                    records.append(record)
                else:
                    with observe_prefill_norms(adapter, task["heads"], enc.query_position) as norms:
                        trial = kernel.run_retrieval_head_behavior_trial(model, tokenizer, adapter, row, heads=task["heads"],
                            condition=task["condition"], anchor_equivalence_id=query["anchor_equivalence_id"],
                            max_new_tokens=cfg["max_new_tokens"], decode_head_ablation_steps=cfg["decode_head_ablation_steps"])
                    validate_trial(trial, task, query, norms)
                    record = {**trial, "dose_k": task["k"], "repeat": task["repeat"], "mode": args.mode,
                        "query_id": query["query_id"], "prefill_zeroed_head_l2_by_layer": norms,
                        "plan_sha256": sha(bank_root / "plan.json"), "generated_at_token_cap": len(trial["generated_token_ids"]) == cfg["max_new_tokens"]}
                assert model.config._attn_implementation == text_cfg._attn_implementation == cfg["attention_backend"]
                record["seconds"] = time.monotonic() - start
                output.write(json.dumps(record, ensure_ascii=True) + "\n")
                output.flush()
                state.update(completed_rows=state["completed_rows"] + 1, seconds=time.monotonic() - tick)
                write(args.output / "status.json", state)
        if args.action == "localize" and not args.smoke:
            write(args.output / "ranking.json", rank_heads(records))
            state["ranking_sha256"] = sha(args.output / "ranking.json")
        state.update(status="COMPLETE", **{f"{name}_sha256": sha(args.output / f"{name}.jsonl")})
    except BaseException as error:
        state.update(status="FAILED", error=repr(error))
        raise
    finally:
        state["seconds"] = time.monotonic() - tick
        write(args.output / "status.json", state)


if __name__ == "__main__":
    main()
