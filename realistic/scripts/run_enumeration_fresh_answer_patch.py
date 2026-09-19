"""Run the Native full-answer-state kernel on frozen Enumeration pairs."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_panel(trials, observed, pair, layer, query, *, clean_reference=None):
    if len(trials) != 2 or len(observed) != 2 or [r["condition"] for r in trials] != ["self_patch", "full_donor_patch"]:
        raise ValueError("Incomplete two-condition panel")
    for row, audit in zip(trials, observed):
        if row["layer"] != layer or row["source_positions"] != [query]:
            raise ValueError("Answer-patch layer or query differs from frozen geometry")
        if audit["positions"] != [query] or audit["hook_applications"] != {str(layer): 1} or audit["observer_calls"] != 1:
            raise ValueError("Answer-patch hook coverage is incorrect")
        if not math.isfinite(audit["realized_delta_norm"]) or not math.isfinite(row["full_delta_norm"]):
            raise ValueError("Nonfinite answer-state perturbation")
        if row["condition"] == "self_patch":
            expected_count = pair["receiver_count"] if clean_reference is None else clean_reference["prediction"]
            if audit["realized_delta_norm"] != 0 or row["prediction"] != expected_count:
                raise ValueError("Self patch changed the clean state or failed Receiver regeneration")
            if clean_reference is not None:
                for key in ("generated_token_ids", "completion_text_raw", "stopped_on_eos", "generation_truncated"):
                    if audit["raw_generation"][key] != clean_reference["generated"][key]:
                        raise ValueError(f"Self patch differs from unmodified query replay: {key}")
        elif audit["realized_delta_norm"] <= 0:
            raise ValueError("Donor patch made no state change")
        if row["completion_text_raw"] != audit["raw_generation"]["completion_text_raw"]:
            raise ValueError("Generated tokens are paired with a different trial")


@contextmanager
def retain_patch_audits(kernel, modeling):
    """Observe the archived one-shot hook without replacing its numerical work."""
    import torch
    original_patch = kernel.generate_with_residual_interventions
    original_generate = modeling.generate_answer_completion
    audits = []

    def patched(model, tokenizer, adapter, encoding, interventions, **options):
        if len(interventions) != 1:
            raise ValueError("Answer-state assay requires a single patch layer")
        layer, (positions, replacement) = next(iter(interventions.items()))
        positions = list(positions)
        before, after, calls = [], [], []

        def observe_before(_module, _inputs, output):
            hidden = modeling._tensor_from_output(output)
            if hidden.shape[1] == encoding.sequence_length:
                before.append(hidden[:, positions, :].detach().float().cpu().clone())

        def observe_after(_module, _inputs, output):
            hidden = modeling._tensor_from_output(output)
            if hidden.shape[1] == encoding.sequence_length:
                actual = hidden[:, positions, :].detach().float().cpu()
                expected = replacement.to(device=hidden.device, dtype=hidden.dtype).detach().float().cpu().reshape(1, len(positions), -1)
                if not torch.equal(actual, expected):
                    raise RuntimeError("Patch hook did not install the exact saved state")
                after.append(actual.clone())
                calls.append(1)

        def generate(*values, **kwargs):
            # The archived function installs its replacement hook first.
            handle = adapter.layers[layer].register_forward_hook(observe_after)
            try:
                return original_generate(*values, **kwargs)
            finally:
                handle.remove()

        handle = adapter.layers[layer].register_forward_hook(observe_before)
        modeling.generate_answer_completion = generate
        try:
            raw = original_patch(model, tokenizer, adapter, encoding, interventions, **options)
        finally:
            modeling.generate_answer_completion = original_generate
            handle.remove()
        if len(before) != 1 or len(after) != 1:
            raise RuntimeError("Patch observer did not see exactly one complete prefill")
        audits.append({"positions": positions, "hook_applications": raw["intervention_hook_applications"],
            "observer_calls": len(calls), "realized_delta_norm": float(torch.linalg.vector_norm(after[0] - before[0])),
            "raw_generation": raw})
        return raw

    kernel.generate_with_residual_interventions = patched
    try:
        yield audits
    finally:
        kernel.generate_with_residual_interventions = original_patch
        modeling.generate_answer_completion = original_generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage", "cache-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pair-ids-file", type=Path)
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
    assert registry["status"] == "FROZEN_BEFORE_ANSWER_PATCH_GPU"
    causal = Path(registry["causal_stage"])
    assert sha(causal / "code_manifest.json") == registry["causal_code_manifest_sha256"]
    code_manifest = read(causal / "code_manifest.json")
    for name, expected in {**code_manifest["original_code_sha256"], **code_manifest["additive_code_sha256"]}.items():
        assert sha(causal / "code" / name) == expected, name
    cell = next(c for c in registry["cells"] if c["model"] == args.model and c["mode"] == args.mode)
    folder = registry_root / args.model / args.mode
    assert sha(folder / "pairs.json") == cell["pairs_sha256"]
    assert sha(folder / "eligibility.json") == cell["eligibility_sha256"]
    source = Path(cell["source_registry"])
    assert sha(source / "manifest.json") == cell["source_registry_sha256"]
    assert sha(source / "adapted_generations.jsonl") == cell["source_generations_sha256"]
    pairs = read(folder / "pairs.json")
    assert len(pairs) == cfg["expected_pairs_per_model_mode"]
    selected = pairs
    if args.pair_ids_file is not None:
        subset_path = args.pair_ids_file.resolve()
        if sha(subset_path) != deployment["subset_sha256"].get(str(subset_path)):
            raise ValueError("Incremental pair subset is not in the frozen manifest")
        identifiers = read(subset_path)
        if len(identifiers) != len(set(identifiers)) or not set(identifiers).issubset({p["pair_id"] for p in pairs}):
            raise ValueError("Incremental pair subset has unknown or duplicate pairs")
        selected = [p for p in pairs if p["pair_id"] in identifiers]
    if args.smoke:
        selected = selected[:int(cfg.get("smoke_pair_count", 1))]
    if not selected:
        raise ValueError("No new pairs to execute")
    eligibility = {row["request_id"]: row for row in read(folder / "eligibility.json")}
    with (source / "adapted_generations.jsonl").open(encoding="utf-8") as handle:
        rows = {row["request_id"]: row for row in map(json.loads, handle)}
    layers = list(range(cfg["num_layers"][args.model]))
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"status": "RUNNING", "completed_rows": 0, "expected_rows": len(selected) * len(layers) * 2,
        "model": args.model, "mode": args.mode, "smoke": args.smoke}
    write(args.output / "status.json", state)
    sys.path[:0] = [str(causal / "code/src"), str(causal / "code")]
    from scripts.enumeration_fresh_geometry import build_read_geometry, json_sha
    import torch
    from realistic_niah_v4 import modeling
    from realistic_niah_v4.spec import resolve_model_spec
    from realistic_niah_v6.kernel import install_v6_kernel_adapters, install_v6_specialized_geometry
    install_v6_kernel_adapters()
    install_v6_specialized_geometry(args.mode)
    from realistic_niah_v5 import causal as kernel
    from realistic_niah_v6.parsing import parse_trace_record
    try:
        model, tokenizer, adapter = modeling.load_registered_model(resolve_model_spec(args.model),
            cache_dir=args.cache_dir, device_map="auto", torch_dtype=cfg["dtype"], attention_backend=cfg["attention_backend"])
        model.eval()
        assert adapter.num_layers == len(layers)
        text_cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        write(args.output / "runtime.json", {"model": args.model, "mode": args.mode, "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"), "python": sys.version,
            "gpu": torch.cuda.get_device_name(), "model_revision": resolve_model_spec(args.model).revision,
            "model_source_sha256": sha(sys.modules[type(model).__module__].__file__),
            "dtype": str(next(model.parameters()).dtype), "backend": cfg["attention_backend"], "layers": layers,
            "max_new_tokens": cfg["max_new_tokens"], "model_load_seconds": time.monotonic() - tick,
            "capture": "One saved answer-query vector per layer and request; all layers captured in the same unmodified forward.",
            "command": sys.argv, "stage_manifest_sha256": sha(args.stage / "manifest.json"),
            "donor_adoption_required": False})
        cache, clean_replays = {}, {}

        def captured(request_id):
            if request_id not in cache:
                row, entry = rows[request_id], eligibility[request_id]
                assert entry["eligible"] and json_sha(row) == entry["adapted_row_sha256"]
                assert sha(entry["geometry_file"]) == entry["geometry_sha256"]
                parsed = parse_trace_record(row)
                assert parsed["parser"]["trace_one_to_one"] and parsed["exact_count"]
                encoding, _, geometry = build_read_geometry(row, tokenizer, mode=args.mode)
                assert json_sha(geometry) == json_sha(entry["read_geometry"])
                _, states = modeling.capture_post_block_states(model, adapter, encoding, [encoding.query_position], layers=layers)
                assert set(states) == set(layers) and all(torch.isfinite(value).all() for value in states.values())
                cache[request_id] = encoding, {layer: value[0] for layer, value in states.items()}
                if cfg.get("record_unmodified_query_replay", False):
                    replay = modeling.generate_answer_completion(model, tokenizer, encoding, max_new_tokens=cfg["max_new_tokens"])
                    metrics = kernel.completion_metrics(replay, gold_count=encoding.count)
                    clean_replays[request_id] = {"request_id": request_id, "gold_count": encoding.count,
                        "prediction": metrics["prediction"], "exact_count": metrics["exact_count"],
                        "query_position": encoding.query_position, "input_ids_sha256": json_sha(list(encoding.input_ids)),
                        "generated": replay, "role": "technical reference; not an eligibility screen"}
                    write(args.output / "clean_query_replays.json", clean_replays)
            return cache[request_id]

        with (args.output / "trials.jsonl").open("x", encoding="utf-8") as output:
            for pair in selected:
                receiver, receiver_states = captured(pair["receiver_request_id"])
                donor, donor_states = captured(pair["donor_request_id"])
                assert receiver.count == pair["receiver_count"] and donor.count == pair["donor_count"]
                assert receiver.seed == donor.seed == pair["seed"]
                for layer in layers:
                    with retain_patch_audits(kernel, modeling) as audits:
                        trials = kernel.run_projected_patch_trials_from_states(model, tokenizer, adapter,
                            receiver, receiver_states[layer], donor, donor_states[layer],
                            receiver_site_id=cfg["site_id"], donor_site_id=cfg["site_id"], layer=layer,
                            basis=None, max_new_tokens=cfg["max_new_tokens"], requested_conditions=cfg["conditions"])
                    reference = clean_replays.get(pair["receiver_request_id"])
                    validate_panel(trials, audits, pair, layer, receiver.query_position, clean_reference=reference)
                    assert model.config._attn_implementation == text_cfg._attn_implementation == cfg["attention_backend"]
                    for trial, audit in zip(trials, audits):
                        output.write(json.dumps({**trial, "pair_id": pair["pair_id"], "alignment_pair_id": pair["alignment_pair_id"],
                            "pair_direction": pair["pair_direction"], "mode": args.mode,
                            "receiver_query_position": receiver.query_position, "donor_query_position": donor.query_position,
                            "hook_audit": audit, "pairs_sha256": cell["pairs_sha256"],
                            **({"receiver_clean_replay": reference,
                                "donor_clean_replay": clean_replays[pair["donor_request_id"]]} if reference is not None else {})}, ensure_ascii=True) + "\n")
                    output.flush()
                    state.update(completed_rows=state["completed_rows"] + 2, pair_id=pair["pair_id"], layer=layer,
                        seconds=time.monotonic() - tick)
                    write(args.output / "status.json", state)
                print(json.dumps(state), flush=True)
        state.update(status="COMPLETE", trials_sha256=sha(args.output / "trials.jsonl"))
    except BaseException as error:
        state.update(status="FAILED", error=repr(error))
        raise
    finally:
        state["seconds"] = time.monotonic() - tick
        write(args.output / "status.json", state)


if __name__ == "__main__":
    main()
