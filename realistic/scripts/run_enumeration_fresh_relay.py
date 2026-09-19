"""Run the frozen common terminal-relay panel with the archived numerical kernel."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import importlib.util
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def selected_relay_tasks(cohorts, model, widths, *, smoke, mode=None):
    tasks = []
    population = cohorts["models"][model]
    if cohorts.get("cohort_policy") == "native_assay_specific_per_mode":
        if mode is None:
            raise ValueError("Native-aligned relay requires an explicit mode")
        population = population["per_mode"][mode]
    for width in widths:
        pairs = population["pairs"][str(width)]
        if len(pairs) != len(set(pairs)) or pairs != sorted(pairs):
            raise ValueError("Frozen relay pairs must be unique and sorted")
        tasks.extend({"width": width, "pair_id": pair} for pair in (pairs[:1] if smoke else pairs))
    return tasks


def validate_relay_panel(trials, audits, generations, pair, width, source_layers):
    expected = {(source, relay) for source in ("self_patch", "full_donor_patch")
                for relay in ("natural_relay", "answer_query_clean_reset", "post_terminal_suffix_clean_reset")}
    actual = {(row["source_condition"], row["relay_condition"]) for row in trials}
    if actual != expected or len(trials) != len(audits) or len(trials) != len(generations) or len(trials) != 6:
        raise ValueError("Relay panel or raw hook/generation coverage is incomplete")
    geometry = pair["widths"][str(width)]
    for row, audit, raw in zip(trials, audits, generations):
        if row["receiver_patch_positions"] != geometry["receiver_patch_positions"] or row["donor_patch_positions"] != geometry["donor_patch_positions"]:
            raise ValueError("GPU source geometry differs from the frozen CPU registry")
        if row["patch_token_count"] != width or row["patch_token_count_capped_by_shorter_span"]:
            raise ValueError("Relay source width was capped or changed")
        if row["source_patch_hook_applications"] != {str(layer): 1 for layer in source_layers}:
            raise ValueError("Source clamp did not fire once at every registered layer")
        norms = audit["source_realized_delta_norms"]
        if set(norms) != {str(layer) for layer in source_layers} or not all(math.isfinite(v) for v in norms.values()):
            raise ValueError("Source norm audit is missing or nonfinite")
        is_self = row["source_condition"] == "self_patch"
        if is_self and any(value != 0 for value in norms.values()):
            raise ValueError("Self source clamp changed its own state")
        if not is_self and not any(value > 0 for value in norms.values()):
            raise ValueError("Donor source clamp made no state change")
        natural = row["relay_condition"] == "natural_relay"
        positions = pair["post_terminal_reset_positions"] if row["relay_condition"] == "post_terminal_suffix_clean_reset" else pair["query_reset_positions"]
        if row["relay_positions"] != positions or row["relay_reset_hook_applications"] != int(not natural):
            raise ValueError("Relay reset positions or hook count changed")
        if not math.isfinite(row["relay_reset_realized_fro_norm"]) or (is_self and row["relay_reset_realized_fro_norm"] != 0):
            raise ValueError("Nonfinite reset, or clean self-reset changed state")
        if not math.isfinite(row["expected_count_utility"]):
            raise ValueError("Nonfinite answer score")
        if row["completion_text_raw"] != raw["completion_text_raw"]:
            raise ValueError("Raw generation is not paired with its outcome row")


@contextmanager
def retain_relay_audits(kernel):
    """Read-only before/after hooks; original replacements and returns are kept."""
    import torch
    original_prefill = kernel._prefill_with_terminal_source_and_relay_reset
    original_generate = kernel.generate_answer_completion_from_prefill
    audits, generations = [], []

    def prefill(model, adapter, encoding, **options):
        positions = list(options["source_positions"])
        replacements = options["source_replacements"]
        before, norms, calls = {}, {}, {}
        handles = []
        old_forward = kernel._prefix_forward
        for layer in replacements:
            def observe_before(_module, _inputs, output, *, layer=layer):
                hidden = kernel._tensor_from_output(output)
                if hidden.shape[1] == encoding.sequence_length:
                    before[layer] = hidden[:, positions, :].detach().float().cpu().clone()
            handles.append(adapter.layers[layer].register_forward_hook(observe_before))

        def forward(*values, **kwargs):
            after_handles = []
            try:
                # The original prefill has installed its patch hooks by now.
                for layer, replacement in replacements.items():
                    def observe_after(_module, _inputs, output, *, layer=layer, replacement=replacement):
                        hidden = kernel._tensor_from_output(output)
                        if hidden.shape[1] != encoding.sequence_length:
                            return
                        actual = hidden[:, positions, :].detach().float().cpu()
                        expected = replacement.to(device=hidden.device, dtype=hidden.dtype).detach().float().cpu().unsqueeze(0)
                        if layer not in before or not torch.equal(actual, expected):
                            raise RuntimeError("Source hook did not install the exact registered replacement")
                        norms[str(layer)] = float(torch.linalg.vector_norm(actual - before[layer]))
                        calls[str(layer)] = calls.get(str(layer), 0) + 1
                    after_handles.append(adapter.layers[layer].register_forward_hook(observe_after))
                return old_forward(*values, **kwargs)
            finally:
                for handle in after_handles:
                    handle.remove()
        kernel._prefix_forward = forward
        try:
            result = original_prefill(model, adapter, encoding, **options)
        finally:
            kernel._prefix_forward = old_forward
            for handle in handles:
                handle.remove()
        if calls != {str(layer): 1 for layer in replacements}:
            raise RuntimeError("Source observer coverage is incomplete")
        audits.append({"source_realized_delta_norms": norms, "observer_calls": calls,
                       "source_positions": positions, "relay_positions": list(options["relay_positions"])})
        return result

    def generate(*values, **options):
        result = original_generate(*values, **options)
        generations.append(result)
        return result

    kernel._prefill_with_terminal_source_and_relay_reset = prefill
    kernel.generate_answer_completion_from_prefill = generate
    try:
        yield audits, generations
    finally:
        kernel._prefill_with_terminal_source_and_relay_reset = original_prefill
        kernel.generate_answer_completion_from_prefill = original_generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage", "cache-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", required=True, choices=("Qwen3-8B", "Gemma4-E4B"))
    parser.add_argument("--mode", required=True, choices=("enumeration_index", "enumeration_bullet", "thinking"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    tick = time.monotonic()
    deployment = read(args.stage / "manifest.json")
    for name, expected in deployment["entrypoints_sha256"].items():
        assert sha(args.stage / "code" / name) == expected, name
    bundle, causal, registry_root = (Path(deployment[name]) for name in ("baseline_bundle", "causal_stage", "registry_root"))
    assert sha(bundle / "manifest.json") == deployment["baseline_manifest_sha256"]
    assert sha(causal / "code_manifest.json") == deployment["causal_code_manifest_sha256"]
    code_manifest = read(causal / "code_manifest.json")
    for name, expected in {**code_manifest["original_code_sha256"], **code_manifest["additive_code_sha256"]}.items():
        assert sha(causal / "code" / name) == expected, name
    cohorts = read(registry_root / "common_cohorts.json")
    assert sha(registry_root / "common_cohorts.json") == deployment["cohorts_sha256"]
    cfg = read(args.stage / "protocol.json")
    assert sha(args.stage / "protocol.json") == deployment["protocol_copy_sha256"]
    pair_folder = registry_root / args.model / args.mode
    pair_manifest = read(pair_folder / "manifest.json")
    assert sha(pair_folder / "manifest.json") == cohorts["manifests"][f"{args.model}/{args.mode}"]
    assert sha(pair_folder / "pairs.json") == pair_manifest["pairs_sha256"]
    source = causal / "registries" / args.model / args.mode
    source_manifest = read(source / "manifest.json")
    assert sha(source / "manifest.json") == pair_manifest["source_registry_sha256"]
    assert sha(source / "adapted_generations.jsonl") == source_manifest["adapted_generations_sha256"]
    assert pair_manifest["compiler_sha256"] == sha(args.stage / "code/scripts/prepare_enumeration_fresh_relay.py")
    tasks = selected_relay_tasks(cohorts, args.model, cfg["widths"], smoke=args.smoke, mode=args.mode)
    if not tasks:
        raise ValueError("No common geometry; record NOT_ESTIMABLE without loading a model")
    pairs = {row["pair_id"]: row for row in read(pair_folder / "pairs.json")}
    with (source / "adapted_generations.jsonl").open(encoding="utf-8") as f:
        rows = {row["request_id"]: row for row in map(json.loads, f)}
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"status": "RUNNING", "completed_rows": 0, "expected_rows": len(tasks) * 6,
             "tasks": tasks, "smoke": args.smoke, "model": args.model, "mode": args.mode}
    write(args.output / "status.json", state)
    sys.path[:0] = [str(causal / "code/src"), str(causal / "code")]
    helper_spec = importlib.util.spec_from_file_location("relay_geometry", args.stage / "code/scripts/prepare_enumeration_fresh_relay.py")
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    import torch
    from realistic_niah_v4.modeling import load_registered_model
    from realistic_niah_v4.spec import resolve_model_spec
    if args.mode != "thinking":
        from realistic_niah_v6.kernel import install_v6_kernel_adapters, install_v6_specialized_geometry
        install_v6_kernel_adapters()
        install_v6_specialized_geometry(args.mode)
    from realistic_niah_v5 import count_stream as kernel
    try:
        model, tokenizer, adapter = load_registered_model(resolve_model_spec(args.model), cache_dir=args.cache_dir,
            device_map="auto", torch_dtype=cfg["dtype"], attention_backend=cfg["attention_backend"])
        model.eval()
        text_cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        layers = cfg["layers_zero_based"][args.model]
        source_layers = list(range(layers["source"], layers["relay"]))
        write(args.output / "runtime.json", {"torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"), "python": sys.version,
            "gpu": torch.cuda.get_device_name(), "model_revision": resolve_model_spec(args.model).revision,
            "model_source_sha256": sha(sys.modules[type(model).__module__].__file__),
            "dtype": str(next(model.parameters()).dtype), "backend": cfg["attention_backend"],
            "layers": adapter.num_layers, "source_layers": source_layers, "relay_layer": layers["relay"],
            "model_load_seconds": time.monotonic() - tick, "command": sys.argv,
            "stage_manifest_sha256": sha(args.stage / "manifest.json"), "behavioral_effect_required": False})
        original_build = kernel.build_answer_source_registry
        geometries = {}
        with (args.output / "trials.jsonl").open("x", encoding="utf-8") as output:
            for task in tasks:
                pair, width = pairs[task["pair_id"]], task["width"]
                assert pair["widths"][str(width)]["eligible"]
                row = rows[pair["request_id"]]
                if row["request_id"] not in geometries:
                    if args.model == "Gemma4-E4B" and args.mode == "thinking":
                        encoding, registry, strategy = helper.build_found_actual_geometry(row, tokenizer)
                    else:
                        encoding, registry = original_build(row, tokenizer, answer_site_id=cfg["answer_site_id"])
                        strategy = "original_full_baseline_parser"
                    geometries[row["request_id"]] = encoding, registry, strategy
                encoding, registry, strategy = geometries[row["request_id"]]
                assert registry.to_dict() == pair["answer_registry"]
                assert strategy == pair["semantic_token_mapping"]
                assert hashlib.sha256(json.dumps(list(encoding.input_ids)).encode()).hexdigest() == pair["input_ids_sha256"]
                def frozen_geometry(source_row, source_tokenizer, **options):
                    assert source_row is row and source_tokenizer is tokenizer
                    assert options["answer_site_id"] == cfg["answer_site_id"]
                    return encoding, registry
                kernel.build_answer_source_registry = frozen_geometry
                try:
                    with retain_relay_audits(kernel) as (audits, generations):
                        trials = kernel.run_terminal_state_relay_reset_trials(model, tokenizer, adapter, row,
                            receiver_occurrence=pair["receiver_occurrence"], donor_occurrence=pair["donor_occurrence"],
                            source_layer=layers["source"], relay_layer=layers["relay"], geometry=f"suffix{width}",
                            answer_site_id=cfg["answer_site_id"], run_greedy=True, max_new_tokens=cfg["max_new_tokens"])
                finally:
                    kernel.build_answer_source_registry = original_build
                validate_relay_panel(trials, audits, generations, pair, width, source_layers)
                assert model.config._attn_implementation == text_cfg._attn_implementation == cfg["attention_backend"]
                for trial, audit, raw in zip(trials, audits, generations):
                    output.write(json.dumps({**trial, "pair_id": task["pair_id"], "width": width,
                        "mode": args.mode, "raw_generation": raw, "source_hook_audit": audit,
                        "semantic_token_mapping": strategy, "pair_registry_sha256": pair_manifest["pairs_sha256"]}, ensure_ascii=True) + "\n")
                output.flush()
                state.update(completed_rows=state["completed_rows"] + len(trials), current_task=task,
                             seconds=time.monotonic() - tick)
                write(args.output / "status.json", state)
                print(json.dumps({key: value for key, value in state.items() if key != "tasks"}), flush=True)
        state.update(status="COMPLETE", trials_sha256=sha(args.output / "trials.jsonl"))
    except BaseException as error:
        state.update(status="FAILED", error=repr(error))
        raise
    finally:
        state["seconds"] = time.monotonic() - tick
        write(args.output / "status.json", state)


if __name__ == "__main__":
    main()
