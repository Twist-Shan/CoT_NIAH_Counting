"""Reuse the frozen aligned Update kernel with fresh common source seeds.

The model stays loaded across the 18 registered k/direction/scope jobs. The
wrappers below only retain otherwise discarded raw generations and hook audits;
they do not change numerical returns, inputs, sampling, or scoring.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from scripts.enumeration_fresh_geometry import json_sha
from scripts.prepare_enumeration_fresh_causal_registry import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--selection-manifest", type=Path,
                        help="Frozen N=10 discovery selection; overrides the historical mixed-N layer only")
    args = parser.parse_args()
    tick = time.monotonic()
    manifest = json.loads((args.registry / "manifest.json").read_text())
    cohorts = json.loads(args.cohorts.read_text())
    cfg = json.loads((args.bundle / "protocol.json").read_text())
    baseline = json.loads((args.bundle / "manifest.json").read_text())
    assert sha(args.bundle / "manifest.json") == manifest["baseline_manifest_sha256"] == cohorts["baseline_manifest_sha256"]
    assert sha(args.bundle / "protocol.json") == baseline["protocol_sha256"]
    for name, expected in baseline["code_sha256"].items():
        if name.startswith(("src/", "scripts/")):
            assert sha(ROOT / name) == expected, name
    for name, expected in manifest["new_entrypoints_sha256"].items():
        assert sha(ROOT / "scripts" / name) == expected, name
    model_label, mode = manifest["model"], manifest["mode"]
    assert sha(args.registry / "manifest.json") == cohorts["registries"][f"{model_label}/{mode}"]["manifest_sha256"]
    generations = args.registry / "adapted_generations.jsonl"
    assert sha(generations) == manifest["adapted_generations_sha256"]
    seeds = cohorts["models"][model_label]["confirmation_seeds"]
    if not seeds:
        raise ValueError("No common Update-eligible confirmation seeds; report coverage, do not replace pool")
    layer = cfg["update"]["layers_one_based"][model_label][mode] - 1
    selection_sha256 = None
    if args.selection_manifest:
        from realistic_niah_v6.update_n10 import validate_selection
        selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        layer = validate_selection(selection, model_label, mode, seeds)
        for cell in selection["cells"]:
            assert sha(Path(cell["selection_path"])) == cell["selection_sha256"]
        selection_sha256 = sha(args.selection_manifest)
    if args.smoke:
        seeds = seeds[:1]
    cohort_mode = ("indexed_positive_control" if mode != "thinking" else
                   "natural_noindex" if model_label == "Qwen3-8B" else "prompt_conditioned_noindex")
    jobs = []
    for k in cfg["update"]["donor_k"]:
        for direction in cfg["update"]["directions"]:
            for scope in cfg["update"]["scopes"]:
                jobs.append({"k": k, "j": k - 1 if direction == "forward" else k + 1,
                             "direction": direction, "scope": scope})
    # Every geometry is exercised on the fixed first common seed in smoke;
    # requiring a favorable behavioral result is deliberately absent.
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"status": "RUNNING", "completed_jobs": 0, "total_jobs": len(jobs),
             "completed_rows": 0, "total_rows": len(jobs) * len(seeds) * 3,
             "smoke": args.smoke, "jobs": []}
    write(args.output / "status.json", state)
    import torch
    from realistic_niah_v4.modeling import load_registered_model
    from realistic_niah_v4.spec import resolve_model_spec
    if mode != "thinking":
        from realistic_niah_v6.kernel import install_v6_kernel_adapters, install_v6_specialized_geometry
        install_v6_kernel_adapters()
        install_v6_specialized_geometry(mode)
    from scripts import run_realistic_niah_v5_natural_aligned_progress_transplant as kernel
    try:
        model, tokenizer, adapter = load_registered_model(resolve_model_spec(model_label),
            cache_dir=args.cache_dir, device_map="auto", torch_dtype="bfloat16", attention_backend="sdpa")
        model.eval()
        text_cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        assert 0 <= layer < adapter.num_layers - 1
        write(args.output / "runtime.json", {
            "model": model_label, "mode": mode, "layers": adapter.num_layers, "patch_layer_zero_based": layer,
            "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
            "python": sys.version, "gpu": torch.cuda.get_device_name(),
            "model_revision": resolve_model_spec(model_label).revision,
            "model_source_sha256": sha(sys.modules[type(model).__module__].__file__),
            "dtype": str(next(model.parameters()).dtype), "backend": "sdpa",
            "registry_sha256": sha(args.registry / "manifest.json"), "cohorts_sha256": sha(args.cohorts),
            "runner_sha256": sha(__file__), "command": sys.argv, "jobs": jobs, "seeds": seeds,
            "n10_selection_manifest_sha256": selection_sha256,
            "model_load_seconds": time.monotonic() - tick,
            "attention_readout": False, "all_three_conditions_generate": True})
        original_load = kernel._experiment_model
        original_generate = kernel.generate_answer_completion_from_prefill
        original_prefill = kernel._chunked_prefill_with_span_replacement
        kernel._experiment_model = lambda _args: (model, tokenizer, adapter)
        try:
            for job in jobs:
                folder = args.output / f"k{job['k']}_{job['direction']}_{job['scope']}"
                folder.mkdir(exist_ok=False)
                patch_scope = "item_span" if job["scope"] == "item_span" else "fixed_suffix"
                width = 4 if job["scope"] == "four_token_tail" else 1
                command = [str(kernel.__file__), "--model", model_label, "--cache-dir", str(args.cache_dir),
                           "--device-map", "auto", "--torch-dtype", "bfloat16", "--attention-backend", "sdpa",
                           "--prefill-chunk-size", str(cfg["prefill_chunk_size"]), "--generations", str(generations),
                           "--cohort-mode", cohort_mode, "--gold-count", "10", "--receiver-occurrence", str(job["j"]),
                           "--donor-occurrence", str(job["k"]), "--layers", str(layer),
                           "--conditions", *cfg["update"]["conditions"],
                           "--generation-conditions", *cfg["update"]["generation_conditions"],
                           "--max-new-tokens", str(cfg["update"]["max_new_tokens"]), "--tail-offset", "0",
                           "--patch-scope", patch_scope, "--patch-width", str(width),
                           "--seeds", *map(str, seeds), "--output", str(folder)]
                write(folder / "command.json", command)
                generated_rows, hook_rows = [], []
                with (folder / "raw_generations.jsonl").open("x") as raw_out, (folder / "prefill_hooks.jsonl").open("x") as hooks_out:
                    def retain_generation(*values, **options):
                        result = original_generate(*values, **options)
                        encoding = values[2]
                        record = {"seed": int(encoding.seed), "condition": cfg["update"]["generation_conditions"][len(generated_rows) % 3],
                                  "query_position": encoding.query_position, **result}
                        generated_rows.append(record)
                        raw_out.write(json.dumps(record, ensure_ascii=True) + "\n")
                        raw_out.flush()
                        return result
                    def retain_hook(*values, **options):
                        result = original_prefill(*values, **options)
                        encoding = values[2]
                        record = {"seed": int(encoding.seed), "layer": options["layer"], "site": options["site"],
                                  "width": int(options["states"].shape[0]), "applications": int(result[1]),
                                  "delta_norm": float(result[2]), "input_ids_sha256": json_sha(list(encoding.input_ids))}
                        assert record["applications"] == 1 and math.isfinite(record["delta_norm"])
                        hook_rows.append(record)
                        hooks_out.write(json.dumps(record) + "\n")
                        hooks_out.flush()
                        return result
                    kernel.generate_answer_completion_from_prefill = retain_generation
                    kernel._chunked_prefill_with_span_replacement = retain_hook
                    saved_argv = sys.argv
                    try:
                        sys.argv = command
                        kernel.main()
                    finally:
                        sys.argv = saved_argv
                with (folder / "trials.jsonl").open() as f:
                    trials = [json.loads(line) for line in f]
                assert len(trials) == len(generated_rows) == len(seeds) * 3
                assert len(hook_rows) == len(trials) * 2
                for index, (trial, raw) in enumerate(zip(trials, generated_rows)):
                    assert trial["seed"] == raw["seed"] and trial["condition"] == raw["condition"]
                    assert trial["completion_text"] == raw["completion_text"]
                    assert trial["patch_applications"] == 1 and trial["generated_token_count"] > 0
                    assert math.isfinite(trial["donor_vs_receiver_sum_logodds"])
                    norms = [hook_rows[index * 2 + offset]["delta_norm"] for offset in (0, 1)]
                    if trial["condition"] == "donor_to_receiver":
                        assert min(norms) > 0, "Target patch made no actual state change"
                    else:
                        assert max(norms) == 0, "Self/native patch altered its own state"
                assert model.config._attn_implementation == text_cfg._attn_implementation == "sdpa"
                audit = {"status": "PASS", "trials": len(trials), "raw_generations": len(generated_rows),
                         "prefill_hook_calls": len(hook_rows), "backend_after": "sdpa",
                         "trials_sha256": sha(folder / "trials.jsonl"), "raw_sha256": sha(folder / "raw_generations.jsonl"),
                         "hooks_sha256": sha(folder / "prefill_hooks.jsonl"), "behavioral_effect_required": False}
                write(folder / "technical_audit.json", audit)
                state["jobs"].append({**job, **audit})
                state.update(completed_jobs=state["completed_jobs"] + 1,
                             completed_rows=state["completed_rows"] + len(trials), seconds=time.monotonic() - tick)
                write(args.output / "status.json", state)
        finally:
            kernel._experiment_model = original_load
            kernel.generate_answer_completion_from_prefill = original_generate
            kernel._chunked_prefill_with_span_replacement = original_prefill
        state["status"] = "COMPLETE"
    except BaseException as error:
        state.update(status="FAILED", error=repr(error))
        raise
    finally:
        state["seconds"] = time.monotonic() - tick
        write(args.output / "status.json", state)


if __name__ == "__main__":
    main()
