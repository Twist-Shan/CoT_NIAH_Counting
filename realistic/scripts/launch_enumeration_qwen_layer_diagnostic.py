"""Freeze and queue a paired Qwen Index layer diagnostic after the primary assays."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_n10_selection(primary, cfg, cell, runtime):
    """Bind the diagnostic to the frozen N10 layer and confirmation cohort."""
    path = primary / "selection_manifest.json"
    digest = sha(path)
    if digest != cfg["primary_selection_manifest_sha256"]:
        raise ValueError("N10 selection manifest hash differs from the protocol")
    selection = read(path)
    if selection["confirmation_used_for_selection"] is not False:
        raise ValueError("N10 selection used confirmation outcomes")
    chosen = next(c for c in selection["cells"] if c["model"] == cfg["model"] and c["mode"] == cfg["mode"])
    if chosen["layer_one_based"] != cfg["reuse_layer_one_based"]:
        raise ValueError("N10 reference layer differs")
    if chosen["confirmation_seeds"] != cell["selected_seeds"]:
        raise ValueError("N10 confirmation seeds differ")
    if read(runtime)["n10_selection_manifest_sha256"] != digest:
        raise ValueError("Reference runtime is not bound to this N10 selection")
    return path, digest


def freeze(args):
    tick = time.monotonic()
    root, stage = args.root.resolve(), args.output.resolve()
    cfg = read(args.protocol)
    assert cfg["layers_one_based"] == [1, 7, 13, 19, 25, 30, 35]
    assert cfg["conditions"] == ["receiver_self", "native_donor", "donor_to_receiver"]
    assert (cfg["model"], cfg["mode"], cfg["scope"], cfg["donor_k"], cfg["reuse_layer_one_based"]) == (
        "Qwen3-8B", "enumeration_index", "item_span", 6, 30)
    assert cfg["directions"] == ["forward", "backward"] and cfg["max_new_tokens"] == 96
    assert cfg["prefill_chunk_size"] == 512
    primary_name = cfg.get("primary_stage", "fresh_native_update_v1")
    assert primary_name in ("fresh_native_update_v1", "fresh_n10_update_v1")
    primary = root / primary_name
    assert read(primary / "gpu_pipeline_status.json")["status"] == "COMPLETE"
    assert read(primary / "assembly_audit.json")["status"] == "PASS"
    main = read(primary / "manifest.json")
    cell = next(c for c in main["cells"] if c["model"] == cfg["model"] and c["mode"] == cfg["mode"])
    assert len(cell["selected_seeds"]) == 10
    registry = Path(cell["registry"])
    registered = read(registry / "manifest.json")
    assert sha(registry / "manifest.json") == cell["registry_sha256"]
    selected = {r["seed"]: r for r in read(registry / "ledger.json") if r["gold_count"] == 10}
    assert all(selected[s]["update_eligible"] for s in cell["selected_seeds"])
    runtime = primary / "jobs/formal" / cfg["model"] / cfg["mode"] / "runtime.json"
    assert read(runtime)["patch_layer_zero_based"] == cfg["reuse_layer_one_based"] - 1
    source_paths = [root / "fresh_v1/manifest.json", root / "fresh_v1/protocol.json",
                    root / "fresh_causal_v1/code_manifest.json", primary / "manifest.json", primary / "assembly_audit.json",
                    registry / "manifest.json", registry / "ledger.json", registry / "adapted_generations.jsonl", runtime]
    assert sha(registry / "adapted_generations.jsonl") == registered["adapted_generations_sha256"]
    selection_hash = None
    if primary_name == "fresh_n10_update_v1":
        selection_path, selection_hash = validate_n10_selection(primary, cfg, cell, runtime)
        source_paths.append(selection_path)
    reused = []
    for direction in cfg["directions"]:
        folder = primary / "primary" / cfg["model"] / cfg["mode"] / f"k6_{direction}_item_span"
        data = rows(folder / "trials.jsonl")
        identities = {(r["seed"], r["condition"]) for r in data}
        assert len(data) == 30 and identities == {(s, c) for s in cell["selected_seeds"] for c in cfg["conditions"]}
        assert all(r["layer"] == 29 and r["donor_occurrence_k"] == 6 and r["patch_scope"] == "item_span" for r in data)
        for name in ("trials.jsonl", "raw_generations.jsonl", "prefill_hooks.jsonl"):
            source_paths.append(folder / name)
        reused.append({"direction": direction, "layer_one_based": 30, "path": str(folder), "rows": 30})
    prerequisite = (primary if primary_name == "fresh_n10_update_v1" else root / "fresh_retrieve_gpu_v2") / "gpu_pipeline_status.json"
    assert prerequisite.exists() and read(prerequisite)["status"] in ("RUNNING", "COMPLETE")
    stage.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, stage / Path(__file__).name)
    shutil.copy2(args.protocol, stage / "protocol.json")
    jobs = [{"id": f"{phase}/{direction}", "phase": phase, "direction": direction,
             "seeds": cell["selected_seeds"][:1] if phase == "smoke" else cell["selected_seeds"],
             "layers_one_based": [x for x in cfg["layers_one_based"] if x != 30],
             "expected_rows": 18 if phase == "smoke" else 180}
            for phase in ("smoke", "formal") for direction in cfg["directions"]]
    write(stage / "jobs.json", jobs)
    manifest = {"schema": "enumeration_qwen_layer_diagnostic_manifest_v1", "status": "FROZEN_BEFORE_DIAGNOSTIC_GPU",
                "created_utc": datetime.now(timezone.utc).isoformat(), "root": str(root), "registry": str(registry),
                "source_runtime": str(runtime), "source_code": str(root / "fresh_causal_v1/code"),
                "selected_seeds": cell["selected_seeds"], "reused": reused,
                "primary_stage": primary_name, "primary_selection_manifest_sha256": selection_hash,
                "prerequisite": str(prerequisite), "prerequisite_complete_phase": (
                    "N10_UPDATE_GPU_COMPLETE_ANALYSIS_PENDING" if primary_name == "fresh_n10_update_v1"
                    else "RETRIEVE_GPU_COMPLETE_ANALYSIS_PENDING"),
                "cache_dir": str(args.cache_dir), "input_sha256": {str(p): sha(p) for p in source_paths},
                "protocol_sha256": sha(stage / "protocol.json"), "jobs_sha256": sha(stage / "jobs.json"),
                "entrypoint_sha256": sha(__file__), "command": sys.argv, "seconds": time.monotonic() - tick}
    write(stage / "manifest.json", manifest)
    verify(stage)
    write(stage / "cpu_validation.json", {"status": "PASS", "scope": "Frozen source hashes, ten-seed eligibility and complete reused L30 grid",
          "gpu_smoke": "PENDING", "manifest_sha256": sha(stage / "manifest.json"), "seconds": time.monotonic() - tick})
    print(json.dumps({"status": manifest["status"], "manifest_sha256": sha(stage / "manifest.json"),
                      "smoke_rows": 36, "new_formal_rows": 360, "reused_rows": 60}), flush=True)


def verify(stage):
    manifest = read(stage / "manifest.json")
    assert sha(stage / Path(__file__).name) == manifest["entrypoint_sha256"]
    assert sha(stage / "protocol.json") == manifest["protocol_sha256"]
    assert sha(stage / "jobs.json") == manifest["jobs_sha256"]
    for path, expected in manifest["input_sha256"].items():
        assert sha(path) == expected, path
    code = read(Path(manifest["root"]) / "fresh_causal_v1/code_manifest.json")
    for name, expected in {**code["original_code_sha256"], **code["additive_code_sha256"]}.items():
        assert sha(Path(manifest["source_code"]) / name) == expected, name
    return manifest


def execute_job(stage, manifest, cfg, job, model, tokenizer, adapter, kernel):
    tick = time.monotonic()
    from scripts.enumeration_fresh_geometry import json_sha
    folder = stage / "jobs" / job["id"]
    folder.mkdir(parents=True, exist_ok=False)
    layers = [x - 1 for x in job["layers_one_based"]]
    command = [str(kernel.__file__), "--model", cfg["model"], "--cache-dir", manifest["cache_dir"],
               "--device-map", "auto", "--torch-dtype", "bfloat16", "--attention-backend", "sdpa",
               "--prefill-chunk-size", str(cfg["prefill_chunk_size"]),
               "--generations", str(Path(manifest["registry"]) / "adapted_generations.jsonl"),
               "--cohort-mode", "indexed_positive_control", "--gold-count", "10",
               "--receiver-occurrence", "5" if job["direction"] == "forward" else "7", "--donor-occurrence", "6",
               "--layers", *map(str, layers), "--conditions", *cfg["conditions"],
               "--generation-conditions", *cfg["conditions"], "--max-new-tokens", str(cfg["max_new_tokens"]),
               "--tail-offset", "0", "--patch-scope", "item_span", "--patch-width", "1",
               "--seeds", *map(str, job["seeds"]), "--output", str(folder)]
    write(folder / "command.json", command)
    original_load, original_generate, original_prefill = (kernel._experiment_model,
        kernel.generate_answer_completion_from_prefill, kernel._chunked_prefill_with_span_replacement)
    raw, hooks = [], []
    try:
        kernel._experiment_model = lambda _args: (model, tokenizer, adapter)
        with (folder / "raw_generations.jsonl").open("x") as raw_file, (folder / "prefill_hooks.jsonl").open("x") as hook_file:
            def generation(*values, **options):
                result = original_generate(*values, **options)
                encoding = values[2]
                record = {"seed": int(encoding.seed), "condition": cfg["conditions"][len(raw) % 3],
                          "layer": layers[(len(raw) // 3) % len(layers)],
                          "query_position": encoding.query_position, **result}
                raw.append(record)
                raw_file.write(json.dumps(record, ensure_ascii=True) + "\n")
                raw_file.flush()
                return result
            def prefill(*values, **options):
                result = original_prefill(*values, **options)
                encoding = values[2]
                record = {"seed": int(encoding.seed), "layer": options["layer"], "site": options["site"],
                          "width": int(options["states"].shape[0]), "applications": int(result[1]),
                          "delta_norm": float(result[2]), "input_ids_sha256": json_sha(list(encoding.input_ids))}
                assert record["applications"] == 1 and math.isfinite(record["delta_norm"])
                hooks.append(record)
                hook_file.write(json.dumps(record) + "\n")
                hook_file.flush()
                return result
            kernel.generate_answer_completion_from_prefill = generation
            kernel._chunked_prefill_with_span_replacement = prefill
            saved_argv = sys.argv
            try:
                sys.argv = command
                kernel.main()
            finally:
                sys.argv = saved_argv
        trials = rows(folder / "trials.jsonl")
        assert len(trials) == len(raw) == job["expected_rows"] and len(hooks) == 2 * len(trials)
        expected = {(s, layer, c) for s in job["seeds"] for layer in layers for c in cfg["conditions"]}
        assert {(r["seed"], r["layer"], r["condition"]) for r in trials} == expected
        for i, (trial, generation_row) in enumerate(zip(trials, raw)):
            for key in ("seed", "layer", "condition", "completion_text", "generated_token_count", "generation_truncated"):
                assert trial[key] == generation_row[key], key
            assert trial["patch_applications"] == 1 and trial["generated_token_count"] > 0
            assert math.isfinite(trial["donor_vs_receiver_sum_logodds"])
            for hook in hooks[2*i:2*i+2]:
                assert hook["seed"] == trial["seed"] and hook["layer"] == trial["layer"]
                assert hook["site"] == trial["shared_commit_position"] and hook["width"] == trial["patch_width"]
                assert (hook["delta_norm"] > 0 if trial["condition"] == "donor_to_receiver" else hook["delta_norm"] == 0)
        assert model.config._attn_implementation == "sdpa"
        result = {"status": "COMPLETE", "completed_rows": len(trials), "expected_rows": job["expected_rows"],
                  "behavioral_effect_required": False, "seconds": time.monotonic() - tick,
                  "files_sha256": {name: sha(folder / name) for name in
                      ("trials.jsonl", "raw_generations.jsonl", "prefill_hooks.jsonl", "geometry_audit.jsonl", "manifest.json")}}
        write(folder / "status.json", result)
        return result
    finally:
        kernel._experiment_model = original_load
        kernel.generate_answer_completion_from_prefill = original_generate
        kernel._chunked_prefill_with_span_replacement = original_prefill


def run(stage):
    start = time.monotonic()
    manifest = verify(stage)
    state = {"status": "RUNNING", "phase": "WAIT_PRIMARY_RETRIEVE_COMPLETE", "pid": os.getpid(), "jobs": []}
    status_path = stage / "gpu_pipeline_status.json"
    with status_path.open("x") as f:
        json.dump(state, f)
    def save():
        state["seconds"] = time.monotonic() - start
        write(status_path, state)
    try:
        while True:
            previous = read(manifest["prerequisite"])
            if previous["status"] in ("FAILED", "SUPERSEDED"):
                raise RuntimeError("Prerequisite failed or was superseded; do not launch diagnostic GPU")
            if previous["status"] == "COMPLETE":
                assert previous["phase"] == manifest["prerequisite_complete_phase"]
                break
            if time.monotonic() - start > 48*3600:
                raise TimeoutError("Primary queue not complete after 48 hours")
            save()
            time.sleep(20)
        verify(stage)
        state["phase"] = "LOAD_MODEL"
        save()
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            os.environ[name] = "1"
        code = Path(manifest["source_code"])
        sys.path[:0] = [str(code / "src"), str(code)]
        import torch
        from realistic_niah_v4.modeling import load_registered_model
        from realistic_niah_v4.spec import resolve_model_spec
        from realistic_niah_v6.kernel import install_v6_kernel_adapters, install_v6_specialized_geometry
        install_v6_kernel_adapters()
        install_v6_specialized_geometry("enumeration_index")
        from scripts import run_realistic_niah_v5_natural_aligned_progress_transplant as kernel
        cfg = read(stage / "protocol.json")
        model, tokenizer, adapter = load_registered_model(resolve_model_spec(cfg["model"]),
            cache_dir=manifest["cache_dir"], device_map="auto", torch_dtype="bfloat16", attention_backend="sdpa")
        model.eval()
        runtime = {"model_revision": resolve_model_spec(cfg["model"]).revision,
                   "model_source_sha256": sha(sys.modules[type(model).__module__].__file__),
                   "dtype": str(next(model.parameters()).dtype), "backend": "sdpa", "layers": adapter.num_layers,
                   "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
                   "gpu": torch.cuda.get_device_name(), "python": sys.version,
                   "entrypoint_sha256": sha(__file__), "source_runtime": manifest["source_runtime"],
                   "seconds_before_jobs": time.monotonic()-start}
        reference = read(manifest["source_runtime"])
        for key in ("model_revision", "model_source_sha256", "dtype", "backend", "layers", "torch", "transformers"):
            assert runtime[key] == reference[key], f"L30 reuse runtime mismatch: {key}"
        write(stage / "runtime.json", runtime)
        for job in read(stage / "jobs.json"):
            row = {**job, "status": "RUNNING"}
            state["jobs"].append(row)
            state["phase"] = job["id"]
            save()
            result = execute_job(stage, manifest, cfg, job, model, tokenizer, adapter, kernel)
            row.update(status="COMPLETE", completed_rows=result["completed_rows"], seconds=result["seconds"])
            save()
        state.update(status="COMPLETE", phase="QWEN_INDEX_LAYER_DIAGNOSTIC_GPU_COMPLETE_ANALYSIS_PENDING")
    except BaseException as error:
        state.update(status="FAILED", error=repr(error))
        raise
    finally:
        save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    f = sub.add_parser("freeze")
    for name in ("root", "protocol", "output", "cache-dir"):
        f.add_argument("--"+name, type=Path, required=True)
    sub.add_parser("run").add_argument("--stage", type=Path, required=True)
    a = p.parse_args()
    freeze(a) if a.action == "freeze" else run(a.stage)


if __name__ == "__main__":
    main()
