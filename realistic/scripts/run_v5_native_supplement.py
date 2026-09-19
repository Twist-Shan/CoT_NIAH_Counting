#!/usr/bin/env python3
"""Run a frozen Native supplement bundle; default is a CPU-only dry run."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(p))
from realistic_niah_v5.native_supplement import (
    MODELS, PROGRESS_SEEDS, compare_replay, digest, read_rows, summarize_backward,
    validate_answer, validate_answer_shards, validate_progress, verify_bundle, write_json, write_rows,
)


def commands(bundle: Path, output: Path, model: str, task: str, cache: Path, device_map: str) -> list[dict]:
    plan = verify_bundle(bundle)
    code = bundle / "code"
    data = bundle / "inputs" / model
    shared = ["--model", model, "--cache-dir", str(cache), "--device-map", device_map,
              "--torch-dtype", plan["torch_dtype"], "--attention-backend", plan["attention_backend"]]
    if task == "answer_dense":
        result = []
        for phase, layers in [("replay", plan["models"][model]["old_layers"]), ("missing", plan["models"][model]["missing_layers"])]:
            path = output / phase / "trials.jsonl"
            args = ["causal-patch", *shared, "--config", str(code / "configs/realistic_niah_v5.json"),
                    "--generations", str(data / "generations.jsonl"), "--pairs", str(data / "pairs.jsonl"),
                    "--output", str(path), "--layers", *map(str, layers),
                    "--receiver-site-id", "answer_query_v3", "--donor-site-id", "answer_query_v3",
                    "--conditions", "self_patch", "full_donor_patch", "--max-new-tokens", "16", "--restartable"]
            result.append({"phase": phase, "entry": "run_realistic_niah_v5.py", "args": args, "output": str(path), "layers": layers})
        return result
    if model != "Gemma4-E4B":
        raise ValueError("gemma_backward applies only to Gemma4-E4B")
    result = []
    for direction, targets in [("forward", [6]), ("backward", [4, 6, 8])]:
        for target in targets:
            receiver = target - 1 if direction == "forward" else target + 1
            phase = "replay_forward_k6" if direction == "forward" else f"backward_k{target}"
            out = output / phase
            args = [*shared, "--generations", str(data / "prompted_cohort.jsonl"),
                    "--cohort-mode", "prompt_conditioned_noindex", "--gold-count", "10",
                    "--receiver-occurrence", str(receiver), "--donor-occurrence", str(target),
                    "--tail-offset", "0", "--patch-scope", "item_span", "--layers", "16",
                    "--conditions", "receiver_self", "native_donor", "donor_to_receiver",
                    "--generation-conditions", "receiver_self", "donor_to_receiver", "--max-new-tokens", "96",
                    "--run-attention", "--targeted-selection", str(code / "configs/realistic_niah_v5_gemma_shared_k6_targeted_selection_frozen.json"),
                    "--targeted-routing", str(code / "configs/realistic_niah_v5_gemma_shared_k6_causal_routes_frozen.json"),
                    "--seeds", *map(str, PROGRESS_SEEDS), "--output", str(out)]
            result.append({"phase": phase, "entry": "run_realistic_niah_v5_natural_aligned_progress_transplant.py", "args": args,
                           "output": str(out / "trials.jsonl"), "target": target, "direction": direction})
    return result


def runtime_snapshot() -> dict:
    import torch
    import transformers
    if not torch.cuda.is_available():
        raise RuntimeError("--execute requires the rented GPU runtime; dry-run works on CPU")
    packages = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "accelerate", "numpy", "pandas")}
    transformer_root = Path(transformers.__file__).parent
    implementation_files = [transformer_root / "modeling_utils.py"]
    for family in ("qwen3", "gemma4"):
        folder = transformer_root / "models" / family
        implementation_files.extend(sorted(folder.glob("*.py")))
    return {
        "python": sys.version, "packages": packages, "cuda": torch.version.cuda,
        "torch_git_version": torch.version.git_version,
        "gpus": [{"name": torch.cuda.get_device_name(i), "memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                 for i in range(torch.cuda.device_count())],
        "transformers_source": digest(Path(transformers.__file__)),
        "transformers_implementation_sha256": {str(p.relative_to(transformer_root)): digest(p) for p in implementation_files},
    }


def worker(path: Path) -> None:
    """Invoke the archived CLI with an observational model-load audit only."""
    job = json.loads(path.read_text())
    entry = Path(job["code"]) / "scripts" / job["entry"]
    spec = importlib.util.spec_from_file_location("supplement_worker_entry", entry)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    attr = "_model" if job["entry"] == "run_realistic_niah_v5.py" else "_experiment_model"
    original = getattr(module, attr)

    def audited_model(args):
        started = time.monotonic()
        model, tokenizer, adapter = original(args)
        if adapter.num_layers != job["expected_layers"]:
            raise ValueError("Loaded model has the wrong decoder layer count")
        config = model.config
        text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
        commit = getattr(config, "_commit_hash", None)
        if commit is not None and commit != job["model_spec"]["revision"]:
            raise ValueError("Loaded model revision differs from the frozen revision")
        audit = {
            "model_spec": job["model_spec"], "observed_revision": commit,
            "model_class": type(model).__name__, "tokenizer_class": type(tokenizer).__name__,
            "effective_attention_backend": getattr(text_config, "_attn_implementation", None),
            "layers": adapter.num_layers, "layer_types": list(adapter.layer_types),
            "device_map": {str(k): str(v) for k, v in getattr(model, "hf_device_map", {}).items()},
            "training": model.training, "dtype": str(next(model.parameters()).dtype),
            "model_source_sha256": digest(Path(sys.modules[type(model).__module__].__file__)),
            "load_seconds": time.monotonic() - started,
        }
        if model.training or audit["effective_attention_backend"] != "sdpa" or audit["dtype"] != "torch.bfloat16":
            raise ValueError(f"Loaded runtime differs from frozen protocol: {audit}")
        write_json(path.parent / "loaded_model.json", audit)
        return model, tokenizer, adapter

    setattr(module, attr, audited_model)
    sys.argv = [str(entry), *job["args"]]
    module.main()


def run_phase(bundle: Path, output: Path, model: str, job: dict, plan: dict) -> None:
    folder = output / job["phase"]
    folder.mkdir(parents=True, exist_ok=True)
    payload = {**job, "code": str(bundle / "code"), "expected_layers": len(plan["models"][model]["all_layers"]),
               "model_spec": plan["models"][model]["model_spec"]}
    write_json(folder / "command.json", payload)
    command = [sys.executable, "-u", str(bundle / "code/scripts/run_v5_native_supplement.py"), "--worker", str(folder / "command.json")]
    started = time.monotonic()
    code = None
    try:
        with (folder / "run.log").open("a", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=bundle / "code", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
        if code:
            raise RuntimeError(f"Phase {job['phase']} failed, exit={code}; see {folder / 'run.log'}")
    finally:
        write_json(folder / "process_status.json", {"status": "COMPLETE" if code == 0 else "FAILED",
                   "exit_code": code, "elapsed_seconds": time.monotonic() - started, "command": command})


def validate_progress_geometry(folder: Path) -> None:
    rows = read_rows(folder / "geometry_audit.jsonl")
    if sorted(r["seed"] for r in rows) != PROGRESS_SEEDS:
        raise ValueError("Progress geometry does not cover all registered seeds")
    for r in rows:
        if not r["endpoint_aligned"] or r["aligned_absolute_site"] != r["aligned_donor_site"]:
            raise ValueError("Progress absolute positions differ")
        if not r["deletion_avoids_prompt_records"] or not r["deletion_avoids_special_tokens"] or r["hidden_state_resampling"]:
            raise ValueError("Progress alignment changed record tokens or resampled states")


def run(args) -> None:
    bundle, output = args.bundle.resolve(), args.output.resolve()
    plan = verify_bundle(bundle)
    jobs = commands(bundle, output, args.model, args.task, args.cache_dir.resolve(), args.device_map)
    if not args.execute:
        print(json.dumps({"status": "DRY_RUN_NO_MODEL_LOAD", "plan_sha256": plan["plan_sha256"], "jobs": jobs}, indent=2))
        return
    if output.is_relative_to(bundle) or bundle.is_relative_to(output):
        raise ValueError("Run outputs and frozen bundle must be separate directories")
    runtime = runtime_snapshot()
    identity = {"plan_sha256": plan["plan_sha256"], "task": args.task, "model": args.model,
                "device_map": args.device_map, "cache_dir": str(args.cache_dir.resolve()), "runtime": runtime}
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "run_manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError("Resume would mix different input/code/runtime; use a new run directory")
    elif any(output.iterdir()):
        raise ValueError("Nonempty output directory lacks a compatible run manifest")
    else:
        write_json(manifest, identity)
    started = time.monotonic()
    try:
        data = bundle / "inputs" / args.model
        all_rows = []
        for job in jobs:
            trials = Path(job["output"])
            if args.task == "answer_dense":
                pairs = read_rows(data / "pairs.jsonl")
                validate_answer_shards(trials.parent / "trials_shards", pairs, job["layers"])
            if not trials.exists():
                run_phase(bundle, output, args.model, job, plan)
            rows = read_rows(trials)
            if args.task == "answer_dense":
                validate_answer(rows, pairs, job["layers"])
                if job["phase"] == "replay":
                    agreement = compare_replay(read_rows(data / "historical_answer_trials.jsonl"), rows)
                    write_json(output / "historical_replay_audit.json", agreement)
                    if agreement["status"] != "PASS":
                        raise ValueError("Historical replay differs; new-layer execution halted for diagnosis")
                all_rows.extend(rows)
            else:
                validate_progress(rows, [job["target"]], direction=job["direction"])
                validate_progress_geometry(trials.parent)
                if job["direction"] == "forward":
                    agreement = compare_replay(read_rows(data / "historical_forward_k6.jsonl"), rows, progress=True)
                    write_json(output / "historical_replay_audit.json", agreement)
                    if agreement["status"] != "PASS":
                        raise ValueError("Gemma forward replay differs; backward execution halted for diagnosis")
                else:
                    all_rows.extend(rows)
        if args.task == "answer_dense":
            validate_answer(all_rows, pairs, plan["models"][args.model]["all_layers"])
            final = output / "dense/trials.jsonl"
            write_rows(final, all_rows)
            command = [sys.executable, str(bundle / "code/scripts/analyze_v5_answer_query_layer_sweep.py"),
                       "--trials", str(final), "--pairs", str(data / "pairs.jsonl"), "--output-dir", str(output / "dense/analysis"),
                       "--expected-layers", *map(str, plan["models"][args.model]["all_layers"])]
            subprocess.run(command, check=True, cwd=bundle / "code")
        else:
            final = output / "backward/trials.jsonl"
            write_rows(final, all_rows)
            write_json(output / "backward/analysis.json", summarize_backward(all_rows))
        write_json(output / "completion_audit.json", {"status": "PASS", "task": args.task, "model": args.model,
                   "trials": len(all_rows), "trials_sha256": digest(final), "elapsed_seconds": time.monotonic() - started,
                   "plan_sha256": plan["plan_sha256"], "fresh_confirmation": False})
    except Exception as error:
        write_json(output / "failure.json", {"status": "FAILED", "reason": str(error), "elapsed_seconds": time.monotonic() - started})
        raise


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--task", choices=("answer_dense", "gemma_backward"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--execute", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
