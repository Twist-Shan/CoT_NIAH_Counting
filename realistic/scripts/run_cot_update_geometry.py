"""Execute frozen update jobs; retain each process log and elapsed time."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=Path("outputs/external/lambda_nfs_CoT-Native-thinking-v5_cache_huggingface"))
    p.add_argument("--phase", choices=("smoke", "primary", "scope", "geometry", "all"), required=True)
    p.add_argument("--models", nargs="+", choices=("Qwen3-8B", "Gemma4-E4B"),
                   help="Run only these models, with a separate per-model status file")
    args = p.parse_args()
    base = args.bundle.resolve()
    manifest = json.loads((base / "manifest.json").read_text())
    if args.models:
        manifest["models"] = {name: manifest["models"][name] for name in args.models}
    for name, expected in manifest["input_code_sha256"].items():
        if sha(base / name) != expected:
            raise RuntimeError(f"Frozen input/code changed: {name}")
    code_hashes = {name[5:]: value for name, value in manifest["input_code_sha256"].items()
                   if name.startswith("code/")}
    jobs = []
    for model, info in manifest["models"].items():
        scopes = ["item_span"] if args.phase in ("smoke", "primary") else manifest["scopes"]
        if args.phase == "scope":
            scopes = [s for s in scopes if s != "item_span"]
        if args.phase != "geometry":
            for scope in scopes:
                for direction in (["forward"] if args.phase == "smoke" else manifest["directions"]):
                    for k in ([6] if args.phase == "smoke" else manifest["donor_k"]):
                        job_id = f"{model}/{scope}/{direction}_k{k}"
                        folder = base / ("smoke" if args.phase == "smoke" else "runs") / job_id
                        j = k - 1 if direction == "forward" else k + 1
                        inputs = base / "inputs" / f"{model}_confirmation.jsonl"
                        selection = base / "code/configs" / f"realistic_niah_v5_{info['bank']}_targeted_selection_frozen.json"
                        routing = base / "code/configs" / f"realistic_niah_v5_{info['bank']}_causal_routes_frozen.json"
                        cli = ["--model", model, "--cache-dir", str(args.cache_dir), "--device-map", "auto",
                               "--torch-dtype", "bfloat16", "--attention-backend", "sdpa", "--prefill-chunk-size", "512",
                               "--generations", str(inputs), "--cohort-mode", info["cohort_mode"], "--gold-count", "10",
                               "--receiver-occurrence", str(j), "--donor-occurrence", str(k), "--tail-offset", "0",
                               "--patch-scope", "item_span" if scope == "item_span" else "fixed_suffix",
                               "--patch-width", "4" if scope == "four_token_tail" else "1",
                               "--layers", str(info["layer_zero_based"]),
                               "--conditions", "receiver_self", "native_donor", "donor_to_receiver",
                               "--generation-conditions", "receiver_self", "donor_to_receiver",
                               "--max-new-tokens", "96", "--run-attention", "--targeted-selection", str(selection),
                               "--targeted-routing", str(routing), "--seeds",
                               *map(str, info["confirmation_seeds"][:1] if args.phase == "smoke" else info["confirmation_seeds"]),
                               "--output", str(folder / "results")]
                        job = dict(backend_variant="fixed", entry="run_realistic_niah_v5_natural_aligned_progress_transplant.py",
                                   loader_name="_experiment_model", args=cli, code_sha256=code_hashes,
                                   input_sha256={str(f): sha(f) for f in (inputs, selection, routing)})
                        jobs.append((job_id, folder, job))
    # Run both models' primary item-span patches before the supplementary scopes.
    model_order = {name: i for i, name in enumerate(manifest["models"])}
    jobs.sort(key=lambda value: ("/item_span/" not in value[0], model_order[value[0].split("/")[0]], value[0]))
    if args.phase in ("geometry", "all"):
        for model in manifest["models"]:
            jobs.append((f"{model}/geometry", base / "geometry" / model, None))
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               PYTHONPATH=str(base / "code/src") + os.pathsep + str(base / "code"))
    started = time.monotonic()
    status = dict(phase=args.phase, state="RUNNING", jobs_total=len(jobs), jobs_complete=0)
    model_suffix = "_" + "_".join(args.models) if args.models else ""
    state_path = base / f"status_{args.phase}{model_suffix}.json"

    def save():
        status["elapsed_seconds"] = time.monotonic() - started
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(state_path)

    try:
        for job_id, folder, job in jobs:
            status["active_job"] = job_id
            save()
            complete = folder / "process_status.json"
            if complete.exists():
                previous = json.loads(complete.read_text())
                if previous["status"] != "COMPLETE":
                    raise RuntimeError(f"Prior job failed; preserve and inspect: {folder}")
                # A complete process must also have its expected result artifact.
                required = folder / ("metrics.json" if job is None else "results/trials.jsonl")
                if not required.exists():
                    raise RuntimeError(f"Completed job lacks results: {folder}")
                print(f"SKIP COMPLETE {job_id}", flush=True)
            else:
                folder.mkdir(parents=True, exist_ok=True)
                if job is None:
                    command = [sys.executable, str(base / "code/scripts/capture_cot_update_geometry.py"),
                               "--bundle", str(base), "--model", job_id.split("/")[0],
                               "--cache-dir", str(args.cache_dir), "--output", str(folder)]
                else:
                    job_path = folder / "command.json"
                    if job_path.exists():
                        raise RuntimeError(f"Unfinished job requires inspection: {folder}")
                    job_path.write_text(json.dumps(job, indent=2) + "\n")
                    command = [sys.executable, str(base / "code/scripts/run_cot_backend_replay.py"), "--job", str(job_path)]
                print(f"START {job_id}", flush=True)
                tick = time.monotonic()
                with (folder / "run.log").open("x") as log:
                    process = subprocess.run(command, cwd=base / "code", env=env, stdout=log, stderr=subprocess.STDOUT)
                elapsed = time.monotonic() - tick
                (folder / "launcher_status.json").write_text(json.dumps(dict(returncode=process.returncode, seconds=elapsed,
                                                                             command=command), indent=2) + "\n")
                if process.returncode:
                    raise RuntimeError(f"Job failed ({process.returncode}): {folder / 'run.log'}")
                print(f"COMPLETE {job_id} seconds={elapsed:.1f}", flush=True)
            status["jobs_complete"] += 1
            save()
        status["state"] = "COMPLETE"
        status.pop("active_job", None)
    except BaseException as exc:
        status.update(state="FAILED", error=str(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
