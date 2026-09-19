"""Complete ten Enumeration Update seeds per cell under Native-style sampling."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def mode_seeds(ledger, candidates, quota):
    """Select within one mode, without looking at another mode or correctness."""
    n10 = [r for r in ledger if r["gold_count"] == 10]
    lookup = {r["seed"]: r for r in n10}
    if len(lookup) != len(n10) or len(candidates) != len(set(candidates)):
        raise ValueError("Duplicate baseline seed")
    if set(candidates) - lookup.keys():
        raise ValueError("Missing registered candidates")
    if quota <= 0:
        raise ValueError("Quota must be positive")
    return [seed for seed in sorted(candidates) if lookup[seed]["update_eligible"]][:quota]


def freeze(a):
    tick = time.monotonic()
    baseline, causal = a.root / "fresh_v1", a.root / "fresh_causal_v1"
    base = read(baseline / "manifest.json")
    old = read(causal / "cohorts.json")
    cfg = read(a.protocol)
    assert old["baseline_manifest_sha256"] == sha(baseline / "manifest.json")
    a.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(a.protocol, a.output / "protocol.json")
    shutil.copy2(__file__, a.output / "launch_enumeration_native_update.py")
    shutil.copy2(a.test_file, a.output / "test_enumeration_native_update.py")
    cells, jobs = [], []
    quota = cfg["update"]["confirmation_quota_per_model_mode"]
    for model in cfg["models"]:
        for mode in cfg["modes"]:
            registry = causal / "registries" / model / mode
            manifest = read(registry / "manifest.json")
            assert sha(registry / "ledger.json") == manifest["ledger_sha256"]
            assert sha(registry / "manifest.json") == old["registries"][f"{model}/{mode}"]["manifest_sha256"]
            selected = mode_seeds(read(registry / "ledger.json"), base["confirmation_candidates"], quota)
            if len(selected) != quota:
                raise ValueError(f"{model}/{mode} needs a frozen reserve scan before continuing")
            old_seeds = set(old["models"][model]["confirmation_seeds"])
            reused = [seed for seed in selected if seed in old_seeds]
            missing = [seed for seed in selected if seed not in old_seeds]
            cohort = deepcopy(old)
            cohort.update(status="FROZEN_BEFORE_PRIMARY_COMPLETION_JOBS", selection="per-mode baseline eligibility",
                          primary_selected_seeds=selected, reused_seeds=reused, missing_seeds=missing,
                          chronology=cfg["chronology"], selection_used_final_correctness=False, intervention_outcomes_accessed=False)
            cohort["models"][model].update(confirmation_seeds=missing, actual_confirmation_n=len(missing))
            cohort_path = a.output / "cohorts" / model / f"{mode}.json"
            write(cohort_path, cohort)
            cells.append({"model": model, "mode": mode, "selected_seeds": selected, "reused_seeds": reused,
                          "missing_seeds": missing, "cohort": str(cohort_path), "cohort_sha256": sha(cohort_path),
                          "registry": str(registry), "registry_sha256": sha(registry / "manifest.json")})
    for phase in ("smoke", "formal"):
        for cell in cells:
            output = a.output / "jobs" / phase / cell["model"] / cell["mode"]
            command = [sys.executable, str(causal / "code/scripts/run_enumeration_fresh_update.py"),
                "--bundle", str(baseline), "--registry", cell["registry"], "--cohorts", cell["cohort"],
                "--cache-dir", str(a.cache_dir), "--output", str(output)]
            if phase == "smoke":
                command.append("--smoke")
            jobs.append({"id": f"{phase}/{cell['model']}/{cell['mode']}", "phase": phase,
                "command": command, "output": str(output),
                "expected_rows": 54 * (min(1, len(cell["missing_seeds"])) if phase == "smoke" else len(cell["missing_seeds"]))})
    write(a.output / "job_manifest.json", {"jobs": jobs, "cells": cells, "all_smokes_precede_formal": True})
    manifest = {"status": "FROZEN_BEFORE_PRIMARY_COMPLETION_JOBS", "causal_stage": str(causal),
        "baseline_bundle": str(baseline), "baseline_manifest_sha256": sha(baseline / "manifest.json"),
        "causal_code_manifest_sha256": sha(causal / "code_manifest.json"), "old_cohorts_sha256": sha(causal / "cohorts.json"),
        "protocol_sha256": sha(a.output / "protocol.json"), "job_manifest_sha256": sha(a.output / "job_manifest.json"),
        "entrypoint_sha256": sha(a.output / "launch_enumeration_native_update.py"),
        "test_sha256": sha(a.output / "test_enumeration_native_update.py"), "cells": cells,
        "primary_expected_rows": 4 * quota * 54, "reused_expected_rows": sum(len(c["reused_seeds"]) for c in cells) * 54,
        "new_formal_expected_rows": sum(len(c["missing_seeds"]) for c in cells) * 54,
        "command": sys.argv, "seconds": time.monotonic() - tick, "created_utc": datetime.now(timezone.utc).isoformat()}
    write(a.output / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("status", "cells", "primary_expected_rows", "reused_expected_rows", "new_formal_expected_rows")}), flush=True)


def verify(stage):
    m = read(stage / "manifest.json")
    for name, key in (("protocol.json", "protocol_sha256"), ("job_manifest.json", "job_manifest_sha256"),
                      ("launch_enumeration_native_update.py", "entrypoint_sha256"), ("test_enumeration_native_update.py", "test_sha256")):
        assert sha(stage / name) == m[key], name
    causal = Path(m["causal_stage"])
    assert sha(causal / "code_manifest.json") == m["causal_code_manifest_sha256"]
    code = read(causal / "code_manifest.json")
    for name, expected in {**code["original_code_sha256"], **code["additive_code_sha256"]}.items():
        assert sha(causal / "code" / name) == expected, name
    for cell in m["cells"]:
        assert sha(cell["cohort"]) == cell["cohort_sha256"]
        assert sha(Path(cell["registry"]) / "manifest.json") == cell["registry_sha256"]
    return m


def assemble(stage, manifest):
    """Retain exact selected rows, full generations and both prefill audits."""
    causal = Path(manifest["causal_stage"])
    audit = {"cells": {}, "source_files_sha256": {}, "primary_rows": 0}
    for cell in manifest["cells"]:
        model, mode = cell["model"], cell["mode"]
        runtimes = []
        for seeds, root in ((cell["reused_seeds"], causal / "gpu_jobs/formal/update"),
                            (cell["missing_seeds"], stage / "jobs/formal")):
            if seeds:
                runtime_path = root / model / mode / "runtime.json"
                runtime = read(runtime_path)
                assert runtime["model"] == model and runtime["mode"] == mode
                assert runtime["registry_sha256"] == cell["registry_sha256"]
                assert runtime["runner_sha256"] == sha(causal / "code/scripts/run_enumeration_fresh_update.py")
                runtimes.append(runtime)
                audit["source_files_sha256"][str(runtime_path)] = sha(runtime_path)
        for other in runtimes[1:]:
            for key in ("model_revision", "model_source_sha256", "dtype", "backend", "torch", "transformers",
                        "layers", "patch_layer_zero_based", "jobs", "attention_readout", "all_three_conditions_generate"):
                assert other[key] == runtimes[0][key], f"Runtime mismatch: {model}/{mode}/{key}"
        cell_rows = 0
        for k in (4, 6, 8):
            for direction in ("forward", "backward"):
                for scope in ("endpoint", "four_token_tail", "item_span"):
                    geometry = f"k{k}_{direction}_{scope}"
                    merged = {name: [] for name in ("trials.jsonl", "raw_generations.jsonl", "prefill_hooks.jsonl")}
                    for seeds, root in ((cell["reused_seeds"], causal / "gpu_jobs/formal/update"),
                                        (cell["missing_seeds"], stage / "jobs/formal")):
                        if not seeds:
                            continue
                        source = root / model / mode / geometry
                        check = read(source / "technical_audit.json")
                        assert check["status"] == "PASS"
                        for name, key in (("trials.jsonl", "trials_sha256"), ("raw_generations.jsonl", "raw_sha256"), ("prefill_hooks.jsonl", "hooks_sha256")):
                            path = source / name
                            assert sha(path) == check[key]
                            audit["source_files_sha256"][str(path)] = check[key]
                            with path.open(encoding="utf-8") as f:
                                chosen = [r for r in map(json.loads, f) if r["seed"] in seeds]
                            expected = (6 if name == "prefill_hooks.jsonl" else 3) * len(seeds)
                            assert len(chosen) == expected
                            merged[name].extend(chosen)
                    identities = {(r["seed"], r["condition"]) for r in merged["trials.jsonl"]}
                    expected = {(seed, condition) for seed in cell["selected_seeds"]
                                for condition in ("receiver_self", "native_donor", "donor_to_receiver")}
                    assert identities == expected and len(merged["trials.jsonl"]) == len(expected)
                    folder = stage / "primary" / model / mode / geometry
                    folder.mkdir(parents=True, exist_ok=False)
                    for name, rows in merged.items():
                        with (folder / name).open("x", encoding="utf-8") as f:
                            for row in rows:
                                f.write(json.dumps(row, ensure_ascii=True) + "\n")
                    cell_rows += len(expected)
        assert cell_rows == len(cell["selected_seeds"]) * 54
        audit["cells"][f"{model}/{mode}"] = {"rows": cell_rows, "seeds": cell["selected_seeds"]}
        audit["primary_rows"] += cell_rows
    assert audit["primary_rows"] == manifest["primary_expected_rows"]
    audit["status"] = "PASS"
    write(stage / "assembly_audit.json", audit)


def run(stage):
    manifest = verify(stage)
    tick = time.monotonic()
    path = stage / "gpu_pipeline_status.json"
    with path.open("x", encoding="utf-8") as f:
        json.dump({"status": "STARTING", "pid": os.getpid()}, f)
    state = {"status": "RUNNING", "phase": "CPU_TESTS", "pid": os.getpid(), "jobs": [], "command": sys.argv}
    def save():
        state["seconds"] = time.monotonic() - tick
        write(path, state)
    save()
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    try:
        with (stage / "cpu_validation.log").open("x", encoding="utf-8") as log:
            test = subprocess.run([sys.executable, "-m", "pytest", "test_enumeration_native_update.py", "-q", "-p", "no:cacheprovider"],
                                  cwd=stage, stdout=log, stderr=subprocess.STDOUT)
        assert test.returncode == 0, "Sampling CPU tests failed"
        state.update(cpu_tests="PASS", phase="WAIT_SHARED_COHORT_READ_UPDATE_COMPLETE")
        save()
        while True:
            previous = read(Path(manifest["causal_stage"]) / "gpu_pipeline_status.json")
            if previous["status"] == "FAILED":
                raise RuntimeError("Earlier Read/Update queue failed; resolve prerequisite first")
            if previous["status"] == "COMPLETE":
                break
            if time.monotonic() - tick > 36 * 3600:
                raise TimeoutError("Earlier queue did not finish in 36 hours")
            time.sleep(20)
            save()
        verify(stage)
        for job in read(stage / "job_manifest.json")["jobs"]:
            record = {**job, "status": "RUNNING"}
            state["jobs"].append(record)
            state["phase"] = job["id"]
            save()
            if job["expected_rows"] == 0:
                record["status"] = "REUSED_NO_NEW_SEEDS"
                save()
                continue
            folder = stage / "logs" / job["id"]
            folder.mkdir(parents=True, exist_ok=False)
            started = time.monotonic()
            with (folder / "process.log").open("x", encoding="utf-8") as log:
                worker = subprocess.Popen(job["command"], env=env, stdout=log, stderr=subprocess.STDOUT)
                record["pid"] = worker.pid
                save()
                record["returncode"] = worker.wait()
            record["seconds"] = time.monotonic() - started
            if record["returncode"]:
                record["status"] = "FAILED"
                raise RuntimeError(f"Primary Update {job['id']} failed")
            result = read(Path(job["output"]) / "status.json")
            assert result["status"] == "COMPLETE" and result["completed_rows"] == job["expected_rows"]
            record.update(status="COMPLETE", rows=result["completed_rows"])
            save()
        state["phase"] = "ASSEMBLE_PRIMARY_SELECTED_ROWS"
        save()
        assemble(stage, manifest)
        state.update(status="COMPLETE", phase="ENUMERATION_UPDATE_PRIMARY_COMPLETE")
    except BaseException as error:
        state.update(status="FAILED", error=repr(error))
        raise
    finally:
        save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    f = sub.add_parser("freeze")
    for name in ("root", "protocol", "test-file", "output", "cache-dir"):
        f.add_argument("--" + name, type=Path, required=True)
    sub.add_parser("run").add_argument("--stage", type=Path, required=True)
    a = p.parse_args()
    freeze(a) if a.action == "freeze" else run(a.stage)


if __name__ == "__main__":
    main()
