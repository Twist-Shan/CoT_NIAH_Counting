"""Compact read-only snapshot across the baseline, registry, and causal queues."""
from __future__ import annotations
import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def tail(path, n=8):
    if not path.exists():
        return []
    with path.open(errors="replace") as f:
        return list(deque(f, maxlen=n))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    base = args.root
    stage = base / "fresh_causal_v1"
    baseline = read(base / "fresh_pipeline_v1_status.json")
    registry = read(stage / "registry_pipeline_status.json")
    causal = read(stage / "gpu_pipeline_status.json")
    cohorts = read(stage / "cohorts.json")
    result = {"utc": datetime.now(timezone.utc).isoformat(),
              "baseline_status": baseline["status"], "baseline_phase": baseline["phase"],
              "baseline_cells": [], "registry_status": registry["status"],
              "registry_cells": registry.get("cells", []),
              "causal_status": causal["status"], "causal_phase": causal["phase"],
              "causal_error": causal.get("error"), "causal_jobs": []}
    pids = {causal["pid"]}
    for job in baseline.get("jobs", []):
        if job["id"].startswith("baseline/"):
            row = read(base / "fresh_v1" / job["id"] / "status.json") or {}
            result["baseline_cells"].append({"id": job["id"], **{key: row[key] for key in ("status", "completed", "total") if key in row}})
    result["baseline_rows"] = sum(row.get("completed", 0) for row in result["baseline_cells"])
    primary_stage = base / "fresh_native_update_v1"
    primary_state = read(primary_stage / "gpu_pipeline_status.json")
    if primary_state:
        if primary_state["status"] == "RUNNING":
            pids.add(primary_state["pid"])
        primary_jobs = []
        for job in primary_state.get("jobs", []):
            item = {key: job[key] for key in ("id", "status", "pid", "expected_rows", "rows", "seconds", "returncode") if key in job}
            item["worker_status"] = read(Path(job["output"]) / "status.json")
            if job["status"] in ("RUNNING", "FAILED"):
                if job.get("pid"):
                    pids.add(job["pid"])
                item["log_tail"] = tail(primary_stage / "logs" / job["id"] / "process.log")
            primary_jobs.append(item)
        primary_manifest = read(primary_stage / "manifest.json")
        result["primary_update"] = {"path": str(primary_stage), "status": primary_state["status"],
            "phase": primary_state["phase"], "pid": primary_state["pid"], "error": primary_state.get("error"),
            "seconds": primary_state.get("seconds"), "jobs": primary_jobs, "cpu_tests": primary_state.get("cpu_tests"),
            "cells": primary_manifest["cells"], "primary_expected_rows": primary_manifest["primary_expected_rows"],
            "reused_expected_rows": primary_manifest["reused_expected_rows"], "new_formal_expected_rows": primary_manifest["new_formal_expected_rows"],
            "manifest_sha256": hashlib.sha256((primary_stage / "manifest.json").read_bytes()).hexdigest(),
            "log_tail": tail(primary_stage / "gpu_pipeline.log") if primary_state["status"] == "FAILED" else []}
    relay_registry = base / ("fresh_relay_native_registry_v1" if (base / "fresh_relay_native_registry_v1").exists() else "fresh_relay_registry_v3")
    if relay_registry.exists():
        relay_cohorts = read(relay_registry / "common_cohorts.json")
        result["relay_registry"] = {"path": str(relay_registry),
            "state": read(relay_registry / "pipeline_status.json"),
            "coverage": {model: value["coverage"] for model, value in relay_cohorts["models"].items()} if relay_cohorts else None,
            "cohort_policy": relay_cohorts.get("cohort_policy") if relay_cohorts else None,
            "per_mode_coverage": {model: {mode: cell["coverage"] for mode, cell in value.get("per_mode", {}).items()} for model, value in relay_cohorts["models"].items()} if relay_cohorts else None,
            "cohorts_sha256": hashlib.sha256((relay_registry / "common_cohorts.json").read_bytes()).hexdigest() if relay_cohorts else None}
    relay_stage = base / ("fresh_relay_native_gpu_v1" if (base / "fresh_relay_native_gpu_v1").exists() else "fresh_relay_gpu_v1")
    relay_state = read(relay_stage / "gpu_pipeline_status.json")
    if relay_state:
        if relay_state["status"] == "RUNNING":
            pids.add(relay_state["pid"])
        relay_jobs = []
        for job in relay_state.get("jobs", []):
            item = {key: job[key] for key in ("id", "status", "pid", "expected_rows", "rows", "seconds", "returncode") if key in job}
            item["worker_status"] = read(Path(job["output"]) / "status.json")
            if job["status"] in ("RUNNING", "FAILED"):
                if job.get("pid"):
                    pids.add(job["pid"])
                item["log_tail"] = tail(relay_stage / "logs" / job["id"] / "process.log")
            relay_jobs.append(item)
        result["relay_gpu"] = {"path": str(relay_stage), "status": relay_state["status"],
            "phase": relay_state["phase"], "pid": relay_state["pid"], "error": relay_state.get("error"),
            "seconds": relay_state.get("seconds"), "jobs": relay_jobs,
            "cpu_validation": read(relay_stage / "cpu_validation.json"),
            "manifest_sha256": hashlib.sha256((relay_stage / "manifest.json").read_bytes()).hexdigest(),
            "log_tail": tail(relay_stage / "gpu_pipeline.log") if relay_state["status"] == "FAILED" else []}
    answer_stage = base / "fresh_answer_patch_gpu_v1"
    answer_state = read(answer_stage / "gpu_pipeline_status.json")
    if answer_state:
        if answer_state["status"] == "RUNNING":
            pids.add(answer_state["pid"])
        answer_jobs = []
        for job in answer_state.get("jobs", []):
            item = {key: job[key] for key in ("id", "status", "pid", "expected_rows", "rows", "seconds", "returncode") if key in job}
            item["worker_status"] = read(Path(job["output"]) / "status.json")
            if job["status"] in ("RUNNING", "FAILED"):
                if job.get("pid"):
                    pids.add(job["pid"])
                item["log_tail"] = tail(answer_stage / "logs" / job["id"] / "process.log")
            answer_jobs.append(item)
        answer_manifest = read(answer_stage / "manifest.json")
        answer_registry = read(Path(answer_manifest["registry_root"]) / "manifest.json")
        result["answer_patch"] = {"path": str(answer_stage), "status": answer_state["status"],
            "phase": answer_state["phase"], "pid": answer_state["pid"], "error": answer_state.get("error"),
            "seconds": answer_state.get("seconds"), "jobs": answer_jobs,
            "cpu_validation": read(answer_stage / "cpu_validation.json"),
            "expected_smoke_rows": answer_manifest["expected_smoke_rows"],
            "expected_formal_rows": answer_manifest["expected_formal_rows"],
            "registry_root": answer_manifest["registry_root"], "cells": answer_registry["cells"],
            "manifest_sha256": hashlib.sha256((answer_stage / "manifest.json").read_bytes()).hexdigest(),
            "registry_manifest_sha256": answer_manifest["registry_manifest_sha256"],
            "log_tail": tail(answer_stage / "gpu_pipeline.log") if answer_state["status"] == "FAILED" else []}
    retrieve_stage = base / "fresh_retrieve_gpu_v1"
    retrieve_state = read(retrieve_stage / "gpu_pipeline_status.json")
    if retrieve_state:
        if retrieve_state["status"] == "RUNNING":
            pids.add(retrieve_state["pid"])
        retrieve_jobs = []
        for job in retrieve_state.get("jobs", []):
            item = {key: job[key] for key in ("id", "status", "pid", "expected_rows", "rows", "seconds", "returncode") if key in job}
            output = Path(job["output"])
            item["worker_status"] = read(output / "status.json")
            if (output / "status.json").exists():
                item["output_mtime_utc"] = datetime.fromtimestamp((output / "status.json").stat().st_mtime, timezone.utc).isoformat()
            if job["status"] in ("RUNNING", "FAILED"):
                if job.get("pid"):
                    pids.add(job["pid"])
                item["log_tail"] = tail(Path(job["log_path"]))
            retrieve_jobs.append(item)
        retrieve_manifest = read(retrieve_stage / "manifest.json")
        retrieve_registry = read(Path(retrieve_manifest["registry_root"]) / "manifest.json")
        result["retrieve"] = {"path": str(retrieve_stage), "status": retrieve_state["status"],
            "phase": retrieve_state["phase"], "pid": retrieve_state["pid"], "error": retrieve_state.get("error"),
            "seconds": retrieve_state.get("seconds"), "jobs": retrieve_jobs,
            "cpu_validation": read(retrieve_stage / "cpu_validation.json"),
            "expected_localization_queries": retrieve_manifest["expected_localization_queries"],
            "expected_formal_rows": retrieve_manifest["expected_formal_rows"],
            "registry_root": retrieve_manifest["registry_root"], "cells": retrieve_registry["cells"],
            "manifest_sha256": hashlib.sha256((retrieve_stage / "manifest.json").read_bytes()).hexdigest(),
            "registry_manifest_sha256": retrieve_manifest["registry_manifest_sha256"],
            "log_tail": tail(retrieve_stage / "gpu_pipeline.log") if retrieve_state["status"] == "FAILED" else []}
    if cohorts:
        result["cohorts"] = cohorts["models"]
        result["cohorts_sha256"] = hashlib.sha256((stage / "cohorts.json").read_bytes()).hexdigest()
    for job in causal.get("jobs", []):
        output = Path(job["output"])
        status = read(output / "status.json") or {}
        item = {key: job[key] for key in ("id", "status", "eligible_inputs", "expected_rows", "pid", "seconds", "rows", "returncode") if key in job}
        item["worker_status"] = {key: status[key] for key in ("status", "completed", "completed_rows", "completed_jobs", "total_rows", "total", "seconds", "error") if key in status}
        if (output / "status.json").exists():
            item["output_mtime_utc"] = datetime.fromtimestamp((output / "status.json").stat().st_mtime, timezone.utc).isoformat()
        if job["status"] in ("RUNNING", "FAILED"):
            if job.get("pid"):
                pids.add(job["pid"])
            item["log_tail"] = tail(stage / "gpu_launch_logs" / job["id"] / "process.log")
        result["causal_jobs"].append(item)
    result["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=15).stdout.strip()
    proc = subprocess.run(["ps", "-p", ",".join(map(str, sorted(pids))), "-o", "pid,etimes,pcpu,rss,args"],
                          capture_output=True, text=True, timeout=10)
    result["processes"] = proc.stdout.strip()
    result["registry_error"] = registry.get("error")
    result["gpu_queue_log_tail"] = tail(stage / "gpu_pipeline.log") if causal["status"] == "FAILED" else []
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
