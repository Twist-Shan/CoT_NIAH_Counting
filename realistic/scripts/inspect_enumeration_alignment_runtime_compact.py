"""Preserve the full v6 runtime snapshot and emit a bounded monitoring view."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def compact_jobs(jobs):
    fields = ("status", "completed", "completed_rows", "completed_jobs", "total", "total_rows",
              "total_jobs", "expected_rows", "seconds", "error", "smoke", "completed_traces",
              "expected_traces", "completed_states", "completed_layers", "expected_layers", "first_trace_repeat_exact")
    for job in jobs:
        worker = job.get("worker_status") or {}
        job["worker_status"] = {key: worker[key] for key in fields if key in worker}
        job.pop("command", None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    inspector = args.root / "inspect_enumeration_alignment_runtime_v6.py"
    raw = subprocess.check_output([sys.executable, str(inspector), "--root", str(args.root)])
    result = json.loads(raw)
    setup_path = args.root / "n10_setup_status.json"
    if setup_path.exists():
        setup = json.loads(setup_path.read_text())
        if setup["status"] == "FAILED":
            log = args.root / "n10_setup.log"
            setup["log_tail"] = log.read_text(errors="replace").splitlines()[-16:] if log.exists() else []
        result["n10_setup"] = setup
        raw = (json.dumps(result, ensure_ascii=True, indent=2) + "\n").encode()
    diagnostic = args.root / "fresh_qwen_index_layer_diagnostic_v1"
    diagnostic_status = diagnostic / "gpu_pipeline_status.json"
    if diagnostic_status.exists():
        state = json.loads(diagnostic_status.read_text())
        manifest = json.loads((diagnostic / "manifest.json").read_text())
        state.update(path=str(diagnostic), manifest_sha256=hashlib.sha256((diagnostic / "manifest.json").read_bytes()).hexdigest(),
                     prerequisite=manifest["prerequisite"], expected_new_smoke_rows=36,
                     expected_new_formal_rows=360, reused_L30_rows=60)
        for job in state.get("jobs", []):
            path = diagnostic / "jobs" / job["id"] / "raw_generations.jsonl"
            if path.exists():
                with path.open() as handle:
                    job["live_generated_rows"] = sum(1 for _ in handle)
        if state.get("pid") and state["status"] == "RUNNING":
            process = subprocess.run(["ps", "-ww", "-p", str(state["pid"]), "-o", "pid=,stat=,etimes=,pcpu=,rss=,args="],
                                     capture_output=True, text=True)
            state["process"] = process.stdout.strip()
            state["process_present"] = process.returncode == 0 and bool(process.stdout.strip())
        if state["status"] == "FAILED":
            log = diagnostic / "gpu_pipeline.log"
            state["log_tail"] = log.read_text(errors="replace").splitlines()[-16:] if log.exists() else []
        result["qwen_layer_diagnostic"] = state
        raw = (json.dumps(result, ensure_ascii=True, indent=2) + "\n").encode()
    n10 = args.root / "fresh_n10_update_v1"
    if (n10 / "gpu_pipeline_status.json").exists():
        state = json.loads((n10 / "gpu_pipeline_status.json").read_text())
        state.update(path=str(n10), manifest_sha256=hashlib.sha256((n10 / "manifest.json").read_bytes()).hexdigest(),
                     expected_reserve_rows=120, expected_discovery_states=800,
                     expected_confirmation_states=400, expected_smoke_rows=216, expected_formal_rows=2160)
        for job in state.get("jobs", []):
            status_path = Path(job["status_path"])
            if status_path.exists():
                job["worker_status"] = json.loads(status_path.read_text())
            if job.get("status") == "FAILED":
                log = n10 / "logs" / job["id"] / "process.log"
                job["log_tail"] = log.read_text(errors="replace").splitlines()[-16:] if log.exists() else []
        if state.get("pid") and state["status"] == "RUNNING":
            process = subprocess.run(["ps", "-ww", "-p", str(state["pid"]), "-o", "pid=,stat=,etimes=,pcpu=,rss=,args="],
                                     capture_output=True, text=True)
            state["process"] = process.stdout.strip()
            state["process_present"] = process.returncode == 0 and bool(process.stdout.strip())
        if state["status"] == "FAILED":
            log = n10 / "gpu_pipeline.log"
            state["log_tail"] = log.read_text(errors="replace").splitlines()[-16:] if log.exists() else []
        result["n10_update"] = state
        raw = (json.dumps(result, ensure_ascii=True, indent=2) + "\n").encode()
    folder = args.root / "monitor_snapshots"
    folder.mkdir(exist_ok=True)
    path = folder / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%fZ}_full.json"
    with path.open("xb") as handle:
        handle.write(raw)
    for name in ("primary_update", "relay_gpu", "answer_patch", "retrieve", "n10_update"):
        stage = result.get(name) or {}
        stage.pop("cells", None)
        compact_jobs(stage.get("jobs", []))
    compact_jobs(result.get("causal_jobs", []))
    result["full_snapshot"] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
