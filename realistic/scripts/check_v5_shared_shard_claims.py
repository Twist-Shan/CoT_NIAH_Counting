#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path


def load_runner(repo: Path):
    path = repo / "scripts" / "run_realistic_niah_v5.py"
    specification = importlib.util.spec_from_file_location("v5_runner", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    runner = load_runner(repo)
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)

    shard = root / "task.jsonl"
    shard.unlink(missing_ok=True)
    shard.with_suffix(".jsonl.claim").unlink(missing_ok=True)
    first = runner._claim_restartable_task(
        shard,
        worker_id="worker-a",
        stale_seconds=3600,
    )
    if first is None:
        raise RuntimeError("First worker failed to claim an empty task")
    second = runner._claim_restartable_task(
        shard,
        worker_id="worker-b",
        stale_seconds=3600,
    )
    if second is not None:
        raise RuntimeError("Two workers claimed the same task")
    runner._atomic_write_jsonl(shard, [{"task": "claimed-once"}])
    first.unlink(missing_ok=True)
    completed = runner._claim_restartable_task(
        shard,
        worker_id="worker-b",
        stale_seconds=3600,
    )
    if completed is not None:
        raise RuntimeError("A completed task was claimed again")

    stale_shard = root / "stale.jsonl"
    stale_shard.unlink(missing_ok=True)
    stale_claim = stale_shard.with_suffix(".jsonl.claim")
    stale_claim.write_text("stale\n", encoding="utf-8")
    old = time.time() - 7200
    os.utime(stale_claim, (old, old))
    reclaimed = runner._claim_restartable_task(
        stale_shard,
        worker_id="worker-c",
        stale_seconds=3600,
    )
    if reclaimed is None:
        raise RuntimeError("A stale task claim was not reclaimed")
    reclaimed.unlink(missing_ok=True)

    payload = {
        "schema_version": "v5_shared_shard_claim_audit_v1",
        "single_winner": True,
        "completed_not_reclaimed": True,
        "stale_claim_recovered": True,
        "atomic_row": json.loads(shard.read_text(encoding="utf-8").strip()),
        "passed": True,
    }
    output = root / "audit.json"
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
