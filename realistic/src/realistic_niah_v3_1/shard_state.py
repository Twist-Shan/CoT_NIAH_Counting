from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .spec import EXPECTED_REQUESTS, EXPECTED_SHARDS, EXPECTED_STIMULI, PROTOCOL_VERSION


BUNDLE_HEADER = (
    "bundle_id",
    "model",
    "logical_shards",
    "requests",
    "worker_id",
    "attempt_id",
    "completed_at_utc",
)
TASK_HEADER = (
    "task_id",
    "model",
    "prompt_mode",
    "worker_id",
    "attempt_id",
    "completed_at_utc",
)
PLAN_HEADER = (
    "bundle_id",
    "priority",
    "model_label",
    "expected_logical_shards",
    "expected_requests",
    "model_revision",
    "prompt_modes",
    "logical_task_ids",
)


def _read_single_row_tsv(path: Path, expected_header: tuple[str, ...]) -> dict[str, str]:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing completion marker: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != expected_header:
            raise RuntimeError(f"Invalid completion marker header: {path}")
        rows = list(reader)
    if len(rows) != 1 or any(value is None or value == "" for value in rows[0].values()):
        raise RuntimeError(f"Completion marker must contain exactly one full row: {path}")
    return rows[0]


def _load_bundle_plan(run_root: Path) -> list[dict[str, str]]:
    path = run_root / "orchestration" / "formal_bundles.tsv"
    if not path.is_file():
        raise RuntimeError(f"Missing formal bundle plan: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != PLAN_HEADER:
            raise RuntimeError(f"Invalid formal bundle plan header: {path}")
        rows = list(reader)
    if len(rows) != 14 or len({row["bundle_id"] for row in rows}) != 14:
        raise RuntimeError("Formal bundle plan must contain 14 unique bundles")
    if sum(int(row["expected_logical_shards"]) for row in rows) != EXPECTED_SHARDS:
        raise RuntimeError("Formal bundle plan does not cover 48 logical shards")
    if sum(int(row["expected_requests"]) for row in rows) != EXPECTED_REQUESTS:
        raise RuntimeError("Formal bundle plan request total is not 161,280")
    return rows


def _validate_manifest(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"Missing shard manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("protocol_version") != PROTOCOL_VERSION
        or int(manifest.get("completed_requests", -1)) != EXPECTED_STIMULI
        or int(manifest.get("expected_requests", -1)) != EXPECTED_STIMULI
        or manifest.get("prompt_payload_storage") != "sha256_only"
    ):
        raise RuntimeError(f"Invalid or incomplete shard manifest: {path}")


def validate_bundle_completion(run_root: str | Path, bundle: dict[str, str]) -> dict[str, Any]:
    root = Path(run_root).resolve()
    state_root = root / "orchestration" / "shard_state"
    bundle_id = bundle["bundle_id"]
    model = bundle["model_label"]
    task_ids = tuple(value for value in bundle["logical_task_ids"].split(",") if value)
    prompt_modes = tuple(value for value in bundle["prompt_modes"].split(",") if value)
    expected_logical_shards = int(bundle["expected_logical_shards"])
    expected_requests = int(bundle["expected_requests"])
    if len(task_ids) != expected_logical_shards or len(prompt_modes) != expected_logical_shards:
        raise RuntimeError(f"Bundle plan cardinality mismatch: {bundle_id}")

    bundle_marker = _read_single_row_tsv(
        state_root / "completed_bundles" / f"{bundle_id}.tsv",
        BUNDLE_HEADER,
    )
    expected_bundle_values = {
        "bundle_id": bundle_id,
        "model": model,
        "logical_shards": str(expected_logical_shards),
        "requests": str(expected_requests),
    }
    for key, expected in expected_bundle_values.items():
        if bundle_marker[key] != expected:
            raise RuntimeError(f"Bundle marker mismatch for {bundle_id}: {key}")

    for task_id, prompt_mode in zip(task_ids, prompt_modes, strict=True):
        marker = _read_single_row_tsv(
            state_root / "completed" / f"{task_id}.tsv",
            TASK_HEADER,
        )
        if (
            marker["task_id"] != task_id
            or marker["model"] != model
            or marker["prompt_mode"] != prompt_mode
        ):
            raise RuntimeError(f"Task completion marker mismatch: {task_id}")
        _validate_manifest(root / "shards" / task_id / "main" / "run_manifest.json")

    return {
        "bundle_id": bundle_id,
        "model": model,
        "logical_shards": expected_logical_shards,
        "requests": expected_requests,
    }


def audit_shard_state(run_root: str | Path, bundle_id: str | None = None) -> dict[str, Any]:
    root = Path(run_root).resolve()
    state_root = root / "orchestration" / "shard_state"
    plan = _load_bundle_plan(root)
    selected = [row for row in plan if bundle_id is None or row["bundle_id"] == bundle_id]
    if bundle_id is not None and not selected:
        raise RuntimeError(f"Unknown formal bundle: {bundle_id}")
    results = [validate_bundle_completion(root, bundle) for bundle in selected]
    if bundle_id is not None:
        return {"passed": True, "bundles": results}

    failed = sorted((state_root / "failed_bundles").glob("*.tsv"))
    if failed:
        raise RuntimeError(f"Failed bundle markers remain: {[path.name for path in failed]}")
    expected_tasks = {
        task_id
        for bundle in plan
        for task_id in bundle["logical_task_ids"].split(",")
        if task_id
    }
    expected_bundles = {bundle["bundle_id"] for bundle in plan}
    observed_tasks = {path.stem for path in (state_root / "completed").glob("*.tsv")}
    observed_bundles = {
        path.stem for path in (state_root / "completed_bundles").glob("*.tsv")
    }
    if observed_tasks != expected_tasks:
        raise RuntimeError(
            f"Task completion marker set mismatch: missing={sorted(expected_tasks - observed_tasks)}, "
            f"extra={sorted(observed_tasks - expected_tasks)}"
        )
    if observed_bundles != expected_bundles:
        raise RuntimeError(
            "Bundle completion marker set mismatch: "
            f"missing={sorted(expected_bundles - observed_bundles)}, "
            f"extra={sorted(observed_bundles - expected_bundles)}"
        )
    if len(expected_tasks) != EXPECTED_SHARDS:
        raise RuntimeError("Expected task set does not contain 48 logical shards")
    return {
        "passed": True,
        "physical_bundles": len(results),
        "logical_shards": len(expected_tasks),
        "requests": sum(result["requests"] for result in results),
    }
