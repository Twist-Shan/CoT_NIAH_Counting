#!/usr/bin/env python3
"""Quarantine a zero-shard V6 confirmation source-write dispatch failure."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "realistic_niah_v6_empty_confirmation_source_recovery_v1"
FAILURE_MARKER = "ValueError: --limit must be positive"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def quarantine_empty_confirmation_source_writes(
    *,
    model_root: Path,
    source: Path,
    failure_log: Path,
    label: str,
) -> dict[str, Any]:
    """Move a proven zero-result failed dispatch aside without deleting it."""

    model_root = model_root.resolve()
    source = source.resolve()
    failure_log = failure_log.resolve()
    if not model_root.is_dir() or not source.is_dir() or not failure_log.is_file():
        raise FileNotFoundError(
            f"Missing model/source/failure log: {model_root}, {source}, {failure_log}"
        )
    if source == model_root or not _within(source, model_root):
        raise ValueError("Source-write recovery target is outside the exact model root")
    if not _within(failure_log, model_root):
        raise ValueError("Failure log is outside the exact model root")
    if not label or any(
        value not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for value in label
    ):
        raise ValueError("Recovery label must contain only lowercase ASCII, digits, _ or -")

    legacy_path = source / "manifest.json"
    adapter_path = source / "v6_adapter_manifest.json"
    legacy = _read_object(legacy_path)
    adapter = _read_object(adapter_path)
    if int(legacy.get("completed_shards", -1)) != 0:
        raise ValueError("Recovery is forbidden when a completed source-write shard exists")
    shards = sorted((source / "shards").glob("*.jsonl")) if (source / "shards").is_dir() else []
    if shards:
        raise ValueError("Recovery is forbidden when a source-write shard exists")
    if adapter.get("command") != "causal-source-writes":
        raise ValueError("Adapter is not a causal-source-writes dispatch")
    if adapter.get("phase") != "confirmation":
        raise ValueError("Adapter is not a confirmation dispatch")
    if adapter.get("run_status") != "DISPATCHED" or adapter.get("status") != "INSTALLED":
        raise ValueError("Adapter is not an interrupted installed dispatch")
    seed_audit = adapter.get("causal_seed_membership_adapter", {})
    if not isinstance(seed_audit, dict) or seed_audit.get("seed_aliasing") is True:
        raise ValueError("Adapter seed-membership audit is absent or aliases seeds")
    if FAILURE_MARKER not in failure_log.read_text(encoding="utf-8", errors="replace"):
        raise ValueError("Failure log does not contain the registered empty-task marker")

    files = [path for path in source.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    quarantine_root = model_root / "quarantine"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    destination = quarantine_root / label
    if destination.exists():
        raise FileExistsError(f"Recovery destination already exists: {destination}")

    source_before = str(source)
    legacy_sha = _sha256(legacy_path)
    adapter_sha = _sha256(adapter_path)
    log_sha = _sha256(failure_log)
    source.rename(destination)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_ZERO_SHARD_CONFIRMATION_ROLE_DISPATCH_QUARANTINED",
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "failure_class": "structural_legacy_seed_role_vocabulary",
        "reason": (
            "the inherited causal-source-writes command filtered the frozen "
            "confirmation registry through its hard-coded development-seed field"
        ),
        "repair": (
            "route effective confirmation true-source seeds through the legacy "
            "development field for this command only"
        ),
        "source_before": source_before,
        "recoverable_destination": str(destination.resolve()),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "preexisting_completed_shards": 0,
        "scientific_artifacts_reused": False,
        "completed_model_trials_recomputed": False,
        "sample_failure": False,
        "seed_replacement_triggered": False,
        "intervention_outcomes_read": False,
        "seed_aliasing": False,
        "deletion_performed": False,
        "legacy_manifest_sha256": legacy_sha,
        "retry_adapter_manifest_sha256": adapter_sha,
        "failure_log": str(failure_log),
        "failure_log_sha256": log_sha,
        "recovery_action": "atomic_same_filesystem_directory_rename_then_clean_dispatch",
    }
    audit_path = quarantine_root / f"{label}.recovery.json"
    temporary = audit_path.with_name(f".{audit_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(audit_path)
    payload["audit_path"] = str(audit_path.resolve())
    payload["audit_sha256"] = _sha256(audit_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--failure-log", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    result = quarantine_empty_confirmation_source_writes(
        model_root=args.model_root,
        source=args.source,
        failure_log=args.failure_log,
        label=args.label,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
