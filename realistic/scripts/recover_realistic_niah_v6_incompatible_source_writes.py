#!/usr/bin/env python3
"""Quarantine, never delete, a V6 source-write directory rejected on resume."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "realistic_niah_v6_incompatible_source_write_recovery_v1"


def _read(path: Path) -> dict[str, Any]:
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


def quarantine_incompatible_source_writes(
    *, model_root: Path, source: Path, label: str
) -> dict[str, Any]:
    model_root = model_root.resolve()
    source = source.resolve()
    if not model_root.is_dir() or not source.is_dir():
        raise FileNotFoundError(f"Missing model/source directory: {model_root}, {source}")
    if source == model_root or not _within(source, model_root):
        raise ValueError("Source-write quarantine target is outside the exact model root")
    if not label or any(value not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for value in label):
        raise ValueError("Quarantine label must contain only lowercase ASCII, digits, _ or -")

    legacy_path = source / "manifest.json"
    adapter_path = source / "v6_adapter_manifest.json"
    legacy = _read(legacy_path)
    adapter = _read(adapter_path)
    existing_generations = str(legacy.get("generations", ""))
    requested_generations = str(adapter.get("materialized_generation_view", ""))
    if not existing_generations or not requested_generations:
        raise ValueError("Both legacy and V6 manifests must identify their generation view")
    if existing_generations == requested_generations:
        raise ValueError("Source-write manifests are compatible; quarantine is not allowed")
    if adapter.get("run_status") != "DISPATCHED" or adapter.get("status") != "INSTALLED":
        raise ValueError("V6 retry adapter manifest is not a dispatched installed adapter")
    if adapter.get("seed_aliasing") is True:
        raise ValueError("V6 retry adapter reports seed aliasing")

    files = [path for path in source.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    quarantine_root = model_root / "quarantine"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    destination = quarantine_root / label
    if destination.exists():
        raise FileExistsError(f"Quarantine destination already exists: {destination}")
    source_before = str(source)
    legacy_sha = _sha256(legacy_path)
    adapter_sha = _sha256(adapter_path)
    source.rename(destination)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_PRE_RESOLVED_INCOMPATIBLE_SHARDS_QUARANTINED",
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sample_failure": False,
        "scientific_artifacts_reused": False,
        "intervention_outcomes_read": False,
        "reason": (
            "resume guard rejected a pre-resolved source-write manifest after the "
            "formal replacement registry changed the materialized generation view"
        ),
        "source_before": source_before,
        "recoverable_destination": str(destination.resolve()),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "legacy_manifest_sha256": legacy_sha,
        "retry_adapter_manifest_sha256": adapter_sha,
        "existing_materialized_generation_view": existing_generations,
        "requested_materialized_generation_view": requested_generations,
        "existing_config_sha256": legacy.get("config_sha256"),
        "requested_v6_config_sha256": adapter.get("v6_config_sha256"),
        "cohort_registry_sha256": adapter.get("cohort_registry_sha256"),
        "recovery_action": "atomic_same_filesystem_directory_rename_then_clean_rerun",
        "deletion_performed": False,
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
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    result = quarantine_incompatible_source_writes(
        model_root=args.model_root, source=args.source, label=args.label
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
