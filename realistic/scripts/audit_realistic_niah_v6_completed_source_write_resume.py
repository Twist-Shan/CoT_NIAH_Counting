#!/usr/bin/env python3
"""Fail-closed audit before reusing a completed V6 source-write shard bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "realistic_niah_v6_completed_source_write_resume_audit_v1"


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_completed_source_write_resume(
    *,
    source: Path,
    model_label: str,
    prompt_mode: str,
    anchor_role: str,
    expected_phase: str = "discovery",
) -> dict[str, Any]:
    source = source.resolve()
    manifest_path = source / "manifest.json"
    adapter_path = source / "v6_adapter_manifest.json"
    manifest = _read(manifest_path)
    adapter = _read(adapter_path)
    expected_manifest = {
        "command": "causal-source-writes",
        "model_label": model_label,
        "anchor_role": anchor_role,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Completed source-write {key} changed: {manifest_path}")
    expected_adapter = {
        "status": "INSTALLED",
        "run_status": "COMPLETE",
        "command": "causal-source-writes",
        "model_label": model_label,
        "prompt_mode": prompt_mode,
        "phase": expected_phase,
    }
    for key, expected in expected_adapter.items():
        if adapter.get(key) != expected:
            raise ValueError(f"Completed V6 adapter {key} changed: {adapter_path}")
    completed = int(manifest.get("completed_shards", -1))
    expected_tasks = int(manifest.get("eligible_anchor_tasks", -2))
    if completed <= 0 or completed != expected_tasks:
        raise ValueError(
            f"Source-write shard quota is incomplete: {completed}/{expected_tasks}"
        )
    shards = sorted((source / "shards").glob("*.jsonl"))
    if len(shards) != completed or any(path.stat().st_size <= 0 for path in shards):
        raise ValueError(
            f"Source-write files do not match completed_shards: {len(shards)}/{completed}"
        )
    if str(manifest.get("generations")) != str(
        adapter.get("materialized_generation_view")
    ):
        raise ValueError("Source-write and V6 adapter generation views disagree")
    cohort = Path(str(adapter.get("cohort_registry", "")))
    if not cohort.is_file() or _sha256(cohort) != adapter.get("cohort_registry_sha256"):
        raise ValueError("Completed source-write cohort registry hash changed")
    seed_audit = adapter.get("causal_seed_membership_adapter", {})
    if not isinstance(seed_audit, dict) or seed_audit.get("seed_aliasing") is not False:
        raise ValueError("Completed source-write adapter does not prove no seed aliasing")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_COMPLETED_SOURCE_WRITES_REUSED_WITHOUT_RECOMPUTATION",
        "model_label": model_label,
        "prompt_mode": prompt_mode,
        "anchor_role": anchor_role,
        "expected_phase": expected_phase,
        "reason": "validated_completed_source_write_resume_after_downstream_failure",
        "sample_failure": False,
        "scientific_artifacts_reused": True,
        "model_outputs_recomputed": False,
        "deletion_performed": False,
        "completed_shards": completed,
        "eligible_anchor_tasks": expected_tasks,
        "source_root": str(source),
        "manifest_sha256": _sha256(manifest_path),
        "v6_adapter_manifest_sha256": _sha256(adapter_path),
        "cohort_registry_sha256": adapter["cohort_registry_sha256"],
        "generation_view": str(manifest["generations"]),
        "recovery_action": (
            "reuse hash- and quota-validated completed source writes; resume only "
            "the downstream panel/plan/behavior/analysis stages"
        ),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument(
        "--prompt-mode", choices=("enumeration_index", "enumeration_bullet"), required=True
    )
    parser.add_argument("--anchor-role", required=True)
    parser.add_argument(
        "--expected-phase",
        choices=("discovery", "confirmation", "diagnostic"),
        default="discovery",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_completed_source_write_resume(
        source=args.source,
        model_label=args.model_label,
        prompt_mode=args.prompt_mode,
        anchor_role=args.anchor_role,
        expected_phase=args.expected_phase,
    )
    _atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
