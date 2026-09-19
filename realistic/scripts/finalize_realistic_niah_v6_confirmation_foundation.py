#!/usr/bin/env python3
"""Finalize a completed V6 confirmation foundation without recomputation.

This recovery entry point is intentionally audit-only.  It validates the
already-written registries, captures, attention outputs, and immutable
confirmation freeze before atomically writing the completion audit and PASS
markers that a launcher could not write after an infrastructure-only failure.
It never invokes generation, capture, attention, or intervention code.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from realistic_niah_v6.suite import validate_confirmation_freeze  # noqa: E402


DISCOVERY_SEEDS = tuple(range(1234, 1254))
CONFIRMATION_SEEDS = tuple(range(1254, 1264))
GOLD_COUNTS = tuple(range(1, 11))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def require_files(paths: Mapping[str, Path]) -> None:
    absent = [f"{name}={path}" for name, path in paths.items() if not path.is_file()]
    if absent:
        raise FileNotFoundError("Missing recovery inputs: " + ", ".join(absent))


def validate_registry(
    rows: Iterable[Mapping[str, Any]],
    *,
    seeds: tuple[int, ...],
    name: str,
) -> list[dict[str, Any]]:
    materialized = [dict(row) for row in rows]
    expected = {(count, seed) for count in GOLD_COUNTS for seed in seeds}
    observed = [
        (int(row["gold_count"]), int(row["analysis_slot_seed"]))
        for row in materialized
    ]
    if len(materialized) != len(expected):
        raise ValueError(
            f"{name} registry has {len(materialized)} rows; expected {len(expected)}"
        )
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise ValueError(f"{name} registry slot identity changed")
    return materialized


def validate_capture_index(
    path: Path,
    *,
    model_label: str,
    prompt_mode: str,
    name: str,
) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != 300:
        raise ValueError(f"{name} capture index has {len(rows)} rows; expected 300")
    request_ids = [str(row.get("request_id", "")) for row in rows]
    if any(not value for value in request_ids) or len(set(request_ids)) != 300:
        raise ValueError(f"{name} capture request identity changed")
    split_counts = Counter(str(row.get("split", "")) for row in rows)
    if split_counts != Counter({"discovery": 200, "confirmation": 100}):
        raise ValueError(f"{name} capture split counts changed: {dict(split_counts)}")
    if any(str(row.get("model_label")) != model_label for row in rows):
        raise ValueError(f"{name} capture model label changed")
    expected_fragment = f"/{prompt_mode}/v6/"
    if any(expected_fragment not in request_id for request_id in request_ids):
        raise ValueError(f"{name} capture prompt mode changed")
    return rows


def validate_adapter(
    value: Mapping[str, Any],
    *,
    model_label: str,
    prompt_mode: str,
    formal: bool,
    generations_sha256: str,
    name: str,
) -> None:
    if value.get("run_status") != "COMPLETE":
        raise ValueError(f"{name} adapter is not COMPLETE")
    if value.get("model_label") != model_label:
        raise ValueError(f"{name} adapter model mismatch")
    if value.get("prompt_mode") != prompt_mode:
        raise ValueError(f"{name} adapter prompt mismatch")
    if value.get("formal_cohort") is not formal:
        raise ValueError(f"{name} adapter formal-cohort flag changed")
    if value.get("generations_sha256") != generations_sha256:
        raise ValueError(f"{name} adapter generation hash mismatch")


def validate_attention_manifest(
    value: Mapping[str, Any],
    *,
    model_label: str,
    prompt_mode: str,
    generations_sha256: str,
    confirmation_registry_sha256: str,
    name: str,
) -> None:
    if int(value.get("requests", -1)) != 100:
        raise ValueError(f"{name} request count changed")
    if value.get("model_label") != model_label:
        raise ValueError(f"{name} model mismatch")
    if value.get("prompt_mode") != prompt_mode:
        raise ValueError(f"{name} prompt mismatch")
    if value.get("seed_role") != "confirmation":
        raise ValueError(f"{name} seed role changed")
    if value.get("formal_cohort") is not True:
        raise ValueError(f"{name} formal-cohort flag changed")
    if value.get("generations_sha256") != generations_sha256:
        raise ValueError(f"{name} generation hash mismatch")
    if value.get("cohort_registry_sha256") != confirmation_registry_sha256:
        raise ValueError(f"{name} cohort-registry hash mismatch")


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--prompt-mode", required=True)
    parser.add_argument("--confirmation-freeze", type=Path, required=True)
    parser.add_argument("--source-log", type=Path, required=True)
    parser.add_argument("--supervisor-script", type=Path, required=True)
    parser.add_argument("--recovery-audit", type=Path, required=True)
    parser.add_argument(
        "--failure-text", default="error reading input file: Stale file handle"
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Atomically write audits and PASS markers after all checks pass.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.model_root.resolve()
    freeze = args.confirmation_freeze.resolve()
    source_log = args.source_log.resolve()
    supervisor_script = args.supervisor_script.resolve()

    required = {
        "generations": root / "generation/generations.jsonl",
        "generation_manifest": root / "generation/manifest_confirmation.json",
        "discovery_registry": root / "replacement/discovery/selected_cells.jsonl",
        "cell_registry": root / "replacement/confirmation/selected_cells.jsonl",
        "cell_mapping": root / "replacement/confirmation/replacement_mapping.jsonl",
        "cell_manifest": root / "replacement/confirmation/manifest.json",
        "broad_registry": root / "replacement/confirmation_broad/selected_cells.jsonl",
        "broad_mapping": root / "replacement/confirmation_broad/coherent_mapping.jsonl",
        "broad_manifest": root / "replacement/confirmation_broad/manifest.json",
        "formal_capture_index": root / "capture/confirmation_formal/capture_index.jsonl",
        "formal_capture": root / "capture/confirmation_formal/v6_adapter_manifest.json",
        "all_capture_index": root
        / "capture/confirmation_all_sample/capture_index.jsonl",
        "all_capture": root
        / "capture/confirmation_all_sample/v6_adapter_manifest.json",
        "confirmation_attention": root / "attention/confirmation_formal.manifest.json",
        "confirmation_answer_query": root
        / "attention/confirmation_answer_query_formal.manifest.json",
        "confirmation_freeze": freeze,
        "source_log": source_log,
        "supervisor_script": supervisor_script,
    }
    require_files(required)

    source_text = source_log.read_text(encoding="utf-8", errors="replace")
    pass_text = "PASS confirmation_answer_query_formal_resolved"
    pass_offset = source_text.rfind(pass_text)
    failure_offset = source_text.rfind(args.failure_text)
    if pass_offset < 0 or failure_offset <= pass_offset:
        raise ValueError(
            "Source log does not prove a final substage PASS followed by the "
            "registered infrastructure-only failure"
        )

    discovery_rows = validate_registry(
        read_jsonl(required["discovery_registry"]),
        seeds=DISCOVERY_SEEDS,
        name="discovery",
    )
    cell_rows = validate_registry(
        read_jsonl(required["cell_registry"]),
        seeds=CONFIRMATION_SEEDS,
        name="cell confirmation",
    )
    broad_rows = validate_registry(
        read_jsonl(required["broad_registry"]),
        seeds=CONFIRMATION_SEEDS,
        name="broad confirmation",
    )
    formal_capture_rows = validate_capture_index(
        required["formal_capture_index"],
        model_label=args.model_label,
        prompt_mode=args.prompt_mode,
        name="formal",
    )
    all_capture_rows = validate_capture_index(
        required["all_capture_index"],
        model_label=args.model_label,
        prompt_mode=args.prompt_mode,
        name="all-sample",
    )

    generations_sha256 = sha256_file(required["generations"])
    confirmation_registry_sha256 = sha256_file(required["cell_registry"])
    discovery_registry_sha256 = sha256_file(required["discovery_registry"])
    formal_adapter = read_json(required["formal_capture"])
    all_adapter = read_json(required["all_capture"])
    attention = read_json(required["confirmation_attention"])
    answer_query = read_json(required["confirmation_answer_query"])
    validate_adapter(
        formal_adapter,
        model_label=args.model_label,
        prompt_mode=args.prompt_mode,
        formal=True,
        generations_sha256=generations_sha256,
        name="formal",
    )
    validate_adapter(
        all_adapter,
        model_label=args.model_label,
        prompt_mode=args.prompt_mode,
        formal=False,
        generations_sha256=generations_sha256,
        name="all-sample",
    )
    formal_registry_hashes = {
        str(value.get("sha256", ""))
        for value in formal_adapter.get("cohort_registries", [])
        if isinstance(value, Mapping)
    }
    expected_registry_hashes = {
        discovery_registry_sha256,
        confirmation_registry_sha256,
    }
    if formal_registry_hashes != expected_registry_hashes:
        raise ValueError("Formal capture registry lineage changed")
    validate_attention_manifest(
        attention,
        model_label=args.model_label,
        prompt_mode=args.prompt_mode,
        generations_sha256=generations_sha256,
        confirmation_registry_sha256=confirmation_registry_sha256,
        name="confirmation attention",
    )
    validate_attention_manifest(
        answer_query,
        model_label=args.model_label,
        prompt_mode=args.prompt_mode,
        generations_sha256=generations_sha256,
        confirmation_registry_sha256=confirmation_registry_sha256,
        name="confirmation answer-query attention",
    )
    if answer_query.get("site_id") != "answer_query_v3":
        raise ValueError("Answer-query endpoint changed")

    for name in ("cell_manifest", "broad_manifest"):
        status = str(read_json(required[name]).get("status", ""))
        if not status.startswith("PASS"):
            raise ValueError(f"{name} is not PASS: {status!r}")

    validated_freeze = validate_confirmation_freeze(
        freeze,
        prompt_mode=args.prompt_mode,
        model_label=args.model_label,
        verify_artifacts=True,
    )
    freeze_validation = {
        "schema_version": "realistic_niah_v6_confirmation_freeze_validation_v1",
        "status": "PASS",
        "prompt_mode": args.prompt_mode,
        "model_label": args.model_label,
        "confirmation_freeze": str(freeze),
        "confirmation_freeze_file_sha256": sha256_file(freeze),
        "content_freeze_sha256": validated_freeze["freeze_sha256"],
        "confirmation_outcomes_read": False,
    }

    completion_artifacts = {
        name: {
            "path": str(required[name].resolve()),
            "sha256": sha256_file(required[name]),
        }
        for name in (
            "generation_manifest",
            "cell_registry",
            "cell_mapping",
            "cell_manifest",
            "broad_registry",
            "broad_mapping",
            "broad_manifest",
            "formal_capture",
            "all_capture",
            "confirmation_attention",
            "confirmation_answer_query",
        )
    }
    completion = {
        "schema_version": (
            "realistic_niah_v6_confirmation_foundation_complete_v2_native_aligned"
        ),
        "status": "CONFIRMATION_FOUNDATION_COMPLETE",
        "model_label": args.model_label,
        "prompt_mode": args.prompt_mode,
        "confirmation_seed_count": 10,
        "cell_count": 100,
        "panel_membership_identity": "analysis_slot_seed",
        "statistical_identity": "true_source_seed",
        "seed_aliasing": False,
        "representation_analysis": {
            "status": "DEFERRED_UNTIL_ALL_FOUR_ORIGINAL_CAPTURES_EXIST",
            "population": "original_registered_all_sample_panel",
            "running_endpoint": "item_end exact four-cell common support",
            "final_endpoint": "answer_query_v3 exact full 300-trajectory panel",
            "legacy_generic_confirmation_scan_required": False,
        },
        "freeze": str(freeze),
        "freeze_sha256": sha256_file(freeze),
        "artifacts": completion_artifacts,
    }

    output_paths = {
        "phase_marker": root / "confirmation-foundation-resolved.COMPLETE",
        "completion_audit": root / "freeze/confirmation_foundation_complete.json",
        "freeze_validation": root
        / "freeze/validation_after_confirmation_foundation.json",
        "foundation_marker": root / "freeze/confirmation-foundation.COMPLETE",
        "recovery_audit": args.recovery_audit.resolve(),
    }
    ready = {
        "schema_version": (
            "realistic_niah_v6_confirmation_foundation_finalize_only_recovery_v1"
        ),
        "status": "PASS_READY_FINALIZE_ONLY_RECOVERY",
        "commit_requested": bool(args.commit),
        "model_label": args.model_label,
        "prompt_mode": args.prompt_mode,
        "checks": {
            "discovery_registry_rows": len(discovery_rows),
            "cell_confirmation_registry_rows": len(cell_rows),
            "broad_confirmation_registry_rows": len(broad_rows),
            "formal_capture_rows": len(formal_capture_rows),
            "all_sample_capture_rows": len(all_capture_rows),
            "attention_requests": int(attention["requests"]),
            "answer_query_requests": int(answer_query["requests"]),
            "freeze_valid": True,
            "final_substage_pass_precedes_infrastructure_failure": True,
            "model_computation_invoked": False,
            "seed_replacement_triggered": False,
            "intervention_outcomes_read_for_recovery": False,
        },
        "source_failure": {
            "classification": "infrastructure_nfs_stale_script_handle",
            "text": args.failure_text,
            "source_log": artifact_record(source_log),
            "supervisor_script": artifact_record(supervisor_script),
        },
        "outputs": {name: str(path) for name, path in output_paths.items()},
    }
    if not args.commit:
        print(json.dumps(ready, indent=2, ensure_ascii=False, sort_keys=True))
        return

    atomic_write_json(output_paths["completion_audit"], completion)
    atomic_write_json(output_paths["freeze_validation"], freeze_validation)
    atomic_write_text(output_paths["phase_marker"], "PASS\n")
    atomic_write_text(output_paths["foundation_marker"], "PASS\n")

    recovered_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    recovery = dict(ready)
    recovery["status"] = "PASS_FINALIZE_ONLY_RECOVERY_COMMITTED"
    recovery["recovered_utc"] = recovered_utc
    recovery["output_artifact_sha256"] = {
        name: sha256_file(path)
        for name, path in output_paths.items()
        if name != "recovery_audit"
    }
    atomic_write_json(output_paths["recovery_audit"], recovery)
    print(json.dumps(recovery, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
