#!/usr/bin/env python3
"""Recover a missing discovery-foundation marker from complete hashed outputs.

This command never runs a model and never edits an existing model output.  It
is intentionally narrow: all five registered foundation jobs and both capture
adapter manifests must already be complete and mutually consistent.  Only then
does it write an infrastructure-recovery audit and the missing PASS marker.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "realistic_niah_v6_foundation_marker_recovery_v1"


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def recover_foundation_marker(
    *,
    model_root: Path,
    model_label: str,
    prompt_mode: str,
    reason: str,
) -> dict[str, Any]:
    root = model_root.resolve()
    marker = root / "discovery-foundation.COMPLETE"
    audit_path = root / "foundation_marker_recovery_audit.json"
    if marker.is_file() or audit_path.is_file():
        if (
            marker.is_file()
            and marker.read_text(encoding="utf-8").strip() == "PASS"
            and audit_path.is_file()
        ):
            existing = _read(audit_path)
            if existing.get("status") == "PASS_OUTPUTS_COMPLETE_MARKER_RECOVERED":
                return existing
        raise ValueError("Foundation marker recovery target is not a clean missing-marker case")

    paths = {
        "formal_capture": root / "capture/formal/capture_manifest.json",
        "formal_capture_adapter": root / "capture/formal/v6_adapter_manifest.json",
        "all_capture": root / "capture/all_sample/capture_manifest.json",
        "all_capture_adapter": root / "capture/all_sample/v6_adapter_manifest.json",
        "formal_attention": root / "attention/discovery_formal.manifest.json",
        "all_attention": root / "attention/discovery_all_sample.manifest.json",
        "formal_answer_query": (
            root / "attention/discovery_answer_query_formal.manifest.json"
        ),
    }
    values = {name: _read(path) for name, path in paths.items()}
    formal_capture = values["formal_capture"]
    all_capture = values["all_capture"]
    formal_attention = values["formal_attention"]
    all_attention = values["all_attention"]
    answer_query = values["formal_answer_query"]
    formal_count = int(formal_capture.get("rows", -1))
    all_count = int(all_capture.get("rows", -1))
    if formal_count <= 0 or all_count != 200 or formal_count > all_count:
        raise ValueError("Foundation capture request counts are incomplete")
    if (
        int(formal_attention.get("requests", -1)) != formal_count
        or int(answer_query.get("requests", -1)) != formal_count
        or int(all_attention.get("requests", -1)) != all_count
    ):
        raise ValueError("Foundation capture and attention request counts disagree")
    for name in ("formal_attention", "all_attention", "formal_answer_query"):
        value = values[name]
        if (
            value.get("model_label") != model_label
            or value.get("prompt_mode") != prompt_mode
            or value.get("seed_role") != "discovery"
            or int(value.get("rows", 0)) <= 0
        ):
            raise ValueError(f"Foundation attention manifest is incomplete: {name}")
    if formal_attention.get("formal_cohort") is not True:
        raise ValueError("Formal attention manifest lost strict-cohort status")
    if answer_query.get("formal_cohort") is not True:
        raise ValueError("Answer-query attention manifest lost strict-cohort status")
    if all_attention.get("formal_cohort") is not False:
        raise ValueError("All-sample attention manifest changed cohort status")
    for name in ("formal_capture", "all_capture"):
        value = values[name]
        if (
            value.get("schema_version") != "realistic_niah_v6_trace_capture_v1"
            or value.get("parser_implementation")
            != "realistic_niah_v6.parse_structured_enumeration_trace"
            or int(value.get("excluded_rows", -1)) != 0
        ):
            raise ValueError(f"Foundation capture manifest is incomplete: {name}")
    for name in ("formal_capture_adapter", "all_capture_adapter"):
        value = values[name]
        if (
            value.get("status") != "INSTALLED"
            or value.get("prompt_mode") != prompt_mode
            or value.get("v5_source_files_modified") is not False
        ):
            raise ValueError(f"Foundation capture adapter is incomplete: {name}")

    for relative in (
        "capture/formal/capture_index.jsonl",
        "capture/all_sample/capture_index.jsonl",
        "attention/discovery_formal.csv",
        "attention/discovery_all_sample.csv",
        "attention/discovery_answer_query_formal.csv",
    ):
        path = root / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_OUTPUTS_COMPLETE_MARKER_RECOVERED",
        "phase": "discovery-foundation",
        "model_label": model_label,
        "prompt_mode": prompt_mode,
        "reason": reason,
        "recovered_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "validated_request_counts": {
            "formal": formal_count,
            "all_sample": all_count,
        },
        "validated_files": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "model_outputs_recomputed": False,
        "intervention_outcomes_read": False,
        "seed_failure_recorded": False,
        "sample_failure": False,
        "seed_replacement_triggered": False,
        "deletion_performed": False,
    }
    _atomic_text(
        audit_path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(marker, "PASS\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt-mode",
        choices=("enumeration_index", "enumeration_bullet"),
        required=True,
    )
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    payload = recover_foundation_marker(
        model_root=args.model_root,
        model_label=args.model,
        prompt_mode=args.prompt_mode,
        reason=args.reason,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
