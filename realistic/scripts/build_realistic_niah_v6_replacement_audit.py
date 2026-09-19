#!/usr/bin/env python3
"""Build one auditable discovery replacement report across modes and models."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


MODES = ("enumeration_index", "enumeration_bullet")
MODELS = ("Qwen3-8B", "Gemma4-E4B")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _cell_summary(root: Path, mode: str, model: str) -> dict[str, Any]:
    directory = root / mode / model / "replacement" / "discovery"
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS_STRICT_QUOTA_FILLED":
        raise ValueError(f"Replacement cell is not PASS: {directory}")
    selected_path = directory / "selected_cells.jsonl"
    mapping_path = directory / "replacement_mapping.jsonl"
    attempts_path = directory / "attempt_ledger.jsonl"
    selected = _read_jsonl(selected_path)
    mappings = _read_jsonl(mapping_path)
    attempts = _read_jsonl(attempts_path)
    quota = int(manifest["quota_per_count"])
    counts = tuple(map(int, manifest["counts"]))
    observed = Counter(int(row["gold_count"]) for row in selected)
    if observed != Counter({count: quota for count in counts}):
        raise ValueError(f"Resolved quota mismatch under {directory}: {observed}")
    if len(mappings) != int(manifest["replacement_count"]):
        raise ValueError(f"Replacement mapping count mismatch under {directory}")
    original_failures = [
        row
        for row in attempts
        if row["candidate_kind"] == "original" and not bool(row["eligible"])
    ]
    reserve_failures = [
        row
        for row in attempts
        if row["candidate_kind"] == "replacement" and not bool(row["eligible"])
    ]
    if len(original_failures) != len(mappings):
        raise ValueError(
            f"Every failed original must have one replacement under {directory}"
        )
    return {
        "model_label": model,
        "prompt_mode": mode,
        "status": manifest["status"],
        "quota_per_count": quota,
        "resolved_rows": len(selected),
        "original_failure_count": len(original_failures),
        "failed_reserve_attempt_count": len(reserve_failures),
        "replacement_count": len(mappings),
        "replacements_by_count": dict(
            sorted(Counter(int(row["gold_count"]) for row in mappings).items())
        ),
        "original_failures": original_failures,
        "failed_reserve_attempts": reserve_failures,
        "replacement_mapping": mappings,
        "artifacts": {
            path.name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in (manifest_path, selected_path, mapping_path, attempts_path)
        },
        "all_sample_panel_replaced": False,
        "panel_membership_identity": "analysis_slot_seed",
        "statistical_identity": "true_source_seed",
        "seed_aliasing": False,
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# V6 discovery failure and replacement audit",
        "",
        f"Status: **{payload['status']}**  ",
        f"Generated UTC: `{payload['generated_utc']}`",
        "",
        "This is a user-authorized protocol amendment made after the original "
        "generations existed. The reserve order was frozen before any reserve "
        "model output. Replacement used only runtime presence and a fresh V6 "
        "strict parse; no hidden state, attention, head ranking, intervention, "
        "or causal result entered selection.",
        "",
        "The original 20-seed panel remains unchanged for all-sample analyses. "
        "Only the strict formal cohort is filled to 20 rows per count. A "
        "replacement keeps its true source seed; `analysis_slot_seed` records "
        "only the failed original cell it fills and is never used to claim a "
        "within-seed trajectory.",
        "",
        "| Model | Prompt mode | Failed originals | Failed reserve attempts | Selected replacements | Counts affected |",
        "|---|---|---:|---:|---:|---|",
    ]
    for cell in payload["cells"]:
        affected = ", ".join(
            f"N{count}:{number}"
            for count, number in cell["replacements_by_count"].items()
        ) or "none"
        lines.append(
            f"| {cell['model_label']} | `{cell['prompt_mode']}` | "
            f"{cell['original_failure_count']} | "
            f"{cell['failed_reserve_attempt_count']} | "
            f"{cell['replacement_count']} | {affected} |"
        )
    lines.extend(
        [
            "",
            "## Exact source-to-slot mappings",
            "",
            "| Model | Prompt mode | Count | Failed slot seed | Failure reason | True replacement seed | Reserve rank |",
            "|---|---|---:|---:|---|---:|---:|",
        ]
    )
    for cell in payload["cells"]:
        for row in cell["replacement_mapping"]:
            reason = ", ".join(map(str, row["original_failure_reasons"]))
            lines.append(
                f"| {cell['model_label']} | `{cell['prompt_mode']}` | "
                f"{row['gold_count']} | {row['analysis_slot_seed']} | {reason} | "
                f"{row['replacement_seed']} | {row['replacement_candidate_rank']} |"
            )
    lines.extend(
        [
            "",
            "Every path and SHA256 is retained in `replacement_audit.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-base", type=Path, required=True)
    parser.add_argument("--replacement-policy", type=Path, required=True)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy = json.loads(args.replacement_policy.read_text(encoding="utf-8"))
    if policy.get("status") != "AMENDMENT_FROZEN_BEFORE_REPLACEMENT_MODEL_OUTPUTS":
        raise ValueError("Replacement amendment status is not frozen")
    pool = json.loads(args.pool_manifest.read_text(encoding="utf-8"))
    if pool.get("status") != "PASS_AMENDMENT_RESERVE_POOL":
        raise ValueError("Replacement pool is not the frozen amendment pool")
    cells = [
        _cell_summary(args.run_base, mode, model)
        for mode in MODES
        for model in MODELS
    ]
    payload = {
        "schema_version": "realistic_niah_v6_replacement_audit_v1",
        "status": "PASS",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_change_class": "prospective_user_authorized_amendment",
        "reserve_order_frozen_before_reserve_model_outputs": True,
        "original_outputs_existed_before_amendment": True,
        "all_sample_panel_replaced": False,
        "formal_quota_per_count": 20,
        "selection_inputs": [
            "generation_presence_or_runtime_status",
            "fresh_v6_parse.strict_causal_eligible",
            "ascending_frozen_amendment_seed_order",
        ],
        "forbidden_selection_inputs_read": False,
        "seed_aliasing": False,
        "total_original_failures": sum(
            int(cell["original_failure_count"]) for cell in cells
        ),
        "total_failed_reserve_attempts": sum(
            int(cell["failed_reserve_attempt_count"]) for cell in cells
        ),
        "total_selected_replacements": sum(
            int(cell["replacement_count"]) for cell in cells
        ),
        "replacement_policy": {
            "path": str(args.replacement_policy.resolve()),
            "sha256": _sha256(args.replacement_policy),
        },
        "replacement_pool_manifest": {
            "path": str(args.pool_manifest.resolve()),
            "sha256": _sha256(args.pool_manifest),
            "stimuli_sha256": pool["stimuli_sha256"],
        },
        "cells": cells,
    }
    json_path = args.output / "replacement_audit.json"
    markdown_path = args.output / "replacement_audit.md"
    _atomic_text(json_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _atomic_text(markdown_path, _markdown(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "total_original_failures": payload["total_original_failures"],
                "total_selected_replacements": payload[
                    "total_selected_replacements"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
