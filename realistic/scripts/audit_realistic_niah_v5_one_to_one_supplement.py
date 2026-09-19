#!/usr/bin/env python3
"""Audit isolated V5 N=10 strict one-to-one supplementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "realistic_niah_v5_one_to_one_supplement_audit_v1"
EXPECTED_OCCURRENCES = list(range(1, 11))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _item_end_occurrences(index_path: Path, row: Mapping[str, Any]) -> list[int]:
    manifest_path = index_path.parent / str(row["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return sorted(
        int(site["occurrence"])
        for site in manifest.get("site_rows", [])
        if str(site.get("site_kind")) == "item_end"
        and site.get("occurrence") is not None
    )


def _strict_rows(index_path: Path, *, model: str) -> list[dict[str, Any]]:
    accepted = []
    for row in _read_jsonl(index_path):
        if str(row.get("model_label")) != model or int(row.get("gold_count", -1)) != 10:
            continue
        occurrences = _item_end_occurrences(index_path, row)
        if bool(row.get("trace_one_to_one")) and occurrences == EXPECTED_OCCURRENCES:
            accepted.append(
                {
                    "seed": int(row["seed"]),
                    "split": str(row["split"]),
                    "request_id": str(row["request_id"]),
                    "capture_index": str(index_path.resolve()),
                    "manifest_path": str(row["manifest_path"]),
                    "states_path": str(row["states_path"]),
                    "exact_count": bool(row.get("exact_count")),
                    "item_end_occurrences": occurrences,
                }
            )
    return accepted


def _batch_attempts(batch: Path, *, model: str) -> list[dict[str, Any]]:
    manifest_path = batch / "dataset" / "candidate_manifest.json"
    parsed_path = batch / "parsed.jsonl"
    capture_index = batch / "capture" / "capture_index.jsonl"
    exclusion_path = batch / "capture" / "capture_exclusions.jsonl"
    completion_path = batch / "batch.complete.json"
    required = (manifest_path, parsed_path, capture_index, completion_path)
    if not all(path.exists() for path in required):
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_by_seed = {
        **{int(seed): "discovery" for seed in manifest["discovery_seeds"]},
        **{int(seed): "confirmation" for seed in manifest["confirmation_seeds"]},
    }
    parsed = {int(row["seed"]): row for row in _read_jsonl(parsed_path)}
    captures = {int(row["seed"]): row for row in _read_jsonl(capture_index)}
    exclusions = {int(row["seed"]): row for row in _read_jsonl(exclusion_path)}
    if set(parsed) != set(split_by_seed):
        raise ValueError(f"Parsed seed mismatch in {batch}: {sorted(parsed)}")
    if set(captures) & set(exclusions):
        raise ValueError(f"Captured/excluded overlap in {batch}")
    if set(captures) | set(exclusions) != set(split_by_seed):
        raise ValueError(f"Capture accounting mismatch in {batch}")

    rows = []
    for seed, split in sorted(split_by_seed.items()):
        parsed_row = parsed[seed]
        parser = dict(parsed_row.get("trace_parse", {}).get("parser", {}))
        capture_row = captures.get(seed)
        exclusion = exclusions.get(seed)
        occurrences = (
            _item_end_occurrences(capture_index, capture_row) if capture_row else []
        )
        parser_one_to_one = bool(parser.get("trace_one_to_one"))
        capture_one_to_one = bool(
            capture_row is not None and capture_row.get("trace_one_to_one")
        )
        accepted = (
            parser_one_to_one
            and capture_one_to_one
            and occurrences == EXPECTED_OCCURRENCES
        )
        reason = "accepted_strict_one_to_one" if accepted else str(
            (exclusion or {}).get("reason_code")
            or parser.get("trace_category")
            or "not_strict_one_to_one"
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "model_label": model,
                "seed": seed,
                "split": split,
                "batch": batch.name,
                "batch_complete_sha256": _sha256(completion_path),
                "parsed_count": int(parsed_row.get("trace_parse", {}).get("parsed_count", -1)),
                "exact_count": bool(parsed_row.get("trace_parse", {}).get("exact_count")),
                "trace_category": parser.get("trace_category"),
                "trace_one_to_one": parser_one_to_one,
                "coverage_count": int(parser.get("coverage_count", 0)),
                "coverage_fraction": float(parser.get("coverage_fraction", 0.0)),
                "duplicate_gold_city_items": int(parser.get("duplicate_gold_city_items", 0)),
                "capture_present": capture_row is not None,
                "capture_trace_one_to_one": capture_one_to_one,
                "item_end_occurrences": occurrences,
                "accepted": accepted,
                "reason": reason,
                "request_id": parsed_row.get("request_id"),
                "capture_index": str(capture_index.resolve()),
                "states_path": (capture_row or {}).get("states_path"),
                "manifest_path": (capture_row or {}).get("manifest_path"),
                "exclusion": exclusion,
                "merge_into_primary_300": False,
            }
        )
    return rows


def audit(
    *,
    model: str,
    primary_capture_index: Path,
    existing_supplement_indexes: tuple[Path, ...],
    supplement_root: Path,
    target_discovery: int,
    target_confirmation: int,
) -> dict[str, Any]:
    primary = _strict_rows(primary_capture_index, model=model)
    existing = []
    for index in existing_supplement_indexes:
        if index.exists():
            existing.extend(_strict_rows(index, model=model))
    fixed = primary + existing
    fixed_seeds = [row["seed"] for row in fixed]
    if len(fixed_seeds) != len(set(fixed_seeds)):
        raise ValueError("Duplicate strict seeds across primary/existing supplement")

    attempts = []
    for batch in sorted((supplement_root / "batches").glob("seed_*")):
        if batch.is_dir():
            attempts.extend(_batch_attempts(batch, model=model))
    attempt_seeds = [row["seed"] for row in attempts]
    if len(attempt_seeds) != len(set(attempt_seeds)):
        raise ValueError("Duplicate candidate seed across supplement batches")
    overlap = sorted(set(fixed_seeds) & set(attempt_seeds))
    if overlap:
        raise ValueError(f"Candidate seeds overlap existing data: {overlap}")

    ledger_path = supplement_root / "attempt_ledger.jsonl"
    existing_ledger = _read_jsonl(ledger_path)
    if len(existing_ledger) > len(attempts):
        raise ValueError("Attempt ledger is longer than completed batch inventory")
    for index, row in enumerate(existing_ledger):
        if row != attempts[index]:
            raise ValueError(f"Attempt ledger mutation detected at row {index + 1}")
    if len(existing_ledger) < len(attempts):
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
            for row in attempts[len(existing_ledger) :]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    accepted_attempts = [row for row in attempts if row["accepted"]]
    counts = Counter((row["split"] for row in fixed))
    counts.update(row["split"] for row in accepted_attempts)
    accepted_path = supplement_root / "accepted_supplement.jsonl"
    _atomic_jsonl(accepted_path, accepted_attempts)
    reason_counts = Counter(row["reason"] for row in attempts)
    audit_value = {
        "schema_version": SCHEMA_VERSION,
        "model_label": model,
        "primary_capture_index": str(primary_capture_index.resolve()),
        "existing_supplement_capture_indexes": [
            str(path.resolve()) for path in existing_supplement_indexes if path.exists()
        ],
        "merge_into_primary_300": False,
        "target": {
            "discovery": target_discovery,
            "confirmation": target_confirmation,
            "total": target_discovery + target_confirmation,
        },
        "primary_strict": dict(Counter(row["split"] for row in primary)),
        "existing_supplement_strict": dict(
            Counter(row["split"] for row in existing)
        ),
        "new_attempts": len(attempts),
        "new_strict": dict(Counter(row["split"] for row in accepted_attempts)),
        "strict_total": {
            "discovery": counts["discovery"],
            "confirmation": counts["confirmation"],
            "total": counts["discovery"] + counts["confirmation"],
        },
        "complete": (
            counts["discovery"] >= target_discovery
            and counts["confirmation"] >= target_confirmation
        ),
        "reason_counts": dict(sorted(reason_counts.items())),
        "candidate_seed_unique": len(attempt_seeds) == len(set(attempt_seeds)),
        "fixed_candidate_seed_disjoint": not overlap,
        "accepted_item_end_occurrences_exact_1_10": all(
            row["item_end_occurrences"] == EXPECTED_OCCURRENCES
            for row in accepted_attempts
        ),
        "attempt_ledger": str(ledger_path.resolve()),
        "attempt_ledger_sha256": _sha256(ledger_path) if ledger_path.exists() else None,
        "accepted_supplement": str(accepted_path.resolve()),
        "accepted_supplement_sha256": _sha256(accepted_path),
    }
    _atomic_json(supplement_root / "supplement_audit.json", audit_value)
    return audit_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--primary-capture-index", type=Path, required=True)
    parser.add_argument(
        "--existing-supplement-capture-index",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--supplement-root", type=Path, required=True)
    parser.add_argument("--target-discovery", type=int, default=20)
    parser.add_argument("--target-confirmation", type=int, default=10)
    args = parser.parse_args()
    value = audit(
        model=args.model,
        primary_capture_index=args.primary_capture_index,
        existing_supplement_indexes=tuple(args.existing_supplement_capture_index),
        supplement_root=args.supplement_root,
        target_discovery=args.target_discovery,
        target_confirmation=args.target_confirmation,
    )
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
