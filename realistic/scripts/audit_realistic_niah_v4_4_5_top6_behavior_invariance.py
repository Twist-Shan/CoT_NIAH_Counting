#!/usr/bin/env python3
"""Audit that Top-6 follow-ups leave non-bank behavioral outputs unchanged.

Experiments 22 and 23 use the frozen retrieval bank only for auxiliary
answer-query readouts.  Their causal interventions are fixed elsewhere.  This
audit compares the original Top-8-derived run with the post-hoc Top-6 rerun at
the row level and permits differences only in the three retrieval-bank fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SPECS = (
    (
        "induction_canonical",
        "induction",
        "detail.jsonl",
        "induction_v3/Gemma4-E4B/detail.jsonl",
        ("seed", "gold_count", "arm"),
        {
            "retrieval_bank_broad_score_mean",
            "retrieval_bank_coverage_mean",
            "retrieval_bank_needle_mass_mean",
        },
    ),
    (
        "induction_synthetic",
        "induction",
        "synthetic_rows.jsonl",
        "induction_v3/Gemma4-E4B/synthetic_rows.jsonl",
        ("layer", "head", "condition", "sequence_index"),
        set(),
    ),
    (
        "noise_factorial",
        "noise",
        "factorial_rows.jsonl",
        "noise_factorial_v2/Gemma4-E4B/factorial_rows.jsonl",
        ("seed", "cell"),
        set(),
    ),
    (
        "outside_context",
        "noise",
        "outside_context_rows.jsonl",
        "noise_factorial_v2/Gemma4-E4B/outside_context_rows.jsonl",
        ("seed", "gold_count", "arm"),
        {
            "retrieval_bank_broad_score_mean",
            "retrieval_bank_coverage_mean",
            "retrieval_bank_needle_mass_mean",
        },
    ),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def keyed(
    rows: Iterable[dict[str, Any]], keys: tuple[str, ...], *, label: str
) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        identity = tuple(row[key] for key in keys)
        if identity in result:
            raise RuntimeError(f"{label}: duplicate identity {identity!r}")
        result[identity] = row
    return result


def strip_allowed(
    row: dict[str, Any], allowed_difference_fields: set[str]
) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in allowed_difference_fields
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-induction-dir", type=Path, required=True)
    parser.add_argument("--old-noise-dir", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit_rows: list[dict[str, Any]] = []
    all_pass = True
    old_roots = {
        "induction": args.old_induction_dir,
        "noise": args.old_noise_dir,
    }
    for (
        label,
        old_root_name,
        old_filename,
        new_relative_path,
        identity_keys,
        allowed_fields,
    ) in SPECS:
        old_path = old_roots[old_root_name] / old_filename
        new_path = args.new_root / new_relative_path
        old_rows = keyed(read_jsonl(old_path), identity_keys, label=f"{label}/old")
        new_rows = keyed(read_jsonl(new_path), identity_keys, label=f"{label}/new")
        identities_match = old_rows.keys() == new_rows.keys()
        allowed_fields_present_both = all(
            allowed_fields <= set(row)
            for rows in (old_rows, new_rows)
            for row in rows.values()
        )
        changed_allowed_fields: set[str] = set()
        mismatches: list[dict[str, Any]] = []
        if identities_match:
            for identity in old_rows:
                for field in allowed_fields:
                    if (
                        field in old_rows[identity]
                        and field in new_rows[identity]
                        and old_rows[identity][field] != new_rows[identity][field]
                    ):
                        changed_allowed_fields.add(field)
                old_behavior = strip_allowed(old_rows[identity], allowed_fields)
                new_behavior = strip_allowed(new_rows[identity], allowed_fields)
                if old_behavior != new_behavior:
                    differing_fields = sorted(
                        key
                        for key in old_behavior.keys() | new_behavior.keys()
                        if old_behavior.get(key) != new_behavior.get(key)
                    )
                    mismatches.append(
                        {
                            "identity": list(identity),
                            "differing_fields": differing_fields,
                        }
                    )
                    if len(mismatches) >= 20:
                        break
        expected_bank_change_observed = (
            not allowed_fields or bool(changed_allowed_fields)
        )
        passed = (
            identities_match
            and allowed_fields_present_both
            and expected_bank_change_observed
            and not mismatches
        )
        all_pass &= passed
        audit_rows.append(
            {
                "label": label,
                "status": "PASS" if passed else "FAIL",
                "old_path": str(old_path),
                "new_path": str(new_path),
                "old_sha256": sha256_file(old_path),
                "new_sha256": sha256_file(new_path),
                "identity_keys": list(identity_keys),
                "old_rows": len(old_rows),
                "new_rows": len(new_rows),
                "identities_match": identities_match,
                "allowed_difference_fields": sorted(allowed_fields),
                "allowed_fields_present_both": allowed_fields_present_both,
                "changed_allowed_fields": sorted(changed_allowed_fields),
                "expected_bank_change_observed": expected_bank_change_observed,
                "non_bank_behavior_exactly_equal": passed,
                "first_mismatches": mismatches,
            }
        )

    payload = {
        "schema_version": "realistic_niah_v4_4_5_top6_behavior_invariance_audit_v1",
        "status": "PASS" if all_pass else "FAIL",
        "definition": (
            "All row identities and all fields other than the explicitly listed "
            "retrieval-bank auxiliary readouts must match exactly."
        ),
        "auditor_sha256": sha256_file(Path(__file__)),
        "comparisons": audit_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
