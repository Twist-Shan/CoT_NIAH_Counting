#!/usr/bin/env python3
"""Freeze one outcome-blind final-transition anchor per registered seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence


DISCOVERY_SEEDS = tuple(range(1234, 1254))
CONFIRMATION_SEEDS = tuple(range(1254, 1264))
REGISTERED_SEEDS = DISCOVERY_SEEDS + CONFIRMATION_SEEDS


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON {path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError("The source final-transition registry is empty")
    return rows


def select_anchor_subset(
    rows: Iterable[dict[str, Any]],
    *,
    seeds: Sequence[int] = REGISTERED_SEEDS,
) -> list[dict[str, Any]]:
    """Choose highest count, then lexicographically smallest request id per seed."""

    registered = tuple(int(value) for value in seeds)
    if len(set(registered)) != len(registered):
        raise ValueError("Registered write-edge seeds must be unique")
    by_seed: dict[int, list[dict[str, Any]]] = {seed: [] for seed in registered}
    for raw in rows:
        if "selection_rank" in raw:
            raise ValueError("Write-edge anchor selection must not use selection_rank")
        seed = int(raw["seed"])
        if seed not in by_seed:
            continue
        count = int(raw["gold_count"])
        if int(raw["from_occurrence"]) != count - 1:
            raise ValueError("A source anchor is not the final N-1 transition")
        if int(raw["to_occurrence"]) != count:
            raise ValueError("A source anchor does not retrieve terminal item N")
        by_seed[seed].append(dict(raw))
    missing = [seed for seed, values in by_seed.items() if not values]
    if missing:
        raise ValueError(f"No final-transition anchor for seeds {missing}")
    selected = []
    for seed in registered:
        candidates = sorted(
            by_seed[seed],
            key=lambda row: (-int(row["gold_count"]), str(row["request_id"])),
        )
        chosen = candidates[0]
        chosen["write_edge_row_selection_rule"] = (
            "highest_gold_count_then_request_id_per_seed"
        )
        chosen["write_edge_outcome_blind"] = True
        chosen["write_edge_selection_rank_used"] = False
        selected.append(chosen)
    if [int(row["seed"]) for row in selected] != list(registered):
        raise RuntimeError("Write-edge anchor subset changed registered seed order")
    if len({str(row["request_id"]) for row in selected}) != len(selected):
        raise ValueError("Write-edge anchor subset contains duplicate requests")
    return selected


def _canonical_sha256(rows: Sequence[dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = _read_jsonl(args.input)
    selected = select_anchor_subset(source)
    canonical_sha = _canonical_sha256(selected)
    audit = {
        "schema_version": "realistic_niah_v5_write_edge_anchor_subset_v1",
        "status": "PASS",
        "source_registry": str(args.input.resolve()),
        "source_registry_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "selection_rule": "highest_gold_count_then_request_id_per_seed",
        "outcome_blind": True,
        "selection_rank_used": False,
        "registered_seeds": list(REGISTERED_SEEDS),
        "discovery_seeds": list(DISCOVERY_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "row_count": len(selected),
        "selected_count_by_seed": {
            str(row["seed"]): int(row["gold_count"]) for row in selected
        },
        "selected_request_by_seed": {
            str(row["seed"]): str(row["request_id"]) for row in selected
        },
        "canonical_rows_sha256": canonical_sha,
    }
    if args.output.exists():
        existing = _read_jsonl(args.output)
        if _canonical_sha256(existing) != canonical_sha:
            raise ValueError("Existing frozen write-edge anchor subset changed")
    else:
        _atomic_jsonl(args.output, selected)
    audit_path = args.output.with_suffix(".audit.json")
    if audit_path.exists():
        existing_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if existing_audit != audit:
            raise ValueError("Existing write-edge anchor audit changed")
    else:
        _atomic_json(audit_path, audit)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
