#!/usr/bin/env python3
"""Freeze an outcome-blind one-row-per-seed count-balanced anchor registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence


DISCOVERY_SEEDS = tuple(range(1234, 1254))
CONFIRMATION_SEEDS = tuple(range(1254, 1264))
ALLOWED_SOURCE_FIELDS = {
    "anchor_equivalence_id",
    "anchor_roles",
    "from_occurrence",
    "gold_count",
    "request_id",
    "seed",
    "target_grammar_class",
    "target_retrieval_surface_variant",
    "to_occurrence",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _read_source(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Balanced-count source registry is empty")
    forbidden = sorted(set().union(*(set(row) for row in rows)) - ALLOWED_SOURCE_FIELDS)
    if forbidden:
        raise ValueError(
            "Balanced-count planning accepts anchor metadata only; forbidden fields: "
            f"{forbidden}"
        )
    identities = [(int(row["seed"]), int(row["gold_count"])) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Balanced-count source has duplicate seed/count rows")
    return rows


def _assign_exact_quota(
    rows: Sequence[dict[str, Any]],
    *,
    seeds: Sequence[int],
    counts: Sequence[int],
) -> list[dict[str, Any]]:
    ordered_seeds = tuple(int(value) for value in seeds)
    ordered_counts = tuple(sorted({int(value) for value in counts}))
    if len(ordered_counts) != len(tuple(counts)):
        raise ValueError("Balanced-count panel has duplicate counts")
    if len(ordered_seeds) % len(ordered_counts):
        raise ValueError("Seed count must be divisible by requested count strata")
    by_identity = {
        (int(row["seed"]), int(row["gold_count"])): dict(row) for row in rows
    }
    quota = len(ordered_seeds) // len(ordered_counts)
    remaining = {count: quota for count in ordered_counts}
    assignment: dict[int, int] = {}

    def search() -> bool:
        if len(assignment) == len(ordered_seeds):
            return all(value == 0 for value in remaining.values())
        candidates: list[tuple[int, int, list[int]]] = []
        for seed in ordered_seeds:
            if seed in assignment:
                continue
            available = [
                count
                for count in ordered_counts
                if remaining[count] > 0 and (seed, count) in by_identity
            ]
            candidates.append((len(available), seed, available))
        _width, seed, available = min(candidates)
        if not available:
            return False
        preferred = ordered_counts[ordered_seeds.index(seed) % len(ordered_counts)]
        for count in sorted(
            available,
            key=lambda value: (
                (value - preferred) % (max(ordered_counts) + 1),
                value,
            ),
        ):
            assignment[seed] = count
            remaining[count] -= 1
            if search():
                return True
            remaining[count] += 1
            del assignment[seed]
        return False

    if not search():
        coverage = {
            str(seed): [count for count in ordered_counts if (seed, count) in by_identity]
            for seed in ordered_seeds
        }
        raise ValueError(f"No exact balanced assignment exists: {coverage}")
    return [by_identity[(seed, assignment[seed])] for seed in ordered_seeds]


def build_plan(
    source: Path,
    *,
    counts: Sequence[int],
    panel_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _read_source(source)
    discovery = _assign_exact_quota(
        rows, seeds=DISCOVERY_SEEDS, counts=counts
    )
    confirmation = _assign_exact_quota(
        rows, seeds=CONFIRMATION_SEEDS, counts=counts
    )
    selected: list[dict[str, Any]] = []
    for phase, phase_rows in (
        ("discovery", discovery),
        ("confirmation", confirmation),
    ):
        for row in phase_rows:
            selected.append(
                {
                    **row,
                    "selection_panel_id": str(panel_id),
                    "selection_phase": phase,
                    "selection_rank_used": False,
                    "outcome_blind": True,
                }
            )
    selected_core = [
        {
            "seed": int(row["seed"]),
            "gold_count": int(row["gold_count"]),
            "request_id": str(row["request_id"]),
            "selection_phase": str(row["selection_phase"]),
        }
        for row in selected
    ]
    count_list = sorted({int(value) for value in counts})
    manifest_core = {
        "schema_version": "realistic_niah_v5_balanced_count_anchor_plan_v1",
        "panel_id": str(panel_id),
        "selection_rule": (
            "deterministic_exact_count_quota_matching_using_anchor_metadata_only"
        ),
        "outcome_blind": True,
        "selection_rank_used": False,
        "source_registry_name": source.name,
        "source_registry_sha256": _sha256(source),
        "counts": count_list,
        "discovery": {
            "seeds": list(DISCOVERY_SEEDS),
            "seed_count": len(DISCOVERY_SEEDS),
            "quota_per_count": len(DISCOVERY_SEEDS) // len(count_list),
        },
        "confirmation": {
            "seeds": list(CONFIRMATION_SEEDS),
            "seed_count": len(CONFIRMATION_SEEDS),
            "quota_per_count": len(CONFIRMATION_SEEDS) // len(count_list),
        },
        "selected_rows": selected_core,
        "selected_rows_sha256": _sha256_json(selected_core),
    }
    return selected, {
        **manifest_core,
        "plan_sha256": _sha256_json(manifest_core),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--counts", required=True)
    parser.add_argument("--panel-id", required=True)
    parser.add_argument("--output-registry", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    counts = tuple(int(value) for value in str(args.counts).split(","))
    rows, manifest = build_plan(
        args.source_registry, counts=counts, panel_id=str(args.panel_id)
    )
    _atomic_text(
        args.output_registry,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    _atomic_text(
        args.output_manifest,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
