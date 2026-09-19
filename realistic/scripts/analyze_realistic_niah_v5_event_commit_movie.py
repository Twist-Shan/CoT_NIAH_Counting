#!/usr/bin/env python3
"""Combine event-commit movie batches and extract the transaction diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "event_commit_movie_analysis_v1"
INVALID_INSERT_VARIANTS = (
    "insert_markerless_valid_payload",
    "insert_marker_neutral_payload",
    "insert_neutral_line",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prediction_counts(values: Sequence[int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(values).items())}


def summarize_event_movie(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pure-JSON copy of the run-time summary, with no model dependencies."""

    clean_groups: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    donor_index: dict[tuple[int, int, str, int], Mapping[str, Any]] = {}
    for row in rows:
        for landmark in row["landmarks"]:
            if str(row["condition"]) == "clean":
                clean_groups[
                    (str(row["event_variant"]), int(row["read_layer"]), str(landmark))
                ].append(row)
            elif str(row["event_variant"]) == "insert_valid_item":
                key = (
                    int(row["seed"]),
                    int(row["read_layer"]),
                    str(landmark),
                    int(row["donor"]),
                )
                if key in donor_index:
                    raise ValueError(f"Duplicate donor landmark row: {key}")
                donor_index[key] = row

    clean_cells = []
    for (variant, layer, landmark), active in sorted(clean_groups.items()):
        predictions = [int(row["probe_prediction"]) for row in active]
        clean_cells.append(
            {
                "event_variant": variant,
                "read_layer": layer,
                "landmark": landmark,
                "n_seeds": len(active),
                "prediction_counts": _prediction_counts(predictions),
                "mean_soft_count": fmean(
                    float(row["probe_softmax_expected_count"]) for row in active
                ),
            }
        )

    pair_groups: dict[
        tuple[int, str], list[tuple[Mapping[str, Any], Mapping[str, Any]]]
    ] = defaultdict(list)
    for seed, layer, landmark in sorted(
        {(seed, layer, landmark) for seed, layer, landmark, _donor in donor_index}
    ):
        low = donor_index.get((seed, layer, landmark, 4))
        high = donor_index.get((seed, layer, landmark, 6))
        if low is None or high is None:
            raise ValueError(
                f"Incomplete donor pair at seed={seed}, layer={layer}, landmark={landmark}"
            )
        pair_groups[(layer, landmark)].append((low, high))

    current_separation: dict[tuple[int, int], float] = {}
    for (layer, landmark), pairs in pair_groups.items():
        if landmark == "current_boundary":
            for low, high in pairs:
                current_separation[(int(low["seed"]), layer)] = float(
                    high["probe_softmax_expected_count"]
                ) - float(low["probe_softmax_expected_count"])

    donor_cells = []
    for (layer, landmark), pairs in sorted(pair_groups.items()):
        separations = [
            float(high["probe_softmax_expected_count"])
            - float(low["probe_softmax_expected_count"])
            for low, high in pairs
        ]
        retention = [
            separation / current_separation[(int(low["seed"]), layer)]
            for (low, _high), separation in zip(pairs, separations)
            if abs(current_separation[(int(low["seed"]), layer)]) > 1e-12
        ]
        donor_cells.append(
            {
                "read_layer": layer,
                "landmark": landmark,
                "n_seed_pairs": len(pairs),
                "donor_invariant_count": sum(
                    int(low["probe_prediction"]) == int(high["probe_prediction"])
                    for low, high in pairs
                ),
                "recurrent_separation_2_count": sum(
                    int(high["probe_prediction"]) - int(low["probe_prediction"]) == 2
                    for low, high in pairs
                ),
                "mean_soft_donor_separation": fmean(separations),
                "mean_within_seed_retention": fmean(retention) if retention else None,
                "donor4_prediction_counts": _prediction_counts(
                    [int(low["probe_prediction"]) for low, _high in pairs]
                ),
                "donor6_prediction_counts": _prediction_counts(
                    [int(high["probe_prediction"]) for _low, high in pairs]
                ),
            }
        )
    return {
        "clean_landmark_cells": clean_cells,
        "valid_item_donor_pair_cells": donor_cells,
    }


def _cell(
    cells: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    layer: int,
    landmark: str,
) -> Mapping[str, Any]:
    matches = [
        cell
        for cell in cells
        if str(cell["event_variant"]) == variant
        and int(cell["read_layer"]) == layer
        and str(cell["landmark"]) == landmark
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one cell for {variant}, L{layer}, {landmark}; found {len(matches)}"
        )
    return matches[0]


def _donor_cell(
    cells: Sequence[Mapping[str, Any]], *, layer: int, landmark: str
) -> Mapping[str, Any]:
    matches = [
        cell
        for cell in cells
        if int(cell["read_layer"]) == layer
        and str(cell["landmark"]) == landmark
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one donor cell for L{layer}, {landmark}; found {len(matches)}"
        )
    return matches[0]


def count_prediction(cell: Mapping[str, Any], label: int) -> int:
    return int(cell["prediction_counts"].get(str(label), 0))


def transaction_diagnostics(summary: Mapping[str, Any]) -> dict[str, Any]:
    clean = summary["clean_landmark_cells"]
    donors = summary["valid_item_donor_pair_cells"]
    valid_intermediate = _cell(
        clean,
        variant="insert_valid_item",
        layer=24,
        landmark="inserted_event_boundary",
    )
    valid_target = _cell(
        clean,
        variant="insert_valid_item",
        layer=24,
        landmark="target_boundary",
    )
    invalid_intermediate = [
        _cell(
            clean,
            variant=variant,
            layer=24,
            landmark="inserted_event_boundary",
        )
        for variant in INVALID_INSERT_VARIANTS
    ]
    invalid_target = [
        _cell(clean, variant=variant, layer=24, landmark="target_boundary")
        for variant in INVALID_INSERT_VARIANTS
    ]
    marker_proposal = _cell(
        clean,
        variant="insert_marker_neutral_payload",
        layer=15,
        landmark="target_boundary",
    )
    payload_without_marker = _cell(
        clean,
        variant="insert_markerless_valid_payload",
        layer=15,
        landmark="target_boundary",
    )
    current = _donor_cell(donors, layer=15, landmark="current_boundary")
    marker_end = _donor_cell(donors, layer=15, landmark="inserted_marker_end")
    inserted_commit = _donor_cell(
        donors, layer=15, landmark="inserted_event_boundary"
    )
    target_commit = _donor_cell(donors, layer=15, landmark="target_boundary")
    n_seeds = int(valid_intermediate["n_seeds"])
    return {
        "n_seeds": n_seeds,
        "valid_item_intermediate_boundary_label6": count_prediction(
            valid_intermediate, 6
        ),
        "valid_item_target_boundary_label7": count_prediction(valid_target, 7),
        "valid_item_target_boundary_not_label6": n_seeds
        - count_prediction(valid_target, 6),
        "invalid_intermediate_label6": {
            variant: count_prediction(cell, 6)
            for variant, cell in zip(INVALID_INSERT_VARIANTS, invalid_intermediate)
        },
        "invalid_target_label6": {
            variant: count_prediction(cell, 6)
            for variant, cell in zip(INVALID_INSERT_VARIANTS, invalid_target)
        },
        "early_marker_only_target_label7": count_prediction(marker_proposal, 7),
        "early_payload_without_marker_target_label7": count_prediction(
            payload_without_marker, 7
        ),
        "donor_current_recurrent_separation2": int(
            current["recurrent_separation_2_count"]
        ),
        "donor_marker_end_invariant": int(marker_end["donor_invariant_count"]),
        "donor_marker_end_retention": float(
            marker_end["mean_within_seed_retention"]
        ),
        "donor_inserted_commit_invariant": int(
            inserted_commit["donor_invariant_count"]
        ),
        "donor_inserted_commit_retention": float(
            inserted_commit["mean_within_seed_retention"]
        ),
        "donor_target_commit_invariant": int(
            target_commit["donor_invariant_count"]
        ),
        "donor_target_commit_retention": float(
            target_commit["mean_within_seed_retention"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    batches = []
    all_rows: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    for path in args.trials:
        rows = read_jsonl(path)
        seeds = {int(row["seed"]) for row in rows}
        overlap = seen_seeds & seeds
        if overlap:
            raise ValueError(f"Seed overlap across movie batches: {sorted(overlap)}")
        seen_seeds |= seeds
        batch_summary = summarize_event_movie(rows)
        batches.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "seeds": sorted(seeds),
                "diagnostics": transaction_diagnostics(batch_summary),
            }
        )
        all_rows.extend(rows)

    combined = summarize_event_movie(all_rows)
    result = {
        "schema_version": SCHEMA_VERSION,
        "batches": batches,
        "combined_seeds": sorted(seen_seeds),
        "combined_diagnostics": transaction_diagnostics(combined),
        "combined_summary": combined,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["combined_diagnostics"], indent=2, sort_keys=True))
    for batch in batches:
        print(f"batch seeds={batch['seeds']}")
        print(json.dumps(batch["diagnostics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
