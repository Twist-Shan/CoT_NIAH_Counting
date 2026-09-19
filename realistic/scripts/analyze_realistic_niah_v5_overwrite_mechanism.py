#!/usr/bin/env python3
"""Summarize donor-state invariance in the v5 list-event edit scan.

The decisive comparison pairs donor 4 and donor 6 within the same seed,
event edit, and read layer.  A literal donor-conditioned successor rule should
make the two next-boundary predictions differ by two.  Contextual
reconstruction instead predicts that they usually agree with each other and
with the clean, edit-specific target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "overwrite_mechanism_analysis_v1"
PRIMARY_VARIANTS = (
    "original",
    "insert_valid_item",
    "delete_prior_valid_item",
)
MATCHED_INSERT_VARIANTS = (
    "insert_valid_item",
    "insert_markerless_valid_payload",
    "insert_marker_neutral_payload",
    "insert_neutral_line",
)
DONORS = (4, 6)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError("Cannot divide by an empty population")
    return float(numerator / denominator)


def _prediction_counts(values: Iterable[int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(values).items())}


def paired_donor_cells(
    rows: Sequence[Mapping[str, Any]],
    *,
    variants: Sequence[str] = PRIMARY_VARIANTS,
    donors: tuple[int, int] = DONORS,
) -> list[dict[str, Any]]:
    """Aggregate paired donor trials by event variant and read layer."""

    allowed = set(variants)
    grouped: dict[tuple[int, str, int], dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        variant = str(row["event_variant"])
        if variant not in allowed:
            continue
        key = (int(row["seed"]), variant, int(row["read_layer"]))
        donor = int(row["donor"])
        if donor in grouped[key]:
            raise ValueError(f"Duplicate donor row for {key}, donor={donor}")
        grouped[key][donor] = row

    incomplete = {key: sorted(pair) for key, pair in grouped.items() if set(pair) != set(donors)}
    if incomplete:
        raise ValueError(f"Incomplete donor pairs: {incomplete}")

    cells: list[dict[str, Any]] = []
    variant_layers = sorted({(variant, layer) for _, variant, layer in grouped})
    donor_low, donor_high = donors
    for variant, layer in variant_layers:
        pairs = [
            pair
            for (seed, cell_variant, cell_layer), pair in sorted(grouped.items())
            if cell_variant == variant and cell_layer == layer
        ]
        flat = [row for pair in pairs for row in pair.values()]
        clean_targets: list[int] = []
        for pair in pairs:
            low_clean = int(pair[donor_low]["clean_target_prediction"])
            high_clean = int(pair[donor_high]["clean_target_prediction"])
            if low_clean != high_clean:
                raise ValueError(
                    f"Clean target differs across donors for {variant}, layer {layer}"
                )
            clean_targets.append(low_clean)

        invariant = [
            int(pair[donor_low]["next_prediction"])
            == int(pair[donor_high]["next_prediction"])
            for pair in pairs
        ]
        discrete_separations = [
            int(pair[donor_high]["next_prediction"])
            - int(pair[donor_low]["next_prediction"])
            for pair in pairs
        ]
        current_soft_separations = [
            float(pair[donor_high]["current_soft"])
            - float(pair[donor_low]["current_soft"])
            for pair in pairs
        ]
        next_soft_separations = [
            float(pair[donor_high]["next_soft"])
            - float(pair[donor_low]["next_soft"])
            for pair in pairs
        ]
        mean_current_separation = fmean(current_soft_separations)
        mean_next_separation = fmean(next_soft_separations)
        cells.append(
            {
                "event_variant": variant,
                "read_layer": layer,
                "n_seed_pairs": len(pairs),
                "n_trials": len(flat),
                "current_donor_exact_count": sum(bool(row["current_donor_exact"]) for row in flat),
                "current_donor_accuracy": _ratio(
                    sum(bool(row["current_donor_exact"]) for row in flat), len(flat)
                ),
                "next_event_exact_count": sum(bool(row["next_event_count_exact"]) for row in flat),
                "next_event_accuracy": _ratio(
                    sum(bool(row["next_event_count_exact"]) for row in flat), len(flat)
                ),
                "reset_to_clean_count": sum(
                    int(row["next_prediction"]) == int(row["clean_target_prediction"])
                    for row in flat
                ),
                "reset_to_clean_accuracy": _ratio(
                    sum(
                        int(row["next_prediction"]) == int(row["clean_target_prediction"])
                        for row in flat
                    ),
                    len(flat),
                ),
                "donor_invariant_count": sum(invariant),
                "donor_invariant_accuracy": _ratio(sum(invariant), len(pairs)),
                "recurrent_separation_exact_count": sum(
                    separation == donor_high - donor_low for separation in discrete_separations
                ),
                "recurrent_separation_accuracy": _ratio(
                    sum(separation == donor_high - donor_low for separation in discrete_separations),
                    len(pairs),
                ),
                "next_discrete_separations": _prediction_counts(discrete_separations),
                "mean_current_soft_donor_separation": mean_current_separation,
                "mean_next_soft_donor_separation": mean_next_separation,
                "soft_separation_retention": (
                    mean_next_separation / mean_current_separation
                    if mean_current_separation != 0.0
                    else None
                ),
                "clean_prediction_counts": _prediction_counts(clean_targets),
                "patched_prediction_counts": _prediction_counts(
                    int(row["next_prediction"]) for row in flat
                ),
            }
        )
    return cells


def clean_prediction_cells(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate the repeated clean run across donors and summarize predictions."""

    deduplicated: dict[tuple[int, str, int], tuple[int, float]] = {}
    for row in rows:
        key = (int(row["seed"]), str(row["event_variant"]), int(row["read_layer"]))
        value = (int(row["clean_target_prediction"]), float(row["clean_target_soft"]))
        if key in deduplicated and deduplicated[key] != value:
            raise ValueError(f"Clean result differs across donor copies for {key}")
        deduplicated[key] = value

    cells: list[dict[str, Any]] = []
    variant_layers = sorted({(variant, layer) for _, variant, layer in deduplicated})
    for variant, layer in variant_layers:
        values = [
            value
            for (seed, cell_variant, cell_layer), value in sorted(deduplicated.items())
            if cell_variant == variant and cell_layer == layer
        ]
        cells.append(
            {
                "event_variant": variant,
                "read_layer": layer,
                "n_seeds": len(values),
                "clean_prediction_counts": _prediction_counts(value[0] for value in values),
                "clean_label_7_count": sum(value[0] == 7 for value in values),
                "clean_label_7_accuracy": _ratio(sum(value[0] == 7 for value in values), len(values)),
                "mean_clean_soft_count": fmean(value[1] for value in values),
            }
        )
    return cells


def subset_rows(
    rows: Sequence[Mapping[str, Any]], *, seeds: set[int]
) -> list[Mapping[str, Any]]:
    return [row for row in rows if int(row["seed"]) in seeds]


def aggregate_primary_layer(cells: Sequence[Mapping[str, Any]], *, layer: int) -> dict[str, Any]:
    selected = [
        cell
        for cell in cells
        if int(cell["read_layer"]) == layer and str(cell["event_variant"]) in PRIMARY_VARIANTS
    ]
    if len(selected) != len(PRIMARY_VARIANTS):
        raise ValueError(f"Expected {len(PRIMARY_VARIANTS)} primary cells at layer {layer}")
    n_trials = sum(int(cell["n_trials"]) for cell in selected)
    n_pairs = sum(int(cell["n_seed_pairs"]) for cell in selected)
    current_soft = fmean(float(cell["mean_current_soft_donor_separation"]) for cell in selected)
    next_soft = fmean(float(cell["mean_next_soft_donor_separation"]) for cell in selected)
    return {
        "read_layer": layer,
        "n_trials": n_trials,
        "n_seed_variant_pairs": n_pairs,
        "current_donor_exact_count": sum(int(cell["current_donor_exact_count"]) for cell in selected),
        "next_event_exact_count": sum(int(cell["next_event_exact_count"]) for cell in selected),
        "reset_to_clean_count": sum(int(cell["reset_to_clean_count"]) for cell in selected),
        "donor_invariant_count": sum(int(cell["donor_invariant_count"]) for cell in selected),
        "recurrent_separation_exact_count": sum(
            int(cell["recurrent_separation_exact_count"]) for cell in selected
        ),
        "mean_current_soft_donor_separation": current_soft,
        "mean_next_soft_donor_separation": next_soft,
        "soft_separation_retention": next_soft / current_soft,
    }


def analyze(
    rows: Sequence[Mapping[str, Any]],
    *,
    discovery_seeds: Sequence[int] = (),
    replication_seeds: Sequence[int] = (),
) -> dict[str, Any]:
    paired = paired_donor_cells(rows)
    analysis: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "primary_variants": list(PRIMARY_VARIANTS),
        "donors": list(DONORS),
        "paired_donor_cells": paired,
        "clean_prediction_cells": clean_prediction_cells(rows),
        "primary_layer_15": aggregate_primary_layer(paired, layer=15),
    }
    if discovery_seeds:
        discovery_cells = paired_donor_cells(
            subset_rows(rows, seeds=set(discovery_seeds))
        )
        analysis["discovery_seeds"] = list(discovery_seeds)
        analysis["discovery_primary_layer_15"] = aggregate_primary_layer(
            discovery_cells, layer=15
        )
    if replication_seeds:
        replication_cells = paired_donor_cells(
            subset_rows(rows, seeds=set(replication_seeds))
        )
        analysis["replication_seeds"] = list(replication_seeds)
        analysis["replication_primary_layer_15"] = aggregate_primary_layer(
            replication_cells, layer=15
        )
    return analysis


def parse_seed_csv(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(part) for part in value.split(","))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--discovery-seeds", type=parse_seed_csv, default=())
    parser.add_argument("--replication-seeds", type=parse_seed_csv, default=())
    args = parser.parse_args()

    rows = read_jsonl(args.trials)
    result = analyze(
        rows,
        discovery_seeds=args.discovery_seeds,
        replication_seeds=args.replication_seeds,
    )
    result["source_trials"] = str(args.trials.resolve())
    result["source_trials_sha256"] = sha256(args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["primary_layer_15"], indent=2, sort_keys=True))
    if "replication_primary_layer_15" in result:
        print("replication:")
        print(json.dumps(result["replication_primary_layer_15"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
