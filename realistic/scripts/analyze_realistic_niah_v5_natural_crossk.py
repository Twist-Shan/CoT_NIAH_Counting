#!/usr/bin/env python3
"""Aggregate natural progress transplants across k and direction."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sign_test(values: list[float]) -> float | None:
    active = [value for value in values if value != 0.0]
    if not active:
        return None
    n = len(active)
    positive = sum(value > 0.0 for value in active)
    tail = sum(math.comb(n, k) for k in range(min(positive, n - positive) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def _paired_cells(directory: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in _read(directory / "trials.jsonl"):
        grouped[(int(row["seed"]), int(row["layer"]))][str(row["condition"])] = row
    cells = []
    for (seed, layer), conditions in grouped.items():
        if {"receiver_self", "native_donor", "donor_to_receiver"} - set(conditions):
            continue
        receiver = conditions["receiver_self"]
        donor = conditions["native_donor"]
        patched = conditions["donor_to_receiver"]
        cell = {
            "directory": str(directory),
            "cohort": directory.parent.name,
            "seed": seed,
            "layer": layer,
            "receiver_occurrence_j": int(patched["receiver_occurrence_j"]),
            "donor_occurrence_k": int(patched["donor_occurrence_k"]),
            "direction": (
                "forward_skip"
                if int(patched["receiver_occurrence_j"])
                < int(patched["donor_occurrence_k"])
                else "backward_rewind"
            ),
            "patch_width": int(patched.get("patch_width", 1)),
            "paired_donor_logodds_shift": float(
                patched["donor_vs_receiver_sum_logodds"]
            )
            - float(receiver["donor_vs_receiver_sum_logodds"]),
            "receiver_baseline_argmax": bool(receiver["receiver_successor_argmax"]),
            "native_donor_argmax": bool(donor["donor_successor_argmax"]),
            "patched_donor_argmax": bool(patched["donor_successor_argmax"]),
        }
        if "donor_vs_receiver_attention_log_ratio" in patched:
            cell["paired_donor_attention_shift"] = float(
                patched["donor_vs_receiver_attention_log_ratio"]
            ) - float(receiver["donor_vs_receiver_attention_log_ratio"])
        if "greedy_donor_successor_adoption" in patched:
            cell["patched_greedy_donor_adoption"] = bool(
                patched["greedy_donor_successor_adoption"]
            )
            cell["patched_first_known_city_ordinal"] = patched.get(
                "first_generated_known_city_ordinal"
            )
            cell["receiver_greedy_retention"] = bool(
                receiver["greedy_receiver_successor_retention"]
            )
            cell["receiver_first_known_city_ordinal"] = receiver.get(
                "first_generated_known_city_ordinal"
            )
        cells.append(cell)
    return cells


def _summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    shifts = [float(cell["paired_donor_logodds_shift"]) for cell in cells]
    attention = [
        float(cell["paired_donor_attention_shift"])
        for cell in cells
        if "paired_donor_attention_shift" in cell
    ]
    by_seed: dict[int, list[float]] = defaultdict(list)
    attention_by_seed: dict[int, list[float]] = defaultdict(list)
    for cell in cells:
        by_seed[int(cell["seed"])].append(float(cell["paired_donor_logodds_shift"]))
        if "paired_donor_attention_shift" in cell:
            attention_by_seed[int(cell["seed"])].append(
                float(cell["paired_donor_attention_shift"])
            )
    seed_means = [mean(values) for values in by_seed.values()]
    attention_seed_means = [mean(values) for values in attention_by_seed.values()]
    generated = [cell for cell in cells if "patched_greedy_donor_adoption" in cell]
    return {
        "cell_count": len(cells),
        "seed_count": len(by_seed),
        "mean_paired_donor_logodds_shift": mean(shifts),
        "median_paired_donor_logodds_shift": median(shifts),
        "positive_logodds_shift_rate": mean(value > 0.0 for value in shifts),
        "cell_level_two_sided_sign_test_p": _sign_test(shifts),
        "seed_cluster_mean_shifts": seed_means,
        "seed_cluster_positive_rate": mean(value > 0.0 for value in seed_means),
        "seed_cluster_two_sided_sign_test_p": _sign_test(seed_means),
        "receiver_baseline_argmax_rate": mean(
            bool(cell["receiver_baseline_argmax"]) for cell in cells
        ),
        "native_donor_argmax_rate": mean(bool(cell["native_donor_argmax"]) for cell in cells),
        "patched_donor_argmax_rate": mean(
            bool(cell["patched_donor_argmax"]) for cell in cells
        ),
        "mean_paired_donor_attention_shift": mean(attention) if attention else None,
        "median_paired_donor_attention_shift": median(attention) if attention else None,
        "positive_attention_shift_rate": (
            mean(value > 0.0 for value in attention) if attention else None
        ),
        "cell_level_attention_two_sided_sign_test_p": (
            _sign_test(attention) if attention else None
        ),
        "seed_cluster_mean_attention_shifts": attention_seed_means,
        "seed_cluster_positive_attention_rate": (
            mean(value > 0.0 for value in attention_seed_means)
            if attention_seed_means
            else None
        ),
        "seed_cluster_attention_two_sided_sign_test_p": (
            _sign_test(attention_seed_means) if attention_seed_means else None
        ),
        "patched_greedy_donor_adoption_rate": (
            mean(bool(cell["patched_greedy_donor_adoption"]) for cell in generated)
            if generated
            else None
        ),
        "receiver_greedy_retention_rate": (
            mean(bool(cell["receiver_greedy_retention"]) for cell in generated)
            if generated
            else None
        ),
        "seed_with_any_greedy_donor_adoption_rate": (
            mean(
                any(
                    bool(cell.get("patched_greedy_donor_adoption", False))
                    for cell in generated
                    if int(cell["seed"]) == seed
                )
                for seed in sorted({int(cell["seed"]) for cell in generated})
            )
            if generated
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory_cells = [(directory, _paired_cells(directory)) for directory in args.directories]
    cells = [cell for _directory, group in directory_cells for cell in group]
    payload = {
        "directory_summaries": [
            {"directory": str(directory), **_summary(group)}
            for directory, group in directory_cells
        ],
        "pooled_summary": _summary(cells),
        "direction_k_summaries": [
            {
                "direction": direction,
                "donor_occurrence_k": donor,
                **_summary(
                    [
                        cell
                        for cell in cells
                        if cell["direction"] == direction
                        and int(cell["donor_occurrence_k"]) == donor
                    ]
                ),
            }
            for donor in sorted({int(cell["donor_occurrence_k"]) for cell in cells})
            for direction in ("forward_skip", "backward_rewind")
        ],
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["pooled_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
