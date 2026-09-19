#!/usr/bin/env python3
"""Effect-size analysis for frozen natural patch scopes and splits."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping

import numpy as np


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _quantile(values: Iterable[float], probability: float) -> float:
    active = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if active.size == 0:
        return float("nan")
    return float(np.quantile(active, float(probability)))


def _paired_cells(directory: Path, *, split: str, scope: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in _read(directory / "trials.jsonl"):
        grouped[(int(row["seed"]), int(row["layer"]))][str(row["condition"])] = row
    cells = []
    for (seed, layer), conditions in grouped.items():
        required = {"receiver_self", "native_donor", "donor_to_receiver"}
        if required - set(conditions):
            raise ValueError(f"Incomplete frozen cell in {directory}: {seed}, L{layer}")
        receiver = conditions["receiver_self"]
        donor = conditions["native_donor"]
        patched = conditions["donor_to_receiver"]
        likelihood_shift = float(patched["donor_vs_receiver_sum_logodds"]) - float(
            receiver["donor_vs_receiver_sum_logodds"]
        )
        attention_shift = float(
            patched["donor_vs_receiver_attention_log_ratio"]
        ) - float(receiver["donor_vs_receiver_attention_log_ratio"])
        patch_norm = float(patched["realized_patch_delta_norm"])
        cells.append(
            {
                "split": split,
                "scope": scope,
                "directory": str(directory),
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
                "patch_width": int(patched["patch_width"]),
                "patch_norm": patch_norm,
                "paired_logodds_shift": likelihood_shift,
                "paired_attention_shift": attention_shift,
                "logodds_shift_per_patch_norm": (
                    likelihood_shift / patch_norm if patch_norm > 0.0 else None
                ),
                "attention_shift_per_patch_norm": (
                    attention_shift / patch_norm if patch_norm > 0.0 else None
                ),
                "receiver_baseline_argmax": bool(
                    receiver["receiver_successor_argmax"]
                ),
                "native_donor_argmax": bool(donor["donor_successor_argmax"]),
                "patched_donor_argmax": bool(patched["donor_successor_argmax"]),
                "patched_greedy_donor_adoption": bool(
                    patched["greedy_donor_successor_adoption"]
                ),
                "patched_first_known_city_ordinal": patched.get(
                    "first_generated_known_city_ordinal"
                ),
                "patched_first_bullet_city_ordinal": patched.get(
                    "first_generated_bullet_city_ordinal"
                ),
                "receiver_greedy_retention": bool(
                    receiver["greedy_receiver_successor_retention"]
                ),
                "receiver_first_known_city_ordinal": receiver.get(
                    "first_generated_known_city_ordinal"
                ),
                "receiver_first_bullet_city_ordinal": receiver.get(
                    "first_generated_bullet_city_ordinal"
                ),
                "equal_length_complete_item": bool(
                    patched.get("equal_length_complete_item", False)
                ),
                "receiver_item_coverage": float(
                    patched.get("receiver_item_coverage", 0.0)
                ),
                "donor_item_coverage": float(patched.get("donor_item_coverage", 0.0)),
            }
        )
    return cells


def _summary(cells: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not cells:
        return {"cell_count": 0}
    likelihood = [float(cell["paired_logodds_shift"]) for cell in cells]
    attention = [float(cell["paired_attention_shift"]) for cell in cells]
    norms = [float(cell["patch_norm"]) for cell in cells]
    normalized_likelihood = [
        float(cell["logodds_shift_per_patch_norm"])
        for cell in cells
        if cell["logodds_shift_per_patch_norm"] is not None
    ]
    normalized_attention = [
        float(cell["attention_shift_per_patch_norm"])
        for cell in cells
        if cell["attention_shift_per_patch_norm"] is not None
    ]
    by_seed: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_seed[int(cell["seed"])].append(cell)
    seed_likelihood = [
        mean(float(cell["paired_logodds_shift"]) for cell in group)
        for group in by_seed.values()
    ]
    seed_attention = [
        mean(float(cell["paired_attention_shift"]) for cell in group)
        for group in by_seed.values()
    ]
    exact_item = [cell for cell in cells if bool(cell["equal_length_complete_item"])]
    first_known_adoption = [
        cell
        for cell in cells
        if cell["patched_first_known_city_ordinal"]
        == int(cell["donor_occurrence_k"]) + 1
    ]
    first_bullet_adoption = [
        cell
        for cell in cells
        if cell["patched_first_bullet_city_ordinal"]
        == int(cell["donor_occurrence_k"]) + 1
    ]
    receiver_first_known_donor_adoption = [
        cell
        for cell in cells
        if cell["receiver_first_known_city_ordinal"]
        == int(cell["donor_occurrence_k"]) + 1
    ]
    receiver_first_bullet_donor_adoption = [
        cell
        for cell in cells
        if cell["receiver_first_bullet_city_ordinal"]
        == int(cell["donor_occurrence_k"]) + 1
    ]
    return {
        "cell_count": len(cells),
        "seed_count": len(by_seed),
        "mean_paired_logodds_shift": mean(likelihood),
        "median_paired_logodds_shift": median(likelihood),
        "q10_paired_logodds_shift": _quantile(likelihood, 0.10),
        "q25_paired_logodds_shift": _quantile(likelihood, 0.25),
        "q75_paired_logodds_shift": _quantile(likelihood, 0.75),
        "q90_paired_logodds_shift": _quantile(likelihood, 0.90),
        "positive_logodds_shift_rate": mean(value > 0.0 for value in likelihood),
        "minimum_seed_mean_logodds_shift": min(seed_likelihood),
        "median_seed_mean_logodds_shift": median(seed_likelihood),
        "mean_paired_attention_shift": mean(attention),
        "median_paired_attention_shift": median(attention),
        "q10_paired_attention_shift": _quantile(attention, 0.10),
        "q90_paired_attention_shift": _quantile(attention, 0.90),
        "positive_attention_shift_rate": mean(value > 0.0 for value in attention),
        "minimum_seed_mean_attention_shift": min(seed_attention),
        "median_seed_mean_attention_shift": median(seed_attention),
        "mean_patch_norm": mean(norms),
        "median_patch_norm": median(norms),
        "mean_logodds_shift_per_patch_norm": mean(normalized_likelihood),
        "median_logodds_shift_per_patch_norm": median(normalized_likelihood),
        "mean_attention_shift_per_patch_norm": mean(normalized_attention),
        "median_attention_shift_per_patch_norm": median(normalized_attention),
        "mean_patch_width": mean(int(cell["patch_width"]) for cell in cells),
        "patch_width_range": [
            min(int(cell["patch_width"]) for cell in cells),
            max(int(cell["patch_width"]) for cell in cells),
        ],
        "receiver_baseline_argmax_rate": mean(
            bool(cell["receiver_baseline_argmax"]) for cell in cells
        ),
        "native_donor_argmax_rate": mean(
            bool(cell["native_donor_argmax"]) for cell in cells
        ),
        "patched_donor_argmax_rate": mean(
            bool(cell["patched_donor_argmax"]) for cell in cells
        ),
        "patched_greedy_donor_adoption_rate": mean(
            bool(cell["patched_greedy_donor_adoption"]) for cell in cells
        ),
        "patched_first_known_city_donor_adoption_count": len(first_known_adoption),
        "patched_first_known_city_donor_adoption_rate": len(first_known_adoption)
        / len(cells),
        "receiver_first_known_city_donor_adoption_count": len(
            receiver_first_known_donor_adoption
        ),
        "receiver_first_known_city_donor_adoption_rate": len(
            receiver_first_known_donor_adoption
        )
        / len(cells),
        "paired_first_known_city_donor_adoption_gain": (
            len(first_known_adoption) - len(receiver_first_known_donor_adoption)
        )
        / len(cells),
        "patched_first_bullet_city_donor_adoption_count": len(
            first_bullet_adoption
        ),
        "patched_first_bullet_city_donor_adoption_rate": len(
            first_bullet_adoption
        )
        / len(cells),
        "receiver_first_bullet_city_donor_adoption_count": len(
            receiver_first_bullet_donor_adoption
        ),
        "receiver_first_bullet_city_donor_adoption_rate": len(
            receiver_first_bullet_donor_adoption
        )
        / len(cells),
        "paired_first_bullet_city_donor_adoption_gain": (
            len(first_bullet_adoption) - len(receiver_first_bullet_donor_adoption)
        )
        / len(cells),
        "receiver_greedy_retention_rate": mean(
            bool(cell["receiver_greedy_retention"]) for cell in cells
        ),
        "seed_with_any_greedy_donor_adoption_rate": mean(
            any(bool(cell["patched_greedy_donor_adoption"]) for cell in group)
            for group in by_seed.values()
        ),
        "equal_length_complete_item_cell_count": len(exact_item),
        "equal_length_complete_item_rate": len(exact_item) / len(cells),
        "equal_length_complete_item_paired_logodds": (
            None
            if not exact_item
            else {
                "mean": mean(
                    float(cell["paired_logodds_shift"]) for cell in exact_item
                ),
                "median": median(
                    float(cell["paired_logodds_shift"]) for cell in exact_item
                ),
                "donor_argmax_rate": mean(
                    bool(cell["patched_donor_argmax"]) for cell in exact_item
                ),
                "greedy_adoption_rate": mean(
                    bool(cell["patched_greedy_donor_adoption"])
                    for cell in exact_item
                ),
            }
        ),
        "mean_receiver_item_coverage": mean(
            float(cell["receiver_item_coverage"]) for cell in cells
        ),
        "mean_donor_item_coverage": mean(
            float(cell["donor_item_coverage"]) for cell in cells
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    all_cells: list[dict[str, Any]] = []
    for split_directory in sorted(path for path in args.root.iterdir() if path.is_dir()):
        split = split_directory.name
        for scope_directory in sorted(
            path for path in split_directory.iterdir() if path.is_dir()
        ):
            scope = scope_directory.name
            for directory in sorted(
                path for path in scope_directory.iterdir() if path.is_dir()
            ):
                if (directory / "trials.jsonl").exists():
                    all_cells.extend(
                        _paired_cells(directory, split=split, scope=scope)
                    )
    if not all_cells:
        raise ValueError(f"No frozen scope cells found under {args.root}")

    splits = sorted({str(cell["split"]) for cell in all_cells})
    scopes = sorted({str(cell["scope"]) for cell in all_cells})
    payload = {
        "schema_version": "natural_patch_scope_frozen_analysis_v1",
        "root": str(args.root),
        "summaries": [
            {
                "split": split,
                "scope": scope,
                **_summary(
                    [
                        cell
                        for cell in all_cells
                        if cell["split"] == split and cell["scope"] == scope
                    ]
                ),
                "by_direction_k": [
                    {
                        "direction": direction,
                        "donor_occurrence_k": donor,
                        **_summary(
                            [
                                cell
                                for cell in all_cells
                                if cell["split"] == split
                                and cell["scope"] == scope
                                and cell["direction"] == direction
                                and int(cell["donor_occurrence_k"]) == donor
                            ]
                        ),
                    }
                    for donor in (4, 6, 8)
                    for direction in ("forward_skip", "backward_rewind")
                ],
            }
            for split in splits
            for scope in scopes
        ],
        "cells": all_cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["summaries"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
