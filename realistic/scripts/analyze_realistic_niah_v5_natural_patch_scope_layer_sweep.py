#!/usr/bin/env python3
"""Compare effect-size layer curves for natural progress patch scopes."""

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
        raise ValueError("Cannot summarize an empty effect-size vector")
    return float(np.quantile(active, float(probability)))


def _baseline_rows(root: Path) -> dict[tuple[str, int], dict[str, dict[str, Any]]]:
    output: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for path in sorted((root / "baseline").glob("*/trials.jsonl")):
        for row in _read(path):
            direction = (
                "forward_skip"
                if int(row["receiver_occurrence_j"]) < int(row["donor_occurrence_k"])
                else "backward_rewind"
            )
            output[(direction, int(row["seed"]))][str(row["condition"])] = row
    if not output:
        raise ValueError(f"No cached native controls found under {root / 'baseline'}")
    return output


def _scope_cells(
    scope_directory: Path,
    baselines: Mapping[tuple[str, int], Mapping[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    trial_paths = sorted(scope_directory.glob("*/trials.jsonl"))
    if not trial_paths:
        raise ValueError(f"No trial directories found under {scope_directory}")
    for path in trial_paths:
        grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in _read(path):
            grouped[(int(row["seed"]), int(row["layer"]))][
                str(row["condition"])
            ] = row
        for (seed, layer), conditions in grouped.items():
            if "donor_to_receiver" not in conditions:
                continue
            patched = conditions["donor_to_receiver"]
            direction = (
                "forward_skip"
                if int(patched["receiver_occurrence_j"])
                < int(patched["donor_occurrence_k"])
                else "backward_rewind"
            )
            native_controls = baselines.get((direction, seed), {})
            receiver = native_controls.get("receiver_self")
            donor = native_controls.get("native_donor")
            if receiver is None or donor is None:
                raise ValueError(
                    f"Missing cached native controls for {direction} seed {seed}"
                )
            raw_shift = float(patched["donor_vs_receiver_sum_logodds"]) - float(
                receiver["donor_vs_receiver_sum_logodds"]
            )
            patch_norm = float(patched["realized_patch_delta_norm"])
            cells.append(
                {
                    "scope": scope_directory.name,
                    "directory": str(path.parent),
                    "seed": seed,
                    "layer": layer,
                    "direction": direction,
                    "receiver_occurrence_j": int(patched["receiver_occurrence_j"]),
                    "donor_occurrence_k": int(patched["donor_occurrence_k"]),
                    "patch_width": int(patched.get("patch_width", 1)),
                    "patch_norm": patch_norm,
                    "paired_logodds_shift": raw_shift,
                    "logodds_shift_per_patch_norm": (
                        raw_shift / patch_norm if patch_norm > 0.0 else None
                    ),
                    "receiver_baseline_argmax": bool(
                        receiver["receiver_successor_argmax"]
                    ),
                    "patched_donor_argmax": bool(patched["donor_successor_argmax"]),
                    "native_donor_argmax": (
                        None if donor is None else bool(donor["donor_successor_argmax"])
                    ),
                    "equal_length_complete_item": bool(
                        patched.get("equal_length_complete_item", False)
                    ),
                    "receiver_item_coverage": float(
                        patched.get("receiver_item_coverage", 0.0)
                    ),
                    "donor_item_coverage": float(
                        patched.get("donor_item_coverage", 0.0)
                    ),
                }
            )
    return cells


def _summarize(cells: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not cells:
        raise ValueError("Cannot summarize zero paired cells")
    shifts = [float(cell["paired_logodds_shift"]) for cell in cells]
    norms = [float(cell["patch_norm"]) for cell in cells]
    normalized = [
        float(cell["logodds_shift_per_patch_norm"])
        for cell in cells
        if cell["logodds_shift_per_patch_norm"] is not None
    ]
    by_seed: dict[int, list[float]] = defaultdict(list)
    for cell in cells:
        by_seed[int(cell["seed"])].append(float(cell["paired_logodds_shift"]))
    seed_means = [mean(values) for values in by_seed.values()]
    native = [
        bool(cell["native_donor_argmax"])
        for cell in cells
        if cell["native_donor_argmax"] is not None
    ]
    return {
        "cell_count": len(cells),
        "seed_count": len(by_seed),
        "mean_paired_logodds_shift": mean(shifts),
        "median_paired_logodds_shift": median(shifts),
        "q10_paired_logodds_shift": _quantile(shifts, 0.10),
        "q25_paired_logodds_shift": _quantile(shifts, 0.25),
        "q75_paired_logodds_shift": _quantile(shifts, 0.75),
        "q90_paired_logodds_shift": _quantile(shifts, 0.90),
        "positive_shift_rate": mean(value > 0.0 for value in shifts),
        "mean_patch_norm": mean(norms),
        "median_patch_norm": median(norms),
        "mean_logodds_shift_per_patch_norm": (
            mean(normalized) if normalized else None
        ),
        "median_logodds_shift_per_patch_norm": (
            median(normalized) if normalized else None
        ),
        "mean_patch_width": mean(int(cell["patch_width"]) for cell in cells),
        "patch_width_range": [
            min(int(cell["patch_width"]) for cell in cells),
            max(int(cell["patch_width"]) for cell in cells),
        ],
        "receiver_baseline_argmax_rate": mean(
            bool(cell["receiver_baseline_argmax"]) for cell in cells
        ),
        "native_donor_argmax_rate": mean(native) if native else None,
        "patched_donor_argmax_rate": mean(
            bool(cell["patched_donor_argmax"]) for cell in cells
        ),
        "seed_mean_effects": seed_means,
        "mean_seed_mean_effect": mean(seed_means),
        "median_seed_mean_effect": median(seed_means),
        "minimum_seed_mean_effect": min(seed_means),
        "equal_length_complete_item_rate": mean(
            bool(cell["equal_length_complete_item"]) for cell in cells
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
    parser.add_argument(
        "--max-causal-layer",
        type=int,
        help=(
            "Largest layer eligible for freezing. By default the final observed "
            "decoder block is excluded because it has no downstream block."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baselines = _baseline_rows(args.root)
    scope_directories = sorted(
        path
        for path in args.root.iterdir()
        if path.is_dir()
        and path.name != "baseline"
        and list(path.glob("*/trials.jsonl"))
    )
    if not scope_directories:
        raise ValueError(f"No completed scope directories under {args.root}")
    all_cells: list[dict[str, Any]] = []
    scope_payloads = []
    for scope_directory in scope_directories:
        cells = _scope_cells(scope_directory, baselines)
        all_cells.extend(cells)
        layers = sorted({int(cell["layer"]) for cell in cells})
        max_causal_layer = (
            max(layers) - 1
            if args.max_causal_layer is None
            else int(args.max_causal_layer)
        )
        layer_summaries = []
        for layer in layers:
            group = [cell for cell in cells if int(cell["layer"]) == layer]
            layer_summaries.append(
                {
                    "layer": layer,
                    **_summarize(group),
                    "directions": {
                        direction: _summarize(
                            [cell for cell in group if cell["direction"] == direction]
                        )
                        for direction in ("forward_skip", "backward_rewind")
                    },
                }
            )
        # Freeze one layer from discovery by a robust effect-size plateau.
        # L35 has no downstream decoder block and is therefore descriptive
        # only. Greedy generation, attention, and confirmation never enter the
        # choice. Both transplant directions must have a positive cell median.
        eligible = [
            row
            for row in layer_summaries
            if int(row["layer"]) <= max_causal_layer
            and float(row["directions"]["forward_skip"]["median_paired_logodds_shift"])
            > 0.0
            and float(
                row["directions"]["backward_rewind"]["median_paired_logodds_shift"]
            )
            > 0.0
        ]
        if not eligible:
            raise ValueError(f"{scope_directory.name}: no bidirectionally positive layer")
        peak_effect = max(float(row["median_seed_mean_effect"]) for row in eligible)
        plateau_threshold = 0.95 * peak_effect
        plateau = [
            row
            for row in eligible
            if float(row["median_seed_mean_effect"]) >= plateau_threshold
        ]
        selected = min(plateau, key=lambda row: int(row["layer"]))
        scope_payloads.append(
            {
                "scope": scope_directory.name,
                "directory": str(scope_directory),
                "selection_rule": (
                    f"among L0-L{max_causal_layer} with positive forward and "
                    "backward cell medians, "
                    "choose the earliest layer reaching 95% of the maximum median "
                    "seed-mean paired donor-directed transition log-odds shift"
                ),
                "max_causal_layer": max_causal_layer,
                "peak_median_seed_mean_effect": peak_effect,
                "plateau_threshold": plateau_threshold,
                "selected_layer": int(selected["layer"]),
                "selected_layer_summary": selected,
                "layer_summaries": layer_summaries,
            }
        )

    payload = {
        "schema_version": "natural_patch_scope_layer_sweep_analysis_v1",
        "root": str(args.root),
        "scope_count": len(scope_payloads),
        "scopes": scope_payloads,
        "cells": all_cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            [
                {
                    "scope": row["scope"],
                    "selected_layer": row["selected_layer"],
                    "selected_layer_summary": row["selected_layer_summary"],
                }
                for row in scope_payloads
            ],
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
