#!/usr/bin/env python3
"""Analyze the V6 natural-patch sweep while retaining negative discovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from analyze_realistic_niah_v5_natural_patch_scope_layer_sweep import (
    _baseline_rows,
    _scope_cells,
    _summarize,
)


def _scope_analysis(
    scope_directory: Path,
    baselines: Mapping[tuple[str, int], Mapping[str, dict[str, Any]]],
    *,
    max_causal_layer: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cells = _scope_cells(scope_directory, baselines)
    layers = sorted({int(cell["layer"]) for cell in cells})
    active_max = max(layers) - 1 if max_causal_layer is None else int(max_causal_layer)
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
    eligible = [
        row
        for row in layer_summaries
        if int(row["layer"]) <= active_max
        and float(row["directions"]["forward_skip"]["median_paired_logodds_shift"])
        > 0.0
        and float(
            row["directions"]["backward_rewind"]["median_paired_logodds_shift"]
        )
        > 0.0
    ]
    common = {
        "scope": scope_directory.name,
        "directory": str(scope_directory.resolve()),
        "selection_rule": (
            f"among L0-L{active_max} with positive forward and backward cell "
            "medians, choose the earliest layer reaching 95% of the maximum "
            "median seed-mean paired donor-directed transition log-odds shift"
        ),
        "max_causal_layer": active_max,
        "layer_summaries": layer_summaries,
        "confirmation_outcomes_read": False,
    }
    if not eligible:
        return (
            {
                **common,
                "status": "NEGATIVE_FROZEN",
                "selected_layer": None,
                "selected_layer_summary": None,
                "peak_median_seed_mean_effect": None,
                "plateau_threshold": None,
                "negative_result_retained": True,
                "negative_reason": "no_bidirectionally_positive_discovery_layer",
            },
            cells,
        )
    peak_effect = max(float(row["median_seed_mean_effect"]) for row in eligible)
    threshold = 0.95 * peak_effect
    selected = min(
        (
            row
            for row in eligible
            if float(row["median_seed_mean_effect"]) >= threshold
        ),
        key=lambda row: int(row["layer"]),
    )
    return (
        {
            **common,
            "status": "FROZEN",
            "selected_layer": int(selected["layer"]),
            "selected_layer_summary": selected,
            "peak_median_seed_mean_effect": peak_effect,
            "plateau_threshold": threshold,
            "negative_result_retained": False,
        },
        cells,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--max-causal-layer", type=int)
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
    scopes = []
    cells = []
    for directory in scope_directories:
        scope, active_cells = _scope_analysis(
            directory,
            baselines,
            max_causal_layer=args.max_causal_layer,
        )
        scopes.append(scope)
        cells.extend(active_cells)
    payload = {
        "schema_version": "realistic_niah_v6_natural_patch_scope_analysis_v1",
        "source_kernel_schema": "natural_patch_scope_layer_sweep_analysis_v1",
        "status": (
            "DISCOVERY_FROZEN"
            if all(scope["status"] == "FROZEN" for scope in scopes)
            else "DISCOVERY_NEGATIVE_RETAINED"
        ),
        "root": str(args.root.resolve()),
        "scope_count": len(scopes),
        "scopes": scopes,
        "cells": cells,
        "selection_split": "discovery",
        "confirmation_outcomes_read": False,
        "negative_results_retained": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selections": {
                    scope["scope"]: scope["selected_layer"] for scope in scopes
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
