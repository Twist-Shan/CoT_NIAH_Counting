#!/usr/bin/env python3
"""Analyze paired donor-directed shifts across a natural-site layer scan."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sign_test_two_sided(values: list[float]) -> float | None:
    nonzero = [value for value in values if value != 0.0]
    n = len(nonzero)
    if n == 0:
        return None
    successes = sum(value > 0.0 for value in nonzero)
    tail = sum(math.comb(n, k) for k in range(0, min(successes, n - successes) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def analyze(directory: Path) -> dict[str, Any]:
    trials = _rows(directory / "trials.jsonl")
    by_cell: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in trials:
        by_cell[(int(row["seed"]), int(row["layer"]))][str(row["condition"])] = row
    layer_rows: list[dict[str, Any]] = []
    for layer in sorted({key[1] for key in by_cell}):
        cells = [value for (seed, active), value in by_cell.items() if active == layer]
        complete = [
            value
            for value in cells
            if {"receiver_self", "native_donor", "donor_to_receiver"} <= set(value)
        ]
        shifts = [
            float(cell["donor_to_receiver"]["donor_vs_receiver_sum_logodds"])
            - float(cell["receiver_self"]["donor_vs_receiver_sum_logodds"])
            for cell in complete
        ]
        layer_rows.append(
            {
                "layer": layer,
                "seed_count": len(complete),
                "mean_paired_donor_shift": mean(shifts),
                "median_paired_donor_shift": median(shifts),
                "positive_shift_rate": mean(value > 0.0 for value in shifts),
                "two_sided_sign_test_p": _sign_test_two_sided(shifts),
                "donor_to_receiver_argmax_rate": mean(
                    bool(cell["donor_to_receiver"]["donor_successor_argmax"])
                    for cell in complete
                ),
                "receiver_self_argmax_rate": mean(
                    bool(cell["receiver_self"]["receiver_successor_argmax"])
                    for cell in complete
                ),
                "native_donor_argmax_rate": mean(
                    bool(cell["native_donor"]["donor_successor_argmax"])
                    for cell in complete
                ),
                "paired_donor_shifts": shifts,
            }
        )
    ranked = sorted(
        layer_rows,
        key=lambda row: (
            float(row["positive_shift_rate"]),
            float(row["mean_paired_donor_shift"]),
        ),
        reverse=True,
    )
    return {
        "directory": str(directory),
        "layer_rows": layer_rows,
        "ranked_layers": ranked,
        "best_layer": ranked[0] if ranked else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = [analyze(directory) for directory in args.directories]
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            [
                {
                    "directory": row["directory"],
                    "top_five": row["ranked_layers"][:5],
                }
                for row in payload
            ],
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
