#!/usr/bin/env python3
"""Summarize held-out natural-anomaly boundary geometry.

The input JSONL files are produced by
``run_realistic_niah_v5_natural_anomaly_geometry.py``.  This script keeps
site-level and seed-level summaries separate and treats the continuous
ordinal coordinate as the primary geometry readout.  In particular, it
checks whether a boundary is closer to:

* the local item ordinal,
* the number of unique cities seen,
* the trace's final answer, or
* the prompt's gold total.

It also compares adjacent-boundary increments in clean traces with the
increments at naturally duplicated items.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


LABELS = (
    "raw_item_ordinal",
    "unique_city_count",
    "final_answer_count",
    "gold_total",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else math.nan


def population_sd(values: Iterable[float]) -> float:
    items = list(values)
    return statistics.pstdev(items) if len(items) > 1 else 0.0


def numeric_summary(values: Iterable[float]) -> dict[str, float | int]:
    items = list(values)
    if not items:
        return {"n": 0}
    return {
        "n": len(items),
        "mean": mean(items),
        "median": statistics.median(items),
        "sd": population_sd(items),
        "min": min(items),
        "max": max(items),
    }


def mae(rows: Iterable[dict[str, Any]], label: str) -> float:
    rows = list(rows)
    return mean(abs(float(row["probe_ordinal_coordinate"]) - float(row[label])) for row in rows)


def group_rows(rows: Iterable[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def adjacent_deltas(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    deltas: list[dict[str, Any]] = []
    for (request_id, layer), group in group_rows(rows, "request_id", "layer").items():
        ordered = sorted(group, key=lambda row: int(row["site_index"]))
        for previous, current in zip(ordered, ordered[1:]):
            if int(current["site_index"]) != int(previous["site_index"]) + 1:
                continue
            deltas.append(
                {
                    "request_id": request_id,
                    "seed": int(current["seed"]),
                    "layer": int(layer),
                    "site_index": int(current["site_index"]),
                    "is_duplicate_city": bool(current["is_duplicate_city"]),
                    "delta": float(current["probe_ordinal_coordinate"])
                    - float(previous["probe_ordinal_coordinate"]),
                }
            )
    return deltas


def pairwise_closeness(
    rows: Iterable[dict[str, Any]],
    left: str,
    right: str,
) -> dict[str, Any]:
    usable = [row for row in rows if float(row[left]) != float(row[right])]
    by_site = {"left": 0, "right": 0, "tie": 0}
    for row in usable:
        coordinate = float(row["probe_ordinal_coordinate"])
        left_error = abs(coordinate - float(row[left]))
        right_error = abs(coordinate - float(row[right]))
        if left_error < right_error:
            by_site["left"] += 1
        elif right_error < left_error:
            by_site["right"] += 1
        else:
            by_site["tie"] += 1

    seed_error: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"left": [], "right": []})
    for row in usable:
        seed = int(row["seed"])
        coordinate = float(row["probe_ordinal_coordinate"])
        seed_error[seed]["left"].append(abs(coordinate - float(row[left])))
        seed_error[seed]["right"].append(abs(coordinate - float(row[right])))
    by_seed = {"left": 0, "right": 0, "tie": 0}
    seed_differences: dict[str, float] = {}
    for seed, errors in sorted(seed_error.items()):
        difference = mean(errors["right"]) - mean(errors["left"])
        seed_differences[str(seed)] = difference
        if difference > 0:
            by_seed["left"] += 1
        elif difference < 0:
            by_seed["right"] += 1
        else:
            by_seed["tie"] += 1
    return {
        "n_sites": len(usable),
        "n_seeds": len(seed_error),
        "site_wins": by_site,
        "seed_wins": by_seed,
        "seed_right_minus_left_mae": seed_differences,
    }


def seed_macro_mae(rows: Iterable[dict[str, Any]], label: str) -> dict[str, Any]:
    by_seed = group_rows(rows, "seed")
    values = {
        str(seed[0]): mae(seed_rows, label)
        for seed, seed_rows in sorted(by_seed.items())
    }
    return {"mean": mean(values.values()), "per_seed": values}


def increment_summary(deltas: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["delta"]) for row in deltas]
    by_seed = group_rows(deltas, "seed")
    seed_means = {
        str(seed[0]): mean(float(row["delta"]) for row in seed_rows)
        for seed, seed_rows in sorted(by_seed.items())
    }
    summary = numeric_summary(values)
    summary.update(
        {
            "mae_to_zero": mean(abs(value) for value in values),
            "mae_to_one": mean(abs(value - 1.0) for value in values),
            "fraction_positive": mean(value > 0.0 for value in values),
            "fraction_near_zero_abs_lt_0.5": mean(abs(value) < 0.5 for value in values),
            "fraction_near_one_abs_error_lt_0.5": mean(abs(value - 1.0) < 0.5 for value in values),
            "seed_macro": numeric_summary(seed_means.values()),
            "seed_means": seed_means,
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anomaly-jsonl", type=Path, required=True)
    parser.add_argument("--clean-jsonl", type=Path, required=True)
    parser.add_argument("--audit-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    anomaly_rows = read_jsonl(args.anomaly_jsonl)
    clean_rows = read_jsonl(args.clean_jsonl)
    audit_rows = read_jsonl(args.audit_jsonl)
    audit_by_request = {row["request_id"]: row for row in audit_rows}

    selected_requests = {row["request_id"] for row in anomaly_rows}
    selected_audit = [audit_by_request[request_id] for request_id in selected_requests]
    anomaly_example_by_request = {
        row["request_id"]: row
        for row in anomaly_rows
    }
    pure_missing_requests = {
        row["request_id"]
        for row in selected_audit
        if int(row["item_count"]) < int(row["gold_count"])
        and int(row["duplicate_gold_city_items"]) == 0
        and int(row["item_count"])
        == int(anomaly_example_by_request[row["request_id"]]["final_answer_count"])
    }

    layers = sorted({int(row["layer"]) for row in anomaly_rows})
    clean_delta_rows = adjacent_deltas(clean_rows)
    anomaly_delta_rows = adjacent_deltas(anomaly_rows)

    result: dict[str, Any] = {
        "schema_version": "realistic_niah_v5_natural_anomaly_geometry_summary_v1",
        "inputs": {
            "anomaly_jsonl": str(args.anomaly_jsonl),
            "clean_jsonl": str(args.clean_jsonl),
            "audit_jsonl": str(args.audit_jsonl),
        },
        "counts": {
            "anomaly_requests": len(selected_requests),
            "anomaly_seeds": len({int(row["seed"]) for row in anomaly_rows}),
            "pure_missing_requests": len(pure_missing_requests),
            "pure_missing_request_ids": sorted(pure_missing_requests),
        },
        "layers": {},
    }

    for layer in layers:
        anomaly = [row for row in anomaly_rows if int(row["layer"]) == layer]
        clean = [row for row in clean_rows if int(row["layer"]) == layer]
        duplicate_sites = [row for row in anomaly if bool(row["is_duplicate_city"])]
        pure_missing = [row for row in anomaly if row["request_id"] in pure_missing_requests]
        final_missing = [row for row in pure_missing if bool(row["is_final_item"])]
        nonforward = [
            row
            for row in anomaly
            if row.get("gold_city_rank") is not None
            and int(row["raw_item_ordinal"]) != int(row["gold_city_rank"])
        ]
        clean_deltas = [row for row in clean_delta_rows if int(row["layer"]) == layer]
        duplicate_deltas = [
            row
            for row in anomaly_delta_rows
            if int(row["layer"]) == layer and bool(row["is_duplicate_city"])
        ]
        result["layers"][str(layer)] = {
            "all_anomaly_sites": {
                "n": len(anomaly),
                "mae": {label: mae(anomaly, label) for label in LABELS},
                "seed_macro_mae": {
                    label: seed_macro_mae(anomaly, label) for label in LABELS
                },
                "raw_vs_final": pairwise_closeness(anomaly, "raw_item_ordinal", "final_answer_count"),
                "raw_vs_gold": pairwise_closeness(anomaly, "raw_item_ordinal", "gold_total"),
            },
            "clean_adjacent_increment": increment_summary(clean_deltas),
            "duplicate_sites": {
                "n": len(duplicate_sites),
                "mae_raw": mae(duplicate_sites, "raw_item_ordinal"),
                "mae_unique": mae(duplicate_sites, "unique_city_count"),
                "raw_vs_unique": pairwise_closeness(
                    duplicate_sites, "raw_item_ordinal", "unique_city_count"
                ),
                "transition_increment": increment_summary(duplicate_deltas),
            },
            "pure_missing_sites": {
                "n": len(pure_missing),
                "mae_raw": mae(pure_missing, "raw_item_ordinal"),
                "mae_gold": mae(pure_missing, "gold_total"),
                "raw_vs_gold": pairwise_closeness(pure_missing, "raw_item_ordinal", "gold_total"),
                "final_boundaries": [
                    {
                        "seed": int(row["seed"]),
                        "raw_item_ordinal": int(row["raw_item_ordinal"]),
                        "gold_total": int(row["gold_total"]),
                        "coordinate": float(row["probe_ordinal_coordinate"]),
                    }
                    for row in sorted(final_missing, key=lambda row: int(row["seed"]))
                ],
            },
            "nonforward_sites": {
                "n": len(nonforward),
                "mae_raw": mae(nonforward, "raw_item_ordinal"),
                "mae_gold_city_rank": mae(nonforward, "gold_city_rank"),
                "raw_vs_gold_city_rank": pairwise_closeness(
                    nonforward, "raw_item_ordinal", "gold_city_rank"
                ),
            },
            "clean_panel": {
                "n": len(clean),
                "mae_raw": mae(clean, "raw_item_ordinal"),
                "rounded_exact": mean(
                    round(float(row["probe_ordinal_coordinate"])) == int(row["raw_item_ordinal"])
                    for row in clean
                ),
                "multiclass_exact": mean(
                    int(row["probe_prediction"]) == int(row["raw_item_ordinal"])
                    for row in clean
                ),
            },
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
