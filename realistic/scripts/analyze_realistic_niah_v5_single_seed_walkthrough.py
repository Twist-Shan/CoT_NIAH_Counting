#!/usr/bin/env python3
"""Summarize a one-seed counter-restoration trajectory without inference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


RESTORE_CONDITIONS = (
    "full_item_restore",
    "counter_carrier_restore",
    "counter_carrier_matched_control",
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError("Cannot write an empty walkthrough table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    temporary.replace(path)


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    return 0.0 if left_scale == 0.0 or right_scale == 0.0 else numerator / (
        left_scale * right_scale
    )


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        raise ValueError("Walkthrough rows are empty")
    if {str(row["condition"]) for row in rows} != {
        "clean",
        "uninformative",
        *RESTORE_CONDITIONS,
    }:
        raise ValueError("Walkthrough conditions changed")
    singleton = {
        key: {row[key] for row in rows}
        for key in ("model_label", "seed", "gold_count", "request_id", "source_layer")
    }
    if any(len(values) != 1 for values in singleton.values()):
        raise ValueError("Walkthrough rows do not describe one frozen case")
    count = int(next(iter(singleton["gold_count"])))
    if len(rows) != 2 + 3 * count:
        raise ValueError("Walkthrough row count changed")
    if any(bool(row.get("answer_query_patched")) for row in rows):
        raise ValueError("Walkthrough patched the final answer query")
    if any(bool(row.get("case_selected_by_outcome")) for row in rows):
        raise ValueError("Walkthrough case selection accessed outcomes")
    if any(not bool(row.get("control_sequence_length_equal")) for row in rows):
        raise ValueError("Walkthrough control changed sequence length")

    table_rows: list[dict[str, Any]] = []
    condition_summaries: dict[str, Any] = {}
    for condition in RESTORE_CONDITIONS:
        active = sorted(
            (row for row in rows if str(row["condition"]) == condition),
            key=lambda row: int(row["restored_occurrence"]),
        )
        occurrences = [int(row["restored_occurrence"]) for row in active]
        if occurrences != list(range(1, count + 1)):
            raise ValueError(f"{condition} does not cover occurrence 1..N")
        candidate = [int(row["predicted_count_among_candidates"]) for row in active]
        expected = [float(row["expected_count"]) for row in active]
        greedy = [row.get("prediction") for row in active]
        target_probabilities = [
            float(row["restored_target_count_probability"]) for row in active
        ]
        target_margins = [float(row["restored_target_count_margin"]) for row in active]
        condition_summaries[condition] = {
            "occurrence_count": count,
            "candidate_exact_path_count": sum(
                observed == target for observed, target in zip(candidate, occurrences)
            ),
            "candidate_exact_path_fraction": sum(
                observed == target for observed, target in zip(candidate, occurrences)
            )
            / count,
            "candidate_mean_absolute_error_to_restored_occurrence": sum(
                abs(observed - target)
                for observed, target in zip(candidate, occurrences)
            )
            / count,
            "greedy_exact_path_count": sum(
                observed == target for observed, target in zip(greedy, occurrences)
            ),
            "greedy_exact_path_fraction": sum(
                observed == target for observed, target in zip(greedy, occurrences)
            )
            / count,
            "candidate_prediction_correlation_with_occurrence": _correlation(
                [float(value) for value in occurrences],
                [float(value) for value in candidate],
            ),
            "expected_count_correlation_with_occurrence": _correlation(
                [float(value) for value in occurrences], expected
            ),
            "mean_restored_target_probability": sum(target_probabilities) / count,
            "mean_restored_target_margin": sum(target_margins) / count,
            "candidate_path": candidate,
            "greedy_path": greedy,
            "expected_count_path": expected,
        }
        for row in active:
            table_rows.append(
                {
                    "model_label": row["model_label"],
                    "seed": int(row["seed"]),
                    "gold_count": int(row["gold_count"]),
                    "condition": condition,
                    "occurrence": int(row["restored_occurrence"]),
                    "carrier_component": row["counter_carrier_component"],
                    "patch_token_count": int(row["patch_token_count"]),
                    "candidate_prediction": int(
                        row["predicted_count_among_candidates"]
                    ),
                    "expected_count": float(row["expected_count"]),
                    "greedy_prediction": row.get("prediction"),
                    "restored_target_probability": float(
                        row["restored_target_count_probability"]
                    ),
                    "restored_target_margin": float(
                        row["restored_target_count_margin"]
                    ),
                }
            )

    baselines = {
        str(row["condition"]): {
            "candidate_prediction": int(row["predicted_count_among_candidates"]),
            "expected_count": float(row["expected_count"]),
            "greedy_prediction": row.get("prediction"),
            "gold_count_probability": float(row["correct_count_probability"]),
            "gold_count_margin": float(row["correct_count_margin"]),
        }
        for row in rows
        if str(row["condition"]) in {"clean", "uninformative"}
    }
    summary = {
        "schema_version": "realistic_niah_v5_single_seed_walkthrough_complete_v1",
        "status": "PASS",
        "model_label": next(iter(singleton["model_label"])),
        "request_id": next(iter(singleton["request_id"])),
        "seed": int(next(iter(singleton["seed"]))),
        "gold_count": count,
        "source_layer": int(next(iter(singleton["source_layer"]))),
        "case_study_not_inferential": True,
        "case_selected_by_outcome": False,
        "answer_query_patched": False,
        "baselines": baselines,
        "conditions": condition_summaries,
        "interpretation_rule": (
            "Use the paths as an illustrative sufficiency check.  Do not attach "
            "confidence intervals, p-values, discovery selection, or population claims."
        ),
    }
    return table_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    table_rows, summary = summarize(rows)
    _atomic_csv(args.output / "walkthrough_table.csv", table_rows)
    _atomic_json(args.output / "walkthrough_complete.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
