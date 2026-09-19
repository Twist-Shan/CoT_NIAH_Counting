#!/usr/bin/env python3
"""Prepare report-facing raw arms for the post-hoc Gemma Top-6 dose."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_SEEDS = tuple(range(1316, 1336))
EXPECTED_COUNTS = tuple(range(1, 6))
BOOTSTRAP_REPETITIONS = 10_000
ELIGIBLE_HEADS = 56


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Not a registered boolean: {value!r}")


def optional_numeric(value: object) -> float | None:
    text = str(value).strip()
    return float(text) if text else None


def stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")


def cluster_bootstrap(
    units: list[tuple[int, str, float]], *, label: str
) -> tuple[float, float, float]:
    if not units or not all(np.isfinite(value) for _, _, value in units):
        raise RuntimeError(f"No finite units for {label}")
    by_seed: dict[int, list[float]] = defaultdict(list)
    for seed, _stimulus_id, value in units:
        by_seed[seed].append(value)
    if tuple(sorted(by_seed)) != EXPECTED_SEEDS:
        raise RuntimeError(f"Expected 20 seed clusters for {label}")
    sums = np.asarray([sum(by_seed[seed]) for seed in EXPECTED_SEEDS], dtype=np.float64)
    counts = np.asarray([len(by_seed[seed]) for seed in EXPECTED_SEEDS], dtype=np.float64)
    rng = np.random.default_rng(stable_seed(label))
    indices = rng.integers(0, 20, size=(BOOTSTRAP_REPETITIONS, 20))
    distribution = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return float(sums.sum() / counts.sum()), float(low), float(high)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detail", type=Path, required=True)
    parser.add_argument("--formal-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_csv_gz(args.detail)
    if len(rows) != 400:
        raise RuntimeError(f"Expected 400 Top-6 rows, got {len(rows)}")
    if {row["model_label"] for row in rows} != {"Gemma4-E4B"}:
        raise RuntimeError("Top-6 detail contains an unexpected model")
    if {int(row["top_n"]) for row in rows} != {6}:
        raise RuntimeError("Top-6 detail contains an unexpected dose")
    if tuple(sorted({int(row["seed"]) for row in rows})) != EXPECTED_SEEDS:
        raise RuntimeError("Top-6 detail contains unexpected seeds")
    if tuple(sorted({int(row["gold_count"]) for row in rows})) != EXPECTED_COUNTS:
        raise RuntimeError("Top-6 detail contains unexpected counts")
    expected_heads = "L29H4,L35H2,L35H7,L35H1,L35H3,L29H2"
    ranked_rows = [row for row in rows if row["condition"] == "ranked"]
    random_rows = [row for row in rows if row["condition"] == "layer_matched_random"]
    if len(ranked_rows) != 100 or len(random_rows) != 300:
        raise RuntimeError("Top-6 ranked/random arm coverage is incomplete")
    if {row["heads"] for row in ranked_rows} != {expected_heads}:
        raise RuntimeError("Top-6 ranked membership drifted")

    metrics: dict[str, dict[str, list[tuple[int, str, float]]]] = {
        "absolute_count_shift": defaultdict(list),
        "clean_correct_to_wrong_rate": defaultdict(list),
    }
    row_counts: dict[tuple[str, str], int] = defaultdict(int)
    for seed in EXPECTED_SEEDS:
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        arms = {
            "ranked": [row for row in seed_rows if row["condition"] == "ranked"],
            "layer_matched_random": [
                row for row in seed_rows if row["condition"] == "layer_matched_random"
            ],
        }
        if len(arms["ranked"]) != 5 or len(arms["layer_matched_random"]) != 15:
            raise RuntimeError(f"Incomplete Top-6 arms for seed {seed}")
        clean_ids: dict[str, set[str]] = {}
        for condition, arm in arms.items():
            by_stimulus: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in arm:
                by_stimulus[row["stimulus_id"]].append(row)
            clean_ids[condition] = set()
            for stimulus_id, stimulus_rows in by_stimulus.items():
                shifts = [
                    value
                    for row in stimulus_rows
                    if (value := optional_numeric(row["generated_count_shift"]))
                    is not None
                ]
                if not shifts:
                    raise RuntimeError(
                        f"No valid count-shift replicate for {stimulus_id}"
                    )
                metrics["absolute_count_shift"][condition].append(
                    (seed, stimulus_id, float(np.mean(np.abs(shifts))))
                )
                reference = stimulus_rows[0]
                if as_bool(reference["baseline_is_correct"]) and as_bool(
                    reference["baseline_format_valid"]
                ):
                    clean_ids[condition].add(stimulus_id)
                    metrics["clean_correct_to_wrong_rate"][condition].append(
                        (
                            seed,
                            stimulus_id,
                            float(
                                np.mean(
                                    [
                                        not as_bool(row["patched_is_correct"])
                                        for row in stimulus_rows
                                    ]
                                )
                            ),
                        )
                    )
            row_counts[("absolute_count_shift", condition)] += len(arm)
            row_counts[("clean_correct_to_wrong_rate", condition)] += sum(
                len(by_stimulus[stimulus_id]) for stimulus_id in clean_ids[condition]
            )
        if clean_ids["ranked"] != clean_ids["layer_matched_random"]:
            raise RuntimeError(f"Clean-correct IDs do not match for seed {seed}")

    raw_rows: list[dict[str, Any]] = []
    for metric, arms in metrics.items():
        for condition in ("ranked", "layer_matched_random"):
            units = arms[condition]
            mean, low, high = cluster_bootstrap(
                units,
                label=f"Gemma4-E4B|K6|{metric}|{condition}|raw-arm",
            )
            raw_rows.append(
                {
                    "model_label": "Gemma4-E4B",
                    "top_n": 6,
                    "eligible_head_count": ELIGIBLE_HEADS,
                    "head_proportion": 6 / ELIGIBLE_HEADS,
                    "metric": metric,
                    "condition": condition,
                    "seed_clusters": 20,
                    "rows": row_counts[(metric, condition)],
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                }
            )

    formal_rows = read_csv(args.formal_summary)
    primary_rows: list[dict[str, Any]] = []
    populations = {
        "absolute_count_shift": "all_examples_signed",
        "clean_correct_to_wrong_rate": "clean_correct_only",
    }
    for metric, population in populations.items():
        formal = next(row for row in formal_rows if row["analysis_population"] == population)
        ranked = {(seed, stimulus_id): value for seed, stimulus_id, value in metrics[metric]["ranked"]}
        random = {(seed, stimulus_id): value for seed, stimulus_id, value in metrics[metric]["layer_matched_random"]}
        if ranked.keys() != random.keys():
            raise RuntimeError(f"Top-6 {metric} ranked/random units do not align")
        effect_units = [
            (seed, stimulus_id, ranked[(seed, stimulus_id)] - random[(seed, stimulus_id)])
            for seed, stimulus_id in ranked
        ]
        effect, _effect_low, _effect_high = cluster_bootstrap(
            effect_units,
            label=f"Gemma4-E4B|K6|{metric}|ranked-minus-random",
        )
        if abs(effect - float(formal["primary_effect"])) > 1e-12:
            raise RuntimeError(f"Top-6 {metric} contrast does not match formal summary")
        primary_rows.append(
            {
                "model_label": "Gemma4-E4B",
                "top_n": 6,
                "analysis_population": population,
                "primary_metric": formal["primary_metric"],
                "primary_effect": effect,
                "ci95_low": float(formal["primary_effect_ci95_low"]),
                "ci95_high": float(formal["primary_effect_ci95_high"]),
                "seed_clusters": 20,
                "primary_ci95_excludes_zero_positive": as_bool(
                    formal["primary_ci95_excludes_zero_positive"]
                ),
                "inference": "post_hoc_seed_cluster_bootstrap",
                "post_hoc_extension": True,
            }
        )

    write_csv(args.output_dir / "top6_raw_arms.csv", raw_rows)
    write_csv(args.output_dir / "top6_primary_statistics.csv", primary_rows)
    audit = {
        "schema_version": "realistic_niah_v4_4_top6_report_extension_v1",
        "status": "PASS",
        "rows": len(rows),
        "seeds": list(EXPECTED_SEEDS),
        "counts": list(EXPECTED_COUNTS),
        "ranked_rows": len(ranked_rows),
        "random_rows": len(random_rows),
        "global_frozen_top6": expected_heads.split(","),
        "raw_arm_rows": len(raw_rows),
        "primary_rows": len(primary_rows),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "primary_inference": "post_hoc_seed_cluster_bootstrap; no exact sign-flip claim",
    }
    (args.output_dir / "report_extension_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
