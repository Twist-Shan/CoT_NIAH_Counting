#!/usr/bin/env python3
"""Merge the final Qwen major-grammar localizers into the frozen hybrid dose grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


GRAMMAR_DIRS = {
    "adjacent_rank_after_city": "adj_citypre_ovnorm",
    "same_unit_rank_before_city": "same_citypre_abs",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_shard(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        rows = [json.loads(text)]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError(f"Expected one row in behavior shard {path}, found {len(rows)}")
    return rows[0]


def failure(row: dict[str, Any]) -> int:
    return int(row.get("behavior_outcome") != "correct_next_needle")


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denom
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_seed_effect(
    anchors: list[dict[str, Any]], *, draws: int = 10000, rng_seed: int = 20260821
) -> tuple[float, float, float]:
    by_seed: dict[int, list[float]] = defaultdict(list)
    for anchor in anchors:
        by_seed[int(anchor["seed"])].append(float(anchor["effect"]))
    seeds = sorted(by_seed)
    if not seeds:
        return float("nan"), float("nan"), float("nan")
    observed = sum(sum(by_seed[s]) for s in seeds) / sum(len(by_seed[s]) for s in seeds)
    rng = random.Random(rng_seed)
    samples: list[float] = []
    for _ in range(draws):
        picked = [rng.choice(seeds) for _ in seeds]
        values = [value for seed in picked for value in by_seed[seed]]
        samples.append(sum(values) / len(values))
    samples.sort()
    lo = samples[int(0.025 * (draws - 1))]
    hi = samples[int(0.975 * (draws - 1))]
    return observed, lo, hi


def analyze_cell(root: Path, grammar: str, bank_size: int) -> dict[str, Any]:
    label = GRAMMAR_DIRS[grammar]
    cell = root / "behavior" / label / f"k{bank_size}"
    manifest = read_json(cell / "manifest.json")
    scheduled = int(manifest["scheduled_anchor_condition_trials"])
    completed = int(manifest["completed_shards"])
    if scheduled != completed:
        raise ValueError(f"Incomplete cell {cell}: {completed}/{scheduled}")
    rows = [read_shard(path) for path in sorted((cell / "shards").glob("trial_*.jsonl"))]
    if len(rows) != completed:
        raise ValueError(f"Shard count mismatch in {cell}: {len(rows)} != {completed}")
    conditions = sorted({str(row["condition"]) for row in rows})
    random_conditions = [condition for condition in conditions if condition != "selected_bank"]
    if len(random_conditions) != 1:
        raise ValueError(f"Expected one random control family in {cell}: {conditions}")
    random_condition = random_conditions[0]

    anchors: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["request_id"]), int(row["from_occurrence"]), int(row["to_occurrence"]))
        anchor = anchors.setdefault(
            key,
            {
                "seed": int(row["seed"]),
                "selected": [],
                "random": [],
            },
        )
        bucket = "selected" if row["condition"] == "selected_bank" else "random"
        anchor[bucket].append(failure(row))
    paired: list[dict[str, Any]] = []
    for key, anchor in anchors.items():
        if len(anchor["selected"]) != 1 or len(anchor["random"]) != 3:
            raise ValueError(f"Unpaired anchor {key} in {cell}: {anchor}")
        selected_failure = anchor["selected"][0]
        random_mean = sum(anchor["random"]) / 3.0
        paired.append(
            {
                "seed": anchor["seed"],
                "selected_failure": selected_failure,
                "random_failure_mean": random_mean,
                "effect": selected_failure - random_mean,
            }
        )
    selected_failures = sum(int(row["selected_failure"]) for row in paired)
    random_failures = sum(sum(anchor["random"]) for anchor in anchors.values())
    n_anchors = len(paired)
    effect, effect_lo, effect_hi = bootstrap_seed_effect(paired)
    selected_rate = selected_failures / n_anchors
    random_rate = random_failures / (3 * n_anchors)
    selected_ci = wilson(selected_failures, n_anchors)
    random_ci = wilson(random_failures, 3 * n_anchors)
    return {
        "grammar": grammar,
        "bank_size": bank_size,
        "confirmation_anchors": n_anchors,
        "selected_failures": selected_failures,
        "selected_trials": n_anchors,
        "selected_failure_rate": selected_rate,
        "selected_wilson_lo": selected_ci[0],
        "selected_wilson_hi": selected_ci[1],
        "random_condition": random_condition,
        "random_failures": random_failures,
        "random_trials": 3 * n_anchors,
        "random_failure_rate": random_rate,
        "random_wilson_lo": random_ci[0],
        "random_wilson_hi": random_ci[1],
        "selected_minus_random": selected_rate - random_rate,
        "seed_bootstrap_effect": effect,
        "seed_bootstrap_lo": effect_lo,
        "seed_bootstrap_hi": effect_hi,
        "bootstrap_draws": 10000,
        "significant_positive_seed_bootstrap": effect_lo > 0.0,
        "selection_anchor_role": "city_pre_d1",
        "intervention_start_anchor_role": "p0_item_end",
        "persistent_ablation": True,
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("reports/v5_native_final_localizers/Qwen3-8B"),
    )
    parser.add_argument(
        "--prior-hybrid",
        type=Path,
        default=Path(
            "reports/v5_native_hybrid_supplement/Qwen3-8B/"
            "analysis_hybrid_supplement_registered_v1/hybrid_dose_grid_complete.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reports/v5_native_final_localizers/analysis"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    completion_path = args.run_root / "qwen_major_grammar_final_dose_complete.json"
    completion = read_json(completion_path)
    if completion.get("status") != "PASS":
        raise ValueError("Final localizer completion is not PASS")
    doses = [int(value) for value in completion["doses"]]
    major_rows = [
        analyze_cell(args.run_root, grammar, bank_size)
        for bank_size in doses
        for grammar in GRAMMAR_DIRS
    ]
    prior = read_json(args.prior_hybrid)
    merged_rows = [dict(row) for row in prior["rows"]]
    replacements = {(row["grammar"], row["bank_size"]): row for row in major_rows}
    for row in merged_rows:
        key = (str(row["grammar"]), int(row["bank_size"]))
        if key not in replacements:
            continue
        replacement = replacements[key]
        row.update(
            {
                "confirmation_anchors": replacement["confirmation_anchors"],
                "selected_failures": replacement["selected_failures"],
                "random_failures": replacement["random_failures"],
                "random_condition": replacement["random_condition"],
                "selection_anchor_role": "city_pre_d1",
                "provenance": "final_city_pre_localizer_p0_persistent_ablation",
            }
        )
    overall: list[dict[str, Any]] = []
    for bank_size in doses:
        for scope in ("all_registered_grammars", "non_exploratory_grammars"):
            subset = [row for row in merged_rows if int(row["bank_size"]) == bank_size]
            if scope == "non_exploratory_grammars":
                subset = [row for row in subset if not bool(row["exploratory"])]
            anchors = sum(int(row["confirmation_anchors"]) for row in subset)
            selected = sum(int(row["selected_failures"]) for row in subset)
            random_failures = sum(int(row["random_failures"]) for row in subset)
            selected_rate = selected / anchors
            random_rate = random_failures / (3 * anchors)
            selected_ci = wilson(selected, anchors)
            random_ci = wilson(random_failures, 3 * anchors)
            overall.append(
                {
                    "bank_size": bank_size,
                    "scope": scope,
                    "confirmation_anchors": anchors,
                    "selected_failures": selected,
                    "selected_trials": anchors,
                    "selected_failure_rate": selected_rate,
                    "selected_wilson_lo": selected_ci[0],
                    "selected_wilson_hi": selected_ci[1],
                    "random_failures": random_failures,
                    "random_trials": 3 * anchors,
                    "random_failure_rate": random_rate,
                    "random_wilson_lo": random_ci[0],
                    "random_wilson_hi": random_ci[1],
                    "selected_minus_random_failure_rate": selected_rate - random_rate,
                    "conservative_unpaired_difference_lo": selected_ci[0] - random_ci[1],
                    "conservative_unpaired_difference_hi": selected_ci[1] - random_ci[0],
                }
            )
    output = {
        "schema_version": "realistic_niah_v5_native_final_localizers_v1",
        "status": "PASS",
        "model_label": "Qwen3-8B",
        "doses": doses,
        "selection": completion["selection"],
        "intervention_start_anchor_role": completion["intervention_start"],
        "persistent_ablation": completion["decode_head_ablation_steps"] == -1,
        "major_grammar_rows": major_rows,
        "rows": merged_rows,
        "overall": overall,
        "source_sha256": {
            str(completion_path): sha256(completion_path),
            str(args.prior_hybrid): sha256(args.prior_hybrid),
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output_root / "qwen_final_merged_dose_grid.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_root / "qwen_final_major_grammar_dose.csv", major_rows)
    write_csv(args.output_root / "qwen_final_merged_grammar_rows.csv", merged_rows)
    write_csv(args.output_root / "qwen_final_overall_dose.csv", overall)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output_path),
                "sha256": sha256(output_path),
                "primary": next(
                    row
                    for row in overall
                    if row["bank_size"] == max(doses)
                    and row["scope"] == "all_registered_grammars"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
