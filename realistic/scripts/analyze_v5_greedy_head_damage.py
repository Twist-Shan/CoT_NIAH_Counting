#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCHEMA = "realistic_niah_v5_greedy_head_damage_analysis_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_sign_flip_p(values: Iterable[float]) -> tuple[float, str, int]:
    vector = np.asarray(list(values), dtype=np.float64)
    vector = vector[np.isfinite(vector)]
    if not len(vector):
        raise ValueError("Sign-flip test needs finite seed effects")
    observed = abs(float(vector.mean()))
    if len(vector) <= 20:
        total = 1 << len(vector)
        extreme = 0
        bit_positions = np.arange(len(vector), dtype=np.uint64)
        for start in range(0, total, 65_536):
            stop = min(total, start + 65_536)
            masks = np.arange(start, stop, dtype=np.uint64)[:, None]
            signs = np.where(((masks >> bit_positions) & 1) == 0, -1.0, 1.0)
            draws = np.abs((signs * vector[None, :]).mean(axis=1))
            extreme += int(np.count_nonzero(draws >= observed - 1e-15))
        return float(extreme / total), "exact_enumeration", int(total)
    repetitions = 1_000_000
    seed = int.from_bytes(
        hashlib.sha256(vector.tobytes()).digest()[:8], "little"
    )
    rng = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    while completed < repetitions:
        active = min(10_000, repetitions - completed)
        signs = rng.choice((-1.0, 1.0), size=(active, len(vector)))
        draws = np.abs((signs * vector[None, :]).mean(axis=1))
        extreme += int(np.count_nonzero(draws >= observed - 1e-15))
        completed += active
    return (
        float((extreme + 1) / (repetitions + 1)),
        "deterministic_monte_carlo",
        repetitions,
    )


def bootstrap_ci(values: Iterable[float], *, label: str) -> tuple[float, float]:
    vector = np.asarray(list(values), dtype=np.float64)
    vector = vector[np.isfinite(vector)]
    if not len(vector):
        raise ValueError("Bootstrap needs finite seed effects")
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(vector), size=(10_000, len(vector)))
    distribution = vector[indices].mean(axis=1)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return float(low), float(high)


def optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return int(value)


def request_id(row: dict[str, Any]) -> str:
    return str(row.get("request_id", row.get("stimulus_id")))


def row_family(row: dict[str, Any]) -> str:
    mechanism = str(row.get("mechanism", ""))
    variant = str(row.get("query_variant", ""))
    if variant:
        return f"{mechanism or 'targeted_retrieval'}::{variant}"
    if mechanism:
        return mechanism
    raise ValueError(f"Intervention row has no mechanism/variant: {request_id(row)}")


def condition_kind(row: dict[str, Any]) -> str:
    condition = str(row.get("condition", ""))
    if condition == "clean":
        return "clean"
    if "layer_matched_random" in condition:
        return "random"
    if condition.endswith("_ranked"):
        return "ranked"
    raise ValueError(f"Unknown damage condition: {condition}")


def valid_shift(intervened: int | None, clean: int | None) -> float:
    return float(abs(intervened - clean)) if intervened is not None and clean is not None else 0.0


def mae_delta(row: dict[str, Any], clean: dict[str, Any]) -> float:
    active = row.get("absolute_error")
    baseline = clean.get("absolute_error")
    if active is None or baseline is None:
        return 0.0
    return float(active) - float(baseline)


def analyze(paths: list[Path], output_dir: Path) -> dict[str, Any]:
    inputs = []
    rows: list[dict[str, Any]] = []
    for path in paths:
        inputs.append({"path": str(path.resolve()), "sha256": sha256(path)})
        for row in read_jsonl(path):
            row = dict(row)
            row["source_path"] = str(path.resolve())
            rows.append(row)
    if not rows:
        raise ValueError("No greedy head-damage rows")
    statuses = pd.Series([str(row.get("status", "ok")) for row in rows]).value_counts()
    ok_rows = [row for row in rows if str(row.get("status", "ok")) == "ok"]
    clean_rows = [row for row in ok_rows if condition_kind(row) == "clean"]
    clean_lookup: dict[str, dict[str, Any]] = {}
    for row in clean_rows:
        key = request_id(row)
        if key in clean_lookup:
            prior = clean_lookup[key]
            fields = ("prediction", "exact_count", "gold_count", "seed")
            if any(prior.get(field) != row.get(field) for field in fields):
                raise ValueError(f"Conflicting clean rows for {key}")
        else:
            clean_lookup[key] = row
    interventions = [row for row in ok_rows if condition_kind(row) != "clean"]
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in interventions:
        key = (row_family(row), int(row.get("bank_size", 0)), request_id(row))
        grouped.setdefault(key, []).append(row)
    paired_rows: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []
    for (family, bank_size, key), frame in sorted(grouped.items()):
        ranked = [row for row in frame if condition_kind(row) == "ranked"]
        random = [row for row in frame if condition_kind(row) == "random"]
        if len(ranked) != 1:
            raise ValueError(f"Expected one ranked row for {family}/K{bank_size}/{key}")
        clean = clean_lookup.get(key)
        if clean is None:
            raise ValueError(f"Missing clean row for {key}")
        treatment = ranked[0]
        base = {
            "model_label": str(treatment.get("model_label")),
            "family": family,
            "bank_size": int(bank_size),
            "request_id": key,
            "seed": int(treatment["seed"]),
            "gold_count": int(treatment["gold_count"]),
            "clean_prediction": optional_int(clean.get("prediction")),
            "clean_exact_count": bool(clean.get("exact_count")),
            "ranked_prediction": optional_int(treatment.get("prediction")),
            "ranked_exact_count": bool(treatment.get("exact_count")),
            "ranked_valid": optional_int(treatment.get("prediction")) is not None,
            "random_replicates": len(random),
            "ranked_heads": json.dumps(treatment.get("heads", []), sort_keys=True),
        }
        if not random:
            unmatched_rows.append({**base, "reason": "no_constructible_layer_matched_control"})
            continue
        if len(random) != 3:
            raise ValueError(
                f"Expected exactly three controls for {family}/K{bank_size}/{key}; "
                f"got {len(random)}"
            )
        clean_prediction = optional_int(clean.get("prediction"))
        ranked_prediction = optional_int(treatment.get("prediction"))
        random_predictions = [optional_int(row.get("prediction")) for row in random]
        ranked_shift = valid_shift(ranked_prediction, clean_prediction)
        random_shifts = [valid_shift(value, clean_prediction) for value in random_predictions]
        ranked_failure = float(not bool(treatment.get("exact_count")))
        random_failures = [float(not bool(row.get("exact_count"))) for row in random]
        ranked_mae = mae_delta(treatment, clean)
        random_maes = [mae_delta(row, clean) for row in random]
        paired_rows.append(
            {
                **base,
                "random_predictions": json.dumps(random_predictions),
                "random_valid_rate": float(np.mean([value is not None for value in random_predictions])),
                "random_exact_rate": float(
                    np.mean([bool(row.get("exact_count")) for row in random])
                ),
                "absolute_count_shift_specificity": ranked_shift - float(np.mean(random_shifts)),
                "correct_to_wrong_specificity": ranked_failure - float(np.mean(random_failures)),
                "mae_specificity": ranked_mae - float(np.mean(random_maes)),
            }
        )
    paired = pd.DataFrame(paired_rows)
    unmatched = pd.DataFrame(unmatched_rows)
    if paired.empty:
        raise ValueError("No ranked treatment had three matched controls")
    seed_rows: list[dict[str, Any]] = []
    for (model, family, bank_size, seed), frame in paired.groupby(
        ["model_label", "family", "bank_size", "seed"], sort=True
    ):
        clean_correct = frame[frame["clean_exact_count"].astype(bool)]
        seed_rows.append(
            {
                "model_label": model,
                "family": family,
                "bank_size": int(bank_size),
                "seed": int(seed),
                "all_examples": int(len(frame)),
                "clean_correct_examples": int(len(clean_correct)),
                "ranked_valid_rate": float(frame["ranked_valid"].mean()),
                "random_valid_rate": float(frame["random_valid_rate"].mean()),
                "all_absolute_shift": float(frame["absolute_count_shift_specificity"].mean()),
                "absolute_error": float(frame["mae_specificity"].mean()),
                "clean_correct_to_wrong": (
                    float(clean_correct["correct_to_wrong_specificity"].mean())
                    if len(clean_correct)
                    else np.nan
                ),
            }
        )
    seed_effects = pd.DataFrame(seed_rows)
    metric_specs = {
        "all_absolute_shift": ("positive", True),
        "clean_correct_to_wrong": ("positive", True),
        "absolute_error": ("positive", False),
    }
    statistic_rows: list[dict[str, Any]] = []
    for (model, family, bank_size), frame in seed_effects.groupby(
        ["model_label", "family", "bank_size"], sort=True
    ):
        for metric, (harmful, primary) in metric_specs.items():
            values = pd.to_numeric(frame[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            low, high = bootstrap_ci(
                values,
                label=f"v5:{model}:{family}:K{bank_size}:{metric}",
            )
            p_value, method, assignments = exact_sign_flip_p(values)
            statistic_rows.append(
                {
                    "model_label": model,
                    "family": family,
                    "bank_size": int(bank_size),
                    "metric": metric,
                    "is_primary_endpoint": bool(primary),
                    "harmful_direction": harmful,
                    "seed_clusters": int(len(values)),
                    "effect": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "sign_flip_p": p_value,
                    "sign_flip_method": method,
                    "sign_flip_assignments": assignments,
                    "positive_seed_fraction": float(np.mean(values > 0)),
                    "nonnegative_seed_fraction": float(np.mean(values >= 0)),
                    "bootstrap_repetitions": 10_000,
                }
            )
    statistics = pd.DataFrame(statistic_rows)
    statistics["holm_p_within_model"] = np.nan
    for _model, indices in statistics.groupby("model_label").groups.items():
        ordered = sorted(
            indices, key=lambda index: statistics.loc[index, "sign_flip_p"]
        )
        running = 0.0
        for rank, index in enumerate(ordered):
            running = max(
                running,
                min(
                    1.0,
                    (len(ordered) - rank)
                    * float(statistics.loc[index, "sign_flip_p"]),
                ),
            )
            statistics.loc[index, "holm_p_within_model"] = running

    answer_families = (
        "answer_prompt_aggregation",
        "answer_trace_aggregation",
        "answer_prompt_and_trace_aggregation",
    )
    answer_frames = []
    answer = paired.loc[paired["family"].isin(answer_families)].copy()
    common = [
        "model_label",
        "bank_size",
        "request_id",
        "seed",
        "gold_count",
        "clean_prediction",
        "clean_exact_count",
    ]
    for family in answer_families:
        active = answer.loc[answer["family"].eq(family)].copy()
        if active.empty:
            continue
        suffix = {
            "answer_prompt_aggregation": "prompt_ablation",
            "answer_trace_aggregation": "trace_ablation",
            "answer_prompt_and_trace_aggregation": "joint_ablation",
        }[family]
        active = active[
            common
            + [
                "ranked_prediction",
                "ranked_exact_count",
                "random_exact_rate",
                "correct_to_wrong_specificity",
                "absolute_count_shift_specificity",
            ]
        ].rename(
            columns={
                "ranked_prediction": f"{suffix}_prediction",
                "ranked_exact_count": f"{suffix}_exact_count",
                "random_exact_rate": f"{suffix}_random_exact_rate",
                "correct_to_wrong_specificity": (
                    f"{suffix}_correct_to_wrong_specificity"
                ),
                "absolute_count_shift_specificity": (
                    f"{suffix}_absolute_count_shift_specificity"
                ),
            }
        )
        answer_frames.append(active)
    factorial = pd.DataFrame()
    if len(answer_frames) == len(answer_families):
        factorial = answer_frames[0]
        for active in answer_frames[1:]:
            factorial = factorial.merge(active, on=common, how="inner", validate="one_to_one")
    output_dir.mkdir(parents=True, exist_ok=True)
    paired_path = output_dir / "paired_request_effects.csv"
    seed_path = output_dir / "seed_effects.csv"
    stats_path = output_dir / "statistics.csv"
    unmatched_path = output_dir / "unmatched_ranked_treatments.csv"
    factorial_path = output_dir / "answer_aggregation_four_conditions.csv"
    paired.to_csv(paired_path, index=False)
    seed_effects.to_csv(seed_path, index=False)
    statistics.to_csv(stats_path, index=False)
    unmatched.to_csv(unmatched_path, index=False)
    factorial.to_csv(factorial_path, index=False)
    audit = {
        "schema_version": SCHEMA,
        "status": "passed",
        "inputs": inputs,
        "behavioral_endpoint": "strict_greedy_complete_numeric_generation",
        "non_thinking_standard": {
            "all_absolute_shift": "|y_ranked-y_clean|-mean_r|y_random_r-y_clean|",
            "clean_correct_to_wrong": "I[y_ranked!=gold]-mean_r I[y_random_r!=gold] conditional on clean correct",
            "absolute_error": "delta_MAE_ranked-mean_r delta_MAE_random",
            "independent_unit": "equal-weight seed mean",
            "uncertainty": "10000 seed-cluster bootstrap",
            "test": "two-sided sign flip of seed effects",
        },
        "rows": len(rows),
        "ok_rows": len(ok_rows),
        "status_counts": {str(key): int(value) for key, value in statuses.items()},
        "paired_requests": int(len(paired)),
        "unmatched_ranked_treatments": int(len(unmatched)),
        "families": sorted(paired["family"].astype(str).unique()),
        "answer_aggregation_four_condition_rows": int(len(factorial)),
        "answer_aggregation_conditions": [
            "clean",
            "prompt_aggregation_ablation",
            "trace_aggregation_ablation",
            "joint_prompt_and_trace_aggregation_ablation",
        ],
        "outputs": {
            "paired": str(paired_path.resolve()),
            "seed_effects": str(seed_path.resolve()),
            "statistics": str(stats_path.resolve()),
            "unmatched": str(unmatched_path.resolve()),
            "answer_aggregation_four_conditions": str(factorial_path.resolve()),
        },
    }
    audit_path = output_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.trials, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
