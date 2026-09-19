#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCHEMA = "realistic_niah_v5_answer_query_extension_analysis_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL {path}:{line_number}: {error}") from error
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def bootstrap_ci(values: np.ndarray, label: str, repetitions: int = 20_000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def sign_flip_p(values: np.ndarray, label: str) -> tuple[float, str, int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if not n:
        return np.nan, "not_estimable", 0
    observed = abs(float(values.mean()))
    if n <= 20:
        total = 1 << n
        extreme = 0
        bits = np.arange(n, dtype=np.uint64)
        for start in range(0, total, 65_536):
            stop = min(total, start + 65_536)
            masks = np.arange(start, stop, dtype=np.uint64)[:, None]
            signs = np.where(((masks >> bits) & 1) == 0, -1.0, 1.0)
            draws = np.abs((signs * values[None, :]).mean(axis=1))
            extreme += int(np.count_nonzero(draws >= observed - 1e-15))
        return float(extreme / total), "exact", total
    repetitions = 1_000_000
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    extreme = 0
    remaining = repetitions
    while remaining:
        size = min(25_000, remaining)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(size, n))
        draws = np.abs((signs * values[None, :]).mean(axis=1))
        extreme += int(np.count_nonzero(draws >= observed - 1e-15))
        remaining -= size
    return float((extreme + 1) / (repetitions + 1)), "monte_carlo", repetitions


def holm(values: Iterable[float]) -> list[float]:
    raw = np.asarray(list(values), dtype=float)
    adjusted = np.full(len(raw), np.nan)
    finite = np.flatnonzero(np.isfinite(raw))
    order = finite[np.argsort(raw[finite])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, raw[index] * (len(order) - rank)))
        adjusted[index] = running
    return adjusted.tolist()


def summarize_seed_effects(
    frame: pd.DataFrame,
    group_columns: list[str],
    value: str,
    *,
    holm_varying_columns: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(group_columns, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        values = group[value].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        label = ":".join(map(str, key))
        low, high = bootstrap_ci(values, label)
        pvalue, method, assignments = sign_flip_p(values, label)
        rows.append(
            {
                **dict(zip(group_columns, key)),
                "effect": float(np.mean(values)) if len(values) else np.nan,
                "ci95_low": low,
                "ci95_high": high,
                "sign_flip_p": pvalue,
                "sign_flip_method": method,
                "sign_flip_assignments": assignments,
                "seed_clusters": int(len(values)),
                "positive_seed_fraction": (
                    float(np.mean(values > 0)) if len(values) else np.nan
                ),
            }
        )
    result = pd.DataFrame(rows)
    if len(result):
        family = [
            column for column in group_columns if column not in holm_varying_columns
        ]
        result["holm_p"] = np.nan
        families = (
            result.groupby(family, dropna=False).groups.values()
            if family
            else [result.index]
        )
        for indices in families:
            result.loc[indices, "holm_p"] = holm(result.loc[indices, "sign_flip_p"])
    return result


def analyze_head_trials(path: Path, cohort_name: str, output: Path) -> dict[str, Path]:
    frame = pd.DataFrame(read_jsonl(path))
    if frame.empty:
        raise ValueError(f"No head trials in {path}")
    frame["baseline_exact_count"] = as_bool(frame["baseline_exact_count"])
    clean = frame.loc[frame["condition"].astype(str).eq("clean")]
    if clean["request_id"].duplicated().any():
        raise ValueError("Head trial clean rows are not request-unique")
    clean_lookup = clean.set_index("request_id")[
        ["target_sequence_log_probability", "target_first_token_probability"]
    ].rename(
        columns={
            "target_sequence_log_probability": "clean_logp",
            "target_first_token_probability": "clean_probability",
        }
    )
    intervention = frame.loc[~frame["condition"].astype(str).eq("clean")].copy()
    intervention = intervention.join(clean_lookup, on="request_id", validate="many_to_one")
    intervention["logp_damage"] = intervention["clean_logp"] - intervention["target_sequence_log_probability"]
    intervention["probability_damage"] = intervention["clean_probability"] - intervention["target_first_token_probability"]
    ranked = intervention.loc[intervention["condition"].astype(str).eq("answer_query_ranked")]
    random = intervention.loc[intervention["condition"].astype(str).eq("layer_matched_random")]
    random_mean = (
        random.groupby(["request_id", "bank_size"], as_index=False)
        .agg(
            random_logp_damage=("logp_damage", "mean"),
            random_probability_damage=("probability_damage", "mean"),
            random_target_needle_raw_mass=("target_needle_raw_mass", "mean"),
            random_target_needle_relative_mass=(
                "target_needle_relative_mass",
                "mean",
            ),
        )
    )
    paired = ranked.merge(random_mean, on=["request_id", "bank_size"], how="left", validate="one_to_one")
    paired["ranked_minus_random_logp_damage"] = paired["logp_damage"] - paired["random_logp_damage"]
    paired["ranked_minus_random_probability_damage"] = paired["probability_damage"] - paired["random_probability_damage"]
    paired["cohort_name"] = cohort_name
    seed_rows = []
    for population in ("all_one_to_one", "baseline_correct_only"):
        active = paired if population == "all_one_to_one" else paired.loc[paired["baseline_exact_count"]]
        for (model, bank_size, seed), group in active.groupby(["model_label", "bank_size", "seed"], sort=True):
            seed_rows.append(
                {
                    "cohort_name": cohort_name,
                    "model_label": model,
                    "analysis_population": population,
                    "bank_size": int(bank_size),
                    "seed": int(seed),
                    "requests": int(len(group)),
                    "ranked_target_needle_raw_mass": float(
                        group["target_needle_raw_mass"].mean()
                    ),
                    "ranked_target_needle_relative_mass": float(
                        group["target_needle_relative_mass"].mean()
                    ),
                    "random_target_needle_raw_mass": float(
                        group["random_target_needle_raw_mass"].mean()
                    ),
                    "random_target_needle_relative_mass": float(
                        group["random_target_needle_relative_mass"].mean()
                    ),
                    "ranked_minus_random_logp_damage": float(group["ranked_minus_random_logp_damage"].mean()),
                    "ranked_minus_random_probability_damage": float(group["ranked_minus_random_probability_damage"].mean()),
                }
            )
    seed_frame = pd.DataFrame(seed_rows)
    summaries = []
    for endpoint in ("ranked_minus_random_logp_damage", "ranked_minus_random_probability_damage"):
        current = summarize_seed_effects(
            seed_frame,
            ["cohort_name", "model_label", "analysis_population", "bank_size"],
            endpoint,
            holm_varying_columns=("bank_size",),
        )
        mass = (
            seed_frame.groupby(
                ["cohort_name", "model_label", "analysis_population", "bank_size"],
                as_index=False,
                dropna=False,
            )
            .agg(
                ranked_target_needle_raw_mass=(
                    "ranked_target_needle_raw_mass",
                    "mean",
                ),
                ranked_target_needle_relative_mass=(
                    "ranked_target_needle_relative_mass",
                    "mean",
                ),
                random_target_needle_raw_mass=(
                    "random_target_needle_raw_mass",
                    "mean",
                ),
                random_target_needle_relative_mass=(
                    "random_target_needle_relative_mass",
                    "mean",
                ),
            )
        )
        current = current.merge(
            mass,
            on=["cohort_name", "model_label", "analysis_population", "bank_size"],
            how="left",
            validate="one_to_one",
        )
        current["endpoint"] = endpoint
        summaries.append(current)
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "detail": output / f"head_ablation_{cohort_name}_paired_detail.csv",
        "seed": output / f"head_ablation_{cohort_name}_seed_effects.csv",
        "summary": output / f"head_ablation_{cohort_name}_statistics.csv",
    }
    paired.to_csv(paths["detail"], index=False)
    seed_frame.to_csv(paths["seed"], index=False)
    summary.to_csv(paths["summary"], index=False)
    return paths


def analyze_execution(path: Path, output: Path) -> dict[str, Path]:
    frame = pd.DataFrame(read_jsonl(path))
    if frame.empty:
        raise ValueError(f"No execution trials in {path}")
    frame["receiver_exact_count"] = as_bool(frame["receiver_exact_count"])
    frame["donor_exact_count"] = as_bool(frame["donor_exact_count"])
    rows = []
    for pair_id, group in frame.groupby("pair_id", sort=True):
        by_condition = {str(row.condition): row for row in group.itertuples(index=False)}
        required = {"self_patch", "full_donor_patch", "projected_donor_patch", "orthogonal_norm_matched"}
        if set(by_condition) != required:
            raise ValueError(f"Execution pair {pair_id} conditions={sorted(by_condition)}")
        self_row = by_condition["self_patch"]
        if pd.isna(self_row.prediction):
            continue
        for treatment in ("full_donor_patch", "projected_donor_patch"):
            treated = by_condition[treatment]
            control = by_condition["orthogonal_norm_matched"]
            if pd.isna(treated.prediction) or pd.isna(control.prediction):
                continue
            self_distance = abs(float(self_row.prediction) - float(self_row.donor_count))
            treated_gain = self_distance - abs(float(treated.prediction) - float(treated.donor_count))
            control_gain = self_distance - abs(float(control.prediction) - float(control.donor_count))
            rows.append(
                {
                    "pair_id": pair_id,
                    "model_label": treated.model_label,
                    "seed": int(treated.seed),
                    "receiver_count": int(treated.gold_count),
                    "donor_count": int(treated.donor_count),
                    "pair_direction": treated.pair_direction,
                    "treatment": treatment,
                    "receiver_exact_count": bool(treated.receiver_exact_count),
                    "donor_exact_count": bool(treated.donor_exact_count),
                    "self_prediction": float(self_row.prediction),
                    "treatment_prediction": float(treated.prediction),
                    "control_prediction": float(control.prediction),
                    "treatment_transport_gain": treated_gain,
                    "control_transport_gain": control_gain,
                    "treatment_minus_control_transport_gain": treated_gain - control_gain,
                    "treatment_minus_control_donor_adoption": float(treated.prediction == treated.donor_count) - float(control.prediction == control.donor_count),
                }
            )
    detail = pd.DataFrame(rows)
    seed_rows = []
    for population in ("all_one_to_one", "baseline_correct_only"):
        active = detail if population == "all_one_to_one" else detail.loc[detail["receiver_exact_count"] & detail["donor_exact_count"]]
        for (model, treatment, seed), group in active.groupby(["model_label", "treatment", "seed"], sort=True):
            seed_rows.append(
                {
                    "model_label": model,
                    "analysis_population": population,
                    "treatment": treatment,
                    "seed": int(seed),
                    "pairs": int(len(group)),
                    "treatment_minus_control_transport_gain": float(group["treatment_minus_control_transport_gain"].mean()),
                    "treatment_minus_control_donor_adoption": float(group["treatment_minus_control_donor_adoption"].mean()),
                }
            )
    seed_frame = pd.DataFrame(seed_rows)
    summaries = []
    for endpoint in ("treatment_minus_control_transport_gain", "treatment_minus_control_donor_adoption"):
        current = summarize_seed_effects(
            seed_frame,
            ["model_label", "analysis_population", "treatment"],
            endpoint,
            holm_varying_columns=("treatment",),
        )
        current["endpoint"] = endpoint
        summaries.append(current)
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "detail": output / "answer_execution_paired_detail.csv",
        "seed": output / "answer_execution_seed_effects.csv",
        "summary": output / "answer_execution_statistics.csv",
    }
    detail.to_csv(paths["detail"], index=False)
    seed_frame.to_csv(paths["seed"], index=False)
    summary.to_csv(paths["summary"], index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-trials-primary", type=Path, required=True)
    parser.add_argument("--head-trials-supplement", type=Path, required=True)
    parser.add_argument("--execution-trials", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    primary = analyze_head_trials(args.head_trials_primary, "primary_confirmation", args.output_dir)
    supplement = analyze_head_trials(args.head_trials_supplement, "supplement_n10_confirmation", args.output_dir)
    execution = analyze_execution(args.execution_trials, args.output_dir)
    inputs = {
        "head_trials_primary": args.head_trials_primary,
        "head_trials_supplement": args.head_trials_supplement,
        "execution_trials": args.execution_trials,
    }
    audit = {
        "schema_version": SCHEMA,
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in inputs.items()
        },
        "outputs": {
            section: {name: str(path.resolve()) for name, path in paths.items()}
            for section, paths in {
                "head_primary": primary,
                "head_supplement": supplement,
                "execution": execution,
            }.items()
        },
        "inference_unit": "seed-cluster mean",
        "selection_policy": "heads and execution layer frozen on discovery only",
        "confirmation_used_for_selection": False,
        "ov_comparison_status": "not_run_by_user_request",
    }
    audit_path = args.output_dir / "answer_query_extension_analysis_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
