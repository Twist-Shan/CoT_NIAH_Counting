from __future__ import annotations

"""Analyze layerwise rank-3 removal against its norm-matched control."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CONDITIONS = ("actual_rank3_remove", "actual_normmatched_orthogonal")
ENDPOINTS = {
    "absolute_error_specificity": ("absolute_error", 1.0),
    "accuracy_damage_specificity": ("correct", -1.0),
}
DAMAGE_ENDPOINTS = {
    "candidate_absolute_error_damage": "candidate_absolute_error_damage_from_clean",
    "control_absolute_error_damage": "control_absolute_error_damage_from_clean",
}


def exact_signflip_p(values: Iterable[float]) -> float:
    vector = np.asarray(list(values), dtype=np.float64)
    vector = vector[np.isfinite(vector)]
    if not 1 <= len(vector) <= 20:
        raise ValueError("exact sign-flip requires 1..20 finite seed effects")
    observed = abs(float(vector.mean()))
    extreme = 0
    total = 1 << len(vector)
    for signs in itertools.product((-1.0, 1.0), repeat=len(vector)):
        draw = abs(float(np.mean(vector * np.asarray(signs))))
        extreme += int(draw >= observed - 1e-15)
    return float(extreme / total)


def bootstrap_ci(values: np.ndarray, *, label: str, draws: int) -> tuple[float, float]:
    stable = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
    rng = np.random.default_rng(stable)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    distribution = values[indices].mean(axis=1)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return float(low), float(high)


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


def resolve_inputs(paths: list[Path], detail_basename: str) -> list[Path]:
    resolved: list[Path] = []
    for path in paths:
        if path.is_dir():
            resolved.extend(sorted(path.rglob(detail_basename)))
        else:
            resolved.append(path)
    if not resolved:
        raise RuntimeError(f"no removal detail files named {detail_basename} found")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--design-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--support-role",
        choices=("prompt_running", "answer_query"),
        default="prompt_running",
    )
    parser.add_argument("--bootstraps", type=int, default=50_000)
    args = parser.parse_args()

    design = json.loads(args.design_config.read_text(encoding="utf-8"))
    design_key = (
        "prompt_removal"
        if args.support_role == "prompt_running"
        else "answer_query_removal"
    )
    prompt_design = design[design_key]
    realized_norm_tolerance = float(
        prompt_design["realized_norm_relative_tolerance"]
    )
    output_prefix = (
        "layerwise_prompt_removal"
        if args.support_role == "prompt_running"
        else "layerwise_answer_query_removal"
    )
    input_paths = resolve_inputs(args.inputs, f"{output_prefix}_detail.csv")
    detail = pd.concat([pd.read_csv(path) for path in input_paths], ignore_index=True)
    keys = ["model_label", "seed", "gold_count", "layer", "condition"]
    if detail.empty:
        raise RuntimeError("empty prompt-removal table")
    duplicate = detail.duplicated(keys, keep=False)
    if duplicate.any():
        conflicting = detail.loc[duplicate].drop_duplicates()
        if conflicting.duplicated(keys, keep=False).any():
            raise RuntimeError("conflicting duplicate prompt-removal rows")
        detail = detail.drop_duplicates(keys, keep="first")
    models = sorted(detail["model_label"].unique())
    seeds = [int(value) for value in design["confirmation_seeds"]]
    counts = [int(value) for value in prompt_design["counts"]]
    expected = {
        (model, seed, count, int(layer), condition)
        for model in models
        for seed in seeds
        for count in counts
        for layer in prompt_design["layers"][model]
        for condition in CONDITIONS
    }
    observed = {
        (str(row.model_label), int(row.seed), int(row.gold_count), int(row.layer), str(row.condition))
        for row in detail.itertuples()
    }
    if observed != expected:
        raise RuntimeError(
            f"output grid mismatch: missing={list(expected-observed)[:5]} "
            f"unexpected={list(observed-expected)[:5]}"
        )
    for column in (
        "correct",
        "absolute_error",
        "clean_correct",
        "clean_absolute_error",
        "target_removed_fro_norm",
        "removed_fro_norm",
        "norm_ratio",
        "normalized_depth",
    ):
        detail[column] = pd.to_numeric(detail[column], errors="raise")

    pivot = detail.pivot(index=keys[:-1], columns="condition")
    candidate = CONDITIONS[0]
    control = CONDITIONS[1]
    paired = pd.DataFrame(index=pivot.index).reset_index()
    for column in (
        "normalized_depth",
        "clean_correct",
        "clean_absolute_error",
    ):
        left = pivot[column][candidate].to_numpy(dtype=float)
        right = pivot[column][control].to_numpy(dtype=float)
        if not np.allclose(left, right, atol=1e-10, rtol=0):
            raise RuntimeError(f"paired conditions disagree on {column}")
        paired[column] = left
    candidate_target = pivot["target_removed_fro_norm"][candidate].to_numpy(float)
    control_target = pivot["target_removed_fro_norm"][control].to_numpy(float)
    target_relative_difference = np.abs(candidate_target - control_target) / np.maximum(
        candidate_target, 1e-12
    )
    max_target_norm_relative_difference = float(target_relative_difference.max())
    max_control_norm_ratio_error = float(
        np.abs(pivot["norm_ratio"][control].to_numpy(float) - 1.0).max()
    )
    if (
        max_target_norm_relative_difference > 5e-4
        or max_control_norm_ratio_error > realized_norm_tolerance
    ):
        raise RuntimeError("norm-matched control audit failed")
    for endpoint, (column, sign) in ENDPOINTS.items():
        paired[endpoint] = sign * (
            pivot[column][candidate].to_numpy(float)
            - pivot[column][control].to_numpy(float)
        )
        paired[f"candidate_{column}_damage_from_clean"] = (
            pivot[column][candidate].to_numpy(float)
            - paired["clean_absolute_error"].to_numpy(float)
            if column == "absolute_error"
            else paired["clean_correct"].to_numpy(float)
            - pivot[column][candidate].to_numpy(float)
        )
        paired[f"control_{column}_damage_from_clean"] = (
            pivot[column][control].to_numpy(float)
            - paired["clean_absolute_error"].to_numpy(float)
            if column == "absolute_error"
            else paired["clean_correct"].to_numpy(float)
            - pivot[column][control].to_numpy(float)
        )

    seed_rows: list[dict[str, float | int | str]] = []
    statistic_rows: list[dict[str, float | int | str]] = []
    populations = {
        "all": np.ones(len(paired), dtype=bool),
        "clean_correct": paired["clean_correct"].to_numpy(bool),
    }
    for population, eligible in populations.items():
        subset = paired.loc[eligible].copy()
        for (model, layer), group in subset.groupby(["model_label", "layer"]):
            for endpoint in ENDPOINTS:
                seed_means = group.groupby("seed")[endpoint].mean()
                expected_seeds = set(seeds)
                if set(seed_means.index.astype(int)) != expected_seeds:
                    raise RuntimeError(
                        f"population {population} lacks a seed for {model}/L{layer}/{endpoint}"
                    )
                values = seed_means.to_numpy(float)
                for seed, value in seed_means.items():
                    seed_rows.append(
                        {
                            "model_label": model,
                            "population": population,
                            "layer": int(layer),
                            "normalized_depth": float(group["normalized_depth"].iloc[0]),
                            "endpoint": endpoint,
                            "seed": int(seed),
                            "effect": float(value),
                            "eligible_examples": int((group["seed"] == seed).sum()),
                        }
                    )
                label = f"{model}/{population}/{layer}/{endpoint}"
                low, high = bootstrap_ci(values, label=label, draws=args.bootstraps)
                statistic_rows.append(
                    {
                        "model_label": model,
                        "population": population,
                        "layer": int(layer),
                        "normalized_depth": float(group["normalized_depth"].iloc[0]),
                        "endpoint": endpoint,
                        "examples": len(group),
                        "seeds": len(values),
                        "mean_effect": float(values.mean()),
                        "bootstrap_95ci_low": low,
                        "bootstrap_95ci_high": high,
                        "exact_seed_signflip_p_two_sided": exact_signflip_p(values),
                        "seed_effect_min": float(values.min()),
                        "seed_effect_max": float(values.max()),
                    }
                )

    statistics = pd.DataFrame(statistic_rows)
    statistics["holm_p_within_model_population_endpoint"] = np.nan
    for _, indices in statistics.groupby(
        ["model_label", "population", "endpoint"]
    ).groups.items():
        statistics.loc[indices, "holm_p_within_model_population_endpoint"] = holm(
            statistics.loc[indices, "exact_seed_signflip_p_two_sided"]
        )
    statistics["significant_holm_0_05"] = (
        statistics["holm_p_within_model_population_endpoint"] <= 0.05
    )

    damage_rows: list[dict[str, float | int | str]] = []
    for population, eligible in populations.items():
        subset = paired.loc[eligible].copy()
        for (model, layer), group in subset.groupby(["model_label", "layer"]):
            for endpoint, column in DAMAGE_ENDPOINTS.items():
                seed_means = group.groupby("seed")[column].mean()
                if set(seed_means.index.astype(int)) != set(seeds):
                    raise RuntimeError(
                        f"population {population} lacks a seed for "
                        f"{model}/L{layer}/{endpoint}"
                    )
                values = seed_means.to_numpy(float)
                low, high = bootstrap_ci(
                    values,
                    label=f"damage/{model}/{population}/{layer}/{endpoint}",
                    draws=args.bootstraps,
                )
                damage_rows.append(
                    {
                        "model_label": model,
                        "population": population,
                        "layer": int(layer),
                        "normalized_depth": float(group["normalized_depth"].iloc[0]),
                        "endpoint": endpoint,
                        "examples": len(group),
                        "seeds": len(values),
                        "mean_damage": float(values.mean()),
                        "bootstrap_95ci_low": low,
                        "bootstrap_95ci_high": high,
                        "seed_damage_min": float(values.min()),
                        "seed_damage_max": float(values.max()),
                    }
                )

    damage_statistics = pd.DataFrame(damage_rows)

    seed_effects = pd.DataFrame(seed_rows)
    trend_rows: list[dict[str, float | int | str]] = []
    for (model, population, endpoint), group in seed_effects.groupby(
        ["model_label", "population", "endpoint"]
    ):
        slopes = []
        for _, seed_group in group.groupby("seed"):
            slopes.append(
                float(
                    np.polyfit(
                        seed_group["normalized_depth"].to_numpy(float),
                        seed_group["effect"].to_numpy(float),
                        deg=1,
                    )[0]
                )
            )
        slope_values = np.asarray(slopes)
        low, high = bootstrap_ci(
            slope_values,
            label=f"trend/{model}/{population}/{endpoint}",
            draws=args.bootstraps,
        )
        trend_rows.append(
            {
                "model_label": model,
                "population": population,
                "endpoint": endpoint,
                "seeds": len(slopes),
                "mean_slope_per_unit_depth": float(slope_values.mean()),
                "bootstrap_95ci_low": low,
                "bootstrap_95ci_high": high,
                "exact_seed_signflip_p_two_sided": exact_signflip_p(slope_values),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output / f"{output_prefix}_paired_examples.csv", index=False)
    seed_effects.to_csv(args.output / f"{output_prefix}_seed_effects.csv", index=False)
    statistics.to_csv(args.output / f"{output_prefix}_statistics.csv", index=False)
    damage_statistics.to_csv(
        args.output / f"{output_prefix}_damage_statistics.csv", index=False
    )
    pd.DataFrame(trend_rows).to_csv(
        args.output / f"{output_prefix}_depth_trends.csv", index=False
    )
    audit = {
        "schema_version": f"realistic_niah_v4_4_layerwise_{design_key}_analysis_v1",
        "status": "PASS",
        "inputs": [str(path) for path in input_paths],
        "models": models,
        "rows": len(detail),
        "paired_examples": len(paired),
        "damage_statistics_rows": len(damage_statistics),
        "confirmation_seeds": seeds,
        "counts": counts,
        "support_role": args.support_role,
        "bootstrap_draws": args.bootstraps,
        "inference_unit": "seed mean across registered counts",
        "primary_effect": "candidate damage minus norm-matched orthogonal-control damage; for absolute error this reduces to candidate absolute error minus control absolute error",
        "multiplicity": prompt_design.get(
            "multiplicity", design["multiplicity"][design_key]
        ),
        "max_target_norm_relative_difference": max_target_norm_relative_difference,
        "max_control_norm_ratio_error": max_control_norm_ratio_error,
        "realized_norm_relative_tolerance": realized_norm_tolerance,
    }
    (args.output / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
