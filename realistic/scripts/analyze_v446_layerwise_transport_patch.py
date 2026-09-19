from __future__ import annotations

"""Analyze layerwise answer-query transport-aligned causal patches."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CONDITIONS = ("aligned_dose_1", "aligned_dose_2", "matched_orthogonal")
CONTRASTS = {
    "aligned_dose_1_minus_orthogonal": (
        "aligned_dose_1",
        "matched_orthogonal",
    ),
    "aligned_dose_2_minus_orthogonal": (
        "aligned_dose_2",
        "matched_orthogonal",
    ),
    "dose_2_minus_dose_1": ("aligned_dose_2", "aligned_dose_1"),
}
METRICS = ("target_donor_fraction", "donor_log_odds_gain")


def exact_signflip_p(values: Iterable[float]) -> float:
    vector = np.asarray(list(values), dtype=float)
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
    samples = values[
        rng.integers(0, len(values), size=(draws, len(values)))
    ].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
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


def resolve_inputs(paths: list[Path]) -> list[Path]:
    resolved: list[Path] = []
    for path in paths:
        if path.is_dir():
            resolved.extend(sorted(path.rglob("layerwise_transport_patch.csv")))
        else:
            resolved.append(path)
    if not resolved:
        raise RuntimeError("no layerwise transport files found")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--design-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=50_000)
    args = parser.parse_args()

    design = json.loads(args.design_config.read_text(encoding="utf-8"))
    transport = design["answer_transport"]
    realized_norm_tolerance = float(
        transport["realized_norm_relative_tolerance"]
    )
    input_paths = resolve_inputs(args.inputs)
    detail = pd.concat([pd.read_csv(path) for path in input_paths], ignore_index=True)
    if detail.empty:
        raise RuntimeError("empty layerwise transport table")
    keys = [
        "model_label",
        "seed",
        "receiver_count",
        "donor_count",
        "source_layer",
        "target_layer",
        "condition",
    ]
    duplicate = detail.duplicated(keys, keep=False)
    if duplicate.any():
        conflicting = detail.loc[duplicate].drop_duplicates()
        if conflicting.duplicated(keys, keep=False).any():
            raise RuntimeError("conflicting duplicate transport rows")
        detail = detail.drop_duplicates(keys, keep="first")
    models = sorted(detail["model_label"].unique())
    seeds = [int(value) for value in design["confirmation_seeds"]]
    pairs = [tuple(map(int, value)) for value in transport["pairs"]]
    expected = {
        (model, seed, receiver, donor, int(source), int(target), condition)
        for model in models
        for seed in seeds
        for receiver, donor in pairs
        for source, target in transport["boundaries"][model]
        for condition in CONDITIONS
    }
    observed = {
        (
            str(row.model_label),
            int(row.seed),
            int(row.receiver_count),
            int(row.donor_count),
            int(row.source_layer),
            int(row.target_layer),
            str(row.condition),
        )
        for row in detail.itertuples()
    }
    if observed != expected:
        raise RuntimeError(
            f"output grid mismatch: missing={list(expected-observed)[:5]} "
            f"unexpected={list(observed-expected)[:5]}"
        )
    numeric = (
        "normalized_depth",
        "replacement_delta_norm",
        "planned_replacement_delta_norm",
        "aligned_dose_1_norm",
        "target_donor_fraction",
        "donor_log_odds_gain",
        "argmax_token_changed",
        "geometry_discovery_centroid_r2",
    )
    for column in numeric:
        detail[column] = pd.to_numeric(detail[column], errors="raise")

    pivot = detail.pivot(index=keys[:-1], columns="condition")
    dose1_norm = pivot["replacement_delta_norm"]["aligned_dose_1"].to_numpy(float)
    dose2_norm = pivot["replacement_delta_norm"]["aligned_dose_2"].to_numpy(float)
    control_norm = pivot["replacement_delta_norm"]["matched_orthogonal"].to_numpy(float)
    realized_norm = detail["replacement_delta_norm"].to_numpy(float)
    planned_norm = detail["planned_replacement_delta_norm"].to_numpy(float)
    max_planned_norm_ratio_error = float(
        np.max(np.abs(realized_norm / np.maximum(planned_norm, 1e-12) - 1.0))
    )
    max_control_norm_ratio_error = float(
        np.max(np.abs(control_norm / np.maximum(dose1_norm, 1e-12) - 1.0))
    )
    max_dose2_norm_ratio_error = float(
        np.max(np.abs(dose2_norm / np.maximum(dose1_norm, 1e-12) - 2.0))
    )
    if (
        max_control_norm_ratio_error > realized_norm_tolerance
        or max_dose2_norm_ratio_error > 2.0 * realized_norm_tolerance
    ):
        raise RuntimeError("transport dose-norm audit failed")

    condition_rows: list[dict[str, float | int | str]] = []
    for (model, source, target, condition), group in detail.groupby(
        ["model_label", "source_layer", "target_layer", "condition"]
    ):
        condition_rows.append(
            {
                "model_label": model,
                "source_layer": int(source),
                "target_layer": int(target),
                "normalized_depth": float(group["normalized_depth"].iloc[0]),
                "condition": condition,
                "rows": len(group),
                "seeds": group["seed"].nunique(),
                "mean_target_donor_fraction": float(group["target_donor_fraction"].mean()),
                "mean_donor_log_odds_gain": float(group["donor_log_odds_gain"].mean()),
                "argmax_change_rate": float(group["argmax_token_changed"].mean()),
                "mean_replacement_delta_norm": float(group["replacement_delta_norm"].mean()),
                "geometry_discovery_centroid_r2": float(
                    group["geometry_discovery_centroid_r2"].iloc[0]
                ),
            }
        )

    seed_rows: list[dict[str, float | int | str]] = []
    statistic_rows: list[dict[str, float | int | str]] = []
    index_frame = pd.DataFrame(index=pivot.index).reset_index()
    for contrast, (left, right) in CONTRASTS.items():
        for metric in METRICS:
            effects = pivot[metric][left].to_numpy(float) - pivot[metric][right].to_numpy(float)
            values = index_frame.copy()
            values["effect"] = effects
            for (model, source, target), group in values.groupby(
                ["model_label", "source_layer", "target_layer"]
            ):
                seed_means = group.groupby("seed")["effect"].mean()
                if set(seed_means.index.astype(int)) != set(seeds):
                    raise RuntimeError(
                        f"missing seed for {model}/L{source}->L{target}/{contrast}/{metric}"
                    )
                vector = seed_means.to_numpy(float)
                normalized_depth = float(
                    detail.loc[
                        (detail["model_label"] == model)
                        & (detail["source_layer"] == source)
                        & (detail["target_layer"] == target),
                        "normalized_depth",
                    ].iloc[0]
                )
                for seed, effect in seed_means.items():
                    seed_rows.append(
                        {
                            "model_label": model,
                            "source_layer": int(source),
                            "target_layer": int(target),
                            "normalized_depth": normalized_depth,
                            "contrast": contrast,
                            "metric": metric,
                            "seed": int(seed),
                            "effect": float(effect),
                            "pairs_per_seed": int((group["seed"] == seed).sum()),
                        }
                    )
                label = f"{model}/{source}/{target}/{contrast}/{metric}"
                low, high = bootstrap_ci(vector, label=label, draws=args.bootstraps)
                statistic_rows.append(
                    {
                        "model_label": model,
                        "source_layer": int(source),
                        "target_layer": int(target),
                        "normalized_depth": normalized_depth,
                        "contrast": contrast,
                        "metric": metric,
                        "seeds": len(vector),
                        "pairs_per_seed": int(len(group) / len(vector)),
                        "mean_contrast": float(vector.mean()),
                        "bootstrap_95ci_low": low,
                        "bootstrap_95ci_high": high,
                        "exact_seed_signflip_p_two_sided": exact_signflip_p(vector),
                        "seed_effect_min": float(vector.min()),
                        "seed_effect_max": float(vector.max()),
                    }
                )

    statistics = pd.DataFrame(statistic_rows)
    statistics["holm_p_within_model_metric_contrast"] = np.nan
    for _, indices in statistics.groupby(["model_label", "metric", "contrast"]).groups.items():
        statistics.loc[indices, "holm_p_within_model_metric_contrast"] = holm(
            statistics.loc[indices, "exact_seed_signflip_p_two_sided"]
        )
    statistics["significant_holm_0_05"] = (
        statistics["holm_p_within_model_metric_contrast"] <= 0.05
    )

    seed_effects = pd.DataFrame(seed_rows)
    trend_rows: list[dict[str, float | int | str]] = []
    for (model, contrast, metric), group in seed_effects.groupby(
        ["model_label", "contrast", "metric"]
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
        vector = np.asarray(slopes)
        low, high = bootstrap_ci(
            vector,
            label=f"trend/{model}/{contrast}/{metric}",
            draws=args.bootstraps,
        )
        trend_rows.append(
            {
                "model_label": model,
                "contrast": contrast,
                "metric": metric,
                "seeds": len(vector),
                "mean_slope_per_unit_depth": float(vector.mean()),
                "bootstrap_95ci_low": low,
                "bootstrap_95ci_high": high,
                "exact_seed_signflip_p_two_sided": exact_signflip_p(vector),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(condition_rows).to_csv(
        args.output / "layerwise_transport_condition_summary.csv", index=False
    )
    seed_effects.to_csv(args.output / "layerwise_transport_seed_effects.csv", index=False)
    statistics.to_csv(args.output / "layerwise_transport_statistics.csv", index=False)
    pd.DataFrame(trend_rows).to_csv(
        args.output / "layerwise_transport_depth_trends.csv", index=False
    )
    audit = {
        "schema_version": "realistic_niah_v4_4_layerwise_transport_analysis_v1",
        "status": "PASS",
        "inputs": [str(path) for path in input_paths],
        "models": models,
        "rows": len(detail),
        "confirmation_seeds": seeds,
        "directed_pairs": pairs,
        "bootstraps": args.bootstraps,
        "inference_unit": "seed mean across four preregistered directed count pairs",
        "primary_endpoint": transport["primary_endpoint"],
        "multiplicity": design["multiplicity"]["answer_transport"],
        "max_control_norm_ratio_error": max_control_norm_ratio_error,
        "max_dose2_norm_ratio_error": max_dose2_norm_ratio_error,
        "max_planned_norm_ratio_error_diagnostic": max_planned_norm_ratio_error,
        "planned_norm_policy": "diagnostic only; realized paired-dose ratios define the causal norm audit",
        "realized_norm_relative_tolerance": realized_norm_tolerance,
    }
    (args.output / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
