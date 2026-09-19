from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"No rows in {path}")
    return pd.DataFrame(rows)


def finite_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.mean()) if len(numeric) else math.nan


def attach_baselines(detail: pd.DataFrame) -> pd.DataFrame:
    key = ["model_label", "seed", "gold_count"]
    baselines = detail[detail["patch_layer"].astype(int).eq(-1)].copy()
    patches = detail[detail["patch_layer"].astype(int).ge(0)].copy()
    if patches.empty:
        raise ValueError("No completed restoration rows")
    clean = baselines[baselines["condition"].eq("clean")][
        key
        + [
            "expected_count",
            "strict_prediction",
            "strict_correct",
            "strict_absolute_error",
        ]
    ].rename(
        columns={
            "expected_count": "clean_expected_count",
            "strict_prediction": "clean_strict_prediction",
            "strict_correct": "clean_strict_correct",
            "strict_absolute_error": "clean_strict_absolute_error",
        }
    )
    corrupt = baselines[baselines["condition"].isin(["needle_corrupt", "ordinary_corrupt"])][
        key
        + [
            "condition",
            "expected_count",
            "strict_prediction",
            "strict_correct",
            "strict_absolute_error",
        ]
    ].rename(
        columns={
            "condition": "baseline_condition",
            "expected_count": "corrupt_expected_count",
            "strict_prediction": "corrupt_strict_prediction",
            "strict_correct": "corrupt_strict_correct",
            "strict_absolute_error": "corrupt_strict_absolute_error",
        }
    )
    patches["baseline_condition"] = np.where(
        patches["patch_kind"].astype(str).str.startswith("needle"),
        "needle_corrupt",
        "ordinary_corrupt",
    )
    merged = patches.merge(clean, on=key, how="left", validate="many_to_one")
    merged = merged.merge(
        corrupt,
        on=key + ["baseline_condition"],
        how="left",
        validate="many_to_one",
    )
    required = [
        "clean_expected_count",
        "corrupt_expected_count",
        "corrupt_strict_absolute_error",
    ]
    if merged[required].isna().any().any():
        raise RuntimeError("A restoration row lacks its clean/corrupted baseline")
    gold = merged["gold_count"].astype(float)
    merged["expected_count_shift"] = (
        merged["expected_count"].astype(float)
        - merged["corrupt_expected_count"].astype(float)
    )
    denominator = (
        merged["clean_expected_count"].astype(float)
        - merged["corrupt_expected_count"].astype(float)
    )
    merged["normalized_recovery"] = np.where(
        denominator.abs() > 1e-8,
        merged["expected_count_shift"] / denominator,
        np.nan,
    )
    merged["expected_absolute_error"] = (
        merged["expected_count"].astype(float) - gold
    ).abs()
    merged["corrupt_expected_absolute_error"] = (
        merged["corrupt_expected_count"].astype(float) - gold
    ).abs()
    merged["expected_absolute_error_reduction"] = (
        merged["corrupt_expected_absolute_error"]
        - merged["expected_absolute_error"]
    )
    merged["strict_absolute_error_reduction"] = (
        merged["corrupt_strict_absolute_error"].astype(float)
        - merged["strict_absolute_error"].astype(float)
    )
    merged["strict_recovered_correct"] = (
        ~merged["corrupt_strict_correct"].astype(bool)
        & merged["strict_correct"].astype(bool)
    )
    return merged


def population_frames(
    frame: pd.DataFrame,
    discovery_seeds: set[int],
    confirmation_seeds: set[int],
) -> list[tuple[str, pd.DataFrame]]:
    discovery = frame[frame["seed"].isin(discovery_seeds)].copy()
    confirmation = frame[frame["seed"].isin(confirmation_seeds)].copy()
    correct = frame[frame["clean_strict_correct"].astype(bool)].copy()
    return [
        ("all", frame),
        ("discovery", discovery),
        ("confirmation", confirmation),
        ("clean_correct", correct),
        (
            "discovery_clean_correct",
            correct[correct["seed"].isin(discovery_seeds)].copy(),
        ),
        (
            "confirmation_clean_correct",
            correct[correct["seed"].isin(confirmation_seeds)].copy(),
        ),
    ]


def summarize(
    restoration: pd.DataFrame,
    discovery_seeds: set[int],
    confirmation_seeds: set[int],
) -> pd.DataFrame:
    populations = population_frames(restoration, discovery_seeds, confirmation_seeds)
    rows: list[dict[str, Any]] = []
    for population, frame in populations:
        for keys, group in frame.groupby(
            ["model_label", "patch_kind", "patch_layer"], sort=True
        ):
            model, patch_kind, layer = keys
            rows.append(
                {
                    "model_label": model,
                    "population": population,
                    "patch_kind": patch_kind,
                    "patch_layer": int(layer),
                    "rows": int(len(group)),
                    "seeds": int(group["seed"].nunique()),
                    "mean_expected_count_shift": finite_mean(
                        group["expected_count_shift"]
                    ),
                    "mean_normalized_recovery": finite_mean(
                        group["normalized_recovery"]
                    ),
                    "mean_expected_absolute_error_reduction": finite_mean(
                        group["expected_absolute_error_reduction"]
                    ),
                    "mean_strict_absolute_error_reduction": finite_mean(
                        group["strict_absolute_error_reduction"]
                    ),
                    "strict_accuracy": float(group["strict_correct"].mean()),
                    "strict_recovered_correct_rate": float(
                        group["strict_recovered_correct"].mean()
                    ),
                    "mean_elapsed_seconds": float(group["elapsed_seconds"].mean()),
                    "max_cuda_allocated_gib": float(
                        group["max_cuda_allocated_bytes"].max() / 2**30
                    ),
                    "max_cuda_reserved_gib": float(
                        group["max_cuda_reserved_bytes"].max() / 2**30
                    ),
                }
            )
    return pd.DataFrame(rows)


def paired_contrasts(restoration: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = ["model_label", "seed", "gold_count", "patch_layer"]
    metrics = [
        "normalized_recovery",
        "expected_absolute_error_reduction",
        "strict_absolute_error_reduction",
    ]
    pivot = restoration.pivot_table(
        index=index, columns="patch_kind", values=metrics, aggfunc="first"
    )
    specificity_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    for values, row in pivot.iterrows():
        model, seed, count, layer = values
        if all((metric, kind) in pivot.columns for metric in metrics for kind in ("needle_full", "ordinary_full")):
            specificity_rows.append(
                {
                    "model_label": model,
                    "seed": int(seed),
                    "gold_count": int(count),
                    "patch_layer": int(layer),
                    **{
                        f"{metric}_specificity": float(
                            row[(metric, "needle_full")]
                            - row[(metric, "ordinary_full")]
                        )
                        for metric in metrics
                    },
                }
            )
        if all((metric, kind) in pivot.columns for metric in metrics for kind in ("needle_full", "needle_endpoint")):
            endpoint_rows.append(
                {
                    "model_label": model,
                    "seed": int(seed),
                    "gold_count": int(count),
                    "patch_layer": int(layer),
                    **{
                        f"{metric}_full_minus_endpoint": float(
                            row[(metric, "needle_full")]
                            - row[(metric, "needle_endpoint")]
                        )
                        for metric in metrics
                    },
                }
            )
    eligibility = restoration[
        index + ["clean_strict_correct"]
    ].drop_duplicates(index)
    specificity = pd.DataFrame(specificity_rows).merge(
        eligibility, on=index, how="left", validate="one_to_one"
    )
    endpoint = pd.DataFrame(endpoint_rows).merge(
        eligibility, on=index, how="left", validate="one_to_one"
    )
    return specificity, endpoint


def summarize_contrast(
    frame: pd.DataFrame,
    suffix: str,
    discovery_seeds: set[int],
    confirmation_seeds: set[int],
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    metrics = [column for column in frame if column.endswith(suffix)]
    rows: list[dict[str, Any]] = []
    for population, population_frame in population_frames(
        frame, discovery_seeds, confirmation_seeds
    ):
        for keys, group in population_frame.groupby(
            ["model_label", "patch_layer"], sort=True
        ):
            model, layer = keys
            row: dict[str, Any] = {
                "model_label": model,
                "population": population,
                "patch_layer": int(layer),
                "rows": int(len(group)),
                "seeds": int(group["seed"].nunique()),
            }
            for metric in metrics:
                values = pd.to_numeric(group[metric], errors="coerce").to_numpy(
                    dtype=float
                )
                values = values[np.isfinite(values)]
                row[f"mean_{metric}"] = (
                    float(values.mean()) if len(values) else np.nan
                )
                row[f"median_{metric}"] = (
                    float(np.median(values)) if len(values) else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def first_sustained_layer(
    layers: np.ndarray,
    values: np.ndarray,
    predicate: Any,
    *,
    start_layer: int,
    consecutive_layers: int = 3,
) -> int | None:
    for index in range(len(layers) - consecutive_layers + 1):
        window_layers = layers[index : index + consecutive_layers]
        window_values = values[index : index + consecutive_layers]
        if int(window_layers[0]) < start_layer:
            continue
        if not np.array_equal(
            window_layers, np.arange(window_layers[0], window_layers[0] + consecutive_layers)
        ):
            continue
        if np.isfinite(window_values).all() and bool(np.all(predicate(window_values))):
            return int(window_layers[0])
    return None


def layerwise_transition_boundaries(
    specificity_summary: pd.DataFrame,
    endpoint_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Freeze descriptive boundaries on discovery and read them on confirmation."""
    metric = "mean_expected_absolute_error_reduction_specificity"
    endpoint_metric = (
        "mean_expected_absolute_error_reduction_full_minus_endpoint"
    )
    rows: list[dict[str, Any]] = []
    discovery = specificity_summary[
        specificity_summary["population"].eq("discovery")
    ]
    confirmation = specificity_summary[
        specificity_summary["population"].eq("confirmation")
    ]
    confirmation_endpoint = endpoint_summary[
        endpoint_summary["population"].eq("confirmation")
    ]
    for model, frame in discovery.groupby("model_label", sort=True):
        frame = frame.sort_values("patch_layer")
        layers = frame["patch_layer"].to_numpy(dtype=int)
        values = frame[metric].to_numpy(dtype=float)
        expected_layers = np.arange(int(layers.min()), int(layers.max()) + 1)
        if not np.array_equal(layers, expected_layers):
            raise RuntimeError(f"{model} does not have a dense layerwise sweep")
        quarter_count = max(1, int(math.ceil(len(layers) / 4)))
        early_layers = layers[:quarter_count]
        early_plateau = float(np.median(values[:quarter_count]))
        half_threshold = 0.5 * early_plateau
        search_start = int(early_layers[-1] + 1)
        half_layer = first_sustained_layer(
            layers,
            values,
            lambda window: window <= half_threshold,
            start_layer=search_start,
        )
        near_zero_layer = first_sustained_layer(
            layers,
            values,
            lambda window: np.abs(window) <= 0.10,
            start_layer=search_start,
        )
        model_confirmation = confirmation[confirmation["model_label"].eq(model)]
        model_endpoint = confirmation_endpoint[
            confirmation_endpoint["model_label"].eq(model)
        ]
        for boundary_name, boundary_layer in (
            ("half_early_plateau", half_layer),
            ("near_zero_0.10_count", near_zero_layer),
        ):
            confirmation_value = math.nan
            endpoint_value = math.nan
            if boundary_layer is not None:
                match = model_confirmation[
                    model_confirmation["patch_layer"].eq(boundary_layer)
                ]
                endpoint_match = model_endpoint[
                    model_endpoint["patch_layer"].eq(boundary_layer)
                ]
                if len(match) == 1:
                    confirmation_value = float(match.iloc[0][metric])
                if len(endpoint_match) == 1:
                    endpoint_value = float(endpoint_match.iloc[0][endpoint_metric])
            rows.append(
                {
                    "model_label": model,
                    "boundary": boundary_name,
                    "discovery_early_layer_start": int(early_layers[0]),
                    "discovery_early_layer_end": int(early_layers[-1]),
                    "discovery_early_plateau": early_plateau,
                    "discovery_half_threshold": half_threshold,
                    "required_consecutive_layers": 3,
                    "frozen_boundary_layer": boundary_layer,
                    "confirmation_specificity_at_boundary": confirmation_value,
                    "confirmation_full_minus_endpoint_at_boundary": endpoint_value,
                }
            )
    return pd.DataFrame(rows)


def analyze_broad(
    broad: pd.DataFrame, detail: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["model_label", "seed", "gold_count", "layer", "head"]
    baselines = broad[broad["patch_layer"].astype(int).eq(-1)].copy()
    patches = broad[broad["patch_layer"].astype(int).ge(0)].copy()
    patches["baseline_condition"] = np.where(
        patches["patch_kind"].astype(str).str.startswith("needle"),
        "needle_corrupt",
        "ordinary_corrupt",
    )
    base = baselines[baselines["condition"].isin(["needle_corrupt", "ordinary_corrupt"])][
        key + ["condition", "needle_mass", "coverage", "broad_score"]
    ].rename(
        columns={
            "condition": "baseline_condition",
            "needle_mass": "baseline_needle_mass",
            "coverage": "baseline_coverage",
            "broad_score": "baseline_broad_score",
        }
    )
    merged = patches.merge(
        base,
        on=key + ["baseline_condition"],
        how="left",
        validate="many_to_one",
    )
    for metric in ("needle_mass", "coverage", "broad_score"):
        merged[f"delta_{metric}"] = (
            merged[metric].astype(float) - merged[f"baseline_{metric}"].astype(float)
        )
    summary = (
        merged.groupby(
            ["model_label", "patch_kind", "patch_layer", "layer", "head"],
            as_index=False,
        )
        .agg(
            rows=("seed", "size"),
            seeds=("seed", "nunique"),
            mean_delta_needle_mass=("delta_needle_mass", "mean"),
            mean_delta_coverage=("delta_coverage", "mean"),
            mean_delta_broad_score=("delta_broad_score", "mean"),
        )
    )
    return merged, summary


def plot_recovery(
    summary: pd.DataFrame,
    output: Path,
    *,
    population: str = "all",
    title_suffix: str = "all evaluated prompts",
) -> None:
    models = list(summary["model_label"].drop_duplicates())
    figure, axes = plt.subplots(len(models), 1, figsize=(10, 4.2 * len(models)), squeeze=False)
    colors = {
        "needle_endpoint": "#d97706",
        "needle_full": "#0f766e",
        "ordinary_full": "#6b7280",
    }
    for axis, model in zip(axes[:, 0], models):
        frame = summary[
            summary["model_label"].eq(model)
            & summary["population"].eq(population)
        ]
        if frame.empty:
            axis.text(0.5, 0.5, "no eligible rows", ha="center", va="center")
            axis.set_axis_off()
            continue
        for kind, group in frame.groupby("patch_kind", sort=False):
            group = group.sort_values("patch_layer")
            axis.plot(
                group["patch_layer"],
                group["mean_expected_absolute_error_reduction"],
                marker="o",
                linewidth=2,
                label=kind,
                color=colors.get(str(kind)),
            )
        axis.axhline(0, color="#111827", linewidth=1, linestyle="--")
        axis.set_title(str(model))
        axis.set_xlabel("Patched post-block layer (zero-based)")
        axis.set_ylabel("Reduction in |expected count - gold|")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle(
        "Layerwise clean-state restoration after matched token corruption\n"
        f"Population: {title_suffix}"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_broad_heatmaps(summary: pd.DataFrame, output: Path) -> None:
    models = list(summary["model_label"].drop_duplicates())
    figure, axes = plt.subplots(len(models), 1, figsize=(11, 4.5 * len(models)), squeeze=False)
    for axis, model in zip(axes[:, 0], models):
        frame = summary[
            summary["model_label"].eq(model)
            & summary["patch_kind"].eq("needle_full")
        ].copy()
        frame["retrieval_head"] = frame.apply(
            lambda row: f"L{int(row['layer'])}H{int(row['head'])}", axis=1
        )
        pivot = frame.pivot_table(
            index="retrieval_head",
            columns="patch_layer",
            values="mean_delta_broad_score",
            aggfunc="mean",
        )
        if pivot.empty:
            axis.text(0.5, 0.5, "no broad rows", ha="center", va="center")
            continue
        magnitude = float(np.nanmax(np.abs(pivot.to_numpy())))
        magnitude = max(magnitude, 1e-6)
        image = axis.imshow(
            pivot.to_numpy(),
            aspect="auto",
            cmap="coolwarm",
            vmin=-magnitude,
            vmax=magnitude,
        )
        axis.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
        axis.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns)
        axis.set_xlabel("Patched post-block layer")
        axis.set_ylabel("Frozen broad head")
        axis.set_title(f"{model}: change in broad score after full-span restoration")
        figure.colorbar(image, ax=axis, label="Delta broad score")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def audit_patch_hook_applications(detail: pd.DataFrame) -> dict[str, Any]:
    """Check the hook count implied by each row's generation path.

    The original implementation ran three patched long-prefix forwards.  The
    cache-reuse implementation retains the patched generation prefill and runs
    only two.  Baseline rows do not install a patch hook and are excluded.
    """

    patched = detail[detail["patch_layer"].astype(int).ge(0)].copy()
    actual = pd.to_numeric(
        patched["patch_hook_applications"], errors="coerce"
    )
    if "strict_generation_reused_prefill" in patched:
        reused = (
            patched["strict_generation_reused_prefill"]
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )
    else:
        reused = pd.Series(False, index=patched.index, dtype=bool)
    expected = pd.Series(np.where(reused, 2, 3), index=patched.index)
    matches = actual.eq(expected)
    return {
        "patched_rows": int(len(patched)),
        "reused_prefill_rows_expected_two": int(reused.sum()),
        "legacy_rows_expected_three": int((~reused).sum()),
        "observed_two": int(actual.eq(2).sum()),
        "observed_three": int(actual.eq(3).sum()),
        "mismatched_rows": int((~matches).sum()),
        "status": "PASS" if bool(matches.all()) else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--experiment-config",
        default="configs/realistic_niah_v4_4_5_span_restoration_canonical.json",
    )
    parser.add_argument("--models", nargs="+", default=["Qwen3-8B", "Gemma4-E4B"])
    args = parser.parse_args()

    config = json.loads(Path(args.experiment_config).read_text(encoding="utf-8"))
    discovery_seeds = {int(value) for value in config["discovery_seeds"]}
    confirmation_seeds = {int(value) for value in config["confirmation_seeds"]}
    root = Path(args.run_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    detail = pd.concat(
        [read_jsonl(root / model / "detail.jsonl") for model in args.models],
        ignore_index=True,
    )
    broad = pd.concat(
        [read_jsonl(root / model / "broad_metrics.jsonl") for model in args.models],
        ignore_index=True,
    )
    restoration = attach_baselines(detail)
    summary = summarize(restoration, discovery_seeds, confirmation_seeds)
    specificity, endpoint = paired_contrasts(restoration)
    specificity_summary = summarize_contrast(
        specificity, "_specificity", discovery_seeds, confirmation_seeds
    )
    endpoint_summary = summarize_contrast(
        endpoint, "_full_minus_endpoint", discovery_seeds, confirmation_seeds
    )
    transitions = layerwise_transition_boundaries(
        specificity_summary, endpoint_summary
    )
    broad_detail, broad_summary = analyze_broad(broad, detail)

    detail.to_csv(output / "raw_detail.csv", index=False)
    restoration.to_csv(output / "restoration_effects.csv", index=False)
    summary.to_csv(output / "layer_summary.csv", index=False)
    specificity.to_csv(output / "needle_minus_ordinary_specificity.csv", index=False)
    specificity_summary.to_csv(
        output / "needle_minus_ordinary_specificity_summary.csv", index=False
    )
    endpoint.to_csv(output / "full_minus_endpoint.csv", index=False)
    endpoint_summary.to_csv(output / "full_minus_endpoint_summary.csv", index=False)
    transitions.to_csv(output / "layerwise_transition_boundaries.csv", index=False)
    broad_detail.to_csv(output / "broad_effects.csv", index=False)
    broad_summary.to_csv(output / "broad_summary.csv", index=False)
    plot_recovery(
        summary,
        output / "layerwise_restoration.png",
        population="all",
        title_suffix="all evaluated prompts",
    )
    plot_recovery(
        summary,
        output / "layerwise_restoration_clean_correct.png",
        population="clean_correct",
        title_suffix="only prompts whose clean strict generation is correct",
    )
    plot_broad_heatmaps(broad_summary, output / "broad_score_change_heatmap.png")
    patch_hook_audit = audit_patch_hook_applications(detail)
    patch_hooks_pass = patch_hook_audit["status"] == "PASS"
    finite_expected = bool(
        np.isfinite(pd.to_numeric(detail["expected_count"], errors="coerce")).all()
    )
    centered = pd.to_numeric(
        detail.get(
            "attention_cache_candidate_centered_logit_max_abs_delta",
            pd.Series(np.nan, index=detail.index),
        ),
        errors="coerce",
    )
    tv = pd.to_numeric(
        detail.get(
            "attention_cache_candidate_probability_total_variation",
            pd.Series(np.nan, index=detail.index),
        ),
        errors="coerce",
    )
    if "attention_cache_equivalence_audited" in detail:
        equivalence_audited = (
            detail["attention_cache_equivalence_audited"]
            .astype("boolean")
            .fillna(True)
            .astype(bool)
        )
    else:
        equivalence_audited = centered.notna() | tv.notna()
    cache_only = ~equivalence_audited
    compared_centered = centered[equivalence_audited].dropna()
    compared_tv = tv[equivalence_audited].dropna()
    max_centered = (
        float(compared_centered.max()) if len(compared_centered) else math.nan
    )
    max_tv = float(compared_tv.max()) if len(compared_tv) else math.nan
    all_pass = bool(patch_hooks_pass and finite_expected)
    audit = {
        "schema_version": "realistic_niah_v4_4_5_span_restoration_analysis_v2",
        "models": list(args.models),
        "discovery_seeds": sorted(discovery_seeds),
        "confirmation_seeds": sorted(confirmation_seeds),
        "detail_rows": int(len(detail)),
        "restoration_rows": int(len(restoration)),
        "all_patch_hooks_expected_applications": patch_hooks_pass,
        "patch_hook_application_audit": patch_hook_audit,
        "finite_expected_counts": finite_expected,
        "attention_protocol": "original V4 prefix-cache final-query eager rows",
        "cache_equivalence_compared_rows": int(equivalence_audited.sum()),
        "cache_only_uncompared_rows": int(cache_only.sum()),
        "historical_compared_max_centered_delta": max_centered,
        "historical_compared_max_probability_total_variation": max_tv,
        "cache_equivalence_is_not_a_final_validity_gate": True,
        "definition": {
            "expected_absolute_error_reduction": (
                "|E_corrupt-count - gold| minus |E_restored-count - gold|"
            ),
            "strict_absolute_error_reduction": (
                "|strict corrupt prediction - gold| minus "
                "|strict restored prediction - gold|; invalid prediction has error 10"
            ),
            "normalized_recovery": (
                "(E_restored - E_corrupt)/(E_clean - E_corrupt), left unclipped"
            ),
            "specificity": "needle restoration effect minus ordinary restoration effect",
            "layerwise_half_boundary": (
                "first of three consecutive layers whose discovery needle-minus-ordinary "
                "expected-error-reduction specificity is at most half the median over "
                "the first quarter of layers"
            ),
            "layerwise_near_zero_boundary": (
                "first of three consecutive layers whose discovery specificity has "
                "absolute magnitude at most 0.10 count"
            ),
        },
        "status": "PASS" if all_pass else "FAIL",
    }
    (output / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not all_pass:
        raise RuntimeError("V4.4.5 span-restoration analysis audit failed")


if __name__ == "__main__":
    main()
