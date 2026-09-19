from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SITES = ("span_end", "span_mean")
MODEL_ORDER = ("Qwen3-8B", "Gemma4-E4B")
RETENTION_METRICS = (
    "correct_distance_specificity",
    "margin_damage_specificity",
    "accuracy_damage_specificity",
)
DEFORMATION_METRICS = (
    "relative_rms_specificity",
    "cosine_specificity",
)
SIGNED_METRICS = RETENTION_METRICS + DEFORMATION_METRICS


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def exact_sign_flip(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not 1 <= len(array) <= 20 or not np.isfinite(array).all():
        raise ValueError("Exact sign-flip requires 1..20 finite paired effects")
    observed = abs(float(array.mean()))
    null = np.asarray(
        [
            float(np.mean(array * np.asarray(signs, dtype=np.float64)))
            for signs in itertools.product((-1.0, 1.0), repeat=len(array))
        ],
        dtype=np.float64,
    )
    return float(np.mean(np.abs(null) >= observed - 1e-15))


def bootstrap_ci(
    values: Sequence[float], *, draws: int, seed: int
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 1 or not np.isfinite(array).all():
        raise ValueError("Bootstrap requires a nonempty finite one-dimensional array")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(array), size=(int(draws), len(array)))
    sampled = array[indices].mean(axis=1)
    low, high = np.quantile(sampled, (0.025, 0.975))
    return float(low), float(high)


def summarize(
    values: Sequence[float], *, draws: int, seed: int, test: bool = True
) -> dict[str, Any]:
    low, high = bootstrap_ci(values, draws=draws, seed=seed)
    return {
        "mean": float(np.mean(values)),
        "ci95_low": low,
        "ci95_high": high,
        "exact_sign_flip_p": exact_sign_flip(values) if test else None,
        "seed_units": int(len(values)),
    }


def holm_adjust(pvalues: Iterable[float]) -> list[float]:
    values = np.asarray(list(pvalues), dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Holm adjustment requires finite one-dimensional p-values")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    family_size = len(values)
    for rank, position in enumerate(order):
        candidate = (family_size - rank) * float(values[position])
        running = max(running, candidate)
        adjusted[position] = min(running, 1.0)
    return [float(value) for value in adjusted]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def candidate_name(pooling: str, metric: str) -> str:
    return f"{pooling}_{metric}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze held-out clean-centroid state retention with frozen windows."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--experiment-config",
        default="configs/realistic_niah_v4_4_5_state_retention.json",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(args.run_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.experiment_config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "realistic_niah_v4_4_5_state_retention_v1":
        raise ValueError("Unexpected state-retention config schema")
    draws = int(config["bootstrap_draws"])
    bootstrap_seed = int(config["bootstrap_seed"])
    confirmation_seeds = tuple(int(value) for value in config["confirmation_seeds"])

    audit: dict[str, Any] = {
        "schema_version": "realistic_niah_v4_4_5_state_retention_analysis_audit_v1",
        "status": "PASS",
        "models": {},
    }
    layerwise_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    level_rows: list[dict[str, Any]] = []
    all_details: dict[str, list[dict[str, Any]]] = {}

    for model_index, model in enumerate(MODEL_ORDER):
        model_root = root / "formal" / model
        complete = json.loads((model_root / "complete.json").read_text(encoding="utf-8"))
        details = read_jsonl(model_root / "layer_seed_metrics.jsonl")
        all_details[model] = details
        layers = tuple(int(value) for value in config["layers"][model])
        expected_keys = {
            (int(seed), int(layer), site)
            for seed in confirmation_seeds
            for layer in layers
            for site in SITES
        }
        keys = [
            (int(row["seed"]), int(row["layer"]), str(row["pooling"]))
            for row in details
        ]
        finite = all(
            math.isfinite(float(row[metric]))
            for row in details
            for metric in SIGNED_METRICS
        )
        model_audit = {
            "run_audit_status": complete.get("status"),
            "rows": len(details),
            "expected_rows": len(expected_keys),
            "unique_keys": len(set(keys)),
            "exact_key_coverage": set(keys) == expected_keys,
            "finite_signed_metrics": finite,
        }
        if not (
            model_audit["run_audit_status"] == "PASS"
            and model_audit["rows"] == model_audit["expected_rows"]
            and model_audit["unique_keys"] == model_audit["expected_rows"]
            and model_audit["exact_key_coverage"]
            and model_audit["finite_signed_metrics"]
        ):
            audit["status"] = "FAIL"
        audit["models"][model] = model_audit

        lookup = {
            (int(row["seed"]), int(row["layer"]), str(row["pooling"])): row
            for row in details
        }
        for site_index, site in enumerate(SITES):
            for layer in layers:
                for metric_index, metric in enumerate(SIGNED_METRICS):
                    values = [
                        float(lookup[(seed, layer, site)][metric])
                        for seed in confirmation_seeds
                    ]
                    stats = summarize(
                        values,
                        draws=draws,
                        seed=(
                            bootstrap_seed
                            + model_index * 100_000
                            + site_index * 10_000
                            + layer * 100
                            + metric_index
                        ),
                    )
                    layerwise_rows.append(
                        {
                            "model_label": model,
                            "pooling": site,
                            "layer": int(layer),
                            "metric": metric,
                            "family": (
                                "retention"
                                if metric in RETENTION_METRICS
                                else "deformation"
                            ),
                            **stats,
                            "scope": "exploratory_layerwise_no_layer_selection",
                        }
                    )

            window_start, window_end = (
                int(value)
                for value in config["independently_frozen_reusable_windows"][model]
            )
            window_layers = tuple(
                layer for layer in layers if window_start <= layer <= window_end
            )
            for metric_index, metric in enumerate(SIGNED_METRICS):
                values = [
                    float(
                        np.mean(
                            [
                                float(lookup[(seed, layer, site)][metric])
                                for layer in window_layers
                            ]
                        )
                    )
                    for seed in confirmation_seeds
                ]
                stats = summarize(
                    values,
                    draws=draws,
                    seed=(
                        bootstrap_seed
                        + model_index * 100_000
                        + site_index * 10_000
                        + 9000
                        + metric_index
                    ),
                )
                window_rows.append(
                    {
                        "model_label": model,
                        "pooling": site,
                        "window_start": window_start,
                        "window_end": window_end,
                        "window_layers": len(window_layers),
                        "metric": metric,
                        "candidate": candidate_name(site, metric),
                        "family": (
                            "retention"
                            if metric in RETENTION_METRICS
                            else "deformation"
                        ),
                        **stats,
                    }
                )

            level_definitions = {
                "correct_distance": (
                    "clean_correct_distance",
                    "needle_correct_distance",
                    "ordinary_correct_distance",
                ),
                "margin": ("clean_margin", "needle_margin", "ordinary_margin"),
                "accuracy": (
                    "clean_accuracy",
                    "needle_accuracy",
                    "ordinary_accuracy",
                ),
                "mad": ("clean_mad", "needle_mad", "ordinary_mad"),
                "relative_rms": (None, "needle_relative_rms", "ordinary_relative_rms"),
                "cosine_distance": (
                    None,
                    "needle_cosine_distance",
                    "ordinary_cosine_distance",
                ),
            }
            for measure_index, (measure, columns) in enumerate(
                level_definitions.items()
            ):
                for condition_index, (condition, column) in enumerate(
                    zip(("clean", "needle_corrupt", "ordinary_corrupt"), columns)
                ):
                    if column is None:
                        continue
                    values = [
                        float(
                            np.mean(
                                [
                                    float(lookup[(seed, layer, site)][column])
                                    for layer in window_layers
                                ]
                            )
                        )
                        for seed in confirmation_seeds
                    ]
                    stats = summarize(
                        values,
                        draws=draws,
                        seed=(
                            bootstrap_seed
                            + model_index * 100_000
                            + site_index * 10_000
                            + 9500
                            + measure_index * 10
                            + condition_index
                        ),
                        test=False,
                    )
                    level_rows.append(
                        {
                            "model_label": model,
                            "pooling": site,
                            "window_start": window_start,
                            "window_end": window_end,
                            "measure": measure,
                            "condition": condition,
                            **stats,
                        }
                    )

    if audit["status"] != "PASS":
        (output / "analysis_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise RuntimeError("Input audit failed before state-retention analysis")

    for model in MODEL_ORDER:
        for family, expected_size in (("retention", 6), ("deformation", 4)):
            indices = [
                index
                for index, row in enumerate(window_rows)
                if row["model_label"] == model and row["family"] == family
            ]
            if len(indices) != expected_size:
                raise RuntimeError(
                    f"{model} {family} family has {len(indices)} not {expected_size} tests"
                )
            adjusted = holm_adjust(
                [float(window_rows[index]["exact_sign_flip_p"]) for index in indices]
            )
            for index, holm_p in zip(indices, adjusted):
                window_rows[index]["holm_p_within_model_family"] = holm_p
                window_rows[index]["positive_and_holm_0_05"] = bool(
                    float(window_rows[index]["mean"]) > 0 and holm_p < 0.05
                )

    window_lookup = {
        (str(row["model_label"]), str(row["candidate"])): row
        for row in window_rows
    }
    selected: str | None = None
    evaluated: list[dict[str, Any]] = []
    for candidate in config["main_display_priority"]:
        model_pass = {
            model: bool(window_lookup[(model, candidate)]["positive_and_holm_0_05"])
            for model in MODEL_ORDER
        }
        evaluated.append({"candidate": candidate, "model_pass": model_pass})
        if selected is None and all(model_pass.values()):
            selected = str(candidate)
    primary = str(config["primary_metric"])
    selection = {
        "schema_version": "realistic_niah_v4_4_5_state_retention_selection_v1",
        "selection_rule": config["main_display_selection_rule"],
        "priority": list(config["main_display_priority"]),
        "evaluated": evaluated,
        "selected_candidate": selected if selected is not None else primary,
        "cross_model_supported_candidate_found": selected is not None,
        "fallback_to_preregistered_primary": selected is None,
        "primary_metric": primary,
    }

    write_csv(output / "layerwise_state_retention.csv", layerwise_rows)
    write_csv(output / "window_state_retention.csv", window_rows)
    write_csv(output / "window_condition_levels.csv", level_rows)
    (output / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": "realistic_niah_v4_4_5_state_retention_analysis_v1",
        "stimulus_sha256": config["stimulus_sha256"],
        "discovery_seeds": config["discovery_seeds"],
        "confirmation_seeds": config["confirmation_seeds"],
        "gold_count": 10,
        "selection": selection,
        "models": {
            model: {
                "window": config["independently_frozen_reusable_windows"][model],
                "candidates": {
                    row["candidate"]: {
                        key: row[key]
                        for key in (
                            "mean",
                            "ci95_low",
                            "ci95_high",
                            "exact_sign_flip_p",
                            "holm_p_within_model_family",
                            "positive_and_holm_0_05",
                        )
                    }
                    for row in window_rows
                    if row["model_label"] == model
                },
            }
            for model in MODEL_ORDER
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    expected_layerwise = sum(len(config["layers"][model]) for model in MODEL_ORDER) * len(
        SITES
    ) * len(SIGNED_METRICS)
    audit.update(
        {
            "layerwise_rows": len(layerwise_rows),
            "expected_layerwise_rows": expected_layerwise,
            "window_rows": len(window_rows),
            "expected_window_rows": len(MODEL_ORDER)
            * len(SITES)
            * len(SIGNED_METRICS),
            "retention_family_size_per_model": 6,
            "deformation_family_size_per_model": 4,
            "bootstrap_draws": draws,
            "exact_sign_flip_assignments_per_test": 2 ** len(confirmation_seeds),
            "selection_file_written": True,
        }
    )
    if not (
        audit["layerwise_rows"] == audit["expected_layerwise_rows"]
        and audit["window_rows"] == audit["expected_window_rows"]
    ):
        audit["status"] = "FAIL"
    (output / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if audit["status"] != "PASS":
        raise RuntimeError("State-retention analysis output audit failed")


if __name__ == "__main__":
    main()
