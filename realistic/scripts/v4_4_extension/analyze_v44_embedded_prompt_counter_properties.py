from __future__ import annotations

"""Measure counter-like geometry in the frozen V4.4 prompt PCA projections.

This is a lightweight, reproducible fallback for the full-hidden-state analysis:
the report HTML already contains discovery-fitted PC1--PC3 coordinates for all
30 canonical seeds, all ten running indices, and every transformer layer.  We
use those frozen coordinates without refitting the PCA.  A separate all-token
projection file supplies a grouped-by-seed absolute-position control.

The resulting quantities are representational, not causal.  All prompts have
final count N=10, so this analysis also cannot identify prefix-count invariance
across different final counts.
"""

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


EPS = 1e-12
MODELS = ("Qwen3-8B", "Gemma4-E4B")
SELECTED_LAYERS = {"Qwen3-8B": 8, "Gemma4-E4B": 9}
COLORS = {"Qwen3-8B": "#0f766e", "Gemma4-E4B": "#7c3aed"}


def extract_embedded_json(document: str, variable_name: str) -> dict[str, Any]:
    marker = f"const {variable_name}="
    start = document.find(marker)
    if start < 0:
        raise RuntimeError(f"Could not locate embedded {variable_name}")
    start += len(marker)
    end = document.find(";\nconst ", start)
    if end < 0:
        end = document.find(";</script>", start)
    if end < 0:
        raise RuntimeError(f"Could not locate end of embedded {variable_name}")
    return json.loads(document[start:end])


def safe_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / max(denominator, EPS))


def centroids(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack([x[y == count].mean(axis=0) for count in range(1, 11)])


def trajectory_metrics(discovery: np.ndarray, confirmation: np.ndarray) -> dict[str, float]:
    levels = np.arange(1, 11, dtype=np.float64)
    centered_levels = levels - levels.mean()
    mean = discovery.mean(axis=0, keepdims=True)
    centered = discovery - mean
    slope = (centered_levels[:, None] * centered).sum(axis=0) / float(
        np.square(centered_levels).sum()
    )
    fitted = mean + centered_levels[:, None] * slope[None, :]
    line_r2 = 1.0 - float(np.square(discovery - fitted).sum()) / max(
        float(np.square(centered).sum()), EPS
    )

    gaps: list[int] = []
    distances: list[float] = []
    for left in range(10):
        for right in range(left + 1, 10):
            gaps.append(right - left)
            distances.append(float(np.linalg.norm(discovery[right] - discovery[left])))

    steps = np.diff(discovery, axis=0)
    step_lengths = np.linalg.norm(steps, axis=1)
    unit_steps = steps / np.maximum(step_lengths[:, None], EPS)
    pairwise_cosines = unit_steps @ unit_steps.T
    upper = pairwise_cosines[np.triu_indices(len(steps), k=1)]
    mean_step = steps.mean(axis=0)
    cosine_to_mean = [safe_cosine(step, mean_step) for step in steps]
    confirmation_steps = np.diff(confirmation, axis=0)
    same_step_cosines = [
        safe_cosine(left, right)
        for left, right in zip(steps, confirmation_steps, strict=True)
    ]
    second_differences = np.diff(steps, axis=0)
    return {
        "trajectory_line_r2": float(line_r2),
        "centroid_distance_vs_count_gap_spearman": float(
            spearmanr(gaps, distances).statistic
        ),
        "adjacent_step_pairwise_cosine_mean": float(np.mean(upper)),
        "adjacent_step_cosine_to_mean_mean": float(np.mean(cosine_to_mean)),
        "adjacent_step_length_cv": float(
            np.std(step_lengths, ddof=1) / max(float(np.mean(step_lengths)), EPS)
        ),
        "second_difference_over_step_norm": float(
            np.mean(np.linalg.norm(second_differences, axis=1))
            / max(float(np.mean(step_lengths)), EPS)
        ),
        "discovery_confirmation_same_step_cosine_mean": float(
            np.mean(same_step_cosines)
        ),
        "discovery_confirmation_same_step_cosine_min": float(
            np.min(same_step_cosines)
        ),
    }


def held_out_metrics(
    x_discovery: np.ndarray,
    y_discovery: np.ndarray,
    x_confirmation: np.ndarray,
    y_confirmation: np.ndarray,
    seed_confirmation: np.ndarray,
) -> dict[str, float]:
    scaler = StandardScaler()
    train = scaler.fit_transform(x_discovery)
    test = scaler.transform(x_confirmation)
    ridge = Ridge(alpha=1.0).fit(train, y_discovery)
    prediction = ridge.predict(test)

    discovery_centroids = np.stack(
        [train[y_discovery == count].mean(axis=0) for count in range(1, 11)]
    )
    distances = np.square(
        test[:, None, :] - discovery_centroids[None, :, :]
    ).sum(axis=2)
    nearest = 1 + np.argmin(distances, axis=1)

    levels = np.arange(1, 11, dtype=np.float64)
    centered_levels = levels - levels.mean()
    centered_centroids = discovery_centroids - discovery_centroids.mean(
        axis=0, keepdims=True
    )
    direction = (centered_levels[:, None] * centered_centroids).sum(axis=0)
    direction /= max(float(np.linalg.norm(direction)), EPS)
    projections = test @ direction
    seed_rhos: list[float] = []
    positive_fractions: list[float] = []
    for seed in sorted(np.unique(seed_confirmation).tolist()):
        mask = seed_confirmation == seed
        order = np.argsort(y_confirmation[mask])
        ordered_y = y_confirmation[mask][order]
        ordered_projection = projections[mask][order]
        seed_rhos.append(float(spearmanr(ordered_y, ordered_projection).statistic))
        positive_fractions.append(float(np.mean(np.diff(ordered_projection) > 0)))

    return {
        "frozen_pc3_ridge_r2": float(r2_score(y_confirmation, prediction)),
        "frozen_pc3_ridge_mad": float(np.mean(np.abs(prediction - y_confirmation))),
        "frozen_pc3_nearest_centroid_accuracy": float(
            np.mean(nearest == y_confirmation)
        ),
        "frozen_pc3_nearest_centroid_mad": float(
            np.mean(np.abs(nearest - y_confirmation))
        ),
        "confirmation_seed_projection_spearman_mean": float(np.mean(seed_rhos)),
        "confirmation_seed_projection_spearman_min": float(np.min(seed_rhos)),
        "confirmation_adjacent_increment_positive_fraction": float(
            np.mean(positive_fractions)
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped_position_control(rows: list[dict[str, str]]) -> dict[str, float]:
    position = np.asarray([float(row["position"]) for row in rows], dtype=np.float64)
    x = np.asarray(
        [[float(row["pc1"]), float(row["pc2"]), float(row["pc3"])] for row in rows],
        dtype=np.float64,
    )
    y = np.asarray([int(row["prefix_count"]) for row in rows], dtype=np.float64)
    groups = np.asarray([int(row["seed"]) for row in rows], dtype=int)
    predictions = {
        "pc3": np.empty(len(rows), dtype=np.float64),
        "position": np.empty(len(rows), dtype=np.float64),
        "residual": np.empty(len(rows), dtype=np.float64),
    }
    splitter = GroupKFold(n_splits=5)
    for train_index, test_index in splitter.split(x, y, groups):
        raw_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        raw_model.fit(x[train_index], y[train_index])
        predictions["pc3"][test_index] = raw_model.predict(x[test_index])

        position_model = make_pipeline(
            PolynomialFeatures(degree=3, include_bias=False),
            StandardScaler(),
            Ridge(alpha=1.0),
        )
        position_model.fit(position[train_index, None], y[train_index])
        predictions["position"][test_index] = position_model.predict(
            position[test_index, None]
        )

        position_mean = float(position[train_index].mean())
        position_scale = max(float(position[train_index].std()), EPS)
        train_p = (position[train_index] - position_mean) / position_scale
        test_p = (position[test_index] - position_mean) / position_scale
        train_design = np.stack(
            [np.ones_like(train_p), train_p, np.square(train_p), np.power(train_p, 3)],
            axis=1,
        )
        test_design = np.stack(
            [np.ones_like(test_p), test_p, np.square(test_p), np.power(test_p, 3)],
            axis=1,
        )
        coefficient = np.linalg.lstsq(train_design, x[train_index], rcond=None)[0]
        train_residual = x[train_index] - train_design @ coefficient
        test_residual = x[test_index] - test_design @ coefficient
        residual_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        residual_model.fit(train_residual, y[train_index])
        predictions["residual"][test_index] = residual_model.predict(test_residual)

    return {
        "position_control_rows": int(len(rows)),
        "position_count_spearman": float(spearmanr(position, y).statistic),
        "all_token_pc3_grouped_ridge_r2": float(r2_score(y, predictions["pc3"])),
        "all_token_pc3_grouped_ridge_mad": float(
            np.mean(np.abs(predictions["pc3"] - y))
        ),
        "position_cubic_grouped_ridge_r2": float(
            r2_score(y, predictions["position"])
        ),
        "position_cubic_grouped_ridge_mad": float(
            np.mean(np.abs(predictions["position"] - y))
        ),
        "position_residual_pc3_grouped_ridge_r2": float(
            r2_score(y, predictions["residual"])
        ),
        "position_residual_pc3_grouped_ridge_mad": float(
            np.mean(np.abs(predictions["residual"] - y))
        ),
    }


def make_plot(rows: list[dict[str, Any]], path: Path) -> None:
    metrics = (
        ("trajectory_line_r2", "centroid line $R^2$"),
        ("centroid_distance_vs_count_gap_spearman", "distance-gap Spearman $\\rho$"),
        ("adjacent_step_pairwise_cosine_mean", "mean pairwise step cosine"),
        (
            "discovery_confirmation_same_step_cosine_mean",
            "discovery-confirmation step cosine",
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 7.5), squeeze=False)
    for axis, (metric, label) in zip(axes.flat, metrics, strict=True):
        for model in MODELS:
            part = sorted(
                [row for row in rows if row["model_label"] == model],
                key=lambda row: int(row["layer"]),
            )
            axis.plot(
                [int(row["layer"]) for row in part],
                [float(row[metric]) for row in part],
                label=model,
                color=COLORS[model],
                marker="o",
                markersize=2.7,
                linewidth=1.5,
            )
            selected = SELECTED_LAYERS[model]
            selected_row = next(row for row in part if int(row["layer"]) == selected)
            axis.scatter(
                [selected], [float(selected_row[metric])], s=55,
                color=COLORS[model], edgecolor="white", linewidth=1.2, zorder=5,
            )
        axis.axhline(0.0, color="#98a2b3", linewidth=0.8, linestyle="--")
        axis.set_xlabel("zero-based transformer layer")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Prompt needle-end counter properties in frozen PC1--PC3")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--all-token-projections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    prompt = extract_embedded_json(
        args.base_report.read_text(encoding="utf-8"), "PROMPT_DATA"
    )
    metric_rows: list[dict[str, Any]] = []
    for key, payload in prompt.items():
        model, layer_text = key.split("|")
        if model not in MODELS:
            continue
        layer = int(layer_text)
        raw = payload["rows"]
        x = np.asarray([row[6:9] for row in raw], dtype=np.float64)
        y = np.asarray([int(row[5]) for row in raw], dtype=int)
        seed = np.asarray([int(row[0]) for row in raw], dtype=int)
        split = np.asarray([str(row[1]) for row in raw])
        discovery = split == "discovery"
        confirmation = split == "confirmation"
        if (int(discovery.sum()), int(confirmation.sum())) != (200, 100):
            raise RuntimeError(
                f"Unexpected split sizes for {key}: "
                f"{discovery.sum()}/{confirmation.sum()}"
            )
        discovery_centroids = centroids(x[discovery], y[discovery])
        confirmation_centroids = centroids(x[confirmation], y[confirmation])
        metric_rows.append(
            {
                "model_label": model,
                "layer": layer,
                "rows_discovery": int(discovery.sum()),
                "rows_confirmation": int(confirmation.sum()),
                "frozen_pc1_variance_fraction": float(payload["explained_variance_ratio"][0]),
                "frozen_pc2_variance_fraction": float(payload["explained_variance_ratio"][1]),
                "frozen_pc3_variance_fraction": float(payload["explained_variance_ratio"][2]),
                "frozen_pc3_total_variance_fraction": float(
                    sum(payload["explained_variance_ratio"][:3])
                ),
                **trajectory_metrics(discovery_centroids, confirmation_centroids),
                **held_out_metrics(
                    x[discovery], y[discovery], x[confirmation], y[confirmation],
                    seed[confirmation],
                ),
            }
        )

    all_token_rows: list[dict[str, str]] = []
    with gzip.open(
        args.all_token_projections, "rt", encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            model = row["model_label"]
            if (
                model in MODELS
                and row["category"] == "needle_endpoint"
                and int(row["layer"]) == SELECTED_LAYERS[model]
            ):
                all_token_rows.append(row)

    selected_rows: list[dict[str, Any]] = []
    for model in MODELS:
        selected = next(
            row
            for row in metric_rows
            if row["model_label"] == model
            and int(row["layer"]) == SELECTED_LAYERS[model]
        )
        controls = grouped_position_control(
            [row for row in all_token_rows if row["model_label"] == model]
        )
        selected_rows.append({**selected, **controls})

    metric_rows.sort(key=lambda row: (row["model_label"], int(row["layer"])))
    write_csv(args.output / "counter_property_metrics_by_layer.csv", metric_rows)
    write_csv(args.output / "selected_layer_counter_properties.csv", selected_rows)
    make_plot(metric_rows, args.output / "counter_properties_by_layer.png")

    required = (
        "trajectory_line_r2",
        "centroid_distance_vs_count_gap_spearman",
        "adjacent_step_pairwise_cosine_mean",
        "discovery_confirmation_same_step_cosine_mean",
        "frozen_pc3_ridge_r2",
    )
    if len(metric_rows) != 78:
        raise RuntimeError(f"Expected 78 model-layer rows, got {len(metric_rows)}")
    if not np.isfinite(
        np.asarray([[float(row[key]) for key in required] for row in metric_rows])
    ).all():
        raise RuntimeError("Non-finite required counter metric")
    audit = {
        "schema_version": "realistic_niah_v4_4_embedded_prompt_counter_properties_v1",
        "status": "PASS",
        "base_report": str(args.base_report),
        "all_token_projection_source": str(args.all_token_projections),
        "models": list(MODELS),
        "metric_rows": len(metric_rows),
        "fit_split": "discovery seeds 1234--1253",
        "evaluation_split": "confirmation seeds 1254--1263",
        "projection_scope": "All geometry and decoding use the discovery-fitted frozen PC1--PC3 embedded in the V4.4 report; PCA is not refit.",
        "position_control": "The all-token confirmation projection is evaluated by 5-fold GroupKFold over seed. Cubic absolute position is fit inside each training fold; PC1--PC3 are residualized against [1,p,p^2,p^3] using training-fold coefficients before ridge decoding.",
        "interpretation_rule": "Decodability alone is not called a counter. Counter-like support additionally requires ordered centroid distances, approximately aligned adjacent updates, and discovery-confirmation stability.",
        "scope_limits": [
            "Projection-level representation analysis only; no causal necessity follows.",
            "The full-hidden-state counter-property job did not complete on the released compute node.",
            "All prompts have final count N=10; cross-final-count prefix invariance is untested.",
            "Absolute token position is strongly coupled to running index; the grouped cubic residual is a robustness control, not a complete deconfounder.",
        ],
        "selected_layers": SELECTED_LAYERS,
        "files": [
            "counter_property_metrics_by_layer.csv",
            "selected_layer_counter_properties.csv",
            "counter_properties_by_layer.png",
        ],
    }
    (args.output / "counter_property_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
