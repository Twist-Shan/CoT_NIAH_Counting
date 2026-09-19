from __future__ import annotations

"""Test whether V4.4 prompt needle-span states have counter-like geometry.

The analysis is deliberately stricter than asking whether count can be decoded.
It measures ordinal geometry, approximate additive updates, held-out-seed
stability, and robustness after removing a cubic function of absolute token
position.  Discovery seeds (1234--1253) fit every map; confirmation seeds
(1254--1263) are used only for evaluation.

The source capture contains N=10 prompts.  Consequently this script can test
running-index structure but cannot by itself prove invariance across different
final needle counts N.  That limitation is recorded in the audit.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


EPS = 1e-12
SUPPORTS = ("endpoint", "last4_mean", "span_mean", "interior_mean")
SUPPORT_LABELS = {
    "endpoint": "needle endpoint",
    "last4_mean": "last 4 span tokens",
    "span_mean": "whole-span mean",
    "interior_mean": "span interior mean",
}
SELECTED_LAYERS = {"Qwen3-8B": 8, "Gemma4-E4B": 9}
COLORS = {
    "endpoint": "#1b9e77",
    "last4_mean": "#d95f02",
    "span_mean": "#7570b3",
    "interior_mean": "#666666",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / max(denominator, EPS))


def count_centroids(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    levels = np.arange(1, 11)
    if set(np.unique(y).tolist()) != set(levels.tolist()):
        raise ValueError(f"Expected count levels 1--10, got {sorted(np.unique(y).tolist())}")
    return np.stack([x[y == level].mean(axis=0, dtype=np.float64) for level in levels])


def trajectory_geometry(discovery: np.ndarray, confirmation: np.ndarray) -> dict[str, float]:
    """Geometry of ten count centroids in the original hidden-state metric."""

    levels = np.arange(1, 11, dtype=np.float64)
    centered_levels = levels - levels.mean()
    centered = discovery - discovery.mean(axis=0, keepdims=True)
    slope = (centered_levels[:, None] * centered).sum(axis=0) / float(np.square(centered_levels).sum())
    fitted = discovery.mean(axis=0, keepdims=True) + centered_levels[:, None] * slope[None, :]
    line_r2 = 1.0 - float(np.square(discovery - fitted).sum()) / max(float(np.square(centered).sum()), EPS)

    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    centroid_variance = np.square(singular)
    rank3_capture = float(centroid_variance[:3].sum() / max(float(centroid_variance.sum()), EPS))

    gaps: list[int] = []
    distances: list[float] = []
    for left in range(10):
        for right in range(left + 1, 10):
            gaps.append(right - left)
            distances.append(float(np.linalg.norm(discovery[right] - discovery[left])))
    distance_gap_rho = float(spearmanr(gaps, distances).statistic)

    steps = np.diff(discovery, axis=0)
    step_lengths = np.linalg.norm(steps, axis=1)
    unit_steps = steps / np.maximum(step_lengths[:, None], EPS)
    pairwise_step_cosines = unit_steps @ unit_steps.T
    upper = pairwise_step_cosines[np.triu_indices(len(steps), k=1)]
    mean_step = steps.mean(axis=0)
    cosine_to_mean = np.asarray([safe_cosine(step, mean_step) for step in steps])
    second_difference = np.diff(steps, axis=0)

    confirmation_steps = np.diff(confirmation, axis=0)
    split_step_cosines = np.asarray(
        [safe_cosine(discovery_step, confirmation_step) for discovery_step, confirmation_step in zip(steps, confirmation_steps)]
    )
    return {
        "trajectory_line_r2": float(line_r2),
        "centroid_rank3_capture": rank3_capture,
        "centroid_distance_vs_count_gap_spearman": distance_gap_rho,
        "adjacent_step_pairwise_cosine_mean": float(upper.mean()),
        "adjacent_step_cosine_to_mean_mean": float(cosine_to_mean.mean()),
        "adjacent_step_length_cv": float(step_lengths.std(ddof=1) / max(float(step_lengths.mean()), EPS)),
        "second_difference_over_step_norm": float(
            np.linalg.norm(second_difference, axis=1).mean() / max(float(step_lengths.mean()), EPS)
        ),
        "discovery_confirmation_same_step_cosine_mean": float(split_step_cosines.mean()),
        "discovery_confirmation_same_step_cosine_min": float(split_step_cosines.min()),
    }


def decode(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    components: int = 32,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    n_components = min(components, len(x_train) - 1, x_train.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=20260814)
    z_train = pca.fit_transform(x_train.astype(np.float32))
    z_test = pca.transform(x_test.astype(np.float32))
    scaler = StandardScaler()
    z_train_scaled = scaler.fit_transform(z_train)
    z_test_scaled = scaler.transform(z_test)

    ridge = Ridge(alpha=1.0)
    ridge.fit(z_train_scaled, y_train)
    continuous = ridge.predict(z_test_scaled)
    rounded = np.clip(np.rint(continuous), 1, 10).astype(int)

    centroids = np.stack([z_train_scaled[y_train == count].mean(axis=0) for count in range(1, 11)])
    squared_distances = np.square(z_test_scaled[:, None, :] - centroids[None, :, :]).sum(axis=2)
    nearest = 1 + np.argmin(squared_distances, axis=1)
    metrics = {
        "ridge_r2": float(r2_score(y_test, continuous)),
        "ridge_mad": float(np.mean(np.abs(continuous - y_test))),
        "ridge_spearman": float(spearmanr(y_test, continuous).statistic),
        "ridge_rounded_exact_accuracy": float(np.mean(rounded == y_test)),
        "nearest_centroid_exact_accuracy": float(np.mean(nearest == y_test)),
        "nearest_centroid_mad": float(np.mean(np.abs(nearest - y_test))),
        "pca32_variance_capture": float(pca.explained_variance_ratio_.sum()),
    }
    artifacts = {
        "continuous": continuous,
        "nearest": nearest,
        "pca_components": pca.components_,
        "pca_mean": pca.mean_,
    }
    return metrics, artifacts


def position_baseline(
    position_train: np.ndarray,
    y_train: np.ndarray,
    position_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    model = make_pipeline(
        PolynomialFeatures(degree=3, include_bias=False),
        StandardScaler(),
        Ridge(alpha=1.0),
    )
    model.fit(position_train[:, None], y_train)
    prediction = model.predict(position_test[:, None])
    return {
        "position_cubic_r2": float(r2_score(y_test, prediction)),
        "position_cubic_mad": float(np.mean(np.abs(prediction - y_test))),
        "position_count_spearman": float(spearmanr(position_test, y_test).statistic),
    }


def position_residuals(
    x_train: np.ndarray,
    x_test: np.ndarray,
    position_train: np.ndarray,
    position_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_design = np.stack(
        [np.ones_like(position_train), position_train, np.square(position_train), np.power(position_train, 3)], axis=1
    ).astype(np.float64)
    test_design = np.stack(
        [np.ones_like(position_test), position_test, np.square(position_test), np.power(position_test, 3)], axis=1
    ).astype(np.float64)
    coefficient = np.linalg.lstsq(train_design, x_train.astype(np.float64), rcond=None)[0]
    return (
        (x_train.astype(np.float64) - train_design @ coefficient).astype(np.float32),
        (x_test.astype(np.float64) - test_design @ coefficient).astype(np.float32),
    )


def per_seed_order_metrics(
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed_test: np.ndarray,
    discovery_centroids: np.ndarray,
) -> dict[str, float]:
    levels = np.arange(1, 11, dtype=np.float64)
    centered_levels = levels - levels.mean()
    centered_centroids = discovery_centroids - discovery_centroids.mean(axis=0, keepdims=True)
    slope = (centered_levels[:, None] * centered_centroids).sum(axis=0)
    slope /= max(float(np.linalg.norm(slope)), EPS)
    projections = x_test.astype(np.float64) @ slope
    seed_rho: list[float] = []
    adjacent_positive: list[float] = []
    for seed in sorted(np.unique(seed_test).tolist()):
        mask = seed_test == seed
        order = np.argsort(y_test[mask])
        values = projections[mask][order]
        gold = y_test[mask][order]
        seed_rho.append(float(spearmanr(gold, values).statistic))
        adjacent_positive.append(float(np.mean(np.diff(values) > 0)))
    return {
        "confirmation_seed_projection_spearman_mean": float(np.mean(seed_rho)),
        "confirmation_seed_projection_spearman_min": float(np.min(seed_rho)),
        "confirmation_adjacent_increment_positive_fraction": float(np.mean(adjacent_positive)),
    }


def extract_model(capture_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: int(row["seed"]))
    layers = np.asarray(rows[0]["layers"], dtype=int)
    seeds = np.asarray([int(row["seed"]) for row in rows], dtype=int)
    splits = np.asarray([str(row["split"]) for row in rows])
    pooled: dict[str, np.ndarray] = {}
    endpoint_positions = np.empty((len(rows), 10), dtype=np.float64)

    for seed_index, row in enumerate(rows):
        with np.load(capture_root / row["path"], allow_pickle=False) as payload:
            states = payload["states"]
            shard_layers = np.asarray(payload["layer_indices"], dtype=int)
            if not np.array_equal(shard_layers, layers):
                raise RuntimeError(f"Layer mismatch in {row['path']}")
            categories = np.asarray(payload["categories"]).astype(str)
            occurrence = np.asarray(payload["occurrence_index"], dtype=int)
            positions = np.asarray(payload["positions"], dtype=int)
            if not pooled:
                hidden = int(states.shape[2])
                pooled = {
                    support: np.empty((len(rows), 10, len(layers), hidden), dtype=np.float32)
                    for support in SUPPORTS
                }
            for count in range(1, 11):
                span_indices = np.flatnonzero(
                    (occurrence == count) & np.isin(categories, ["needle_interior", "needle_endpoint"])
                )
                endpoint_indices = np.flatnonzero((occurrence == count) & (categories == "needle_endpoint"))
                interior_indices = np.flatnonzero((occurrence == count) & (categories == "needle_interior"))
                if len(endpoint_indices) != 1 or not len(interior_indices):
                    raise RuntimeError(
                        f"Expected one endpoint and nonempty interior for seed={row['seed']} count={count}; "
                        f"got endpoint={len(endpoint_indices)}, interior={len(interior_indices)}"
                    )
                span_indices = span_indices[np.argsort(positions[span_indices])]
                last4_indices = span_indices[-min(4, len(span_indices)) :]
                pooled["endpoint"][seed_index, count - 1] = states[:, endpoint_indices[0], :].astype(np.float32)
                pooled["last4_mean"][seed_index, count - 1] = states[:, last4_indices, :].mean(
                    axis=1, dtype=np.float32
                )
                pooled["span_mean"][seed_index, count - 1] = states[:, span_indices, :].mean(
                    axis=1, dtype=np.float32
                )
                pooled["interior_mean"][seed_index, count - 1] = states[:, interior_indices, :].mean(
                    axis=1, dtype=np.float32
                )
                endpoint_positions[seed_index, count - 1] = float(positions[endpoint_indices[0]]) / max(
                    float(row["sequence_length"] - 1), 1.0
                )
        print(f"[counter-properties:extract] {row['model_label']} seed={row['seed']}", flush=True)
    return {
        "layers": layers,
        "seeds": seeds,
        "splits": splits,
        "pooled": pooled,
        "endpoint_positions": endpoint_positions,
    }


def plot_layerwise(frame: pd.DataFrame, path: Path) -> None:
    metrics = (
        ("trajectory_line_r2", "centroid trajectory linear $R^2$"),
        ("centroid_distance_vs_count_gap_spearman", "distance vs. count gap Spearman $\\rho$"),
        ("ridge_r2", "held-out count ridge $R^2$"),
    )
    models = [model for model in ("Qwen3-8B", "Gemma4-E4B") if model in set(frame["model_label"])]
    figure, axes = plt.subplots(len(models), len(metrics), figsize=(15, 4.2 * len(models)), squeeze=False)
    for row_index, model in enumerate(models):
        model_frame = frame[frame["model_label"] == model]
        for column_index, (metric, label) in enumerate(metrics):
            axis = axes[row_index, column_index]
            for support in SUPPORTS:
                part = model_frame[model_frame["support"] == support].sort_values("layer")
                axis.plot(
                    part["layer"],
                    part[metric],
                    marker="o",
                    markersize=3.5,
                    linewidth=1.6,
                    label=SUPPORT_LABELS[support],
                    color=COLORS[support],
                )
            axis.axhline(0.0, color="#bbbbbb", linewidth=0.8)
            axis.set_xlabel("zero-based transformer layer")
            axis.set_ylabel(label)
            axis.set_title(f"{model}: {label}")
            axis.grid(alpha=0.2)
            if row_index == 0 and column_index == 0:
                axis.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_selected_manifolds(
    selected: dict[tuple[str, str], dict[str, np.ndarray]],
    path: Path,
    projection_path: Path,
) -> tuple[bool, str | None]:
    models = [model for model in ("Qwen3-8B", "Gemma4-E4B") if (model, "endpoint") in selected]
    supports = ("endpoint", "span_mean")
    projection_rows: list[dict[str, Any]] = []
    for model in models:
        for support in supports:
            payload = selected[(model, support)]
            x_train = payload["x_train"]
            x_test = payload["x_test"]
            y_test = payload["y_test"]
            seed_test = payload["seed_test"]
            pca = PCA(n_components=3, svd_solver="randomized", random_state=20260814)
            pca.fit(x_train.astype(np.float32))
            projected = pca.transform(x_test.astype(np.float32))
            for index in range(len(projected)):
                projection_rows.append(
                    {
                        "model_label": model,
                        "layer": SELECTED_LAYERS[model],
                        "support": support,
                        "seed": int(seed_test[index]),
                        "running_index": int(y_test[index]),
                        "pc1": float(projected[index, 0]),
                        "pc2": float(projected[index, 1]),
                        "pc3": float(projected[index, 2]),
                        "pc1_variance_fraction": float(pca.explained_variance_ratio_[0]),
                        "pc2_variance_fraction": float(pca.explained_variance_ratio_[1]),
                        "pc3_variance_fraction": float(pca.explained_variance_ratio_[2]),
                    }
                )
    projections = pd.DataFrame(projection_rows)
    projections.to_csv(projection_path, index=False)

    # Some compute images accidentally combine a pip Matplotlib with a system
    # mpl_toolkits.  Preserve the numerical analysis and projection table even
    # if only the optional 3-D renderer is broken; the same CSV can be rendered
    # in any clean Matplotlib environment without recomputing activations.
    try:
        figure = plt.figure(figsize=(13, 5.5 * len(models)))
        plot_index = 1
        for model in models:
            for support in supports:
                part = projections[
                    (projections["model_label"] == model) & (projections["support"] == support)
                ]
                axis = figure.add_subplot(len(models), len(supports), plot_index, projection="3d")
                plot_index += 1
                scatter = axis.scatter(
                    part["pc1"], part["pc2"], part["pc3"],
                    c=part["running_index"], cmap="viridis", s=20, alpha=0.65,
                )
                centroids = (
                    part.groupby("running_index")[["pc1", "pc2", "pc3"]]
                    .mean()
                    .sort_index()
                    .to_numpy()
                )
                axis.plot(centroids[:, 0], centroids[:, 1], centroids[:, 2], color="black", marker="o", linewidth=2)
                first = part.iloc[0]
                variance = 100.0 * np.asarray(
                    [first["pc1_variance_fraction"], first["pc2_variance_fraction"], first["pc3_variance_fraction"]]
                )
                axis.set_xlabel(f"PC1 ({variance[0]:.1f}%)")
                axis.set_ylabel(f"PC2 ({variance[1]:.1f}%)")
                axis.set_zlabel(f"PC3 ({variance[2]:.1f}%)")
                axis.set_title(f"{model} L{SELECTED_LAYERS[model]}: {SUPPORT_LABELS[support]}")
                figure.colorbar(scatter, ax=axis, shrink=0.60, pad=0.12, label="running index")
        figure.tight_layout()
        figure.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(figure)
        return True, None
    except Exception as error:  # pragma: no cover - environment-specific renderer failure
        plt.close("all")
        return False, f"{type(error).__name__}: {error}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["Qwen3-8B", "Gemma4-E4B"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    index = read_jsonl(args.capture_root / "capture_index.jsonl")
    metric_rows: list[dict[str, Any]] = []
    selected: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    source_rows = 0
    for model in args.models:
        rows = [row for row in index if str(row["model_label"]) == model]
        if len(rows) != 30:
            raise RuntimeError(f"Expected 30 capture shards for {model}, got {len(rows)}")
        extracted = extract_model(args.capture_root, rows)
        source_rows += len(rows)
        layers = extracted["layers"]
        seeds = extracted["seeds"]
        splits = extracted["splits"]
        pooled = extracted["pooled"]
        positions = extracted["endpoint_positions"]
        y_grid = np.broadcast_to(np.arange(1, 11, dtype=int)[None, :], positions.shape)
        seed_grid = np.broadcast_to(seeds[:, None], positions.shape)
        split_grid = np.broadcast_to(splits[:, None], positions.shape)
        y = y_grid.reshape(-1)
        seed = seed_grid.reshape(-1)
        split = split_grid.reshape(-1)
        position = positions.reshape(-1)
        train = split == "discovery"
        test = split == "confirmation"
        if (int(train.sum()), int(test.sum())) != (200, 100):
            raise RuntimeError(f"Unexpected split sizes for {model}: train={train.sum()} test={test.sum()}")

        for layer_index, layer in enumerate(layers.tolist()):
            for support in SUPPORTS:
                x = pooled[support][:, :, layer_index, :].reshape(len(y), -1)
                discovery_centroids = count_centroids(x[train], y[train])
                confirmation_centroids = count_centroids(x[test], y[test])
                decoding, _ = decode(x[train], y[train], x[test], y[test])
                residual_train, residual_test = position_residuals(
                    x[train], x[test], position[train], position[test]
                )
                residual_decoding, _ = decode(
                    residual_train, y[train], residual_test, y[test]
                )
                metric_rows.append(
                    {
                        "model_label": model,
                        "layer": int(layer),
                        "support": support,
                        "support_definition": SUPPORT_LABELS[support],
                        "rows_discovery": int(train.sum()),
                        "rows_confirmation": int(test.sum()),
                        **decoding,
                        **{f"position_residual_{key}": value for key, value in residual_decoding.items()},
                        **position_baseline(position[train], y[train], position[test], y[test]),
                        **trajectory_geometry(discovery_centroids, confirmation_centroids),
                        **per_seed_order_metrics(x[test], y[test], seed[test], discovery_centroids),
                    }
                )
                if int(layer) == SELECTED_LAYERS.get(model) and support in {"endpoint", "span_mean"}:
                    selected[(model, support)] = {
                        "x_train": x[train].copy(),
                        "x_test": x[test].copy(),
                        "y_test": y[test].copy(),
                        "seed_test": seed[test].copy(),
                    }
                print(f"[counter-properties] {model} L{layer} {support}", flush=True)
        del extracted, pooled

    metrics = pd.DataFrame(metric_rows).sort_values(["model_label", "layer", "support"])
    metrics.to_csv(args.output / "counter_property_metrics.csv", index=False)
    plot_layerwise(metrics, args.output / "counter_properties_by_layer.png")
    rendered_3d, render_error = plot_selected_manifolds(
        selected,
        args.output / "selected_counter_manifolds_3d.png",
        args.output / "selected_counter_manifold_projections.csv",
    )
    selected_rows = metrics[
        metrics.apply(lambda row: int(row["layer"]) == SELECTED_LAYERS.get(str(row["model_label"]), -1), axis=1)
    ]
    selected_rows.to_csv(args.output / "selected_layer_counter_properties.csv", index=False)

    required_finite = [
        "ridge_r2",
        "ridge_mad",
        "nearest_centroid_exact_accuracy",
        "position_residual_ridge_r2",
        "trajectory_line_r2",
        "centroid_distance_vs_count_gap_spearman",
        "adjacent_step_pairwise_cosine_mean",
        "discovery_confirmation_same_step_cosine_mean",
    ]
    if len(metrics) != sum(len(row["layers"]) for row in index if int(row["seed"]) == 1234) * len(SUPPORTS):
        raise RuntimeError(f"Unexpected metric row count: {len(metrics)}")
    if not np.isfinite(metrics[required_finite].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite required counter-property metric")
    audit = {
        "schema_version": "realistic_niah_v4_4_prompt_counter_properties_v1",
        "status": "PASS",
        "source_capture": str(args.capture_root),
        "source_capture_rows": source_rows,
        "models": args.models,
        "supports": {
            "endpoint": "single final token of each active needle span",
            "last4_mean": "unweighted mean of the final min(4, span_length) tokens, including the endpoint",
            "span_mean": "unweighted mean of every token in the active needle span",
            "interior_mean": "unweighted mean of every active-needle token except the endpoint",
        },
        "fit_split": "discovery seeds 1234--1253",
        "evaluation_split": "confirmation seeds 1254--1263",
        "rows_per_model_layer_support": {"discovery": 200, "confirmation": 100},
        "decoding": "PCA32 fit on discovery only; StandardScaler and ridge/nearest-centroid fit on discovery only",
        "ordered_geometry": "Spearman correlation across 45 centroid pairs between Euclidean distance and absolute count gap",
        "additive_geometry": "line R2, adjacent-step cosine alignment, step-length CV, and normalized second differences of ten discovery centroids",
        "cross_split_stability": "cosine between each of nine discovery centroid steps and the corresponding confirmation step",
        "position_control": "absolute endpoint token position divided by sequence length; cubic position-only ridge and hidden-state residualization against [1,p,p^2,p^3] are fit on discovery only",
        "selected_visualization_layers": SELECTED_LAYERS,
        "selected_3d_rendered_on_analysis_host": rendered_3d,
        "selected_3d_render_error": render_error,
        "critical_scope_limit": "All source prompts have final count N=10. Running index is therefore correlated with absolute ordinal/position slot; position residualization reduces but does not eliminate this identification problem. Cross-final-count prefix invariance is not tested.",
        "interpretation_rule": "Decodability alone is insufficient. Counter-like evidence requires ordered distance, approximately aligned updates, held-out-seed stability, and information beyond the absolute-position control. Causal necessity is evaluated separately by interventions.",
        "metric_rows": len(metrics),
        "files": [
            "counter_property_metrics.csv",
            "selected_layer_counter_properties.csv",
            "counter_properties_by_layer.png",
            "selected_counter_manifolds_3d.png",
            "selected_counter_manifold_projections.csv",
        ],
    }
    atomic_json(args.output / "counter_property_analysis_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
