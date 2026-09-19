from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from collections import defaultdict
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler


RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stable_rank(centered: np.ndarray) -> float:
    singular = np.linalg.svd(centered, compute_uv=False)
    if not len(singular) or singular[0] <= 1e-12:
        return math.nan
    return float(np.square(singular).sum() / float(singular[0] ** 2))


def rank_capture(matrix: np.ndarray, rank: int = 3) -> float:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = np.square(singular)
    denominator = float(variance.sum())
    return (
        float(variance[: int(rank)].sum() / denominator)
        if denominator > 1e-12
        else math.nan
    )


def eta_squared(matrix: np.ndarray, labels: np.ndarray) -> float:
    global_mean = matrix.mean(axis=0)
    total = float(np.square(matrix - global_mean).sum())
    between = 0.0
    for label in np.unique(labels):
        group = matrix[labels == label]
        between += len(group) * float(np.square(group.mean(axis=0) - global_mean).sum())
    return between / total if total > 1e-12 else math.nan


def cosine_nearest_centroid(
    train: np.ndarray,
    train_labels: np.ndarray,
    test: np.ndarray,
) -> np.ndarray:
    counts = np.asarray(sorted(np.unique(train_labels)), dtype=int)
    centroids = np.stack([train[train_labels == count].mean(axis=0) for count in counts])
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    normalized = test / np.maximum(np.linalg.norm(test, axis=1, keepdims=True), 1e-12)
    return counts[np.argmax(normalized @ centroids.T, axis=1)]


def choose_ridge_alpha(
    matrix: np.ndarray, labels: np.ndarray, seeds: np.ndarray
) -> float:
    losses: dict[float, list[float]] = {alpha: [] for alpha in RIDGE_ALPHAS}
    for heldout_seed in sorted(np.unique(seeds)):
        train = seeds != heldout_seed
        valid = ~train
        if not train.any() or not valid.any():
            continue
        for alpha in RIDGE_ALPHAS:
            prediction = Ridge(alpha=alpha).fit(matrix[train], labels[train]).predict(
                matrix[valid]
            )
            losses[alpha].extend(np.abs(prediction - labels[valid]).tolist())
    return min(RIDGE_ALPHAS, key=lambda alpha: float(np.mean(losses[alpha])))


def max_principal_angle(left: np.ndarray, right: np.ndarray) -> float:
    singular = np.linalg.svd(left @ right.T, compute_uv=False)
    singular = np.clip(singular, -1.0, 1.0)
    return float(np.degrees(np.arccos(singular.min())))


def bootstrap_basis_angle(
    matrix: np.ndarray,
    seeds: np.ndarray,
    reference_components: np.ndarray,
    *,
    draws: int,
    random_seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(int(random_seed))
    unique = np.unique(seeds)
    values: list[float] = []
    for _ in range(int(draws)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(seeds == seed) for seed in sampled])
        if len(indices) < 4:
            continue
        components = PCA(n_components=3).fit(matrix[indices]).components_
        values.append(max_principal_angle(reference_components, components))
    if not values:
        return math.nan, math.nan
    return float(np.median(values)), float(np.quantile(values, 0.95))


def load_bank_states(
    run_root: Path,
    models: Sequence[str],
    head_registry: Mapping[str, Sequence[Sequence[int]]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in models:
        registered_by_layer: dict[int, list[int]] = defaultdict(list)
        if head_registry is not None:
            for layer, head in head_registry[str(model)]:
                registered_by_layer[int(layer)].append(int(head))
        detail_path = run_root / model / "detail.jsonl"
        for row in read_jsonl(detail_path):
            if row["condition"] != "clean" or int(row["patch_layer"]) != -1:
                continue
            state_path = run_root / model / str(row["state_path"])
            payload = torch.load(state_path, map_location="cpu", weights_only=True)
            writes = payload["selected_head_writes"]
            if head_registry is None:
                layer_values = {
                    int(str(key).split(".")[0][1:]): value
                    for key, value in writes.items()
                    if str(key).endswith(".bank_o")
                }
            else:
                layer_values = {}
                for layer, heads in registered_by_layer.items():
                    keys = [f"L{layer}H{head}.o" for head in heads]
                    missing = [key for key in keys if key not in writes]
                    if missing:
                        raise KeyError(
                            f"{state_path} lacks registered head writes: {missing}"
                        )
                    layer_values[layer] = torch.stack(
                        [writes[key].detach().float() for key in keys]
                    ).sum(dim=0)
            for layer, value in layer_values.items():
                rows.append(
                    {
                        "model_label": str(model),
                        "seed": int(row["seed"]),
                        "gold_count": int(row["gold_count"]),
                        "layer": layer,
                        "strict_correct": bool(row["strict_correct"]),
                        "vector": value.detach().float().numpy(),
                    }
                )
    if not rows:
        raise ValueError("No clean broad-bank output states were found")
    return pd.DataFrame(rows)


def geometry_metrics(
    states: pd.DataFrame,
    discovery_seeds: set[int],
    confirmation_seeds: set[int],
    *,
    bootstrap_draws: int,
) -> tuple[pd.DataFrame, dict[tuple[str, int], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    fitted: dict[tuple[str, int], dict[str, Any]] = {}
    for (model, layer), group in states.groupby(["model_label", "layer"], sort=True):
        discovery = group[group["seed"].isin(discovery_seeds)].copy()
        confirmation = group[group["seed"].isin(confirmation_seeds)].copy()
        if discovery.empty or confirmation.empty:
            raise RuntimeError(f"Missing discovery/confirmation rows for {model} L{layer}")
        x_train = np.stack(discovery["vector"].to_list()).astype(np.float64)
        y_train = discovery["gold_count"].to_numpy(dtype=int)
        seed_train = discovery["seed"].to_numpy(dtype=int)
        x_test = np.stack(confirmation["vector"].to_list()).astype(np.float64)
        y_test = confirmation["gold_count"].to_numpy(dtype=int)
        pca = PCA(n_components=3).fit(x_train)
        predictive_pca = PCA(
            n_components=min(32, len(x_train) - 1, x_train.shape[1]),
            svd_solver="randomized",
            random_state=20260813,
        ).fit(x_train)
        predictive_train = predictive_pca.transform(x_train)
        predictive_test = predictive_pca.transform(x_test)
        component_scale = predictive_train.std(axis=0)
        scale_floor = max(1e-10, float(component_scale.max()) * 1e-6)
        retained = component_scale > scale_floor
        if not retained.any():
            retained[0] = True
        predictive_train = predictive_train[:, retained]
        predictive_test = predictive_test[:, retained]
        scaler = StandardScaler().fit(predictive_train)
        predictive_train = scaler.transform(predictive_train)
        predictive_test = scaler.transform(predictive_test)
        centroids = np.stack(
            [x_train[y_train == count].mean(axis=0) for count in sorted(np.unique(y_train))]
        )
        alpha = choose_ridge_alpha(
            predictive_train, y_train.astype(float), seed_train
        )
        ridge = Ridge(alpha=alpha).fit(predictive_train, y_train)
        ridge_prediction = ridge.predict(predictive_test)
        classifier = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=3000,
            random_state=20260813,
        ).fit(predictive_train, y_train)
        exact_prediction = classifier.predict(predictive_test)
        centroid_prediction = cosine_nearest_centroid(
            predictive_train, y_train, predictive_test
        )
        median_angle, p95_angle = bootstrap_basis_angle(
            x_train,
            seed_train,
            pca.components_,
            draws=bootstrap_draws,
            random_seed=20260813 + int(layer),
        )
        silhouette = (
            float(silhouette_score(predictive_test, y_test, metric="cosine"))
            if len(np.unique(y_test)) > 1 and len(x_test) > len(np.unique(y_test))
            else math.nan
        )
        rows.append(
            {
                "model_label": model,
                "layer": int(layer),
                "discovery_rows": int(len(discovery)),
                "confirmation_rows": int(len(confirmation)),
                "rank3_all": rank_capture(x_train, 3),
                "rank3_centroids": rank_capture(centroids, 3),
                "stable_rank": stable_rank(x_train - x_train.mean(axis=0)),
                "ridge_alpha": float(alpha),
                "ridge_r2": float(r2_score(y_test, ridge_prediction)),
                "ridge_mad": float(np.mean(np.abs(ridge_prediction - y_test))),
                "exact_classifier_accuracy": float(
                    accuracy_score(y_test, exact_prediction)
                ),
                "exact_classifier_mad": float(
                    np.mean(np.abs(exact_prediction - y_test))
                ),
                "nearest_centroid_accuracy": float(
                    accuracy_score(y_test, centroid_prediction)
                ),
                "nearest_centroid_mad": float(
                    np.mean(np.abs(centroid_prediction - y_test))
                ),
                "eta2_count": eta_squared(x_test, y_test),
                "cosine_silhouette": silhouette,
                "bootstrap_max_principal_angle_median_deg": median_angle,
                "bootstrap_max_principal_angle_p95_deg": p95_angle,
                "pca_pc1_variance": float(pca.explained_variance_ratio_[0]),
                "pca_pc2_variance": float(pca.explained_variance_ratio_[1]),
                "pca_pc3_variance": float(pca.explained_variance_ratio_[2]),
                "predictive_pca_components_fitted": int(
                    predictive_pca.n_components_
                ),
                "predictive_pca_components_retained": int(retained.sum()),
                "predictive_component_scale_floor": float(scale_floor),
                "predictive_pca_variance_capture": float(
                    predictive_pca.explained_variance_ratio_.sum()
                ),
            }
        )
        fitted[(str(model), int(layer))] = {
            "pca": pca,
            "discovery": discovery,
            "confirmation": confirmation,
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
        }
    return pd.DataFrame(rows), fitted


def project_isometric(points: np.ndarray) -> np.ndarray:
    yaw = np.deg2rad(38.0)
    pitch = np.deg2rad(24.0)
    rotation_z = np.asarray(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    rotation_x = np.asarray(
        [[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]]
    )
    rotated = points @ rotation_z.T @ rotation_x.T
    return rotated[:, :2]


def plot_frozen_pca(
    fitted: dict[tuple[str, int], dict[str, Any]], output: Path
) -> None:
    keys = sorted(fitted)
    figure, axes = plt.subplots(
        len(keys), 1, figsize=(9, max(5.0, 4.8 * len(keys))), squeeze=False
    )
    palette = plt.get_cmap("turbo", 10)
    for axis, key in zip(axes[:, 0], keys):
        payload = fitted[key]
        pca: PCA = payload["pca"]
        train = pca.transform(payload["x_train"])
        test = pca.transform(payload["x_test"])
        train_2d = project_isometric(train)
        test_2d = project_isometric(test)
        for count in range(1, 11):
            train_mask = payload["y_train"] == count
            test_mask = payload["y_test"] == count
            axis.scatter(
                train_2d[train_mask, 0],
                train_2d[train_mask, 1],
                s=24,
                alpha=0.30,
                color=palette(count - 1),
            )
            axis.scatter(
                test_2d[test_mask, 0],
                test_2d[test_mask, 1],
                s=32,
                alpha=0.75,
                marker="o",
                facecolors="none",
                edgecolors=[palette(count - 1)],
                label=str(count) if key == keys[0] else None,
            )
        model, layer = key
        variance = 100 * pca.explained_variance_ratio_
        axis.set_title(
            f"{model} L{layer} broad-bank output; frozen discovery PCA "
            f"({variance[0]:.1f}%, {variance[1]:.1f}%, {variance[2]:.1f}%)"
        )
        axis.set_xlabel("Isometric projection horizontal (PC1/PC2 mixture)")
        axis.set_ylabel("Isometric projection vertical (PC2/PC3 mixture)")
        axis.grid(alpha=0.2)
    if keys:
        axes[0, 0].legend(
            title="Count", ncol=10, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.25)
        )
    figure.tight_layout()
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--experiment-config",
        default="configs/realistic_niah_v4_4_5_span_restoration.json",
    )
    parser.add_argument("--models", nargs="+", default=["Qwen3-8B", "Gemma4-E4B"])
    parser.add_argument("--bootstrap-draws", type=int, default=200)
    parser.add_argument(
        "--sum-registered-head-writes",
        action="store_true",
        help=(
            "Rebuild each layer bank from the per-head .o tensors named in "
            "experiment-config instead of reading the precomputed .bank_o tensor."
        ),
    )
    args = parser.parse_args()

    config = json.loads(Path(args.experiment_config).read_text(encoding="utf-8"))
    discovery = {int(value) for value in config["discovery_seeds"]}
    confirmation = {int(value) for value in config["confirmation_seeds"]}
    root = Path(args.run_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    head_registry = config["retrieval_heads"] if args.sum_registered_head_writes else None
    states = load_bank_states(root, args.models, head_registry=head_registry)
    metrics, fitted = geometry_metrics(
        states,
        discovery,
        confirmation,
        bootstrap_draws=int(args.bootstrap_draws),
    )
    metrics.to_csv(output / "retrieval_geometry_metrics.csv", index=False)
    plot_frozen_pca(fitted, output / "retrieval_frozen_pca_3d_isometric.png")
    basis_payload: dict[str, Any] = {}
    for (model, layer), payload in fitted.items():
        pca: PCA = payload["pca"]
        basis_payload[f"{model}.L{layer}"] = {
            "mean": torch.from_numpy(pca.mean_.astype(np.float32)),
            "components": torch.from_numpy(pca.components_.astype(np.float32)),
            "explained_variance_ratio": torch.from_numpy(
                pca.explained_variance_ratio_.astype(np.float32)
            ),
        }
    torch.save(basis_payload, output / "retrieval_bases.pt")
    audit = {
        "schema_version": "realistic_niah_v4_4_5_retrieval_geometry_analysis_v1",
        "models": list(args.models),
        "discovery_seeds": sorted(discovery),
        "confirmation_seeds": sorted(confirmation),
        "rows": int(len(states)),
        "layer_metrics": int(len(metrics)),
        "basis_fit_population": "discovery clean natural forward only",
        "classifier_site": "answer-query broad-bank post-O output",
        "bank_construction": (
            "sum_registered_per_head_post_o_writes"
            if args.sum_registered_head_writes
            else "saved_precomputed_bank_o"
        ),
        "registered_heads": (
            {model: config["retrieval_heads"][model] for model in args.models}
            if args.sum_registered_head_writes
            else None
        ),
        "predictive_preprocessing": (
            "discovery-fitted PCA32 followed by discovery-fitted StandardScaler"
        ),
        "nearest_centroid_metric": (
            "cosine distance to discovery count centroid in standardized PCA32 space"
        ),
        "status": "PASS",
    }
    (output / "geometry_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
