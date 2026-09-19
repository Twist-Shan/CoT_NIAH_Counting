"""Cross-layer counter-channel analyses for the Realistic NIAH V4.4 campaign.

The module deliberately consumes a small, server-independent layer-manifest
format.  GPU capture jobs may remain on FileStream; CPU analyses only need one
``.npz`` shard per model/role/layer with the following required arrays:

``states`` [rows, hidden], ``count`` [rows], ``seed`` [rows].

Optional one-dimensional arrays (``sample_id``, ``correct``, ``prediction``
and any noise covariates) are carried through by :func:`load_layer_dataset`.
All train/test splits are grouped by seed to prevent prompt-family leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class LayerStateDataset:
    model_label: str
    role: str
    layer: int
    states: np.ndarray
    count: np.ndarray
    seed: np.ndarray
    metadata: pd.DataFrame
    source: Path

    def validate(self) -> None:
        if self.states.ndim != 2:
            raise ValueError(f"states must be [rows, hidden], got {self.states.shape}")
        rows = self.states.shape[0]
        if len(self.count) != rows or len(self.seed) != rows or len(self.metadata) != rows:
            raise ValueError("state/count/seed/metadata row counts disagree")
        if rows == 0 or self.states.shape[1] == 0:
            raise ValueError("empty layer dataset")
        if not np.isfinite(self.states).all():
            raise ValueError(f"non-finite hidden state in {self.source}")
        if not np.isfinite(self.count.astype(float)).all():
            raise ValueError(f"non-finite count label in {self.source}")
        if len(np.unique(self.count)) < 2:
            raise ValueError("counter analysis requires at least two count classes")
        if len(np.unique(self.seed)) < 2:
            raise ValueError("grouped analysis requires at least two seeds")


def read_layer_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("datasets") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("layer manifest must contain a non-empty datasets list")
    required = {"model_label", "role", "layer", "path"}
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"manifest row {index} is missing {sorted(missing)}")
    return [dict(row) for row in rows]


def load_layer_dataset(
    manifest_path: str | Path, row: Mapping[str, Any]
) -> LayerStateDataset:
    manifest_path = Path(manifest_path)
    shard = Path(str(row["path"]))
    if not shard.is_absolute():
        shard = manifest_path.parent / shard
    if not shard.is_file():
        raise FileNotFoundError(shard)
    with np.load(shard, allow_pickle=False) as payload:
        missing = {"states", "count", "seed"} - set(payload.files)
        if missing:
            raise ValueError(f"{shard} is missing arrays {sorted(missing)}")
        states = np.asarray(payload["states"], dtype=np.float32)
        count = np.asarray(payload["count"], dtype=np.int64)
        seed = np.asarray(payload["seed"], dtype=np.int64)
        metadata = {
            key: np.asarray(payload[key])
            for key in payload.files
            if key not in {"states", "count", "seed"}
            and np.asarray(payload[key]).ndim == 1
            and len(np.asarray(payload[key])) == len(count)
        }
    frame = pd.DataFrame(metadata)
    frame.insert(0, "seed", seed)
    frame.insert(1, "count", count)
    dataset = LayerStateDataset(
        model_label=str(row["model_label"]),
        role=str(row["role"]),
        layer=int(row["layer"]),
        states=states,
        count=count,
        seed=seed,
        metadata=frame,
        source=shard,
    )
    dataset.validate()
    return dataset


def count_centroids(states: np.ndarray, count: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.unique(count)
    centroids = np.stack([states[count == label].mean(axis=0) for label in labels])
    return labels.astype(np.int64), centroids.astype(np.float64, copy=False)


def count_axis(states: np.ndarray, count: np.ndarray) -> np.ndarray:
    labels, centroids = count_centroids(states, count)
    x = labels.astype(np.float64) - float(labels.mean())
    centered = centroids - centroids.mean(axis=0, keepdims=True)
    axis = x @ centered
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError("count-aligned axis has zero norm")
    return axis / norm


def count_subspace(
    states: np.ndarray, count: np.ndarray, *, rank: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    _, centroids = count_centroids(states, count)
    centered = centroids - centroids.mean(axis=0, keepdims=True)
    maximum_rank = min(centered.shape[0] - 1, centered.shape[1])
    if not 1 <= int(rank) <= maximum_rank:
        raise ValueError(f"rank must be in [1,{maximum_rank}], got {rank}")
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[: int(rank)].T.copy()
    explained = singular_values**2
    explained = explained / max(float(explained.sum()), 1e-12)
    return basis, explained[: int(rank)]


def subspace_overlap(left: np.ndarray, right: np.ndarray) -> float:
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("subspace bases must be [hidden, rank] in a shared residual space")
    denominator = float(min(left.shape[1], right.shape[1]))
    return float(np.linalg.norm(left.T @ right, ord="fro") ** 2 / denominator)


def principal_angles_degrees(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    singular = np.linalg.svd(left.T @ right, compute_uv=False)
    singular = np.clip(singular, 0.0, 1.0)
    return np.degrees(np.arccos(singular))


def oriented_axis_angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    """Return the 0--180 degree angle between increasing-count axes.

    Both axes are oriented by increasing numeric count in :func:`count_axis`,
    so folding the sign with ``abs`` would hide a genuine reversal.
    """

    cosine = float(np.clip(np.dot(left, right), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def unoriented_axis_angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    """Return the 0--90 degree angle between the two unoriented axis lines."""

    oriented = oriented_axis_angle_degrees(left, right)
    return float(min(oriented, 180.0 - oriented))


def subset_layer_dataset(
    dataset: LayerStateDataset, mask: np.ndarray
) -> LayerStateDataset:
    """Return a validated row subset while preserving dataset provenance."""

    selected = np.asarray(mask, dtype=bool)
    if selected.ndim != 1 or len(selected) != len(dataset.count):
        raise ValueError("dataset subset mask must have one entry per row")
    subset = LayerStateDataset(
        model_label=dataset.model_label,
        role=dataset.role,
        layer=dataset.layer,
        states=dataset.states[selected],
        count=dataset.count[selected],
        seed=dataset.seed[selected],
        metadata=dataset.metadata.loc[selected].reset_index(drop=True),
        source=dataset.source,
    )
    subset.validate()
    return subset


def grouped_ridge_predictions(
    train_states: np.ndarray,
    train_count: np.ndarray,
    test_states: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )
    model.fit(train_states, train_count.astype(float))
    return np.asarray(model.predict(test_states), dtype=np.float64)


def _fold_count(groups: np.ndarray, requested: int) -> int:
    return max(2, min(int(requested), len(np.unique(groups))))


def cluster_bootstrap_mean_ci(
    values: np.ndarray,
    *,
    draws: int = 10_000,
    random_state: int = 442,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (math.nan, math.nan)
    rng = np.random.default_rng(random_state)
    means = rng.choice(values, size=(int(draws), len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def grouped_cross_layer_decode(
    train: LayerStateDataset,
    test: LayerStateDataset,
    *,
    folds: int = 5,
    alpha: float = 1.0,
) -> dict[str, float]:
    """Fit on one layer/role and test the same held-out seeds elsewhere."""

    common_seeds = np.intersect1d(np.unique(train.seed), np.unique(test.seed))
    if len(common_seeds) < 2:
        raise ValueError("cross-layer decoding needs at least two shared seeds")
    splitter = GroupKFold(n_splits=_fold_count(common_seeds, folds))
    seed_rows = np.arange(len(common_seeds))
    target_all: list[np.ndarray] = []
    prediction_all: list[np.ndarray] = []
    for train_seed_idx, test_seed_idx in splitter.split(seed_rows, groups=common_seeds):
        fit_seeds = common_seeds[train_seed_idx]
        held_seeds = common_seeds[test_seed_idx]
        fit_mask = np.isin(train.seed, fit_seeds)
        held_mask = np.isin(test.seed, held_seeds)
        prediction_all.append(
            grouped_ridge_predictions(
                train.states[fit_mask],
                train.count[fit_mask],
                test.states[held_mask],
                alpha=alpha,
            )
        )
        target_all.append(test.count[held_mask].astype(float))
    target = np.concatenate(target_all)
    prediction = np.concatenate(prediction_all)
    rounded = np.clip(np.rint(prediction), target.min(), target.max())
    return {
        "rows": int(len(target)),
        "r2": float(r2_score(target, prediction)),
        "mae": float(np.mean(np.abs(prediction - target))),
        "rounded_accuracy": float(np.mean(rounded == target)),
        "rounded_mae": float(np.mean(np.abs(rounded - target))),
    }


def _classifier_factories(
    *, pca_components: int, random_state: int, n_jobs: int
) -> dict[str, Any]:
    def projected(estimator: Any, *, scale: bool = True) -> Pipeline:
        steps: list[tuple[str, Any]] = []
        if scale:
            steps.append(("scale", StandardScaler()))
        steps.append(("pca", PCA(n_components=int(pca_components), random_state=random_state)))
        steps.append(("classifier", estimator))
        return Pipeline(steps)

    models: dict[str, Any] = {
        "logistic_l2": projected(
            LogisticRegression(C=1.0, max_iter=4000, solver="lbfgs")
        ),
        "ridge_classifier": projected(RidgeClassifier(alpha=1.0)),
        "linear_svm": projected(LinearSVC(C=1.0, dual="auto", max_iter=10000)),
        "rbf_svm": projected(SVC(C=10.0, gamma="scale")),
        "nearest_centroid": projected(NearestCentroid(shrink_threshold=0.1)),
        "shrinkage_lda": projected(
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        ),
        "gaussian_nb": projected(GaussianNB(var_smoothing=1e-8)),
        "random_forest": projected(
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=random_state,
                n_jobs=n_jobs,
            ),
            scale=False,
        ),
        "extra_trees": projected(
            ExtraTreesClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=random_state,
                n_jobs=n_jobs,
            ),
            scale=False,
        ),
        "hist_gradient_boosting": projected(
            HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=250,
                max_leaf_nodes=31,
                random_state=random_state,
            ),
            scale=False,
        ),
    }
    for neighbors in (1, 3, 5, 7, 9):
        for metric in ("euclidean", "cosine"):
            models[f"knn_k{neighbors}_{metric}"] = projected(
                KNeighborsClassifier(
                    n_neighbors=neighbors,
                    metric=metric,
                    weights="distance",
                    n_jobs=n_jobs,
                )
            )
    return models


def benchmark_classifiers(
    dataset: LayerStateDataset,
    *,
    algorithms: Sequence[str] | None = None,
    folds: int = 5,
    pca_components: int = 32,
    random_state: int = 442,
    n_jobs: int = 1,
) -> pd.DataFrame:
    maximum_components = min(
        int(pca_components),
        dataset.states.shape[1],
        dataset.states.shape[0] - math.ceil(dataset.states.shape[0] / _fold_count(dataset.seed, folds)) - 1,
    )
    if maximum_components < 1:
        raise ValueError("not enough training rows for PCA classification")
    factories = _classifier_factories(
        pca_components=maximum_components,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    selected = list(algorithms) if algorithms else list(factories)
    unknown = sorted(set(selected) - set(factories))
    if unknown:
        raise ValueError(f"unknown classifiers: {unknown}; choices={sorted(factories)}")
    rows: list[dict[str, Any]] = []
    for name in selected:
        prediction_frame = _classifier_oof_predictions(
            dataset,
            factories[name],
            folds=folds,
            algorithm=name,
        )
        y_true = prediction_frame["gold_count"].to_numpy(np.int64)
        y_pred = prediction_frame["predicted_count"].to_numpy(np.int64)
        seeds = prediction_frame["seed"].to_numpy(np.int64)
        seed_accuracy = (
            prediction_frame.assign(
                _correct=prediction_frame["gold_count"]
                == prediction_frame["predicted_count"]
            )
            .groupby("seed", sort=True)["_correct"]
            .mean()
            .to_numpy(float)
        )
        accuracy_low, accuracy_high = cluster_bootstrap_mean_ci(
            seed_accuracy,
            random_state=random_state,
        )
        rows.append(
            {
                "model_label": dataset.model_label,
                "role": dataset.role,
                "layer": dataset.layer,
                "algorithm": name,
                "rows": len(y_true),
                "seeds": len(np.unique(seeds)),
                "count_classes": json.dumps(
                    [int(value) for value in np.unique(y_true)]
                ),
                "count_class_count": int(len(np.unique(y_true))),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "accuracy_ci95_low": accuracy_low,
                "accuracy_ci95_high": accuracy_high,
                "seed_accuracy_sd": float(np.std(seed_accuracy, ddof=1)),
                "chance_accuracy": float(1.0 / len(np.unique(y_true))),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "count_mae": float(np.mean(np.abs(y_pred - y_true))),
                "signed_error": float(np.mean(y_pred - y_true)),
                "pca_components": maximum_components,
            }
        )
    return pd.DataFrame(rows)


def _classifier_oof_predictions(
    dataset: LayerStateDataset,
    estimator: Any,
    *,
    folds: int,
    algorithm: str,
) -> pd.DataFrame:
    splitter = GroupKFold(n_splits=_fold_count(dataset.seed, folds))
    output: list[pd.DataFrame] = []
    for fold, (train_index, test_index) in enumerate(
        splitter.split(dataset.states, dataset.count, dataset.seed)
    ):
        fitted = clone(estimator)
        fitted.fit(dataset.states[train_index], dataset.count[train_index])
        predicted = np.asarray(
            fitted.predict(dataset.states[test_index]), dtype=np.int64
        )
        frame = dataset.metadata.iloc[test_index].copy()
        frame["gold_count"] = dataset.count[test_index]
        frame["predicted_count"] = predicted
        frame["fold"] = int(fold)
        frame["algorithm"] = algorithm
        frame["model_label"] = dataset.model_label
        frame["role"] = dataset.role
        frame["layer"] = dataset.layer
        output.append(frame)
    return pd.concat(output, ignore_index=True)


def classifier_oof_predictions(
    dataset: LayerStateDataset,
    algorithm: str,
    *,
    folds: int = 5,
    pca_components: int = 32,
    random_state: int = 442,
    n_jobs: int = 1,
) -> pd.DataFrame:
    maximum_components = min(
        int(pca_components),
        dataset.states.shape[1],
        dataset.states.shape[0]
        - math.ceil(dataset.states.shape[0] / _fold_count(dataset.seed, folds))
        - 1,
    )
    factories = _classifier_factories(
        pca_components=maximum_components,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    if algorithm not in factories:
        raise ValueError(f"unknown classifier {algorithm}; choices={sorted(factories)}")
    return _classifier_oof_predictions(
        dataset,
        factories[algorithm],
        folds=folds,
        algorithm=algorithm,
    )


def leave_group_out_noise(
    dataset: LayerStateDataset,
    basis: np.ndarray | None = None,
    *,
    rank: int = 3,
) -> pd.DataFrame:
    """Compute count-conditional residual energies without held-seed leakage.

    If ``basis`` is supplied it is treated as a genuinely frozen external
    basis (for example discovery-fit evaluated on confirmation). Otherwise a
    fresh count subspace is fitted on the non-held seeds inside every fold.
    """

    frozen_projector = None if basis is None else basis @ basis.T
    output: list[dict[str, Any]] = []
    for held_seed in np.unique(dataset.seed):
        train = dataset.seed != held_seed
        if frozen_projector is None:
            fold_basis, _ = count_subspace(
                dataset.states[train], dataset.count[train], rank=rank
            )
            projector = fold_basis @ fold_basis.T
            basis_fit = "leave_seed_out"
        else:
            projector = frozen_projector
            basis_fit = "frozen_external"
        test_indices = np.flatnonzero(dataset.seed == held_seed)
        for index in test_indices:
            label = dataset.count[index]
            centroid_rows = dataset.states[train & (dataset.count == label)]
            if len(centroid_rows) == 0:
                raise ValueError(f"no training centroid for count={label}, seed={held_seed}")
            residual = dataset.states[index].astype(np.float64) - centroid_rows.mean(axis=0)
            parallel = projector @ residual
            orthogonal = residual - parallel
            row = dataset.metadata.iloc[index].to_dict()
            row.update(
                {
                    "model_label": dataset.model_label,
                    "role": dataset.role,
                    "layer": dataset.layer,
                    "noise_total": float(np.dot(residual, residual)),
                    "noise_parallel": float(np.dot(parallel, parallel)),
                    "noise_orthogonal": float(np.dot(orthogonal, orthogonal)),
                    "residual_norm": float(np.linalg.norm(residual)),
                    "noise_basis_fit": basis_fit,
                }
            )
            output.append(row)
    return pd.DataFrame(output)


def _noise_preprocessor(
    numeric: Sequence[str], categorical: Sequence[str]
) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                list(numeric),
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                list(categorical),
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def benchmark_noise_models(
    frame: pd.DataFrame,
    *,
    target: str,
    group_column: str,
    numeric: Sequence[str],
    categorical: Sequence[str],
    folds: int = 5,
    random_state: int = 442,
    n_jobs: int = 1,
    algorithms: Sequence[str] | None = None,
) -> pd.DataFrame:
    required = {target, group_column, *numeric, *categorical}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"noise table is missing columns {sorted(missing)}")
    groups = frame[group_column].to_numpy()
    y = np.log1p(pd.to_numeric(frame[target], errors="raise").to_numpy(float))
    x = frame[list(dict.fromkeys([*numeric, *categorical]))]
    preprocessor = _noise_preprocessor(numeric, categorical)
    models: dict[str, Any] = {
        "elastic_net": ElasticNet(alpha=0.01, l1_ratio=0.25, max_iter=10000),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=n_jobs,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=n_jobs,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=300,
            max_leaf_nodes=31,
            random_state=random_state,
        ),
    }
    selected = list(models) if algorithms is None else [str(name) for name in algorithms]
    unknown = sorted(set(selected) - set(models))
    if unknown:
        raise ValueError(f"unknown noise algorithms: {unknown}; choices={sorted(models)}")
    splitter = GroupKFold(n_splits=_fold_count(groups, folds))
    rows: list[dict[str, Any]] = []
    for name in selected:
        model = models[name]
        truth: list[np.ndarray] = []
        prediction: list[np.ndarray] = []
        for train_index, test_index in splitter.split(x, y, groups):
            pipeline = Pipeline(
                [("features", clone(preprocessor)), ("regressor", clone(model))]
            )
            pipeline.fit(x.iloc[train_index], y[train_index])
            truth.append(y[test_index])
            prediction.append(np.asarray(pipeline.predict(x.iloc[test_index])))
        y_true = np.concatenate(truth)
        y_pred = np.concatenate(prediction)
        rows.append(
            {
                "target": target,
                "model": name,
                "rows": len(y_true),
                "groups": len(np.unique(groups)),
                "heldout_r2_log1p": float(r2_score(y_true, y_pred)),
                "heldout_mae_log1p": float(np.mean(np.abs(y_pred - y_true))),
                "numeric_factors": json.dumps(list(numeric)),
                "categorical_factors": json.dumps(list(categorical)),
            }
        )
    return pd.DataFrame(rows)


def projected_patch(
    receiver: np.ndarray, donor: np.ndarray, basis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return count-subspace and complement donor patches for matched controls."""

    receiver = np.asarray(receiver, dtype=np.float64)
    donor = np.asarray(donor, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    if receiver.shape != donor.shape or receiver.ndim != 1:
        raise ValueError("receiver and donor must be same-width vectors")
    if basis.ndim != 2 or basis.shape[0] != receiver.shape[0]:
        raise ValueError("basis must be [hidden, rank]")
    delta = donor - receiver
    projected = basis @ (basis.T @ delta)
    return receiver + projected, receiver + (delta - projected)


def remove_count_component(
    state: np.ndarray, center: np.ndarray, basis: np.ndarray, *, dose: float = 1.0
) -> np.ndarray:
    if not 0.0 <= float(dose) <= 1.0:
        raise ValueError("removal dose must be in [0,1]")
    state = np.asarray(state, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    if state.shape != center.shape or basis.shape[0] != state.shape[0]:
        raise ValueError("state, center and basis widths disagree")
    centered = state - center
    return state - float(dose) * basis @ (basis.T @ centered)
