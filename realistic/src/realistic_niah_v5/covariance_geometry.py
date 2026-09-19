"""Covariance-aware held-out geometry diagnostics for count representations.

The functions in this module deliberately separate four geometry questions:

* isotropic SNR: are class centroids large relative to total within-class power?
* Fisher trace: are centroids separated after accounting for anisotropic noise?
* Mahalanobis silhouette: do individual states cluster under a discovery-frozen
  within-class metric?
* ordinal RSA: do centroid distances increase with the count gap?

All preprocessing and the Mahalanobis metric are fitted on discovery rows.  A
caller may select a layer from discovery metrics and then read the matching
confirmation metric without using confirmation for selection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ScatterEstimate:
    """Class-balanced centroid and residual covariance estimates."""

    between: np.ndarray
    within: np.ndarray
    centroids: np.ndarray
    grand_centroid: np.ndarray
    support: dict[int, int]


def class_balanced_scatter(
    values: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[int],
) -> ScatterEstimate:
    """Estimate class-balanced between- and within-class covariance matrices."""

    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    retained = tuple(map(int, classes))
    if x.ndim != 2:
        raise ValueError(f"Expected a matrix of states, got shape {x.shape}")
    missing = [label for label in retained if not np.any(y == label)]
    if missing:
        raise ValueError(f"Missing class labels: {missing}")
    centroids = np.stack([x[y == label].mean(axis=0) for label in retained])
    grand = centroids.mean(axis=0)
    centered_centroids = centroids - grand
    between = np.einsum(
        "ki,kj->ij", centered_centroids, centered_centroids
    ) / len(retained)
    within_terms = []
    support: dict[int, int] = {}
    for class_index, label in enumerate(retained):
        group = x[y == label]
        support[label] = int(len(group))
        residual = group - centroids[class_index]
        within_terms.append(residual.T @ residual / len(group))
    within = np.mean(np.stack(within_terms), axis=0)
    return ScatterEstimate(
        between=np.asarray(0.5 * (between + between.T), dtype=np.float64),
        within=np.asarray(0.5 * (within + within.T), dtype=np.float64),
        centroids=centroids,
        grand_centroid=grand,
        support=support,
    )


def regularized_precision(
    within: np.ndarray,
    *,
    relative_ridge: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return precision/inverse-square-root matrices with a trace-scaled ridge."""

    covariance = np.asarray(within, dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T)
    dimension = int(covariance.shape[0])
    if covariance.shape != (dimension, dimension) or dimension == 0:
        raise ValueError(f"Invalid covariance shape {covariance.shape}")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(
        np.finfo(np.float64).eps,
        abs(float(np.trace(covariance))) / dimension,
        float(np.linalg.norm(covariance, ord="fro")) / dimension,
    )
    ridge = max(
        float(relative_ridge) * scale,
        -float(eigenvalues[0]) + float(relative_ridge) * scale,
    )
    regularized_eigenvalues = np.maximum(eigenvalues + ridge, np.finfo(float).eps)
    precision = (
        eigenvectors * (1.0 / regularized_eigenvalues)[None, :]
    ) @ eigenvectors.T
    inverse_sqrt = (
        eigenvectors * (1.0 / np.sqrt(regularized_eigenvalues))[None, :]
    ) @ eigenvectors.T
    condition = float(regularized_eigenvalues[-1] / regularized_eigenvalues[0])
    return precision, inverse_sqrt, float(ridge), condition


def isotropic_snr(scatter: ScatterEstimate) -> tuple[float, float]:
    """Return trace-between / trace-within and its value in decibels."""

    signal = float(np.trace(scatter.between))
    noise = float(np.trace(scatter.within))
    if signal <= 0 or noise <= np.finfo(float).eps:
        return np.nan, np.nan
    ratio = float(signal / noise)
    return ratio, float(10.0 * np.log10(ratio))


def fisher_trace(
    scatter: ScatterEstimate,
    *,
    relative_ridge: float = 1e-6,
) -> tuple[float, float, float]:
    """Return tr((Sigma_W + ridge I)^-1 Sigma_B), ridge, and condition."""

    precision, _inverse_sqrt, ridge, condition = regularized_precision(
        scatter.within, relative_ridge=relative_ridge
    )
    value = float(np.trace(precision @ scatter.between))
    return max(0.0, value), ridge, condition


def frozen_fisher_trace(
    reference_within: np.ndarray,
    evaluation_scatter: ScatterEstimate,
    *,
    relative_ridge: float = 1e-6,
) -> tuple[float, float]:
    """Evaluate signal with a precision matrix fitted on separate reference rows.

    The second return value is the mean evaluation within-class variance after
    applying the reference precision.  It should be near one when the reference
    and evaluation noise distributions agree.
    """

    precision, _inverse_sqrt, _ridge, _condition = regularized_precision(
        reference_within, relative_ridge=relative_ridge
    )
    value = float(np.trace(precision @ evaluation_scatter.between))
    noise_calibration = float(
        np.trace(precision @ evaluation_scatter.within)
        / evaluation_scatter.within.shape[0]
    )
    return max(0.0, value), noise_calibration


def class_balanced_silhouette(
    values: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[int],
) -> float:
    """Average Euclidean silhouette equally across classes rather than states."""

    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    per_state = silhouette_samples(x, y, metric="euclidean")
    return float(np.mean([per_state[y == int(label)].mean() for label in classes]))


def ordinal_centroid_rsa(
    values: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[int],
) -> float:
    """Spearman rho between centroid distance and absolute count difference."""

    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    retained = tuple(map(int, classes))
    centroids = np.stack([x[y == label].mean(axis=0) for label in retained])
    distances = []
    count_gaps = []
    for left in range(len(retained)):
        for right in range(left + 1, len(retained)):
            distances.append(float(np.linalg.norm(centroids[left] - centroids[right])))
            count_gaps.append(abs(retained[left] - retained[right]))
    statistic = spearmanr(count_gaps, distances).statistic
    return float(statistic)


def _split_rows(
    states: np.ndarray,
    metadata: pd.DataFrame,
    classes: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    class_mask = metadata["occurrence"].astype(int).isin(classes).to_numpy()
    discovery = metadata["split"].astype(str).eq("discovery").to_numpy() & class_mask
    confirmation = (
        metadata["split"].astype(str).eq("confirmation").to_numpy() & class_mask
    )
    discovery_y = metadata.loc[discovery, "occurrence"].to_numpy(dtype=int)
    confirmation_y = metadata.loc[confirmation, "occurrence"].to_numpy(dtype=int)
    for name, values in (("discovery", discovery_y), ("confirmation", confirmation_y)):
        missing = sorted(set(map(int, classes)) - set(values.tolist()))
        if missing:
            raise ValueError(f"{name} rows lack classes {missing}")
    return (
        np.asarray(states[discovery], dtype=np.float32),
        discovery_y,
        np.asarray(states[confirmation], dtype=np.float32),
        confirmation_y,
    )


def evaluate_covariance_geometry_layer(
    states: np.ndarray,
    metadata: pd.DataFrame,
    classes: Sequence[int],
    *,
    pca_dim: int = 16,
    random_state: int = 0,
    relative_ridge: float = 1e-6,
    discovery_cv_folds: int = 5,
) -> dict[str, Any]:
    """Fit PCA/noise metric on discovery and evaluate both data splits.

    Fisher trace is computed from each split's class-balanced covariance after a
    discovery-fitted PCA.  Mahalanobis silhouette and ordinal RSA additionally
    use the inverse square root of discovery within-class covariance, which is
    frozen before confirmation is transformed.
    """

    discovery_x, discovery_y, confirmation_x, confirmation_y = _split_rows(
        states, metadata, classes
    )
    scaler = StandardScaler().fit(discovery_x)
    discovery_scaled = scaler.transform(discovery_x)
    confirmation_scaled = scaler.transform(confirmation_x)
    components = min(
        int(pca_dim),
        int(len(discovery_x) - len(classes)),
        int(discovery_x.shape[1]),
    )
    if components < 2:
        raise ValueError("Discovery rows support fewer than two PCA components")
    pca = PCA(
        n_components=components,
        svd_solver="randomized",
        whiten=False,
        random_state=random_state,
    ).fit(discovery_scaled)
    discovery_projected = pca.transform(discovery_scaled)
    confirmation_projected = pca.transform(confirmation_scaled)

    # Reproduce the existing PCA-whitened isotropic SNR without fitting a second
    # PCA: sklearn whitening divides each score by sqrt(explained_variance_).
    pca_scale = np.sqrt(np.maximum(pca.explained_variance_, np.finfo(float).eps))
    discovery_pca_whitened = discovery_projected / pca_scale
    confirmation_pca_whitened = confirmation_projected / pca_scale
    discovery_isotropic_scatter = class_balanced_scatter(
        discovery_pca_whitened, discovery_y, classes
    )
    confirmation_isotropic_scatter = class_balanced_scatter(
        confirmation_pca_whitened, confirmation_y, classes
    )
    discovery_snr, discovery_snr_db = isotropic_snr(discovery_isotropic_scatter)
    confirmation_snr, confirmation_snr_db = isotropic_snr(
        confirmation_isotropic_scatter
    )

    discovery_scatter = class_balanced_scatter(
        discovery_projected, discovery_y, classes
    )
    confirmation_scatter = class_balanced_scatter(
        confirmation_projected, confirmation_y, classes
    )
    discovery_fisher, discovery_ridge, discovery_condition = fisher_trace(
        discovery_scatter, relative_ridge=relative_ridge
    )
    confirmation_fisher, confirmation_ridge, confirmation_condition = fisher_trace(
        confirmation_scatter, relative_ridge=relative_ridge
    )

    _precision, discovery_inverse_sqrt, _ridge, _condition = regularized_precision(
        discovery_scatter.within, relative_ridge=relative_ridge
    )
    confirmation_frozen_fisher, confirmation_noise_calibration = frozen_fisher_trace(
        discovery_scatter.within,
        confirmation_scatter,
        relative_ridge=relative_ridge,
    )
    discovery_mahalanobis = discovery_projected @ discovery_inverse_sqrt
    confirmation_mahalanobis = confirmation_projected @ discovery_inverse_sqrt
    discovery_silhouette = class_balanced_silhouette(
        discovery_mahalanobis, discovery_y, classes
    )
    confirmation_silhouette = class_balanced_silhouette(
        confirmation_mahalanobis, confirmation_y, classes
    )
    discovery_rsa = ordinal_centroid_rsa(
        discovery_mahalanobis, discovery_y, classes
    )
    confirmation_rsa = ordinal_centroid_rsa(
        confirmation_mahalanobis, confirmation_y, classes
    )

    discovery_frame = metadata.loc[
        metadata["split"].astype(str).eq("discovery")
        & metadata["occurrence"].astype(int).isin(classes)
    ].reset_index(drop=True)
    discovery_seeds = discovery_frame["seed"].to_numpy(dtype=int)
    unique_seeds = np.unique(discovery_seeds)
    fold_count = min(int(discovery_cv_folds), int(len(unique_seeds)))
    if fold_count < 2:
        raise ValueError("Discovery covariance geometry requires at least two seeds")
    fold_metrics: list[dict[str, float]] = []
    invalid_fold_support: list[dict[str, list[int]]] = []
    splitter = GroupKFold(n_splits=fold_count)
    for train_index, test_index in splitter.split(
        discovery_projected, discovery_y, groups=discovery_seeds
    ):
        train_y = discovery_y[train_index]
        test_y = discovery_y[test_index]
        missing_train = sorted(set(map(int, classes)) - set(train_y.tolist()))
        missing_test = sorted(set(map(int, classes)) - set(test_y.tolist()))
        if missing_train or missing_test:
            # Sparse token sites (for example, an explicit-marker-only site)
            # need not occur for every count in every seed.  Their full-split
            # and frozen-confirmation covariance metrics remain well defined,
            # but a grouped discovery fold that omits a class does not.  Keep
            # the usable folds and make the reduced OOF support explicit in
            # the returned audit fields instead of aborting the whole sweep.
            invalid_fold_support.append(
                {"missing_train": missing_train, "missing_test": missing_test}
            )
            continue
        test_isotropic = class_balanced_scatter(
            discovery_pca_whitened[test_index], test_y, classes
        )
        _fold_snr, fold_snr_db = isotropic_snr(test_isotropic)
        train_scatter = class_balanced_scatter(
            discovery_projected[train_index], train_y, classes
        )
        test_scatter = class_balanced_scatter(
            discovery_projected[test_index], test_y, classes
        )
        fold_fisher, fold_noise_calibration = frozen_fisher_trace(
            train_scatter.within,
            test_scatter,
            relative_ridge=relative_ridge,
        )
        _fold_precision, fold_inverse_sqrt, _fold_ridge, _fold_condition = (
            regularized_precision(
                train_scatter.within, relative_ridge=relative_ridge
            )
        )
        test_mahalanobis = discovery_projected[test_index] @ fold_inverse_sqrt
        fold_metrics.append(
            {
                "isotropic_snr_db": fold_snr_db,
                "fisher_trace": fold_fisher,
                "fisher_noise_calibration": fold_noise_calibration,
                "mahalanobis_silhouette": class_balanced_silhouette(
                    test_mahalanobis, test_y, classes
                ),
                "ordinal_rsa": ordinal_centroid_rsa(
                    test_mahalanobis, test_y, classes
                ),
            }
        )

    def fold_mean(name: str) -> float:
        if not fold_metrics:
            return float("nan")
        return float(np.mean([row[name] for row in fold_metrics]))

    def fold_std(name: str) -> float:
        if len(fold_metrics) < 2:
            return float("nan")
        return float(np.std([row[name] for row in fold_metrics], ddof=1))

    return {
        "pca_components": int(components),
        "pca_explained_variance_ratio_sum": float(
            np.sum(pca.explained_variance_ratio_)
        ),
        "discovery_rows": int(len(discovery_y)),
        "confirmation_rows": int(len(confirmation_y)),
        "discovery_support_min": int(min(discovery_scatter.support.values())),
        "discovery_support_max": int(max(discovery_scatter.support.values())),
        "confirmation_support_min": int(min(confirmation_scatter.support.values())),
        "confirmation_support_max": int(max(confirmation_scatter.support.values())),
        "discovery_isotropic_snr": discovery_snr,
        "discovery_isotropic_snr_db": discovery_snr_db,
        "confirmation_isotropic_snr": confirmation_snr,
        "confirmation_isotropic_snr_db": confirmation_snr_db,
        "discovery_fisher_trace": discovery_fisher,
        "confirmation_fisher_trace": confirmation_fisher,
        "confirmation_fisher_trace_frozen": confirmation_frozen_fisher,
        "confirmation_fisher_noise_calibration": confirmation_noise_calibration,
        "discovery_covariance_ridge": discovery_ridge,
        "confirmation_covariance_ridge": confirmation_ridge,
        "discovery_covariance_condition": discovery_condition,
        "confirmation_covariance_condition": confirmation_condition,
        "discovery_mahalanobis_silhouette": discovery_silhouette,
        "confirmation_mahalanobis_silhouette": confirmation_silhouette,
        "discovery_ordinal_rsa": discovery_rsa,
        "confirmation_ordinal_rsa": confirmation_rsa,
        "discovery_cv_requested_fold_count": int(fold_count),
        "discovery_cv_fold_count": int(len(fold_metrics)),
        "discovery_cv_invalid_fold_count": int(len(invalid_fold_support)),
        "discovery_cv_invalid_fold_support": json.dumps(
            invalid_fold_support, sort_keys=True
        ),
        "discovery_oof_isotropic_snr_db": fold_mean("isotropic_snr_db"),
        "discovery_oof_isotropic_snr_db_fold_sd": fold_std("isotropic_snr_db"),
        "discovery_oof_fisher_trace": fold_mean("fisher_trace"),
        "discovery_oof_fisher_trace_fold_sd": fold_std("fisher_trace"),
        "discovery_oof_fisher_noise_calibration": fold_mean(
            "fisher_noise_calibration"
        ),
        "discovery_oof_mahalanobis_silhouette": fold_mean(
            "mahalanobis_silhouette"
        ),
        "discovery_oof_mahalanobis_silhouette_fold_sd": fold_std(
            "mahalanobis_silhouette"
        ),
        "discovery_oof_ordinal_rsa": fold_mean("ordinal_rsa"),
        "discovery_oof_ordinal_rsa_fold_sd": fold_std("ordinal_rsa"),
        "metric_definitions": {
            "fisher_trace": "tr((Sigma_W + ridge I)^-1 Sigma_B) in discovery-fitted PCA space",
            "frozen_fisher_trace": "evaluation Sigma_B measured with a within-covariance precision fitted on separate discovery rows",
            "mahalanobis_silhouette": "class-balanced silhouette after whitening by discovery within-class covariance",
            "ordinal_rsa": "Spearman rho across centroid pairs: frozen-Mahalanobis distance versus absolute count gap",
        },
    }


SELECTION_METRICS = {
    "isotropic_snr": (
        "discovery_oof_isotropic_snr_db",
        "confirmation_isotropic_snr_db",
    ),
    "fisher_trace": (
        "discovery_oof_fisher_trace",
        "confirmation_fisher_trace_frozen",
    ),
    "mahalanobis_silhouette": (
        "discovery_oof_mahalanobis_silhouette",
        "confirmation_mahalanobis_silhouette",
    ),
    "ordinal_rsa": ("discovery_oof_ordinal_rsa", "confirmation_ordinal_rsa"),
}


def select_discovery_winners(per_layer: pd.DataFrame) -> pd.DataFrame:
    """Select one layer per dataset and metric using discovery columns only."""

    group_columns = ["model_label", "endpoint", "mode"]
    selected_rows: list[dict[str, Any]] = []
    for group_key, frame in per_layer.groupby(group_columns, sort=True):
        for selector, (discovery_column, confirmation_column) in SELECTION_METRICS.items():
            ordered = frame.sort_values(
                [discovery_column, "layer"], ascending=[False, True]
            )
            winner = ordered.iloc[0]
            selected_rows.append(
                {
                    "model_label": group_key[0],
                    "endpoint": group_key[1],
                    "mode": group_key[2],
                    "selector": selector,
                    "discovery_metric": discovery_column,
                    "confirmation_metric": confirmation_column,
                    "selected_layer": int(winner["layer"]),
                    "discovery_value": float(winner[discovery_column]),
                    "confirmation_value": float(winner[confirmation_column]),
                    "pca_components": int(winner["pca_components"]),
                    "discovery_rows": int(winner["discovery_rows"]),
                    "confirmation_rows": int(winner["confirmation_rows"]),
                }
            )
    return pd.DataFrame(selected_rows)
