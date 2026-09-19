"""Discovery-fitted Fisher/LDA display geometry.

This module builds a supervised diagnostic view without allowing confirmation
rows to influence preprocessing, covariance estimation, or discriminant axes.
The construction is:

1. fit a feature-wise StandardScaler on discovery rows;
2. fit an unwhitened PCA on discovery rows;
3. estimate class-balanced discovery within-class covariance;
4. whiten the PCA scores by that regularized covariance;
5. diagonalize the discovery between-class covariance in the whitened space.

The leading three eigenvectors are the Fisher/LDA display axes.  Confirmation
rows are only transformed after the full display map has been frozen.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .covariance_geometry import (
    class_balanced_scatter,
    class_balanced_silhouette,
    regularized_precision,
)


def _split_mask(
    metadata: pd.DataFrame, classes: Sequence[int], split: str
) -> np.ndarray:
    return (
        metadata["split"].astype(str).eq(split)
        & metadata["occurrence"].astype(int).isin(tuple(map(int, classes)))
    ).to_numpy()


def _canonicalize_axis_signs(axes: np.ndarray) -> np.ndarray:
    """Remove the arbitrary sign ambiguity of symmetric eigendecomposition."""

    result = np.asarray(axes, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0:
            result[:, column] *= -1.0
    return result


def _centroid_rows(
    coordinates: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[int],
) -> list[list[float | int]]:
    rows: list[list[float | int]] = []
    for label in map(int, classes):
        group = coordinates[labels == label]
        if len(group) == 0:
            continue
        mean = group.mean(axis=0)
        rows.append(
            [
                label,
                round(float(mean[0]), 6),
                round(float(mean[1]), 6),
                round(float(mean[2]), 6),
                int(len(group)),
            ]
        )
    return rows


def _radius_gap_ratio(
    discovery_coordinates: np.ndarray,
    discovery_labels: np.ndarray,
    confirmation_coordinates: np.ndarray,
    confirmation_labels: np.ndarray,
    classes: Sequence[int],
) -> tuple[float, list[dict[str, float | int]]]:
    retained = tuple(map(int, classes))
    centroids = {
        label: discovery_coordinates[discovery_labels == label].mean(axis=0)
        for label in retained
    }
    ratios: list[float] = []
    rows: list[dict[str, float | int]] = []
    for label in retained:
        adjacent = [
            float(np.linalg.norm(centroids[label] - centroids[other]))
            for other in (label - 1, label + 1)
            if other in centroids
        ]
        gap = min(adjacent)
        group = confirmation_coordinates[confirmation_labels == label]
        radius = float(np.mean(np.linalg.norm(group - centroids[label], axis=1)))
        ratio = radius / gap if gap > np.finfo(float).eps else float("nan")
        ratios.append(ratio)
        rows.append(
            {
                "occurrence": label,
                "confirmation_states": int(len(group)),
                "mean_radius_to_discovery_centroid": radius,
                "nearest_adjacent_discovery_centroid_gap": gap,
                "radius_gap_ratio": ratio,
            }
        )
    return float(np.nanmean(ratios)), rows


def discovery_fitted_fisher_lda3(
    states: np.ndarray,
    metadata: pd.DataFrame,
    classes: Sequence[int],
    *,
    pca_dim: int = 16,
    relative_ridge: float = 1e-6,
    random_state: int = 0,
) -> dict[str, Any]:
    """Return a frozen Fisher/LDA3 display payload for all registered rows.

    The returned point coordinates include discovery and confirmation rows for
    interactive auditing.  Every fitted quantity is a function of discovery
    rows only.  ``confirmation_*`` diagnostics are evaluated after freezing.
    """

    x = np.asarray(states, dtype=np.float32)
    frame = metadata.reset_index(drop=True)
    if x.ndim != 2 or len(x) != len(frame):
        raise ValueError(
            f"State/metadata mismatch: states={x.shape}, metadata={len(frame)}"
        )
    retained = tuple(map(int, classes))
    discovery = _split_mask(frame, retained, "discovery")
    confirmation = _split_mask(frame, retained, "confirmation")
    discovery_y = frame.loc[discovery, "occurrence"].to_numpy(dtype=int)
    confirmation_y = frame.loc[confirmation, "occurrence"].to_numpy(dtype=int)
    for name, labels in (("discovery", discovery_y), ("confirmation", confirmation_y)):
        missing = sorted(set(retained) - set(labels.tolist()))
        if missing:
            raise ValueError(f"{name} rows lack classes {missing}")

    scaler = StandardScaler().fit(x[discovery])
    scaled = scaler.transform(x)
    components = min(
        int(pca_dim),
        int(discovery.sum() - len(retained)),
        int(x.shape[1]),
    )
    if components < 3:
        raise ValueError("Fisher/LDA3 requires at least three PCA components")
    pca = PCA(
        n_components=components,
        svd_solver="randomized",
        whiten=False,
        random_state=random_state,
    ).fit(scaled[discovery])
    pca_scores = pca.transform(scaled)

    discovery_scatter = class_balanced_scatter(
        pca_scores[discovery], discovery_y, retained
    )
    _precision, inverse_sqrt, ridge, condition = regularized_precision(
        discovery_scatter.within, relative_ridge=relative_ridge
    )
    whitened = (pca_scores - discovery_scatter.grand_centroid) @ inverse_sqrt
    fisher_between = inverse_sqrt @ discovery_scatter.between @ inverse_sqrt
    fisher_between = 0.5 * (fisher_between + fisher_between.T)
    eigenvalues, eigenvectors = np.linalg.eigh(fisher_between)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    axes = _canonicalize_axis_signs(eigenvectors[:, order[:3]])
    coordinates = whitened @ axes

    discovery_coordinates = coordinates[discovery]
    confirmation_coordinates = coordinates[confirmation]
    discovery_silhouette = class_balanced_silhouette(
        discovery_coordinates, discovery_y, retained
    )
    confirmation_silhouette = class_balanced_silhouette(
        confirmation_coordinates, confirmation_y, retained
    )
    radius_gap, radius_rows = _radius_gap_ratio(
        discovery_coordinates,
        discovery_y,
        confirmation_coordinates,
        confirmation_y,
        retained,
    )

    fisher_trace = float(eigenvalues.sum())
    top3_trace = float(eigenvalues[:3].sum())
    top3_fraction = top3_trace / fisher_trace if fisher_trace > 0 else float("nan")
    points = []
    for index, row in enumerate(frame.itertuples(index=False)):
        value = coordinates[index]
        points.append(
            [
                str(row.split),
                int(row.seed),
                int(row.occurrence),
                round(float(value[0]), 6),
                round(float(value[1]), 6),
                round(float(value[2]), 6),
                int(row.gold_count),
            ]
        )

    return {
        "fit": {
            "fit_split": "discovery only",
            "pipeline": (
                "discovery StandardScaler -> discovery PCA16 (unwhitened) -> "
                "class-balanced discovery within-covariance whitening -> "
                "top-3 discovery between-class eigenvectors"
            ),
            "pca_components": int(components),
            "pca_explained_variance_ratio_sum": float(
                np.sum(pca.explained_variance_ratio_)
            ),
            "relative_covariance_ridge": float(relative_ridge),
            "absolute_covariance_ridge": float(ridge),
            "regularized_within_condition_number": float(condition),
            "fisher_eigenvalues": [float(value) for value in eigenvalues],
            "top3_fisher_trace": top3_trace,
            "total_fisher_trace": fisher_trace,
            "top3_fisher_trace_fraction": top3_fraction,
        },
        "metrics": {
            "discovery_lda3_class_balanced_silhouette": discovery_silhouette,
            "confirmation_lda3_class_balanced_silhouette": confirmation_silhouette,
            "confirmation_lda3_radius_gap_ratio": radius_gap,
            "confirmation_lda3_radius_gap_by_class": radius_rows,
        },
        "points": points,
        "discovery_centroids": _centroid_rows(
            discovery_coordinates, discovery_y, retained
        ),
        "confirmation_centroids": _centroid_rows(
            confirmation_coordinates, confirmation_y, retained
        ),
        "axis_labels": ["F1", "F2", "F3"],
        "warning": (
            "Supervised diagnostic: axes maximize discovery count separation. "
            "Only frozen confirmation geometry is evidence of generalization."
        ),
    }


__all__ = ["discovery_fitted_fisher_lda3"]
