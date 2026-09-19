"""Low-rank coordinate maps for layerwise count-geometry analysis.

The functions in this module are deliberately independent of model loading.
They operate on paired NumPy state matrices and keep the PCA gauge explicit so
that resampled maps can be compared only after basis alignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CentroidGeometry:
    """A rank-k coordinate system fitted to class centroids."""

    classes: np.ndarray
    centroids: np.ndarray
    center: np.ndarray
    basis: np.ndarray
    singular_values: np.ndarray
    variance_capture: float

    def coordinates(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=np.float64)
        return (values - self.center) @ self.basis


@dataclass(frozen=True)
class LayerMapFit:
    """A ridge map between source and target centroid coordinates."""

    source: CentroidGeometry
    target: CentroidGeometry
    matrix: np.ndarray
    orthogonal_factor: np.ndarray
    stretch_factor: np.ndarray
    proper_rotation: np.ndarray
    orthogonal_determinant: float
    ridge: float
    training_r2: float
    training_normalized_rmse: float


def _validate_states_labels(
    states: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(states, dtype=np.float64)
    targets = np.asarray(labels)
    if values.ndim != 2:
        raise ValueError(f"states must be rank two, got {values.shape}")
    if targets.ndim != 1 or len(targets) != len(values):
        raise ValueError("labels must be one dimensional and match states")
    if not np.isfinite(values).all():
        raise ValueError("states contain non-finite values")
    return values, targets


def fit_centroid_geometry(
    states: np.ndarray,
    labels: np.ndarray,
    *,
    rank: int = 3,
    required_classes: np.ndarray | None = None,
) -> CentroidGeometry:
    """Fit a PCA basis to class centroids, never to the full state cloud."""

    values, targets = _validate_states_labels(states, labels)
    classes = np.unique(targets) if required_classes is None else np.asarray(required_classes)
    if len(classes) <= rank:
        raise ValueError(f"rank {rank} requires more than {rank} classes")
    missing = [value for value in classes if not np.any(targets == value)]
    if missing:
        raise ValueError(f"missing classes: {missing}")
    centroids = np.stack([values[targets == value].mean(axis=0) for value in classes])
    center = centroids.mean(axis=0)
    centered = centroids - center
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) < rank or singular_values[rank - 1] <= 1e-12:
        raise ValueError("centroid geometry is rank deficient")
    basis = vt[:rank].T
    total = float(np.square(singular_values).sum())
    capture = float(np.square(singular_values[:rank]).sum() / max(total, 1e-24))
    return CentroidGeometry(
        classes=np.asarray(classes).copy(),
        centroids=centroids,
        center=center,
        basis=basis,
        singular_values=singular_values,
        variance_capture=capture,
    )


def matrix_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Multivariate R2 with one shared denominator across coordinates."""

    truth = np.asarray(observed, dtype=np.float64)
    estimate = np.asarray(predicted, dtype=np.float64)
    if truth.shape != estimate.shape:
        raise ValueError(f"shape mismatch: {truth.shape} != {estimate.shape}")
    residual = float(np.square(truth - estimate).sum())
    centered = truth - truth.mean(axis=0, keepdims=True)
    total = float(np.square(centered).sum())
    return 1.0 - residual / max(total, 1e-24)


def normalized_rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Frobenius RMSE normalized by the centered target RMS."""

    truth = np.asarray(observed, dtype=np.float64)
    estimate = np.asarray(predicted, dtype=np.float64)
    if truth.shape != estimate.shape:
        raise ValueError(f"shape mismatch: {truth.shape} != {estimate.shape}")
    numerator = float(np.square(truth - estimate).mean())
    denominator = float(np.square(truth - truth.mean(axis=0, keepdims=True)).mean())
    return float(np.sqrt(numerator / max(denominator, 1e-24)))


def orthogonal_procrustes(
    source: np.ndarray, target: np.ndarray, *, proper: bool = False
) -> np.ndarray:
    """Return Q minimizing ``||source @ Q - target||_F``.

    When ``proper`` is true, the determinant is constrained to +1. The default
    O(k) solution is appropriate for PCA gauges, where sign flips are arbitrary.
    """

    left = np.asarray(source, dtype=np.float64)
    right = np.asarray(target, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("source and target must be equally shaped matrices")
    u, _, vt = np.linalg.svd(left.T @ right, full_matrices=False)
    if proper and np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1.0
    return u @ vt


def polar_factors(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return O(k) polar factor, PSD stretch, nearest SO(k), and det(O)."""

    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("polar decomposition requires a square matrix")
    u, singular, vt = np.linalg.svd(value, full_matrices=False)
    orthogonal = u @ vt
    stretch = (vt.T * singular) @ vt
    proper_u = u.copy()
    if np.linalg.det(proper_u @ vt) < 0:
        proper_u[:, -1] *= -1.0
    proper_rotation = proper_u @ vt
    return orthogonal, stretch, proper_rotation, float(np.linalg.det(orthogonal))


def fit_layer_map(
    source_states: np.ndarray,
    target_states: np.ndarray,
    labels: np.ndarray,
    *,
    rank: int = 3,
    ridge_relative_scale: float = 1e-3,
    required_classes: np.ndarray | None = None,
) -> LayerMapFit:
    """Fit a discovery-only map between adjacent-layer count coordinates."""

    source_values, targets = _validate_states_labels(source_states, labels)
    target_values, target_labels = _validate_states_labels(target_states, labels)
    if source_values.shape[0] != target_values.shape[0]:
        raise ValueError("source and target row counts differ")
    if not np.array_equal(targets, target_labels):
        raise ValueError("source and target labels differ")
    source = fit_centroid_geometry(
        source_values, targets, rank=rank, required_classes=required_classes
    )
    target = fit_centroid_geometry(
        target_values, targets, rank=rank, required_classes=source.classes
    )
    source_coordinates = source.coordinates(source.centroids)
    target_coordinates = target.coordinates(target.centroids)
    gram = source_coordinates.T @ source_coordinates
    ridge = float(ridge_relative_scale) * max(
        float(np.trace(gram) / rank), 1e-12
    )
    matrix = np.linalg.solve(
        gram + ridge * np.eye(rank), source_coordinates.T @ target_coordinates
    )
    predicted = source_coordinates @ matrix
    orthogonal, stretch, proper_rotation, determinant = polar_factors(matrix)
    return LayerMapFit(
        source=source,
        target=target,
        matrix=matrix,
        orthogonal_factor=orthogonal,
        stretch_factor=stretch,
        proper_rotation=proper_rotation,
        orthogonal_determinant=determinant,
        ridge=ridge,
        training_r2=matrix_r2(target_coordinates, predicted),
        training_normalized_rmse=normalized_rmse(target_coordinates, predicted),
    )


def evaluate_layer_map(
    fit: LayerMapFit,
    source_states: np.ndarray,
    target_states: np.ndarray,
) -> dict[str, float]:
    """Evaluate a frozen map on paired states in its fitted coordinate gauges."""

    source = np.asarray(source_states, dtype=np.float64)
    target = np.asarray(target_states, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or len(source) != len(target):
        raise ValueError("paired evaluation states must be rank-two and row matched")
    source_coordinates = fit.source.coordinates(source)
    target_coordinates = fit.target.coordinates(target)
    predicted = source_coordinates @ fit.matrix
    return {
        "r2": matrix_r2(target_coordinates, predicted),
        "normalized_rmse": normalized_rmse(target_coordinates, predicted),
        "direct_identity_r2": matrix_r2(target_coordinates, source_coordinates),
    }


def align_resampled_map_to_reference(
    resampled: LayerMapFit, reference: LayerMapFit
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gauge-align a resampled map to the full-discovery reference bases."""

    if resampled.matrix.shape != reference.matrix.shape:
        raise ValueError("map ranks differ")
    source_alignment = orthogonal_procrustes(
        resampled.source.basis, reference.source.basis
    )
    target_alignment = orthogonal_procrustes(
        resampled.target.basis, reference.target.basis
    )
    aligned = source_alignment.T @ resampled.matrix @ target_alignment
    return aligned, source_alignment, target_alignment


def principal_angles_degrees(left_basis: np.ndarray, right_basis: np.ndarray) -> np.ndarray:
    """Return canonical subspace angles in degrees."""

    left = np.asarray(left_basis, dtype=np.float64)
    right = np.asarray(right_basis, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("bases must share their ambient dimension")
    singular = np.linalg.svd(left.T @ right, compute_uv=False)
    singular = np.clip(singular, -1.0, 1.0)
    return np.degrees(np.arccos(singular))


def proper_rotation_geodesic_degrees(left: np.ndarray, right: np.ndarray) -> float:
    """Geodesic angle between two 3D proper rotations."""

    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.shape != (3, 3) or second.shape != (3, 3):
        raise ValueError("geodesic rotation angle is defined here only for rank three")
    if np.linalg.det(first) < 0.999 or np.linalg.det(second) < 0.999:
        raise ValueError("both inputs must be proper rotations")
    relative = first.T @ second
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def relative_frobenius_error(observed: np.ndarray, reference: np.ndarray) -> float:
    value = np.asarray(observed, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    if value.shape != target.shape:
        raise ValueError("matrix shapes differ")
    return float(
        np.linalg.norm(value - target, ord="fro")
        / max(float(np.linalg.norm(target, ord="fro")), 1e-12)
    )


def consecutive_full_operator_metrics(
    first: LayerMapFit, second: LayerMapFit
) -> dict[str, float]:
    """Compare consecutive full-space low-rank transport operators.

    A coordinate map reconstructs to ``T = U_source A U_target.T`` in the
    shared residual-stream feature space.  The operator is invariant to every
    admissible PCA gauge transformation.  This function evaluates its inner
    product using only rank-sized cross-Gram matrices and never materializes a
    hidden-size by hidden-size matrix.
    """

    if first.matrix.shape != second.matrix.shape:
        raise ValueError("consecutive maps must have the same rank")
    if (
        first.source.basis.shape[0] != first.target.basis.shape[0]
        or first.source.basis.shape[0] != second.source.basis.shape[0]
        or first.source.basis.shape[0] != second.target.basis.shape[0]
    ):
        raise ValueError("consecutive maps must share an ambient hidden space")
    source_cross = first.source.basis.T @ second.source.basis
    target_cross = second.target.basis.T @ first.target.basis
    inner = float(
        np.trace(first.matrix.T @ source_cross @ second.matrix @ target_cross)
    )
    first_norm = float(np.linalg.norm(first.matrix, ord="fro"))
    second_norm = float(np.linalg.norm(second.matrix, ord="fro"))
    cosine = inner / max(first_norm * second_norm, 1e-12)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    difference_squared = max(
        first_norm**2 + second_norm**2 - 2.0 * inner, 0.0
    )
    relative_drift = float(
        np.sqrt(difference_squared)
        / max((first_norm + second_norm) / 2.0, 1e-12)
    )
    return {
        "full_operator_inner_product_to_next": inner,
        "full_operator_cosine_to_next": cosine,
        "full_operator_relative_drift_to_next": relative_drift,
    }
