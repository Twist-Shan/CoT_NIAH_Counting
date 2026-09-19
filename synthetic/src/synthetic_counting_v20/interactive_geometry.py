"""Compatibility helpers for v20/v21 milestone manifold artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SITES_BY_MODE = {
    "nonthinking": ("final_answer",),
    "thinking": ("final_answer", "trace_index", "trace_marker"),
}


def _orient_basis(
    basis: np.ndarray,
    coordinates: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose deterministic PCA signs without changing the fitted subspace."""

    basis = basis.copy()
    coordinates = coordinates.copy()
    for axis in range(basis.shape[0]):
        if axis == 0 and np.std(coordinates[:, axis]) > 1e-12:
            correlation = np.corrcoef(labels.astype(float), coordinates[:, axis])[0, 1]
            flip = bool(np.isfinite(correlation) and correlation < 0)
        else:
            pivot = int(np.argmax(np.abs(basis[axis])))
            flip = bool(basis[axis, pivot] < 0)
        if flip:
            basis[axis] *= -1
            coordinates[:, axis] *= -1
    return basis, coordinates


def mean_first_pca(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    components: int = 6,
) -> dict[str, np.ndarray | float]:
    """Fit PCA to class centroids and return geometry used by the widget."""

    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    unique = np.unique(labels)
    if values.ndim != 2 or labels.ndim != 1 or len(values) != len(labels):
        raise ValueError("values must be [examples, hidden] and labels must align")
    if len(unique) < 2:
        raise ValueError("mean-first PCA needs at least two semantic classes")
    means = np.stack([values[labels == label].mean(axis=0) for label in unique])
    centered = means - means.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    available = min(components, len(vt))
    basis = vt[:available]
    coordinates = centered @ basis.T
    basis, coordinates = _orient_basis(basis, coordinates, unique)
    if available < components:
        coordinates = np.pad(coordinates, ((0, 0), (0, components - available)))
    variance = singular**2
    total_variance = float(variance.sum())
    if total_variance > 1e-12:
        full_ratio = variance / total_variance
    else:
        full_ratio = np.zeros_like(variance)
    ratio = np.pad(full_ratio[:components], (0, max(0, components - len(full_ratio))))
    displacement = np.diff(means, axis=0)
    norms = np.linalg.norm(displacement, axis=1)
    if len(displacement) > 1:
        adjacent_denominator = norms[:-1] * norms[1:]
        valid = adjacent_denominator > 1e-12
        if np.any(valid):
            adjacent = np.sum(displacement[:-1][valid] * displacement[1:][valid], axis=1)
            adjacent_cosine = float((adjacent / adjacent_denominator[valid]).mean())
        else:
            adjacent_cosine = 0.0
    else:
        adjacent_cosine = float("nan")
    chord = float(np.linalg.norm(means[-1] - means[0]))
    arc = float(norms.sum())
    effective_dimension = (
        float(1.0 / np.square(full_ratio).sum()) if total_variance > 1e-12 else 0.0
    )
    return {
        "labels": unique,
        "coordinates": coordinates,
        "variance": ratio,
        "effective_dimension": effective_dimension,
        "adjacent_cosine": adjacent_cosine,
        "straightness": chord / arc if arc > 1e-12 else 0.0,
    }


def build_interactive_geometry_table(run_dir: str | Path) -> pd.DataFrame:
    """Read the compact milestone sample coordinates produced by phase analysis."""

    run_dir = Path(run_dir)
    path = (
        run_dir
        / "analysis"
        / "phase_transition"
        / "tables"
        / "milestone_manifold_cloud_3d.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"phase manifold table is missing: {path}; run the phase stage first"
        )
    return pd.read_csv(path)


def write_interactive_geometry_table(run_dir: str | Path) -> Path:
    """Atomically persist the checkpoint-wise coordinate table."""

    run_dir = Path(run_dir)
    output = (
        run_dir
        / "analysis"
        / "phase_transition"
        / "tables"
        / "interactive_hidden_state_pca.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    build_interactive_geometry_table(run_dir).to_csv(temporary, index=False)
    temporary.replace(output)
    return output
