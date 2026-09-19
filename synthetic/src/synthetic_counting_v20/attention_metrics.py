"""Shared attention-profile metrics for v20 mechanism analyses."""

from __future__ import annotations

import math

import numpy as np


def broad_profile_metrics(values: np.ndarray) -> dict[str, np.ndarray]:
    """Return mass/coverage diagnostics for target-occurrence weights.

    The last axis indexes the ``N`` target occurrences.  The primary broad
    score follows the realistic-NiaH reports:

    ``M = sum(a_i)``, ``C_eff = exp(H(a / M)) / N``, ``B_eff = M * C_eff``.

    Normalized entropy is retained as a separately named diagnostic.  Unlike
    the historical v20 score, effective coverage gives a head that uniformly
    covers ``K`` of ``N`` occurrences a coverage of exactly ``K/N`` and treats
    the valid ``N=1`` case as full coverage when mass is positive.
    """

    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError("values must have a non-empty occurrence axis")
    if np.any(values < 0):
        raise ValueError("attention weights must be non-negative")

    occurrence_count = int(values.shape[-1])
    total_mass = values.sum(axis=-1)
    positive_mass = total_mass > 0
    probabilities = np.divide(
        values,
        total_mass[..., None],
        out=np.zeros_like(values, dtype=np.float64),
        where=positive_mass[..., None],
    )
    log_probabilities = np.zeros_like(probabilities)
    np.log(probabilities, out=log_probabilities, where=probabilities > 0)
    entropy = -(probabilities * log_probabilities).sum(axis=-1)
    effective_number = np.where(positive_mass, np.exp(entropy), 0.0)
    effective_coverage = effective_number / float(occurrence_count)

    if occurrence_count == 1:
        normalized_entropy = positive_mass.astype(np.float64)
        legacy_normalized_entropy = np.zeros_like(total_mass, dtype=np.float64)
    else:
        normalized_entropy = entropy / math.log(occurrence_count)
        legacy_normalized_entropy = normalized_entropy

    return {
        "total_target_mass": total_mass,
        "effective_number": effective_number,
        "effective_coverage": effective_coverage,
        "normalized_entropy": normalized_entropy,
        "broad_score": total_mass * effective_coverage,
        "entropy_broad_score": total_mass * normalized_entropy,
        "legacy_entropy_broad_score": total_mass * legacy_normalized_entropy,
    }


__all__ = ["broad_profile_metrics"]
