from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_realistic_niah_v5_anthropic_count_manifold import (
    build_natural_reference,
    gauge_scores,
    transition_metrics,
    vector_match_metrics,
)


def _natural_rows(curved: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in range(4):
        context = np.asarray([0.2 * seed, -0.1 * seed, 0.05 * seed, 0.0])
        for count in range(1, 11):
            if curved:
                theta = count / 3.0
                signal = np.asarray([np.cos(theta), np.sin(theta), count / 10.0, 0.0])
            else:
                signal = np.asarray([float(count), -float(count), 0.0, 0.0])
            rows.append(
                {
                    "seed": seed,
                    "layer": 2,
                    "gold_total": 10,
                    "raw_item_ordinal": count,
                    "probe_scores": (signal + context).tolist(),
                }
            )
    return rows


def test_gauge_scores_removes_common_offset() -> None:
    assert np.allclose(gauge_scores([3.0, 4.0, 5.0]), [-1.0, 0.0, 1.0])


def test_linear_natural_trajectory_is_rank_one_and_fixed_increment() -> None:
    reference = build_natural_reference(_natural_rows(curved=False), layer=2)
    assert reference.summary["pc1_variance_fraction"] == pytest.approx(1.0)
    assert reference.summary["linear_count_trajectory_r2"] == pytest.approx(1.0)
    assert reference.summary["fixed_increment_vector_r2_over_sample_steps"] == pytest.approx(
        1.0
    )


def test_curved_natural_trajectory_uses_more_than_one_dimension() -> None:
    reference = build_natural_reference(_natural_rows(curved=True), layer=2)
    assert reference.summary["pc1_variance_fraction"] < 0.9
    assert reference.summary["rank90"] >= 2
    assert reference.summary["linear_count_trajectory_r2"] < 0.9


def test_transition_metrics_identifies_aligned_and_orthogonal_changes() -> None:
    reference = build_natural_reference(_natural_rows(curved=False), layer=2)
    tangent = reference.tangents[3]
    aligned = transition_metrics(2.0 * tangent, tangent, reference)
    assert aligned["same_tangent_cosine"] == pytest.approx(1.0)
    assert aligned["reference_scale"] == pytest.approx(2.0)
    assert aligned["orthogonal_over_reference"] == pytest.approx(0.0, abs=1e-10)

    orthogonal = np.asarray([0.0, 0.0, 1.0, -1.0])
    result = transition_metrics(orthogonal, tangent, reference)
    assert result["same_tangent_cosine"] == pytest.approx(0.0, abs=1e-10)


def test_vector_match_metrics_reports_direction_scale_and_extra_component() -> None:
    target = np.asarray([1.0, -1.0, 0.0])
    observed = np.asarray([0.5, -0.5, 1.0])
    result = vector_match_metrics(observed, target)
    assert result["matched_textual_scale"] == pytest.approx(0.5)
    assert result["matched_textual_orthogonal_over_target"] == pytest.approx(
        1.0 / np.sqrt(2.0)
    )
