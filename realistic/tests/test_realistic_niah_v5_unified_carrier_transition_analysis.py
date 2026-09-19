from __future__ import annotations

import pytest

from scripts.analyze_realistic_niah_v5_unified_carrier_transition import (
    cluster_bootstrap_analysis,
    validate_balanced_trials,
)


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in (1, 2, 3):
        for dose in (-1, 1):
            for carrier, next_scale, logodds in (
                ("residual_count_subspace", 0.5, 0.0),
                ("residual_count_plus_kv", 1.0, 0.2),
            ):
                rows.append(
                    {
                        "seed": seed,
                        "receiver": 5,
                        "dose": dose,
                        "carrier": carrier,
                        "current_shift": 2.0 * dose,
                        "next_shift": next_scale * dose,
                        "current_exact": True,
                        "next_exact": carrier == "residual_count_plus_kv",
                        "donor_vs_receiver_mean_logodds_change": logodds,
                    }
                )
    return rows


def test_cluster_bootstrap_preserves_paired_carrier_contrast() -> None:
    analysis = cluster_bootstrap_analysis(
        _rows(), draws=100, bootstrap_seed=7
    )
    residual = analysis["carriers"]["residual_count_subspace"]
    distributed = analysis["carriers"]["residual_count_plus_kv"]
    contrast = analysis["contrasts"]["distributed_minus_residual"]
    assert residual["mean_signed_current_shift"] == pytest.approx(2.0)
    assert residual["current_to_next_slope"] == pytest.approx(0.25)
    assert distributed["current_to_next_slope"] == pytest.approx(0.5)
    assert contrast["difference"]["mean_signed_next_shift"] == pytest.approx(0.5)
    assert contrast["difference"]["current_to_next_slope"] == pytest.approx(0.25)


def test_balanced_trial_validation_rejects_duplicate_cell() -> None:
    rows = _rows()
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="Duplicate trial cell"):
        validate_balanced_trials(rows)
