from __future__ import annotations

import pytest

from scripts.analyze_realistic_niah_v5_recurrence_operator_scan import (
    analyze_operator_scan,
    validate_trials,
)


def _rows(operator: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in (1, 2, 3):
        for receiver in (3, 5):
            for donor in (receiver - 1, receiver + 1):
                if operator == "reset":
                    next_prediction = receiver + 1
                    next_soft = float(receiver + 1)
                elif operator == "plus_one":
                    next_prediction = donor + 1
                    next_soft = float(donor + 1)
                else:
                    raise ValueError(operator)
                rows.append(
                    {
                        "seed": seed,
                        "receiver": receiver,
                        "donor": donor,
                        "dose": donor - receiver,
                        "carrier": "whole_state",
                        "current_prediction": donor,
                        "next_prediction": next_prediction,
                        "clean_next_prediction": receiver + 1,
                        "current_soft": float(donor),
                        "next_soft": next_soft,
                        "clean_current_soft": float(receiver),
                        "clean_next_soft": float(receiver + 1),
                        "current_shift": float(donor - receiver),
                        "next_shift": float(next_soft - (receiver + 1)),
                    }
                )
    return rows


def test_analysis_ranks_reset_for_overwritten_state() -> None:
    analysis = analyze_operator_scan(
        _rows("reset"), bootstrap_draws=100, bootstrap_seed=4
    )
    result = analysis["carriers"]["whole_state"]
    assert result["metrics"]["reset_accuracy"] == pytest.approx(1.0)
    assert result["metrics"]["next_target_rate"] == pytest.approx(0.0)
    assert result["leave_one_seed_out_soft_rmse"]["reset_to_clean_next"] == pytest.approx(
        0.0
    )


def test_analysis_ranks_plus_one_for_true_increment() -> None:
    analysis = analyze_operator_scan(
        _rows("plus_one"), bootstrap_draws=100, bootstrap_seed=4
    )
    result = analysis["carriers"]["whole_state"]
    assert result["metrics"]["next_target_rate"] == pytest.approx(1.0)
    assert result["metrics"]["reset_accuracy"] == pytest.approx(0.0)


def test_validation_rejects_duplicate_operator_cell() -> None:
    rows = _rows("reset")
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="Duplicate operator-scan cell"):
        validate_trials(rows)
