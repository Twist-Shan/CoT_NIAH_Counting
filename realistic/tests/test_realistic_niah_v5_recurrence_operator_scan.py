from __future__ import annotations

import pytest

from scripts.run_realistic_niah_v5_recurrence_operator_scan import (
    summarize_operator_rows,
    valid_scan_pairs,
)


def test_valid_scan_pairs_filters_donors_without_successors() -> None:
    pairs = valid_scan_pairs((3, 7), (-6, -2, -1, 1, 2, 6))
    assert pairs == (
        (3, 1, -2),
        (3, 2, -1),
        (3, 4, 1),
        (3, 5, 2),
        (3, 9, 6),
        (7, 1, -6),
        (7, 5, -2),
        (7, 6, -1),
        (7, 8, 1),
        (7, 9, 2),
    )


def test_operator_summary_prefers_reset_when_next_state_is_overwritten() -> None:
    rows = []
    for donor, current in ((2, 2.0), (4, 4.0), (8, 8.0)):
        rows.append(
            {
                "carrier": "whole_state",
                "donor": donor,
                "current_prediction": donor,
                "next_prediction": 6,
                "current_soft": current,
                "next_soft": 6.0,
                "clean_current_soft": 5.0,
                "clean_next_soft": 6.0,
            }
        )
    summary = summarize_operator_rows(rows)["whole_state"]
    assert summary["current_target_exact"] == 3
    assert summary["next_target_exact"] == 0
    assert summary["operator_rmse"]["reset_to_clean_next"] == pytest.approx(0.0)
    assert summary["operator_rmse"]["plus_one"] > 1.0


def test_valid_scan_pairs_rejects_zero_dose() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        valid_scan_pairs((5,), (0, 1))
