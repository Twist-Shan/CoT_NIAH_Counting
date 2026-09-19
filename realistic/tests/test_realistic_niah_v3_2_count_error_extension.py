import numpy as np
import pandas as pd
import pytest

from scripts.analyze_realistic_niah_v3_2_count_error_extension import (
    deviation_summary,
    fit_continuous_candidate,
    selected_bias_influence,
)
from scripts.analyze_realistic_niah_v3_2_empirical_laws import Candidate


def test_deviation_summary_symmetrically_trims_absolute_errors_for_mae() -> None:
    errors = [-100.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 100.0]
    result = deviation_summary(errors)

    assert result["n_parseable"] == 10
    assert result["trim_count_each_tail"] == 1
    assert result["conditional_mae"] == np.mean(np.abs(errors))
    assert result["trimmed_conditional_mae_10"] == np.mean(
        sorted(np.abs(errors))[1:-1]
    )
    assert result["raw_signed_mean"] == np.mean(errors)
    assert result["trimmed_signed_mean"] == np.mean(sorted(errors)[1:-1])
    assert result["max_abs_raw_error"] == 100.0
    assert result["max_abs_retained_error"] == 3.0
    assert result["max_abs_retained_for_mae"] == 100.0


def test_trimmed_mae_fit_uses_identity_scale_predictions() -> None:
    n_levels = tuple(range(1, 15))
    l_levels = tuple(1000 * index for index in range(1, 9))
    frame = pd.DataFrame(
        [(n, length) for n in n_levels for length in l_levels],
        columns=["N", "L"],
    )
    frame["L_k"] = frame["L"] / 1000.0
    frame["trimmed_conditional_mae_10"] = (
        0.02 * frame["N"] + 0.03 * frame["L_k"]
    )
    frame["n_parseable"] = 30
    frame["n_total"] = 30
    candidate = Candidate("N__L_k", ("N", "L_k"))

    metrics, coefficients = fit_continuous_candidate(
        frame,
        candidate,
        outcome_family="trimmed_conditional_mae_10",
        outcome_column="trimmed_conditional_mae_10",
        n_levels=n_levels,
        l_levels=l_levels,
    )

    assert metrics["model_scale"] == "trimmed_conditional_mae_10_identity"
    assert metrics["inverse_link"] == "identity"
    assert metrics["minimum_prediction"] == pytest.approx(
        frame["trimmed_conditional_mae_10"].min(), abs=1e-10
    )
    assert metrics["cv_r2"] > 0.9
    assert {row["term"] for row in coefficients} == {"intercept", "N", "L_k"}


def test_cooks_distance_flags_a_deliberately_influential_bias_cell() -> None:
    n_levels = tuple(range(1, 15))
    l_levels = tuple(1000 * index for index in range(1, 9))
    frame = pd.DataFrame(
        [(n, length) for n in n_levels for length in l_levels],
        columns=["N", "L"],
    )
    frame["L_k"] = frame["L"] / 1000.0
    frame["trimmed_signed_bias_10"] = (
        0.10 * frame["N"]
        + 0.03 * frame["L_k"]
        + 0.01 * np.sin(np.arange(len(frame)))
    )
    outlier = (frame["N"] == 14) & (frame["L"] == 8000)
    frame.loc[outlier, "trimmed_signed_bias_10"] += 50.0
    candidate = Candidate("N__L_k", ("N", "L_k"))

    result = selected_bias_influence(frame, candidate)

    assert result["diagnostic_estimable"] is True
    assert result["influential_cells"] >= 1
    assert result["max_cooks_N"] == 14
    assert result["max_cooks_L"] == 8000
    assert result["surface_max_abs_change_after_dropping_influential"] > 0
