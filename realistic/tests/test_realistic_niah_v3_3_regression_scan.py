from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analyze_realistic_niah_v3_3_regression_scan import (
    FIXED_N_CANDIDATE_IDS,
    N_FIXED_LENGTH_TERMS,
    add_predictors,
    coefficient_prediction,
    n_fixed_design,
)


def test_fixed_n10_registry_is_length_only() -> None:
    assert FIXED_N_CANDIDATE_IDS == ("intercept", "L_k", "logL")


def test_predictor_definitions_match_v3_2() -> None:
    frame = add_predictors(pd.DataFrame({"N": [2], "L": [8000]}))
    row = frame.iloc[0]
    assert row["L_k"] == 8.0
    assert np.isclose(row["logN"], np.log(2.0))
    assert np.isclose(row["logL"], np.log(8.0))
    assert row["invN"] == 0.5
    assert row["N_x_L_k"] == 16.0


def test_coefficient_prediction_uses_registered_terms() -> None:
    frame = add_predictors(pd.DataFrame({"N": [2, 4], "L": [1000, 2000]}))
    coefficients = pd.DataFrame(
        {
            "term": ["intercept", "N", "logL"],
            "estimate": [1.0, 2.0, -0.5],
        }
    )
    expected = 1.0 + 2.0 * frame["N"].to_numpy() - 0.5 * frame["logL"].to_numpy()
    assert np.allclose(coefficient_prediction(frame, coefficients), expected)


def test_predictors_can_be_recomputed_on_cell_tables() -> None:
    cells = add_predictors(pd.DataFrame({"N": [10], "L": [100000]}))
    assert cells.loc[0, "invN"] == 0.1
    assert cells.loc[0, "invN_x_L_k"] == 10.0
    assert np.isclose(cells.loc[0, "invN_x_logL"], 0.1 * np.log(100.0))


def test_n_fixed_design_has_one_intercept_per_registered_n() -> None:
    frame = add_predictors(
        pd.DataFrame({"N": [1, 2, 1], "L": [1000, 2000, 4000]})
    )
    matrix, names = n_fixed_design(frame, (1, 2), "logL")
    assert names == ("alpha_N=1", "alpha_N=2", "beta_logL")
    assert matrix.shape == (3, 3)
    assert np.allclose(matrix[:, :2], [[1, 0], [0, 1], [1, 0]])
    assert np.allclose(matrix[:, 2], np.log([1.0, 2.0, 4.0]))


def test_n_fixed_registry_compares_linear_and_log_length() -> None:
    assert N_FIXED_LENGTH_TERMS == (None, "L_k", "logL")
