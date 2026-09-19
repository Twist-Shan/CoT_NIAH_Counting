import json

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_realistic_niah_v3_2_inverse_n_candidate_extension import (
    DEFAULT_EXTENSION_CONFIG,
    add_inverse_count_terms,
    load_inverse_candidates,
)


def test_inverse_count_registry_is_hierarchical_and_has_five_candidates() -> None:
    candidates = load_inverse_candidates(DEFAULT_EXTENSION_CONFIG)

    assert len(candidates) == 5
    assert candidates[0].id == "invN"
    assert candidates[3].parent == "invN__L_k"
    assert candidates[3].interaction == "invN_x_L_k"
    assert set(candidates[3].terms) == {"invN", "L_k", "invN_x_L_k"}
    assert candidates[4].parent == "invN__logL"
    assert set(candidates[4].terms) == {"invN", "logL", "invN_x_logL"}


def test_inverse_count_terms_are_computed_exactly() -> None:
    frame = pd.DataFrame(
        {
            "N": [1, 2, 20],
            "L_k": [1.0, 5.0, 20.0],
            "logL": np.log([1.0, 5.0, 20.0]),
        }
    )
    result = add_inverse_count_terms(frame)

    np.testing.assert_allclose(result["invN"], [1.0, 0.5, 0.05])
    np.testing.assert_allclose(result["invN_x_L_k"], [1.0, 2.5, 1.0])
    np.testing.assert_allclose(
        result["invN_x_logL"], result["invN"] * result["logL"]
    )


def test_inverse_count_rejects_nonpositive_n() -> None:
    frame = pd.DataFrame({"N": [0], "L_k": [1.0], "logL": [0.0]})
    with pytest.raises(ValueError, match="N > 0"):
        add_inverse_count_terms(frame)


def test_extension_config_declares_no_bootstrap_or_selection_change() -> None:
    payload = json.loads(DEFAULT_EXTENSION_CONFIG.read_text(encoding="utf-8"))

    assert payload["bootstrap_repetitions"] == 0
    assert payload["base_registry_size"] == 13
    assert "Identical to frozen V3.2" in payload["selection_and_validation"]
