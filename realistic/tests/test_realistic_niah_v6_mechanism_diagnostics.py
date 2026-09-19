from __future__ import annotations

import pandas as pd

from scripts.analyze_realistic_niah_v5_local_terminal_token_state_bridge import (
    _optional_constant_string,
)
from scripts.analyze_realistic_niah_v6_targeted_counter_write_diagnostic import (
    analyze as analyze_counter_write_diagnostic,
)
from realistic_niah_v5.causal import target_city_head_ablation_geometry
from realistic_niah_v5.targeted_counter_write import _head_ablation_positions


def test_target_city_support_exposes_query_last_pre_city_alias() -> None:
    geometry = target_city_head_ablation_geometry(
        prompt_token_count=100,
        query_output_token_index=20,
        target_output_token_start=21,
        target_output_token_end=24,
        scope="registered_query_through_city_prefix",
    )
    assert geometry["registered_query_full_sequence_token"] == 120
    assert geometry["last_pre_city_predictor_full_sequence_token"] == 120
    assert geometry["registered_query_equals_last_pre_city_predictor"] is True
    assert geometry["head_ablation_positions"] == [120, 121, 122]
    assert geometry["score_positions"] == [120, 121, 122]


def test_target_city_support_keeps_interstitial_query_positions() -> None:
    geometry = target_city_head_ablation_geometry(
        prompt_token_count=10,
        query_output_token_index=5,
        target_output_token_start=8,
        target_output_token_end=10,
        scope="registered_query_through_city_prefix",
    )
    assert geometry["registered_query_to_last_pre_city_distance"] == 2
    assert geometry["head_ablation_positions"] == [15, 16, 17, 18]
    assert geometry["score_positions"] == [17, 18]


def test_targeted_counter_write_query_through_carrier_scope_is_inclusive() -> None:
    assert _head_ablation_positions(
        query_position=10,
        carrier_positions=(13, 14, 15),
        scope="query_local",
    ) == (10,)
    assert _head_ablation_positions(
        query_position=10,
        carrier_positions=(13, 14, 15),
        scope="query_through_carrier",
    ) == (10, 11, 12, 13, 14, 15)


def test_targeted_counter_write_rejects_carrier_before_query() -> None:
    try:
        _head_ablation_positions(
            query_position=10,
            carrier_positions=(9, 11),
            scope="query_through_carrier",
        )
    except ValueError as error:
        assert "strictly after" in str(error)
    else:
        raise AssertionError("invalid carrier geometry was accepted")


def test_local_terminal_analysis_propagates_only_constant_diagnostic_label() -> None:
    label = "POSTHOC_DIAGNOSTIC_SPLIT_REUSE"
    assert _optional_constant_string(pd.DataFrame({"analysis_status": [label, label]}), "analysis_status") == label
    assert _optional_constant_string(pd.DataFrame({"x": [1]}), "analysis_status") is None
    try:
        _optional_constant_string(
            pd.DataFrame({"analysis_status": [label, "UNREGISTERED"]}),
            "analysis_status",
        )
    except ValueError as error:
        assert "one constant analysis_status" in str(error)
    else:
        raise AssertionError("mixed diagnostic labels were accepted")


def _counter_write_frame() -> pd.DataFrame:
    rows = []
    conditions = {
        "clean": (0.0, 0.0),
        "selected_mask": (2.0, 3.0),
        "random_mask_r1": (0.5, 0.5),
        "random_mask_r2": (0.5, 0.5),
        "random_mask_r3": (0.5, 0.5),
        "selected_mask_clean_carrier_restore": (0.0, 0.5),
        "selected_mask_matched_position_state_control": (0.0, 2.5),
    }
    for seed in range(3):
        for condition, (carrier, boundary) in conditions.items():
            rows.append(
                {
                    "experiment_id": "teacher_forced_targeted_counter_write",
                    "model_label": "Gemma4-E4B",
                    "seed": seed,
                    "request_id": f"request-{seed}",
                    "condition": condition,
                    "teacher_forced_trace_tokens": True,
                    "selection_rank_used": False,
                    "head_ablation_scope": "query_through_carrier",
                    "head_ablation_position_count": 4,
                    "carrier_state_rms_distance_mean_downstream": carrier,
                    "boundary_state_rms_distance_to_clean_final": boundary,
                }
            )
    return pd.DataFrame(rows)


def test_counter_write_diagnostic_requires_and_detects_spanning_support() -> None:
    _effects, claims = analyze_counter_write_diagnostic(
        _counter_write_frame(),
        phase="confirmation",
        expected_seeds=3,
        expected_scope="query_through_carrier",
        random_seed=7,
    )
    assert claims["targeted_counter_write_strong_gate_pass"] is True
    assert claims["original_query_local_null_retained"] is True


def test_counter_write_diagnostic_rejects_query_local_rows() -> None:
    frame = _counter_write_frame()
    frame["head_ablation_scope"] = "query_local"
    try:
        analyze_counter_write_diagnostic(
            frame,
            phase="confirmation",
            expected_seeds=3,
            expected_scope="query_through_carrier",
            random_seed=7,
        )
    except ValueError as error:
        assert "scope changed" in str(error)
    else:
        raise AssertionError("query-local rows entered the spanning diagnostic")
