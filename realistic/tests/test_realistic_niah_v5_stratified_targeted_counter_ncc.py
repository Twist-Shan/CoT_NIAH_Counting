from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from realistic_niah_v5.stratified_targeted_counter_ncc import (
    _validate_causal_reach,
    grammar_timing,
    stratified_endpoint_positions,
)
from scripts.analyze_realistic_niah_v5_stratified_targeted_counter_ncc import (
    _correct_margin,
)


def _span(start: int, end: int) -> dict[str, object]:
    return {
        "status": "ok",
        "full_sequence_token_start": start,
        "full_sequence_token_end": end,
    }


def _index(position: int) -> dict[str, object]:
    return {
        "status": "ok",
        "full_sequence_token_index": position,
    }


def test_city_to_rank_endpoints_precede_and_exclude_marker() -> None:
    registry = SimpleNamespace(trace_items=((10, 20), (20, 40)))
    event = {
        "grammar_class": "adjacent_rank_after_city",
        "sites": {
            "city_target_span": _span(25, 27),
            "pre_marker_state": _index(30),
            "rank_evidence_core_span": _span(31, 33),
        },
    }
    assert grammar_timing(event) == "rank_after_city"
    endpoints = stratified_endpoint_positions(
        registry, event, occurrence=2, timing="rank_after_city"
    )
    assert endpoints == {
        "pre_marker_exact": (30,),
        "pre_marker_suffix4": (27, 28, 29, 30),
    }
    assert not set(endpoints["pre_marker_suffix4"]) & {31, 32}


def test_city_to_rank_rejects_teacher_forced_marker_as_endpoint() -> None:
    registry = SimpleNamespace(trace_items=((20, 40),))
    event = {
        "grammar_class": "same_unit_rank_after_city",
        "sites": {
            "city_target_span": _span(25, 27),
            "pre_marker_state": _index(31),
            "rank_evidence_core_span": _span(31, 33),
        },
    }
    with pytest.raises(ValueError, match="precede the marker"):
        stratified_endpoint_positions(
            registry, event, occurrence=1, timing="rank_after_city"
        )


def test_rank_to_city_keeps_city_through_commit_tail() -> None:
    registry = SimpleNamespace(trace_items=((10, 20), (20, 40)))
    event = {
        "grammar_class": "same_unit_rank_before_city",
        "sites": {
            "city_target_span": _span(25, 27),
            "post_update_commit_state": _index(35),
        },
    }
    assert stratified_endpoint_positions(
        registry, event, occurrence=2, timing="rank_before_city"
    ) == {"city_to_commit": tuple(range(25, 36))}


def test_causal_reach_requires_disjoint_exact_layer_matched_controls() -> None:
    banks = [
        {"condition": "clean", "heads": []},
        {"condition": "selected_bank", "heads": [(2, 0), (4, 0)]},
        {"condition": "layer_matched_random", "heads": [(2, 1), (4, 1)]},
        {"condition": "layer_matched_random", "heads": [(2, 2), (4, 2)]},
        {"condition": "layer_matched_random", "heads": [(2, 3), (4, 3)]},
    ]
    assert _validate_causal_reach(banks, capture_start_layer=5) == 4
    with pytest.raises(ValueError, match="one layer above"):
        _validate_causal_reach(banks, capture_start_layer=4)
    overlapping = [dict(row) for row in banks]
    overlapping[2] = {
        "condition": "layer_matched_random",
        "heads": [(2, 0), (4, 1)],
    }
    with pytest.raises(ValueError, match="overlaps"):
        _validate_causal_reach(overlapping, capture_start_layer=5)


def test_correct_centroid_margin_has_expected_sign() -> None:
    distances = np.asarray([[1.0, 4.0], [5.0, 2.0]])
    classes = np.asarray([1, 2])
    labels = np.asarray([1, 1])
    margins = _correct_margin(distances, classes, labels)
    assert margins.tolist() == [3.0, -3.0]
