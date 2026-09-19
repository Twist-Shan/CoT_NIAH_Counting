from __future__ import annotations

import numpy as np
import pytest

from realistic_niah_v5.kv_counter_transition import (
    history_occurrences,
    item_bin_positions,
    normalized_span_bins,
    projection_kinds,
)
from scripts.run_realistic_niah_v5_kv_counter_transition import (
    BankSpec,
    adjacent_count_tangent,
    build_kv_directions,
    default_bank_specs,
    orthogonal_kv_tangents,
    summarize,
)
from scripts.run_realistic_niah_v5_kv_transition_movie import (
    progress_bin,
    summarize_movie,
    transition_position_metadata,
)


def test_normalized_span_bins_cover_every_token_once() -> None:
    groups = normalized_span_bins(10, 21, bins=4)
    assert [len(group) for group in groups] == [2, 3, 3, 3]
    assert tuple(position for group in groups for position in group) == tuple(
        range(10, 21)
    )
    with pytest.raises(ValueError, match="one token per bin"):
        normalized_span_bins(0, 3, bins=4)


def test_item_bins_and_history_scopes_are_outcome_independent() -> None:
    items = [(index * 6, (index + 1) * 6) for index in range(10)]
    positions = item_bin_positions(items, bins=3)
    assert positions[1] == ((0, 1), (2, 3), (4, 5))
    assert positions[10] == ((54, 55), (56, 57), (58, 59))
    assert history_occurrences(6, "all_history") == (1, 2, 3, 4, 5, 6)
    assert history_occurrences(6, "last_4") == (3, 4, 5, 6)
    assert projection_kinds("kv") == ("k", "v")


def test_default_bank_specs_freeze_full_and_three_layer_bands() -> None:
    specs = default_bank_specs(range(14, 24))
    by_name = {spec.name: spec for spec in specs}
    assert by_name["all_history_kv"].layers == tuple(range(14, 24))
    assert by_name["all_history_kv_early"].layers == (14, 15, 16)
    assert by_name["all_history_kv_middle"].layers == (17, 18, 19)
    assert by_name["all_history_kv_late"].layers == (20, 21, 22, 23)


def test_adjacent_tangent_is_central_and_projected() -> None:
    states = np.zeros((2, 10, 3), dtype=np.float32)
    states[:, :, 0] = np.arange(1, 11, dtype=np.float32)
    states[:, :, 1] = 100.0 * np.arange(1, 11, dtype=np.float32)
    basis = np.asarray([[1.0], [0.0], [0.0]], dtype=np.float32)
    assert adjacent_count_tangent(states, occurrence=5, basis=basis).tolist() == [
        1.0,
        0.0,
        0.0,
    ]
    assert adjacent_count_tangent(states, occurrence=1, basis=basis).tolist() == [
        1.0,
        0.0,
        0.0,
    ]


def test_build_kv_directions_covers_full_selected_spans() -> None:
    positions = {
        occurrence: ((occurrence * 10,), (occurrence * 10 + 1,))
        for occurrence in range(1, 11)
    }
    tangents = {
        (14, kind, bin_index, occurrence): np.asarray(
            [occurrence, bin_index + 1], dtype=np.float32
        )
        for kind in ("k", "v")
        for bin_index in range(2)
        for occurrence in range(1, 11)
    }
    spec = BankSpec("test", "last_4", "kv", (14,))
    directions = build_kv_directions(
        spec,
        receiver=5,
        dose=-1,
        scale=2.0,
        bins_by_occurrence=positions,
        tangents=tangents,
    )
    assert set(directions) == {(14, "k"), (14, "v")}
    assert set(directions[(14, "k")]) == {
        20,
        21,
        30,
        31,
        40,
        41,
        50,
        51,
    }
    assert directions[(14, "v")][51].tolist() == [-10.0, -4.0]


def test_orthogonal_kv_tangents_match_norm_outside_count_span() -> None:
    basis = np.asarray([[1.0], [0.0], [0.0], [0.0]], dtype=np.float32)
    tangent = np.asarray([2.0, 0.0, 0.0, 0.0], dtype=np.float32)
    controls = orthogonal_kv_tangents(
        {(14, "v", 0): basis},
        {(14, "v", 0, 5): tangent},
        seed=7,
    )
    control = controls[(14, "v", 0, 5)]
    assert float(control @ basis[:, 0]) == pytest.approx(0.0, abs=1e-6)
    assert float(control @ tangent) == pytest.approx(0.0, abs=1e-6)
    assert np.linalg.norm(control) == pytest.approx(np.linalg.norm(tangent))


def test_summary_retention_uses_current_displacement_as_denominator() -> None:
    rows = []
    for dose in (0, -1, 1):
        for horizon in (0, 1):
            natural = 5 + horizon
            shift = 0.0 if dose == 0 else dose * (2.0 if horizon == 0 else 1.0)
            prediction = natural if dose == 0 else natural + dose
            rows.append(
                {
                    "bank": "bank",
                    "seed": 1,
                    "receiver": 5,
                    "dose": dose,
                    "horizon": horizon,
                    "probe_softmax_expected_count": natural + shift,
                    "probe_prediction": prediction,
                    "exact": True,
                }
            )
    result = summarize(rows)["bank"]
    assert result["pooled_current_to_next_retention"] == pytest.approx(0.5)
    assert result["discrete"]["horizon_1"]["exact"] == 2


def test_transition_movie_positions_include_every_native_token() -> None:
    items = [(index * 5, index * 5 + 4) for index in range(10)]
    metadata = transition_position_metadata(
        items, receiver=1, current_boundary=3, next_boundary=8
    )
    assert [row["position"] for row in metadata] == list(range(3, 9))
    assert metadata[0]["role"] == "current_boundary"
    assert metadata[1]["role"] == "pre_item_separator"
    assert metadata[-1]["role"] == "next_boundary"
    assert metadata[0]["transition_progress"] == 0.0
    assert metadata[-1]["transition_progress"] == 1.0


def test_progress_bins_reserve_formal_boundary_endpoints() -> None:
    assert progress_bin(0.0, bins=12) == 0
    assert progress_bin(0.001, bins=12) == 1
    assert progress_bin(0.999, bins=12) == 10
    assert progress_bin(1.0, bins=12) == 11


def test_movie_summary_recovers_natural_increment_and_half_retention() -> None:
    rows = []
    for seed in (11, 12):
        for position, active_bin, clean_soft in ((100, 0, 5.0), (110, 1, 6.0)):
            rows.append(
                {
                    "seed": seed,
                    "receiver": 5,
                    "read_layer": 24,
                    "bank": "clean",
                    "dose": 0,
                    "position": position,
                    "progress_bin": active_bin,
                    "probe_softmax_expected_count": clean_soft,
                    "probe_scores": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0 if active_bin == 0 else 0.0,
                        0.0 if active_bin == 0 else 1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "probe_prediction": 5 if active_bin == 0 else 6,
                    "is_current_boundary": active_bin == 0,
                    "is_next_boundary": active_bin == 1,
                }
            )
        for dose in (-1, 1):
            for position, active_bin, clean_soft, retention in (
                (100, 0, 5.0, 1.0),
                (110, 1, 6.0, 0.5),
            ):
                rows.append(
                    {
                        "seed": seed,
                        "receiver": 5,
                        "read_layer": 24,
                        "bank": "all_history_kv",
                        "dose": dose,
                        "position": position,
                        "progress_bin": active_bin,
                        "probe_softmax_expected_count": clean_soft + dose * retention,
                        "is_current_boundary": active_bin == 0,
                        "is_next_boundary": active_bin == 1,
                        "exact_transition_target": True,
                        "prediction_changed_from_clean": True,
                        "directionally_correct_change": True,
                    }
                )
    summary = summarize_movie(rows, progress_bins=2, bootstrap_samples=50)
    natural = summary["natural_increment"]["L24"]
    assert natural[0]["estimate"] == pytest.approx(0.0)
    assert natural[1]["estimate"] == pytest.approx(1.0)
    margin = summary["natural_adjacent_margin_k_plus_1_minus_k"]["L24"]
    assert margin[0]["estimate"] == pytest.approx(-1.0)
    assert margin[1]["estimate"] == pytest.approx(1.0)
    retention = summary["normalized_offset_retention"]["all_history_kv"]["L24"]
    assert retention[0]["estimate"] == pytest.approx(1.0)
    assert retention[1]["estimate"] == pytest.approx(0.5)
