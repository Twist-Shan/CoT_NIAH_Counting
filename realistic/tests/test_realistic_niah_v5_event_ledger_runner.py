from __future__ import annotations

import pytest

from scripts.run_realistic_niah_v5_event_ledger_factorial import (
    binary_cells,
    summarize_cache_seed,
    summarize_textual_seed,
)


def test_cache_summary_recovers_three_independent_entries() -> None:
    rows = []
    effects = (0.2, 0.3, 0.4)
    for bits in binary_cells(3):
        rows.append(
            {
                "seed": 9,
                "family": "marker_V_all_layers",
                "landmark": "target_boundary",
                "read_layer": 24,
                "marker_bits": list(bits),
                "donor_axis_progress": sum(bit * effect for bit, effect in zip(bits, effects)),
            }
        )
    summary = summarize_cache_seed(rows)
    assert summary["singleton_effects"] == pytest.approx(effects)
    assert summary["factorial_main_effects"] == pytest.approx(effects)
    assert summary["full_subset_progress"] == pytest.approx(0.9)
    assert summary["additivity_error"] == pytest.approx(0.0)
    assert summary["all_singletons_positive"]


def test_textual_summary_recovers_hamming_slope_and_main_effects() -> None:
    rows = []
    for bits in binary_cells(3):
        hamming = sum(bits)
        rows.append(
            {
                "seed": 10,
                "landmark": "target_boundary",
                "read_layer": 24,
                "marker_bits": list(bits),
                "valid_marker_count": hamming,
                "endpoint_axis_progress": hamming / 3,
                "probe_softmax_expected_count": 6 + 0.8 * hamming,
                "probe_prediction_exact": True,
            }
        )
    summary = summarize_textual_seed(rows)
    assert summary["axis_progress_per_valid_marker"] == pytest.approx(1 / 3)
    assert summary["probe_expected_count_per_valid_marker"] == pytest.approx(0.8)
    assert summary["axis_factorial_main_effects"] == pytest.approx([1 / 3] * 3)
    assert summary["all_axis_main_effects_positive"]
    assert summary["hamming_level_axis_monotone"]
