from __future__ import annotations

import pytest

from scripts.run_realistic_niah_v5_event_ledger_behavior import (
    summarize_cache_seed,
    summarize_textual_seed,
)
from scripts.run_realistic_niah_v5_event_ledger_factorial import binary_cells


def test_behavior_textual_summary_recovers_half_count_slope() -> None:
    rows = []
    for bits in binary_cells(3):
        hamming = sum(bits)
        rows.append(
            {
                "seed": 3,
                "subset_id": "".join(map(str, bits)),
                "marker_bits": list(bits),
                "candidate_expected_count": 6 + 0.5 * hamming,
                "candidate_exact": hamming % 2 == 0,
            }
        )
    summary = summarize_textual_seed(rows)
    assert summary["candidate_expected_count_per_valid_marker"] == pytest.approx(0.5)
    assert summary["clean_endpoint_expected_count_contrast"] == pytest.approx(1.5)
    assert summary["candidate_factorial_main_effects"] == pytest.approx([0.5] * 3)
    assert summary["candidate_exact_accuracy"] == pytest.approx(0.5)


def test_behavior_cache_summary_recovers_entry_progress() -> None:
    effects = (0.1, 0.2, 0.3)
    rows = [
        {
            "seed": 4,
            "family": "marker_V_all_layers",
            "marker_bits": list(bits),
            "behavior_axis_progress": sum(bit * effect for bit, effect in zip(bits, effects)),
            "candidate_exact": True,
        }
        for bits in binary_cells(3)
    ]
    summary = summarize_cache_seed(rows)
    assert summary["singleton_effects"] == pytest.approx(effects)
    assert summary["full_subset_progress"] == pytest.approx(0.6)
    assert summary["additivity_error"] == pytest.approx(0.0)
