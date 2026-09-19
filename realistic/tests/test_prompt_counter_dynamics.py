from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from realistic_niah_v4.prompt_counter_dynamics import (
    ATTENTION_METRICS,
    _all_head_sample_metrics,
    _bootstrap_seed_correlation,
    _linear_slope,
    _residualized_correlation,
)


def test_occurrence_adjustment_recovers_seed_level_association() -> None:
    rows = []
    for seed in range(10):
        seed_effect = float(seed - 4.5)
        for occurrence in range(1, 11):
            rows.append(
                {
                    "seed": seed,
                    "query_occurrence": occurrence,
                    "dispersion": 10.0 * occurrence + seed_effect,
                    "counter_noise": -3.0 * occurrence + 2.0 * seed_effect,
                }
            )
    frame = pd.DataFrame(rows)
    estimate = _residualized_correlation(frame, "dispersion")
    low, high = _bootstrap_seed_correlation(
        frame,
        "dispersion",
        seed=17,
        replicates=500,
    )
    assert estimate == pytest.approx(1.0)
    assert low > 0.99 and high <= 1.0 + 1e-12


def test_linear_slope_uses_the_full_normalized_occurrence_range() -> None:
    x = np.linspace(0.0, 1.0, 10)
    assert _linear_slope(x, 2.5 * x - 1.0) == pytest.approx(2.5)


def test_all_head_control_averages_without_discovery_selection() -> None:
    rows = []
    for head, value in ((0, 1.0), (1, 3.0)):
        row = {
            "model": "Qwen3-8B",
            "design_variant": "v4.1",
            "seed": 1254,
            "split": "confirmation",
            "query_occurrence": 2,
            "layer": 4,
            "head": head,
            "needle_end_total_mass": value,
            "needle_end_effective_number": value,
            "needle_end_relative_coverage": value,
            "needle_end_current_share": value,
            "needle_span_total_mass": value + 1.0,
            "needle_span_effective_number": value + 1.0,
            "needle_span_relative_coverage": value + 1.0,
            "needle_span_current_share": value + 1.0,
        }
        for metric in ATTENTION_METRICS[:3]:
            row[metric] = value
        rows.append(row)
    result = _all_head_sample_metrics(
        pd.DataFrame(rows),
        {("v4.1", 1254): "wrong"},
    )
    assert len(result) == 2
    endpoint = result[result["hidden_pooling"] == "span_end"].iloc[0]
    span = result[result["hidden_pooling"] == "span_mean"].iloc[0]
    assert endpoint["needle_total_mass"] == pytest.approx(2.0)
    assert span["needle_total_mass"] == pytest.approx(3.0)
    assert endpoint["row_effective_fraction"] == pytest.approx(2.0)
