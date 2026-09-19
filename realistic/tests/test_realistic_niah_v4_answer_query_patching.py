from __future__ import annotations

import pandas as pd
import pytest

from realistic_niah_v4.answer_query_patching import (
    DIRECTED_COUNT_PAIRS,
    EXPECTED_LAYERS,
    _matches_answer_query_design,
    _seed_estimate,
    add_answer_query_metrics,
)


def test_registered_answer_query_design_is_exact() -> None:
    design = {
        "family": "generation_residual_patching_v1",
        "model_label": "Qwen3-8B",
        "answer_format": "numeric",
        "behavior_metric": "strict_greedy_complete_numeric_generation",
        "confirmation_variants": ["v4.1", "v4.2", "v4.3", "v4.4"],
        "confirmation_seeds": list(range(1254, 1264)),
        "confirmation_counts": list(range(1, 11)),
        "layers": list(EXPECTED_LAYERS["Qwen3-8B"]),
        "directed_count_pairs": [list(pair) for pair in DIRECTED_COUNT_PAIRS],
        "sites": ["answer_query"],
        "needle_protocols": ["single_layer"],
    }
    assert _matches_answer_query_design("Qwen3-8B", design)
    design["sites"] = ["toggled_needle_end"]
    assert not _matches_answer_query_design("Qwen3-8B", design)


def test_answer_query_metrics_condition_donor_adoption_on_eligibility() -> None:
    detail = pd.DataFrame(
        {
            "baseline_predicted_count": [4, 4],
            "donor_baseline_predicted_count": [6, 4],
            "patched_predicted_count": [6, 5],
            "receiver_count": [5, 6],
            "donor_count": [6, 5],
            "generated_count_shift": [2, 1],
            "patched_format_valid": [True, True],
            "prediction_changed": [True, True],
            "moved_toward_donor_gold": [True, True],
            "follows_donor_gold": [True, True],
            "follows_donor_prediction": [True, False],
        }
    )
    enriched = add_answer_query_metrics(detail)
    first = enriched.iloc[0]
    assert first["canonical_pair"] == "5<->6"
    assert bool(first["donor_prediction_eligible"])
    assert first["donor_prediction_adopted"] == pytest.approx(1.0)
    assert first["donor_prediction_distance_reduction"] == pytest.approx(1.0)
    assert first["donor_prediction_transport_fraction"] == pytest.approx(1.0)
    assert first["direction_aligned_shift"] == pytest.approx(2.0)

    second = enriched.iloc[1]
    assert not bool(second["donor_prediction_eligible"])
    assert pd.isna(second["donor_prediction_adopted"])
    assert pd.isna(second["donor_prediction_distance_reduction"])
    assert pd.isna(second["donor_prediction_transport_fraction"])
    assert second["direction_aligned_shift"] == pytest.approx(-1.0)


def test_answer_query_metrics_preserve_invalid_greedy_outputs() -> None:
    detail = pd.DataFrame(
        {
            "baseline_predicted_count": [4],
            "donor_baseline_predicted_count": [6],
            "patched_predicted_count": [float("nan")],
            "receiver_count": [5],
            "donor_count": [6],
            "generated_count_shift": [float("nan")],
            "patched_format_valid": [False],
            "prediction_changed": [float("nan")],
            "moved_toward_donor_gold": [float("nan")],
            "follows_donor_gold": [float("nan")],
            "follows_donor_prediction": [float("nan")],
        }
    )
    enriched = add_answer_query_metrics(detail).iloc[0]
    assert enriched["patched_valid_numeric"] == pytest.approx(0.0)
    assert pd.isna(enriched["prediction_changed_numeric"])
    assert pd.isna(enriched["follows_donor_prediction_numeric"])
    assert enriched["donor_prediction_adopted"] == pytest.approx(0.0)


def test_seed_estimate_reports_sparse_eligible_clusters() -> None:
    frame = pd.DataFrame(
        {
            "seed": list(range(1254, 1264)),
            "metric": [float("nan"), *([1.0] * 9)],
        }
    )
    with pytest.raises(ValueError, match="expected 10 seed clusters"):
        _seed_estimate(frame, "metric", label="strict", repetitions=100)
    estimate = _seed_estimate(
        frame,
        "metric",
        label="sparse",
        repetitions=100,
        expected_seed_count=None,
    )
    assert estimate["seed_clusters"] == 9
    assert estimate["estimate"] == pytest.approx(1.0)
    assert estimate["ci95_low"] == pytest.approx(1.0)
    assert estimate["ci95_high"] == pytest.approx(1.0)
