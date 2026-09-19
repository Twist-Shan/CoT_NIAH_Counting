from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from realistic_niah_v4.partitioned_attention import (
    assess_rank1_partitioning,
    depth_bin_masses,
    occurrence_attention_values,
    partition_sample_metrics,
    phenotype_bank_coverage,
)
from realistic_niah_v4.prompts import TokenSpan


def _span(slot: int, start: int, end: int) -> TokenSpan:
    return TokenSpan(
        slot_index=slot,
        start=start,
        end=end,
        active=True,
        kind="needle",
        canonical_length=end - start,
        model_token_length=end - start,
    )


def test_occurrence_attention_values_distinguish_end_and_mean() -> None:
    row = np.asarray([0.0, 1.0, 9.0, 0.0, 2.0, 4.0, 0.0])
    spans = (_span(1, 11, 13), _span(2, 14, 16))
    np.testing.assert_allclose(
        occurrence_attention_values(row, spans, key_start=10, pooling="span_end"),
        [9.0, 4.0],
    )
    np.testing.assert_allclose(
        occurrence_attention_values(row, spans, key_start=10, pooling="span_mean"),
        [5.0, 3.0],
    )


def test_partition_metrics_reject_local_aggregation_for_one_spike() -> None:
    metrics = partition_sample_metrics(
        np.asarray([99.0, 1.0, 1.0, 1.0]),
        np.asarray([0.10, 0.20, 0.60, 0.80]),
        partitions=2,
    )
    assert metrics["winner_occurrence_index"] == 1
    assert metrics["winner_depth_bin"] == 1
    assert metrics["first_occurrence_share"] == pytest.approx(99 / 102)
    assert metrics["effective_number"] < 1.1
    assert metrics["local_needle_count"] == 2
    assert metrics["local_effective_fraction"] < 0.52


def test_depth_bin_masses_preserve_row_sum() -> None:
    row = np.arange(1, 13, dtype=float)
    masses = depth_bin_masses(row, bins=4)
    np.testing.assert_allclose(masses, [6.0, 15.0, 24.0, 33.0])
    assert masses.sum() == pytest.approx(row.sum())


def test_assessment_separates_endpoint_selector_from_span_mean() -> None:
    common = {
        "model_label": "Qwen3-8B",
        "design_variant": "v4.2",
        "head_rank": 1,
        "layer": 29,
        "head": 3,
        "winner_depth_bin_mode_frequency": 0.45,
        "local_needle_count_mean": 2.5,
        "local_effective_number_mean": 1.0,
    }
    summary = pd.DataFrame(
        [
            {
                **common,
                "pooling": "span_end",
                "first_occurrence_share_mean": 0.99,
                "winner_is_first_mean": 1.0,
                "local_effective_fraction_mean": 0.51,
                "effective_number_mean": 1.02,
            },
            {
                **common,
                "pooling": "span_mean",
                "first_occurrence_share_mean": 0.28,
                "winner_is_first_mean": 0.70,
                "local_effective_fraction_mean": 0.78,
                "effective_number_mean": 5.4,
            },
        ]
    )
    depth = pd.DataFrame(
        {
            "design_variant": ["v4.2"] * 4,
            "head_rank": [1] * 4,
            "depth_bin": [0, 1, 2, 3],
            "row_mass_fraction": [0.20, 0.16, 0.14, 0.10],
        }
    )
    result = assess_rank1_partitioning(summary, depth, depth_bins=4)
    item = result["assessments"][0]
    assert item["classification"] == (
        "first_occurrence_endpoint_selector_with_broader_span_mean"
    )
    assert item["first_occurrence_endpoint_selector"] is True
    assert item["near_uniform_endpoint_aggregation_inside_winning_partition"] is False
    assert item["broader_span_mean_distribution"] is True


def test_phenotype_bank_combines_complementary_endpoint_heads() -> None:
    occurrence = pd.DataFrame(
        [
            {
                "stimulus_id": stimulus,
                "model_label": "model",
                "design_variant": "v4.1",
                "pooling": "span_end",
                "head_rank": rank,
                "layer": 0,
                "head": rank,
                "occurrence_index": occurrence_index,
                "normalized_share": value,
                "raw_attention_value": value,
            }
            for stimulus in ("a", "b")
            for rank, values in ((1, (1.0, 0.0)), (2, (0.0, 1.0)))
            for occurrence_index, value in enumerate(values, start=1)
        ]
    )
    phenotypes = pd.DataFrame(
        {
            "model_label": ["model", "model"],
            "design_variant": ["v4.1", "v4.1"],
            "head_rank": [1, 2],
            "layer": [0, 0],
            "head": [1, 2],
            "phenotype": ["complementary", "complementary"],
        }
    )
    result = phenotype_bank_coverage(occurrence, phenotypes).iloc[0]
    assert result["head_count"] == 2
    assert result["equal_head_profile_effective_number"] == pytest.approx(2.0)
    assert result["raw_attention_ensemble_effective_number"] == pytest.approx(2.0)
