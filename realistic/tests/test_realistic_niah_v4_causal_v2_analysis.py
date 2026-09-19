from __future__ import annotations

import pandas as pd
import pytest

from realistic_niah_v4.causal_v2_analysis import (
    paired_control_adjusted_transport,
    summarize_ablation_sweep,
    summarize_layer_k_transport,
)


def test_paired_transport_and_layer_k_summary_use_matched_controls() -> None:
    rows = []
    for seed in (1254, 1255):
        for condition, value in (
            ("donor_transport", 0.8),
            ("self_patch", 0.0),
            ("same_count_seed", 0.2),
        ):
            rows.append(
                {
                    "model_label": "toy",
                    "site": "answer_query",
                    "patch_protocol": "single_layer",
                    "start_layer": 3,
                    "k": 2,
                    "condition": condition,
                    "seed": seed,
                    "receiver_count": 4,
                    "donor_count": 6,
                    "target_direction": "increase",
                    "strict_normalized_transport": value,
                    "patched_format_valid": True,
                    "transport_numeric_valid": True,
                    "status": "ok",
                }
            )
    paired = paired_control_adjusted_transport(
        pd.DataFrame(rows), family="answer_patching"
    )
    assert paired["control_adjusted_transport"].tolist() == pytest.approx([0.7, 0.7])
    summary = summarize_layer_k_transport(
        paired, family="answer_patching", bootstrap_repetitions=200
    )
    assert summary.iloc[0]["mean_control_adjusted_transport"] == pytest.approx(0.7)
    assert summary.iloc[0]["seeds"] == 2


def test_ablation_summary_is_ranked_minus_random_mean() -> None:
    rows = []
    for stimulus, seed in (("a", 1254), ("b", 1255)):
        rows.append(
            {
                "model_label": "toy",
                "stimulus_id": stimulus,
                "seed": seed,
                "head_bank": "broad_aggregation",
                "top_n": 1,
                "condition": "ranked",
                "accuracy_delta": -1.0,
                "absolute_error_delta": 2.0,
                "prediction_changed": 1.0,
                "patched_format_valid": True,
                "ranked_random_head_overlap": 1,
            }
        )
        for replicate, accuracy in enumerate((-0.2, -0.4, -0.6)):
            rows.append(
                {
                    "model_label": "toy",
                    "stimulus_id": stimulus,
                    "seed": seed,
                    "head_bank": "broad_aggregation",
                    "top_n": 1,
                    "condition": "layer_matched_random",
                    "random_replicate": replicate,
                    "accuracy_delta": accuracy,
                    "absolute_error_delta": 0.5,
                    "prediction_changed": 0.5,
                    "patched_format_valid": True,
                    "ranked_random_head_overlap": 0,
                }
            )
    summary = summarize_ablation_sweep(pd.DataFrame(rows))
    assert summary.iloc[0]["accuracy_effect"] == pytest.approx(-0.6)
    assert summary.iloc[0]["absolute_error_effect"] == pytest.approx(1.5)
    assert summary.iloc[0]["random_overlap_mean"] == pytest.approx(0.0)
