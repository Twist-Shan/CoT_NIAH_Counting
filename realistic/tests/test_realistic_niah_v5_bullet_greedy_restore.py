from __future__ import annotations

import pandas as pd

import realistic_niah_v5.bullet_greedy_restore as greedy
from scripts.analyze_realistic_niah_v5_bullet_greedy_restore import (
    _confirmation,
    _freeze_top_three,
    _prediction_proportions,
)


def test_greedy_outcome_uses_target_k_not_source_gold(monkeypatch) -> None:
    monkeypatch.setattr(
        greedy,
        "generate_answer_completion_from_prefill",
        lambda *args, **kwargs: {
            "generated_token_ids": [17],
            "generated_token_count": 1,
            "generation_eos_token_ids": [],
            "stopped_on_eos": False,
            "generation_truncated": False,
            "completion_text_raw": " 2",
            "completion_text": " 2",
            "full_answer_text": "Total: 2",
        },
    )
    monkeypatch.setattr(
        greedy,
        "completion_metrics",
        lambda result, *, gold_count: {
            "prediction": 2,
            "exact_count": gold_count == 2,
            "signed_error": 2 - gold_count,
            "absolute_error": abs(2 - gold_count),
        },
    )
    outcomes = greedy._greedy_integer_outcomes(
        object(), object(), object(), object(), target_k=2, max_new_tokens=2
    )
    assert outcomes["greedy_prediction"] == 2
    assert outcomes["greedy_running_exact"] is True
    assert outcomes["greedy_output_is_integer_1_to_10"] is True


def test_greedy_discovery_selects_exact_gain_not_margin() -> None:
    layer = pd.DataFrame(
        [
            {"source_layer": 0, "mean_restoration_exact_gain": 0.2, "restored_exact_accuracy": 0.4},
            {"source_layer": 4, "mean_restoration_exact_gain": 0.3, "restored_exact_accuracy": 0.4},
            {"source_layer": 8, "mean_restoration_exact_gain": 0.3, "restored_exact_accuracy": 0.5},
            {"source_layer": 12, "mean_restoration_exact_gain": 0.1, "restored_exact_accuracy": 0.9},
        ]
    )
    frozen = _freeze_top_three(
        layer,
        plan={"model_label": "Qwen3-8B", "seed_count": 20, "rows": []},
    )
    assert frozen["source_layers"] == [8, 4, 0]


def test_greedy_confirmation_uses_seed_as_unit() -> None:
    rows = []
    for seed in range(10):
        for occurrence in range(1, 11):
            for layer in (4, 8, 12):
                rows.append(
                    {
                        "seed": seed,
                        "target_occurrence": occurrence,
                        "source_layer": layer,
                        "source_exact": 1.0,
                        "blank_exact": 0.0,
                        "restored_exact": 1.0,
                        "restoration_exact_gain": 1.0,
                    }
                )
    per_layer, seed_layer, summary = _confirmation(
        pd.DataFrame(rows), frozen_layers=(4, 8, 12)
    )
    assert len(seed_layer) == 30
    assert summary["layer_results_are_not_averaged"] is True
    assert len(summary["layers"]) == 3
    assert (per_layer["mean_exact_gain"] == 1.0).all()
    assert (per_layer["restored_exact_accuracy"] == 1.0).all()


def test_prediction_proportions_count_out_of_range_as_invalid() -> None:
    derived = pd.DataFrame(
        [
            {
                "seed": seed,
                "target_occurrence": 1,
                "source_layer": 4,
                "source_prediction": source,
                "blank_prediction": blank,
                "restored_prediction": restored,
            }
            for seed, source, blank, restored in (
                (1, 1, 0, 11),
                (2, 2, None, 2),
            )
        ]
    )
    proportions = _prediction_proportions(derived)
    invalid = proportions[
        proportions["label"].eq("invalid_or_outside_1_to_10")
    ].set_index("condition")
    assert invalid.loc["source_reference", "count"] == 0
    assert invalid.loc["blank_reference", "count"] == 2
    assert invalid.loc["source_list_item_k_to_blank_restoration", "count"] == 1
