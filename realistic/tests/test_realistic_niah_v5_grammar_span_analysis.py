from __future__ import annotations

import json

import scripts.analyze_realistic_niah_v5_grammar_span_decomposition as analysis
from realistic_niah_v5.terminal_token_state import (
    REGISTERED_GRAMMAR_SPAN_DECOMPOSITION_CONDITIONS,
)


def test_grammar_span_analysis_consumes_mean_effect_bootstrap_schema(
    tmp_path, monkeypatch
) -> None:
    plan = {"plan_sha256": "frozen-plan"}
    (tmp_path / "frozen_row_plan.json").write_text(json.dumps(plan))
    shards = tmp_path / "shards"
    shards.mkdir()
    for offset, seed in enumerate(range(1234, 1254)):
        timing = "rank_after_city" if offset % 2 == 0 else "rank_before_city"
        rows = []
        for condition in REGISTERED_GRAMMAR_SPAN_DECOMPOSITION_CONDITIONS:
            value = 0.0
            if condition == "clean" or condition.endswith("_restore"):
                value = 1.0
            elif condition.endswith("_matched_random"):
                value = 0.25
            rows.append(
                {
                    "condition": condition,
                    "model_label": "Qwen3-8B",
                    "seed": seed,
                    "gold_count": 8,
                    "request_id": f"request-{seed}",
                    "grammar_timing_stratum": timing,
                    "terminal_grammar_class": (
                        "adjacent_rank_after_city"
                        if timing == "rank_after_city"
                        else "adjacent_rank_before_city"
                    ),
                    "selection_rank_used": False,
                    "outcome_blind": True,
                    "target_is_terminal": True,
                    "all_trace_items_replaced": True,
                    "control_sequence_length_equal": True,
                    "span_selection_uses_outcome": False,
                    "row_plan_sha256": "frozen-plan",
                    "correct_count_margin": value,
                    "expected_count_utility": value,
                    "correct_count_probability": value,
                    "exact_count": value,
                }
            )
        (shards / f"{seed}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )

    def fake_bootstrap(values, *, samples, seed):
        mean = float(sum(values) / len(values))
        return {
            "mean_effect": mean,
            "ci_low": mean - 0.1,
            "ci_high": mean + 0.1,
            "n_seeds": len(values),
        }

    monkeypatch.setattr(analysis, "bootstrap_seed_mean_ci", fake_bootstrap)
    monkeypatch.setattr(analysis, "sign_flip_pvalue", lambda values: 1.0)
    _effects, claims, audit = analysis.analyze(
        tmp_path,
        phase="discovery",
        bootstrap_samples=10,
        random_seed=1,
    )
    assert audit["status"] == "PASS"
    assert claims["largest_split_geometry"] == "marker_core"
    assert claims["largest_split_restoration"]["mean_effect"] == 1.0
    assert claims["largest_split_matched_random_specificity"]["mean_effect"] == 0.75
