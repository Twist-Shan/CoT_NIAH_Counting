from __future__ import annotations

from scripts.run_realistic_niah_v5 import build_parser
from realistic_niah_v5.causal import target_city_head_ablation_geometry


def test_target_city_support_audits_query_alias_and_full_prefix() -> None:
    geometry = target_city_head_ablation_geometry(
        prompt_token_count=100,
        query_output_token_index=20,
        target_output_token_start=21,
        target_output_token_end=24,
        scope="registered_query_through_city_prefix",
    )
    assert geometry["registered_query_equals_last_pre_city_predictor"] is True
    assert geometry["registered_query_to_last_pre_city_distance"] == 0
    assert geometry["head_ablation_positions"] == [120, 121, 122]
    assert geometry["score_positions"] == [120, 121, 122]


def test_target_city_support_preserves_interstitial_predictors() -> None:
    geometry = target_city_head_ablation_geometry(
        prompt_token_count=10,
        query_output_token_index=5,
        target_output_token_start=8,
        target_output_token_end=10,
        scope="registered_query_through_city_prefix",
    )
    assert geometry["registered_query_to_last_pre_city_distance"] == 2
    assert geometry["head_ablation_positions"] == [15, 16, 17, 18]
    assert geometry["score_positions"] == [17, 18]


def test_v2_cli_exposes_scoped_city_lesion_and_final_transition_capture() -> None:
    parser = build_parser()
    heads = parser.parse_args(
        [
            "causal-heads",
            "--config",
            "config.json",
            "--model",
            "Qwen3-8B",
            "--cache-dir",
            "cache",
            "--generations",
            "generations.jsonl",
            "--plan",
            "plan.csv",
            "--output",
            "output",
            "--head-ablation-scope",
            "registered_query_through_city_prefix",
        ]
    )
    assert heads.head_ablation_scope == "registered_query_through_city_prefix"
    writes = parser.parse_args(
        [
            "causal-source-writes",
            "--config",
            "config.json",
            "--model",
            "Gemma4-E4B",
            "--cache-dir",
            "cache",
            "--generations",
            "generations.jsonl",
            "--output",
            "output",
            "--counts",
            "10",
            "--final-transition-only",
        ]
    )
    assert writes.counts == [10]
    assert writes.final_transition_only is True
