from __future__ import annotations

from scripts.analyze_realistic_niah_v5_targeted_counter_logit_margin import analyze
from realistic_niah_v5.targeted_counter_logit_margin import _first_divergence


def test_first_divergence_rejects_prefix_relations() -> None:
    assert _first_divergence((1, 2, 4), (1, 2, 3)) == 2
    assert _first_divergence((1, 2), (1, 2, 3)) is None
    assert _first_divergence((1, 2), (1, 2)) is None


def _condition(name: str, margin: float, *, gold: int) -> dict[str, object]:
    clean = name == "clean"
    return {
        "condition": name,
        "receiver_bank_sha256": "clean" if clean else "selected" if name == "selected_mask" else name,
        "receiver_head_count": 0 if clean else 1,
        "head_ablation_layer_applications": {} if clean else {"0": 1},
        "head_ablation_selected_post_zero_max_abs": 0.0,
        "predicted_count_among_candidates": gold,
        "correct_count_margin": margin,
        "local_rank_adjacent_sequence_margin": None,
    }


def _row(seed: int, *, phase: str, gold: int = 6) -> dict[str, object]:
    return {
        "schema_version": "realistic_niah_v5_targeted_counter_logit_margin_capture_v2",
        "experiment_id": "targeted_retrieval_query_to_direct_count_logit_margin",
        "request_id": f"request-{seed}",
        "model_label": "Qwen3-8B",
        "seed": seed,
        "gold_count": gold,
        "timing_branch": "rank_before_city",
        "endpoint_names": ["final_answer_sequence_margin"],
        "answer_query_is_downstream_of_targeted_query": True,
        "candidate_answer_tokens_run_without_head_hooks": True,
        "outcome_blind_panel": True,
        "selection_rank_used": False,
        "no_decoder_fit_or_layer_selection": True,
        "mechanism_split": phase,
        "conditions": [
            _condition("clean", 3.0, gold=gold),
            _condition("selected_mask", 1.0, gold=gold),
            _condition("random_mask_r1", 2.8, gold=gold),
            _condition("random_mask_r2", 2.7, gold=gold),
            _condition("random_mask_r3", 2.9, gold=gold),
        ],
    }


def _plan(seeds: list[int], *, role: str) -> dict[str, object]:
    return {
        "schema_version": "realistic_niah_v5_targeted_counter_logit_margin_plan_v2",
        "model_label": "Qwen3-8B",
        "timing_branch": "rank_before_city",
        "seed_role": role,
        "seeds": seeds,
        "selected_bank_sha256": "selected",
    }


def test_direct_margin_analysis_uses_selected_minus_random_gate() -> None:
    discovery_seeds = [1, 2, 3]
    confirmation_seeds = [4, 5, 6]
    _long, effects, claim = analyze(
        [_row(seed, phase="development") for seed in discovery_seeds],
        _plan(discovery_seeds, role="development"),
        [_row(seed, phase="confirmation") for seed in confirmation_seeds],
        _plan(confirmation_seeds, role="confirmation"),
        timing="rank_before_city",
    )
    primary = claim["primary_endpoint_result"]
    assert primary["readout_validity"]["pass"] is True
    assert primary["selected_mask_changes_margin_directionally"] is True
    assert primary["selected_mask_more_damaging_than_random"] is True
    assert primary["effect_status"] == "INTERVAL_CONFIRMED_DIRECTIONAL_SPECIFIC_SUPPORT"
    assert set(effects["selected_margin_loss"]) == {2.0}
    assert all(effects["selected_vs_random_specificity"] > 1.7)
