from __future__ import annotations

from realistic_niah_v4.causal_audit import _matches_screen_design


def _common(model: str) -> dict[str, object]:
    return {
        "model_label": model,
        "behavior_metric": "strict_greedy_complete_numeric_generation",
        "confirmation_variants": ["v4.1", "v4.2", "v4.3", "v4.4"],
        "confirmation_seeds": list(range(1254, 1264)),
    }


def test_screen_ablation_design_match_is_strict() -> None:
    design = {
        **_common("Qwen3-8B"),
        "family": "generation_head_ablation_v1",
        "confirmation_counts": [7, 8, 9, 10],
        "poolings": ["span_end"],
        "scopes": ["answer_query"],
        "top_ns": [4, 8],
        "random_replicates": 1,
    }
    assert _matches_screen_design("ablation", "Qwen3-8B", design)
    assert not _matches_screen_design(
        "ablation", "Qwen3-8B", {**design, "poolings": ["span_mean"]}
    )


def test_screen_patching_and_steering_layers_are_model_specific() -> None:
    patching = {
        **_common("Gemma4-E4B"),
        "family": "generation_residual_patching_v1",
        "confirmation_counts": list(range(1, 11)),
        "layers": [10, 20, 31],
        "sites": ["toggled_needle_end"],
        "needle_protocols": ["cumulative_from_layer"],
        "directed_count_pairs": [
            [5, 6],
            [6, 5],
            [7, 8],
            [8, 7],
            [9, 10],
            [10, 9],
        ],
    }
    assert _matches_screen_design("patching", "Gemma4-E4B", patching)
    assert not _matches_screen_design(
        "patching", "Gemma4-E4B", {**patching, "layers": [9, 18, 26]}
    )

    steering = {
        **_common("Gemma4-E4B"),
        "family": "geometric_steering_v1",
        "confirmation_counts": list(range(1, 11)),
        "discovery_seeds": list(range(1234, 1254)),
        "layers": [10, 20, 31],
        "methods": ["centroid_delta"],
        "alphas": [1.0],
        "orthogonal_random_replicates": 1,
        "directed_count_pairs": [
            [7, 8],
            [8, 7],
            [9, 10],
            [10, 9],
            [5, 10],
            [10, 5],
        ],
    }
    assert _matches_screen_design("steering", "Gemma4-E4B", steering)
    assert not _matches_screen_design(
        "steering", "Gemma4-E4B", {**steering, "alphas": [0.5]}
    )
