from __future__ import annotations

import pandas as pd

from scripts.analyze_realistic_niah_v6_full_item_multihop import (
    CONDITIONS,
    DIRECTIONS,
    MODES,
    MODELS,
    analyze,
    exact_prefix_depth,
    reparse_trials,
)


def test_exact_prefix_depth_never_skips_or_repairs() -> None:
    assert exact_prefix_depth([7, 8, 9, 10], [7, 8, 9, 10]) == 4
    assert exact_prefix_depth([7, 8, 10], [7, 8, 9, 10]) == 2
    assert exact_prefix_depth([5, 7, 8, 9, 10], [7, 8, 9, 10]) == 0
    assert exact_prefix_depth([], [7, 8, 9, 10]) == 0


def _synthetic_factorial() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mode in MODES:
        for model in MODELS:
            for direction in DIRECTIONS:
                receiver = 5 if direction == "forward_skip" else 7
                receiver_path = list(range(receiver + 1, 11))
                donor_path = [7, 8, 9, 10]
                for seed in range(10):
                    for condition in CONDITIONS:
                        observed = (
                            receiver_path
                            if condition == "receiver_self"
                            else donor_path
                        )
                        rows.append(
                            {
                                "prompt_mode": mode,
                                "model_label": model,
                                "direction": direction,
                                "seed": seed,
                                "request_id": f"{mode}/{model}/{direction}/{seed}",
                                "condition": condition,
                                "gold_count": 10,
                                "receiver_occurrence_j": receiver,
                                "donor_occurrence_k": 6,
                                "donor_successor": 7,
                                "receiver_successor": receiver + 1,
                                "completion_text": "synthetic",
                                "generated_known_city_ordinals_any_surface": observed,
                                "generation_truncated": False,
                                "first_generated_known_city_ordinal": observed[0],
                                "patch_scope": "item_span",
                                "ambiguous_known_city_bullet_lines": [],
                                "generated_token_count": 12,
                                "run_manifest_sha256": "a" * 64,
                                "trials_sha256": "b" * 64,
                            }
                        )
    return pd.DataFrame(rows)


def test_multihop_reparse_retains_full_factorial_and_closes_depth_four() -> None:
    reparsed = reparse_trials(_synthetic_factorial())
    effects, claims = analyze(
        reparsed,
        bootstrap_samples=200,
        random_seed=606831,
    )

    assert len(reparsed) == 240
    assert len(effects) == 80
    assert claims["all_cells_primary_depth4_directional"] is True
    assert claims["all_cells_primary_depth4_strong_gate_pass"] is True
    assert claims["failure_taxonomy"]["patched_depth4_failures"] == 0
    assert all(
        row["depth_4"]["patched_rate"] == 1.0
        and row["depth_4"]["receiver_self_donor_rate"] == 0.0
        for row in claims["cell_summaries"]
    )
