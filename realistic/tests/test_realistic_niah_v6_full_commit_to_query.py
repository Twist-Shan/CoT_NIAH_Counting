from __future__ import annotations

import pandas as pd

from scripts import analyze_realistic_niah_v5_commit_state_to_targeted_query as legacy
from scripts.analyze_realistic_niah_v6_full_commit_to_query import (
    SCHEMA_VERSION,
    analyze_with_true_source_seeds,
)


def _frame(seeds: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    conditions = (
        "self_patch",
        "full_donor_patch",
        "count_subspace_transplant",
        "norm_matched_orthogonal_patch",
    )
    for seed in seeds:
        for offset in (-1, 1):
            for condition in conditions:
                treatment = condition == "full_donor_patch"
                rows.append(
                    {
                        "experiment_id": "p0_count_state_to_targeted_retrieval",
                        "model_label": "Gemma4-E4B",
                        "seed": seed,
                        "pair_sha256": f"{seed}/{offset}",
                        "condition": condition,
                        "donor_offset": offset,
                        "selection_rank_used": False,
                        "donor_minus_receiver_successor_attention_mass": (
                            2.0 if treatment else 0.0
                        ),
                        "donor_vs_receiver_city_log_odds": 3.0 if treatment else 0.0,
                        "donor_city_adoption": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_v6_full_commit_adapter_uses_replacement_source_seeds_without_leaking_state() -> None:
    replacement_seeds = tuple(range(1364, 1374))
    original = legacy.COUNT_STREAM_CONFIRMATION_SEEDS
    estimands, seed_effects, gates = analyze_with_true_source_seeds(
        _frame(replacement_seeds),
        phase="confirmation",
        true_source_seeds=replacement_seeds,
        bootstrap_samples=500,
        random_seed=20260831,
    )
    assert legacy.COUNT_STREAM_CONFIRMATION_SEEDS == original
    assert set(seed_effects["seed"].astype(int)) == set(replacement_seeds)
    assert not estimands.empty
    assert gates["schema_version"] == SCHEMA_VERSION
    assert gates["strong_direct_gate_pass"] is True
    assert gates["model_trials_recomputed"] is False
    assert gates["frozen_k_changed"] is False


def test_v6_full_commit_adapter_rejects_seed_aliasing() -> None:
    seeds = tuple(range(1364, 1374))
    duplicated = seeds[:-1] + (seeds[-2],)
    try:
        analyze_with_true_source_seeds(
            _frame(seeds),
            phase="confirmation",
            true_source_seeds=duplicated,
            bootstrap_samples=50,
            random_seed=1,
        )
    except ValueError as error:
        assert "unique seeds" in str(error)
    else:
        raise AssertionError("duplicate true-source seed contract was accepted")
