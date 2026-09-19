from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_realistic_niah_v5_commit_state_to_targeted_query.py"
SPEC = importlib.util.spec_from_file_location("commit_query_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_full_commit_analysis_supports_non_greedy_confirmation_panel() -> None:
    rows = []
    conditions = (
        "self_patch",
        "full_donor_patch",
        "count_subspace_transplant",
        "norm_matched_orthogonal_patch",
    )
    direct = {
        "self_patch": 0.0,
        "full_donor_patch": 1.0,
        "count_subspace_transplant": 0.25,
        "norm_matched_orthogonal_patch": 0.1,
    }
    city = {
        "self_patch": 0.0,
        "full_donor_patch": 0.8,
        "count_subspace_transplant": 0.2,
        "norm_matched_orthogonal_patch": 0.1,
    }
    for seed in range(1234, 1254):
        for offset in (-3, -2, -1, 1, 2, 3):
            for condition in conditions:
                rows.append(
                    {
                        "experiment_id": "p0_count_state_to_targeted_retrieval",
                        "model_label": "Synthetic",
                        "seed": seed,
                        "pair_sha256": f"{seed}:{offset}",
                        "donor_offset": offset,
                        "condition": condition,
                        "selection_rank_used": False,
                        "donor_minus_receiver_successor_attention_mass": direct[condition],
                        "donor_vs_receiver_city_log_odds": city[condition],
                    }
                )
    estimands, seed_effects, result = MODULE.analyze(
        pd.DataFrame(rows),
        phase="discovery",
        bootstrap_samples=200,
        random_seed=7,
    )
    assert not estimands.empty
    assert not seed_effects.empty
    assert result["directional_signal_pass"] is True
    assert result["strong_direct_gate_pass"] is True
    assert all("greedy_city_adoption" not in name for name in estimands["estimand"])
