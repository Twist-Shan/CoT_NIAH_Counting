from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from realistic_niah_v5.native_loop import (
    choose_shuffled_commit_donor_occurrence,
    full_commit_specificity_condition_states,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = (
    ROOT / "scripts" / "analyze_realistic_niah_v5_full_commit_specificity.py"
)
SPEC = importlib.util.spec_from_file_location(
    "full_commit_specificity_analysis", ANALYSIS_SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)

FREEZE_SCRIPT = (
    ROOT / "scripts" / "freeze_realistic_niah_v5_full_commit_specificity_plan.py"
)
FREEZE_SPEC = importlib.util.spec_from_file_location(
    "full_commit_specificity_freeze", FREEZE_SCRIPT
)
assert FREEZE_SPEC is not None and FREEZE_SPEC.loader is not None
FREEZE = importlib.util.module_from_spec(FREEZE_SPEC)
FREEZE_SPEC.loader.exec_module(FREEZE)


def test_shuffled_commit_prefers_mirrored_equal_distance_control() -> None:
    assert (
        choose_shuffled_commit_donor_occurrence(
            gold_count=10,
            receiver_occurrence=5,
            donor_occurrence=6,
            random_seed=7,
        )
        == 4
    )
    fallback = choose_shuffled_commit_donor_occurrence(
        gold_count=6,
        receiver_occurrence=2,
        donor_occurrence=5,
        random_seed=7,
    )
    assert fallback == 4


def test_full_commit_controls_match_complete_delta_norm() -> None:
    receiver = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    donor = np.asarray([3.0, 1.0, 7.0, 2.0], dtype=np.float32)
    shuffled = np.asarray([0.0, 4.0, 2.0, 6.0], dtype=np.float32)
    states, audit = full_commit_specificity_condition_states(
        receiver,
        donor,
        shuffled_donor_state=shuffled,
        random_seed=11,
    )
    full_delta = donor - receiver
    full_norm = np.linalg.norm(full_delta)
    for index in range(3):
        delta = (
            states[f"full_delta_norm_matched_orthogonal_r{index}"].numpy()
            - receiver
        )
        np.testing.assert_allclose(np.linalg.norm(delta), full_norm, rtol=2e-5)
        np.testing.assert_allclose(np.dot(delta, full_delta), 0.0, atol=2e-5)
    np.testing.assert_allclose(
        states["opposite_full_delta_patch"].numpy(), receiver - full_delta
    )
    np.testing.assert_allclose(
        states["shuffled_natural_donor_patch"].numpy(), shuffled
    )
    assert audit["full_delta_random_replicates"] == 3
    assert audit["full_delta_random_max_abs_cosine"] <= 2e-5


def test_specificity_plan_freezes_wrong_ordinal_natural_donor() -> None:
    frame = pd.DataFrame(
        [
            {
                "panel_kind": "p0_local",
                "pair_sha256": "pair-a",
                "seed": 1234,
                "gold_count": 10,
                "receiver_occurrence": 5,
                "donor_occurrence": 6,
                "donor_offset": 1,
                "selection_rank_used": False,
            },
            {
                "panel_kind": "p0_local",
                "pair_sha256": "pair-b",
                "seed": 1234,
                "gold_count": 10,
                "receiver_occurrence": 5,
                "donor_occurrence": 4,
                "donor_offset": -1,
                "selection_rank_used": False,
            },
        ]
    )
    frozen = FREEZE.freeze_plan(
        frame, donor_offsets=(-1, 1), random_seed=20260821
    )
    assert list(frozen["shuffled_donor_occurrence"]) == [6, 4]
    assert frozen["shuffled_absolute_distance_matched"].all()
    assert frozen["specificity_pair_sha256"].nunique() == 2


def test_full_commit_specificity_analysis_uses_seed_level_primary_gates() -> None:
    rows = []
    conditions = (
        "clean",
        "self_patch",
        "full_donor_patch",
        "full_delta_norm_matched_orthogonal_r0",
        "full_delta_norm_matched_orthogonal_r1",
        "full_delta_norm_matched_orthogonal_r2",
        "opposite_full_delta_patch",
        "shuffled_natural_donor_patch",
    )
    intended = {
        "clean": 0.0,
        "self_patch": 0.0,
        "full_donor_patch": 2.0,
        "full_delta_norm_matched_orthogonal_r0": 0.1,
        "full_delta_norm_matched_orthogonal_r1": 0.0,
        "full_delta_norm_matched_orthogonal_r2": -0.1,
        "opposite_full_delta_patch": -1.0,
        "shuffled_natural_donor_patch": -0.5,
    }
    identity = {
        **{condition: value for condition, value in intended.items()},
        "full_donor_patch": 2.5,
        "shuffled_natural_donor_patch": -2.5,
    }
    for seed in range(1234, 1254):
        for offset in (-1, 1):
            for condition in conditions:
                is_random = condition.startswith(
                    "full_delta_norm_matched_orthogonal_r"
                )
                rows.append(
                    {
                        "model_label": "Synthetic",
                        "seed": seed,
                        "pair_sha256": f"{seed}:{offset}",
                        "donor_offset": offset,
                        "condition": condition,
                        "selection_rank_used": False,
                        "donor_minus_receiver_successor_attention_mass": intended[
                            condition
                        ],
                        "donor_minus_shuffled_successor_attention_mass": identity[
                            condition
                        ],
                        "donor_vs_shuffled_donor_city_log_odds": identity[
                            condition
                        ],
                        "condition_full_donor_delta_norm_ratio": (
                            1.0 if is_random else 0.0
                        ),
                        "condition_full_donor_delta_cosine": 0.0,
                        "condition_is_natural_commit_state": bool(
                            condition == "shuffled_natural_donor_patch"
                        ),
                    }
                )
    estimands, seed_effects, result = ANALYSIS.analyze(
        pd.DataFrame(rows),
        phase="discovery",
        bootstrap_samples=200,
        random_seed=13,
    )
    assert len(estimands) == 5
    assert not seed_effects.empty
    assert result["full_commit_specificity_pass"] is True
    assert result["seed_count"] == 20


def test_confirmation_analysis_accepts_frozen_forward_backfill_seeds() -> None:
    registered = (1254, 1255, 1256, 1258, 1259, 1260, 1261, 1262, 1263, 1264)
    rows = []
    conditions = (
        "self_patch",
        "full_donor_patch",
        "full_delta_norm_matched_orthogonal_r0",
        "full_delta_norm_matched_orthogonal_r1",
        "full_delta_norm_matched_orthogonal_r2",
        "opposite_full_delta_patch",
        "shuffled_natural_donor_patch",
    )
    for seed in registered:
        for offset in (-1, 1):
            for condition in conditions:
                is_full = condition == "full_donor_patch"
                is_shuffled = condition == "shuffled_natural_donor_patch"
                is_random = condition.startswith(
                    "full_delta_norm_matched_orthogonal_r"
                )
                rows.append(
                    {
                        "model_label": "Synthetic",
                        "seed": seed,
                        "pair_sha256": f"{seed}:{offset}",
                        "donor_offset": offset,
                        "condition": condition,
                        "selection_rank_used": False,
                        "donor_minus_receiver_successor_attention_mass": (
                            2.0 if is_full else 0.0
                        ),
                        "donor_minus_shuffled_successor_attention_mass": (
                            2.0 if is_full else (-2.0 if is_shuffled else 0.0)
                        ),
                        "donor_vs_shuffled_donor_city_log_odds": (
                            2.0 if is_full else (-2.0 if is_shuffled else 0.0)
                        ),
                        "condition_full_donor_delta_norm_ratio": (
                            1.0 if is_random else 0.0
                        ),
                        "condition_full_donor_delta_cosine": 0.0,
                        "condition_is_natural_commit_state": is_shuffled,
                    }
                )
    _, _, result = ANALYSIS.analyze(
        pd.DataFrame(rows),
        phase="confirmation",
        bootstrap_samples=100,
        random_seed=17,
        registered_seeds=registered,
    )
    assert result["registered_seeds"] == list(registered)
    assert (
        result["seed_contract_source"]
        == "frozen_structural_gate_with_forward_backfill"
    )
