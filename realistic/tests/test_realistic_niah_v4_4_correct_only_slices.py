from __future__ import annotations

import pandas as pd

from realistic_niah_v4.correct_only_slices import (
    clean_correct_ablation_rows,
    clean_correct_patching_rows,
)


def test_clean_correct_patching_requires_receiver_semantic_donor_and_state_source() -> None:
    frame = pd.DataFrame(
        [
            {
                "row": "treatment_ok",
                "baseline_is_correct": True,
                "baseline_format_valid": True,
                "donor_baseline_outcome": "correct",
                "state_donor_baseline_outcome": "correct",
                "status": "ok",
            },
            {
                "row": "wrong_receiver",
                "baseline_is_correct": False,
                "baseline_format_valid": True,
                "donor_baseline_outcome": "correct",
                "state_donor_baseline_outcome": "correct",
                "status": "ok",
            },
            {
                "row": "wrong_semantic_donor",
                "baseline_is_correct": True,
                "baseline_format_valid": True,
                "donor_baseline_outcome": "wrong",
                "state_donor_baseline_outcome": "correct",
                "status": "ok",
            },
            {
                "row": "wrong_control_source",
                "baseline_is_correct": True,
                "baseline_format_valid": True,
                "donor_baseline_outcome": "correct",
                "state_donor_baseline_outcome": "wrong",
                "status": "ok",
            },
            {
                "row": "invalid_receiver",
                "baseline_is_correct": True,
                "baseline_format_valid": False,
                "donor_baseline_outcome": "correct",
                "state_donor_baseline_outcome": "correct",
                "status": "ok",
            },
            {
                "row": "failed_intervention",
                "baseline_is_correct": True,
                "baseline_format_valid": True,
                "donor_baseline_outcome": "correct",
                "state_donor_baseline_outcome": "correct",
                "status": "failed",
            },
        ]
    )
    result = clean_correct_patching_rows(frame)
    assert result["row"].tolist() == ["treatment_ok"]


def test_clean_correct_ablation_uses_only_preintervention_baseline() -> None:
    frame = pd.DataFrame(
        [
            {
                "row": "eligible_even_if_ablated_wrong",
                "baseline_is_correct": "True",
                "baseline_format_valid": "true",
                "patched_is_correct": False,
            },
            {
                "row": "baseline_wrong",
                "baseline_is_correct": "False",
                "baseline_format_valid": "true",
                "patched_is_correct": True,
            },
            {
                "row": "baseline_invalid",
                "baseline_is_correct": "True",
                "baseline_format_valid": "false",
                "patched_is_correct": True,
            },
        ]
    )
    result = clean_correct_ablation_rows(frame)
    assert result["row"].tolist() == ["eligible_even_if_ablated_wrong"]

