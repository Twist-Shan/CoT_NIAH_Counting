from __future__ import annotations

import math

import pandas as pd

from realistic_niah_v4_4_4.analysis import (
    _holm_adjust,
    add_candidate_specificity,
    build_seed_metrics,
    exact_sign_flip_p,
    primary_decision,
    summarize_seed_metrics,
)
from realistic_niah_v4_4_4.spec import V444Config


def test_exact_sign_flip_uses_all_twenty_seed_assignments() -> None:
    values = [1.0] * 20
    assert exact_sign_flip_p(values, alternative="greater") == 1.0 / (2**20)
    assert exact_sign_flip_p(values, alternative="less") == 1.0


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    adjusted = _holm_adjust([0.01, 0.04, 0.03, math.nan])
    assert adjusted[:3] == [0.03, 0.06, 0.06]
    assert math.isnan(adjusted[3])


def _synthetic_details() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    natural_rows = []
    directed_rows = []
    mediation_rows = []
    definitions = [
        ("candidate", "candidate_core", 1.0, 0.50, 0.30, -0.40, 0.80, 0.10, 0.70),
        ("control0", "matched_control", 0.2, 0.10, 0.05, -0.05, 0.20, 0.10, 0.15),
        ("control1", "matched_control", 0.2, 0.10, 0.05, -0.05, 0.20, 0.10, 0.15),
        ("control2", "matched_control", 0.2, 0.10, 0.05, -0.05, 0.20, 0.10, 0.15),
        ("control3", "matched_control", 0.2, 0.10, 0.05, -0.05, 0.20, 0.10, 0.15),
    ]
    for seed in range(20):
        for (
            set_id,
            role,
            natural_slope,
            injection_slope,
            removal_error,
            removal_margin,
            patch,
            block,
            control,
        ) in definitions:
            for count in range(1, 11):
                natural_rows.append(
                    {
                        "seed": seed,
                        "gold_count": count,
                        "set_id": set_id,
                        "set_role": role,
                        "natural_carrier_coefficient": natural_slope * count,
                    }
                )
            for count in (2, 5, 8):
                for beta in (-2.0, -1.0, 0.0, 1.0, 2.0):
                    directed_rows.append(
                        {
                            "seed": seed,
                            "gold_count": count,
                            "set_id": set_id,
                            "set_role": role,
                            "intervention": f"natural_ov_z_injection_beta_{beta:+g}",
                            "beta": beta,
                            "delta_expected_count": injection_slope * beta,
                            "delta_expected_count_absolute_error": 0.0,
                            "delta_correct_margin": 0.0,
                        }
                    )
                for intervention, error, margin in (
                    ("natural_ov_count_axis_removal", removal_error, removal_margin),
                    ("equal_output_norm_set_span_orthogonal_removal", 0.0, 0.0),
                ):
                    directed_rows.append(
                        {
                            "seed": seed,
                            "gold_count": count,
                            "set_id": set_id,
                            "set_role": role,
                            "intervention": intervention,
                            "beta": math.nan,
                            "delta_expected_count": 0.0,
                            "delta_expected_count_absolute_error": error,
                            "delta_correct_margin": margin,
                        }
                    )
            for condition, value in (
                ("donor_z_patch", patch),
                ("donor_z_patch_natural_axis_block", block),
                ("donor_z_patch_orthogonal_control", control),
            ):
                mediation_rows.append(
                    {
                        "seed": seed,
                        "set_id": set_id,
                        "set_role": role,
                        "intervention": condition,
                        "continuous_normalized_transport": value,
                    }
                )
    selection = {
        "candidate": {"set_id": "candidate", "set_role": "candidate_core", "heads": [16, 19]},
        "matched_controls": [
            {"set_id": f"control{index}", "set_role": "matched_control", "heads": [index, index + 1]}
            for index in range(4)
        ],
    }
    return (
        pd.DataFrame(natural_rows),
        pd.DataFrame(directed_rows),
        pd.DataFrame(mediation_rows),
        selection,
    )


def test_primary_conjunction_supports_clean_synthetic_transporter() -> None:
    natural, directed, mediation, selection = _synthetic_details()
    config = V444Config(bootstrap_repetitions=1_000)
    metrics = build_seed_metrics(natural, directed, mediation)
    metrics = add_candidate_specificity(metrics, selection)
    summary = summarize_seed_metrics(metrics, config)
    decision = primary_decision(summary, config)
    assert decision["full_natural_ov_transporter_support"] is True
    assert decision["global_intersection_union_p"] == 1.0 / (2**20)


def test_primary_conjunction_fails_when_mediation_is_absent() -> None:
    natural, directed, mediation, selection = _synthetic_details()
    mediation.loc[
        mediation["set_role"].eq("candidate_core")
        & mediation["intervention"].eq("donor_z_patch_natural_axis_block"),
        "continuous_normalized_transport",
    ] = 0.7
    config = V444Config(bootstrap_repetitions=1_000)
    metrics = add_candidate_specificity(
        build_seed_metrics(natural, directed, mediation), selection
    )
    decision = primary_decision(summarize_seed_metrics(metrics, config), config)
    assert decision["full_natural_ov_transporter_support"] is False
    assert decision["families"]["path_mediation"]["passes_alpha"] is False

