from __future__ import annotations

import pandas as pd

from scripts.analyze_realistic_niah_v6_targeted_city_likelihood import analyze


def _frame(seed_count: int = 10) -> pd.DataFrame:
    rows = []
    arms = (
        ("clean", 0, 0.0),
        ("selected_bank", 0, -2.0),
        ("layer_matched_random", 1, -0.2),
        ("layer_matched_random", 2, -0.1),
        ("layer_matched_random", 3, -0.3),
    )
    for seed in range(2000, 2000 + seed_count):
        for condition, repeat, value in arms:
            rows.append(
                {
                    "model_label": "Gemma4-E4B",
                    "seed": seed,
                    "request_id": f"request/{seed}",
                    "anchor_equivalence_id": f"anchor/{seed}",
                    "condition": condition,
                    "repeat": repeat,
                    "status": "ok",
                    "target_city_log_probability": value,
                    "target_city_logit_margin": value * 2,
                    "target_city_log_odds": value * 3,
                    "target_city_teacher_forced_exact": condition == "clean",
                }
            )
    return pd.DataFrame(rows)


def test_targeted_city_likelihood_requires_selected_damage_and_specificity() -> None:
    effects, claims = analyze(
        _frame(),
        phase="confirmation",
        expected_seeds=10,
        bootstrap_samples=500,
        random_seed=4,
    )
    assert len(effects) == 10 * 3 * 2
    assert claims["directional_specific_signal"] is True
    assert claims["strong_interval_gate_pass"] is True
    primary = {row["estimand"]: row for row in claims["primary_estimands"]}
    assert primary["selected_damage"]["mean_effect"] == 2.0
    assert primary["selected_vs_random_specificity"]["mean_effect"] == 1.8


def test_targeted_city_likelihood_rejects_missing_random_repeat() -> None:
    frame = _frame()
    frame = frame.loc[
        ~(
            frame["condition"].eq("layer_matched_random")
            & frame["repeat"].eq(3)
        )
    ]
    try:
        analyze(
            frame,
            phase="confirmation",
            expected_seeds=10,
            bootstrap_samples=50,
            random_seed=1,
        )
    except ValueError as error:
        assert "factorial changed" in str(error)
    else:
        raise AssertionError("missing random repeat was accepted")
