from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.analyze_realistic_niah_v5_targeted_retrieval import (
    _analyze,
    _analyze_count_strata,
    _anchor_table,
    _registry_relation,
)


def _trial_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in (1254, 1255):
        shared = {
            "model_label": "Qwen3-8B",
            "seed": seed,
            "request_id": f"request-{seed}",
            "from_occurrence": 1,
            "to_occurrence": 2,
            "split": "confirmation",
            "gold_count": 2,
            "routed_target_grammar_class": "adjacent_rank_before_city",
            "target_retrieval_surface_variant": "rank_before_city_compact",
            "query_site_id": "1->2@route-q10",
            "status": "ok",
            "trial_complete": True,
        }
        rows.append(
            {
                **shared,
                "condition": "clean",
                "repeat": 0,
                "correct_next_needle": True,
                "behavior_outcome": "correct_next_needle",
            }
        )
        rows.append(
            {
                **shared,
                "condition": "selected_bank",
                "repeat": 0,
                "correct_next_needle": False,
                "behavior_outcome": "no_identifiable_city_record",
            }
        )
        for repeat in (1, 2, 3):
            rows.append(
                {
                    **shared,
                    "condition": "layer_matched_random",
                    "repeat": repeat,
                    "correct_next_needle": True,
                    "behavior_outcome": "correct_next_needle",
                }
            )
    return rows


def test_five_arm_anchor_pairing_and_seed_equal_effect() -> None:
    trials = pd.DataFrame(_trial_rows())
    anchors, audit = _anchor_table(
        trials,
        bank_size=125,
        clean_reference=None,
        require_complete=True,
    )
    assert audit["complete_five_arm_anchor_units"] == 2
    assert anchors["selected_minus_random_failure"].tolist() == [1.0, 1.0]
    trials["registered_bank_size"] = 125
    estimands, arms, seed_effects, _flow, _modes = _analyze(
        anchors,
        trials,
        bootstrap_samples=100,
    )
    primary = estimands.loc[
        estimands["evaluation_scope"].eq("confirmation")
        & estimands["analysis_population"].eq("all_examples")
        & estimands["grammar_class"].eq("pooled")
    ].iloc[0]
    assert primary["mean"] == pytest.approx(1.0)
    assert primary["n_seeds"] == 2
    assert len(seed_effects) > 0
    raw = arms.loc[
        arms["evaluation_scope"].eq("confirmation")
        & arms["analysis_population"].eq("all_examples")
        & arms["grammar_class"].eq("pooled")
    ].set_index("arm")
    assert raw.loc["selected_bank", "mean"] == pytest.approx(1.0)
    assert raw.loc["layer_matched_random_mean", "mean"] == pytest.approx(0.0)
    macro = estimands.loc[
        estimands["evaluation_scope"].eq("confirmation")
        & estimands["analysis_population"].eq("all_examples")
        & estimands["grammar_class"].eq("macro_primary_grammars")
    ].iloc[0]
    assert macro["mean"] == pytest.approx(1.0)
    count_estimands, count_arms = _analyze_count_strata(
        anchors,
        bootstrap_samples=100,
    )
    assert set(count_estimands["gold_count"]) == {2}
    assert not count_arms.empty


def test_incomplete_random_repeat_is_rejected() -> None:
    rows = _trial_rows()
    rows = [
        row
        for row in rows
        if not (
            row["seed"] == 1255
            and row["condition"] == "layer_matched_random"
            and row["repeat"] == 3
        )
    ]
    with pytest.raises(ValueError, match="incomplete five-arm"):
        _anchor_table(
            pd.DataFrame(rows),
            bank_size=125,
            clean_reference=None,
            require_complete=True,
        )


def test_confirmation_registry_must_be_exact_reference_subset(tmp_path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    rows = [
        {
            "request_id": "a",
            "seed": 1234,
            "gold_count": 2,
            "from_occurrence": 1,
            "to_occurrence": 2,
        },
        {
            "request_id": "b",
            "seed": 1254,
            "gold_count": 2,
            "from_occurrence": 1,
            "to_occurrence": 2,
        },
    ]
    (reference / "selected_anchor_registry.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (candidate / "selected_anchor_registry.jsonl").write_text(
        json.dumps(rows[1], sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert _registry_relation(candidate, reference) == "exact_subset"
    changed = {**rows[1], "gold_count": 3}
    (candidate / "selected_anchor_registry.jsonl").write_text(
        json.dumps(changed, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not an exact row-wise subset"):
        _registry_relation(candidate, reference)
