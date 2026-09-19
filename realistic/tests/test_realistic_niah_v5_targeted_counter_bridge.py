from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYZER_PATH = (
    ROOT / "scripts" / "analyze_realistic_niah_v5_targeted_counter_bridge.py"
)
SPEC = importlib.util.spec_from_file_location("targeted_counter_analyzer", ANALYZER_PATH)
assert SPEC is not None and SPEC.loader is not None
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def _csv(values: list[float]) -> str:
    return ",".join(f"{value:.9g}" for value in values)


def _probability_profile(gold_probability: float, competitor_probability: float) -> list[float]:
    remainder = 1.0 - gold_probability - competitor_probability
    values = [remainder / 8.0] * 10
    values[7] = gold_probability
    values[6] = competitor_probability
    return values


def _row(
    *,
    seed: int,
    condition: str,
    repeat: int,
    utility: float,
    gold_probability: float,
    competitor_probability: float,
    nonmarker_accuracy: float,
    suffix_exact: bool,
    state_distance: float,
) -> dict[str, object]:
    if condition.startswith("selected_generated"):
        receiver = "selected_bank"
    elif condition.startswith("layer_matched_random"):
        receiver = "layer_matched_random"
    else:
        receiver = "clean"
    probabilities = _probability_profile(gold_probability, competitor_probability)
    scores = [math.log(value) for value in probabilities]
    return {
        "model_label": "Qwen3-8B",
        "seed": seed,
        "gold_count": 8,
        "request_id": f"request-{seed}",
        "condition": condition,
        "receiver_generation_condition": receiver,
        "receiver_generation_repeat": repeat,
        "selection_rank_used": False,
        "outcome_blind": True,
        "teacher_forced_terminal_suffix": False,
        "post_terminal_suffix_teacher_forced": True,
        "matched_position_control_enabled": True,
        "matched_position_control_equal_token_budget": True,
        "state_patch_excludes_answer_query": True,
        "state_patch_geometry": "grammar_counter_carrier",
        "row_plan_sha256": "plan",
        "grammar_timing_stratum": "rank_after_city",
        "counter_carrier_component": "marker_core",
        "reference_suffix_exact": suffix_exact,
        "reference_terminal_nonmarker_token_accuracy": nonmarker_accuracy,
        "candidate_probabilities": _csv(probabilities),
        "candidate_log_scores": _csv(scores),
        "expected_count": 8.0 + utility,
        "expected_count_utility": utility,
        "correct_count_probability": gold_probability,
        "correct_count_margin": scores[7] - max(scores[:7] + scores[8:]),
        "exact_count": gold_probability > competitor_probability,
        "strict_count_utility": utility,
        "receiver_carrier_distance_to_clean_by_layer": {
            "19": state_distance,
            "20": state_distance,
        },
    }


def test_targeted_counter_analysis_detects_distribution_and_position_mediation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "discovery"
    shards = root / "shards"
    shards.mkdir(parents=True)
    (root / "frozen_row_plan.json").write_text(
        json.dumps({"plan_sha256": "plan"}), encoding="utf-8"
    )
    for seed in range(1234, 1254):
        rows = [
            _row(
                seed=seed,
                condition="clean_generated_self_state",
                repeat=0,
                utility=0.0,
                gold_probability=0.80,
                competitor_probability=0.10,
                nonmarker_accuracy=1.0,
                suffix_exact=True,
                state_distance=0.0,
            ),
            _row(
                seed=seed,
                condition="selected_generated_self_state",
                repeat=0,
                utility=-1.0,
                gold_probability=0.20,
                competitor_probability=0.65,
                nonmarker_accuracy=0.4,
                suffix_exact=False,
                state_distance=3.0,
            ),
            _row(
                seed=seed,
                condition="selected_generated_clean_state_restore",
                repeat=0,
                utility=-0.1,
                gold_probability=0.72,
                competitor_probability=0.15,
                nonmarker_accuracy=0.4,
                suffix_exact=False,
                state_distance=3.0,
            ),
            _row(
                seed=seed,
                condition="selected_generated_matched_position_state_control",
                repeat=0,
                utility=-0.8,
                gold_probability=0.28,
                competitor_probability=0.58,
                nonmarker_accuracy=0.4,
                suffix_exact=False,
                state_distance=3.0,
            ),
            _row(
                seed=seed,
                condition="clean_generated_selected_state_occlusion",
                repeat=0,
                utility=-0.7,
                gold_probability=0.30,
                competitor_probability=0.55,
                nonmarker_accuracy=1.0,
                suffix_exact=True,
                state_distance=0.0,
            ),
        ]
        for repeat in (1, 2, 3):
            rows.extend(
                [
                    _row(
                        seed=seed,
                        condition="layer_matched_random_generated_self_state",
                        repeat=repeat,
                        utility=-0.2,
                        gold_probability=0.70,
                        competitor_probability=0.18,
                        nonmarker_accuracy=0.9,
                        suffix_exact=True,
                        state_distance=1.0,
                    ),
                    _row(
                        seed=seed,
                        condition="layer_matched_random_generated_clean_state_restore",
                        repeat=repeat,
                        utility=-0.1,
                        gold_probability=0.75,
                        competitor_probability=0.14,
                        nonmarker_accuracy=0.9,
                        suffix_exact=True,
                        state_distance=1.0,
                    ),
                ]
            )
        assert len(rows) == 11
        (shards / f"seed-{seed}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    effects, claims, audit = ANALYZER.analyze(
        root,
        phase="discovery",
        bootstrap_samples=500,
        random_seed=7,
    )
    assert len(effects) == 20
    assert claims["targeted_counter_directional_signal_pass"] is True
    assert claims["targeted_counter_strong_gate_pass"] is True
    assert audit["conditions_per_seed"] == 11
    primary = {row["estimand"]: row for row in claims["primary_estimands"]}
    assert primary["selected_clean_state_distribution_recovery"]["mean_effect"] > 0
    assert primary["distribution_recovery_position_specificity"]["mean_effect"] > 0
    assert primary["targeted_carrier_state_deformation_specificity"]["mean_effect"] > 0
