#!/usr/bin/env python3
"""Analyze targeted retrieval -> grammar-specific counter state -> answer."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.causal import (  # noqa: E402
    bootstrap_seed_mean_ci,
    sign_flip_pvalue,
)


OUTCOMES = (
    "expected_count_utility",
    "correct_count_probability",
    "correct_count_margin",
    "exact_count",
    "strict_count_utility",
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _summary(
    values: Sequence[float],
    *,
    estimand: str,
    outcome: str,
    bootstrap_samples: int,
    random_seed: int,
    primary: bool,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"Targeted-counter estimand {estimand} is not finite")
    result = bootstrap_seed_mean_ci(
        array, samples=int(bootstrap_samples), seed=int(random_seed)
    )
    result.update(
        {
            "estimand": estimand,
            "outcome": outcome,
            "n_seeds": len(array),
            "p_value": sign_flip_pvalue(array),
            "positive_mean": bool(result["mean_effect"] > 0.0),
            "positive_95pct_ci": bool(result["ci_low"] > 0.0),
            "primary": bool(primary),
        }
    )
    return result


def _vector(value: Any, *, name: str) -> np.ndarray:
    values = np.asarray([float(item) for item in str(value).split(",")], dtype=float)
    if values.shape != (10,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain ten finite count-candidate values")
    return values


def _tv(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.abs(left - right).sum())


def _centered_rmse(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - float(left.mean())
    right_centered = right - float(right.mean())
    return float(np.sqrt(np.mean((left_centered - right_centered) ** 2)))


def _mean_layer_distance(value: Any) -> float:
    if not isinstance(value, dict) or not value:
        raise ValueError("Carrier state-distance audit is missing")
    values = np.asarray([float(item) for item in value.values()], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Carrier state-distance audit is non-finite")
    return float(values.mean())


def analyze(
    root: Path,
    *,
    phase: str,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    plan = json.loads((root / "frozen_row_plan.json").read_text(encoding="utf-8"))
    expected = 20 if phase == "discovery" else 10
    files = sorted((root / "shards").glob("*.jsonl"))
    if len(files) != expected:
        raise ValueError(f"Expected {expected} shards, observed {len(files)}")
    rows = [
        json.loads(line)
        for path in files
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frame = pd.DataFrame(rows)
    if len(frame) != expected * 11:
        raise ValueError("Targeted-counter bridge must have eleven rows per seed")
    if frame["seed"].nunique() != expected:
        raise ValueError("Targeted-counter seed contract changed")
    if frame["selection_rank_used"].map(bool).any():
        raise ValueError("Targeted-counter bridge used selection_rank")
    if not frame["outcome_blind"].map(bool).all():
        raise ValueError("Targeted-counter bridge is not outcome-blind")
    if frame["teacher_forced_terminal_suffix"].map(bool).any():
        raise ValueError("Targeted-counter bridge teacher-forced the terminal item")
    if not frame["post_terminal_suffix_teacher_forced"].map(bool).all():
        raise ValueError("Targeted-counter post-terminal alignment changed")
    if not frame["matched_position_control_enabled"].map(bool).all():
        raise ValueError("Targeted-counter matched-position control is disabled")
    if not frame["matched_position_control_equal_token_budget"].map(bool).all():
        raise ValueError("Targeted-counter state control changed token budget")
    if not frame["state_patch_excludes_answer_query"].map(bool).all():
        raise ValueError("Targeted-counter patch touched the answer query")
    if set(frame["row_plan_sha256"].astype(str)) != {str(plan["plan_sha256"])}:
        raise ValueError("Targeted-counter frozen row-plan hash changed")
    if frame["model_label"].astype(str).nunique() != 1:
        raise ValueError("Targeted-counter analysis mixed models")
    if frame["state_patch_geometry"].astype(str).nunique() != 1:
        raise ValueError("Targeted-counter analysis mixed geometries")
    model_label = str(frame.iloc[0]["model_label"])
    geometry = str(frame.iloc[0]["state_patch_geometry"])

    sample_rows: list[dict[str, Any]] = []
    for sample_key, group in frame.groupby(
        ["model_label", "seed", "gold_count", "request_id"], sort=True
    ):
        def cell(condition: str) -> pd.DataFrame:
            return group.loc[group["condition"].astype(str).eq(condition)]

        clean = cell("clean_generated_self_state")
        selected = cell("selected_generated_self_state")
        selected_restore = cell("selected_generated_clean_state_restore")
        position_control = cell(
            "selected_generated_matched_position_state_control"
        )
        clean_occluded = cell("clean_generated_selected_state_occlusion")
        random_self = cell("layer_matched_random_generated_self_state").sort_values(
            "receiver_generation_repeat"
        )
        random_restore = cell(
            "layer_matched_random_generated_clean_state_restore"
        ).sort_values("receiver_generation_repeat")
        if not (
            len(clean)
            == len(selected)
            == len(selected_restore)
            == len(position_control)
            == len(clean_occluded)
            == 1
            and len(random_self) == len(random_restore) == 3
        ):
            raise ValueError("Targeted-counter factorial cells changed")
        if random_self["receiver_generation_repeat"].astype(int).tolist() != [1, 2, 3]:
            raise ValueError("Targeted-counter random receiver repeats changed")

        clean_row = clean.iloc[0]
        selected_row = selected.iloc[0]
        restore_row = selected_restore.iloc[0]
        control_row = position_control.iloc[0]
        occluded_row = clean_occluded.iloc[0]

        clean_prob = _vector(clean_row["candidate_probabilities"], name="clean probabilities")
        selected_prob = _vector(
            selected_row["candidate_probabilities"], name="selected probabilities"
        )
        restore_prob = _vector(
            restore_row["candidate_probabilities"], name="restored probabilities"
        )
        control_prob = _vector(
            control_row["candidate_probabilities"], name="control probabilities"
        )
        random_prob = [
            _vector(row.candidate_probabilities, name="random probabilities")
            for row in random_self.itertuples(index=False)
        ]
        random_restore_prob = [
            _vector(row.candidate_probabilities, name="random restore probabilities")
            for row in random_restore.itertuples(index=False)
        ]

        clean_score = _vector(clean_row["candidate_log_scores"], name="clean scores")
        selected_score = _vector(
            selected_row["candidate_log_scores"], name="selected scores"
        )
        restore_score = _vector(
            restore_row["candidate_log_scores"], name="restored scores"
        )
        control_score = _vector(
            control_row["candidate_log_scores"], name="control scores"
        )

        selected_tv = _tv(selected_prob, clean_prob)
        restore_tv = _tv(restore_prob, clean_prob)
        control_tv = _tv(control_prob, clean_prob)
        random_tv = np.asarray([_tv(value, clean_prob) for value in random_prob])
        random_restore_tv = np.asarray(
            [
                _tv(value, clean_prob)
                for value in random_restore_prob
            ]
        )
        selected_distribution_recovery = selected_tv - restore_tv
        random_distribution_recovery = float(
            np.mean(random_tv - random_restore_tv)
        )

        selected_state_distance = _mean_layer_distance(
            selected_row["receiver_carrier_distance_to_clean_by_layer"]
        )
        random_state_distance = float(
            np.mean(
                [
                    _mean_layer_distance(value)
                    for value in random_self[
                        "receiver_carrier_distance_to_clean_by_layer"
                    ]
                ]
            )
        )
        result: dict[str, Any] = {
            "model_label": sample_key[0],
            "seed": int(sample_key[1]),
            "gold_count": int(sample_key[2]),
            "request_id": sample_key[3],
            "grammar_timing_stratum": str(clean_row.get("grammar_timing_stratum", "none")),
            "counter_carrier_component": str(clean_row["counter_carrier_component"]),
            "clean_reference_suffix_exact": float(
                bool(clean_row["reference_suffix_exact"])
            ),
            "targeted_terminal_nonmarker_damage": float(
                random_self["reference_terminal_nonmarker_token_accuracy"]
                .astype(float)
                .mean()
                - float(selected_row["reference_terminal_nonmarker_token_accuracy"])
            ),
            "targeted_carrier_state_deformation_specificity": float(
                selected_state_distance - random_state_distance
            ),
            "selected_clean_state_distribution_recovery": float(
                selected_distribution_recovery
            ),
            "random_clean_state_distribution_recovery": float(
                random_distribution_recovery
            ),
            "distribution_recovery_bank_specificity": float(
                selected_distribution_recovery - random_distribution_recovery
            ),
            "distribution_recovery_position_specificity": float(
                control_tv - restore_tv
            ),
            "position_control_distribution_recovery": float(
                selected_tv - control_tv
            ),
            "selected_answer_distribution_damage_specificity": float(
                selected_tv - float(random_tv.mean())
            ),
            "selected_clean_state_centered_score_recovery": float(
                _centered_rmse(selected_score, clean_score)
                - _centered_rmse(restore_score, clean_score)
            ),
            "centered_score_recovery_position_specificity": float(
                _centered_rmse(control_score, clean_score)
                - _centered_rmse(restore_score, clean_score)
            ),
            "selected_clean_state_expected_count_recovery": float(
                abs(float(selected_row["expected_count"]) - float(clean_row["expected_count"]))
                - abs(float(restore_row["expected_count"]) - float(clean_row["expected_count"]))
            ),
        }
        for outcome in OUTCOMES:
            clean_value = float(clean_row[outcome])
            selected_value = float(selected_row[outcome])
            restore_value = float(restore_row[outcome])
            control_value = float(control_row[outcome])
            occluded_value = float(occluded_row[outcome])
            random_values = random_self[outcome].astype(float).to_numpy()
            random_restore_values = random_restore[outcome].astype(float).to_numpy()
            result[f"targeted_answer_damage__{outcome}"] = float(
                random_values.mean() - selected_value
            )
            result[f"selected_clean_state_restoration__{outcome}"] = float(
                restore_value - selected_value
            )
            result[f"state_position_specificity__{outcome}"] = float(
                restore_value - control_value
            )
            result[f"restoration_bank_specificity__{outcome}"] = float(
                (restore_value - selected_value)
                - np.mean(random_restore_values - random_values)
            )
            result[f"selected_state_occlusion__{outcome}"] = float(
                clean_value - occluded_value
            )
        sample_rows.append(result)

    effects = pd.DataFrame(sample_rows).sort_values("seed").reset_index(drop=True)
    if len(effects) != expected:
        raise ValueError("Targeted-counter sample aggregation changed")

    primary_columns = (
        ("targeted_terminal_nonmarker_damage", "token_accuracy"),
        (
            "targeted_carrier_state_deformation_specificity",
            "carrier_state_distance",
        ),
        (
            "selected_clean_state_distribution_recovery",
            "candidate_distribution_tv",
        ),
        (
            "distribution_recovery_position_specificity",
            "candidate_distribution_tv",
        ),
    )
    diagnostic_columns = (
        ("distribution_recovery_bank_specificity", "candidate_distribution_tv"),
        ("position_control_distribution_recovery", "candidate_distribution_tv"),
        ("selected_answer_distribution_damage_specificity", "candidate_distribution_tv"),
        ("selected_clean_state_centered_score_recovery", "centered_log_score_rmse"),
        ("centered_score_recovery_position_specificity", "centered_log_score_rmse"),
        ("selected_clean_state_expected_count_recovery", "expected_count"),
    )
    summaries: list[dict[str, Any]] = []
    for index, (column, outcome) in enumerate((*primary_columns, *diagnostic_columns)):
        summaries.append(
            _summary(
                effects[column].to_numpy(float),
                estimand=column,
                outcome=outcome,
                bootstrap_samples=bootstrap_samples,
                random_seed=random_seed + index * 101,
                primary=column in {name for name, _ in primary_columns},
            )
        )
    for outcome_index, outcome in enumerate(OUTCOMES):
        for estimand_index, estimand in enumerate(
            (
                "targeted_answer_damage",
                "selected_clean_state_restoration",
                "state_position_specificity",
                "restoration_bank_specificity",
                "selected_state_occlusion",
            )
        ):
            column = f"{estimand}__{outcome}"
            summaries.append(
                _summary(
                    effects[column].to_numpy(float),
                    estimand=estimand,
                    outcome=outcome,
                    bootstrap_samples=bootstrap_samples,
                    random_seed=(
                        random_seed + 2000 + outcome_index * 1009 + estimand_index * 103
                    ),
                    primary=False,
                )
            )
    primary = [row for row in summaries if row["primary"]]
    clean_adequacy = _summary(
        effects["clean_reference_suffix_exact"].to_numpy(float),
        estimand="clean_reference_suffix_exact",
        outcome="exact_suffix",
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed + 9091,
        primary=False,
    )
    directional_pass = bool(
        clean_adequacy["ci_low"] >= 0.50
        and all(row["positive_mean"] for row in primary)
    )
    strong_pass = bool(
        clean_adequacy["ci_low"] >= 0.50
        and all(row["positive_95pct_ci"] for row in primary)
    )
    grammar_counts = {
        str(key): int(value)
        for key, value in effects["grammar_timing_stratum"].value_counts().items()
    }
    count_strata = [
        {
            "gold_count": int(gold_count),
            "n_seeds": int(len(group)),
            "means": {
                column: float(group[column].mean())
                for column, _ in (*primary_columns, *diagnostic_columns)
            },
        }
        for gold_count, group in effects.groupby("gold_count", sort=True)
    ]
    claims = {
        "schema_version": "realistic_niah_v5_targeted_counter_bridge_analysis_v1",
        "phase": phase,
        "model_label": model_label,
        "state_patch_geometry": geometry,
        "seed_count": expected,
        "selection_rank_used": False,
        "outcome_blind": True,
        "grammar_timing_counts": grammar_counts,
        "clean_replay_adequacy": clean_adequacy,
        "primary_estimands": primary,
        "diagnostic_estimands": [row for row in summaries if not row["primary"]],
        "all_estimands": summaries,
        "count_strata": count_strata,
        "targeted_counter_directional_signal_pass": directional_pass,
        "targeted_counter_strong_gate_pass": strong_pass,
        "allowed_claim_if_confirmation_passes": (
            f"{model_label}'s frozen targeted bank perturbs the grammar-specific "
            "terminal counter carrier, and restoring clean carrier states moves "
            "the full count-candidate answer distribution back toward clean more "
            "than an equal-token matched-position state control."
        ),
        "restriction": (
            "The terminal suffix is free-running under a fixed token budget; the "
            "post-terminal suffix is teacher-forced for answer-site alignment. "
            "Distribution recovery establishes one mediator pathway, not exclusivity."
        ),
    }
    audit = {
        "status": "PASS",
        "phase": phase,
        "model_label": model_label,
        "state_patch_geometry": geometry,
        "seed_count": expected,
        "trial_rows": len(frame),
        "conditions_per_seed": 11,
        "selection_rank_used": False,
        "outcome_blind": True,
        "teacher_forced_terminal_suffix": False,
        "post_terminal_suffix_teacher_forced": True,
        "matched_position_control": True,
        "state_patch_excludes_answer_query": True,
        "targeted_counter_directional_signal_pass": directional_pass,
        "targeted_counter_strong_gate_pass": strong_pass,
    }
    return effects, claims, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--phase", choices=["discovery", "confirmation"], required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260821)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    effects, claims, audit = analyze(
        args.input,
        phase=str(args.phase),
        bootstrap_samples=int(args.bootstrap_samples),
        random_seed=int(args.random_seed),
    )
    _atomic_csv(args.output / "seed_effects.csv", effects)
    _atomic_json(args.output / "claim_gates.json", claims)
    _atomic_json(args.output / "audit.json", audit)
    print(json.dumps(claims, sort_keys=True))


if __name__ == "__main__":
    main()
