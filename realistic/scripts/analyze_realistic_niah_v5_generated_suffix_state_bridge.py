#!/usr/bin/env python3
"""Analyze a frozen free-running suffix -> state -> answer mediation panel."""

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
        raise ValueError(f"Generated-suffix estimand {estimand} is not finite")
    result = bootstrap_seed_mean_ci(
        array, samples=int(bootstrap_samples), seed=int(random_seed)
    )
    result.update(
        {
            "estimand": estimand,
            "outcome": outcome,
            "n_seeds": len(array),
            "p_value": sign_flip_pvalue(array),
            "positive_95pct_ci": bool(result["ci_low"] > 0.0),
            "primary": bool(primary),
        }
    )
    return result


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
    if len(frame) != expected * 10:
        raise ValueError("Generated-suffix bridge must have ten rows per seed")
    if frame["seed"].nunique() != expected:
        raise ValueError("Generated-suffix seed contract changed")
    if frame["selection_rank_used"].map(bool).any():
        raise ValueError("Generated-suffix bridge used selection_rank")
    for column in ("outcome_blind",):
        if not frame[column].map(bool).all():
            raise ValueError(f"Generated-suffix audit failed for {column}")
    if frame["teacher_forced_terminal_suffix"].map(bool).any():
        raise ValueError("Generated-suffix bridge silently teacher-forced the item")
    if not frame["post_terminal_suffix_teacher_forced"].map(bool).all():
        raise ValueError("Generated-suffix post-terminal control changed")
    if frame["model_label"].astype(str).nunique() != 1:
        raise ValueError("Generated-suffix analysis mixed models")
    model_label = str(frame.iloc[0]["model_label"])
    if frame["state_patch_geometry"].astype(str).nunique() != 1:
        raise ValueError("Generated-suffix analysis mixed state geometries")
    state_patch_geometry = str(frame.iloc[0]["state_patch_geometry"])
    if not frame["state_patch_excludes_answer_query"].map(bool).all():
        raise ValueError("Generated-suffix state patch touched the answer query")
    if set(frame["row_plan_sha256"].astype(str)) != {str(plan["plan_sha256"])}:
        raise ValueError("Generated-suffix frozen row-plan hash changed")
    if not (
        frame["free_running_token_budget"].astype(int).to_numpy()
        == frame["generated_suffix_token_count"].astype(int).to_numpy()
    ).all():
        raise ValueError("Generated-suffix token budget was not preserved")

    sample_rows: list[dict[str, Any]] = []
    for sample_key, group in frame.groupby(
        ["model_label", "seed", "gold_count", "request_id"], sort=True
    ):
        def cell(condition: str, *, repeat: int | None = None) -> pd.DataFrame:
            active = group.loc[group["condition"].astype(str).eq(condition)]
            if repeat is not None:
                active = active.loc[
                    active["receiver_generation_repeat"].astype(int).eq(int(repeat))
                ]
            return active

        clean = cell("clean_generated_self_state")
        selected = cell("selected_generated_self_state")
        selected_restore = cell("selected_generated_clean_state_restore")
        clean_occluded = cell("clean_generated_selected_state_occlusion")
        random_self = cell("layer_matched_random_generated_self_state").sort_values(
            "receiver_generation_repeat"
        )
        random_restore = cell(
            "layer_matched_random_generated_clean_state_restore"
        ).sort_values("receiver_generation_repeat")
        if not (
            len(clean) == len(selected) == len(selected_restore) == len(clean_occluded) == 1
            and len(random_self) == len(random_restore) == 3
        ):
            raise ValueError("Generated-suffix factorial cells changed")
        if random_self["receiver_generation_repeat"].astype(int).tolist() != [1, 2, 3]:
            raise ValueError("Generated-suffix random self repeats changed")
        if random_restore["receiver_generation_repeat"].astype(int).tolist() != [
            1,
            2,
            3,
        ]:
            raise ValueError("Generated-suffix random restore repeats changed")
        result: dict[str, Any] = {
            "model_label": sample_key[0],
            "seed": int(sample_key[1]),
            "gold_count": int(sample_key[2]),
            "request_id": sample_key[3],
            "clean_reference_suffix_exact": float(
                bool(clean.iloc[0]["reference_suffix_exact"])
            ),
            "clean_reference_suffix_token_accuracy": float(
                clean.iloc[0]["reference_suffix_token_accuracy"]
            ),
            "clean_reference_terminal_nonmarker_token_accuracy": float(
                clean.iloc[0]["reference_terminal_nonmarker_token_accuracy"]
            ),
            "targeted_suffix_token_damage": float(
                random_self["reference_suffix_token_accuracy"].astype(float).mean()
                - float(selected.iloc[0]["reference_suffix_token_accuracy"])
            ),
            "targeted_terminal_nonmarker_damage": float(
                random_self["reference_terminal_nonmarker_token_accuracy"]
                .astype(float)
                .mean()
                - float(
                    selected.iloc[0]["reference_terminal_nonmarker_token_accuracy"]
                )
            ),
            "selected_early_eos": float(bool(selected.iloc[0]["early_eos_generated"])),
            "mean_random_early_eos": float(
                random_self["early_eos_generated"].map(bool).astype(float).mean()
            ),
        }
        for outcome in OUTCOMES:
            clean_value = float(clean.iloc[0][outcome])
            selected_value = float(selected.iloc[0][outcome])
            selected_restore_value = float(selected_restore.iloc[0][outcome])
            clean_occluded_value = float(clean_occluded.iloc[0][outcome])
            random_values = random_self[outcome].astype(float).to_numpy()
            random_restore_values = random_restore[outcome].astype(float).to_numpy()
            selected_restoration = selected_restore_value - selected_value
            random_restoration = float(
                np.mean(random_restore_values - random_values)
            )
            result[f"targeted_answer_damage__{outcome}"] = float(
                random_values.mean() - selected_value
            )
            result[f"selected_clean_state_restoration__{outcome}"] = float(
                selected_restoration
            )
            result[f"random_clean_state_restoration__{outcome}"] = float(
                random_restoration
            )
            result[f"restoration_specificity__{outcome}"] = float(
                selected_restoration - random_restoration
            )
            result[f"selected_state_occlusion__{outcome}"] = float(
                clean_value - clean_occluded_value
            )
            result[f"clean_selected_gap__{outcome}"] = float(
                clean_value - selected_value
            )
        sample_rows.append(result)
    effects = pd.DataFrame(sample_rows).sort_values("seed").reset_index(drop=True)
    if len(effects) != expected:
        raise ValueError("Generated-suffix sample aggregation changed")

    summaries: list[dict[str, Any]] = []
    basic_estimands = (
        ("clean_reference_suffix_exact", "exact_suffix", False),
        ("targeted_suffix_token_damage", "token_accuracy", False),
        ("targeted_terminal_nonmarker_damage", "token_accuracy", True),
    )
    for index, (column, outcome, primary) in enumerate(basic_estimands):
        summaries.append(
            _summary(
                effects[column].to_numpy(float),
                estimand=column,
                outcome=outcome,
                bootstrap_samples=bootstrap_samples,
                random_seed=random_seed + index * 101,
                primary=primary,
            )
        )
    outcome_estimands = (
        ("targeted_answer_damage", True),
        ("selected_clean_state_restoration", True),
        ("selected_state_occlusion", True),
        ("restoration_specificity", False),
        ("clean_selected_gap", False),
    )
    for outcome_index, outcome in enumerate(OUTCOMES):
        for estimand_index, (estimand, primary) in enumerate(outcome_estimands):
            column = f"{estimand}__{outcome}"
            summaries.append(
                _summary(
                    effects[column].to_numpy(float),
                    estimand=estimand,
                    outcome=outcome,
                    bootstrap_samples=bootstrap_samples,
                    random_seed=(
                        random_seed + 1000 + outcome_index * 1009 + estimand_index * 103
                    ),
                    primary=bool(primary and outcome == "expected_count_utility"),
                )
            )
    adequacy = next(
        item for item in summaries if item["estimand"] == "clean_reference_suffix_exact"
    )
    primary = [item for item in summaries if item["primary"]]
    primary_ids = {
        (item["estimand"], item["outcome"]) for item in primary
    }
    expected_primary = {
        ("targeted_terminal_nonmarker_damage", "token_accuracy"),
        ("targeted_answer_damage", "expected_count_utility"),
        ("selected_clean_state_restoration", "expected_count_utility"),
        ("selected_state_occlusion", "expected_count_utility"),
    }
    passed = bool(
        primary_ids == expected_primary
        and adequacy["ci_low"] >= 0.50
        and all(item["positive_95pct_ci"] for item in primary)
    )
    count_strata: list[dict[str, Any]] = []
    stratum_columns = [
        "clean_reference_suffix_exact",
        "targeted_terminal_nonmarker_damage",
        "targeted_answer_damage__expected_count_utility",
        "selected_clean_state_restoration__expected_count_utility",
        "random_clean_state_restoration__expected_count_utility",
        "restoration_specificity__expected_count_utility",
        "selected_state_occlusion__expected_count_utility",
        "targeted_answer_damage__correct_count_margin",
        "selected_clean_state_restoration__correct_count_margin",
        "restoration_specificity__correct_count_margin",
    ]
    for gold_count, stratum in effects.groupby("gold_count", sort=True):
        count_strata.append(
            {
                "gold_count": int(gold_count),
                "n_seeds": int(len(stratum)),
                "means": {
                    column: float(stratum[column].astype(float).mean())
                    for column in stratum_columns
                },
            }
        )
    bank_size = int(
        frame.loc[
            frame["receiver_generation_condition"].astype(str).eq("selected_bank"),
            "receiver_heads",
        ]
        .map(len)
        .iloc[0]
    )
    claims = {
        "schema_version": "realistic_niah_v5_generated_suffix_state_analysis_v2",
        "phase": phase,
        "model_label": model_label,
        "selected_bank_size": bank_size,
        "state_patch_geometry": state_patch_geometry,
        "seed_count": expected,
        "selection_rank_used": False,
        "outcome_blind": True,
        "clean_replay_adequacy": {
            **adequacy,
            "pass": bool(adequacy["ci_low"] >= 0.50),
            "rule": "clean fixed-budget suffix exact-replay CI low >= 0.50",
        },
        "primary_estimands": primary,
        "diagnostic_estimands": [item for item in summaries if not item["primary"]],
        "all_estimands": summaries,
        "count_strata": count_strata,
        "generated_suffix_state_bridge_pass": passed,
        "allowed_claim_if_confirmation_passes": (
            f"{model_label} Top-{bank_size} retrieval heads causally affect the "
            "generated terminal item, and a cumulative residual-state clamp over "
            f"the preregistered {state_patch_geometry} geometry mediates the "
            "answer-count effect."
        ),
        "restriction": (
            "The terminal suffix uses a fixed token budget and the post-terminal "
            "grammar suffix remains teacher forced to preserve answer-site alignment."
        ),
    }
    audit = {
        "status": "PASS",
        "phase": phase,
        "seed_count": expected,
        "trial_rows": len(frame),
        "conditions_per_seed": 10,
        "selection_rank_used": False,
        "outcome_blind": True,
        "teacher_forced_terminal_suffix": False,
        "post_terminal_suffix_teacher_forced": True,
        "fixed_token_budget": True,
        "model_label": model_label,
        "state_patch_geometry": state_patch_geometry,
        "state_patch_excludes_answer_query": True,
        "generated_suffix_state_bridge_pass": passed,
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
