#!/usr/bin/env python3
"""Analyze full-state specificity for the native 5.3 commit intervention.

This extension intentionally makes no low-dimensional count-subspace claim.
It asks whether the ordinal-matched natural donor has a donor-specific routing
effect beyond (i) complete-delta norm-matched generic perturbations, (ii) the
antipodal complete delta, and (iii) a wrong-ordinal natural commit state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.causal import bootstrap_seed_mean_ci, sign_flip_pvalue  # noqa: E402
from realistic_niah_v5.count_stream import (  # noqa: E402
    COUNT_STREAM_CONFIRMATION_SEEDS,
    COUNT_STREAM_DISCOVERY_SEEDS,
)


SCHEMA_VERSION = "realistic_niah_v5_full_commit_specificity_analysis_v1"
PAIR_KEYS = ["model_label", "seed", "pair_sha256"]
PRIMARY_OFFSETS = (-1, 1)
SELF = "self_patch"
FULL = "full_donor_patch"
OPPOSITE = "opposite_full_delta_patch"
SHUFFLED = "shuffled_natural_donor_patch"
RANDOM_CONTROLS = tuple(
    f"full_delta_norm_matched_orthogonal_r{index}" for index in range(3)
)
REQUIRED_CONDITIONS = {SELF, FULL, OPPOSITE, SHUFFLED, *RANDOM_CONTROLS}


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


def _read_trials(path: Path) -> pd.DataFrame:
    files = [path] if path.is_file() else sorted((path / "shards").glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No specificity shards under {path}")
    rows: list[dict[str, Any]] = []
    for file in files:
        with file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Full-commit specificity panel is empty")
    return frame


def _seed_summary(
    values: pd.DataFrame,
    *,
    estimand: str,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    seed_effects = (
        values.groupby(["model_label", "seed"], as_index=False)
        .agg(effect=("pair_effect", "mean"), pair_count=("pair_sha256", "nunique"))
    )
    array = seed_effects["effect"].to_numpy(dtype=float)
    summary = bootstrap_seed_mean_ci(
        array, samples=int(bootstrap_samples), seed=int(random_seed)
    )
    summary.update(
        {
            "estimand": estimand,
            "pair_count": int(values["pair_sha256"].nunique()),
            "p_value": sign_flip_pvalue(array),
        }
    )
    seed_effects.insert(0, "estimand", estimand)
    return summary, seed_effects


def _condition_contrast(
    frame: pd.DataFrame,
    *,
    estimand: str,
    outcome: str,
    treatment: str,
    control: str,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    selected = frame.loc[frame["condition"].isin((treatment, control))].copy()
    selected[outcome] = pd.to_numeric(selected[outcome], errors="raise").astype(float)
    if selected.duplicated(PAIR_KEYS + ["condition"]).any():
        raise ValueError(f"{estimand} contains duplicate pair/condition rows")
    wide = selected.pivot(index=PAIR_KEYS, columns="condition", values=outcome)
    wide = wide.dropna(subset=[treatment, control]).copy()
    wide["pair_effect"] = wide[treatment] - wide[control]
    values = wide.reset_index()
    summary, seeds = _seed_summary(
        values,
        estimand=estimand,
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed,
    )
    summary.update(
        {
            "outcome": outcome,
            "treatment": treatment,
            "control": control,
        }
    )
    return summary, seeds


def _full_vs_random_mean(
    frame: pd.DataFrame,
    *,
    estimand: str,
    outcome: str,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    selected = frame.loc[
        frame["condition"].isin((FULL, *RANDOM_CONTROLS))
    ].copy()
    selected[outcome] = pd.to_numeric(selected[outcome], errors="raise").astype(float)
    if selected.duplicated(PAIR_KEYS + ["condition"]).any():
        raise ValueError(f"{estimand} contains duplicate pair/condition rows")
    wide = selected.pivot(index=PAIR_KEYS, columns="condition", values=outcome)
    wide = wide.dropna(subset=[FULL, *RANDOM_CONTROLS]).copy()
    wide["random_mean"] = wide[list(RANDOM_CONTROLS)].mean(axis=1)
    wide["pair_effect"] = wide[FULL] - wide["random_mean"]
    values = wide.reset_index()
    summary, seeds = _seed_summary(
        values,
        estimand=estimand,
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed,
    )
    summary.update(
        {
            "outcome": outcome,
            "treatment": FULL,
            "control": "mean_three_full_delta_norm_matched_orthogonal",
            "random_control_conditions": list(RANDOM_CONTROLS),
        }
    )
    return summary, seeds


def analyze(
    frame: pd.DataFrame,
    *,
    phase: str,
    bootstrap_samples: int,
    random_seed: int,
    registered_seeds: tuple[int, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if "selection_rank" in frame.columns:
        raise ValueError("Specificity trials contain selection_rank")
    if frame.get("selection_rank_used", pd.Series(False, index=frame.index)).map(
        lambda value: str(value).strip().lower() in {"1", "true", "yes"}
    ).any():
        raise ValueError("Specificity trials used selection rank")
    work = frame.loc[
        pd.to_numeric(frame["donor_offset"], errors="raise")
        .astype(int)
        .isin(PRIMARY_OFFSETS)
    ].copy()
    observed_conditions = set(work["condition"].astype(str))
    missing = sorted(REQUIRED_CONDITIONS - observed_conditions)
    if missing:
        raise ValueError(f"Specificity panel lacks conditions: {missing}")
    default_seeds = tuple(
        COUNT_STREAM_DISCOVERY_SEEDS
        if phase == "discovery"
        else COUNT_STREAM_CONFIRMATION_SEEDS
    )
    if registered_seeds is None:
        expected_seed_sequence = default_seeds
        seed_contract_source = "historical_fixed_partition"
    else:
        expected_seed_sequence = tuple(int(value) for value in registered_seeds)
        if len(set(expected_seed_sequence)) != len(expected_seed_sequence):
            raise ValueError("Registered seed override contains duplicates")
        if len(expected_seed_sequence) != len(default_seeds):
            raise ValueError(
                f"{phase} requires {len(default_seeds)} registered seeds, "
                f"got {len(expected_seed_sequence)}"
            )
        if tuple(sorted(expected_seed_sequence)) != expected_seed_sequence:
            raise ValueError("Registered seed override must be increasing")
        seed_contract_source = "frozen_structural_gate_with_forward_backfill"
    expected_seeds = set(expected_seed_sequence)
    observed_seeds = set(pd.to_numeric(work["seed"], errors="raise").astype(int))
    if observed_seeds != expected_seeds:
        raise ValueError(
            f"{phase} seed contract mismatch: expected={sorted(expected_seeds)} "
            f"observed={sorted(observed_seeds)}"
        )

    random_rows = work.loc[work["condition"].isin(RANDOM_CONTROLS)]
    norm_error = (
        pd.to_numeric(
            random_rows["condition_full_donor_delta_norm_ratio"], errors="raise"
        ).astype(float)
        - 1.0
    ).abs()
    cosine = pd.to_numeric(
        random_rows["condition_full_donor_delta_cosine"], errors="raise"
    ).astype(float).abs()
    if float(norm_error.max()) > 2e-5 or float(cosine.max()) > 2e-5:
        raise ValueError("Full-delta random-control audit failed")
    if not work.loc[work["condition"].eq(SHUFFLED), "condition_is_natural_commit_state"].map(bool).all():
        raise ValueError("Shuffled control is not marked as a natural commit")

    specs: list[
        tuple[
            str,
            Callable[..., tuple[dict[str, Any], pd.DataFrame]],
            dict[str, Any],
        ]
    ] = [
        (
            "full_vs_self_intended_successor_routing",
            _condition_contrast,
            {
                "outcome": "donor_minus_receiver_successor_attention_mass",
                "treatment": FULL,
                "control": SELF,
            },
        ),
        (
            "full_vs_full_norm_random_intended_successor_routing",
            _full_vs_random_mean,
            {"outcome": "donor_minus_receiver_successor_attention_mass"},
        ),
        (
            "full_vs_opposite_intended_successor_routing",
            _condition_contrast,
            {
                "outcome": "donor_minus_receiver_successor_attention_mass",
                "treatment": FULL,
                "control": OPPOSITE,
            },
        ),
        (
            "ordinal_donor_identity_double_difference",
            _condition_contrast,
            {
                "outcome": "donor_minus_shuffled_successor_attention_mass",
                "treatment": FULL,
                "control": SHUFFLED,
            },
        ),
        (
            "ordinal_donor_identity_city_log_odds",
            _condition_contrast,
            {
                "outcome": "donor_vs_shuffled_donor_city_log_odds",
                "treatment": FULL,
                "control": SHUFFLED,
            },
        ),
    ]
    summaries: list[dict[str, Any]] = []
    seed_frames: list[pd.DataFrame] = []
    for index, (name, function, kwargs) in enumerate(specs):
        summary, seeds = function(
            work,
            estimand=name,
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed + index,
            **kwargs,
        )
        summaries.append(summary)
        seed_frames.append(seeds)
    diagnostic_summaries: list[dict[str, Any]] = []
    if "shuffled_donor_offset" in work.columns:
        donor_distance = pd.to_numeric(
            work["donor_offset"], errors="raise"
        ).astype(int).abs()
        shuffled_distance = pd.to_numeric(
            work["shuffled_donor_offset"], errors="raise"
        ).astype(int).abs()
        distance_matched = work.loc[donor_distance.eq(shuffled_distance)].copy()
        if not distance_matched.empty:
            summary, seeds = _condition_contrast(
                distance_matched,
                estimand="distance_matched_ordinal_identity_double_difference",
                outcome="donor_minus_shuffled_successor_attention_mass",
                treatment=FULL,
                control=SHUFFLED,
                bootstrap_samples=bootstrap_samples,
                random_seed=random_seed + len(specs),
            )
            summary.update(
                {
                    "diagnostic_only": True,
                    "distance_match_rule": (
                        "abs(shuffled_donor_offset) == abs(donor_offset)"
                    ),
                }
            )
            seeds["diagnostic_only"] = True
            diagnostic_summaries.append(summary)
            seed_frames.append(seeds)
    by_name = {row["estimand"]: row for row in summaries}
    primary_names = (
        "full_vs_self_intended_successor_routing",
        "full_vs_full_norm_random_intended_successor_routing",
        "ordinal_donor_identity_double_difference",
    )
    primary = [by_name[name] for name in primary_names]
    pass_gate = all(float(row["ci_low"]) > 0.0 for row in primary)
    result = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "registered_offsets": list(PRIMARY_OFFSETS),
        "registered_seeds": list(expected_seed_sequence),
        "seed_contract_source": seed_contract_source,
        "seed_count": len(expected_seeds),
        "pair_count": int(work["pair_sha256"].nunique()),
        "outcome_blind": True,
        "selection_rank_used": False,
        "full_commit_specificity_pass": bool(pass_gate),
        "claim_if_passed": (
            "An ordinal-matched natural commit state causally redirects the "
            "next retrieval toward its own successor beyond complete-delta "
            "norm-matched generic perturbations and a wrong-ordinal natural "
            "commit control."
        ),
        "restriction": (
            "This is a distributed full-state successor-control claim, not a "
            "linear count-subspace or explicit arithmetic-addition claim."
        ),
        "control_audit": {
            "random_max_relative_norm_error": float(norm_error.max()),
            "random_max_abs_full_delta_cosine": float(cosine.max()),
            "random_replicates": len(RANDOM_CONTROLS),
        },
        "primary_estimands": primary,
        "estimands": summaries,
        "diagnostic_estimands": diagnostic_summaries,
    }
    return (
        pd.DataFrame([*summaries, *diagnostic_summaries]),
        pd.concat(seed_frames, ignore_index=True),
        result,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--phase", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260824)
    parser.add_argument(
        "--registered-seeds",
        type=int,
        nargs="+",
        help=(
            "Frozen structurally eligible seed list. Use only for the "
            "predeclared forward-backfill rule."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    estimands, seed_effects, result = analyze(
        _read_trials(args.trials),
        phase=args.phase,
        bootstrap_samples=int(args.bootstrap_samples),
        random_seed=int(args.random_seed),
        registered_seeds=(
            tuple(int(value) for value in args.registered_seeds)
            if args.registered_seeds is not None
            else None
        ),
    )
    _atomic_csv(args.output / "estimands.csv", estimands)
    _atomic_csv(args.output / "seed_effects.csv", seed_effects)
    _atomic_json(args.output / "claim_gates.json", result)
    print(
        json.dumps(
            {
                "full_commit_specificity_pass": result[
                    "full_commit_specificity_pass"
                ],
                "seed_count": result["seed_count"],
                "pair_count": result["pair_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
