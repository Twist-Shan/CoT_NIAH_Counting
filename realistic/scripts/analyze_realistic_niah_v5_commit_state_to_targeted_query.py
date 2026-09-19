#!/usr/bin/env python3
"""Exploratory full commit-state -> targeted-query analysis.

This analysis is deliberately separate from the original native-loop gate.  The
old gate tested a low-dimensional count-subspace intervention and required a
query-attention effect, a teacher-forced city-logit effect, and a greedy city
adoption effect simultaneously.  Here the direct mechanistic endpoint is the
frozen targeted-bank attention shift caused by a *full* donor commit state.

The discovery contrast is post-hoc with respect to the already sealed native-
loop panel.  Any confirmation opened from it must therefore use the untouched
10-seed confirmation split and be labelled prospective confirmation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

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


DIRECT_OUTCOME = "donor_minus_receiver_successor_attention_mass"
CITY_OUTCOME = "donor_vs_receiver_city_log_odds"
GREEDY_OUTCOME = "donor_city_adoption"


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
        raise FileNotFoundError(f"No native-loop shards under {path}")
    rows: list[dict[str, Any]] = []
    for file in files:
        for line in file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    frame = frame.loc[
        frame["experiment_id"].astype(str).eq(
            "p0_count_state_to_targeted_retrieval"
        )
    ].copy()
    if frame.empty:
        raise ValueError("No P0 commit-to-targeted-query rows")
    if "selection_rank" in frame.columns:
        raise ValueError("Formal commit-state rows contain selection_rank")
    if frame.get("selection_rank_used", pd.Series(False, index=frame.index)).map(
        lambda value: str(value).strip().lower() in {"1", "true", "yes"}
    ).any():
        raise ValueError("Formal commit-state rows used selection rank")
    return frame


def _paired_contrast(
    frame: pd.DataFrame,
    *,
    estimand: str,
    outcome: str,
    treatment: str,
    control: str,
    offsets: tuple[int, ...],
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    selected = frame.loc[
        frame["condition"].isin((treatment, control))
        & pd.to_numeric(frame["donor_offset"], errors="raise")
        .astype(int)
        .isin(offsets)
    ].copy()
    selected[outcome] = pd.to_numeric(
        selected[outcome], errors="coerce"
    ).astype(float)
    selected = selected.loc[selected[outcome].notna()].copy()
    keys = ["model_label", "seed", "pair_sha256"]
    if selected.duplicated(keys + ["condition"]).any():
        raise ValueError(f"{estimand} has duplicate pair/condition rows")
    wide = selected.pivot(index=keys, columns="condition", values=outcome)
    if treatment not in wide or control not in wide:
        raise ValueError(f"{estimand} lacks a registered arm")
    wide = wide.dropna(subset=[treatment, control]).copy()
    wide["pair_effect"] = wide[treatment] - wide[control]
    seed_effects = (
        wide.reset_index()
        .groupby(["model_label", "seed"], as_index=False)
        .agg(effect=("pair_effect", "mean"), pair_count=("pair_sha256", "nunique"))
    )
    values = seed_effects["effect"].to_numpy(dtype=float)
    summary = bootstrap_seed_mean_ci(
        values, samples=int(bootstrap_samples), seed=int(random_seed)
    )
    summary.update(
        {
            "estimand": estimand,
            "outcome": outcome,
            "treatment": treatment,
            "control": control,
            "offsets": list(offsets),
            "pair_count": int(len(wide)),
            "p_value": sign_flip_pvalue(values),
        }
    )
    seed_effects.insert(0, "estimand", estimand)
    return summary, seed_effects


def analyze(
    frame: pd.DataFrame,
    *,
    phase: str,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    expected = set(
        COUNT_STREAM_DISCOVERY_SEEDS
        if phase == "discovery"
        else COUNT_STREAM_CONFIRMATION_SEEDS
    )
    observed = set(pd.to_numeric(frame["seed"], errors="raise").astype(int))
    if observed != expected:
        raise ValueError(
            f"{phase} seed contract mismatch: expected={sorted(expected)} "
            f"observed={sorted(observed)}"
        )
    required = {
        "self_patch",
        "full_donor_patch",
        "count_subspace_transplant",
        "norm_matched_orthogonal_patch",
    }
    observed_conditions = set(frame["condition"].astype(str))
    if not required <= observed_conditions:
        raise ValueError(
            f"Commit-state panel lacks conditions: {sorted(required-observed_conditions)}"
        )

    observed_offsets = {
        int(value)
        for value in pd.to_numeric(frame["donor_offset"], errors="raise")
        .astype(int)
        .unique()
        if int(value) != 0
    }
    distances = sorted({abs(value) for value in observed_offsets})
    if 1 not in distances:
        raise ValueError("Commit-state panel must include donor offsets -1/+1")

    specs: list[tuple[str, str, str, str, tuple[int, ...]]] = []
    for distance in distances:
        offsets = tuple(
            value for value in (-distance, distance) if value in observed_offsets
        )
        outcomes = [
            ("targeted_attention", DIRECT_OUTCOME),
            ("city_log_odds", CITY_OUTCOME),
        ]
        if GREEDY_OUTCOME in frame and frame[GREEDY_OUTCOME].notna().any():
            outcomes.append(("greedy_city_adoption", GREEDY_OUTCOME))
        for outcome_name, outcome in outcomes:
            for control_name, control in (
                ("self", "self_patch"),
                ("orthogonal", "norm_matched_orthogonal_patch"),
                ("count_subspace", "count_subspace_transplant"),
            ):
                specs.append(
                    (
                        f"full_commit_{outcome_name}_vs_{control_name}_distance_{distance}",
                        outcome,
                        "full_donor_patch",
                        control,
                        offsets,
                    )
                )

    summaries: list[dict[str, Any]] = []
    effects: list[pd.DataFrame] = []
    for index, (name, outcome, treatment, control, offsets) in enumerate(specs):
        summary, seed_effect = _paired_contrast(
            frame,
            estimand=name,
            outcome=outcome,
            treatment=treatment,
            control=control,
            offsets=offsets,
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed + index,
        )
        summaries.append(summary)
        effects.append(seed_effect)

    estimands = pd.DataFrame(summaries)
    seed_effects = pd.concat(effects, ignore_index=True)
    by_name = {row["estimand"]: row for row in summaries}
    direct_primary = [
        by_name["full_commit_targeted_attention_vs_self_distance_1"],
        by_name["full_commit_targeted_attention_vs_orthogonal_distance_1"],
    ]
    directional = bool(all(float(row["mean_effect"]) > 0.0 for row in direct_primary))
    strong = bool(all(float(row["ci_low"]) > 0.0 for row in direct_primary))
    result = {
        "schema_version": "realistic_niah_v5_full_commit_to_targeted_query_v1",
        "phase": phase,
        "analysis_status": (
            "POSTHOC_SEALED_DISCOVERY"
            if phase == "discovery"
            else "PROSPECTIVE_CONFIRMATION"
        ),
        "registered_seeds": sorted(expected),
        "seed_count": len(expected),
        "outcome_blind": True,
        "selection_rank_used": False,
        "direct_mechanistic_endpoint": DIRECT_OUTCOME,
        "directional_signal_pass": directional,
        "strong_direct_gate_pass": strong,
        "confirmation_eligible": bool(phase == "discovery" and directional),
        "claim_if_confirmed": (
            "A full commit hidden state causally changes how the frozen targeted "
            "retrieval bank routes attention at the next query."
        ),
        "restriction": (
            "City logits and greedy city adoption are downstream diagnostics, not "
            "required for the direct commit-state-to-query-attention edge."
        ),
        "primary_estimands": direct_primary,
        "estimands": summaries,
    }
    return estimands, seed_effects, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--phase", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260822)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = _read_trials(args.trials)
    estimands, seed_effects, result = analyze(
        frame,
        phase=args.phase,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
    )
    _atomic_csv(args.output / "estimands.csv", estimands)
    _atomic_csv(args.output / "seed_effects.csv", seed_effects)
    _atomic_json(args.output / "claim_gates.json", result)
    print(
        json.dumps(
            {
                "directional_signal_pass": result["directional_signal_pass"],
                "strong_direct_gate_pass": result["strong_direct_gate_pass"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
