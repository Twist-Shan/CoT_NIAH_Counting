#!/usr/bin/env python3
"""Analyze the minimal native-thinking count loop with seed-level inference."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

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
from realistic_niah_v5.count_stream import (  # noqa: E402
    COUNT_STREAM_CONFIRMATION_SEEDS,
    COUNT_STREAM_DISCOVERY_SEEDS,
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


def _read_trials(paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        files = [path] if path.is_file() else sorted((path / "shards").glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No native-loop shards under {path}")
        for file in files:
            for line in file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        raise ValueError("Native-loop analysis received no trial rows")
    frame = pd.DataFrame(rows)
    if "selection_rank" in frame.columns:
        raise ValueError("Formal native-loop outcomes contain selection_rank")
    if frame.get("selection_rank_used", pd.Series(False, index=frame.index)).map(
        lambda value: str(value).strip().lower() in {"1", "true", "yes"}
    ).any():
        raise ValueError("Formal native-loop outcomes used selection rank")
    return frame


def _paired_seed_contrast(
    frame: pd.DataFrame,
    *,
    estimand: str,
    outcome: str,
    treatment: str,
    control: str,
    filters: dict[str, Any] | None = None,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    selected = frame.copy()
    for column, value in (filters or {}).items():
        if callable(value):
            selected = selected.loc[value(selected[column])]
        elif isinstance(value, (tuple, list, set)):
            selected = selected.loc[selected[column].isin(value)]
        else:
            selected = selected.loc[selected[column].eq(value)]
    selected = selected.loc[selected["condition"].isin([treatment, control])].copy()
    if selected.empty:
        raise ValueError(f"Estimand {estimand} has no rows")
    selected[outcome] = pd.to_numeric(
        selected[outcome], errors="coerce"
    ).astype(float)
    selected = selected.loc[selected[outcome].notna()]
    keys = ["model_label", "seed", "pair_sha256"]
    duplicate = selected.duplicated(keys + ["condition"])
    if duplicate.any():
        raise ValueError(f"Estimand {estimand} has duplicate pair/condition rows")
    wide = selected.pivot(index=keys, columns="condition", values=outcome)
    if treatment not in wide or control not in wide:
        raise ValueError(f"Estimand {estimand} lacks a registered arm")
    wide = wide.dropna(subset=[treatment, control]).copy()
    wide["pair_effect"] = wide[treatment] - wide[control]
    seed_effects = (
        wide.reset_index()
        .groupby(["model_label", "seed"], as_index=False)
        .agg(
            effect=("pair_effect", "mean"),
            pair_count=("pair_sha256", "nunique"),
        )
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
            "p_value": sign_flip_pvalue(values),
            "pair_count": int(len(wide)),
            "higher_is_supportive": True,
            "gate_pass": bool(summary["ci_low"] > 0.0),
            "filters": json.dumps(filters or {}, sort_keys=True, default=str),
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
    expected_seeds = (
        set(COUNT_STREAM_DISCOVERY_SEEDS)
        if phase == "discovery"
        else set(COUNT_STREAM_CONFIRMATION_SEEDS)
    )
    observed_seeds = set(pd.to_numeric(frame["seed"], errors="raise").astype(int))
    if observed_seeds != expected_seeds:
        raise ValueError(
            f"{phase} native-loop seed contract mismatch: "
            f"expected={sorted(expected_seeds)} observed={sorted(observed_seeds)}"
        )
    if set(frame["mechanism_split"].astype(str)) != {
        "development" if phase == "discovery" else "confirmation"
    }:
        raise ValueError("Native-loop mechanism split disagrees with analysis phase")

    specifications = [
        {
            "estimand": "p0_attention_ordinal_steering",
            "experiment_id": "p0_count_state_to_targeted_retrieval",
            "outcome": "donor_minus_receiver_successor_attention_mass",
            "treatment": "count_subspace_transplant",
            "control": "norm_matched_orthogonal_patch",
            "filters": {"donor_offset": (-1, 1)},
            "claim_group": "commit_to_retrieval",
            "primary": True,
        },
        {
            "estimand": "p0_city_log_odds_steering",
            "experiment_id": "p0_count_state_to_targeted_retrieval",
            "outcome": "donor_vs_receiver_city_log_odds",
            "treatment": "count_subspace_transplant",
            "control": "norm_matched_orthogonal_patch",
            "filters": {"donor_offset": (-1, 1)},
            "claim_group": "commit_to_retrieval",
            "primary": True,
        },
        {
            "estimand": "p0_first_city_steering",
            "experiment_id": "p0_count_state_to_targeted_retrieval",
            "outcome": "donor_city_adoption",
            "treatment": "count_subspace_transplant",
            "control": "norm_matched_orthogonal_patch",
            "filters": {"donor_offset": (-1, 1)},
            "claim_group": "commit_to_retrieval",
            "primary": True,
        },
        {
            "estimand": "natural_backstep_repeat",
            "experiment_id": "p0_count_state_to_targeted_retrieval",
            "outcome": "donor_city_adoption",
            "treatment": "count_subspace_transplant",
            "control": "norm_matched_orthogonal_patch",
            "filters": {"donor_offset": -1},
            "claim_group": "update",
            "primary": True,
        },
        {
            "estimand": "count_component_restoration",
            "experiment_id": "p0_count_state_to_targeted_retrieval",
            "outcome": "receiver_city_retention",
            "treatment": "count_component_restored",
            "control": "count_component_removed",
            "filters": {"donor_offset": -1},
            "claim_group": "update",
            "primary": True,
        },
        {
            "estimand": "terminal_state_causes_stop",
            "experiment_id": "endpoint_state_update_stop_transplant",
            "outcome": "stopped_before_known_city",
            "treatment": "count_subspace_transplant",
            "control": "norm_matched_orthogonal_patch",
            "filters": {"panel_kind": "terminal_injection"},
            "claim_group": "stop",
            "primary": True,
        },
        {
            "estimand": "nonterminal_state_causes_continue",
            "experiment_id": "endpoint_state_update_stop_transplant",
            "outcome": "donor_successor_adoption",
            "treatment": "count_subspace_transplant",
            "control": "norm_matched_orthogonal_patch",
            "filters": {"panel_kind": "nonterminal_injection"},
            "claim_group": "stop",
            "primary": True,
        },
    ]
    p0_offsets = set(
        pd.to_numeric(
            frame.loc[
                frame["experiment_id"]
                .astype(str)
                .eq("p0_count_state_to_targeted_retrieval"),
                "donor_offset",
            ],
            errors="raise",
        ).astype(int)
    )
    for distance in (2, 3):
        signed_offsets = (-distance, distance)
        if not set(signed_offsets).issubset(p0_offsets):
            continue
        for suffix, outcome in (
            ("attention_ordinal_steering", "donor_minus_receiver_successor_attention_mass"),
            ("city_log_odds_steering", "donor_vs_receiver_city_log_odds"),
            ("first_city_steering", "donor_city_adoption"),
        ):
            specifications.append(
                {
                    "estimand": f"p0_{suffix}_distance_{distance}",
                    "experiment_id": "p0_count_state_to_targeted_retrieval",
                    "outcome": outcome,
                    "treatment": "count_subspace_transplant",
                    "control": "norm_matched_orthogonal_patch",
                    "filters": {"donor_offset": signed_offsets},
                    "claim_group": "dose_robustness",
                    "primary": False,
                }
            )
    summaries: list[dict[str, Any]] = []
    effects: list[pd.DataFrame] = []
    for index, spec in enumerate(specifications):
        active = frame.loc[
            frame["experiment_id"].astype(str).eq(spec["experiment_id"])
        ]
        summary, seed_effect = _paired_seed_contrast(
            active,
            estimand=spec["estimand"],
            outcome=spec["outcome"],
            treatment=spec["treatment"],
            control=spec["control"],
            filters=spec["filters"],
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed + index,
        )
        summary["claim_group"] = spec["claim_group"]
        summary["primary"] = bool(spec["primary"])
        summaries.append(summary)
        effects.append(seed_effect)
    estimands = pd.DataFrame(summaries)
    seed_effects = pd.concat(effects, ignore_index=True)
    group_pass = {
        group: bool(values["gate_pass"].all())
        for group, values in estimands.loc[estimands["primary"]].groupby(
            "claim_group"
        )
    }
    gates = {
        "schema_version": "realistic_niah_v5_native_loop_analysis_v1",
        "phase": phase,
        "registered_seeds": sorted(expected_seeds),
        "seed_count": len(expected_seeds),
        "selection_rank_used": False,
        "group_gates": group_pass,
        "commit_to_retrieval_pass": bool(group_pass["commit_to_retrieval"]),
        "update_pass": bool(group_pass["update"]),
        "stop_pass": bool(group_pass["stop"]),
        "native_loop_pass": bool(all(group_pass.values())),
        "dose_robustness_estimands": [
            summary
            for summary in summaries
            if summary["claim_group"] == "dose_robustness"
        ],
        "estimands": summaries,
    }
    return estimands, seed_effects, gates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, nargs="+", required=True)
    parser.add_argument("--phase", choices=["discovery", "confirmation"], required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260821)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = _read_trials(args.trials)
    estimands, seed_effects, gates = analyze(
        frame,
        phase=args.phase,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
    )
    _atomic_csv(args.output / "estimands.csv", estimands)
    _atomic_csv(args.output / "seed_effects.csv", seed_effects)
    _atomic_json(args.output / "claim_gates.json", gates)
    print(json.dumps({"native_loop_pass": gates["native_loop_pass"]}))


if __name__ == "__main__":
    main()
