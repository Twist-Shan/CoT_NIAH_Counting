#!/usr/bin/env python3
"""Analyze the fixed-token targeted-query -> grammar counter-state write assay."""

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


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _read(path: Path) -> pd.DataFrame:
    files = [path] if path.is_file() else sorted((path / "shards").glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No counter-write shards under {path}")
    rows: list[dict[str, Any]] = []
    for file in files:
        for line in file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    if set(frame["experiment_id"].astype(str)) != {
        "teacher_forced_targeted_counter_write"
    }:
        raise ValueError("Counter-write experiment id changed")
    if "selection_rank" in frame or frame["selection_rank_used"].astype(bool).any():
        raise ValueError("Counter-write analysis forbids selection_rank")
    return frame


def _summary(values: pd.Series, *, name: str, seed: int) -> dict[str, Any]:
    array = pd.to_numeric(values, errors="raise").astype(float).to_numpy()
    result = bootstrap_seed_mean_ci(array, samples=10_000, seed=int(seed))
    result.update(
        {
            "estimand": name,
            "p_value": sign_flip_pvalue(array),
            "higher_is_supportive": True,
        }
    )
    return result


def analyze(frame: pd.DataFrame, *, phase: str, random_seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    expected = set(
        COUNT_STREAM_DISCOVERY_SEEDS
        if phase == "discovery"
        else COUNT_STREAM_CONFIRMATION_SEEDS
    )
    observed = set(pd.to_numeric(frame["seed"], errors="raise").astype(int))
    if observed != expected:
        raise ValueError(
            f"{phase} counter-write seed contract mismatch: "
            f"expected={sorted(expected)} observed={sorted(observed)}"
        )
    expected_conditions = {
        "clean",
        "selected_mask",
        "random_mask_r1",
        "random_mask_r2",
        "random_mask_r3",
        "selected_mask_clean_carrier_restore",
        "selected_mask_matched_position_state_control",
    }
    if set(frame["condition"].astype(str)) != expected_conditions:
        raise ValueError("Counter-write factorial changed")
    if not frame["teacher_forced_trace_tokens"].astype(bool).all():
        raise ValueError("Counter-write trace tokens were not fixed")
    counts = frame.groupby("seed")["condition"].nunique()
    if not counts.eq(7).all():
        raise ValueError("Counter-write seed lacks a factorial arm")

    carrier = "carrier_state_rms_distance_mean_downstream"
    boundary = "boundary_state_rms_distance_to_clean_final"
    for column in (carrier, boundary):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    wide_carrier = frame.pivot(index="seed", columns="condition", values=carrier)
    wide_boundary = frame.pivot(index="seed", columns="condition", values=boundary)
    random_conditions = ["random_mask_r1", "random_mask_r2", "random_mask_r3"]
    random_carrier = wide_carrier[random_conditions].mean(axis=1)
    random_boundary = wide_boundary[random_conditions].mean(axis=1)
    effects = pd.DataFrame(index=wide_carrier.index)
    effects.index.name = "seed"
    effects["selected_carrier_deformation"] = wide_carrier["selected_mask"]
    effects["selected_carrier_deformation_specificity"] = (
        wide_carrier["selected_mask"] - random_carrier
    )
    effects["selected_boundary_deformation"] = wide_boundary["selected_mask"]
    effects["selected_boundary_deformation_specificity"] = (
        wide_boundary["selected_mask"] - random_boundary
    )
    effects["clean_carrier_restoration"] = (
        wide_boundary["selected_mask"]
        - wide_boundary["selected_mask_clean_carrier_restore"]
    )
    effects["restoration_position_specificity"] = (
        wide_boundary["selected_mask_matched_position_state_control"]
        - wide_boundary["selected_mask_clean_carrier_restore"]
    )
    effects = effects.reset_index()
    estimand_names = [column for column in effects.columns if column != "seed"]
    summaries = [
        _summary(effects[name], name=name, seed=int(random_seed) + index)
        for index, name in enumerate(estimand_names)
    ]
    by_name = {row["estimand"]: row for row in summaries}
    primary_names = (
        "selected_carrier_deformation",
        "clean_carrier_restoration",
        "restoration_position_specificity",
    )
    directional = bool(
        all(float(by_name[name]["mean_effect"]) > 0.0 for name in primary_names)
    )
    strong = bool(all(float(by_name[name]["ci_low"]) > 0.0 for name in primary_names))
    grammar_counts = (
        frame.drop_duplicates("seed")["grammar_timing_stratum"]
        .astype(str)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    result = {
        "schema_version": "realistic_niah_v5_targeted_counter_write_analysis_v1",
        "phase": phase,
        "seed_count": len(expected),
        "registered_seeds": sorted(expected),
        "outcome_blind": True,
        "selection_rank_used": False,
        "teacher_forced_trace_tokens": True,
        "grammar_timing_counts": grammar_counts,
        "primary_estimand_ids": list(primary_names),
        "targeted_counter_write_directional_pass": directional,
        "targeted_counter_write_strong_gate_pass": strong,
        "targeted_specificity_diagnostics": [
            by_name["selected_carrier_deformation_specificity"],
            by_name["selected_boundary_deformation_specificity"],
        ],
        "primary_estimands": [by_name[name] for name in primary_names],
        "all_estimands": summaries,
        "allowed_claim_if_confirmation_passes": (
            "With trace tokens fixed, the frozen targeted retrieval bank changes "
            "the grammar-specific marker/tail carrier, and restoring that clean "
            "carrier normalizes the later commit state more than an equal-token "
            "near-depth state control."
        ),
        "restriction": (
            "Random-bank specificity is reported separately; the primary mediation "
            "claim does not assert that no other attention heads can perturb state."
        ),
    }
    return effects, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--phase", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--random-seed", type=int, default=20260822)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    effects, result = analyze(_read(args.input), phase=args.phase, random_seed=args.random_seed)
    _atomic_csv(args.output / "seed_effects.csv", effects)
    _atomic_json(args.output / "claim_gates.json", result)
    _atomic_json(
        args.output / "audit.json",
        {
            "status": "PASS",
            "seed_count": result["seed_count"],
            "conditions_per_seed": 7,
            "teacher_forced_trace_tokens": True,
            "selection_rank_used": False,
        },
    )
    print(
        json.dumps(
            {
                "directional": result["targeted_counter_write_directional_pass"],
                "strong": result["targeted_counter_write_strong_gate_pass"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
