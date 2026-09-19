#!/usr/bin/env python3
"""Analyze the V6 query-through-carrier post-hoc diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.causal import (  # noqa: E402
    bootstrap_seed_mean_ci,
    sign_flip_pvalue,
)


DIAGNOSTIC_LABEL = "POSTHOC_DIAGNOSTIC_SPLIT_REUSE"
CONDITIONS = {
    "clean",
    "selected_mask",
    "random_mask_r1",
    "random_mask_r2",
    "random_mask_r3",
    "selected_mask_clean_carrier_restore",
    "selected_mask_matched_position_state_control",
}


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


def _read(path: Path) -> pd.DataFrame:
    files = [path] if path.is_file() else sorted((path / "shards").glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No counter-write shards under {path}")
    rows = [
        json.loads(line)
        for file in files
        for line in file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame(rows)


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


def analyze(
    frame: pd.DataFrame,
    *,
    phase: str,
    expected_seeds: int,
    expected_scope: str,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "experiment_id",
        "model_label",
        "seed",
        "request_id",
        "condition",
        "teacher_forced_trace_tokens",
        "selection_rank_used",
        "head_ablation_scope",
        "head_ablation_position_count",
        "carrier_state_rms_distance_mean_downstream",
        "boundary_state_rms_distance_to_clean_final",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Counter-write diagnostic lacks fields: {missing}")
    if set(frame["experiment_id"].astype(str)) != {
        "teacher_forced_targeted_counter_write"
    }:
        raise ValueError("Counter-write experiment id changed")
    if "selection_rank" in frame or frame["selection_rank_used"].map(bool).any():
        raise ValueError("Counter-write diagnostic forbids selection_rank")
    if not frame["teacher_forced_trace_tokens"].map(bool).all():
        raise ValueError("Counter-write trace tokens were not fixed")
    if set(frame["head_ablation_scope"].astype(str)) != {expected_scope}:
        raise ValueError("Counter-write head-ablation scope changed")
    if expected_scope == "query_through_carrier" and not frame[
        "head_ablation_position_count"
    ].astype(int).gt(1).all():
        raise ValueError("Decode-aligned carrier lesion did not span multiple positions")
    if set(frame["condition"].astype(str)) != CONDITIONS:
        raise ValueError("Counter-write factorial changed")
    if frame["seed"].astype(int).nunique() != int(expected_seeds):
        raise ValueError("Counter-write true-source seed count changed")
    if frame.groupby("seed")["condition"].nunique().ne(len(CONDITIONS)).any():
        raise ValueError("Counter-write seed lacks a factorial arm")
    if frame.groupby("seed")["request_id"].nunique().ne(1).any():
        raise ValueError("Counter-write diagnostic mixes requests within a seed")

    carrier = "carrier_state_rms_distance_mean_downstream"
    boundary = "boundary_state_rms_distance_to_clean_final"
    for column in (carrier, boundary):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    wide_carrier = frame.pivot(index="seed", columns="condition", values=carrier)
    wide_boundary = frame.pivot(index="seed", columns="condition", values=boundary)
    random_conditions = ["random_mask_r1", "random_mask_r2", "random_mask_r3"]
    effects = pd.DataFrame(index=wide_carrier.index)
    effects.index.name = "seed"
    effects["selected_carrier_deformation"] = wide_carrier["selected_mask"]
    effects["selected_carrier_deformation_specificity"] = (
        wide_carrier["selected_mask"]
        - wide_carrier[random_conditions].mean(axis=1)
    )
    effects["selected_boundary_deformation"] = wide_boundary["selected_mask"]
    effects["selected_boundary_deformation_specificity"] = (
        wide_boundary["selected_mask"]
        - wide_boundary[random_conditions].mean(axis=1)
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
    names = [column for column in effects.columns if column != "seed"]
    summaries = [
        _summary(effects[name], name=name, seed=int(random_seed) + index)
        for index, name in enumerate(names)
    ]
    by_name = {row["estimand"]: row for row in summaries}
    primary_names = (
        "selected_carrier_deformation",
        "clean_carrier_restoration",
        "restoration_position_specificity",
    )
    directional = all(float(by_name[name]["mean_effect"]) > 0 for name in primary_names)
    strong = all(float(by_name[name]["ci_low"]) > 0 for name in primary_names)
    claims = {
        "schema_version": "realistic_niah_v6_targeted_counter_write_diagnostic_v1",
        "analysis_status": DIAGNOSTIC_LABEL,
        "phase": phase,
        "model_label": str(frame["model_label"].iloc[0]),
        "seed_count": int(expected_seeds),
        "head_ablation_scope": expected_scope,
        "original_query_local_null_retained": True,
        "frozen_k_changed": False,
        "selection_rank_used": False,
        "teacher_forced_trace_tokens": True,
        "primary_estimand_ids": list(primary_names),
        "targeted_counter_write_directional_pass": bool(directional),
        "targeted_counter_write_strong_gate_pass": bool(strong),
        "targeted_specificity_diagnostics": [
            by_name["selected_carrier_deformation_specificity"],
            by_name["selected_boundary_deformation_specificity"],
        ],
        "primary_estimands": [by_name[name] for name in primary_names],
        "all_estimands": summaries,
        "qualification": (
            "This support window was specified after inspection of the original "
            "query-local null and reuses the frozen split. It diagnoses support "
            "mismatch and does not replace the original V6 gate."
        ),
    }
    return effects, claims


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--phase", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--expected-seeds", type=int, required=True)
    parser.add_argument(
        "--expected-scope", choices=("query_local", "query_through_carrier"), required=True
    )
    parser.add_argument("--random-seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    effects, claims = analyze(
        _read(args.input),
        phase=str(args.phase),
        expected_seeds=int(args.expected_seeds),
        expected_scope=str(args.expected_scope),
        random_seed=int(args.random_seed),
    )
    _atomic_csv(args.output / "seed_effects.csv", effects)
    _atomic_json(args.output / "claim_gates.json", claims)
    _atomic_json(
        args.output / "audit.json",
        {
            "status": "PASS",
            "analysis_status": DIAGNOSTIC_LABEL,
            "phase": str(args.phase),
            "seed_count": int(args.expected_seeds),
            "conditions_per_seed": len(CONDITIONS),
            "head_ablation_scope": str(args.expected_scope),
            "teacher_forced_trace_tokens": True,
            "selection_rank_used": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "directional": claims["targeted_counter_write_directional_pass"],
                "strong": claims["targeted_counter_write_strong_gate_pass"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
