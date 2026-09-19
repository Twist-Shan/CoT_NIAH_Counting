#!/usr/bin/env python3
"""Analyze terminal item-token -> full-span state -> answer mediation."""

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
from realistic_niah_v5.terminal_token_state import (  # noqa: E402
    REGISTERED_TERMINAL_TOKEN_STATE_CONDITIONS,
)


CONTRASTS = {
    "terminal_token_necessity": {
        "clean": 1.0,
        "clean_terminal_token_ablation": -1.0,
    },
    "terminal_token_sufficiency": {
        "terminal_token_restore": 1.0,
        "uninformative": -1.0,
    },
    "token_written_state_sufficiency": {
        "uninformative_terminal_state_restore": 1.0,
        "uninformative": -1.0,
    },
    "token_effect_requires_terminal_state": {
        "terminal_token_restore": 1.0,
        "terminal_token_restore_state_occluded": -1.0,
    },
    "terminal_marker_token_sufficiency": {
        "terminal_marker_token_restore": 1.0,
        "uninformative": -1.0,
    },
    "terminal_nonmarker_token_sufficiency": {
        "terminal_nonmarker_token_restore": 1.0,
        "uninformative": -1.0,
    },
}
OUTCOMES = (
    "expected_count_utility",
    "correct_count_probability",
    "correct_count_margin",
    "exact_count",
)
PRIMARY = (
    "terminal_token_necessity",
    "terminal_token_sufficiency",
    "token_written_state_sufficiency",
    "token_effect_requires_terminal_state",
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


def analyze(
    root: Path,
    *,
    phase: str,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    plan = json.loads((root / "frozen_row_plan.json").read_text(encoding="utf-8"))
    files = sorted((root / "shards").glob("*.jsonl"))
    expected = 20 if phase == "discovery" else 10
    if len(files) != expected:
        raise ValueError(f"Expected {expected} shards, observed {len(files)}")
    rows = [
        json.loads(line)
        for path in files
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frame = pd.DataFrame(rows)
    if len(frame) != expected * len(REGISTERED_TERMINAL_TOKEN_STATE_CONDITIONS):
        raise ValueError("Terminal token-state trial row count changed")
    if set(frame["condition"].astype(str)) != set(
        REGISTERED_TERMINAL_TOKEN_STATE_CONDITIONS
    ):
        raise ValueError("Terminal token-state conditions changed")
    if frame["selection_rank_used"].map(bool).any():
        raise ValueError("Terminal token-state bridge used selection_rank")
    for column in (
        "outcome_blind",
        "target_is_terminal",
        "all_trace_items_replaced",
        "control_sequence_length_equal",
        "input_token_restoration_uses_outcome",
    ):
        values = frame[column].map(bool)
        expected_value = False if column == "input_token_restoration_uses_outcome" else True
        if not values.eq(expected_value).all():
            raise ValueError(f"Terminal token-state audit failed for {column}")
    if frame["seed"].nunique() != expected:
        raise ValueError("Terminal token-state effective seed count changed")
    if set(frame["row_plan_sha256"].astype(str)) != {str(plan["plan_sha256"])}:
        raise ValueError("Terminal token-state row plan hash changed")

    seed_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    index = ["model_label", "seed", "gold_count", "request_id"]
    for outcome_index, outcome in enumerate(OUTCOMES):
        active = frame.copy()
        active[outcome] = pd.to_numeric(active[outcome], errors="coerce")
        wide = active.pivot(index=index, columns="condition", values=outcome)
        for contrast_index, (estimand, coefficients) in enumerate(CONTRASTS.items()):
            pair = wide.dropna(subset=list(coefficients)).copy()
            effect = pd.Series(0.0, index=pair.index, dtype=float)
            for condition, coefficient in coefficients.items():
                effect += float(coefficient) * pair[condition].astype(float)
            values = effect.to_numpy(float)
            summary = bootstrap_seed_mean_ci(
                values,
                samples=int(bootstrap_samples),
                seed=int(random_seed) + outcome_index * 1009 + contrast_index * 101,
            )
            summary.update(
                {
                    "estimand": estimand,
                    "outcome": outcome,
                    "coefficients": coefficients,
                    "n_seeds": len(values),
                    "p_value": sign_flip_pvalue(values),
                    "positive_95pct_ci": bool(summary["ci_low"] > 0.0),
                    "primary": outcome == "expected_count_utility"
                    and estimand in PRIMARY,
                }
            )
            summaries.append(summary)
            for key, value in effect.items():
                seed_rows.append(
                    {
                        "model_label": key[0],
                        "seed": int(key[1]),
                        "gold_count": int(key[2]),
                        "request_id": key[3],
                        "estimand": estimand,
                        "outcome": outcome,
                        "effect": float(value),
                    }
                )
    primary = [value for value in summaries if value["primary"]]
    claims = {
        "schema_version": "realistic_niah_v5_terminal_token_state_analysis_v1",
        "phase": phase,
        "model_label": str(frame["model_label"].iloc[0]),
        "seed_count": expected,
        "selection_rank_used": False,
        "outcome_blind": True,
        "primary_outcome": "expected_count_utility",
        "primary_estimands": primary,
        "diagnostic_estimands": [
            value
            for value in summaries
            if value["outcome"] == "expected_count_utility"
            and value["estimand"] not in PRIMARY
        ],
        "all_estimands": summaries,
        "terminal_token_state_mediation_pass": bool(
            len(primary) == len(PRIMARY)
            and all(value["positive_95pct_ci"] for value in primary)
        ),
        "effect_size_interpretation": (
            "Expected-count utility is negative absolute count error; positive "
            "effects improve count accuracy in count units."
        ),
    }
    audit = {
        "status": "PASS",
        "phase": phase,
        "seed_count": expected,
        "trial_rows": len(frame),
        "conditions_per_seed": len(REGISTERED_TERMINAL_TOKEN_STATE_CONDITIONS),
        "selection_rank_used": False,
        "outcome_blind": True,
    }
    return pd.DataFrame(seed_rows), claims, audit


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
