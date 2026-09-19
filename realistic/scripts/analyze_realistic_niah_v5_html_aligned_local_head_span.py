#!/usr/bin/env python3
"""Analyze multi-token targeted-head mediation across query-to-city paths."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

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


PRIMARY_OUTCOME = "target_vs_best_other_city_log_odds"
SOURCE = "uninformative_target_restore"
CARRIER = "uninformative"
INTACT = "intact_full_path"
SELECTED = "selected_mask_full_path"
RANDOM = "layer_matched_random_mask_full_path"
RESTORE = "selected_restore_full_path_from_restored_state"
CONTRASTS: dict[str, dict[tuple[str, str], float]] = {
    "full_span_state_sufficiency": {
        (SOURCE, INTACT): 1.0,
        (CARRIER, INTACT): -1.0,
    },
    "selected_full_path_mask_interaction": {
        (SOURCE, INTACT): 1.0,
        (CARRIER, INTACT): -1.0,
        (SOURCE, SELECTED): -1.0,
        (CARRIER, SELECTED): 1.0,
    },
    "random_full_path_mask_interaction": {
        (SOURCE, INTACT): 1.0,
        (CARRIER, INTACT): -1.0,
        (SOURCE, RANDOM): -1.0,
        (CARRIER, RANDOM): 1.0,
    },
    "selected_vs_random_full_path_specificity": {
        (SOURCE, RANDOM): 1.0,
        (CARRIER, RANDOM): -1.0,
        (SOURCE, SELECTED): -1.0,
        (CARRIER, SELECTED): 1.0,
    },
    "selected_full_path_output_restoration": {
        (CARRIER, RESTORE): 1.0,
        (CARRIER, INTACT): -1.0,
    },
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


def _read(root: Path) -> pd.DataFrame:
    files = sorted((root / "shards").glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No HTML local head-span shards under {root}")
    rows = []
    for path in files:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        if len(lines) != 7:
            raise ValueError(f"{path} has {len(lines)} arms instead of 7")
        rows.extend(json.loads(line) for line in lines)
    return pd.DataFrame(rows)


def _audit(frame: pd.DataFrame, plan: Mapping[str, Any], phase: str) -> None:
    expected_seeds = (
        COUNT_STREAM_DISCOVERY_SEEDS
        if phase == "discovery"
        else COUNT_STREAM_CONFIRMATION_SEEDS
    )
    if len(frame) != len(expected_seeds) * 7:
        raise ValueError("HTML local head-span row count changed")
    if tuple(sorted(frame["seed"].astype(int).unique())) != tuple(expected_seeds):
        raise ValueError("HTML local head-span seed contract changed")
    if int(plan.get("planned_arms_per_row", -1)) != 7:
        raise ValueError("HTML local head-span frozen arm count changed")
    if str(plan.get("head_token_geometry")) != "query_plus_full_path":
        raise ValueError("HTML local head-span frozen geometry changed")
    selected = {
        int(seed): int(count)
        for seed, count in plan["selected_count_by_seed"].items()
    }
    observed = {
        int(seed): int(group["gold_count"].astype(int).iloc[0])
        for seed, group in frame.groupby("seed")
    }
    if observed != selected:
        raise ValueError("HTML local head-span selected counts changed")
    if frame.duplicated(["request_id", "state_condition", "head_condition"]).any():
        raise ValueError("HTML local head-span arms are duplicated")
    if set(frame["head_token_geometry"].astype(str)) != {
        "query_plus_full_path"
    }:
        raise ValueError("HTML local head-span runtime geometry changed")
    if set(frame["head_intervention_scope"].astype(str)) != {
        "registered_query_plus_every_teacher_forced_path_input"
    }:
        raise ValueError("HTML local head-span intervention scope changed")
    if not frame["target_is_two_before_terminal"].map(bool).all():
        raise ValueError("HTML local head-span target site changed")
    if not frame["intervening_item_count"].astype(int).eq(1).all():
        raise ValueError("HTML local head-span propagation distance changed")
    if set(frame["patch_geometry"].astype(str)) != {
        "full_item_span_same_position"
    }:
        raise ValueError("HTML local head-span state geometry changed")
    if set(frame["patch_layer_mode"].astype(str)) != {"cumulative_clamp"}:
        raise ValueError("HTML local head-span layer mode changed")
    if frame["selection_rank_used"].map(bool).any():
        raise ValueError("HTML local head-span used selection_rank")
    if not frame["outcome_blind"].map(bool).all():
        raise ValueError("HTML local head-span is not outcome blind")
    if set(frame["row_plan_sha256"].astype(str)) != {str(plan["plan_sha256"])}:
        raise ValueError("HTML local head-span plan hash changed")
    if set(frame["targeted_bank_sha256"].astype(str)) != {
        str(plan["targeted_bank_sha256"])
    }:
        raise ValueError("HTML local head-span targeted bank changed")


def _estimand(
    frame: pd.DataFrame,
    *,
    name: str,
    coefficients: Mapping[tuple[str, str], float],
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    active = frame.copy()
    active[PRIMARY_OUTCOME] = pd.to_numeric(active[PRIMARY_OUTCOME], errors="raise")
    wide = active.pivot(
        index=["model_label", "seed", "gold_count", "request_id"],
        columns=["state_condition", "head_condition"],
        values=PRIMARY_OUTCOME,
    )
    required = list(coefficients)
    missing = [arm for arm in required if arm not in wide.columns]
    if missing:
        raise ValueError(f"{name} is missing arms {missing}")
    wide = wide.dropna(subset=required).copy()
    effect = pd.Series(0.0, index=wide.index, dtype=float)
    for arm, coefficient in coefficients.items():
        effect += float(coefficient) * wide[arm].astype(float)
    per_seed = effect.rename("effect").reset_index()
    values = per_seed["effect"].to_numpy(float)
    summary = bootstrap_seed_mean_ci(
        values, samples=int(bootstrap_samples), seed=int(random_seed)
    )
    summary.update(
        {
            "estimand": name,
            "outcome": PRIMARY_OUTCOME,
            "coefficients": {
                f"{state}|{head}": coefficient
                for (state, head), coefficient in coefficients.items()
            },
            "n_seeds": int(len(per_seed)),
            "pair_count": int(len(per_seed)),
            "p_value": sign_flip_pvalue(values),
            "higher_is_supportive": True,
            "positive_95pct_ci": bool(summary["ci_low"] > 0.0),
        }
    )
    per_seed.insert(0, "estimand", name)
    return summary, per_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--phase", choices=["discovery", "confirmation"], required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260821)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads((args.input / "frozen_row_plan.json").read_text(encoding="utf-8"))
    frame = _read(args.input)
    _audit(frame, plan, args.phase)
    summaries = []
    seed_frames = []
    for index, (name, coefficients) in enumerate(CONTRASTS.items()):
        summary, seed_frame = _estimand(
            frame,
            name=name,
            coefficients=coefficients,
            bootstrap_samples=args.bootstrap_samples,
            random_seed=args.random_seed + index * 101,
        )
        summaries.append(summary)
        seed_frames.append(seed_frame)
    by_name = {value["estimand"]: value for value in summaries}
    primary_names = (
        "full_span_state_sufficiency",
        "selected_full_path_mask_interaction",
        "selected_full_path_output_restoration",
    )
    result = {
        "schema_version": "realistic_niah_v5_html_local_head_span_analysis_v1",
        "phase": args.phase,
        "model_label": str(frame["model_label"].iloc[0]),
        "seed_count": int(frame["seed"].nunique()),
        "row_plan_sha256": str(plan["plan_sha256"]),
        "targeted_bank_sha256": str(plan["targeted_bank_sha256"]),
        "selection_rank_used": False,
        "outcome_blind": True,
        "head_token_geometry": "query_plus_full_path",
        "primary_outcome": PRIMARY_OUTCOME,
        "estimands": summaries,
        "primary_estimands": [by_name[name] for name in primary_names],
        "complete_local_head_span_pass": bool(
            all(by_name[name]["positive_95pct_ci"] for name in primary_names)
        ),
        "random_specificity_is_secondary": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_csv(args.output / "seed_effects.csv", pd.concat(seed_frames, ignore_index=True))
    _atomic_json(args.output / "claim_gates.json", result)
    _atomic_json(
        args.output / "audit.json",
        {
            "status": "PASS",
            "phase": args.phase,
            "seed_count": int(frame["seed"].nunique()),
            "arms_per_seed": 7,
            "trial_rows": int(len(frame)),
            "selection_rank_used": False,
            "outcome_blind": True,
            "head_token_geometry": "query_plus_full_path",
        },
    )


if __name__ == "__main__":
    main()
