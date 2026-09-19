#!/usr/bin/env python3
"""Analyze old-HTML-aligned terminal full-span ablation/restoration."""

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

from realistic_niah_v5.causal import (  # noqa: E402
    bootstrap_seed_mean_ci,
    sign_flip_pvalue,
)
from realistic_niah_v5.count_stream import (  # noqa: E402
    COUNT_STREAM_CONFIRMATION_SEEDS,
    COUNT_STREAM_DISCOVERY_SEEDS,
    REGISTERED_HTML_ALIGNED_SPAN_CONDITIONS,
)


CONTRASTS = {
    "terminal_span_necessity": {
        "clean": 1.0,
        "clean_target_ablation": -1.0,
    },
    "terminal_span_sufficiency": {
        "uninformative_target_restore": 1.0,
        "uninformative": -1.0,
    },
    "clean_uninformative_gap": {
        "clean": 1.0,
        "uninformative": -1.0,
    },
}
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


def _read_trials(root: Path) -> pd.DataFrame:
    files = sorted((root / "shards").glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No HTML-aligned shards under {root}")
    rows: list[dict[str, Any]] = []
    for path in files:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        if len(lines) != len(REGISTERED_HTML_ALIGNED_SPAN_CONDITIONS):
            raise ValueError(f"{path} does not contain the four frozen conditions")
        rows.extend(json.loads(line) for line in lines)
    frame = pd.DataFrame(rows)
    return frame


def _audit(frame: pd.DataFrame, plan: dict[str, Any], phase: str) -> None:
    expected_seeds = (
        COUNT_STREAM_DISCOVERY_SEEDS
        if phase == "discovery"
        else COUNT_STREAM_CONFIRMATION_SEEDS
    )
    expected_rows = len(expected_seeds) * len(REGISTERED_HTML_ALIGNED_SPAN_CONDITIONS)
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, observed {len(frame)}")
    if tuple(sorted(frame["seed"].astype(int).unique())) != tuple(expected_seeds):
        raise ValueError("HTML-aligned seed contract changed")
    selected_count_by_seed = {
        int(seed): int(count)
        for seed, count in plan["selected_count_by_seed"].items()
    }
    observed_count_by_seed = {
        int(seed): int(group["gold_count"].astype(int).iloc[0])
        for seed, group in frame.groupby("seed")
    }
    if observed_count_by_seed != selected_count_by_seed:
        raise ValueError("HTML-aligned selected counts changed")
    if set(frame["condition"].astype(str)) != set(
        REGISTERED_HTML_ALIGNED_SPAN_CONDITIONS
    ):
        raise ValueError("HTML-aligned condition contract changed")
    keys = ["request_id", "condition"]
    if frame.duplicated(keys).any():
        raise ValueError("HTML-aligned outcome rows are duplicated")
    required_truth = (
        "target_is_terminal",
        "all_trace_items_replaced",
        "control_sequence_length_equal",
        "control_attention_mask_equal",
        "outcome_blind",
    )
    for column in required_truth:
        if not frame[column].map(bool).all():
            raise ValueError(f"HTML-aligned audit failed for {column}")
    if frame["selection_rank_used"].map(bool).any():
        raise ValueError("HTML-aligned experiment used selection_rank")
    if set(frame["patch_geometry"].astype(str)) != {
        "full_item_span_same_position"
    }:
        raise ValueError("HTML-aligned geometry changed")
    if set(frame["patch_layer_mode"].astype(str)) != {"cumulative_clamp"}:
        raise ValueError("HTML-aligned layer mode changed")
    if set(frame["row_plan_sha256"].astype(str)) != {str(plan["plan_sha256"])}:
        raise ValueError("HTML-aligned row-plan hash changed")


def _summaries(
    frame: pd.DataFrame, *, bootstrap_samples: int, random_seed: int
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    seed_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    index = ["model_label", "seed", "gold_count", "request_id"]
    for outcome in OUTCOMES:
        if outcome not in frame:
            continue
        active = frame.copy()
        active[outcome] = pd.to_numeric(active[outcome], errors="coerce")
        wide = active.pivot(index=index, columns="condition", values=outcome)
        for contrast_index, (estimand, coefficients) in enumerate(CONTRASTS.items()):
            required = list(coefficients)
            pair = wide.dropna(subset=required).copy()
            effect = pd.Series(0.0, index=pair.index, dtype=float)
            for condition, coefficient in coefficients.items():
                effect += float(coefficient) * pair[condition].astype(float)
            pairs = effect.rename("pair_effect").reset_index()
            per_seed = pairs.rename(columns={"pair_effect": "effect"})[
                ["model_label", "seed", "gold_count", "effect"]
            ]
            if per_seed["seed"].nunique() != len(per_seed):
                raise ValueError(f"{estimand}/{outcome} has duplicate seed rows")
            values = per_seed["effect"].to_numpy(float)
            summary = bootstrap_seed_mean_ci(
                values,
                samples=int(bootstrap_samples),
                seed=int(random_seed) + contrast_index * 101,
            )
            summary.update(
                {
                    "estimand": estimand,
                    "outcome": outcome,
                    "coefficients": coefficients,
                    "n_seeds": int(len(per_seed)),
                    "pair_count": int(len(pairs)),
                    "p_value": sign_flip_pvalue(values),
                    "higher_is_supportive": True,
                    "positive_95pct_ci": bool(summary["ci_low"] > 0.0),
                }
            )
            summaries.append(summary)
            for row in per_seed.itertuples(index=False):
                seed_rows.append(
                    {
                        "estimand": estimand,
                        "outcome": outcome,
                        "model_label": row.model_label,
                        "seed": int(row.seed),
                        "effect": float(row.effect),
                        "gold_count": int(row.gold_count),
                    }
                )
    return pd.DataFrame(seed_rows), summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--phase", choices=["discovery", "confirmation"], required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260821)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads((args.input / "frozen_row_plan.json").read_text(encoding="utf-8"))
    frame = _read_trials(args.input)
    _audit(frame, plan, args.phase)
    seed_effects, summaries = _summaries(
        frame,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
    )
    primary = [
        value
        for value in summaries
        if value["outcome"] == "expected_count_utility"
        and value["estimand"] in {
            "terminal_span_necessity",
            "terminal_span_sufficiency",
        }
    ]
    result = {
        "schema_version": "realistic_niah_v5_html_aligned_terminal_analysis_v1",
        "phase": args.phase,
        "model_label": str(frame["model_label"].iloc[0]),
        "seed_count": int(frame["seed"].nunique()),
        "row_selection_rule": str(plan["row_selection_rule"]),
        "selected_count_by_seed": plan["selected_count_by_seed"],
        "row_plan_sha256": str(plan["plan_sha256"]),
        "selection_rank_used": False,
        "outcome_blind": True,
        "primary_outcome": "expected_count_utility",
        "primary_estimands": primary,
        "all_estimands": summaries,
        "html_aligned_terminal_span_pass": bool(
            len(primary) == 2
            and all(value["positive_95pct_ci"] for value in primary)
        ),
        "effect_size_interpretation": (
            "Expected-count utility is negative absolute count error; positive "
            "necessity/sufficiency effects are improvements measured in count units."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_csv(args.output / "seed_effects.csv", seed_effects)
    _atomic_json(args.output / "claim_gates.json", result)
    _atomic_json(
        args.output / "audit.json",
        {
            "status": "PASS",
            "phase": args.phase,
            "seed_count": int(frame["seed"].nunique()),
            "rows_per_seed": 1,
            "trial_rows": int(len(frame)),
            "selection_rank_used": False,
            "outcome_blind": True,
        },
    )


if __name__ == "__main__":
    main()
