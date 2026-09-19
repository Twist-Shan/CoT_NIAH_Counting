#!/usr/bin/env python3
"""Analyze grammar-timed terminal-span decomposition by effect size and stratum."""

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
    REGISTERED_GRAMMAR_SPAN_DECOMPOSITION_CONDITIONS,
    REGISTERED_GRAMMAR_SPAN_DECOMPOSITION_GEOMETRIES,
)


OUTCOMES = (
    "correct_count_margin",
    "expected_count_utility",
    "correct_count_probability",
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


def analyze(
    root: Path,
    *,
    phase: str,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    plan = json.loads((root / "frozen_row_plan.json").read_text(encoding="utf-8"))
    files = sorted((root / "shards").glob("*.jsonl"))
    expected = int(plan.get("seed_count", 20 if phase == "discovery" else 10))
    if len(files) != expected:
        raise ValueError(f"Expected {expected} grammar-span shards, observed {len(files)}")
    rows: list[dict[str, Any]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    frame = pd.DataFrame(rows)
    expected_rows = expected * len(REGISTERED_GRAMMAR_SPAN_DECOMPOSITION_CONDITIONS)
    if len(frame) != expected_rows:
        raise ValueError("Grammar-span trial row count changed")
    if set(frame["condition"].astype(str)) != set(
        REGISTERED_GRAMMAR_SPAN_DECOMPOSITION_CONDITIONS
    ):
        raise ValueError("Grammar-span conditions changed")
    if frame["selection_rank_used"].map(bool).any():
        raise ValueError("Grammar-span experiment used selection_rank")
    for column in (
        "outcome_blind",
        "target_is_terminal",
        "all_trace_items_replaced",
        "control_sequence_length_equal",
    ):
        if not frame[column].map(bool).all():
            raise ValueError(f"Grammar-span audit failed for {column}")
    if frame["span_selection_uses_outcome"].map(bool).any():
        raise ValueError("Grammar-span site selection accessed outcomes")
    if frame["seed"].nunique() != expected:
        raise ValueError("Grammar-span effective seed count changed")
    if set(frame["row_plan_sha256"].astype(str)) != {str(plan["plan_sha256"])}:
        raise ValueError("Grammar-span row plan hash changed")
    seed_grammar = frame[["seed", "grammar_timing_stratum"]].drop_duplicates()
    observed_balance = seed_grammar["grammar_timing_stratum"].value_counts().to_dict()
    expected_balance = {
        str(key): int(value) for key, value in plan["timing_counts"].items()
    }
    if observed_balance != expected_balance:
        raise ValueError("Grammar-span timing balance changed")

    index = [
        "model_label",
        "seed",
        "gold_count",
        "request_id",
        "grammar_timing_stratum",
        "terminal_grammar_class",
    ]
    effect_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    strata = ("all", "rank_after_city", "rank_before_city")
    for outcome_index, outcome in enumerate(OUTCOMES):
        active = frame.copy()
        active[outcome] = pd.to_numeric(active[outcome], errors="coerce")
        wide = active.pivot(index=index, columns="condition", values=outcome)
        for geometry_index, geometry in enumerate(
            REGISTERED_GRAMMAR_SPAN_DECOMPOSITION_GEOMETRIES
        ):
            contrasts = {
                "restoration": {
                    f"{geometry}_restore": 1.0,
                    "uninformative": -1.0,
                },
                "matched_random_specificity": {
                    f"{geometry}_restore": 1.0,
                    f"{geometry}_matched_random": -1.0,
                },
            }
            for contrast_index, (contrast, coefficients) in enumerate(
                contrasts.items()
            ):
                pair = wide.dropna(subset=list(coefficients)).copy()
                effect = pd.Series(0.0, index=pair.index, dtype=float)
                for condition, coefficient in coefficients.items():
                    effect += float(coefficient) * pair[condition].astype(float)
                for stratum_index, stratum in enumerate(strata):
                    selected = (
                        effect
                        if stratum == "all"
                        else effect[
                            effect.index.get_level_values(
                                "grammar_timing_stratum"
                            ).astype(str)
                            == stratum
                        ]
                    )
                    values = selected.to_numpy(float)
                    summary = bootstrap_seed_mean_ci(
                        values,
                        samples=int(bootstrap_samples),
                        seed=(
                            int(random_seed)
                            + outcome_index * 10007
                            + geometry_index * 1009
                            + contrast_index * 101
                            + stratum_index * 17
                        ),
                    )
                    summary.update(
                        {
                            "outcome": outcome,
                            "geometry": geometry,
                            "contrast": contrast,
                            "stratum": stratum,
                            "coefficients": coefficients,
                            "n_seeds": len(values),
                            "p_value": sign_flip_pvalue(values),
                            "positive_mean": bool(summary["mean_effect"] > 0.0),
                            "positive_95pct_ci": bool(summary["ci_low"] > 0.0),
                            "formal_overall_panel": stratum == "all",
                            "grammar_stratum_diagnostic": stratum != "all",
                        }
                    )
                    summaries.append(summary)
                for key, value in effect.items():
                    effect_rows.append(
                        {
                            "model_label": key[0],
                            "seed": int(key[1]),
                            "gold_count": int(key[2]),
                            "request_id": key[3],
                            "grammar_timing_stratum": key[4],
                            "terminal_grammar_class": key[5],
                            "outcome": outcome,
                            "geometry": geometry,
                            "contrast": contrast,
                            "effect": float(value),
                        }
                    )
    primary = [
        value
        for value in summaries
        if value["outcome"] == "correct_count_margin"
        and value["stratum"] == "all"
    ]
    restoration_ranking = sorted(
        (
            value
            for value in primary
            if value["contrast"] == "restoration"
        ),
        key=lambda value: float(value["mean_effect"]),
        reverse=True,
    )
    specificity_by_geometry = {
        str(value["geometry"]): value
        for value in primary
        if value["contrast"] == "matched_random_specificity"
    }
    split_ranking = [
        value for value in restoration_ranking if value["geometry"] != "full_item"
    ]
    largest_split = split_ranking[0]
    largest_split_specificity = specificity_by_geometry[str(largest_split["geometry"])]
    claims = {
        "schema_version": "realistic_niah_v5_grammar_span_decomposition_analysis_v1",
        "phase": phase,
        "model_label": str(frame["model_label"].iloc[0]),
        "seed_count": expected,
        "timing_counts": observed_balance,
        "selection_rank_used": False,
        "outcome_blind": True,
        "primary_effect_size_outcome": "correct_count_margin",
        "primary_estimands": primary,
        "effect_size_ranking": restoration_ranking,
        "largest_split_geometry": str(largest_split["geometry"]),
        "largest_split_restoration": largest_split,
        "largest_split_matched_random_specificity": largest_split_specificity,
        "descriptive_split_signal": bool(
            largest_split["positive_mean"]
            and largest_split_specificity["positive_mean"]
        ),
        "grammar_stratum_estimands": [
            value
            for value in summaries
            if value["outcome"] == "correct_count_margin"
            and value["stratum"] != "all"
        ],
        "all_estimands": summaries,
        "interpretation": (
            "Correct-count margin is the primary effect-size readout. Confidence "
            "intervals and sign-flip p-values are retained, but the diagnostic "
            "ranking is based on the sealed mean effect. Grammar strata are "
            "diagnostic subdivisions of the frozen overall panel rather than "
            "separate experiments."
        ),
    }
    audit = {
        "status": "PASS",
        "phase": phase,
        "seed_count": expected,
        "trial_rows": len(frame),
        "conditions_per_seed": len(
            REGISTERED_GRAMMAR_SPAN_DECOMPOSITION_CONDITIONS
        ),
        "timing_counts": observed_balance,
        "selection_rank_used": False,
        "outcome_blind": True,
    }
    return pd.DataFrame(effect_rows), claims, audit


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
