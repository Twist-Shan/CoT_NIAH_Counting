#!/usr/bin/env python3
"""Analyze a frozen-bank continuous next-city retrieval diagnostic."""

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


SCHEMA_VERSION = "realistic_niah_v6_targeted_city_likelihood_analysis_v1"
OUTCOMES = (
    "target_city_log_probability",
    "target_city_logit_margin",
    "target_city_log_odds",
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def _read_rows(root: Path) -> pd.DataFrame:
    files = [root] if root.is_file() else sorted((root / "shards").glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No targeted likelihood shards under {root}")
    rows = [
        json.loads(line)
        for path in files
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Targeted likelihood diagnostic has no rows")
    return frame


def analyze(
    frame: pd.DataFrame,
    *,
    phase: str,
    expected_seeds: int,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "model_label",
        "seed",
        "request_id",
        "anchor_equivalence_id",
        "condition",
        "repeat",
        "status",
        *OUTCOMES,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Targeted likelihood rows lack fields: {missing}")
    if set(frame["status"].astype(str)) != {"ok"}:
        raise ValueError("Targeted likelihood diagnostic contains incomplete rows")
    if "selection_rank" in frame.columns:
        raise ValueError("Targeted likelihood diagnostic contains selection_rank")
    if frame.get("selection_rank_used", pd.Series(False, index=frame.index)).map(
        bool
    ).any():
        raise ValueError("Targeted likelihood diagnostic used selection rank")
    observed_seed_count = frame["seed"].astype(int).nunique()
    if observed_seed_count != int(expected_seeds):
        raise ValueError(
            f"Expected {expected_seeds} true source seeds, observed {observed_seed_count}"
        )
    conditions = set(frame["condition"].astype(str))
    if not {"clean", "selected_bank"} <= conditions:
        raise ValueError("Targeted likelihood diagnostic lacks clean or selected bank")
    random_conditions = conditions - {"clean", "selected_bank"}
    if len(random_conditions) != 1:
        raise ValueError(
            "Targeted likelihood diagnostic requires one frozen random-control family"
        )
    random_condition = next(iter(random_conditions))

    keys = ["model_label", "seed", "request_id", "anchor_equivalence_id"]
    active = frame.copy()
    active["arm"] = active["condition"].astype(str)
    random_mask = active["condition"].astype(str).eq(random_condition)
    active.loc[random_mask, "arm"] = active.loc[random_mask].apply(
        lambda row: f"random_r{int(row['repeat'])}", axis=1
    )
    if active.duplicated(keys + ["arm"]).any():
        raise ValueError("Targeted likelihood diagnostic duplicates an anchor arm")
    expected_arms = {"clean", "selected_bank", "random_r1", "random_r2", "random_r3"}
    observed_arms = set(active["arm"].astype(str))
    if observed_arms != expected_arms:
        raise ValueError(
            f"Targeted likelihood factorial changed: {sorted(observed_arms)}"
        )

    summaries: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for outcome_index, outcome in enumerate(OUTCOMES):
        active[outcome] = pd.to_numeric(active[outcome], errors="raise").astype(float)
        wide = active.pivot(index=keys, columns="arm", values=outcome)
        wide = wide.dropna(subset=sorted(expected_arms)).copy()
        wide["selected_damage"] = wide["clean"] - wide["selected_bank"]
        wide["selected_vs_random_specificity"] = (
            wide[["random_r1", "random_r2", "random_r3"]].mean(axis=1)
            - wide["selected_bank"]
        )
        for estimand_index, estimand in enumerate(
            ("selected_damage", "selected_vs_random_specificity")
        ):
            per_seed = (
                wide.reset_index()
                .groupby(["model_label", "seed"], as_index=False)
                .agg(effect=(estimand, "mean"), anchor_count=("request_id", "count"))
            )
            values = per_seed["effect"].to_numpy(float)
            summary = bootstrap_seed_mean_ci(
                values,
                samples=int(bootstrap_samples),
                seed=int(random_seed) + outcome_index * 101 + estimand_index,
            )
            summary.update(
                {
                    "outcome": outcome,
                    "estimand": estimand,
                    "n_seeds": len(values),
                    "anchor_count": len(wide),
                    "p_value": sign_flip_pvalue(values),
                    "positive_95pct_ci": bool(summary["ci_low"] > 0.0),
                    "primary": outcome == "target_city_log_probability",
                }
            )
            summaries.append(summary)
            for row in per_seed.to_dict("records"):
                seed_rows.append(
                    {
                        "model_label": str(row["model_label"]),
                        "seed": int(row["seed"]),
                        "outcome": outcome,
                        "estimand": estimand,
                        "effect": float(row["effect"]),
                        "anchor_count": int(row["anchor_count"]),
                    }
                )

    primary = [row for row in summaries if row["primary"]]
    if len(primary) != 2:
        raise RuntimeError("Targeted likelihood primary estimand count changed")
    directional = all(float(row["mean_effect"]) > 0.0 for row in primary)
    strong = all(bool(row["positive_95pct_ci"]) for row in primary)
    clean = active.loc[active["arm"].eq("clean")]
    claims = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "analysis_status": "POSTHOC_DIAGNOSTIC_SPLIT_REUSE",
        "model_label": str(frame["model_label"].iloc[0]),
        "seed_count": int(expected_seeds),
        "anchor_count": int(len(wide)),
        "random_control_family": random_condition,
        "primary_outcome": "target_city_log_probability",
        "primary_estimands": primary,
        "all_estimands": summaries,
        "directional_specific_signal": bool(directional),
        "strong_interval_gate_pass": bool(strong),
        "clean_teacher_forced_city_exact_rate": (
            float(clean["target_city_teacher_forced_exact"].map(bool).mean())
            if "target_city_teacher_forced_exact" in clean
            else None
        ),
        "frozen_k_changed": False,
        "model_trials_recomputed_from_v6_baseline": False,
        "selection_rank_used": False,
        "qualification": (
            "This endpoint was registered after the binary V6 outcomes were "
            "inspected and reuses the frozen split. It diagnoses ceiling-limited "
            "binary behavior but does not replace the original gate."
        ),
    }
    return pd.DataFrame(seed_rows), claims


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--phase", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--expected-seeds", type=int, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    effects, claims = analyze(
        _read_rows(args.trials),
        phase=args.phase,
        expected_seeds=args.expected_seeds,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
    )
    _atomic_csv(args.output / "seed_effects.csv", effects)
    _atomic_json(args.output / "claim_gates.json", claims)
    print(
        json.dumps(
            {
                "status": "PASS",
                "directional_specific_signal": claims["directional_specific_signal"],
                "strong_interval_gate_pass": claims["strong_interval_gate_pass"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
