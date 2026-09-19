#!/usr/bin/env python3
"""Re-estimate terminal-relay effects on identical model-common trial keys."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEYS = ["seed", "gold_count", "donor_offset"]
OUTCOMES = (
    "patch_damage_natural",
    "specific_mediation__post_terminal_suffix",
    "specific_mediation__answer_query",
    "patch_damage__post_terminal_suffix",
)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _prepare(path: Path, expected_model: str, expected_phase: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(KEYS + list(OUTCOMES) + ["model_label", "mechanism_split"]) - set(
        frame.columns
    )
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if set(frame["model_label"].astype(str)) != {expected_model}:
        raise ValueError(f"{path} has the wrong model label")
    if set(frame["mechanism_split"].astype(str)) != {expected_phase}:
        raise ValueError(f"{path} has the wrong phase")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{path} contains duplicate common-support keys")
    return frame[KEYS + list(OUTCOMES)].copy()


def _bootstrap(
    seed_frame: pd.DataFrame,
    *,
    samples: int,
    random_seed: int,
) -> dict[str, dict[str, float]]:
    seeds = seed_frame["seed"].to_numpy(dtype=int)
    values = seed_frame.drop(columns="seed").to_numpy(dtype=float)
    columns = list(seed_frame.columns[1:])
    if len(seeds) < 2 or not np.isfinite(values).all():
        raise ValueError("Common-support bootstrap requires finite effects for >=2 seeds")
    rng = np.random.default_rng(int(random_seed))
    indices = rng.integers(0, len(seeds), size=(int(samples), len(seeds)))
    draws = values[indices].mean(axis=1)
    output: dict[str, dict[str, float]] = {}
    for index, column in enumerate(columns):
        output[column] = {
            "estimate": float(values[:, index].mean()),
            "ci_low": float(np.quantile(draws[:, index], 0.025)),
            "ci_high": float(np.quantile(draws[:, index], 0.975)),
        }
    natural_index = columns.index("patch_damage_natural")
    suffix_index = columns.index("specific_mediation__post_terminal_suffix")
    query_index = columns.index("specific_mediation__answer_query")
    valid = np.abs(draws[:, natural_index]) > 1e-8
    if valid.mean() < 0.99:
        raise ValueError("Natural common-support effect is too close to zero")
    for label, index in (
        ("post_terminal_suffix_explained_fraction", suffix_index),
        ("answer_query_explained_fraction", query_index),
    ):
        ratios = draws[valid, index] / draws[valid, natural_index]
        point = values[:, index].mean() / values[:, natural_index].mean()
        output[label] = {
            "estimate": float(point),
            "ci_low": float(np.quantile(ratios, 0.025)),
            "ci_high": float(np.quantile(ratios, 0.975)),
        }
    return output


def analyze(
    qwen: pd.DataFrame,
    gemma: pd.DataFrame,
    *,
    phase: str,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    merged = qwen.merge(
        gemma,
        on=KEYS,
        how="inner",
        suffixes=("__qwen", "__gemma"),
        validate="one_to_one",
    ).sort_values(KEYS, kind="stable")
    if merged.empty:
        raise ValueError("The two models have no common-support trial keys")
    expected_seeds = 20 if phase == "development" else 10
    if merged["seed"].nunique() != expected_seeds:
        raise ValueError("Common support lost a canonical seed")
    cell_counts = (
        merged.groupby(["gold_count", "donor_offset"], as_index=False)
        .agg(pair_count=("seed", "size"), seed_count=("seed", "nunique"))
        .sort_values(["gold_count", "donor_offset"], kind="stable")
    )
    models: dict[str, Any] = {}
    for model_key, suffix in (("Qwen3-8B", "qwen"), ("Gemma4-E4B", "gemma")):
        columns = [f"{outcome}__{suffix}" for outcome in OUTCOMES]
        seed_frame = merged.groupby("seed", as_index=False)[columns].mean()
        seed_frame = seed_frame.rename(
            columns={f"{outcome}__{suffix}": outcome for outcome in OUTCOMES}
        )
        effects = _bootstrap(
            seed_frame,
            samples=bootstrap_samples,
            random_seed=random_seed,
        )
        models[model_key] = {
            "seed_count": int(seed_frame["seed"].nunique()),
            "common_pair_count": int(len(merged)),
            "effects": effects,
            "pathway_primary_pass": bool(
                effects["patch_damage_natural"]["ci_low"] > 0
                and effects["specific_mediation__post_terminal_suffix"]["ci_low"] > 0
            ),
            "answer_query_secondary_pass": bool(
                effects["specific_mediation__answer_query"]["ci_low"] > 0
            ),
        }
    summary = {
        "schema_version": "realistic_niah_v5_terminal_relay_common_support_v1",
        "status": "PASS",
        "phase": phase,
        "matching_keys": KEYS,
        "common_pair_count_per_model": int(len(merged)),
        "common_seed_count": int(merged["seed"].nunique()),
        "cell_count": int(len(cell_counts)),
        "selection_uses_outcomes": False,
        "interpretation": (
            "Model-common robustness analysis on identical seed, count, and donor-offset "
            "keys. It does not replace the within-model canonical analysis."
        ),
        "models": models,
    }
    return merged, cell_counts, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--gemma", type=Path, required=True)
    parser.add_argument("--phase", choices=["development", "confirmation"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260828)
    args = parser.parse_args()
    qwen = _prepare(args.qwen, "Qwen3-8B", args.phase)
    gemma = _prepare(args.gemma, "Gemma4-E4B", args.phase)
    pairs, cells, summary = analyze(
        qwen,
        gemma,
        phase=args.phase,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
    )
    _atomic_csv(args.output / "common_support_pairs.csv", pairs)
    _atomic_csv(args.output / "common_support_cell_counts.csv", cells)
    _atomic_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
