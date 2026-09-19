#!/usr/bin/env python3
"""Analyze Qwen's complementary residual-relay and direct-reread factorial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


PATCHES = ("self_patch", "full_donor_patch")
RELAYS = ("natural_relay", "post_terminal_suffix_clean_reset")
MASKS = ("clean", "block_trace_items", "block_trace_items_matched_control")
EQUIVALENCE_BOUND = 0.20
GEOMETRY_REASON = (
    "not applicable: a trace item is shorter than the requested suffix8 geometry"
)


def _read_shards(path: Path) -> pd.DataFrame:
    files = sorted((path / "shards").glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL shards under {path}")
    rows: list[dict[str, Any]] = []
    for file in files:
        with file.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return pd.DataFrame(rows)


def complementary_pair_effects(trials: pd.DataFrame) -> pd.DataFrame:
    ok = trials.loc[trials["status"].eq("ok")].copy()
    identity = [
        "model_label",
        "seed",
        "request_id",
        "gold_count",
        "mechanism_split",
        "pair_sha256",
        "donor_offset",
        "source_layer",
        "relay_layer",
    ]
    duplicated = ok.duplicated(
        identity + ["patch_condition", "relay_condition", "mask_condition"],
        keep=False,
    )
    if duplicated.any():
        raise ValueError("Complementary readout contains duplicate factorial cells")
    expected = {
        (patch, relay, mask)
        for patch in PATCHES
        for relay in RELAYS
        for mask in MASKS
    }
    outcomes = [
        value
        for value in ("correct_count_margin", "exact_count", "strict_count_utility")
        if value in ok.columns
    ]
    if "correct_count_margin" not in outcomes:
        raise ValueError("Complementary readout lacks correct_count_margin")
    wide = ok.pivot(
        index=identity,
        columns=["patch_condition", "relay_condition", "mask_condition"],
        values=outcomes,
    )
    observed = set(
        zip(
            wide.columns.get_level_values(1),
            wide.columns.get_level_values(2),
            wide.columns.get_level_values(3),
        )
    )
    if observed != expected:
        raise ValueError(
            f"Complementary cells differ: missing={expected - observed}, "
            f"extra={observed - expected}"
        )
    output = wide.index.to_frame(index=False)

    def cell(outcome: str, patch: str, relay: str, mask: str) -> np.ndarray:
        return pd.to_numeric(
            wide[(outcome, patch, relay, mask)], errors="coerce"
        ).to_numpy(dtype=float)

    for outcome in outcomes:
        damages: dict[tuple[str, str], np.ndarray] = {}
        for relay in RELAYS:
            for mask in MASKS:
                damages[(relay, mask)] = cell(
                    outcome, "self_patch", relay, mask
                ) - cell(outcome, "full_donor_patch", relay, mask)
                output[f"{outcome}__patch_damage__{relay}__{mask}"] = damages[
                    (relay, mask)
                ]
        natural_clean = damages[("natural_relay", "clean")]
        natural_true = damages[("natural_relay", "block_trace_items")]
        natural_matched = damages[
            ("natural_relay", "block_trace_items_matched_control")
        ]
        reset_true = damages[
            ("post_terminal_suffix_clean_reset", "block_trace_items")
        ]
        reset_matched = damages[
            (
                "post_terminal_suffix_clean_reset",
                "block_trace_items_matched_control",
            )
        ]
        output[f"{outcome}__relay_mediation_matched"] = (
            natural_matched - reset_matched
        )
        output[f"{outcome}__source_mediation_natural"] = (
            natural_matched - natural_true
        )
        output[f"{outcome}__source_mediation_after_relay"] = (
            reset_matched - reset_true
        )
        output[f"{outcome}__combined_mediation"] = natural_clean - reset_true
    return output


def _seed_values(effects: pd.DataFrame, column: str) -> np.ndarray:
    return (
        effects.groupby("seed", as_index=False)[column]
        .mean()[column]
        .to_numpy(dtype=float)
    )


def _bootstrap_mean(
    values: np.ndarray, *, samples: int, random_seed: int
) -> dict[str, float]:
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Complementary bootstrap requires finite seed effects")
    rng = np.random.default_rng(int(random_seed))
    indices = rng.integers(0, len(values), size=(int(samples), len(values)))
    draws = values[indices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "seed_count": int(len(values)),
    }


def _summary(
    effects: pd.DataFrame,
    column: str,
    *,
    bootstrap_samples: int,
    random_seed: int,
) -> dict[str, float]:
    return _bootstrap_mean(
        _seed_values(effects, column),
        samples=bootstrap_samples,
        random_seed=random_seed,
    )


def _combined_ratio_gate(
    effects: pd.DataFrame, *, bootstrap_samples: int, random_seed: int
) -> dict[str, Any]:
    denominator = "correct_count_margin__patch_damage__natural_relay__clean"
    numerator = (
        "correct_count_margin__patch_damage__"
        "post_terminal_suffix_clean_reset__block_trace_items"
    )
    seed_frame = effects.groupby("seed", as_index=False)[
        [denominator, numerator]
    ].mean()
    values = seed_frame[[denominator, numerator]].to_numpy(dtype=float)
    rng = np.random.default_rng(int(random_seed))
    indices = rng.integers(
        0, len(values), size=(int(bootstrap_samples), len(values))
    )
    draws = values[indices].mean(axis=1)
    valid = np.abs(draws[:, 0]) > 1e-8
    if valid.mean() < 0.99:
        raise ValueError("Natural patch damage is too small for ratio gate")
    ratios = np.abs(draws[valid, 1]) / np.abs(draws[valid, 0])
    point = abs(float(values[:, 1].mean())) / abs(float(values[:, 0].mean()))
    high = float(np.quantile(ratios, 0.975))
    return {
        "estimate": point,
        "ci_low": float(np.quantile(ratios, 0.025)),
        "ci_high": high,
        "relative_equivalence_bound": EQUIVALENCE_BOUND,
        "pass": high < EQUIVALENCE_BOUND,
        "rule": "combined residual ratio CI high < 0.20",
    }


def complementary_claim_gates(
    effects: pd.DataFrame,
    *,
    phase: str,
    bootstrap_samples: int,
    random_seed: int,
) -> dict[str, Any]:
    columns = {
        "storage_main_effect": (
            "correct_count_margin__patch_damage__natural_relay__clean",
            "ci_low > 0",
        ),
        "matched_source_control_preserves_patch": (
            "correct_count_margin__patch_damage__natural_relay__"
            "block_trace_items_matched_control",
            "ci_low > 0",
        ),
        "source_only_leaves_residual": (
            "correct_count_margin__patch_damage__natural_relay__"
            "block_trace_items",
            "ci_low > 0",
        ),
        "relay_only_leaves_residual": (
            "correct_count_margin__patch_damage__"
            "post_terminal_suffix_clean_reset__block_trace_items_matched_control",
            "ci_low > 0",
        ),
        "residual_relay_contribution": (
            "correct_count_margin__relay_mediation_matched",
            "ci_low > 0",
        ),
        "direct_reread_contribution_after_relay": (
            "correct_count_margin__source_mediation_after_relay",
            "ci_low > 0",
        ),
    }
    gates: dict[str, Any] = {}
    for index, (gate_id, (column, rule)) in enumerate(columns.items()):
        summary = _summary(
            effects,
            column,
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed + index,
        )
        gates[gate_id] = {
            **summary,
            "pass": summary["ci_low"] > 0,
            "rule": rule,
        }
    gates["joint_cut_residual_equivalence"] = _combined_ratio_gate(
        effects,
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed + 100,
    )
    primary = tuple(gates)

    greedy_ids: list[str] = []
    if "exact_count__combined_mediation" in effects.columns:
        for offset, (gate_id, column) in enumerate(
            (
                (
                    "greedy_exact_count_patch_effect",
                    "exact_count__patch_damage__natural_relay__clean",
                ),
                (
                    "greedy_exact_count_combined_mediation",
                    "exact_count__combined_mediation",
                ),
            )
        ):
            summary = _summary(
                effects,
                column,
                bootstrap_samples=bootstrap_samples,
                random_seed=random_seed + 200 + offset,
            )
            gates[gate_id] = {
                **summary,
                "pass": summary["ci_low"] > 0,
                "rule": "supplementary greedy exact-count CI low > 0",
                "role": "supplementary_final_output",
            }
            greedy_ids.append(gate_id)
    return {
        "phase": phase,
        "confirmatory": phase == "confirmation",
        "primary_gate_ids": list(primary),
        "complementary_readout_pass": all(
            bool(gates[name]["pass"]) for name in primary
        ),
        "supplementary_greedy_gate_ids": greedy_ids,
        "greedy_exact_count_support_pass": (
            all(bool(gates[name]["pass"]) for name in greedy_ids)
            if greedy_ids
            else None
        ),
        "gates": gates,
        "allowed_claim_if_confirmation_passes": (
            "Qwen reads a terminal trace count state through complementary "
            "post-terminal residual and direct trace-source attention routes; "
            "each route alone is insufficient, while their joint removal "
            "eliminates the state-patch effect."
        ),
        "restriction": (
            "The assay identifies two functional route classes, not a minimal "
            "head circuit or a uniquely scalar counter."
        ),
    }


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["discovery", "confirmation"], required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260821)
    args = parser.parse_args(argv)

    trials = _read_shards(args.trials.resolve())
    expected_seed_count = 20 if args.phase == "discovery" else 10
    observed_seeds = sorted(int(value) for value in trials["seed"].unique())
    if len(observed_seeds) != expected_seed_count:
        raise ValueError(
            f"{args.phase} requires {expected_seed_count} seeds, got {observed_seeds}"
        )
    if "selection_rank" in trials.columns:
        raise ValueError("Formal complementary readout must not use selection_rank")
    not_applicable = trials.loc[trials["status"].eq("not_applicable")]
    if not not_applicable.empty:
        if set(not_applicable["exclusion_reason"].astype(str)) != {GEOMETRY_REASON}:
            raise ValueError("Complementary readout has an unregistered exclusion")
        counts = not_applicable.groupby("pair_sha256").size()
        if not (counts == 12).all():
            raise ValueError("Geometry N/A pairs do not contain all 12 cells")
    ok = trials.loc[trials["status"].eq("ok")]
    ok_counts = ok.groupby("pair_sha256").size()
    if not len(ok_counts) or not (ok_counts == 12).all():
        raise ValueError("Every eligible complementary pair must contain 12 cells")
    effects = complementary_pair_effects(trials)
    claims = complementary_claim_gates(
        effects,
        phase=args.phase,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
    )
    output = args.output.resolve()
    _atomic_csv(output / "pair_effects.csv", effects)
    _atomic_json(output / "claim_gates.json", claims)
    _atomic_json(
        output / "audit.json",
        {
            "status": "PASS",
            "phase": args.phase,
            "seed_count": len(observed_seeds),
            "seeds": observed_seeds,
            "pair_count": int(effects["pair_sha256"].nunique()),
            "planned_pair_count": int(trials["pair_sha256"].nunique()),
            "geometry_not_applicable_pair_count": int(
                not_applicable["pair_sha256"].nunique()
            ),
            "trial_rows": int(len(trials)),
            "cells_per_pair": 12,
            "selection_rank_used": False,
            "complementary_readout_pass": claims["complementary_readout_pass"],
            "relative_equivalence_bound": EQUIVALENCE_BOUND,
            "bootstrap_samples": int(args.bootstrap_samples),
        },
    )
    print(json.dumps(claims, sort_keys=True))


if __name__ == "__main__":
    main()
