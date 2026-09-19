#!/usr/bin/env python3
"""Analyze terminal-state propagation through a frozen pre-answer relay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


GEOMETRY_REASON = (
    "not applicable: a trace item is shorter than the requested suffix8 geometry"
)
RELAY_CONDITIONS = (
    "natural_relay",
    "answer_query_clean_reset",
    "post_terminal_suffix_clean_reset",
)
RELATIVE_EQUIVALENCE_BOUND = 0.20


def _read_shards(path: Path) -> pd.DataFrame:
    files = sorted((path / "shards").glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL shards under {path}")
    rows: list[dict[str, Any]] = []
    for file in files:
        rows.extend(
            json.loads(line)
            for line in file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return pd.DataFrame(rows)


def relay_pair_effects(trials: pd.DataFrame) -> pd.DataFrame:
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
    duplicated = trials.duplicated(
        identity + ["source_condition", "relay_condition"], keep=False
    )
    if duplicated.any():
        raise ValueError("Relay mediation contains duplicate factorial cells")
    wide = trials.pivot(
        index=identity,
        columns=["source_condition", "relay_condition"],
        values="correct_count_margin",
    )
    expected = {
        (source, relay)
        for source in ("self_patch", "full_donor_patch")
        for relay in RELAY_CONDITIONS
    }
    if set(wide.columns) != expected:
        raise ValueError(
            f"Relay factorial cells differ: missing={expected - set(wide.columns)}"
        )
    output = wide.index.to_frame(index=False)
    self_natural = wide[("self_patch", "natural_relay")].to_numpy(dtype=float)
    donor_natural = wide[("full_donor_patch", "natural_relay")].to_numpy(
        dtype=float
    )
    output["patch_damage_natural"] = self_natural - donor_natural
    for relay in RELAY_CONDITIONS[1:]:
        self_reset = wide[("self_patch", relay)].to_numpy(dtype=float)
        donor_reset = wide[("full_donor_patch", relay)].to_numpy(dtype=float)
        label = relay.removesuffix("_clean_reset")
        output[f"patch_damage__{label}"] = self_reset - donor_reset
        output[f"specific_mediation__{label}"] = (
            (self_natural - donor_natural) - (self_reset - donor_reset)
        )
        output[f"self_reset_damage__{label}"] = self_natural - self_reset
    for outcome in ("exact_count", "strict_count_utility"):
        if outcome not in trials.columns:
            continue
        outcome_wide = trials.pivot(
            index=identity,
            columns=["source_condition", "relay_condition"],
            values=outcome,
        )
        if set(outcome_wide.columns) != expected:
            raise ValueError(
                f"Relay {outcome} cells differ: "
                f"missing={expected - set(outcome_wide.columns)}"
            )
        outcome_self_natural = outcome_wide[
            ("self_patch", "natural_relay")
        ].to_numpy(dtype=float)
        outcome_donor_natural = outcome_wide[
            ("full_donor_patch", "natural_relay")
        ].to_numpy(dtype=float)
        output[f"{outcome}__patch_damage_natural"] = (
            outcome_self_natural - outcome_donor_natural
        )
        for relay in RELAY_CONDITIONS[1:]:
            label = relay.removesuffix("_clean_reset")
            outcome_self_reset = outcome_wide[("self_patch", relay)].to_numpy(
                dtype=float
            )
            outcome_donor_reset = outcome_wide[
                ("full_donor_patch", relay)
            ].to_numpy(dtype=float)
            outcome_reset_damage = outcome_self_reset - outcome_donor_reset
            output[f"{outcome}__patch_damage__{label}"] = outcome_reset_damage
            output[f"{outcome}__specific_mediation__{label}"] = (
                (outcome_self_natural - outcome_donor_natural)
                - outcome_reset_damage
            )
            output[f"{outcome}__self_reset_damage__{label}"] = (
                outcome_self_natural - outcome_self_reset
            )
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
        raise ValueError("Relay bootstrap requires finite seed effects")
    rng = np.random.default_rng(int(random_seed))
    indices = rng.integers(0, len(values), size=(int(samples), len(values)))
    draws = values[indices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "seed_count": int(len(values)),
    }


def _ratio_gate(
    effects: pd.DataFrame,
    *,
    numerator: str,
    denominator: str,
    bootstrap_samples: int,
    random_seed: int,
) -> dict[str, Any]:
    seed_frame = effects.groupby("seed", as_index=False)[
        [numerator, denominator]
    ].mean()
    values = seed_frame[[numerator, denominator]].to_numpy(dtype=float)
    rng = np.random.default_rng(int(random_seed))
    indices = rng.integers(
        0, len(values), size=(int(bootstrap_samples), len(values))
    )
    draws = values[indices].mean(axis=1)
    valid = np.abs(draws[:, 1]) > 1e-8
    if valid.mean() < 0.99:
        raise ValueError("Natural patch effect is too close to zero for a ratio gate")
    ratios = np.abs(draws[valid, 0]) / np.abs(draws[valid, 1])
    point = abs(float(values[:, 0].mean())) / abs(float(values[:, 1].mean()))
    high = float(np.quantile(ratios, 0.975))
    return {
        "estimate": point,
        "ci_low": float(np.quantile(ratios, 0.025)),
        "ci_high": high,
        "relative_equivalence_bound": RELATIVE_EQUIVALENCE_BOUND,
        "pass": high < RELATIVE_EQUIVALENCE_BOUND,
        "rule": "residual ratio CI high < 0.20",
    }


def relay_claim_gates(
    effects: pd.DataFrame,
    *,
    phase: str,
    bootstrap_samples: int,
    random_seed: int,
) -> dict[str, Any]:
    natural = _bootstrap_mean(
        _seed_values(effects, "patch_damage_natural"),
        samples=bootstrap_samples,
        random_seed=random_seed,
    )
    suffix_mediation = _bootstrap_mean(
        _seed_values(effects, "specific_mediation__post_terminal_suffix"),
        samples=bootstrap_samples,
        random_seed=random_seed + 1,
    )
    query_mediation = _bootstrap_mean(
        _seed_values(effects, "specific_mediation__answer_query"),
        samples=bootstrap_samples,
        random_seed=random_seed + 2,
    )
    suffix_ratio = _ratio_gate(
        effects,
        numerator="patch_damage__post_terminal_suffix",
        denominator="patch_damage_natural",
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed + 3,
    )
    self_ratio = _ratio_gate(
        effects,
        numerator="self_reset_damage__post_terminal_suffix",
        denominator="patch_damage_natural",
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed + 4,
    )
    greedy_natural = None
    greedy_suffix_mediation = None
    if "exact_count__patch_damage_natural" in effects.columns:
        greedy_natural = _bootstrap_mean(
            _seed_values(effects, "exact_count__patch_damage_natural"),
            samples=bootstrap_samples,
            random_seed=random_seed + 5,
        )
        greedy_suffix_mediation = _bootstrap_mean(
            _seed_values(
                effects,
                "exact_count__specific_mediation__post_terminal_suffix",
            ),
            samples=bootstrap_samples,
            random_seed=random_seed + 6,
        )
    gates = {
        "terminal_state_patch_effect": {
            **natural,
            "pass": natural["ci_low"] > 0,
            "rule": "natural terminal donor patch damage CI low > 0",
        },
        "post_terminal_suffix_specific_mediation": {
            **suffix_mediation,
            "pass": suffix_mediation["ci_low"] > 0,
            "rule": "patch-by-clean-suffix-reset interaction CI low > 0",
        },
        "post_terminal_suffix_residual_equivalence": suffix_ratio,
        "self_reset_is_nondamaging": self_ratio,
        "answer_query_only_mediation": {
            **query_mediation,
            "pass": query_mediation["ci_low"] > 0,
            "rule": "secondary: query-only reset interaction CI low > 0",
        },
    }
    greedy_gate_ids: list[str] = []
    if greedy_natural is not None and greedy_suffix_mediation is not None:
        gates.update(
            {
                "greedy_exact_count_patch_effect": {
                    **greedy_natural,
                    "pass": greedy_natural["ci_low"] > 0,
                    "rule": "supplementary exact-count patch damage ci_low > 0",
                    "role": "supplementary_final_output",
                },
                "greedy_exact_count_suffix_mediation": {
                    **greedy_suffix_mediation,
                    "pass": greedy_suffix_mediation["ci_low"] > 0,
                    "rule": "supplementary exact-count suffix mediation ci_low > 0",
                    "role": "supplementary_final_output",
                },
            }
        )
        greedy_gate_ids = [
            "greedy_exact_count_patch_effect",
            "greedy_exact_count_suffix_mediation",
        ]
    primary = (
        "terminal_state_patch_effect",
        "post_terminal_suffix_specific_mediation",
        "post_terminal_suffix_residual_equivalence",
        "self_reset_is_nondamaging",
    )
    return {
        "phase": phase,
        "confirmatory": phase == "confirmation",
        "primary_gate_ids": list(primary),
        "residual_relay_pass": all(bool(gates[name]["pass"]) for name in primary),
        "supplementary_greedy_gate_ids": greedy_gate_ids,
        "greedy_exact_count_support_pass": (
            all(bool(gates[name]["pass"]) for name in greedy_gate_ids)
            if greedy_gate_ids
            else None
        ),
        "gates": gates,
        "allowed_claim_if_confirmation_passes": (
            "A terminal trace count state propagates through the post-terminal "
            "residual suffix before determining the answer count."
        ),
        "restriction": (
            "The assay identifies the complete post-terminal suffix as a relay; "
            "the query-only secondary arm determines whether one last position "
            "is sufficient."
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
    status_by_pair = trials.groupby("pair_sha256")["status"].agg(
        lambda values: tuple(sorted(set(str(value) for value in values)))
    )
    if not status_by_pair.isin([("ok",), ("not_applicable",)]).all():
        raise ValueError("Relay analysis found a mixed or invalid pair status")
    excluded = trials.loc[trials["status"].astype(str).eq("not_applicable")]
    if not excluded.empty:
        if set(excluded["exclusion_reason"].astype(str)) != {GEOMETRY_REASON}:
            raise ValueError("Relay analysis found an unregistered exclusion reason")
        if not (excluded.groupby("pair_sha256").size() == 6).all():
            raise ValueError("A relay N/A pair must contain all six factorial cells")
    expected_seed_count = 20 if args.phase == "discovery" else 10
    planned_seeds = sorted(set(int(seed) for seed in trials["seed"]))
    if len(planned_seeds) != expected_seed_count:
        raise ValueError("Relay analysis lost a canonical seed")
    planned_pairs = int(trials["pair_sha256"].nunique())
    excluded_pairs = int(excluded["pair_sha256"].nunique())
    trials = trials.loc[trials["status"].astype(str).eq("ok")].copy()
    expected_split = "development" if args.phase == "discovery" else "confirmation"
    if set(trials["mechanism_split"].astype(str)) != {expected_split}:
        raise ValueError("Relay analysis received the wrong split")
    eligible_seeds = sorted(set(int(seed) for seed in trials["seed"]))
    if not eligible_seeds:
        raise ValueError("Relay analysis has no geometry-eligible canonical seed")
    geometry_not_applicable_full_seeds = sorted(
        set(planned_seeds) - set(eligible_seeds)
    )
    if "selection_rank" in trials.columns:
        raise ValueError("Formal relay trials must not use selection_rank")
    if set(trials["source_layer"].astype(int)) - {19, 16}:
        raise ValueError("Unexpected terminal source layer")
    if set(trials["relay_layer"].astype(int)) - {26, 34}:
        raise ValueError("Unexpected frozen relay layer")
    effects = relay_pair_effects(trials)
    claims = relay_claim_gates(
        effects,
        phase=str(args.phase),
        bootstrap_samples=int(args.bootstrap_samples),
        random_seed=int(args.random_seed),
    )
    output = args.output.resolve()
    _atomic_csv(output / "pair_effects.csv", effects)
    _atomic_json(output / "claim_gates.json", claims)
    per_seed = effects.groupby("seed")["pair_sha256"].nunique()
    _atomic_json(
        output / "audit.json",
        {
            "status": "PASS",
            "phase": str(args.phase),
            "trial_rows": int(len(trials)),
            "planned_pair_count": planned_pairs,
            "eligible_pair_count": int(effects["pair_sha256"].nunique()),
            "geometry_not_applicable_pair_count": excluded_pairs,
            "geometry_not_applicable_reason": GEOMETRY_REASON,
            "planned_seed_count": len(planned_seeds),
            "seed_count": int(effects["seed"].nunique()),
            "geometry_not_applicable_full_seed_count": len(
                geometry_not_applicable_full_seeds
            ),
            "geometry_not_applicable_full_seeds": geometry_not_applicable_full_seeds,
            "pairs_per_seed_min": int(per_seed.min()),
            "pairs_per_seed_max": int(per_seed.max()),
            "source_patch_stops_before_relay": True,
            "selection_rank_used": False,
            "residual_relay_pass": bool(claims["residual_relay_pass"]),
        },
    )


if __name__ == "__main__":
    main()
