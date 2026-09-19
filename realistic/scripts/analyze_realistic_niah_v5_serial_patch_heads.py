#!/usr/bin/env python3
"""Analyze the terminal-state -> frozen-heads -> answer serial factorial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


SELF = "self_patch"
DONOR = "full_donor_patch"
CLEAN = "clean"
SELECTED = "selected_bank"
RANDOM = "layer_matched_random"
RELATIVE_EQUIVALENCE_BOUND = 0.20
HIGHER_IS_BETTER = (
    "correct_count_margin",
    "correct_count_probability",
    "correct_count_log_score",
    "expected_count_utility",
    "exact_count",
    "strict_count_utility",
)


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


def serial_patch_head_effects(trials: pd.DataFrame) -> pd.DataFrame:
    """Return one paired causal-effect row per terminal donor/receiver pair."""

    required = {
        "pair_sha256",
        "request_id",
        "model_label",
        "seed",
        "gold_count",
        "donor_offset",
        "patch_condition",
        "head_condition",
        "head_repeat",
        "status",
        "correct_count_margin",
        "expected_count",
    }
    missing = sorted(required - set(trials.columns))
    if missing:
        raise ValueError(f"Serial head trials are missing {missing}")
    bad = trials.loc[~trials["status"].astype(str).eq("ok")]
    if not bad.empty:
        raise ValueError(
            "Serial confirmation does not permit post-hoc exclusion of "
            f"not-applicable/error pairs: rows={len(bad)}"
        )
    if set(trials["patch_condition"].astype(str)) != {SELF, DONOR}:
        raise ValueError("Serial trials have the wrong patch arms")
    if set(trials["head_condition"].astype(str)) != {
        CLEAN,
        SELECTED,
        RANDOM,
    }:
        raise ValueError("Serial trials have the wrong head-readout arms")

    metadata_columns = [
        "pair_sha256",
        "request_id",
        "model_label",
        "seed",
        "gold_count",
        "donor_offset",
        "receiver_occurrence",
        "donor_occurrence",
        "mechanism_split",
    ]
    outcome_columns = [
        value for value in HIGHER_IS_BETTER if value in trials.columns
    ]
    rows: list[dict[str, Any]] = []
    random_repeat_counts: set[int] = set()
    for pair_sha, group in trials.groupby("pair_sha256", sort=False):
        metadata = {
            column: group[column].iloc[0]
            for column in metadata_columns
            if column in group.columns
        }
        for column, value in metadata.items():
            if group[column].astype(str).nunique() != 1:
                raise ValueError(f"Pair {pair_sha} mixes metadata column {column}")
        random_repeats = sorted(
            int(value)
            for value in group.loc[
                group["head_condition"].astype(str).eq(RANDOM), "head_repeat"
            ].unique()
        )
        if not random_repeats:
            raise ValueError(f"Pair {pair_sha} has no matched-random repeats")
        random_repeat_counts.add(len(random_repeats))

        def scalar(outcome: str, patch: str, head: str) -> float:
            selected = group.loc[
                group["patch_condition"].astype(str).eq(patch)
                & group["head_condition"].astype(str).eq(head),
                outcome,
            ]
            expected_rows = len(random_repeats) if head == RANDOM else 1
            if len(selected) != expected_rows:
                raise ValueError(
                    f"Pair {pair_sha} {patch}/{head} has {len(selected)} rows, "
                    f"expected {expected_rows}"
                )
            values = pd.to_numeric(selected, errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Pair {pair_sha} {outcome} is non-finite")
            return float(values.mean())

        result: dict[str, Any] = {**metadata}
        offset = float(metadata["donor_offset"])
        if offset == 0:
            raise ValueError("Serial donor offset cannot be zero")
        for outcome in outcome_columns:
            self_clean = scalar(outcome, SELF, CLEAN)
            donor_clean = scalar(outcome, DONOR, CLEAN)
            self_selected = scalar(outcome, SELF, SELECTED)
            donor_selected = scalar(outcome, DONOR, SELECTED)
            self_random = scalar(outcome, SELF, RANDOM)
            donor_random = scalar(outcome, DONOR, RANDOM)
            clean_damage = self_clean - donor_clean
            selected_damage = self_selected - donor_selected
            random_damage = self_random - donor_random
            prefix = f"{outcome}__"
            result[prefix + "patch_damage_clean"] = clean_damage
            result[prefix + "patch_damage_selected"] = selected_damage
            result[prefix + "patch_damage_random"] = random_damage
            result[prefix + "specific_head_mediation"] = (
                random_damage - selected_damage
            )
            result[prefix + "target_attenuation_vs_clean"] = (
                clean_damage - selected_damage
            )
            result[prefix + "random_change_vs_clean"] = random_damage - clean_damage
            result[prefix + "selected_head_main_damage_self"] = (
                self_clean - self_selected
            )
            result[prefix + "random_head_main_damage_self"] = (
                self_clean - self_random
            )
            result[prefix + "selected_head_specific_main_damage"] = (
                self_random - self_selected
            )

        self_expected_clean = scalar("expected_count", SELF, CLEAN)
        donor_expected_clean = scalar("expected_count", DONOR, CLEAN)
        self_expected_selected = scalar("expected_count", SELF, SELECTED)
        donor_expected_selected = scalar("expected_count", DONOR, SELECTED)
        self_expected_random = scalar("expected_count", SELF, RANDOM)
        donor_expected_random = scalar("expected_count", DONOR, RANDOM)
        adoption_clean = (donor_expected_clean - self_expected_clean) / offset
        adoption_selected = (
            donor_expected_selected - self_expected_selected
        ) / offset
        adoption_random = (donor_expected_random - self_expected_random) / offset
        result["expected_count__adoption_clean"] = adoption_clean
        result["expected_count__adoption_selected"] = adoption_selected
        result["expected_count__adoption_random"] = adoption_random
        result["expected_count__specific_head_mediation"] = (
            adoption_random - adoption_selected
        )
        result["expected_count__target_attenuation_vs_clean"] = (
            adoption_clean - adoption_selected
        )
        rows.append(result)
    if len(random_repeat_counts) != 1:
        raise ValueError("Pairs use different numbers of matched-random banks")
    return pd.DataFrame(rows)


def _bootstrap_summary(
    seed_effects: pd.DataFrame,
    *,
    samples: int,
    random_seed: int,
    grouping: str,
) -> pd.DataFrame:
    metric_columns = [
        column
        for column in seed_effects.columns
        if "__" in column and pd.api.types.is_numeric_dtype(seed_effects[column])
    ]
    if not metric_columns:
        raise ValueError("No numeric serial-effect metrics are available")
    values = seed_effects[metric_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Seed-level serial effects contain non-finite values")
    rng = np.random.default_rng(int(random_seed))
    indices = rng.integers(0, len(seed_effects), size=(int(samples), len(seed_effects)))
    draws = values[indices].mean(axis=1)
    rows: list[dict[str, Any]] = []
    for column_index, metric in enumerate(metric_columns):
        distribution = draws[:, column_index]
        rows.append(
            {
                "grouping": grouping,
                "metric": metric,
                "mean_seed_equal": float(values[:, column_index].mean()),
                "ci_low": float(np.quantile(distribution, 0.025)),
                "ci_high": float(np.quantile(distribution, 0.975)),
                "seed_count": int(len(seed_effects)),
            }
        )

    clean_name = "correct_count_margin__patch_damage_clean"
    selected_name = "correct_count_margin__patch_damage_selected"
    if clean_name not in metric_columns or selected_name not in metric_columns:
        raise ValueError("Correct-count patch damage is required for equivalence")
    clean_index = metric_columns.index(clean_name)
    selected_index = metric_columns.index(selected_name)
    clean_draw = draws[:, clean_index]
    selected_draw = draws[:, selected_index]
    valid = np.abs(clean_draw) > 1e-8
    if valid.mean() < 0.99:
        raise ValueError("Clean patch effect is too close to zero for a ratio gate")
    residual_ratio = np.abs(selected_draw[valid]) / np.abs(clean_draw[valid])
    point_clean = float(values[:, clean_index].mean())
    point_selected = float(values[:, selected_index].mean())
    point_ratio = abs(point_selected) / abs(point_clean)
    rows.append(
        {
            "grouping": grouping,
            "metric": "correct_count_margin__selected_residual_ratio",
            "mean_seed_equal": float(point_ratio),
            "ci_low": float(np.quantile(residual_ratio, 0.025)),
            "ci_high": float(np.quantile(residual_ratio, 0.975)),
            "seed_count": int(len(seed_effects)),
        }
    )
    attenuation = 1.0 - residual_ratio
    rows.append(
        {
            "grouping": grouping,
            "metric": "correct_count_margin__selected_attenuation_fraction",
            "mean_seed_equal": float(1.0 - point_ratio),
            "ci_low": float(np.quantile(attenuation, 0.025)),
            "ci_high": float(np.quantile(attenuation, 0.975)),
            "seed_count": int(len(seed_effects)),
        }
    )
    return pd.DataFrame(rows)


def summarize_serial_effects(
    effects: pd.DataFrame,
    *,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_metrics = [
        column
        for column in effects.columns
        if "__" in column and pd.api.types.is_numeric_dtype(effects[column])
    ]
    metadata = effects[["seed", "model_label", "mechanism_split"]].drop_duplicates(
        "seed"
    )
    seed_effects = effects.groupby("seed", as_index=False)[seed_metrics].mean()
    seed_effects = seed_effects.merge(metadata, on="seed", how="left", validate="one_to_one")
    summaries = [
        _bootstrap_summary(
            seed_effects,
            samples=bootstrap_samples,
            random_seed=random_seed,
            grouping="overall",
        )
    ]
    for offset_index, (offset, group) in enumerate(effects.groupby("donor_offset")):
        local = group.groupby("seed", as_index=False)[seed_metrics].mean()
        summaries.append(
            _bootstrap_summary(
                local,
                samples=bootstrap_samples,
                random_seed=random_seed + 1000 + offset_index,
                grouping=f"offset={int(offset):+d}",
            )
        )
    return seed_effects, pd.concat(summaries, ignore_index=True)


def _claim_gates(summary: pd.DataFrame, *, phase: str) -> dict[str, Any]:
    overall = summary.loc[summary["grouping"].eq("overall")].set_index("metric")

    def row(metric: str) -> dict[str, float]:
        if metric not in overall.index:
            raise ValueError(f"Serial summary is missing {metric}")
        value = overall.loc[metric]
        return {
            "estimate": float(value["mean_seed_equal"]),
            "ci_low": float(value["ci_low"]),
            "ci_high": float(value["ci_high"]),
        }

    storage = row("correct_count_margin__patch_damage_clean")
    random_patch = row("correct_count_margin__patch_damage_random")
    mediation = row("correct_count_margin__specific_head_mediation")
    residual = row("correct_count_margin__selected_residual_ratio")
    head_main = row("correct_count_margin__selected_head_specific_main_damage")
    adoption = row("expected_count__adoption_clean")
    adoption_mediation = row("expected_count__specific_head_mediation")
    gates = {
        "storage_main_effect": {
            **storage,
            "pass": storage["ci_low"] > 0,
            "rule": "ci_low > 0",
        },
        "random_bank_preserves_patch": {
            **random_patch,
            "pass": random_patch["ci_low"] > 0,
            "rule": "ci_low > 0",
        },
        "selected_bank_specific_mediation": {
            **mediation,
            "pass": mediation["ci_low"] > 0,
            "rule": "ci_low > 0",
        },
        "selected_bank_residual_equivalence": {
            **residual,
            "relative_equivalence_bound": RELATIVE_EQUIVALENCE_BOUND,
            "pass": residual["ci_high"] < RELATIVE_EQUIVALENCE_BOUND,
            "rule": "residual_ratio_ci_high < 0.20",
        },
        "selected_bank_normal_behavior_main_effect": {
            **head_main,
            "pass": head_main["ci_low"] > 0,
            "rule": "ci_low > 0",
            "role": "secondary",
        },
        "donor_count_adoption": {
            **adoption,
            "pass": adoption["ci_low"] > 0,
            "rule": "ci_low > 0",
            "role": "secondary",
        },
        "donor_adoption_specific_mediation": {
            **adoption_mediation,
            "pass": adoption_mediation["ci_low"] > 0,
            "rule": "ci_low > 0",
            "role": "secondary",
        },
    }
    primary_ids = (
        "storage_main_effect",
        "random_bank_preserves_patch",
        "selected_bank_specific_mediation",
        "selected_bank_residual_equivalence",
    )
    serial_pass = all(bool(gates[name]["pass"]) for name in primary_ids)
    return {
        "phase": phase,
        "confirmatory": phase == "confirmation",
        "primary_gate_ids": list(primary_ids),
        "serial_readout_pass": bool(serial_pass),
        "allowed_claim_if_confirmation_passes": (
            "A terminal trace count state causally affects the answer and is "
            "read out serially through the frozen answer-query trace-head bank."
        ),
        "restriction": (
            "This identifies the route from the patched trace state; it does "
            "not exclude a parallel prompt-recount pathway."
        ),
        "gates": gates,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["discovery", "confirmation"], required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260820)
    args = parser.parse_args(argv)

    trials = _read_shards(args.trials.resolve())
    expected_split = "development" if args.phase == "discovery" else "confirmation"
    observed_splits = set(trials["mechanism_split"].astype(str))
    if observed_splits != {expected_split}:
        raise ValueError(
            f"{args.phase} analysis received mechanism splits {observed_splits}"
        )
    effects = serial_patch_head_effects(trials)
    expected_seed_count = 20 if args.phase == "discovery" else 10
    observed_seeds = sorted(int(value) for value in effects["seed"].unique())
    if len(observed_seeds) != expected_seed_count:
        raise ValueError(
            f"{args.phase} requires {expected_seed_count} seeds, observed "
            f"{observed_seeds}"
        )
    seed_effects, summary = summarize_serial_effects(
        effects,
        bootstrap_samples=int(args.bootstrap_samples),
        random_seed=int(args.random_seed),
    )
    claims = _claim_gates(summary, phase=args.phase)
    output = args.output.resolve()
    _atomic_csv(output / "pair_effects.csv", effects)
    _atomic_csv(output / "seed_effects.csv", seed_effects)
    _atomic_csv(output / "seed_equal_summary.csv", summary)
    _atomic_json(output / "claim_gates.json", claims)
    _atomic_json(
        output / "audit.json",
        {
            "status": "PASS",
            "phase": args.phase,
            "trial_rows": int(len(trials)),
            "pair_count": int(effects["pair_sha256"].nunique()),
            "seed_count": int(len(observed_seeds)),
            "seeds": observed_seeds,
            "bootstrap_samples": int(args.bootstrap_samples),
            "relative_equivalence_bound": RELATIVE_EQUIVALENCE_BOUND,
            "serial_readout_pass": bool(claims["serial_readout_pass"]),
        },
    )


if __name__ == "__main__":
    main()
