#!/usr/bin/env python3
"""Analyze whether answer-source masking amplifies or occludes donor patches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


SELF = "self_patch"
DONOR = "full_donor_patch"
SOURCE_FAMILIES = {
    "trace_items": (
        "block_trace_items",
        "block_trace_items_matched_control",
    ),
    "prompt_records": (
        "block_prompt_records",
        "block_prompt_records_matched_control",
    ),
}
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
    rows = []
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


def full_state_patch_source_effects(trials: pd.DataFrame) -> pd.DataFrame:
    ok = trials.loc[trials["status"].eq("ok")].copy()
    required_masks = {"clean"}
    for treatment, control in SOURCE_FAMILIES.values():
        required_masks.update([treatment, control])
    if not {SELF, DONOR} <= set(ok["patch_condition"].astype(str)):
        raise ValueError("Missing self/full-donor patch arms")
    if not required_masks <= set(ok["mask_condition"].astype(str)):
        raise ValueError("Missing registered source-mask arms")
    identity = [
        "model_label",
        "seed",
        "request_id",
        "pair_sha256",
        "gold_count",
        "receiver_occurrence",
        "donor_occurrence",
        "donor_offset",
        "patch_geometry",
        "patch_layer_mode",
        "layer",
    ]
    identity.extend(
        column
        for column in (
            "selection_cell_id",
            "selection_rank",
            "within_cell_index",
            "selection_policy",
            "mechanism_split",
            "mask_scope",
        )
        if column in ok.columns
    )
    outcomes = [value for value in HIGHER_IS_BETTER if value in ok.columns]
    outcomes.append("expected_count")
    prediction_field = next(
        (value for value in ("prediction", "predicted_count") if value in ok.columns),
        None,
    )
    if prediction_field is not None:
        outcomes.append(prediction_field)
    duplicates = ok.duplicated(
        identity + ["patch_condition", "mask_condition"], keep=False
    )
    if duplicates.any():
        raise ValueError("Duplicate patch/mask cells exist within a pair")
    wide = ok.pivot(
        index=identity,
        columns=["patch_condition", "mask_condition"],
        values=outcomes,
    )
    expected_cells = {
        (patch, mask) for patch in (SELF, DONOR) for mask in required_masks
    }
    observed_cells = set(
        zip(wide.columns.get_level_values(1), wide.columns.get_level_values(2))
    )
    if not expected_cells <= observed_cells:
        raise ValueError(f"Missing cells: {sorted(expected_cells - observed_cells)}")
    output = wide.index.to_frame(index=False)

    def value(outcome: str, patch: str, mask: str) -> np.ndarray:
        return pd.to_numeric(
            wide[(outcome, patch, mask)], errors="coerce"
        ).to_numpy(dtype=float)

    for outcome in HIGHER_IS_BETTER:
        if outcome not in outcomes:
            continue
        output[f"{outcome}__patch_damage_clean"] = (
            value(outcome, SELF, "clean") - value(outcome, DONOR, "clean")
        )
        for family, (treatment, control) in SOURCE_FAMILIES.items():
            damage_treatment = value(outcome, SELF, treatment) - value(
                outcome, DONOR, treatment
            )
            damage_control = value(outcome, SELF, control) - value(
                outcome, DONOR, control
            )
            output[f"{outcome}__{family}__patch_damage_true_mask"] = damage_treatment
            output[f"{outcome}__{family}__patch_damage_matched_mask"] = damage_control
            output[f"{outcome}__{family}__specific_interaction"] = (
                damage_treatment - damage_control
            )
    donor_offset = output["donor_offset"].to_numpy(dtype=float)
    for outcome in ["expected_count"] + (
        [prediction_field] if prediction_field is not None else []
    ):
        output[f"{outcome}__shift_clean"] = value(
            outcome, DONOR, "clean"
        ) - value(outcome, SELF, "clean")
        output[f"{outcome}__adoption_clean"] = (
            output[f"{outcome}__shift_clean"].to_numpy(dtype=float) / donor_offset
        )
        for family, (treatment, control) in SOURCE_FAMILIES.items():
            shift_treatment = value(outcome, DONOR, treatment) - value(
                outcome, SELF, treatment
            )
            shift_control = value(outcome, DONOR, control) - value(
                outcome, SELF, control
            )
            adoption_treatment = shift_treatment / donor_offset
            adoption_control = shift_control / donor_offset
            output[f"{outcome}__{family}__adoption_true_mask"] = adoption_treatment
            output[f"{outcome}__{family}__adoption_matched_mask"] = adoption_control
            output[f"{outcome}__{family}__adoption_interaction"] = (
                adoption_treatment - adoption_control
            )
    return output


def _bootstrap(values: np.ndarray, *, samples: int, seed: int) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(finite, (int(samples), len(finite)), replace=True).mean(axis=1)
    return float(finite.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize_effects(
    effects: pd.DataFrame, *, bootstrap_samples: int, random_seed: int
) -> pd.DataFrame:
    metrics = [
        column
        for column in effects.columns
        if "__" in column and pd.api.types.is_numeric_dtype(effects[column])
    ]
    rows = []
    groupings = [("overall", effects)] + [
        (f"offset={int(offset):+d}", group)
        for offset, group in effects.groupby("donor_offset")
    ]
    counter = 0
    for grouping, frame in groupings:
        for metric in metrics:
            active = frame[["seed", metric]].copy()
            active[metric] = pd.to_numeric(active[metric], errors="coerce")
            active = active.loc[np.isfinite(active[metric])]
            if active.empty:
                continue
            seed_values = active.groupby("seed", as_index=False)[metric].mean()
            mean, low, high = _bootstrap(
                seed_values[metric].to_numpy(),
                samples=bootstrap_samples,
                seed=int(random_seed) + counter,
            )
            counter += 1
            parts = metric.split("__")
            rows.append(
                {
                    "grouping": grouping,
                    "outcome": parts[0],
                    "source_family": parts[1] if len(parts) == 3 else "none",
                    "estimand": parts[-1],
                    "mean_seed_equal": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "mean_pair": float(active[metric].mean()),
                    "median_pair": float(active[metric].median()),
                    "pair_count": len(active),
                    "seed_count": seed_values["seed"].nunique(),
                }
            )
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260820)
    args = parser.parse_args(argv)
    trials = _read_shards(args.trials)
    effects = full_state_patch_source_effects(trials)
    summary = summarize_effects(
        effects,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
    )
    output = args.output.resolve()
    effects_path = output / "pair_effects.csv"
    summary_path = output / "seed_equal_summary.csv"
    _atomic_csv(effects_path, effects)
    _atomic_csv(summary_path, summary)
    ok = trials.loc[trials["status"].eq("ok")]
    per_shard_cells = ok.groupby(["pair_sha256", "patch_geometry"]).size()
    audit = {
        "status": (
            "PASS"
            if len(per_shard_cells) > 0 and (per_shard_cells == 10).all()
            else "FAIL"
        ),
        "trial_rows": len(trials),
        "ok_rows": len(ok),
        "not_applicable_rows": int(trials["status"].eq("not_applicable").sum()),
        "eligible_pair_geometries": len(per_shard_cells),
        "pair_count": effects["pair_sha256"].nunique(),
        "seed_count": effects["seed"].nunique(),
        "rows_per_eligible_pair_geometry_min": (
            int(per_shard_cells.min()) if len(per_shard_cells) else 0
        ),
        "rows_per_eligible_pair_geometry_max": (
            int(per_shard_cells.max()) if len(per_shard_cells) else 0
        ),
        "interpretation": {
            "positive_prompt_adoption_interaction": (
                "prompt masking exposes a donor state normally corrected by an "
                "independent prompt recount"
            ),
            "negative_trace_adoption_interaction": (
                "trace masking blocks readout of the patched trace state"
            ),
        },
    }
    _atomic_json(output / "audit.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError("Full-state/source audit failed")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
