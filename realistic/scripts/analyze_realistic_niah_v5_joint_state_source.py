#!/usr/bin/env python3
"""Audit and summarize the count-state x answer-source causal factorial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


STATE_CLEAN = "clean"
STATE_ALIGNED = "aligned_running_state_removal"
STATE_CONTROL = "norm_matched_orthogonal_removal"
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


def _bootstrap_seed_mean(
    values: np.ndarray, *, samples: int, random_seed: int
) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(int(random_seed))
    draws = rng.choice(
        finite, size=(int(samples), len(finite)), replace=True
    ).mean(axis=1)
    return (
        float(np.mean(finite)),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def factorial_request_effects(trials: pd.DataFrame) -> pd.DataFrame:
    ok = trials.loc[trials["status"].eq("ok")].copy()
    required_states = {STATE_CLEAN, STATE_ALIGNED, STATE_CONTROL}
    required_masks = {"clean"}
    for treatment, control in SOURCE_FAMILIES.values():
        required_masks.update([treatment, control])
    observed_states = set(ok["state_condition"].astype(str))
    observed_masks = set(ok["mask_condition"].astype(str))
    if not required_states <= observed_states:
        raise ValueError(f"Missing state arms: {required_states - observed_states}")
    if not required_masks <= observed_masks:
        raise ValueError(f"Missing mask arms: {required_masks - observed_masks}")
    identity = [
        "model_label",
        "seed",
        "request_id",
        "gold_count",
        "dataset_split",
        "state_source_layer",
        "state_source_scope",
    ]
    outcomes = [value for value in HIGHER_IS_BETTER if value in ok.columns]
    if not outcomes:
        raise ValueError("No registered higher-is-better outcomes are present")
    duplicates = ok.duplicated(
        identity + ["state_condition", "mask_condition"], keep=False
    )
    if duplicates.any():
        raise ValueError("Duplicate state/mask cells exist within a request")
    wide = ok.pivot(
        index=identity,
        columns=["state_condition", "mask_condition"],
        values=outcomes,
    )
    expected_cells = {
        (state, mask)
        for state in required_states
        for mask in required_masks
    }
    observed_cells = set(zip(wide.columns.get_level_values(1), wide.columns.get_level_values(2)))
    missing_cells = expected_cells - observed_cells
    if missing_cells:
        raise ValueError(f"The factorial is missing cells: {sorted(missing_cells)}")
    output = wide.index.to_frame(index=False)
    for metric in outcomes:
        value = lambda state, mask: pd.to_numeric(  # noqa: E731
            wide[(metric, state, mask)], errors="coerce"
        ).to_numpy(dtype=float)
        clean = value(STATE_CLEAN, "clean")
        aligned_clean = value(STATE_ALIGNED, "clean")
        control_clean = value(STATE_CONTROL, "clean")
        output[f"{metric}__raw_state_damage"] = clean - aligned_clean
        output[f"{metric}__orthogonal_state_damage"] = clean - control_clean
        output[f"{metric}__specific_state_damage"] = control_clean - aligned_clean
        for family, (treatment, control) in SOURCE_FAMILIES.items():
            aligned_treatment = value(STATE_ALIGNED, treatment)
            aligned_control = value(STATE_ALIGNED, control)
            orthogonal_treatment = value(STATE_CONTROL, treatment)
            orthogonal_control = value(STATE_CONTROL, control)
            output[f"{metric}__{family}__mask_damage_aligned"] = (
                aligned_control - aligned_treatment
            )
            output[f"{metric}__{family}__mask_damage_orthogonal"] = (
                orthogonal_control - orthogonal_treatment
            )
            output[f"{metric}__{family}__state_damage_true_mask"] = (
                orthogonal_treatment - aligned_treatment
            )
            output[f"{metric}__{family}__state_damage_matched_mask"] = (
                orthogonal_control - aligned_control
            )
            output[f"{metric}__{family}__specific_interaction"] = (
                (orthogonal_treatment - aligned_treatment)
                - (orthogonal_control - aligned_control)
            )
    return output


def seed_equal_summary(
    effects: pd.DataFrame, *, bootstrap_samples: int, random_seed: int
) -> pd.DataFrame:
    metric_columns = [
        column
        for column in effects.columns
        if "__" in column and pd.api.types.is_numeric_dtype(effects[column])
    ]
    rows = []
    for index, column in enumerate(metric_columns):
        finite = pd.to_numeric(effects[column], errors="coerce")
        active = effects.loc[np.isfinite(finite), ["seed"]].copy()
        active["effect"] = finite.loc[active.index]
        if active.empty:
            continue
        by_seed = active.groupby("seed", as_index=False)["effect"].mean()
        mean, low, high = _bootstrap_seed_mean(
            by_seed["effect"].to_numpy(),
            samples=bootstrap_samples,
            random_seed=int(random_seed) + index,
        )
        parts = column.split("__")
        rows.append(
            {
                "outcome": parts[0],
                "source_family": parts[1] if len(parts) == 3 else "none",
                "estimand": parts[-1],
                "mean_seed_equal": mean,
                "ci_low": low,
                "ci_high": high,
                "mean_request": float(active["effect"].mean()),
                "median_request": float(active["effect"].median()),
                "request_count": len(active),
                "seed_count": by_seed["seed"].nunique(),
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
    effects = factorial_request_effects(trials)
    summary = seed_equal_summary(
        effects,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
    )
    output = args.output.resolve()
    effects_path = output / "request_effects.csv"
    summary_path = output / "seed_equal_summary.csv"
    _atomic_csv(effects_path, effects)
    _atomic_csv(summary_path, summary)
    expected_rows_per_request = 3 * 5
    per_request = trials.groupby("request_id").size()
    hook_failures = trials.loc[
        trials["state_condition"].ne(STATE_CLEAN)
        & pd.to_numeric(
            trials["state_intervention_hook_applications"], errors="coerce"
        ).ne(1)
    ]
    audit = {
        "status": "PASS"
        if (per_request == expected_rows_per_request).all() and hook_failures.empty
        else "FAIL",
        "trial_rows": len(trials),
        "request_count": trials["request_id"].nunique(),
        "seed_count": trials["seed"].nunique(),
        "expected_rows_per_request": expected_rows_per_request,
        "rows_per_request_min": int(per_request.min()),
        "rows_per_request_max": int(per_request.max()),
        "state_hook_failure_rows": len(hook_failures),
        "artifacts": {
            "request_effects": effects_path.name,
            "seed_equal_summary": summary_path.name,
        },
        "interaction_interpretation": {
            "positive_prompt_records": (
                "prompt recount and stored state are complementary/redundant: "
                "prompt masking amplifies count-state removal"
            ),
            "negative_trace_items": (
                "trace attention mediates readout of the stored state: blocking "
                "trace sources occludes count-state removal"
            ),
        },
    }
    _atomic_json(output / "audit.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError("Joint-factorial audit failed")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
