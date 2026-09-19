#!/usr/bin/env python3
"""Analyze free-greedy integer marker-scrubbed restoration trials."""

from __future__ import annotations

import argparse
import hashlib
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
    holm_adjust,
    sign_flip_pvalue,
)


CONDITIONS = {
    "source_reference",
    "blank_reference",
    "source_list_item_k_to_blank_restoration",
}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read(root: Path, *, phase: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((root / "frozen_trial_plan.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or str(plan.get("phase")) != str(phase):
        raise ValueError("Greedy trial manifest/plan is not sealed")
    files = sorted((root / "shards").glob("*.jsonl"))
    if len(files) != int(plan["seed_count"]):
        raise ValueError("Greedy shard count differs from plan")
    frame = pd.DataFrame(
        [
            json.loads(line)
            for path in files
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    )
    if set(frame["condition"].astype(str)) != CONDITIONS:
        raise ValueError("Greedy conditions changed")
    if set(frame["experiment_id"].astype(str)) != {
        "marker_scrubbed_list_greedy_integer_restoration"
    }:
        raise ValueError("Greedy experiment id changed")
    if frame["candidate_scoring_used"].astype(bool).any():
        raise ValueError("Corrected greedy rerun used candidate scoring")
    if frame["diagnostic_suffix_used"].astype(bool).any():
        raise ValueError("Corrected greedy rerun used a diagnostic suffix")
    expected_seeds = {int(value["seed"]) for value in plan["rows"]}
    if set(frame["seed"].astype(int)) != expected_seeds:
        raise ValueError("Greedy seeds differ from plan")
    return frame, plan


def _derive(frame: pd.DataFrame) -> pd.DataFrame:
    layers = sorted(
        frame.loc[frame["source_layer"].astype(int).ge(0), "source_layer"]
        .astype(int)
        .unique()
    )
    rows: list[dict[str, Any]] = []
    for (seed, target), active in frame.groupby(["seed", "target_occurrence"], sort=True):
        source_rows = active[active["condition"].eq("source_reference")]
        blank_rows = active[active["condition"].eq("blank_reference")]
        if len(source_rows) != 1 or len(blank_rows) != 1:
            raise ValueError("Every seed/k needs one Source and Blank")
        source = source_rows.iloc[0]
        blank = blank_rows.iloc[0]
        for layer in layers:
            restored_rows = active[
                active["condition"].eq("source_list_item_k_to_blank_restoration")
                & active["source_layer"].astype(int).eq(int(layer))
            ]
            if len(restored_rows) != 1:
                raise ValueError("Every seed/k/layer needs one restoration")
            restored = restored_rows.iloc[0]
            rows.append(
                {
                    "seed": int(seed),
                    "target_occurrence": int(target),
                    "source_layer": int(layer),
                    "source_prediction": source["greedy_prediction"],
                    "blank_prediction": blank["greedy_prediction"],
                    "restored_prediction": restored["greedy_prediction"],
                    "source_exact": float(source["greedy_running_exact"]),
                    "blank_exact": float(blank["greedy_running_exact"]),
                    "restored_exact": float(restored["greedy_running_exact"]),
                    "restoration_exact_gain": float(
                        float(restored["greedy_running_exact"])
                        - float(blank["greedy_running_exact"])
                    ),
                    "source_valid_integer": float(source["greedy_output_is_integer_1_to_10"]),
                    "blank_valid_integer": float(blank["greedy_output_is_integer_1_to_10"]),
                    "restored_valid_integer": float(
                        restored["greedy_output_is_integer_1_to_10"]
                    ),
                    "source_completion": str(source["completion_text_raw"]),
                    "blank_completion": str(blank["completion_text_raw"]),
                    "restored_completion": str(restored["completion_text_raw"]),
                }
            )
    derived = pd.DataFrame(rows)
    expected = (
        derived["seed"].nunique()
        * derived["target_occurrence"].nunique()
        * derived["source_layer"].nunique()
    )
    if len(derived) != expected or derived["target_occurrence"].nunique() != 10:
        raise ValueError("Greedy derived panel is incomplete")
    return derived


def _layer_metrics(derived: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_layer = derived.groupby(["seed", "source_layer"], as_index=False).agg(
        source_exact=("source_exact", "mean"),
        blank_exact=("blank_exact", "mean"),
        restored_exact=("restored_exact", "mean"),
        exact_gain=("restoration_exact_gain", "mean"),
        source_valid=("source_valid_integer", "mean"),
        blank_valid=("blank_valid_integer", "mean"),
        restored_valid=("restored_valid_integer", "mean"),
    )
    layer = seed_layer.groupby("source_layer", as_index=False).agg(
        seed_count=("seed", "nunique"),
        source_exact_accuracy=("source_exact", "mean"),
        blank_exact_accuracy=("blank_exact", "mean"),
        restored_exact_accuracy=("restored_exact", "mean"),
        mean_restoration_exact_gain=("exact_gain", "mean"),
        source_valid_integer_rate=("source_valid", "mean"),
        blank_valid_integer_rate=("blank_valid", "mean"),
        restored_valid_integer_rate=("restored_valid", "mean"),
    )
    return seed_layer, layer.sort_values("source_layer").reset_index(drop=True)


def _freeze_top_three(layer: pd.DataFrame, *, plan: dict[str, Any]) -> dict[str, Any]:
    ranked = layer.sort_values(
        ["mean_restoration_exact_gain", "restored_exact_accuracy", "source_layer"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    top = ranked.iloc[:3]
    return {
        "schema_version": "realistic_niah_v5_greedy_frozen_layers_v1",
        "status": "FROZEN_FROM_GREEDY_DISCOVERY",
        "model_label": str(plan["model_label"]),
        "source_layers": [int(value) for value in top["source_layer"]],
        "selection_primary": "seed_equal_mean_greedy_exact_gain",
        "tie_break_1": "descending_restored_greedy_exact",
        "tie_break_2": "ascending_layer_index",
        "discovery_plan_sha256": hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "ranked_layers": ranked.to_dict(orient="records"),
        "confirmation_results_accessed": False,
    }


def _confirmation(
    derived: pd.DataFrame, *, frozen_layers: tuple[int, ...]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    seed_layer = derived.groupby(["seed", "source_layer"], as_index=False).agg(
        exact_gain=("restoration_exact_gain", "mean"),
        source_exact=("source_exact", "mean"),
        blank_exact=("blank_exact", "mean"),
        restored_exact=("restored_exact", "mean"),
    )
    layer_rows = []
    raw_p = []
    for layer in frozen_layers:
        active = seed_layer[seed_layer["source_layer"].astype(int).eq(int(layer))]
        values = active["exact_gain"].to_numpy(float)
        ci = bootstrap_seed_mean_ci(values, samples=10_000, seed=20260825 + int(layer))
        p_value = sign_flip_pvalue(values)
        raw_p.append(p_value)
        layer_rows.append(
            {
                "source_layer": int(layer),
                "seed_count": len(values),
                "mean_exact_gain": float(ci["mean_effect"]),
                "bootstrap_ci_low": float(ci["ci_low"]),
                "bootstrap_ci_high": float(ci["ci_high"]),
                "two_sided_sign_flip_p": float(p_value),
                "source_exact_accuracy": float(active["source_exact"].mean()),
                "blank_exact_accuracy": float(active["blank_exact"].mean()),
                "restored_exact_accuracy": float(active["restored_exact"].mean()),
                "exact_gap_closure": (
                    float(
                        (active["restored_exact"].mean() - active["blank_exact"].mean())
                        / (active["source_exact"].mean() - active["blank_exact"].mean())
                    )
                    if active["source_exact"].mean()
                    != active["blank_exact"].mean()
                    else None
                ),
            }
        )
    for row, adjusted in zip(layer_rows, holm_adjust(raw_p)):
        row["holm_p_across_three_layers"] = float(adjusted)
        row["positive_support"] = bool(
            float(row["bootstrap_ci_low"]) > 0 and float(adjusted) < 0.05
        )
    per_layer = pd.DataFrame(layer_rows)
    return per_layer, seed_layer, {
        "primary_estimand": (
            "separately for each frozen layer: within each seed, mean over "
            "k=1..10 of I[restored greedy integer=k]-I[blank greedy integer=k]"
        ),
        "statistical_unit": "seed",
        "layer_results_are_not_averaged": True,
        "multiplicity_control": "Holm correction across the three frozen layers",
        "layers": per_layer.to_dict(orient="records"),
    }


def _prediction_proportions(derived: pd.DataFrame) -> pd.DataFrame:
    def append_distribution(
        records: list[dict[str, Any]],
        *,
        condition: str,
        layer: int,
        values: pd.Series,
    ) -> None:
        numeric = pd.to_numeric(values, errors="coerce")
        valid = numeric.between(1, 10, inclusive="both")
        for prediction in range(1, 11):
            count = int(numeric.eq(prediction).sum())
            records.append(
                {
                    "condition": condition,
                    "source_layer": int(layer),
                    "predicted_integer": prediction,
                    "label": str(prediction),
                    "count": count,
                    "proportion": count / len(numeric),
                }
            )
        invalid_count = int((~valid).sum())
        records.append(
            {
                "condition": condition,
                "source_layer": int(layer),
                "predicted_integer": None,
                "label": "invalid_or_outside_1_to_10",
                "count": invalid_count,
                "proportion": invalid_count / len(numeric),
            }
        )

    references = derived.drop_duplicates(["seed", "target_occurrence"])
    records: list[dict[str, Any]] = []
    for condition, frame, column, layer in (
        ("source_reference", references, "source_prediction", -1),
        ("blank_reference", references, "blank_prediction", -1),
    ):
        append_distribution(
            records,
            condition=condition,
            layer=layer,
            values=frame[column],
        )
    for layer, frame in derived.groupby("source_layer", sort=True):
        append_distribution(
            records,
            condition="source_list_item_k_to_blank_restoration",
            layer=int(layer),
            values=frame["restored_prediction"],
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--phase", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--frozen-layers", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame, plan = _read(args.input, phase=str(args.phase))
    derived = _derive(frame)
    seed_layer, layer = _layer_metrics(derived)
    proportions = _prediction_proportions(derived)
    args.output.mkdir(parents=True, exist_ok=True)
    derived.to_csv(args.output / "seed_k_layer_metrics.csv", index=False)
    seed_layer.to_csv(args.output / "seed_layer_metrics.csv", index=False)
    layer.to_csv(args.output / "layer_metrics.csv", index=False)
    proportions.to_csv(args.output / "predicted_integer_proportions.csv", index=False)
    if args.phase == "discovery":
        frozen = _freeze_top_three(layer, plan=plan)
        _atomic_json(args.output / "frozen_layers.json", frozen)
        result = {
            "schema_version": "realistic_niah_v5_greedy_analysis_v1",
            "status": "PASS",
            "phase": "discovery",
            "model_label": str(plan["model_label"]),
            "seed_count": int(plan["seed_count"]),
            "frozen_source_layers": frozen["source_layers"],
            "selection_primary": "free_greedy_integer_exact_gain",
            "confirmation_results_accessed": False,
        }
    else:
        if args.frozen_layers is None:
            raise ValueError("Greedy confirmation requires frozen layers")
        frozen = json.loads(args.frozen_layers.read_text(encoding="utf-8"))
        frozen_layers = tuple(int(value) for value in frozen["source_layers"])
        per_layer, seed_layer_confirmation, primary = _confirmation(
            derived, frozen_layers=frozen_layers
        )
        per_layer.to_csv(args.output / "confirmation_per_layer.csv", index=False)
        seed_layer_confirmation.to_csv(
            args.output / "confirmation_seed_layer.csv", index=False
        )
        result = {
            "schema_version": "realistic_niah_v5_greedy_analysis_v1",
            "status": "PASS",
            "phase": "confirmation",
            "model_label": str(plan["model_label"]),
            "seed_count": int(plan["seed_count"]),
            "frozen_source_layers": list(frozen_layers),
            "primary_per_layer": primary,
            "formal_status": "corrected_readout_rerun_same_frozen_cohort",
            "independent_new_confirmation_claim_allowed": False,
        }
    _atomic_json(args.output / "analysis.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
