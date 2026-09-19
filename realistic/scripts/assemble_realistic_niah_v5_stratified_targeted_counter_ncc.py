#!/usr/bin/env python3
"""Assemble four timing-specific NCC analyses without pooling raw margins."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED = (
    ("Qwen3-8B", "rank_after_city", "qwen_after"),
    ("Qwen3-8B", "rank_before_city", "qwen_before"),
    ("Gemma4-E4B", "rank_after_city", "gemma_after"),
    ("Gemma4-E4B", "rank_before_city", "gemma_before"),
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load(path: Path) -> dict[str, Any]:
    active = path / "claim_gates.json" if path.is_dir() else path
    return json.loads(active.read_text(encoding="utf-8"))


def _condition(result: dict[str, Any], name: str) -> dict[str, Any]:
    rows = result["primary_endpoint_result"]["condition_metrics"]
    matched = [row for row in rows if str(row["condition"]) == name]
    if len(matched) != 1:
        raise ValueError(f"Missing unique condition {name}")
    return matched[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for _model, _timing, argument in EXPECTED:
        parser.add_argument(f"--{argument.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    branch_rows: list[dict[str, Any]] = []
    full_results: dict[str, dict[str, Any]] = {}
    for model, timing, argument in EXPECTED:
        result = _load(getattr(args, argument))
        if (
            result.get("status") != "PASS"
            or str(result.get("model_label")) != model
            or str(result.get("timing_branch")) != timing
        ):
            raise ValueError(f"Analysis mismatch for {model} {timing}")
        primary = result["primary_endpoint_result"]
        raw = primary["raw_primary_estimand"]
        raw_specificity = primary["raw_specificity_estimand"]
        standardized = primary["standardized_primary_estimand"]
        standardized_specificity = primary["standardized_specificity_estimand"]
        clean = _condition(result, "clean")
        selected = _condition(result, "selected_mask")
        key = f"{model}:{timing}"
        full_results[key] = result
        branch_rows.append(
            {
                "model_label": model,
                "timing_branch": timing,
                "primary_endpoint": result["primary_endpoint"],
                "development_seed_count": result["development_seed_count"],
                "confirmation_seed_count": result["confirmation_seed_count"],
                "selected_layer": primary["selected_layer"],
                "discovery_oof_ncc_balanced_accuracy": primary[
                    "selected_layer_discovery_metrics"
                ]["grouped_oof_ncc_balanced_accuracy"],
                "clean_confirmation_exact_accuracy": clean["exact_accuracy"],
                "clean_confirmation_balanced_accuracy": primary[
                    "readout_validity"
                ]["clean_confirmation_balanced_accuracy"],
                "clean_confirmation_mean_correct_centroid_margin": primary[
                    "readout_validity"
                ]["clean_confirmation_mean_correct_centroid_margin"],
                "readout_validity_pass": primary["readout_validity"]["pass"],
                "selected_confirmation_exact_accuracy": selected["exact_accuracy"],
                "raw_selected_margin_loss": raw["mean_effect"],
                "raw_selected_margin_loss_ci_low": raw["ci_low"],
                "raw_selected_margin_loss_ci_high": raw["ci_high"],
                "raw_selected_vs_random_specificity": raw_specificity["mean_effect"],
                "standardized_selected_margin_loss": standardized["mean_effect"],
                "standardized_selected_margin_loss_ci_low": standardized["ci_low"],
                "standardized_selected_margin_loss_ci_high": standardized["ci_high"],
                "standardized_selected_vs_random_specificity": standardized_specificity[
                    "mean_effect"
                ],
                "standardized_specificity_ci_low": standardized_specificity["ci_low"],
                "standardized_specificity_ci_high": standardized_specificity["ci_high"],
                "directional": primary["selected_mask_changes_ncc_directionally"],
                "more_damaging_than_random": primary[
                    "selected_mask_more_damaging_than_random"
                ],
                "effect_status": primary["ncc_effect_status"],
            }
        )

    branches = pd.DataFrame(branch_rows)
    model_synthesis: dict[str, dict[str, Any]] = {}
    for model, active in branches.groupby("model_label", sort=False):
        active = active.sort_values("timing_branch")
        directional_specific = (
            active["directional"].astype(bool)
            & active["more_damaging_than_random"].astype(bool)
            & active["readout_validity_pass"].astype(bool)
        )
        supported = int(directional_specific.sum())
        if supported == 2:
            status = "BOTH_TIMING_BRANCHES_DIRECTIONAL_SPECIFIC"
        elif supported == 1:
            status = "TIMING_ASYMMETRIC_DIRECTIONAL_SPECIFIC"
        else:
            status = "NEITHER_TIMING_BRANCH_DIRECTIONAL_SPECIFIC"
        model_synthesis[str(model)] = {
            "status": status,
            "supported_branch_count": supported,
            "branch_count": 2,
            "equal_branch_descriptive_mean_standardized_margin_loss": float(
                active["standardized_selected_margin_loss"].mean()
            ),
            "equal_branch_descriptive_mean_standardized_specificity": float(
                active["standardized_selected_vs_random_specificity"].mean()
            ),
            "warning": (
                "The equal-branch means are descriptive summaries of distinct, "
                "unpaired estimands; no pooled inferential test is attached."
            ),
        }

    timing_correspondence: dict[str, dict[str, Any]] = {}
    for timing, active in branches.groupby("timing_branch", sort=False):
        active = active.sort_values("model_label")
        signs = [
            bool(row.directional)
            and bool(row.more_damaging_than_random)
            and bool(row.readout_validity_pass)
            for row in active.itertuples(index=False)
        ]
        timing_correspondence[str(timing)] = {
            "models_agree_on_directional_specific_gate": len(set(signs)) == 1,
            "both_models_directional_specific": all(signs),
            "neither_model_directional_specific": not any(signs),
        }

    synthesis = {
        "schema_version": "realistic_niah_v5_stratified_targeted_counter_ncc_synthesis_v2",
        "status": "PASS",
        "primary_reporting_unit": "model_x_timing_branch",
        "raw_margins_pooled": False,
        "standardized_effects_are_discovery_oof_margin_sd_units": True,
        "directional_specific_support_requires_valid_clean_readout": True,
        "branches_use_unpaired_maximal_eligible_fixed_phase_cohorts": True,
        "model_synthesis": model_synthesis,
        "cross_model_timing_correspondence": timing_correspondence,
        "branch_results": {
            key: {
                "primary_endpoint": value["primary_endpoint"],
                "primary_endpoint_result": value["primary_endpoint_result"],
                "development_seeds": value["development_seeds"],
                "confirmation_seeds": value["confirmation_seeds"],
            }
            for key, value in full_results.items()
        },
        "reporting_rule": (
            "Report the four branch rows first; use equal-branch standardized "
            "means only as descriptive synthesis and never pool raw margins."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    branches.to_csv(args.output / "branch_summary.csv", index=False)
    _atomic_json(args.output / "stratified_ncc_synthesis.json", synthesis)
    print(json.dumps({"status": "PASS", "branch_count": len(branches)}))


if __name__ == "__main__":
    main()
