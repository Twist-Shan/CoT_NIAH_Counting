#!/usr/bin/env python3
"""Assemble four direct count-logit margin branches without pooling them."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


BRANCHES = (
    ("Qwen3-8B", "rank_after_city"),
    ("Qwen3-8B", "rank_before_city"),
    ("Gemma4-E4B", "rank_after_city"),
    ("Gemma4-E4B", "rank_before_city"),
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    full: dict[str, Any] = {}
    for model, timing in BRANCHES:
        path = args.root / model / timing / "analysis" / "claim_gates.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        if (
            result.get("schema_version")
            != "realistic_niah_v5_targeted_counter_logit_margin_analysis_v1"
            or result.get("status") != "PASS"
            or str(result.get("model_label")) != model
            or str(result.get("timing_branch")) != timing
        ):
            raise ValueError(f"Logit-margin analysis mismatch: {path}")
        primary = result["primary_endpoint_result"]
        development = primary["development"]
        confirmation = primary["confirmation"]
        loss = confirmation["selected_margin_loss"]
        specificity = confirmation["selected_vs_random_specificity"]
        key = f"{model}:{timing}"
        full[key] = result
        rows.append(
            {
                "model_label": model,
                "timing_branch": timing,
                "development_seed_count": development["seed_count"],
                "confirmation_seed_count": confirmation["seed_count"],
                "development_clean_margin": development["clean_mean_margin"],
                "development_clean_accuracy": development["clean_accuracy"],
                "development_selected_margin_loss": development[
                    "selected_margin_loss"
                ]["mean_effect"],
                "development_specificity": development[
                    "selected_vs_random_specificity"
                ]["mean_effect"],
                "confirmation_clean_margin": confirmation["clean_mean_margin"],
                "confirmation_clean_accuracy": confirmation["clean_accuracy"],
                "readout_validity_pass": primary["readout_validity"]["pass"],
                "confirmation_selected_margin_loss": loss["mean_effect"],
                "selected_margin_loss_ci_low": loss["ci_low"],
                "selected_margin_loss_ci_high": loss["ci_high"],
                "selected_margin_loss_sign_flip_p": loss[
                    "p_value_two_sided_sign_flip"
                ],
                "confirmation_specificity": specificity["mean_effect"],
                "specificity_ci_low": specificity["ci_low"],
                "specificity_ci_high": specificity["ci_high"],
                "specificity_sign_flip_p": specificity[
                    "p_value_two_sided_sign_flip"
                ],
                "selected_loss_fraction_of_clean_margin": (
                    loss["mean_effect"] / confirmation["clean_mean_margin"]
                ),
                "specificity_fraction_of_clean_margin": (
                    specificity["mean_effect"] / confirmation["clean_mean_margin"]
                ),
                "effect_status": primary["effect_status"],
                "interval_confirmed": primary[
                    "bootstrap_interval_excludes_zero_for_both_gates"
                ],
            }
        )

    frame = pd.DataFrame(rows)
    gemma = frame.loc[frame["model_label"] == "Gemma4-E4B"]
    qwen = frame.loc[frame["model_label"] == "Qwen3-8B"]
    synthesis = {
        "schema_version": "realistic_niah_v5_targeted_counter_logit_margin_synthesis_v1",
        "status": "PASS",
        "primary_reporting_unit": "model_x_timing_branch",
        "raw_margins_pooled": False,
        "all_four_clean_readouts_valid": bool(frame["readout_validity_pass"].all()),
        "all_four_clean_candidate_accuracies_one": bool(
            (frame["confirmation_clean_accuracy"] == 1.0).all()
        ),
        "qwen_has_no_directional_specific_branch": bool(
            (qwen["effect_status"] == "NO_DIRECTIONAL_SPECIFIC_EVIDENCE").all()
        ),
        "gemma_both_branches_positive_in_discovery_and_confirmation": bool(
            (
                (gemma["development_selected_margin_loss"] > 0)
                & (gemma["development_specificity"] > 0)
                & (gemma["confirmation_selected_margin_loss"] > 0)
                & (gemma["confirmation_specificity"] > 0)
            ).all()
        ),
        "gemma_both_specificity_bootstrap_intervals_exclude_zero": bool(
            (gemma["specificity_ci_low"] > 0).all()
        ),
        "gemma_selected_loss_intervals_exclude_zero": bool(
            (gemma["selected_margin_loss_ci_low"] > 0).all()
        ),
        "any_branch_passes_both_interval_gates": bool(
            frame["interval_confirmed"].any()
        ),
        "branch_results": full,
        "reporting_rule": (
            "Report all four branches separately. A positive specificity with a "
            "selected-loss interval crossing zero is directional-specific evidence, "
            "not interval-confirmed direct damage."
        ),
        "interpretation": (
            "Gemma shows small, cross-split targeted-bank-specific reductions in "
            "final answer count margin in both timing branches; Qwen does not. "
            "No branch passes both registered interval gates."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "branch_summary.csv", index=False)
    _atomic_json(args.output / "targeted_logit_margin_synthesis.json", synthesis)
    print(json.dumps({"status": "PASS", "branch_count": len(frame)}))


if __name__ == "__main__":
    main()
