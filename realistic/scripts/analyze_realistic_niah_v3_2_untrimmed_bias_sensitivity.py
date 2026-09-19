#!/usr/bin/env python3
"""Sensitivity analysis for untrimmed signed-bias laws in V3.2.

This script changes exactly one estimand relative to the frozen V3.2 bias
analysis: within each comparison-slot x prompt-mode x N x L cell, it averages
all parseable signed errors instead of symmetrically trimming 10% from each
tail.  The candidate registry, condition folds, selection gates, slot-specific
coefficients, and leave-one-model-out (LOMO) validation are unchanged.

The output is exploratory and is written to a separate directory; it never
overwrites the confirmatory V3.2 tables.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import analyze_realistic_niah_v3_2_empirical_laws as core


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "anvil_realistic_niah_v3_1_20260819_formal"
    / "analysis"
    / "v3_2_untrimmed_bias_sensitivity"
)
DEFAULT_TRIMMED_TABLES = (
    ROOT
    / "outputs"
    / "anvil_realistic_niah_v3_1_20260819_formal"
    / "analysis"
    / "v3_2_empirical_law"
    / "tables"
)
FAMILY = "mean_signed_bias_untrimmed"
MODES = ("enumeration_index", "enumeration_bullet", "native_thinking")
KEYS = ["comparison_slot", "prompt_mode", "N", "L"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=core.DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=core.DEFAULT_CONFIG)
    parser.add_argument("--freeze", type=Path, default=core.DEFAULT_FREEZE)
    parser.add_argument("--trimmed-tables", type=Path, default=DEFAULT_TRIMMED_TABLES)
    return parser.parse_args()


def build_untrimmed_cells(requests: pd.DataFrame) -> pd.DataFrame:
    cells = core.build_cells(requests)
    parsed = requests.loc[
        requests["parse_success"].astype(bool) & requests["signed_deviation"].notna()
    ].copy()
    grouped = (
        parsed.groupby(KEYS, sort=True, observed=True)["signed_deviation"]
        .agg(
            mean_signed_bias_untrimmed="mean",
            median_signed_bias="median",
            signed_error_sd="std",
            mean_absolute_error=lambda values: float(np.mean(np.abs(values))),
        )
        .reset_index()
    )
    cells = cells.merge(grouped, on=KEYS, how="left", validate="one_to_one")
    if cells["mean_signed_bias_untrimmed"].isna().any():
        raise ValueError("Untrimmed cell aggregation left missing values")
    cells["trimmed_minus_untrimmed"] = (
        cells["trimmed_signed_bias_10"] - cells["mean_signed_bias_untrimmed"]
    )
    return cells


def fit_all(
    cells: pd.DataFrame,
    candidates: tuple[core.Candidate, ...],
    n_levels: tuple[int, ...],
    l_levels: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    core.BIAS_FAMILY = FAMILY
    core.LOMO_FAMILIES = (FAMILY,)
    metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, str]] = []
    for (slot, mode), block in cells.loc[cells["prompt_mode"].isin(MODES)].groupby(
        ["comparison_slot", "prompt_mode"], sort=True, observed=True
    ):
        eligible = block.loc[block["bias_law_eligible"].astype(bool)].copy()
        eligible["trimmed_signed_bias_10"] = eligible["mean_signed_bias_untrimmed"]
        for candidate in candidates:
            try:
                fit, terms = core.fit_bias_candidate(
                    eligible, candidate, n_levels, l_levels
                )
                metric_rows.append(
                    {"comparison_slot": slot, "prompt_mode": mode, **fit}
                )
                coefficient_rows.extend(
                    {"comparison_slot": slot, "prompt_mode": mode, **term}
                    for term in terms
                )
            except Exception as error:  # preserve the complete audit trail
                failure_rows.append(
                    {
                        "comparison_slot": str(slot),
                        "prompt_mode": str(mode),
                        "outcome_family": FAMILY,
                        "candidate": candidate.id,
                        "error": repr(error),
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    coefficients = core.apply_coefficient_bh(pd.DataFrame(coefficient_rows))
    return metrics, coefficients, pd.DataFrame(failure_rows)


def mode_diagnostics(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mode, block in cells.loc[cells["prompt_mode"].isin(MODES)].groupby(
        "prompt_mode", sort=True, observed=True
    ):
        rows.append(
            {
                "prompt_mode": mode,
                "cells": int(len(block)),
                "zero_fraction_trimmed": float(
                    np.mean(np.isclose(block["trimmed_signed_bias_10"], 0.0))
                ),
                "zero_fraction_untrimmed": float(
                    np.mean(np.isclose(block["mean_signed_bias_untrimmed"], 0.0))
                ),
                "median_abs_trimmed_bias": float(
                    np.median(np.abs(block["trimmed_signed_bias_10"]))
                ),
                "median_abs_untrimmed_bias": float(
                    np.median(np.abs(block["mean_signed_bias_untrimmed"]))
                ),
                "median_abs_estimand_change": float(
                    np.median(np.abs(block["trimmed_minus_untrimmed"]))
                ),
                "correlation_trimmed_untrimmed": float(
                    block[["trimmed_signed_bias_10", "mean_signed_bias_untrimmed"]]
                    .corr()
                    .iloc[0, 1]
                ),
            }
        )
    return pd.DataFrame(rows)


def comparison_table(selected: pd.DataFrame, trimmed_tables: Path) -> pd.DataFrame:
    untrimmed = selected[
        [
            "prompt_mode",
            "selected_candidate",
            "median_primary_score",
            "q25_primary_score",
            "lomo_formula_stability",
            "lomo_median_held_primary_score",
            "evidence_reading",
        ]
    ].copy()
    untrimmed = untrimmed.add_prefix("untrimmed_").rename(
        columns={"untrimmed_prompt_mode": "prompt_mode"}
    )
    trimmed = pd.read_csv(trimmed_tables / "selected_mode_laws.csv")
    trimmed = trimmed.loc[
        trimmed["outcome_family"].eq("trimmed_signed_bias_10")
        & trimmed["prompt_mode"].isin(MODES),
        [
            "prompt_mode",
            "selected_candidate",
            "median_primary_score",
            "q25_primary_score",
            "lomo_formula_stability",
            "lomo_median_held_primary_score",
            "evidence_reading",
        ],
    ].add_prefix("trimmed_").rename(columns={"trimmed_prompt_mode": "prompt_mode"})
    result = trimmed.merge(untrimmed, on="prompt_mode", validate="one_to_one")
    result["delta_median_cv_r2"] = (
        result["untrimmed_median_primary_score"]
        - result["trimmed_median_primary_score"]
    )
    result["delta_lomo_median_held_r2"] = (
        result["untrimmed_lomo_median_held_primary_score"]
        - result["trimmed_lomo_median_held_primary_score"]
    )
    return result


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = core.verify_freeze(args.config.resolve(), args.freeze.resolve())
    if core.file_sha256(args.input.resolve()) != config["immutable_input"][
        "request_level_sha256"
    ]:
        raise ValueError("Request table SHA-256 does not match the V3.2 freeze")
    requests = pd.read_csv(args.input.resolve())
    core.validate_requests(requests, config)
    cells = build_untrimmed_cells(requests)
    candidates = core.load_candidates(config)
    n_levels = tuple(int(value) for value in config["immutable_input"]["N_levels"])
    l_levels = tuple(int(value) for value in config["immutable_input"]["L_levels"])
    metrics, coefficients, failures = fit_all(
        cells, candidates, n_levels, l_levels
    )
    if len(failures):
        raise RuntimeError(f"Untrimmed fitting produced {len(failures)} failures")
    summary = core.summarize_candidates(metrics, coefficients, candidates)
    selected = core.select_all(summary, candidates)
    lomo = core.lomo_selection(metrics, coefficients, candidates, selected)
    selected = core.add_lomo_summary(selected, lomo)
    selected_metrics = metrics.merge(
        selected[["outcome_family", "prompt_mode", "selected_candidate"]],
        left_on=["outcome_family", "prompt_mode", "candidate"],
        right_on=["outcome_family", "prompt_mode", "selected_candidate"],
        validate="many_to_one",
    )
    selected_coefficients = coefficients.merge(
        selected[["outcome_family", "prompt_mode", "selected_candidate"]],
        left_on=["outcome_family", "prompt_mode", "candidate"],
        right_on=["outcome_family", "prompt_mode", "selected_candidate"],
        validate="many_to_one",
    )

    output = args.output.resolve()
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    cells.to_csv(tables / "cell_outcomes_untrimmed.csv.gz", index=False, compression="gzip")
    metrics.to_csv(tables / "candidate_fit_metrics.csv", index=False)
    coefficients.to_csv(tables / "candidate_coefficients.csv", index=False)
    failures.to_csv(tables / "candidate_fit_failures.csv", index=False)
    summary.to_csv(tables / "mode_candidate_summary.csv", index=False)
    selected.to_csv(tables / "selected_mode_laws.csv", index=False)
    lomo.to_csv(tables / "lomo_structure_selection.csv", index=False)
    selected_metrics.to_csv(tables / "selected_model_fit_metrics.csv", index=False)
    selected_coefficients.to_csv(
        tables / "selected_model_coefficients.csv", index=False
    )
    diagnostics = mode_diagnostics(cells)
    diagnostics.to_csv(tables / "estimand_diagnostics.csv", index=False)
    comparison = comparison_table(selected, args.trimmed_tables.resolve())
    comparison.to_csv(tables / "trimmed_vs_untrimmed_comparison.csv", index=False)

    manifest = {
        "schema_version": "realistic_niah_v3_2_untrimmed_bias_sensitivity_v1",
        "status": "complete",
        "exploratory": True,
        "changed_estimand_only": True,
        "input": str(args.input.resolve()),
        "input_sha256": core.file_sha256(args.input.resolve()),
        "modes": list(MODES),
        "candidate_count": len(candidates),
        "condition_folds": 5,
        "lomo": True,
        "bootstrap_repetitions": 0,
        "selected_laws": selected.to_dict("records"),
        "elapsed_seconds": time.perf_counter() - started,
    }
    core.write_json(output / "analysis_manifest.json", manifest)
    print(json.dumps(core.json_safe(manifest), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
