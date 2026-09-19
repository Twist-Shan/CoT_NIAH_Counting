#!/usr/bin/env python3
"""Add 10%-trimmed conditional-MAE laws and bias diagnostics to V3.2.

This is an additive post-generation extension.  It reads the same frozen
161,280-request table and the same 13-candidate registry as the V3.2 focused
analysis, but it never overwrites the frozen V3.2 output directory.  The new
continuous outcome is a symmetric 10%-trimmed cell-level conditional MAE,
defined only over parseable integer responses.  Absolute errors are sorted and
the smallest and largest floor(0.10*m) observations are removed before taking
the mean.  Bias diagnostics operate on the preregistered 10%-trimmed cell bias
and quantify both request-tail removal and OLS cell influence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_realistic_niah_v3_2_empirical_laws import (
    BIAS_FAMILY,
    DEFAULT_CONFIG,
    DEFAULT_FREEZE,
    DEFAULT_INPUT,
    Candidate,
    apply_coefficient_bh,
    build_cells,
    choose_formula,
    condition_fold,
    continuous_metrics,
    design_matrix,
    evidence_reading,
    file_sha256,
    fit_ols,
    json_safe,
    load_candidates,
    select_all,
    summarize_candidates,
    utc_now,
    validate_requests,
    write_json,
)


DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "anvil_realistic_niah_v3_1_20260819_formal"
    / "analysis"
    / "v3_2_trimmed_count_error_extension"
)
DEFAULT_FORMAL_ANALYSIS = (
    ROOT
    / "outputs"
    / "anvil_realistic_niah_v3_1_20260819_formal"
    / "analysis"
    / "v3_2_empirical_law"
)
DEFAULT_EXTENSION_CONFIG = (
    ROOT / "configs" / "realistic_niah_v3_2_inverse_n_candidate_extension.json"
)
MAE_FAMILY = "trimmed_conditional_mae_10"


def load_parent_frozen_config(config_path: Path, freeze_path: Path) -> dict[str, Any]:
    """Verify the immutable V3.2 config and input lock for this new estimand.

    The MAE estimand is intentionally post-freeze, so the historical methods
    document is not treated as if it had preregistered this change.  We still
    fail closed unless the parent config itself and its immutable request hash
    exactly match the recorded V3.2 freeze.
    """
    config = json.loads(config_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected_config_hash = freeze.get("files", {}).get(
        "configs/realistic_niah_v3_2_empirical_law_analysis.json"
    )
    if file_sha256(config_path) != expected_config_hash:
        raise ValueError("Parent V3.2 analysis config no longer matches its freeze")
    if (
        config.get("immutable_input", {}).get("request_level_sha256")
        != freeze.get("immutable_request_level_sha256")
    ):
        raise ValueError("Parent V3.2 request hash disagrees with its freeze")
    return config


def load_inverse_candidates(path: Path) -> tuple[Candidate, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = tuple(
        Candidate(
            id=str(item["id"]),
            terms=tuple(str(term) for term in item["terms"]),
            parent=(str(item["parent"]) if item.get("parent") else None),
            interaction=(
                str(item["interaction"])
                if item.get("interaction")
                else None
            ),
        )
        for item in payload["candidate_registry_addition"]
    )
    if len(candidates) != 5 or any(
        not candidate.id.startswith("invN") for candidate in candidates
    ):
        raise ValueError("Unexpected inverse-count candidate registry")
    return candidates


def add_inverse_count_terms(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    n = result["N"].to_numpy(dtype=float)
    if not np.all(np.isfinite(n)) or np.any(n <= 0):
        raise ValueError("The 1/N candidates require finite N > 0")
    result["invN"] = 1.0 / n
    result["invN_x_L_k"] = result["invN"] * result["L_k"]
    result["invN_x_logL"] = result["invN"] * result["logL"]
    return result


def deviation_summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "n_parseable": 0,
            "conditional_mae": math.nan,
            "trimmed_conditional_mae_10": math.nan,
            "conditional_rmse": math.nan,
            "raw_signed_mean": math.nan,
            "trimmed_signed_mean": math.nan,
            "trim_count_each_tail": 0,
            "max_abs_raw_error": math.nan,
            "max_abs_retained_error": math.nan,
            "max_abs_retained_for_mae": math.nan,
        }
    ordered = np.sort(array)
    ordered_absolute = np.sort(np.abs(array))
    trim = int(math.floor(0.10 * len(ordered)))
    retained = ordered[trim : len(ordered) - trim] if trim else ordered
    retained_absolute = (
        ordered_absolute[trim : len(ordered_absolute) - trim]
        if trim
        else ordered_absolute
    )
    return {
        "n_parseable": int(len(array)),
        "conditional_mae": float(np.mean(np.abs(array))),
        "trimmed_conditional_mae_10": float(np.mean(retained_absolute)),
        "conditional_rmse": float(np.sqrt(np.mean(np.square(array)))),
        "raw_signed_mean": float(np.mean(array)),
        "trimmed_signed_mean": float(np.mean(retained)),
        "trim_count_each_tail": trim,
        "max_abs_raw_error": float(np.max(np.abs(array))),
        "max_abs_retained_error": float(np.max(np.abs(retained))),
        "max_abs_retained_for_mae": float(np.max(retained_absolute)),
    }


def build_count_error_cells(requests: pd.DataFrame) -> pd.DataFrame:
    base = build_cells(requests)
    keys = ["comparison_slot", "prompt_mode", "N", "L"]
    rows: list[dict[str, Any]] = []
    for key, group in requests.groupby(keys, sort=True, observed=True):
        deviations = group.loc[
            group["parse_success"].astype(bool), "signed_deviation"
        ].dropna()
        rows.append({**dict(zip(keys, key, strict=True)), **deviation_summary(deviations)})
    extra = pd.DataFrame(rows)
    cells = base.merge(
        extra.drop(columns=["n_parseable", "trim_count_each_tail"]),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    cells["mae_law_eligible"] = cells["bias_law_eligible"].astype(bool)
    return cells


def fit_continuous_candidate(
    frame: pd.DataFrame,
    candidate: Candidate,
    *,
    outcome_family: str,
    outcome_column: str,
    n_levels: tuple[int, ...],
    l_levels: tuple[int, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y = frame[outcome_column].to_numpy(dtype=float)
    if np.any(y < 0):
        raise ValueError(f"{outcome_column} contains negative values")
    # V3.2 reports and models the symmetric 10%-trimmed Conditional MAE in its
    # native count-error units.  The identity-scale OLS law is deliberately
    # allowed to reveal misspecification through negative fitted values rather
    # than silently applying a log transformation or clipping.
    model_y = y
    x = design_matrix(frame, candidate)
    folds = condition_fold(frame, n_levels, l_levels)
    oof = np.full(len(frame), np.nan, dtype=float)
    fold_failures = 0
    for fold in range(5):
        test = folds == fold
        train = ~test
        if not np.any(test) or x[train].shape[0] <= x[train].shape[1]:
            fold_failures += 1
            continue
        try:
            fit = fit_ols(model_y[train], x[train], robust=False)
            oof[test] = fit.predict(x[test])
        except Exception:
            fold_failures += 1
    if np.isnan(oof).any():
        raise RuntimeError(
            f"{outcome_family}/{candidate.id} left {int(np.isnan(oof).sum())} OOF rows"
        )
    full = fit_ols(model_y, x, robust=True)
    cv = continuous_metrics(y, oof)
    full_prediction = np.asarray(full.predict(x), dtype=float)
    in_sample = continuous_metrics(y, full_prediction)
    metrics = {
        "outcome_family": outcome_family,
        "candidate": candidate.id,
        "predictors": len(candidate.terms),
        "parent": candidate.parent,
        "interaction": candidate.interaction,
        "n_rows": int(len(frame)),
        "n_conditions": int(frame[["N", "L"]].drop_duplicates().shape[0]),
        "converged": True,
        "fold_failures": fold_failures,
        "primary_loss": cv["mae"],
        "primary_score": cv["r2"],
        "cv_mae": cv["mae"],
        "cv_rmse": cv["rmse"],
        "cv_r2": cv["r2"],
        "in_sample_mae": in_sample["mae"],
        "in_sample_rmse": in_sample["rmse"],
        "in_sample_r2": in_sample["r2"],
        "aic": float(full.aic),
        "bic": float(full.bic),
        "mean_parse_rate": float(frame["n_parseable"].sum() / frame["n_total"].sum()),
        "minimum_parseable": int(frame["n_parseable"].min()),
        "minimum_prediction": float(np.min(full_prediction)),
        "model_scale": "trimmed_conditional_mae_10_identity",
        "inverse_link": "identity",
    }
    outcome_sd = float(np.std(model_y, ddof=0))
    coefficients: list[dict[str, Any]] = []
    for index, term in enumerate(("intercept", *candidate.terms)):
        estimate = float(full.params[index])
        standardized = (
            math.nan
            if term == "intercept" or outcome_sd <= 0
            else estimate * float(frame[term].std(ddof=0)) / outcome_sd
        )
        coefficients.append(
            {
                "outcome_family": outcome_family,
                "candidate": candidate.id,
                "term": term,
                "estimate": estimate,
                "standard_error": float(full.bse[index]),
                "p_value": float(full.pvalues[index]),
                "ci95_low": float(full.conf_int()[index, 0]),
                "ci95_high": float(full.conf_int()[index, 1]),
                "standardized_effect": standardized,
            }
        )
    return metrics, coefficients


def lomo_for_family(
    metrics: pd.DataFrame,
    coefficients: pd.DataFrame,
    candidates: tuple[Candidate, ...],
    selected: pd.DataFrame,
    family: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_slots = set(metrics["comparison_slot"].unique())
    for mode in sorted(metrics["prompt_mode"].unique()):
        full = selected.loc[
            selected["outcome_family"].eq(family)
            & selected["prompt_mode"].eq(mode)
        ].iloc[0]
        for omitted in sorted(all_slots):
            summary = summarize_candidates(
                metrics,
                coefficients,
                candidates,
                slots=all_slots.difference({omitted}),
            )
            winner = choose_formula(summary, family, mode, candidates)
            held = metrics.loc[
                metrics["outcome_family"].eq(family)
                & metrics["prompt_mode"].eq(mode)
                & metrics["comparison_slot"].eq(omitted)
                & metrics["candidate"].eq(winner["candidate"])
            ]
            rows.append(
                {
                    "outcome_family": family,
                    "prompt_mode": mode,
                    "omitted_comparison_slot": omitted,
                    "selected_without_slot": winner["candidate"],
                    "full_selected_candidate": full["selected_candidate"],
                    "agrees_with_full_selection": bool(
                        winner["candidate"] == full["selected_candidate"]
                    ),
                    "held_slot_primary_score": (
                        float(held.iloc[0]["primary_score"])
                        if not held.empty
                        else math.nan
                    ),
                    "held_slot_primary_loss": (
                        float(held.iloc[0]["primary_loss"])
                        if not held.empty
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def add_generic_lomo_summary(
    selected: pd.DataFrame, lomo: pd.DataFrame
) -> pd.DataFrame:
    summary = (
        lomo.groupby(["outcome_family", "prompt_mode"], as_index=False)
        .agg(
            lomo_formula_stability=("agrees_with_full_selection", "mean"),
            lomo_median_held_primary_score=("held_slot_primary_score", "median"),
            lomo_q25_held_primary_score=(
                "held_slot_primary_score",
                lambda value: value.quantile(0.25),
            ),
            lomo_median_held_primary_loss=("held_slot_primary_loss", "median"),
        )
    )
    result = selected.merge(
        summary,
        on=["outcome_family", "prompt_mode"],
        how="left",
        validate="one_to_one",
    )
    result["evidence_reading"] = result.apply(evidence_reading, axis=1)
    return result


def selected_bias_influence(
    frame: pd.DataFrame, candidate: Candidate
) -> dict[str, Any]:
    y = frame["trimmed_signed_bias_10"].to_numpy(dtype=float)
    x = design_matrix(frame, candidate)
    fit = sm.OLS(y, x).fit()
    influence = fit.get_influence()
    cooks = np.asarray(influence.cooks_distance[0], dtype=float)
    leverage = np.asarray(influence.hat_matrix_diag, dtype=float)
    studentized = np.asarray(influence.resid_studentized_external, dtype=float)
    n_rows, n_parameters = x.shape
    cook_threshold = 4.0 / n_rows
    leverage_threshold = 2.0 * n_parameters / n_rows
    finite_cooks = np.isfinite(cooks)
    diagnostic_estimable = bool(finite_cooks.any())
    flagged = finite_cooks & (cooks > cook_threshold)
    retained = ~flagged
    if retained.sum() <= n_parameters:
        retained = np.ones(n_rows, dtype=bool)
    reduced = sm.OLS(y[retained], x[retained]).fit()
    full_surface = np.asarray(fit.predict(x), dtype=float)
    reduced_surface = np.asarray(reduced.predict(x), dtype=float)
    surface_delta = reduced_surface - full_surface
    if np.std(full_surface) > 0 and np.std(reduced_surface) > 0:
        surface_correlation = float(np.corrcoef(full_surface, reduced_surface)[0, 1])
    else:
        surface_correlation = math.nan
    flagged_labels = [
        f"N={int(row.N)},L={int(row.L)}"
        for row in frame.loc[flagged, ["N", "L"]].itertuples(index=False)
    ]
    if diagnostic_estimable:
        max_index = int(np.nanargmax(cooks))
        max_row = frame.iloc[max_index]
        max_cooks_d = float(cooks[max_index])
        max_cooks_n: float | int = int(max_row["N"])
        max_cooks_l: float | int = int(max_row["L"])
    else:
        max_cooks_d = math.nan
        max_cooks_n = math.nan
        max_cooks_l = math.nan
    finite_studentized = np.abs(studentized[np.isfinite(studentized)])
    return {
        "n_cells": n_rows,
        "n_parameters": n_parameters,
        "diagnostic_estimable": diagnostic_estimable,
        "cook_threshold_4_over_n": cook_threshold,
        "influential_cells": int(flagged.sum()),
        "influential_fraction": float(flagged.mean()),
        "flagged_cells": ";".join(flagged_labels),
        "max_cooks_d": max_cooks_d,
        "max_cooks_N": max_cooks_n,
        "max_cooks_L": max_cooks_l,
        "max_abs_external_studentized_residual": float(
            np.max(finite_studentized) if len(finite_studentized) else math.nan
        ),
        "high_leverage_cells": int(np.sum(leverage > leverage_threshold)),
        "leverage_threshold_2p_over_n": leverage_threshold,
        "surface_correlation_after_dropping_influential": surface_correlation,
        "surface_rmse_after_dropping_influential": float(
            np.sqrt(np.mean(np.square(surface_delta)))
        ),
        "surface_max_abs_change_after_dropping_influential": float(
            np.max(np.abs(surface_delta))
        ),
    }


def summarize_request_tails(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mode, block in cells.groupby("prompt_mode", sort=True):
        difference = (
            block["raw_signed_mean"] - block["trimmed_signed_bias_10"]
        ).abs()
        raw = block["raw_signed_mean"].to_numpy(dtype=float)
        trimmed = block["trimmed_signed_bias_10"].to_numpy(dtype=float)
        correlation = (
            float(np.corrcoef(raw, trimmed)[0, 1])
            if np.std(raw) > 0 and np.std(trimmed) > 0
            else math.nan
        )
        rows.append(
            {
                "prompt_mode": mode,
                "cells": int(len(block)),
                "parseable_requests": int(block["n_parseable"].sum()),
                "requests_trimmed_from_each_tail": int(
                    block["trim_count_each_tail"].sum()
                ),
                "max_abs_raw_request_error": float(block["max_abs_raw_error"].max()),
                "max_abs_error_retained_after_cell_trimming": float(
                    block["max_abs_retained_error"].max()
                ),
                "median_abs_raw_vs_trimmed_cell_bias_change": float(
                    difference.median()
                ),
                "q95_abs_raw_vs_trimmed_cell_bias_change": float(
                    difference.quantile(0.95)
                ),
                "max_abs_raw_vs_trimmed_cell_bias_change": float(difference.max()),
                "pearson_raw_vs_trimmed_cell_bias": correlation,
            }
        )
    return pd.DataFrame(rows)


def summarize_influence(diagnostics: pd.DataFrame) -> pd.DataFrame:
    return (
        diagnostics.groupby("prompt_mode", as_index=False)
        .agg(
            slots=("comparison_slot", "nunique"),
            estimable_slots=("diagnostic_estimable", "sum"),
            cells_evaluated=("n_cells", "sum"),
            influential_cells=("influential_cells", "sum"),
            median_influential_fraction=("influential_fraction", "median"),
            max_cooks_d=("max_cooks_d", "max"),
            median_surface_correlation=(
                "surface_correlation_after_dropping_influential",
                "median",
            ),
            minimum_surface_correlation=(
                "surface_correlation_after_dropping_influential",
                "min",
            ),
            median_surface_rmse=(
                "surface_rmse_after_dropping_influential",
                "median",
            ),
            maximum_surface_change=(
                "surface_max_abs_change_after_dropping_influential",
                "max",
            ),
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument(
        "--extension-config", type=Path, default=DEFAULT_EXTENSION_CONFIG
    )
    parser.add_argument("--formal-analysis", type=Path, default=DEFAULT_FORMAL_ANALYSIS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = load_parent_frozen_config(
        args.config.resolve(), args.freeze.resolve()
    )
    candidates = load_candidates(config) + load_inverse_candidates(
        args.extension_config.resolve()
    )
    input_path = args.input.resolve()
    output = args.output.resolve()
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    if file_sha256(input_path) != config["immutable_input"]["request_level_sha256"]:
        raise ValueError("Request table SHA-256 does not match the V3.2 freeze")
    requests = pd.read_csv(input_path)
    validate_requests(requests, config)
    cells = add_inverse_count_terms(build_count_error_cells(requests))
    cells.to_csv(tables / "count_error_cells.csv.gz", index=False, compression="gzip")
    n_levels = tuple(int(value) for value in config["immutable_input"]["N_levels"])
    l_levels = tuple(int(value) for value in config["immutable_input"]["L_levels"])

    metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for (slot, mode), block in cells.groupby(
        ["comparison_slot", "prompt_mode"], sort=True, observed=True
    ):
        eligible = block.loc[block["mae_law_eligible"].astype(bool)].copy()
        for candidate in candidates:
            try:
                fit, terms = fit_continuous_candidate(
                    eligible,
                    candidate,
                    outcome_family=MAE_FAMILY,
                    outcome_column="trimmed_conditional_mae_10",
                    n_levels=n_levels,
                    l_levels=l_levels,
                )
                metric_rows.append(
                    {"comparison_slot": slot, "prompt_mode": mode, **fit}
                )
                coefficient_rows.extend(
                    {"comparison_slot": slot, "prompt_mode": mode, **term}
                    for term in terms
                )
            except Exception as error:
                failures.append(
                    {
                        "comparison_slot": str(slot),
                        "prompt_mode": str(mode),
                        "candidate": candidate.id,
                        "error": repr(error),
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    coefficients = apply_coefficient_bh(pd.DataFrame(coefficient_rows))
    failure_table = pd.DataFrame(failures)
    metrics.to_csv(tables / "mae_candidate_fit_metrics.csv", index=False)
    coefficients.to_csv(tables / "mae_candidate_coefficients.csv", index=False)
    failure_table.to_csv(tables / "mae_candidate_failures.csv", index=False)
    if failures:
        raise RuntimeError(
            f"Trimmed conditional-MAE fitting produced {len(failures)} failures"
        )

    summary = summarize_candidates(metrics, coefficients, candidates)
    selected = select_all(summary, candidates)
    lomo = lomo_for_family(metrics, coefficients, candidates, selected, MAE_FAMILY)
    selected = add_generic_lomo_summary(selected, lomo)
    selected_metrics = metrics.merge(
        selected[["outcome_family", "prompt_mode", "selected_candidate"]],
        left_on=["outcome_family", "prompt_mode", "candidate"],
        right_on=["outcome_family", "prompt_mode", "selected_candidate"],
        how="inner",
        validate="many_to_one",
    )
    selected_coefficients = coefficients.merge(
        selected[["outcome_family", "prompt_mode", "selected_candidate"]],
        left_on=["outcome_family", "prompt_mode", "candidate"],
        right_on=["outcome_family", "prompt_mode", "selected_candidate"],
        how="inner",
        validate="many_to_one",
    )
    summary.to_csv(tables / "mae_mode_candidate_summary.csv", index=False)
    selected.to_csv(tables / "mae_selected_mode_laws.csv", index=False)
    lomo.to_csv(tables / "mae_lomo_structure_selection.csv", index=False)
    selected_metrics.to_csv(tables / "mae_selected_model_fit_metrics.csv", index=False)
    selected_coefficients.to_csv(
        tables / "mae_selected_model_coefficients.csv", index=False
    )

    formal_tables = args.formal_analysis.resolve() / "tables"
    formal_selected = pd.read_csv(formal_tables / "selected_mode_laws.csv")
    candidate_map = {candidate.id: candidate for candidate in candidates}
    influence_rows: list[dict[str, Any]] = []
    for (slot, mode), block in cells.loc[
        cells["bias_law_eligible"].astype(bool)
    ].groupby(["comparison_slot", "prompt_mode"], sort=True, observed=True):
        winner = formal_selected.loc[
            formal_selected["outcome_family"].eq(BIAS_FAMILY)
            & formal_selected["prompt_mode"].eq(mode),
            "selected_candidate",
        ].iloc[0]
        influence_rows.append(
            {
                "comparison_slot": slot,
                "prompt_mode": mode,
                "selected_candidate": winner,
                **selected_bias_influence(block, candidate_map[winner]),
            }
        )
    influence = pd.DataFrame(influence_rows)
    influence_summary = summarize_influence(influence)
    request_tails = summarize_request_tails(cells)
    influence.to_csv(tables / "bias_cell_influence_diagnostics.csv", index=False)
    influence_summary.to_csv(tables / "bias_influence_summary.csv", index=False)
    request_tails.to_csv(tables / "bias_request_tail_diagnostics.csv", index=False)

    manifest = {
        "schema_version": "realistic_niah_v3_2_trimmed_count_error_extension_v1",
        "status": "complete",
        "created_utc": utc_now(),
        "input": str(input_path),
        "input_sha256": file_sha256(input_path),
        "formal_analysis_manifest": str(
            args.formal_analysis.resolve() / "analysis_manifest.json"
        ),
        "formal_analysis_manifest_sha256": file_sha256(
            args.formal_analysis.resolve() / "analysis_manifest.json"
        ),
        "parent_v3_2_freeze": str(args.freeze.resolve()),
        "parent_v3_2_config_verified": True,
        "post_freeze_estimand_change": True,
        "config_sha256": file_sha256(args.config.resolve()),
        "inverse_count_extension_config": str(args.extension_config.resolve()),
        "inverse_count_extension_config_sha256": file_sha256(
            args.extension_config.resolve()
        ),
        "candidate_registry_size": len(candidates),
        "requests": int(len(requests)),
        "cells": int(len(cells)),
        "outcome": {
            "id": MAE_FAMILY,
            "definition": "sort absolute errors over parseable responses, remove floor(0.10*m) from each tail, and average the retained values within each slot-mode-N-L cell",
            "minimum_parseable": 20,
            "symmetric_trim_fraction_each_tail": 0.10,
            "fit": "identity-scale OLS for symmetric 10%-trimmed conditional MAE, with HC3 coefficient covariance",
            "inverse_link": "identity",
            "validation": "five-fold held-condition CV",
            "lomo": True,
        },
        "bias_influence": {
            "request_tail_control": "10% symmetric trimming within each cell",
            "cell_influence_rule": "Cook's D > 4/n (descriptive heuristic, not a hypothesis test)",
            "sensitivity_refit": "same selected topology after dropping Cook-flagged cells",
        },
        "bootstrap_repetitions": 0,
        "selected_laws": selected[
            [
                "prompt_mode",
                "selected_candidate",
                "median_primary_score",
                "q25_primary_score",
                "median_primary_loss",
                "lomo_formula_stability",
                "lomo_median_held_primary_score",
            ]
        ].to_dict("records"),
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output / "analysis_manifest.json", json_safe(manifest))
    write_json(
        output / "analysis_state.json",
        {
            "schema_version": "realistic_niah_v3_2_trimmed_count_error_extension_state_v1",
            "stage": "complete",
            "updated_at_utc": utc_now(),
            "mae_selected_laws": len(selected),
            "bias_influence_rows": len(influence),
            "elapsed_seconds": manifest["elapsed_seconds"],
        },
    )
    print(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
