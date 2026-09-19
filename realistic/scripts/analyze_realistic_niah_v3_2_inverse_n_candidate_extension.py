#!/usr/bin/env python3
"""Extend the frozen V3.2 candidate registry with the inverse-count basis 1/N.

The frozen 13-candidate analysis is never overwritten.  This additive audit
fits only five new hierarchical candidates, merges them with the exact frozen
fit tables, and reruns the unchanged V3.2 selection and LOMO procedures for
accuracy, 10%-trimmed signed bias, and conditional MAE.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import statsmodels

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_realistic_niah_v3_2_count_error_extension import (
    MAE_FAMILY,
    add_generic_lomo_summary,
    build_count_error_cells,
    fit_continuous_candidate,
    lomo_for_family,
    selected_bias_influence,
    summarize_influence,
    summarize_request_tails,
)
from scripts.analyze_realistic_niah_v3_2_empirical_laws import (
    BIAS_FAMILY,
    DEFAULT_CONFIG,
    DEFAULT_FREEZE,
    DEFAULT_INPUT,
    Candidate,
    add_lomo_summary,
    apply_coefficient_bh,
    build_cells,
    file_sha256,
    fit_slot_task,
    json_safe,
    lomo_selection,
    load_candidates,
    run_beta_binomial,
    select_all,
    summarize_candidates,
    utc_now,
    validate_requests,
    verify_freeze,
    write_json,
)


DEFAULT_EXTENSION_CONFIG = (
    ROOT / "configs" / "realistic_niah_v3_2_inverse_n_candidate_extension.json"
)
DEFAULT_FORMAL_ANALYSIS = (
    ROOT
    / "outputs"
    / "anvil_realistic_niah_v3_1_20260819_formal"
    / "analysis"
    / "v3_2_empirical_law"
)
DEFAULT_COUNT_ERROR_ANALYSIS = (
    ROOT
    / "outputs"
    / "anvil_realistic_niah_v3_1_20260819_formal"
    / "analysis"
    / "v3_2_count_error_extension"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "anvil_realistic_niah_v3_1_20260819_formal"
    / "analysis"
    / "v3_2_inverse_n_candidate_extension"
)


def load_inverse_candidates(path: Path) -> tuple[Candidate, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = tuple(
        Candidate(
            id=str(item["id"]),
            terms=tuple(str(term) for term in item["terms"]),
            parent=(str(item["parent"]) if item.get("parent") else None),
            interaction=(
                str(item["interaction"]) if item.get("interaction") else None
            ),
        )
        for item in payload["candidate_registry_addition"]
    )
    expected = (
        "invN",
        "invN__L_k",
        "invN__logL",
        "invN__L_k__invN_x_L_k",
        "invN__logL__invN_x_logL",
    )
    if tuple(candidate.id for candidate in candidates) != expected:
        raise ValueError("Unexpected inverse-count candidate registry")
    return candidates


def add_inverse_count_terms(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    n = result["N"].to_numpy(dtype=float)
    if not np.all(np.isfinite(n)) or np.any(n <= 0):
        raise ValueError("The 1/N candidate requires finite N > 0")
    inv_n = 1.0 / n
    result["invN"] = inv_n
    result["invN_x_L_k"] = inv_n * result["L_k"].to_numpy(dtype=float)
    result["invN_x_logL"] = inv_n * result["logL"].to_numpy(dtype=float)
    return result


def fit_new_accuracy_bias_candidates(
    requests: pd.DataFrame,
    cells: pd.DataFrame,
    candidates: tuple[Candidate, ...],
    n_levels: tuple[int, ...],
    l_levels: tuple[int, ...],
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    request_columns = [
        "N",
        "L",
        "exact_count",
        "L_k",
        "logN",
        "logL",
        "invN",
        "invN_x_L_k",
        "invN_x_logL",
    ]
    candidate_payload = [
        {
            "id": candidate.id,
            "terms": candidate.terms,
            "parent": candidate.parent,
            "interaction": candidate.interaction,
        }
        for candidate in candidates
    ]
    payloads: list[dict[str, Any]] = []
    keys = (
        requests[["comparison_slot", "prompt_mode"]]
        .drop_duplicates()
        .sort_values(["comparison_slot", "prompt_mode"])
        .itertuples(index=False, name=None)
    )
    for slot, mode in keys:
        payloads.append(
            {
                "slot": slot,
                "mode": mode,
                "requests": requests.loc[
                    requests["comparison_slot"].eq(slot)
                    & requests["prompt_mode"].eq(mode),
                    request_columns,
                ].reset_index(drop=True),
                "cells": cells.loc[
                    cells["comparison_slot"].eq(slot)
                    & cells["prompt_mode"].eq(mode)
                ].reset_index(drop=True),
                "candidates": candidate_payload,
                "n_levels": n_levels,
                "l_levels": l_levels,
            }
        )
    metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    if workers == 1:
        iterator = (fit_slot_task(payload) for payload in payloads)
        for result in iterator:
            metric_rows.extend(result["metrics"])
            coefficient_rows.extend(result["coefficients"])
            failure_rows.extend(result["failures"])
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(fit_slot_task, payload) for payload in payloads]
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                metric_rows.extend(result["metrics"])
                coefficient_rows.extend(result["coefficients"])
                failure_rows.extend(result["failures"])
                print(
                    json.dumps(
                        {
                            "stage": "inverse_candidate_fitting",
                            "completed_tasks": completed,
                            "total_tasks": len(payloads),
                            "slot": result["slot"],
                            "mode": result["mode"],
                        }
                    ),
                    flush=True,
                )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(failure_rows),
    )


def fit_new_mae_candidates(
    cells: pd.DataFrame,
    candidates: tuple[Candidate, ...],
    n_levels: tuple[int, ...],
    l_levels: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
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
                    outcome_column="conditional_mae",
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
                failure_rows.append(
                    {
                        "comparison_slot": str(slot),
                        "prompt_mode": str(mode),
                        "outcome_family": MAE_FAMILY,
                        "candidate": candidate.id,
                        "error": repr(error),
                    }
                )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(failure_rows),
    )


def comparison_table(
    baseline: pd.DataFrame, expanded: pd.DataFrame, label: str
) -> pd.DataFrame:
    left = baseline[
        [
            "outcome_family",
            "prompt_mode",
            "selected_candidate",
            "median_primary_score",
            "q25_primary_score",
            "lomo_formula_stability",
        ]
    ].rename(
        columns={
            "selected_candidate": "baseline_selected_candidate",
            "median_primary_score": "baseline_median_primary_score",
            "q25_primary_score": "baseline_q25_primary_score",
            "lomo_formula_stability": "baseline_lomo_formula_stability",
        }
    )
    right = expanded[
        [
            "outcome_family",
            "prompt_mode",
            "selected_candidate",
            "median_primary_score",
            "q25_primary_score",
            "lomo_formula_stability",
        ]
    ].rename(
        columns={
            "selected_candidate": "expanded_selected_candidate",
            "median_primary_score": "expanded_median_primary_score",
            "q25_primary_score": "expanded_q25_primary_score",
            "lomo_formula_stability": "expanded_lomo_formula_stability",
        }
    )
    result = left.merge(
        right,
        on=["outcome_family", "prompt_mode"],
        how="outer",
        validate="one_to_one",
    )
    result.insert(0, "analysis_block", label)
    result["selection_changed"] = (
        result["baseline_selected_candidate"]
        != result["expanded_selected_candidate"]
    )
    result["selected_is_inverse_count"] = result[
        "expanded_selected_candidate"
    ].str.startswith("invN", na=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument(
        "--extension-config", type=Path, default=DEFAULT_EXTENSION_CONFIG
    )
    parser.add_argument(
        "--formal-analysis", type=Path, default=DEFAULT_FORMAL_ANALYSIS
    )
    parser.add_argument(
        "--count-error-analysis", type=Path, default=DEFAULT_COUNT_ERROR_ANALYSIS
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2))
    )
    parser.add_argument("--skip-beta-binomial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = verify_freeze(args.config.resolve(), args.freeze.resolve())
    base_candidates = load_candidates(config)
    inverse_candidates = load_inverse_candidates(args.extension_config.resolve())
    candidates = (*base_candidates, *inverse_candidates)
    input_path = args.input.resolve()
    output = args.output.resolve()
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    if file_sha256(input_path) != config["immutable_input"]["request_level_sha256"]:
        raise ValueError("Request table SHA-256 does not match the V3.2 freeze")

    requests = pd.read_csv(input_path)
    validate_requests(requests, config)
    requests = add_inverse_count_terms(requests)
    cells = add_inverse_count_terms(build_cells(requests))
    count_error_cells = add_inverse_count_terms(build_count_error_cells(requests))
    cells.to_csv(tables / "cell_outcomes.csv.gz", index=False, compression="gzip")
    count_error_cells.to_csv(
        tables / "count_error_cells.csv.gz", index=False, compression="gzip"
    )
    n_levels = tuple(int(value) for value in config["immutable_input"]["N_levels"])
    l_levels = tuple(int(value) for value in config["immutable_input"]["L_levels"])

    new_metrics, new_coefficients, new_failures = fit_new_accuracy_bias_candidates(
        requests,
        cells,
        inverse_candidates,
        n_levels,
        l_levels,
        max(1, args.workers),
    )
    new_coefficients = apply_coefficient_bh(new_coefficients)
    new_metrics.to_csv(tables / "inverse_candidate_fit_metrics.csv", index=False)
    new_coefficients.to_csv(
        tables / "inverse_candidate_coefficients.csv", index=False
    )
    new_failures.to_csv(tables / "inverse_candidate_failures.csv", index=False)
    if len(new_failures):
        raise RuntimeError(
            f"Inverse-count accuracy/bias fitting produced {len(new_failures)} failures"
        )

    formal_tables = args.formal_analysis.resolve() / "tables"
    base_metrics = pd.read_csv(formal_tables / "candidate_fit_metrics.csv")
    base_coefficients = pd.read_csv(formal_tables / "candidate_coefficients.csv")
    base_selected = pd.read_csv(formal_tables / "selected_mode_laws.csv")
    combined_metrics = pd.concat([base_metrics, new_metrics], ignore_index=True)
    combined_coefficients = apply_coefficient_bh(
        pd.concat([base_coefficients, new_coefficients], ignore_index=True)
    )
    summary = summarize_candidates(combined_metrics, combined_coefficients, candidates)
    selected = select_all(summary, candidates)
    lomo = lomo_selection(combined_metrics, combined_coefficients, candidates, selected)
    selected = add_lomo_summary(selected, lomo)
    selected_metrics = combined_metrics.merge(
        selected[["outcome_family", "prompt_mode", "selected_candidate"]],
        left_on=["outcome_family", "prompt_mode", "candidate"],
        right_on=["outcome_family", "prompt_mode", "selected_candidate"],
        how="inner",
        validate="many_to_one",
    )
    selected_coefficients = combined_coefficients.merge(
        selected[["outcome_family", "prompt_mode", "selected_candidate"]],
        left_on=["outcome_family", "prompt_mode", "candidate"],
        right_on=["outcome_family", "prompt_mode", "selected_candidate"],
        how="inner",
        validate="many_to_one",
    )
    combined_metrics.to_csv(tables / "candidate_fit_metrics.csv", index=False)
    combined_coefficients.to_csv(tables / "candidate_coefficients.csv", index=False)
    summary.to_csv(tables / "mode_candidate_summary.csv", index=False)
    selected.to_csv(tables / "selected_mode_laws.csv", index=False)
    lomo.to_csv(tables / "lomo_structure_selection.csv", index=False)
    selected_metrics.to_csv(tables / "selected_model_fit_metrics.csv", index=False)
    selected_coefficients.to_csv(
        tables / "selected_model_coefficients.csv", index=False
    )

    base_mae_tables = args.count_error_analysis.resolve() / "tables"
    base_mae_metrics = pd.read_csv(base_mae_tables / "mae_candidate_fit_metrics.csv")
    base_mae_coefficients = pd.read_csv(
        base_mae_tables / "mae_candidate_coefficients.csv"
    )
    base_mae_selected = pd.read_csv(base_mae_tables / "mae_selected_mode_laws.csv")
    new_mae_metrics, new_mae_coefficients, new_mae_failures = (
        fit_new_mae_candidates(
            count_error_cells,
            inverse_candidates,
            n_levels,
            l_levels,
        )
    )
    new_mae_coefficients = apply_coefficient_bh(new_mae_coefficients)
    new_mae_metrics.to_csv(
        tables / "mae_inverse_candidate_fit_metrics.csv", index=False
    )
    new_mae_coefficients.to_csv(
        tables / "mae_inverse_candidate_coefficients.csv", index=False
    )
    new_mae_failures.to_csv(
        tables / "mae_inverse_candidate_failures.csv", index=False
    )
    if len(new_mae_failures):
        raise RuntimeError(
            f"Inverse-count conditional-MAE fitting produced {len(new_mae_failures)} failures"
        )
    mae_metrics = pd.concat([base_mae_metrics, new_mae_metrics], ignore_index=True)
    mae_coefficients = apply_coefficient_bh(
        pd.concat([base_mae_coefficients, new_mae_coefficients], ignore_index=True)
    )
    mae_summary = summarize_candidates(mae_metrics, mae_coefficients, candidates)
    mae_selected = select_all(mae_summary, candidates)
    mae_lomo = lomo_for_family(
        mae_metrics, mae_coefficients, candidates, mae_selected, MAE_FAMILY
    )
    mae_selected = add_generic_lomo_summary(mae_selected, mae_lomo)
    mae_selected_metrics = mae_metrics.merge(
        mae_selected[["outcome_family", "prompt_mode", "selected_candidate"]],
        left_on=["outcome_family", "prompt_mode", "candidate"],
        right_on=["outcome_family", "prompt_mode", "selected_candidate"],
        how="inner",
        validate="many_to_one",
    )
    mae_selected_coefficients = mae_coefficients.merge(
        mae_selected[["outcome_family", "prompt_mode", "selected_candidate"]],
        left_on=["outcome_family", "prompt_mode", "candidate"],
        right_on=["outcome_family", "prompt_mode", "selected_candidate"],
        how="inner",
        validate="many_to_one",
    )
    mae_metrics.to_csv(tables / "mae_candidate_fit_metrics.csv", index=False)
    mae_coefficients.to_csv(tables / "mae_candidate_coefficients.csv", index=False)
    mae_summary.to_csv(tables / "mae_mode_candidate_summary.csv", index=False)
    mae_selected.to_csv(tables / "mae_selected_mode_laws.csv", index=False)
    mae_lomo.to_csv(tables / "mae_lomo_structure_selection.csv", index=False)
    mae_selected_metrics.to_csv(
        tables / "mae_selected_model_fit_metrics.csv", index=False
    )
    mae_selected_coefficients.to_csv(
        tables / "mae_selected_model_coefficients.csv", index=False
    )

    candidate_map = {candidate.id: candidate for candidate in candidates}
    influence_rows: list[dict[str, Any]] = []
    for (slot, mode), block in count_error_cells.loc[
        count_error_cells["bias_law_eligible"].astype(bool)
    ].groupby(["comparison_slot", "prompt_mode"], sort=True, observed=True):
        winner = selected.loc[
            selected["outcome_family"].eq(BIAS_FAMILY)
            & selected["prompt_mode"].eq(mode),
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
    request_tails = summarize_request_tails(count_error_cells)
    influence.to_csv(tables / "bias_cell_influence_diagnostics.csv", index=False)
    influence_summary.to_csv(tables / "bias_influence_summary.csv", index=False)
    request_tails.to_csv(tables / "bias_request_tail_diagnostics.csv", index=False)

    beta_metrics = pd.DataFrame()
    beta_coefficients = pd.DataFrame()
    beta_failures = pd.DataFrame()
    if not args.skip_beta_binomial:
        beta_metrics, beta_coefficients, beta_failures = run_beta_binomial(
            cells, selected, candidates, n_levels, l_levels
        )
        beta_metrics.to_csv(tables / "beta_binomial_fit_metrics.csv", index=False)
        beta_coefficients.to_csv(
            tables / "beta_binomial_coefficients.csv", index=False
        )
        beta_failures.to_csv(tables / "beta_binomial_failures.csv", index=False)
        if len(beta_failures):
            raise RuntimeError(
                f"Expanded Beta-Binomial robustness produced {len(beta_failures)} failures"
            )
    else:
        for name in (
            "beta_binomial_fit_metrics.csv",
            "beta_binomial_coefficients.csv",
            "beta_binomial_failures.csv",
        ):
            pd.DataFrame().to_csv(tables / name, index=False)

    comparisons = pd.concat(
        [
            comparison_table(base_selected, selected, "accuracy_and_bias"),
            comparison_table(base_mae_selected, mae_selected, "conditional_mae"),
        ],
        ignore_index=True,
    )
    comparisons.to_csv(tables / "baseline_vs_inverse_n_selection.csv", index=False)

    manifest = {
        "schema_version": "realistic_niah_v3_2_inverse_n_candidate_extension_v1",
        "status": "complete",
        "analysis_version": "V3.2 + inverse-count candidate extension",
        "created_utc": utc_now(),
        "input": str(input_path),
        "input_sha256": file_sha256(input_path),
        "base_config_sha256": file_sha256(args.config.resolve()),
        "extension_config": str(args.extension_config.resolve()),
        "extension_config_sha256": file_sha256(args.extension_config.resolve()),
        "formal_analysis_manifest_sha256": file_sha256(
            args.formal_analysis.resolve() / "analysis_manifest.json"
        ),
        "count_error_manifest_sha256": file_sha256(
            args.count_error_analysis.resolve() / "analysis_manifest.json"
        ),
        "requests": int(len(requests)),
        "cells": int(len(cells)),
        "base_candidates": len(base_candidates),
        "added_candidates": [candidate.id for candidate in inverse_candidates],
        "expanded_candidates": len(candidates),
        "selection_and_validation_changed": False,
        "bootstrap_repetitions": 0,
        "selection_changes": int(comparisons["selection_changed"].sum()),
        "inverse_count_selections": int(
            comparisons["selected_is_inverse_count"].sum()
        ),
        "selected_laws": selected[
            [
                "outcome_family",
                "prompt_mode",
                "selected_candidate",
                "median_primary_score",
                "q25_primary_score",
                "median_primary_loss",
                "lomo_formula_stability",
            ]
        ].to_dict("records"),
        "mae_selected_laws": mae_selected[
            [
                "outcome_family",
                "prompt_mode",
                "selected_candidate",
                "median_primary_score",
                "q25_primary_score",
                "median_primary_loss",
                "lomo_formula_stability",
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
            "schema_version": "realistic_niah_v3_2_inverse_n_candidate_extension_state_v1",
            "stage": "complete",
            "updated_at_utc": utc_now(),
            "selection_changes": manifest["selection_changes"],
            "inverse_count_selections": manifest["inverse_count_selections"],
            "elapsed_seconds": manifest["elapsed_seconds"],
        },
    )
    print(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
