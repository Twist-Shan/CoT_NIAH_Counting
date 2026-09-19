#!/usr/bin/env python3
"""Evaluate frozen V3.2 laws and run exploratory V3.3 regression scans.

The primary V3.3 result is a genuine out-of-range evaluation: frozen V3.2
coefficients are scored on the 25k--100k holdout without refitting.  Candidate
scans and the fixed-N=10 length regressions are explicitly exploratory.  They
use the V3.2 five-fold held-condition rule and no bootstrap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.special import expit


REPO_ROOT = Path(
    os.environ.get("NIAH_REPO_ROOT", Path(__file__).resolve().parents[1])
).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_realistic_niah_v3_2_count_error_extension import (  # noqa: E402
    MAE_FAMILY,
    build_count_error_cells,
    fit_continuous_candidate,
    load_inverse_candidates,
)
from scripts.analyze_realistic_niah_v3_2_empirical_laws import (  # noqa: E402
    BIAS_FAMILY,
    HEADLINE_ACCURACY,
    Candidate,
    accuracy_intercept_oof,
    binary_metrics,
    clip_probability,
    condition_fold,
    continuous_metrics,
    design_matrix,
    fit_glm,
    fit_ols,
    fit_accuracy_candidate,
    fit_bias_candidate,
    load_candidates,
)


MODELS = ("Gemma4-31B", "Qwen3-32B")
MODES = ("direct", "native_thinking")
OLD_LENGTHS = (1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000)
HOLDOUT_LENGTHS = (25000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000)
N_LEVELS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20)
SEEDS = tuple(range(1234, 1264))
FIXED_N = 10
FIXED_N_CANDIDATE_IDS = ("intercept", "L_k", "logL")
N_FIXED_LENGTH_TERMS = (None, "L_k", "logL")
EXPECTED_OLD_ROWS = len(MODELS) * len(MODES) * len(OLD_LENGTHS) * len(N_LEVELS) * len(SEEDS)
EXPECTED_HOLDOUT_ROWS = (
    len(MODELS) * len(MODES) * len(HOLDOUT_LENGTHS) * len(N_LEVELS) * len(SEEDS)
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def add_predictors(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["N"] = pd.to_numeric(result["N"], errors="raise").astype(int)
    result["L"] = pd.to_numeric(result["L"], errors="raise").astype(int)
    if (result["N"] <= 0).any() or (result["L"] <= 0).any():
        raise ValueError("N and L must be positive")
    result["L_k"] = result["L"] / 1000.0
    result["logN"] = np.log(result["N"].astype(float))
    result["logL"] = np.log(result["L_k"])
    result["N_x_L_k"] = result["N"] * result["L_k"]
    result["logN_x_logL"] = result["logN"] * result["logL"]
    result["N_x_logL"] = result["N"] * result["logL"]
    result["logN_x_L_k"] = result["logN"] * result["L_k"]
    result["invN"] = 1.0 / result["N"]
    result["invN_x_L_k"] = result["invN"] * result["L_k"]
    result["invN_x_logL"] = result["invN"] * result["logL"]
    return result


def normalize_requests(path: Path, source: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    aliases = {
        "signed_error": "signed_deviation",
        "absolute_error": "absolute_deviation",
    }
    for old, new in aliases.items():
        if new not in frame and old in frame:
            frame = frame.rename(columns={old: new})
    if "comparison_slot" not in frame:
        frame["comparison_slot"] = frame["model_label"]
    required = {
        "request_id",
        "model_label",
        "model_id",
        "model_revision",
        "prompt_mode",
        "seed",
        "N",
        "L",
        "predicted_count",
        "parse_success",
        "exact_count",
        "signed_deviation",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} request input is missing {missing}")
    frame = frame.loc[
        frame["model_label"].isin(MODELS) & frame["prompt_mode"].isin(MODES)
    ].copy()
    frame["parse_success"] = frame["parse_success"].astype(bool)
    frame["exact_count"] = frame["exact_count"].astype(bool)
    frame["source_range"] = source
    return add_predictors(frame)


def validate_grid(
    frame: pd.DataFrame,
    *,
    source: str,
    expected_rows: int,
    expected_lengths: tuple[int, ...],
) -> None:
    checks = {
        "rows": len(frame),
        "unique_request_ids": frame["request_id"].nunique(),
        "models": sorted(frame["model_label"].unique().tolist()),
        "modes": sorted(frame["prompt_mode"].unique().tolist()),
        "N": sorted(frame["N"].unique().tolist()),
        "L": sorted(frame["L"].unique().tolist()),
        "seeds": sorted(frame["seed"].unique().tolist()),
    }
    expected = {
        "rows": expected_rows,
        "unique_request_ids": expected_rows,
        "models": sorted(MODELS),
        "modes": sorted(MODES),
        "N": list(N_LEVELS),
        "L": list(expected_lengths),
        "seeds": list(SEEDS),
    }
    if checks != expected:
        raise ValueError(f"{source} grid audit failed: {checks} != {expected}")
    cell_counts = frame.groupby(
        ["model_label", "prompt_mode", "N", "L"], observed=True
    ).size()
    if not (cell_counts == len(SEEDS)).all():
        raise ValueError(f"{source} cells do not all contain {len(SEEDS)} seeds")


def coefficient_prediction(
    frame: pd.DataFrame, coefficient_rows: pd.DataFrame
) -> np.ndarray:
    prediction = np.zeros(len(frame), dtype=float)
    for row in coefficient_rows.itertuples(index=False):
        term = str(row.term)
        values = np.ones(len(frame)) if term == "intercept" else frame[term].to_numpy(float)
        prediction += float(row.estimate) * values
    return prediction


def n_fixed_design(
    frame: pd.DataFrame,
    n_levels: tuple[int, ...],
    length_term: str | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build a full-rank N-fixed-effect design without a global intercept."""

    observed = frame["N"].to_numpy(dtype=int)
    indicators = np.column_stack(
        [(observed == level).astype(float) for level in n_levels]
    )
    names: tuple[str, ...] = tuple(f"alpha_N={level}" for level in n_levels)
    if length_term is None:
        return indicators, names
    if length_term not in {"L_k", "logL"}:
        raise ValueError(f"Unsupported shared length term: {length_term}")
    length = frame[length_term].to_numpy(dtype=float)[:, None]
    return np.column_stack([indicators, length]), (*names, f"beta_{length_term}")


def continuous_mean_oof(y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    prediction = np.full(len(y), np.nan, dtype=float)
    for fold in range(5):
        test = folds == fold
        prediction[test] = float(np.mean(y[~test]))
    return prediction


def _coefficient_rows(
    full: Any,
    names: tuple[str, ...],
    *,
    outcome: str,
    length_term: str | None,
    model_specification: str,
    evaluation_scheme: str,
) -> list[dict[str, Any]]:
    confidence = np.asarray(full.conf_int(), dtype=float)
    return [
        {
            "outcome_family": outcome,
            "length_term": length_term or "none",
            "model_specification": model_specification,
            "evaluation_scheme": evaluation_scheme,
            "term": name,
            "estimate": float(full.params[index]),
            "standard_error": float(full.bse[index]),
            "p_value": float(full.pvalues[index]),
            "ci95_low": float(confidence[index, 0]),
            "ci95_high": float(confidence[index, 1]),
        }
        for index, name in enumerate(names)
    ]


def fit_n_fixed_accuracy_cv(
    frame: pd.DataFrame,
    length_term: str | None,
    n_levels: tuple[int, ...],
    l_levels: tuple[int, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y = frame["exact_count"].astype(float).to_numpy()
    x, names = n_fixed_design(frame, n_levels, length_term)
    x_n, _ = n_fixed_design(frame, n_levels, None)
    folds = condition_fold(frame, n_levels, l_levels)
    oof = np.full(len(frame), np.nan, dtype=float)
    n_fixed_oof = np.full(len(frame), np.nan, dtype=float)
    for fold in range(5):
        test = folds == fold
        train = ~test
        fit = fit_glm(y[train], x[train], HEADLINE_ACCURACY, robust=False)
        oof[test] = fit.predict(x[test])
        baseline_fit = fit_glm(
            y[train], x_n[train], HEADLINE_ACCURACY, robust=False
        )
        n_fixed_oof[test] = baseline_fit.predict(x_n[test])
    if np.isnan(oof).any() or np.isnan(n_fixed_oof).any():
        raise RuntimeError("N-fixed accuracy CV left missing OOF predictions")
    oof = clip_probability(oof)
    n_fixed_oof = clip_probability(n_fixed_oof)
    global_oof = accuracy_intercept_oof(frame, folds)
    metrics = binary_metrics(y, oof)
    n_baseline = binary_metrics(y, n_fixed_oof)
    global_baseline = binary_metrics(y, global_oof)
    full = fit_glm(y, x, HEADLINE_ACCURACY, robust=True)
    in_sample = binary_metrics(y, full.predict(x))
    model_specification = (
        "alpha_N" if length_term is None else f"alpha_N + beta_{length_term}"
    )
    evaluation_scheme = "combined_5fold_held_condition_cv"
    row = {
        "outcome_family": HEADLINE_ACCURACY,
        "length_term": length_term or "none",
        "model_specification": model_specification,
        "evaluation_scheme": evaluation_scheme,
        "n_rows": len(frame),
        "n_conditions": int(frame[["N", "L"]].drop_duplicates().shape[0]),
        "n_parameters": x.shape[1],
        "primary_loss": metrics["log_loss"],
        "primary_score": 1.0 - metrics["log_loss"] / global_baseline["log_loss"],
        "cv_log_loss": metrics["log_loss"],
        "cv_brier_score": metrics["brier_score"],
        "cv_d2_vs_global_intercept": 1.0
        - metrics["log_loss"] / global_baseline["log_loss"],
        "cv_d2_vs_N_fixed": 1.0
        - metrics["log_loss"] / n_baseline["log_loss"],
        "delta_log_loss_vs_N_fixed": n_baseline["log_loss"]
        - metrics["log_loss"],
        "in_sample_log_loss": in_sample["log_loss"],
        "aic": float(full.aic),
        "bic": float(getattr(full, "bic_llf", math.nan)),
    }
    return row, _coefficient_rows(
        full,
        names,
        outcome=HEADLINE_ACCURACY,
        length_term=length_term,
        model_specification=model_specification,
        evaluation_scheme=evaluation_scheme,
    )


def fit_n_fixed_continuous_cv(
    frame: pd.DataFrame,
    outcome: str,
    length_term: str | None,
    n_levels: tuple[int, ...],
    l_levels: tuple[int, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y = frame[outcome].to_numpy(dtype=float)
    x, names = n_fixed_design(frame, n_levels, length_term)
    x_n, _ = n_fixed_design(frame, n_levels, None)
    folds = condition_fold(frame, n_levels, l_levels)
    oof = np.full(len(frame), np.nan, dtype=float)
    n_fixed_oof = np.full(len(frame), np.nan, dtype=float)
    for fold in range(5):
        test = folds == fold
        train = ~test
        oof[test] = fit_ols(y[train], x[train], robust=False).predict(x[test])
        n_fixed_oof[test] = fit_ols(
            y[train], x_n[train], robust=False
        ).predict(x_n[test])
    if np.isnan(oof).any() or np.isnan(n_fixed_oof).any():
        raise RuntimeError("N-fixed continuous CV left missing OOF predictions")
    metrics = continuous_metrics(y, oof)
    n_baseline = continuous_metrics(y, n_fixed_oof)
    global_baseline = continuous_metrics(y, continuous_mean_oof(y, folds))
    full = fit_ols(y, x, robust=True)
    full_prediction = np.asarray(full.predict(x), dtype=float)
    in_sample = continuous_metrics(y, full_prediction)
    model_sse = float(np.sum(np.square(y - oof)))
    n_fixed_sse = float(np.sum(np.square(y - n_fixed_oof)))
    model_specification = (
        "alpha_N" if length_term is None else f"alpha_N + beta_{length_term}"
    )
    evaluation_scheme = "combined_5fold_held_condition_cv"
    row = {
        "outcome_family": outcome,
        "length_term": length_term or "none",
        "model_specification": model_specification,
        "evaluation_scheme": evaluation_scheme,
        "n_rows": len(frame),
        "n_conditions": int(frame[["N", "L"]].drop_duplicates().shape[0]),
        "n_parameters": x.shape[1],
        "primary_loss": metrics["mae"],
        "primary_score": metrics["r2"],
        "cv_mae": metrics["mae"],
        "cv_rmse": metrics["rmse"],
        "cv_r2_vs_global_mean": metrics["r2"],
        "cv_r2_of_global_oof_baseline": global_baseline["r2"],
        "cv_relative_sse_reduction_vs_N_fixed": (
            1.0 - model_sse / n_fixed_sse if n_fixed_sse > 0 else math.nan
        ),
        "delta_cv_mae_vs_N_fixed": n_baseline["mae"] - metrics["mae"],
        "in_sample_mae": in_sample["mae"],
        "in_sample_r2": in_sample["r2"],
        "minimum_in_sample_prediction": float(np.min(full_prediction)),
        "aic": float(full.aic),
        "bic": float(full.bic),
    }
    return row, _coefficient_rows(
        full,
        names,
        outcome=outcome,
        length_term=length_term,
        model_specification=model_specification,
        evaluation_scheme=evaluation_scheme,
    )


def fit_n_fixed_accuracy_holdout(
    train: pd.DataFrame,
    test: pd.DataFrame,
    length_term: str | None,
    n_levels: tuple[int, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y_train = train["exact_count"].astype(float).to_numpy()
    y_test = test["exact_count"].astype(float).to_numpy()
    x_train, names = n_fixed_design(train, n_levels, length_term)
    x_test, _ = n_fixed_design(test, n_levels, length_term)
    x_n_train, _ = n_fixed_design(train, n_levels, None)
    x_n_test, _ = n_fixed_design(test, n_levels, None)
    full = fit_glm(y_train, x_train, HEADLINE_ACCURACY, robust=True)
    prediction = clip_probability(full.predict(x_test))
    n_prediction = clip_probability(
        fit_glm(y_train, x_n_train, HEADLINE_ACCURACY, robust=False).predict(x_n_test)
    )
    global_prediction = np.full(len(test), float(np.mean(y_train)))
    metrics = binary_metrics(y_test, prediction)
    n_baseline = binary_metrics(y_test, n_prediction)
    global_baseline = binary_metrics(y_test, global_prediction)
    model_specification = (
        "alpha_N" if length_term is None else f"alpha_N + beta_{length_term}"
    )
    evaluation_scheme = "fit_v3_1_score_v3_3_holdout"
    row = {
        "outcome_family": HEADLINE_ACCURACY,
        "length_term": length_term or "none",
        "model_specification": model_specification,
        "evaluation_scheme": evaluation_scheme,
        "n_rows": len(test),
        "n_conditions": int(test[["N", "L"]].drop_duplicates().shape[0]),
        "n_parameters": x_train.shape[1],
        "primary_loss": metrics["log_loss"],
        "primary_score": 1.0 - metrics["log_loss"] / global_baseline["log_loss"],
        "holdout_observed_accuracy": float(np.mean(y_test)),
        "holdout_mean_predicted_accuracy": float(np.mean(prediction)),
        "holdout_log_loss": metrics["log_loss"],
        "holdout_brier_score": metrics["brier_score"],
        "holdout_d2_vs_global_train_intercept": 1.0
        - metrics["log_loss"] / global_baseline["log_loss"],
        "holdout_d2_vs_N_fixed": 1.0
        - metrics["log_loss"] / n_baseline["log_loss"],
        "delta_holdout_log_loss_vs_N_fixed": n_baseline["log_loss"]
        - metrics["log_loss"],
    }
    return row, _coefficient_rows(
        full,
        names,
        outcome=HEADLINE_ACCURACY,
        length_term=length_term,
        model_specification=model_specification,
        evaluation_scheme=evaluation_scheme,
    )


def fit_n_fixed_continuous_holdout(
    train: pd.DataFrame,
    test: pd.DataFrame,
    outcome: str,
    length_term: str | None,
    n_levels: tuple[int, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y_train = train[outcome].to_numpy(dtype=float)
    y_test = test[outcome].to_numpy(dtype=float)
    x_train, names = n_fixed_design(train, n_levels, length_term)
    x_test, _ = n_fixed_design(test, n_levels, length_term)
    x_n_train, _ = n_fixed_design(train, n_levels, None)
    x_n_test, _ = n_fixed_design(test, n_levels, None)
    full = fit_ols(y_train, x_train, robust=True)
    prediction = np.asarray(full.predict(x_test), dtype=float)
    n_prediction = np.asarray(
        fit_ols(y_train, x_n_train, robust=False).predict(x_n_test), dtype=float
    )
    global_prediction = np.full(len(test), float(np.mean(y_train)))
    metrics = continuous_metrics(y_test, prediction)
    n_baseline = continuous_metrics(y_test, n_prediction)
    global_baseline = continuous_metrics(y_test, global_prediction)
    model_sse = float(np.sum(np.square(y_test - prediction)))
    n_fixed_sse = float(np.sum(np.square(y_test - n_prediction)))
    model_specification = (
        "alpha_N" if length_term is None else f"alpha_N + beta_{length_term}"
    )
    evaluation_scheme = "fit_v3_1_score_v3_3_holdout"
    row = {
        "outcome_family": outcome,
        "length_term": length_term or "none",
        "model_specification": model_specification,
        "evaluation_scheme": evaluation_scheme,
        "n_rows": len(test),
        "n_conditions": int(test[["N", "L"]].drop_duplicates().shape[0]),
        "n_parameters": x_train.shape[1],
        "primary_loss": metrics["mae"],
        "primary_score": metrics["r2"],
        "holdout_observed_mean": float(np.mean(y_test)),
        "holdout_predicted_mean": float(np.mean(prediction)),
        "holdout_mae": metrics["mae"],
        "holdout_rmse": metrics["rmse"],
        "holdout_r2_vs_observed_holdout_mean": metrics["r2"],
        "holdout_r2_of_global_train_mean": global_baseline["r2"],
        "holdout_relative_sse_reduction_vs_N_fixed": (
            1.0 - model_sse / n_fixed_sse if n_fixed_sse > 0 else math.nan
        ),
        "delta_holdout_mae_vs_N_fixed": n_baseline["mae"] - metrics["mae"],
        "minimum_holdout_prediction": float(np.min(prediction)),
    }
    return row, _coefficient_rows(
        full,
        names,
        outcome=outcome,
        length_term=length_term,
        model_specification=model_specification,
        evaluation_scheme=evaluation_scheme,
    )


def fit_n_fixed_length_models(
    old: pd.DataFrame,
    holdout: pd.DataFrame,
    combined: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    old_cells = add_predictors(build_count_error_cells(old))
    holdout_cells = add_predictors(build_count_error_cells(holdout))
    combined_cells = add_predictors(build_count_error_cells(combined))
    metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    combined_l_levels = tuple(sorted(combined["L"].unique().astype(int)))
    for model in MODELS:
        for mode in MODES:
            request_sets = {
                name: frame.loc[
                    frame["model_label"].eq(model)
                    & frame["prompt_mode"].eq(mode)
                ]
                for name, frame in {
                    "old": old,
                    "holdout": holdout,
                    "combined": combined,
                }.items()
            }
            cell_sets = {
                name: frame.loc[
                    frame["comparison_slot"].eq(model)
                    & frame["prompt_mode"].eq(mode)
                    & frame["bias_law_eligible"].astype(bool)
                ]
                for name, frame in {
                    "old": old_cells,
                    "holdout": holdout_cells,
                    "combined": combined_cells,
                }.items()
            }
            for outcome in (HEADLINE_ACCURACY, MAE_FAMILY, BIAS_FAMILY):
                for length_term in N_FIXED_LENGTH_TERMS:
                    prefix = {"model_label": model, "prompt_mode": mode}
                    try:
                        if outcome == HEADLINE_ACCURACY:
                            cv_metric, cv_coef = fit_n_fixed_accuracy_cv(
                                request_sets["combined"],
                                length_term,
                                N_LEVELS,
                                combined_l_levels,
                            )
                            hold_metric, hold_coef = fit_n_fixed_accuracy_holdout(
                                request_sets["old"],
                                request_sets["holdout"],
                                length_term,
                                N_LEVELS,
                            )
                        else:
                            cv_metric, cv_coef = fit_n_fixed_continuous_cv(
                                cell_sets["combined"],
                                outcome,
                                length_term,
                                N_LEVELS,
                                combined_l_levels,
                            )
                            hold_metric, hold_coef = fit_n_fixed_continuous_holdout(
                                cell_sets["old"],
                                cell_sets["holdout"],
                                outcome,
                                length_term,
                                N_LEVELS,
                            )
                        metric_rows.extend(
                            [{**prefix, **cv_metric}, {**prefix, **hold_metric}]
                        )
                        coefficient_rows.extend(
                            {**prefix, **row} for row in (*cv_coef, *hold_coef)
                        )
                    except Exception as exc:
                        failures.append(
                            {
                                **prefix,
                                "outcome_family": outcome,
                                "length_term": length_term or "none",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(failures),
    )


def evaluate_frozen_accuracy(
    holdout: pd.DataFrame, coefficients: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for mode in MODES:
            block = holdout.loc[
                holdout["model_label"].eq(model) & holdout["prompt_mode"].eq(mode)
            ]
            coef = coefficients.loc[
                coefficients["comparison_slot"].eq(model)
                & coefficients["prompt_mode"].eq(mode)
                & coefficients["outcome_family"].eq(HEADLINE_ACCURACY)
            ]
            if coef.empty:
                raise ValueError(f"Missing frozen accuracy coefficients for {model}/{mode}")
            probability = expit(coefficient_prediction(block, coef))
            for length, part in [("all", block), *list(block.groupby("L", sort=True))]:
                index = part.index
                position = block.index.get_indexer(index)
                metrics = binary_metrics(
                    part["exact_count"].astype(float).to_numpy(), probability[position]
                )
                rows.append(
                    {
                        "model_label": model,
                        "prompt_mode": mode,
                        "L": length,
                        "candidate": str(coef["candidate"].iloc[0]),
                        "n_requests": len(part),
                        "observed_accuracy": float(part["exact_count"].mean()),
                        "mean_predicted_probability": float(np.mean(probability[position])),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def evaluate_frozen_mae(
    holdout_cells: pd.DataFrame, coefficients: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eligible = holdout_cells.loc[holdout_cells["mae_law_eligible"].astype(bool)]
    for model in MODELS:
        for mode in MODES:
            block = eligible.loc[
                eligible["comparison_slot"].eq(model)
                & eligible["prompt_mode"].eq(mode)
            ]
            coef = coefficients.loc[
                coefficients["comparison_slot"].eq(model)
                & coefficients["prompt_mode"].eq(mode)
                & coefficients["outcome_family"].eq(MAE_FAMILY)
            ]
            if coef.empty:
                raise ValueError(f"Missing frozen MAE coefficients for {model}/{mode}")
            prediction = coefficient_prediction(block, coef)
            for length, part in [("all", block), *list(block.groupby("L", sort=True))]:
                index = part.index
                position = block.index.get_indexer(index)
                observed = part[MAE_FAMILY].to_numpy(float)
                predicted = prediction[position]
                metrics = continuous_metrics(observed, predicted)
                rows.append(
                    {
                        "model_label": model,
                        "prompt_mode": mode,
                        "L": length,
                        "candidate": str(coef["candidate"].iloc[0]),
                        "n_cells": len(part),
                        "observed_mean": float(np.mean(observed)),
                        "predicted_mean": float(np.mean(predicted)),
                        "mean_absolute_prediction_error": float(
                            np.mean(np.abs(observed - predicted))
                        ),
                        "minimum_prediction": float(np.min(predicted)),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def fit_one_candidate(
    requests: pd.DataFrame,
    cells: pd.DataFrame,
    candidate: Candidate,
    outcome: str,
    n_levels: tuple[int, ...],
    l_levels: tuple[int, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if outcome == HEADLINE_ACCURACY:
        return fit_accuracy_candidate(
            requests, candidate, HEADLINE_ACCURACY, n_levels, l_levels
        )
    eligible = cells.loc[cells["bias_law_eligible"].astype(bool)]
    if outcome == BIAS_FAMILY:
        return fit_bias_candidate(eligible, candidate, n_levels, l_levels)
    if outcome == MAE_FAMILY:
        return fit_continuous_candidate(
            eligible,
            candidate,
            outcome_family=MAE_FAMILY,
            outcome_column=MAE_FAMILY,
            n_levels=n_levels,
            l_levels=l_levels,
        )
    raise ValueError(outcome)


def scan_candidates(
    requests: pd.DataFrame,
    candidates: tuple[Candidate, ...],
    *,
    dataset: str,
    fixed_n: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_requests = requests if fixed_n is None else requests.loc[requests["N"].eq(fixed_n)]
    # build_count_error_cells preserves the original V3.2 predictors but the
    # inverse-N extension was added later.  Recompute the full predictor set on
    # the aggregated cell table so accuracy, MAE, and bias see the same 18-law
    # registry.
    cells = add_predictors(build_count_error_cells(selected_requests))
    metrics: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    n_levels = tuple(sorted(selected_requests["N"].unique().astype(int).tolist()))
    l_levels = tuple(sorted(selected_requests["L"].unique().astype(int).tolist()))
    for model in MODELS:
        for mode in MODES:
            request_block = selected_requests.loc[
                selected_requests["model_label"].eq(model)
                & selected_requests["prompt_mode"].eq(mode)
            ]
            cell_block = cells.loc[
                cells["comparison_slot"].eq(model) & cells["prompt_mode"].eq(mode)
            ]
            for outcome in (HEADLINE_ACCURACY, MAE_FAMILY, BIAS_FAMILY):
                for candidate in candidates:
                    try:
                        metric, terms = fit_one_candidate(
                            request_block,
                            cell_block,
                            candidate,
                            outcome,
                            n_levels,
                            l_levels,
                        )
                        prefix = {
                            "dataset": dataset,
                            "model_label": model,
                            "prompt_mode": mode,
                            "fixed_N": fixed_n,
                        }
                        metrics.append({**prefix, **metric})
                        coefficients.extend({**prefix, **term} for term in terms)
                    except Exception as exc:  # retain complete failure accounting
                        failures.append(
                            {
                                "dataset": dataset,
                                "model_label": model,
                                "prompt_mode": mode,
                                "fixed_N": fixed_n,
                                "outcome_family": outcome,
                                "candidate": candidate.id,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
    return pd.DataFrame(metrics), pd.DataFrame(coefficients), pd.DataFrame(failures)


def select_candidates(metrics: pd.DataFrame, candidates: tuple[Candidate, ...]) -> pd.DataFrame:
    rank = {candidate.id: index for index, candidate in enumerate(candidates)}
    rows: list[dict[str, Any]] = []
    keys = ["dataset", "model_label", "prompt_mode", "fixed_N", "outcome_family"]
    for key, block in metrics.groupby(keys, dropna=False, sort=True):
        block = block.loc[np.isfinite(block["primary_score"])].copy()
        if block.empty:
            continue
        best_score = float(block["primary_score"].max())
        near = block.loc[block["primary_score"] >= best_score - 0.02].copy()
        near["registry_order"] = near["candidate"].map(rank)
        winner = near.sort_values(
            ["predictors", "primary_loss", "primary_score", "registry_order"],
            ascending=[True, True, False, True],
        ).iloc[0]
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "selected_candidate": winner["candidate"],
                "selected_cv_score": float(winner["primary_score"]),
                "selected_cv_loss": float(winner["primary_loss"]),
                "best_cv_score": best_score,
                "near_best_tolerance": 0.02,
                "selection_rule": "best CV score minus 0.02, then fewer terms/lower loss/registry order",
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-input", type=Path, required=True)
    parser.add_argument("--holdout-input", type=Path, required=True)
    parser.add_argument("--accuracy-coefficients", type=Path, required=True)
    parser.add_argument("--mae-coefficients", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--inverse-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    write_json(output / "analysis_state.json", {"stage": "loading", "updated_at_utc": utc_now()})

    old = normalize_requests(args.old_input.resolve(), "v3_1_1k_20k")
    holdout = normalize_requests(args.holdout_input.resolve(), "v3_3_25k_100k")
    validate_grid(old, source="v3_1_1k_20k", expected_rows=EXPECTED_OLD_ROWS, expected_lengths=OLD_LENGTHS)
    validate_grid(
        holdout,
        source="v3_3_25k_100k",
        expected_rows=EXPECTED_HOLDOUT_ROWS,
        expected_lengths=HOLDOUT_LENGTHS,
    )
    combined = pd.concat([old, holdout], ignore_index=True, sort=False)
    if combined["request_id"].duplicated().any():
        raise ValueError("Duplicate request IDs after combining V3.1 and V3.3")

    base_config = json.loads(args.base_config.read_text(encoding="utf-8"))
    candidates = load_candidates(base_config) + load_inverse_candidates(args.inverse_config)
    candidate_map = {candidate.id: candidate for candidate in candidates}
    fixed_candidates = tuple(candidate_map[item] for item in FIXED_N_CANDIDATE_IDS)

    holdout_cells = build_count_error_cells(holdout)
    accuracy_coefficients = pd.read_csv(args.accuracy_coefficients)
    mae_coefficients = pd.read_csv(args.mae_coefficients)
    frozen_accuracy = evaluate_frozen_accuracy(holdout, accuracy_coefficients)
    frozen_mae = evaluate_frozen_mae(holdout_cells, mae_coefficients)
    frozen_accuracy.to_csv(tables / "frozen_v3_2_accuracy_holdout_metrics.csv", index=False)
    frozen_mae.to_csv(tables / "frozen_v3_2_mae_holdout_metrics.csv", index=False)

    write_json(output / "analysis_state.json", {"stage": "candidate_scan", "updated_at_utc": utc_now()})
    scan_parts = []
    coefficient_parts = []
    failure_parts = []
    for dataset, frame in (("holdout_only", holdout), ("combined_1k_100k", combined)):
        metric, coefficient, failure = scan_candidates(
            frame, candidates, dataset=dataset
        )
        scan_parts.append(metric)
        coefficient_parts.append(coefficient)
        failure_parts.append(failure)

    write_json(output / "analysis_state.json", {"stage": "fixed_N10_scan", "updated_at_utc": utc_now()})
    fixed_metrics, fixed_coefficients, fixed_failures = scan_candidates(
        combined,
        fixed_candidates,
        dataset="combined_1k_100k_fixed_N10",
        fixed_n=FIXED_N,
    )
    scan_parts.append(fixed_metrics)
    coefficient_parts.append(fixed_coefficients)
    failure_parts.append(fixed_failures)

    write_json(
        output / "analysis_state.json",
        {"stage": "N_fixed_shared_length", "updated_at_utc": utc_now()},
    )
    n_fixed_metrics, n_fixed_coefficients, n_fixed_failures = (
        fit_n_fixed_length_models(old, holdout, combined)
    )

    metrics = pd.concat(scan_parts, ignore_index=True)
    coefficients = pd.concat(coefficient_parts, ignore_index=True)
    nonempty_failures = [part for part in failure_parts if not part.empty]
    failures = (
        pd.concat(nonempty_failures, ignore_index=True)
        if nonempty_failures
        else pd.DataFrame(columns=["dataset", "model_label", "prompt_mode", "fixed_N", "outcome_family", "candidate", "error"])
    )
    selections = pd.concat(
        [
            select_candidates(metrics.loc[metrics["dataset"].eq("holdout_only")], candidates),
            select_candidates(metrics.loc[metrics["dataset"].eq("combined_1k_100k")], candidates),
            select_candidates(
                metrics.loc[metrics["dataset"].eq("combined_1k_100k_fixed_N10")],
                fixed_candidates,
            ),
        ],
        ignore_index=True,
    )
    metrics.to_csv(tables / "exploratory_candidate_metrics.csv", index=False)
    coefficients.to_csv(tables / "exploratory_candidate_coefficients.csv", index=False)
    failures.to_csv(tables / "exploratory_candidate_failures.csv", index=False)
    selections.to_csv(tables / "exploratory_selected_laws.csv", index=False)
    n_fixed_metrics.to_csv(tables / "n_fixed_shared_length_metrics.csv", index=False)
    n_fixed_coefficients.to_csv(
        tables / "n_fixed_shared_length_coefficients.csv", index=False
    )
    n_fixed_failures.to_csv(
        tables / "n_fixed_shared_length_failures.csv", index=False
    )

    manifest = {
        "schema_version": "realistic_niah_v3_3_regression_scan_v2",
        "completed_at_utc": utc_now(),
        "confirmatory_analysis": "Frozen V3.2 accuracy and trimmed-MAE coefficients scored on the untouched 25k-100k holdout",
        "exploratory_analysis": [
            "separate 18-candidate scans for each model and prompt mode on holdout only",
            "separate 18-candidate scans after adding the holdout to V3.1",
            "fixed N=10 length-only scan over intercept, L/1000, and ln(L/1000)",
            "N fixed effects with a shared L/1000 or ln(L/1000) slope, evaluated by combined-range held-condition CV",
            "the same N-fixed shared-slope forms fitted on V3.1 and scored on the V3.3 long-context holdout",
        ],
        "selection": {
            "folds": 5,
            "rule": "V3.2 held-condition folds; within 0.02 of best CV score choose simpler candidate",
            "bootstrap_repetitions": 0,
            "fixed_N10_candidates": list(FIXED_N_CANDIDATE_IDS),
            "N_fixed_shared_length_candidates": [
                "alpha_N",
                "alpha_N + beta_L_k",
                "alpha_N + beta_logL",
            ],
        },
        "estimands": {
            "accuracy": "request-level exact-count Bernoulli-logit",
            "mae": "cell-level symmetric 10%-trimmed conditional MAE on parseable responses",
            "bias": "cell-level symmetric 10%-trimmed signed error on parseable responses",
        },
        "inputs": {
            "old": {"path": args.old_input.resolve(), "sha256": file_sha256(args.old_input.resolve()), "rows": len(old)},
            "holdout": {"path": args.holdout_input.resolve(), "sha256": file_sha256(args.holdout_input.resolve()), "rows": len(holdout)},
            "accuracy_coefficients_sha256": file_sha256(args.accuracy_coefficients.resolve()),
            "mae_coefficients_sha256": file_sha256(args.mae_coefficients.resolve()),
            "base_config_sha256": file_sha256(args.base_config.resolve()),
            "inverse_config_sha256": file_sha256(args.inverse_config.resolve()),
        },
        "outputs": {
            "candidate_metric_rows": len(metrics),
            "candidate_coefficient_rows": len(coefficients),
            "candidate_failures": len(failures),
            "selected_laws": len(selections),
            "N_fixed_metric_rows": len(n_fixed_metrics),
            "N_fixed_coefficient_rows": len(n_fixed_coefficients),
            "N_fixed_failures": len(n_fixed_failures),
        },
    }
    write_json(output / "analysis_manifest.json", manifest)
    write_json(
        output / "analysis_state.json",
        {"stage": "complete", "updated_at_utc": utc_now(), **manifest["outputs"]},
    )
    print(json.dumps(json_safe(manifest["outputs"]), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
