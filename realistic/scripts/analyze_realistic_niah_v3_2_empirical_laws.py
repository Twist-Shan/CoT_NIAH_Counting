#!/usr/bin/env python3
"""Run the frozen Realistic NiaH V3.2 focused empirical-law analysis.

The script implements the contract in
``configs/realistic_niah_v3_2_empirical_law_analysis.json``.  It deliberately
contains no bootstrap or nested held-seed/N/L reselection.  Accuracy candidates
are Bernoulli GLMs with logit, probit, and cloglog links; conditional bias is
the eligible cell-level 10% trimmed signed bias fit by OLS.  Model selection
uses the earlier focused empirical-law held-condition procedure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.optimize import minimize
from scipy.special import betaln, expit, gammaln
import statsmodels
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "outputs"
    / "anvil_realistic_niah_v3_1_20260819_formal"
    / "analysis"
    / "v3_1_behavior_empirical_law"
    / "tables"
    / "request_level.csv.gz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "anvil_realistic_niah_v3_1_20260819_formal"
    / "analysis"
    / "v3_2_empirical_law"
)
DEFAULT_CONFIG = (
    ROOT / "configs" / "realistic_niah_v3_2_empirical_law_analysis.json"
)
DEFAULT_FREEZE = (
    ROOT
    / "configs"
    / "realistic_niah_v3_2_empirical_law_analysis.freeze.json"
)

ACCURACY_FAMILIES = (
    "accuracy_bernoulli_logit",
    "accuracy_bernoulli_probit",
    "accuracy_bernoulli_cloglog",
)
BIAS_FAMILY = "trimmed_signed_bias_10"
HEADLINE_ACCURACY = "accuracy_bernoulli_logit"
LOMO_FAMILIES = (HEADLINE_ACCURACY, BIAS_FAMILY)

PRACTICAL_EFFECT_THRESHOLD = 0.10
SPECIAL_SIGNIFICANT_FRACTION_THRESHOLD = 0.50
SPECIAL_CV_GAIN_THRESHOLD = 0.02
SPECIAL_CV_GAIN_Q_THRESHOLD = 0.05
MEDIAN_CV_TOLERANCE = 0.02
Q25_CV_TOLERANCE = 0.05
PROBABILITY_EPSILON = 1e-9


@dataclass(frozen=True)
class Candidate:
    id: str
    terms: tuple[str, ...]
    parent: str | None = None
    interaction: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Convert analysis objects to strict, portable JSON values.

    Missing estimates are expected for deliberately skipped robustness checks
    (for example, a benchmark run without LOMO).  JSON has no NaN/Inf values,
    so represent those as ``null`` instead of emitting non-standard JSON.
    """
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def write_state(output: Path, **payload: Any) -> None:
    write_json(
        output / "analysis_state.json",
        {
            "schema_version": "realistic_niah_v3_2_analysis_state_v1",
            "updated_at_utc": utc_now(),
            **payload,
        },
    )


def verify_freeze(config_path: Path, freeze_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    for relative, expected in freeze["files"].items():
        observed = file_sha256(ROOT / relative)
        if observed != expected:
            raise ValueError(
                f"V3.2 freeze mismatch for {relative}: {observed} != {expected}"
            )
    if config["analysis_version"] != "V3.2":
        raise ValueError("Analysis config is not V3.2")
    return config


def load_candidates(config: dict[str, Any]) -> tuple[Candidate, ...]:
    candidates = tuple(
        Candidate(
            id=str(item["id"]),
            terms=tuple(str(term) for term in item["terms"]),
            parent=(str(item["parent"]) if item.get("parent") else None),
            interaction=(
                str(item["interaction"]) if item.get("interaction") else None
            ),
        )
        for item in config["candidate_registry"]
    )
    if len(candidates) != 13 or candidates[0].id != "intercept":
        raise ValueError("Unexpected V3.2 candidate registry")
    return candidates


def validate_requests(requests: pd.DataFrame, config: dict[str, Any]) -> None:
    required = {
        "request_id",
        "comparison_slot",
        "prompt_mode",
        "seed",
        "N",
        "L",
        "predicted_count",
        "parse_success",
        "exact_count",
        "signed_deviation",
        "L_k",
        "logN",
        "logL",
        "N_x_L_k",
        "logN_x_logL",
        "N_x_logL",
        "logN_x_L_k",
    }
    missing = sorted(required.difference(requests.columns))
    if missing:
        raise ValueError(f"Request table is missing required columns: {missing}")
    immutable = config["immutable_input"]
    checks = {
        "requests": len(requests),
        "unique_request_ids": requests["request_id"].nunique(),
        "comparison_slots": requests["comparison_slot"].nunique(),
        "model_mode_slots": requests[
            ["comparison_slot", "prompt_mode"]
        ].drop_duplicates().shape[0],
        "N_levels": sorted(requests["N"].astype(int).unique().tolist()),
        "L_levels": sorted(requests["L"].astype(int).unique().tolist()),
        "prompt_modes": sorted(requests["prompt_mode"].astype(str).unique()),
    }
    expected = {
        "requests": immutable["requests"],
        "unique_request_ids": immutable["requests"],
        "comparison_slots": immutable["comparison_slots"],
        "model_mode_slots": immutable["model_mode_slots"],
        "N_levels": sorted(immutable["N_levels"]),
        "L_levels": sorted(immutable["L_levels"]),
        "prompt_modes": sorted(immutable["prompt_modes"]),
    }
    if checks != expected:
        raise ValueError(f"V3.2 immutable-input audit failed: {checks} != {expected}")
    if requests["request_id"].duplicated().any():
        raise ValueError("Duplicate request IDs")


def symmetric_trimmed_mean(values: Iterable[float]) -> float:
    array = np.sort(np.asarray(list(values), dtype=float))
    array = array[np.isfinite(array)]
    if not len(array):
        return math.nan
    trim = int(math.floor(0.10 * len(array)))
    kept = array[trim : len(array) - trim] if trim else array
    return float(np.mean(kept))


def build_cells(requests: pd.DataFrame) -> pd.DataFrame:
    keys = ["comparison_slot", "prompt_mode", "N", "L"]
    rows: list[dict[str, Any]] = []
    for key, group in requests.groupby(keys, sort=True, observed=True):
        parsed = group.loc[group["parse_success"].astype(bool), "signed_deviation"]
        parsed_values = parsed.dropna().to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "n_total": int(len(group)),
                "n_correct": int(group["exact_count"].astype(bool).sum()),
                "n_parseable": int(len(parsed_values)),
                "parsed_exact_accuracy": float(group["exact_count"].astype(bool).mean()),
                "trim_count_each_tail": int(math.floor(0.10 * len(parsed_values))),
                "trimmed_signed_bias_10": symmetric_trimmed_mean(parsed_values),
                "bias_law_eligible": bool(len(parsed_values) >= 20),
                "L_k": float(group["L_k"].iloc[0]),
                "logN": float(group["logN"].iloc[0]),
                "logL": float(group["logL"].iloc[0]),
                "N_x_L_k": float(group["N_x_L_k"].iloc[0]),
                "logN_x_logL": float(group["logN_x_logL"].iloc[0]),
                "N_x_logL": float(group["N_x_logL"].iloc[0]),
                "logN_x_L_k": float(group["logN_x_L_k"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def bh_adjust(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().astype(float)
    if valid.empty:
        return result
    order = np.argsort(valid.to_numpy())
    ranked = valid.to_numpy()[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result.loc[valid.index[order]] = adjusted
    return result


def condition_fold(frame: pd.DataFrame, n_levels: tuple[int, ...], l_levels: tuple[int, ...]) -> np.ndarray:
    n_index = {value: index for index, value in enumerate(n_levels)}
    l_index = {value: index for index, value in enumerate(l_levels)}
    return np.asarray(
        [
            (n_index[int(n)] + l_index[int(length)]) % 5
            for n, length in zip(frame["N"], frame["L"], strict=True)
        ],
        dtype=int,
    )


def design_matrix(frame: pd.DataFrame, candidate: Candidate) -> np.ndarray:
    if not candidate.terms:
        return np.ones((len(frame), 1), dtype=float)
    values = frame.loc[:, list(candidate.terms)].to_numpy(dtype=float)
    return np.column_stack([np.ones(len(frame), dtype=float), values])


def link_for_family(family: str) -> Any:
    if family.endswith("_logit"):
        return sm.families.links.Logit()
    if family.endswith("_probit"):
        return sm.families.links.Probit()
    if family.endswith("_cloglog"):
        return sm.families.links.CLogLog()
    raise ValueError(f"Unknown accuracy family: {family}")


def clip_probability(value: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(value, dtype=float), PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)


def binary_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    p = clip_probability(probability)
    observed = np.asarray(y, dtype=float)
    log_loss = -float(np.mean(observed * np.log(p) + (1.0 - observed) * np.log1p(-p)))
    brier = float(np.mean(np.square(observed - p)))
    accuracy = float(np.mean((p >= 0.5) == (observed >= 0.5)))
    calibration_intercept = math.nan
    calibration_slope = math.nan
    if 0 < observed.sum() < len(observed):
        logit_p = np.log(p / (1.0 - p))
        try:
            calibration = sm.GLM(
                observed,
                np.column_stack([np.ones(len(observed)), logit_p]),
                family=sm.families.Binomial(),
            ).fit(maxiter=200, disp=0)
            calibration_intercept = float(calibration.params[0])
            calibration_slope = float(calibration.params[1])
        except Exception:
            pass
    return {
        "log_loss": log_loss,
        "brier_score": brier,
        "accuracy": accuracy,
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
    }


def continuous_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    observed = np.asarray(y, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    residual = observed - predicted
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    denominator = float(np.sum(np.square(observed - observed.mean())))
    r2 = 1.0 - float(np.sum(np.square(residual))) / denominator if denominator > 0 else math.nan
    return {"mae": mae, "rmse": rmse, "r2": r2}


def fit_glm(y: np.ndarray, x: np.ndarray, family: str, *, robust: bool) -> Any:
    model = sm.GLM(y, x, family=sm.families.Binomial(link=link_for_family(family)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(
            maxiter=300,
            tol=1e-10,
            disp=0,
            cov_type=("HC3" if robust else "nonrobust"),
        )


def fit_ols(y: np.ndarray, x: np.ndarray, *, robust: bool) -> Any:
    return sm.OLS(y, x).fit(cov_type=("HC3" if robust else "nonrobust"))


def accuracy_intercept_oof(
    frame: pd.DataFrame,
    folds: np.ndarray,
) -> np.ndarray:
    y = frame["exact_count"].astype(float).to_numpy()
    prediction = np.full(len(frame), np.nan, dtype=float)
    for fold in range(5):
        test = folds == fold
        train = ~test
        probability = float(np.mean(y[train]))
        prediction[test] = probability
    return clip_probability(prediction)


def fit_accuracy_candidate(
    frame: pd.DataFrame,
    candidate: Candidate,
    family: str,
    n_levels: tuple[int, ...],
    l_levels: tuple[int, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y = frame["exact_count"].astype(float).to_numpy()
    x = design_matrix(frame, candidate)
    folds = condition_fold(frame, n_levels, l_levels)
    baseline = accuracy_intercept_oof(frame, folds)
    oof = np.full(len(frame), np.nan, dtype=float)
    fold_failures = 0
    for fold in range(5):
        test = folds == fold
        train = ~test
        try:
            result = fit_glm(y[train], x[train], family, robust=False)
            oof[test] = result.predict(x[test])
        except Exception:
            fold_failures += 1
    if np.isnan(oof).any():
        raise RuntimeError(
            f"{family}/{candidate.id} left {int(np.isnan(oof).sum())} OOF rows"
        )
    full = fit_glm(y, x, family, robust=True)
    in_sample = binary_metrics(y, full.predict(x))
    cv = binary_metrics(y, oof)
    baseline_metrics = binary_metrics(y, baseline)
    cv_d2 = 1.0 - cv["log_loss"] / baseline_metrics["log_loss"]
    in_sample_baseline = np.full(len(y), y.mean(), dtype=float)
    in_sample_d2 = 1.0 - in_sample["log_loss"] / binary_metrics(
        y, in_sample_baseline
    )["log_loss"]
    metrics = {
        "outcome_family": family,
        "candidate": candidate.id,
        "predictors": len(candidate.terms),
        "parent": candidate.parent,
        "interaction": candidate.interaction,
        "n_rows": int(len(frame)),
        "n_conditions": int(frame[["N", "L"]].drop_duplicates().shape[0]),
        "converged": bool(getattr(full, "converged", True)),
        "fold_failures": fold_failures,
        "primary_loss": cv["log_loss"],
        "primary_score": cv_d2,
        "cv_log_loss": cv["log_loss"],
        "cv_d2": cv_d2,
        "cv_brier_score": cv["brier_score"],
        "cv_accuracy": cv["accuracy"],
        "cv_calibration_intercept": cv["calibration_intercept"],
        "cv_calibration_slope": cv["calibration_slope"],
        "in_sample_log_loss": in_sample["log_loss"],
        "in_sample_d2": in_sample_d2,
        "in_sample_brier_score": in_sample["brier_score"],
        "aic": float(full.aic),
        "bic": float(getattr(full, "bic_llf", math.nan)),
    }
    coefficients: list[dict[str, Any]] = []
    term_names = ("intercept", *candidate.terms)
    for index, term in enumerate(term_names):
        effect = float(full.params[index])
        if term == "intercept":
            standardized = math.nan
        else:
            standardized = effect * float(frame[term].std(ddof=0))
        coefficients.append(
            {
                "outcome_family": family,
                "candidate": candidate.id,
                "term": term,
                "estimate": effect,
                "standard_error": float(full.bse[index]),
                "p_value": float(full.pvalues[index]),
                "ci95_low": float(full.conf_int()[index, 0]),
                "ci95_high": float(full.conf_int()[index, 1]),
                "standardized_effect": standardized,
            }
        )
    return metrics, coefficients


def fit_bias_candidate(
    frame: pd.DataFrame,
    candidate: Candidate,
    n_levels: tuple[int, ...],
    l_levels: tuple[int, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y = frame["trimmed_signed_bias_10"].to_numpy(dtype=float)
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
            result = fit_ols(y[train], x[train], robust=False)
            oof[test] = result.predict(x[test])
        except Exception:
            fold_failures += 1
    if np.isnan(oof).any():
        raise RuntimeError(
            f"bias/{candidate.id} left {int(np.isnan(oof).sum())} OOF rows"
        )
    full = fit_ols(y, x, robust=True)
    cv = continuous_metrics(y, oof)
    in_sample = continuous_metrics(y, full.predict(x))
    metrics = {
        "outcome_family": BIAS_FAMILY,
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
    }
    coefficients: list[dict[str, Any]] = []
    term_names = ("intercept", *candidate.terms)
    outcome_sd = float(np.std(y, ddof=0))
    for index, term in enumerate(term_names):
        effect = float(full.params[index])
        if term == "intercept" or outcome_sd <= 0:
            standardized = math.nan
        else:
            standardized = (
                effect * float(frame[term].std(ddof=0)) / outcome_sd
            )
        coefficients.append(
            {
                "outcome_family": BIAS_FAMILY,
                "candidate": candidate.id,
                "term": term,
                "estimate": effect,
                "standard_error": float(full.bse[index]),
                "p_value": float(full.pvalues[index]),
                "ci95_low": float(full.conf_int()[index, 0]),
                "ci95_high": float(full.conf_int()[index, 1]),
                "standardized_effect": standardized,
            }
        )
    return metrics, coefficients


def fit_slot_task(payload: dict[str, Any]) -> dict[str, Any]:
    slot = str(payload["slot"])
    mode = str(payload["mode"])
    requests = payload["requests"]
    cells = payload["cells"]
    candidates = tuple(Candidate(**item) for item in payload["candidates"])
    n_levels = tuple(payload["n_levels"])
    l_levels = tuple(payload["l_levels"])
    metrics: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for family in ACCURACY_FAMILIES:
        for candidate in candidates:
            try:
                fit, terms = fit_accuracy_candidate(
                    requests, candidate, family, n_levels, l_levels
                )
                metrics.append({"comparison_slot": slot, "prompt_mode": mode, **fit})
                coefficients.extend(
                    {"comparison_slot": slot, "prompt_mode": mode, **term}
                    for term in terms
                )
            except Exception as error:
                failures.append(
                    {
                        "comparison_slot": slot,
                        "prompt_mode": mode,
                        "outcome_family": family,
                        "candidate": candidate.id,
                        "error": repr(error),
                    }
                )
    eligible = cells.loc[cells["bias_law_eligible"].astype(bool)].copy()
    for candidate in candidates:
        try:
            fit, terms = fit_bias_candidate(
                eligible, candidate, n_levels, l_levels
            )
            metrics.append({"comparison_slot": slot, "prompt_mode": mode, **fit})
            coefficients.extend(
                {"comparison_slot": slot, "prompt_mode": mode, **term}
                for term in terms
            )
        except Exception as error:
            failures.append(
                {
                    "comparison_slot": slot,
                    "prompt_mode": mode,
                    "outcome_family": BIAS_FAMILY,
                    "candidate": candidate.id,
                    "error": repr(error),
                }
            )
    return {
        "slot": slot,
        "mode": mode,
        "metrics": metrics,
        "coefficients": coefficients,
        "failures": failures,
    }


def apply_coefficient_bh(coefficients: pd.DataFrame) -> pd.DataFrame:
    result = coefficients.copy()
    result["hc3_q"] = np.nan
    non_intercept = result["term"].ne("intercept")
    result.loc[non_intercept, "hc3_q"] = (
        result.loc[non_intercept]
        .groupby(
            ["outcome_family", "prompt_mode", "candidate", "term"],
            group_keys=False,
        )["p_value"]
        .apply(bh_adjust)
        .reindex(result.loc[non_intercept].index)
    )
    return result


def summarize_candidates(
    metrics: pd.DataFrame,
    coefficients: pd.DataFrame,
    candidates: tuple[Candidate, ...],
    *,
    slots: set[str] | None = None,
) -> pd.DataFrame:
    fit_subset = metrics.copy()
    coefficient_subset = coefficients.copy()
    if slots is not None:
        fit_subset = fit_subset.loc[fit_subset["comparison_slot"].isin(slots)]
        coefficient_subset = coefficient_subset.loc[
            coefficient_subset["comparison_slot"].isin(slots)
        ]
        coefficient_subset = apply_coefficient_bh(coefficient_subset)
    rows: list[dict[str, Any]] = []
    families = sorted(fit_subset["outcome_family"].unique())
    modes = sorted(fit_subset["prompt_mode"].unique())
    for family in families:
        for mode in modes:
            for candidate in candidates:
                block = fit_subset.loc[
                    fit_subset["outcome_family"].eq(family)
                    & fit_subset["prompt_mode"].eq(mode)
                    & fit_subset["candidate"].eq(candidate.id)
                ]
                if block.empty:
                    continue
                effects = coefficient_subset.loc[
                    coefficient_subset["outcome_family"].eq(family)
                    & coefficient_subset["prompt_mode"].eq(mode)
                    & coefficient_subset["candidate"].eq(candidate.id)
                    & coefficient_subset["term"].ne("intercept")
                ]
                by_term = effects.groupby("term")["standardized_effect"].apply(
                    lambda value: float(np.nanmedian(np.abs(value)))
                )
                minimum_effect = float(by_term.min()) if len(by_term) else math.nan
                special_effect = math.nan
                significant_fraction = math.nan
                if candidate.interaction:
                    special = effects.loc[effects["term"].eq(candidate.interaction)]
                    special_effect = float(
                        np.nanmedian(np.abs(special["standardized_effect"]))
                    )
                    significant_fraction = float(np.mean(special["hc3_q"] < 0.05))
                cv_gains = np.asarray([], dtype=float)
                gain_p = math.nan
                if candidate.parent:
                    parent = fit_subset.loc[
                        fit_subset["outcome_family"].eq(family)
                        & fit_subset["prompt_mode"].eq(mode)
                        & fit_subset["candidate"].eq(candidate.parent),
                        ["comparison_slot", "primary_score"],
                    ].rename(columns={"primary_score": "parent_score"})
                    gains = block[["comparison_slot", "primary_score"]].merge(
                        parent,
                        on="comparison_slot",
                        how="inner",
                        validate="one_to_one",
                    )
                    cv_gains = (
                        gains["primary_score"] - gains["parent_score"]
                    ).to_numpy(dtype=float)
                    valid = cv_gains[np.isfinite(cv_gains)]
                    if len(valid) and np.any(np.abs(valid) > 1e-12):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            gain_p = float(
                                stats.wilcoxon(
                                    valid,
                                    alternative="greater",
                                    zero_method="wilcox",
                                ).pvalue
                            )
                rows.append(
                    {
                        "outcome_family": family,
                        "prompt_mode": mode,
                        "candidate": candidate.id,
                        "predictors": len(candidate.terms),
                        "parent": candidate.parent,
                        "interaction": candidate.interaction,
                        "models": int(block["comparison_slot"].nunique()),
                        "median_primary_score": float(block["primary_score"].median()),
                        "q25_primary_score": float(block["primary_score"].quantile(0.25)),
                        "mean_primary_score": float(block["primary_score"].mean()),
                        "median_primary_loss": float(block["primary_loss"].median()),
                        "mean_primary_loss": float(block["primary_loss"].mean()),
                        "minimum_term_median_abs_standardized_effect": minimum_effect,
                        "special_term_median_abs_standardized_effect": special_effect,
                        "special_term_significant_fraction": significant_fraction,
                        "median_cv_score_gain_over_parent": (
                            float(np.nanmedian(cv_gains)) if len(cv_gains) else math.nan
                        ),
                        "special_cv_gain_p": gain_p,
                    }
                )
    summary = pd.DataFrame(rows)
    summary["special_cv_gain_q"] = np.nan
    mask = summary["interaction"].notna() & summary["special_cv_gain_p"].notna()
    summary.loc[mask, "special_cv_gain_q"] = (
        summary.loc[mask]
        .groupby(["outcome_family", "prompt_mode"], group_keys=False)[
            "special_cv_gain_p"
        ]
        .apply(bh_adjust)
        .reindex(summary.loc[mask].index)
    )
    summary["practical_effect_pass"] = (
        summary["candidate"].eq("intercept")
        | (
            summary["minimum_term_median_abs_standardized_effect"]
            >= PRACTICAL_EFFECT_THRESHOLD
        )
    )
    summary["special_support_pass"] = True
    interaction = summary["interaction"].notna()
    summary.loc[interaction, "special_support_pass"] = (
        (
            summary["special_term_median_abs_standardized_effect"]
            >= PRACTICAL_EFFECT_THRESHOLD
        )
        & (
            summary["special_term_significant_fraction"]
            >= SPECIAL_SIGNIFICANT_FRACTION_THRESHOLD
        )
        & (
            summary["median_cv_score_gain_over_parent"]
            >= SPECIAL_CV_GAIN_THRESHOLD
        )
        & (summary["special_cv_gain_q"] <= SPECIAL_CV_GAIN_Q_THRESHOLD)
    )[interaction]
    summary["selection_gate_pass"] = (
        summary["practical_effect_pass"] & summary["special_support_pass"]
    )
    return summary


def choose_formula(
    summary: pd.DataFrame,
    family: str,
    mode: str,
    candidates: tuple[Candidate, ...],
) -> pd.Series:
    block = summary.loc[
        summary["outcome_family"].eq(family)
        & summary["prompt_mode"].eq(mode)
        & summary["median_primary_score"].notna()
    ].copy()
    eligible = block.loc[block["selection_gate_pass"]].copy()
    if eligible.empty:
        eligible = block.copy()
    best_median = float(eligible["median_primary_score"].max())
    near = eligible.loc[
        eligible["median_primary_score"] >= best_median - MEDIAN_CV_TOLERANCE
    ].copy()
    best_q25 = float(near["q25_primary_score"].max())
    near = near.loc[
        near["q25_primary_score"] >= best_q25 - Q25_CV_TOLERANCE
    ].copy()
    rank = {candidate.id: index for index, candidate in enumerate(candidates)}
    near["_rank"] = near["candidate"].map(rank)
    near = near.sort_values(
        [
            "predictors",
            "median_primary_loss",
            "median_primary_score",
            "q25_primary_score",
            "_rank",
        ],
        ascending=[True, True, False, False, True],
    )
    return near.iloc[0].drop(labels="_rank")


def select_all(
    summary: pd.DataFrame,
    candidates: tuple[Candidate, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family in sorted(summary["outcome_family"].unique()):
        for mode in sorted(summary["prompt_mode"].unique()):
            winner = choose_formula(summary, family, mode, candidates)
            rows.append({**winner.to_dict(), "selected_candidate": winner["candidate"]})
    return pd.DataFrame(rows)


def evidence_reading(row: pd.Series) -> str:
    median = float(row["median_primary_score"])
    q25 = float(row["q25_primary_score"])
    stability = float(row.get("lomo_formula_stability", math.nan))
    if median >= 0.50 and q25 >= 0.25 and stability >= 0.75:
        return "Strong cross-model support"
    if median >= 0.20 and q25 > 0 and stability >= 0.50:
        return "Tentative cross-model support"
    if median > 0 and q25 > 0:
        return "Weak; structure unstable"
    return "No reliable shared law"


def lomo_selection(
    metrics: pd.DataFrame,
    coefficients: pd.DataFrame,
    candidates: tuple[Candidate, ...],
    selected: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_slots = set(metrics["comparison_slot"].unique())
    for family in LOMO_FAMILIES:
        for mode in sorted(metrics["prompt_mode"].unique()):
            full = selected.loc[
                selected["outcome_family"].eq(family)
                & selected["prompt_mode"].eq(mode)
            ].iloc[0]
            for omitted in sorted(all_slots):
                retained = all_slots.difference({omitted})
                summary = summarize_candidates(
                    metrics,
                    coefficients,
                    candidates,
                    slots=retained,
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


def add_lomo_summary(selected: pd.DataFrame, lomo: pd.DataFrame) -> pd.DataFrame:
    result = selected.copy()
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
    result = result.merge(
        summary,
        on=["outcome_family", "prompt_mode"],
        how="left",
        validate="one_to_one",
    )
    result["evidence_reading"] = result.apply(
        lambda row: (
            evidence_reading(row)
            if row["outcome_family"] in LOMO_FAMILIES
            else "Robustness family; not headline evidence"
        ),
        axis=1,
    )
    return result


def beta_binomial_nll(
    theta: np.ndarray,
    x: np.ndarray,
    successes: np.ndarray,
    totals: np.ndarray,
) -> float:
    beta = theta[:-1]
    log_kappa = float(theta[-1])
    probability = clip_probability(expit(x @ beta))
    kappa = float(np.exp(np.clip(log_kappa, -8.0, 16.0)))
    alpha = probability * kappa
    beta_shape = (1.0 - probability) * kappa
    log_probability = (
        gammaln(totals + 1.0)
        - gammaln(successes + 1.0)
        - gammaln(totals - successes + 1.0)
        + betaln(successes + alpha, totals - successes + beta_shape)
        - betaln(alpha, beta_shape)
    )
    return -float(np.sum(log_probability))


def minimize_beta_binomial(
    start: np.ndarray,
    x: np.ndarray,
    successes: np.ndarray,
    totals: np.ndarray,
    bounds: list[tuple[float, float]],
    *,
    ftol: float,
) -> Any:
    """Minimize one frozen Beta-Binomial likelihood robustly.

    L-BFGS-B is fast but occasionally exits with SciPy's ``ABNORMAL`` line
    search status on otherwise finite likelihood surfaces.  Powell is an
    objective-equivalent, derivative-free fallback; a final L-BFGS-B pass
    from the Powell solution recovers its inverse-Hessian approximation when
    possible.  This changes only the numerical optimizer, not the estimand,
    likelihood, folds, bounds, or selected law.
    """
    options = {"maxiter": 2000, "ftol": ftol, "maxls": 50}
    primary = minimize(
        beta_binomial_nll,
        start,
        args=(x, successes, totals),
        method="L-BFGS-B",
        bounds=bounds,
        options=options,
    )
    if primary.success and np.isfinite(primary.fun):
        primary["optimizer"] = "L-BFGS-B"
        return primary
    fallback_start = (
        np.asarray(primary.x, dtype=float)
        if np.isfinite(primary.x).all()
        else np.asarray(start, dtype=float)
    )
    fallback = minimize(
        beta_binomial_nll,
        fallback_start,
        args=(x, successes, totals),
        method="Powell",
        bounds=bounds,
        options={"maxiter": 4000, "ftol": max(ftol, 1e-9), "xtol": 1e-7},
    )
    if not fallback.success or not np.isfinite(fallback.fun):
        return primary
    refinement = minimize(
        beta_binomial_nll,
        np.asarray(fallback.x, dtype=float),
        args=(x, successes, totals),
        method="L-BFGS-B",
        bounds=bounds,
        options=options,
    )
    if (
        refinement.success
        and np.isfinite(refinement.fun)
        and refinement.fun <= fallback.fun + 1e-6
    ):
        refinement["optimizer"] = "Powell+L-BFGS-B"
        return refinement
    fallback["optimizer"] = "Powell"
    return fallback


def fit_beta_binomial(
    frame: pd.DataFrame,
    candidate: Candidate,
    n_levels: tuple[int, ...],
    l_levels: tuple[int, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    x = design_matrix(frame, candidate)
    successes = frame["n_correct"].to_numpy(dtype=float)
    totals = frame["n_total"].to_numpy(dtype=float)
    initial_glm = sm.GLM(
        successes / totals,
        x,
        family=sm.families.Binomial(),
        freq_weights=totals,
    ).fit(maxiter=300, disp=0)
    theta0 = np.concatenate([np.asarray(initial_glm.params), [math.log(100.0)]])
    bounds = [(-50.0, 50.0)] * x.shape[1] + [(-8.0, 16.0)]
    full = minimize_beta_binomial(
        theta0,
        x,
        successes,
        totals,
        bounds,
        ftol=1e-11,
    )
    if not full.success:
        raise RuntimeError(f"Beta-Binomial full fit failed: {full.message}")
    folds = condition_fold(frame, n_levels, l_levels)
    probability = np.full(len(frame), np.nan, dtype=float)
    predictive_nll = 0.0
    for fold in range(5):
        test = folds == fold
        train = ~test
        train_fit = minimize_beta_binomial(
            np.asarray(full.x, dtype=float),
            x[train],
            successes[train],
            totals[train],
            bounds,
            ftol=1e-10,
        )
        if not train_fit.success:
            raise RuntimeError(f"Beta-Binomial fold {fold} failed: {train_fit.message}")
        probability[test] = expit(x[test] @ train_fit.x[:-1])
        predictive_nll += beta_binomial_nll(
            train_fit.x, x[test], successes[test], totals[test]
        )
    p = clip_probability(probability)
    request_log_loss = -float(
        np.sum(successes * np.log(p) + (totals - successes) * np.log1p(-p))
        / np.sum(totals)
    )
    request_brier = float(
        np.sum(successes * np.square(1.0 - p) + (totals - successes) * np.square(p))
        / np.sum(totals)
    )
    kappa = float(np.exp(full.x[-1]))
    metrics = {
        "candidate": candidate.id,
        "converged": bool(full.success),
        "cv_beta_binomial_nlpd_per_cell": float(predictive_nll / len(frame)),
        "cv_request_log_loss_from_mean": request_log_loss,
        "cv_request_brier_from_mean": request_brier,
        "kappa": kappa,
        "rho": float(1.0 / (kappa + 1.0)),
        "full_negative_log_likelihood": float(full.fun),
        "full_optimizer": str(full.get("optimizer", "unknown")),
    }
    try:
        covariance = np.asarray(full.hess_inv.todense(), dtype=float)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    except (AttributeError, ValueError):
        standard_errors = np.full(len(full.x), np.nan, dtype=float)
    coefficients: list[dict[str, Any]] = []
    for index, term in enumerate(("intercept", *candidate.terms, "log_kappa")):
        estimate = float(full.x[index])
        standard_error = float(standard_errors[index])
        coefficients.append(
            {
                "candidate": candidate.id,
                "term": term,
                "estimate": estimate,
                "standard_error": standard_error,
                "ci95_low": estimate - 1.96 * standard_error,
                "ci95_high": estimate + 1.96 * standard_error,
            }
        )
    return metrics, coefficients


def run_beta_binomial(
    cells: pd.DataFrame,
    selected: pd.DataFrame,
    candidates: tuple[Candidate, ...],
    n_levels: tuple[int, ...],
    l_levels: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_id = {candidate.id: candidate for candidate in candidates}
    metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    headline = selected.loc[selected["outcome_family"].eq(HEADLINE_ACCURACY)]
    for choice in headline.itertuples(index=False):
        mode = str(choice.prompt_mode)
        candidate = by_id[str(choice.selected_candidate)]
        for slot in sorted(cells["comparison_slot"].unique()):
            frame = cells.loc[
                cells["comparison_slot"].eq(slot)
                & cells["prompt_mode"].eq(mode)
            ].copy()
            try:
                metrics, coefficients = fit_beta_binomial(
                    frame, candidate, n_levels, l_levels
                )
                metric_rows.append(
                    {"comparison_slot": slot, "prompt_mode": mode, **metrics}
                )
                coefficient_rows.extend(
                    {"comparison_slot": slot, "prompt_mode": mode, **row}
                    for row in coefficients
                )
            except Exception as error:
                failure_rows.append(
                    {
                        "comparison_slot": slot,
                        "prompt_mode": mode,
                        "candidate": candidate.id,
                        "error": repr(error),
                    }
                )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(failure_rows),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) // 2)),
    )
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--skip-beta-binomial", action="store_true")
    parser.add_argument("--skip-lomo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = verify_freeze(args.config.resolve(), args.freeze.resolve())
    candidates = load_candidates(config)
    input_path = args.input.resolve()
    output = args.output.resolve()
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    if file_sha256(input_path) != config["immutable_input"]["request_level_sha256"]:
        raise ValueError("Request table SHA-256 does not match the V3.2 freeze")
    write_state(output, stage="loading_requests", input=str(input_path))
    requests = pd.read_csv(input_path)
    validate_requests(requests, config)
    cells = build_cells(requests)
    cells.to_csv(tables / "cell_outcomes.csv.gz", index=False, compression="gzip")
    n_levels = tuple(int(value) for value in config["immutable_input"]["N_levels"])
    l_levels = tuple(int(value) for value in config["immutable_input"]["L_levels"])
    task_keys = list(
        requests[["comparison_slot", "prompt_mode"]]
        .drop_duplicates()
        .sort_values(["comparison_slot", "prompt_mode"])
        .itertuples(index=False, name=None)
    )
    if args.max_tasks is not None:
        task_keys = task_keys[: args.max_tasks]
    payloads = []
    candidate_payload = [
        {
            "id": candidate.id,
            "terms": candidate.terms,
            "parent": candidate.parent,
            "interaction": candidate.interaction,
        }
        for candidate in candidates
    ]
    request_columns = [
        "N",
        "L",
        "exact_count",
        "L_k",
        "logN",
        "logL",
        "N_x_L_k",
        "logN_x_logL",
        "N_x_logL",
        "logN_x_L_k",
    ]
    for slot, mode in task_keys:
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
    write_state(
        output,
        stage="candidate_fitting",
        tasks_completed=0,
        tasks_total=len(payloads),
        workers=args.workers,
    )
    metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    if args.workers == 1:
        iterator = (fit_slot_task(payload) for payload in payloads)
        for completed, result in enumerate(iterator, start=1):
            metric_rows.extend(result["metrics"])
            coefficient_rows.extend(result["coefficients"])
            failure_rows.extend(result["failures"])
            print(
                json.dumps(
                    {
                        "stage": "candidate_fitting",
                        "completed_tasks": completed,
                        "total_tasks": len(payloads),
                        "slot": result["slot"],
                        "mode": result["mode"],
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                    }
                ),
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(fit_slot_task, payload) for payload in payloads]
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                metric_rows.extend(result["metrics"])
                coefficient_rows.extend(result["coefficients"])
                failure_rows.extend(result["failures"])
                print(
                    json.dumps(
                        {
                            "stage": "candidate_fitting",
                            "completed_tasks": completed,
                            "total_tasks": len(payloads),
                            "slot": result["slot"],
                            "mode": result["mode"],
                            "elapsed_seconds": round(time.perf_counter() - started, 3),
                        }
                    ),
                    flush=True,
                )
                write_state(
                    output,
                    stage="candidate_fitting",
                    tasks_completed=completed,
                    tasks_total=len(payloads),
                    workers=args.workers,
                    failures=len(failure_rows),
                )
    metrics = pd.DataFrame(metric_rows)
    coefficients = apply_coefficient_bh(pd.DataFrame(coefficient_rows))
    failures = pd.DataFrame(failure_rows)
    metrics.to_csv(tables / "candidate_fit_metrics.csv", index=False)
    coefficients.to_csv(tables / "candidate_coefficients.csv", index=False)
    failures.to_csv(tables / "candidate_fit_failures.csv", index=False)
    if failures.shape[0]:
        raise RuntimeError(f"Candidate fitting produced {len(failures)} failures")
    summary = summarize_candidates(metrics, coefficients, candidates)
    selected = select_all(summary, candidates)
    if args.skip_lomo or args.max_tasks is not None:
        lomo = pd.DataFrame()
        selected["lomo_formula_stability"] = math.nan
        selected["evidence_reading"] = "LOMO not run"
    else:
        write_state(output, stage="lomo_selection")
        lomo = lomo_selection(metrics, coefficients, candidates, selected)
        selected = add_lomo_summary(selected, lomo)
    summary.to_csv(tables / "mode_candidate_summary.csv", index=False)
    selected.to_csv(tables / "selected_mode_laws.csv", index=False)
    lomo.to_csv(tables / "lomo_structure_selection.csv", index=False)
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
    selected_metrics.to_csv(tables / "selected_model_fit_metrics.csv", index=False)
    selected_coefficients.to_csv(
        tables / "selected_model_coefficients.csv", index=False
    )
    beta_metrics = pd.DataFrame()
    beta_coefficients = pd.DataFrame()
    beta_failures = pd.DataFrame()
    if not args.skip_beta_binomial and args.max_tasks is None:
        write_state(output, stage="beta_binomial_robustness")
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
                f"Beta-Binomial robustness produced {len(beta_failures)} failures"
            )
    manifest = {
        "schema_version": "realistic_niah_v3_2_analysis_manifest_v1",
        "analysis_version": "V3.2",
        "created_utc": utc_now(),
        "config": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config.resolve()),
        "freeze": str(args.freeze.resolve()),
        "input": str(input_path),
        "input_sha256": file_sha256(input_path),
        "requests": len(requests),
        "cells": len(cells),
        "tasks": len(payloads),
        "workers": args.workers,
        "candidate_fit_rows": len(metrics),
        "candidate_coefficient_rows": len(coefficients),
        "candidate_failures": len(failures),
        "lomo_rows": len(lomo),
        "beta_binomial_fit_rows": len(beta_metrics),
        "beta_binomial_failures": len(beta_failures),
        "bootstrap_repetitions": 0,
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "selected_laws": selected[
            [
                "outcome_family",
                "prompt_mode",
                "selected_candidate",
                "median_primary_score",
                "q25_primary_score",
                "median_primary_loss",
                "lomo_formula_stability",
                "evidence_reading",
            ]
        ].to_dict("records"),
    }
    write_json(output / "analysis_manifest.json", manifest)
    write_state(
        output,
        stage="complete",
        candidate_fit_rows=len(metrics),
        selected_laws=len(selected),
        lomo_rows=len(lomo),
        beta_binomial_fit_rows=len(beta_metrics),
        elapsed_seconds=manifest["elapsed_seconds"],
    )
    print(
        json.dumps(json_safe({"stage": "complete", **manifest}), allow_nan=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
