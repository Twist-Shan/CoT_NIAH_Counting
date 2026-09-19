from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, expit, gammaln
from scipy.stats import betabinom, binom, chi2, kstest, norm, t
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from .analysis import accuracy_condition_table, bias_condition_table


@dataclass(frozen=True)
class Candidate:
    name: str
    features: tuple[str, ...]
    parent: str | None = None
    interaction_feature: str | None = None


CANDIDATES = (
    Candidate("intercept", ()),
    Candidate("N", ("N",)),
    Candidate("L", ("L_k",)),
    Candidate("logN", ("logN",)),
    Candidate("logL", ("logL",)),
    Candidate("linear_additive", ("N", "L_k")),
    Candidate("log_additive", ("logN", "logL")),
    Candidate("N_logL_additive", ("N", "logL")),
    Candidate("logN_L_additive", ("logN", "L_k")),
    Candidate(
        "linear_interaction",
        ("N", "L_k", "N_x_L_k"),
        parent="linear_additive",
        interaction_feature="N_x_L_k",
    ),
    Candidate(
        "log_interaction",
        ("logN", "logL", "logN_x_logL"),
        parent="log_additive",
        interaction_feature="logN_x_logL",
    ),
    Candidate(
        "N_logL_interaction",
        ("N", "logL", "N_x_logL"),
        parent="N_logL_additive",
        interaction_feature="N_x_logL",
    ),
    Candidate(
        "logN_L_interaction",
        ("logN", "L_k", "logN_x_L_k"),
        parent="logN_L_additive",
        interaction_feature="logN_x_L_k",
    ),
)
CANDIDATE_BY_NAME = {candidate.name: candidate for candidate in CANDIDATES}
OUTCOME_MODELS = ("bernoulli", "binomial", "beta_binomial", "bias")
_FIT_BACKEND = "scipy"
_FIT_DEVICE = "cpu"


def configure_fit_backend(*, backend: str = "scipy", device: str = "cpu") -> None:
    """Configure the process-wide law-fitting backend used by nested analyses."""

    if backend not in {"scipy", "torch"}:
        raise ValueError("fit backend must be scipy or torch")
    if device not in {"cpu", "cuda"}:
        raise ValueError("analysis device must be cpu or cuda")
    if backend == "scipy" and device != "cpu":
        raise ValueError("The scipy backend is CPU-only")
    if backend == "torch":
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA analysis requested but torch.cuda is unavailable")
    global _FIT_BACKEND, _FIT_DEVICE
    _FIT_BACKEND = backend
    _FIT_DEVICE = device


def fit_backend_metadata() -> dict[str, str]:
    return {"backend": _FIT_BACKEND, "device": _FIT_DEVICE}


def _minimum_successful_bootstraps(replicates: int) -> int:
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    return min(replicates, max(20, int(math.ceil(0.8 * replicates))))


@dataclass(frozen=True)
class FeatureScaler:
    means: dict[str, float]
    scales: dict[str, float]

    @classmethod
    def fit(cls, frame: pd.DataFrame, features: Iterable[str]) -> "FeatureScaler":
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        for feature in features:
            values = frame[feature].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite values in {feature}")
            means[feature] = float(values.mean())
            scale = float(values.std(ddof=0))
            scales[feature] = scale if scale > 0 else 1.0
        return cls(means=means, scales=scales)

    def values(self, frame: pd.DataFrame, feature: str) -> np.ndarray:
        return (
            frame[feature].to_numpy(dtype=float) - self.means[feature]
        ) / self.scales[feature]


@dataclass
class FittedLaw:
    candidate: Candidate
    outcome_model: str
    levels: tuple[str, ...]
    scaler: FeatureScaler
    beta: np.ndarray
    covariance: np.ndarray
    converged: bool
    log_kappa: np.ndarray | None = None

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        x, _ = design_matrix(frame, self.candidate, self.levels, self.scaler)
        return np.clip(expit(x @ self.beta), 1e-9, 1 - 1e-9)

    def predict_bias(self, frame: pd.DataFrame) -> np.ndarray:
        x, _ = design_matrix(frame, self.candidate, self.levels, self.scaler)
        return x @ self.beta


def design_matrix(
    frame: pd.DataFrame,
    candidate: Candidate,
    levels: tuple[str, ...],
    scaler: FeatureScaler,
) -> tuple[np.ndarray, list[str]]:
    labels = frame["comparison_slot"].astype(str)
    columns: list[np.ndarray] = []
    names: list[str] = []
    indicators: dict[str, np.ndarray] = {}
    for level in levels:
        indicator = (labels == level).to_numpy(dtype=float)
        if not indicator.any():
            raise ValueError(f"Design frame has no rows for registered level {level}")
        indicators[level] = indicator
        columns.append(indicator)
        names.append(f"alpha[{level}]")
    for feature in candidate.features:
        values = scaler.values(frame, feature)
        for level in levels:
            columns.append(indicators[level] * values)
            names.append(f"beta[{level}]:{feature}")
    return np.column_stack(columns), names


def _fit_ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    degrees = max(1, len(y) - x.shape[1])
    sigma2 = float(residual @ residual) / degrees
    covariance = sigma2 * np.linalg.pinv(x.T @ x)
    return beta, covariance, bool(np.isfinite(beta).all())


def _fit_binomial(
    x: np.ndarray,
    successes: np.ndarray,
    totals: np.ndarray,
    *,
    intercept_columns: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    ridge = 1e-8

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = x @ beta
        loss = float((totals * np.logaddexp(0.0, eta) - successes * eta).sum())
        probability = expit(eta)
        gradient = x.T @ (totals * probability - successes)
        loss += 0.5 * ridge * float(beta[intercept_columns:] @ beta[intercept_columns:])
        gradient[intercept_columns:] += ridge * beta[intercept_columns:]
        return loss, gradient

    result = minimize(
        lambda value: objective(value)[0],
        np.zeros(x.shape[1], dtype=float),
        jac=lambda value: objective(value)[1],
        method="L-BFGS-B",
        options={"maxiter": 3000, "ftol": 1e-11},
    )
    beta = np.asarray(result.x, dtype=float)
    probability = np.clip(expit(x @ beta), 1e-9, 1 - 1e-9)
    weight = totals * probability * (1.0 - probability)
    information = x.T @ (x * weight[:, None])
    if x.shape[1] > intercept_columns:
        information[intercept_columns:, intercept_columns:] += ridge * np.eye(
            x.shape[1] - intercept_columns
        )
    covariance = np.linalg.pinv(information)
    return beta, covariance, bool(result.success and np.isfinite(beta).all())


def _fit_beta_binomial(
    x: np.ndarray,
    successes: np.ndarray,
    totals: np.ndarray,
    model_index: np.ndarray,
    *,
    intercept_columns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    beta_start, _, _ = _fit_binomial(
        x,
        successes,
        totals,
        intercept_columns=intercept_columns,
    )
    model_count = int(model_index.max()) + 1
    start = np.concatenate([beta_start, np.full(model_count, math.log(100.0))])
    ridge = 1e-8

    def objective(theta: np.ndarray) -> float:
        beta = theta[: x.shape[1]]
        log_kappa = theta[x.shape[1] :]
        mu = np.clip(expit(x @ beta), 1e-9, 1 - 1e-9)
        kappa = np.exp(np.clip(log_kappa[model_index], -8.0, 20.0))
        a = mu * kappa
        b = (1.0 - mu) * kappa
        log_probability = (
            gammaln(totals + 1)
            - gammaln(successes + 1)
            - gammaln(totals - successes + 1)
            + betaln(successes + a, totals - successes + b)
            - betaln(a, b)
        )
        penalty = (
            0.5 * ridge * float(beta[intercept_columns:] @ beta[intercept_columns:])
        )
        return float(-log_probability.sum() + penalty)

    bounds = [(None, None)] * x.shape[1] + [(-8.0, 20.0)] * model_count
    result = minimize(
        objective,
        start,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 4000, "ftol": 1e-10},
    )
    if not result.success or not np.isfinite(result.fun):
        fallback = minimize(
            objective,
            np.asarray(result.x if np.isfinite(result.x).all() else start),
            method="Powell",
            bounds=bounds,
            options={"maxiter": 4000, "ftol": 1e-9, "xtol": 1e-7},
        )
        if np.isfinite(fallback.fun) and (
            not np.isfinite(result.fun) or fallback.fun <= result.fun + 1e-7
        ):
            result = fallback
    theta = np.asarray(result.x, dtype=float)
    beta = theta[: x.shape[1]]
    log_kappa = theta[x.shape[1] :]
    try:
        covariance = np.asarray(result.hess_inv.todense(), dtype=float)[
            : x.shape[1], : x.shape[1]
        ]
    except (AttributeError, ValueError):
        covariance = np.full((x.shape[1], x.shape[1]), np.nan)
    return (
        beta,
        covariance,
        log_kappa,
        bool(result.success and np.isfinite(theta).all()),
    )


def fit_law(
    frame: pd.DataFrame,
    candidate: Candidate,
    outcome_model: str,
    *,
    levels: tuple[str, ...] | None = None,
    backend: str | None = None,
    device: str | None = None,
) -> FittedLaw:
    resolved_backend = backend or _FIT_BACKEND
    resolved_device = device or _FIT_DEVICE
    if resolved_backend == "torch":
        from .torch_laws import fit_law_torch

        return fit_law_torch(
            frame,
            candidate,
            outcome_model,
            levels=levels,
            device=resolved_device,
        )
    if resolved_backend != "scipy" or resolved_device != "cpu":
        raise ValueError(
            f"Unsupported fit backend/device: {resolved_backend}/{resolved_device}"
        )
    if outcome_model not in OUTCOME_MODELS:
        raise ValueError(f"Unknown outcome model: {outcome_model}")
    if frame.empty:
        raise ValueError("Cannot fit an empty law frame")
    resolved_levels = levels or tuple(
        sorted(frame["comparison_slot"].astype(str).unique())
    )
    scaler = FeatureScaler.fit(frame, candidate.features)
    x, _ = design_matrix(frame, candidate, resolved_levels, scaler)
    if outcome_model == "bias":
        y = frame["trimmed_signed_bias_10"].to_numpy(dtype=float)
        beta, covariance, converged = _fit_ols(x, y)
        return FittedLaw(
            candidate,
            outcome_model,
            resolved_levels,
            scaler,
            beta,
            covariance,
            converged,
        )
    if outcome_model == "bernoulli":
        successes = frame["exact_count"].to_numpy(dtype=float)
        totals = np.ones(len(frame), dtype=float)
    else:
        successes = frame["n_correct_parsed"].to_numpy(dtype=float)
        totals = frame["n_total"].to_numpy(dtype=float)
    beta, covariance, converged = _fit_binomial(
        x,
        successes,
        totals,
        intercept_columns=len(resolved_levels),
    )
    if outcome_model != "beta_binomial":
        return FittedLaw(
            candidate,
            outcome_model,
            resolved_levels,
            scaler,
            beta,
            covariance,
            converged,
        )
    model_lookup = {level: index for index, level in enumerate(resolved_levels)}
    model_index = (
        frame["comparison_slot"].astype(str).map(model_lookup).to_numpy(dtype=int)
    )
    beta, covariance, log_kappa, converged = _fit_beta_binomial(
        x,
        successes,
        totals,
        model_index,
        intercept_columns=len(resolved_levels),
    )
    return FittedLaw(
        candidate,
        outcome_model,
        resolved_levels,
        scaler,
        beta,
        covariance,
        converged,
        log_kappa=log_kappa,
    )


def coefficient_table(fit: FittedLaw) -> pd.DataFrame:
    _, names = design_matrix(
        pd.DataFrame(
            {
                "comparison_slot": list(fit.levels),
                **{
                    feature: [fit.scaler.means[feature]] * len(fit.levels)
                    for feature in fit.candidate.features
                },
            }
        ),
        fit.candidate,
        fit.levels,
        fit.scaler,
    )
    standard_error = np.sqrt(np.clip(np.diag(fit.covariance), 0.0, None))
    rows: list[dict[str, Any]] = []
    level_count = len(fit.levels)
    for level_index, level in enumerate(fit.levels):
        scaled_alpha = float(fit.beta[level_index])
        raw_alpha = scaled_alpha
        variance_alpha = float(fit.covariance[level_index, level_index])
        for feature_index, feature in enumerate(fit.candidate.features):
            index = level_count + feature_index * level_count + level_index
            raw_alpha -= (
                float(fit.beta[index])
                * fit.scaler.means[feature]
                / fit.scaler.scales[feature]
            )
        rows.append(
            {
                "comparison_slot": level,
                "term": "intercept",
                "estimate_original_scale": raw_alpha,
                "estimate_standardized": scaled_alpha,
                "standard_error_standardized": math.sqrt(max(0.0, variance_alpha)),
            }
        )
        for feature_index, feature in enumerate(fit.candidate.features):
            index = level_count + feature_index * level_count + level_index
            estimate = float(fit.beta[index])
            error = float(standard_error[index])
            statistic = estimate / error if error > 0 else math.nan
            distribution = norm if fit.outcome_model != "bias" else t
            if fit.outcome_model == "bias":
                p_value = 2.0 * distribution.sf(abs(statistic), df=max(1, len(names)))
            else:
                p_value = 2.0 * distribution.sf(abs(statistic))
            rows.append(
                {
                    "comparison_slot": level,
                    "term": feature,
                    "estimate_original_scale": estimate / fit.scaler.scales[feature],
                    "estimate_standardized": estimate,
                    "standard_error_standardized": error,
                    "p_value_asymptotic": float(p_value),
                    "ci95_low_standardized": estimate - 1.96 * error,
                    "ci95_high_standardized": estimate + 1.96 * error,
                }
            )
    if fit.log_kappa is not None:
        for level, value in zip(fit.levels, fit.log_kappa):
            kappa = float(np.exp(value))
            rows.append(
                {
                    "comparison_slot": level,
                    "term": "beta_binomial_kappa",
                    "estimate_original_scale": kappa,
                    "estimate_standardized": float(value),
                    "rho": 1.0 / (kappa + 1.0),
                }
            )
    result = pd.DataFrame(rows)
    result.insert(0, "outcome_model", fit.outcome_model)
    result.insert(1, "candidate", fit.candidate.name)
    return result


def _beta_binomial_log_probability(
    successes: np.ndarray,
    totals: np.ndarray,
    mu: np.ndarray,
    kappa: np.ndarray,
) -> np.ndarray:
    a = mu * kappa
    b = (1.0 - mu) * kappa
    return (
        gammaln(totals + 1)
        - gammaln(successes + 1)
        - gammaln(totals - successes + 1)
        + betaln(successes + a, totals - successes + b)
        - betaln(a, b)
    )


def _fold_assignments(seeds: Iterable[int], n_splits: int) -> dict[int, int]:
    ordered = sorted({int(seed) for seed in seeds})
    if n_splits < 2 or n_splits > len(ordered):
        raise ValueError("Invalid number of seed folds")
    return {seed: index % n_splits for index, seed in enumerate(ordered)}


def _analysis_frame(
    requests: pd.DataFrame, outcome_model: str, *, full: bool
) -> pd.DataFrame:
    if outcome_model == "bernoulli":
        return requests.copy()
    if outcome_model in {"binomial", "beta_binomial"}:
        return accuracy_condition_table(requests)
    seed_count = int(requests["seed"].nunique())
    minimum = (
        20 if full and seed_count >= 30 else max(1, int(math.ceil(2 * seed_count / 3)))
    )
    cells = bias_condition_table(requests, minimum_parseable=minimum)
    return cells.loc[cells["bias_law_eligible"]].copy()


def _score_fit(fit: FittedLaw, test: pd.DataFrame) -> dict[str, float]:
    if fit.outcome_model == "bias":
        observed = test["trimmed_signed_bias_10"].to_numpy(dtype=float)
        predicted = fit.predict_bias(test)
        return {
            "primary_loss": float(mean_absolute_error(observed, predicted)),
            "mae": float(mean_absolute_error(observed, predicted)),
            "rmse": float(math.sqrt(mean_squared_error(observed, predicted))),
            "r2": (
                float(r2_score(observed, predicted))
                if len(observed) > 1 and not np.allclose(observed, observed[0])
                else math.nan
            ),
        }
    probability = fit.predict_probability(test)
    if fit.outcome_model == "bernoulli":
        observed = test["exact_count"].to_numpy(dtype=int)
        loss = float(log_loss(observed, probability, labels=[0, 1]))
        return {
            "primary_loss": loss,
            "log_loss": loss,
            "brier": float(brier_score_loss(observed, probability)),
        }
    successes = test["n_correct_parsed"].to_numpy(dtype=float)
    totals = test["n_total"].to_numpy(dtype=float)
    if fit.outcome_model == "binomial":
        logp = (
            gammaln(totals + 1)
            - gammaln(successes + 1)
            - gammaln(totals - successes + 1)
            + successes * np.log(probability)
            + (totals - successes) * np.log1p(-probability)
        )
    else:
        assert fit.log_kappa is not None
        lookup = {level: index for index, level in enumerate(fit.levels)}
        indices = test["comparison_slot"].astype(str).map(lookup).to_numpy(dtype=int)
        kappa = np.exp(fit.log_kappa[indices])
        logp = _beta_binomial_log_probability(successes, totals, probability, kappa)
    observed_rate = successes / totals
    return {
        "primary_loss": float(-logp.mean()),
        "negative_log_predictive_density": float(-logp.mean()),
        "brier": float(np.mean((observed_rate - probability) ** 2)),
    }


def probability_distribution_diagnostics(
    fit: FittedLaw,
    cells: pd.DataFrame,
    *,
    random_seed: int = 20_260_807,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if fit.outcome_model not in {"binomial", "beta_binomial"}:
        raise ValueError("Distribution diagnostics require a cell probability model")
    probability = fit.predict_probability(cells)
    successes = cells["n_correct_parsed"].to_numpy(dtype=int)
    totals = cells["n_total"].to_numpy(dtype=int)
    if fit.outcome_model == "beta_binomial":
        assert fit.log_kappa is not None
        lookup = {level: index for index, level in enumerate(fit.levels)}
        indices = cells["comparison_slot"].astype(str).map(lookup).to_numpy(dtype=int)
        kappa = np.exp(fit.log_kappa[indices])
        a = probability * kappa
        b = (1.0 - probability) * kappa
        cdf_at = betabinom.cdf(successes, totals, a, b)
        cdf_before = betabinom.cdf(successes - 1, totals, a, b)

        def predictive_quantile(probability_level: float) -> np.ndarray:
            return betabinom.ppf(probability_level, totals, a, b)
    else:
        kappa = np.full(len(cells), np.inf)
        cdf_at = binom.cdf(successes, totals, probability)
        cdf_before = binom.cdf(successes - 1, totals, probability)

        def predictive_quantile(probability_level: float) -> np.ndarray:
            return binom.ppf(probability_level, totals, probability)

    rng = np.random.default_rng(random_seed)
    randomized_pit = cdf_before + rng.random(len(cells)) * (cdf_at - cdf_before)
    detail = cells[
        [
            "comparison_slot",
            "model_label",
            "prompt_mode",
            "N",
            "L",
            "n_total",
            "n_correct_parsed",
        ]
    ].copy()
    detail["predicted_accuracy"] = probability
    detail["observed_accuracy"] = successes / totals
    detail["kappa"] = kappa
    detail["rho"] = np.where(np.isfinite(kappa), 1.0 / (kappa + 1.0), 0.0)
    detail["randomized_pit"] = randomized_pit
    coverage: list[dict[str, Any]] = []
    for level in (0.50, 0.80, 0.95):
        tail = (1.0 - level) / 2.0
        low = np.asarray(predictive_quantile(tail), dtype=float)
        high = np.asarray(predictive_quantile(1.0 - tail), dtype=float)
        covered = (successes >= low) & (successes <= high)
        detail[f"pi{int(level * 100)}_low"] = low
        detail[f"pi{int(level * 100)}_high"] = high
        detail[f"pi{int(level * 100)}_covered"] = covered
        coverage.append(
            {
                "nominal_coverage": level,
                "empirical_coverage": float(covered.mean()),
                "mean_interval_width": float(np.mean(high - low)),
            }
        )
    bins = pd.qcut(
        pd.Series(probability),
        q=min(10, max(2, int(pd.Series(probability).nunique()))),
        duplicates="drop",
    )
    calibration = (
        detail.assign(probability_bin=bins.astype(str).to_numpy())
        .groupby("probability_bin", sort=False, dropna=False)
        .agg(
            cells=("n_total", "size"),
            predicted_accuracy=("predicted_accuracy", "mean"),
            observed_accuracy=("observed_accuracy", "mean"),
        )
        .reset_index()
    )
    ks = kstest(randomized_pit, "uniform")
    summary = {
        "outcome_model": fit.outcome_model,
        "candidate": fit.candidate.name,
        "cells": len(cells),
        "randomized_pit_mean": float(np.mean(randomized_pit)),
        "randomized_pit_variance": float(np.var(randomized_pit, ddof=1)),
        "randomized_pit_uniform_ks_statistic": float(ks.statistic),
        "randomized_pit_uniform_ks_p_value": float(ks.pvalue),
        "brier": float(np.mean((successes / totals - probability) ** 2)),
        "coverage": coverage,
    }
    return detail, calibration, summary


def cross_validate_candidate(
    requests: pd.DataFrame,
    *,
    prompt_mode: str,
    candidate: Candidate,
    outcome_model: str,
    n_splits: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, FittedLaw]:
    mode_rows = requests.loc[requests["prompt_mode"] == prompt_mode].copy()
    splits = n_splits or (3 if outcome_model == "bias" else 5)
    assignment = _fold_assignments(mode_rows["seed"], splits)
    levels = tuple(sorted(mode_rows["comparison_slot"].astype(str).unique()))
    fold_rows: list[dict[str, Any]] = []
    converged = True
    for fold in range(splits):
        test_seeds = {seed for seed, target in assignment.items() if target == fold}
        train_requests = mode_rows.loc[~mode_rows["seed"].isin(test_seeds)]
        test_requests = mode_rows.loc[mode_rows["seed"].isin(test_seeds)]
        train = _analysis_frame(train_requests, outcome_model, full=False)
        test = _analysis_frame(test_requests, outcome_model, full=False)
        fit = fit_law(train, candidate, outcome_model, levels=levels)
        converged = converged and fit.converged
        fold_rows.append(
            {"fold": fold, **_score_fit(fit, test), "test_rows": len(test)}
        )
    folds = pd.DataFrame(fold_rows)
    full = _analysis_frame(mode_rows, outcome_model, full=True)
    full_fit = fit_law(full, candidate, outcome_model, levels=levels)
    result = {
        "prompt_mode": prompt_mode,
        "outcome_model": outcome_model,
        "candidate": candidate.name,
        "feature_count": len(candidate.features),
        "parent": candidate.parent,
        "interaction_feature": candidate.interaction_feature,
        "cv_folds": splits,
        "cv_primary_loss_mean": float(folds["primary_loss"].mean()),
        "cv_primary_loss_sd": float(folds["primary_loss"].std(ddof=1)),
        "converged": bool(converged and full_fit.converged),
        "full_rows": len(full),
    }
    for metric in (
        "log_loss",
        "brier",
        "negative_log_predictive_density",
        "mae",
        "rmse",
        "r2",
    ):
        if metric in folds:
            result[f"cv_{metric}_mean"] = float(folds[metric].mean())
            result[f"cv_{metric}_sd"] = float(folds[metric].std(ddof=1))
    coefficients = coefficient_table(full_fit)
    coefficients.insert(0, "prompt_mode", prompt_mode)
    return result, coefficients, full_fit


def _holm(values: pd.Series) -> pd.Series:
    result = pd.Series(index=values.index, dtype=float)
    ordered = values.sort_values()
    running = 0.0
    count = len(ordered)
    for rank, (index, value) in enumerate(ordered.items()):
        adjusted = min(1.0, float(value) * (count - rank))
        running = max(running, adjusted)
        result.loc[index] = running
    return result


def _resample_seed_clusters(
    requests: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    seeds = np.asarray(sorted(requests["seed"].unique()), dtype=int)
    sampled = rng.choice(seeds, size=len(seeds), replace=True)
    parts: list[pd.DataFrame] = []
    for pseudo_seed, seed in enumerate(sampled):
        part = requests.loc[requests["seed"] == seed].copy()
        part["seed"] = pseudo_seed
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _interaction_vector(fit: FittedLaw) -> np.ndarray:
    feature = fit.candidate.interaction_feature
    if feature is None:
        return np.empty(0)
    feature_index = fit.candidate.features.index(feature)
    start = len(fit.levels) + feature_index * len(fit.levels)
    return fit.beta[start : start + len(fit.levels)]


def bootstrap_interaction_test(
    requests: pd.DataFrame,
    *,
    prompt_mode: str,
    candidate: Candidate,
    outcome_model: str,
    replicates: int = 2_000,
    random_seed: int = 20_260_807,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if candidate.interaction_feature is None:
        raise ValueError("Bootstrap interaction test requires an interaction candidate")
    mode_rows = requests.loc[requests["prompt_mode"] == prompt_mode].copy()
    full = _analysis_frame(mode_rows, outcome_model, full=True)
    observed_fit = fit_law(full, candidate, outcome_model)
    observed = _interaction_vector(observed_fit)
    rng = np.random.default_rng(random_seed)
    draws: list[np.ndarray] = []
    failed = 0
    for _ in range(replicates):
        sampled = _resample_seed_clusters(mode_rows, rng)
        frame = _analysis_frame(sampled, outcome_model, full=True)
        try:
            fit = fit_law(frame, candidate, outcome_model, levels=observed_fit.levels)
        except (ValueError, np.linalg.LinAlgError):
            failed += 1
            continue
        if fit.converged:
            draws.append(_interaction_vector(fit))
        else:
            failed += 1
    if len(draws) < _minimum_successful_bootstraps(replicates):
        raise RuntimeError("Too many failed interaction bootstrap fits")
    matrix = np.vstack(draws)
    covariance = np.cov(matrix, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    rank = int(np.linalg.matrix_rank(covariance))
    statistic = float(observed @ np.linalg.pinv(covariance) @ observed)
    p_value = float(chi2.sf(statistic, df=max(1, rank)))
    coefficient_rows: list[dict[str, Any]] = []
    for index, level in enumerate(observed_fit.levels):
        draw = matrix[:, index]
        tail = min(float((draw <= 0).mean()), float((draw >= 0).mean()))
        coefficient_rows.append(
            {
                "comparison_slot": level,
                "estimate_standardized": float(observed[index]),
                "bootstrap_ci95_low": float(np.quantile(draw, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(draw, 0.975)),
                "bootstrap_two_sided_p_value": min(1.0, 2.0 * tail),
                "sign_stability": float(max((draw > 0).mean(), (draw < 0).mean())),
            }
        )
    summary = {
        "prompt_mode": prompt_mode,
        "outcome_model": outcome_model,
        "candidate": candidate.name,
        "interaction_feature": candidate.interaction_feature,
        "bootstrap_replicates_requested": replicates,
        "bootstrap_replicates_successful": len(draws),
        "bootstrap_replicates_failed": failed,
        "joint_wald_statistic": statistic,
        "joint_degrees_of_freedom": rank,
        "joint_bootstrap_covariance_p_value": p_value,
    }
    return summary, pd.DataFrame(coefficient_rows)


def fit_candidate_grid(
    requests: pd.DataFrame,
    *,
    outcome_models: tuple[str, ...] = OUTCOME_MODELS,
    interaction_bootstrap_replicates: int = 2_000,
    preapproved_interactions: set[tuple[str, str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparisons: list[dict[str, Any]] = []
    coefficients: list[pd.DataFrame] = []
    modes = tuple(sorted(requests["prompt_mode"].unique()))
    for outcome_model in outcome_models:
        for mode in modes:
            for candidate in CANDIDATES:
                result, terms, _ = cross_validate_candidate(
                    requests,
                    prompt_mode=mode,
                    candidate=candidate,
                    outcome_model=outcome_model,
                )
                comparisons.append(result)
                coefficients.append(terms)
    comparison = pd.DataFrame(comparisons)

    tests: list[dict[str, Any]] = []
    interaction_coefficients: list[pd.DataFrame] = []
    if interaction_bootstrap_replicates > 0:
        for (outcome_model, mode), group in comparison.groupby(
            ["outcome_model", "prompt_mode"]
        ):
            lookup = group.set_index("candidate")["cv_primary_loss_mean"]
            for candidate in CANDIDATES:
                if candidate.parent is None:
                    continue
                improves = float(lookup[candidate.name]) < float(
                    lookup[candidate.parent]
                )
                if not improves:
                    tests.append(
                        {
                            "prompt_mode": mode,
                            "outcome_model": outcome_model,
                            "candidate": candidate.name,
                            "interaction_feature": candidate.interaction_feature,
                            "improves_parent_held_seed_loss": False,
                            "joint_bootstrap_covariance_p_value": 1.0,
                        }
                    )
                    continue
                summary, term_draws = bootstrap_interaction_test(
                    requests,
                    prompt_mode=str(mode),
                    candidate=candidate,
                    outcome_model=str(outcome_model),
                    replicates=interaction_bootstrap_replicates,
                )
                summary["improves_parent_held_seed_loss"] = True
                tests.append(summary)
                term_draws.insert(0, "prompt_mode", mode)
                term_draws.insert(1, "outcome_model", outcome_model)
                term_draws.insert(2, "candidate", candidate.name)
                interaction_coefficients.append(term_draws)
    test_frame = pd.DataFrame(tests)
    if not test_frame.empty:
        test_frame["holm_p_value"] = test_frame.groupby(
            ["outcome_model", "prompt_mode"], group_keys=False
        )["joint_bootstrap_covariance_p_value"].apply(_holm)
        test_frame["interaction_eligible"] = test_frame[
            "improves_parent_held_seed_loss"
        ].astype(bool) & (test_frame["holm_p_value"] < 0.05)
    selected_rows: list[dict[str, Any]] = []
    for (outcome_model, mode), group in comparison.groupby(
        ["outcome_model", "prompt_mode"]
    ):
        eligible = group.loc[group["converged"].astype(bool)].copy()
        if not test_frame.empty:
            eligibility = test_frame.loc[
                (test_frame["outcome_model"] == outcome_model)
                & (test_frame["prompt_mode"] == mode)
            ].set_index("candidate")["interaction_eligible"]
            is_interaction = eligible["interaction_feature"].notna()
            eligible = eligible.loc[
                ~is_interaction
                | eligible["candidate"].map(eligibility).fillna(False).astype(bool)
            ]
        elif preapproved_interactions is not None:
            approved = {
                candidate
                for approved_outcome, approved_mode, candidate in preapproved_interactions
                if approved_outcome == outcome_model and approved_mode == mode
            }
            eligible = eligible.loc[
                eligible["interaction_feature"].isna()
                | eligible["candidate"].isin(approved)
            ]
        else:
            eligible = eligible.loc[eligible["interaction_feature"].isna()]
        best = eligible.loc[eligible["cv_primary_loss_mean"].idxmin()]
        tolerance = float(best["cv_primary_loss_sd"]) / math.sqrt(
            float(best["cv_folds"])
        )
        near = eligible.loc[
            eligible["cv_primary_loss_mean"]
            <= float(best["cv_primary_loss_mean"]) + tolerance
        ]
        order = {candidate.name: index for index, candidate in enumerate(CANDIDATES)}
        near = near.assign(registry_order=near["candidate"].map(order))
        selected = (
            near.sort_values(["feature_count", "registry_order"])
            .iloc[0]
            .drop(labels="registry_order")
        )
        selected_rows.append(selected.to_dict())
    selected = pd.DataFrame(selected_rows)
    coefficient_frame = pd.concat(coefficients, ignore_index=True)
    interaction_coefficient_frame = (
        pd.concat(interaction_coefficients, ignore_index=True)
        if interaction_coefficients
        else pd.DataFrame()
    )
    if not interaction_coefficient_frame.empty:
        interaction_coefficient_frame["model_specific_holm_p_value"] = (
            interaction_coefficient_frame.groupby(
                ["outcome_model", "prompt_mode", "candidate"],
                group_keys=False,
            )["bootstrap_two_sided_p_value"].apply(_holm)
        )
    if not test_frame.empty and not interaction_coefficient_frame.empty:
        interaction_output = test_frame.merge(
            interaction_coefficient_frame,
            on=["prompt_mode", "outcome_model", "candidate"],
            how="left",
            suffixes=("", "_coefficient"),
        )
    else:
        interaction_output = test_frame
    return comparison, selected, coefficient_frame, interaction_output


def bootstrap_reselection_stability(
    requests: pd.DataFrame,
    *,
    preapproved_interactions: set[tuple[str, str, str]],
    outcome_models: tuple[str, ...] = OUTCOME_MODELS,
    replicates: int = 2_000,
    random_seed: int = 20_260_807,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)
    selections: list[pd.DataFrame] = []
    failed = 0
    for replicate in range(replicates):
        sampled = _resample_seed_clusters(requests, rng)
        try:
            _, selected, _, _ = fit_candidate_grid(
                sampled,
                outcome_models=outcome_models,
                interaction_bootstrap_replicates=0,
                preapproved_interactions=preapproved_interactions,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            failed += 1
            continue
        selected.insert(0, "bootstrap_replicate", replicate)
        selections.append(selected)
    if len(selections) < _minimum_successful_bootstraps(replicates):
        raise RuntimeError("Too many failed full-reselection bootstrap replicates")
    draws = pd.concat(selections, ignore_index=True)
    frequency = (
        draws.groupby(["outcome_model", "prompt_mode", "candidate"], sort=True)
        .size()
        .rename("selected_replicates")
        .reset_index()
    )
    totals = frequency.groupby(["outcome_model", "prompt_mode"])[
        "selected_replicates"
    ].transform("sum")
    frequency["selection_frequency"] = frequency["selected_replicates"] / totals
    frequency["successful_replicates"] = len(selections)
    frequency["failed_replicates"] = failed
    frequency["interaction_was_preapproved"] = [
        (str(row.outcome_model), str(row.prompt_mode), str(row.candidate))
        in preapproved_interactions
        for row in frequency.itertuples(index=False)
    ]
    return frequency, draws


def validate_held_axis(
    requests: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    axis: str,
) -> pd.DataFrame:
    if axis not in {"N", "L"}:
        raise ValueError("held axis must be N or L")
    rows: list[dict[str, Any]] = []
    for selection in selected.itertuples(index=False):
        mode_rows = requests.loc[
            requests["prompt_mode"] == selection.prompt_mode
        ].copy()
        outcome_model = str(selection.outcome_model)
        candidate = CANDIDATE_BY_NAME[str(selection.candidate)]
        levels = tuple(sorted(mode_rows["comparison_slot"].astype(str).unique()))
        for held_value in sorted(mode_rows[axis].unique()):
            train_requests = mode_rows.loc[mode_rows[axis] != held_value]
            test_requests = mode_rows.loc[mode_rows[axis] == held_value]
            train = _analysis_frame(train_requests, outcome_model, full=True)
            test = _analysis_frame(test_requests, outcome_model, full=True)
            fit = fit_law(train, candidate, outcome_model, levels=levels)
            metrics = _score_fit(fit, test)
            boundary_values = (mode_rows[axis].min(), mode_rows[axis].max())
            rows.append(
                {
                    "axis": axis,
                    "held_value": held_value,
                    "validation_kind": (
                        "boundary_extrapolation"
                        if held_value in boundary_values
                        else "interpolation"
                    ),
                    "prompt_mode": selection.prompt_mode,
                    "outcome_model": outcome_model,
                    "candidate": candidate.name,
                    **metrics,
                    "test_rows": len(test),
                }
            )
    return pd.DataFrame(rows)


def nested_seed_validation(
    requests: pd.DataFrame,
    *,
    outcome_models: tuple[str, ...] = OUTCOME_MODELS,
    interaction_bootstrap_replicates: int = 2_000,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outcome_model in outcome_models:
        for mode in sorted(requests["prompt_mode"].unique()):
            mode_rows = requests.loc[requests["prompt_mode"] == mode].copy()
            outer_splits = 3 if outcome_model == "bias" else 5
            assignment = _fold_assignments(mode_rows["seed"], outer_splits)
            levels = tuple(sorted(mode_rows["comparison_slot"].astype(str).unique()))
            for fold in range(outer_splits):
                test_seeds = {
                    seed for seed, target in assignment.items() if target == fold
                }
                train_requests = mode_rows.loc[~mode_rows["seed"].isin(test_seeds)]
                test_requests = mode_rows.loc[mode_rows["seed"].isin(test_seeds)]
                _, selected, _, _ = fit_candidate_grid(
                    train_requests,
                    outcome_models=(outcome_model,),
                    interaction_bootstrap_replicates=(interaction_bootstrap_replicates),
                )
                candidate = CANDIDATE_BY_NAME[str(selected.iloc[0]["candidate"])]
                train = _analysis_frame(train_requests, outcome_model, full=False)
                test = _analysis_frame(test_requests, outcome_model, full=False)
                fit = fit_law(
                    train,
                    candidate,
                    outcome_model,
                    levels=levels,
                )
                rows.append(
                    {
                        "validation": "nested_held_seed",
                        "fold": fold,
                        "held_seeds": ",".join(
                            str(seed) for seed in sorted(test_seeds)
                        ),
                        "prompt_mode": mode,
                        "outcome_model": outcome_model,
                        "selected_candidate_in_outer_training": candidate.name,
                        **_score_fit(fit, test),
                        "test_rows": len(test),
                    }
                )
    return pd.DataFrame(rows)


def nested_held_axis_validation(
    requests: pd.DataFrame,
    *,
    axis: str,
    outcome_models: tuple[str, ...] = OUTCOME_MODELS,
    interaction_bootstrap_replicates: int = 2_000,
) -> pd.DataFrame:
    if axis not in {"N", "L"}:
        raise ValueError("held axis must be N or L")
    rows: list[dict[str, Any]] = []
    for outcome_model in outcome_models:
        for mode in sorted(requests["prompt_mode"].unique()):
            mode_rows = requests.loc[requests["prompt_mode"] == mode].copy()
            levels = tuple(sorted(mode_rows["comparison_slot"].astype(str).unique()))
            boundary_values = (mode_rows[axis].min(), mode_rows[axis].max())
            for held_value in sorted(mode_rows[axis].unique()):
                train_requests = mode_rows.loc[mode_rows[axis] != held_value]
                test_requests = mode_rows.loc[mode_rows[axis] == held_value]
                _, selected, _, _ = fit_candidate_grid(
                    train_requests,
                    outcome_models=(outcome_model,),
                    interaction_bootstrap_replicates=(interaction_bootstrap_replicates),
                )
                candidate = CANDIDATE_BY_NAME[str(selected.iloc[0]["candidate"])]
                train = _analysis_frame(train_requests, outcome_model, full=True)
                test = _analysis_frame(test_requests, outcome_model, full=True)
                fit = fit_law(
                    train,
                    candidate,
                    outcome_model,
                    levels=levels,
                )
                rows.append(
                    {
                        "validation": f"nested_held_{axis}",
                        "held_value": held_value,
                        "validation_kind": (
                            "boundary_extrapolation"
                            if held_value in boundary_values
                            else "interpolation"
                        ),
                        "prompt_mode": mode,
                        "outcome_model": outcome_model,
                        "selected_candidate_in_outer_training": candidate.name,
                        **_score_fit(fit, test),
                        "test_rows": len(test),
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_selected_coefficients(
    requests: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    replicates: int = 2_000,
    random_seed: int = 20_260_807,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    output: list[dict[str, Any]] = []
    for selection_index, selection in enumerate(selected.itertuples(index=False)):
        mode = str(selection.prompt_mode)
        outcome_model = str(selection.outcome_model)
        candidate = CANDIDATE_BY_NAME[str(selection.candidate)]
        mode_rows = requests.loc[requests["prompt_mode"] == mode].copy()
        full = _analysis_frame(mode_rows, outcome_model, full=True)
        observed_fit = fit_law(full, candidate, outcome_model)
        draws: list[np.ndarray] = []
        kappa_draws: list[np.ndarray] = []
        failed = 0
        for _ in range(replicates):
            sampled = _resample_seed_clusters(mode_rows, rng)
            frame = _analysis_frame(sampled, outcome_model, full=True)
            try:
                fit = fit_law(
                    frame,
                    candidate,
                    outcome_model,
                    levels=observed_fit.levels,
                )
            except (ValueError, np.linalg.LinAlgError):
                failed += 1
                continue
            if not fit.converged:
                failed += 1
                continue
            draws.append(fit.beta)
            if fit.log_kappa is not None:
                kappa_draws.append(np.exp(fit.log_kappa))
        if len(draws) < _minimum_successful_bootstraps(replicates):
            raise RuntimeError(
                f"Too many failed coefficient bootstraps for {mode}/{outcome_model}"
            )
        matrix = np.vstack(draws)
        names = [f"intercept[{level}]" for level in observed_fit.levels]
        names.extend(
            f"{feature}[{level}]"
            for feature in candidate.features
            for level in observed_fit.levels
        )
        for index, name in enumerate(names):
            draw = matrix[:, index]
            tail = min(float((draw <= 0).mean()), float((draw >= 0).mean()))
            output.append(
                {
                    "prompt_mode": mode,
                    "outcome_model": outcome_model,
                    "candidate": candidate.name,
                    "term": name,
                    "estimate_standardized": float(observed_fit.beta[index]),
                    "bootstrap_ci95_low": float(np.quantile(draw, 0.025)),
                    "bootstrap_ci95_high": float(np.quantile(draw, 0.975)),
                    "bootstrap_two_sided_p_value": min(1.0, 2.0 * tail),
                    "sign_stability": float(max((draw > 0).mean(), (draw < 0).mean())),
                    "bootstrap_replicates_successful": len(draws),
                    "bootstrap_replicates_failed": failed,
                    "selection_index": selection_index,
                }
            )
        if observed_fit.log_kappa is not None:
            kappa_matrix = np.vstack(kappa_draws)
            for index, level in enumerate(observed_fit.levels):
                draw = kappa_matrix[:, index]
                observed_kappa = float(np.exp(observed_fit.log_kappa[index]))
                output.append(
                    {
                        "prompt_mode": mode,
                        "outcome_model": outcome_model,
                        "candidate": candidate.name,
                        "term": f"kappa[{level}]",
                        "estimate_standardized": observed_kappa,
                        "bootstrap_ci95_low": float(np.quantile(draw, 0.025)),
                        "bootstrap_ci95_high": float(np.quantile(draw, 0.975)),
                        "bootstrap_two_sided_p_value": math.nan,
                        "sign_stability": 1.0,
                        "bootstrap_replicates_successful": len(draws),
                        "bootstrap_replicates_failed": failed,
                        "selection_index": selection_index,
                    }
                )
    return pd.DataFrame(output)


def leave_one_model_out_structure(
    requests: pd.DataFrame,
    *,
    outcome_models: tuple[str, ...] = OUTCOME_MODELS,
    interaction_bootstrap_replicates: int = 0,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for omitted in sorted(requests["comparison_slot"].unique()):
        subset = requests.loc[requests["comparison_slot"] != omitted]
        _, selected, _, _ = fit_candidate_grid(
            subset,
            outcome_models=outcome_models,
            interaction_bootstrap_replicates=interaction_bootstrap_replicates,
        )
        selected.insert(0, "omitted_comparison_slot", omitted)
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)
