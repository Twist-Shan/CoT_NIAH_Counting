from __future__ import annotations

"""Explicit probability models and diagnostics for exact-count accuracy.

Accuracy is binary at request level.  Requests sharing model slot, N, and L
are therefore represented as a success count out of a known number of paired
seeds.  The registered search compares three Binomial links and one
Beta-Binomial model that permits extra-binomial seed heterogeneity.
"""

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import (
    betabinom,
    binom,
    cramervonmises,
    norm,
    shapiro,
)


@dataclass(frozen=True)
class AccuracyFamily:
    name: str
    distribution: str
    link: str
    extra_parameters: int = 0


ACCURACY_FAMILIES = (
    AccuracyFamily("binomial_logit", "binomial", "logit"),
    AccuracyFamily("binomial_probit", "binomial", "probit"),
    AccuracyFamily("binomial_cloglog", "binomial", "cloglog"),
    AccuracyFamily(
        "beta_binomial_logit",
        "beta_binomial",
        "logit",
        extra_parameters=1,
    ),
)
ACCURACY_FAMILY_BY_NAME = {family.name: family for family in ACCURACY_FAMILIES}


@dataclass(frozen=True)
class AccuracyFit:
    beta: np.ndarray
    probability: np.ndarray
    standard_error: np.ndarray
    p_value: np.ndarray
    converged: bool
    concentration: float | None
    log_likelihood: float


def inverse_link(eta: np.ndarray, link: str) -> np.ndarray:
    eta = np.asarray(eta, dtype=float)
    if link == "logit":
        probability = expit(eta)
    elif link == "probit":
        probability = norm.cdf(eta)
    elif link == "cloglog":
        bounded = np.clip(eta, -30.0, 20.0)
        probability = -np.expm1(-np.exp(bounded))
    else:
        raise ValueError(f"Unsupported accuracy link: {link}")
    return np.clip(probability, 1e-8, 1.0 - 1e-8)


def _inverse_link_derivative(eta: np.ndarray, link: str) -> np.ndarray:
    eta = np.asarray(eta, dtype=float)
    if link == "logit":
        probability = expit(eta)
        return probability * (1.0 - probability)
    if link == "probit":
        return norm.pdf(eta)
    if link == "cloglog":
        bounded = np.clip(eta, -30.0, 20.0)
        exponential = np.exp(bounded)
        return np.exp(bounded - exponential)
    raise ValueError(f"Unsupported accuracy link: {link}")


def _validate_counts(
    successes: np.ndarray,
    trials: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    if successes.shape != trials.shape or successes.ndim != 1:
        raise ValueError("successes and trials must be aligned one-dimensional arrays")
    if (
        not np.isfinite(successes).all()
        or not np.isfinite(trials).all()
        or (trials <= 0).any()
        or (successes < 0).any()
        or (successes > trials).any()
    ):
        raise ValueError("Invalid Binomial success/trial counts")
    if not np.allclose(successes, np.rint(successes)) or not np.allclose(
        trials,
        np.rint(trials),
    ):
        raise ValueError("Binomial success/trial counts must be integers")
    return np.rint(successes).astype(int), np.rint(trials).astype(int)


def predictive_log_likelihood(
    successes: np.ndarray,
    trials: np.ndarray,
    probability: np.ndarray,
    family: AccuracyFamily,
    *,
    concentration: float | None = None,
) -> float:
    successes, trials = _validate_counts(successes, trials)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    if family.distribution == "binomial":
        values = binom.logpmf(successes, trials, probability)
    elif family.distribution == "beta_binomial":
        if concentration is None or not math.isfinite(concentration) or concentration <= 0:
            raise ValueError("Beta-Binomial concentration must be positive")
        alpha = probability * concentration
        beta = (1.0 - probability) * concentration
        values = betabinom.logpmf(successes, trials, alpha, beta)
    else:
        raise ValueError(f"Unsupported accuracy distribution: {family.distribution}")
    if not np.isfinite(values).all():
        return -math.inf
    return float(values.sum())


def _binomial_start(
    x: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
    link: str,
) -> np.ndarray:
    ridge = 1e-8

    def objective(beta: np.ndarray) -> float:
        probability = inverse_link(x @ beta, link)
        value = -float(binom.logpmf(successes, trials, probability).sum())
        return value + 0.5 * ridge * float(beta[1:] @ beta[1:])

    result = minimize(
        objective,
        np.zeros(x.shape[1], dtype=float),
        method="L-BFGS-B",
        options={"maxiter": 2_000, "ftol": 1e-10},
    )
    return np.asarray(result.x, dtype=float)


def _numerical_hessian(
    objective: Callable[[np.ndarray], float],
    theta: np.ndarray,
) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    dimensions = len(theta)
    steps = 1e-4 * (1.0 + np.abs(theta))
    hessian = np.zeros((dimensions, dimensions), dtype=float)
    center = float(objective(theta))
    for index in range(dimensions):
        high = theta.copy()
        low = theta.copy()
        high[index] += steps[index]
        low[index] -= steps[index]
        hessian[index, index] = (
            float(objective(high)) - 2.0 * center + float(objective(low))
        ) / (steps[index] ** 2)
        for other in range(index):
            pp = theta.copy()
            pm = theta.copy()
            mp = theta.copy()
            mm = theta.copy()
            pp[index] += steps[index]
            pp[other] += steps[other]
            pm[index] += steps[index]
            pm[other] -= steps[other]
            mp[index] -= steps[index]
            mp[other] += steps[other]
            mm[index] -= steps[index]
            mm[other] -= steps[other]
            value = (
                float(objective(pp))
                - float(objective(pm))
                - float(objective(mp))
                + float(objective(mm))
            ) / (4.0 * steps[index] * steps[other])
            hessian[index, other] = hessian[other, index] = value
    return hessian


def fit_accuracy_distribution(
    x: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
    family: AccuracyFamily,
    *,
    compute_inference: bool = True,
) -> AccuracyFit:
    """Fit one registered Binomial or Beta-Binomial response surface."""

    x = np.asarray(x, dtype=float)
    successes, trials = _validate_counts(successes, trials)
    if x.ndim != 2 or len(x) != len(successes):
        raise ValueError("Accuracy design matrix does not align with counts")
    ridge = 1e-8
    start_beta = _binomial_start(x, successes, trials, family.link)

    if family.distribution == "binomial":
        def objective(beta: np.ndarray) -> float:
            probability = inverse_link(x @ beta, family.link)
            loss = -float(binom.logpmf(successes, trials, probability).sum())
            return loss + 0.5 * ridge * float(beta[1:] @ beta[1:])

        result = minimize(
            objective,
            start_beta,
            method="L-BFGS-B",
            options={"maxiter": 2_000, "ftol": 1e-10},
        )
        beta = np.asarray(result.x, dtype=float)
        probability = inverse_link(x @ beta, family.link)
        derivative = _inverse_link_derivative(x @ beta, family.link)
        weight = trials * derivative**2 / (probability * (1.0 - probability))
        information = x.T @ (x * weight[:, None])
        if x.shape[1] > 1:
            information[1:, 1:] += ridge * np.eye(x.shape[1] - 1)
        covariance = np.linalg.pinv(information)
        concentration = None
        log_likelihood = predictive_log_likelihood(
            successes,
            trials,
            probability,
            family,
        )
        converged = bool(result.success)
    elif family.distribution == "beta_binomial":
        initial = np.concatenate((start_beta, [math.log(20.0)]))

        def objective(theta: np.ndarray) -> float:
            beta = theta[:-1]
            concentration = math.exp(float(theta[-1]))
            probability = inverse_link(x @ beta, family.link)
            log_likelihood = predictive_log_likelihood(
                successes,
                trials,
                probability,
                family,
                concentration=concentration,
            )
            penalty = 0.5 * ridge * float(beta[1:] @ beta[1:])
            return -log_likelihood + penalty

        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=[(None, None)] * x.shape[1] + [(-5.0, 12.0)],
            options={"maxiter": 2_000, "ftol": 1e-10},
        )
        theta = np.asarray(result.x, dtype=float)
        beta = theta[:-1]
        concentration = math.exp(float(theta[-1]))
        probability = inverse_link(x @ beta, family.link)
        if compute_inference:
            hessian = _numerical_hessian(objective, theta)
            covariance = np.linalg.pinv(hessian)[
                : x.shape[1], : x.shape[1]
            ]
            hessian_finite = bool(np.isfinite(hessian).all())
        else:
            covariance = np.full(
                (x.shape[1], x.shape[1]),
                np.nan,
                dtype=float,
            )
            hessian_finite = True
        log_likelihood = predictive_log_likelihood(
            successes,
            trials,
            probability,
            family,
            concentration=concentration,
        )
        converged = bool(result.success and hessian_finite)
    else:
        raise ValueError(f"Unsupported accuracy distribution: {family.distribution}")

    standard_error = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    statistic = np.divide(
        beta,
        standard_error,
        out=np.zeros_like(beta),
        where=standard_error > 0,
    )
    p_value = 2.0 * norm.sf(np.abs(statistic))
    return AccuracyFit(
        beta=beta,
        probability=probability,
        standard_error=standard_error,
        p_value=p_value,
        converged=converged,
        concentration=concentration,
        log_likelihood=log_likelihood,
    )


def randomized_quantile_residuals(
    successes: np.ndarray,
    trials: np.ndarray,
    probability: np.ndarray,
    family: AccuracyFamily,
    *,
    concentration: float | None,
    random_seed: int,
) -> np.ndarray:
    """Dunn-Smyth residuals for a discrete fitted accuracy distribution."""

    successes, trials = _validate_counts(successes, trials)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    if family.distribution == "binomial":
        lower = binom.cdf(successes - 1, trials, probability)
        upper = binom.cdf(successes, trials, probability)
    elif family.distribution == "beta_binomial":
        if concentration is None or concentration <= 0:
            raise ValueError("Beta-Binomial concentration must be positive")
        alpha = probability * concentration
        beta = (1.0 - probability) * concentration
        lower = betabinom.cdf(successes - 1, trials, alpha, beta)
        upper = betabinom.cdf(successes, trials, alpha, beta)
    else:
        raise ValueError(f"Unsupported accuracy distribution: {family.distribution}")
    rng = np.random.default_rng(random_seed)
    uniform = lower + rng.random(len(lower)) * np.maximum(upper - lower, 0.0)
    return norm.ppf(np.clip(uniform, 1e-8, 1 - 1e-8))


def quantile_residual_diagnostics(residuals: np.ndarray) -> dict[str, float | int]:
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < 3:
        return {
            "qq_residuals": int(len(residuals)),
            "qq_correlation_r2": math.nan,
            "residual_mean": math.nan,
            "residual_sd": math.nan,
            "shapiro_w": math.nan,
            "shapiro_p_value": math.nan,
            "cramer_von_mises_statistic": math.nan,
            "cramer_von_mises_p_value": math.nan,
        }
    ordered = np.sort(residuals)
    theoretical = norm.ppf((np.arange(len(ordered)) + 0.5) / len(ordered))
    correlation = float(np.corrcoef(theoretical, ordered)[0, 1])
    shapiro_rows = residuals[:5_000]
    shapiro_result = shapiro(shapiro_rows)
    cvm_result = cramervonmises(residuals, "norm")
    return {
        "qq_residuals": int(len(residuals)),
        "qq_correlation_r2": correlation**2,
        "residual_mean": float(residuals.mean()),
        "residual_sd": float(residuals.std(ddof=1)),
        "shapiro_w": float(shapiro_result.statistic),
        "shapiro_p_value": float(shapiro_result.pvalue),
        "cramer_von_mises_statistic": float(cvm_result.statistic),
        "cramer_von_mises_p_value": float(cvm_result.pvalue),
    }
