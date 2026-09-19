from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .laws import (
    OUTCOME_MODELS,
    Candidate,
    FeatureScaler,
    FittedLaw,
    design_matrix,
)


def _torch_device(name: str):
    import torch

    if name not in {"cpu", "cuda"}:
        raise ValueError("Torch law device must be cpu or cuda")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA analysis requested but torch.cuda is unavailable")
    return torch.device(name)


def _fit_ols_torch(
    x: np.ndarray,
    y: np.ndarray,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray, bool]:
    import torch

    target = _torch_device(device)
    tx = torch.tensor(x, dtype=torch.float64, device=target)
    ty = torch.tensor(y, dtype=torch.float64, device=target)
    beta = torch.linalg.lstsq(tx, ty).solution
    residual = ty - tx @ beta
    degrees = max(1, len(y) - x.shape[1])
    sigma2 = (residual @ residual) / degrees
    covariance = sigma2 * torch.linalg.pinv(tx.T @ tx)
    return (
        beta.detach().cpu().numpy(),
        covariance.detach().cpu().numpy(),
        bool(torch.isfinite(beta).all().item()),
    )


def _fit_binomial_torch(
    x: np.ndarray,
    successes: np.ndarray,
    totals: np.ndarray,
    *,
    intercept_columns: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, bool]:
    import torch
    import torch.nn.functional as functional

    target = _torch_device(device)
    tx = torch.tensor(x, dtype=torch.float64, device=target)
    ts = torch.tensor(successes, dtype=torch.float64, device=target)
    tt = torch.tensor(totals, dtype=torch.float64, device=target)
    beta = torch.zeros(tx.shape[1], dtype=torch.float64, device=target)
    beta.requires_grad_(True)
    ridge = 1e-8
    optimizer = torch.optim.LBFGS(
        [beta],
        lr=1.0,
        max_iter=1_000,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        eta = tx @ beta
        loss = (tt * functional.softplus(eta) - ts * eta).sum()
        if beta.numel() > intercept_columns:
            loss = loss + 0.5 * ridge * (beta[intercept_columns:] ** 2).sum()
        loss.backward()
        return loss

    try:
        final_loss = optimizer.step(closure)
        finite = bool(
            torch.isfinite(beta).all().item()
            and torch.isfinite(final_loss).all().item()
        )
    except RuntimeError:
        finite = False
    with torch.no_grad():
        probability = torch.sigmoid(tx @ beta).clamp(1e-9, 1.0 - 1e-9)
        weight = tt * probability * (1.0 - probability)
        information = tx.T @ (tx * weight[:, None])
        if tx.shape[1] > intercept_columns:
            information[intercept_columns:, intercept_columns:] += ridge * torch.eye(
                tx.shape[1] - intercept_columns,
                dtype=torch.float64,
                device=target,
            )
        covariance = torch.linalg.pinv(information)
    return (
        beta.detach().cpu().numpy(),
        covariance.detach().cpu().numpy(),
        finite,
    )


def _fit_beta_binomial_torch(
    x: np.ndarray,
    successes: np.ndarray,
    totals: np.ndarray,
    model_index: np.ndarray,
    *,
    intercept_columns: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    import torch

    target = _torch_device(device)
    beta_start, _, _ = _fit_binomial_torch(
        x,
        successes,
        totals,
        intercept_columns=intercept_columns,
        device=device,
    )
    model_count = int(model_index.max()) + 1
    start = np.concatenate([beta_start, np.full(model_count, math.log(100.0))])
    tx = torch.tensor(x, dtype=torch.float64, device=target)
    ts = torch.tensor(successes, dtype=torch.float64, device=target)
    tt = torch.tensor(totals, dtype=torch.float64, device=target)
    tm = torch.tensor(model_index, dtype=torch.long, device=target)
    theta = torch.tensor(start, dtype=torch.float64, device=target)
    theta.requires_grad_(True)
    ridge = 1e-8

    def objective(value):
        beta = value[: tx.shape[1]]
        log_kappa = value[tx.shape[1] :].clamp(-8.0, 20.0)
        mu = torch.sigmoid(tx @ beta).clamp(1e-9, 1.0 - 1e-9)
        kappa = torch.exp(log_kappa[tm])
        a = mu * kappa
        b = (1.0 - mu) * kappa
        log_combination = (
            torch.lgamma(tt + 1.0)
            - torch.lgamma(ts + 1.0)
            - torch.lgamma(tt - ts + 1.0)
        )
        log_beta_observed = (
            torch.lgamma(ts + a)
            + torch.lgamma(tt - ts + b)
            - torch.lgamma(tt + a + b)
        )
        log_beta_prior = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
        loss = -(log_combination + log_beta_observed - log_beta_prior).sum()
        if beta.numel() > intercept_columns:
            loss = loss + 0.5 * ridge * (beta[intercept_columns:] ** 2).sum()
        return loss

    optimizer = torch.optim.LBFGS(
        [theta],
        lr=0.8,
        max_iter=1_500,
        tolerance_grad=1e-8,
        tolerance_change=1e-11,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = objective(theta)
        loss.backward()
        return loss

    try:
        final_loss = optimizer.step(closure)
        finite = bool(
            torch.isfinite(theta).all().item()
            and torch.isfinite(final_loss).all().item()
        )
    except RuntimeError:
        finite = False
    final = theta.detach().clone()
    final[x.shape[1] :] = final[x.shape[1] :].clamp(-8.0, 20.0)
    try:
        hessian = torch.autograd.functional.hessian(objective, final)
        covariance = torch.linalg.pinv(hessian)[: x.shape[1], : x.shape[1]]
    except RuntimeError:
        covariance = torch.full(
            (x.shape[1], x.shape[1]),
            float("nan"),
            dtype=torch.float64,
            device=target,
        )
    return (
        final[: x.shape[1]].cpu().numpy(),
        covariance.cpu().numpy(),
        final[x.shape[1] :].cpu().numpy(),
        finite,
    )


def fit_law_torch(
    frame: pd.DataFrame,
    candidate: Candidate,
    outcome_model: str,
    *,
    levels: tuple[str, ...] | None = None,
    device: str = "cuda",
) -> FittedLaw:
    """Numerically matched Torch backend for CPU or CUDA law fitting."""

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
        beta, covariance, converged = _fit_ols_torch(
            x,
            frame["trimmed_signed_bias_10"].to_numpy(dtype=float),
            device=device,
        )
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
    beta, covariance, converged = _fit_binomial_torch(
        x,
        successes,
        totals,
        intercept_columns=len(resolved_levels),
        device=device,
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
    beta, covariance, log_kappa, converged = _fit_beta_binomial_torch(
        x,
        successes,
        totals,
        model_index,
        intercept_columns=len(resolved_levels),
        device=device,
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
