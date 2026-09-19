"""Discovery-frozen count-subspace removal and norm-matched controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, MutableMapping

import numpy as np
import torch


@dataclass(frozen=True)
class PromptRemovalGeometry:
    classes: np.ndarray
    centroids: np.ndarray
    basis: np.ndarray
    control_basis: np.ndarray
    centroid_variance_capture: float


def fit_prompt_removal_geometry(
    states: np.ndarray,
    counts: np.ndarray,
    *,
    rank: int = 3,
    required_classes: np.ndarray | None = None,
) -> PromptRemovalGeometry:
    """Fit the count-centroid basis and an orthogonal within-count control."""

    values = np.asarray(states, dtype=np.float64)
    labels = np.asarray(counts)
    if values.ndim != 2 or labels.ndim != 1 or len(values) != len(labels):
        raise ValueError("states and counts must be row matched")
    if not np.isfinite(values).all():
        raise ValueError("states contain non-finite values")
    classes = np.unique(labels) if required_classes is None else np.asarray(required_classes)
    if len(classes) <= rank:
        raise ValueError("too few classes for requested rank")
    missing = [value for value in classes if not np.any(labels == value)]
    if missing:
        raise ValueError(f"missing classes: {missing}")
    centroids = np.stack([values[labels == value].mean(0) for value in classes])
    centered = centroids - centroids.mean(0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    if singular[rank - 1] <= 1e-12:
        raise ValueError("count-centroid geometry is rank deficient")
    basis = vt[:rank].T

    lookup = {value: centroids[index] for index, value in enumerate(classes)}
    residual = values - np.stack([lookup[value] for value in labels])
    residual -= (residual @ basis) @ basis.T
    _, control_singular, control_vt = np.linalg.svd(residual, full_matrices=False)
    if control_singular[rank - 1] <= 1e-12:
        raise ValueError("within-count control geometry is rank deficient")
    control = control_vt[:rank].T
    control -= basis @ (basis.T @ control)
    control, _ = np.linalg.qr(control, mode="reduced")
    control = control[:, :rank]
    if np.linalg.norm(basis.T @ control, ord="fro") > 1e-7:
        raise RuntimeError("control basis is not orthogonal to count basis")
    total = float(np.square(singular).sum())
    capture = float(np.square(singular[:rank]).sum() / max(total, 1e-24))
    return PromptRemovalGeometry(
        classes=np.asarray(classes).copy(),
        centroids=centroids,
        basis=basis,
        control_basis=control,
        centroid_variance_capture=capture,
    )


def projected_delta(selected: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project within-prompt endpoint differences onto a frozen basis."""

    if selected.ndim != 3:
        raise ValueError("selected states must have shape [batch, positions, hidden]")
    # Perform the geometry in fp32 even when the model residual is bf16.  The
    # final replacement is quantized below and audited in the model dtype.
    work = selected.float()
    axes = basis.to(device=selected.device, dtype=torch.float32)
    if axes.ndim != 2 or axes.shape[0] != selected.shape[-1]:
        raise ValueError("basis width does not match selected states")
    centered = work - work.mean(dim=1, keepdim=True)
    return (centered @ axes) @ axes.T


def normmatched_orthogonal_delta(
    selected: torch.Tensor,
    target_basis: torch.Tensor,
    control_basis: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a control delta with the candidate norm in every batch item."""

    target = projected_delta(selected, target_basis)
    nuisance = projected_delta(selected, control_basis)
    target_norm = torch.linalg.vector_norm(target, dim=(1, 2), keepdim=True)
    nuisance_norm = torch.linalg.vector_norm(nuisance, dim=(1, 2), keepdim=True)
    if torch.any((nuisance_norm <= epsilon) & (target_norm > epsilon)):
        raise RuntimeError("orthogonal control has zero norm for a nonzero target")
    scaled = nuisance * target_norm / torch.clamp(nuisance_norm, min=epsilon)
    return scaled, target


def single_position_projected_delta(
    selected: torch.Tensor,
    basis: torch.Tensor,
    center: torch.Tensor,
) -> torch.Tensor:
    """Project a single-token state relative to a discovery-frozen center."""

    if selected.ndim != 3 or selected.shape[1] != 1:
        raise ValueError("single-position states must have shape [batch, 1, hidden]")
    work = selected.float()
    axes = basis.to(device=selected.device, dtype=torch.float32)
    origin = center.to(device=selected.device, dtype=torch.float32)
    if axes.ndim != 2 or axes.shape[0] != selected.shape[-1]:
        raise ValueError("basis width does not match selected states")
    if origin.ndim != 1 or origin.shape[0] != selected.shape[-1]:
        raise ValueError("center width does not match selected states")
    centered = work - origin.view(1, 1, -1)
    return (centered @ axes) @ axes.T


def _realized_replacement(
    selected: torch.Tensor, delta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    replacement = (selected.float() - delta.float()).to(dtype=selected.dtype)
    realized = selected.float() - replacement.float()
    return replacement, realized


def _closest_realized_norm_replacement(
    selected: torch.Tensor,
    direction: torch.Tensor,
    target_norm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    direction_norm = torch.linalg.vector_norm(direction)
    if direction_norm <= 1e-12 and target_norm > 1e-12:
        raise RuntimeError("orthogonal control has zero norm for a nonzero target")
    unit = direction / torch.clamp(direction_norm, min=1e-12)

    def candidate_at(scale: float) -> tuple[torch.Tensor, torch.Tensor, float]:
        candidate_replacement, candidate_delta = _realized_replacement(
            selected, unit * float(scale)
        )
        candidate_norm = float(torch.linalg.vector_norm(candidate_delta).detach().cpu())
        return candidate_replacement, candidate_delta, candidate_norm

    # A bf16-written norm is a staircase, not a continuous function of the
    # fp32 scale. Bracket the target and retain the closest realizable point.
    target_scalar = float(target_norm.detach().cpu())
    low = 0.0
    high = max(target_scalar, 1e-12)
    candidates = [candidate_at(low), candidate_at(high)]
    for _ in range(20):
        if candidates[-1][2] >= target_scalar:
            break
        high *= 2.0
        candidates.append(candidate_at(high))
    else:
        raise RuntimeError("could not bracket the realized control norm")
    for _ in range(40):
        midpoint = (low + high) / 2.0
        current = candidate_at(midpoint)
        candidates.append(current)
        if current[2] < target_scalar:
            low = midpoint
        else:
            high = midpoint
    replacement, realized_delta, _ = min(
        candidates, key=lambda value: abs(value[2] - target_scalar)
    )
    return replacement, realized_delta


def _record_realized_norms(
    measurements: MutableMapping[str, float],
    realized_delta: torch.Tensor,
    target_norm: torch.Tensor,
) -> None:
    realized_norm_value = (
        torch.linalg.vector_norm(realized_delta).detach().float().cpu().item()
    )
    target_norm_value = target_norm.detach().float().cpu().item()
    measurements["removed_fro_norm"] = float(realized_norm_value)
    measurements["target_removed_fro_norm"] = float(target_norm_value)
    measurements["norm_ratio"] = float(
        realized_norm_value / max(target_norm_value, 1e-12)
    )


def make_prompt_removal_transform(
    geometry: PromptRemovalGeometry,
    condition: str,
    measurements: MutableMapping[str, float],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a hook transform and record its realized Frobenius norm."""

    basis = torch.from_numpy(geometry.basis.astype(np.float32))
    control = torch.from_numpy(geometry.control_basis.astype(np.float32))
    if condition not in {"actual_rank3_remove", "actual_normmatched_orthogonal"}:
        raise ValueError(f"unsupported removal condition: {condition}")

    def transform(selected: torch.Tensor) -> torch.Tensor:
        target = projected_delta(selected, basis)
        target_replacement, realized_target = _realized_replacement(selected, target)
        target_norm = torch.linalg.vector_norm(realized_target)
        if condition == "actual_rank3_remove":
            replacement = target_replacement
            realized_delta = realized_target
        else:
            nuisance = projected_delta(selected, control)
            replacement, realized_delta = _closest_realized_norm_replacement(
                selected, nuisance, target_norm
            )
        _record_realized_norms(measurements, realized_delta, target_norm)
        return replacement

    return transform


def make_answer_query_removal_transform(
    geometry: PromptRemovalGeometry,
    condition: str,
    measurements: MutableMapping[str, float],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Remove the answer-query count coordinate relative to its global center."""

    basis = torch.from_numpy(geometry.basis.astype(np.float32))
    control = torch.from_numpy(geometry.control_basis.astype(np.float32))
    center = torch.from_numpy(geometry.centroids.mean(axis=0).astype(np.float32))
    if condition not in {"actual_rank3_remove", "actual_normmatched_orthogonal"}:
        raise ValueError(f"unsupported removal condition: {condition}")

    def transform(selected: torch.Tensor) -> torch.Tensor:
        target = single_position_projected_delta(selected, basis, center)
        target_replacement, realized_target = _realized_replacement(selected, target)
        target_norm = torch.linalg.vector_norm(realized_target)
        if condition == "actual_rank3_remove":
            replacement = target_replacement
            realized_delta = realized_target
        else:
            nuisance = single_position_projected_delta(selected, control, center)
            replacement, realized_delta = _closest_realized_norm_replacement(
                selected, nuisance, target_norm
            )
        _record_realized_norms(measurements, realized_delta, target_norm)
        return replacement

    return transform
