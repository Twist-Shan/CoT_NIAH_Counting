"""Dynamic residual-subspace interventions for the V4.4 counter channel."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import torch

from .modeling import (
    DecoderAdapter,
    generate_with_residual_transforms,
)
from .prompts import PromptEncoding


def _basis_on(basis: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    value = basis.to(device=reference.device, dtype=torch.float32)
    if value.ndim != 2 or value.shape[0] != reference.shape[-1]:
        raise ValueError("basis must have shape [hidden, rank]")
    gram = value.T @ value
    identity = torch.eye(value.shape[1], device=value.device, dtype=value.dtype)
    if not torch.allclose(gram, identity, atol=2e-3, rtol=2e-3):
        raise ValueError("counter basis is not orthonormal")
    return value


def projected_donor_transform(
    donor_state: torch.Tensor,
    basis: torch.Tensor,
):
    donor = donor_state.detach().float()

    def transform(selected: torch.Tensor) -> torch.Tensor:
        active_basis = _basis_on(basis, selected)
        target = donor.to(device=selected.device, dtype=torch.float32)
        if target.ndim == 1:
            target = target.unsqueeze(0)
        if target.shape != selected.shape[1:]:
            raise ValueError(
                f"donor state shape {tuple(target.shape)} does not match "
                f"selected {tuple(selected.shape[1:])}"
            )
        delta = target.unsqueeze(0) - selected.float()
        projected = (delta @ active_basis) @ active_basis.T
        return (selected.float() + projected).to(selected.dtype)

    return transform


def fixed_delta_transform(delta: torch.Tensor):
    value = delta.detach().float()

    def transform(selected: torch.Tensor) -> torch.Tensor:
        active = value.to(device=selected.device, dtype=torch.float32)
        if active.ndim == 1:
            active = active.unsqueeze(0)
        if active.shape != selected.shape[1:]:
            raise ValueError("fixed delta and selected residual shapes disagree")
        return (selected.float() + active.unsqueeze(0)).to(selected.dtype)

    return transform


def removal_transform(
    center: torch.Tensor,
    basis: torch.Tensor,
    *,
    dose: float = 1.0,
):
    if not 0.0 <= float(dose) <= 1.0:
        raise ValueError("removal dose must lie in [0,1]")
    center_value = center.detach().float()

    def transform(selected: torch.Tensor) -> torch.Tensor:
        active_basis = _basis_on(basis, selected)
        active_center = center_value.to(device=selected.device, dtype=torch.float32)
        if active_center.ndim == 1:
            active_center = active_center.unsqueeze(0)
        if active_center.shape != selected.shape[1:]:
            raise ValueError("removal center and selected residual shapes disagree")
        centered = selected.float() - active_center.unsqueeze(0)
        component = (centered @ active_basis) @ active_basis.T
        return (selected.float() - float(dose) * component).to(selected.dtype)

    return transform


def norm_matched_orthogonal_delta(
    delta: torch.Tensor,
    basis: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    value = delta.detach().float()
    active_basis = basis.detach().float()
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if active_basis.ndim != 2 or active_basis.shape[0] != value.shape[-1]:
        raise ValueError("basis and delta widths disagree")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    random = torch.randn(value.shape, generator=generator, dtype=torch.float32)
    random = random - (random @ active_basis) @ active_basis.T
    random = random - (
        torch.sum(random * value, dim=-1, keepdim=True)
        / torch.clamp(torch.sum(value * value, dim=-1, keepdim=True), min=1e-12)
    ) * value
    random_norm = torch.linalg.vector_norm(random, dim=-1, keepdim=True)
    target_norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    if torch.any(random_norm <= 1e-10):
        raise RuntimeError("failed to sample a nonzero orthogonal control delta")
    return random * (target_norm / random_norm)


def stable_intervention_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def run_counter_subspace_conditions(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    source_layer: int,
    source_positions: Sequence[int],
    receiver_source_state: torch.Tensor,
    donor_source_state: torch.Tensor,
    source_basis: torch.Tensor,
    mediator_layer: int | None = None,
    mediator_positions: Sequence[int] | None = None,
    mediator_center: torch.Tensor | None = None,
    mediator_basis: torch.Tensor | None = None,
    removal_dose: float = 1.0,
    random_seed: int = 442,
    max_new_tokens: int = 16,
) -> dict[str, dict[str, Any]]:
    """Run sufficiency, matched control, and optional serial mediation.

    The projected patch and random control have exactly matched post-block
    delta norms.  If a mediator is supplied, the later removal is applied to
    the *causally changed* activation, not to a cached clean activation.
    """

    receiver = receiver_source_state.detach().float()
    donor = donor_source_state.detach().float()
    basis = source_basis.detach().float()
    if receiver.ndim == 1:
        receiver = receiver.unsqueeze(0)
    if donor.ndim == 1:
        donor = donor.unsqueeze(0)
    if receiver.shape != donor.shape:
        raise ValueError("receiver and donor source-state shapes disagree")
    if len(tuple(source_positions)) != receiver.shape[0]:
        raise ValueError("source position/state counts disagree")
    projected_delta = ((donor - receiver) @ basis) @ basis.T
    random_delta = norm_matched_orthogonal_delta(
        projected_delta, basis, seed=random_seed
    )
    conditions: dict[str, dict[int, tuple[Sequence[int], Any]]] = {
        "projected_patch": {
            int(source_layer): (
                tuple(source_positions),
                projected_donor_transform(donor, basis),
            )
        },
        "orthogonal_norm_matched": {
            int(source_layer): (
                tuple(source_positions),
                fixed_delta_transform(random_delta),
            )
        },
    }
    mediator_supplied = all(
        value is not None
        for value in (
            mediator_layer,
            mediator_positions,
            mediator_center,
            mediator_basis,
        )
    )
    if any(
        value is not None
        for value in (
            mediator_layer,
            mediator_positions,
            mediator_center,
            mediator_basis,
        )
    ) and not mediator_supplied:
        raise ValueError("mediator layer, positions, center and basis are all required")
    if mediator_supplied:
        if int(mediator_layer) <= int(source_layer):
            raise ValueError("serial mediator layer must follow source layer")
        conditions["projected_patch_plus_removal"] = {
            int(source_layer): (
                tuple(source_positions),
                projected_donor_transform(donor, basis),
            ),
            int(mediator_layer): (
                tuple(int(value) for value in mediator_positions),
                removal_transform(
                    mediator_center,
                    mediator_basis,
                    dose=removal_dose,
                ),
            ),
        }
        conditions["removal_only"] = {
            int(mediator_layer): (
                tuple(int(value) for value in mediator_positions),
                removal_transform(
                    mediator_center,
                    mediator_basis,
                    dose=removal_dose,
                ),
            )
        }

    output: dict[str, dict[str, Any]] = {}
    for condition, transforms in conditions.items():
        output[condition] = generate_with_residual_transforms(
            model,
            tokenizer,
            adapter,
            encoding,
            transforms,
            max_new_tokens=max_new_tokens,
        )
    output["_audit"] = {
        "source_layer": int(source_layer),
        "source_positions": [int(value) for value in source_positions],
        "source_rank": int(basis.shape[1]),
        "projected_delta_norm": float(torch.linalg.vector_norm(projected_delta)),
        "random_delta_norm": float(torch.linalg.vector_norm(random_delta)),
        "mediator_layer": None if mediator_layer is None else int(mediator_layer),
        "removal_dose": float(removal_dose),
    }
    return output
