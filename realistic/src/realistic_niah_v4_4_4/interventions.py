from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from realistic_niah_v4.modeling import DecoderAdapter
from realistic_niah_v4.prompts import PromptEncoding
from realistic_niah_v4_4_3.geometry import deterministic_orthogonal_direction
from realistic_niah_v4_4_3.interventions import (
    CausalOutput,
    QueryBundle,
    head_output_from_z,
    head_z,
)
from realistic_niah_v4_4_3.set_interventions import run_with_set_z_replacements


def stacked_set_z(
    bundle: QueryBundle,
    adapter: DecoderAdapter,
    *,
    layer: int,
    heads: Sequence[int],
) -> torch.Tensor:
    registered = tuple(int(head) for head in heads)
    if not registered or len(set(registered)) != len(registered):
        raise ValueError("Set heads must be unique and nonempty")
    return torch.stack(
        [
            head_z(bundle, adapter, layer=int(layer), head=head)
            .detach()
            .float()
            .cpu()
            for head in registered
        ],
        dim=0,
    )


def set_output_from_stacked_z(
    adapter: DecoderAdapter,
    *,
    layer: int,
    heads: Sequence[int],
    stacked_z: torch.Tensor,
) -> torch.Tensor:
    registered = tuple(int(head) for head in heads)
    width = int(adapter.head_dims[int(layer)])
    values = stacked_z.detach().float().cpu()
    if values.shape != (len(registered), width):
        raise ValueError(
            f"Expected stacked Z {(len(registered), width)}, got {tuple(values.shape)}"
        )
    return torch.stack(
        [
            head_output_from_z(
                adapter,
                layer=int(layer),
                head=head,
                z=values[offset],
            )
            for offset, head in enumerate(registered)
        ],
        dim=0,
    ).sum(dim=0)


def natural_axis_diagnostics(
    adapter: DecoderAdapter,
    *,
    layer: int,
    heads: Sequence[int],
    z_count_steps: torch.Tensor,
) -> dict[str, Any]:
    step_output = set_output_from_stacked_z(
        adapter,
        layer=int(layer),
        heads=heads,
        stacked_z=z_count_steps,
    )
    norm = torch.linalg.vector_norm(step_output)
    if not torch.isfinite(norm) or float(norm) <= 1e-8:
        raise RuntimeError("Natural OV output step is degenerate")
    return {
        "output_step": step_output,
        "output_unit": step_output / norm,
        "output_step_norm": float(norm),
    }


def natural_carrier_coefficient(
    adapter: DecoderAdapter,
    *,
    bundle: QueryBundle,
    layer: int,
    heads: Sequence[int],
    z_count_steps: torch.Tensor,
    z_center: torch.Tensor,
) -> dict[str, float]:
    actual_z = stacked_set_z(bundle, adapter, layer=int(layer), heads=heads)
    center = z_center.detach().float().cpu()
    if center.shape != actual_z.shape:
        raise ValueError("Natural carrier Z center has the wrong shape")
    centered_output = set_output_from_stacked_z(
        adapter,
        layer=int(layer),
        heads=heads,
        stacked_z=actual_z - center,
    )
    axis = natural_axis_diagnostics(
        adapter,
        layer=int(layer),
        heads=heads,
        z_count_steps=z_count_steps,
    )
    projection = float(torch.dot(centered_output, axis["output_unit"]))
    return {
        "natural_carrier_output_projection": projection,
        "natural_carrier_coefficient": projection / float(axis["output_step_norm"]),
        "natural_centered_set_output_norm": float(
            torch.linalg.vector_norm(centered_output)
        ),
        "natural_output_step_norm": float(axis["output_step_norm"]),
    }


def _orthogonal_equal_output_norm_delta(
    adapter: DecoderAdapter,
    *,
    layer: int,
    heads: Sequence[int],
    z_count_steps: torch.Tensor,
    target_output_norm: float,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    steps = z_count_steps.detach().float().cpu()
    axis = natural_axis_diagnostics(
        adapter,
        layer=int(layer),
        heads=heads,
        z_count_steps=steps,
    )
    output_unit = axis["output_unit"]
    step_output_norm = float(axis["output_step_norm"])
    probe = deterministic_orthogonal_direction(
        steps.reshape(-1), label=f"{label}:z-probe"
    ).reshape_as(steps)
    probe_output = set_output_from_stacked_z(
        adapter,
        layer=int(layer),
        heads=heads,
        stacked_z=probe,
    )
    parallel = float(torch.dot(probe_output, output_unit))
    control = probe - (parallel / step_output_norm) * steps
    control_output = set_output_from_stacked_z(
        adapter,
        layer=int(layer),
        heads=heads,
        stacked_z=control,
    )
    control_norm = torch.linalg.vector_norm(control_output)
    if abs(float(target_output_norm)) <= 1e-12:
        return torch.zeros_like(control), torch.zeros_like(control_output)
    if not torch.isfinite(control_norm) or float(control_norm) <= 1e-8:
        raise RuntimeError("Cannot construct stable in-span orthogonal control")
    control = control * (abs(float(target_output_norm)) / float(control_norm))
    control_output = set_output_from_stacked_z(
        adapter,
        layer=int(layer),
        heads=heads,
        stacked_z=control,
    )
    return control, control_output


def natural_ov_mediation_deltas(
    adapter: DecoderAdapter,
    *,
    receiver: QueryBundle,
    donor: QueryBundle,
    layer: int,
    heads: Sequence[int],
    z_count_steps: torch.Tensor,
    orthogonal_label: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Build donor-Z patch, natural-axis block, and matched control in Z-space.

    The block removes only the donor-minus-receiver output component parallel
    to the frozen natural OV axis.  It therefore preserves the receiver's
    baseline component and directly tests mediation of the donor patch effect.
    """

    receiver_z = stacked_set_z(
        receiver, adapter, layer=int(layer), heads=heads
    )
    donor_z = stacked_set_z(donor, adapter, layer=int(layer), heads=heads)
    patch_delta = donor_z - receiver_z
    patch_output = set_output_from_stacked_z(
        adapter,
        layer=int(layer),
        heads=heads,
        stacked_z=patch_delta,
    )
    axis = natural_axis_diagnostics(
        adapter,
        layer=int(layer),
        heads=heads,
        z_count_steps=z_count_steps,
    )
    coefficient = float(torch.dot(patch_output, axis["output_unit"]))
    block_delta = (-coefficient / float(axis["output_step_norm"])) * (
        z_count_steps.detach().float().cpu()
    )
    control_delta, control_output = _orthogonal_equal_output_norm_delta(
        adapter,
        layer=int(layer),
        heads=heads,
        z_count_steps=z_count_steps,
        target_output_norm=abs(coefficient),
        label=orthogonal_label,
    )
    block_output = set_output_from_stacked_z(
        adapter,
        layer=int(layer),
        heads=heads,
        stacked_z=block_delta,
    )
    residual_parallel = float(
        torch.dot(patch_output + block_output, axis["output_unit"])
    )
    control_norm = float(torch.linalg.vector_norm(control_output))
    diagnostics = {
        "patch_output_norm": float(torch.linalg.vector_norm(patch_output)),
        "patch_natural_axis_coefficient": coefficient,
        "axis_block_output_norm": float(torch.linalg.vector_norm(block_output)),
        "orthogonal_control_output_norm": control_norm,
        "blocked_patch_residual_axis_component": residual_parallel,
        "orthogonal_control_axis_cosine": (
            float(torch.dot(control_output, axis["output_unit"]))
            / max(control_norm, 1e-12)
        ),
    }
    return donor_z, block_delta, control_delta, diagnostics


def natural_ov_mediation_logits(
    model: Any,
    adapter: DecoderAdapter,
    receiver_encoding: PromptEncoding,
    *,
    receiver: QueryBundle,
    donor: QueryBundle,
    layer: int,
    heads: Sequence[int],
    z_count_steps: torch.Tensor,
    orthogonal_label: str,
) -> tuple[dict[str, CausalOutput], dict[str, float]]:
    donor_z, block_delta, control_delta, diagnostics = natural_ov_mediation_deltas(
        adapter,
        receiver=receiver,
        donor=donor,
        layer=int(layer),
        heads=heads,
        z_count_steps=z_count_steps,
        orthogonal_label=orthogonal_label,
    )
    registered = tuple(int(head) for head in heads)

    def replacements(values: torch.Tensor) -> Mapping[int, torch.Tensor]:
        return {
            head: values[offset]
            for offset, head in enumerate(registered)
        }

    return (
        {
            "donor_z_patch": run_with_set_z_replacements(
                model,
                adapter,
                receiver_encoding,
                layer=int(layer),
                replacements=replacements(donor_z),
            ),
            "donor_z_patch_natural_axis_block": run_with_set_z_replacements(
                model,
                adapter,
                receiver_encoding,
                layer=int(layer),
                replacements=replacements(donor_z + block_delta),
            ),
            "donor_z_patch_orthogonal_control": run_with_set_z_replacements(
                model,
                adapter,
                receiver_encoding,
                layer=int(layer),
                replacements=replacements(donor_z + control_delta),
            ),
        },
        diagnostics,
    )


def max_abs_tensor_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError("Tensor delta operands have different shapes")
    values = torch.abs(left.detach().float().cpu() - right.detach().float().cpu())
    return float(values.max()) if values.numel() else 0.0


def finite_diagnostics(payload: Mapping[str, float]) -> None:
    invalid = {
        key: value
        for key, value in payload.items()
        if not math.isfinite(float(value))
    }
    if invalid:
        raise RuntimeError(f"Non-finite natural-OV diagnostics: {invalid}")
