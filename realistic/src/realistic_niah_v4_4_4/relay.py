from __future__ import annotations

import hashlib
import math
from typing import Mapping, Sequence

import torch

from realistic_niah_v4.modeling import DecoderAdapter
from realistic_niah_v4.prompts import PromptEncoding
from realistic_niah_v4_4_3.geometry import query_to_kv_head
from realistic_niah_v4_4_3.interventions import QueryBundle

from .interventions import (
    _orthogonal_equal_output_norm_delta,
    natural_axis_diagnostics,
    set_output_from_stacked_z,
    stacked_set_z,
)


def _registered_heads(heads: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(head) for head in heads)
    if not result or len(set(result)) != len(result):
        raise ValueError("Relay head sets must be nonempty and unique")
    return result


def _value_matrix(bundle: QueryBundle, *, layer: int) -> torch.Tensor:
    if int(layer) not in bundle.value_by_layer:
        raise ValueError("Relay bundle has no captured V states")
    values = bundle.value_by_layer[int(layer)].detach().float().cpu()
    if values.ndim != 2:
        raise ValueError("Relay V states must have shape [position,kv_width]")
    return values


def _attention_row(
    bundle: QueryBundle, *, layer: int, head: int
) -> tuple[torch.Tensor, int]:
    if int(layer) not in bundle.alpha_by_layer:
        raise ValueError("Relay bundle has no captured attention row")
    rows = bundle.alpha_by_layer[int(layer)].detach().float().cpu()
    if rows.ndim != 2 or not 0 <= int(head) < rows.shape[0]:
        raise ValueError("Relay attention row has an incompatible shape")
    start = int(bundle.alpha_key_start_by_layer[int(layer)])
    return rows[int(head)], start


def _kv_slice(
    values: torch.Tensor,
    adapter: DecoderAdapter,
    *,
    layer: int,
    head: int,
) -> torch.Tensor:
    width = int(adapter.head_dims[int(layer)])
    if values.shape[-1] % width:
        raise ValueError("Relay V width is not divisible by the head width")
    kv_heads = int(values.shape[-1]) // width
    kv_head = query_to_kv_head(
        query_head=int(head),
        query_heads=int(adapter.num_heads[int(layer)]),
        kv_heads=kv_heads,
    )
    start = kv_head * width
    return values[:, start : start + width]


def _row_indices(
    positions: Sequence[int], *, key_start: int, key_length: int
) -> torch.Tensor:
    absolute = tuple(sorted({int(position) for position in positions}))
    if not absolute:
        raise ValueError("A relay position set cannot be empty")
    relative = torch.as_tensor(
        [position - int(key_start) for position in absolute], dtype=torch.long
    )
    if int(relative.min()) < 0 or int(relative.max()) >= int(key_length):
        raise ValueError("Relay positions lie outside the captured attention keys")
    return relative


def edge_z_from_values(
    alpha_bundle: QueryBundle,
    value_bundle: QueryBundle,
    adapter: DecoderAdapter,
    *,
    layer: int,
    heads: Sequence[int],
    positions: Sequence[int],
) -> torch.Tensor:
    """Read a position set with ``alpha_bundle`` routing and ``value_bundle`` V.

    The result is the exact selected-edge contribution at the true pre-O Z
    boundary for ordinary attention.  In particular, using receiver alpha and
    donor V changes content without changing Q, K, or the attention weights.
    """

    registered = _registered_heads(heads)
    values = _value_matrix(value_bundle, layer=int(layer))
    outputs: list[torch.Tensor] = []
    for head in registered:
        row, key_start = _attention_row(
            alpha_bundle, layer=int(layer), head=int(head)
        )
        if key_start < 0 or key_start + len(row) > len(values):
            raise ValueError("Relay attention/V key axes do not align")
        indices = _row_indices(
            positions, key_start=key_start, key_length=len(row)
        )
        selected_row = row[indices]
        source_values = _kv_slice(
            values[key_start : key_start + len(row)],
            adapter,
            layer=int(layer),
            head=int(head),
        )[indices]
        outputs.append(torch.einsum("k,kd->d", selected_row, source_values))
    return torch.stack(outputs, dim=0)


def all_source_edge_z(
    bundle: QueryBundle,
    adapter: DecoderAdapter,
    *,
    layer: int,
    heads: Sequence[int],
) -> torch.Tensor:
    row, key_start = _attention_row(bundle, layer=int(layer), head=int(heads[0]))
    return edge_z_from_values(
        bundle,
        bundle,
        adapter,
        layer=int(layer),
        heads=heads,
        positions=range(key_start, key_start + len(row)),
    )


def contribution_reconstruction_diagnostics(
    bundle: QueryBundle,
    adapter: DecoderAdapter,
    *,
    layer: int,
    heads: Sequence[int],
) -> dict[str, float]:
    reconstructed = all_source_edge_z(
        bundle, adapter, layer=int(layer), heads=heads
    )
    captured = stacked_set_z(bundle, adapter, layer=int(layer), heads=heads)
    delta = torch.linalg.vector_norm(reconstructed - captured)
    reference = torch.linalg.vector_norm(captured)
    return {
        "edge_z_reconstruction_l2": float(delta),
        "edge_z_reconstruction_relative_l2": float(
            delta / max(float(reference), 1e-12)
        ),
        "captured_set_z_l2": float(reference),
    }


def source_contribution_vector(
    bundle: QueryBundle,
    adapter: DecoderAdapter,
    *,
    layer: int,
    heads: Sequence[int],
    output_axis: torch.Tensor,
) -> torch.Tensor:
    """Return each absolute key position's additive output-axis contribution."""

    registered = _registered_heads(heads)
    axis = output_axis.detach().float().cpu()
    axis_norm = torch.linalg.vector_norm(axis)
    if axis.ndim != 1 or not torch.isfinite(axis_norm) or float(axis_norm) <= 0:
        raise ValueError("Relay output axis must be a finite nonzero vector")
    axis = axis / axis_norm
    values = _value_matrix(bundle, layer=int(layer))
    contributions = torch.zeros(len(values), dtype=torch.float32)
    output_weight = adapter.output_projections[int(layer)].weight.detach().float()
    width = int(adapter.head_dims[int(layer)])
    for head in registered:
        row, key_start = _attention_row(bundle, layer=int(layer), head=int(head))
        source_values = _kv_slice(
            values[key_start : key_start + len(row)],
            adapter,
            layer=int(layer),
            head=int(head),
        )
        start = int(head) * width
        readout = output_weight[:, start : start + width].T @ axis.to(
            device=output_weight.device, dtype=torch.float32
        )
        value_scores = source_values.to(
            device=readout.device, dtype=torch.float32
        ) @ readout
        contributions[key_start : key_start + len(row)] += (
            row.to(value_scores.device) * value_scores
        ).detach().cpu()
    return contributions


def _slot_positions(encoding: PromptEncoding) -> set[int]:
    return {
        position
        for span in encoding.slot_spans
        for position in range(int(span.start), int(span.end))
    }


def resolve_position_set(
    encoding: PromptEncoding,
    name: str,
    *,
    contribution: torch.Tensor | None = None,
) -> tuple[int, ...]:
    """Resolve a preregistered semantic or contribution-ranked position set."""

    query = int(encoding.query_position)
    slots = _slot_positions(encoding)
    if name == "answer_query_self":
        positions = (query,)
    elif name.startswith("pre_query_non_slot_tail_"):
        width = int(name.rsplit("_", 1)[1])
        positions = tuple(
            position
            for position in range(max(0, query - width), query)
            if position not in slots
        )
    elif name == "active_needle_endpoints":
        positions = tuple(sorted({int(span.end) - 1 for span in encoding.needle_spans}))
    elif name == "active_needle_spans":
        positions = tuple(
            sorted(
                {
                    position
                    for span in encoding.needle_spans
                    for position in range(int(span.start), int(span.end))
                }
            )
        )
    elif name.startswith("non_slot_top_"):
        if contribution is None:
            raise ValueError("Ranked relay sets require a contribution vector")
        values = contribution.detach().float().cpu()
        if values.ndim != 1 or len(values) != encoding.sequence_length:
            raise ValueError("Relay contribution vector has the wrong length")
        size = int(name.rsplit("_", 1)[1])
        eligible = [
            position
            for position in range(query)
            if position not in slots
        ]
        if len(eligible) < size:
            raise ValueError("Too few non-slot positions for a ranked relay set")
        ranked = sorted(
            eligible,
            key=lambda position: (-abs(float(values[position])), position),
        )
        positions = tuple(sorted(ranked[:size]))
    else:
        raise ValueError(f"Unknown relay position set: {name}")
    if not positions:
        raise ValueError(f"Relay position set {name} resolved to no positions")
    if min(positions) < 0 or max(positions) >= encoding.sequence_length:
        raise ValueError("Resolved relay position lies outside the prompt")
    return positions


def positions_sha256(positions: Sequence[int]) -> str:
    encoded = ",".join(str(int(value)) for value in positions).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def relay_carrier_coefficient(
    adapter: DecoderAdapter,
    *,
    layer: int,
    heads: Sequence[int],
    edge_z: torch.Tensor,
    edge_z_center: torch.Tensor,
    edge_z_count_step: torch.Tensor,
) -> dict[str, float]:
    values = edge_z.detach().float().cpu()
    center = edge_z_center.detach().float().cpu()
    steps = edge_z_count_step.detach().float().cpu()
    if values.shape != center.shape or values.shape != steps.shape:
        raise ValueError("Relay carrier tensors have incompatible shapes")
    axis = natural_axis_diagnostics(
        adapter,
        layer=int(layer),
        heads=heads,
        z_count_steps=steps,
    )
    centered_output = set_output_from_stacked_z(
        adapter,
        layer=int(layer),
        heads=heads,
        stacked_z=values - center,
    )
    projection = float(torch.dot(centered_output, axis["output_unit"]))
    return {
        "relay_carrier_output_projection": projection,
        "relay_carrier_coefficient": projection
        / float(axis["output_step_norm"]),
        "relay_centered_output_norm": float(
            torch.linalg.vector_norm(centered_output)
        ),
        "relay_output_step_norm": float(axis["output_step_norm"]),
    }


def receiver_alpha_donor_v_delta(
    receiver: QueryBundle,
    donor: QueryBundle,
    adapter: DecoderAdapter,
    *,
    layer: int,
    heads: Sequence[int],
    positions: Sequence[int],
) -> torch.Tensor:
    receiver_values = _value_matrix(receiver, layer=int(layer))
    donor_values = _value_matrix(donor, layer=int(layer))
    if receiver_values.shape != donor_values.shape:
        raise ValueError("Receiver/donor V states require exact position alignment")
    receiver_edge = edge_z_from_values(
        receiver,
        receiver,
        adapter,
        layer=int(layer),
        heads=heads,
        positions=positions,
    )
    donor_edge = edge_z_from_values(
        receiver,
        donor,
        adapter,
        layer=int(layer),
        heads=heads,
        positions=positions,
    )
    return donor_edge - receiver_edge


def stacked_delta_mapping(
    heads: Sequence[int], stacked_delta: torch.Tensor
) -> dict[int, torch.Tensor]:
    registered = _registered_heads(heads)
    values = stacked_delta.detach().float().cpu()
    if values.ndim != 2 or values.shape[0] != len(registered):
        raise ValueError("Stacked relay delta has the wrong shape")
    return {head: values[offset] for offset, head in enumerate(registered)}


def global_axis_patch_diagnostics(
    adapter: DecoderAdapter,
    *,
    layer: int,
    heads: Sequence[int],
    patch_delta_z: torch.Tensor,
    global_z_count_steps: torch.Tensor,
) -> dict[str, float]:
    patch_output = set_output_from_stacked_z(
        adapter,
        layer=int(layer),
        heads=heads,
        stacked_z=patch_delta_z,
    )
    axis = natural_axis_diagnostics(
        adapter,
        layer=int(layer),
        heads=heads,
        z_count_steps=global_z_count_steps,
    )
    coefficient = float(torch.dot(patch_output, axis["output_unit"]))
    return {
        "relay_patch_output_norm": float(torch.linalg.vector_norm(patch_output)),
        "relay_patch_global_axis_projection": coefficient,
        "relay_patch_global_axis_coefficient": coefficient
        / float(axis["output_step_norm"]),
        "relay_patch_global_axis_cosine": coefficient
        / max(float(torch.linalg.vector_norm(patch_output)), 1e-12),
    }


def natural_axis_block_for_patch_delta(
    adapter: DecoderAdapter,
    *,
    layer: int,
    heads: Sequence[int],
    patch_delta_z: torch.Tensor,
    global_z_count_steps: torch.Tensor,
    orthogonal_label: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    patch = patch_delta_z.detach().float().cpu()
    steps = global_z_count_steps.detach().float().cpu()
    if patch.shape != steps.shape:
        raise ValueError("Relay patch/global step shapes disagree")
    axis = natural_axis_diagnostics(
        adapter, layer=int(layer), heads=heads, z_count_steps=steps
    )
    patch_output = set_output_from_stacked_z(
        adapter, layer=int(layer), heads=heads, stacked_z=patch
    )
    coefficient = float(torch.dot(patch_output, axis["output_unit"]))
    block = (-coefficient / float(axis["output_step_norm"])) * steps
    control, control_output = _orthogonal_equal_output_norm_delta(
        adapter,
        layer=int(layer),
        heads=heads,
        z_count_steps=steps,
        target_output_norm=abs(coefficient),
        label=orthogonal_label,
    )
    block_output = set_output_from_stacked_z(
        adapter, layer=int(layer), heads=heads, stacked_z=block
    )
    residual = float(
        torch.dot(patch_output + block_output, axis["output_unit"])
    )
    control_norm = float(torch.linalg.vector_norm(control_output))
    diagnostics = {
        **global_axis_patch_diagnostics(
            adapter,
            layer=int(layer),
            heads=heads,
            patch_delta_z=patch,
            global_z_count_steps=steps,
        ),
        "relay_axis_block_output_norm": float(
            torch.linalg.vector_norm(block_output)
        ),
        "relay_axis_control_output_norm": control_norm,
        "relay_axis_block_residual_projection": residual,
        "relay_axis_control_cosine": float(
            torch.dot(control_output, axis["output_unit"])
        )
        / max(control_norm, 1e-12),
    }
    return block, control, diagnostics


def relay_removal_deltas(
    adapter: DecoderAdapter,
    *,
    layer: int,
    heads: Sequence[int],
    edge_z: torch.Tensor,
    edge_z_center: torch.Tensor,
    edge_z_count_step: torch.Tensor,
    orthogonal_label: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    values = edge_z.detach().float().cpu()
    center = edge_z_center.detach().float().cpu()
    steps = edge_z_count_step.detach().float().cpu()
    if values.shape != center.shape or values.shape != steps.shape:
        raise ValueError("Relay removal tensors have incompatible shapes")
    axis = natural_axis_diagnostics(
        adapter, layer=int(layer), heads=heads, z_count_steps=steps
    )
    centered_output = set_output_from_stacked_z(
        adapter,
        layer=int(layer),
        heads=heads,
        stacked_z=values - center,
    )
    coefficient = float(torch.dot(centered_output, axis["output_unit"]))
    removal = (-coefficient / float(axis["output_step_norm"])) * steps
    control, control_output = _orthogonal_equal_output_norm_delta(
        adapter,
        layer=int(layer),
        heads=heads,
        z_count_steps=steps,
        target_output_norm=abs(coefficient),
        label=orthogonal_label,
    )
    removal_output = set_output_from_stacked_z(
        adapter, layer=int(layer), heads=heads, stacked_z=removal
    )
    control_norm = float(torch.linalg.vector_norm(control_output))
    diagnostics = {
        "relay_natural_component": coefficient,
        "relay_removal_output_norm": float(
            torch.linalg.vector_norm(removal_output)
        ),
        "relay_removal_control_output_norm": control_norm,
        "relay_removal_residual_projection": float(
            torch.dot(centered_output + removal_output, axis["output_unit"])
        ),
        "relay_removal_control_axis_cosine": float(
            torch.dot(control_output, axis["output_unit"])
        )
        / max(control_norm, 1e-12),
    }
    if not all(math.isfinite(float(value)) for value in diagnostics.values()):
        raise RuntimeError("Relay removal produced non-finite diagnostics")
    return removal, control, diagnostics
