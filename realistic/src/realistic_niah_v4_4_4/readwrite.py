from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from realistic_niah_v4.modeling import (
    DecoderAdapter,
    _bounded_logits_kwargs,
    _encoding_tensors,
)
from realistic_niah_v4.prompts import PromptEncoding
from realistic_niah_v4_4_3.interventions import (
    CausalOutput,
    QueryBundle,
    _replace_output_tensor,
    _score_candidate_sequences,
    _tensor_from_output,
    capture_query_bundle,
)
from realistic_niah_v4_4_4.interventions import (
    natural_axis_diagnostics,
    set_output_from_stacked_z,
    stacked_set_z,
)
from realistic_niah_v4_4_4.relay import edge_z_from_values


PARTITION_NAMES = (
    "slot_tokens",
    "pre_query_non_slot_early",
    "pre_query_non_slot_tail",
    "answer_query_self",
)
READ_COMPONENT_NAMES = ("full", "value", "routing")


def _registered_heads(heads: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(head) for head in heads)
    if not result or len(set(result)) != len(result):
        raise ValueError("Read/write head sets must be nonempty and unique")
    return result


def stable_position_partition(
    encoding: PromptEncoding, *, tail_width: int = 64
) -> dict[str, tuple[int, ...]]:
    """Return a disjoint, exhaustive partition of the answer-query key axis.

    Slot membership is defined by the registered slot spans, not whether a slot
    is active in a particular count.  This keeps donor and receiver groups
    semantically aligned when the active needle count changes.
    """

    if tail_width <= 0:
        raise ValueError("Position-partition tail width must be positive")
    query = int(encoding.query_position)
    if query != int(encoding.sequence_length) - 1:
        raise ValueError("Read/write supplement requires the final-token answer query")
    slots = {
        position
        for span in encoding.slot_spans
        for position in range(int(span.start), int(span.end))
    }
    if any(position < 0 or position >= query for position in slots):
        raise ValueError("A registered slot overlaps or follows the answer query")
    non_slot = [position for position in range(query) if position not in slots]
    tail_start = max(0, query - int(tail_width))
    groups = {
        "slot_tokens": tuple(sorted(slots)),
        "pre_query_non_slot_early": tuple(
            position for position in non_slot if position < tail_start
        ),
        "pre_query_non_slot_tail": tuple(
            position for position in non_slot if position >= tail_start
        ),
        "answer_query_self": (query,),
    }
    flattened = [position for name in PARTITION_NAMES for position in groups[name]]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("Read/write position partition is not disjoint")
    expected = set(range(query + 1))
    if set(flattened) != expected:
        missing = sorted(expected - set(flattened))
        extra = sorted(set(flattened) - expected)
        raise RuntimeError(
            f"Read/write position partition is not exhaustive: missing={missing[:5]} "
            f"extra={extra[:5]}"
        )
    if any(not groups[name] for name in PARTITION_NAMES):
        raise RuntimeError("A read/write position-partition group is empty")
    return {"all_positions": tuple(range(query + 1)), **groups}


def four_way_edge_states(
    receiver: QueryBundle,
    donor: QueryBundle,
    adapter: DecoderAdapter,
    *,
    layer: int,
    heads: Sequence[int],
    positions: Sequence[int],
) -> dict[str, torch.Tensor]:
    """Capture z(alpha_R,V_R), z(alpha_R,V_D), z(alpha_D,V_R), z(alpha_D,V_D)."""

    registered = _registered_heads(heads)
    receiver_values = receiver.value_by_layer[int(layer)]
    donor_values = donor.value_by_layer[int(layer)]
    if receiver_values.shape != donor_values.shape:
        raise ValueError("Read decomposition requires aligned receiver/donor V axes")
    receiver_alpha = receiver.alpha_by_layer[int(layer)]
    donor_alpha = donor.alpha_by_layer[int(layer)]
    if receiver_alpha.shape != donor_alpha.shape:
        raise ValueError("Read decomposition requires aligned receiver/donor alpha axes")
    if (
        receiver.alpha_key_start_by_layer[int(layer)]
        != donor.alpha_key_start_by_layer[int(layer)]
    ):
        raise ValueError("Read decomposition requires an identical attention key start")
    return {
        "rr": edge_z_from_values(
            receiver,
            receiver,
            adapter,
            layer=int(layer),
            heads=registered,
            positions=positions,
        ),
        "rd": edge_z_from_values(
            receiver,
            donor,
            adapter,
            layer=int(layer),
            heads=registered,
            positions=positions,
        ),
        "dr": edge_z_from_values(
            donor,
            receiver,
            adapter,
            layer=int(layer),
            heads=registered,
            positions=positions,
        ),
        "dd": edge_z_from_values(
            donor,
            donor,
            adapter,
            layer=int(layer),
            heads=registered,
            positions=positions,
        ),
    }


def shapley_read_decomposition(
    receiver: QueryBundle,
    donor: QueryBundle,
    adapter: DecoderAdapter,
    *,
    layer: int,
    heads: Sequence[int],
    positions: Sequence[int],
    closure_relative_tolerance: float = 1e-5,
    anchor_to_captured_endpoints: bool = False,
) -> dict[str, torch.Tensor | float]:
    """Exactly decompose an edge-state change into value and routing terms.

    For the all-position primary analysis, ``anchor_to_captured_endpoints``
    replaces the eager alpha-V receiver/donor endpoints with the actual fused
    pre-O states.  The two Shapley terms then close exactly onto the causal
    donor-Z patch.  Eager endpoint reconstruction error remains explicit as a
    numerical sensitivity diagnostic.
    """

    states = four_way_edge_states(
        receiver,
        donor,
        adapter,
        layer=int(layer),
        heads=heads,
        positions=positions,
    )
    rr_eager, rd, dr, dd_eager = (
        states[name] for name in ("rr", "rd", "dr", "dd")
    )
    endpoint_diagnostics: dict[str, float] = {
        "endpoint_anchor_used": float(bool(anchor_to_captured_endpoints))
    }
    if anchor_to_captured_endpoints:
        key_start = int(receiver.alpha_key_start_by_layer[int(layer)])
        key_length = int(receiver.alpha_by_layer[int(layer)].shape[-1])
        expected_positions = tuple(range(key_start, key_start + key_length))
        registered_positions = tuple(sorted({int(value) for value in positions}))
        if registered_positions != expected_positions:
            raise ValueError(
                "Captured endpoint anchoring is only defined for all key positions"
            )
        rr = stacked_set_z(
            receiver, adapter, layer=int(layer), heads=heads
        ).detach().float().cpu()
        dd = stacked_set_z(
            donor, adapter, layer=int(layer), heads=heads
        ).detach().float().cpu()
        for label, eager, actual in (
            ("receiver", rr_eager, rr),
            ("donor", dd_eager, dd),
        ):
            error = float(torch.linalg.vector_norm(eager - actual))
            reference = float(torch.linalg.vector_norm(actual))
            endpoint_diagnostics[f"{label}_endpoint_reconstruction_l2"] = error
            endpoint_diagnostics[
                f"{label}_endpoint_reconstruction_relative_l2"
            ] = error / max(reference, 1e-12)
    else:
        rr, dd = rr_eager, dd_eager
    value_receiver = rd - rr
    value_donor = dd - dr
    routing_receiver = dr - rr
    routing_donor = dd - rd
    value = 0.5 * (value_receiver + value_donor)
    routing = 0.5 * (routing_receiver + routing_donor)
    full = dd - rr
    interaction = value_donor - value_receiver
    closure = full - value - routing
    closure_l2 = float(torch.linalg.vector_norm(closure))
    reference_l2 = float(torch.linalg.vector_norm(full))
    relative = closure_l2 / max(reference_l2, 1e-12)
    if not math.isfinite(relative) or relative > float(closure_relative_tolerance):
        raise RuntimeError(
            f"Read Shapley closure failed: relative_l2={relative:.6g}"
        )
    return {
        **states,
        "rr_eager": rr_eager,
        "dd_eager": dd_eager,
        "rr": rr,
        "dd": dd,
        "full": full,
        "value": value,
        "routing": routing,
        "value_receiver_anchored": value_receiver,
        "value_donor_anchored": value_donor,
        "routing_receiver_anchored": routing_receiver,
        "routing_donor_anchored": routing_donor,
        "alpha_value_interaction": interaction,
        "closure_l2": closure_l2,
        "closure_relative_l2": relative,
        **endpoint_diagnostics,
    }


def read_component_output_diagnostics(
    adapter: DecoderAdapter,
    *,
    layer: int,
    heads: Sequence[int],
    components: Mapping[str, torch.Tensor | float],
    global_z_count_steps: torch.Tensor,
    count_gap: int,
) -> dict[str, float]:
    if int(count_gap) == 0:
        raise ValueError("Read-component count gap cannot be zero")
    axis = natural_axis_diagnostics(
        adapter,
        layer=int(layer),
        heads=heads,
        z_count_steps=global_z_count_steps,
    )
    result = {
        "closure_l2": float(components["closure_l2"]),
        "closure_relative_l2": float(components["closure_relative_l2"]),
    }
    for key in (
        "endpoint_anchor_used",
        "receiver_endpoint_reconstruction_l2",
        "receiver_endpoint_reconstruction_relative_l2",
        "donor_endpoint_reconstruction_l2",
        "donor_endpoint_reconstruction_relative_l2",
    ):
        if key in components:
            result[key] = float(components[key])
    for name in READ_COMPONENT_NAMES:
        values = components[name]
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"Read component {name} is not a tensor")
        output = set_output_from_stacked_z(
            adapter,
            layer=int(layer),
            heads=heads,
            stacked_z=values,
        )
        norm = float(torch.linalg.vector_norm(output))
        projection = float(torch.dot(output, axis["output_unit"]))
        result[f"{name}_output_norm"] = norm
        result[f"{name}_global_axis_projection"] = projection
        result[f"{name}_global_axis_cosine"] = projection / max(norm, 1e-12)
        result[f"{name}_mechanical_transport"] = (
            projection / float(axis["output_step_norm"]) / int(count_gap)
        )
    interaction = components["alpha_value_interaction"]
    if not isinstance(interaction, torch.Tensor):
        raise TypeError("Read interaction is not a tensor")
    interaction_output = set_output_from_stacked_z(
        adapter,
        layer=int(layer),
        heads=heads,
        stacked_z=interaction,
    )
    result["alpha_value_interaction_output_norm"] = float(
        torch.linalg.vector_norm(interaction_output)
    )
    return result


def fit_count_intercept_and_step(
    states: torch.Tensor, counts: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """OLS intercept at count zero and one-count slope for arbitrary state tails."""

    values = states.detach().float().cpu()
    labels = torch.as_tensor(tuple(int(count) for count in counts), dtype=torch.float32)
    if values.ndim < 2 or values.shape[0] != len(labels):
        raise ValueError("Count-axis states and labels have incompatible shapes")
    centered = labels - labels.mean()
    denominator = float(torch.dot(centered, centered))
    if denominator <= 0:
        raise ValueError("Count-axis labels are constant")
    broadcast = centered.reshape((len(centered),) + (1,) * (values.ndim - 1))
    slope = torch.sum(broadcast * values, dim=0) / denominator
    intercept = values.mean(dim=0) - labels.mean() * slope
    if not torch.isfinite(slope).all() or not torch.isfinite(intercept).all():
        raise RuntimeError("Count-axis OLS produced non-finite values")
    return intercept, slope


def residual_axis_coefficient(
    state: torch.Tensor, *, intercept: torch.Tensor, step: torch.Tensor
) -> float:
    values = state.detach().float().cpu()
    center = intercept.detach().float().cpu()
    direction = step.detach().float().cpu()
    if values.shape != center.shape or values.shape != direction.shape:
        raise ValueError("Residual state/count-axis shapes disagree")
    denominator = float(torch.dot(direction.flatten(), direction.flatten()))
    if denominator <= 0:
        raise ValueError("Residual count step is zero")
    return float(
        torch.dot((values - center).flatten(), direction.flatten()) / denominator
    )


@dataclass(frozen=True)
class TracedCausalOutput:
    causal_output: CausalOutput
    residual_by_layer: dict[int, torch.Tensor]


@torch.inference_mode()
def capture_query_bundle_and_trace(
    model: nn.Module,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    mediator_layer: int,
    trace_layers: Sequence[int],
    cache_logit_tolerance: float = 0.5,
) -> tuple[QueryBundle, dict[int, torch.Tensor]]:
    """Capture the L28 Q/K/V bundle and natural post-block query trajectory."""

    selected_layers = tuple(sorted({int(value) for value in trace_layers}))
    if not selected_layers or selected_layers[0] < int(mediator_layer):
        raise ValueError("Natural trace layers must be downstream of the mediator")
    if any(not 0 <= value < adapter.num_layers for value in selected_layers):
        raise ValueError("A natural trace layer is outside the decoder")
    residuals: dict[int, torch.Tensor] = {}
    calls = {layer: 0 for layer in selected_layers}

    def make_hook(trace_layer: int):
        def hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            hidden = _tensor_from_output(output)
            if hidden.ndim != 3:
                raise RuntimeError("A natural decoder trace returned no hidden tensor")
            # capture_query_bundle also performs cached continuation and a
            # one-query attention rerun.  Only the original full prefill is a
            # state on the registered prompt coordinate system.
            if hidden.shape[1] != encoding.sequence_length:
                return
            residuals[trace_layer] = (
                hidden[0, encoding.query_position].detach().float().cpu()
            )
            calls[trace_layer] += 1

        return hook

    handles = [
        adapter.layers[layer].register_forward_hook(make_hook(layer))
        for layer in selected_layers
    ]
    try:
        bundle = capture_query_bundle(
            model,
            adapter,
            encoding,
            layers=(int(mediator_layer),),
            capture_attention=True,
            capture_values=True,
            cache_logit_tolerance=float(cache_logit_tolerance),
        )
    finally:
        for handle in handles:
            handle.remove()
    violations = {layer: count for layer, count in calls.items() if count != 1}
    if violations:
        raise RuntimeError(f"Natural query trace hook mismatch: {violations}")
    return bundle, residuals


@torch.inference_mode()
def run_with_set_z_deltas_and_trace(
    model: nn.Module,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    layer: int,
    deltas: Mapping[int, torch.Tensor],
    trace_layers: Sequence[int],
) -> TracedCausalOutput:
    """Apply a true pre-O set delta and capture downstream post-block states."""

    registered = tuple(sorted(int(head) for head in deltas))
    if not registered:
        raise ValueError("A traced set-Z intervention needs at least one head")
    if len(registered) != len(deltas):
        raise ValueError("Traced set-Z heads must be unique")
    selected_layers = tuple(sorted({int(value) for value in trace_layers}))
    if not selected_layers or selected_layers[0] < int(layer):
        raise ValueError("Trace layers must be nonempty and downstream of the intervention")
    if any(not 0 <= value < adapter.num_layers for value in selected_layers):
        raise ValueError("A requested trace layer is outside the decoder")
    width = int(adapter.head_dims[int(layer)])
    applied = 0
    residuals: dict[int, torch.Tensor] = {}
    attention_output: torch.Tensor | None = None

    def z_hook(_module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
        nonlocal applied
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("o_proj received no positional tensor")
        values = args[0]
        if values.ndim != 3 or values.shape[1] != encoding.sequence_length:
            raise RuntimeError("Traced set-Z intervention requires full prefill")
        patched = values.clone()
        for head in registered:
            addition = deltas[head].to(device=values.device, dtype=values.dtype)
            if addition.shape != (width,):
                raise RuntimeError("A traced set-Z delta has the wrong width")
            start = int(head) * width
            patched[:, encoding.query_position, start : start + width] += addition
        applied += 1
        return (patched, *args[1:])

    def make_trace_hook(trace_layer: int):
        def hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            hidden = _tensor_from_output(output)
            if hidden.ndim != 3:
                raise RuntimeError("A traced decoder block returned no hidden tensor")
            if hidden.shape[1] != encoding.sequence_length:
                return
            residuals[trace_layer] = (
                hidden[0, encoding.query_position].detach().float().cpu()
            )

        return hook

    def attention_output_hook(
        _module: nn.Module, _args: tuple[Any, ...], output: Any
    ) -> None:
        nonlocal attention_output
        hidden = _tensor_from_output(output)
        if hidden.ndim != 3 or hidden.shape[1] != encoding.sequence_length:
            raise RuntimeError("Traced attention output requires full prefill")
        attention_output = (
            hidden[0, encoding.query_position].detach().float().cpu()
        )

    handles = [
        adapter.output_projections[int(layer)].register_forward_pre_hook(z_hook),
        adapter.attentions[int(layer)].register_forward_hook(attention_output_hook),
    ]
    handles.extend(
        adapter.layers[trace_layer].register_forward_hook(make_trace_hook(trace_layer))
        for trace_layer in selected_layers
    )
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        prefill_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            **_bounded_logits_kwargs(model),
        )
    finally:
        for handle in handles:
            handle.remove()
    if applied != 1:
        raise RuntimeError(f"Traced set-Z delta applied {applied} times")
    if attention_output is None:
        raise RuntimeError("Traced set-Z run captured no post-O attention output")
    missing = sorted(set(selected_layers) - set(residuals))
    if missing:
        raise RuntimeError(f"Traced set-Z run missed decoder layers: {missing}")
    scored = _score_candidate_sequences(model, encoding, prefill_output)
    return TracedCausalOutput(
        causal_output=CausalOutput(
            logits=scored.logits,
            candidate_log_scores=scored.candidate_log_scores,
            attention_output=attention_output,
        ),
        residual_by_layer=residuals,
    )


def stacked_delta_mapping(
    heads: Sequence[int], stacked_delta: torch.Tensor
) -> dict[int, torch.Tensor]:
    registered = _registered_heads(heads)
    values = stacked_delta.detach().float().cpu()
    if values.ndim != 2 or values.shape[0] != len(registered):
        raise ValueError("A stacked read/write delta has the wrong shape")
    return {head: values[offset] for offset, head in enumerate(registered)}


def full_set_delta_from_bundles(
    receiver: QueryBundle,
    donor: QueryBundle,
    adapter: DecoderAdapter,
    *,
    layer: int,
    heads: Sequence[int],
) -> torch.Tensor:
    """Return the exact total selected-set donor-minus-receiver pre-O delta."""

    return stacked_set_z(
        donor, adapter, layer=int(layer), heads=heads
    ) - stacked_set_z(receiver, adapter, layer=int(layer), heads=heads)


def write_central_difference_diagnostics(
    plus: Mapping[int, torch.Tensor],
    minus: Mapping[int, torch.Tensor],
    *,
    beta: float,
    downstream_steps: Mapping[int, torch.Tensor],
) -> list[dict[str, float | int]]:
    """Project a symmetric causal write trace onto natural downstream axes."""

    if beta <= 0:
        raise ValueError("Write central-difference beta must be positive")
    if set(plus) != set(minus) or set(plus) != set(downstream_steps):
        raise ValueError("Write traces and downstream axes cover different layers")
    rows: list[dict[str, float | int]] = []
    for layer in sorted(plus):
        delta = (
            plus[layer].detach().float().cpu()
            - minus[layer].detach().float().cpu()
        ) / (2.0 * float(beta))
        step = downstream_steps[layer].detach().float().cpu()
        if delta.shape != step.shape:
            raise ValueError("Write trace/count-step hidden widths disagree")
        delta_norm = float(torch.linalg.vector_norm(delta))
        step_norm = float(torch.linalg.vector_norm(step))
        if step_norm <= 0:
            raise ValueError("A downstream natural count step is zero")
        projection = float(torch.dot(delta.flatten(), step.flatten()) / step_norm)
        rows.append(
            {
                "layer": int(layer),
                "central_delta_norm": delta_norm,
                "natural_count_axis_projection": projection,
                "natural_count_axis_coefficient": projection / step_norm,
                "natural_count_axis_cosine": projection / max(delta_norm, 1e-12),
            }
        )
    return rows
