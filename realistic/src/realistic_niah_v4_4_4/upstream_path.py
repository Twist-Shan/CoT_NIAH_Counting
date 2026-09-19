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
    _score_candidate_sequences,
    capture_query_bundle,
)
from realistic_niah_v4_4_4.interventions import (
    _orthogonal_equal_output_norm_delta,
    set_output_from_stacked_z,
)
from realistic_niah_v4_4_4.readwrite import four_way_edge_states


HeadKey = tuple[int, int]


@dataclass(frozen=True)
class MultiSiteBundle:
    """Natural answer-query bundle plus selected head states at slot queries.

    Full hidden states and raw attention are deliberately transient.  Callers
    persist only scalar summaries derived from this object.
    """

    query: QueryBundle
    slot_positions: tuple[int, ...]
    slot_z_by_head: dict[HeadKey, torch.Tensor]


@dataclass(frozen=True)
class PathCausalOutput:
    causal_output: CausalOutput
    late_z_before: torch.Tensor
    late_z_after: torch.Tensor


def slot_positions(encoding: PromptEncoding) -> tuple[int, ...]:
    spans = sorted(encoding.slot_spans, key=lambda span: int(span.slot_index))
    positions = tuple(
        position
        for span in spans
        for position in range(int(span.start), int(span.end))
    )
    if not positions or len(positions) != len(set(positions)):
        raise ValueError("Registered slot positions must be nonempty and disjoint")
    if min(positions) < 0 or max(positions) >= int(encoding.query_position):
        raise ValueError("A slot position overlaps or follows the answer query")
    return positions


def assert_aligned_pair(receiver: PromptEncoding, donor: PromptEncoding) -> None:
    if receiver.sequence_length != donor.sequence_length:
        raise ValueError("Receiver/donor prompts do not share a token coordinate system")
    if receiver.query_position != donor.query_position:
        raise ValueError("Receiver/donor answer-query positions disagree")
    receiver_spans = tuple(
        (int(span.slot_index), int(span.start), int(span.end))
        for span in sorted(receiver.slot_spans, key=lambda value: int(value.slot_index))
    )
    donor_spans = tuple(
        (int(span.slot_index), int(span.start), int(span.end))
        for span in sorted(donor.slot_spans, key=lambda value: int(value.slot_index))
    )
    if receiver_spans != donor_spans:
        raise ValueError("Receiver/donor slot token coordinates are not aligned")


def _head_slice(adapter: DecoderAdapter, *, layer: int, head: int) -> slice:
    width = int(adapter.head_dims[int(layer)])
    return slice(int(head) * width, (int(head) + 1) * width)


def query_head_z(
    bundle: QueryBundle,
    adapter: DecoderAdapter,
    *,
    layer: int,
    head: int,
) -> torch.Tensor:
    return bundle.z_by_layer[int(layer)][
        _head_slice(adapter, layer=int(layer), head=int(head))
    ].detach().float().cpu()


@torch.inference_mode()
def capture_multisite_bundle(
    model: nn.Module,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    early_heads: Sequence[HeadKey],
    mediator_layer: int,
    cache_logit_tolerance: float = math.inf,
) -> MultiSiteBundle:
    """Capture natural early slot-Z, answer-query alpha/V/Z, and L28 Z."""

    registered = tuple((int(layer), int(head)) for layer, head in early_heads)
    if not registered or len(registered) != len(set(registered)):
        raise ValueError("Early head registry must be nonempty and unique")
    positions = slot_positions(encoding)
    heads_by_layer: dict[int, list[int]] = {}
    for layer, head in registered:
        heads_by_layer.setdefault(layer, []).append(head)
    captured: dict[HeadKey, torch.Tensor] = {}
    calls = {layer: 0 for layer in heads_by_layer}
    handles = []
    index = torch.as_tensor(positions, dtype=torch.long)
    for layer, heads in heads_by_layer.items():
        expected = int(adapter.num_heads[layer]) * int(adapter.head_dims[layer])

        def hook(
            _module: nn.Module,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
            heads: tuple[int, ...] = tuple(heads),
            expected: int = expected,
        ) -> None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("o_proj received no positional tensor")
            value = args[0]
            # capture_query_bundle also performs shorter cached forwards.
            if value.ndim != 3 or value.shape[1] != encoding.sequence_length:
                return
            if value.shape[0] != 1 or value.shape[-1] != expected:
                raise RuntimeError("Unexpected pre-O tensor during slot capture")
            local_index = index.to(device=value.device)
            for head in heads:
                captured[(layer, head)] = value[
                    0, local_index, _head_slice(adapter, layer=layer, head=head)
                ].detach().to(device="cpu", dtype=torch.float16)
            calls[layer] += 1

        handles.append(adapter.output_projections[layer].register_forward_pre_hook(hook))
    try:
        query = capture_query_bundle(
            model,
            adapter,
            encoding,
            layers=tuple(sorted({layer for layer, _head in registered} | {int(mediator_layer)})),
            capture_attention=True,
            capture_values=True,
            cache_logit_tolerance=float(cache_logit_tolerance),
        )
    finally:
        for handle in handles:
            handle.remove()
    violations = {layer: count for layer, count in calls.items() if count != 1}
    missing = sorted(set(registered) - set(captured))
    if violations or missing:
        raise RuntimeError(f"Slot capture mismatch: calls={violations}, missing={missing}")
    return MultiSiteBundle(query=query, slot_positions=positions, slot_z_by_head=captured)


def broad_retrieval_score(
    bundle: QueryBundle,
    encoding: PromptEncoding,
    *,
    layer: int,
    head: int,
) -> dict[str, float]:
    """V4.4.2 broad score: mass times effective occurrence coverage / N."""

    row = bundle.alpha_by_layer[int(layer)][int(head)].detach().float().cpu()
    key_start = int(bundle.alpha_key_start_by_layer[int(layer)])
    masses: list[float] = []
    for span in encoding.needle_spans:
        start = int(span.start) - key_start
        end = int(span.end) - key_start
        if start < 0 or end > len(row) or start >= end:
            raise ValueError("A needle span lies outside the captured attention row")
        masses.append(float(row[start:end].sum()))
    if len(masses) != int(encoding.count):
        raise ValueError("Active needle span count disagrees with gold count")
    total = float(sum(masses))
    if total <= 0:
        coverage = 0.0
    else:
        probabilities = torch.as_tensor(masses, dtype=torch.float64) / total
        probabilities = probabilities[probabilities > 0]
        entropy = float(-(probabilities * torch.log(probabilities)).sum())
        coverage = math.exp(entropy) / len(masses)
    return {
        "needle_attention_mass": total,
        "occurrence_coverage": coverage,
        "broad_retrieval_score": total * coverage,
    }


def group_heads_by_layer(heads: Sequence[HeadKey]) -> dict[int, tuple[int, ...]]:
    grouped: dict[int, list[int]] = {}
    for layer, head in heads:
        grouped.setdefault(int(layer), []).append(int(head))
    return {layer: tuple(values) for layer, values in sorted(grouped.items())}


def slot_edge_qk_deltas(
    receiver: MultiSiteBundle,
    donor: MultiSiteBundle,
    receiver_encoding: PromptEncoding,
    donor_encoding: PromptEncoding,
    adapter: DecoderAdapter,
    *,
    heads: Sequence[HeadKey],
) -> dict[HeadKey, torch.Tensor]:
    """Donor-routing / receiver-V delta on registered slot edges only.

    This changes alpha-selected slot contributions at the true pre-O boundary
    while holding V fixed to the receiver.  It is an edge-specific QK routing
    patch, not a full head-output patch.
    """

    assert_aligned_pair(receiver_encoding, donor_encoding)
    positions = slot_positions(receiver_encoding)
    result: dict[HeadKey, torch.Tensor] = {}
    for layer, layer_heads in group_heads_by_layer(heads).items():
        states = four_way_edge_states(
            receiver.query,
            donor.query,
            adapter,
            layer=layer,
            heads=layer_heads,
            positions=positions,
        )
        delta = (states["dr"] - states["rr"]).detach().float().cpu()
        for offset, head in enumerate(layer_heads):
            result[(layer, head)] = delta[offset]
    return result


def answer_query_full_replacements(
    donor: MultiSiteBundle,
    adapter: DecoderAdapter,
    *,
    heads: Sequence[HeadKey],
) -> dict[HeadKey, torch.Tensor]:
    return {
        (int(layer), int(head)): query_head_z(
            donor.query, adapter, layer=int(layer), head=int(head)
        )
        for layer, head in heads
    }


def slot_state_replacements(
    donor: MultiSiteBundle,
    *,
    heads: Sequence[HeadKey],
) -> dict[HeadKey, torch.Tensor]:
    return {
        (int(layer), int(head)): donor.slot_z_by_head[(int(layer), int(head))].float()
        for layer, head in heads
    }


def stacked_late_z(
    full_z: torch.Tensor,
    adapter: DecoderAdapter,
    *,
    layer: int,
    heads: Sequence[int],
) -> torch.Tensor:
    values = full_z.detach().float().cpu()
    expected = int(adapter.num_heads[int(layer)]) * int(adapter.head_dims[int(layer)])
    if values.ndim != 1 or len(values) != expected:
        raise ValueError("Full late Z has an incompatible shape")
    return torch.stack(
        [values[_head_slice(adapter, layer=int(layer), head=int(head))] for head in heads],
        dim=0,
    )


def stacked_delta_mapping(
    *, layer: int, heads: Sequence[int], stacked: torch.Tensor
) -> dict[HeadKey, torch.Tensor]:
    values = stacked.detach().float().cpu()
    if values.ndim != 2 or values.shape[0] != len(heads):
        raise ValueError("Stacked delta/head registry shapes disagree")
    return {
        (int(layer), int(head)): values[offset]
        for offset, head in enumerate(heads)
    }


@torch.inference_mode()
def run_path_intervention(
    model: nn.Module,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    mediator_layer: int,
    query_deltas: Mapping[HeadKey, torch.Tensor] | None = None,
    query_replacements: Mapping[HeadKey, torch.Tensor] | None = None,
    slot_replacements: Mapping[HeadKey, torch.Tensor] | None = None,
    late_replacements: Mapping[HeadKey, torch.Tensor] | None = None,
    late_deltas: Mapping[HeadKey, torch.Tensor] | None = None,
) -> PathCausalOutput:
    """Apply multi-layer pre-O path patches in one full causal prefill."""

    query_deltas = dict(query_deltas or {})
    query_replacements = dict(query_replacements or {})
    slot_replacements = dict(slot_replacements or {})
    late_replacements = dict(late_replacements or {})
    late_deltas = dict(late_deltas or {})
    if set(query_deltas) & set(query_replacements):
        raise ValueError("A head cannot receive both query addition and replacement")
    if any(layer == int(mediator_layer) for layer, _head in (
        set(query_deltas) | set(query_replacements) | set(slot_replacements)
    )):
        raise ValueError("Upstream patches must be strictly separate from L28")
    if set(late_replacements) & set(late_deltas):
        raise ValueError("A late head cannot receive both replacement and delta")
    if any(
        layer != int(mediator_layer)
        for layer, _head in set(late_replacements) | set(late_deltas)
    ):
        raise ValueError("Late interventions must all target the mediator layer")
    positions = slot_positions(encoding)
    slot_index = torch.as_tensor(positions, dtype=torch.long)
    layers = sorted(
        {int(mediator_layer)}
        | {layer for layer, _head in query_deltas}
        | {layer for layer, _head in query_replacements}
        | {layer for layer, _head in slot_replacements}
    )
    calls = {layer: 0 for layer in layers}
    late_before: torch.Tensor | None = None
    late_after: torch.Tensor | None = None
    handles = []
    for layer in layers:
        expected = int(adapter.num_heads[layer]) * int(adapter.head_dims[layer])

        def hook(
            _module: nn.Module,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
            expected: int = expected,
        ) -> tuple[Any, ...]:
            nonlocal late_before, late_after
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("o_proj received no positional tensor")
            value = args[0]
            if (
                value.ndim != 3
                or value.shape[0] != 1
                or value.shape[1] != encoding.sequence_length
                or value.shape[-1] != expected
            ):
                raise RuntimeError("Path intervention requires an aligned full prefill")
            patched = value.clone()
            local_slot_index = slot_index.to(device=value.device)
            for (target_layer, head), replacement in slot_replacements.items():
                if target_layer != layer:
                    continue
                source = replacement.to(device=value.device, dtype=value.dtype)
                width = int(adapter.head_dims[layer])
                if source.shape != (len(positions), width):
                    raise RuntimeError("A slot-state replacement has the wrong shape")
                patched[0, local_slot_index, _head_slice(adapter, layer=layer, head=head)] = source
            for (target_layer, head), replacement in query_replacements.items():
                if target_layer != layer:
                    continue
                source = replacement.to(device=value.device, dtype=value.dtype)
                width = int(adapter.head_dims[layer])
                if source.shape != (width,):
                    raise RuntimeError("An answer-query replacement has the wrong shape")
                patched[0, encoding.query_position, _head_slice(adapter, layer=layer, head=head)] = source
            for (target_layer, head), delta in query_deltas.items():
                if target_layer != layer:
                    continue
                addition = delta.to(device=value.device, dtype=value.dtype)
                width = int(adapter.head_dims[layer])
                if addition.shape != (width,):
                    raise RuntimeError("An answer-query delta has the wrong shape")
                patched[0, encoding.query_position, _head_slice(adapter, layer=layer, head=head)] += addition
            if layer == int(mediator_layer):
                late_before = patched[0, encoding.query_position].detach().float().cpu()
                for (_target_layer, head), replacement in late_replacements.items():
                    source = replacement.to(device=value.device, dtype=value.dtype)
                    width = int(adapter.head_dims[layer])
                    if source.shape != (width,):
                        raise RuntimeError("A late replacement has the wrong shape")
                    patched[0, encoding.query_position, _head_slice(adapter, layer=layer, head=head)] = source
                for (_target_layer, head), delta in late_deltas.items():
                    addition = delta.to(device=value.device, dtype=value.dtype)
                    width = int(adapter.head_dims[layer])
                    if addition.shape != (width,):
                        raise RuntimeError("A late delta has the wrong shape")
                    patched[0, encoding.query_position, _head_slice(adapter, layer=layer, head=head)] += addition
                late_after = patched[0, encoding.query_position].detach().float().cpu()
            calls[layer] += 1
            return (patched, *args[1:])

        handles.append(adapter.output_projections[layer].register_forward_pre_hook(hook))
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
    violations = {layer: count for layer, count in calls.items() if count != 1}
    if violations or late_before is None or late_after is None:
        raise RuntimeError(f"Path hook application mismatch: {violations}")
    scored = _score_candidate_sequences(model, encoding, prefill_output)
    return PathCausalOutput(
        causal_output=CausalOutput(
            logits=scored.logits,
            candidate_log_scores=scored.candidate_log_scores,
        ),
        late_z_before=late_before,
        late_z_after=late_after,
    )


def late_block_and_control(
    adapter: DecoderAdapter,
    *,
    layer: int,
    heads: Sequence[int],
    induced_delta: torch.Tensor,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Exact selected-set restoration and an equal-output-norm orthogonal control."""

    induced = induced_delta.detach().float().cpu()
    if induced.ndim != 2 or induced.shape[0] != len(heads):
        raise ValueError("Induced late delta/head-set shapes disagree")
    block = -induced
    induced_output = set_output_from_stacked_z(
        adapter, layer=int(layer), heads=heads, stacked_z=induced
    )
    output_norm = float(torch.linalg.vector_norm(induced_output))
    if output_norm <= 1e-12:
        control = torch.zeros_like(induced)
        control_output = torch.zeros_like(induced_output)
    else:
        control, control_output = _orthogonal_equal_output_norm_delta(
            adapter,
            layer=int(layer),
            heads=heads,
            z_count_steps=induced,
            target_output_norm=output_norm,
            label=label,
        )
    control_norm = float(torch.linalg.vector_norm(control_output))
    cosine = 0.0
    if output_norm > 1e-12 and control_norm > 1e-12:
        cosine = float(torch.dot(induced_output, control_output) / (output_norm * control_norm))
    return block, control, {
        "late_induced_z_norm": float(torch.linalg.vector_norm(induced)),
        "late_induced_output_norm": output_norm,
        "late_control_output_norm": control_norm,
        "late_control_output_cosine_to_induced": cosine,
    }
