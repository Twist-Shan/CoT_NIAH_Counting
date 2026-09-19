from __future__ import annotations

"""Runtime helpers for the preregistered non-thinking follow-ups 22/23."""

import copy
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from realistic_niah_v4.modeling import (
    DecoderAdapter,
    _accepts_keyword,
    _attention_tensor,
    _bounded_logits_kwargs,
    _encoding_tensors,
    _tensor_from_output,
)
from realistic_niah_v4.prompts import PromptEncoding
from realistic_niah_v4_4_3.interventions import (
    CausalOutput,
    _clone_tensor_tree,
    _output_logits,
    _score_candidate_sequences,
    clone_prefill_output_for_scoring,
)


HeadPosition = tuple[int, int, int]


def _single_query_attention_row(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    elif isinstance(value, (tuple, list)):
        tensors = [item for item in value if isinstance(item, torch.Tensor)]
        if not tensors:
            raise RuntimeError("Selected attention output contains no tensor")
        tensor = tensors[0]
    else:
        raise RuntimeError(f"Unsupported attention output {type(value).__name__}")
    if tensor.ndim == 5:
        batch, heads, blocks, queries, keys = tensor.shape
        if blocks * queries != 1:
            raise RuntimeError("Selected attention returned multiple query cells")
        tensor = tensor.reshape(batch, heads, 1, keys)
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[2] != 1:
        raise RuntimeError(
            f"Expected [1,heads,1,keys] attention, got {tuple(tensor.shape)}"
        )
    return tensor[0, :, 0].detach().float().cpu()


@torch.inference_mode()
def selective_position_attention_outputs(
    model: nn.Module,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    query_position: int,
    layers: Sequence[int],
) -> tuple[dict[int, torch.Tensor], dict[int, int]]:
    """Reconstruct selected attention rows from a causal prefix cache.

    Only the selected modules use eager attention for the one-token query; all
    other layers retain their native backend.  This is the same cache-only
    attention policy used by the V4.4.5 campaign and intentionally performs no
    second equivalence comparison.
    """

    selected = tuple(sorted({int(layer) for layer in layers}))
    query = int(query_position)
    if not selected or not 0 < query < int(encoding.sequence_length):
        raise ValueError("A valid query position and at least one layer are required")
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    prefix_output = model(
        input_ids=input_ids[:, :query],
        attention_mask=attention_mask[:, :query],
        use_cache=True,
        output_attentions=False,
        **_bounded_logits_kwargs(model),
    )
    past = getattr(prefix_output, "past_key_values", None)
    if past is None:
        raise RuntimeError("Prefix reconstruction returned no KV cache")
    kwargs: dict[str, Any] = {
        "input_ids": input_ids[:, query : query + 1],
        "attention_mask": attention_mask[:, : query + 1],
        "past_key_values": past,
        "use_cache": False,
        "output_attentions": False,
        **_bounded_logits_kwargs(model),
    }
    if _accepts_keyword(model, "position_ids"):
        kwargs["position_ids"] = torch.tensor(
            [[query]], dtype=torch.long, device=input_ids.device
        )
    if _accepts_keyword(model, "cache_position"):
        kwargs["cache_position"] = torch.tensor(
            [query], dtype=torch.long, device=input_ids.device
        )
    shared = getattr(prefix_output, "shared_kv_states", None)
    if shared is not None and _accepts_keyword(model, "shared_kv_states"):
        kwargs["shared_kv_states"] = shared

    original_configs: dict[int, Any] = {}
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer in selected:
        attention = adapter.attentions[layer]
        original = getattr(attention, "config", None)
        if original is None or not hasattr(original, "_attn_implementation"):
            raise RuntimeError("Selected attention exposes no backend config")
        patched = copy.copy(original)
        patched._attn_implementation = "eager"
        original_configs[layer] = original
        attention.config = patched

        def hook(
            _module: nn.Module,
            _args: tuple[Any, ...],
            output: Any,
            *,
            layer: int = layer,
        ) -> None:
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise RuntimeError("Selected attention returned no weight slot")
            captured[layer] = _single_query_attention_row(output[1])

        handles.append(attention.register_forward_hook(hook))
    try:
        model(**kwargs)
    finally:
        for handle in handles:
            handle.remove()
        for layer, original in original_configs.items():
            adapter.attentions[layer].config = original
    missing = sorted(set(selected) - set(captured))
    if missing:
        raise RuntimeError(f"Selected layers produced no attention rows: {missing}")
    starts = {
        layer: query + 1 - int(row.shape[-1]) for layer, row in captured.items()
    }
    if any(value < 0 for value in starts.values()):
        raise RuntimeError("A reconstructed attention key axis is too long")
    return captured, starts


@contextmanager
def position_pre_o_deltas(
    adapter: DecoderAdapter,
    *,
    deltas: Mapping[HeadPosition, torch.Tensor],
):
    """Add frozen natural edge deltas at absolute prompt positions.

    The hook applies to a full prompt or a zero-based prefix containing the
    registered position.  It skips one-token cached queries and candidate
    continuations, whose local tensor axis does not contain the absolute site.
    """

    registered = {
        (int(layer), int(head), int(position)): value.detach().float().cpu()
        for (layer, head, position), value in deltas.items()
    }
    if not registered:
        yield {"applications": 0, "sites": 0}
        return
    grouped: dict[int, list[tuple[int, int, torch.Tensor]]] = {}
    for (layer, head, position), value in registered.items():
        grouped.setdefault(layer, []).append((head, position, value))
    applications = 0
    touched: dict[HeadPosition, int] = {key: 0 for key in registered}
    handles = []
    for layer, edits in grouped.items():
        width = int(adapter.head_dims[layer])
        expected = int(adapter.num_heads[layer]) * width

        def hook(
            _module: nn.Module,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
            edits: tuple[tuple[int, int, torch.Tensor], ...] = tuple(edits),
            width: int = width,
            expected: int = expected,
        ) -> tuple[Any, ...] | None:
            nonlocal applications
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("o_proj received no positional tensor")
            value = args[0]
            if value.ndim != 3 or value.shape[-1] != expected:
                raise RuntimeError("Unexpected pre-O tensor shape")
            applicable = [edit for edit in edits if edit[1] < value.shape[1]]
            if not applicable:
                return None
            patched = value.clone()
            for head, position, delta in applicable:
                if delta.shape != (width,):
                    raise RuntimeError("A registered edge delta has the wrong width")
                start = int(head) * width
                patched[:, int(position), start : start + width] += delta.to(
                    device=value.device, dtype=value.dtype
                )
                touched[(layer, int(head), int(position))] += 1
            applications += 1
            return (patched, *args[1:])

        handles.append(adapter.output_projections[layer].register_forward_pre_hook(hook))
    audit = {"applications": 0, "sites": len(registered), "site_calls": touched}
    try:
        yield audit
    finally:
        for handle in handles:
            handle.remove()
        audit["applications"] = int(applications)


@contextmanager
def capture_endpoint_states(
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    layer: int,
):
    endpoints = tuple(int(span.end) - 1 for span in encoding.needle_spans)
    captured: list[torch.Tensor] = []

    def hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        value = _tensor_from_output(output)
        if value.ndim == 3 and value.shape[1] == encoding.sequence_length:
            captured.append(value[0, torch.as_tensor(endpoints, device=value.device)].detach().float().cpu())

    handle = adapter.layers[int(layer)].register_forward_hook(hook)
    audit: dict[str, Any] = {"calls": 0, "states": None, "endpoints": endpoints}
    try:
        yield audit
    finally:
        handle.remove()
        audit["calls"] = len(captured)
        if captured:
            audit["states"] = captured[-1]


def _shared_output(output: Any) -> Any | None:
    return getattr(output, "shared_kv_states", None)


@torch.inference_mode()
def run_answer_query_edge_arm(
    model: nn.Module,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    source_layer: int,
    source_deltas: Mapping[int, torch.Tensor],
    readout_layers: Sequence[int],
    state_layers: Sequence[int] = (),
) -> tuple[
    CausalOutput,
    Any,
    dict[int, torch.Tensor],
    dict[int, int],
    dict[int, torch.Tensor],
    dict[str, int],
]:
    """Run one causal answer query with a source pre-O edge intervention.

    The natural prefix is cached first.  The final query is then executed once
    with the source-head delta and selected later attention rows captured in
    that same forward.  Returned cache/logits are therefore the causal arm,
    not a separately reconstructed diagnostic pass.
    """

    query = int(encoding.query_position)
    if query != int(encoding.sequence_length) - 1:
        raise ValueError("Answer-query edge arms require the final prompt token")
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    prefix = model(
        input_ids=input_ids[:, :query],
        attention_mask=attention_mask[:, :query],
        use_cache=True,
        output_attentions=False,
        **_bounded_logits_kwargs(model),
    )
    past = getattr(prefix, "past_key_values", None)
    if past is None:
        raise RuntimeError("Answer-query prefix returned no cache")
    selected = tuple(sorted({int(value) for value in readout_layers}))
    selected_states = tuple(sorted({int(value) for value in state_layers}))
    originals: dict[int, Any] = {}
    rows: dict[int, torch.Tensor] = {}
    handles = []
    states: dict[int, torch.Tensor] = {}
    for layer in selected:
        attention = adapter.attentions[layer]
        original = getattr(attention, "config", None)
        if original is None or not hasattr(original, "_attn_implementation"):
            raise RuntimeError("Readout attention exposes no backend config")
        patched = copy.copy(original)
        patched._attn_implementation = "eager"
        originals[layer] = original
        attention.config = patched

        def attention_hook(
            _module: nn.Module,
            _args: tuple[Any, ...],
            output: Any,
            *,
            layer: int = layer,
        ) -> None:
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise RuntimeError("Readout attention returned no weight slot")
            rows[layer] = _single_query_attention_row(output[1])

        handles.append(attention.register_forward_hook(attention_hook))
    for layer in selected_states:
        def state_hook(
            _module: nn.Module,
            _args: tuple[Any, ...],
            output: Any,
            *,
            layer: int = layer,
        ) -> None:
            value = _tensor_from_output(output)
            if value.ndim != 3 or value.shape[:2] != (1, 1):
                raise RuntimeError("Answer-query state hook saw an unexpected shape")
            states[layer] = value[0, 0].detach().float().cpu()

        handles.append(adapter.layers[layer].register_forward_hook(state_hook))

    width = int(adapter.head_dims[int(source_layer)])
    expected = int(adapter.num_heads[int(source_layer)]) * width
    source_calls = 0

    def source_hook(_module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
        nonlocal source_calls
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("Source o_proj received no tensor")
        value = args[0]
        if value.ndim != 3 or value.shape[1] != 1 or value.shape[-1] != expected:
            raise RuntimeError("Answer-query source hook saw an unexpected shape")
        patched = value.clone()
        for head, delta in source_deltas.items():
            addition = delta.detach().to(device=value.device, dtype=value.dtype)
            if addition.shape != (width,):
                raise RuntimeError("Source edge delta has the wrong width")
            start = int(head) * width
            patched[:, 0, start : start + width] += addition
        source_calls += 1
        return (patched, *args[1:])

    source_handle = adapter.output_projections[int(source_layer)].register_forward_pre_hook(source_hook)
    kwargs: dict[str, Any] = {
        "input_ids": input_ids[:, query : query + 1],
        "attention_mask": attention_mask,
        "past_key_values": past,
        "use_cache": True,
        "output_attentions": False,
        **_bounded_logits_kwargs(model),
    }
    if _accepts_keyword(model, "position_ids"):
        kwargs["position_ids"] = torch.tensor([[query]], dtype=torch.long, device=input_ids.device)
    if _accepts_keyword(model, "cache_position"):
        kwargs["cache_position"] = torch.tensor([query], dtype=torch.long, device=input_ids.device)
    shared = _shared_output(prefix)
    if shared is not None and _accepts_keyword(model, "shared_kv_states"):
        kwargs["shared_kv_states"] = shared
    try:
        output = model(**kwargs)
    finally:
        source_handle.remove()
        for handle in handles:
            handle.remove()
        for layer, original in originals.items():
            adapter.attentions[layer].config = original
    if source_calls != 1:
        raise RuntimeError(f"Source edge hook applied {source_calls} times")
    missing = sorted(set(selected) - set(rows))
    if missing:
        raise RuntimeError(f"Readout attention rows are missing: {missing}")
    missing_states = sorted(set(selected_states) - set(states))
    if missing_states:
        raise RuntimeError(f"Answer-query state rows are missing: {missing_states}")
    key_starts = {layer: query + 1 - int(row.shape[-1]) for layer, row in rows.items()}
    scoring_output = clone_prefill_output_for_scoring(output)
    scored = _score_candidate_sequences(model, encoding, scoring_output)
    return (
        scored,
        output,
        rows,
        key_starts,
        states,
        {"source_hook_applications": int(source_calls)},
    )
