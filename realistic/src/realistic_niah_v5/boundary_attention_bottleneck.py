"""Full-trace boundary-state readout through an explicit attention bottleneck.

The native trace is kept at its original length and the native ``Total:``
site stays at its original position.  Only the attention graph of the native
terminal suffix is changed: suffix tokens may read the scrubbed prompt, one
selected list boundary, and earlier suffix tokens.  They cannot read future
items, the rest of the reasoning, or a post-list recap.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import torch

from realistic_niah_v4.modeling import (
    DecoderAdapter,
    _bounded_logits_kwargs,
    _encoding_tensors,
)

from .encoding import NativeTraceEncoding
from .indexed_counter_patch import minimal_terminal_suffix_token_ids


_INTEGER_RE = re.compile(r"(?<!\d)(10|[1-9])(?!\d)")


def full_native_terminal_suffix_positions(
    row: Mapping[str, Any],
    tokenizer: Any,
    encoding: NativeTraceEncoding,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Locate the token-exact native closing delimiter plus ``Total:``."""

    suffix_ids, audit = minimal_terminal_suffix_token_ids(row, tokenizer)
    start = int(encoding.prompt_token_count) + int(
        audit["minimal_terminal_suffix_start_output_token"]
    )
    end = int(encoding.prompt_token_count) + int(
        audit["minimal_terminal_suffix_end_output_token"]
    )
    positions = tuple(range(start, end))
    observed = tuple(int(encoding.input_ids[position]) for position in positions)
    if observed != tuple(int(value) for value in suffix_ids):
        raise ValueError(
            "The minimal native suffix is not contiguous in the full encoding; "
            "a fragment-aware bottleneck is required for this row"
        )
    if not positions or positions[-1] != int(encoding.query_position):
        raise ValueError("The full native suffix must end at the answer query")
    return positions, {
        **audit,
        "full_native_suffix_start": start,
        "full_native_suffix_end": end,
        "full_native_suffix_positions": list(positions),
        "full_native_suffix_is_contiguous": True,
        "full_trace_length_preserved": True,
        "native_query_position_preserved": True,
    }


def select_post_item_boundary_position(
    encoding: NativeTraceEncoding,
    registry: Any,
    tokenizer: Any,
    *,
    occurrence: int,
) -> tuple[int, dict[str, Any]]:
    """Select the first count-neutral separator after item k when available.

    If tokenization leaves no gap between adjacent registered items, fall back
    to the final token of item k.  The choice uses token geometry/text only.
    """

    k = int(occurrence)
    items = tuple(registry.trace_items)
    if not 1 <= k <= len(items):
        raise ValueError("Boundary occurrence is outside the registered list")
    start, end = (int(value) for value in items[k - 1])
    next_start = (
        int(items[k][0]) if k < len(items) else int(encoding.query_position)
    )
    position = end - 1
    kind = "item_endpoint_fallback"
    decoded = tokenizer.decode(
        [int(encoding.input_ids[position])],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if end < next_start:
        candidate_text = tokenizer.decode(
            [int(encoding.input_ids[end])],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if candidate_text and not any(character.isalnum() for character in candidate_text):
            position = end
            kind = "first_post_item_separator"
            decoded = candidate_text
    if not start <= min(position, end - 1) < end:
        raise RuntimeError("Selected boundary is inconsistent with item geometry")
    return position, {
        "boundary_occurrence": k,
        "boundary_position": position,
        "boundary_kind": kind,
        "boundary_token_text": decoded,
        "item_start": start,
        "item_end": end,
        "next_item_start_or_query": next_start,
    }


def memory_geometry_positions(
    encoding: NativeTraceEncoding,
    registry: Any,
    tokenizer: Any,
    *,
    occurrence: int,
    geometry: str,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Compile a predeclared memory-size geometry through occurrence k."""

    k = int(occurrence)
    items = tuple((int(start), int(end)) for start, end in registry.trace_items)
    if not 1 <= k <= len(items):
        raise ValueError("Memory-geometry occurrence is outside the list")
    start, end = items[k - 1]
    if geometry == "post_item_boundary":
        boundary, boundary_audit = select_post_item_boundary_position(
            encoding, registry, tokenizer, occurrence=k
        )
        positions = (boundary,)
        detail = boundary_audit
    elif geometry == "item_endpoint":
        positions = (end - 1,)
        detail = {"boundary_kind": "registered_item_endpoint"}
    elif geometry == "item_suffix4":
        positions = tuple(range(max(start, end - 4), end))
        detail = {"boundary_kind": "registered_item_suffix4"}
    elif geometry == "full_item":
        positions = tuple(range(start, end))
        detail = {"boundary_kind": "registered_full_item"}
    elif geometry == "all_boundaries_through_k":
        values = [
            select_post_item_boundary_position(
                encoding, registry, tokenizer, occurrence=index
            )[0]
            for index in range(1, k + 1)
        ]
        positions = tuple(values)
        detail = {"boundary_kind": "all_post_item_boundaries_through_k"}
    elif geometry == "all_items_through_k":
        positions = tuple(
            position
            for item_start, item_end in items[:k]
            for position in range(item_start, item_end)
        )
        detail = {"boundary_kind": "all_registered_items_through_k"}
    elif geometry == "list_prefix_through_k":
        post_boundary, _boundary_audit = select_post_item_boundary_position(
            encoding, registry, tokenizer, occurrence=k
        )
        positions = tuple(range(int(items[0][0]), post_boundary + 1))
        detail = {"boundary_kind": "contiguous_list_prefix_through_k"}
    else:
        raise ValueError(f"Unknown memory geometry: {geometry}")
    if not positions or len(set(positions)) != len(positions):
        raise RuntimeError("Memory geometry must be unique and nonempty")
    return positions, {
        "memory_geometry": geometry,
        "memory_token_count": len(positions),
        "memory_positions": list(positions),
        "target_occurrence": k,
        "item_start": start,
        "item_end": end,
        **detail,
    }


def build_standard_4d_causal_mask(
    encoding: NativeTraceEncoding,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Build the boolean 4D mask equivalent of the encoding's causal mask."""

    length = int(encoding.sequence_length)
    active = torch.as_tensor(
        encoding.attention_mask, dtype=torch.bool, device=device
    )
    if active.ndim != 1 or int(active.numel()) != length:
        raise ValueError("Encoding attention mask has invalid geometry")
    causal = torch.ones((length, length), dtype=torch.bool, device=device).tril_()
    causal &= active.view(1, length)
    return causal.view(1, 1, length, length)


def build_suffix_attention_bottleneck_mask(
    encoding: NativeTraceEncoding,
    *,
    boundary_positions: Sequence[int],
    suffix_positions: Sequence[int],
    scaffold_end: int | None = None,
    device: torch.device,
) -> torch.Tensor:
    """Cut every route from non-boundary trace tokens into the suffix."""

    boundary = tuple(sorted({int(value) for value in boundary_positions}))
    suffix = tuple(sorted({int(value) for value in suffix_positions}))
    length = int(encoding.sequence_length)
    prompt_end = int(encoding.prompt_token_count)
    context_end = prompt_end if scaffold_end is None else int(scaffold_end)
    query = int(encoding.query_position)
    if not boundary or not suffix:
        raise ValueError("Boundary and suffix positions must both be nonempty")
    if suffix[-1] != query or suffix != tuple(range(suffix[0], query + 1)):
        raise ValueError("The bottleneck suffix must be contiguous through the query")
    if not prompt_end <= context_end <= min(boundary):
        raise ValueError("Scaffold end must lie between prompt and memory")
    if min(boundary) < prompt_end or max(boundary) >= suffix[0]:
        raise ValueError("Boundary must lie inside the trace before the suffix")

    mask = build_standard_4d_causal_mask(encoding, device=device)
    active = torch.as_tensor(
        encoding.attention_mask, dtype=torch.bool, device=device
    )
    scaffold_keys = torch.arange(context_end, device=device)[active[:context_end]]
    boundary_keys = torch.as_tensor(boundary, dtype=torch.long, device=device)
    for suffix_offset, query_position in enumerate(suffix):
        row = mask[0, 0, query_position]
        row.zero_()
        row[scaffold_keys] = True
        row[boundary_keys] = active[boundary_keys]
        prior_suffix = torch.as_tensor(
            suffix[: suffix_offset + 1], dtype=torch.long, device=device
        )
        row[prior_suffix] = active[prior_suffix]
        if not bool(row[query_position]):
            raise RuntimeError("A suffix query was denied self-attention")
    return mask


def build_transition_attention_bottleneck_mask(
    encoding: NativeTraceEncoding,
    *,
    scaffold_end: int,
    donor_boundary_positions: Sequence[int],
    transition_positions: Sequence[int],
    next_boundary_positions: Sequence[int],
    suffix_positions: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    """Force one counted/no-op transition through a transplanted boundary.

    Transition tokens can read only the scrubbed scaffold, the donor boundary,
    and earlier tokens in the transition.  The terminal suffix can read only
    the scrubbed scaffold, the resulting next boundary, and suffix self-state.
    """

    donor = tuple(sorted({int(value) for value in donor_boundary_positions}))
    transition = tuple(sorted({int(value) for value in transition_positions}))
    next_boundary = tuple(sorted({int(value) for value in next_boundary_positions}))
    suffix = tuple(sorted({int(value) for value in suffix_positions}))
    context_end = int(scaffold_end)
    if not donor or not transition or not next_boundary or not suffix:
        raise ValueError("Transition bottleneck groups must all be nonempty")
    if not (
        int(encoding.prompt_token_count)
        <= context_end
        <= min(donor)
        < min(transition)
        <= max(transition)
        <= min(next_boundary)
        < min(suffix)
    ):
        raise ValueError("Transition bottleneck token groups are not causal")
    if not set(next_boundary) <= set(transition):
        raise ValueError("The resulting boundary must be part of the transition")
    if suffix[-1] != int(encoding.query_position):
        raise ValueError("Transition suffix must end at the native query")

    mask = build_standard_4d_causal_mask(encoding, device=device)
    active = torch.as_tensor(
        encoding.attention_mask, dtype=torch.bool, device=device
    )
    scaffold_keys = torch.arange(context_end, device=device)[active[:context_end]]
    donor_keys = torch.as_tensor(donor, dtype=torch.long, device=device)
    for offset, query_position in enumerate(transition):
        row = mask[0, 0, query_position]
        row.zero_()
        row[scaffold_keys] = True
        row[donor_keys] = active[donor_keys]
        prior_transition = torch.as_tensor(
            transition[: offset + 1], dtype=torch.long, device=device
        )
        row[prior_transition] = active[prior_transition]
    next_keys = torch.as_tensor(next_boundary, dtype=torch.long, device=device)
    for offset, query_position in enumerate(suffix):
        row = mask[0, 0, query_position]
        row.zero_()
        row[scaffold_keys] = True
        row[next_keys] = active[next_keys]
        prior_suffix = torch.as_tensor(
            suffix[: offset + 1], dtype=torch.long, device=device
        )
        row[prior_suffix] = active[prior_suffix]
    return mask


@torch.inference_mode()
def prefill_with_custom_attention_mask(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    attention_mask_4d: torch.Tensor,
    patch_positions: Sequence[int] = (),
    patch_layer: int | None = None,
    replacement_states: torch.Tensor | None = None,
) -> tuple[Any, int, float]:
    """Run full prefill with an optional one-shot decoder-input transplant."""

    selected = tuple(int(value) for value in patch_positions)
    do_patch = patch_layer is not None
    if do_patch != (replacement_states is not None):
        raise ValueError("Patch layer and replacement states must be supplied together")
    if do_patch and not selected:
        raise ValueError("A transplant requires at least one patch position")
    if do_patch and not 0 <= int(patch_layer) < int(adapter.num_layers):
        raise ValueError("Patch layer is outside the decoder")
    states = (
        torch.as_tensor(replacement_states).detach().float().cpu()
        if replacement_states is not None
        else None
    )
    if states is not None and (
        states.ndim != 2 or int(states.shape[0]) != len(selected)
    ):
        raise ValueError("Replacement states must have shape [positions, hidden]")

    applications = 0
    realized_norm = 0.0

    def hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
        nonlocal applications, realized_norm
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("Decoder block input is not a positional tensor")
        hidden = args[0]
        if hidden.ndim != 3 or int(hidden.shape[1]) != encoding.sequence_length:
            return None
        assert states is not None
        before = hidden[:, list(selected), :]
        replacement = states.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(0)
        if replacement.shape != before.shape:
            raise RuntimeError("Boundary transplant hidden width mismatch")
        patched = hidden.clone()
        patched[:, list(selected), :] = replacement
        realized_norm = float(
            torch.linalg.vector_norm(before.float() - replacement.float())
            .detach()
            .cpu()
        )
        applications += 1
        return (patched, *args[1:])

    handle = None
    if do_patch:
        handle = adapter.layers[int(patch_layer)].register_forward_pre_hook(hook)
    try:
        input_ids, _attention_mask = _encoding_tensors(model, encoding)
        mask = attention_mask_4d.to(device=input_ids.device)
        if tuple(mask.shape) != (1, 1, encoding.sequence_length, encoding.sequence_length):
            raise ValueError("Custom attention mask has invalid full-prefill shape")
        prefill = model(
            input_ids=input_ids,
            attention_mask=mask,
            use_cache=True,
            **_bounded_logits_kwargs(model),
        )
    finally:
        if handle is not None:
            handle.remove()
    if applications != (1 if do_patch else 0):
        raise RuntimeError(
            "Boundary transplant must apply exactly once when requested; "
            f"observed {applications}"
        )
    return prefill, applications, realized_norm


def _candidate_next_token_distribution(
    logits: torch.Tensor,
    encoding: NativeTraceEncoding,
) -> tuple[dict[str, float], int | None]:
    first_ids: dict[int, int] = {}
    for count, token_ids in encoding.count_candidate_answer_token_ids:
        if token_ids:
            first_ids[int(count)] = int(token_ids[0])
    if set(first_ids) != set(range(1, 11)) or len(set(first_ids.values())) != 10:
        return {}, None
    counts = tuple(range(1, 11))
    selected = torch.stack([logits[first_ids[count]].float() for count in counts])
    probabilities = torch.softmax(selected, dim=0).detach().cpu().tolist()
    distribution = {
        str(count): float(probability)
        for count, probability in zip(counts, probabilities)
    }
    return distribution, int(counts[int(torch.argmax(selected).detach().cpu())])


@torch.inference_mode()
def greedy_integer_from_bottleneck_prefill(
    model: Any,
    tokenizer: Any,
    encoding: NativeTraceEncoding,
    prefill: Any,
    *,
    boundary_positions: Sequence[int],
    suffix_positions: Sequence[int],
    scaffold_end: int | None = None,
    target_count: int,
    max_new_tokens: int = 2,
) -> dict[str, Any]:
    """Greedily decode while preserving the same graph cut after prefill."""

    if int(max_new_tokens) < 1:
        raise ValueError("Greedy bottleneck decoding needs at least one token")
    logits = prefill.logits[0, -1]
    distribution, candidate_argmax = _candidate_next_token_distribution(
        logits, encoding
    )
    cache = prefill.past_key_values
    generated: list[int] = []
    current_logits = logits
    base_length = int(encoding.sequence_length)
    context_end = (
        int(encoding.prompt_token_count)
        if scaffold_end is None
        else int(scaffold_end)
    )
    active_scaffold = tuple(
        position
        for position in range(context_end)
        if int(encoding.attention_mask[position])
    )
    fixed_keys = tuple(
        sorted(
            set(active_scaffold)
            | {int(value) for value in boundary_positions}
            | {int(value) for value in suffix_positions}
        )
    )
    for step in range(int(max_new_tokens)):
        token_id = int(torch.argmax(current_logits).detach().cpu())
        generated.append(token_id)
        if step + 1 >= int(max_new_tokens):
            break
        input_ids = torch.tensor(
            [[token_id]], dtype=torch.long, device=current_logits.device
        )
        key_length = base_length + len(generated)
        decode_mask = torch.zeros(
            (1, 1, 1, key_length), dtype=torch.bool, device=input_ids.device
        )
        decode_mask[0, 0, 0, list(fixed_keys)] = True
        decode_mask[0, 0, 0, base_length:key_length] = True
        position_ids = torch.tensor(
            [[key_length - 1]], dtype=torch.long, device=input_ids.device
        )
        output = model(
            input_ids=input_ids,
            attention_mask=decode_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            **_bounded_logits_kwargs(model),
        )
        cache = output.past_key_values
        current_logits = output.logits[0, -1]

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    match = _INTEGER_RE.search(text)
    prediction = int(match.group(1)) if match is not None else None
    return {
        "greedy_generated_token_ids": generated,
        "greedy_generated_text": text,
        "greedy_prediction": prediction,
        "greedy_exact": bool(prediction is not None and prediction == int(target_count)),
        "candidate_1_to_10_probability": distribution,
        "candidate_argmax": candidate_argmax,
        "candidate_argmax_exact": bool(
            candidate_argmax is not None and candidate_argmax == int(target_count)
        ),
    }
