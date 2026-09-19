"""Diagnostics for structural scrubbing and post-item counter readout sites."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import re
from typing import Any, Mapping, Sequence

import torch

from realistic_niah_v4.modeling import DecoderAdapter, _encoding_tensors

from .bullet_counterfactual_restore import (
    _COUNT_NEUTRAL_BANNED_SUBSTRINGS,
    _replace_positions_from_pool,
    _safe_ordinary_prompt_token_pool,
    _span_positions,
    audit_complete_marker_scrubbable_list,
    build_marker_scrubbed_list_registry,
    build_scrubbed_source_and_blank,
)
from .causal_sites import build_output_token_map
from .encoding import NativeTraceEncoding
from .count_stream import _prefix_forward
from .indexed_counter_patch import minimal_terminal_suffix_token_ids


_PRELIST_INDEXED_LINE_RE = re.compile(r"^\s*(?:10|[1-9])[.)]\s+")
_EXPLICIT_VALUE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[0-9]+|zero|one|two|three|four|five|six|seven|"
    r"eight|nine|ten)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_COUNT_VALUE_CONTEXT_RE = re.compile(
    r"\b(?:count|total|subtotal|record|records|match|matches|item|items|city|cities)\b",
    re.IGNORECASE,
)


def _single_token_id(tokenizer: Any, text: str) -> int:
    ids = tuple(int(value) for value in tokenizer.encode(text, add_special_tokens=False))
    if len(ids) != 1:
        raise ValueError(f"Expected {text!r} to encode as one token, observed {ids}")
    return ids[0]


def _contains_alphanumeric(tokenizer: Any, token_id: int) -> bool:
    text = tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return any(character.isalnum() for character in text)


def build_structure_preserving_source_and_blank(
    encoding: NativeTraceEncoding,
    registry: Any,
    tokenizer: Any,
    *,
    random_seed: int,
    marker_kind: str,
    keep_explicit_indices: bool,
) -> tuple[NativeTraceEncoding, NativeTraceEncoding, dict[str, Any]]:
    """Scrub semantic evidence while retaining punctuation and list newlines.

    This fixes a diagnostic flaw in the original construction, which replaced
    every non-item token in the trace and therefore destroyed the line breaks
    separating otherwise preserved list items.
    """

    original = tuple(int(value) for value in encoding.input_ids)
    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", ())}
    prompt_positions = set(_span_positions(registry.prompt_records))
    first_item_start = int(registry.trace_items[0][0])
    last_item_end = int(registry.trace_items[-1][1])
    preitem_positions = set(
        range(int(registry.prompt_token_count), first_item_start)
    )
    semantic_base_positions = tuple(
        sorted(
            position
            for position in prompt_positions | preitem_positions
            if original[position] not in special_ids
            and _contains_alphanumeric(tokenizer, original[position])
        )
    )
    pool = _safe_ordinary_prompt_token_pool(encoding, registry, tokenizer)
    source_ids, base_changed = _replace_positions_from_pool(
        original,
        semantic_base_positions,
        pool,
        salt=f"{encoding.request_id}|structural-source|{int(random_seed)}",
        tokenizer=tokenizer,
    )
    source_values = list(source_ids)
    index_positions: tuple[int, ...] = ()
    if str(marker_kind) == "indexed" and not bool(keep_explicit_indices):
        index_positions = tuple(
            position
            for position in _span_positions(getattr(registry, "trace_markers", ()))
            if original[position] not in special_ids
        )
        dash_id = _single_token_id(tokenizer, "-")
        for position in index_positions:
            source_values[position] = dash_id
    source_ids = tuple(source_values)

    # The receiver deliberately has no line/list structure. The Source retains
    # the original separators, while the receiver list envelope is replaced by
    # ordinary count-neutral tokens at the same positions.
    receiver_positions = tuple(
        position
        for position in range(first_item_start, last_item_end)
        if source_ids[position] not in special_ids
    )
    blank_ids, blank_changed = _replace_positions_from_pool(
        source_ids,
        receiver_positions,
        pool,
        salt=f"{encoding.request_id}|structural-blank|{int(random_seed)}",
        tokenizer=tokenizer,
    )
    source = replace(encoding, input_ids=source_ids)
    blank = replace(encoding, input_ids=blank_ids)
    return source, blank, {
        "scrub_variant": "structure_preserving_semantic_base_scrub",
        "interitem_structure_preserved_in_source": True,
        "interitem_structure_preserved_in_blank": False,
        "semantic_base_token_count": len(semantic_base_positions),
        "explicit_index_token_count": len(index_positions),
        "explicit_indices_kept": bool(keep_explicit_indices),
        "base_changed_token_count": int(base_changed),
        "blank_changed_token_count": int(blank_changed),
    }


def _raw_line_spans(raw: str) -> tuple[tuple[int, int, str], ...]:
    result: list[tuple[int, int, str]] = []
    cursor = 0
    for value in raw.splitlines(keepends=True):
        content = value.rstrip("\r\n")
        result.append((cursor, cursor + len(content), content))
        cursor += len(value)
    if cursor < len(raw):
        result.append((cursor, len(raw), raw[cursor:]))
    return tuple(result)


def _replace_prompt_record_spans_with_contiguous_neutral_text(
    token_ids: Sequence[int],
    registry: Any,
    tokenizer: Any,
    *,
    salt: str,
) -> tuple[tuple[int, ...], int]:
    """Copy outcome-blind, coherent non-record prompt fragments by token span."""

    output = [int(value) for value in token_ids]
    forbidden = set(_span_positions(registry.prompt_records))
    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", ())}
    prompt_end = int(registry.prompt_token_count)
    token_safe: list[bool] = [False] * prompt_end
    for position in range(1, prompt_end):
        if position in forbidden or output[position] in special_ids:
            continue
        text = tokenizer.decode(
            [output[position]],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        lowered = text.casefold()
        token_safe[position] = bool(
            text
            and not any(character.isdigit() for character in text)
            and not any(value in lowered for value in _COUNT_NEUTRAL_BANNED_SUBSTRINGS)
        )

    changed = 0
    for span_index, (raw_start, raw_end) in enumerate(registry.prompt_records):
        start, end = int(raw_start), int(raw_end)
        length = end - start
        candidates: list[int] = []
        for candidate_start in range(1, prompt_end - length + 1):
            candidate_end = candidate_start + length
            if not all(token_safe[candidate_start:candidate_end]):
                continue
            candidate_ids = output[candidate_start:candidate_end]
            decoded = tokenizer.decode(
                candidate_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            lowered = decoded.casefold()
            if (
                not any(character.isalpha() for character in decoded)
                or any(character.isdigit() for character in decoded)
                or any(
                    value in lowered for value in _COUNT_NEUTRAL_BANNED_SUBSTRINGS
                )
            ):
                continue
            candidates.append(candidate_start)
        if not candidates:
            raise ValueError(
                f"No contiguous neutral prompt replacement for record span {span_index}"
            )
        digest = hashlib.sha256(f"{salt}|{span_index}".encode("utf-8")).digest()
        selected = candidates[int.from_bytes(digest[:8], "big") % len(candidates)]
        replacement = output[selected : selected + length]
        before = output[start:end]
        output[start:end] = replacement
        changed += sum(left != right for left, right in zip(before, replacement))
    return tuple(output), changed


def build_targeted_explicit_count_scrub_source_and_blank(
    row: Mapping[str, Any],
    encoding: NativeTraceEncoding,
    registry: Any,
    tokenizer: Any,
    *,
    random_seed: int,
    marker_kind: str,
    mask_index_punctuation: bool,
    first_item_char_override: int | None = None,
) -> tuple[NativeTraceEncoding, NativeTraceEncoding, dict[str, Any]]:
    """Scrub explicit count evidence while preserving count-neutral prose.

    Selection is raw-text/token-geometry based and never reads the generated
    final answer or any patch outcome.  Prompt needle records are always
    scrubbed.  Before the retained list, we scrub alphanumeric tokens on
    numbered-enumeration lines, lines repeating a registered gold city, and
    lines jointly containing a numeric value and count/total/record language.
    """

    original = tuple(int(value) for value in encoding.input_ids)
    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", ())}
    prompt_positions = set(_span_positions(registry.prompt_records))
    raw = str(row.get("raw_output_text", ""))
    parser = row.get("trace_parse", {}).get("parser", {})
    reasoning_start = int(parser.get("reasoning_start_char", 0))
    first_item_char = (
        int(first_item_char_override)
        if first_item_char_override is not None
        else int(audit_complete_marker_scrubbable_list(row)["item_char_spans"][0][0])
    )
    token_map = build_output_token_map(row, tokenizer)
    gold_cities = tuple(
        str(record.get("city", "")).casefold()
        for record in row.get("gold_records", ())
        if str(record.get("city", ""))
    )
    prelist_positions: set[int] = set()
    selected_line_count = 0
    selection_reasons = {
        "numbered_enumeration": 0,
        "gold_city_recap": 0,
        "explicit_count_value": 0,
    }
    for line_number, (start, end, text) in enumerate(_raw_line_spans(raw), start=1):
        if start < reasoning_start or end > first_item_char or end <= start:
            continue
        reasons: list[str] = []
        if _PRELIST_INDEXED_LINE_RE.search(text):
            reasons.append("numbered_enumeration")
        folded = text.casefold()
        if any(city in folded for city in gold_cities):
            reasons.append("gold_city_recap")
        if _EXPLICIT_VALUE_RE.search(text) and _COUNT_VALUE_CONTEXT_RE.search(text):
            reasons.append("explicit_count_value")
        if not reasons:
            continue
        mapped = token_map.span(f"targeted_prelist_line:{line_number}", start, end)
        if mapped.get("status") != "ok":
            raise ValueError(f"Cannot token-map targeted prelist line {line_number}")
        positions = range(
            int(encoding.prompt_token_count) + int(mapped["output_token_start"]),
            int(encoding.prompt_token_count) + int(mapped["output_token_end"]),
        )
        prelist_positions.update(
            position
            for position in positions
            if original[position] not in special_ids
            and _contains_alphanumeric(tokenizer, original[position])
        )
        selected_line_count += 1
        for reason in set(reasons):
            selection_reasons[reason] += 1

    pool = _safe_ordinary_prompt_token_pool(encoding, registry, tokenizer)
    prompt_scrubbed_ids, prompt_changed = (
        _replace_prompt_record_spans_with_contiguous_neutral_text(
            original,
            registry,
            tokenizer,
            salt=f"{encoding.request_id}|targeted-prompt|{int(random_seed)}",
        )
    )
    source_ids, prelist_changed = _replace_positions_from_pool(
        prompt_scrubbed_ids,
        tuple(sorted(prelist_positions)),
        pool,
        salt=f"{encoding.request_id}|targeted-prelist|{int(random_seed)}",
        tokenizer=tokenizer,
    )
    # The two sets are disjoint by construction.
    base_changed = int(prompt_changed) + int(prelist_changed)
    source_values = list(source_ids)
    index_positions: tuple[int, ...] = ()
    index_prefix_punctuation_positions: tuple[int, ...] = ()
    index_positions = tuple(
        position
        for position in _span_positions(getattr(registry, "trace_markers", ()))
        if original[position] not in special_ids
    )
    if index_positions:
        dash_id = _single_token_id(tokenizer, "-")
        for position in index_positions:
            source_values[position] = dash_id
    if str(marker_kind) == "indexed":
        dash_id = _single_token_id(tokenizer, "-")
        # Replacing only `1` in `1. City` leaves the unnatural token sequence
        # `- . City`.  Canonicalize the leading prefix to `-  City` without
        # changing sequence length: digit -> dash and period -> one-space.
        # Other explicit within-item ordinals remain dash-scrubbed above.
        space_id = _single_token_id(tokenizer, " ")
        punctuation_positions: list[int] = []
        for item_start, item_end in registry.trace_items:
            start = int(item_start)
            if start + 1 >= int(item_end):
                continue
            leading = tokenizer.decode(
                [original[start]],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ).strip()
            punctuation = tokenizer.decode(
                [original[start + 1]],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if leading in {str(value) for value in range(1, 11)} and punctuation == ".":
                source_values[start] = dash_id
                source_values[start + 1] = space_id
                punctuation_positions.append(start + 1)
        index_prefix_punctuation_positions = tuple(punctuation_positions)
    source_ids = tuple(source_values)

    first_item_start = int(registry.trace_items[0][0])
    last_item_end = int(registry.trace_items[-1][1])
    receiver_positions = tuple(
        position
        for position in range(first_item_start, last_item_end)
        if source_ids[position] not in special_ids
    )
    blank_ids, blank_changed = _replace_positions_from_pool(
        source_ids,
        receiver_positions,
        pool,
        salt=f"{encoding.request_id}|targeted-blank|{int(random_seed)}",
        tokenizer=tokenizer,
    )
    shared_attention_mask = list(encoding.attention_mask)
    if bool(mask_index_punctuation):
        for position in index_prefix_punctuation_positions:
            shared_attention_mask[position] = 0
    shared_attention_mask_tuple = tuple(int(value) for value in shared_attention_mask)
    return replace(
        encoding,
        input_ids=source_ids,
        attention_mask=shared_attention_mask_tuple,
    ), replace(
        encoding,
        input_ids=blank_ids,
        attention_mask=shared_attention_mask_tuple,
    ), {
        "scrub_variant": "targeted_explicit_count_evidence_scrub",
        "interitem_structure_preserved_in_source": True,
        "interitem_structure_preserved_in_blank": False,
        "prompt_record_scrub_token_count": len(prompt_positions),
        "prompt_record_scrub_mode": "contiguous_count_neutral_prompt_fragments",
        "prompt_record_changed_token_count": int(prompt_changed),
        "targeted_prelist_scrub_token_count": len(prelist_positions),
        "targeted_prelist_changed_token_count": int(prelist_changed),
        "targeted_prelist_line_count": selected_line_count,
        "targeted_prelist_selection_reasons": selection_reasons,
        "explicit_index_token_count": len(index_positions),
        "canonicalized_index_prefix_punctuation_token_count": len(
            index_prefix_punctuation_positions
        ),
        "indexed_prefix_rendering": "digit_to_dash_and_period_to_space_equal_length",
        "index_prefix_punctuation_attention_masked": bool(mask_index_punctuation),
        "attention_masked_index_punctuation_token_count": (
            len(index_prefix_punctuation_positions)
            if bool(mask_index_punctuation)
            else 0
        ),
        "explicit_indices_kept": False,
        "base_changed_token_count": int(base_changed),
        "blank_changed_token_count": int(blank_changed),
        "selection_uses_final_answer": False,
        "selection_uses_patch_outcome": False,
    }


def terminal_suffix_with_optional_newline(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    prepend_newline: bool,
) -> tuple[int, ...]:
    suffix, _audit = minimal_terminal_suffix_token_ids(row, tokenizer)
    if not prepend_newline:
        return tuple(int(value) for value in suffix)
    newline = tuple(int(value) for value in tokenizer.encode("\n", add_special_tokens=False))
    if not newline:
        raise ValueError("Tokenizer supplied no token for a line break")
    return newline + tuple(int(value) for value in suffix)


def build_item_early_stop_with_suffix(
    encoding: NativeTraceEncoding,
    registry: Any,
    *,
    target_occurrence: int,
    terminal_suffix_token_ids: Sequence[int],
) -> tuple[NativeTraceEncoding, dict[str, Any]]:
    occurrence = int(target_occurrence)
    items = tuple(registry.trace_items)
    if not 1 <= occurrence <= len(items):
        raise ValueError("Early-stop occurrence is outside the list")
    item_start, item_end = items[occurrence - 1]
    suffix = tuple(int(value) for value in terminal_suffix_token_ids)
    early_ids = tuple(encoding.input_ids[: int(item_end)]) + suffix
    early_mask = tuple(encoding.attention_mask[: int(item_end)]) + (1,) * len(suffix)
    visible_items = tuple(encoding.trace_item_spans[:occurrence])
    early = replace(
        encoding,
        input_ids=early_ids,
        attention_mask=early_mask,
        query_position=len(early_ids) - 1,
        trace_item_spans=visible_items,
        slot_spans=visible_items,
        needle_spans=visible_items,
    )
    boundary_start = int(item_end)
    return early, {
        "target_occurrence": occurrence,
        "item_start": int(item_start),
        "item_end": int(item_end),
        "suffix_start": boundary_start,
        "query_position": int(early.query_position),
        "suffix_token_count": len(suffix),
        "sequence_length": int(early.sequence_length),
    }


def build_diagnostic_bases(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    random_seed: int,
    construction: str,
) -> tuple[NativeTraceEncoding, NativeTraceEncoding, Any, dict[str, Any]]:
    trace_audit = audit_complete_marker_scrubbable_list(row)
    clean, registry, registry_audit = build_marker_scrubbed_list_registry(
        row, tokenizer, trace_audit=trace_audit
    )
    if construction == "current":
        source, blank, scrub_audit = build_scrubbed_source_and_blank(
            clean, registry, tokenizer, random_seed=int(random_seed)
        )
    elif construction in {"structural_indices_scrubbed", "structural_indices_intact"}:
        source, blank, scrub_audit = build_structure_preserving_source_and_blank(
            clean,
            registry,
            tokenizer,
            random_seed=int(random_seed),
            marker_kind=str(trace_audit["marker_kind"]),
            keep_explicit_indices=construction == "structural_indices_intact",
        )
    elif construction in {
        "targeted_explicit_count_scrub",
        "targeted_explicit_count_scrub_masked_index_punctuation",
    }:
        source, blank, scrub_audit = (
            build_targeted_explicit_count_scrub_source_and_blank(
                row,
                clean,
                registry,
                tokenizer,
                random_seed=int(random_seed),
                marker_kind=str(trace_audit["marker_kind"]),
                mask_index_punctuation=(
                    construction
                    == "targeted_explicit_count_scrub_masked_index_punctuation"
                ),
            )
        )
    else:
        raise ValueError(f"Unknown diagnostic construction: {construction}")
    return source, blank, registry, {
        **trace_audit,
        **registry_audit,
        **scrub_audit,
        "construction": construction,
    }


def _layer_output_hidden(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported decoder-block output type: {type(output).__name__}")


def _replace_layer_output_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise TypeError(f"Unsupported decoder-block output type: {type(output).__name__}")


@torch.inference_mode()
def capture_last_decoder_block_output_states(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    positions: Sequence[int],
) -> torch.Tensor:
    selected = tuple(int(value) for value in positions)
    captured: torch.Tensor | None = None

    def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
        nonlocal captured
        hidden = _layer_output_hidden(output)
        if hidden.ndim == 3 and hidden.shape[1] == encoding.sequence_length:
            captured = hidden[0, list(selected)].detach().float().cpu()

    handle = adapter.layers[-1].register_forward_hook(hook)
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        _prefix_forward(model, adapter, input_ids, attention_mask)
    finally:
        handle.remove()
    if captured is None:
        raise RuntimeError("Last decoder block output capture did not apply")
    return captured


@torch.inference_mode()
def prefill_with_last_decoder_block_output_replacement(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    positions: Sequence[int],
    replacement_states: torch.Tensor,
) -> tuple[Any, int]:
    selected = tuple(int(value) for value in positions)
    states = torch.as_tensor(replacement_states).detach().float().cpu()
    applications = 0

    def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
        nonlocal applications
        hidden = _layer_output_hidden(output)
        if hidden.ndim != 3 or hidden.shape[1] != encoding.sequence_length:
            return output
        replacement = states.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(0)
        before = hidden[:, list(selected), :]
        if before.shape != replacement.shape:
            raise RuntimeError("Last-block output replacement shape mismatch")
        patched = hidden.clone()
        patched[:, list(selected), :] = replacement
        applications += 1
        return _replace_layer_output_hidden(output, patched)

    handle = adapter.layers[-1].register_forward_hook(hook)
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        prefill = _prefix_forward(model, adapter, input_ids, attention_mask)
    finally:
        handle.remove()
    if applications != 1:
        raise RuntimeError(
            f"Last decoder block output patch applied {applications} times"
        )
    return prefill, applications


@torch.inference_mode()
def prefill_with_layerwise_decoder_block_input_replacements(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    positions: Sequence[int],
    replacements: Mapping[int, torch.Tensor],
) -> tuple[Any, dict[int, int], dict[int, float]]:
    """Clamp the same small span at multiple decoder-block inputs."""

    selected = tuple(int(value) for value in positions)
    replacement_map = {
        int(layer): torch.as_tensor(states).detach().float().cpu()
        for layer, states in replacements.items()
    }
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Layerwise input clamp positions must be unique and nonempty")
    applications = {layer: 0 for layer in replacement_map}
    norms = {layer: 0.0 for layer in replacement_map}
    handles = []
    for layer, states in sorted(replacement_map.items()):
        if not 0 <= layer < int(adapter.num_layers):
            raise ValueError(f"Layerwise input clamp L{layer} is outside the decoder")
        if states.ndim != 2 or int(states.shape[0]) != len(selected):
            raise ValueError("Layerwise input states must have [positions, hidden]")

        def hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
            states: torch.Tensor = states,
        ) -> tuple[Any, ...] | None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Decoder block input is not a positional tensor")
            hidden = args[0]
            if hidden.ndim != 3 or hidden.shape[1] != encoding.sequence_length:
                return None
            replacement = states.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(0)
            before = hidden[:, list(selected), :]
            if before.shape != replacement.shape:
                raise RuntimeError("Layerwise input replacement shape mismatch")
            patched = hidden.clone()
            patched[:, list(selected), :] = replacement
            norms[layer] = float(
                torch.linalg.vector_norm(before.float() - replacement.float())
                .detach()
                .cpu()
            )
            applications[layer] += 1
            return (patched, *args[1:])

        handles.append(adapter.layers[layer].register_forward_pre_hook(hook))
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        prefill = _prefix_forward(model, adapter, input_ids, attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    violations = sorted(layer for layer, count in applications.items() if count != 1)
    if violations:
        raise RuntimeError(f"Layerwise input clamp missed layers {violations}")
    return prefill, applications, norms
