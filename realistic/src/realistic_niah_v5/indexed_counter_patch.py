"""Old-HTML-style trace patching on the frozen explicit-progress N=10 panel.

This module intentionally lives beside, rather than inside, the no-index
counter experiment.  The intervention is useful evidence that an item-k trace
state carries the running value k, but an explicit numbered/count marker can be
part of the patched span.  It therefore cannot distinguish a latent counter
from a representation of the visible marker.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from typing import Any, Mapping, Sequence

import torch

from realistic_niah_v4.modeling import (
    DecoderAdapter,
    _bounded_logits_kwargs,
    _encoding_tensors,
)

from .count_stream import (
    _prefill_with_layerwise_state_replacements,
    _score_and_generate_prefill,
    _sha256_json,
    _prefix_forward,
    build_answer_source_registry,
)
from .encoding import NativeTraceEncoding
from .unnumbered_counter_restore import build_fully_uninformative_encoding


_COUNT_WORD_RE = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten)\b",
    re.IGNORECASE,
)
_TOTAL_QUERY_RE = re.compile(r"Total\s*:\s*", re.IGNORECASE)
_NATIVE_CHANNEL_CLOSES = ("</think>", "<channel|>")
_EXPLICIT_MARKER_KINDS = frozenset(
    {"indexed", "inline_count", "completion_recap", "ordinal"}
)


def audit_original_explicit_progress_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Audit only frozen-panel geometry; never select on the final answer.

    The V5 hybrid parser registers the longest contiguous 1..M item episode.
    Partial episodes are retained, as in the former HTML's available-span
    tables, but every registered item must have a monotone non-empty span.
    """

    parser = row.get("trace_parse", {}).get("parser", {})
    starts = [int(value) for value in parser.get("item_start_chars", ())]
    ends = [int(value) for value in parser.get("item_end_chars", ())]
    item_count = int(parser.get("item_count", 0))
    marker_kind = str(parser.get("marker_kind", ""))
    reasons: list[str] = []
    if int(row.get("gold_count", -1)) != 10:
        reasons.append("source_count_not_10")
    if item_count < 1 or len(starts) != item_count or len(ends) != item_count:
        reasons.append("item_registry_incomplete")
    previous_end = -1
    for occurrence, (start, end) in enumerate(zip(starts, ends), start=1):
        if not 0 <= start < end or start < previous_end:
            reasons.append(f"invalid_or_nonmonotone_span:{occurrence}")
        previous_end = end
    return {
        "status": "PASS" if not reasons else "FAIL",
        "eligible": not reasons,
        "reasons": reasons,
        "marker_kind": marker_kind,
        "parsed_item_count": item_count,
        "gold_count": int(row.get("gold_count", -1)),
        "trace_one_to_one": bool(parser.get("trace_one_to_one")),
        "explicit_progress_marker": marker_kind in _EXPLICIT_MARKER_KINDS,
        "available_span_analysis": True,
        "selection_uses_final_answer": False,
        "prompt_modified": False,
    }


def minimal_terminal_suffix_token_ids(
    row: Mapping[str, Any], tokenizer: Any
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Extract only the native channel close and the ``Total:`` query.

    Natural traces often put a recap such as ``there are ten`` between their
    last parsed item and the answer.  Gemma can also put that recap *after* its
    ``<channel|>`` boundary.  Reusing the whole contiguous tail would leak the
    result.  We therefore retain token-exact fragments for the model-native
    channel close and the ``Total:`` query, but remove any intervening
    non-whitespace prose.  The answer site ends before the answer digit.
    """

    raw = str(row.get("raw_output_text", ""))
    parser = row.get("trace_parse", {}).get("parser", {})
    reasoning_end = int(parser.get("reasoning_end_char", -1))
    query_match = _TOTAL_QUERY_RE.search(raw, reasoning_end)
    if query_match is None:
        raise ValueError("Indexed early stop has no Total: answer query after reasoning")
    query_start = int(query_match.start())
    answer_end = int(query_match.end())
    if not 0 <= reasoning_end < answer_end <= len(raw):
        raise ValueError("Indexed early-stop terminal suffix bounds are invalid")
    channel_close = next(
        (
            value
            for value in _NATIVE_CHANNEL_CLOSES
            if raw.startswith(value, reasoning_end)
        ),
        None,
    )
    if channel_close is None:
        raise ValueError("Indexed early stop cannot identify the native channel close")
    close_end = reasoning_end + len(channel_close)
    if close_end > query_start:
        raise ValueError("Indexed early-stop channel close crosses the Total query")
    interstitial_text = raw[close_end:query_start]
    remove_interstitial = bool(interstitial_text.strip())
    retained_interstitial = "" if remove_interstitial else interstitial_text
    query_text = raw[query_start:answer_end]
    suffix_text = channel_close + retained_interstitial + query_text
    suffix_query_match = _TOTAL_QUERY_RE.search(suffix_text)
    if suffix_query_match is None or suffix_query_match.end() != len(suffix_text):
        raise ValueError("Indexed early-stop suffix does not end at Total:")
    suffix_digit_match = re.search(r"\d", suffix_text)
    suffix_count_word_match = _COUNT_WORD_RE.search(suffix_text)
    if suffix_digit_match is not None or suffix_count_word_match is not None:
        matched = (
            suffix_digit_match.group(0)
            if suffix_digit_match is not None
            else suffix_count_word_match.group(0)
        )
        raise ValueError(
            "Indexed early-stop minimal suffix leaks a candidate count: "
            f"match={matched!r}"
        )
    baseline_ids = tuple(int(value) for value in row.get("output_token_ids", ()))
    tokenized = tokenizer(
        raw, add_special_tokens=False, return_offsets_mapping=True
    )
    full_ids = tuple(int(value) for value in tokenized["input_ids"])
    offsets = tuple((int(start), int(end)) for start, end in tokenized["offset_mapping"])
    if not baseline_ids or full_ids != baseline_ids or len(offsets) != len(full_ids):
        raise ValueError("Indexed early-stop raw output is not token-exact")

    def boundary_token_index(char_index: int) -> int:
        starts = [index for index, (start, _end) in enumerate(offsets) if start == char_index]
        if starts:
            return int(starts[0])
        ends = [index + 1 for index, (_start, end) in enumerate(offsets) if end == char_index]
        if ends:
            return int(ends[-1])
        raise ValueError(f"Character boundary {char_index} is not token-aligned")

    suffix_start_token = boundary_token_index(reasoning_end)
    close_end_token = boundary_token_index(close_end)
    query_start_token = boundary_token_index(query_start)
    suffix_end_token = boundary_token_index(answer_end)
    if remove_interstitial:
        token_ids = (
            baseline_ids[suffix_start_token:close_end_token]
            + baseline_ids[query_start_token:suffix_end_token]
        )
    else:
        token_ids = baseline_ids[suffix_start_token:suffix_end_token]
    if not token_ids:
        raise ValueError("Indexed early-stop suffix tokenized to an empty sequence")
    decoded_suffix = tokenizer.decode(token_ids, skip_special_tokens=False)
    if decoded_suffix != suffix_text:
        raise ValueError(
            "Indexed early-stop token fragments do not decode to the minimal suffix"
        )
    return token_ids, {
        "minimal_terminal_suffix_text_sha256": hashlib.sha256(
            suffix_text.encode("utf-8")
        ).hexdigest(),
        "minimal_terminal_suffix_token_ids_sha256": _sha256_json(token_ids),
        "minimal_terminal_suffix_token_count": len(token_ids),
        "minimal_terminal_suffix_start_output_token": suffix_start_token,
        "minimal_terminal_suffix_end_output_token": suffix_end_token,
        "minimal_terminal_suffix_contains_candidate_digit": False,
        "minimal_terminal_suffix_contains_count_word": False,
        "natural_recap_removed": True,
        "interstitial_nonwhitespace_removed": remove_interstitial,
        "removed_interstitial_text_sha256": hashlib.sha256(
            interstitial_text.encode("utf-8")
        ).hexdigest(),
        "removed_interstitial_contains_candidate_digit": bool(
            re.search(r"\d", interstitial_text)
        ),
        "removed_interstitial_contains_count_word": bool(
            _COUNT_WORD_RE.search(interstitial_text)
        ),
        "minimal_terminal_suffix_fragment_count": 2 if remove_interstitial else 1,
        "terminal_suffix_source": (
            "saved_output_token_ids_channel_close_plus_Total_query_fragments"
            if remove_interstitial
            else "saved_output_token_ids_reasoning_end_to_Total_colon_and_query_space"
        ),
    }


def build_minimal_item_early_stop_encoding(
    encoding: NativeTraceEncoding,
    registry: Any,
    *,
    target_occurrence: int,
    terminal_suffix_token_ids: Sequence[int],
) -> tuple[NativeTraceEncoding, dict[str, Any]]:
    """Keep the causal prefix through item k, then ask for the count directly."""

    occurrence = int(target_occurrence)
    items = tuple(registry.trace_items)
    if not 1 <= occurrence <= len(items):
        raise ValueError("Indexed early-stop target is outside the parsed episode")
    item_start, item_end = items[occurrence - 1]
    query = int(registry.query_position)
    if not (
        int(encoding.prompt_token_count)
        <= int(item_start)
        < int(item_end)
        <= query
        < int(encoding.sequence_length)
    ):
        raise ValueError("Indexed early-stop item/query ordering is invalid")
    suffix = tuple(int(value) for value in terminal_suffix_token_ids)
    if not suffix:
        raise ValueError("Indexed early-stop suffix is empty")
    original_ids = tuple(int(value) for value in encoding.input_ids)
    early_ids = original_ids[: int(item_end)] + suffix
    early_mask = tuple(int(value) for value in encoding.attention_mask[: int(item_end)]) + (
        1,
    ) * len(suffix)
    new_query = len(early_ids) - 1
    visible_items = tuple(encoding.trace_item_spans[:occurrence])
    result = replace(
        encoding,
        input_ids=early_ids,
        attention_mask=early_mask,
        query_position=new_query,
        trace_item_spans=visible_items,
        slot_spans=visible_items,
        needle_spans=visible_items,
    )
    if result.input_ids[: int(item_end)] != original_ids[: int(item_end)]:
        raise RuntimeError("Indexed early stop changed the item-k causal prefix")
    return result, {
        "early_stop_target_occurrence": occurrence,
        "early_stop_item_start": int(item_start),
        "early_stop_item_end": int(item_end),
        "early_stop_original_query_position": query,
        "early_stop_query_position": new_query,
        "future_trace_tokens_present": False,
        "future_trace_items_removed": len(items) - occurrence,
        "item_positions_unchanged": True,
        "readout_mode": "immediate_item_k_minimal_native_terminal_suffix",
    }


@torch.inference_mode()
def capture_decoder_block_input_states(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    positions: Sequence[int],
    *,
    layers: Sequence[int],
) -> dict[int, torch.Tensor]:
    """Capture the residual stream entering each selected decoder block."""

    selected_positions = tuple(int(value) for value in positions)
    selected_layers = tuple(sorted({int(value) for value in layers}))
    if not selected_positions or len(set(selected_positions)) != len(
        selected_positions
    ):
        raise ValueError("Block-input capture positions must be unique and nonempty")
    if not selected_layers or any(
        not 0 <= layer < int(adapter.num_layers) for layer in selected_layers
    ):
        raise ValueError("Block-input capture layers are invalid")
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer in selected_layers:

        def hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
        ) -> None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Decoder block input is not a positional tensor")
            hidden = args[0]
            if hidden.ndim != 3 or hidden.shape[1] != encoding.sequence_length:
                return
            if max(selected_positions) >= int(hidden.shape[1]):
                raise RuntimeError("A block-input capture position is out of bounds")
            captured[layer] = (
                hidden[0, list(selected_positions)].detach().float().cpu()
            )

        handles.append(adapter.layers[layer].register_forward_pre_hook(hook))
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **_bounded_logits_kwargs(model),
        )
    finally:
        for handle in handles:
            handle.remove()
    missing = sorted(set(selected_layers) - set(captured))
    if missing:
        raise RuntimeError(f"Block-input capture missed layers {missing}")
    return captured


@torch.inference_mode()
def prefill_with_single_decoder_block_input_replacement(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    positions: Sequence[int],
    layer: int,
    replacement_states: torch.Tensor,
) -> tuple[Any, int, float]:
    """Patch one block input during prefill and let all later layers recompute."""

    selected_positions = tuple(int(value) for value in positions)
    active_layer = int(layer)
    states = torch.as_tensor(replacement_states).detach().float().cpu()
    if not selected_positions or len(set(selected_positions)) != len(
        selected_positions
    ):
        raise ValueError("Single-layer patch positions must be unique and nonempty")
    if not 0 <= active_layer < int(adapter.num_layers):
        raise ValueError("Single-layer patch layer is outside the decoder")
    if states.ndim != 2 or int(states.shape[0]) != len(selected_positions):
        raise ValueError("Single-layer states must have shape [positions, hidden]")
    applications = 0
    realized_norm = 0.0

    def hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
        nonlocal applications, realized_norm
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("Decoder block input is not a positional tensor")
        hidden = args[0]
        if hidden.ndim != 3 or hidden.shape[1] != encoding.sequence_length:
            return None
        before = hidden[:, list(selected_positions), :]
        replacement = states.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(0)
        if replacement.shape != before.shape:
            raise RuntimeError("Single-layer replacement width disagrees with model")
        patched = hidden.clone()
        patched[:, list(selected_positions), :] = replacement
        realized_norm = float(
            torch.linalg.vector_norm(before.float() - replacement.float())
            .detach()
            .cpu()
        )
        applications += 1
        return (patched, *args[1:])

    handle = adapter.layers[active_layer].register_forward_pre_hook(hook)
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        prefill = _prefix_forward(model, adapter, input_ids, attention_mask)
    finally:
        handle.remove()
    if applications != 1:
        raise RuntimeError(
            "A single decoder-block-input patch must apply exactly once; "
            f"observed {applications}"
        )
    return prefill, applications, realized_norm


@torch.inference_mode()
def run_indexed_counter_early_stop_patch_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    source_layers: Sequence[int],
    target_occurrences: Sequence[int],
    random_seed: int,
    answer_site_id: str = "answer_query_v3",
) -> list[dict[str, Any]]:
    """Run paired full-item restoration and ablation at an immediate readout."""

    trace_audit = audit_original_explicit_progress_row(row)
    if not trace_audit["eligible"]:
        raise ValueError(f"Indexed trace audit failed: {trace_audit['reasons']}")
    clean_full, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    receiver_full, receiver_audit = build_fully_uninformative_encoding(
        clean_full, registry, tokenizer, random_seed=int(random_seed)
    )
    suffix_ids, suffix_audit = minimal_terminal_suffix_token_ids(row, tokenizer)
    if int(suffix_ids[-1]) != int(clean_full.input_ids[registry.query_position]):
        raise ValueError("Minimal early-stop suffix changed the answer-query token")

    layers = tuple(sorted({int(value) for value in source_layers}))
    if not layers or layers[0] < 0 or layers[-1] >= int(adapter.num_layers):
        raise ValueError("Indexed patch source-layer registry is invalid")
    targets = tuple(sorted({int(value) for value in target_occurrences}))
    if not targets or min(targets) < 1 or max(targets) > len(registry.trace_items):
        raise ValueError("Indexed patch target registry is invalid")
    rows: list[dict[str, Any]] = []

    for occurrence in targets:
        clean, clean_early_audit = build_minimal_item_early_stop_encoding(
            clean_full,
            registry,
            target_occurrence=occurrence,
            terminal_suffix_token_ids=suffix_ids,
        )
        receiver, receiver_early_audit = build_minimal_item_early_stop_encoding(
            receiver_full,
            registry,
            target_occurrence=occurrence,
            terminal_suffix_token_ids=suffix_ids,
        )
        if clean_early_audit != receiver_early_audit:
            raise RuntimeError("Clean/corrupt early-stop geometries disagree")
        if (
            clean.query_position != receiver.query_position
            or clean.input_ids[clean.query_position]
            != receiver.input_ids[receiver.query_position]
        ):
            raise RuntimeError("Clean/corrupt early-stop answer queries are misaligned")
        start, end = registry.trace_items[occurrence - 1]
        positions = tuple(range(int(start), int(end)))
        clean_capture = capture_decoder_block_input_states(
            model, adapter, clean, positions, layers=layers
        )
        corrupt_capture = capture_decoder_block_input_states(
            model, adapter, receiver, positions, layers=layers
        )

        def score(active: NativeTraceEncoding) -> dict[str, Any]:
            prefill, _readout, _applications, _norms = (
                _prefill_with_layerwise_state_replacements(
                    model,
                    adapter,
                    active,
                    positions=positions,
                    replacements=None,
                    readout_layers=(int(adapter.num_layers) - 1,),
                    readout_positions=(int(active.query_position),),
                )
            )
            return _score_and_generate_prefill(
                model,
                tokenizer,
                active,
                prefill,
                run_greedy=False,
                max_new_tokens=1,
            )

        clean_outcomes = score(clean)
        corrupt_outcomes = score(receiver)
        common = {
            "schema_version": "realistic_niah_v5_indexed_counter_early_stop_patch_v2",
            "experiment_id": "indexed_old_html_counter_early_stop_patch",
            "request_id": str(clean.request_id),
            "model_label": str(clean.model_label),
            "seed": int(clean.seed),
            "dataset_split": str(clean.split),
            "gold_count": int(clean.count),
            "answer_site_id": answer_site_id,
            "registered_target_occurrences": list(targets),
            "target_occurrence": int(occurrence),
            "patch_geometry": "full_trace_item_same_position",
            "patch_layer_mode": "single_decoder_block_input",
            "upper_layers_recomputed_after_patch": True,
            "readout_mode": "immediate_item_k_minimal_native_terminal_suffix",
            "receiver_prompt_and_visible_trace_items_uninformative": True,
            "prompt_modified": False,
            "source_prompt_is_frozen_original": True,
            "visible_progress_confound_allowed": True,
            "internal_counter_without_visible_index_claim_allowed": False,
            "controlled_running_state_sufficiency_claim_allowed": True,
            "outcome_blind": True,
            "selection_rank_used": False,
            "registry_sha256": registry.to_dict()["registry_sha256"],
            **trace_audit,
            **receiver_audit,
            **suffix_audit,
            **clean_early_audit,
        }
        rows.extend(
            [
                {
                    **common,
                    "condition": "clean_early_stop_reference",
                    "source_layer": -1,
                    "patch_token_count": 0,
                    **clean_outcomes,
                },
                {
                    **common,
                    "condition": "corrupt_early_stop_reference",
                    "source_layer": -1,
                    "patch_token_count": 0,
                    **corrupt_outcomes,
                },
            ]
        )
        for source_layer in layers:
            for condition, active, capture in (
                ("clean_item_restore_into_corrupt", receiver, clean_capture),
                ("corrupt_item_ablate_into_clean", clean, corrupt_capture),
            ):
                prefill, applications, realized = (
                    prefill_with_single_decoder_block_input_replacement(
                        model,
                        adapter,
                        active,
                        positions=positions,
                        layer=int(source_layer),
                        replacement_states=capture[int(source_layer)],
                    )
                )
                outcomes = _score_and_generate_prefill(
                    model,
                    tokenizer,
                    active,
                    prefill,
                    run_greedy=False,
                    max_new_tokens=1,
                )
                rows.append(
                    {
                        **common,
                        "condition": condition,
                        "source_layer": int(source_layer),
                        "patch_layers": [int(source_layer)],
                        "patch_layer_count": 1,
                        "patch_site": "decoder_block_input",
                        "upper_layers_recomputed_after_patch": True,
                        "patch_token_count": len(positions),
                        "receiver_positions_sha256": _sha256_json(positions),
                        "donor_positions_sha256": _sha256_json(positions),
                        "donor_receiver_positions_identical": True,
                        "donor_receiver_span_lengths_equal": True,
                        "patch_hook_applications": {
                            str(source_layer): int(applications)
                        },
                        "patch_realized_fro_norm_by_layer": {
                            str(source_layer): float(realized)
                        },
                        **outcomes,
                    }
                )
    return rows
