"""Paper-aligned counting-mechanism interventions for native thinking traces.

The utilities in this module implement the common causal substrate for four
experiment families:

* CountScope-style causal decoding into a fixed, semantically blank receiver;
* last-k to first-k continued-counting and last-k to last-k maximum-count
  transplants;
* mean position-difference (linear-additivity) steering; and
* separator/marker state collapse.

The paper implementation clamps decoder-block *inputs* at every selected
layer.  Existing V5 restoration code mainly patches post-block states, so the
pre-block intervention below is deliberately separate and explicitly audited.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from realistic_niah_v4.modeling import (
    DecoderAdapter,
    _accepts_keyword,
    _encoding_tensors,
)
from realistic_niah_v4.prompts import TokenSpan
from realistic_niah_v4_4_3.interventions import (
    _output_logits,
    _repeat_batch_tree,
    clone_prefill_output_for_scoring,
)
from realistic_niah_v4_4_5.restoration import (
    generate_answer_completion_from_prefill,
)

from .causal import completion_metrics
from .causal_sites import build_output_token_map
from .count_stream import AnswerSourceRegistry, _positions_to_spans, _prefix_forward
from .encoding import NativeTraceEncoding, build_native_trace_encoding
from .tstar_prefix import build_tstar_prefix_context
from .unnumbered_counter_restore import (
    build_fully_uninformative_encoding,
)


SCHEMA_VERSION = "realistic_niah_v5_counting_mechanism_transfer_v2"
REGIONS = (
    "marker",
    "opening",
    "payload",
    "closing",
    "post_item",
    "nonmarker",
    "full",
    "both",
)
INTERVENTION_KINDS = ("replace", "add")


def _first_pass_metadata_keys(
    row: Mapping[str, Any],
    *,
    selection_population: str = "first_pass_noindex_enumeration",
    eligibility_field: str = "primary_eligible_prefix_clean",
) -> tuple[str, str]:
    audit_keys = [
        str(key)
        for key, value in row.items()
        if str(key).endswith("_format_audit")
        and isinstance(value, Mapping)
        and str(eligibility_field) in value
    ]
    cohort_keys = [
        str(key)
        for key, value in row.items()
        if str(key).endswith("_cohort")
        and isinstance(value, Mapping)
        and str(value.get("selection_population", ""))
        == str(selection_population)
    ]
    if len(audit_keys) != 1 or len(cohort_keys) != 1:
        raise ValueError(
            "Expected exactly one frozen first-pass audit and cohort record "
            f"for {selection_population}/{eligibility_field}: "
            f"audits={audit_keys} cohorts={cohort_keys}"
        )
    return audit_keys[0], cohort_keys[0]


def build_first_pass_tstar_answer_source_registry(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    answer_site_id: str = "answer_query_v3",
    candidate_counts: Sequence[int] = tuple(range(1, 11)),
    selection_population: str = "first_pass_noindex_enumeration",
    eligibility_field: str = "primary_eligible_prefix_clean",
) -> tuple[NativeTraceEncoding, AnswerSourceRegistry]:
    """Compile the frozen first evidence pass, excluding every later token.

    The resulting source ends at the smallest whole-token prefix covering the
    K-th first score-supported event and then appends only the same minimal
    ``Total:`` query used by the early-stop readouts.  For occurrences 2..K,
    the closest punctuation/whitespace-only token between consecutive events
    is registered as that event's separator marker.  If the tokenizer exposes
    no such standalone token, the marker assay is explicitly unavailable for
    that occurrence rather than borrowing a semantic token.
    """

    audit_key, cohort_key = _first_pass_metadata_keys(
        row,
        selection_population=str(selection_population),
        eligibility_field=str(eligibility_field),
    )
    stored_audit = row[audit_key]
    if not bool(stored_audit.get(str(eligibility_field))):
        raise ValueError("Frozen row is not first-pass prefix-clean eligible")
    context = build_tstar_prefix_context(
        row,
        tokenizer,
        audit_key=audit_key,
        cohort_key=cohort_key,
        eligibility_field=str(eligibility_field),
    )
    base = build_native_trace_encoding(
        row,
        tokenizer,
        site_id=answer_site_id,
        candidate_counts=tuple(int(value) for value in candidate_counts),
    )
    prompt_count = int(context["prompt_token_count"])
    prefix_ids = tuple(int(value) for value in context["output_prefix_token_ids"])
    suffix_text = "\n</think>\n\nTotal:"
    suffix_ids = tuple(
        int(value)
        for value in tokenizer.encode(suffix_text, add_special_tokens=False)
    )
    if not suffix_ids:
        raise RuntimeError("First-pass answer-query suffix tokenized to nothing")
    input_ids = tuple(int(value) for value in context["input_ids"]) + suffix_ids
    attention_mask = tuple(int(value) for value in context["attention_mask"]) + (
        1,
    ) * len(suffix_ids)
    query_position = len(input_ids) - 1

    token_map = build_output_token_map(row, tokenizer)
    occurrence_rows = tuple(context["first_occurrences"])
    semantic_spans = tuple(
        (
            prompt_count + int(value["output_token_start"]),
            prompt_count + int(value["output_token_end"]),
        )
        for value in occurrence_rows
    )
    if len(semantic_spans) != int(context["fixed_count"]):
        raise ValueError("First-pass token registry does not match fixed count")

    marker_positions: set[int] = set()
    item_spans: list[tuple[int, int]] = []
    marker_by_occurrence: dict[str, int | None] = {}
    previous_output_end: int | None = None
    for occurrence, (value, semantic_span) in enumerate(
        zip(occurrence_rows, semantic_spans), start=1
    ):
        output_start = int(value["output_token_start"])
        output_end = int(value["output_token_end"])
        marker_output_position: int | None = None
        if previous_output_end is not None:
            for output_position in range(previous_output_end, output_start):
                left, right = token_map.offsets[output_position]
                surface = str(row["raw_output_text"])[int(left) : int(right)]
                if surface and not any(character.isalnum() for character in surface):
                    marker_output_position = output_position
            if marker_output_position is not None:
                marker_positions.add(prompt_count + marker_output_position)
        item_start = (
            prompt_count + marker_output_position
            if marker_output_position is not None
            else int(semantic_span[0])
        )
        item_end = int(semantic_span[1])
        if item_spans and item_start < item_spans[-1][1]:
            raise ValueError("First-pass event regions overlap after marker assignment")
        item_spans.append((item_start, item_end))
        marker_by_occurrence[str(occurrence)] = (
            None
            if marker_output_position is None
            else prompt_count + marker_output_position
        )
        previous_output_end = output_end

    item_positions = set(_positions(item_spans))
    context_positions = set(range(prompt_count, query_position))
    nonmarker_positions = item_positions - marker_positions
    registry = AnswerSourceRegistry(
        request_id=str(base.request_id),
        answer_site_id="first_pass_tstar_minimal_total_query_v1",
        sequence_length=len(input_ids),
        prompt_token_count=prompt_count,
        query_position=query_position,
        prompt_records=tuple(
            (int(span.start), int(span.end)) for span in base.prompt_record_spans
        ),
        trace_context=((prompt_count, query_position),),
        trace_items=tuple(item_spans),
        trace_other=_positions_to_spans(context_positions - item_positions),
        trace_markers=_positions_to_spans(marker_positions),
        trace_nonmarkers=_positions_to_spans(nonmarker_positions),
        earlier_trace_items=tuple(item_spans[:-1]),
        terminal_trace_item=(tuple(item_spans[-1]),),
    )
    registry.validate()
    selected_site = {
        **dict(base.selected_site),
        "site_id": "first_pass_tstar_minimal_total_query_v1",
        "source_answer_site_id": answer_site_id,
        "selection_population": str(selection_population),
        "future_recap_available_to_context": False,
        "t_star_char": int(context["t_star_char"]),
        "output_token_end": int(context["output_token_end"]),
        "token_boundary_right_spill_chars": int(
            context["token_boundary_right_spill_chars"]
        ),
        "separator_marker_positions_by_occurrence": marker_by_occurrence,
    }
    raw_prefix = str(context["raw_prefix_text"]) + suffix_text
    trace_spans = tuple(
        TokenSpan(
            slot_index=index,
            start=int(start),
            end=int(end),
            active=True,
            kind="first_pass_native_trace_item",
            canonical_length=int(end) - int(start),
            model_token_length=int(end) - int(start),
        )
        for index, (start, end) in enumerate(item_spans, start=1)
    )
    encoding = replace(
        base,
        split=str(context["split"]),
        count=int(context["fixed_count"]),
        text=str(row.get("rendered_prompt", "")) + raw_prefix,
        generation_prompt=str(row.get("rendered_prompt", "")) + raw_prefix,
        input_ids=input_ids,
        attention_mask=attention_mask,
        query_position=query_position,
        prompt_token_count=prompt_count,
        raw_prefix_text=raw_prefix,
        selected_site=selected_site,
        trace_item_spans=trace_spans,
        slot_spans=trace_spans,
        needle_spans=trace_spans,
        hard_negative_spans=(),
    )
    if encoding.input_ids[: prompt_count + len(prefix_ids)] != tuple(
        int(value) for value in context["input_ids"]
    ):
        raise RuntimeError("First-pass registry changed the frozen t_star prefix")
    return encoding, registry


def _sha256_jsonable(value: Any) -> str:
    import json

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _positions(spans: Sequence[Sequence[int]]) -> tuple[int, ...]:
    return tuple(
        position
        for start, end in spans
        for position in range(int(start), int(end))
    )


def occurrence_region_positions(
    registry: Any,
    occurrence: int,
    region: str,
) -> tuple[int, ...]:
    """Return one occurrence's exact token positions for a registered region."""

    active_region = "full" if str(region) == "both" else str(region)
    if active_region not in set(REGIONS) - {"both"}:
        raise ValueError(f"Unknown counting-transfer region: {region}")
    index = int(occurrence) - 1
    items = tuple((int(start), int(end)) for start, end in registry.trace_items)
    if not 0 <= index < len(items):
        raise ValueError("Occurrence is outside the registered trace")
    start, end = items[index]
    if active_region == "post_item":
        # Decoder block inputs at ``end`` are the first states that have seen
        # the complete item.  They are therefore the natural site for testing
        # whether an item-boundary state is carried into the next event.
        return (end,)
    item = tuple(range(start, end))
    marker_set = set(int(value) for value in registry.positions("trace_markers"))
    marker = tuple(value for value in item if value in marker_set)
    if active_region == "marker" and not marker:
        raise ValueError(f"Occurrence {occurrence} has no registered marker token")
    nonmarker = tuple(value for value in item if value not in marker_set)
    opening = (nonmarker[0],) if nonmarker else (item[0],)
    closing = (nonmarker[-1],) if nonmarker else (item[-1],)
    closing_set = set(closing)
    payload = tuple(value for value in nonmarker if value not in closing_set)
    groups = {
        "marker": marker,
        "opening": opening,
        "payload": payload,
        "closing": closing,
        "nonmarker": nonmarker,
        "full": item,
    }
    selected = tuple(groups[active_region])
    if not selected:
        raise ValueError(
            f"Occurrence {occurrence} exposes no tokens for region {active_region}"
        )
    return selected


def occurrence_region_groups(
    registry: Any,
    occurrences: Sequence[int],
    region: str,
) -> tuple[tuple[int, ...], ...]:
    values = tuple(int(value) for value in occurrences)
    if not values or len(set(values)) != len(values):
        raise ValueError("Region occurrences must be unique and nonempty")
    return tuple(
        occurrence_region_positions(registry, occurrence, region)
        for occurrence in values
    )


@dataclass(frozen=True)
class RegionAlignment:
    """Token correspondence for one or more source/receiver event regions."""

    receiver_positions: tuple[int, ...]
    donor_positions: tuple[int, ...]
    source_group_widths: tuple[int, ...]
    receiver_group_widths: tuple[int, ...]

    def validate(self) -> None:
        if not self.receiver_positions:
            raise ValueError("Region alignment is empty")
        if len(self.receiver_positions) != len(self.donor_positions):
            raise ValueError("Donor/receiver token arity differs")
        if len(set(self.receiver_positions)) != len(self.receiver_positions):
            raise ValueError("Receiver positions must be unique")
        if len(self.source_group_widths) != len(self.receiver_group_widths):
            raise ValueError("Source/receiver event arity differs")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "receiver_token_count": len(self.receiver_positions),
            "unique_donor_token_count": len(set(self.donor_positions)),
            "source_group_widths": list(self.source_group_widths),
            "receiver_group_widths": list(self.receiver_group_widths),
            "donor_positions_sha256": _sha256_jsonable(self.donor_positions),
            "receiver_positions_sha256": _sha256_jsonable(
                self.receiver_positions
            ),
            "alignment_rule": "within_event_normalized_midpoint_nearest",
        }


def align_region_groups(
    source_groups: Sequence[Sequence[int]],
    receiver_groups: Sequence[Sequence[int]],
) -> RegionAlignment:
    """Align unequal-width event spans without crossing event boundaries.

    Repeated donor positions are allowed when a receiver span is wider than its
    source.  Receiver positions remain one-to-one and are always unique.
    """

    sources = tuple(tuple(int(value) for value in group) for group in source_groups)
    receivers = tuple(
        tuple(int(value) for value in group) for group in receiver_groups
    )
    if not sources or len(sources) != len(receivers):
        raise ValueError("Source and receiver must contain the same event count")
    donor_positions: list[int] = []
    receiver_positions: list[int] = []
    for source, receiver in zip(sources, receivers):
        if not source or not receiver:
            raise ValueError("Every aligned event region must be nonempty")
        source_width = len(source)
        receiver_width = len(receiver)
        for target_index, receiver_position in enumerate(receiver):
            relative_midpoint = (target_index + 0.5) / receiver_width
            source_index = min(
                source_width - 1,
                int(np.floor(relative_midpoint * source_width)),
            )
            donor_positions.append(source[source_index])
            receiver_positions.append(receiver_position)
    result = RegionAlignment(
        receiver_positions=tuple(receiver_positions),
        donor_positions=tuple(donor_positions),
        source_group_widths=tuple(len(group) for group in sources),
        receiver_group_widths=tuple(len(group) for group in receivers),
    )
    result.validate()
    return result


def align_occurrence_regions(
    registry: Any,
    *,
    source_occurrences: Sequence[int],
    receiver_occurrences: Sequence[int],
    region: str,
) -> RegionAlignment:
    return align_region_groups(
        occurrence_region_groups(registry, source_occurrences, region),
        occurrence_region_groups(registry, receiver_occurrences, region),
    )


def gather_aligned_states(
    captures: Mapping[int, torch.Tensor],
    capture_positions: Sequence[int],
    donor_positions: Sequence[int],
) -> dict[int, torch.Tensor]:
    """Gather a possibly repeated donor-token mapping from unique captures."""

    positions = tuple(int(value) for value in capture_positions)
    if len(set(positions)) != len(positions):
        raise ValueError("Capture positions must be unique")
    lookup = {position: index for index, position in enumerate(positions)}
    missing = sorted(set(int(value) for value in donor_positions) - set(lookup))
    if missing:
        raise ValueError(f"Aligned donor positions were not captured: {missing}")
    indices = [lookup[int(value)] for value in donor_positions]
    result: dict[int, torch.Tensor] = {}
    for layer, states in captures.items():
        value = torch.as_tensor(states).detach().float().cpu()
        if value.ndim != 2 or value.shape[0] != len(positions):
            raise ValueError("Captured block-input states have the wrong shape")
        result[int(layer)] = value[indices].clone()
    return result


@torch.inference_mode()
def prefill_with_block_input_intervention(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    positions: Sequence[int],
    layer_values: Mapping[int, torch.Tensor] | None,
    intervention_kind: str,
    readout_layers: Sequence[int] = (),
    readout_positions: Sequence[int] = (),
    norm_rescale_replacement: bool = False,
) -> tuple[Any, dict[int, torch.Tensor], dict[int, int], dict[int, float]]:
    """Run one prefill with audited pre-block replacement or additive hooks."""

    kind = str(intervention_kind)
    if kind not in INTERVENTION_KINDS:
        raise ValueError(f"Unknown block-input intervention kind: {kind}")
    selected_positions = tuple(int(value) for value in positions)
    if not selected_positions or len(set(selected_positions)) != len(
        selected_positions
    ):
        raise ValueError("Intervention positions must be unique and nonempty")
    if min(selected_positions) < 0 or max(selected_positions) >= encoding.sequence_length:
        raise ValueError("Intervention position is outside the encoding")
    values = {
        int(layer): torch.as_tensor(states).detach().float().cpu()
        for layer, states in (layer_values or {}).items()
    }
    for layer, states in values.items():
        if not 0 <= layer < int(adapter.num_layers):
            raise ValueError(f"Intervention layer L{layer} is outside the decoder")
        if states.ndim != 2 or states.shape[0] != len(selected_positions):
            raise ValueError("Layer values must have shape [positions, hidden]")
    capture_layers = tuple(sorted({int(value) for value in readout_layers}))
    capture_positions = tuple(int(value) for value in readout_positions)
    if bool(capture_layers) != bool(capture_positions):
        raise ValueError("Readout layers and positions must be jointly present")
    if any(not 0 <= layer < int(adapter.num_layers) for layer in capture_layers):
        raise ValueError("Readout layer is outside the decoder")
    if capture_positions and (
        min(capture_positions) < 0
        or max(capture_positions) >= encoding.sequence_length
    ):
        raise ValueError("Readout position is outside the encoding")

    active_layers = tuple(sorted(set(values) | set(capture_layers)))
    applications = {layer: 0 for layer in values}
    realized_norms = {layer: 0.0 for layer in values}
    captures: dict[int, torch.Tensor] = {}
    handles = []
    epsilon = 1e-6
    for layer in active_layers:

        def hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
        ) -> tuple[Any, ...] | None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Decoder block input is not a positional tensor")
            hidden = args[0]
            if hidden.ndim != 3 or int(hidden.shape[1]) != encoding.sequence_length:
                return None
            active = hidden
            if layer in values:
                if int(hidden.shape[0]) != 1:
                    raise RuntimeError("Counting-transfer hooks require batch size one")
                before = hidden[:, list(selected_positions), :]
                supplied = values[layer].to(
                    device=hidden.device, dtype=hidden.dtype
                ).unsqueeze(0)
                if supplied.shape != before.shape:
                    raise RuntimeError("Intervention hidden width disagrees with model")
                if kind == "replace":
                    replacement = supplied
                    if norm_rescale_replacement:
                        target_norm = torch.linalg.vector_norm(
                            before.float(), dim=-1, keepdim=True
                        )
                        donor_norm = torch.linalg.vector_norm(
                            replacement.float(), dim=-1, keepdim=True
                        )
                        replacement = replacement * (
                            target_norm / torch.clamp(donor_norm, min=epsilon)
                        ).to(replacement.dtype)
                else:
                    replacement = before + supplied
                active = hidden.clone()
                active[:, list(selected_positions), :] = replacement
                realized_norms[layer] = float(
                    torch.linalg.vector_norm(
                        (replacement - before).float()
                    ).detach().cpu()
                )
                applications[layer] += 1
            if layer in capture_layers:
                captures[layer] = (
                    active[0, list(capture_positions)].detach().float().cpu()
                )
            if active is hidden:
                return None
            return (active, *args[1:])

        handles.append(adapter.layers[layer].register_forward_pre_hook(hook))
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        prefill = _prefix_forward(model, adapter, input_ids, attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    bad = sorted(layer for layer, count in applications.items() if count != 1)
    if bad:
        raise RuntimeError(f"Block-input interventions did not apply once: {bad}")
    missing = sorted(set(capture_layers) - set(captures))
    if missing:
        raise RuntimeError(f"Block-input readout missed layers {missing}")
    return prefill, captures, applications, realized_norms


def candidate_metrics(
    candidate_log_scores: Mapping[int, float],
    *,
    target_count: int,
    original_count: int | None = None,
) -> dict[str, Any]:
    """Generalized candidate metrics for counts beyond the legacy 1--10 grid."""

    counts = np.asarray(
        sorted(int(value) for value in candidate_log_scores), dtype=int
    )
    if counts.size < 2 or len(set(counts.tolist())) != int(counts.size):
        raise ValueError("Candidate count registry must contain distinct values")
    target = int(target_count)
    if target not in set(counts.tolist()):
        raise ValueError(f"Target count {target} is outside the candidate registry")
    scores = np.asarray(
        [float(candidate_log_scores[int(value)]) for value in counts], dtype=float
    )
    if not np.isfinite(scores).all():
        raise ValueError("Candidate scores contain non-finite values")
    shifted = scores - float(scores.max())
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    target_index = int(np.flatnonzero(counts == target)[0])
    other_scores = np.delete(scores, target_index)
    probability_map = {
        str(int(count)): float(probability)
        for count, probability in zip(counts, probabilities)
    }
    score_map = {
        str(int(count)): float(score) for count, score in zip(counts, scores)
    }
    original_probability = None
    if original_count is not None:
        original = int(original_count)
        if original not in set(counts.tolist()):
            raise ValueError("Original count is outside the candidate registry")
        original_probability = probability_map[str(original)]
    return {
        "target_count": target,
        "original_count": None if original_count is None else int(original_count),
        "predicted_count_among_candidates": int(counts[int(scores.argmax())]),
        "target_is_candidate_argmax": bool(int(counts[int(scores.argmax())]) == target),
        "target_log_score": float(scores[target_index]),
        "target_margin": float(scores[target_index] - other_scores.max()),
        "target_probability": float(probabilities[target_index]),
        "original_probability": original_probability,
        "expected_count": float(np.sum(probabilities * counts.astype(float))),
        "candidate_counts": [int(value) for value in counts],
        "candidate_log_scores_by_count": score_map,
        "candidate_probabilities_by_count": probability_map,
    }


@torch.inference_mode()
def score_count_candidate_sequences(
    model: Any,
    encoding: NativeTraceEncoding,
    prefill_output: Any,
) -> Any:
    """Score a dynamic contiguous 1..N answer-candidate registry.

    The upstream V4.4.3 helper is deliberately frozen to exactly ten
    candidates.  Natural no-index traces have variable gold counts, while the
    mechanism-transfer plan registers a wider 1..18 evaluation range.  This
    local scorer preserves the upstream cache-branching and sequence scoring
    semantics without inheriting the unrelated ten-candidate assertion.
    """

    prefill_logits = _output_logits(prefill_output)[0, -1].detach().float()
    past = getattr(prefill_output, "past_key_values", None)
    if past is None:
        raise RuntimeError("Candidate scoring prefill returned no KV cache")
    candidates = sorted(
        (int(count), tuple(int(token) for token in tokens))
        for count, tokens in encoding.count_candidate_token_ids
    )
    counts = [count for count, _tokens in candidates]
    if len(counts) < 2 or counts != list(range(1, len(counts) + 1)):
        raise RuntimeError("Candidate counts must be a contiguous registry 1..N")
    if any(len(tokens) < 2 for _count, tokens in candidates):
        raise RuntimeError("Every candidate must include answer and termination tokens")
    if len({tokens for _count, tokens in candidates}) != len(candidates):
        raise RuntimeError("Candidate token sequences must be unique")

    max_inputs = max(len(tokens) - 1 for _count, tokens in candidates)
    device = prefill_logits.device
    positions = torch.arange(
        int(encoding.sequence_length),
        int(encoding.sequence_length) + max_inputs,
        dtype=torch.long,
        device=device,
    )
    shared = getattr(prefill_output, "shared_kv_states", None)
    first_log_probs = torch.log_softmax(prefill_logits, dim=-1)
    scores: dict[int, float] = {}
    # Branching all 18 continuations at once replicates a long-context KV cache
    # 18 times and can exceed an 80 GB H100.  Fixed four-way chunks preserve
    # exactly the same per-candidate scores while bounding peak cache memory.
    candidate_batch_size = 4
    for batch_start in range(0, len(candidates), candidate_batch_size):
        active = candidates[batch_start : batch_start + candidate_batch_size]
        batch = len(active)
        continuation_ids = torch.zeros(
            (batch, max_inputs), dtype=torch.long, device=device
        )
        continuation_mask = torch.zeros_like(continuation_ids)
        for row, (_count, tokens) in enumerate(active):
            inputs = tokens[:-1]
            continuation_ids[row, : len(inputs)] = torch.tensor(
                inputs, dtype=torch.long, device=device
            )
            continuation_mask[row, : len(inputs)] = 1
        active_past = copy.deepcopy(past)
        repeater = getattr(active_past, "batch_repeat_interleave", None)
        if not callable(repeater):
            raise RuntimeError("Transformers cache cannot branch candidate continuations")
        repeater(batch)
        base_mask = torch.tensor(
            [encoding.attention_mask], dtype=torch.long, device=device
        ).repeat(batch, 1)
        attention_mask = torch.cat((base_mask, continuation_mask), dim=1)
        kwargs: dict[str, Any] = {
            "input_ids": continuation_ids,
            "attention_mask": attention_mask,
            "past_key_values": active_past,
            "use_cache": False,
        }
        if _accepts_keyword(model, "position_ids"):
            kwargs["position_ids"] = positions.unsqueeze(0).expand(batch, -1)
        if _accepts_keyword(model, "cache_position"):
            kwargs["cache_position"] = positions
        if shared is not None and _accepts_keyword(model, "shared_kv_states"):
            kwargs["shared_kv_states"] = _repeat_batch_tree(shared, batch)
        continuation_output = model(**kwargs)
        continuation_logits = _output_logits(continuation_output).detach().float()
        if continuation_logits.shape[:2] != (batch, max_inputs):
            raise RuntimeError("Candidate continuation logits have the wrong shape")
        continuation_log_probs = torch.log_softmax(continuation_logits, dim=-1)
        for row, (count, tokens) in enumerate(active):
            score = first_log_probs[tokens[0]]
            for target_offset, token in enumerate(tokens[1:]):
                score = score + continuation_log_probs[row, target_offset, token]
            scores[count] = float(score.detach().cpu())
        del active_past, continuation_output, continuation_logits
        del continuation_log_probs, continuation_ids, continuation_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return SimpleNamespace(
        logits=prefill_logits.detach().float().cpu(),
        candidate_log_scores=scores,
    )


@torch.inference_mode()
def score_prefill(
    model: Any,
    tokenizer: Any,
    encoding: NativeTraceEncoding,
    prefill: Any,
    *,
    target_count: int,
    original_count: int | None,
    run_greedy: bool,
    max_new_tokens: int,
) -> dict[str, Any]:
    scoring = clone_prefill_output_for_scoring(prefill)
    output = score_count_candidate_sequences(model, encoding, scoring)
    result = candidate_metrics(
        output.candidate_log_scores,
        target_count=int(target_count),
        original_count=original_count,
    )
    if run_greedy:
        completion = generate_answer_completion_from_prefill(
            model,
            tokenizer,
            encoding,
            prefill,
            max_new_tokens=int(max_new_tokens),
        )
        greedy = completion_metrics(completion, gold_count=int(target_count))
        result.update({f"greedy_{key}": value for key, value in greedy.items()})
        result["greedy_target_adoption"] = bool(greedy["exact_count"])
    return result


def prefix_through_boundary(
    encoding: NativeTraceEncoding, boundary: int
) -> NativeTraceEncoding:
    """Return a teacher-forced prefix whose final token is ``boundary``."""

    end = int(boundary) + 1
    if not 0 < end <= int(encoding.sequence_length):
        raise ValueError("Carrier boundary is outside the encoding")
    return replace(
        encoding,
        input_ids=tuple(encoding.input_ids[:end]),
        attention_mask=tuple(encoding.attention_mask[:end]),
        query_position=end - 1,
    )


def item_candidate_tokens(
    encoding: NativeTraceEncoding, registry: Any
) -> dict[int, tuple[int, ...]]:
    """Extract receiver-native item strings used for successor scoring."""

    result = {
        occurrence: tuple(
            int(value) for value in encoding.input_ids[int(start) : int(end)]
        )
        for occurrence, (start, end) in enumerate(registry.trace_items, start=1)
    }
    expected = set(range(1, len(registry.trace_items) + 1))
    if len(expected) < 2 or set(result) != expected or any(
        len(tokens) < 1 for tokens in result.values()
    ):
        raise ValueError(
            "Native next-item candidates must cover every nonempty trace item"
        )
    if len(set(result.values())) != len(result):
        raise ValueError("Native next-item candidate token sequences are not unique")
    return result


def _clone_prefill_for_native_candidates(prefill: Any) -> Any:
    past = getattr(prefill, "past_key_values", None)
    if past is None:
        raise RuntimeError("Patched candidate prefill returned no KV cache")
    values: dict[str, Any] = {
        "logits": _output_logits(prefill),
        "past_key_values": copy.deepcopy(past),
    }
    shared = getattr(prefill, "shared_kv_states", None)
    if shared is not None:
        values["shared_kv_states"] = copy.deepcopy(shared)
    return type("CandidatePrefill", (), values)()


@torch.inference_mode()
def score_native_item_candidates(
    model: Any,
    prefix: NativeTraceEncoding,
    prefill: Any,
    candidates: Mapping[int, Sequence[int]],
    *,
    target: int,
    baseline: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Score all receiver-native item strings from one patched prefix cache."""

    branch = _clone_prefill_for_native_candidates(prefill)
    prefill_logits = _output_logits(branch)[0, -1].detach().float()
    past = branch.past_key_values
    candidate_count = len(candidates)
    if candidate_count < 2 or set(candidates) != set(range(1, candidate_count + 1)):
        raise ValueError("Native-item candidates must be contiguous occurrences 1..N")
    ordered = [
        (occurrence, tuple(int(value) for value in candidates[occurrence]))
        for occurrence in range(1, candidate_count + 1)
    ]
    max_inputs = max(len(tokens) - 1 for _occurrence, tokens in ordered)
    if max_inputs < 1:
        raise ValueError(
            "At least one native-item candidate must have a continuation token"
        )
    device = prefill_logits.device
    continuation_ids = torch.zeros(
        (candidate_count, max_inputs), dtype=torch.long, device=device
    )
    continuation_mask = torch.zeros_like(continuation_ids)
    for row, (_occurrence, tokens) in enumerate(ordered):
        values = tokens[:-1]
        continuation_ids[row, : len(values)] = torch.tensor(values, device=device)
        continuation_mask[row, : len(values)] = 1
    repeater = getattr(past, "batch_repeat_interleave", None)
    if not callable(repeater):
        raise RuntimeError("Transformers cache cannot branch native item candidates")
    repeater(candidate_count)
    base_mask = torch.tensor(
        [prefix.attention_mask], dtype=torch.long, device=device
    ).repeat(candidate_count, 1)
    attention_mask = torch.cat((base_mask, continuation_mask), dim=1)
    kwargs: dict[str, Any] = {
        "input_ids": continuation_ids,
        "attention_mask": attention_mask,
        "past_key_values": past,
        "use_cache": False,
    }
    positions = torch.arange(
        int(prefix.sequence_length),
        int(prefix.sequence_length) + max_inputs,
        dtype=torch.long,
        device=device,
    )
    if _accepts_keyword(model, "position_ids"):
        kwargs["position_ids"] = positions.unsqueeze(0).expand(
            candidate_count, -1
        )
    if _accepts_keyword(model, "cache_position"):
        kwargs["cache_position"] = positions
    shared = getattr(branch, "shared_kv_states", None)
    if shared is not None and _accepts_keyword(model, "shared_kv_states"):
        kwargs["shared_kv_states"] = _repeat_batch_tree(shared, candidate_count)
    continuation = model(**kwargs)
    logits = _output_logits(continuation).detach().float()
    first_log_probs = torch.log_softmax(prefill_logits, dim=-1)
    later_log_probs = torch.log_softmax(logits, dim=-1)
    sum_scores: list[float] = []
    mean_scores: list[float] = []
    for row, (_occurrence, tokens) in enumerate(ordered):
        score = first_log_probs[tokens[0]]
        for offset, token in enumerate(tokens[1:]):
            score = score + later_log_probs[row, offset, token]
        value = float(score.detach().cpu())
        sum_scores.append(value)
        mean_scores.append(value / len(tokens))
    target_index = int(target) - 1
    if not 0 <= target_index < candidate_count:
        raise ValueError("Native-item target is outside the receiver trace")
    other_mean = [
        value for index, value in enumerate(mean_scores) if index != target_index
    ]
    other_sum = [
        value for index, value in enumerate(sum_scores) if index != target_index
    ]
    mean_shifted = np.asarray(mean_scores) - float(max(mean_scores))
    probabilities = np.exp(mean_shifted)
    probabilities /= probabilities.sum()
    result = {
        "target_occurrence": int(target),
        "predicted_occurrence_mean_logprob": int(np.argmax(mean_scores)) + 1,
        "predicted_occurrence_sum_logprob": int(np.argmax(sum_scores)) + 1,
        "target_exact_mean_logprob": bool(
            int(np.argmax(mean_scores)) == target_index
        ),
        "target_exact_sum_logprob": bool(
            int(np.argmax(sum_scores)) == target_index
        ),
        "target_mean_logprob_margin": float(
            mean_scores[target_index] - max(other_mean)
        ),
        "target_sum_logprob_margin": float(
            sum_scores[target_index] - max(other_sum)
        ),
        "target_probability_mean_logprob": float(probabilities[target_index]),
        "mean_logprob_scores": [float(value) for value in mean_scores],
        "sum_logprob_scores": [float(value) for value in sum_scores],
        "candidate_token_counts": [len(tokens) for _key, tokens in ordered],
    }
    if baseline is not None:
        baseline_mean = np.asarray(baseline["mean_logprob_scores"], dtype=float)
        baseline_sum = np.asarray(baseline["sum_logprob_scores"], dtype=float)
        expected_shape = (candidate_count,)
        if baseline_mean.shape != expected_shape or baseline_sum.shape != expected_shape:
            raise ValueError(
                "Native-item baseline must match the dynamic candidate count"
            )
        delta_mean = np.asarray(mean_scores, dtype=float) - baseline_mean
        delta_sum = np.asarray(sum_scores, dtype=float) - baseline_sum
        other_delta_mean = np.delete(delta_mean, target_index)
        other_delta_sum = np.delete(delta_sum, target_index)
        delta_shifted = delta_mean - float(np.max(delta_mean))
        delta_probabilities = np.exp(delta_shifted)
        delta_probabilities /= delta_probabilities.sum()
        result.update(
            {
                "baseline_corrected": True,
                "delta_mean_logprob_scores": delta_mean.tolist(),
                "delta_sum_logprob_scores": delta_sum.tolist(),
                "predicted_occurrence_delta_mean_logprob": (
                    int(np.argmax(delta_mean)) + 1
                ),
                "predicted_occurrence_delta_sum_logprob": (
                    int(np.argmax(delta_sum)) + 1
                ),
                "target_exact_delta_mean_logprob": bool(
                    int(np.argmax(delta_mean)) == target_index
                ),
                "target_exact_delta_sum_logprob": bool(
                    int(np.argmax(delta_sum)) == target_index
                ),
                "target_delta_mean_logprob_margin": float(
                    delta_mean[target_index] - np.max(other_delta_mean)
                ),
                "target_delta_sum_logprob_margin": float(
                    delta_sum[target_index] - np.max(other_delta_sum)
                ),
                "target_probability_delta_mean_logprob": float(
                    delta_probabilities[target_index]
                ),
            }
        )
    return result


def paper_causal_influence(
    baseline: Mapping[str, Any],
    patched: Mapping[str, Any],
    *,
    expected_count: int,
    original_count: int,
) -> float:
    """Compute the paper's CI, including its explicit one-half factor."""

    expected_key = str(int(expected_count))
    original_key = str(int(original_count))
    baseline_prob = baseline["candidate_probabilities_by_count"]
    patched_prob = patched["candidate_probabilities_by_count"]
    return 0.5 * (
        (float(patched_prob[expected_key]) - float(baseline_prob[expected_key]))
        + (float(baseline_prob[original_key]) - float(patched_prob[original_key]))
    )


def continued_count_expected(
    source_end_count: int,
    target_item_count: int,
    k: int,
) -> int:
    source = int(source_end_count)
    target = int(target_item_count)
    width = int(k)
    if source < width or target < width or width < 1:
        raise ValueError("Invalid continued-counting source/target/k geometry")
    return source + target - width


def maximum_latent_count_expected(
    source_end_count: int,
    target_end_count: int,
    k: int,
) -> int:
    source = int(source_end_count)
    target = int(target_end_count)
    width = int(k)
    if source < width or target < width or width < 1:
        raise ValueError("Invalid maximum-count source/target/k geometry")
    return max(source, target - width)


def prompt_scrubbed_encoding(
    clean: NativeTraceEncoding,
    registry: Any,
    tokenizer: Any,
    *,
    random_seed: int,
) -> tuple[NativeTraceEncoding, dict[str, Any]]:
    """Scrub prompt records while restoring every teacher-forced trace token."""

    blank, audit = build_fully_uninformative_encoding(
        clean, registry, tokenizer, random_seed=int(random_seed)
    )
    values = list(int(value) for value in blank.input_ids)
    trace_positions = tuple(int(value) for value in registry.positions("trace_items"))
    for position in trace_positions:
        values[position] = int(clean.input_ids[position])
    result = replace(blank, input_ids=tuple(values))
    prompt_positions = tuple(int(value) for value in registry.positions("prompt_records"))
    changed_prompt = sum(
        int(result.input_ids[position]) != int(clean.input_ids[position])
        for position in prompt_positions
    )
    if changed_prompt <= 0:
        raise RuntimeError("Prompt-scrubbed target changed no prompt-record tokens")
    if any(
        result.input_ids[position] != clean.input_ids[position]
        for position in trace_positions
    ):
        raise RuntimeError("Prompt scrub changed a trace token")
    return result, {
        **audit,
        "target_context": "prompt_records_scrubbed_trace_preserved",
        "prompt_record_changed_token_count": int(changed_prompt),
        "trace_tokens_restored_exactly": True,
        "target_input_ids_sha256": _sha256_jsonable(result.input_ids),
    }


def build_immediate_count_query_encoding(
    encoding: NativeTraceEncoding,
    registry: Any,
    tokenizer: Any,
    *,
    target_occurrence: int,
) -> tuple[NativeTraceEncoding, dict[str, Any]]:
    """Query the count immediately after event ``k`` with a minimal suffix.

    Natural traces can contain a post-list recap such as "there are eight
    cities" before the original ``Total:`` query.  Reusing that terminal
    suffix would leak the answer into the causal readout.  This constructor
    keeps the exact prefix through event ``k`` and appends only the literal
    reasoning-channel close and answer-query grammar.
    """

    occurrence = int(target_occurrence)
    items = tuple((int(start), int(end)) for start, end in registry.trace_items)
    if not 1 <= occurrence <= len(items):
        raise ValueError("Immediate-query occurrence is outside the trace")
    item_start, item_end = items[occurrence - 1]
    terminal_end = int(items[-1][1])
    query = int(registry.query_position)
    if not (
        int(encoding.prompt_token_count)
        <= item_start
        < item_end
        <= terminal_end
        <= query
        < int(encoding.sequence_length)
    ):
        raise ValueError("Immediate-query item/terminal/query ordering is invalid")

    suffix_text = "\n</think>\n\nTotal:"
    suffix_ids = tuple(
        int(value)
        for value in tokenizer.encode(suffix_text, add_special_tokens=False)
    )
    if not suffix_ids:
        raise RuntimeError("Minimal immediate-query suffix tokenized to nothing")
    decoded_suffix = tokenizer.decode(
        list(suffix_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if any(character.isdigit() for character in decoded_suffix):
        raise RuntimeError("Minimal immediate-query suffix contains a count digit")
    count_words = set(
        "zero one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen seventeen eighteen".split()
    )
    normalized_words = {
        value.strip(".,:;!?()[]{}<>").lower() for value in decoded_suffix.split()
    }
    if count_words & normalized_words:
        raise RuntimeError("Minimal immediate-query suffix contains a count word")

    original_ids = tuple(int(value) for value in encoding.input_ids)
    original_mask = tuple(int(value) for value in encoding.attention_mask)
    immediate_ids = original_ids[:item_end] + suffix_ids
    immediate_mask = original_mask[:item_end] + (1,) * len(suffix_ids)
    visible_items = tuple(encoding.trace_item_spans[:occurrence])
    result = replace(
        encoding,
        input_ids=immediate_ids,
        attention_mask=immediate_mask,
        query_position=len(immediate_ids) - 1,
        trace_item_spans=visible_items,
        slot_spans=visible_items,
        needle_spans=visible_items,
    )
    if result.input_ids[:item_end] != original_ids[:item_end]:
        raise RuntimeError("Immediate query changed the causal event prefix")
    return result, {
        "early_stop_target_occurrence": occurrence,
        "early_stop_item_start": item_start,
        "early_stop_item_end": item_end,
        "early_stop_original_terminal_end": terminal_end,
        "early_stop_original_query_position": query,
        "early_stop_query_position": int(result.query_position),
        "early_stop_suffix_mode": "minimal_literal_reasoning_close_and_total_query",
        "early_stop_suffix_token_count": len(suffix_ids),
        "early_stop_suffix_sha256": _sha256_jsonable(suffix_ids),
        "post_event_original_suffix_tokens_retained": 0,
        "post_event_original_tokens_removed": query + 1 - item_end,
        "future_trace_token_count_removed": terminal_end - item_end,
        "future_trace_items_removed": len(items) - occurrence,
        "future_trace_tokens_present": False,
        "item_positions_unchanged": True,
        "terminal_suffix_contains_candidate_digit": False,
        "terminal_suffix_contains_candidate_word": False,
        "outcome_fields_accessed": False,
    }


def countscope_blank_encoding(
    clean: NativeTraceEncoding,
    registry: Any,
    tokenizer: Any,
    *,
    receiver_occurrence: int,
    random_seed: int,
) -> tuple[NativeTraceEncoding, dict[str, Any]]:
    """Build a one-prefix CountScope receiver with structure but no semantics."""

    blank, audit = build_fully_uninformative_encoding(
        clean, registry, tokenizer, random_seed=int(random_seed)
    )
    values = list(int(value) for value in blank.input_ids)
    marker_positions = set(int(value) for value in registry.positions("trace_markers"))
    structural: list[int] = []
    for position in registry.positions("trace_items"):
        token_id = int(clean.input_ids[int(position)])
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if int(position) in marker_positions or not any(
            character.isalnum() for character in decoded
        ):
            values[int(position)] = token_id
            structural.append(int(position))
    structured = replace(blank, input_ids=tuple(values))
    receiver, early_audit = build_immediate_count_query_encoding(
        structured,
        registry,
        tokenizer,
        target_occurrence=int(receiver_occurrence),
    )
    visible_end = int(registry.trace_items[int(receiver_occurrence) - 1][1])
    semantic_positions = [
        position
        for position in range(int(clean.prompt_token_count), visible_end)
        if position not in set(structural)
    ]
    if not semantic_positions:
        raise RuntimeError("CountScope receiver exposes no blank semantic tokens")
    unchanged_semantics = sum(
        int(receiver.input_ids[position]) == int(clean.input_ids[position])
        for position in semantic_positions
    )
    if unchanged_semantics == len(semantic_positions):
        raise RuntimeError("CountScope receiver failed to blank item semantics")
    return receiver, {
        **audit,
        **early_audit,
        "target_context": "countscope_structural_single_prefix_receiver",
        "receiver_occurrence": int(receiver_occurrence),
        "prompt_records_scrubbed": True,
        "item_alphanumeric_semantics_scrubbed": True,
        "structural_trace_positions_restored": len(structural),
        "visible_semantic_position_count": len(semantic_positions),
        "unchanged_visible_semantic_positions": int(unchanged_semantics),
        "target_input_ids_sha256": _sha256_jsonable(receiver.input_ids),
    }


def norm_matched_orthogonal_control(
    delta: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Sample one deterministic delta orthogonal to the real position delta."""

    value = torch.as_tensor(delta).detach().float().cpu()
    if value.ndim != 2 or value.shape[0] < 1:
        raise ValueError("Geometry delta must have shape [positions, hidden]")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    random = torch.randn(value.shape, generator=generator, dtype=torch.float32)
    coefficient = torch.sum(random * value, dim=-1, keepdim=True) / torch.clamp(
        torch.sum(value * value, dim=-1, keepdim=True), min=1e-12
    )
    random = random - coefficient * value
    random_norm = torch.linalg.vector_norm(random, dim=-1, keepdim=True)
    target_norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    if torch.any(random_norm <= 1e-10):
        raise RuntimeError("Could not construct an orthogonal geometry control")
    return random * (target_norm / random_norm)


def layerwise_centroid_deltas(
    centroids: Mapping[int, Mapping[int, torch.Tensor]],
    *,
    receiver_occurrence: int,
    target_occurrence: int,
) -> dict[int, torch.Tensor]:
    receiver = int(receiver_occurrence)
    target = int(target_occurrence)
    result: dict[int, torch.Tensor] = {}
    for layer, by_occurrence in centroids.items():
        if receiver not in by_occurrence or target not in by_occurrence:
            raise ValueError("Geometry centroids lack a requested occurrence")
        left = torch.as_tensor(by_occurrence[receiver]).detach().float().cpu()
        right = torch.as_tensor(by_occurrence[target]).detach().float().cpu()
        if left.shape != right.shape:
            raise ValueError("Geometry centroid shapes disagree")
        if left.ndim == 1:
            left = left.unsqueeze(0)
            right = right.unsqueeze(0)
        result[int(layer)] = right - left
    return result
