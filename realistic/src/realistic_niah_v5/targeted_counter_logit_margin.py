"""Direct count-logit margins after one frozen retrieval-query head mask.

This assay complements the centroid-based NCC readout with two output-space
estimands:

* both timing branches score the complete native answer candidates 1..10 at
  the registered final answer query;
* ``rank_after_city`` additionally branches at the marker-free pre-marker
  prefix and scores the complete final rank marker N against the immediately
  preceding same-grammar marker N-1.

The head mask is applied only at the frozen N-1 -> N retrieval query during
the teacher-forced prefill.  Candidate answer tokens run without hooks and
therefore test persistence of that exact lesion rather than continued damage.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace
from typing import Any, Mapping, Sequence

import torch

from realistic_niah_v4.modeling import (
    DecoderAdapter,
    _accepts_keyword,
    _encoding_tensors,
)
from realistic_niah_v4_4_3.interventions import _score_candidate_sequences

from .causal_sites import compile_causal_site_plan
from .count_stream import (
    _score_and_generate_prefill,
    _uses_shared_kv,
    build_answer_source_registry,
)
from .integrated_bridge import _final_post_marker_position, _validated_heads
from .stratified_targeted_counter_ncc import grammar_timing
from .targeted_counter_ncc import NCC_CONDITIONS, _normalize_banks
from .terminal_token_state import _site_positions


LOGIT_MARGIN_ENDPOINTS: dict[str, tuple[str, ...]] = {
    "rank_after_city": (
        "final_answer_sequence_margin",
        "local_rank_adjacent_sequence_margin",
    ),
    "rank_before_city": ("final_answer_sequence_margin",),
}


def _first_divergence(
    correct_ids: Sequence[int], rival_ids: Sequence[int]
) -> int | None:
    """Return the first unequal token index, or None for a prefix relation."""

    correct = tuple(int(value) for value in correct_ids)
    rival = tuple(int(value) for value in rival_ids)
    for index, (left, right) in enumerate(zip(correct, rival)):
        if left != right:
            return int(index)
    return None


def _local_rank_contrast(
    encoding: Any,
    tokenizer: Any,
    events: Sequence[Mapping[str, Any]],
    *,
    timing: str,
    targeted_query: int,
) -> dict[str, Any]:
    """Compile the N-vs-(N-1) first-discriminative marker-token contrast."""

    if timing != "rank_after_city":
        return {"available": False, "reason": "rank_marker_precedes_targeted_query"}
    gold = int(encoding.count)
    if gold < 2 or len(events) != gold:
        raise ValueError("Local rank contrast requires a complete count >= 2 trace")
    correct_event = events[-1]
    rival_event = events[-2]
    correct_class = str(correct_event.get("grammar_class", ""))
    rival_class = str(rival_event.get("grammar_class", ""))
    if correct_class != rival_class:
        return {
            "available": False,
            "reason": "adjacent_previous_marker_uses_different_grammar_class",
            "correct_grammar_class": correct_class,
            "rival_grammar_class": rival_class,
        }
    correct_positions = _site_positions(
        correct_event.get("sites", {}).get("rank_evidence_core_span"),
        role="rank_evidence_core_span",
    )
    rival_positions = _site_positions(
        rival_event.get("sites", {}).get("rank_evidence_core_span"),
        role="rank_evidence_core_span",
    )
    correct_ids = tuple(int(encoding.input_ids[position]) for position in correct_positions)
    rival_ids = tuple(int(encoding.input_ids[position]) for position in rival_positions)
    divergence = _first_divergence(correct_ids, rival_ids)
    if divergence is None:
        return {
            "available": False,
            "reason": "marker_token_sequences_equal_or_prefix_related",
            "correct_token_count": len(correct_ids),
            "rival_token_count": len(rival_ids),
        }
    prefix_position = int(correct_positions[0]) - 1
    if prefix_position <= int(targeted_query):
        raise ValueError("Local rank margin is not downstream of the targeted query")
    correct_token = int(correct_ids[divergence])
    rival_token = int(rival_ids[divergence])
    if correct_token == rival_token:
        raise RuntimeError("Local rank discriminative tokens unexpectedly agree")
    shared_ids = correct_ids[:divergence]
    return {
        "available": True,
        "correct_count": gold,
        "rival_count": gold - 1,
        "grammar_class": correct_class,
        "prefix_position": prefix_position,
        "first_divergence_index": int(divergence),
        "shared_prefix_token_ids": list(shared_ids),
        "shared_prefix_text": tokenizer.decode(
            list(shared_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "correct_token_id": correct_token,
        "correct_token_text": tokenizer.decode(
            [correct_token],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "rival_token_id": rival_token,
        "rival_token_text": tokenizer.decode(
            [rival_token],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "correct_marker_text": tokenizer.decode(
            list(correct_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "rival_marker_text": tokenizer.decode(
            list(rival_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "correct_marker_token_ids": list(correct_ids),
        "rival_marker_token_ids": list(rival_ids),
        "marker_candidates_not_teacher_forced_during_scoring": True,
    }


def _validate_factorial_banks(banks: Sequence[Mapping[str, Any]]) -> None:
    selected = next(row for row in banks if row["condition"] == "selected_bank")
    clean = next(row for row in banks if row["condition"] == "clean")
    if clean["heads"]:
        raise ValueError("Clean logit-margin condition unexpectedly contains heads")
    selected_heads = tuple((int(a), int(b)) for a, b in selected["heads"])
    selected_layers = Counter(layer for layer, _head in selected_heads)
    selected_set = set(selected_heads)
    randoms = [row for row in banks if row["condition"] == "layer_matched_random"]
    if [int(row["repeat"]) for row in randoms] != [1, 2, 3]:
        raise ValueError("Logit-margin random-bank repeats changed")
    for row in randoms:
        heads = tuple((int(a), int(b)) for a, b in row["heads"])
        if Counter(layer for layer, _head in heads) != selected_layers:
            raise ValueError("Logit-margin random bank is not exactly layer matched")
        if selected_set & set(heads):
            raise ValueError("Logit-margin random bank overlaps the selected bank")


@torch.inference_mode()
def _prefill_with_query_head_ablation(
    model: Any,
    adapter: DecoderAdapter,
    encoding: Any,
    *,
    heads: Sequence[tuple[int, int]],
    hook_position: int,
    score_positions: Sequence[int],
) -> tuple[Any, dict[int, torch.Tensor], dict[str, Any]]:
    """Run one bounded-logit cached prefill with an audited exact-query mask."""

    expected_length = int(encoding.sequence_length)
    scores = tuple(sorted({int(value) for value in score_positions}))
    if not scores or scores[0] < 0 or scores[-1] >= expected_length:
        raise ValueError("A logit-margin score position is outside the prefill")
    if not 0 <= int(hook_position) < expected_length:
        raise ValueError("The targeted retrieval query is outside the prefill")
    if min(scores) <= int(hook_position):
        raise ValueError("Every logit-margin endpoint must follow the targeted query")
    if not _accepts_keyword(model, "logits_to_keep"):
        raise RuntimeError("Bounded suffix logits are required for this long-context assay")

    by_layer = _validated_heads(adapter, heads)
    applications = {layer: 0 for layer in by_layer}
    maxima = {layer: 0.0 for layer in by_layer}
    handles = []
    for layer, layer_heads in by_layer.items():
        head_dim = int(adapter.head_dims[layer])
        expected_width = int(adapter.num_heads[layer]) * head_dim

        def hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
            layer_heads: tuple[int, ...] = layer_heads,
            head_dim: int = head_dim,
            expected_width: int = expected_width,
        ) -> tuple[Any, ...] | None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Logit-margin hook saw no attention-head tensor")
            value = args[0]
            if value.ndim != 3 or int(value.shape[-1]) != expected_width:
                raise RuntimeError("Logit-margin pre-o tensor shape changed")
            if int(value.shape[1]) != expected_length:
                return None
            if applications[layer] != 0:
                raise RuntimeError("Logit-margin head hook applied more than once")
            patched = value.clone()
            for head in layer_heads:
                left = int(head) * head_dim
                patched[:, int(hook_position), left : left + head_dim] = 0
            selected = torch.cat(
                [
                    patched[
                        :, int(hook_position), int(head) * head_dim : (int(head) + 1) * head_dim
                    ].detach().reshape(-1)
                    for head in layer_heads
                ]
            )
            maxima[layer] = float(selected.abs().max().float().cpu())
            applications[layer] += 1
            return (patched, *args[1:])

        handles.append(adapter.output_projections[layer].register_forward_pre_hook(hook))

    input_ids, attention_mask = _encoding_tensors(model, encoding)
    keep = expected_length - int(min(scores))
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": True,
        "output_attentions": False,
        "logits_to_keep": int(keep),
    }
    if _uses_shared_kv(adapter):
        kwargs["return_shared_kv_states"] = True
    try:
        output = model(**kwargs)
    finally:
        for handle in handles:
            handle.remove()

    bad = {layer: count for layer, count in applications.items() if count != 1}
    if bad:
        raise RuntimeError(f"Logit-margin head ablation did not apply once: {bad}")
    if any(value != 0.0 for value in maxima.values()):
        raise RuntimeError("Logit-margin selected head slices were not exactly zero")
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise RuntimeError("Logit-margin prefill returned no sequence logits")
    if int(logits.shape[1]) == expected_length:
        offset = 0
    elif int(logits.shape[1]) == keep:
        offset = expected_length - keep
    else:
        raise RuntimeError(
            f"Unexpected bounded-logit length {logits.shape[1]} (expected {keep})"
        )
    selected_logits = {
        position: logits[0, position - offset].detach().float().cpu()
        for position in scores
    }
    return output, selected_logits, {
        "head_ablation_layer_applications": {
            str(layer): int(count) for layer, count in sorted(applications.items())
        },
        "head_ablation_selected_post_zero_max_abs": (
            max(maxima.values()) if maxima else 0.0
        ),
        "head_ablation_position": int(hook_position),
        "bounded_logits_to_keep": int(keep),
        "score_positions": list(scores),
    }


def _score_map(outcomes: Mapping[str, Any]) -> dict[int, float]:
    counts = [int(value) for value in str(outcomes["candidate_counts"]).split(",")]
    scores = [float(value) for value in str(outcomes["candidate_log_scores"]).split(",")]
    if counts != list(range(1, 11)) or len(scores) != 10:
        raise ValueError("Answer candidates must cover counts 1..10")
    if not all(math.isfinite(value) for value in scores):
        raise ValueError("Answer candidate log scores must be finite")
    return dict(zip(counts, scores))


@torch.inference_mode()
def run_targeted_counter_logit_margin(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    banks: Sequence[Mapping[str, Any]],
    targeted_site: Mapping[str, Any],
    timing: str,
    selected_bank_size: int,
    answer_site_id: str = "answer_query_v3",
) -> dict[str, Any]:
    """Score direct count margins under clean/selected/three-random conditions."""

    if timing not in LOGIT_MARGIN_ENDPOINTS:
        raise ValueError(f"Unknown logit-margin timing branch: {timing}")
    encoding, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    gold_count = int(encoding.count)
    if gold_count < 2 or len(registry.trace_items) != gold_count:
        raise ValueError("Logit margin requires a complete trace with count >= 2")
    targeted_query, specification = _final_post_marker_position(
        row, gold_count=gold_count, targeted_site=targeted_site
    )
    plan = compile_causal_site_plan(row, tokenizer)
    events = list(plan.get("events", ()))
    if len(events) != gold_count or grammar_timing(events[-1]) != timing:
        raise ValueError("Logit-margin final grammar event changed timing")
    if int(encoding.query_position) <= int(targeted_query):
        raise ValueError("Final answer query must follow the targeted retrieval query")

    local = _local_rank_contrast(
        encoding,
        tokenizer,
        events,
        timing=timing,
        targeted_query=targeted_query,
    )
    score_positions = [int(encoding.query_position)]

    normalized = _normalize_banks(
        adapter, banks, selected_size=int(selected_bank_size)
    )
    _validate_factorial_banks(normalized)
    maximum_head_layer = max(
        int(layer) for bank in normalized for layer, _head in bank["heads"]
    )
    if maximum_head_layer >= int(adapter.num_layers) - 1:
        raise ValueError("The complete bank leaves no later layer for causal propagation")

    candidate_lengths = {
        int(count): len(tuple(int(value) for value in ids))
        for count, ids in encoding.count_candidate_answer_token_ids
    }
    if set(candidate_lengths) != set(range(1, 11)) or min(candidate_lengths.values()) < 1:
        raise ValueError("Answer candidate token registry changed")

    condition_results: list[dict[str, Any]] = []
    for bank in normalized:
        condition = (
            "clean"
            if bank["condition"] == "clean"
            else "selected_mask"
            if bank["condition"] == "selected_bank"
            else f"random_mask_r{bank['repeat']}"
        )
        prefill, _answer_logits, audit = _prefill_with_query_head_ablation(
            model,
            adapter,
            encoding,
            heads=bank["heads"],
            hook_position=int(targeted_query),
            score_positions=score_positions,
        )
        outcomes = _score_and_generate_prefill(
            model,
            tokenizer,
            encoding,
            prefill,
            run_greedy=False,
            max_new_tokens=1,
        )
        scores = _score_map(outcomes)
        normalized_scores = {
            count: float(value) / int(candidate_lengths[count])
            for count, value in scores.items()
        }
        wrong_normalized = [
            value for count, value in normalized_scores.items() if count != gold_count
        ]
        local_margin: float | None = None
        local_probability: float | None = None
        local_length_normalized_margin: float | None = None
        local_audit: dict[str, Any] | None = None
        if local["available"]:
            prefix_position = int(local["prefix_position"])
            correct_marker_ids = tuple(
                int(value) for value in local["correct_marker_token_ids"]
            )
            rival_marker_ids = tuple(
                int(value) for value in local["rival_marker_token_ids"]
            )
            local_encoding = replace(
                encoding,
                input_ids=tuple(encoding.input_ids[: prefix_position + 1]),
                attention_mask=tuple(encoding.attention_mask[: prefix_position + 1]),
                query_position=prefix_position,
                count_candidate_answer_token_ids=(
                    (gold_count, correct_marker_ids),
                    (gold_count - 1, rival_marker_ids),
                ),
            )
            local_prefill, _local_logits, local_audit = (
                _prefill_with_query_head_ablation(
                    model,
                    adapter,
                    local_encoding,
                    heads=bank["heads"],
                    hook_position=int(targeted_query),
                    score_positions=(prefix_position,),
                )
            )
            marker_scores = _score_candidate_sequences(
                model, local_encoding, local_prefill
            ).candidate_log_scores
            local_margin = float(
                marker_scores[gold_count] - marker_scores[gold_count - 1]
            )
            local_probability = float(torch.sigmoid(torch.tensor(local_margin)))
            local_length_normalized_margin = float(
                marker_scores[gold_count] / len(correct_marker_ids)
                - marker_scores[gold_count - 1] / len(rival_marker_ids)
            )
        condition_results.append(
            {
                "condition": condition,
                "receiver_bank_condition": bank["condition"],
                "receiver_bank_repeat": int(bank["repeat"]),
                "receiver_bank_sha256": bank["bank_sha256"],
                "receiver_head_count": len(bank["heads"]),
                "predicted_count_among_candidates": int(
                    outcomes["predicted_count_among_candidates"]
                ),
                "correct_count_log_score": float(outcomes["correct_count_log_score"]),
                "correct_count_margin": float(outcomes["correct_count_margin"]),
                "correct_count_probability": float(outcomes["correct_count_probability"]),
                "expected_count": float(outcomes["expected_count"]),
                "candidate_counts": str(outcomes["candidate_counts"]),
                "candidate_log_scores": str(outcomes["candidate_log_scores"]),
                "candidate_probabilities": str(outcomes["candidate_probabilities"]),
                "length_normalized_correct_count_margin": float(
                    normalized_scores[gold_count] - max(wrong_normalized)
                ),
                "adjacent_answer_count_margin": float(
                    scores[gold_count] - scores[gold_count - 1]
                ),
                "local_rank_adjacent_sequence_margin": local_margin,
                "local_rank_adjacent_sequence_probability": local_probability,
                "local_rank_adjacent_length_normalized_margin": (
                    local_length_normalized_margin
                ),
                "local_rank_prefill_hook_audit": local_audit,
                **audit,
            }
        )

    observed = tuple(row["condition"] for row in condition_results)
    if observed != NCC_CONDITIONS:
        raise RuntimeError(f"Logit-margin condition order changed: {observed}")
    return {
        "schema_version": "realistic_niah_v5_targeted_counter_logit_margin_capture_v2",
        "experiment_id": "targeted_retrieval_query_to_direct_count_logit_margin",
        "request_id": str(encoding.request_id),
        "model_label": str(encoding.model_label),
        "seed": int(encoding.seed),
        "dataset_split": str(encoding.split),
        "gold_count": gold_count,
        "timing_branch": timing,
        "targeted_query_position": int(targeted_query),
        "targeted_from_occurrence": int(specification["from_occurrence"]),
        "targeted_to_occurrence": int(specification["to_occurrence"]),
        "targeted_anchor_equivalence_id": str(specification["anchor_equivalence_id"]),
        "answer_query_position": int(encoding.query_position),
        "answer_query_is_downstream_of_targeted_query": True,
        "answer_candidate_token_lengths": candidate_lengths,
        "primary_endpoint": "final_answer_sequence_margin",
        "endpoint_names": list(LOGIT_MARGIN_ENDPOINTS[timing]),
        "local_rank_contrast": local,
        "conditions": condition_results,
        "mask_scope": "exact_targeted_retrieval_query_prefill_only",
        "candidate_answer_tokens_run_without_head_hooks": True,
        "candidate_answer_scoring": "full_autoregressive_sequence_log_probability_1_to_10",
        "teacher_forced_trace_tokens": True,
        "outcome_blind_panel": True,
        "selection_rank_used": False,
        "no_decoder_fit_or_layer_selection": True,
        "maximum_ablated_head_layer": maximum_head_layer,
        "later_propagation_layer_exists": True,
        "confirmation_used_for_registration": False,
        "causal_claim_scope": "targeted_query_head_mask_to_direct_count_output_margin",
        "registry_sha256": registry.to_dict()["registry_sha256"],
        "causal_site_plan_schema_version": plan["schema_version"],
    }
