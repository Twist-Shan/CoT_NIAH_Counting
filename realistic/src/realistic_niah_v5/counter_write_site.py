"""Exploratory localization of a native trace counter write site.

This module deliberately treats grammar boundaries as competing hypotheses.
It does not assume that ``pre_marker`` (or any other parser site) is where a
counter is written.  For one frozen teacher-forced trajectory it:

1. captures full residual-stream states at the same grammar site for several
   running-count occurrences;
2. stops a receiver trajectory immediately at that site and appends only the
   model's native channel-closing/``Total:`` bridge;
3. replaces the receiver state with a donor occurrence at one or more layers;
4. scores all answer counts 1..10.

An eligible write site must move the answer distribution toward both earlier
and later donors.  A one-sided or surface-token-only effect is diagnostic, not
evidence that the counter is written there.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import Any, Mapping, Sequence

import torch

from realistic_niah_v4.modeling import DecoderAdapter, capture_post_block_states

from .causal_sites import build_output_token_map
from .count_stream import (
    _prefill_with_layerwise_state_replacements,
    _score_and_generate_prefill,
    build_answer_source_registry,
)
from .encoding import NativeTraceEncoding
from .parsing import (
    align_trace_sites,
    find_trace_count_sequence,
    gold_records,
    infer_model_family,
    output_token_ids,
    raw_output_text,
    trace_char_sites,
)


COUNTER_WRITE_SITE_SCHEMA_VERSION = "realistic_niah_v5_counter_write_site_pilot_v1"
REGISTERED_COUNTER_WRITE_SITES = (
    "pre_city",
    "city_end",
    "pre_marker",
    "marker_end",
    "item_end",
)


def _candidate_score_map(outcomes: Mapping[str, Any]) -> dict[int, float]:
    counts = [int(value) for value in str(outcomes["candidate_counts"]).split(",")]
    scores = [
        float(value) for value in str(outcomes["candidate_log_scores"]).split(",")
    ]
    if counts != list(range(1, 11)) or len(scores) != len(counts):
        raise ValueError("Counter write-site outcomes must score counts 1..10")
    if not all(math.isfinite(value) for value in scores):
        raise ValueError("Counter write-site candidate scores must be finite")
    return dict(zip(counts, scores))


def directional_count_metrics(
    baseline: Mapping[str, Any],
    patched: Mapping[str, Any],
    *,
    receiver_progress: int,
    donor_progress: int,
) -> dict[str, float | int | bool]:
    """Measure whether a patch moves the count distribution toward its donor."""

    receiver = int(receiver_progress)
    donor = int(donor_progress)
    if not 1 <= receiver <= 10 or not 1 <= donor <= 10 or receiver == donor:
        raise ValueError("Receiver/donor progress must be distinct counts in 1..10")
    baseline_scores = _candidate_score_map(baseline)
    patched_scores = _candidate_score_map(patched)
    before_log_odds = baseline_scores[donor] - baseline_scores[receiver]
    after_log_odds = patched_scores[donor] - patched_scores[receiver]
    direction = 1 if donor > receiver else -1
    expected_shift = float(patched["expected_count"]) - float(
        baseline["expected_count"]
    )
    return {
        "receiver_progress_count": receiver,
        "donor_progress_count": donor,
        "donor_direction": direction,
        "baseline_donor_vs_receiver_log_odds": float(before_log_odds),
        "patched_donor_vs_receiver_log_odds": float(after_log_odds),
        "donor_vs_receiver_log_odds_effect": float(
            after_log_odds - before_log_odds
        ),
        "expected_count_shift": float(expected_shift),
        "donor_aligned_expected_count_shift": float(direction * expected_shift),
        "moves_expected_count_toward_donor": bool(direction * expected_shift > 0),
    }


def target_count_restoration_metrics(
    baseline: Mapping[str, Any],
    patched: Mapping[str, Any],
    *,
    target_progress: int,
) -> dict[str, float | int | bool]:
    """Measure restoration of one registered progress value in a blank receiver."""

    target = int(target_progress)
    if not 1 <= target <= 10:
        raise ValueError("Target progress must be a count in 1..10")
    baseline_scores = _candidate_score_map(baseline)
    patched_scores = _candidate_score_map(patched)

    def metrics(scores: Mapping[int, float]) -> tuple[float, float]:
        ordered = [float(scores[count]) for count in range(1, 11)]
        maximum = max(ordered)
        weights = [math.exp(value - maximum) for value in ordered]
        normalizer = sum(weights)
        probability = weights[target - 1] / normalizer
        margin = float(scores[target]) - max(
            float(value) for count, value in scores.items() if int(count) != target
        )
        return probability, margin

    baseline_probability, baseline_margin = metrics(baseline_scores)
    patched_probability, patched_margin = metrics(patched_scores)
    baseline_distance = abs(float(baseline["expected_count"]) - target)
    patched_distance = abs(float(patched["expected_count"]) - target)
    return {
        "target_progress_count": target,
        "baseline_target_probability": float(baseline_probability),
        "patched_target_probability": float(patched_probability),
        "target_probability_effect": float(
            patched_probability - baseline_probability
        ),
        "baseline_target_margin": float(baseline_margin),
        "patched_target_margin": float(patched_margin),
        "target_margin_effect": float(patched_margin - baseline_margin),
        "baseline_expected_distance_to_target": float(baseline_distance),
        "patched_expected_distance_to_target": float(patched_distance),
        "expected_distance_improvement": float(
            baseline_distance - patched_distance
        ),
        "target_probability_increased": bool(
            patched_probability > baseline_probability
        ),
        "target_margin_increased": bool(patched_margin > baseline_margin),
    }


def build_site_early_stop_encoding(
    encoding: NativeTraceEncoding,
    *,
    cut_position: int,
    terminal_suffix_start: int,
    site_id: str,
) -> tuple[NativeTraceEncoding, dict[str, Any]]:
    """Cut after one site and append only the native terminal answer bridge."""

    cut = int(cut_position)
    suffix_start = int(terminal_suffix_start)
    query = int(encoding.query_position)
    if not (
        int(encoding.prompt_token_count) <= cut < suffix_start <= query
    ):
        raise ValueError("Site cut, terminal suffix, and answer query are misordered")
    original_ids = tuple(int(value) for value in encoding.input_ids)
    original_mask = tuple(int(value) for value in encoding.attention_mask)
    cut_end = cut + 1
    suffix_end = query + 1
    early_ids = original_ids[:cut_end] + original_ids[suffix_start:suffix_end]
    early_mask = original_mask[:cut_end] + original_mask[suffix_start:suffix_end]
    new_query = len(early_ids) - 1
    if early_ids[:cut_end] != original_ids[:cut_end]:
        raise RuntimeError("Early-stop construction changed the causal receiver prefix")
    if early_ids[cut_end:] != original_ids[suffix_start:suffix_end]:
        raise RuntimeError("Early-stop construction changed the terminal bridge")
    result = replace(
        encoding,
        input_ids=early_ids,
        attention_mask=early_mask,
        query_position=new_query,
        trace_item_spans=tuple(
            span for span in encoding.trace_item_spans if int(span.end) <= cut_end
        ),
        slot_spans=tuple(
            span for span in encoding.slot_spans if int(span.end) <= cut_end
        ),
        needle_spans=tuple(
            span for span in encoding.needle_spans if int(span.end) <= cut_end
        ),
    )
    return result, {
        "early_stop_site_id": str(site_id),
        "early_stop_cut_position": cut,
        "early_stop_original_query_position": query,
        "early_stop_query_position": new_query,
        "terminal_suffix_original_start": suffix_start,
        "terminal_suffix_token_count": suffix_end - suffix_start,
        "future_original_tokens_removed": suffix_start - cut_end,
        "future_original_tokens_present": False,
        "prompt_tokens_changed": False,
        "receiver_prefix_tokens_changed": False,
    }


def compile_counter_write_site_plan(
    row: Mapping[str, Any],
    tokenizer: Any,
    encoding: NativeTraceEncoding,
    *,
    site_kinds: Sequence[str] = REGISTERED_COUNTER_WRITE_SITES,
    answer_site_id: str = "answer_query_v3",
) -> dict[str, Any]:
    """Resolve exact baseline-token positions for all competing write sites."""

    requested = tuple(str(value) for value in site_kinds)
    unknown = sorted(set(requested) - set(REGISTERED_COUNTER_WRITE_SITES))
    if unknown:
        raise ValueError(f"Unknown counter write sites: {unknown}")
    family = infer_model_family(row)
    raw = raw_output_text(row)
    parser = find_trace_count_sequence(
        raw, model_family=family, gold_records=gold_records(row)
    )
    aligned = align_trace_sites(
        tokenizer,
        raw_text=raw,
        baseline_output_token_ids=output_token_ids(row),
        sites=trace_char_sites(raw, parser),
    )
    answer_sites = [
        site for site in aligned if site.char_site.site_id == str(answer_site_id)
    ]
    if len(answer_sites) != 1 or not answer_sites[0].alignment_eligible:
        raise ValueError("The registered answer query is not exactly aligned")
    answer = answer_sites[0]
    output_map = build_output_token_map(row, tokenizer)
    reasoning_end = int(parser.reasoning_end_char or 0)
    terminal = output_map.span(
        "minimal_native_terminal_bridge",
        reasoning_end,
        int(answer.char_site.char_end),
    )
    if terminal.get("status") != "ok":
        raise ValueError(f"Cannot align the minimal terminal bridge: {terminal}")
    if not terminal.get("exact_char_start") or not terminal.get("exact_char_end"):
        raise ValueError("Minimal terminal bridge must begin/end on exact token boundaries")
    suffix_text = str(terminal["char_text"])
    if re.search(r"\d", suffix_text):
        raise ValueError("Minimal terminal bridge leaks a numeric count")
    suffix_output_start = int(terminal["output_token_start"])
    suffix_output_end = int(terminal["output_token_end"])
    expected_query_end = int(encoding.query_position) - int(
        encoding.prompt_token_count
    ) + 1
    if suffix_output_end != expected_query_end:
        raise RuntimeError("Terminal bridge does not end at the encoded answer query")

    positions: dict[str, dict[int, dict[str, Any]]] = {
        kind: {} for kind in requested
    }
    for site in aligned:
        kind = str(site.char_site.site_kind)
        occurrence = site.char_site.occurrence
        if kind not in positions or occurrence is None:
            continue
        if not site.alignment_eligible:
            continue
        covering = output_map.span(
            str(site.char_site.site_id),
            int(site.char_site.char_start),
            int(site.char_site.char_end),
        )
        if covering.get("status") != "ok":
            continue
        output_end = int(covering["output_token_end"])
        if output_end < 1 or output_end > suffix_output_start:
            continue
        positions[kind][int(occurrence)] = {
            **site.to_dict(),
            "full_sequence_endpoint": int(encoding.prompt_token_count) + output_end - 1,
            "baseline_position_alignment": (
                "exact_character_boundary"
                if site.literal_token_end is not None
                else "actual_baseline_token_covering_character_site"
            ),
            "covering_token_audit": covering,
            "putative_progress_count": (
                int(occurrence) - 1 if kind == "pre_city" else int(occurrence)
            ),
        }
    missing = [kind for kind, values in positions.items() if not values]
    if missing:
        raise ValueError(f"No exact baseline-token sites for {missing}")
    return {
        "schema_version": COUNTER_WRITE_SITE_SCHEMA_VERSION,
        "request_id": str(encoding.request_id),
        "site_positions": positions,
        "terminal_suffix_full_start": int(encoding.prompt_token_count)
        + suffix_output_start,
        "terminal_suffix_text": suffix_text,
        "terminal_suffix_token_text": str(terminal["token_text"]),
        "terminal_suffix_contains_numeric_count": False,
        "terminal_suffix_audit": terminal,
        "answer_site": answer.to_dict(),
        "parser_marker_kind": str(parser.marker_kind),
        "parser_item_count": int(parser.item_count),
        "parser_trace_one_to_one": bool(parser.trace_one_to_one),
    }


@torch.inference_mode()
def run_counter_write_site_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    source_layers: Sequence[int],
    receiver_occurrences: Sequence[int],
    donor_offsets: Sequence[int] = (-1, 1),
    span_widths: Sequence[int] = (1, 4),
    site_kinds: Sequence[str] = REGISTERED_COUNTER_WRITE_SITES,
    answer_site_id: str = "answer_query_v3",
) -> list[dict[str, Any]]:
    """Run a same-trajectory, early-stop full-state write-site ladder."""

    encoding, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    plan = compile_counter_write_site_plan(
        row,
        tokenizer,
        encoding,
        site_kinds=site_kinds,
        answer_site_id=answer_site_id,
    )
    layers = tuple(sorted({int(value) for value in source_layers}))
    if not layers or layers[0] < 0 or layers[-1] >= int(adapter.num_layers) - 1:
        raise ValueError("Counter write-site source layers are invalid")
    receivers = tuple(int(value) for value in receiver_occurrences)
    offsets = tuple(int(value) for value in donor_offsets)
    widths = tuple(sorted({int(value) for value in span_widths}))
    if not receivers or not offsets or 0 in offsets or not widths or widths[0] < 1:
        raise ValueError("Receiver, donor-offset, and span-width grids are invalid")

    requested_cells: list[tuple[str, int, int, int, int, int]] = []
    capture_positions: set[int] = set()
    for kind in site_kinds:
        by_occurrence = plan["site_positions"][str(kind)]
        for receiver in receivers:
            receiver_site = by_occurrence.get(receiver)
            if receiver_site is None:
                continue
            for offset in offsets:
                donor = receiver + offset
                donor_site = by_occurrence.get(donor)
                if donor_site is None:
                    continue
                receiver_end = int(receiver_site["full_sequence_endpoint"])
                donor_end = int(donor_site["full_sequence_endpoint"])
                for width in widths:
                    if (
                        receiver_end - width + 1 < int(encoding.prompt_token_count)
                        or donor_end - width + 1 < int(encoding.prompt_token_count)
                    ):
                        continue
                    requested_cells.append(
                        (str(kind), receiver, donor, offset, width, receiver_end)
                    )
                    capture_positions.update(range(donor_end - width + 1, donor_end + 1))
    if not requested_cells:
        raise ValueError("No exact donor/receiver site cells survived compilation")

    patch_layers = tuple(range(min(layers), int(adapter.num_layers)))
    ordered_capture_positions = tuple(sorted(capture_positions))
    _logits, captured = capture_post_block_states(
        model,
        adapter,
        encoding,
        ordered_capture_positions,
        layers=patch_layers,
    )
    capture_index = {
        position: index for index, position in enumerate(ordered_capture_positions)
    }
    baseline_cache: dict[tuple[str, int], tuple[NativeTraceEncoding, dict[str, Any], dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for kind, receiver, donor, offset, width, receiver_end in requested_cells:
        cache_key = (kind, receiver)
        if cache_key not in baseline_cache:
            early, early_audit = build_site_early_stop_encoding(
                encoding,
                cut_position=receiver_end,
                terminal_suffix_start=int(plan["terminal_suffix_full_start"]),
                site_id=f"{kind}:{receiver}",
            )
            baseline_prefill, _readouts, _applications, _norms = (
                _prefill_with_layerwise_state_replacements(
                    model,
                    adapter,
                    early,
                    positions=(receiver_end,),
                    replacements=None,
                    readout_layers=(int(adapter.num_layers) - 1,),
                    readout_positions=(int(early.query_position),),
                )
            )
            baseline_outcomes = _score_and_generate_prefill(
                model,
                tokenizer,
                early,
                baseline_prefill,
                run_greedy=False,
                max_new_tokens=1,
            )
            baseline_cache[cache_key] = (early, early_audit, baseline_outcomes)
        early, early_audit, baseline_outcomes = baseline_cache[cache_key]
        donor_end = int(
            plan["site_positions"][kind][donor]["full_sequence_endpoint"]
        )
        receiver_positions = tuple(range(receiver_end - width + 1, receiver_end + 1))
        donor_positions = tuple(range(donor_end - width + 1, donor_end + 1))
        donor_indices = [capture_index[position] for position in donor_positions]
        receiver_token_ids = tuple(
            int(encoding.input_ids[position]) for position in receiver_positions
        )
        donor_token_ids = tuple(
            int(encoding.input_ids[position]) for position in donor_positions
        )
        token_matches = sum(
            left == right for left, right in zip(receiver_token_ids, donor_token_ids)
        )
        receiver_progress = int(
            plan["site_positions"][kind][receiver]["putative_progress_count"]
        )
        donor_progress = int(
            plan["site_positions"][kind][donor]["putative_progress_count"]
        )
        if not (1 <= receiver_progress <= 10 and 1 <= donor_progress <= 10):
            continue
        for source_layer in layers:
            active_layers = tuple(range(source_layer, int(adapter.num_layers)))
            replacements = {
                layer: captured[layer][donor_indices].clone()
                for layer in active_layers
            }
            prefill, _readouts, applications, realized_norms = (
                _prefill_with_layerwise_state_replacements(
                    model,
                    adapter,
                    early,
                    positions=receiver_positions,
                    replacements=replacements,
                    readout_layers=(int(adapter.num_layers) - 1,),
                    readout_positions=(int(early.query_position),),
                )
            )
            patched_outcomes = _score_and_generate_prefill(
                model,
                tokenizer,
                early,
                prefill,
                run_greedy=False,
                max_new_tokens=1,
            )
            rows.append(
                {
                    "schema_version": COUNTER_WRITE_SITE_SCHEMA_VERSION,
                    "experiment_id": "same_trajectory_counter_write_site_ladder",
                    "status": "ok",
                    "request_id": str(encoding.request_id),
                    "model_label": str(encoding.model_label),
                    "seed": int(encoding.seed),
                    "gold_count": int(encoding.count),
                    "answer_site_id": str(answer_site_id),
                    "site_kind": kind,
                    "receiver_occurrence": receiver,
                    "donor_occurrence": donor,
                    "donor_offset": offset,
                    "source_layer": source_layer,
                    "patch_layers": list(active_layers),
                    "patch_layer_mode": "cumulative_clamp_source_through_last",
                    "patch_span_width": width,
                    "patch_geometry": "right_aligned_site_suffix",
                    "receiver_positions": list(receiver_positions),
                    "donor_positions": list(donor_positions),
                    "receiver_token_ids": list(receiver_token_ids),
                    "donor_token_ids": list(donor_token_ids),
                    "donor_receiver_token_match_count": token_matches,
                    "donor_receiver_token_match_fraction": token_matches / width,
                    "donor_receiver_surface_tokens_identical": bool(
                        token_matches == width
                    ),
                    "patch_hook_applications": {
                        str(key): value for key, value in applications.items()
                    },
                    "patch_realized_fro_norm_by_layer": {
                        str(key): value for key, value in realized_norms.items()
                    },
                    "baseline_expected_count": float(
                        baseline_outcomes["expected_count"]
                    ),
                    "patched_expected_count": float(
                        patched_outcomes["expected_count"]
                    ),
                    "baseline_candidate_log_scores": str(
                        baseline_outcomes["candidate_log_scores"]
                    ),
                    "patched_candidate_log_scores": str(
                        patched_outcomes["candidate_log_scores"]
                    ),
                    "outcome_blind": True,
                    "selection_rank_used": False,
                    "prompt_modified": False,
                    "causal_claim_scope": (
                        "exploratory_same_trajectory_site_localization_not_confirmation"
                    ),
                    **directional_count_metrics(
                        baseline_outcomes,
                        patched_outcomes,
                        receiver_progress=receiver_progress,
                        donor_progress=donor_progress,
                    ),
                    **early_audit,
                    "terminal_suffix_text": str(plan["terminal_suffix_text"]),
                    "terminal_suffix_contains_numeric_count": False,
                    "parser_marker_kind": str(plan["parser_marker_kind"]),
                    "parser_trace_one_to_one": bool(
                        plan["parser_trace_one_to_one"]
                    ),
                    "registry_sha256": registry.to_dict()["registry_sha256"],
                }
            )
    if not rows:
        raise RuntimeError("Counter write-site ladder produced no trial rows")
    return rows


@torch.inference_mode()
def run_counter_write_site_uninformative_restore_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    source_layers: Sequence[int],
    target_occurrences: Sequence[int],
    span_widths: Sequence[int] = (1, 4),
    site_kinds: Sequence[str] = REGISTERED_COUNTER_WRITE_SITES,
    random_seed: int,
    answer_site_id: str = "answer_query_v3",
) -> list[dict[str, Any]]:
    """Restore clean site states into a same-position no-needle receiver.

    This is the direct site-localization analogue of the historical HTML
    experiment.  Prompt records and all parsed trace items are replaced with
    same-length ordinary prompt windows before the receiver is early-stopped.
    Thus the final query has no needle evidence to recount.
    """

    from .unnumbered_counter_restore import build_fully_uninformative_encoding

    clean, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    receiver, receiver_audit = build_fully_uninformative_encoding(
        clean,
        registry,
        tokenizer,
        random_seed=int(random_seed),
    )
    plan = compile_counter_write_site_plan(
        row,
        tokenizer,
        clean,
        site_kinds=site_kinds,
        answer_site_id=answer_site_id,
    )
    layers = tuple(sorted({int(value) for value in source_layers}))
    if not layers or layers[0] < 0 or layers[-1] >= int(adapter.num_layers) - 1:
        raise ValueError("Counter restore source layers are invalid")
    targets = tuple(int(value) for value in target_occurrences)
    widths = tuple(sorted({int(value) for value in span_widths}))
    if not targets or not widths or widths[0] < 1:
        raise ValueError("Counter restore target/width grids are invalid")

    requested_cells: list[tuple[str, int, int, int]] = []
    capture_positions: set[int] = set()
    for kind in site_kinds:
        by_occurrence = plan["site_positions"][str(kind)]
        for occurrence in targets:
            site = by_occurrence.get(occurrence)
            if site is None:
                continue
            endpoint = int(site["full_sequence_endpoint"])
            target_progress = int(site["putative_progress_count"])
            if not 1 <= target_progress <= 10:
                continue
            for width in widths:
                if endpoint - width + 1 < int(clean.prompt_token_count):
                    continue
                requested_cells.append(
                    (str(kind), occurrence, width, endpoint)
                )
                capture_positions.update(range(endpoint - width + 1, endpoint + 1))
    if not requested_cells:
        raise ValueError("No no-needle counter restore cells survived compilation")

    patch_layers = tuple(range(min(layers), int(adapter.num_layers)))
    ordered_positions = tuple(sorted(capture_positions))
    _logits, clean_capture = capture_post_block_states(
        model,
        adapter,
        clean,
        ordered_positions,
        layers=patch_layers,
    )
    position_index = {
        position: index for index, position in enumerate(ordered_positions)
    }
    baseline_cache: dict[
        tuple[str, int],
        tuple[NativeTraceEncoding, dict[str, Any], dict[str, Any]],
    ] = {}
    rows: list[dict[str, Any]] = []
    for kind, occurrence, width, endpoint in requested_cells:
        cache_key = (kind, occurrence)
        if cache_key not in baseline_cache:
            early_receiver, early_audit = build_site_early_stop_encoding(
                receiver,
                cut_position=endpoint,
                terminal_suffix_start=int(plan["terminal_suffix_full_start"]),
                site_id=f"{kind}:{occurrence}",
            )
            baseline_prefill, _readout, _applications, _norms = (
                _prefill_with_layerwise_state_replacements(
                    model,
                    adapter,
                    early_receiver,
                    positions=(endpoint,),
                    replacements=None,
                    readout_layers=(int(adapter.num_layers) - 1,),
                    readout_positions=(int(early_receiver.query_position),),
                )
            )
            baseline_outcomes = _score_and_generate_prefill(
                model,
                tokenizer,
                early_receiver,
                baseline_prefill,
                run_greedy=False,
                max_new_tokens=1,
            )
            baseline_cache[cache_key] = (
                early_receiver,
                early_audit,
                baseline_outcomes,
            )
        early_receiver, early_audit, baseline_outcomes = baseline_cache[cache_key]
        positions = tuple(range(endpoint - width + 1, endpoint + 1))
        indices = [position_index[position] for position in positions]
        clean_token_ids = tuple(int(clean.input_ids[position]) for position in positions)
        receiver_token_ids = tuple(
            int(receiver.input_ids[position]) for position in positions
        )
        token_matches = sum(
            left == right for left, right in zip(clean_token_ids, receiver_token_ids)
        )
        target_progress = int(
            plan["site_positions"][kind][occurrence]["putative_progress_count"]
        )
        for source_layer in layers:
            active_layers = tuple(range(source_layer, int(adapter.num_layers)))
            replacements = {
                layer: clean_capture[layer][indices].clone()
                for layer in active_layers
            }
            prefill, _readout, applications, realized_norms = (
                _prefill_with_layerwise_state_replacements(
                    model,
                    adapter,
                    early_receiver,
                    positions=positions,
                    replacements=replacements,
                    readout_layers=(int(adapter.num_layers) - 1,),
                    readout_positions=(int(early_receiver.query_position),),
                )
            )
            patched_outcomes = _score_and_generate_prefill(
                model,
                tokenizer,
                early_receiver,
                prefill,
                run_greedy=False,
                max_new_tokens=1,
            )
            rows.append(
                {
                    "schema_version": COUNTER_WRITE_SITE_SCHEMA_VERSION,
                    "experiment_id": (
                        "no_needle_same_position_counter_write_site_restore"
                    ),
                    "status": "ok",
                    "request_id": str(clean.request_id),
                    "model_label": str(clean.model_label),
                    "seed": int(clean.seed),
                    "gold_count": int(clean.count),
                    "answer_site_id": str(answer_site_id),
                    "site_kind": kind,
                    "target_occurrence": occurrence,
                    "source_layer": source_layer,
                    "patch_layers": list(active_layers),
                    "patch_layer_mode": "cumulative_clamp_source_through_last",
                    "patch_span_width": width,
                    "patch_geometry": "same_position_right_aligned_site_suffix",
                    "patch_positions": list(positions),
                    "clean_token_ids": list(clean_token_ids),
                    "receiver_token_ids": list(receiver_token_ids),
                    "clean_receiver_token_match_count": token_matches,
                    "clean_receiver_token_match_fraction": token_matches / width,
                    "patch_hook_applications": {
                        str(key): value for key, value in applications.items()
                    },
                    "patch_realized_fro_norm_by_layer": {
                        str(key): value for key, value in realized_norms.items()
                    },
                    "baseline_expected_count": float(
                        baseline_outcomes["expected_count"]
                    ),
                    "patched_expected_count": float(
                        patched_outcomes["expected_count"]
                    ),
                    "baseline_candidate_log_scores": str(
                        baseline_outcomes["candidate_log_scores"]
                    ),
                    "patched_candidate_log_scores": str(
                        patched_outcomes["candidate_log_scores"]
                    ),
                    "outcome_blind": True,
                    "selection_rank_used": False,
                    "prompt_template_modified": False,
                    "prompt_and_trace_needles_removed_in_receiver": True,
                    "causal_claim_scope": (
                        "exploratory_no_needle_site_state_sufficiency_not_confirmation"
                    ),
                    **target_count_restoration_metrics(
                        baseline_outcomes,
                        patched_outcomes,
                        target_progress=target_progress,
                    ),
                    **early_audit,
                    **receiver_audit,
                    "terminal_suffix_text": str(plan["terminal_suffix_text"]),
                    "terminal_suffix_contains_numeric_count": False,
                    "parser_marker_kind": str(plan["parser_marker_kind"]),
                    "parser_trace_one_to_one": bool(
                        plan["parser_trace_one_to_one"]
                    ),
                    "registry_sha256": registry.to_dict()["registry_sha256"],
                }
            )
    if not rows:
        raise RuntimeError("No-needle counter write-site restore produced no rows")
    return rows
