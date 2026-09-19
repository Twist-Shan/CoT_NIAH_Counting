"""One-seed, full-trajectory counter restoration walkthrough.

This is the Native-thinking analogue of the illustrative restoration panel in
``NIAH-counting.html``.  It is deliberately a case study rather than a second
inferential experiment: one metadata-frozen confirmation trace is followed
from occurrence 1 through occurrence N.  Prompt records and the complete trace
context are first replaced by same-length ordinary prompt text.  We then restore, one
occurrence at a time, either the whole clean item or the grammar-aware counter
carrier while leaving the final answer query untouched.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from typing import Any, Mapping, Sequence

import torch

from realistic_niah_v4.modeling import DecoderAdapter, capture_post_block_states

from .causal_sites import compile_causal_site_plan
from .count_stream import (
    COUNT_STREAM_SCHEMA_VERSION,
    AnswerSourceRegistry,
    NativeTraceEncoding,
    _full_state_patch_layers,
    _prefill_with_layerwise_state_replacements,
    _score_and_generate_prefill,
    _sha256_json,
    build_answer_source_registry,
)
from .terminal_token_state import _site_positions


WALKTHROUGH_CONDITIONS = (
    "clean",
    "uninformative",
    "full_item_restore",
    "counter_carrier_restore",
    "counter_carrier_matched_control",
)


def _span_positions(spans: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    return tuple(
        position
        for start, end in spans
        for position in range(int(start), int(end))
    )


def build_uninformative_prompt_trace_encoding(
    encoding: NativeTraceEncoding,
    registry: AnswerSourceRegistry,
    tokenizer: Any,
    *,
    random_seed: int,
) -> tuple[NativeTraceEncoding, dict[str, Any]]:
    """Replace prompt records and the full trace with equal-length ordinary text."""

    original = tuple(int(value) for value in encoding.input_ids)
    # V1 replaced only parser-observed items.  That left unparsed trace text
    # (including possible terminal summaries) intact and therefore allowed a
    # final-count leak.  The HTML-aligned V2 control removes the complete CoT
    # context while preserving every absolute token position.
    sources = tuple(registry.prompt_records) + tuple(registry.trace_context)
    if not sources:
        raise ValueError("Single-seed walkthrough has no source spans")
    source_positions = set(_span_positions(sources))
    prompt_record_positions = set(registry.positions("prompt_records"))
    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", ())}
    max_width = max(int(end) - int(start) for start, end in sources)
    candidates: list[tuple[int, tuple[int, ...]]] = []
    for start in range(1, int(registry.prompt_token_count) - max_width + 1):
        positions = tuple(range(start, start + max_width))
        if set(positions) & prompt_record_positions:
            continue
        window = tuple(original[position] for position in positions)
        if any(token in special_ids for token in window):
            continue
        candidates.append((start, window))
    if not candidates:
        raise ValueError("No ordinary prompt window can replace the registered sources")

    active = list(original)
    selected_starts: list[int] = []
    changed_by_span: list[int] = []
    for span_index, (span_start, span_end) in enumerate(sources):
        start, end = int(span_start), int(span_end)
        width = end - start
        original_span = original[start:end]
        digest = hashlib.sha256(
            (
                f"{encoding.request_id}|{int(random_seed)}|{span_index}|"
                f"{start}|{end}"
            ).encode("utf-8")
        ).digest()
        initial = int.from_bytes(digest[:8], "big") % len(candidates)
        selected_start = -1
        replacement: tuple[int, ...] | None = None
        for offset in range(len(candidates)):
            candidate_start, candidate_window = candidates[
                (initial + offset) % len(candidates)
            ]
            candidate = candidate_window[:width]
            if candidate != original_span:
                selected_start = int(candidate_start)
                replacement = candidate
                break
        if replacement is None:
            raise ValueError("Every ordinary control equals a registered source span")
        active[start:end] = replacement
        selected_starts.append(selected_start)
        changed_by_span.append(
            sum(left != right for left, right in zip(original_span, replacement))
        )

    changed = sum(original[position] != active[position] for position in source_positions)
    if changed != sum(changed_by_span) or changed <= 0:
        raise RuntimeError("Prompt/trace uninformative replacement audit failed")
    control = replace(encoding, input_ids=tuple(active))
    return control, {
        "control_construction": "same_length_ordinary_prompt_windows",
        "control_retokenized": False,
        "control_sequence_length_equal": bool(
            control.sequence_length == encoding.sequence_length
        ),
        "answer_query_token_preserved": bool(
            control.input_ids[registry.query_position]
            == encoding.input_ids[registry.query_position]
        ),
        "prompt_record_count": len(registry.prompt_records),
        "trace_context_span_count": len(registry.trace_context),
        "trace_item_count": len(registry.trace_items),
        "source_span_count": len(sources),
        "source_token_count": len(source_positions),
        "changed_source_token_count": int(changed),
        "changed_source_token_fraction": float(changed / len(source_positions)),
        "ordinary_window_starts": selected_starts,
        "ordinary_window_starts_sha256": _sha256_json(selected_starts),
        "control_input_ids_sha256": _sha256_json(control.input_ids),
    }


def occurrence_counter_geometry(
    registry: AnswerSourceRegistry,
    event: Mapping[str, Any],
    occurrence: int,
) -> tuple[dict[str, tuple[int, ...]], dict[str, Any]]:
    """Return the full item and grammar-aware carrier for one occurrence."""

    index = int(occurrence) - 1
    if not 0 <= index < len(registry.trace_items):
        raise ValueError("Walkthrough occurrence lies outside the trace")
    item_start, item_end = registry.trace_items[index]
    full_item = tuple(range(int(item_start), int(item_end)))
    item_set = set(full_item)
    sites = event.get("sites", {})
    marker = _site_positions(
        sites.get("rank_evidence_core_span"), role="rank_evidence_core_span"
    )
    city = _site_positions(sites.get("city_target_span"), role="city_target_span")
    commit = _site_positions(
        sites.get("post_update_commit_state"), role="post_update_commit_state"
    )
    grammar = str(event.get("grammar_class", ""))
    if "rank_after_city" in grammar:
        timing = "rank_after_city"
        carrier = marker
        carrier_name = "marker_core"
    elif "rank_before_city" in grammar:
        timing = "rank_before_city"
        carrier = tuple(range(city[0], commit[-1] + 1))
        carrier_name = "city_to_commit_tail"
    else:
        raise ValueError(f"Unsupported walkthrough grammar: {grammar!r}")
    for name, positions in (("full_item", full_item), ("counter_carrier", carrier)):
        if not positions or not set(positions) <= item_set:
            raise ValueError(f"Walkthrough {name} is outside occurrence {occurrence}")
        if max(positions) >= int(registry.query_position):
            raise ValueError(f"Walkthrough {name} reaches the answer query")
    return {
        "full_item": full_item,
        "counter_carrier": carrier,
    }, {
        "occurrence_grammar_class": grammar,
        "grammar_timing_stratum": timing,
        "counter_carrier_component": carrier_name,
        "item_span": [int(item_start), int(item_end)],
    }


def matched_ordinary_positions(
    registry: AnswerSourceRegistry, receivers: Sequence[int]
) -> tuple[int, ...]:
    """Choose equal-budget prompt-background donor states for a carrier."""

    targets = tuple(int(value) for value in receivers)
    forbidden = set(registry.positions("prompt_records"))
    candidates = {
        position
        for position in range(1, int(registry.prompt_token_count))
        if position not in forbidden
    }
    if len(candidates) < len(targets):
        raise ValueError("Not enough ordinary positions for walkthrough control")
    result: list[int] = []
    available = set(candidates)
    for target in targets:
        donor = min(available, key=lambda value: (abs(value - target), value))
        result.append(int(donor))
        available.remove(donor)
    return tuple(result)


def _target_count_metrics(outcomes: Mapping[str, Any], target: int) -> dict[str, Any]:
    counts = [int(value) for value in str(outcomes["candidate_counts"]).split(",")]
    scores = [float(value) for value in str(outcomes["candidate_log_scores"]).split(",")]
    probabilities = [
        float(value) for value in str(outcomes["candidate_probabilities"]).split(",")
    ]
    if counts != list(range(1, 11)) or len(scores) != 10 or len(probabilities) != 10:
        raise ValueError("Walkthrough count-vector schema changed")
    index = counts.index(int(target))
    other = max(score for offset, score in enumerate(scores) if offset != index)
    predicted = int(outcomes["predicted_count_among_candidates"])
    greedy = outcomes.get("prediction")
    return {
        "restored_target_count": int(target),
        "restored_target_count_log_score": float(scores[index]),
        "restored_target_count_margin": float(scores[index] - other),
        "restored_target_count_probability": float(probabilities[index]),
        "candidate_prediction_matches_restored_target": bool(predicted == int(target)),
        "candidate_prediction_absolute_error_from_restored_target": abs(
            predicted - int(target)
        ),
        "greedy_prediction_matches_restored_target": bool(greedy == int(target)),
        "greedy_prediction_absolute_error_from_restored_target": (
            None if greedy is None else abs(int(greedy) - int(target))
        ),
    }


@torch.inference_mode()
def run_single_seed_walkthrough_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    source_layer: int,
    random_seed: int,
    answer_site_id: str = "answer_query_v3",
    max_new_tokens: int = 16,
) -> list[dict[str, Any]]:
    """Restore every occurrence in one frozen trace and score the final count."""

    clean, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    if int(clean.count) < 2 or len(registry.trace_items) != int(clean.count):
        raise ValueError("Walkthrough requires a complete multi-item trace")
    control, control_audit = build_uninformative_prompt_trace_encoding(
        clean, registry, tokenizer, random_seed=int(random_seed)
    )
    causal_plan = compile_causal_site_plan(row, tokenizer)
    events = list(causal_plan.get("events", ()))
    if len(events) != int(clean.count):
        raise ValueError("Walkthrough causal events do not match the trace count")
    patch_layers = _full_state_patch_layers(
        source_layer=int(source_layer),
        num_layers=int(adapter.num_layers),
        layer_mode="cumulative_clamp",
    )

    geometries: dict[int, dict[str, tuple[int, ...]]] = {}
    geometry_audits: dict[int, dict[str, Any]] = {}
    matched: dict[int, tuple[int, ...]] = {}
    capture_positions: set[int] = set()
    for occurrence, event in enumerate(events, start=1):
        geometry, audit = occurrence_counter_geometry(registry, event, occurrence)
        control_positions = matched_ordinary_positions(
            registry, geometry["counter_carrier"]
        )
        geometries[occurrence] = geometry
        geometry_audits[occurrence] = audit
        matched[occurrence] = control_positions
        capture_positions.update(geometry["full_item"])
        capture_positions.update(control_positions)
    ordered_capture = tuple(sorted(capture_positions))
    position_index = {position: index for index, position in enumerate(ordered_capture)}
    _logits, clean_capture = capture_post_block_states(
        model,
        adapter,
        clean,
        ordered_capture,
        layers=patch_layers,
    )

    common = {
        "schema_version": COUNT_STREAM_SCHEMA_VERSION,
        "experiment_id": "single_seed_full_trajectory_counter_walkthrough",
        "request_id": clean.request_id,
        "model_label": clean.model_label,
        "seed": int(clean.seed),
        "dataset_split": clean.split,
        "gold_count": int(clean.count),
        "answer_site_id": answer_site_id,
        "source_layer": int(source_layer),
        "patch_layer_mode": "cumulative_clamp",
        "patch_layers": list(patch_layers),
        "patch_layer_count": len(patch_layers),
        "answer_query_patched": False,
        "selection_rank_used": False,
        "case_selected_by_outcome": False,
        "case_study_not_inferential": True,
        "causal_claim_scope": "illustrative_full_trajectory_state_sufficiency",
        "registry_sha256": registry.to_dict()["registry_sha256"],
        **control_audit,
    }

    def evaluate(
        *,
        condition: str,
        active: NativeTraceEncoding,
        occurrence: int,
        positions: Sequence[int],
        donor_positions: Sequence[int] | None,
    ) -> dict[str, Any]:
        replacements = None
        if donor_positions is not None:
            indices = [position_index[int(value)] for value in donor_positions]
            replacements = {
                layer: clean_capture[layer][indices].clone() for layer in patch_layers
            }
        prefill, _captures, applications, norms = (
            _prefill_with_layerwise_state_replacements(
                model,
                adapter,
                active,
                positions=positions,
                replacements=replacements,
                readout_layers=(int(adapter.num_layers) - 1,),
                readout_positions=(int(registry.query_position),),
            )
        )
        outcomes = _score_and_generate_prefill(
            model,
            tokenizer,
            active,
            prefill,
            run_greedy=True,
            max_new_tokens=int(max_new_tokens),
        )
        target = int(occurrence) if occurrence > 0 else int(clean.count)
        aggregate_norm = float(math.sqrt(sum(value * value for value in norms.values())))
        row_result = {
            **common,
            "condition": condition,
            "status": "ok",
            "restored_occurrence": int(occurrence),
            "patch_positions": [int(value) for value in positions],
            "patch_positions_sha256": _sha256_json(list(positions)),
            "patch_token_count": len(tuple(positions)),
            "state_donor_positions": (
                [] if donor_positions is None else [int(value) for value in donor_positions]
            ),
            "state_donor_positions_sha256": _sha256_json(
                [] if donor_positions is None else list(donor_positions)
            ),
            "patch_hook_applications": {
                str(layer): int(value) for layer, value in sorted(applications.items())
            },
            "patch_realized_fro_norm_by_layer": {
                str(layer): float(value) for layer, value in sorted(norms.items())
            },
            "patch_realized_aggregate_fro_norm": aggregate_norm,
            **outcomes,
            **_target_count_metrics(outcomes, target),
        }
        if occurrence > 0:
            row_result.update(geometry_audits[occurrence])
        return row_result

    reference_positions = geometries[int(clean.count)]["counter_carrier"]
    rows = [
        evaluate(
            condition="clean",
            active=clean,
            occurrence=0,
            positions=reference_positions,
            donor_positions=None,
        ),
        evaluate(
            condition="uninformative",
            active=control,
            occurrence=0,
            positions=reference_positions,
            donor_positions=None,
        ),
    ]
    for occurrence in range(1, int(clean.count) + 1):
        geometry = geometries[occurrence]
        rows.extend(
            [
                evaluate(
                    condition="full_item_restore",
                    active=control,
                    occurrence=occurrence,
                    positions=geometry["full_item"],
                    donor_positions=geometry["full_item"],
                ),
                evaluate(
                    condition="counter_carrier_restore",
                    active=control,
                    occurrence=occurrence,
                    positions=geometry["counter_carrier"],
                    donor_positions=geometry["counter_carrier"],
                ),
                evaluate(
                    condition="counter_carrier_matched_control",
                    active=control,
                    occurrence=occurrence,
                    positions=geometry["counter_carrier"],
                    donor_positions=matched[occurrence],
                ),
            ]
        )
    expected_rows = 2 + 3 * int(clean.count)
    if len(rows) != expected_rows:
        raise RuntimeError("Single-seed walkthrough factorial changed")
    return rows
