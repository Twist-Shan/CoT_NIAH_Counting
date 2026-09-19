"""Terminal item-token -> full-span state -> count-answer mediation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import torch

from realistic_niah_v4.modeling import DecoderAdapter, capture_post_block_states

from .causal_sites import compile_causal_site_plan
from .count_stream import (
    COUNT_STREAM_SCHEMA_VERSION,
    NativeTraceEncoding,
    _full_state_patch_layers,
    _prefill_with_layerwise_state_replacements,
    _score_and_generate_prefill,
    _sha256_json,
    build_answer_source_registry,
    build_html_aligned_uninformative_trace_encoding,
)


REGISTERED_TERMINAL_TOKEN_STATE_CONDITIONS = (
    "clean",
    "clean_terminal_token_ablation",
    "uninformative",
    "terminal_token_restore",
    "terminal_marker_token_restore",
    "terminal_nonmarker_token_restore",
    "uninformative_terminal_state_restore",
    "terminal_token_restore_state_occluded",
)

REGISTERED_LOCAL_TERMINAL_TOKEN_STATE_CONDITIONS = (
    "clean",
    "terminal_token_ablation",
    "terminal_marker_token_ablation",
    "terminal_nonmarker_token_ablation",
    "ablated_terminal_state_restore",
    "clean_terminal_state_occluded",
)

REGISTERED_GRAMMAR_SPAN_DECOMPOSITION_GEOMETRIES = (
    "full_item",
    "marker_core",
    "retrieved_city",
    "grammar_terminal_update",
    "boundary_commit",
)

REGISTERED_GRAMMAR_SPAN_DECOMPOSITION_CONDITIONS = (
    "clean",
    "uninformative",
    *tuple(
        condition
        for geometry in REGISTERED_GRAMMAR_SPAN_DECOMPOSITION_GEOMETRIES
        for condition in (
            f"{geometry}_restore",
            f"{geometry}_matched_random",
        )
    ),
)

REGISTERED_GRAMMAR_TIMING_STRATA = (
    "rank_after_city",
    "rank_before_city",
)


def _replace_positions(
    receiver: NativeTraceEncoding,
    donor: NativeTraceEncoding,
    positions: Sequence[int],
) -> NativeTraceEncoding:
    active = list(receiver.input_ids)
    donor_ids = tuple(int(value) for value in donor.input_ids)
    normalized = tuple(sorted({int(value) for value in positions}))
    if not normalized or normalized[0] < 0 or normalized[-1] >= len(active):
        raise ValueError("Terminal token-state replacement positions are invalid")
    for position in normalized:
        active[position] = donor_ids[position]
    return replace(receiver, input_ids=tuple(active))


def _restore_or_identity(
    receiver: NativeTraceEncoding,
    donor: NativeTraceEncoding,
    positions: Sequence[int],
) -> NativeTraceEncoding:
    """Restore a diagnostic partition, using identity for an empty partition."""

    normalized = tuple(sorted({int(value) for value in positions}))
    if not normalized:
        return receiver
    return _replace_positions(receiver, donor, normalized)


def _site_positions(
    site: Mapping[str, Any] | None,
    *,
    role: str,
) -> tuple[int, ...]:
    """Return exact full-sequence positions for one compiled semantic site."""

    if site is None or str(site.get("status", "")) != "ok":
        raise ValueError(f"Grammar span decomposition has no resolved {role} site")
    if site.get("full_sequence_token_start") is not None:
        start = int(site["full_sequence_token_start"])
        end = int(site["full_sequence_token_end"])
        if not 0 <= start < end:
            raise ValueError(f"Grammar span decomposition has an invalid {role} span")
        return tuple(range(start, end))
    if site.get("full_sequence_token_index") is not None:
        return (int(site["full_sequence_token_index"]),)
    raise ValueError(f"Grammar span decomposition cannot map {role} to tokens")


def _grammar_timed_geometry_positions(
    registry: Any,
    terminal_event: Mapping[str, Any],
) -> tuple[dict[str, tuple[int, ...]], dict[str, Any]]:
    """Compile semantic and grammar-timed terminal state-patch geometries.

    The last semantic component differs by trace grammar.  For rank-after-city
    traces it is the explicit rank/count marker; for rank-before-city traces it
    is the retrieved city.  ``grammar_terminal_update`` starts at that component
    and continues through the parser-registered item commit token.
    """

    terminal_positions = tuple(registry.positions("terminal_trace_item"))
    terminal_set = set(terminal_positions)
    if not terminal_positions:
        raise ValueError("Grammar span decomposition has an empty terminal item")
    sites = terminal_event.get("sites", {})
    marker_positions = _site_positions(
        sites.get("rank_evidence_core_span"), role="rank_evidence_core_span"
    )
    city_positions = _site_positions(
        sites.get("city_target_span"), role="city_target_span"
    )
    commit_positions = _site_positions(
        sites.get("post_update_commit_state"), role="post_update_commit_state"
    )
    grammar_class = str(terminal_event.get("grammar_class", ""))
    if "rank_after_city" in grammar_class:
        timing_stratum = "rank_after_city"
        update_start = marker_positions[0]
        terminal_component = "rank_evidence_core_span"
    elif "rank_before_city" in grammar_class:
        timing_stratum = "rank_before_city"
        update_start = city_positions[0]
        terminal_component = "city_target_span"
    else:
        raise ValueError(
            "Grammar span decomposition requires an explicit rank-before/after-city "
            f"terminal event, observed {grammar_class!r}"
        )
    update_end = commit_positions[-1] + 1
    if not update_start < update_end:
        raise ValueError("Grammar-timed update span does not precede the item commit")
    geometries = {
        "full_item": terminal_positions,
        "marker_core": marker_positions,
        "retrieved_city": city_positions,
        "grammar_terminal_update": tuple(range(update_start, update_end)),
        "boundary_commit": commit_positions,
    }
    if tuple(geometries) != REGISTERED_GRAMMAR_SPAN_DECOMPOSITION_GEOMETRIES:
        raise RuntimeError("Grammar span geometry order changed")
    for geometry, positions in geometries.items():
        if not positions or not set(positions) <= terminal_set:
            raise ValueError(
                f"Grammar span geometry {geometry} is not inside the terminal item"
            )
        if max(positions) >= int(registry.query_position):
            raise ValueError(f"Grammar span geometry {geometry} reaches the answer query")
    return geometries, {
        "terminal_grammar_class": grammar_class,
        "grammar_timing_stratum": timing_stratum,
        "grammar_terminal_component": terminal_component,
        "grammar_terminal_update_start": int(update_start),
        "grammar_terminal_update_end": int(update_end),
    }


def _matched_state_donor_positions(
    registry: Any,
    receiver_positions: Sequence[int],
) -> tuple[int, ...]:
    """Choose an equal-budget, depth-matched non-item state donor control."""

    receivers = tuple(sorted({int(value) for value in receiver_positions}))
    terminal = set(registry.positions("terminal_trace_item"))
    prompt_records = set(registry.positions("prompt_records"))
    candidates = set(registry.positions("trace_other")) - terminal
    candidates.update(
        position
        for position in range(1, int(registry.prompt_token_count))
        if position not in prompt_records
    )
    candidates -= terminal
    if len(candidates) < len(receivers):
        raise ValueError("Not enough ordinary positions for a matched state donor")
    chosen: list[int] = []
    available = set(candidates)
    for receiver in receivers:
        donor = min(
            available,
            key=lambda candidate: (abs(int(candidate) - receiver), int(candidate)),
        )
        chosen.append(int(donor))
        available.remove(donor)
    result = tuple(chosen)
    if len(result) != len(receivers) or set(result) & terminal:
        raise RuntimeError("Matched state donor budget or exclusion audit failed")
    return result


@torch.inference_mode()
def run_terminal_token_state_bridge_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    layer: int,
    random_seed: int,
    answer_site_id: str = "answer_query_v3",
    run_greedy: bool = True,
    max_new_tokens: int = 16,
) -> list[dict[str, Any]]:
    """Test whether terminal item tokens write the count-relevant terminal state."""

    clean, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    uninformative, control_audit = build_html_aligned_uninformative_trace_encoding(
        clean,
        registry,
        tokenizer,
        random_seed=int(random_seed),
    )
    terminal_positions = tuple(registry.positions("terminal_trace_item"))
    terminal_set = set(terminal_positions)
    marker_positions = tuple(
        position
        for position in registry.positions("trace_markers")
        if position in terminal_set
    )
    nonmarker_positions = tuple(
        position
        for position in registry.positions("trace_nonmarkers")
        if position in terminal_set
    )
    if not terminal_positions:
        raise ValueError("Terminal token-state bridge has an empty terminal item")
    if set(marker_positions) & set(nonmarker_positions):
        raise RuntimeError("Terminal marker/nonmarker partitions overlap")
    if set(marker_positions) | set(nonmarker_positions) != terminal_set:
        raise RuntimeError("Terminal marker/nonmarker partitions do not cover the item")

    terminal_restore = _replace_positions(
        uninformative, clean, terminal_positions
    )
    marker_restore = _restore_or_identity(uninformative, clean, marker_positions)
    nonmarker_restore = _restore_or_identity(
        uninformative, clean, nonmarker_positions
    )
    clean_terminal_ablation = _replace_positions(
        clean, uninformative, terminal_positions
    )
    patch_layers = _full_state_patch_layers(
        source_layer=int(layer),
        num_layers=int(adapter.num_layers),
        layer_mode="cumulative_clamp",
    )
    _unused, uninformative_capture = capture_post_block_states(
        model,
        adapter,
        uninformative,
        terminal_positions,
        layers=patch_layers,
    )
    _unused, restored_capture = capture_post_block_states(
        model,
        adapter,
        terminal_restore,
        terminal_positions,
        layers=patch_layers,
    )
    uninformative_states = {
        patch_layer: uninformative_capture[patch_layer].clone()
        for patch_layer in patch_layers
    }
    restored_states = {
        patch_layer: restored_capture[patch_layer].clone()
        for patch_layer in patch_layers
    }
    state_delta_norms = {
        str(patch_layer): float(
            torch.linalg.vector_norm(
                restored_states[patch_layer] - uninformative_states[patch_layer]
            )
        )
        for patch_layer in patch_layers
    }
    conditions: dict[
        str,
        tuple[NativeTraceEncoding, Mapping[int, torch.Tensor] | None],
    ] = {
        "clean": (clean, None),
        "clean_terminal_token_ablation": (clean_terminal_ablation, None),
        "uninformative": (uninformative, None),
        "terminal_token_restore": (terminal_restore, None),
        "terminal_marker_token_restore": (marker_restore, None),
        "terminal_nonmarker_token_restore": (nonmarker_restore, None),
        "uninformative_terminal_state_restore": (
            uninformative,
            restored_states,
        ),
        "terminal_token_restore_state_occluded": (
            terminal_restore,
            uninformative_states,
        ),
    }
    rows: list[dict[str, Any]] = []
    for condition in REGISTERED_TERMINAL_TOKEN_STATE_CONDITIONS:
        encoding, replacements = conditions[condition]
        prefill, _captures, applications, realized_norms = (
            _prefill_with_layerwise_state_replacements(
                model,
                adapter,
                encoding,
                positions=terminal_positions,
                replacements=replacements,
                readout_layers=(int(adapter.num_layers) - 1,),
                readout_positions=(registry.query_position,),
            )
        )
        outcomes = _score_and_generate_prefill(
            model,
            tokenizer,
            encoding,
            prefill,
            run_greedy=run_greedy,
            max_new_tokens=int(max_new_tokens),
        )
        rows.append(
            {
                "schema_version": COUNT_STREAM_SCHEMA_VERSION,
                "experiment_id": "terminal_token_to_fullspan_state_bridge",
                "condition": condition,
                "status": "ok",
                "request_id": clean.request_id,
                "model_label": clean.model_label,
                "seed": clean.seed,
                "dataset_split": clean.split,
                "gold_count": clean.count,
                "answer_site_id": answer_site_id,
                "target_occurrence": len(registry.trace_items),
                "target_is_terminal": True,
                "patch_geometry": "terminal_full_item_span_same_position",
                "patch_layer_mode": "cumulative_clamp",
                "layer": int(layer),
                "patch_layers": list(patch_layers),
                "patch_token_count": len(terminal_positions),
                "terminal_positions_sha256": _sha256_json(terminal_positions),
                "terminal_marker_token_count": len(marker_positions),
                "terminal_nonmarker_token_count": len(nonmarker_positions),
                "terminal_marker_partition_empty": not bool(marker_positions),
                "terminal_nonmarker_partition_empty": not bool(nonmarker_positions),
                "terminal_marker_positions_sha256": _sha256_json(marker_positions),
                "terminal_nonmarker_positions_sha256": _sha256_json(
                    nonmarker_positions
                ),
                "terminal_token_state_delta_norm_by_layer": state_delta_norms,
                "patch_hook_applications": {
                    str(key): value for key, value in sorted(applications.items())
                },
                "patch_realized_fro_norm_by_layer": {
                    str(key): value for key, value in sorted(realized_norms.items())
                },
                "input_token_restoration_uses_outcome": False,
                "causal_claim_scope": (
                    "terminal_item_token_content_to_fullspan_hidden_state_to_answer"
                ),
                "registry_sha256": registry.to_dict()["registry_sha256"],
                **control_audit,
                **outcomes,
            }
        )
    return rows


@torch.inference_mode()
def run_local_terminal_token_state_bridge_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    layer: int,
    random_seed: int,
    answer_site_id: str = "answer_query_v3",
    run_greedy: bool = True,
    max_new_tokens: int = 16,
) -> list[dict[str, Any]]:
    """Mediate a local terminal-item ablation through its full-span state."""

    clean, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    background, control_audit = build_html_aligned_uninformative_trace_encoding(
        clean,
        registry,
        tokenizer,
        random_seed=int(random_seed),
    )
    terminal_positions = tuple(registry.positions("terminal_trace_item"))
    terminal_set = set(terminal_positions)
    marker_positions = tuple(
        position
        for position in registry.positions("trace_markers")
        if position in terminal_set
    )
    nonmarker_positions = tuple(
        position
        for position in registry.positions("trace_nonmarkers")
        if position in terminal_set
    )
    if not terminal_positions:
        raise ValueError("Local terminal token-state bridge has an empty item")
    if set(marker_positions) & set(nonmarker_positions):
        raise RuntimeError("Local terminal marker/nonmarker partitions overlap")
    if set(marker_positions) | set(nonmarker_positions) != terminal_set:
        raise RuntimeError(
            "Local terminal marker/nonmarker partitions do not cover the item"
        )
    terminal_ablation = _replace_positions(
        clean, background, terminal_positions
    )
    marker_ablation = _restore_or_identity(clean, background, marker_positions)
    nonmarker_ablation = _restore_or_identity(
        clean, background, nonmarker_positions
    )
    patch_layers = _full_state_patch_layers(
        source_layer=int(layer),
        num_layers=int(adapter.num_layers),
        layer_mode="cumulative_clamp",
    )
    _unused, clean_capture = capture_post_block_states(
        model,
        adapter,
        clean,
        terminal_positions,
        layers=patch_layers,
    )
    _unused, ablated_capture = capture_post_block_states(
        model,
        adapter,
        terminal_ablation,
        terminal_positions,
        layers=patch_layers,
    )
    clean_states = {
        patch_layer: clean_capture[patch_layer].clone()
        for patch_layer in patch_layers
    }
    ablated_states = {
        patch_layer: ablated_capture[patch_layer].clone()
        for patch_layer in patch_layers
    }
    state_delta_norms = {
        str(patch_layer): float(
            torch.linalg.vector_norm(
                clean_states[patch_layer] - ablated_states[patch_layer]
            )
        )
        for patch_layer in patch_layers
    }
    conditions: dict[
        str,
        tuple[NativeTraceEncoding, Mapping[int, torch.Tensor] | None],
    ] = {
        "clean": (clean, None),
        "terminal_token_ablation": (terminal_ablation, None),
        "terminal_marker_token_ablation": (marker_ablation, None),
        "terminal_nonmarker_token_ablation": (nonmarker_ablation, None),
        "ablated_terminal_state_restore": (terminal_ablation, clean_states),
        "clean_terminal_state_occluded": (clean, ablated_states),
    }
    rows: list[dict[str, Any]] = []
    for condition in REGISTERED_LOCAL_TERMINAL_TOKEN_STATE_CONDITIONS:
        encoding, replacements = conditions[condition]
        prefill, _captures, applications, realized_norms = (
            _prefill_with_layerwise_state_replacements(
                model,
                adapter,
                encoding,
                positions=terminal_positions,
                replacements=replacements,
                readout_layers=(int(adapter.num_layers) - 1,),
                readout_positions=(registry.query_position,),
            )
        )
        outcomes = _score_and_generate_prefill(
            model,
            tokenizer,
            encoding,
            prefill,
            run_greedy=run_greedy,
            max_new_tokens=int(max_new_tokens),
        )
        rows.append(
            {
                "schema_version": COUNT_STREAM_SCHEMA_VERSION,
                "experiment_id": "local_terminal_token_to_fullspan_state_bridge",
                "condition": condition,
                "status": "ok",
                "request_id": clean.request_id,
                "model_label": clean.model_label,
                "seed": clean.seed,
                "dataset_split": clean.split,
                "gold_count": clean.count,
                "answer_site_id": answer_site_id,
                "target_occurrence": len(registry.trace_items),
                "target_is_terminal": True,
                "earlier_trace_tokens_remain_clean": True,
                "patch_geometry": "terminal_full_item_span_same_position",
                "patch_layer_mode": "cumulative_clamp",
                "layer": int(layer),
                "patch_layers": list(patch_layers),
                "patch_token_count": len(terminal_positions),
                "terminal_positions_sha256": _sha256_json(terminal_positions),
                "terminal_marker_token_count": len(marker_positions),
                "terminal_nonmarker_token_count": len(nonmarker_positions),
                "terminal_marker_partition_empty": not bool(marker_positions),
                "terminal_nonmarker_partition_empty": not bool(nonmarker_positions),
                "terminal_token_state_delta_norm_by_layer": state_delta_norms,
                "patch_hook_applications": {
                    str(key): value for key, value in sorted(applications.items())
                },
                "patch_realized_fro_norm_by_layer": {
                    str(key): value for key, value in sorted(realized_norms.items())
                },
                "terminal_token_ablation_uses_outcome": False,
                "causal_claim_scope": (
                    "local_terminal_item_token_content_to_fullspan_state_to_answer"
                ),
                "registry_sha256": registry.to_dict()["registry_sha256"],
                **control_audit,
                **outcomes,
            }
        )
    return rows


@torch.inference_mode()
def run_grammar_span_decomposition_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    layer: int,
    random_seed: int,
    registered_grammar_class: str,
    registered_timing_stratum: str,
    answer_site_id: str = "answer_query_v3",
    run_greedy: bool = True,
    max_new_tokens: int = 16,
) -> list[dict[str, Any]]:
    """Decompose an old-HTML full-span counter patch by semantic trace role."""

    clean, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    uninformative, control_audit = build_html_aligned_uninformative_trace_encoding(
        clean,
        registry,
        tokenizer,
        random_seed=int(random_seed),
    )
    causal_plan = compile_causal_site_plan(row, tokenizer)
    events = list(causal_plan.get("events", ()))
    if len(events) != len(registry.trace_items):
        raise ValueError("Causal event count disagrees with answer-source registry")
    terminal_event = events[-1]
    geometries, grammar_audit = _grammar_timed_geometry_positions(
        registry, terminal_event
    )
    if str(registered_grammar_class) != str(grammar_audit["terminal_grammar_class"]):
        raise ValueError("Frozen anchor grammar disagrees with the terminal event")
    if str(registered_timing_stratum) != str(
        grammar_audit["grammar_timing_stratum"]
    ):
        raise ValueError("Frozen anchor timing stratum disagrees with the terminal event")

    matched_donors = {
        geometry: _matched_state_donor_positions(registry, positions)
        for geometry, positions in geometries.items()
    }
    patch_layers = _full_state_patch_layers(
        source_layer=int(layer),
        num_layers=int(adapter.num_layers),
        layer_mode="cumulative_clamp",
    )
    capture_positions = tuple(
        sorted(
            {
                position
                for positions in (*geometries.values(), *matched_donors.values())
                for position in positions
            }
        )
    )
    _unused, clean_capture = capture_post_block_states(
        model,
        adapter,
        clean,
        capture_positions,
        layers=patch_layers,
    )
    target_union = tuple(
        sorted({position for positions in geometries.values() for position in positions})
    )
    _unused, control_capture = capture_post_block_states(
        model,
        adapter,
        uninformative,
        target_union,
        layers=patch_layers,
    )
    clean_index = {position: index for index, position in enumerate(capture_positions)}
    control_index = {position: index for index, position in enumerate(target_union)}

    replacements: dict[str, dict[int, torch.Tensor]] = {}
    delta_norms: dict[str, dict[str, float]] = {}
    for geometry, receiver_positions in geometries.items():
        donor_positions = matched_donors[geometry]
        true_states: dict[int, torch.Tensor] = {}
        random_states: dict[int, torch.Tensor] = {}
        geometry_norms: dict[str, float] = {}
        for patch_layer in patch_layers:
            clean_targets = clean_capture[patch_layer][
                [clean_index[position] for position in receiver_positions]
            ].clone()
            clean_random = clean_capture[patch_layer][
                [clean_index[position] for position in donor_positions]
            ].clone()
            control_targets = control_capture[patch_layer][
                [control_index[position] for position in receiver_positions]
            ]
            true_states[patch_layer] = clean_targets
            random_states[patch_layer] = clean_random
            geometry_norms[str(patch_layer)] = float(
                torch.linalg.vector_norm(clean_targets - control_targets)
            )
        replacements[f"{geometry}_restore"] = true_states
        replacements[f"{geometry}_matched_random"] = random_states
        delta_norms[geometry] = geometry_norms

    rows: list[dict[str, Any]] = []
    for condition in REGISTERED_GRAMMAR_SPAN_DECOMPOSITION_CONDITIONS:
        if condition in {"clean", "uninformative"}:
            active_encoding = clean if condition == "clean" else uninformative
            active_geometry = "none"
            receiver_positions = geometries["boundary_commit"]
            donor_positions: tuple[int, ...] = ()
            active_replacements = None
            replacement_source = "none"
        else:
            matched_random = condition.endswith("_matched_random")
            suffix = "_matched_random" if matched_random else "_restore"
            active_geometry = condition[: -len(suffix)]
            receiver_positions = geometries[active_geometry]
            donor_positions = (
                matched_donors[active_geometry]
                if matched_random
                else receiver_positions
            )
            active_encoding = uninformative
            active_replacements = replacements[condition]
            replacement_source = (
                "clean_depth_matched_nonitem_positions"
                if matched_random
                else "clean_same_semantic_positions"
            )
        prefill, _captures, applications, realized_norms = (
            _prefill_with_layerwise_state_replacements(
                model,
                adapter,
                active_encoding,
                positions=receiver_positions,
                replacements=active_replacements,
                readout_layers=(int(adapter.num_layers) - 1,),
                readout_positions=(registry.query_position,),
            )
        )
        outcomes = _score_and_generate_prefill(
            model,
            tokenizer,
            active_encoding,
            prefill,
            run_greedy=run_greedy,
            max_new_tokens=int(max_new_tokens),
        )
        rows.append(
            {
                "schema_version": COUNT_STREAM_SCHEMA_VERSION,
                "experiment_id": "grammar_timed_terminal_span_decomposition",
                "condition": condition,
                "status": "ok",
                "request_id": clean.request_id,
                "model_label": clean.model_label,
                "seed": clean.seed,
                "dataset_split": clean.split,
                "gold_count": clean.count,
                "answer_site_id": answer_site_id,
                "target_occurrence": len(registry.trace_items),
                "target_is_terminal": True,
                "patch_geometry": active_geometry,
                "patch_layer_mode": "cumulative_clamp",
                "layer": int(layer),
                "patch_layers": list(patch_layers),
                "patch_token_count": (
                    0 if active_geometry == "none" else len(receiver_positions)
                ),
                "receiver_positions_sha256": _sha256_json(receiver_positions),
                "state_replacement_source": replacement_source,
                "state_donor_token_count": len(donor_positions),
                "state_donor_positions_sha256": _sha256_json(donor_positions),
                "matched_random_equal_token_budget": bool(
                    not condition.endswith("_matched_random")
                    or len(donor_positions) == len(receiver_positions)
                ),
                "clean_control_state_delta_norm_by_layer": (
                    {} if active_geometry == "none" else delta_norms[active_geometry]
                ),
                "patch_hook_applications": {
                    str(key): value for key, value in sorted(applications.items())
                },
                "patch_realized_fro_norm_by_layer": {
                    str(key): value for key, value in sorted(realized_norms.items())
                },
                "span_selection_uses_outcome": False,
                "selection_rank_used": False,
                "causal_claim_scope": (
                    "grammar_timed_terminal_semantic_state_sufficiency"
                ),
                "registry_sha256": registry.to_dict()["registry_sha256"],
                "causal_site_plan_schema_version": causal_plan["schema_version"],
                **grammar_audit,
                **control_audit,
                **outcomes,
            }
        )
    return rows
