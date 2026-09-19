"""Timing-stratified targeted-query -> counter-state NCC capture.

The mixed-carrier NCC pooled two different computational questions.  This
module keeps them separate:

* ``rank_after_city`` (City -> rank) reads the state immediately *before* the
  teacher-forced rank marker, so the readout cannot trivially contain the
  marker/count token itself.  A four-token pre-marker suffix is retained as a
  registered secondary endpoint.
* ``rank_before_city`` (rank -> City) reads the retrieved-city through commit
  tail, matching the previously frozen estimand.

Only post-block layers strictly above every ablated head layer are captured.
That restriction is necessary because a query-head intervention at layer h
cannot affect a later token's state until a subsequent layer.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from realistic_niah_v4.modeling import DecoderAdapter, capture_post_block_states

from .causal_sites import compile_causal_site_plan
from .count_stream import build_answer_source_registry
from .integrated_bridge import (
    _capture_states_with_query_head_ablation,
    _final_post_marker_position,
)
from .targeted_counter_ncc import NCC_CONDITIONS, _normalize_banks
from .terminal_token_state import _site_positions


STRATIFIED_NCC_ENDPOINTS: dict[str, tuple[str, ...]] = {
    "rank_after_city": ("pre_marker_exact", "pre_marker_suffix4"),
    "rank_before_city": ("city_to_commit",),
}


def grammar_timing(event: Mapping[str, Any]) -> str | None:
    """Map an explicit compiled grammar event to one of the two NCC strata."""

    grammar_class = str(event.get("grammar_class", ""))
    if "rank_after_city" in grammar_class:
        return "rank_after_city"
    if "rank_before_city" in grammar_class:
        return "rank_before_city"
    return None


def stratified_endpoint_positions(
    registry: Any,
    event: Mapping[str, Any],
    *,
    occurrence: int,
    timing: str,
) -> dict[str, tuple[int, ...]]:
    """Compile one event's registered endpoint positions.

    The City -> rank endpoints end at ``pre_marker_state`` and therefore omit
    every token in ``rank_evidence_core_span``.  The rank -> City endpoint is
    the inclusive city-to-commit tail used by the original assay.
    """

    if timing not in STRATIFIED_NCC_ENDPOINTS:
        raise ValueError(f"Unknown stratified NCC timing: {timing}")
    if grammar_timing(event) != timing:
        raise ValueError("Event timing does not match the requested NCC branch")
    index = int(occurrence) - 1
    if not 0 <= index < len(registry.trace_items):
        raise ValueError("Stratified NCC occurrence is outside the trace registry")
    item_start, item_end = (int(value) for value in registry.trace_items[index])
    item_positions = set(range(item_start, item_end))
    sites = event.get("sites", {})

    if timing == "rank_after_city":
        city = _site_positions(sites.get("city_target_span"), role="city_target_span")
        marker = _site_positions(
            sites.get("rank_evidence_core_span"), role="rank_evidence_core_span"
        )
        pre_marker = _site_positions(
            sites.get("pre_marker_state"), role="pre_marker_state"
        )
        if len(pre_marker) != 1:
            raise ValueError("City -> rank requires one exact pre-marker state")
        endpoint = int(pre_marker[0])
        if not int(city[-1]) < endpoint < int(marker[0]):
            raise ValueError(
                "City -> rank endpoint must follow the city and precede the marker"
            )
        suffix4 = tuple(range(endpoint - 3, endpoint + 1))
        positions = {
            "pre_marker_exact": (endpoint,),
            "pre_marker_suffix4": suffix4,
        }
        if set(suffix4) & set(marker):
            raise ValueError("City -> rank suffix4 leaked a rank-marker token")
    else:
        city = _site_positions(sites.get("city_target_span"), role="city_target_span")
        commit = _site_positions(
            sites.get("post_update_commit_state"), role="post_update_commit_state"
        )
        if int(city[0]) > int(commit[-1]):
            raise ValueError("Rank -> City commit precedes the retrieved city")
        positions = {
            "city_to_commit": tuple(range(int(city[0]), int(commit[-1]) + 1))
        }

    if tuple(positions) != STRATIFIED_NCC_ENDPOINTS[timing]:
        raise RuntimeError("Stratified NCC endpoint order changed")
    for name, active in positions.items():
        if not active or not set(active) <= item_positions:
            raise ValueError(f"Stratified endpoint {name} escapes its trace item")
    return positions


def _validate_causal_reach(
    banks: Sequence[Mapping[str, Any]], *, capture_start_layer: int
) -> int:
    selected = next(row for row in banks if row["condition"] == "selected_bank")
    selected_heads = tuple((int(a), int(b)) for a, b in selected["heads"])
    if not selected_heads:
        raise ValueError("Stratified NCC selected bank is empty")
    selected_layers = Counter(layer for layer, _head in selected_heads)
    selected_set = set(selected_heads)
    for random_bank in (
        row for row in banks if row["condition"] == "layer_matched_random"
    ):
        random_heads = tuple((int(a), int(b)) for a, b in random_bank["heads"])
        if Counter(layer for layer, _head in random_heads) != selected_layers:
            raise ValueError("Stratified NCC random bank is not exactly layer matched")
        if selected_set & set(random_heads):
            raise ValueError("Stratified NCC random bank overlaps the selected bank")
    maximum = max(layer for row in banks for layer, _head in row["heads"])
    if int(capture_start_layer) != int(maximum) + 1:
        raise ValueError(
            "Stratified NCC must start exactly one layer above the complete bank"
        )
    return int(maximum)


def _pool_positions(
    states: Mapping[int, torch.Tensor],
    *,
    layers: Sequence[int],
    lookup: Mapping[int, int],
    endpoints: Mapping[str, Sequence[int]],
) -> np.ndarray:
    """Return endpoint x layer x hidden mean-pooled vectors."""

    return np.stack(
        [
            np.stack(
                [
                    states[int(layer)][[lookup[int(position)] for position in positions]]
                    .mean(dim=0)
                    .numpy()
                    for layer in layers
                ],
                axis=0,
            )
            for positions in endpoints.values()
        ],
        axis=0,
    )


@torch.inference_mode()
def capture_stratified_targeted_counter_ncc(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    banks: Sequence[Mapping[str, Any]],
    targeted_site: Mapping[str, Any],
    timing: str,
    capture_start_layer: int,
    selected_bank_size: int,
    answer_site_id: str = "answer_query_v3",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Capture clean bases and intervention endpoints for one timing branch."""

    if timing not in STRATIFIED_NCC_ENDPOINTS:
        raise ValueError(f"Unknown stratified NCC timing: {timing}")
    encoding, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    gold_count = int(encoding.count)
    if gold_count < 2 or len(registry.trace_items) != gold_count:
        raise ValueError("Stratified NCC requires a complete trace with count >= 2")
    targeted_query, specification = _final_post_marker_position(
        row, gold_count=gold_count, targeted_site=targeted_site
    )
    plan = compile_causal_site_plan(row, tokenizer)
    events = list(plan.get("events", ()))
    if len(events) != gold_count:
        raise ValueError("Stratified NCC causal event count disagrees with the trace")
    if grammar_timing(events[-1]) != timing:
        raise ValueError("Frozen final transition belongs to the other NCC branch")
    registered_timing = targeted_site.get("grammar_span_timing_stratum")
    if registered_timing is not None and str(registered_timing) != timing:
        raise ValueError("Frozen targeted-site timing changed")

    endpoint_names = STRATIFIED_NCC_ENDPOINTS[timing]
    event_endpoints: list[dict[str, tuple[int, ...]]] = []
    basis_occurrences: list[int] = []
    basis_grammar_classes: list[str] = []
    for occurrence, event in enumerate(events, start=1):
        if grammar_timing(event) != timing:
            continue
        event_endpoints.append(
            stratified_endpoint_positions(
                registry, event, occurrence=occurrence, timing=timing
            )
        )
        basis_occurrences.append(int(occurrence))
        basis_grammar_classes.append(str(event.get("grammar_class", "")))
    if not event_endpoints or basis_occurrences[-1] != gold_count:
        raise ValueError("Stratified NCC lacks its registered final event")
    final_endpoints = event_endpoints[-1]
    if min(position for span in final_endpoints.values() for position in span) <= int(
        targeted_query
    ):
        raise ValueError("Stratified NCC final endpoint must be downstream of retrieval")

    normalized = _normalize_banks(
        adapter, banks, selected_size=int(selected_bank_size)
    )
    maximum_head_layer = _validate_causal_reach(
        normalized, capture_start_layer=int(capture_start_layer)
    )
    layers = tuple(range(int(capture_start_layer), int(adapter.num_layers)))
    if not layers:
        raise ValueError("Stratified NCC has no causally reachable capture layer")

    union_positions = tuple(
        sorted(
            {
                int(position)
                for endpoints in event_endpoints
                for span in endpoints.values()
                for position in span
            }
        )
    )
    _unused, clean_states = capture_post_block_states(
        model, adapter, encoding, union_positions, layers=layers
    )
    union_lookup = {position: index for index, position in enumerate(union_positions)}
    clean_basis = np.stack(
        [
            _pool_positions(
                clean_states,
                layers=layers,
                lookup=union_lookup,
                endpoints=endpoints,
            )
            for endpoints in event_endpoints
        ],
        axis=0,
    ).astype(np.float16)

    final_positions = tuple(
        sorted({position for span in final_endpoints.values() for position in span})
    )
    final_lookup = {position: index for index, position in enumerate(final_positions)}
    by_condition: list[np.ndarray] = []
    condition_rows: list[dict[str, Any]] = []
    for bank in normalized:
        condition = (
            "clean"
            if bank["condition"] == "clean"
            else "selected_mask"
            if bank["condition"] == "selected_bank"
            else f"random_mask_r{bank['repeat']}"
        )
        if condition == "clean":
            state = clean_states
            lookup = union_lookup
            audit: Mapping[str, Any] = {
                "head_ablation_layer_applications": {},
                "head_ablation_selected_post_zero_max_abs": 0.0,
            }
        else:
            state, audit = _capture_states_with_query_head_ablation(
                model,
                adapter,
                encoding,
                capture_positions=final_positions,
                capture_layers=layers,
                heads=bank["heads"],
                hook_positions=int(targeted_query),
            )
            lookup = final_lookup
        by_condition.append(
            _pool_positions(
                state,
                layers=layers,
                lookup=lookup,
                endpoints=final_endpoints,
            ).astype(np.float16)
        )
        condition_rows.append(
            {
                "condition": condition,
                "receiver_bank_condition": bank["condition"],
                "receiver_bank_repeat": int(bank["repeat"]),
                "receiver_bank_sha256": bank["bank_sha256"],
                "receiver_head_count": len(bank["heads"]),
                "head_ablation_layer_applications": dict(
                    audit.get("head_ablation_layer_applications", {})
                ),
                "head_ablation_selected_post_zero_max_abs": float(
                    audit.get("head_ablation_selected_post_zero_max_abs", 0.0)
                ),
            }
        )
    observed_conditions = tuple(item["condition"] for item in condition_rows)
    if observed_conditions != NCC_CONDITIONS:
        raise RuntimeError(f"Stratified NCC condition order changed: {observed_conditions}")

    # by_condition is C x E x L x H; archive endpoint-first for analysis.
    final_vectors = np.stack(by_condition, axis=0).transpose(1, 0, 2, 3)
    arrays = {
        "clean_basis": clean_basis,
        "final_vectors": final_vectors.astype(np.float16),
        "endpoint_names": np.asarray(endpoint_names),
        "layers": np.asarray(layers, dtype=np.int16),
        "occurrences": np.asarray(basis_occurrences, dtype=np.int16),
        "condition_names": np.asarray(observed_conditions),
    }
    metadata = {
        "schema_version": "realistic_niah_v5_stratified_targeted_counter_ncc_capture_v1",
        "experiment_id": "teacher_forced_stratified_targeted_counter_ncc",
        "request_id": str(encoding.request_id),
        "model_label": str(encoding.model_label),
        "seed": int(encoding.seed),
        "dataset_split": str(encoding.split),
        "gold_count": gold_count,
        "timing_branch": timing,
        "endpoint_names": list(endpoint_names),
        "primary_endpoint": endpoint_names[0],
        "basis_occurrences": basis_occurrences,
        "basis_grammar_classes": basis_grammar_classes,
        "basis_event_count": len(basis_occurrences),
        "capture_start_layer": int(capture_start_layer),
        "maximum_ablated_head_layer": maximum_head_layer,
        "layers": list(layers),
        "all_capture_layers_strictly_above_all_ablated_heads": True,
        "carrier_pooling": "mean_over_registered_endpoint_tokens",
        "city_to_rank_marker_tokens_excluded": timing == "rank_after_city",
        "targeted_query_position": int(targeted_query),
        "targeted_from_occurrence": int(specification["from_occurrence"]),
        "targeted_to_occurrence": int(specification["to_occurrence"]),
        "targeted_anchor_equivalence_id": str(specification["anchor_equivalence_id"]),
        "conditions": condition_rows,
        "teacher_forced_trace_tokens": True,
        "outcome_blind": True,
        "selection_rank_used": False,
        "confirmation_used_for_fit_or_layer_selection": False,
        "causal_claim_scope": "timing_specific_targeted_query_head_mask_to_counter_state_geometry",
        "registry_sha256": registry.to_dict()["registry_sha256"],
        "causal_site_plan_schema_version": plan["schema_version"],
    }
    return arrays, metadata
