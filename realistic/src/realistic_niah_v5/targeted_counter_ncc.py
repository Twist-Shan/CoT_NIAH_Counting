"""Hidden-vector capture for the targeted-query -> counter-carrier NCC assay.

The earlier teacher-forced write experiment only retained RMS distances.  This
module retains the actual grammar-timed carrier vectors.  Clean vectors from
every completed item in the discovery trace form the nearest-centroid basis;
the frozen final transition is then re-run under the selected and three
layer-matched-random retrieval-head masks.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from realistic_niah_v4.modeling import DecoderAdapter, capture_post_block_states

from .causal_sites import compile_causal_site_plan
from .count_stream import build_answer_source_registry
from .integrated_bridge import (
    _capture_states_with_query_head_ablation,
    _final_post_marker_position,
    _validated_heads,
)
from .terminal_token_state import _site_positions


NCC_CONDITIONS = (
    "clean",
    "selected_mask",
    "random_mask_r1",
    "random_mask_r2",
    "random_mask_r3",
)


def transition_carrier_positions(
    registry: Any,
    event: Mapping[str, Any],
    *,
    occurrence: int,
) -> tuple[tuple[int, ...], str, str]:
    """Return the grammar-timed carrier for one completed trace item.

    Rank-after-city traces carry the update in the rank marker.  Rank-before-
    city traces carry it from the retrieved city through the registered commit.
    The assay deliberately rejects implicit grammars: the frozen 5.1 panel is
    balanced over these two explicit timing strata, so silently mixing a third
    token geometry would change the estimand.
    """

    index = int(occurrence) - 1
    if not 0 <= index < len(registry.trace_items):
        raise ValueError("Carrier occurrence is outside the trace registry")
    item_start, item_end = registry.trace_items[index]
    sites = event.get("sites", {})
    marker = _site_positions(
        sites.get("rank_evidence_core_span"), role="rank_evidence_core_span"
    )
    city = _site_positions(sites.get("city_target_span"), role="city_target_span")
    commit = _site_positions(
        sites.get("post_update_commit_state"), role="post_update_commit_state"
    )
    grammar_class = str(event.get("grammar_class", ""))
    if "rank_after_city" in grammar_class:
        positions = tuple(marker)
        component = "marker_core"
        timing = "rank_after_city"
    elif "rank_before_city" in grammar_class:
        positions = tuple(range(int(city[0]), int(commit[-1]) + 1))
        component = "city_to_commit_tail"
        timing = "rank_before_city"
    else:
        raise ValueError(
            "Targeted-counter NCC requires an explicit rank-before/after carrier, "
            f"observed {grammar_class!r}"
        )
    if not positions or not set(positions) <= set(range(int(item_start), int(item_end))):
        raise ValueError("Grammar-timed carrier escapes its registered trace item")
    return positions, component, timing


def _normalize_banks(
    adapter: DecoderAdapter,
    banks: Sequence[Mapping[str, Any]],
    *,
    selected_size: int,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in banks:
        condition = str(raw["condition"])
        repeat = int(raw.get("repeat", 0))
        heads = tuple((int(a), int(b)) for a, b in raw.get("heads", ()))
        _validated_heads(adapter, heads)
        normalized.append(
            {
                "condition": condition,
                "repeat": repeat,
                "heads": heads,
                "bank_sha256": str(raw.get("bank_sha256", "clean")),
            }
        )
    counts = {
        name: sum(row["condition"] == name for row in normalized)
        for name in ("clean", "selected_bank", "layer_matched_random")
    }
    if counts != {"clean": 1, "selected_bank": 1, "layer_matched_random": 3}:
        raise ValueError(f"NCC bank factorial changed: {counts}")
    selected = next(row for row in normalized if row["condition"] == "selected_bank")
    if len(selected["heads"]) != int(selected_size):
        raise ValueError("NCC selected-bank size changed")
    clean = next(row for row in normalized if row["condition"] == "clean")
    randoms = sorted(
        (row for row in normalized if row["condition"] == "layer_matched_random"),
        key=lambda row: int(row["repeat"]),
    )
    return [clean, selected, *randoms]


@torch.inference_mode()
def capture_targeted_counter_ncc(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    banks: Sequence[Mapping[str, Any]],
    targeted_site: Mapping[str, Any],
    source_layer: int,
    selected_bank_size: int,
    answer_site_id: str = "answer_query_v3",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Capture discovery-basis and final-intervention carrier vectors."""

    encoding, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    gold_count = int(encoding.count)
    if gold_count < 2 or len(registry.trace_items) != gold_count:
        raise ValueError("NCC requires a complete trace with at least two items")
    targeted_query, specification = _final_post_marker_position(
        row, gold_count=gold_count, targeted_site=targeted_site
    )
    plan = compile_causal_site_plan(row, tokenizer)
    events = list(plan.get("events", ()))
    if len(events) != gold_count:
        raise ValueError("NCC causal event count disagrees with the trace")

    carriers: list[tuple[int, ...]] = []
    components: list[str] = []
    timings: list[str] = []
    for occurrence, event in enumerate(events, start=1):
        positions, component, timing = transition_carrier_positions(
            registry, event, occurrence=occurrence
        )
        carriers.append(positions)
        components.append(component)
        timings.append(timing)
    final_positions = carriers[-1]
    if min(final_positions) <= int(targeted_query):
        raise ValueError("The final NCC carrier must be downstream of retrieval")
    if timings[-1] != str(
        targeted_site.get("grammar_span_timing_stratum", timings[-1])
    ):
        # The targeted registry normally has no timing field; when one is
        # supplied, treat it as an immutable audit field.
        raise ValueError("Frozen NCC timing stratum changed")

    normalized = _normalize_banks(
        adapter, banks, selected_size=int(selected_bank_size)
    )
    layers = tuple(range(int(source_layer), int(adapter.num_layers)))
    union_positions = tuple(sorted({p for span in carriers for p in span}))
    _unused, clean_states = capture_post_block_states(
        model, adapter, encoding, union_positions, layers=layers
    )
    union_index = {position: index for index, position in enumerate(union_positions)}
    clean_basis = np.stack(
        [
            np.stack(
                [
                    clean_states[layer][[union_index[p] for p in positions]]
                    .mean(dim=0)
                    .numpy()
                    for layer in layers
                ],
                axis=0,
            )
            for positions in carriers
        ],
        axis=0,
    ).astype(np.float16)

    final_vectors: list[np.ndarray] = []
    condition_rows: list[dict[str, Any]] = []
    final_union_indices = [union_index[p] for p in final_positions]
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
            audit: Mapping[str, Any] = {
                "head_hook_applications": {},
                "head_zeroed_output_norm_by_layer": {},
            }
            vector = np.stack(
                [state[layer][final_union_indices].mean(dim=0).numpy() for layer in layers],
                axis=0,
            )
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
            vector = np.stack(
                [state[layer].mean(dim=0).numpy() for layer in layers], axis=0
            )
        final_vectors.append(vector.astype(np.float16))
        condition_rows.append(
            {
                "condition": condition,
                "receiver_bank_condition": bank["condition"],
                "receiver_bank_repeat": int(bank["repeat"]),
                "receiver_bank_sha256": bank["bank_sha256"],
                "receiver_head_count": len(bank["heads"]),
                "head_hook_applications": dict(
                    audit.get(
                        "head_ablation_layer_applications",
                        audit.get("head_hook_applications", {}),
                    )
                ),
                "head_ablation_selected_post_zero_max_abs": float(
                    audit.get("head_ablation_selected_post_zero_max_abs", 0.0)
                ),
            }
        )
    observed_conditions = tuple(row["condition"] for row in condition_rows)
    if set(observed_conditions) != set(NCC_CONDITIONS):
        raise RuntimeError(f"NCC condition order changed: {observed_conditions}")

    arrays = {
        "clean_basis": clean_basis,
        "final_vectors": np.stack(final_vectors, axis=0).astype(np.float16),
        "layers": np.asarray(layers, dtype=np.int16),
        "occurrences": np.arange(1, gold_count + 1, dtype=np.int16),
        "condition_names": np.asarray(observed_conditions),
    }
    metadata = {
        "schema_version": "realistic_niah_v5_targeted_counter_ncc_capture_v1",
        "experiment_id": "teacher_forced_targeted_counter_ncc",
        "request_id": str(encoding.request_id),
        "model_label": str(encoding.model_label),
        "seed": int(encoding.seed),
        "dataset_split": str(encoding.split),
        "gold_count": gold_count,
        "source_layer": int(source_layer),
        "layers": list(layers),
        "carrier_components": components,
        "grammar_timing_strata": timings,
        "final_carrier_component": components[-1],
        "final_grammar_timing_stratum": timings[-1],
        "carrier_pooling": "mean_over_registered_grammar_timed_tokens",
        "targeted_query_position": int(targeted_query),
        "targeted_from_occurrence": int(specification["from_occurrence"]),
        "targeted_to_occurrence": int(specification["to_occurrence"]),
        "targeted_anchor_equivalence_id": str(specification["anchor_equivalence_id"]),
        "conditions": condition_rows,
        "teacher_forced_trace_tokens": True,
        "outcome_blind": True,
        "selection_rank_used": False,
        "confirmation_used_for_fit_or_layer_selection": False,
        "causal_claim_scope": "targeted_query_head_mask_to_carrier_count_geometry",
        "registry_sha256": registry.to_dict()["registry_sha256"],
        "causal_site_plan_schema_version": plan["schema_version"],
    }
    return arrays, metadata
