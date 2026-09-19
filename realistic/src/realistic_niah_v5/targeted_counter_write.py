"""Teacher-forced targeted retrieval -> grammar counter-carrier write assay.

The generated-suffix bridge mixes two questions: whether the retrieval bank
changes the generated token path and whether it writes a downstream hidden
state.  This module fixes every token in the trace and intervenes only at the
registered targeted query.  It then measures the grammar-specific carrier
(rank marker for rank-after-city; city-to-commit tail for rank-before-city) and
tests whether cumulatively restoring the clean carrier normalizes the later
commit state.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from realistic_niah_v4.modeling import (
    DecoderAdapter,
    _replace_output_tensor,
    _tensor_from_output,
)

from .causal_sites import compile_causal_site_plan
from .count_stream import COUNT_STREAM_SCHEMA_VERSION, _sha256_json, build_answer_source_registry
from .integrated_bridge import (
    _capture_states_with_query_head_ablation,
    _final_post_marker_position,
    _validated_heads,
)
from .terminal_token_state import (
    _grammar_timed_geometry_positions,
    _matched_state_donor_positions,
)


_MODEL_CONTRACTS = {
    "Qwen3-8B": {"source_layer": 19, "num_layers": 36, "bank_size": 128},
    "Gemma4-E4B": {"source_layer": 16, "num_layers": 42, "bank_size": 6},
}


def _rms_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    delta = left.detach().float() - right.detach().float()
    return float(torch.linalg.vector_norm(delta).cpu()) / math.sqrt(delta.numel())


def _head_ablation_positions(
    *,
    query_position: int,
    carrier_positions: Sequence[int],
    scope: str,
) -> tuple[int, ...]:
    """Compile the exact teacher-forced support of a retrieval-head lesion."""

    query = int(query_position)
    carriers = tuple(sorted({int(value) for value in carrier_positions}))
    if not carriers or carriers[0] <= query:
        raise ValueError("Counter carrier must lie strictly after the query")
    if scope == "query_local":
        return (query,)
    if scope == "query_through_carrier":
        return tuple(range(query, carriers[-1] + 1))
    raise ValueError(f"Unknown targeted-counter head-ablation scope: {scope}")


@torch.inference_mode()
def _capture_with_query_ablation_and_carrier_clamp(
    model: Any,
    adapter: DecoderAdapter,
    encoding: Any,
    *,
    capture_positions: Sequence[int],
    capture_layers: Sequence[int],
    heads: Sequence[tuple[int, int]],
    head_ablation_positions: Sequence[int],
    carrier_positions: Sequence[int],
    replacements: Mapping[int, torch.Tensor],
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Apply the frozen head lesion plus a layerwise carrier replacement."""

    carriers = tuple(int(value) for value in carrier_positions)
    if not carriers:
        raise ValueError("Counter write carrier clamp is empty")
    applications = {int(layer): 0 for layer in replacements}
    realized = {int(layer): 0.0 for layer in replacements}
    handles = []
    for raw_layer, raw_states in sorted(replacements.items()):
        layer = int(raw_layer)
        states = raw_states.detach().clone()

        def hook(
            _module: Any,
            _args: tuple[Any, ...],
            output: Any,
            *,
            layer: int = layer,
            states: torch.Tensor = states,
        ) -> Any:
            hidden = _tensor_from_output(output)
            if hidden.ndim != 3 or int(hidden.shape[1]) != int(encoding.sequence_length):
                return output
            replacement = states.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(0)
            before = hidden[:, list(carriers), :]
            if before.shape != replacement.shape:
                raise RuntimeError("Counter carrier clamp shape changed")
            patched = hidden.clone()
            patched[:, list(carriers), :] = replacement
            realized[layer] = _rms_distance(before, replacement)
            applications[layer] += 1
            return _replace_output_tensor(output, patched)

        handles.append(adapter.layers[layer].register_forward_hook(hook))
    try:
        states, head_audit = _capture_states_with_query_head_ablation(
            model,
            adapter,
            encoding,
            capture_positions=capture_positions,
            capture_layers=capture_layers,
            heads=heads,
            hook_positions=tuple(int(value) for value in head_ablation_positions),
        )
    finally:
        for handle in handles:
            handle.remove()
    bad = {layer: count for layer, count in applications.items() if count != 1}
    if bad:
        raise RuntimeError(f"Counter carrier clamps did not apply once: {bad}")
    return states, {
        **head_audit,
        "carrier_clamp_layer_applications": {
            str(layer): int(value) for layer, value in sorted(applications.items())
        },
        "carrier_clamp_realized_rms_by_layer": {
            str(layer): float(value) for layer, value in sorted(realized.items())
        },
    }


@torch.inference_mode()
def run_targeted_counter_write_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    banks: Sequence[Mapping[str, Any]],
    targeted_site: Mapping[str, Any],
    source_layer: int,
    head_ablation_scope: str = "query_local",
    answer_site_id: str = "answer_query_v3",
) -> list[dict[str, Any]]:
    """Mask a targeted query and measure/restore its fixed-trace carrier state."""

    encoding, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    model_label = str(encoding.model_label)
    contract = _MODEL_CONTRACTS.get(model_label)
    if contract is None:
        raise ValueError(f"Unsupported targeted-counter model: {model_label}")
    if int(source_layer) != int(contract["source_layer"]) or int(adapter.num_layers) != int(
        contract["num_layers"]
    ):
        raise ValueError("Targeted-counter write layer contract changed")
    gold_count = int(encoding.count)
    if gold_count < 2 or len(registry.trace_items) != gold_count:
        raise ValueError("Targeted-counter write requires a complete trace")
    targeted_query, specification = _final_post_marker_position(
        row, gold_count=gold_count, targeted_site=targeted_site
    )
    causal_plan = compile_causal_site_plan(row, tokenizer)
    events = list(causal_plan.get("events", ()))
    if len(events) != len(registry.trace_items):
        raise ValueError("Counter write causal event count changed")
    geometries, grammar_audit = _grammar_timed_geometry_positions(
        registry, events[-1]
    )
    if grammar_audit["grammar_timing_stratum"] == "rank_after_city":
        carrier_positions = geometries["marker_core"]
        carrier_component = "marker_core"
    else:
        carrier_positions = geometries["grammar_terminal_update"]
        carrier_component = "city_to_commit_tail"
    boundary_positions = geometries["boundary_commit"]
    if min(carrier_positions) <= int(targeted_query):
        raise ValueError("Counter carrier does not lie strictly after targeted query")
    lesion_positions = _head_ablation_positions(
        query_position=int(targeted_query),
        carrier_positions=carrier_positions,
        scope=str(head_ablation_scope),
    )
    if max(boundary_positions) >= int(registry.query_position):
        raise ValueError("Counter boundary reaches the answer query")
    matched_positions = _matched_state_donor_positions(registry, carrier_positions)
    if len(matched_positions) != len(carrier_positions):
        raise RuntimeError("Counter matched-position control changed token budget")

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for raw in banks:
        condition = str(raw["condition"])
        repeat = int(raw.get("repeat", 0))
        heads = tuple((int(a), int(b)) for a, b in raw.get("heads", ()))
        _validated_heads(adapter, heads)
        identity = (condition, repeat)
        if identity in seen:
            raise ValueError(f"Duplicate counter write bank: {identity}")
        seen.add(identity)
        normalized.append(
            {
                "bank_id": f"{condition}:r{repeat}",
                "condition": condition,
                "repeat": repeat,
                "heads": heads,
                "bank_sha256": str(raw.get("bank_sha256", "clean")),
            }
        )
    counts = {
        name: sum(bank["condition"] == name for bank in normalized)
        for name in ("clean", "selected_bank", "layer_matched_random")
    }
    if counts != {"clean": 1, "selected_bank": 1, "layer_matched_random": 3}:
        raise ValueError(f"Counter write bank contract changed: {counts}")
    selected = next(bank for bank in normalized if bank["condition"] == "selected_bank")
    if len(selected["heads"]) != int(contract["bank_size"]):
        raise ValueError("Counter write selected bank size changed")

    patch_layers = tuple(range(int(source_layer), int(adapter.num_layers) - 1))
    capture_layers = tuple(range(int(source_layer), int(adapter.num_layers)))
    capture_positions = tuple(
        sorted(set(carrier_positions) | set(boundary_positions) | set(matched_positions))
    )
    position_index = {position: index for index, position in enumerate(capture_positions)}
    carrier_indices = [position_index[position] for position in carrier_positions]
    boundary_indices = [position_index[position] for position in boundary_positions]
    matched_indices = [position_index[position] for position in matched_positions]

    captures: dict[str, dict[int, torch.Tensor]] = {}
    audits: dict[str, dict[str, Any]] = {}
    for bank in normalized:
        state, audit = _capture_states_with_query_head_ablation(
            model,
            adapter,
            encoding,
            capture_positions=capture_positions,
            capture_layers=capture_layers,
            heads=bank["heads"],
            hook_positions=lesion_positions,
        )
        captures[bank["bank_id"]] = state
        audits[bank["bank_id"]] = audit

    clean_bank = next(bank for bank in normalized if bank["condition"] == "clean")
    clean_capture = captures[clean_bank["bank_id"]]
    clean_carrier = {
        layer: clean_capture[layer][carrier_indices].clone() for layer in patch_layers
    }
    matched_carrier = {
        layer: clean_capture[layer][matched_indices].clone() for layer in patch_layers
    }
    selected_clean_restore, selected_restore_audit = (
        _capture_with_query_ablation_and_carrier_clamp(
            model,
            adapter,
            encoding,
            capture_positions=capture_positions,
            capture_layers=capture_layers,
            heads=selected["heads"],
            head_ablation_positions=lesion_positions,
            carrier_positions=carrier_positions,
            replacements=clean_carrier,
        )
    )
    selected_matched_control, matched_control_audit = (
        _capture_with_query_ablation_and_carrier_clamp(
            model,
            adapter,
            encoding,
            capture_positions=capture_positions,
            capture_layers=capture_layers,
            heads=selected["heads"],
            head_ablation_positions=lesion_positions,
            carrier_positions=carrier_positions,
            replacements=matched_carrier,
        )
    )

    common = {
        "schema_version": COUNT_STREAM_SCHEMA_VERSION,
        "experiment_id": "teacher_forced_targeted_counter_write",
        "request_id": str(encoding.request_id),
        "model_label": model_label,
        "seed": int(encoding.seed),
        "dataset_split": str(encoding.split),
        "gold_count": gold_count,
        "answer_site_id": answer_site_id,
        "targeted_query_position": int(targeted_query),
        "targeted_from_occurrence": int(specification["from_occurrence"]),
        "targeted_to_occurrence": int(specification["to_occurrence"]),
        "targeted_anchor_equivalence_id": str(specification["anchor_equivalence_id"]),
        "source_layer": int(source_layer),
        "patch_layers": list(patch_layers),
        "carrier_component": carrier_component,
        "carrier_positions": list(carrier_positions),
        "carrier_positions_sha256": _sha256_json(carrier_positions),
        "carrier_token_count": len(carrier_positions),
        "boundary_positions": list(boundary_positions),
        "boundary_positions_sha256": _sha256_json(boundary_positions),
        "matched_position_token_count": len(matched_positions),
        "matched_positions_sha256": _sha256_json(matched_positions),
        "teacher_forced_trace_tokens": True,
        "query_local_head_mask": str(head_ablation_scope) == "query_local",
        "head_ablation_scope": str(head_ablation_scope),
        "head_ablation_positions": list(lesion_positions),
        "head_ablation_position_count": len(lesion_positions),
        "head_ablation_positions_sha256": _sha256_json(lesion_positions),
        "selection_rank_used": False,
        "outcome_blind": True,
        "causal_claim_scope": (
            "targeted_query_to_grammar_counter_carrier_to_commit_state"
            if str(head_ablation_scope) == "query_local"
            else "targeted_query_through_carrier_to_counter_carrier_to_commit_state"
        ),
        "registry_sha256": registry.to_dict()["registry_sha256"],
        **grammar_audit,
    }

    def row_for(
        condition: str,
        bank: Mapping[str, Any],
        state: Mapping[int, torch.Tensor],
        audit: Mapping[str, Any],
        *,
        mediator: str,
    ) -> dict[str, Any]:
        carrier_distances = {
            str(layer): _rms_distance(
                state[layer][carrier_indices], clean_capture[layer][carrier_indices]
            )
            for layer in capture_layers
        }
        boundary_final = _rms_distance(
            state[int(adapter.num_layers) - 1][boundary_indices],
            clean_capture[int(adapter.num_layers) - 1][boundary_indices],
        )
        return {
            **common,
            "condition": condition,
            "status": "ok",
            "receiver_bank_condition": str(bank["condition"]),
            "receiver_bank_repeat": int(bank["repeat"]),
            "receiver_bank_sha256": str(bank["bank_sha256"]),
            "receiver_heads": [list(value) for value in bank["heads"]],
            "mediator_condition": mediator,
            "carrier_state_rms_distance_to_clean_by_layer": carrier_distances,
            "carrier_state_rms_distance_mean_downstream": float(
                sum(carrier_distances[str(layer)] for layer in capture_layers)
                / len(capture_layers)
            ),
            "boundary_state_rms_distance_to_clean_final": boundary_final,
            **dict(audit),
        }

    rows = [
        row_for(
            "clean",
            clean_bank,
            clean_capture,
            audits[clean_bank["bank_id"]],
            mediator="self_state",
        )
    ]
    for bank in normalized:
        if bank["condition"] == "clean":
            continue
        condition = (
            "selected_mask"
            if bank["condition"] == "selected_bank"
            else f"random_mask_r{int(bank['repeat'])}"
        )
        rows.append(
            row_for(
                condition,
                bank,
                captures[bank["bank_id"]],
                audits[bank["bank_id"]],
                mediator="self_state",
            )
        )
    rows.append(
        row_for(
            "selected_mask_clean_carrier_restore",
            selected,
            selected_clean_restore,
            selected_restore_audit,
            mediator="clean_same_semantic_carrier",
        )
    )
    rows.append(
        row_for(
            "selected_mask_matched_position_state_control",
            selected,
            selected_matched_control,
            matched_control_audit,
            mediator="clean_equal_token_near_depth_nonitem_state",
        )
    )
    if len(rows) != 7:
        raise RuntimeError("Counter write factorial changed")
    return rows
