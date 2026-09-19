"""Frozen targeted-write -> terminal-state -> answer-readout bridge assays.

The behavioral endpoint and the state/readout factorials are intentionally kept
as separate estimands.  This module supplies the missing controlled bridge: it
ablates the frozen targeted-retrieval bank at the final N-1 -> N query while the
clean trace is teacher forced, captures the resulting terminal hidden state,
transfers that state into an otherwise clean receiver, and then applies the
model-specific confirmed readout cut.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import Any, Mapping, Sequence

import torch

from realistic_niah_v4.modeling import (
    DecoderAdapter,
    _encoding_tensors,
    _replace_output_tensor,
    _tensor_from_output,
    capture_post_block_states,
)
from realistic_niah_v4_4_3.interventions import clone_prefill_output_for_scoring

from .count_stream import (
    COUNT_STREAM_SCHEMA_VERSION,
    _prefix_forward,
    _query_forward_from_prefix,
    _score_and_generate_prefill,
    _sha256_json,
    answer_source_mask,
    build_answer_source_registry,
    trace_patch_geometry_positions,
)
from .parsing import prompt_token_ids


INTEGRATED_BRIDGE_READOUT_CONDITIONS = ("natural", "matched_control", "cut")


def _targeted_anchor_role(model_label: str) -> str:
    roles = {"Qwen3-8B": "post_marker", "Gemma4-E4B": "p0_item_end"}
    try:
        return roles[str(model_label)]
    except KeyError as exc:
        raise ValueError(f"No frozen targeted anchor role for {model_label}") from exc


def _final_post_marker_position(
    row: Mapping[str, Any], *, gold_count: int, targeted_site: Mapping[str, Any]
) -> tuple[int, dict[str, Any]]:
    if str(targeted_site.get("request_id")) != str(
        row.get("request_id", row.get("stimulus_id"))
    ):
        raise ValueError("Integrated targeted site belongs to another request")
    if int(targeted_site["from_occurrence"]) != int(gold_count) - 1 or int(
        targeted_site["to_occurrence"]
    ) != int(gold_count):
        raise ValueError(
            "Integrated bridge site must be the frozen final N-1 -> N transition"
        )
    anchor_id = str(targeted_site["anchor_equivalence_id"])
    matched = re.fullmatch(
        rf"{int(gold_count) - 1}->{int(gold_count)}@route-q([0-9]+)", anchor_id
    )
    if matched is None:
        raise ValueError(f"Malformed frozen routed anchor: {anchor_id}")
    query_output_index = int(matched.group(1))
    specification = {
        **dict(targeted_site),
        "query_output_token_index": query_output_index,
        "anchor_equivalence_id": anchor_id,
    }
    full_position = len(prompt_token_ids(row)) + query_output_index
    return int(full_position), specification


def _post_query_receiver_positions(
    targeted_query_position: int, receiver_positions: Sequence[int]
) -> tuple[int, ...]:
    """Return causally downstream state positions or mark the geometry N/A."""

    positions = tuple(int(position) for position in receiver_positions)
    if not positions:
        raise ValueError("not applicable: terminal state receiver is empty")
    post_query = tuple(
        position for position in positions if position > int(targeted_query_position)
    )
    if int(targeted_query_position) > min(positions) or not post_query:
        raise ValueError(
            "not applicable: targeted query must not follow the terminal receiver "
            "and the state span must retain at least one strictly post-query token"
        )
    return post_query


def _validated_heads(
    adapter: DecoderAdapter, heads: Sequence[tuple[int, int]]
) -> dict[int, tuple[int, ...]]:
    by_layer: dict[int, set[int]] = {}
    for raw_layer, raw_head in heads:
        layer = int(raw_layer)
        head = int(raw_head)
        if not 0 <= layer < int(adapter.num_layers):
            raise ValueError(f"Invalid integrated-bridge layer L{layer}")
        if not 0 <= head < int(adapter.num_heads[layer]):
            raise ValueError(f"Invalid integrated-bridge head L{layer}H{head}")
        by_layer.setdefault(layer, set()).add(head)
    return {layer: tuple(sorted(values)) for layer, values in sorted(by_layer.items())}


@torch.inference_mode()
def _capture_states_with_query_head_ablation(
    model: Any,
    adapter: DecoderAdapter,
    encoding: Any,
    *,
    capture_positions: Sequence[int],
    capture_layers: Sequence[int],
    heads: Sequence[tuple[int, int]],
    hook_positions: int | Sequence[int],
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Capture post-block states after one exact pre-o head-bank ablation."""

    by_layer = _validated_heads(adapter, heads)
    if not by_layer:
        _logits, states = capture_post_block_states(
            model,
            adapter,
            encoding,
            capture_positions,
            layers=capture_layers,
        )
        return states, {
            "head_ablation_layer_applications": {},
            "head_ablation_selected_post_zero_max_abs": 0.0,
        }
    expected_length = int(encoding.sequence_length)
    if isinstance(hook_positions, int):
        positions = (int(hook_positions),)
    else:
        positions = tuple(sorted({int(value) for value in hook_positions}))
    if not positions or positions[0] < 0 or positions[-1] >= expected_length:
        raise ValueError("Integrated-bridge head-ablation positions are out of range")
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
                raise RuntimeError("Integrated bridge saw no attention head tensor")
            value = args[0]
            if value.ndim != 3 or int(value.shape[-1]) != expected_width:
                raise RuntimeError("Integrated-bridge pre-o tensor shape changed")
            if int(value.shape[1]) != expected_length or applications[layer] != 0:
                return None
            patched = value.clone()
            for head in layer_heads:
                left = int(head) * head_dim
                patched[:, list(positions), left : left + head_dim] = 0
            selected = torch.cat(
                [
                    patched[
                        :, list(positions), int(head) * head_dim : (int(head) + 1) * head_dim
                    ].detach().reshape(-1)
                    for head in layer_heads
                ]
            )
            maxima[layer] = float(selected.abs().max().float().cpu())
            applications[layer] += 1
            return (patched, *args[1:])

        handles.append(adapter.output_projections[layer].register_forward_pre_hook(hook))
    try:
        _logits, states = capture_post_block_states(
            model,
            adapter,
            encoding,
            capture_positions,
            layers=capture_layers,
        )
    finally:
        for handle in handles:
            handle.remove()
    bad = {layer: count for layer, count in applications.items() if count != 1}
    if bad:
        raise RuntimeError(f"Integrated head ablation did not apply once: {bad}")
    if any(value != 0.0 for value in maxima.values()):
        raise RuntimeError("Integrated selected head slices were not exactly zero")
    return states, {
        "head_ablation_layer_applications": {
            str(layer): int(count) for layer, count in sorted(applications.items())
        },
        "head_ablation_selected_post_zero_max_abs": max(maxima.values()),
        "head_ablation_position_count": len(positions),
        "head_ablation_positions": list(positions),
    }


@torch.inference_mode()
def run_integrated_serial_bridge_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    banks: Sequence[Mapping[str, Any]],
    targeted_site: Mapping[str, Any],
    patch_layers: Sequence[int],
    model_label: str,
    geometry: str = "suffix8",
    relay_layer: int | None = None,
    write_window: str = "exact_query",
    bridge_design: str = "transfer",
    answer_site_id: str = "answer_query_v3",
    run_greedy: bool = True,
    max_new_tokens: int = 16,
) -> list[dict[str, Any]]:
    """Cross frozen write interventions, terminal states, and readout cuts."""

    label = str(model_label)
    if label not in {"Qwen3-8B", "Gemma4-E4B"}:
        raise ValueError(f"No frozen integrated bridge for {label}")
    design = str(bridge_design)
    if design not in {"transfer", "restoration"}:
        raise ValueError(f"Unknown integrated bridge design: {bridge_design}")
    frozen_layers = tuple(int(value) for value in patch_layers)
    if not frozen_layers or tuple(sorted(set(frozen_layers))) != frozen_layers:
        raise ValueError("Integrated patch layers must be unique and increasing")
    if any(not 0 <= layer < int(adapter.num_layers) for layer in frozen_layers):
        raise ValueError("An integrated patch layer is outside model depth")
    if label == "Qwen3-8B":
        if frozen_layers != tuple(range(19, 26)) or int(relay_layer or -1) != 26:
            raise ValueError("Qwen integrated bridge is frozen to L19:25 -> L26")
        relay = 26
    else:
        if frozen_layers != tuple(range(16, 42)) or relay_layer is not None:
            raise ValueError("Gemma integrated bridge is frozen to L16:41")
        relay = None

    encoding, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    gold_count = int(encoding.count)
    if gold_count < 2 or len(registry.trace_items) != gold_count:
        raise ValueError("Integrated bridge requires a complete count >= 2 trace")
    if design == "restoration" and str(geometry) == "full_span":
        receiver_span = registry.trace_items[-1]
        receiver_positions = tuple(
            range(int(receiver_span[0]), int(receiver_span[1]))
        )
        if not receiver_positions:
            raise ValueError("not applicable: terminal trace item is empty")
        geometry_audit = {
            "patch_geometry": "full_span",
            "patch_position_alignment": "same_sequence_exact_positions",
            "patch_token_count": len(receiver_positions),
            "receiver_span": list(receiver_span),
            "receiver_span_token_count": len(receiver_positions),
            "donor_span": list(receiver_span),
            "donor_span_token_count": len(receiver_positions),
            "donor_receiver_span_lengths_equal": True,
            "receiver_patch_positions": list(receiver_positions),
            "donor_patch_positions": list(receiver_positions),
            "receiver_patch_positions_sha256": _sha256_json(receiver_positions),
            "donor_patch_positions_sha256": _sha256_json(receiver_positions),
        }
    else:
        (
            receiver_positions,
            _unused_donor,
            geometry_audit,
        ) = trace_patch_geometry_positions(
            registry,
            receiver_occurrence=gold_count,
            donor_occurrence=gold_count - 1,
            geometry=geometry,
        )
    targeted_query_position, specification = _final_post_marker_position(
        row, gold_count=gold_count, targeted_site=targeted_site
    )
    post_query_receiver_positions = _post_query_receiver_positions(
        targeted_query_position, receiver_positions
    )
    query = int(registry.query_position)
    if str(write_window) == "exact_query":
        write_positions = (targeted_query_position,)
    elif str(write_window) == "query_through_trace":
        write_positions = tuple(range(targeted_query_position, query))
    else:
        raise ValueError(f"Unknown integrated write window: {write_window}")
    terminal_end = int(registry.trace_items[-1][1])
    relay_positions = tuple(range(terminal_end, query + 1)) if relay is not None else ()
    prefix_relay_positions = relay_positions[:-1]
    capture_positions = tuple(
        dict.fromkeys(receiver_positions + relay_positions)
    )
    capture_layers = frozen_layers + (() if relay is None else (relay,))
    lookup = {position: index for index, position in enumerate(capture_positions)}

    donor_states: dict[str, dict[int, torch.Tensor]] = {}
    donor_audits: dict[str, dict[str, Any]] = {}
    normalized_banks: list[dict[str, Any]] = []
    for raw_bank in banks:
        condition = str(raw_bank["condition"])
        repeat = int(raw_bank.get("repeat", 0))
        bank_id = f"{condition}:r{repeat}"
        heads = tuple((int(a), int(b)) for a, b in raw_bank.get("heads", ()))
        states, audit = _capture_states_with_query_head_ablation(
            model,
            adapter,
            encoding,
            capture_positions=capture_positions,
            capture_layers=capture_layers,
            heads=heads,
            hook_positions=write_positions,
        )
        donor_states[bank_id] = {
            layer: states[layer][
                [lookup[position] for position in receiver_positions]
            ].clone()
            for layer in frozen_layers
        }
        donor_audits[bank_id] = audit
        normalized_banks.append(
            {
                "bank_id": bank_id,
                "condition": condition,
                "repeat": repeat,
                "heads": heads,
                "bank_sha256": str(raw_bank.get("bank_sha256", "clean")),
            }
        )
        if condition == "clean" and relay is not None:
            clean_relay_states = states[relay].clone()
    if sum(bank["condition"] == "clean" for bank in normalized_banks) != 1:
        raise ValueError("Integrated bridge requires exactly one clean state donor")
    if sum(bank["condition"] == "selected_bank" for bank in normalized_banks) != 1:
        raise ValueError("Integrated bridge requires exactly one selected bank")
    random_banks = [
        bank for bank in normalized_banks if bank["condition"] == "layer_matched_random"
    ]
    if len(random_banks) != 3:
        raise ValueError("Integrated bridge requires exactly three random banks")

    input_ids, attention_mask = _encoding_tensors(model, encoding)
    clean_prefix_relay = None
    clean_query_relay = None
    if relay is not None:
        clean_prefix_relay = clean_relay_states[
            [lookup[position] for position in prefix_relay_positions]
        ].clone()
        clean_query_relay = clean_relay_states[[lookup[query]]].clone()

    def patched_prefix(
        replacements: Mapping[int, torch.Tensor],
        *,
        reset_relay: bool,
        receiver_heads: Sequence[tuple[int, int]] = (),
    ) -> tuple[Any, dict[int, int], int, float, dict[str, Any]]:
        patch_applications = {layer: 0 for layer in replacements}
        relay_applications = 0
        relay_norm = 0.0
        handles = []
        receiver_by_layer = _validated_heads(adapter, receiver_heads)
        write_applications = {layer: 0 for layer in receiver_by_layer}
        write_maxima = {layer: 0.0 for layer in receiver_by_layer}
        for active_layer, layer_heads in receiver_by_layer.items():
            head_dim = int(adapter.head_dims[active_layer])
            expected_width = int(adapter.num_heads[active_layer]) * head_dim

            def receiver_write_hook(
                _module: Any,
                args: tuple[Any, ...],
                *,
                active_layer: int = active_layer,
                layer_heads: tuple[int, ...] = layer_heads,
                head_dim: int = head_dim,
                expected_width: int = expected_width,
            ) -> tuple[Any, ...] | None:
                if not args or not isinstance(args[0], torch.Tensor):
                    raise RuntimeError("Integrated restoration saw no head tensor")
                value = args[0]
                if value.ndim != 3 or int(value.shape[-1]) != expected_width:
                    raise RuntimeError("Integrated restoration pre-o shape changed")
                if int(value.shape[1]) != query or write_applications[active_layer] != 0:
                    return None
                patched = value.clone()
                for head in layer_heads:
                    left = int(head) * head_dim
                    patched[:, list(write_positions), left : left + head_dim] = 0
                selected = torch.cat(
                    [
                        patched[
                            :,
                            list(write_positions),
                            int(head) * head_dim : (int(head) + 1) * head_dim,
                        ]
                        .detach()
                        .reshape(-1)
                        for head in layer_heads
                    ]
                )
                write_maxima[active_layer] = float(
                    selected.abs().max().float().cpu()
                )
                write_applications[active_layer] += 1
                return (patched, *args[1:])

            handles.append(
                adapter.output_projections[active_layer].register_forward_pre_hook(
                    receiver_write_hook
                )
            )
        for active_layer, replacement_states in sorted(replacements.items()):

            def state_hook(
                _module: Any,
                _args: tuple[Any, ...],
                output: Any,
                *,
                active_layer: int = active_layer,
                replacement_states: torch.Tensor = replacement_states,
            ) -> Any:
                hidden = _tensor_from_output(output)
                if hidden.ndim != 3 or int(hidden.shape[1]) != query:
                    return output
                replacement = replacement_states.to(
                    device=hidden.device, dtype=hidden.dtype
                ).unsqueeze(0)
                before = hidden[:, list(receiver_positions), :]
                if replacement.shape != before.shape:
                    raise RuntimeError("Integrated state replacement shape changed")
                patched = hidden.clone()
                patched[:, list(receiver_positions), :] = replacement
                patch_applications[active_layer] += 1
                return _replace_output_tensor(output, patched)

            handles.append(adapter.layers[active_layer].register_forward_hook(state_hook))
        if reset_relay and prefix_relay_positions:

            def prefix_relay_hook(
                _module: Any, _args: tuple[Any, ...], output: Any
            ) -> Any:
                nonlocal relay_applications, relay_norm
                hidden = _tensor_from_output(output)
                if hidden.ndim != 3 or int(hidden.shape[1]) != query:
                    return output
                replacement = clean_prefix_relay.to(
                    device=hidden.device, dtype=hidden.dtype
                ).unsqueeze(0)
                before = hidden[:, list(prefix_relay_positions), :]
                patched = hidden.clone()
                patched[:, list(prefix_relay_positions), :] = replacement
                relay_norm = float(
                    torch.linalg.vector_norm(before.float() - replacement.float())
                    .detach()
                    .cpu()
                )
                relay_applications += 1
                return _replace_output_tensor(output, patched)

            handles.append(adapter.layers[relay].register_forward_hook(prefix_relay_hook))
        try:
            prefix = _prefix_forward(
                model,
                adapter,
                input_ids[:, :query],
                attention_mask[:, :query],
            )
        finally:
            for handle in handles:
                handle.remove()
        bad = {layer: count for layer, count in patch_applications.items() if count != 1}
        if bad:
            raise RuntimeError(f"Integrated state hooks did not apply once: {bad}")
        if relay_applications != int(bool(reset_relay and prefix_relay_positions)):
            raise RuntimeError("Integrated prefix relay hook count changed")
        bad_write = {
            layer: count
            for layer, count in write_applications.items()
            if count != 1
        }
        if bad_write:
            raise RuntimeError(
                f"Integrated restoration write hooks did not apply once: {bad_write}"
            )
        if any(value != 0.0 for value in write_maxima.values()):
            raise RuntimeError("Integrated restoration head slices were not zero")
        write_audit = {
            "receiver_head_ablation_layer_applications": {
                str(layer): int(count)
                for layer, count in sorted(write_applications.items())
            },
            "receiver_head_ablation_selected_post_zero_max_abs": max(
                write_maxima.values(), default=0.0
            ),
        }
        return (
            prefix,
            patch_applications,
            relay_applications,
            relay_norm,
            write_audit,
        )

    def query_branch(
        prefix: Any, *, mask: torch.Tensor, reset_relay: bool
    ) -> tuple[Any, int, float]:
        applications = 0
        realized_norm = 0.0
        handle = None
        if reset_relay:

            def query_relay_hook(
                _module: Any, _args: tuple[Any, ...], output: Any
            ) -> Any:
                nonlocal applications, realized_norm
                hidden = _tensor_from_output(output)
                if hidden.ndim != 3 or int(hidden.shape[1]) != 1:
                    return output
                replacement = clean_query_relay.to(
                    device=hidden.device, dtype=hidden.dtype
                ).unsqueeze(0)
                before = hidden[:, :1, :]
                patched = hidden.clone()
                patched[:, :1, :] = replacement
                realized_norm = float(
                    torch.linalg.vector_norm(before.float() - replacement.float())
                    .detach()
                    .cpu()
                )
                applications += 1
                return _replace_output_tensor(output, patched)

            handle = adapter.layers[relay].register_forward_hook(query_relay_hook)
        try:
            output = _query_forward_from_prefix(
                model,
                adapter,
                encoding,
                prefix_output=prefix,
                query_attention_mask=mask,
            )
        finally:
            if handle is not None:
                handle.remove()
        if applications != int(reset_relay):
            raise RuntimeError("Integrated query relay hook count changed")
        return output, applications, realized_norm

    common = {
        "schema_version": COUNT_STREAM_SCHEMA_VERSION,
        "experiment_id": (
            "integrated_targeted_state_readout_bridge"
            if design == "transfer"
            else "integrated_targeted_mediator_restoration"
        ),
        "bridge_design": design,
        "request_id": encoding.request_id,
        "model_label": label,
        "seed": int(encoding.seed),
        "dataset_split": encoding.split,
        "gold_count": gold_count,
        "targeted_from_occurrence": int(specification["from_occurrence"]),
        "targeted_to_occurrence": int(specification["to_occurrence"]),
        "targeted_anchor_role": _targeted_anchor_role(label),
        "targeted_anchor_equivalence_id": str(specification["anchor_equivalence_id"]),
        "targeted_query_position": targeted_query_position,
        "write_window": str(write_window),
        "write_ablation_position_count": len(write_positions),
        "write_ablation_stops_before_answer_query": bool(write_positions[-1] < query),
        "targeted_query_in_state_receiver": bool(
            targeted_query_position in set(receiver_positions)
        ),
        "post_query_state_position_count": len(post_query_receiver_positions),
        "post_query_state_positions": list(post_query_receiver_positions),
        "state_patch_layers": list(frozen_layers),
        "state_patch_geometry": geometry,
        "state_transfer_alignment": "same_sequence_exact_positions",
        "teacher_forced_trace": True,
        "readout_application": "answer_query_and_numeric_answer_tokens",
        **geometry_audit,
    }
    rows: list[dict[str, Any]] = []
    if design == "transfer":
        factorial_cells = [
            {
                "receiver_bank": None,
                "mediator_bank": bank,
                "mediator_condition": "transferred_state",
            }
            for bank in normalized_banks
        ]
    else:
        clean_bank = next(
            bank for bank in normalized_banks if bank["condition"] == "clean"
        )
        factorial_cells = [
            {
                "receiver_bank": receiver_bank,
                "mediator_bank": (
                    receiver_bank if mediator_condition == "self_state" else clean_bank
                ),
                "mediator_condition": mediator_condition,
            }
            for receiver_bank in normalized_banks
            for mediator_condition in ("self_state", "clean_state_restore")
        ]
    for cell in factorial_cells:
        receiver_bank = cell["receiver_bank"]
        mediator_bank = cell["mediator_bank"]
        for readout_condition in INTEGRATED_BRIDGE_READOUT_CONDITIONS:
            reset_relay = bool(label == "Qwen3-8B" and readout_condition == "cut")
            mask_condition = {
                "natural": "clean",
                "matched_control": "block_trace_items_matched_control",
                "cut": "block_trace_items",
            }[readout_condition]
            (
                prefix,
                patch_apps,
                prefix_relay_apps,
                prefix_norm,
                receiver_write_audit,
            ) = patched_prefix(
                donor_states[mediator_bank["bank_id"]],
                reset_relay=reset_relay,
                receiver_heads=(
                    () if receiver_bank is None else receiver_bank["heads"]
                ),
            )
            mask, mask_audit = answer_source_mask(registry, condition=mask_condition)
            output, query_relay_apps, query_norm = query_branch(
                clone_prefill_output_for_scoring(prefix),
                mask=mask,
                reset_relay=reset_relay,
            )
            scoring_encoding = replace(
                encoding,
                attention_mask=tuple(int(value) for value in mask[0].tolist()),
            )
            greedy_for_cell = bool(
                run_greedy
                and (
                    design == "transfer"
                    or (
                        receiver_bank is not None
                        and receiver_bank["condition"] == "clean"
                        and cell["mediator_condition"] == "self_state"
                        and readout_condition == "natural"
                    )
                )
            )
            outcomes = _score_and_generate_prefill(
                model,
                tokenizer,
                scoring_encoding,
                output,
                run_greedy=greedy_for_cell,
                max_new_tokens=max_new_tokens,
            )
            rows.append(
                {
                    **common,
                    "write_condition": mediator_bank["condition"],
                    "write_repeat": mediator_bank["repeat"],
                    "bank_sha256": mediator_bank["bank_sha256"],
                    "bank_size": len(mediator_bank["heads"]),
                    "receiver_write_condition": (
                        "clean_receiver"
                        if receiver_bank is None
                        else receiver_bank["condition"]
                    ),
                    "receiver_write_repeat": (
                        0 if receiver_bank is None else receiver_bank["repeat"]
                    ),
                    "receiver_bank_sha256": (
                        "clean_receiver"
                        if receiver_bank is None
                        else receiver_bank["bank_sha256"]
                    ),
                    "mediator_condition": cell["mediator_condition"],
                    "mediator_state_source": mediator_bank["condition"],
                    "mediator_state_source_repeat": mediator_bank["repeat"],
                    "greedy_generation_run": greedy_for_cell,
                    "readout_condition": readout_condition,
                    "mask_condition": mask_condition,
                    "state_patch_hook_applications": {
                        str(layer): int(count) for layer, count in sorted(patch_apps.items())
                    },
                    "prefix_relay_reset_hook_applications": int(prefix_relay_apps),
                    "query_relay_reset_hook_applications": int(query_relay_apps),
                    "relay_reset_realized_fro_norm": float(
                        math.sqrt(prefix_norm * prefix_norm + query_norm * query_norm)
                    ),
                    "status": "ok",
                    **donor_audits[mediator_bank["bank_id"]],
                    **receiver_write_audit,
                    **mask_audit,
                    **outcomes,
                }
            )
    expected = len(factorial_cells) * len(INTEGRATED_BRIDGE_READOUT_CONDITIONS)
    if len(rows) != expected:
        raise RuntimeError(f"Integrated bridge emitted {len(rows)} rows, expected {expected}")
    return rows
