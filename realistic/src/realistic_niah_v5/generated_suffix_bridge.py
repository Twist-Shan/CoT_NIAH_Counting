"""Free-running targeted retrieval -> terminal token -> state -> count bridge.

The earlier integrated bridge teacher-forced the clean terminal item after the
targeted retrieval query.  That can hide a genuine retrieval-to-state edge:
even when the retrieval heads are ablated, the correct city tokens are still
fed back into the model.  This module instead greedily generates a fixed-length
suffix after the frozen query, replays those generated tokens at the original
positions, and then mediates their answer effect through a preregistered
cumulative full-state clamp.  The same primitive supports Gemma's confirmed
L16:41 terminal-item bridge and Qwen's L19:35 geometry diagnostics.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import torch

from realistic_niah_v4.modeling import (
    DecoderAdapter,
    _accepts_keyword,
    _bounded_logits_kwargs,
    _encoding_tensors,
    _extract_shared_kv_states,
    capture_post_block_states,
)
from realistic_niah_v4_4_5.restoration import _generation_eos_ids

from .count_stream import (
    COUNT_STREAM_SCHEMA_VERSION,
    NativeTraceEncoding,
    _full_state_patch_layers,
    _prefill_with_layerwise_state_replacements,
    _prefix_forward,
    _score_and_generate_prefill,
    _sha256_json,
    build_answer_source_registry,
)
from .causal_sites import compile_causal_site_plan
from .integrated_bridge import _final_post_marker_position, _validated_heads
from .terminal_token_state import (
    _grammar_timed_geometry_positions,
    _matched_state_donor_positions,
)


REGISTERED_GENERATED_SUFFIX_STATE_CONDITIONS = (
    "clean_generated_self_state",
    "selected_generated_self_state",
    "layer_matched_random_generated_self_state",
    "selected_generated_clean_state_restore",
    "layer_matched_random_generated_clean_state_restore",
    "clean_generated_selected_state_occlusion",
)

REGISTERED_STATE_PATCH_GEOMETRIES = (
    "terminal_span",
    "generated_suffix_span",
    "terminal_prefix_span",
)

REGISTERED_TARGETED_COUNTER_GEOMETRIES = (
    "grammar_counter_carrier",
    "grammar_counter_tail",
    "terminal_last4",
    "terminal_last8",
)

_MODEL_STATE_CONTRACTS = {
    "Gemma4-E4B": {"num_layers": 42, "source_layer": 16, "bank_size": 6},
    "Qwen3-8B": {"num_layers": 36, "source_layer": 19, "bank_size": 128},
}


def _replace_fixed_suffix(
    encoding: NativeTraceEncoding,
    *,
    start: int,
    stop: int,
    token_ids: Sequence[int],
) -> NativeTraceEncoding:
    """Replace ``[start, stop)`` without changing sequence length or positions."""

    left = int(start)
    right = int(stop)
    replacement = tuple(int(value) for value in token_ids)
    if not 0 <= left < right <= int(encoding.sequence_length):
        raise ValueError("Generated suffix replay range is invalid")
    if len(replacement) != right - left:
        raise ValueError("Generated suffix replay must preserve the token budget")
    active = list(int(value) for value in encoding.input_ids)
    active[left:right] = replacement
    if len(active) != int(encoding.sequence_length):
        raise RuntimeError("Generated suffix replay changed sequence length")
    return replace(encoding, input_ids=tuple(active))


@torch.inference_mode()
def _prefill_through_targeted_query(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    targeted_query_position: int,
    heads: Sequence[tuple[int, int]],
) -> tuple[Any, dict[str, Any]]:
    """Prefill through one frozen query while zeroing exact pre-O head slices."""

    query = int(targeted_query_position)
    prefix_length = query + 1
    if not 0 <= query < int(encoding.query_position):
        raise ValueError("Targeted retrieval query must precede the answer query")
    grouped = _validated_heads(adapter, heads)
    applications = {layer: 0 for layer in grouped}
    maxima = {layer: 0.0 for layer in grouped}
    handles = []
    for layer, layer_heads in grouped.items():
        width = int(adapter.head_dims[layer])
        expected_width = int(adapter.num_heads[layer]) * width

        def hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
            layer_heads: tuple[int, ...] = layer_heads,
            width: int = width,
            expected_width: int = expected_width,
        ) -> tuple[Any, ...] | None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Generated-suffix bridge saw no pre-O tensor")
            value = args[0]
            if value.ndim != 3 or int(value.shape[-1]) != expected_width:
                raise RuntimeError("Generated-suffix pre-O tensor shape changed")
            if int(value.shape[1]) != prefix_length or applications[layer] != 0:
                return None
            patched = value.clone()
            for head in layer_heads:
                left = int(head) * width
                patched[:, query, left : left + width] = 0
            selected = torch.cat(
                [
                    patched[:, query, int(head) * width : (int(head) + 1) * width]
                    .detach()
                    .reshape(-1)
                    for head in layer_heads
                ]
            )
            maxima[layer] = float(selected.abs().max().float().cpu())
            applications[layer] += 1
            return (patched, *args[1:])

        handles.append(adapter.output_projections[layer].register_forward_pre_hook(hook))
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        output = _prefix_forward(
            model,
            adapter,
            input_ids[:, :prefix_length],
            attention_mask[:, :prefix_length],
        )
    finally:
        for handle in handles:
            handle.remove()
    bad = {layer: count for layer, count in applications.items() if count != 1}
    if bad:
        raise RuntimeError(f"Generated-suffix head hooks did not apply once: {bad}")
    if any(value != 0.0 for value in maxima.values()):
        raise RuntimeError("Generated-suffix selected head slices were not zero")
    return output, {
        "head_ablation_layer_applications": {
            str(layer): int(count) for layer, count in sorted(applications.items())
        },
        "head_ablation_selected_post_zero_max_abs": max(maxima.values(), default=0.0),
    }


@torch.inference_mode()
def _greedy_fixed_token_budget_from_prefill(
    model: Any,
    tokenizer: Any,
    *,
    prefill_output: Any,
    prefix_attention_mask: Sequence[int],
    token_budget: int,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Greedily decode exactly ``token_budget`` tokens, retaining early EOS.

    Continuing after an early EOS is deliberate: every arm must replay the same
    number of positions.  Early EOS is recorded as a diagnostic rather than
    becoming an outcome-dependent exclusion.
    """

    budget = int(token_budget)
    if budget < 1:
        raise ValueError("Generated suffix token budget must be positive")
    logits = getattr(prefill_output, "logits", None)
    past = getattr(prefill_output, "past_key_values", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise RuntimeError("Generated-suffix prefill returned no logits")
    if int(logits.shape[0]) != 1 or past is None:
        raise RuntimeError("Generated-suffix prefill returned no reusable cache")
    next_logits = logits[0, -1].detach().float()
    device = next_logits.device
    base_mask = torch.tensor(
        [tuple(int(value) for value in prefix_attention_mask)],
        dtype=torch.long,
        device=device,
    )
    prefix_length = int(base_mask.shape[1])
    shared = _extract_shared_kv_states(prefill_output)
    generated: list[int] = []
    eos_set = set(_generation_eos_ids(model, tokenizer))
    first_eos_offset: int | None = None
    for step in range(budget):
        token = int(torch.argmax(next_logits).item())
        generated.append(token)
        if token in eos_set and first_eos_offset is None:
            first_eos_offset = int(step)
        if step == budget - 1:
            break
        token_ids = torch.tensor([[token]], dtype=torch.long, device=device)
        generated_mask = torch.ones(
            (1, len(generated)), dtype=base_mask.dtype, device=device
        )
        attention_mask = torch.cat((base_mask, generated_mask), dim=1)
        position = prefix_length + int(step)
        kwargs: dict[str, Any] = {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
            "past_key_values": past,
            "use_cache": True,
            **_bounded_logits_kwargs(model),
        }
        if _accepts_keyword(model, "position_ids"):
            kwargs["position_ids"] = torch.tensor(
                [[position]], dtype=torch.long, device=device
            )
        if _accepts_keyword(model, "cache_position"):
            kwargs["cache_position"] = torch.tensor(
                [position], dtype=torch.long, device=device
            )
        if shared is not None and _accepts_keyword(model, "shared_kv_states"):
            kwargs["shared_kv_states"] = shared
        output = model(**kwargs)
        updated_past = getattr(output, "past_key_values", None)
        if updated_past is None:
            raise RuntimeError("Generated-suffix cached step returned no KV cache")
        past = updated_past
        updated_shared = _extract_shared_kv_states(output)
        if updated_shared is not None:
            shared = updated_shared
        step_logits = getattr(output, "logits", None)
        if not isinstance(step_logits, torch.Tensor) or step_logits.ndim != 3:
            raise RuntimeError("Generated-suffix cached step returned no logits")
        next_logits = step_logits[0, -1].detach().float()
    return tuple(generated), {
        "fixed_token_budget": budget,
        "generated_suffix_token_count": len(generated),
        "eos_token_ids": sorted(eos_set),
        "early_eos_generated": first_eos_offset is not None,
        "first_eos_offset": first_eos_offset,
        "generation_ignored_early_eos_for_alignment": first_eos_offset is not None,
    }


def _token_accuracy(
    generated: Sequence[int],
    reference: Sequence[int],
    offsets: Sequence[int] | None = None,
) -> float:
    left = tuple(int(value) for value in generated)
    right = tuple(int(value) for value in reference)
    if len(left) != len(right) or not left:
        raise ValueError("Token accuracy requires equal nonempty sequences")
    selected = tuple(range(len(left))) if offsets is None else tuple(int(x) for x in offsets)
    if not selected:
        return 1.0
    if min(selected) < 0 or max(selected) >= len(left):
        raise ValueError("Token-accuracy offsets are out of range")
    return float(sum(left[index] == right[index] for index in selected) / len(selected))


def _causal_terminal_suffix_positions(
    *,
    terminal_start: int,
    terminal_stop: int,
    replay_start: int,
) -> tuple[int, ...]:
    """Return terminal-item positions strictly downstream of the frozen query."""

    left = max(int(terminal_start), int(replay_start))
    right = int(terminal_stop)
    if left >= right:
        raise ValueError("Targeted query leaves no causal terminal-item suffix")
    return tuple(range(left, right))


@torch.inference_mode()
def run_generated_suffix_state_bridge_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    banks: Sequence[Mapping[str, Any]],
    targeted_site: Mapping[str, Any],
    source_layer: int = 16,
    state_patch_geometry: str = "terminal_span",
    include_matched_position_control: bool = False,
    answer_site_id: str = "answer_query_v3",
    run_greedy_answer: bool = True,
    max_new_tokens: int = 16,
) -> list[dict[str, Any]]:
    """Test targeted heads -> generated suffix -> residual state -> answer."""

    encoding, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    model_label = str(encoding.model_label)
    if model_label not in _MODEL_STATE_CONTRACTS:
        raise ValueError(f"Unsupported generated-suffix model: {model_label}")
    contract = _MODEL_STATE_CONTRACTS[model_label]
    if (
        int(source_layer) != int(contract["source_layer"])
        or int(adapter.num_layers) != int(contract["num_layers"])
    ):
        raise ValueError(
            "Generated-suffix state contract changed: "
            f"{model_label} requires L{contract['source_layer']}:"
            f"{int(contract['num_layers']) - 1}"
        )
    geometry = str(state_patch_geometry)
    if geometry not in (
        *REGISTERED_STATE_PATCH_GEOMETRIES,
        *REGISTERED_TARGETED_COUNTER_GEOMETRIES,
    ):
        raise ValueError(f"Unknown generated-suffix state geometry: {geometry}")
    gold_count = int(encoding.count)
    if gold_count < 2 or len(registry.trace_items) != gold_count:
        raise ValueError("Generated-suffix bridge requires a complete count >= 2 trace")
    targeted_query, specification = _final_post_marker_position(
        row, gold_count=gold_count, targeted_site=targeted_site
    )
    terminal_start, terminal_stop = registry.trace_items[-1]
    replay_start = int(targeted_query) + 1
    replay_stop = int(terminal_stop)
    if not 0 <= replay_start < replay_stop < int(registry.query_position):
        raise ValueError("Generated terminal suffix geometry is not causal")
    replay_positions = tuple(range(replay_start, replay_stop))
    terminal_positions = _causal_terminal_suffix_positions(
        terminal_start=int(terminal_start),
        terminal_stop=replay_stop,
        replay_start=replay_start,
    )
    base_geometries = {
        "terminal_span": terminal_positions,
        "generated_suffix_span": replay_positions,
        "terminal_prefix_span": tuple(
            range(max(int(terminal_start), replay_start), int(registry.query_position))
        ),
    }
    semantic_audit: dict[str, Any] = {
        "counter_geometry_is_grammar_specific": False,
        "counter_carrier_component": "none",
    }
    if geometry in REGISTERED_TARGETED_COUNTER_GEOMETRIES:
        causal_plan = compile_causal_site_plan(row, tokenizer)
        events = list(causal_plan.get("events", ()))
        if len(events) != len(registry.trace_items):
            raise ValueError(
                "Generated-suffix counter geometry disagrees with causal event count"
            )
        semantic_geometries, grammar_audit = _grammar_timed_geometry_positions(
            registry, events[-1]
        )
        if geometry == "grammar_counter_carrier":
            if grammar_audit["grammar_timing_stratum"] == "rank_after_city":
                mediator_positions = semantic_geometries["marker_core"]
                carrier_component = "marker_core"
            else:
                mediator_positions = semantic_geometries["grammar_terminal_update"]
                carrier_component = "city_to_commit_tail"
        elif geometry == "grammar_counter_tail":
            mediator_positions = semantic_geometries["grammar_terminal_update"]
            carrier_component = "grammar_terminal_update"
        elif geometry == "terminal_last4":
            mediator_positions = terminal_positions[-min(4, len(terminal_positions)) :]
            carrier_component = "last4_terminal_tokens"
        elif geometry == "terminal_last8":
            mediator_positions = terminal_positions[-min(8, len(terminal_positions)) :]
            carrier_component = "last8_terminal_tokens"
        else:  # pragma: no cover - guarded by the registered tuple
            raise RuntimeError(f"Unhandled targeted-counter geometry: {geometry}")
        semantic_audit = {
            **grammar_audit,
            "counter_geometry_is_grammar_specific": bool(
                geometry in {"grammar_counter_carrier", "grammar_counter_tail"}
            ),
            "counter_carrier_component": carrier_component,
        }
    else:
        mediator_positions = base_geometries[geometry]
    if (
        not mediator_positions
        or min(mediator_positions) < replay_start
        or max(mediator_positions) >= int(registry.query_position)
    ):
        raise ValueError("Generated-suffix mediator geometry is not causal")
    matched_position_donors = (
        _matched_state_donor_positions(registry, mediator_positions)
        if include_matched_position_control
        else ()
    )
    reference_suffix = tuple(
        int(encoding.input_ids[position]) for position in replay_positions
    )
    nonmarkers = set(registry.positions("trace_nonmarkers"))
    terminal_nonmarker_offsets = tuple(
        position - replay_start
        for position in terminal_positions
        if position in nonmarkers
    )
    if not terminal_nonmarker_offsets:
        raise ValueError("Generated-suffix bridge found no terminal nonmarker tokens")

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for raw in banks:
        condition = str(raw["condition"])
        repeat = int(raw.get("repeat", 0))
        heads = tuple((int(a), int(b)) for a, b in raw.get("heads", ()))
        _validated_heads(adapter, heads)
        identity = (condition, repeat)
        if identity in seen:
            raise ValueError(f"Duplicate generated-suffix bank arm: {identity}")
        seen.add(identity)
        if condition == "clean" and heads:
            raise ValueError("Clean generated-suffix arm must have no heads")
        if condition != "clean" and not heads:
            raise ValueError("Ablated generated-suffix arm requires heads")
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
        raise ValueError(f"Generated-suffix bank contract changed: {counts}")
    selected_size = len(
        next(bank for bank in normalized if bank["condition"] == "selected_bank")[
            "heads"
        ]
    )
    if selected_size != int(contract["bank_size"]):
        raise ValueError(
            f"{model_label} generated-suffix bank must have "
            f"{contract['bank_size']} heads, observed {selected_size}"
        )

    replay_by_bank: dict[str, NativeTraceEncoding] = {}
    generation_audit_by_bank: dict[str, dict[str, Any]] = {}
    states_by_bank: dict[str, dict[int, torch.Tensor]] = {}
    patch_layers = _full_state_patch_layers(
        source_layer=int(source_layer),
        num_layers=int(adapter.num_layers),
        layer_mode="cumulative_clamp",
    )
    for bank in normalized:
        prefix, head_audit = _prefill_through_targeted_query(
            model,
            adapter,
            encoding,
            targeted_query_position=int(targeted_query),
            heads=bank["heads"],
        )
        generated, generation_audit = _greedy_fixed_token_budget_from_prefill(
            model,
            tokenizer,
            prefill_output=prefix,
            prefix_attention_mask=encoding.attention_mask[:replay_start],
            token_budget=len(replay_positions),
        )
        replay = _replace_fixed_suffix(
            encoding,
            start=replay_start,
            stop=replay_stop,
            token_ids=generated,
        )
        replay_by_bank[bank["bank_id"]] = replay
        generation_audit_by_bank[bank["bank_id"]] = {
            **head_audit,
            **generation_audit,
            "generated_suffix_token_ids": list(generated),
            "generated_suffix_sha256": _sha256_json(generated),
            "reference_suffix_exact": bool(generated == reference_suffix),
            "reference_suffix_token_accuracy": _token_accuracy(
                generated, reference_suffix
            ),
            "reference_terminal_nonmarker_token_accuracy": _token_accuracy(
                generated,
                reference_suffix,
                terminal_nonmarker_offsets,
            ),
        }
        _unused, captured = capture_post_block_states(
            model,
            adapter,
            replay,
            mediator_positions,
            layers=patch_layers,
        )
        states_by_bank[bank["bank_id"]] = {
            layer: captured[layer].clone() for layer in patch_layers
        }

    clean_bank = next(bank for bank in normalized if bank["condition"] == "clean")
    selected_bank = next(
        bank for bank in normalized if bank["condition"] == "selected_bank"
    )
    clean_states = states_by_bank[clean_bank["bank_id"]]
    carrier_distance_to_clean = {
        bank["bank_id"]: {
            str(layer): float(
                torch.linalg.vector_norm(
                    states_by_bank[bank["bank_id"]][layer] - clean_states[layer]
                )
                / max(1.0, float(len(mediator_positions)) ** 0.5)
            )
            for layer in patch_layers
        }
        for bank in normalized
    }
    matched_position_states: dict[int, torch.Tensor] | None = None
    if include_matched_position_control:
        _unused, matched_capture = capture_post_block_states(
            model,
            adapter,
            replay_by_bank[clean_bank["bank_id"]],
            matched_position_donors,
            layers=patch_layers,
        )
        matched_position_states = {
            layer: matched_capture[layer].clone() for layer in patch_layers
        }
    factorial: list[dict[str, Any]] = []
    for bank in normalized:
        factorial.append(
            {
                "condition": {
                    "clean": "clean_generated_self_state",
                    "selected_bank": "selected_generated_self_state",
                    "layer_matched_random": (
                        "layer_matched_random_generated_self_state"
                    ),
                }[bank["condition"]],
                "receiver_bank": bank,
                "mediator_bank": bank,
                "mediator_condition": "self_state",
            }
        )
        if bank["condition"] != "clean":
            factorial.append(
                {
                    "condition": (
                        "selected_generated_clean_state_restore"
                        if bank["condition"] == "selected_bank"
                        else "layer_matched_random_generated_clean_state_restore"
                    ),
                    "receiver_bank": bank,
                    "mediator_bank": clean_bank,
                    "mediator_condition": "clean_state_restore",
                }
            )
    factorial.append(
        {
            "condition": "clean_generated_selected_state_occlusion",
            "receiver_bank": clean_bank,
            "mediator_bank": selected_bank,
            "mediator_condition": "selected_state_occlusion",
        }
    )
    if include_matched_position_control:
        factorial.append(
            {
                "condition": "selected_generated_matched_position_state_control",
                "receiver_bank": selected_bank,
                "mediator_bank": clean_bank,
                "mediator_condition": "matched_position_state_control",
            }
        )
    expected_arms = 11 if include_matched_position_control else 10
    if len(factorial) != expected_arms:
        raise RuntimeError(
            f"Generated-suffix bridge must emit {expected_arms} arms per sample"
        )

    common = {
        "schema_version": COUNT_STREAM_SCHEMA_VERSION,
        "experiment_id": "generated_suffix_targeted_state_bridge",
        "request_id": encoding.request_id,
        "model_label": encoding.model_label,
        "seed": int(encoding.seed),
        "dataset_split": encoding.split,
        "gold_count": gold_count,
        "answer_site_id": answer_site_id,
        "targeted_from_occurrence": int(specification["from_occurrence"]),
        "targeted_to_occurrence": int(specification["to_occurrence"]),
        "targeted_anchor_equivalence_id": str(
            specification["anchor_equivalence_id"]
        ),
        "targeted_query_position": int(targeted_query),
        "free_running_replay_start": replay_start,
        "free_running_replay_stop": replay_stop,
        "free_running_token_budget": len(replay_positions),
        "terminal_span": [int(terminal_start), int(terminal_stop)],
        "causal_terminal_suffix_span": [
            int(terminal_positions[0]),
            int(terminal_positions[-1]) + 1,
        ],
        "terminal_patch_token_count": len(terminal_positions),
        "terminal_nonmarker_token_count": len(terminal_nonmarker_offsets),
        "state_patch_geometry": geometry,
        "state_patch_positions": list(mediator_positions),
        "state_patch_token_count": len(mediator_positions),
        "state_patch_excludes_answer_query": bool(
            int(registry.query_position) not in mediator_positions
        ),
        "replay_alignment": "same_sequence_fixed_token_budget_exact_positions",
        "teacher_forced_terminal_suffix": False,
        "post_terminal_suffix_teacher_forced": True,
        "state_source_layer": int(source_layer),
        "state_patch_layers": list(patch_layers),
        "state_patch_layer_mode": "cumulative_clamp",
        "matched_position_control_enabled": bool(include_matched_position_control),
        "matched_position_control_equal_token_budget": bool(
            not include_matched_position_control
            or len(matched_position_donors) == len(mediator_positions)
        ),
        "matched_position_donor_count": len(matched_position_donors),
        "matched_position_donors_sha256": _sha256_json(matched_position_donors),
        **semantic_audit,
        "causal_claim_scope": (
            f"top{selected_size}_targeted_query_to_generated_terminal_content_to_"
            f"L{int(source_layer)}_{int(adapter.num_layers) - 1}_"
            f"{geometry}_state_to_answer"
        ),
        "registry_sha256": registry.to_dict()["registry_sha256"],
    }
    rows: list[dict[str, Any]] = []
    for cell in factorial:
        receiver = cell["receiver_bank"]
        mediator = cell["mediator_bank"]
        replay = replay_by_bank[receiver["bank_id"]]
        if cell["mediator_condition"] == "self_state":
            replacements = None
        elif cell["mediator_condition"] == "matched_position_state_control":
            if matched_position_states is None:
                raise RuntimeError("Matched-position state control was not captured")
            replacements = matched_position_states
        else:
            replacements = states_by_bank[mediator["bank_id"]]
        prefill, _captures, applications, realized_norms = (
            _prefill_with_layerwise_state_replacements(
                model,
                adapter,
                replay,
                positions=mediator_positions,
                replacements=replacements,
                readout_layers=(int(adapter.num_layers) - 1,),
                readout_positions=(int(registry.query_position),),
            )
        )
        outcomes = _score_and_generate_prefill(
            model,
            tokenizer,
            replay,
            prefill,
            run_greedy=bool(run_greedy_answer),
            max_new_tokens=int(max_new_tokens),
        )
        audit = generation_audit_by_bank[receiver["bank_id"]]
        rows.append(
            {
                **common,
                "condition": str(cell["condition"]),
                "status": "ok",
                "receiver_generation_condition": str(receiver["condition"]),
                "receiver_generation_repeat": int(receiver["repeat"]),
                "receiver_bank_sha256": str(receiver["bank_sha256"]),
                "receiver_heads": [list(value) for value in receiver["heads"]],
                "mediator_condition": str(cell["mediator_condition"]),
                "mediator_state_source_condition": str(mediator["condition"]),
                "mediator_state_source_repeat": int(mediator["repeat"]),
                "mediator_bank_sha256": str(mediator["bank_sha256"]),
                "receiver_carrier_distance_to_clean_by_layer": (
                    carrier_distance_to_clean[receiver["bank_id"]]
                ),
                "state_patch_hook_applications": {
                    str(key): int(value) for key, value in sorted(applications.items())
                },
                "state_patch_realized_fro_norm_by_layer": {
                    str(key): float(value)
                    for key, value in sorted(realized_norms.items())
                },
                **audit,
                **outcomes,
            }
        )
    return rows
