from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from realistic_niah_v4.modeling import (
    DecoderAdapter,
    _replace_output_tensor,
    _tensor_from_output,
    generate_answer_completion,
    position_attention_outputs,
)

from .causal import (
    _continuation_metrics,
    _first_generated_gold_city,
    _fixed_target_head_write_intervention_logits,
    capture_source_specific_head_writes_multi,
    completion_metrics,
)
from .count_stream import (
    AnswerSourceRegistry,
    build_answer_source_registry,
    source_attention_metrics,
)
from .encoding import NativeTraceEncoding, build_native_causal_encoding
from .parsing import gold_records


TOKEN_LEVEL_ABLATION_SCHEMA_VERSION = "realistic_niah_v5_token_level_ablation_v1"


ANSWER_TOKEN_BLANK_CONDITIONS = (
    "clean",
    "prompt_all_blank",
    "prompt_records_blank",
    "trace_all_blank",
    "prompt_and_trace_blank",
)


TARGETING_TRACE_BLANK_CONDITIONS = (
    "clean",
    "early_half_trace_blank",
    "cumulative_trace_blank",
    "recent_transition_blank",
    "full_trace_blank",
    "early_half_trace_matched_control",
    "cumulative_trace_matched_control",
    "recent_transition_matched_control",
    "full_trace_matched_control",
)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _positions_to_spans(positions: Sequence[int]) -> tuple[tuple[int, int], ...]:
    values = sorted({int(value) for value in positions})
    if not values:
        return ()
    spans: list[tuple[int, int]] = []
    start = values[0]
    previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            spans.append((start, previous + 1))
            start = value
        previous = value
    spans.append((start, previous + 1))
    return tuple(spans)


def _span_positions(spans: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    return tuple(
        position
        for start, end in spans
        for position in range(int(start), int(end))
    )


def _clip_spans(
    spans: Sequence[tuple[int, int]], *, lower: int, upper: int
) -> tuple[tuple[int, int], ...]:
    clipped = []
    for start, end in spans:
        left = max(int(start), int(lower))
        right = min(int(end), int(upper))
        if left < right:
            clipped.append((left, right))
    return tuple(clipped)


@dataclass(frozen=True)
class TokenBlankRegistry:
    """Position-preserving token groups for answer and retrieval queries.

    Position zero is retained as an unconditional BOS/system boundary.  Every
    registered blank position is strictly before the query, so the exact query
    token and all teacher-forced target tokens are invariant across arms.
    """

    request_id: str
    query_position: int
    sequence_length: int
    prompt_token_count: int
    prompt_all: tuple[tuple[int, int], ...]
    prompt_records: tuple[tuple[int, int], ...]
    trace_all: tuple[tuple[int, int], ...]
    early_half_trace: tuple[tuple[int, int], ...]
    cumulative_trace: tuple[tuple[int, int], ...]
    recent_transition: tuple[tuple[int, int], ...]
    ordinary_prompt_control_pool: tuple[tuple[int, int], ...]
    visible_trace_item_count: int
    transition_occurrence: int | None

    def positions(self, group: str) -> tuple[int, ...]:
        if not hasattr(self, group):
            raise KeyError(f"Unknown token blank group: {group}")
        raw = getattr(self, group)
        if not isinstance(raw, tuple):
            raise KeyError(f"Token blank field is not a span group: {group}")
        return _span_positions(raw)

    def validate(self) -> None:
        query = int(self.query_position)
        if not 0 < int(self.prompt_token_count) <= query < int(self.sequence_length):
            raise ValueError("Invalid prompt/query/sequence boundaries")
        group_names = (
            "prompt_all",
            "prompt_records",
            "trace_all",
            "early_half_trace",
            "cumulative_trace",
            "recent_transition",
            "ordinary_prompt_control_pool",
        )
        groups = {name: set(self.positions(name)) for name in group_names}
        if any(position <= 0 or position >= query for position in groups["prompt_all"]):
            raise ValueError("Prompt-all blank positions cross BOS or query")
        if any(
            position < self.prompt_token_count or position >= query
            for position in groups["trace_all"]
        ):
            raise ValueError("Trace blank positions cross prompt/query boundaries")
        if not groups["prompt_records"] <= groups["prompt_all"]:
            raise ValueError("Prompt records must be contained in prompt-all")
        if groups["cumulative_trace"] & groups["recent_transition"]:
            raise ValueError("Cumulative and recent trace groups overlap")
        if (
            groups["cumulative_trace"] | groups["recent_transition"]
            != groups["trace_all"]
        ):
            raise ValueError(
                "Cumulative/recent groups must exactly partition visible trace"
            )
        if not groups["early_half_trace"] <= groups["cumulative_trace"]:
            raise ValueError("Early-half trace must be a cumulative-trace subset")
        if groups["ordinary_prompt_control_pool"] & groups["prompt_records"]:
            raise ValueError("Ordinary prompt controls overlap prompt records")
        if not groups["ordinary_prompt_control_pool"] <= groups["prompt_all"]:
            raise ValueError("Ordinary prompt controls must stay in the prompt")
        if int(self.visible_trace_item_count) < 1:
            raise ValueError("At least one visible trace item is required")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": TOKEN_LEVEL_ABLATION_SCHEMA_VERSION,
            "request_id": self.request_id,
            "query_position": int(self.query_position),
            "sequence_length": int(self.sequence_length),
            "prompt_token_count": int(self.prompt_token_count),
            "visible_trace_item_count": int(self.visible_trace_item_count),
            "transition_occurrence": self.transition_occurrence,
        }
        for name in (
            "prompt_all",
            "prompt_records",
            "trace_all",
            "early_half_trace",
            "cumulative_trace",
            "recent_transition",
            "ordinary_prompt_control_pool",
        ):
            spans = [list(value) for value in getattr(self, name)]
            payload[name] = spans
            payload[f"{name}_token_count"] = len(self.positions(name))
        payload["registry_sha256"] = _sha256_json(payload)
        return payload


def build_token_blank_registry_from_spans(
    *,
    request_id: str,
    query_position: int,
    sequence_length: int,
    prompt_token_count: int,
    prompt_record_spans: Sequence[tuple[int, int]],
    trace_item_spans: Sequence[tuple[int, int]],
    transition_occurrence: int | None,
) -> TokenBlankRegistry:
    """Build the cumulative/recent 2x2 without any model-dependent guessing."""

    query = int(query_position)
    prompt_count = int(prompt_token_count)
    ordered_items = tuple(sorted((int(start), int(end)) for start, end in trace_item_spans))
    if transition_occurrence is not None:
        occurrence = int(transition_occurrence)
        if not 1 <= occurrence <= len(ordered_items):
            raise ValueError(
                f"Transition occurrence {occurrence} is outside 1..{len(ordered_items)}"
            )
        # At city-pre/P2, tokens belonging to item k+1 may already be visible.
        # The causal source split must nevertheless end at completed item k.
        ordered_items = ordered_items[:occurrence]
    visible = _clip_spans(ordered_items, lower=prompt_count, upper=query)
    if not visible:
        raise ValueError("No complete or partial trace item is visible at the query")
    visible = tuple(sorted(visible))
    if transition_occurrence is not None and len(visible) != int(transition_occurrence):
        raise ValueError(
            "The retrieval query does not expose exactly the completed items 1..k: "
            f"visible={len(visible)} k={int(transition_occurrence)}"
        )
    recent_start = int(visible[-1][0])
    trace_all = ((prompt_count, query),)
    cumulative = ((prompt_count, recent_start),) if prompt_count < recent_start else ()
    recent = ((recent_start, query),) if recent_start < query else ()
    half_count = len(visible) // 2
    if half_count:
        half_end = min(query, int(visible[half_count - 1][1]))
        early_half = ((prompt_count, half_end),) if prompt_count < half_end else ()
    else:
        early_half = ()
    prompt_all = ((1, prompt_count),) if prompt_count > 1 else ()
    prompt_records = _clip_spans(
        prompt_record_spans, lower=1, upper=prompt_count
    )
    record_positions = set(_span_positions(prompt_records))
    ordinary_positions = [
        position
        for position in range(1, prompt_count)
        if position not in record_positions
    ]
    registry = TokenBlankRegistry(
        request_id=str(request_id),
        query_position=query,
        sequence_length=int(sequence_length),
        prompt_token_count=prompt_count,
        prompt_all=prompt_all,
        prompt_records=prompt_records,
        trace_all=trace_all,
        early_half_trace=early_half,
        cumulative_trace=cumulative,
        recent_transition=recent,
        ordinary_prompt_control_pool=_positions_to_spans(ordinary_positions),
        visible_trace_item_count=len(visible),
        transition_occurrence=(
            None if transition_occurrence is None else int(transition_occurrence)
        ),
    )
    registry.validate()
    return registry


def build_token_blank_registry(
    row: Mapping[str, Any],
    tokenizer: Any,
    specification: Mapping[str, Any],
    *,
    answer_site_id: str = "answer_query_v3",
) -> tuple[NativeTraceEncoding, TokenBlankRegistry, AnswerSourceRegistry]:
    """Compile token groups at one grammar-aware retrieval anchor."""

    query_output = int(specification["query_output_token_index"])
    query_encoding = build_native_causal_encoding(
        row,
        tokenizer,
        query_output_token_index=query_output,
        sequence_output_token_end=query_output + 1,
        selected_site=specification,
    )
    _answer_encoding, answer_registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    if int(answer_registry.prompt_token_count) != int(query_encoding.prompt_token_count):
        raise RuntimeError("Answer/retrieval prompt boundaries disagree")
    registry = build_token_blank_registry_from_spans(
        request_id=query_encoding.request_id,
        query_position=query_encoding.query_position,
        sequence_length=query_encoding.sequence_length,
        prompt_token_count=query_encoding.prompt_token_count,
        prompt_record_spans=answer_registry.prompt_records,
        trace_item_spans=answer_registry.trace_items,
        transition_occurrence=specification.get("from_occurrence"),
    )
    return query_encoding, registry, answer_registry


def _matched_control_positions(
    registry: TokenBlankRegistry,
    *,
    target_group: str,
    repeat: int,
) -> tuple[int, ...]:
    targets = registry.positions(target_group)
    candidates = list(registry.positions("ordinary_prompt_control_pool"))
    if not targets:
        raise ValueError(f"Target group is empty: {target_group}")
    if len(candidates) < len(targets):
        raise ValueError(
            f"Ordinary prompt control pool has {len(candidates)} tokens but "
            f"{target_group} needs {len(targets)}"
        )
    salt = f"{registry.request_id}:{target_group}:{int(repeat)}"
    ranked = sorted(
        candidates,
        key=lambda position: (
            hashlib.sha256(f"{salt}:{position}".encode("utf-8")).digest(),
            position,
        ),
    )
    return tuple(sorted(ranked[: len(targets)]))


def token_blank_condition(
    registry: TokenBlankRegistry,
    condition: str,
    *,
    control_repeat: int = 1,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Resolve one registered treatment/control to exact absolute positions."""

    value = str(condition)
    direct_groups = {
        "clean": (),
        "prompt_all_blank": ("prompt_all",),
        "prompt_records_blank": ("prompt_records",),
        "trace_all_blank": ("trace_all",),
        "prompt_and_trace_blank": ("prompt_all", "trace_all"),
        "early_half_trace_blank": ("early_half_trace",),
        "cumulative_trace_blank": ("cumulative_trace",),
        "recent_transition_blank": ("recent_transition",),
        "full_trace_blank": ("cumulative_trace", "recent_transition"),
    }
    matched_groups = {
        "early_half_trace_matched_control": "early_half_trace",
        "cumulative_trace_matched_control": "cumulative_trace",
        "recent_transition_matched_control": "recent_transition",
        "full_trace_matched_control": "trace_all",
    }
    if value in direct_groups:
        source_groups = direct_groups[value]
        positions = tuple(
            sorted(
                {
                    position
                    for group in source_groups
                    for position in registry.positions(group)
                }
            )
        )
        matched_for = None
        if value != "clean" and not positions:
            raise ValueError(
                f"Condition is not applicable because {list(source_groups)} is empty"
            )
    elif value in matched_groups:
        matched_for = matched_groups[value]
        source_groups = ()
        positions = _matched_control_positions(
            registry, target_group=matched_for, repeat=int(control_repeat)
        )
    else:
        raise ValueError(f"Unknown token blank condition: {condition}")
    if any(position <= 0 or position >= registry.query_position for position in positions):
        raise RuntimeError("A token blank condition touched BOS/query/future tokens")
    audit = {
        "condition": value,
        "blank_source_groups": list(source_groups),
        "matched_control_for": matched_for,
        "control_repeat": int(control_repeat) if matched_for is not None else 0,
        "blank_token_count": len(positions),
        "blank_positions_sha256": _sha256_json(list(positions)),
        "blank_intervention": "zero_token_state_after_embedding_and_every_decoder_block",
        "token_deletion_used": False,
        "sequence_length_preserved": True,
        "query_token_preserved": True,
    }
    return positions, audit


@contextmanager
def blank_token_states(
    model: Any,
    adapter: DecoderAdapter,
    positions: Sequence[int],
) -> Iterator[dict[str, Any]]:
    """Zero selected token states while retaining every sequence position.

    Zeroing the embedding and the post-block residual at every layer prevents a
    blanked token from rebuilding and relaying source content at later layers.
    One-token cached query/decode calls are never interpreted as absolute token
    position zero and are therefore left untouched.
    """

    selected = tuple(sorted({int(value) for value in positions}))
    audit: dict[str, Any] = {
        "blank_embedding_hook_applications": 0,
        "blank_layer_hook_applications": {str(layer): 0 for layer in range(adapter.num_layers)},
        "blank_prefill_sequence_lengths": [],
    }
    if not selected:
        yield audit
        return
    embeddings = model.get_input_embeddings()
    if embeddings is None:
        raise RuntimeError("Model exposes no input embedding module")
    handles = []

    def active_positions(time: int) -> list[int]:
        # Cached query/decode calls have local length one, not absolute index 0.
        if int(time) <= 1:
            return []
        return [position for position in selected if position < int(time)]

    def embedding_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            raise RuntimeError("Embedding blank hook expected [batch,time,hidden]")
        active = active_positions(int(output.shape[1]))
        if not active:
            return output
        patched = output.clone()
        patched[:, active, :] = 0
        audit["blank_embedding_hook_applications"] += 1
        audit["blank_prefill_sequence_lengths"].append(int(output.shape[1]))
        return patched

    handles.append(embeddings.register_forward_hook(embedding_hook))
    for layer in range(int(adapter.num_layers)):

        def layer_hook(
            _module: Any,
            _args: tuple[Any, ...],
            output: Any,
            *,
            layer: int = layer,
        ) -> Any:
            hidden = _tensor_from_output(output)
            active = active_positions(int(hidden.shape[1]))
            if not active:
                return output
            patched = hidden.clone()
            patched[:, active, :] = 0
            audit["blank_layer_hook_applications"][str(layer)] += 1
            return _replace_output_tensor(output, patched)

        handles.append(adapter.layers[layer].register_forward_hook(layer_hook))
    try:
        yield audit
    finally:
        for handle in handles:
            handle.remove()
    if int(audit["blank_embedding_hook_applications"]) < 1:
        raise RuntimeError("Token blank intervention never reached a prefill")
    missing = [
        int(layer)
        for layer, count in audit["blank_layer_hook_applications"].items()
        if int(count) < 1
    ]
    if missing:
        raise RuntimeError(f"Token blank intervention missed decoder layers {missing}")


def _validate_heads(
    adapter: DecoderAdapter, heads: Sequence[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    normalized = tuple((int(layer), int(head)) for layer, head in heads)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("A frozen nonempty unique head bank is required")
    for layer, head in normalized:
        if not 0 <= layer < int(adapter.num_layers):
            raise ValueError(f"Invalid head layer {layer}")
        if not 0 <= head < int(adapter.num_heads[layer]):
            raise ValueError(f"Invalid head L{layer}H{head}")
    return normalized


def _bank_attention_summary(
    frame: Any, heads: Sequence[tuple[int, int]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    wanted = {(int(layer), int(head)) for layer, head in heads}
    selected = frame.loc[
        frame.apply(lambda row: (int(row["layer"]), int(row["head"])) in wanted, axis=1)
    ].copy()
    observed = {
        (int(row.layer), int(row.head)) for row in selected.itertuples(index=False)
    }
    if observed != wanted or len(selected) != len(wanted):
        raise RuntimeError("Source-write capture did not return the frozen head bank")
    target = selected["target_source_attention_mass"].to_numpy(float)
    all_gold = selected["all_gold_source_attention_mass"].to_numpy(float)
    relative = selected["target_source_relative_attention_mass"].to_numpy(float)
    margin = selected[
        "target_minus_max_wrong_source_attention_mass"
    ].to_numpy(float)
    finite_relative = relative[np.isfinite(relative)]
    summary = {
        "bank_target_attention_mass_sum": float(np.sum(target)),
        "bank_all_gold_attention_mass_sum": float(np.sum(all_gold)),
        "bank_target_attention_share_of_gold_mass": (
            float(np.sum(target) / np.sum(all_gold))
            if float(np.sum(all_gold)) > 0.0
            else np.nan
        ),
        "bank_mean_head_target_relative_mass": (
            float(np.mean(finite_relative)) if len(finite_relative) else np.nan
        ),
        "bank_mean_target_minus_max_wrong_mass": float(np.mean(margin)),
        "bank_target_top1_fraction": float(
            selected["target_source_attention_top1"].astype(bool).mean()
        ),
        "bank_target_unique_top1_fraction": float(
            selected["target_source_attention_unique_top1"].astype(bool).mean()
        ),
        "bank_source_specific_ov_write_norm_sum": float(
            selected["source_specific_ov_write_norm"].astype(float).sum()
        ),
    }
    columns = (
        "layer",
        "head",
        "target_source_attention_mass",
        "all_gold_source_attention_mass",
        "target_source_relative_attention_mass",
        "max_wrong_source_attention_mass",
        "target_minus_max_wrong_source_attention_mass",
        "target_source_attention_rank",
        "target_source_attention_top1",
        "target_source_attention_unique_top1",
        "source_specific_ov_write_norm",
    )
    details = selected.loc[:, list(columns)].sort_values(["layer", "head"])
    return summary, details.to_dict("records")


@torch.inference_mode()
def run_targeting_trace_token_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    specification: Mapping[str, Any],
    *,
    heads: Sequence[tuple[int, int]],
    conditions: Sequence[str] = TARGETING_TRACE_BLANK_CONDITIONS,
    control_repeat: int = 1,
    score_target: bool = True,
    run_greedy: bool = False,
    max_new_tokens: int = 32,
) -> list[dict[str, Any]]:
    """Measure whether cumulative or recent trace tokens create targeting."""

    bank = _validate_heads(adapter, heads)
    query_encoding, registry, _answer_registry = build_token_blank_registry(
        row, tokenizer, specification
    )
    query_output = int(specification["query_output_token_index"])
    target_start_output = int(specification["target_output_token_start"])
    target_end_output = int(specification["target_output_token_end"])
    target_encoding = build_native_causal_encoding(
        row,
        tokenizer,
        query_output_token_index=query_output,
        sequence_output_token_end=target_end_output,
        selected_site=specification,
    )
    target_start = int(target_encoding.prompt_token_count) + target_start_output
    target_end = int(target_encoding.prompt_token_count) + target_end_output
    target_ids = tuple(int(value) for value in specification["target_token_ids"])
    target_city = str(specification["target_city"])
    all_cities = tuple(str(value["city"]) for value in gold_records(row))
    active_layers = tuple(sorted({layer for layer, _head in bank}))
    registry_payload = registry.to_dict()
    common = {
        "schema_version": TOKEN_LEVEL_ABLATION_SCHEMA_VERSION,
        "experiment_id": "targeting_trace_token_blank",
        "request_id": query_encoding.request_id,
        "model_label": query_encoding.model_label,
        "seed": int(query_encoding.seed),
        "dataset_split": query_encoding.split,
        "gold_count": int(query_encoding.count),
        "anchor_equivalence_id": str(specification["anchor_equivalence_id"]),
        "registry_anchor_equivalence_id": str(
            specification.get("registry_anchor_equivalence_id", "")
        ),
        "anchor_roles": list(specification.get("anchor_roles", [])),
        "registry_anchor_roles": list(
            specification.get("registry_anchor_roles", [])
        ),
        "target_grammar_class": str(specification.get("target_grammar_class", "")),
        "from_occurrence": int(specification["from_occurrence"]),
        "to_occurrence": int(specification["to_occurrence"]),
        "target_city": target_city,
        "query_full_sequence_token": int(query_encoding.query_position),
        "target_full_sequence_token_start": target_start,
        "target_full_sequence_token_end": target_end,
        "bank_size": len(bank),
        "bank_heads": [list(value) for value in bank],
        "bank_sha256": _sha256_json([list(value) for value in bank]),
        "token_blank_registry_sha256": registry_payload["registry_sha256"],
        "visible_trace_item_count": int(registry.visible_trace_item_count),
        "early_half_trace_token_count": len(registry.positions("early_half_trace")),
        "cumulative_trace_token_count": len(registry.positions("cumulative_trace")),
        "recent_transition_token_count": len(registry.positions("recent_transition")),
    }
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        try:
            positions, blank_audit = token_blank_condition(
                registry, condition, control_repeat=int(control_repeat)
            )
        except ValueError as error:
            rows.append(
                {
                    **common,
                    "condition": str(condition),
                    "status": "not_applicable",
                    "exclusion_reason": str(error),
                }
            )
            continue
        with blank_token_states(model, adapter, positions) as attention_hook_audit:
            captured = capture_source_specific_head_writes_multi(
                model,
                adapter,
                query_encoding,
                source_cities=[target_city],
                attention_audit_cities=all_cities,
                layers=active_layers,
            )
        metrics, _writes = captured[target_city]
        bank_summary, head_details = _bank_attention_summary(metrics, bank)
        target_score: dict[str, Any] = {}
        score_hook_audit: dict[str, Any] | None = None
        if score_target:
            with blank_token_states(model, adapter, positions) as score_hook_audit:
                logits = _fixed_target_head_write_intervention_logits(
                    model,
                    adapter,
                    target_encoding,
                    hook_position=int(query_encoding.query_position),
                    target_full_sequence_token_start=target_start,
                    target_full_sequence_token_end=target_end,
                )
            target_score = _continuation_metrics(logits, target_ids)
        behavioral: dict[str, Any] = {}
        if run_greedy:
            with blank_token_states(model, adapter, positions) as generation_hook_audit:
                generated = generate_answer_completion(
                    model,
                    tokenizer,
                    query_encoding,
                    max_new_tokens=int(max_new_tokens),
                )
            generated_city, generated_city_start = _first_generated_gold_city(
                str(generated["completion_text"]), all_cities
            )
            behavioral = {
                "completion_text": str(generated["completion_text"]),
                "generation_truncated": bool(generated["generation_truncated"]),
                "first_generated_gold_city": generated_city,
                "first_generated_gold_city_char_start": generated_city_start,
                "target_city_retrieved": bool(
                    generated_city is not None
                    and generated_city.casefold() == target_city.casefold()
                ),
                "generation_blank_hook_audit": generation_hook_audit,
            }
        cumulative_present = str(condition) not in {
            "cumulative_trace_blank",
            "full_trace_blank",
        }
        recent_present = str(condition) not in {
            "recent_transition_blank",
            "full_trace_blank",
        }
        rows.append(
            {
                **common,
                "condition": str(condition),
                "status": "ok",
                "cumulative_trace_present": bool(cumulative_present),
                "recent_transition_present": bool(recent_present),
                **blank_audit,
                **bank_summary,
                **target_score,
                **behavioral,
                "head_metrics": head_details,
                "attention_blank_hook_audit": attention_hook_audit,
                "score_blank_hook_audit": score_hook_audit,
            }
        )
    return rows


def _answer_head_metrics(
    attention_rows: Sequence[torch.Tensor],
    key_starts: Sequence[int],
    registry: AnswerSourceRegistry,
    heads: Sequence[tuple[int, int]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    for layer, head in heads:
        prompt = source_attention_metrics(
            attention_rows[layer],
            key_start=key_starts[layer],
            spans=registry.prompt_records,
        )[head]
        trace = source_attention_metrics(
            attention_rows[layer],
            key_start=key_starts[layer],
            spans=registry.trace_context,
        )[head]
        items = source_attention_metrics(
            attention_rows[layer],
            key_start=key_starts[layer],
            spans=registry.trace_items,
        )[head]
        details.append(
            {
                "layer": int(layer),
                "head": int(head),
                "prompt_records_mass": float(prompt["mass"]),
                "prompt_records_coverage": float(prompt["coverage"]),
                "prompt_records_broad_score": float(prompt["broad_score"]),
                "trace_context_mass": float(trace["mass"]),
                "trace_context_coverage": float(trace["coverage"]),
                "trace_context_broad_score": float(trace["broad_score"]),
                "trace_items_mass": float(items["mass"]),
                "trace_items_coverage": float(items["coverage"]),
                "trace_items_broad_score": float(items["broad_score"]),
            }
        )
    summary = {}
    for group in ("prompt_records", "trace_context", "trace_items"):
        summary[f"bank_{group}_mass_sum"] = float(
            sum(float(row[f"{group}_mass"]) for row in details)
        )
        summary[f"bank_{group}_broad_score_mean"] = float(
            np.mean([float(row[f"{group}_broad_score"]) for row in details])
        )
    return summary, details


@torch.inference_mode()
def run_answer_token_blank_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    heads: Sequence[tuple[int, int]],
    conditions: Sequence[str] = ANSWER_TOKEN_BLANK_CONDITIONS,
    answer_site_id: str = "answer_query_v3",
    max_new_tokens: int = 16,
) -> list[dict[str, Any]]:
    """Blank prompt/trace token states and inspect the frozen answer bank."""

    bank = _validate_heads(adapter, heads)
    encoding, answer_registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    registry = build_token_blank_registry_from_spans(
        request_id=encoding.request_id,
        query_position=encoding.query_position,
        sequence_length=encoding.sequence_length,
        prompt_token_count=encoding.prompt_token_count,
        prompt_record_spans=answer_registry.prompt_records,
        trace_item_spans=answer_registry.trace_items,
        transition_occurrence=None,
    )
    registry_payload = registry.to_dict()
    common = {
        "schema_version": TOKEN_LEVEL_ABLATION_SCHEMA_VERSION,
        "experiment_id": "answer_token_source_blank",
        "request_id": encoding.request_id,
        "model_label": encoding.model_label,
        "seed": int(encoding.seed),
        "dataset_split": encoding.split,
        "gold_count": int(encoding.count),
        "answer_site_id": answer_site_id,
        "query_full_sequence_token": int(encoding.query_position),
        "bank_size": len(bank),
        "bank_heads": [list(value) for value in bank],
        "bank_sha256": _sha256_json([list(value) for value in bank]),
        "token_blank_registry_sha256": registry_payload["registry_sha256"],
    }
    first_gold_ids = dict(encoding.count_candidate_answer_token_ids)[encoding.count]
    if not first_gold_ids:
        raise RuntimeError("Gold answer candidate has no first token")
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        positions, blank_audit = token_blank_condition(registry, condition)
        with blank_token_states(model, adapter, positions) as attention_hook_audit:
            attention_rows, key_starts, logits = position_attention_outputs(
                model, adapter, encoding, int(encoding.query_position)
            )
        bank_summary, head_details = _answer_head_metrics(
            attention_rows, key_starts, answer_registry, bank
        )
        gold_first_logp = float(
            torch.log_softmax(logits.float(), dim=-1)[int(first_gold_ids[0])]
        )
        with blank_token_states(model, adapter, positions) as generation_hook_audit:
            generated = generate_answer_completion(
                model,
                tokenizer,
                encoding,
                max_new_tokens=int(max_new_tokens),
            )
        rows.append(
            {
                **common,
                "condition": str(condition),
                "status": "ok",
                **blank_audit,
                **bank_summary,
                "gold_first_answer_token_log_probability": gold_first_logp,
                **completion_metrics(generated, gold_count=encoding.count),
                "head_metrics": head_details,
                "attention_blank_hook_audit": attention_hook_audit,
                "generation_blank_hook_audit": generation_hook_audit,
            }
        )
    return rows
