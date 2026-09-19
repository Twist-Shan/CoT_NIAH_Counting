"""Additional v20 causal interventions aligned to the realistic NiaH reports."""

from __future__ import annotations

import contextlib
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from .data import V20Example, render_v20
from .v10_port_analysis import (
    RunContext,
    _balanced,
    _count_band,
    _forward,
    _non_target_token,
    _normalized_recovery,
    _paired_count_margin,
    _prelayer_residual_patch,
    _replace_tokens,
    _residual_patch,
)


def _matched_non_target_position(example: V20Example) -> int:
    targets = set(example.needle_positions)
    if not targets:
        raise ValueError("task example lacks target positions")
    target = max(targets)
    candidates = [index for index in range(len(example.seq_tokens)) if index not in targets]
    if not candidates:
        raise ValueError("prompt contains no matched non-target position")
    return min(candidates, key=lambda index: (abs(index - target), index))


def run_nonthinking_prompt_evidence_restoration(
    ctx: RunContext,
    *,
    examples_per_count: int = 3,
) -> pd.DataFrame:
    """Delete one true prompt occurrence and restore its residual representation.

    The target record in synthetic v20 is one character, so a single-position
    patch is the exact analogue of the realistic report's full-record/span
    restoration.  Content- and location-matched patches prevent interpreting a
    generic residual injection as evidence-specific rescue.
    """

    eligible = [example for example in ctx.heldout_examples if int(example.count or 0) >= 2]
    # ``_balanced`` expects a contiguous count range, which is 2..30 here.
    selected = _balanced(eligible, examples_per_count)
    clean_items = [render_v20(example, ctx.vocab, "nonthinking") for example in selected]
    target_positions = [item.prompt_needle_positions[-1] for item in clean_items]
    control_prompt_offsets = [_matched_non_target_position(example) for example in selected]
    control_positions = [
        item.spans.prompt_start + offset
        for item, offset in zip(clean_items, control_prompt_offsets, strict=True)
    ]
    corrupt_items = [
        _replace_tokens(
            item,
            {target: _non_target_token(example, ctx.vocab)},
            ctx.vocab,
        )
        for example, item, target in zip(
            selected, clean_items, target_positions, strict=True
        )
    ]
    model = ctx.models["nonthinking"]
    clean_output = _forward(
        model,
        clean_items,
        ctx.vocab,
        ctx.device,
        output_hidden_states=True,
    )
    corrupt_output = _forward(model, corrupt_items, ctx.vocab, ctx.device)
    alternatives = [int(example.count or 0) - 1 for example in selected]
    clean_margin = _paired_count_margin(
        clean_output.logits, clean_items, ctx.vocab, alternatives
    )
    corrupt_margin = _paired_count_margin(
        corrupt_output.logits, corrupt_items, ctx.vocab, alternatives
    )
    rows: list[dict[str, Any]] = []

    def record(layer: int, intervention: str, output) -> None:
        patched_margin = _paired_count_margin(
            output.logits, corrupt_items, ctx.vocab, alternatives
        )
        recovery = _normalized_recovery(clean_margin, corrupt_margin, patched_margin)
        for index, example in enumerate(selected):
            rows.append(
                {
                    "layer": int(layer),
                    "intervention": intervention,
                    "count": int(example.count or 0),
                    "count_band": _count_band(int(example.count or 0)),
                    "deleted_occurrence": int(example.count or 0),
                    "clean_margin_n_vs_n_minus_1": clean_margin[index],
                    "corrupt_margin_n_vs_n_minus_1": corrupt_margin[index],
                    "patched_margin_n_vs_n_minus_1": patched_margin[index],
                    "normalized_recovery": recovery[index],
                    "clean_correct_pairwise": float(clean_margin[index] > 0),
                    "corrupt_correct_pairwise": float(corrupt_margin[index] > 0),
                    "patched_correct_pairwise": float(patched_margin[index] > 0),
                }
            )

    record(-1, "corrupt_baseline", corrupt_output)
    assert clean_output.hidden_states is not None
    embedding = clean_output.hidden_states[0]
    target_embeddings = torch.stack(
        [embedding[row, position] for row, position in enumerate(target_positions)]
    )
    control_embeddings = torch.stack(
        [embedding[row, position] for row, position in enumerate(control_positions)]
    )
    with _prelayer_residual_patch(model, 1, target_positions, target_embeddings):
        output = _forward(model, corrupt_items, ctx.vocab, ctx.device)
    record(0, "target_embedding_at_deleted_target", output)
    with _prelayer_residual_patch(model, 1, target_positions, control_embeddings):
        output = _forward(model, corrupt_items, ctx.vocab, ctx.device)
    record(0, "ordinary_embedding_at_deleted_target_control", output)
    with _prelayer_residual_patch(model, 1, control_positions, target_embeddings):
        output = _forward(model, corrupt_items, ctx.vocab, ctx.device)
    record(0, "target_embedding_at_ordinary_location_control", output)

    for layer in range(1, len(model.layers) + 1):
        clean_hidden = clean_output.hidden_states[layer]
        target_vectors = torch.stack(
            [clean_hidden[row, position] for row, position in enumerate(target_positions)]
        )
        control_vectors = torch.stack(
            [clean_hidden[row, position] for row, position in enumerate(control_positions)]
        )

        with _residual_patch(model, layer, target_positions, target_vectors):
            output = _forward(model, corrupt_items, ctx.vocab, ctx.device)
        record(layer, "target_state_at_deleted_target", output)

        with _residual_patch(model, layer, target_positions, control_vectors):
            output = _forward(model, corrupt_items, ctx.vocab, ctx.device)
        record(layer, "ordinary_state_at_deleted_target_control", output)

        with _residual_patch(model, layer, control_positions, target_vectors):
            output = _forward(model, corrupt_items, ctx.vocab, ctx.device)
        record(layer, "target_state_at_ordinary_location_control", output)
    return pd.DataFrame(rows)


def _trace_pairs(
    examples: Sequence[V20Example],
    *,
    examples_per_k: int,
) -> list[tuple[V20Example, V20Example, V20Example, int]]:
    buckets: dict[int, list[V20Example]] = {}
    for example in examples:
        buckets.setdefault(int(example.count or 0), []).append(example)
    pairs = []
    for k in range(2, 29):
        donors = buckets.get(k, [])[:examples_per_k]
        receivers = buckets.get(k + 2, [])[: examples_per_k + 1]
        if len(donors) < examples_per_k or len(receivers) < examples_per_k + 1:
            raise ValueError(f"insufficient held-out examples for trace pair k={k}")
        for index, donor in enumerate(donors):
            receiver = receivers[index]
            control = receivers[(index + 1) % len(receivers)]
            pairs.append((donor, receiver, control, k))
    return pairs


def run_thinking_trace_scope_restoration(
    ctx: RunContext,
    *,
    examples_per_k: int = 3,
) -> pd.DataFrame:
    """Patch terminal progress states into a continuing trace at matched index k."""

    pairs = _trace_pairs(ctx.heldout_examples, examples_per_k=examples_per_k)
    model = ctx.models["thinking"]
    donor_items = [render_v20(donor, ctx.vocab, "thinking") for donor, _, _, _ in pairs]
    receiver_items = [
        render_v20(receiver, ctx.vocab, "thinking") for _, receiver, _, _ in pairs
    ]
    control_items = [
        render_v20(control, ctx.vocab, "thinking") for _, _, control, _ in pairs
    ]
    receiver_output = _forward(
        model,
        receiver_items,
        ctx.vocab,
        ctx.device,
        output_hidden_states=True,
    )
    donor_output = _forward(
        model,
        donor_items,
        ctx.vocab,
        ctx.device,
        output_hidden_states=True,
    )
    control_output = _forward(
        model,
        control_items,
        ctx.vocab,
        ctx.device,
        output_hidden_states=True,
    )
    assert donor_output.hidden_states is not None
    assert control_output.hidden_states is not None

    index_positions = [
        item.spans.trace_index_positions[k - 1]
        for item, (_, _, _, k) in zip(receiver_items, pairs, strict=True)
    ]
    marker_positions = [
        item.spans.trace_marker_positions[k - 1]
        for item, (_, _, _, k) in zip(receiver_items, pairs, strict=True)
    ]
    donor_index_positions = [
        item.spans.trace_index_positions[k - 1]
        for item, (_, _, _, k) in zip(donor_items, pairs, strict=True)
    ]
    donor_marker_positions = [
        item.spans.trace_marker_positions[k - 1]
        for item, (_, _, _, k) in zip(donor_items, pairs, strict=True)
    ]
    control_index_positions = [
        item.spans.trace_index_positions[k - 1]
        for item, (_, _, _, k) in zip(control_items, pairs, strict=True)
    ]
    control_marker_positions = [
        item.spans.trace_marker_positions[k - 1]
        for item, (_, _, _, k) in zip(control_items, pairs, strict=True)
    ]
    close_id = ctx.vocab.token_to_id["</Think>"]
    next_ids = [ctx.vocab.token_to_id[ctx.vocab.number_token(k + 1)] for *_, k in pairs]

    def margins(output) -> np.ndarray:
        return np.asarray(
            [
                float(
                    (
                        output.logits[row, marker_positions[row], close_id]
                        - output.logits[row, marker_positions[row], next_id]
                    )
                    .detach()
                    .cpu()
                )
                for row, next_id in enumerate(next_ids)
            ]
        )

    baseline = margins(receiver_output)
    rows: list[dict[str, Any]] = []

    def record(layer: int, scope: str, donor_kind: str, output) -> None:
        changed = margins(output)
        for index, (donor, receiver, _control, k) in enumerate(pairs):
            rows.append(
                {
                    "layer": int(layer),
                    "scope": scope,
                    "donor_kind": donor_kind,
                    "k": int(k),
                    "donor_total": int(donor.count or 0),
                    "receiver_total": int(receiver.count or 0),
                    "baseline_close_minus_continue_margin": baseline[index],
                    "patched_close_minus_continue_margin": changed[index],
                    "close_margin_shift": changed[index] - baseline[index],
                    "baseline_close_decision": float(baseline[index] > 0),
                    "patched_close_decision": float(changed[index] > 0),
                }
            )

    record(0, "none", "baseline_continuing", receiver_output)
    for layer in range(1, len(model.layers) + 1):
        donor_index = torch.stack(
            [
                donor_output.hidden_states[layer][row, position]
                for row, position in enumerate(donor_index_positions)
            ]
        )
        donor_marker = torch.stack(
            [
                donor_output.hidden_states[layer][row, position]
                for row, position in enumerate(donor_marker_positions)
            ]
        )
        control_index = torch.stack(
            [
                control_output.hidden_states[layer][row, position]
                for row, position in enumerate(control_index_positions)
            ]
        )
        control_marker = torch.stack(
            [
                control_output.hidden_states[layer][row, position]
                for row, position in enumerate(control_marker_positions)
            ]
        )

        with _residual_patch(model, layer, index_positions, donor_index):
            output = _forward(model, receiver_items, ctx.vocab, ctx.device)
        record(layer, "index_only", "terminal_total_equals_k", output)

        with _residual_patch(model, layer, marker_positions, donor_marker):
            output = _forward(model, receiver_items, ctx.vocab, ctx.device)
        record(layer, "marker_only", "terminal_total_equals_k", output)

        with contextlib.ExitStack() as stack:
            stack.enter_context(_residual_patch(model, layer, index_positions, donor_index))
            stack.enter_context(_residual_patch(model, layer, marker_positions, donor_marker))
            output = _forward(model, receiver_items, ctx.vocab, ctx.device)
        record(layer, "index_plus_marker", "terminal_total_equals_k", output)

        with contextlib.ExitStack() as stack:
            stack.enter_context(_residual_patch(model, layer, index_positions, control_index))
            stack.enter_context(_residual_patch(model, layer, marker_positions, control_marker))
            output = _forward(model, receiver_items, ctx.vocab, ctx.device)
        record(layer, "index_plus_marker", "continuing_same_total_control", output)
    return pd.DataFrame(rows)


__all__ = [
    "run_nonthinking_prompt_evidence_restoration",
    "run_thinking_trace_scope_restoration",
]
