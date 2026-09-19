from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from synthetic_niah_v5.data import (
    BaseExample,
    balanced_examples,
    render_nonthinking,
    render_thinking,
    trace_prediction_queries,
)
from synthetic_niah_v5.evaluation import parse_thinking_generation, trace_metric_dict
from synthetic_niah_v5.vocab import MARKER_TOKENS, NOISE_TOKENS, Vocab, index_token

from .v5_2_switch_diagnostics import load_v5_state, resolve_v5_run_dir


Head = tuple[int, int]


def _entropy(weights: np.ndarray, *, normalized: bool = True) -> float:
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    if total <= 0 or len(weights) <= 1:
        return 0.0
    probs = weights / total
    value = float(-(probs * np.log(np.maximum(probs, 1e-12))).sum())
    return value / math.log(len(weights)) if normalized else value


def _margin(logits: torch.Tensor, target_id: int, competitor_ids: Iterable[int]) -> float:
    values = logits.detach().float().cpu()
    competitors = [int(idx) for idx in competitor_ids if int(idx) != int(target_id)]
    if not competitors:
        return math.nan
    return float(values[int(target_id)] - values[competitors].max())


def _pred_from_subset(logits: torch.Tensor, ids: list[int]) -> int:
    values = logits.detach().float().cpu()[ids]
    return int(ids[int(values.argmax())])


def _head_mask(model: torch.nn.Module, device: str | torch.device, heads: Iterable[Head]) -> torch.Tensor:
    n_layer = int(model.config.n_layer)
    n_head = int(model.config.n_head)
    mask = torch.ones((n_layer, n_head), dtype=torch.float32, device=device)
    for layer, head in heads:
        if not (0 <= int(layer) < n_layer and 0 <= int(head) < n_head):
            raise ValueError(f"Invalid 0-based head L{layer}H{head} for {n_layer}x{n_head} model.")
        mask[int(layer), int(head)] = 0.0
    return mask


def _head_label(heads: Iterable[Head]) -> str:
    return " ".join(f"L{layer}H{head}" for layer, head in heads)


def _attention_categories(rendered, row: np.ndarray) -> dict[str, float]:
    spans = rendered.spans
    needles = rendered.prompt_needle_token_positions
    needle_set = set(needles)
    prompt = list(range(spans.seq_start, spans.seq_end_exclusive))
    noise = [pos for pos in prompt if pos not in needle_set]
    trace_indices = spans.trace_index_positions
    trace_markers = spans.trace_marker_positions
    needle_weights = row[needles] if needles else np.array([], dtype=float)
    needle_mass = float(needle_weights.sum()) if len(needle_weights) else 0.0
    return {
        "bos_mass": float(row[spans.bos_pos]),
        "mode_token_mass": float(row[spans.mode_pos]),
        "prompt_needles_mass": needle_mass,
        "prompt_noise_mass": float(row[noise].sum()) if noise else 0.0,
        "think_open_mass": float(row[spans.think_open_pos]),
        "trace_indices_mass": float(row[trace_indices].sum()) if trace_indices else 0.0,
        "trace_markers_mass": float(row[trace_markers].sum()) if trace_markers else 0.0,
        "last_trace_marker_mass": float(row[trace_markers[-1]]) if trace_markers else 0.0,
        "needle_entropy_normalized": _entropy(needle_weights),
        "needle_effective_number": float(math.exp(_entropy(needle_weights, normalized=False))) if len(needle_weights) else 0.0,
        "broad_aggregation_score": needle_mass * _entropy(needle_weights),
    }


@torch.no_grad()
def collect_attention_signatures(
    model,
    vocab: Vocab,
    examples: list[BaseExample],
    device: str | torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for example_idx, ex in enumerate(examples):
        for rendered in (render_nonthinking(ex, vocab), render_thinking(ex, vocab, trace_indices=True)):
            ids = torch.tensor([rendered.input_ids], dtype=torch.long, device=device)
            output = model(input_ids=ids, output_attentions=True)
            attentions = [item[0].detach().float().cpu().numpy() for item in output.attentions or []]
            queries: list[tuple[str, int, int | None]] = [("final_count_query", rendered.spans.think_close_pos, None)]
            if rendered.variant == "thinking":
                queries.extend(
                    ("trace_marker_query", int(item["prediction_query_pos"]), int(item["k"]) - 1)
                    for item in trace_prediction_queries(rendered)
                )
                queries.extend(
                    ("successor_query", int(pos), k)
                    for k, pos in enumerate(rendered.spans.trace_marker_positions)
                )
            for layer, layer_attn in enumerate(attentions):
                for head in range(layer_attn.shape[0]):
                    for query_kind, query_pos, correct_idx in queries:
                        row = layer_attn[head, query_pos]
                        metrics = _attention_categories(rendered, row)
                        correct_mass = math.nan
                        correct_top1 = math.nan
                        diagonal = math.nan
                        next_prompt_mass = math.nan
                        if correct_idx is not None and correct_idx < len(rendered.prompt_needle_token_positions):
                            needle_positions = rendered.prompt_needle_token_positions
                            weights = row[needle_positions]
                            correct_mass = float(weights[correct_idx])
                            correct_top1 = float(int(np.argmax(weights) == correct_idx))
                            diagonal = correct_mass / max(float(weights.sum()), 1e-12)
                            if query_kind == "successor_query" and correct_idx + 1 < len(needle_positions):
                                next_prompt_mass = float(row[needle_positions[correct_idx + 1]])
                        rows.append(
                            {
                                "example_idx": example_idx,
                                "mode": rendered.variant,
                                "count": ex.count,
                                "query_kind": query_kind,
                                "query_k": math.nan if correct_idx is None else correct_idx + 1,
                                "layer": layer,
                                "head": head,
                                "correct_prompt_needle_mass": correct_mass,
                                "correct_top1": correct_top1,
                                "diagonal_dominance": diagonal,
                                "next_prompt_needle_mass": next_prompt_mass,
                                **metrics,
                            }
                        )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["mode", "query_kind", "layer", "head"], as_index=False)
        .mean(numeric_only=True)
        .drop(columns=["example_idx"], errors="ignore")
    )
    return detail, summary


def select_head_groups(summary: pd.DataFrame, *, seed: int, n_layer: int, n_head: int) -> dict[str, list[Head]]:
    def top(query: str, mode: str, metric: str, n: int) -> list[Head]:
        part = summary[(summary.query_kind == query) & (summary["mode"] == mode)]
        part = part.sort_values(metric, ascending=False).head(n)
        return [(int(row.layer), int(row.head)) for row in part.itertuples(index=False)]

    all_heads = [(layer, head) for layer in range(n_layer) for head in range(n_head)]
    rng = random.Random(seed)
    targeted = top("trace_marker_query", "thinking", "correct_prompt_needle_mass", min(4, len(all_heads)))
    direct = top("final_count_query", "nonthinking", "broad_aggregation_score", min(4, len(all_heads)))
    trace_readout = top("final_count_query", "thinking", "trace_markers_mass", min(4, len(all_heads)))
    groups: dict[str, list[Head]] = {}
    for prefix, ordered in (("targeted", targeted), ("direct_broad", direct), ("trace_readout", trace_readout)):
        for n in (1, 2, 4):
            if len(ordered) >= n:
                groups[f"{prefix}_top{n}"] = ordered[:n]
    for n in (1, 2, 4):
        for replicate in range(5):
            groups[f"random_{n}_rep{replicate}"] = sorted(rng.sample(all_heads, n))
    for layer in range(n_layer):
        groups[f"layer{layer}_all"] = [(layer, head) for head in range(n_head)]
    groups["all_heads"] = all_heads
    return groups


def _teacher_forced_metrics(model, rendered, vocab: Vocab, device, *, head_mask=None) -> dict[str, float]:
    ids = torch.tensor([rendered.input_ids], dtype=torch.long, device=device)
    logits = model(input_ids=ids, head_mask=head_mask).logits[0]
    count_logits = logits[rendered.spans.think_close_pos]
    count_pred_id = _pred_from_subset(count_logits, vocab.count_ids)
    metrics = {
        "count_margin": _margin(count_logits, vocab.count_id(len(rendered.gold_trace_markers)), vocab.count_ids),
        "count_accuracy": float(count_pred_id == vocab.count_id(len(rendered.gold_trace_markers))),
        "trace_marker_margin": math.nan,
        "trace_marker_accuracy": math.nan,
        "successor_margin": math.nan,
        "successor_accuracy": math.nan,
    }
    if rendered.variant != "thinking":
        return metrics
    marker_margins: list[float] = []
    marker_acc: list[float] = []
    successor_margins: list[float] = []
    successor_acc: list[float] = []
    queries = trace_prediction_queries(rendered)
    successor_ids = [vocab.token_to_id[index_token(k)] for k in range(1, 11)] + [vocab.think_close_id]
    for idx, query in enumerate(queries):
        target_marker = vocab.token_to_id[rendered.gold_trace_markers[idx]]
        query_logits = logits[int(query["prediction_query_pos"])]
        marker_margins.append(_margin(query_logits, target_marker, vocab.marker_ids))
        marker_acc.append(float(_pred_from_subset(query_logits, vocab.marker_ids) == target_marker))
        successor_logits = logits[int(query["target_marker_pos"])]
        successor_target = vocab.think_close_id if idx + 1 == len(queries) else vocab.token_to_id[index_token(idx + 2)]
        successor_margins.append(_margin(successor_logits, successor_target, successor_ids))
        successor_acc.append(float(_pred_from_subset(successor_logits, successor_ids) == successor_target))
    metrics.update(
        {
            "trace_marker_margin": float(np.mean(marker_margins)),
            "trace_marker_accuracy": float(np.mean(marker_acc)),
            "successor_margin": float(np.mean(successor_margins)),
            "successor_accuracy": float(np.mean(successor_acc)),
        }
    )
    return metrics


@torch.no_grad()
def run_head_ablation(
    model,
    vocab: Vocab,
    examples: list[BaseExample],
    groups: dict[str, list[Head]],
    device: str | torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for example_idx, ex in enumerate(examples):
        rendered_by_mode = {
            "nonthinking": render_nonthinking(ex, vocab),
            "thinking": render_thinking(ex, vocab, trace_indices=True),
        }
        for mode, rendered in rendered_by_mode.items():
            clean = _teacher_forced_metrics(model, rendered, vocab, device)
            for group_name, heads in {"none": [], **groups}.items():
                masked = clean if not heads else _teacher_forced_metrics(
                    model, rendered, vocab, device, head_mask=_head_mask(model, device, heads)
                )
                rows.append(
                    {
                        "example_idx": example_idx,
                        "mode": mode,
                        "count": ex.count,
                        "group_name": group_name,
                        "n_masked_heads": len(heads),
                        "masked_heads": _head_label(heads),
                        **{f"clean_{key}": value for key, value in clean.items()},
                        **{f"masked_{key}": value for key, value in masked.items()},
                        **{f"drop_{key}": clean[key] - masked[key] for key in clean},
                    }
                )
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["mode", "group_name", "n_masked_heads", "masked_heads"], as_index=False).mean(numeric_only=True)
    return detail, summary


@torch.no_grad()
def _greedy_with_head_mask(model, prefix: list[int], vocab: Vocab, device, head_mask, max_new_tokens: int) -> list[int]:
    ids = torch.tensor([prefix], dtype=torch.long, device=device)
    generated: list[int] = []
    saw_close = False
    after_close = 0
    for _ in range(int(max_new_tokens)):
        next_id = int(model(input_ids=ids, head_mask=head_mask).logits[0, -1].argmax().detach().cpu())
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        if next_id == vocab.eos_id:
            break
        if saw_close:
            after_close += 1
            if after_close >= 2:
                break
        elif next_id == vocab.think_close_id:
            saw_close = True
    return generated


@torch.no_grad()
def run_behavioral_ablation(
    model,
    vocab: Vocab,
    examples: list[BaseExample],
    groups: dict[str, list[Head]],
    device: str | torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep = {
        "none",
        "targeted_top2",
        "targeted_top4",
        "direct_broad_top2",
        "direct_broad_top4",
        "trace_readout_top2",
        "trace_readout_top4",
        *[name for name in groups if name.startswith("layer")],
        *[name for name in groups if name.startswith("random_4_")],
        "all_heads",
    }
    selected = {name: heads for name, heads in {"none": [], **groups}.items() if name in keep}
    rows: list[dict[str, Any]] = []
    for example_idx, ex in enumerate(examples):
        prefixes = {
            "nonthinking": vocab.encode(["<BOS>", "<THINK_OFF>", *ex.seq_tokens, "<Think/>"]),
            "thinking": vocab.encode(["<BOS>", "<THINK_ON>", *ex.seq_tokens, "<Think/>"]),
        }
        for mode, prefix in prefixes.items():
            for group_name, heads in selected.items():
                head_mask = None if not heads else _head_mask(model, device, heads)
                generated = _greedy_with_head_mask(model, prefix, vocab, device, head_mask, max_new_tokens=26)
                tokens = vocab.decode(generated)
                parsed = parse_thinking_generation(tokens, ex.needle_markers if mode == "thinking" else [])
                metrics = trace_metric_dict(parsed, ex.needle_markers if mode == "thinking" else [])
                rows.append(
                    {
                        "example_idx": example_idx,
                        "mode": mode,
                        "count": ex.count,
                        "group_name": group_name,
                        "n_masked_heads": len(heads),
                        "masked_heads": _head_label(heads),
                        "pred_count": -1 if parsed.final_count is None else parsed.final_count,
                        "final_accuracy": float(parsed.final_count == ex.count),
                        "trace_exact": metrics["trace_exact"],
                        "trace_precision": metrics["trace_marker_precision"],
                        "trace_recall": metrics["trace_marker_recall"],
                        "premature_close": metrics["premature_close_rate"],
                        "missing_close": metrics["missing_close_rate"],
                        "generated": " ".join(tokens),
                    }
                )
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["mode", "group_name", "n_masked_heads", "masked_heads"], as_index=False).mean(numeric_only=True)
    return detail, summary


def marker_identity_corruption(ex: BaseExample, k: int) -> BaseExample:
    idx = int(k) - 1
    markers = list(ex.needle_markers)
    old = markers[idx]
    replacement_marker = next(token for token in MARKER_TOKENS if token != old)
    markers[idx] = replacement_marker
    tokens = list(ex.seq_tokens)
    tokens[ex.needle_positions[idx]] = replacement_marker
    return replace(ex, needle_markers=markers, seq_tokens=tokens)


def delete_last_needle(ex: BaseExample) -> BaseExample:
    if ex.count <= 1:
        raise ValueError("Need count >= 2 for delete-one corruption.")
    tokens = list(ex.seq_tokens)
    pos = ex.needle_positions[-1]
    tokens[pos] = NOISE_TOKENS[(pos + ex.count) % len(NOISE_TOKENS)]
    return replace(
        ex,
        count=ex.count - 1,
        needle_positions=list(ex.needle_positions[:-1]),
        needle_markers=list(ex.needle_markers[:-1]),
        seq_tokens=tokens,
    )


def _capture_cproj_inputs(model, ids: torch.Tensor, layers: set[int]) -> dict[int, torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer in layers:
        def hook(_module, args, layer=layer):
            captured[layer] = args[0].detach().clone()
        handles.append(model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(hook))
    try:
        model(input_ids=ids)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def _patched_forward(
    model,
    receiver_ids: torch.Tensor,
    donor_inputs: dict[int, torch.Tensor],
    heads: list[Head],
    donor_pos: int,
    receiver_pos: int,
) -> torch.Tensor:
    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(layer, []).append(head)
    head_dim = int(model.config.n_embd) // int(model.config.n_head)
    handles = []
    for layer, layer_heads in by_layer.items():
        def pre_hook(_module, args, layer=layer, layer_heads=tuple(layer_heads)):
            value = args[0].clone()
            donor = donor_inputs[layer].to(value.device)
            for head in layer_heads:
                start = head * head_dim
                end = start + head_dim
                value[:, receiver_pos, start:end] = donor[:, donor_pos, start:end]
            return (value, *args[1:])
        handles.append(model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(pre_hook))
    try:
        return model(input_ids=receiver_ids).logits[0]
    finally:
        for handle in handles:
            handle.remove()


def _residual_patched_forward(
    model,
    receiver_ids: torch.Tensor,
    donor_hidden: torch.Tensor,
    layer: int,
    receiver_pos: int,
) -> torch.Tensor:
    def hook(_module, _args, output):
        if isinstance(output, tuple):
            hidden = output[0].clone()
            hidden[:, receiver_pos] = donor_hidden.to(hidden.device)
            return (hidden, *output[1:])
        hidden = output.clone()
        hidden[:, receiver_pos] = donor_hidden.to(hidden.device)
        return hidden
    handle = model.transformer.h[layer].register_forward_hook(hook)
    try:
        return model(input_ids=receiver_ids).logits[0]
    finally:
        handle.remove()


def _normalized_recovery(clean: float, corrupt: float, patched: float) -> float:
    denominator = clean - corrupt
    return (patched - corrupt) / denominator if abs(denominator) > 1e-8 else math.nan


@torch.no_grad()
def run_patching(
    model,
    vocab: Vocab,
    examples: list[BaseExample],
    groups: dict[str, list[Head]],
    device: str | torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    patch_groups = {
        name: heads
        for name, heads in groups.items()
        if name in {"targeted_top1", "targeted_top2", "targeted_top4", "direct_broad_top1", "direct_broad_top2", "direct_broad_top4", "trace_readout_top2", "trace_readout_top4", "all_heads"}
        or name.startswith(("random_2_", "random_4_"))
    }
    layers = set(range(int(model.config.n_layer)))
    for example_idx, clean_ex in enumerate(examples):
        if clean_ex.count < 2:
            continue

        # Retrieval identity: one prompt marker changes, count and absolute positions stay fixed.
        k = clean_ex.count
        corrupt_ex = marker_identity_corruption(clean_ex, k)
        clean_r = render_thinking(clean_ex, vocab, trace_indices=True)
        corrupt_r = render_thinking(corrupt_ex, vocab, trace_indices=True)
        clean_ids = torch.tensor([clean_r.input_ids], dtype=torch.long, device=device)
        corrupt_ids = torch.tensor([corrupt_r.input_ids], dtype=torch.long, device=device)
        query_pos = clean_r.spans.trace_index_positions[k - 1]
        clean_target = vocab.token_to_id[clean_ex.needle_markers[k - 1]]
        corrupt_target = vocab.token_to_id[corrupt_ex.needle_markers[k - 1]]
        clean_logits = model(input_ids=clean_ids, output_hidden_states=True)
        corrupt_logits = model(input_ids=corrupt_ids).logits[0]
        clean_margin = _margin(clean_logits.logits[0, query_pos], clean_target, [clean_target, corrupt_target])
        corrupt_margin = _margin(corrupt_logits[query_pos], clean_target, [clean_target, corrupt_target])
        donor_inputs = _capture_cproj_inputs(model, clean_ids, layers)
        for group_name, heads in patch_groups.items():
            patched = _patched_forward(model, corrupt_ids, donor_inputs, heads, query_pos, query_pos)
            patched_margin = _margin(patched[query_pos], clean_target, [clean_target, corrupt_target])
            rows.append(
                {
                    "example_idx": example_idx,
                    "experiment": "retrieval_identity",
                    "group_name": group_name,
                    "n_patched_heads": len(heads),
                    "patched_heads": _head_label(heads),
                    "clean_margin": clean_margin,
                    "corrupt_margin": corrupt_margin,
                    "patched_margin": patched_margin,
                    "normalized_recovery": _normalized_recovery(clean_margin, corrupt_margin, patched_margin),
                    "donor_pos": query_pos,
                    "receiver_pos": query_pos,
                    "position_matched": 1.0,
                }
            )
        wrong_pos_heads = groups.get("targeted_top2", groups.get("targeted_top1", []))
        if wrong_pos_heads and query_pos > 0:
            patched = _patched_forward(model, corrupt_ids, donor_inputs, wrong_pos_heads, query_pos - 1, query_pos)
            patched_margin = _margin(patched[query_pos], clean_target, [clean_target, corrupt_target])
            rows.append(
                {
                    "example_idx": example_idx,
                    "experiment": "retrieval_identity",
                    "group_name": "targeted_wrong_donor_position",
                    "n_patched_heads": len(wrong_pos_heads),
                    "patched_heads": _head_label(wrong_pos_heads),
                    "clean_margin": clean_margin,
                    "corrupt_margin": corrupt_margin,
                    "patched_margin": patched_margin,
                    "normalized_recovery": _normalized_recovery(clean_margin, corrupt_margin, patched_margin),
                    "donor_pos": query_pos - 1,
                    "receiver_pos": query_pos,
                    "position_matched": 0.0,
                }
            )
        for layer in layers:
            donor_hidden = clean_logits.hidden_states[layer + 1][:, query_pos].detach()
            patched = _residual_patched_forward(model, corrupt_ids, donor_hidden, layer, query_pos)
            patched_margin = _margin(patched[query_pos], clean_target, [clean_target, corrupt_target])
            rows.append(
                {
                    "example_idx": example_idx,
                    "experiment": "retrieval_identity_residual",
                    "group_name": f"resid_after_layer{layer}",
                    "n_patched_heads": int(model.config.n_head),
                    "patched_heads": f"residual L{layer}",
                    "clean_margin": clean_margin,
                    "corrupt_margin": corrupt_margin,
                    "patched_margin": patched_margin,
                    "normalized_recovery": _normalized_recovery(clean_margin, corrupt_margin, patched_margin),
                    "donor_pos": query_pos,
                    "receiver_pos": query_pos,
                    "position_matched": 1.0,
                }
            )

        # Count path: delete the final needle. Patch at the count-prediction state.
        corrupt_count_ex = delete_last_needle(clean_ex)
        for mode in ("nonthinking", "thinking"):
            clean_count_r = render_nonthinking(clean_ex, vocab) if mode == "nonthinking" else render_thinking(clean_ex, vocab, trace_indices=True)
            corrupt_count_r = render_nonthinking(corrupt_count_ex, vocab) if mode == "nonthinking" else render_thinking(corrupt_count_ex, vocab, trace_indices=True)
            clean_count_ids = torch.tensor([clean_count_r.input_ids], dtype=torch.long, device=device)
            corrupt_count_ids = torch.tensor([corrupt_count_r.input_ids], dtype=torch.long, device=device)
            clean_pos = clean_count_r.spans.think_close_pos
            corrupt_pos = corrupt_count_r.spans.think_close_pos
            clean_output = model(input_ids=clean_count_ids, output_hidden_states=True)
            corrupt_output = model(input_ids=corrupt_count_ids)
            clean_margin = _margin(clean_output.logits[0, clean_pos], vocab.count_id(clean_ex.count), [vocab.count_id(clean_ex.count), vocab.count_id(corrupt_count_ex.count)])
            corrupt_margin = _margin(corrupt_output.logits[0, corrupt_pos], vocab.count_id(clean_ex.count), [vocab.count_id(clean_ex.count), vocab.count_id(corrupt_count_ex.count)])
            donor_inputs = _capture_cproj_inputs(model, clean_count_ids, layers)
            for group_name, heads in patch_groups.items():
                patched = _patched_forward(model, corrupt_count_ids, donor_inputs, heads, clean_pos, corrupt_pos)
                patched_margin = _margin(patched[corrupt_pos], vocab.count_id(clean_ex.count), [vocab.count_id(clean_ex.count), vocab.count_id(corrupt_count_ex.count)])
                rows.append(
                    {
                        "example_idx": example_idx,
                        "experiment": f"{mode}_count_readout",
                        "group_name": group_name,
                        "n_patched_heads": len(heads),
                        "patched_heads": _head_label(heads),
                        "clean_margin": clean_margin,
                        "corrupt_margin": corrupt_margin,
                        "patched_margin": patched_margin,
                        "normalized_recovery": _normalized_recovery(clean_margin, corrupt_margin, patched_margin),
                        "donor_pos": clean_pos,
                        "receiver_pos": corrupt_pos,
                        "position_matched": float(clean_pos == corrupt_pos),
                    }
                )
            for layer in layers:
                donor_hidden = clean_output.hidden_states[layer + 1][:, clean_pos].detach()
                patched = _residual_patched_forward(model, corrupt_count_ids, donor_hidden, layer, corrupt_pos)
                patched_margin = _margin(patched[corrupt_pos], vocab.count_id(clean_ex.count), [vocab.count_id(clean_ex.count), vocab.count_id(corrupt_count_ex.count)])
                rows.append(
                    {
                        "example_idx": example_idx,
                        "experiment": f"{mode}_count_readout_residual",
                        "group_name": f"resid_after_layer{layer}",
                        "n_patched_heads": int(model.config.n_head),
                        "patched_heads": f"residual L{layer}",
                        "clean_margin": clean_margin,
                        "corrupt_margin": corrupt_margin,
                        "patched_margin": patched_margin,
                        "normalized_recovery": _normalized_recovery(clean_margin, corrupt_margin, patched_margin),
                        "donor_pos": clean_pos,
                        "receiver_pos": corrupt_pos,
                        "position_matched": float(clean_pos == corrupt_pos),
                    }
                )
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["experiment", "group_name", "n_patched_heads", "patched_heads"], as_index=False).mean(numeric_only=True)
    return detail, summary


def render_trace_override(ex: BaseExample, vocab: Vocab, trace_count: int) -> tuple[list[int], int]:
    trace: list[str] = []
    for k in range(1, int(trace_count) + 1):
        trace.extend([index_token(k), ex.needle_markers[(k - 1) % len(ex.needle_markers)]])
    tokens = ["<BOS>", "<THINK_ON>", *ex.seq_tokens, "<Think/>", *trace, "</Think>"]
    return vocab.encode(tokens), len(tokens) - 1


@torch.no_grad()
def run_trace_conflict(model, vocab: Vocab, examples: list[BaseExample], device) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for example_idx, ex in enumerate(examples):
        for trace_count in range(1, 11):
            prefix, query_pos = render_trace_override(ex, vocab, trace_count)
            ids = torch.tensor([prefix], dtype=torch.long, device=device)
            logits = model(input_ids=ids).logits[0, query_pos]
            pred_id = _pred_from_subset(logits, vocab.count_ids)
            pred_count = vocab.count_from_id(pred_id)
            rows.append(
                {
                    "example_idx": example_idx,
                    "prompt_count": ex.count,
                    "forced_trace_count": trace_count,
                    "pred_count": pred_count,
                    "follows_prompt": float(pred_count == ex.count),
                    "follows_trace": float(pred_count == trace_count),
                    "prompt_margin": _margin(logits, vocab.count_id(ex.count), vocab.count_ids),
                    "trace_margin": _margin(logits, vocab.count_id(trace_count), vocab.count_ids),
                }
            )
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["prompt_count", "forced_trace_count"], as_index=False).mean(numeric_only=True)
    return detail, summary


@torch.no_grad()
def run_progress_state_transplant(
    model,
    vocab: Vocab,
    examples: list[BaseExample],
    device: str | torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = [ex for ex in examples if ex.count >= 4]
    rows: list[dict[str, Any]] = []
    index_ids = [vocab.token_to_id[index_token(k)] for k in range(1, 11)] + [vocab.think_close_id]
    for example_idx, ex in enumerate(eligible):
        rendered = render_thinking(ex, vocab, trace_indices=True)
        ids = torch.tensor([rendered.input_ids], dtype=torch.long, device=device)
        output = model(input_ids=ids, output_hidden_states=True)
        receiver_k = min(2, ex.count - 2)
        receiver_pos = rendered.spans.trace_marker_positions[receiver_k - 1]
        receiver_target = vocab.token_to_id[index_token(receiver_k + 1)]
        base_logits = output.logits[0, receiver_pos]
        for donor_offset in (-1, 1):
            donor_k = receiver_k + donor_offset
            donor_pos = rendered.spans.trace_marker_positions[donor_k - 1]
            donor_target = vocab.token_to_id[index_token(donor_k + 1)]
            base_donor_margin = _margin(base_logits, donor_target, [receiver_target, donor_target])
            for layer in range(int(model.config.n_layer)):
                donor_hidden = output.hidden_states[layer + 1][:, donor_pos].detach()
                patched = _residual_patched_forward(model, ids, donor_hidden, layer, receiver_pos)
                patched_logits = patched[receiver_pos]
                patched_donor_margin = _margin(patched_logits, donor_target, [receiver_target, donor_target])
                rows.append(
                    {
                        "example_idx": example_idx,
                        "donor_kind": "same_example_progress_shift",
                        "receiver_k": receiver_k,
                        "donor_k": donor_k,
                        "donor_offset": donor_offset,
                        "layer": layer,
                        "base_donor_margin": base_donor_margin,
                        "patched_donor_margin": patched_donor_margin,
                        "margin_shift_toward_donor": patched_donor_margin - base_donor_margin,
                        "patched_predicts_donor_next": float(_pred_from_subset(patched_logits, index_ids) == donor_target),
                        "patched_predicts_receiver_next": float(_pred_from_subset(patched_logits, index_ids) == receiver_target),
                        "position_matched": 0.0,
                    }
                )

        # Same-progress, same-position donor control from a different prompt.
        control_ex = eligible[(example_idx + 1) % len(eligible)]
        control_rendered = render_thinking(control_ex, vocab, trace_indices=True)
        control_ids = torch.tensor([control_rendered.input_ids], dtype=torch.long, device=device)
        control_output = model(input_ids=control_ids, output_hidden_states=True)
        control_pos = control_rendered.spans.trace_marker_positions[receiver_k - 1]
        for layer in range(int(model.config.n_layer)):
            donor_hidden = control_output.hidden_states[layer + 1][:, control_pos].detach()
            patched = _residual_patched_forward(model, ids, donor_hidden, layer, receiver_pos)
            patched_logits = patched[receiver_pos]
            receiver_margin = _margin(patched_logits, receiver_target, index_ids)
            rows.append(
                {
                    "example_idx": example_idx,
                    "donor_kind": "different_prompt_same_progress_control",
                    "receiver_k": receiver_k,
                    "donor_k": receiver_k,
                    "donor_offset": 0,
                    "layer": layer,
                    "base_donor_margin": _margin(base_logits, receiver_target, index_ids),
                    "patched_donor_margin": receiver_margin,
                    "margin_shift_toward_donor": receiver_margin - _margin(base_logits, receiver_target, index_ids),
                    "patched_predicts_donor_next": float(_pred_from_subset(patched_logits, index_ids) == receiver_target),
                    "patched_predicts_receiver_next": float(_pred_from_subset(patched_logits, index_ids) == receiver_target),
                    "position_matched": 1.0,
                }
            )
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["donor_kind", "donor_offset", "layer", "position_matched"], as_index=False).mean(numeric_only=True)
    return detail, summary


def make_plots(outputs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    def heatmap(ax, pivot: pd.DataFrame, *, vmin: float, vmax: float, title: str, fmt: str = ".2f") -> None:
        values = pivot.to_numpy(dtype=float)
        image = ax.imshow(values, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)), labels=[str(value) for value in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), labels=[str(value) for value in pivot.index])
        ax.set_xlabel(str(pivot.columns.name or "column"))
        ax.set_ylabel(str(pivot.index.name or "row"))
        ax.set_title(title)
        midpoint = (vmin + vmax) / 2
        for y, row in enumerate(values):
            for x, value in enumerate(row):
                if np.isfinite(value):
                    ax.text(x, y, format(value, fmt), ha="center", va="center", color="white" if value < midpoint else "black", fontsize=9)
        return image

    def barh(ax, labels: list[str], values: list[float], title: str, color: str) -> None:
        y = np.arange(len(labels))
        ax.barh(y, values, color=color)
        ax.set_yticks(y, labels=labels)
        ax.set_title(title)
        ax.axvline(0, color="black", lw=1)

    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    attention = outputs["attention_head_summary"]
    metrics = [
        ("nonthinking", "final_count_query", "broad_aggregation_score", "Direct: broad needle aggregation"),
        ("thinking", "trace_marker_query", "correct_prompt_needle_mass", "CoT: targeted k-to-k retrieval"),
        ("thinking", "final_count_query", "trace_markers_mass", "CoT final readout: trace-marker mass"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for ax, (mode, query, metric, title) in zip(axes, metrics):
        part = attention[(attention["mode"] == mode) & (attention.query_kind == query)]
        pivot = part.pivot(index="layer", columns="head", values=metric)
        image = heatmap(ax, pivot, vmin=0, vmax=1, title=title)
    fig.colorbar(image, ax=axes, fraction=.025, pad=.02)
    fig.savefig(figures / "attention_mechanism_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    ablation = outputs["head_ablation_summary"]
    for mode, metric, filename in [
        ("nonthinking", "drop_count_margin", "nonthinking_ablation.png"),
        ("thinking", "drop_trace_marker_margin", "thinking_retrieval_ablation.png"),
        ("thinking", "drop_count_margin", "thinking_readout_ablation.png"),
    ]:
        part = ablation[(ablation["mode"] == mode) & (ablation.group_name != "none")].sort_values(metric, ascending=False).head(14)
        fig, ax = plt.subplots(figsize=(9, max(4, .38 * len(part))))
        ordered = part.sort_values(metric, ascending=True)
        barh(ax, ordered.group_name.astype(str).tolist(), ordered[metric].astype(float).tolist(), f"{mode}: causal head-mask effect", "#3b73d9")
        ax.set_xlabel(metric)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    behavior = outputs["behavioral_ablation_summary"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, mode in zip(axes, ["nonthinking", "thinking"]):
        part = behavior[behavior["mode"] == mode].sort_values("final_accuracy", ascending=True)
        labels = part.group_name.astype(str).tolist()
        values = part.final_accuracy.astype(float).tolist()
        y = np.arange(len(labels))
        ax.barh(y, values, color="#e07a3f")
        ax.set_yticks(y, labels=labels)
        if mode == "thinking":
            for idx, row in enumerate(part.itertuples(index=False)):
                value = float(row.final_accuracy)
                if value >= .35:
                    ax.text(value - .02, idx, f"trace={float(row.trace_exact):.2f}", va="center", ha="right", fontsize=8)
                else:
                    ax.text(value + .02, idx, f"trace={float(row.trace_exact):.2f}", va="center", ha="left", fontsize=8)
        ax.set_xlim(0, 1.03)
        ax.set_title(f"Free-running behavior: {mode}")
    fig.savefig(figures / "behavioral_head_ablation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    patch = outputs["patching_summary"]
    patch_sets = {
        "patching_retrieval.png": ["retrieval_identity", "retrieval_identity_residual"],
        "patching_nonthinking_readout.png": ["nonthinking_count_readout", "nonthinking_count_readout_residual"],
        "patching_thinking_readout.png": ["thinking_count_readout", "thinking_count_readout_residual"],
    }
    for filename, experiments in patch_sets.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
        for ax, experiment in zip(axes, experiments):
            part = patch[patch.experiment == experiment].sort_values("normalized_recovery", ascending=False).head(12)
            ordered = part.sort_values("normalized_recovery", ascending=True)
            barh(ax, ordered.group_name.astype(str).tolist(), ordered.normalized_recovery.astype(float).tolist(), experiment, "#1f9d68")
            ax.axvline(1, color="gray", lw=1, ls="--")
            ax.set_xlabel("normalized recovery")
        fig.savefig(figures / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    conflict = outputs["trace_conflict_summary"].pivot(index="prompt_count", columns="forced_trace_count", values="pred_count")
    fig, ax = plt.subplots(figsize=(9, 6))
    image = heatmap(ax, conflict, vmin=1, vmax=10, title="Final count under prompt/trace conflict", fmt=".1f")
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    ax.set_xlabel("forced trace count")
    ax.set_ylabel("prompt count")
    fig.tight_layout()
    fig.savefig(figures / "trace_conflict_pred_count.png", dpi=180)
    plt.close(fig)

    progress = outputs["progress_transplant_summary"]
    fig, ax = plt.subplots(figsize=(9, 5))
    shifted = progress[progress.donor_kind == "same_example_progress_shift"]
    layers = sorted(shifted.layer.unique())
    offsets = sorted(shifted.donor_offset.unique())
    width = .8 / max(1, len(offsets))
    for offset_idx, offset in enumerate(offsets):
        part = shifted[shifted.donor_offset == offset].set_index("layer").reindex(layers)
        x = np.arange(len(layers)) + (offset_idx - (len(offsets) - 1) / 2) * width
        ax.bar(x, part.margin_shift_toward_donor.astype(float), width=width, label=f"donor offset {int(offset):+d}")
    ax.set_xticks(range(len(layers)), labels=[str(layer) for layer in layers])
    ax.legend()
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Progress-state transplant: shift toward k-1 or k+1 donor")
    ax.set_ylabel("change in donor-next-token margin")
    fig.tight_layout()
    fig.savefig(figures / "progress_state_transplant.png", dpi=180)
    plt.close(fig)


@torch.no_grad()
def run_v5_3_mechanism_causal(
    run_dir: str | Path,
    *,
    attention_examples_per_count: int = 30,
    ablation_examples_per_count: int = 10,
    generation_examples_per_count: int = 3,
    patch_examples_per_count: int = 10,
    conflict_examples_per_count: int = 10,
    device: str | None = None,
    seed_offset: int = 93_000,
) -> dict[str, pd.DataFrame]:
    run_dir = resolve_v5_run_dir(run_dir)
    cfg, vocab, model = load_v5_state(run_dir, device=device)
    if not bool(cfg.get("trace_indices")):
        raise ValueError("v5.3 requires the corrected indexed trace checkpoint (trace_indices=true).")
    train = cfg["train"]
    seq_len = int(train["seq_len"])
    seed = int(train["seed"]) + int(seed_offset)
    out_dir = run_dir / "v5_3_mechanism_causal"
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    print(f"[v5.3] attention signatures: {attention_examples_per_count} examples/count", flush=True)
    attention_examples = balanced_examples(seq_len, attention_examples_per_count, seed)
    attention_detail, attention_summary = collect_attention_signatures(model, vocab, attention_examples, cfg["device"])
    groups = select_head_groups(
        attention_summary,
        seed=seed + 1,
        n_layer=int(model.config.n_layer),
        n_head=int(model.config.n_head),
    )
    (out_dir / "head_groups.json").write_text(json.dumps({key: [list(head) for head in value] for key, value in groups.items()}, indent=2), encoding="utf-8")

    print(f"[v5.3] teacher-forced head ablation: {ablation_examples_per_count} examples/count", flush=True)
    ablation_examples = balanced_examples(seq_len, ablation_examples_per_count, seed + 2)
    ablation_detail, ablation_summary = run_head_ablation(model, vocab, ablation_examples, groups, cfg["device"])
    print(f"[v5.3] free-running head ablation: {generation_examples_per_count} examples/count", flush=True)
    generation_examples = balanced_examples(seq_len, generation_examples_per_count, seed + 5)
    behavior_detail, behavior_summary = run_behavioral_ablation(model, vocab, generation_examples, groups, cfg["device"])
    print(f"[v5.3] clean-to-corrupt head/residual patching: {patch_examples_per_count} examples/count", flush=True)
    patch_examples = balanced_examples(seq_len, patch_examples_per_count, seed + 3, count_min=2, count_max=10)
    patch_detail, patch_summary = run_patching(model, vocab, patch_examples, groups, cfg["device"])
    print(f"[v5.3] trace conflict and progress transplant: {conflict_examples_per_count} examples/count", flush=True)
    conflict_examples = balanced_examples(seq_len, conflict_examples_per_count, seed + 4)
    conflict_detail, conflict_summary = run_trace_conflict(model, vocab, conflict_examples, cfg["device"])
    progress_detail, progress_summary = run_progress_state_transplant(model, vocab, conflict_examples, cfg["device"])

    outputs = {
        "attention_rows": attention_detail,
        "attention_head_summary": attention_summary,
        "head_ablation_rows": ablation_detail,
        "head_ablation_summary": ablation_summary,
        "behavioral_ablation_rows": behavior_detail,
        "behavioral_ablation_summary": behavior_summary,
        "patching_rows": patch_detail,
        "patching_summary": patch_summary,
        "trace_conflict_rows": conflict_detail,
        "trace_conflict_summary": conflict_summary,
        "progress_transplant_rows": progress_detail,
        "progress_transplant_summary": progress_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(tables / f"{name}.csv", index=False)
    make_plots(outputs, out_dir)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "source_run": str(run_dir),
                "attention_examples_per_count": attention_examples_per_count,
                "ablation_examples_per_count": ablation_examples_per_count,
                "generation_examples_per_count": generation_examples_per_count,
                "patch_examples_per_count": patch_examples_per_count,
                "conflict_examples_per_count": conflict_examples_per_count,
                "seed_offset": seed_offset,
                "device": cfg["device"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[v5.3] complete: {out_dir}", flush=True)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v5.3 attention/ablation/patching mechanism diagnostics")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--attention-examples-per-count", type=int, default=30)
    parser.add_argument("--ablation-examples-per-count", type=int, default=10)
    parser.add_argument("--generation-examples-per-count", type=int, default=3)
    parser.add_argument("--patch-examples-per-count", type=int, default=10)
    parser.add_argument("--conflict-examples-per-count", type=int, default=10)
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = run_v5_3_mechanism_causal(
        args.run_dir,
        attention_examples_per_count=args.attention_examples_per_count,
        ablation_examples_per_count=args.ablation_examples_per_count,
        generation_examples_per_count=args.generation_examples_per_count,
        patch_examples_per_count=args.patch_examples_per_count,
        conflict_examples_per_count=args.conflict_examples_per_count,
        device=args.device,
    )
    print({key: len(value) for key, value in outputs.items()})


if __name__ == "__main__":
    main()
