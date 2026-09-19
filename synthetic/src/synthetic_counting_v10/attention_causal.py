from __future__ import annotations

import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from .config import V10Config
from .core import Example, Rendered, Vocab, balanced_examples, count_prediction, margin, render
from .training import load_final_model


Head = tuple[int, int]


def _entropy(weights: np.ndarray, normalized: bool = True) -> float:
    values = np.asarray(weights, dtype=float)
    total = float(values.sum())
    if total <= 1e-12 or len(values) <= 1:
        return 0.0
    probabilities = values / total
    entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())
    return entropy / math.log(len(values)) if normalized else entropy


def attention_categories(rendered: Rendered, row: np.ndarray) -> dict[str, float]:
    spans = rendered.spans
    needle_positions = list(rendered.prompt_needle_positions)
    needle_set = set(needle_positions)
    prompt_positions = list(range(spans.prompt_start, spans.prompt_end_exclusive))
    noise_positions = [pos for pos in prompt_positions if pos not in needle_set]
    needle_weights = row[needle_positions] if needle_positions else np.asarray([], dtype=float)
    needle_mass = float(needle_weights.sum()) if len(needle_weights) else 0.0
    categories = {
        "bos_mass": float(row[spans.bos_pos]),
        "prompt_needles_mass": needle_mass,
        "prompt_noise_mass": float(row[noise_positions].sum()) if noise_positions else 0.0,
        "think_open_mass": float(row[spans.think_pos]) if spans.think_pos is not None else 0.0,
        "trace_indices_mass": float(row[spans.trace_index_positions].sum()) if spans.trace_index_positions else 0.0,
        "trace_markers_mass": float(row[spans.trace_marker_positions].sum()) if spans.trace_marker_positions else 0.0,
        "think_close_mass": float(row[spans.think_close_pos]) if spans.think_close_pos is not None else 0.0,
        "ans_mass": float(row[spans.ans_pos]),
        "needle_entropy_normalized": _entropy(needle_weights),
        "needle_effective_number": float(math.exp(_entropy(needle_weights, normalized=False))) if len(needle_weights) else 0.0,
    }
    categories["broad_attention_score"] = needle_mass * categories["needle_entropy_normalized"]
    categories["classified_mass"] = sum(
        categories[name]
        for name in (
            "bos_mass",
            "prompt_needles_mass",
            "prompt_noise_mass",
            "think_open_mass",
            "trace_indices_mass",
            "trace_markers_mass",
            "think_close_mass",
            "ans_mass",
        )
    )
    categories["other_or_query_self_mass"] = max(0.0, 1.0 - categories["classified_mass"])
    return categories


@torch.no_grad()
def collect_attention(
    model,
    vocab: Vocab,
    examples: list[Example],
    mode: str,
    device: str | torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for example_idx, example in enumerate(examples):
        item = render(example, vocab, mode)
        ids = torch.tensor([item.input_ids], dtype=torch.long, device=device)
        output = model(input_ids=ids, output_attentions=True)
        attentions = [layer[0].detach().float().cpu().numpy() for layer in output.attentions or []]
        queries: list[tuple[str, int, int | None]] = [("final_count_query", item.spans.ans_pos, None)]
        if mode == "thinking":
            queries.extend(
                ("targeted_retrieval_query", pos, k)
                for k, pos in enumerate(item.spans.trace_index_positions)
            )
            queries.extend(
                ("successor_query", pos, k)
                for k, pos in enumerate(item.spans.trace_marker_positions)
            )
        for layer, layer_attention in enumerate(attentions):
            for head in range(layer_attention.shape[0]):
                for query_kind, query_pos, k in queries:
                    row = layer_attention[head, query_pos]
                    values = attention_categories(item, row)
                    correct_mass = math.nan
                    correct_top1 = math.nan
                    diagonal = math.nan
                    next_needle_mass = math.nan
                    if k is not None and k < len(item.prompt_needle_positions):
                        weights = row[item.prompt_needle_positions]
                        correct_mass = float(weights[k])
                        correct_top1 = float(int(np.argmax(weights) == k))
                        diagonal = correct_mass / max(float(weights.sum()), 1e-12)
                        if query_kind == "successor_query" and k + 1 < len(item.prompt_needle_positions):
                            next_needle_mass = float(row[item.prompt_needle_positions[k + 1]])
                    rows.append(
                        {
                            "example_idx": example_idx,
                            "mode": mode,
                            "count": example.count,
                            "query_kind": query_kind,
                            "query_k": math.nan if k is None else k + 1,
                            "layer": layer,
                            "head": head,
                            "correct_prompt_needle_mass": correct_mass,
                            "correct_top1": correct_top1,
                            "diagonal_dominance": diagonal,
                            "next_prompt_needle_mass": next_needle_mass,
                            **values,
                        }
                    )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["mode", "query_kind", "layer", "head"], as_index=False)
        .mean(numeric_only=True)
        .drop(columns=["example_idx"], errors="ignore")
    )
    return detail, summary


def _sorted_heads(summary: pd.DataFrame, query_kind: str, metric: str) -> list[Head]:
    rows = summary[summary.query_kind == query_kind].sort_values(metric, ascending=False)
    return [(int(row.layer), int(row.head)) for row in rows.itertuples(index=False)]


def build_head_rankings(
    nonthinking_summary: pd.DataFrame,
    thinking_summary: pd.DataFrame,
    seed: int,
) -> dict[str, list[Head]]:
    direct = _sorted_heads(nonthinking_summary, "final_count_query", "broad_attention_score")
    targeted = _sorted_heads(thinking_summary, "targeted_retrieval_query", "correct_prompt_needle_mass")
    trace_readout = _sorted_heads(thinking_summary, "final_count_query", "trace_markers_mass")
    successor = _sorted_heads(thinking_summary, "successor_query", "next_prompt_needle_mass")
    all_heads = sorted(set(direct) | set(targeted) | set(trace_readout) | set(successor))
    random_heads = list(all_heads)
    random.Random(seed).shuffle(random_heads)
    return {
        "direct_broad": direct,
        "targeted_retrieval": targeted,
        "trace_readout": trace_readout,
        "successor": successor,
        "random": random_heads,
    }


def _head_mask(model, heads: Iterable[Head], device: str | torch.device) -> torch.Tensor:
    mask = torch.ones((int(model.config.n_layer), int(model.config.n_head)), dtype=torch.float32, device=device)
    for layer, head in heads:
        mask[int(layer), int(head)] = 0.0
    return mask


def _head_label(heads: Iterable[Head]) -> str:
    return " ".join(f"L{layer + 1}H{head}" for layer, head in heads)


@torch.no_grad()
def _teacher_forced_metrics(
    model,
    vocab: Vocab,
    examples: list[Example],
    mode: str,
    head_mask: torch.Tensor | None,
    device: str | torch.device,
) -> dict[str, float]:
    count_accuracy: list[float] = []
    count_margins: list[float] = []
    marker_accuracy: list[float] = []
    marker_margins: list[float] = []
    index_accuracy: list[float] = []
    for example in examples:
        item = render(example, vocab, mode)
        ids = torch.tensor([item.input_ids], dtype=torch.long, device=device)
        logits = model(input_ids=ids, head_mask=head_mask).logits[0]
        pred, _, _ = count_prediction(logits[item.spans.ans_pos], vocab)
        count_accuracy.append(float(pred == example.count))
        count_margins.append(margin(logits[item.spans.ans_pos], vocab.number_id(example.count), vocab.number_ids))
        if mode == "thinking":
            for k, (index_pos, marker_pos) in enumerate(
                zip(item.spans.trace_index_positions, item.spans.trace_marker_positions), start=1
            ):
                target_marker = item.input_ids[marker_pos]
                marker_logits = logits[index_pos]
                marker_accuracy.append(float(int(marker_logits.argmax().item()) == target_marker))
                marker_margins.append(margin(marker_logits, target_marker, vocab.marker_ids))
                query_pos = item.spans.think_pos if k == 1 else item.spans.trace_marker_positions[k - 2]
                index_accuracy.append(float(int(logits[query_pos].argmax().item()) == vocab.number_id(k)))
    return {
        "final_count_accuracy": float(np.mean(count_accuracy)),
        "final_count_margin": float(np.mean(count_margins)),
        "trace_marker_accuracy": float(np.mean(marker_accuracy)) if marker_accuracy else math.nan,
        "trace_marker_margin": float(np.mean(marker_margins)) if marker_margins else math.nan,
        "trace_index_accuracy": float(np.mean(index_accuracy)) if index_accuracy else math.nan,
    }


@torch.no_grad()
def run_topn_ablation(
    models: dict[str, Any],
    vocab: Vocab,
    examples: list[Example],
    rankings: dict[str, list[Head]],
    device: str | torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    specifications = {
        "nonthinking": ("direct_broad", "random"),
        "thinking": ("targeted_retrieval", "trace_readout", "successor", "random"),
    }
    for mode, ranking_names in specifications.items():
        model = models[mode]
        baseline = _teacher_forced_metrics(model, vocab, examples, mode, None, device)
        rows.append({"mode": mode, "ranking": "none", "top_n": 0, "masked_heads": "", **baseline})
        for ranking_name in ranking_names:
            ranking = rankings[ranking_name]
            for top_n in range(1, len(ranking) + 1):
                heads = ranking[:top_n]
                metrics = _teacher_forced_metrics(model, vocab, examples, mode, _head_mask(model, heads, device), device)
                rows.append(
                    {
                        "mode": mode,
                        "ranking": ranking_name,
                        "top_n": top_n,
                        "masked_heads": _head_label(heads),
                        **metrics,
                    }
                )
    detail = pd.DataFrame(rows)
    baseline = detail[detail.top_n == 0].set_index("mode")
    for metric in (
        "final_count_accuracy",
        "final_count_margin",
        "trace_marker_accuracy",
        "trace_marker_margin",
        "trace_index_accuracy",
    ):
        detail[f"drop_{metric}"] = detail.apply(
            lambda row: float(baseline.loc[row["mode"], metric] - row[metric])
            if np.isfinite(row[metric]) and np.isfinite(baseline.loc[row["mode"], metric])
            else math.nan,
            axis=1,
        )
    return detail, detail.copy()


def marker_identity_corruption(example: Example, k: int, vocab: Vocab) -> Example:
    index = int(k) - 1
    markers = list(example.needle_markers)
    old = markers[index]
    replacement = next(marker for marker in vocab.markers if marker != old)
    markers[index] = replacement
    tokens = list(example.seq_tokens)
    tokens[example.needle_positions[index]] = replacement
    return replace(example, needle_markers=markers, seq_tokens=tokens)


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


def _patched_head_forward(
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
        def hook(_module, args, layer=layer, layer_heads=tuple(layer_heads)):
            value = args[0].clone()
            donor = donor_inputs[layer].to(value.device)
            for head in layer_heads:
                start = int(head) * head_dim
                value[:, receiver_pos, start : start + head_dim] = donor[:, donor_pos, start : start + head_dim]
            return (value, *args[1:])

        handles.append(model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(hook))
    try:
        return model(input_ids=receiver_ids).logits[0]
    finally:
        for handle in handles:
            handle.remove()


def normalized_recovery(clean: float, corrupt: float, patched: float) -> float:
    denominator = clean - corrupt
    return (patched - corrupt) / denominator if abs(denominator) > 1e-8 else math.nan


@torch.no_grad()
def run_retrieval_patching(
    model,
    vocab: Vocab,
    examples: list[Example],
    rankings: dict[str, list[Head]],
    device: str | torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    targeted = rankings["targeted_retrieval"]
    top_ns = sorted(set([1, 2, 4, 8, 12, len(targeted)]))
    layers = set(range(int(model.config.n_layer)))
    for example_idx, clean_example in enumerate(examples):
        k = clean_example.count
        corrupt_example = marker_identity_corruption(clean_example, k, vocab)
        clean_item = render(clean_example, vocab, "thinking")
        corrupt_item = render(corrupt_example, vocab, "thinking")
        clean_ids = torch.tensor([clean_item.input_ids], dtype=torch.long, device=device)
        corrupt_ids = torch.tensor([corrupt_item.input_ids], dtype=torch.long, device=device)
        query_pos = clean_item.spans.trace_index_positions[k - 1]
        clean_target = vocab.token_to_id[clean_example.needle_markers[k - 1]]
        corrupt_target = vocab.token_to_id[corrupt_example.needle_markers[k - 1]]
        candidates = [clean_target, corrupt_target]
        clean_logits = model(input_ids=clean_ids).logits[0, query_pos]
        corrupt_logits = model(input_ids=corrupt_ids).logits[0, query_pos]
        clean_margin = margin(clean_logits, clean_target, candidates)
        corrupt_margin = margin(corrupt_logits, clean_target, candidates)
        donor = _capture_cproj_inputs(model, clean_ids, layers)
        for ranking_name in ("targeted_retrieval", "random"):
            ranking = rankings[ranking_name]
            for top_n in top_ns:
                heads = ranking[:top_n]
                patched_logits = _patched_head_forward(model, corrupt_ids, donor, heads, query_pos, query_pos)[query_pos]
                patched_margin = margin(patched_logits, clean_target, candidates)
                rows.append(
                    {
                        "example_idx": example_idx,
                        "count": clean_example.count,
                        "ranking": ranking_name,
                        "top_n": top_n,
                        "patched_heads": _head_label(heads),
                        "donor_position_control": "matched_query",
                        "clean_margin": clean_margin,
                        "corrupt_margin": corrupt_margin,
                        "patched_margin": patched_margin,
                        "normalized_recovery": normalized_recovery(clean_margin, corrupt_margin, patched_margin),
                    }
                )
        wrong_heads = targeted[: min(4, len(targeted))]
        if query_pos > 0 and wrong_heads:
            patched_logits = _patched_head_forward(
                model, corrupt_ids, donor, wrong_heads, query_pos - 1, query_pos
            )[query_pos]
            patched_margin = margin(patched_logits, clean_target, candidates)
            rows.append(
                {
                    "example_idx": example_idx,
                    "count": clean_example.count,
                    "ranking": "targeted_wrong_donor_position",
                    "top_n": len(wrong_heads),
                    "patched_heads": _head_label(wrong_heads),
                    "donor_position_control": "previous_token",
                    "clean_margin": clean_margin,
                    "corrupt_margin": corrupt_margin,
                    "patched_margin": patched_margin,
                    "normalized_recovery": normalized_recovery(clean_margin, corrupt_margin, patched_margin),
                }
            )
    detail = pd.DataFrame(rows)
    summary = detail.groupby(
        ["ranking", "top_n", "patched_heads", "donor_position_control"], as_index=False
    ).mean(numeric_only=True)
    return detail, summary


def _paired_example(cfg: V10Config, vocab: Vocab, count: int, seed: int) -> Example:
    return balanced_examples(cfg, vocab, 1, seed, count_min=count, count_max=count)[0]


@torch.no_grad()
def run_count_offset_head_patching(
    models: dict[str, Any],
    cfg: V10Config,
    vocab: Vocab,
    rankings: dict[str, list[Head]],
    device: str | torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    top_ns = [n for n in (1, 2, 4, 8, 12, 16) if n <= cfg.n_layer * cfg.n_head]
    for mode in cfg.modes:
        model = models[mode]
        primary_ranking = "direct_broad" if mode == "nonthinking" else "trace_readout"
        layers = set(range(cfg.n_layer))
        for receiver_count in range(cfg.count_min, cfg.count_max + 1):
            receiver = _paired_example(cfg, vocab, receiver_count, cfg.seed + 200_000 + receiver_count)
            receiver_item = render(receiver, vocab, mode)
            receiver_ids = torch.tensor([receiver_item.input_ids], dtype=torch.long, device=device)
            receiver_pos = receiver_item.spans.ans_pos
            baseline_logits = model(input_ids=receiver_ids).logits[0, receiver_pos]
            baseline_pred, baseline_expected, _ = count_prediction(baseline_logits, vocab)
            for offset in cfg.patch_offsets:
                donor_count = receiver_count + int(offset)
                if not cfg.count_min <= donor_count <= cfg.count_max:
                    continue
                donor = _paired_example(cfg, vocab, donor_count, cfg.seed + 300_000 + donor_count)
                donor_item = render(donor, vocab, mode)
                donor_ids = torch.tensor([donor_item.input_ids], dtype=torch.long, device=device)
                donor_pos = donor_item.spans.ans_pos
                donor_inputs = _capture_cproj_inputs(model, donor_ids, layers)
                for ranking_name in (primary_ranking, "random"):
                    ranking = rankings[ranking_name]
                    for top_n in top_ns:
                        heads = ranking[:top_n]
                        patched_logits = _patched_head_forward(
                            model, receiver_ids, donor_inputs, heads, donor_pos, receiver_pos
                        )[receiver_pos]
                        pred, expected, _ = count_prediction(patched_logits, vocab)
                        rows.append(
                            {
                                "mode": mode,
                                "ranking": ranking_name,
                                "receiver_count": receiver_count,
                                "donor_count": donor_count,
                                "donor_offset": offset,
                                "top_n": top_n,
                                "patched_heads": _head_label(heads),
                                "baseline_pred": baseline_pred,
                                "baseline_expected": baseline_expected,
                                "patched_pred": pred,
                                "patched_expected": expected,
                                "causal_expected_shift": expected - baseline_expected,
                                "follows_donor": float(pred == donor_count),
                                "follows_receiver": float(pred == receiver_count),
                                "position_delta": donor_pos - receiver_pos,
                            }
                        )
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["mode", "ranking", "donor_offset", "top_n"], as_index=False).mean(numeric_only=True)
    return detail, summary


def save_head_rankings(rankings: dict[str, list[Head]], path: Path) -> None:
    path.write_text(
        json.dumps({name: [[layer, head] for layer, head in heads] for name, heads in rankings.items()}, indent=2),
        encoding="utf-8",
    )


def run_attention_causal(
    cfg: V10Config,
    vocab: Vocab,
    run_dir: Path,
) -> dict[str, pd.DataFrame]:
    out_dir = run_dir / "analysis" / "attention_causal"
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    models = {mode: load_final_model(cfg, vocab, run_dir, mode) for mode in cfg.modes}
    attention_examples = balanced_examples(
        cfg, vocab, cfg.attention_examples_per_count, cfg.seed + 70_000
    )
    non_detail, non_summary = collect_attention(models["nonthinking"], vocab, attention_examples, "nonthinking", cfg.device)
    think_detail, think_summary = collect_attention(models["thinking"], vocab, attention_examples, "thinking", cfg.device)
    rankings = build_head_rankings(non_summary, think_summary, cfg.seed + 71_000)
    save_head_rankings(rankings, out_dir / "head_rankings.json")

    causal_examples = balanced_examples(
        cfg, vocab, cfg.attention_causal_examples_per_count, cfg.seed + 72_000
    )
    ablation_rows, ablation_summary = run_topn_ablation(models, vocab, causal_examples, rankings, cfg.device)
    retrieval_rows, retrieval_summary = run_retrieval_patching(
        models["thinking"], vocab, causal_examples, rankings, cfg.device
    )
    offset_rows, offset_summary = run_count_offset_head_patching(models, cfg, vocab, rankings, cfg.device)
    outputs = {
        "attention_rows": pd.concat([non_detail, think_detail], ignore_index=True),
        "attention_head_summary": pd.concat([non_summary, think_summary], ignore_index=True),
        "topn_ablation_rows": ablation_rows,
        "topn_ablation_summary": ablation_summary,
        "retrieval_patching_rows": retrieval_rows,
        "retrieval_patching_summary": retrieval_summary,
        "count_offset_head_patching_rows": offset_rows,
        "count_offset_head_patching_summary": offset_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(tables / f"{name}.csv", index=False)
    for model in models.values():
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs
