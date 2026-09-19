from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .attention_causal import Head, attention_categories
from .core import Vocab, balanced_examples, count_bin, count_prediction, margin, render
from .hidden_state_patching import _post_block_states
from .report_followups import load_rankings, load_run
from .training import load_final_model


MECHANISM_LABELS = {
    "nonthinking_broad": "Non-thinking broad aggregation",
    "cot_targeted": "CoT k-to-k retrieval",
    "cot_readout": "CoT trace readout",
}
BIN_ORDER = ["1-10", "11-20", "21-30"]


def _head_label(head: Head) -> str:
    return f"L{int(head[0]) + 1}H{int(head[1])}"


def _safe_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else math.nan


def _count_or_marker_margin(
    logits: torch.Tensor,
    vocab: Vocab,
    target_id: int,
    competitor_ids: list[int],
) -> float:
    return margin(logits, int(target_id), [int(value) for value in competitor_ids])


def _standardized_geometry(
    state: torch.Tensor,
    bank: dict[str, Any],
    label: int,
) -> dict[str, float]:
    value = state.detach().float().cpu().numpy().reshape(-1)
    z = (value - bank["mean"]) / bank["std"]
    distances = {
        int(label): float(np.linalg.norm(z - centroid) / math.sqrt(len(z)))
        for label, centroid in bank["centroids"].items()
    }
    true_distance = distances[int(label)]
    wrong_distance = min(
        distance for candidate, distance in distances.items()
        if candidate != int(label)
    )
    nearest = min(distances, key=distances.get)
    return {
        "centroid_margin": float(wrong_distance - true_distance),
        "nearest_centroid_correct": float(nearest == int(label)),
        "nearest_centroid_count": float(nearest),
    }


@torch.no_grad()
def _build_centroid_bank(
    model,
    cfg,
    vocab: Vocab,
    mode: str,
    site: str,
    label_kind: str,
    examples_per_count: int,
    seed: int,
) -> dict[int, dict[str, Any]]:
    examples = balanced_examples(cfg, vocab, int(examples_per_count), int(seed))
    if label_kind == "marker_identity":
        # A random balanced-count sample need not cover every marker class,
        # especially in debug runs.  Explicitly balance the final marker while
        # preserving each example's count, positions, and all other tokens.
        marker_balanced = []
        for marker_index, marker in enumerate(vocab.markers):
            for repeat in range(max(1, int(examples_per_count))):
                base = examples[(marker_index * max(1, int(examples_per_count)) + repeat) % len(examples)]
                seq_tokens = list(base.seq_tokens)
                needle_markers = list(base.needle_markers)
                seq_tokens[base.needle_positions[-1]] = marker
                needle_markers[-1] = marker
                marker_balanced.append(
                    replace(base, seq_tokens=seq_tokens, needle_markers=needle_markers)
                )
        examples = marker_balanced
    rows: dict[int, dict[int, list[np.ndarray]]] = {
        layer: {} for layer in range(int(cfg.n_layer))
    }
    for example in examples:
        item = render(example, vocab, mode)
        position = (
            item.spans.ans_pos
            if site == "final"
            else item.spans.trace_index_positions[-1]
        )
        ids = torch.tensor([item.input_ids], dtype=torch.long, device=cfg.device)
        _, states = _post_block_states(model, ids, int(position))
        for layer, state in enumerate(states):
            label = (
                int(vocab.token_to_id[example.needle_markers[-1]])
                if label_kind == "marker_identity"
                else int(example.count)
            )
            rows[layer].setdefault(label, []).append(
                state.detach().float().cpu().numpy().reshape(-1)
            )

    banks: dict[int, dict[str, Any]] = {}
    for layer, by_count in rows.items():
        all_values = np.stack([value for values in by_count.values() for value in values])
        mean = all_values.mean(axis=0)
        std = all_values.std(axis=0)
        std[std < 1e-5] = 1.0
        centroids = {
            count: ((np.stack(values).mean(axis=0) - mean) / std)
            for count, values in by_count.items()
        }
        banks[layer] = {"mean": mean, "std": std, "centroids": centroids}
    return banks


@torch.no_grad()
def _forward_with_local_head_mask(
    model,
    input_ids: torch.Tensor,
    heads: Iterable[Head],
    query_position: int,
    state_position: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(int(layer), []).append(int(head))
    head_dim = int(model.config.n_embd) // int(model.config.n_head)
    captured: dict[int, torch.Tensor] = {}
    handles = []

    for layer, layer_heads in by_layer.items():
        def mask_hook(_module, args, layer_heads=tuple(layer_heads)):
            value = args[0].clone()
            for head in layer_heads:
                start = int(head) * head_dim
                value[:, int(query_position), start : start + head_dim] = 0.0
            return (value, *args[1:])

        handles.append(model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(mask_hook))

    for layer, block in enumerate(model.transformer.h):
        def state_hook(_module, _args, output, layer=layer):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer] = hidden[:, int(state_position), :].detach().clone()

        handles.append(block.register_forward_hook(state_hook))

    try:
        logits = model(input_ids=input_ids).logits[0].detach()
    finally:
        for handle in handles:
            handle.remove()
    return logits, [captured[layer] for layer in range(len(model.transformer.h))]


def _interventions(primary: list[Head], random_heads: list[Head]) -> list[tuple[str, list[Head]]]:
    top4 = set(primary[:4])
    noncandidate4 = [head for head in random_heads if head not in top4][:4]
    top1 = primary[0]
    same_layer_controls = [
        head for head in reversed(primary)
        if head[0] == top1[0] and head != top1
    ]
    matched_top1 = same_layer_controls[:1] or noncandidate4[:1]
    return [
        ("clean", []),
        ("candidate_top1", list(primary[:1])),
        ("matched_control_top1", list(matched_top1)),
        ("candidate_top4", list(primary[:4])),
        ("noncandidate4_control", list(noncandidate4)),
    ]


@torch.no_grad()
def run_head_to_state(
    models: dict[str, Any],
    cfg,
    vocab: Vocab,
    rankings: dict[str, list[Head]],
    *,
    centroid_examples_per_count: int,
    eval_examples_per_count: int,
    seed: int,
) -> pd.DataFrame:
    banks = {
        "nonthinking_broad": _build_centroid_bank(
            models["nonthinking"], cfg, vocab, "nonthinking", "final",
            "count",
            centroid_examples_per_count, seed + 11,
        ),
        "cot_targeted": _build_centroid_bank(
            models["thinking"], cfg, vocab, "thinking", "progress",
            "marker_identity",
            centroid_examples_per_count, seed + 17,
        ),
        "cot_readout": _build_centroid_bank(
            models["thinking"], cfg, vocab, "thinking", "final",
            "count",
            centroid_examples_per_count, seed + 23,
        ),
    }
    examples = balanced_examples(cfg, vocab, int(eval_examples_per_count), seed + 101)
    specs = [
        ("nonthinking_broad", "nonthinking", "direct_broad", "final"),
        ("cot_targeted", "thinking", "targeted_retrieval", "progress"),
        ("cot_readout", "thinking", "trace_readout", "final"),
    ]
    rows: list[dict[str, Any]] = []
    for mechanism, mode, ranking_name, site in specs:
        model = models[mode]
        for example_index, example in enumerate(examples):
            item = render(example, vocab, mode)
            if site == "progress":
                query_position = int(item.spans.trace_index_positions[-1])
                target_id = vocab.token_to_id[example.needle_markers[-1]]
                competitor_ids = vocab.marker_ids
            else:
                query_position = int(item.spans.ans_pos)
                target_id = vocab.number_id(example.count)
                competitor_ids = vocab.number_ids
            ids = torch.tensor([item.input_ids], dtype=torch.long, device=cfg.device)
            for intervention, heads in _interventions(
                rankings[ranking_name], rankings.get("random", [])
            ):
                logits, states = _forward_with_local_head_mask(
                    model, ids, heads, query_position, query_position
                )
                query_logits = logits[query_position]
                predicted = (
                    count_prediction(query_logits, vocab)[0]
                    if site == "final"
                    else int(torch.argmax(query_logits[vocab.marker_ids]).item())
                )
                output_correct = (
                    float(predicted == int(example.count))
                    if site == "final"
                    else float(vocab.marker_ids[predicted] == int(target_id))
                )
                output_margin = _count_or_marker_margin(
                    query_logits, vocab, target_id, competitor_ids
                )
                for eval_layer, state in enumerate(states):
                    geometry_label = int(target_id) if site == "progress" else int(example.count)
                    geometry = _standardized_geometry(
                        state, banks[mechanism][eval_layer], geometry_label
                    )
                    rows.append({
                        "mechanism": mechanism,
                        "mode": mode,
                        "site": site,
                        "example_index": int(example_index),
                        "count": int(example.count),
                        "count_bin": count_bin(example.count),
                        "intervention": intervention,
                        "masked_heads": ",".join(_head_label(head) for head in heads),
                        "n_masked_heads": len(heads),
                        "eval_layer": int(eval_layer + 1),
                        "query_position": int(query_position),
                        "geometry_target": "marker_identity" if site == "progress" else "count",
                        "geometry_label": geometry_label,
                        "output_correct": output_correct,
                        "output_margin": output_margin,
                        **geometry,
                    })
    detail = pd.DataFrame(rows)
    clean = detail[detail.intervention == "clean"][
        ["mechanism", "example_index", "eval_layer", "centroid_margin", "output_margin"]
    ].rename(columns={"centroid_margin": "clean_centroid_margin", "output_margin": "clean_output_margin"})
    detail = detail.merge(clean, on=["mechanism", "example_index", "eval_layer"], how="left")
    detail["centroid_margin_drop"] = detail.clean_centroid_margin - detail.centroid_margin
    detail["output_margin_drop"] = detail.clean_output_margin - detail.output_margin
    return detail


@torch.no_grad()
def _pre_block_states(
    model,
    input_ids: torch.Tensor,
    position: int,
) -> list[torch.Tensor]:
    """Capture the residual stream immediately before every Transformer layer."""
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer, block in enumerate(model.transformer.h):
        def capture_hook(_module, args, layer=layer):
            captured[layer] = args[0][:, int(position), :].detach().clone()

        handles.append(block.register_forward_pre_hook(capture_hook))
    try:
        model(input_ids=input_ids)
    finally:
        for handle in handles:
            handle.remove()
    return [captured[layer] for layer in range(len(model.transformer.h))]


@torch.no_grad()
def _forward_with_residual_patch_and_attention(
    model,
    receiver_ids: torch.Tensor,
    patch_before_layer: int,
    receiver_position: int,
    donor_state: torch.Tensor,
) -> tuple[torch.Tensor, list[np.ndarray]]:
    """Patch the residual entering one layer, then expose that layer's attention onward."""
    def patch_hook(_module, args):
        hidden = args[0].clone()
        hidden[:, int(receiver_position), :] = donor_state.to(hidden.device, hidden.dtype)
        return (hidden, *args[1:])

    handle = model.transformer.h[int(patch_before_layer)].register_forward_pre_hook(patch_hook)
    try:
        output = model(input_ids=receiver_ids, output_attentions=True)
        attentions = [
            layer[0].detach().float().cpu().numpy() for layer in (output.attentions or [])
        ]
        return output.logits[0].detach(), attentions
    finally:
        handle.remove()


@torch.no_grad()
def _clean_forward_attention(model, ids: torch.Tensor) -> tuple[torch.Tensor, list[np.ndarray]]:
    output = model(input_ids=ids, output_attentions=True)
    return output.logits[0].detach(), [
        layer[0].detach().float().cpu().numpy() for layer in (output.attentions or [])
    ]


def _attention_signature(
    mechanism: str,
    item,
    row: np.ndarray,
    *,
    receiver_progress: int | None = None,
    donor_progress: int | None = None,
) -> dict[str, float]:
    categories = attention_categories(item, row)
    values = {
        "broad_attention_score": categories["broad_attention_score"],
        "prompt_needles_mass": categories["prompt_needles_mass"],
        "trace_markers_mass": categories["trace_markers_mass"],
        "last_trace_marker_mass": (
            float(row[item.spans.trace_marker_positions[-1]])
            if item.spans.trace_marker_positions else 0.0
        ),
        "receiver_progress_needle_mass": math.nan,
        "donor_progress_needle_mass": math.nan,
        "progress_routing_preference": math.nan,
    }
    if mechanism == "cot_targeted" and receiver_progress and donor_progress:
        receiver_pos = item.prompt_needle_positions[int(receiver_progress) - 1]
        donor_pos = item.prompt_needle_positions[int(donor_progress) - 1]
        values["receiver_progress_needle_mass"] = float(row[receiver_pos])
        values["donor_progress_needle_mass"] = float(row[donor_pos])
        values["progress_routing_preference"] = float(row[donor_pos] - row[receiver_pos])
    return values


def _state_to_head_scenarios(cfg) -> list[dict[str, int | str]]:
    candidates = [
        ("1-10", 5, 8, 3, 6),
        ("11-20", 15, 18, 6, 12),
        ("21-30", 25, 28, 10, 20),
    ]
    rows = []
    for bin_name, receiver_count, donor_count, receiver_progress, donor_progress in candidates:
        donor_count = min(int(donor_count), int(cfg.count_max))
        donor_progress = min(int(donor_progress), donor_count)
        if not (cfg.count_min <= receiver_count <= cfg.count_max):
            continue
        if not (cfg.count_min <= donor_count <= cfg.count_max):
            continue
        if receiver_progress > donor_count or donor_progress > donor_count:
            continue
        rows.append({
            "count_bin": bin_name,
            "receiver_count": int(receiver_count),
            "donor_count": int(donor_count),
            "targeted_count": int(donor_count),
            "receiver_progress": int(receiver_progress),
            "donor_progress": int(donor_progress),
        })
    return rows


@torch.no_grad()
def run_state_to_head(
    models: dict[str, Any],
    cfg,
    vocab: Vocab,
    rankings: dict[str, list[Head]],
    *,
    examples_per_bin: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    specs = [
        ("nonthinking_broad", "nonthinking", "direct_broad", "final"),
        ("cot_targeted", "thinking", "targeted_retrieval", "progress"),
        ("cot_readout", "thinking", "trace_readout", "final"),
    ]
    for scenario in _state_to_head_scenarios(cfg):
        bin_name = str(scenario["count_bin"])
        for pair_index in range(int(examples_per_bin)):
            for mechanism, mode, ranking_name, site in specs:
                if mechanism == "cot_targeted":
                    receiver_count = int(scenario["targeted_count"])
                    donor_count = receiver_count
                else:
                    receiver_count = int(scenario["receiver_count"])
                    donor_count = int(scenario["donor_count"])
                base_seed = seed + receiver_count * 1_000 + pair_index * 10
                receiver = balanced_examples(
                    cfg, vocab, 1, base_seed + 1,
                    count_min=receiver_count, count_max=receiver_count,
                )[0]
                same_donor = balanced_examples(
                    cfg, vocab, 1, base_seed + 2,
                    count_min=receiver_count, count_max=receiver_count,
                )[0]
                shifted_donor = balanced_examples(
                    cfg, vocab, 1, base_seed + 3,
                    count_min=donor_count, count_max=donor_count,
                )[0]
                model = models[mode]
                receiver_item = render(receiver, vocab, mode)
                same_item = render(same_donor, vocab, mode)
                shifted_item = render(shifted_donor, vocab, mode)
                if site == "progress":
                    receiver_progress = int(scenario["receiver_progress"])
                    donor_progress = int(scenario["donor_progress"])
                    receiver_pos = receiver_item.spans.trace_index_positions[receiver_progress - 1]
                    same_pos = same_item.spans.trace_index_positions[receiver_progress - 1]
                    shifted_pos = shifted_item.spans.trace_index_positions[donor_progress - 1]
                else:
                    receiver_progress = None
                    donor_progress = None
                    receiver_pos = receiver_item.spans.ans_pos
                    same_pos = same_item.spans.ans_pos
                    shifted_pos = shifted_item.spans.ans_pos

                receiver_ids = torch.tensor([receiver_item.input_ids], dtype=torch.long, device=cfg.device)
                same_ids = torch.tensor([same_item.input_ids], dtype=torch.long, device=cfg.device)
                shifted_ids = torch.tensor([shifted_item.input_ids], dtype=torch.long, device=cfg.device)
                clean_logits, clean_attn = _clean_forward_attention(model, receiver_ids)
                same_states = _pre_block_states(model, same_ids, int(same_pos))
                shifted_states = _pre_block_states(model, shifted_ids, int(shifted_pos))
                primary = set(rankings[ranking_name][:4])

                for patch_layer in range(int(cfg.n_layer)):
                    variants = [
                        ("clean", clean_logits, clean_attn),
                        (
                            "same_state_control",
                            *_forward_with_residual_patch_and_attention(
                                model, receiver_ids, patch_layer, int(receiver_pos), same_states[patch_layer]
                            ),
                        ),
                        (
                            "shifted_state_patch",
                            *_forward_with_residual_patch_and_attention(
                                model, receiver_ids, patch_layer, int(receiver_pos), shifted_states[patch_layer]
                            ),
                        ),
                    ]
                    for intervention, logits, attentions in variants:
                        for attention_layer in range(patch_layer, int(cfg.n_layer)):
                            for head in range(int(cfg.n_head)):
                                query_row = attentions[attention_layer][head, int(receiver_pos)]
                                signature = _attention_signature(
                                    mechanism,
                                    receiver_item,
                                    query_row,
                                    receiver_progress=receiver_progress,
                                    donor_progress=donor_progress,
                                )
                                query_logits = logits[int(receiver_pos)]
                                pred = count_prediction(query_logits, vocab)[0] if site == "final" else math.nan
                                rows.append({
                                    "mechanism": mechanism,
                                    "mode": mode,
                                    "site": site,
                                    "count_bin": bin_name,
                                    "pair_index": int(pair_index),
                                    "receiver_count": int(receiver.count),
                                    "donor_count": int(shifted_donor.count),
                                    "receiver_progress": receiver_progress,
                                    "donor_progress": donor_progress,
                                    "intervention": intervention,
                                    "patch_before_layer": int(patch_layer + 1),
                                    "attention_layer": int(attention_layer + 1),
                                    "head": int(head),
                                    "head_label": _head_label((attention_layer, head)),
                                    "is_candidate_top4": float((attention_layer, head) in primary),
                                    "query_position": int(receiver_pos),
                                    "predicted_count": pred,
                                    "predicted_receiver_count": (
                                        float(pred == int(receiver.count)) if site == "final" else math.nan
                                    ),
                                    "predicted_shifted_donor_count": (
                                        float(pred == int(shifted_donor.count)) if site == "final" else math.nan
                                    ),
                                    "count_margin": (
                                        _count_or_marker_margin(
                                            query_logits, vocab, vocab.number_id(receiver.count), vocab.number_ids
                                        ) if site == "final" else math.nan
                                    ),
                                    **signature,
                                })
    detail = pd.DataFrame(rows)
    metric_by_mechanism = {
        "nonthinking_broad": "broad_attention_score",
        "cot_targeted": "progress_routing_preference",
        "cot_readout": "trace_markers_mass",
    }
    detail["signature_metric"] = detail.apply(
        lambda row: row[metric_by_mechanism[row.mechanism]], axis=1
    )
    clean = detail[detail.intervention == "clean"][
        ["mechanism", "count_bin", "pair_index", "patch_before_layer", "attention_layer", "head", "signature_metric", "count_margin"]
    ].rename(columns={"signature_metric": "clean_signature_metric", "count_margin": "clean_count_margin"})
    detail = detail.merge(
        clean,
        on=["mechanism", "count_bin", "pair_index", "patch_before_layer", "attention_layer", "head"],
        how="left",
    )
    detail["signature_shift"] = detail.signature_metric - detail.clean_signature_metric
    detail["count_margin_shift"] = detail.count_margin - detail.clean_count_margin
    return detail


def summarize_head_to_state(detail: pd.DataFrame) -> pd.DataFrame:
    return (
        detail.groupby(
            ["mechanism", "count_bin", "intervention", "eval_layer"], as_index=False
        )
        .agg(
            n=("example_index", "nunique"),
            nearest_centroid_accuracy=("nearest_centroid_correct", "mean"),
            mean_centroid_margin=("centroid_margin", "mean"),
            mean_centroid_margin_drop=("centroid_margin_drop", "mean"),
            output_accuracy=("output_correct", "mean"),
            mean_output_margin=("output_margin", "mean"),
            mean_output_margin_drop=("output_margin_drop", "mean"),
        )
    )


def summarize_state_to_head(detail: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for scope, scoped in (
        ("candidate_top4", detail[detail.is_candidate_top4 == 1.0]),
        ("all_downstream", detail),
    ):
        summary = (
            scoped.groupby(
                ["mechanism", "count_bin", "intervention", "patch_before_layer"],
                as_index=False,
            )
            .agg(
                n_pairs=("pair_index", "nunique"),
                n_heads=("head_label", "nunique"),
                mean_signature=("signature_metric", "mean"),
                mean_signature_shift=("signature_shift", "mean"),
                mean_abs_signature_shift=("signature_shift", lambda values: float(np.abs(values).mean())),
                max_abs_signature_shift=("signature_shift", lambda values: float(np.abs(values).max())),
                mean_count_margin_shift=("count_margin_shift", "mean"),
                receiver_count_retention=("predicted_receiver_count", "mean"),
                shifted_donor_adoption=("predicted_shifted_donor_count", "mean"),
            )
        )
        summary["head_scope"] = scope
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


def _plot_head_to_state(summary: pd.DataFrame, out: Path) -> None:
    mechanisms = list(MECHANISM_LABELS)
    colors = {"1-10": "#2f6fed", "11-20": "#e66a1f", "21-30": "#1b9e77"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for axis, mechanism in zip(axes, mechanisms):
        subset = summary[(summary.mechanism == mechanism) & (summary.eval_layer == summary.eval_layer.max())]
        interventions = ["clean", "candidate_top1", "matched_control_top1", "candidate_top4", "noncandidate4_control"]
        x = np.arange(len(interventions))
        width = 0.24
        for index, bin_name in enumerate(BIN_ORDER):
            values = []
            for intervention in interventions:
                rows = subset[(subset.count_bin == bin_name) & (subset.intervention == intervention)]
                values.append(float(rows.nearest_centroid_accuracy.iloc[0]) if len(rows) else math.nan)
            axis.bar(x + (index - 1) * width, values, width, label=bin_name, color=colors[bin_name])
        axis.set_title(MECHANISM_LABELS[mechanism])
        axis.set_xticks(x, ["clean", "top-1", "matched-1", "top-4", "noncand-4"], rotation=20, ha="right")
        axis.set_ylim(0, 1.04)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("nearest-centroid semantic-state accuracy")
    axes[-1].legend(title="gold count bin", loc="lower left")
    fig.suptitle("Head intervention -> downstream hidden-state geometry", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_head_to_output(summary: pd.DataFrame, out: Path) -> None:
    mechanisms = list(MECHANISM_LABELS)
    colors = {"1-10": "#2f6fed", "11-20": "#e66a1f", "21-30": "#1b9e77"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for axis, mechanism in zip(axes, mechanisms):
        subset = summary[(summary.mechanism == mechanism) & (summary.eval_layer == summary.eval_layer.max())]
        interventions = ["clean", "candidate_top1", "matched_control_top1", "candidate_top4", "noncandidate4_control"]
        x = np.arange(len(interventions))
        width = 0.24
        for index, bin_name in enumerate(BIN_ORDER):
            values = []
            for intervention in interventions:
                rows = subset[(subset.count_bin == bin_name) & (subset.intervention == intervention)]
                values.append(float(rows.output_accuracy.iloc[0]) if len(rows) else math.nan)
            axis.bar(x + (index - 1) * width, values, width, label=bin_name, color=colors[bin_name])
        axis.set_title(MECHANISM_LABELS[mechanism])
        axis.set_xticks(x, ["clean", "top-1", "matched-1", "top-4", "noncand-4"], rotation=20, ha="right")
        axis.set_ylim(0, 1.04)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("local output accuracy")
    axes[-1].legend(title="gold count bin", loc="lower left")
    fig.suptitle("Head intervention -> marker/final-count output", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_state_to_head(summary: pd.DataFrame, out: Path) -> None:
    mechanisms = list(MECHANISM_LABELS)
    colors = {"1-10": "#2f6fed", "11-20": "#e66a1f", "21-30": "#1b9e77"}
    styles = {"same_state_control": "--", "shifted_state_patch": "-"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    for axis, mechanism in zip(axes, mechanisms):
        scope = "all_downstream" if mechanism == "nonthinking_broad" else "candidate_top4"
        subset = summary[
            (summary.mechanism == mechanism)
            & (summary.intervention != "clean")
            & (summary.head_scope == scope)
        ]
        for bin_name in BIN_ORDER:
            for intervention in ("same_state_control", "shifted_state_patch"):
                rows = subset[(subset.count_bin == bin_name) & (subset.intervention == intervention)].sort_values("patch_before_layer")
                if rows.empty:
                    continue
                axis.plot(
                    rows.patch_before_layer,
                    rows.mean_signature_shift,
                    marker="o",
                    linestyle=styles[intervention],
                    color=colors[bin_name],
                    alpha=1.0 if intervention == "shifted_state_patch" else 0.55,
                    label=f"{bin_name} | {'shifted' if intervention == 'shifted_state_patch' else 'same-state ctrl'}",
                )
        axis.axhline(0, color="#555", linewidth=1)
        axis.set_title(MECHANISM_LABELS[mechanism])
        axis.set_xlabel("residual patched before Layer")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("downstream attention-signature shift")
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle("Hidden-state transplant -> downstream attention routing", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_mediation(head_summary: pd.DataFrame, out: Path) -> None:
    rows = head_summary[
        (head_summary.intervention != "clean")
        & (head_summary.eval_layer == head_summary.eval_layer.max())
    ].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    for axis, mechanism in zip(axes, MECHANISM_LABELS):
        subset = rows[rows.mechanism == mechanism]
        for intervention, marker, color in [
            ("candidate_top1", "o", "#2f6fed"),
            ("matched_control_top1", "x", "#999999"),
            ("candidate_top4", "s", "#d62728"),
            ("noncandidate4_control", "^", "#777777"),
        ]:
            part = subset[subset.intervention == intervention]
            axis.scatter(
                part.mean_centroid_margin_drop,
                part.mean_output_margin_drop,
                marker=marker,
                color=color,
                s=70,
                label=intervention.replace("candidate_", ""),
            )
            for row in part.itertuples(index=False):
                axis.annotate(row.count_bin, (row.mean_centroid_margin_drop, row.mean_output_margin_drop), fontsize=8, xytext=(4, 3), textcoords="offset points")
        axis.axhline(0, color="#aaa", linewidth=1)
        axis.axvline(0, color="#aaa", linewidth=1)
        axis.set_title(MECHANISM_LABELS[mechanism])
        axis.set_xlabel("hidden centroid-margin drop")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("marker/final logit-margin drop")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Does head-induced geometry damage track output damage?", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_bidirectional_analysis(
    run_dir: str | Path,
    *,
    centroid_examples_per_count: int = 4,
    eval_examples_per_count: int = 2,
    state_to_head_examples_per_bin: int = 3,
    device: str | None = None,
) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    cfg, vocab = load_run(run_dir, device=device)
    rankings = load_rankings(run_dir)
    models = {
        mode: load_final_model(cfg, vocab, run_dir, mode).to(cfg.device).eval()
        for mode in cfg.modes
    }
    output_dir = run_dir / "analysis" / "head_state_bidirectional"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    head_detail = run_head_to_state(
        models, cfg, vocab, rankings,
        centroid_examples_per_count=centroid_examples_per_count,
        eval_examples_per_count=eval_examples_per_count,
        seed=cfg.seed + 91_000,
    )
    state_detail = run_state_to_head(
        models, cfg, vocab, rankings,
        examples_per_bin=state_to_head_examples_per_bin,
        seed=cfg.seed + 92_000,
    )
    head_summary = summarize_head_to_state(head_detail)
    state_summary = summarize_state_to_head(state_detail)

    outputs = {
        "head_to_state_detail": head_detail,
        "head_to_state_summary": head_summary,
        "state_to_head_detail": state_detail,
        "state_to_head_summary": state_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(table_dir / f"{name}.csv", index=False)

    _plot_head_to_state(head_summary, figure_dir / "head_to_state_geometry.png")
    _plot_head_to_output(head_summary, figure_dir / "head_to_state_output.png")
    _plot_state_to_head(state_summary, figure_dir / "state_to_head_routing.png")
    _plot_mediation(head_summary, figure_dir / "head_state_mediation.png")

    manifest = {
        "analysis": "bidirectional causal link between attention heads and hidden states",
        "centroid_examples_per_exact_count": int(centroid_examples_per_count),
        "head_to_state_eval_examples_per_exact_count": int(eval_examples_per_count),
        "state_to_head_pairs_per_count_bin": int(state_to_head_examples_per_bin),
        "head_to_state_interventions": [
            "clean", "candidate_top1", "matched_control_top1",
            "candidate_top4", "noncandidate4_control",
        ],
        "state_to_head_interventions": ["clean", "same_state_control", "shifted_state_patch"],
        "state_to_head_scopes": {
            "candidate_top4": "mean over the four pre-registered mechanism candidates",
            "all_downstream": "mean over every head at or after the patched layer",
        },
        "geometry_targets": {
            "nonthinking_broad": "exact scalar count",
            "cot_targeted": "marker identity at the final k-to-k trace query",
            "cot_readout": "exact scalar count",
        },
        "candidate_heads": {
            "nonthinking_broad": [_head_label(head) for head in rankings["direct_broad"][:4]],
            "cot_targeted": [_head_label(head) for head in rankings["targeted_retrieval"][:4]],
            "cot_readout": [_head_label(head) for head in rankings["trace_readout"][:4]],
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return outputs
