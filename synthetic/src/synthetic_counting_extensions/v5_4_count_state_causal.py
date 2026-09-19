from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from synthetic_niah_v5.data import BaseExample, balanced_examples, render_nonthinking, render_thinking
from synthetic_niah_v5.vocab import MARKER_TOKENS, Vocab, index_token

from .v5_2_switch_diagnostics import load_v5_state, resolve_v5_run_dir
from .v5_3_mechanism_causal import (
    Head,
    _capture_cproj_inputs,
    _margin,
    _normalized_recovery,
    _patched_forward,
    _pred_from_subset,
    delete_last_needle,
)


SITE_NONTHINKING = "nonthinking_close"
SITE_THINKING_FIXED = "thinking_fixed_trace_close"
SITE_THINKING_NATURAL = "thinking_natural_close"
PRIMARY_SITES = (SITE_NONTHINKING, SITE_THINKING_FIXED)


@dataclass(frozen=True)
class SiteBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    query_positions: torch.Tensor
    counts: np.ndarray


@dataclass(frozen=True)
class Direction:
    vector: np.ndarray
    step_size: float
    method: str
    source_site: str
    layer: int


def render_fixed_trace_prefix(
    example: BaseExample,
    vocab: Vocab,
    trace_count: int,
    *,
    marker_template: Iterable[str] | None = None,
) -> tuple[list[int], int]:
    """Render a position-matched post-retrieval query with a fixed trace canvas.

    The prompt count varies, but every example has exactly ``trace_count`` index/marker
    pairs. A constant marker template prevents the trace content from leaking the prompt
    count. This sequence is deliberately counterfactual and is used only as a causal
    control, not as an in-distribution accuracy benchmark.
    """

    if not 1 <= int(trace_count) <= 10:
        raise ValueError("trace_count must be in 1..10")
    markers = list(marker_template or MARKER_TOKENS[: int(trace_count)])
    if not markers:
        raise ValueError("marker_template must contain at least one marker")
    trace: list[str] = []
    for k in range(1, int(trace_count) + 1):
        trace.extend([index_token(k), markers[(k - 1) % len(markers)]])
    tokens = ["<BOS>", "<THINK_ON>", *example.seq_tokens, "<Think/>", *trace, "</Think>"]
    return vocab.encode(tokens), len(tokens) - 1


def render_site(
    example: BaseExample,
    vocab: Vocab,
    site: str,
    *,
    fixed_trace_count: int,
) -> tuple[list[int], int]:
    if site == SITE_NONTHINKING:
        rendered = render_nonthinking(example, vocab)
        return rendered.input_ids, rendered.spans.think_close_pos
    if site == SITE_THINKING_NATURAL:
        rendered = render_thinking(example, vocab, trace_indices=True)
        return rendered.input_ids, rendered.spans.think_close_pos
    if site == SITE_THINKING_FIXED:
        return render_fixed_trace_prefix(example, vocab, fixed_trace_count)
    raise ValueError(f"Unknown count-state site: {site}")


def make_site_batch(
    examples: list[BaseExample],
    vocab: Vocab,
    site: str,
    device: str | torch.device,
    *,
    fixed_trace_count: int,
) -> SiteBatch:
    rendered = [render_site(ex, vocab, site, fixed_trace_count=fixed_trace_count) for ex in examples]
    max_len = max(len(ids) for ids, _ in rendered)
    input_ids = torch.full((len(examples), max_len), vocab.pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long, device=device)
    positions = torch.empty(len(examples), dtype=torch.long, device=device)
    for row, (ids, pos) in enumerate(rendered):
        length = len(ids)
        input_ids[row, :length] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[row, :length] = 1
        positions[row] = int(pos)
    return SiteBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        query_positions=positions,
        counts=np.asarray([ex.count for ex in examples], dtype=int),
    )


def _gather_positions(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, positions]


@torch.no_grad()
def capture_block_residuals(model, batch: SiteBatch) -> tuple[torch.Tensor, list[torch.Tensor]]:
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer, block in enumerate(model.transformer.h):
        def hook(_module, _args, output, layer=layer):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer] = _gather_positions(hidden, batch.query_positions).detach().float().cpu()

        handles.append(block.register_forward_hook(hook))
    try:
        output = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        logits = _gather_positions(output.logits, batch.query_positions).detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return logits, [captured[layer] for layer in range(len(model.transformer.h))]


@torch.no_grad()
def forward_with_residual_intervention(
    model,
    batch: SiteBatch,
    *,
    layer: int,
    additive: torch.Tensor | None = None,
    replacement: torch.Tensor | None = None,
) -> torch.Tensor:
    if (additive is None) == (replacement is None):
        raise ValueError("Provide exactly one of additive or replacement")

    def hook(_module, _args, output):
        is_tuple = isinstance(output, tuple)
        hidden = (output[0] if is_tuple else output).clone()
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        if additive is not None:
            update = additive.to(hidden.device)
            if update.ndim == 1:
                update = update.unsqueeze(0).expand(hidden.shape[0], -1)
            hidden[rows, batch.query_positions] += update
        else:
            value = replacement.to(hidden.device)
            if value.ndim == 1:
                value = value.unsqueeze(0).expand(hidden.shape[0], -1)
            hidden[rows, batch.query_positions] = value
        return (hidden, *output[1:]) if is_tuple else hidden

    handle = model.transformer.h[int(layer)].register_forward_hook(hook)
    try:
        output = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        return _gather_positions(output.logits, batch.query_positions).detach().float().cpu()
    finally:
        handle.remove()


def _count_predictions(logits: torch.Tensor, vocab: Vocab) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = logits[:, vocab.count_ids]
    probs = torch.softmax(subset, dim=-1).numpy()
    count_values = np.arange(1, len(vocab.count_ids) + 1, dtype=float)
    expected = probs @ count_values
    pred = probs.argmax(axis=1) + 1
    return pred.astype(int), expected, probs


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    denominator = float(((y - y.mean()) ** 2).sum())
    if denominator <= 1e-12:
        return math.nan
    return 1.0 - float(((y - pred) ** 2).sum()) / denominator


def _fit_direction(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    rng: np.random.Generator,
    ridge_alpha: float = 1.0,
    source_site: str,
    layer: int,
) -> tuple[Direction, dict[str, float]]:
    counts = sorted(int(value) for value in np.unique(y))
    means = {count: x[y == count].mean(axis=0) for count in counts}
    deltas = [means[b] - means[a] for a, b in zip(counts[:-1], counts[1:]) if b == a + 1]
    if not deltas:
        raise ValueError("Need adjacent count classes to estimate a direction")

    if method == "adjacent_mean":
        raw = np.mean(deltas, axis=0)
    elif method in {"ridge", "shuffled_ridge"}:
        target = y.astype(float).copy()
        if method == "shuffled_ridge":
            rng.shuffle(target)
        xc = x - x.mean(axis=0, keepdims=True)
        yc = target - target.mean()
        raw = np.linalg.solve(xc.T @ xc + ridge_alpha * np.eye(x.shape[1]), xc.T @ yc)
    else:
        raise ValueError(method)

    norm = float(np.linalg.norm(raw))
    if norm <= 1e-12:
        raise ValueError(f"Degenerate {method} direction")
    unit = raw / norm
    projections = x @ unit
    if np.corrcoef(projections, y.astype(float))[0, 1] < 0:
        unit = -unit
        projections = -projections
    projected_steps = np.asarray([float(delta @ unit) for delta in deltas])
    step_size = float(np.mean(projected_steps))

    delta_units = [delta / max(float(np.linalg.norm(delta)), 1e-12) for delta in deltas]
    pairwise_cos = [
        float(delta_units[i] @ delta_units[j])
        for i in range(len(delta_units))
        for j in range(i + 1, len(delta_units))
    ]
    design = np.column_stack([np.ones(len(projections)), projections])
    beta, *_ = np.linalg.lstsq(design, y.astype(float), rcond=None)
    pred = design @ beta
    summary = {
        "direction_raw_norm": norm,
        "step_size": step_size,
        "projection_r2_train": _r2(y.astype(float), pred),
        "adjacent_delta_cosine_mean": float(np.mean(pairwise_cos)) if pairwise_cos else math.nan,
        "adjacent_delta_cosine_min": float(np.min(pairwise_cos)) if pairwise_cos else math.nan,
        "adjacent_projected_step_std": float(projected_steps.std()),
    }
    return Direction(unit.astype(np.float32), step_size, method, source_site, int(layer)), summary


@torch.no_grad()
def collect_states(
    model,
    vocab: Vocab,
    examples: list[BaseExample],
    site: str,
    device: str | torch.device,
    *,
    fixed_trace_count: int,
    batch_size: int,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    layer_parts: list[list[np.ndarray]] = [[] for _ in model.transformer.h]
    baseline_rows: list[dict[str, Any]] = []
    for start in range(0, len(examples), int(batch_size)):
        chunk = examples[start : start + int(batch_size)]
        batch = make_site_batch(chunk, vocab, site, device, fixed_trace_count=fixed_trace_count)
        logits, states = capture_block_residuals(model, batch)
        pred, expected, _ = _count_predictions(logits, vocab)
        for layer, values in enumerate(states):
            layer_parts[layer].append(values.numpy())
        for offset, ex in enumerate(chunk):
            baseline_rows.append(
                {
                    "example_idx": start + offset,
                    "site": site,
                    "count": ex.count,
                    "query_position": int(batch.query_positions[offset].item()),
                    "pred_count": int(pred[offset]),
                    "expected_count": float(expected[offset]),
                    "accuracy": float(pred[offset] == ex.count),
                }
            )
    return [np.concatenate(parts, axis=0) for parts in layer_parts], pd.DataFrame(baseline_rows)


def estimate_directions(
    train_states: dict[str, list[np.ndarray]],
    train_counts: np.ndarray,
    *,
    eval_states: dict[str, list[np.ndarray]] | None = None,
    eval_counts: np.ndarray | None = None,
    seed: int,
) -> tuple[
    dict[tuple[str, int, str], Direction],
    dict[tuple[str, int, int], np.ndarray],
    pd.DataFrame,
    pd.DataFrame,
]:
    rng = np.random.default_rng(seed)
    directions: dict[tuple[str, int, str], Direction] = {}
    centroids: dict[tuple[str, int, int], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for site, layers in train_states.items():
        for layer, x in enumerate(layers):
            for count in sorted(int(value) for value in np.unique(train_counts)):
                centroids[(site, layer, count)] = x[train_counts == count].mean(axis=0).astype(np.float32)
            for method in ("adjacent_mean", "ridge", "shuffled_ridge"):
                direction, metrics = _fit_direction(
                    x,
                    train_counts,
                    method=method,
                    rng=rng,
                    source_site=site,
                    layer=layer,
                )
                directions[(site, layer, method)] = direction
                if eval_states is not None and eval_counts is not None:
                    train_projection = x @ direction.vector
                    design = np.column_stack([np.ones(len(train_projection)), train_projection])
                    beta, *_ = np.linalg.lstsq(design, train_counts.astype(float), rcond=None)
                    eval_projection = eval_states[site][layer] @ direction.vector
                    eval_prediction = np.column_stack([np.ones(len(eval_projection)), eval_projection]) @ beta
                    metrics["projection_r2_heldout"] = _r2(eval_counts.astype(float), eval_prediction)
                    metrics["projection_mae_heldout"] = float(np.mean(np.abs(eval_prediction - eval_counts)))
                rows.append({"site": site, "layer": layer, "method": method, **metrics})

    cross_rows: list[dict[str, Any]] = []
    for layer in range(len(next(iter(train_states.values())))):
        for method in ("adjacent_mean", "ridge"):
            non = directions[(SITE_NONTHINKING, layer, method)].vector
            think = directions[(SITE_THINKING_FIXED, layer, method)].vector
            cross_rows.append(
                {
                    "layer": layer,
                    "method": method,
                    "site_a": SITE_NONTHINKING,
                    "site_b": SITE_THINKING_FIXED,
                    "cosine": float(non @ think),
                }
            )
    return directions, centroids, pd.DataFrame(rows), pd.DataFrame(cross_rows)


def _random_orthogonal(direction: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    value = rng.normal(size=direction.shape)
    value -= float(value @ direction) * direction
    return (value / max(float(np.linalg.norm(value)), 1e-12)).astype(np.float32)


def make_count_manifold_plots(
    centroids: dict[tuple[str, int, int], np.ndarray],
    out_dir: Path,
) -> pd.DataFrame:
    """Plot count-centroid trajectories and quantify low-dimensional fidelity."""
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    projections: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    sites = (SITE_NONTHINKING, SITE_THINKING_FIXED)
    n_layers = max(layer for _, layer, _ in centroids) + 1

    for site in sites:
        for layer in range(n_layers):
            values = np.stack([centroids[(site, layer, count)] for count in range(1, 11)])
            centered = values - values.mean(axis=0, keepdims=True)
            _, singular, components = np.linalg.svd(centered, full_matrices=False)
            variance = singular**2
            raw_variance_ratio = variance / max(float(variance.sum()), 1e-12)
            variance_ratio = np.pad(raw_variance_ratio, (0, max(0, 6 - len(raw_variance_ratio))))
            raw_coordinates = centered @ components[:6].T
            coordinates = np.pad(raw_coordinates, ((0, 0), (0, max(0, 6 - raw_coordinates.shape[1]))))
            projections[(site, layer)] = (coordinates, variance_ratio)
            deltas = np.diff(values, axis=0)
            consecutive_cosines = np.sum(deltas[:-1] * deltas[1:], axis=1) / np.maximum(
                np.linalg.norm(deltas[:-1], axis=1) * np.linalg.norm(deltas[1:], axis=1),
                1e-12,
            )
            turning_angles = np.degrees(np.arccos(np.clip(consecutive_cosines, -1.0, 1.0)))
            cumulative = np.cumsum(variance_ratio)
            rows.append(
                {
                    "site": site,
                    "layer": layer,
                    "pc1_variance": float(variance_ratio[0]),
                    "pc1_pc2_variance": float(cumulative[min(1, len(cumulative) - 1)]),
                    "pc1_pc2_pc3_variance": float(cumulative[min(2, len(cumulative) - 1)]),
                    "pc1_to_pc6_variance": float(cumulative[min(5, len(cumulative) - 1)]),
                    "pcs_for_90pct": int(np.searchsorted(cumulative, 0.90) + 1),
                    "effective_dimension": float(variance.sum() ** 2 / max(float(np.sum(variance**2)), 1e-12)),
                    "mean_turning_angle_degrees": float(turning_angles.mean()),
                    "path_to_chord_ratio": float(
                        np.linalg.norm(deltas, axis=1).sum()
                        / max(float(np.linalg.norm(values[-1] - values[0])), 1e-12)
                    ),
                }
            )

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(len(sites), n_layers, figsize=(4.2 * n_layers, 7.6), constrained_layout=True)
    for row_idx, site in enumerate(sites):
        for layer in range(n_layers):
            ax = axes[row_idx, layer]
            coordinates, variance_ratio = projections[(site, layer)]
            ax.plot(coordinates[:, 0], coordinates[:, 1], color="C0", alpha=0.45, linewidth=1.5)
            for idx, count in enumerate(range(1, 11)):
                ax.scatter(coordinates[idx, 0], coordinates[idx, 1], color="C0", s=34, zorder=3)
                ax.annotate(str(count), coordinates[idx, :2], xytext=(5, 4), textcoords="offset points", fontsize=9)
                if idx < 9:
                    delta = coordinates[idx + 1, :2] - coordinates[idx, :2]
                    ax.arrow(
                        coordinates[idx, 0],
                        coordinates[idx, 1],
                        delta[0],
                        delta[1],
                        width=0.0,
                        head_width=max(float(np.linalg.norm(delta)) * 0.08, 0.015),
                        length_includes_head=True,
                        color="C1",
                        alpha=0.75,
                    )
            ax.axhline(0, color="0.75", linewidth=0.8)
            ax.axvline(0, color="0.75", linewidth=0.8)
            ax.set_title(f"{site}, Layer {layer + 1}\n2D variance={variance_ratio[:2].sum():.1%}")
            ax.set_xlabel("centroid PCA 1")
            ax.set_ylabel("centroid PCA 2")
    fig.suptitle("Count-centroid trajectories; arrows are adjacent count differences", fontsize=15)
    fig.savefig(figures / "count_centroid_manifold_2d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    selected_layers = sorted(set([0, n_layers - 1]))
    fig = plt.figure(figsize=(7.2 * len(selected_layers), 6.2 * len(sites)), constrained_layout=True)
    plot_idx = 1
    for site in sites:
        for layer in selected_layers:
            ax = fig.add_subplot(len(sites), len(selected_layers), plot_idx, projection="3d")
            plot_idx += 1
            coordinates, variance_ratio = projections[(site, layer)]
            ax.plot(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], color="C0", alpha=0.55)
            for idx, count in enumerate(range(1, 11)):
                ax.scatter(*coordinates[idx, :3], color="C0", s=35)
                ax.text(*coordinates[idx, :3], str(count), fontsize=9)
                if idx < 9:
                    delta = coordinates[idx + 1, :3] - coordinates[idx, :3]
                    ax.quiver(*coordinates[idx, :3], *delta, color="C1", arrow_length_ratio=0.15, alpha=0.8)
            ax.set_title(f"{site}, Layer {layer + 1}; 3D variance={variance_ratio[:3].sum():.1%}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
    fig.suptitle("Early versus late count-centroid geometry in 3D", fontsize=15)
    fig.savefig(figures / "count_centroid_manifold_3d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    final_layer = n_layers - 1
    fig = plt.figure(figsize=(14.5, 11.5), constrained_layout=True)
    plot_idx = 1
    for site in sites:
        coordinates, variance_ratio = projections[(site, final_layer)]
        for start_pc in (0, 3):
            ax = fig.add_subplot(len(sites), 2, plot_idx, projection="3d")
            plot_idx += 1
            subspace = coordinates[:, start_pc : start_pc + 3]
            ax.plot(subspace[:, 0], subspace[:, 1], subspace[:, 2], color="C0", alpha=0.55)
            for idx, count in enumerate(range(1, 11)):
                ax.scatter(*subspace[idx], color="C0", s=35)
                ax.text(*subspace[idx], str(count), fontsize=9)
                if idx < 9:
                    delta = subspace[idx + 1] - subspace[idx]
                    ax.quiver(*subspace[idx], *delta, color="C1", arrow_length_ratio=0.15, alpha=0.8)
            retained = variance_ratio[start_pc : start_pc + 3].sum()
            ax.set_title(
                f"{site}, Layer {final_layer + 1}; "
                f"PC{start_pc + 1}-{start_pc + 3} variance={retained:.1%}"
            )
            ax.set_xlabel(f"PC{start_pc + 1}")
            ax.set_ylabel(f"PC{start_pc + 2}")
            ax.set_zlabel(f"PC{start_pc + 3}")
    six_variance = [projections[(site, final_layer)][1][:6].sum() for site in sites]
    fig.suptitle(
        "Layer-4 count geometry across two 3D subspaces; "
        f"PC1-6 retain {six_variance[0]:.1%} / {six_variance[1]:.1%}",
        fontsize=15,
    )
    fig.savefig(figures / "count_centroid_six_pc_3d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


@torch.no_grad()
def run_direction_steering(
    model,
    vocab: Vocab,
    examples: list[BaseExample],
    directions: dict[tuple[str, int, str], Direction],
    centroids: dict[tuple[str, int, int], np.ndarray],
    device: str | torch.device,
    *,
    fixed_trace_count: int,
    batch_size: int,
    alphas: Iterable[float],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    eval_examples = [ex for ex in examples if 2 <= ex.count <= 9]
    target_sites = (SITE_NONTHINKING, SITE_THINKING_FIXED, SITE_THINKING_NATURAL)
    for target_site in target_sites:
        for layer in range(int(model.config.n_layer)):
            source_specs: list[tuple[str, str, Direction | None]] = []
            own_site = SITE_THINKING_FIXED if target_site == SITE_THINKING_NATURAL else target_site
            for method in ("adjacent_mean", "ridge", "shuffled_ridge"):
                direction = directions[(own_site, layer, method)]
                source_specs.append((own_site, method, direction))
            if target_site != SITE_NONTHINKING:
                for method in ("adjacent_mean", "ridge"):
                    source_specs.append((SITE_NONTHINKING, f"cross_{method}", directions[(SITE_NONTHINKING, layer, method)]))

            own = directions[(own_site, layer, "adjacent_mean")]
            random_vector = _random_orthogonal(own.vector, seed + layer * 31 + len(target_site))
            source_specs.append((own_site, "random_orthogonal", Direction(random_vector, own.step_size, "random_orthogonal", own_site, layer)))
            source_specs.append((own_site, "centroid_transport", None))
            if target_site != SITE_NONTHINKING:
                source_specs.append((SITE_NONTHINKING, "cross_centroid_transport", None))

            for source_site, label, direction in source_specs:
                for alpha in alphas:
                    for start in range(0, len(eval_examples), int(batch_size)):
                        chunk = eval_examples[start : start + int(batch_size)]
                        batch = make_site_batch(chunk, vocab, target_site, device, fixed_trace_count=fixed_trace_count)
                        if direction is None:
                            rounded_alpha = int(round(float(alpha)))
                            update_np = np.stack(
                                [
                                    centroids[(source_site, layer, int(np.clip(ex.count + rounded_alpha, 1, 10)))]
                                    - centroids[(source_site, layer, ex.count)]
                                    for ex in chunk
                                ],
                                axis=0,
                            )
                            update = torch.tensor(update_np, dtype=torch.float32, device=device)
                        else:
                            update = torch.tensor(
                                float(alpha) * float(direction.step_size) * direction.vector,
                                dtype=torch.float32,
                                device=device,
                            )
                        logits = forward_with_residual_intervention(model, batch, layer=layer, additive=update)
                        pred, expected, _ = _count_predictions(logits, vocab)
                        for offset, ex in enumerate(chunk):
                            desired = int(np.clip(ex.count + int(round(float(alpha))), 1, 10))
                            rows.append(
                                {
                                    "example_idx": start + offset,
                                    "target_site": target_site,
                                    "source_site": source_site,
                                    "direction_method": label,
                                    "layer": layer,
                                    "alpha": float(alpha),
                                    "count": ex.count,
                                    "desired_count": desired,
                                    "pred_count": int(pred[offset]),
                                    "expected_count": float(expected[offset]),
                                    "gold_accuracy": float(pred[offset] == ex.count),
                                    "desired_accuracy": float(pred[offset] == desired),
                                    "pred_shift": float(pred[offset] - ex.count),
                                    "expected_shift": float(expected[offset] - ex.count),
                                }
                            )
    detail = pd.DataFrame(rows)
    baseline = detail[detail.alpha.eq(0)][
        ["example_idx", "target_site", "source_site", "direction_method", "layer", "pred_count", "expected_count"]
    ].rename(columns={"pred_count": "baseline_pred_count", "expected_count": "baseline_expected_count"})
    detail = detail.merge(
        baseline,
        on=["example_idx", "target_site", "source_site", "direction_method", "layer"],
        how="left",
        validate="many_to_one",
    )
    detail["causal_pred_shift"] = detail.pred_count - detail.baseline_pred_count
    detail["causal_expected_shift"] = detail.expected_count - detail.baseline_expected_count
    summary = (
        detail.groupby(["target_site", "source_site", "direction_method", "layer", "alpha"], as_index=False)
        .mean(numeric_only=True)
        .drop(columns=["example_idx"], errors="ignore")
    )
    return detail, summary


def _paired_examples(examples: list[BaseExample]) -> tuple[list[BaseExample], list[BaseExample]]:
    by_count = {count: [ex for ex in examples if ex.count == count] for count in range(1, 11)}
    receivers: list[BaseExample] = []
    donors: list[BaseExample] = []
    for count in range(1, 10):
        size = min(len(by_count[count]), len(by_count[count + 1]))
        receivers.extend(by_count[count][:size])
        donors.extend(by_count[count + 1][:size])
    return receivers, donors


@torch.no_grad()
def run_state_swaps(
    model,
    vocab: Vocab,
    examples: list[BaseExample],
    device: str | torch.device,
    *,
    fixed_trace_count: int,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    receivers, donors = _paired_examples(examples)
    rows: list[dict[str, Any]] = []
    for site in PRIMARY_SITES:
        for start in range(0, len(receivers), int(batch_size)):
            receiver_chunk = receivers[start : start + int(batch_size)]
            donor_chunk = donors[start : start + int(batch_size)]
            receiver_batch = make_site_batch(receiver_chunk, vocab, site, device, fixed_trace_count=fixed_trace_count)
            donor_batch = make_site_batch(donor_chunk, vocab, site, device, fixed_trace_count=fixed_trace_count)
            baseline_logits, _ = capture_block_residuals(model, receiver_batch)
            baseline_pred, baseline_expected, _ = _count_predictions(baseline_logits, vocab)
            _, donor_states = capture_block_residuals(model, donor_batch)
            for layer, states in enumerate(donor_states):
                logits = forward_with_residual_intervention(
                    model,
                    receiver_batch,
                    layer=layer,
                    replacement=states.to(device),
                )
                pred, expected, _ = _count_predictions(logits, vocab)
                for offset, (receiver, donor) in enumerate(zip(receiver_chunk, donor_chunk)):
                    rows.append(
                        {
                            "site": site,
                            "control": "plus_one_donor",
                            "layer": layer,
                            "receiver_count": receiver.count,
                            "donor_count": donor.count,
                            "pred_count": int(pred[offset]),
                            "expected_count": float(expected[offset]),
                            "baseline_pred_count": int(baseline_pred[offset]),
                            "baseline_expected_count": float(baseline_expected[offset]),
                            "follows_donor": float(pred[offset] == donor.count),
                            "follows_receiver": float(pred[offset] == receiver.count),
                            "expected_shift": float(expected[offset] - receiver.count),
                            "causal_expected_shift": float(expected[offset] - baseline_expected[offset]),
                        }
                    )

        # Same-count, different-prompt donor control.
        controls: list[BaseExample] = []
        control_donors: list[BaseExample] = []
        for count in range(1, 11):
            same_count = [ex for ex in examples if ex.count == count]
            if len(same_count) < 2:
                continue
            controls.extend(same_count)
            control_donors.extend(same_count[1:] + same_count[:1])
        for start in range(0, len(controls), int(batch_size)):
            receiver_chunk = controls[start : start + int(batch_size)]
            donor_chunk = control_donors[start : start + int(batch_size)]
            receiver_batch = make_site_batch(receiver_chunk, vocab, site, device, fixed_trace_count=fixed_trace_count)
            donor_batch = make_site_batch(donor_chunk, vocab, site, device, fixed_trace_count=fixed_trace_count)
            baseline_logits, _ = capture_block_residuals(model, receiver_batch)
            baseline_pred, baseline_expected, _ = _count_predictions(baseline_logits, vocab)
            _, donor_states = capture_block_residuals(model, donor_batch)
            for layer, states in enumerate(donor_states):
                logits = forward_with_residual_intervention(model, receiver_batch, layer=layer, replacement=states.to(device))
                pred, expected, _ = _count_predictions(logits, vocab)
                for offset, receiver in enumerate(receiver_chunk):
                    donor = donor_chunk[offset]
                    if donor.count != receiver.count:
                        raise AssertionError("Same-count control pairing failed")
                    rows.append(
                        {
                            "site": site,
                            "control": "same_count_different_prompt",
                            "layer": layer,
                            "receiver_count": receiver.count,
                            "donor_count": donor.count,
                            "pred_count": int(pred[offset]),
                            "expected_count": float(expected[offset]),
                            "baseline_pred_count": int(baseline_pred[offset]),
                            "baseline_expected_count": float(baseline_expected[offset]),
                            "follows_donor": float(pred[offset] == donor.count),
                            "follows_receiver": float(pred[offset] == receiver.count),
                            "expected_shift": float(expected[offset] - receiver.count),
                            "causal_expected_shift": float(expected[offset] - baseline_expected[offset]),
                        }
                    )
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["site", "control", "layer"], as_index=False).mean(numeric_only=True)
    return detail, summary


def _load_head_groups(run_dir: Path) -> dict[str, list[Head]]:
    path = run_dir / "v5_3_mechanism_causal" / "head_groups.json"
    if not path.is_file():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    keep = ("direct_broad_top2", "direct_broad_top4", "trace_readout_top4", "targeted_top4")
    return {name: [tuple(int(v) for v in pair) for pair in obj[name]] for name in keep if name in obj}


@torch.no_grad()
def run_count_mediation(
    model,
    vocab: Vocab,
    examples: list[BaseExample],
    directions: dict[tuple[str, int, str], Direction],
    head_groups: dict[str, list[Head]],
    device: str | torch.device,
    *,
    fixed_trace_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    layers = set(range(int(model.config.n_layer)))
    for example_idx, clean_ex in enumerate(examples):
        if clean_ex.count < 2:
            continue
        corrupt_ex = delete_last_needle(clean_ex)
        for site in PRIMARY_SITES:
            clean_ids_list, clean_pos = render_site(clean_ex, vocab, site, fixed_trace_count=fixed_trace_count)
            corrupt_ids_list, corrupt_pos = render_site(corrupt_ex, vocab, site, fixed_trace_count=fixed_trace_count)
            if clean_pos != corrupt_pos or len(clean_ids_list) != len(corrupt_ids_list):
                raise AssertionError(f"Count mediation requires position matching at {site}")
            clean_ids = torch.tensor([clean_ids_list], dtype=torch.long, device=device)
            corrupt_ids = torch.tensor([corrupt_ids_list], dtype=torch.long, device=device)
            clean_mask = torch.ones_like(clean_ids)
            corrupt_mask = torch.ones_like(corrupt_ids)
            clean_batch = SiteBatch(clean_ids, clean_mask, torch.tensor([clean_pos], device=device), np.asarray([clean_ex.count]))
            corrupt_batch = SiteBatch(corrupt_ids, corrupt_mask, torch.tensor([corrupt_pos], device=device), np.asarray([corrupt_ex.count]))
            clean_logits, clean_states = capture_block_residuals(model, clean_batch)
            corrupt_logits, corrupt_states = capture_block_residuals(model, corrupt_batch)
            target_ids = [vocab.count_id(clean_ex.count), vocab.count_id(corrupt_ex.count)]
            clean_margin = _margin(clean_logits[0], vocab.count_id(clean_ex.count), target_ids)
            corrupt_margin = _margin(corrupt_logits[0], vocab.count_id(clean_ex.count), target_ids)

            for layer, (clean_state, corrupt_state) in enumerate(zip(clean_states, corrupt_states)):
                direction = directions[(site, layer, "adjacent_mean")]
                state_step_units = float(((clean_state[0] - corrupt_state[0]).numpy() @ direction.vector) / max(abs(direction.step_size), 1e-12))
                patched_logits = forward_with_residual_intervention(
                    model,
                    corrupt_batch,
                    layer=layer,
                    replacement=clean_state.to(device),
                )
                patched_margin = _margin(patched_logits[0], vocab.count_id(clean_ex.count), target_ids)
                rows.append(
                    {
                        "example_idx": example_idx,
                        "site": site,
                        "component_type": "residual",
                        "component_name": f"resid_after_layer{layer}",
                        "layer": layer,
                        "clean_count": clean_ex.count,
                        "corrupt_count": corrupt_ex.count,
                        "state_step_units": state_step_units,
                        "clean_margin": clean_margin,
                        "corrupt_margin": corrupt_margin,
                        "patched_margin": patched_margin,
                        "normalized_recovery": _normalized_recovery(clean_margin, corrupt_margin, patched_margin),
                    }
                )

            if head_groups:
                donor_inputs = _capture_cproj_inputs(model, clean_ids, layers)
                for group_name, heads in head_groups.items():
                    patched = _patched_forward(model, corrupt_ids, donor_inputs, heads, clean_pos, corrupt_pos)
                    patched_margin = _margin(patched[corrupt_pos], vocab.count_id(clean_ex.count), target_ids)
                    rows.append(
                        {
                            "example_idx": example_idx,
                            "site": site,
                            "component_type": "head_group",
                            "component_name": group_name,
                            "layer": math.nan,
                            "clean_count": clean_ex.count,
                            "corrupt_count": corrupt_ex.count,
                            "state_step_units": math.nan,
                            "clean_margin": clean_margin,
                            "corrupt_margin": corrupt_margin,
                            "patched_margin": patched_margin,
                            "normalized_recovery": _normalized_recovery(clean_margin, corrupt_margin, patched_margin),
                        }
                    )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["site", "component_type", "component_name"], as_index=False)
        .mean(numeric_only=True)
        .drop(columns=["example_idx"], errors="ignore")
    )
    return detail, summary


def make_plots(outputs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    geometry = outputs["direction_geometry"]
    part = geometry[geometry.method.isin(["adjacent_mean", "ridge"])].copy()
    part["layer_display"] = part["layer"].astype(int) + 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    r2_metric = "projection_r2_heldout" if "projection_r2_heldout" in part.columns else "projection_r2_train"
    sns.lineplot(data=part, x="layer_display", y=r2_metric, hue="site", style="method", marker="o", ax=axes[0])
    axes[0].set_title("Count-direction linear readability")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("held-out projection R²")
    sns.lineplot(data=part, x="layer_display", y="adjacent_delta_cosine_mean", hue="site", style="method", marker="o", ax=axes[1])
    axes[1].set_title("Are adjacent count differences parallel?")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("mean cosine among μ(n+1)-μ(n)")
    fig.savefig(figures / "count_direction_geometry.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    steering = outputs["steering_summary"]
    selected = steering[
        steering.direction_method.isin(
            ["adjacent_mean", "cross_adjacent_mean", "centroid_transport", "cross_centroid_transport", "random_orthogonal"]
        )
    ]
    sites = (SITE_NONTHINKING, SITE_THINKING_FIXED, SITE_THINKING_NATURAL)
    shown_layers = (0, int(steering.layer.max()))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.2), constrained_layout=True, sharex=True, sharey=True)
    for row, layer in enumerate(shown_layers):
        for col, site in enumerate(sites):
            ax = axes[row, col]
            panel = selected[(selected.target_site == site) & (selected.layer == layer)]
            sns.lineplot(data=panel, x="alpha", y="causal_expected_shift", hue="direction_method", marker="o", ax=ax)
            ax.axhline(0, color="black", lw=1)
            ax.set_title(f"{site}, Layer {layer + 1}")
            ax.set_ylabel("mean expected-count shift")
            if row != 0 and ax.get_legend() is not None:
                ax.get_legend().remove()
    fig.savefig(figures / "count_direction_steering.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    swaps = outputs["state_swap_summary"].copy()
    swaps["layer_display"] = swaps["layer"].astype(int) + 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True, sharey=True)
    for ax, site in zip(axes, PRIMARY_SITES):
        sns.lineplot(data=swaps[swaps.site == site], x="layer_display", y="follows_donor", hue="control", marker="o", ax=ax)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(site)
        ax.set_xlabel("Layer")
        ax.set_ylabel("prediction follows donor count")
    fig.savefig(figures / "count_state_swap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    mediation = outputs["mediation_summary"]
    residual = mediation[mediation.component_type == "residual"].copy()
    residual["layer_display"] = residual.component_name.str.extract(r"(\d+)").astype(int) + 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    sns.lineplot(data=residual, x="layer_display", y="state_step_units", hue="site", marker="o", ax=axes[0])
    axes[0].axhline(1, color="gray", ls="--", lw=1)
    axes[0].set_title("Needle deletion projected onto count direction")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("clean−corrupt projection / one count step")
    sns.lineplot(data=residual, x="layer_display", y="normalized_recovery", hue="site", marker="o", ax=axes[1])
    axes[1].axhline(1, color="gray", ls="--", lw=1)
    axes[1].set_title("Clean residual restores the deleted-needle answer")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("normalized logit-margin recovery")
    fig.savefig(figures / "count_residual_mediation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    heads = mediation[mediation.component_type == "head_group"]
    if not heads.empty:
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        sns.barplot(data=heads, x="component_name", y="normalized_recovery", hue="site", ax=ax)
        ax.axhline(1, color="gray", ls="--", lw=1)
        ax.set_xlabel("clean head-output group patched at the count query")
        ax.set_ylabel("normalized logit-margin recovery")
        ax.tick_params(axis="x", rotation=25)
        ax.set_title("Which heads write the causal count state?")
        fig.savefig(figures / "count_head_mediation.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


@torch.no_grad()
def run_v5_4_count_state_causal(
    run_dir: str | Path,
    *,
    train_examples_per_count: int = 100,
    eval_examples_per_count: int = 20,
    mediation_examples_per_count: int = 10,
    fixed_trace_count: int = 5,
    batch_size: int = 64,
    alphas: Iterable[float] = (-2, -1, 0, 1, 2),
    seed_offset: int = 121_000,
    device: str | None = None,
) -> dict[str, pd.DataFrame]:
    run_dir = resolve_v5_run_dir(run_dir)
    cfg, vocab, model = load_v5_state(run_dir, device=device)
    train_cfg = cfg["train"]
    seq_len = int(train_cfg["seq_len"])
    seed = int(train_cfg["seed"]) + int(seed_offset)
    device = cfg["device"]
    out_dir = run_dir / "v5_4_count_state_causal"
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    print("[v5.4] collecting independent train/eval count states", flush=True)
    train_examples = balanced_examples(seq_len, train_examples_per_count, seed)
    eval_examples = balanced_examples(seq_len, eval_examples_per_count, seed + 1)
    train_states: dict[str, list[np.ndarray]] = {}
    eval_states: dict[str, list[np.ndarray]] = {}
    baseline_frames: list[pd.DataFrame] = []
    for site in PRIMARY_SITES:
        states, baseline = collect_states(
            model,
            vocab,
            train_examples,
            site,
            device,
            fixed_trace_count=fixed_trace_count,
            batch_size=batch_size,
        )
        train_states[site] = states
        baseline["split"] = "direction_train"
        baseline_frames.append(baseline)
        site_eval_states, eval_baseline = collect_states(
            model,
            vocab,
            eval_examples,
            site,
            device,
            fixed_trace_count=fixed_trace_count,
            batch_size=batch_size,
        )
        eval_baseline["split"] = "causal_eval"
        baseline_frames.append(eval_baseline)
        eval_states[site] = site_eval_states

    directions, centroids, geometry, cross = estimate_directions(
        train_states,
        np.asarray([ex.count for ex in train_examples], dtype=int),
        eval_states=eval_states,
        eval_counts=np.asarray([ex.count for ex in eval_examples], dtype=int),
        seed=seed + 2,
    )
    np.savez(
        out_dir / "directions.npz",
        **{
            f"{site}__L{layer}__{method}": direction.vector
            for (site, layer, method), direction in directions.items()
        },
    )
    np.savez(
        out_dir / "count_centroids.npz",
        **{
            f"{site}__L{layer}__C{count}": value
            for (site, layer, count), value in centroids.items()
        },
    )
    manifold_geometry = make_count_manifold_plots(centroids, out_dir)

    print("[v5.4] causal direction steering in non-thinking and thinking", flush=True)
    steering_rows, steering_summary = run_direction_steering(
        model,
        vocab,
        eval_examples,
        directions,
        centroids,
        device,
        fixed_trace_count=fixed_trace_count,
        batch_size=batch_size,
        alphas=alphas,
        seed=seed + 3,
    )

    print("[v5.4] position-matched full-state swaps", flush=True)
    swap_rows, swap_summary = run_state_swaps(
        model,
        vocab,
        eval_examples,
        device,
        fixed_trace_count=fixed_trace_count,
        batch_size=batch_size,
    )

    print("[v5.4] paired needle-delete residual/head mediation", flush=True)
    mediation_examples = balanced_examples(seq_len, mediation_examples_per_count, seed + 4, count_min=2, count_max=10)
    mediation_rows, mediation_summary = run_count_mediation(
        model,
        vocab,
        mediation_examples,
        directions,
        _load_head_groups(run_dir),
        device,
        fixed_trace_count=fixed_trace_count,
    )

    outputs = {
        "baseline_accuracy": pd.concat(baseline_frames, ignore_index=True),
        "direction_geometry": geometry,
        "manifold_geometry": manifold_geometry,
        "cross_mode_direction_cosine": cross,
        "steering_rows": steering_rows,
        "steering_summary": steering_summary,
        "state_swap_rows": swap_rows,
        "state_swap_summary": swap_summary,
        "mediation_rows": mediation_rows,
        "mediation_summary": mediation_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(tables / f"{name}.csv", index=False)
    make_plots(outputs, out_dir)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "source_run": str(run_dir),
                "train_examples_per_count": train_examples_per_count,
                "eval_examples_per_count": eval_examples_per_count,
                "mediation_examples_per_count": mediation_examples_per_count,
                "fixed_trace_count": fixed_trace_count,
                "batch_size": batch_size,
                "alphas": list(alphas),
                "seed_offset": seed_offset,
                "device": str(device),
                "position_control": (
                    "thinking_fixed_trace_close uses a constant trace template with fixed length; "
                    "prompt count varies while the close token remains at one absolute position"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[v5.4] outputs: {out_dir}", flush=True)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v5.4 causal count-state direction, swap, and mediation analysis")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--train-examples-per-count", type=int, default=100)
    parser.add_argument("--eval-examples-per-count", type=int, default=20)
    parser.add_argument("--mediation-examples-per-count", type=int, default=10)
    parser.add_argument("--fixed-trace-count", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--alphas", default="-2,-1,0,1,2")
    parser.add_argument("--seed-offset", type=int, default=121_000)
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_v5_4_count_state_causal(
        args.run_dir,
        train_examples_per_count=args.train_examples_per_count,
        eval_examples_per_count=args.eval_examples_per_count,
        mediation_examples_per_count=args.mediation_examples_per_count,
        fixed_trace_count=args.fixed_trace_count,
        batch_size=args.batch_size,
        alphas=[float(value) for value in args.alphas.split(",") if value.strip()],
        seed_offset=args.seed_offset,
        device=args.device,
    )


if __name__ == "__main__":
    main()
