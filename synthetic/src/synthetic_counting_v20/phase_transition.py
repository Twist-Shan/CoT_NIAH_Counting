"""Streaming phase-transition, attention-role, and manifold diagnostics.

The analysis deliberately avoids raw per-token exports.  Every checkpoint shard
is loaded once, reduced to sufficient statistics, and released before the next
shard is opened.  Sample-level 3-D coordinates and causal interventions are
kept only at the configured milestone steps.
"""

from __future__ import annotations

import contextlib
import html
import json
import math
from pathlib import Path
from typing import Any, Iterator, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from .config import V20Config
from .data import V20Example, V20Rendered, V20Vocab, collate_v20, render_v20
from .model import TinyPositionCausalLM, build_model
from .training import checkpoint_steps


Head = tuple[int, int]  # one-based layer, zero-based head


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _balanced_split(
    examples: Sequence[V20Example],
    count_max: int,
    per_count: int,
    *,
    offset: int = 0,
) -> list[V20Example]:
    buckets = {count: [] for count in range(1, count_max + 1)}
    for example in examples:
        count = int(example.count or 0)
        if count in buckets:
            buckets[count].append(example)
    result: list[V20Example] = []
    for count, values in buckets.items():
        selected = values[offset : offset + per_count]
        if len(selected) != per_count:
            raise ValueError(
                f"phase split needs {offset + per_count} examples for count={count}; "
                f"found {len(values)}"
            )
        result.extend(selected)
    return result


def _batches(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _forward(
    model: TinyPositionCausalLM,
    items: Sequence[V20Rendered],
    vocab: V20Vocab,
    device: str,
    *,
    attention: bool = False,
    hidden: bool = False,
):
    ids, _, mask = collate_v20(list(items), vocab, device)
    with torch.inference_mode():
        return model(
            input_ids=ids,
            attention_mask=mask,
            output_attentions=attention,
            output_hidden_states=hidden,
        )


def _targeted_and_successor_sums(
    model: TinyPositionCausalLM,
    items: Sequence[V20Rendered],
    vocab: V20Vocab,
    device: str,
    *,
    output: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-head sums and denominators for two operational head roles.

    Targeted retrieval is measured at trace-query anchors and asks how much
    attention lands on the matching k-th prompt needle.  Marker successor is
    measured at marker k as the absolute attention mass returning to the
    immediately preceding trace-query group k.  The query is a numeric index in
    v20/v21 and the repeated separator token in v22.
    """

    if output is None:
        output = _forward(model, items, vocab, device, attention=True)
    assert output.attentions is not None
    shape = (model.config.n_layer, model.config.n_head)
    targeted = np.zeros(shape, dtype=np.float64)
    successor = np.zeros(shape, dtype=np.float64)
    targeted_n = np.zeros(shape, dtype=np.int64)
    successor_n = np.zeros(shape, dtype=np.int64)
    for row, item in enumerate(items):
        assert item.spans is not None
        for occurrence, (query, prompt_key) in enumerate(
            zip(
                item.spans.trace_query_positions,
                item.prompt_needle_positions,
                strict=True,
            )
        ):
            for layer, weights in enumerate(output.attentions):
                targeted[layer] += (
                    weights[row, :, query, prompt_key].detach().float().cpu().numpy()
                )
                targeted_n[layer] += 1
            marker_query = item.spans.trace_marker_positions[occurrence]
            past_groups = item.spans.trace_query_token_groups[: occurrence + 1]
            current_group = past_groups[-1]
            for layer, weights in enumerate(output.attentions):
                current_mass = weights[row, :, marker_query, list(current_group)].sum(-1)
                successor[layer] += current_mass.detach().float().cpu().numpy()
                successor_n[layer] += 1
    return targeted, targeted_n, successor, successor_n


def _select_fixed_heads(
    cfg: V20Config,
    vocab: V20Vocab,
    model: TinyPositionCausalLM,
    examples: Sequence[V20Example],
) -> tuple[dict[str, Head], pd.DataFrame]:
    targeted = np.zeros((cfg.n_layer, cfg.n_head), dtype=np.float64)
    successor = np.zeros_like(targeted)
    targeted_n = np.zeros_like(targeted, dtype=np.int64)
    successor_n = np.zeros_like(targeted, dtype=np.int64)
    items = [render_v20(example, vocab, "thinking") for example in examples]
    batch_size = min(8, cfg.analysis_batch_size)
    for batch in _batches(items, batch_size):
        values = _targeted_and_successor_sums(model, batch, vocab, cfg.device)
        targeted += values[0]
        targeted_n += values[1]
        successor += values[2]
        successor_n += values[3]
    scores = {
        "targeted_retrieval": targeted / np.maximum(targeted_n, 1),
        "marker_successor": successor / np.maximum(successor_n, 1),
    }
    rows: list[dict[str, Any]] = []
    fixed: dict[str, Head] = {}
    for role, matrix in scores.items():
        order = np.argsort(matrix.reshape(-1))[::-1]
        for rank, flat in enumerate(order, start=1):
            layer, head = np.unravel_index(int(flat), matrix.shape)
            rows.append(
                {
                    "role": role,
                    "rank": rank,
                    "layer": int(layer + 1),
                    "head": int(head),
                    "selection_score": float(matrix[layer, head]),
                    "selection_split": "heldout_head_selection",
                    "selection_checkpoint": cfg.train_steps,
                }
            )
        best = int(order[0])
        layer, head = np.unravel_index(best, matrix.shape)
        fixed[role] = (int(layer + 1), int(head))
    return fixed, pd.DataFrame(rows)


def _fixed_head_k_rows(
    output: Any,
    items: Sequence[V20Rendered],
    fixed: dict[str, Head],
    *,
    step: int,
) -> list[dict[str, Any]]:
    assert output.attentions is not None
    rows: list[dict[str, Any]] = []
    for row, item in enumerate(items):
        assert item.spans is not None and item.count is not None
        for occurrence, (index_query, prompt_key, marker_query, group) in enumerate(
            zip(
                item.spans.trace_query_positions,
                item.prompt_needle_positions,
                item.spans.trace_marker_positions,
                item.spans.trace_query_token_groups,
                strict=True,
            ),
            start=1,
        ):
            targeted_layer, targeted_head = fixed["targeted_retrieval"]
            successor_layer, successor_head = fixed["marker_successor"]
            targeted_mass = output.attentions[targeted_layer - 1][
                row, targeted_head, index_query, prompt_key
            ]
            successor_mass = output.attentions[successor_layer - 1][
                row, successor_head, marker_query, list(group)
            ].sum()
            rows.extend(
                (
                    {
                        "step": step,
                        "role": "targeted_retrieval",
                        "count": int(item.count),
                        "k": occurrence,
                        "score": float(targeted_mass.detach().float().cpu()),
                    },
                    {
                        "step": step,
                        "role": "marker_successor",
                        "count": int(item.count),
                        "k": occurrence,
                        "score": float(successor_mass.detach().float().cpu()),
                    },
                )
            )
    return rows


def _target_positions_and_ids(
    item: V20Rendered,
    vocab: V20Vocab,
    outcome: str,
) -> tuple[list[int], list[int], list[int]]:
    assert item.spans is not None and item.count is not None
    if outcome == "targeted_retrieval":
        queries = list(item.spans.trace_query_positions)
        targets = [item.input_ids[position] for position in item.spans.trace_marker_positions]
        ks = list(range(1, int(item.count) + 1))
        return queries, targets, ks
    if outcome == "marker_successor":
        queries = list(item.spans.trace_marker_positions)
        targets = []
        for k in range(1, int(item.count) + 1):
            if k < int(item.count):
                token = (
                    "<Sep>"
                    if vocab.trace_format == "separator"
                    else vocab.number_tokens(k + 1)[0]
                )
            else:
                token = "</Think>"
            targets.append(vocab.token_to_id[token])
        return queries, targets, list(range(1, int(item.count) + 1))
    raise ValueError(outcome)


def _prediction_rows(
    output: Any,
    items: Sequence[V20Rendered],
    vocab: V20Vocab,
    *,
    step: int,
    mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row, item in enumerate(items):
        assert item.spans is not None and item.count is not None
        answer_correct = all(
            int(output.logits[row, position - 1].argmax()) == item.input_ids[position]
            for position in item.spans.count_positions
        )
        rows.append(
            {
                "step": step,
                "mode": mode,
                "outcome": "final_answer_teacher_forced_exact",
                "count": int(item.count),
                "k": math.nan,
                "correct": float(answer_correct),
            }
        )
        if mode != "thinking":
            continue
        for outcome in ("targeted_retrieval", "marker_successor"):
            queries, targets, ks = _target_positions_and_ids(item, vocab, outcome)
            for query, target, k in zip(queries, targets, ks, strict=True):
                rows.append(
                    {
                        "step": step,
                        "mode": mode,
                        "outcome": (
                            "trace_marker_teacher_forced"
                            if outcome == "targeted_retrieval"
                            else "marker_next_token_teacher_forced"
                        ),
                        "count": int(item.count),
                        "k": k,
                        "correct": float(int(output.logits[row, query].argmax()) == target),
                    }
                )
        # Semantic k->k+1 exactness spans all digit tokens in v21.  In v20 this
        # reduces to the single marker-position prediction above.
        for k in range(1, int(item.count)):
            next_group = item.spans.trace_query_token_groups[k]
            exact = all(
                int(output.logits[row, position - 1].argmax()) == item.input_ids[position]
                for position in next_group
            )
            rows.append(
                {
                    "step": step,
                    "mode": mode,
                    "outcome": "semantic_k_to_k_plus_1_teacher_forced_exact",
                    "count": int(item.count),
                    "k": k,
                    "correct": float(exact),
                }
            )
    return rows


def _geometry_rows(
    vectors: dict[tuple[str, int], list[np.ndarray]],
    labels: dict[tuple[str, int], list[int]],
    *,
    step: int,
    mode: str,
    keep_cloud: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    clouds: list[dict[str, Any]] = []
    for (site, layer), values in vectors.items():
        x = np.stack(values).astype(np.float64)
        y = np.asarray(labels[(site, layer)], dtype=int)
        unique = np.unique(y)
        centroids = np.stack([x[y == value].mean(0) for value in unique])
        centered = centroids - centroids.mean(0, keepdims=True)
        _, singular, basis = np.linalg.svd(centered, full_matrices=False)
        variance = singular**2
        ratio = variance / max(float(variance.sum()), 1e-12)
        displacement = np.diff(centroids, axis=0)
        norms = np.linalg.norm(displacement, axis=1)
        adjacent_cosine = (
            np.sum(displacement[:-1] * displacement[1:], axis=1)
            / np.maximum(norms[:-1] * norms[1:], 1e-12)
            if len(displacement) > 1
            else np.asarray([], dtype=float)
        )
        within = np.mean(
            [np.mean(np.sum((x[y == value] - centroids[index]) ** 2, axis=1))
             for index, value in enumerate(unique)]
        )
        adjacent_sq = float(np.mean(norms**2)) if len(norms) else math.nan
        arc = float(norms.sum())
        chord = float(np.linalg.norm(centroids[-1] - centroids[0]))
        metrics.append(
            {
                "step": step,
                "mode": mode,
                "site": site,
                "layer": layer,
                "classes": len(unique),
                "centroid_pc1_variance_fraction": float(ratio[0]),
                "centroid_pc1_to_pc3_variance_fraction": float(ratio[:3].sum()),
                "centroid_effective_dimension": float(
                    1.0 / max(float(np.sum(ratio**2)), 1e-12)
                ),
                "mean_adjacent_centroid_distance": float(np.mean(norms)),
                "mean_adjacent_step_cosine": (
                    float(np.mean(adjacent_cosine)) if len(adjacent_cosine) else math.nan
                ),
                "path_straightness_chord_over_arc": chord / max(arc, 1e-12),
                "within_k_scatter": float(within),
                "adjacent_between_over_within": adjacent_sq / max(float(within), 1e-12),
            }
        )
        if keep_cloud:
            coordinates = (x - centroids.mean(0, keepdims=True)) @ basis[:3].T
            for index, (label, point) in enumerate(zip(y, coordinates, strict=True)):
                clouds.append(
                    {
                        "step": step,
                        "mode": mode,
                        "site": site,
                        "layer": layer,
                        "sample": index,
                        "k": int(label),
                        "pc1": float(point[0]),
                        "pc2": float(point[1]) if point.shape[0] > 1 else 0.0,
                        "pc3": float(point[2]) if point.shape[0] > 2 else 0.0,
                    }
                )
    return metrics, clouds


@contextlib.contextmanager
def _local_head_zero(
    model: TinyPositionCausalLM,
    head: Head,
    positions: Sequence[Sequence[int]],
) -> Iterator[None]:
    layer, selected_head = head
    width = model.config.n_embd // model.config.n_head
    module = model.layers[layer - 1].attention.output

    def hook(_module, args):
        hidden = args[0].clone()
        start, end = selected_head * width, (selected_head + 1) * width
        for row, row_positions in enumerate(positions):
            for position in row_positions:
                hidden[row, int(position), start:end] = 0
        return (hidden, *args[1:])

    handle = module.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _outcome_summary(
    output: Any,
    items: Sequence[V20Rendered],
    vocab: V20Vocab,
    outcome: str,
) -> tuple[float, float]:
    correct: list[float] = []
    margins: list[float] = []
    for row, item in enumerate(items):
        queries, targets, _ = _target_positions_and_ids(item, vocab, outcome)
        for query, target in zip(queries, targets, strict=True):
            logits = output.logits[row, query].float()
            alternatives = torch.cat((logits[:target], logits[target + 1 :]))
            correct.append(float(int(logits.argmax()) == target))
            margins.append(float((logits[target] - alternatives.max()).detach().cpu()))
    return float(np.mean(correct)), float(np.mean(margins))


def _causal_rows(
    cfg: V20Config,
    vocab: V20Vocab,
    model: TinyPositionCausalLM,
    items: Sequence[V20Rendered],
    fixed: dict[str, Head],
    controls: dict[str, Head],
    *,
    step: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role, head in fixed.items():
        control = controls[role]
        positions = []
        for item in items:
            assert item.spans is not None
            positions.append(
                list(item.spans.trace_query_positions)
                if role == "targeted_retrieval"
                else list(item.spans.trace_marker_positions)
            )
        baseline = _forward(model, items, vocab, cfg.device)
        base_accuracy, base_margin = _outcome_summary(baseline, items, vocab, role)
        for intervention, selected in (("baseline", None), ("fixed_head_zero", head), ("same_layer_control_zero", control)):
            if selected is None:
                accuracy, margin = base_accuracy, base_margin
            else:
                with _local_head_zero(model, selected, positions):
                    changed = _forward(model, items, vocab, cfg.device)
                accuracy, margin = _outcome_summary(changed, items, vocab, role)
            rows.append(
                {
                    "step": step,
                    "role": role,
                    "intervention": intervention,
                    "layer": selected[0] if selected else math.nan,
                    "head": selected[1] if selected else math.nan,
                    "accuracy": accuracy,
                    "margin": margin,
                    "accuracy_change_from_baseline": accuracy - base_accuracy,
                    "margin_change_from_baseline": margin - base_margin,
                    "intervention_scope": "selected head slice only at role query positions",
                }
            )
    return rows


def build_training_token_exposure(cfg: V20Config, run_dir: Path) -> pd.DataFrame:
    """Expand cumulative accepted-count histograms into exposure for every k."""

    path = run_dir / "tables" / "train_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, usecols=lambda name: name in {
        "position_encoding", "mode", "step", "cumulative_sampling_json"
    })
    rows: list[dict[str, Any]] = []
    for item in frame.itertuples(index=False):
        state = json.loads(item.cumulative_sampling_json)
        thinking = str(item.mode) == "thinking"
        accepted = {
            count: int(state.get("accepted_counts", {}).get(str(count), 0))
            for count in range(1, cfg.count_max_threshold + 1)
        }
        for k in range(1, cfg.count_max_threshold + 1):
            semantic = sum(value for count, value in accepted.items() if count >= k)
            continue_exposure = sum(value for count, value in accepted.items() if count > k)
            width = len(str(k)) if cfg.count_tokenization == "digitwise" else 1
            query_width = 1 if cfg.trace_format == "separator" else width
            query_exposure = semantic * query_width if thinking else 0
            rows.append(
                {
                    "position_encoding": item.position_encoding,
                    "mode": item.mode,
                    "step": int(item.step),
                    "k": k,
                    "accepted_examples_with_total_at_least_k": semantic,
                    "trace_query_token_exposure": query_exposure,
                    "trace_index_token_exposure": (
                        query_exposure if thinking and cfg.trace_format == "indexed" else 0
                    ),
                    "trace_delimiter_token_exposure": (
                        query_exposure if thinking and cfg.trace_format == "separator" else 0
                    ),
                    "marker_token_exposure": semantic if thinking else 0,
                    "continue_target_exposure_after_marker_k": continue_exposure if thinking else 0,
                    "close_target_exposure_after_marker_k": accepted[k] if thinking else 0,
                    "final_answer_example_exposure": accepted[k],
                    "final_answer_digit_token_exposure": accepted[k] * width,
                }
            )
    return pd.DataFrame(rows)


def plot_training_token_exposure(
    frame: pd.DataFrame,
    run_dir: Path,
    *,
    trace_format: str = "indexed",
) -> None:
    if frame.empty:
        return
    sns.set_theme(style="whitegrid", context="notebook")
    modes = [mode for mode in ("thinking", "nonthinking") if mode in set(frame["mode"])]
    figure, axes = plt.subplots(len(modes), 2, figsize=(14, 4.2 * len(modes)), squeeze=False)
    for row, mode in enumerate(modes):
        subset = frame[frame["mode"] == mode]
        heat_metric = (
            "trace_query_token_exposure"
            if mode == "thinking"
            else "final_answer_digit_token_exposure"
        )
        pivot = subset.pivot_table(
            index="k", columns="step", values=heat_metric, aggfunc="last"
        )
        sns.heatmap(pivot, ax=axes[row, 0], cmap="viridis", cbar_kws={"label": "cumulative token exposure"})
        axes[row, 0].set_title(f"{mode}: {heat_metric.replace('_', ' ')}")
        axes[row, 0].set_xlabel("training step")
        axes[row, 0].set_ylabel("semantic count index k")
        final_step = int(subset["step"].max())
        final = subset[subset["step"] == final_step].sort_values("k")
        if mode == "thinking":
            query_label = "separator token" if trace_format == "separator" else "index token"
            axes[row, 1].plot(
                final["k"],
                final["trace_query_token_exposure"],
                marker="o",
                label=query_label,
            )
            axes[row, 1].plot(final["k"], final["continue_target_exposure_after_marker_k"], marker="o", label="continue target")
            axes[row, 1].plot(final["k"], final["close_target_exposure_after_marker_k"], marker="o", label="close target")
        else:
            axes[row, 1].plot(final["k"], final["final_answer_example_exposure"], marker="o", label="answer examples")
            axes[row, 1].plot(final["k"], final["final_answer_digit_token_exposure"], marker="o", label="answer digit tokens")
        axes[row, 1].set_title(f"{mode}: exposure at step {final_step}")
        axes[row, 1].set_xlabel("k")
        axes[row, 1].set_ylabel("cumulative occurrences")
        axes[row, 1].legend()
    figure.tight_layout()
    path = run_dir / "figures" / "training_token_exposure_by_k.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_phase_diagnostics(
    behavior: pd.DataFrame,
    attention: pd.DataFrame,
    geometry: pd.DataFrame,
    causality: pd.DataFrame,
    run_dir: Path,
) -> None:
    """Compact, non-overlapping figures whose axes match stored definitions."""

    sns.set_theme(style="whitegrid", context="notebook")
    outcomes = [
        "final_answer_teacher_forced_exact",
        "trace_marker_teacher_forced",
        "marker_next_token_teacher_forced",
        "semantic_k_to_k_plus_1_teacher_forced_exact",
    ]
    available = [value for value in outcomes if value in set(behavior.get("outcome", []))]
    if available:
        columns = 2
        rows = math.ceil(len(available) / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(16, 4.5 * rows), squeeze=False, constrained_layout=True)
        flat_axes = list(axes.flat)
        for axis, outcome in zip(flat_axes, available, strict=False):
            subset = behavior[behavior["outcome"] == outcome]
            if outcome == "final_answer_teacher_forced_exact" and "thinking" in set(subset["mode"]):
                subset = subset[subset["mode"] == "thinking"]
            pivot = subset.pivot_table(index="count", columns="step", values="accuracy", aggfunc="mean")
            sns.heatmap(pivot, ax=axis, vmin=0, vmax=1, cmap="mako", cbar_kws={"label": "teacher-forced accuracy"})
            axis.set_title(outcome.replace("_", " "))
            axis.set_xlabel("training step")
            axis.set_ylabel("true total count n")
        for axis in flat_axes[len(available) :]:
            axis.remove()
        _save_figure(figure, run_dir / "figures" / "dense_phase_behavior_by_count.png")

    fixed = attention[attention.get("is_fixed_role_head", 0) == 1] if not attention.empty else attention
    if not fixed.empty:
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.3), constrained_layout=True)
        for axis, role in zip(axes, ("targeted_retrieval", "marker_successor"), strict=True):
            subset = fixed[fixed["role"] == role]
            for (layer, head), line in subset.groupby(["layer", "head"]):
                axis.plot(line["step"], line["score"], marker="o", markersize=3, label=f"L{layer}H{head}")
            axis.set_title(role.replace("_", " "))
            axis.set_xlabel("training step")
            axis.set_ylabel("fixed-head role score")
            axis.set_ylim(bottom=0)
            axis.legend(loc="best")
        _save_figure(figure, run_dir / "figures" / "dense_fixed_head_emergence.png")

    marker = geometry[(geometry.get("mode") == "thinking") & (geometry.get("site") == "trace_marker")] if not geometry.empty else geometry
    if not marker.empty:
        metrics = (
            ("adjacent_between_over_within", "adjacent between/within separation"),
            ("mean_adjacent_step_cosine", "mean adjacent step cosine"),
            ("centroid_effective_dimension", "centroid effective dimension"),
        )
        figure, axes = plt.subplots(1, 3, figsize=(17, 4.3), constrained_layout=True)
        for axis, (metric, label) in zip(axes, metrics, strict=True):
            for layer, line in marker.groupby("layer"):
                line = line.sort_values("step")
                axis.plot(line["step"], line[metric], label=f"layer {layer}")
            axis.set_title(label)
            axis.set_xlabel("training step")
            axis.set_ylabel(metric)
        axes[-1].legend(loc="best")
        _save_figure(figure, run_dir / "figures" / "dense_marker_manifold_emergence.png")

    changed = causality[causality.get("intervention") != "baseline"] if not causality.empty else causality
    if not changed.empty:
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.3), constrained_layout=True)
        for axis, role in zip(axes, ("targeted_retrieval", "marker_successor"), strict=True):
            subset = changed[changed["role"] == role]
            for intervention, line in subset.groupby("intervention"):
                axis.plot(line["step"], line["margin_change_from_baseline"], marker="o", label=intervention)
            axis.axhline(0, color="black", linewidth=1)
            axis.set_title(f"local causal effect: {role.replace('_', ' ')}")
            axis.set_xlabel("training step")
            axis.set_ylabel("change in correct-token logit margin")
            axis.legend(loc="best")
        _save_figure(figure, run_dir / "figures" / "milestone_local_head_causality.png")


def write_interactive_manifold_html(cloud: pd.DataFrame, output: Path) -> None:
    """Write one selectable 3-D centroid-PCA view without raw hidden vectors."""

    if cloud.empty:
        return
    compact = cloud.copy()
    for column in ("pc1", "pc2", "pc3"):
        compact[column] = compact[column].round(6)
    records = compact.to_dict(orient="records")
    defaults = {
        "mode": "thinking" if "thinking" in set(compact["mode"]) else str(compact.iloc[-1]["mode"]),
        "site": "trace_marker" if "trace_marker" in set(compact["site"]) else str(compact.iloc[-1]["site"]),
        "layer": int(compact["layer"].max()),
        "step": int(compact["step"].max()),
    }
    payload = json.dumps(records, separators=(",", ":"), ensure_ascii=True)
    default_json = json.dumps(defaults, separators=(",", ":"))
    title = html.escape("v20/v21 hidden-state manifold: selectable centroid-PCA 3D view")
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body{{font-family:Inter,Segoe UI,sans-serif;margin:0;background:#f6f8fb;color:#17324d}}
main{{max-width:1500px;margin:auto;padding:24px}} .controls{{display:flex;gap:14px;flex-wrap:wrap;margin:12px 0}}
label{{font-weight:600}} select{{margin-left:6px;padding:5px 8px}} #plot{{height:780px;background:white;border:1px solid #dce4ec}}
.note{{line-height:1.55;background:white;padding:14px 18px;border-left:4px solid #2a6f97}}
</style></head><body><main><h1>{title}</h1>
<p class="note">Each option is fit independently at one checkpoint/site/layer. PC1–PC3 are axes from PCA on the k-centroids, and sample states are projected into that centroid-defined basis. Color and the connected centroid path encode semantic progress k; distances are comparable within a selected panel, not across independently fitted panels.</p>
<div class="controls"><label>mode<select id="mode"></select></label><label>site<select id="site"></select></label><label>layer<select id="layer"></select></label><label>step<select id="step"></select></label></div>
<div id="plot"></div></main><script>
const DATA={payload}; const DEFAULTS={default_json};
const fields=['mode','site','layer','step'];
for(const field of fields){{const select=document.getElementById(field); const values=[...new Set(DATA.map(d=>d[field]))].sort((a,b)=>typeof a==='number'?a-b:String(a).localeCompare(String(b))); for(const value of values){{const option=document.createElement('option'); option.value=value; option.textContent=value; select.appendChild(option);}} select.value=DEFAULTS[field]; select.onchange=render;}}
function render(){{const chosen={{}}; for(const field of fields) chosen[field]=document.getElementById(field).value; const rows=DATA.filter(d=>String(d.mode)===chosen.mode&&String(d.site)===chosen.site&&String(d.layer)===chosen.layer&&String(d.step)===chosen.step); const grouped=new Map(); for(const row of rows){{if(!grouped.has(row.k))grouped.set(row.k,[]); grouped.get(row.k).push(row);}} const ks=[...grouped.keys()].sort((a,b)=>a-b); const centers=ks.map(k=>{{const g=grouped.get(k); return {{k,x:g.reduce((s,d)=>s+d.pc1,0)/g.length,y:g.reduce((s,d)=>s+d.pc2,0)/g.length,z:g.reduce((s,d)=>s+d.pc3,0)/g.length}};}}); const traces=[{{type:'scatter3d',mode:'markers',name:'sample states',x:rows.map(d=>d.pc1),y:rows.map(d=>d.pc2),z:rows.map(d=>d.pc3),text:rows.map(d=>'k='+d.k),marker:{{size:3,opacity:.45,color:rows.map(d=>d.k),colorscale:'Viridis',colorbar:{{title:'k'}}}}}},{{type:'scatter3d',mode:'lines+markers+text',name:'k centroids',x:centers.map(d=>d.x),y:centers.map(d=>d.y),z:centers.map(d=>d.z),text:centers.map(d=>d.k),textposition:'top center',line:{{width:6,color:'#d1495b'}},marker:{{size:5,color:'#d1495b'}}}}]; Plotly.react('plot',traces,{{title:`${{chosen.mode}} | ${{chosen.site}} | layer ${{chosen.layer}} | step ${{chosen.step}}`,scene:{{xaxis:{{title:'centroid-PCA PC1'}},yaxis:{{title:'centroid-PCA PC2'}},zaxis:{{title:'centroid-PCA PC3'}}}},margin:{{l:0,r:0,t:55,b:0}},legend:{{x:.01,y:.99}}}},{{responsive:true}});}}
render();</script></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output)


def _phase_candidates(
    behavior_by_count: pd.DataFrame,
    behavior_by_k: pd.DataFrame,
    attention: pd.DataFrame,
    geometry: pd.DataFrame,
    causality: pd.DataFrame,
    exposure: pd.DataFrame,
) -> pd.DataFrame:
    """Candidate transition locations; these are diagnostics, not proof."""

    rows: list[dict[str, Any]] = []

    def add_behavior(frame: pd.DataFrame, axis: str) -> None:
        keys = ["mode", "outcome", axis]
        for values, line in frame.groupby(keys, dropna=False):
            line = line.sort_values("step")
            steps = line["step"].to_numpy(dtype=int)
            metric = line["accuracy"].to_numpy(dtype=float)
            if len(metric) < 2:
                continue
            slopes = np.diff(metric) / np.maximum(np.diff(steps), 1) * 100.0
            steep = int(np.argmax(slopes))
            sustained = math.nan
            for index in range(len(metric)):
                tail = metric[index : index + 3]
                if len(tail) == 3 and bool(np.all(tail >= 0.8)):
                    sustained = int(steps[index])
                    break
            mode, outcome, coordinate = values
            query_exposure_value = math.nan
            index_exposure_value = math.nan
            if axis == "k" and not math.isnan(sustained):
                match = exposure[
                    (exposure["mode"] == mode)
                    & (exposure["k"] == int(coordinate))
                    & (exposure["step"] == int(sustained))
                ]
                if not match.empty:
                    query_exposure_value = float(
                        match.iloc[-1]["trace_query_token_exposure"]
                    )
                    index_exposure_value = float(
                        match.iloc[-1]["trace_index_token_exposure"]
                    )
            rows.append(
                {
                    "evidence_family": "behavior",
                    "metric": str(outcome),
                    "mode": str(mode),
                    "axis": axis,
                    "axis_value": coordinate,
                    "layer": math.nan,
                    "candidate_step": int(steps[steep + 1]) if slopes[steep] > 0 else math.nan,
                    "change_statistic": float(slopes[steep]),
                    "criterion": "maximum positive accuracy change per 100 steps",
                    "sustained_80pct_step": sustained,
                    "trace_query_token_exposure_at_sustained_step": query_exposure_value,
                    "trace_index_token_exposure_at_sustained_step": index_exposure_value,
                    "interpretation": "behavioral transition candidate only",
                }
            )

    add_behavior(behavior_by_count, "count")
    if not behavior_by_k.empty:
        add_behavior(behavior_by_k, "k")

    fixed = attention[attention["is_fixed_role_head"] == 1]
    for role, line in fixed.groupby("role"):
        line = line.sort_values("step")
        steps = line["step"].to_numpy(dtype=int)
        values = line["score"].to_numpy(dtype=float)
        if len(values) < 2:
            continue
        slopes = np.diff(values) / np.maximum(np.diff(steps), 1) * 100.0
        index = int(np.argmax(slopes))
        rows.append(
            {
                "evidence_family": "attention_role",
                "metric": "fixed_head_role_score",
                "mode": "thinking",
                "axis": str(role),
                "axis_value": math.nan,
                "layer": int(line.iloc[0]["layer"]),
                "candidate_step": int(steps[index + 1]) if slopes[index] > 0 else math.nan,
                "change_statistic": float(slopes[index]),
                "criterion": "maximum positive role-score change per 100 steps",
                "sustained_80pct_step": math.nan,
                "trace_query_token_exposure_at_sustained_step": math.nan,
                "trace_index_token_exposure_at_sustained_step": math.nan,
                "interpretation": "descriptive head-emergence candidate",
            }
        )

    marker = geometry[(geometry["mode"] == "thinking") & (geometry["site"] == "trace_marker")]
    for metric in (
        "adjacent_between_over_within",
        "mean_adjacent_step_cosine",
        "centroid_effective_dimension",
    ):
        for layer, line in marker.groupby("layer"):
            line = line.sort_values("step")
            steps = line["step"].to_numpy(dtype=int)
            values = line[metric].to_numpy(dtype=float)
            if len(values) < 2:
                continue
            slopes = np.diff(values) / np.maximum(np.diff(steps), 1) * 100.0
            index = int(np.argmax(np.abs(slopes)))
            rows.append(
                {
                    "evidence_family": "manifold_geometry",
                    "metric": metric,
                    "mode": "thinking",
                    "axis": "trace_marker",
                    "axis_value": math.nan,
                    "layer": int(layer),
                    "candidate_step": int(steps[index + 1]),
                    "change_statistic": float(slopes[index]),
                    "criterion": "largest absolute signed geometry change per 100 steps",
                    "sustained_80pct_step": math.nan,
                    "trace_query_token_exposure_at_sustained_step": math.nan,
                    "trace_index_token_exposure_at_sustained_step": math.nan,
                    "interpretation": "largest absolute geometry reorganization",
                }
            )

    fixed_causal = causality[causality["intervention"] == "fixed_head_zero"]
    for role, line in fixed_causal.groupby("role"):
        line = line.sort_values("step")
        if line.empty:
            continue
        strongest = line.iloc[int(np.argmin(line["margin_change_from_baseline"].to_numpy()))]
        causal_change = float(strongest["margin_change_from_baseline"])
        rows.append(
            {
                "evidence_family": "local_causality",
                "metric": "correct_token_margin_change",
                "mode": "thinking",
                "axis": str(role),
                "axis_value": math.nan,
                "layer": int(strongest["layer"]),
                "candidate_step": int(strongest["step"]) if causal_change < 0 else math.nan,
                "change_statistic": causal_change,
                "criterion": "most negative local-ablation margin change among milestones",
                "sustained_80pct_step": math.nan,
                "trace_query_token_exposure_at_sustained_step": math.nan,
                "trace_index_token_exposure_at_sustained_step": math.nan,
                "interpretation": "milestone with largest negative local-ablation effect",
            }
        )
    return pd.DataFrame(rows)


def run_phase_transition_analysis(
    cfg: V20Config,
    vocab: V20Vocab,
    run_dir: str | Path,
    heldout_examples: Sequence[V20Example],
) -> Path:
    """Run dense post-hoc diagnostics without materializing token-level tables."""

    run_dir = Path(run_dir)
    output_dir = run_dir / "analysis" / "phase_transition"
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    selection = _balanced_split(
        heldout_examples,
        cfg.count_max_threshold,
        cfg.phase_head_selection_examples_per_count,
    )
    reporting = _balanced_split(
        heldout_examples,
        cfg.count_max_threshold,
        cfg.phase_examples_per_count,
        offset=cfg.phase_head_selection_examples_per_count,
    )
    reporting_items = {
        mode: [render_v20(example, vocab, mode) for example in reporting]
        for mode in cfg.modes
    }

    final_model = build_model(cfg, vocab, "rope", cfg.device).eval()
    final_index = checkpoint_steps(run_dir, "rope", "thinking")
    if not final_index:
        raise FileNotFoundError("thinking dense snapshot index is missing")
    final_shard = final_index[-1][1]
    final_payload = torch.load(final_shard, map_location="cpu", weights_only=False)
    final_model.load_state_dict(final_payload["model_state_dicts"][str(cfg.train_steps)])
    fixed, ranking = _select_fixed_heads(cfg, vocab, final_model, selection)
    controls: dict[str, Head] = {}
    for role, selected in fixed.items():
        role_rows = ranking[
            (ranking["role"] == role)
            & (ranking["layer"] == selected[0])
            & (ranking["head"] != selected[1])
        ].copy()
        selected_score = float(
            ranking[
                (ranking["role"] == role)
                & (ranking["layer"] == selected[0])
                & (ranking["head"] == selected[1])
            ].iloc[0]["selection_score"]
        )
        role_rows["score_distance"] = (role_rows["selection_score"] - selected_score).abs()
        matched = role_rows.sort_values("score_distance").iloc[0]
        controls[role] = (int(matched["layer"]), int(matched["head"]))
    _atomic_csv(ranking, table_dir / "fixed_head_rankings.csv")
    _atomic_json({role: {"layer": head[0], "head": head[1]} for role, head in fixed.items()}, output_dir / "fixed_head_roles.json")
    del final_payload, final_model

    prediction_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    fixed_k_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    cloud_rows: list[dict[str, Any]] = []
    causal_rows: list[dict[str, Any]] = []
    trace_query_site = (
        "trace_separator" if cfg.trace_format == "separator" else "trace_index"
    )
    for mode in cfg.modes:
        entries = checkpoint_steps(run_dir, "rope", mode)
        by_shard: dict[Path, list[int]] = {}
        for step, shard in entries:
            by_shard.setdefault(shard, []).append(step)
        model = build_model(cfg, vocab, "rope", cfg.device).eval()
        for shard, steps in by_shard.items():
            payload = torch.load(shard, map_location="cpu", weights_only=False)
            for step in sorted(steps):
                model.load_state_dict(payload["model_state_dicts"][str(step)])
                vectors: dict[tuple[str, int], list[np.ndarray]] = {}
                labels: dict[tuple[str, int], list[int]] = {}
                targeted = np.zeros((cfg.n_layer, cfg.n_head), dtype=np.float64)
                successor = np.zeros_like(targeted)
                targeted_n = np.zeros_like(targeted, dtype=np.int64)
                successor_n = np.zeros_like(targeted, dtype=np.int64)
                items = reporting_items[mode]
                for batch in _batches(items, min(8, cfg.analysis_batch_size)):
                    output = _forward(
                        model,
                        batch,
                        vocab,
                        cfg.device,
                        attention=mode == "thinking",
                        hidden=True,
                    )
                    prediction_rows.extend(
                        _prediction_rows(output, batch, vocab, step=step, mode=mode)
                    )
                    assert output.hidden_states is not None
                    for row, item in enumerate(batch):
                        assert item.spans is not None and item.count is not None
                        for layer, hidden in enumerate(output.hidden_states):
                            key = ("final_answer", layer)
                            vectors.setdefault(key, []).append(
                                hidden[row, item.spans.ans_pos].detach().float().cpu().numpy()
                            )
                            labels.setdefault(key, []).append(int(item.count))
                            if mode == "thinking":
                                for k, position in enumerate(item.spans.trace_query_positions, start=1):
                                    key = (trace_query_site, layer)
                                    vectors.setdefault(key, []).append(
                                        hidden[row, position].detach().float().cpu().numpy()
                                    )
                                    labels.setdefault(key, []).append(k)
                                for k, position in enumerate(item.spans.trace_marker_positions, start=1):
                                    key = ("trace_marker", layer)
                                    vectors.setdefault(key, []).append(
                                        hidden[row, position].detach().float().cpu().numpy()
                                    )
                                    labels.setdefault(key, []).append(k)
                    if mode == "thinking":
                        values = _targeted_and_successor_sums(
                            model, batch, vocab, cfg.device, output=output
                        )
                        targeted += values[0]
                        targeted_n += values[1]
                        successor += values[2]
                        successor_n += values[3]
                        fixed_k_rows.extend(
                            _fixed_head_k_rows(output, batch, fixed, step=step)
                        )
                metrics, clouds = _geometry_rows(
                    vectors,
                    labels,
                    step=step,
                    mode=mode,
                    keep_cloud=step in set(cfg.phase_cloud_steps),
                )
                geometry_rows.extend(metrics)
                cloud_rows.extend(clouds)
                if mode == "thinking":
                    for role, sums, denominators in (
                        ("targeted_retrieval", targeted, targeted_n),
                        ("marker_successor", successor, successor_n),
                    ):
                        values = sums / np.maximum(denominators, 1)
                        for layer in range(cfg.n_layer):
                            for head in range(cfg.n_head):
                                attention_rows.append(
                                    {
                                        "step": step,
                                        "role": role,
                                        "layer": layer + 1,
                                        "head": head,
                                        "score": float(values[layer, head]),
                                        "is_fixed_role_head": float(fixed[role] == (layer + 1, head)),
                                        "observations": int(denominators[layer, head]),
                                    }
                                )
                    if step in set(cfg.phase_cloud_steps):
                        causal_rows.extend(
                            _causal_rows(
                                cfg, vocab, model, items, fixed, controls, step=step
                            )
                        )
            del payload
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    predictions = pd.DataFrame(prediction_rows)
    prediction_summary = predictions.groupby(
        ["step", "mode", "outcome", "count"], as_index=False, dropna=False
    ).agg(accuracy=("correct", "mean"), observations=("correct", "size"))
    prediction_by_k = predictions[predictions["k"].notna()].groupby(
        ["step", "mode", "outcome", "k"], as_index=False
    ).agg(accuracy=("correct", "mean"), observations=("correct", "size"))
    fixed_by_k = pd.DataFrame(fixed_k_rows).groupby(
        ["step", "role", "k"], as_index=False
    ).agg(score=("score", "mean"), observations=("score", "size"))
    tables = {
        "dense_behavior_by_count.csv": prediction_summary,
        "dense_behavior_by_k.csv": prediction_by_k,
        "dense_fixed_head_dynamics.csv": pd.DataFrame(attention_rows),
        "dense_fixed_head_by_k.csv": fixed_by_k,
        "dense_manifold_geometry.csv": pd.DataFrame(geometry_rows),
        "milestone_manifold_cloud_3d.csv": pd.DataFrame(cloud_rows),
        "milestone_local_head_causality.csv": pd.DataFrame(causal_rows),
    }
    for name, frame in tables.items():
        _atomic_csv(frame, table_dir / name)
    exposure = build_training_token_exposure(cfg, run_dir)
    _atomic_csv(exposure, run_dir / "tables" / "training_token_exposure_by_k.csv")
    candidates = _phase_candidates(
        tables["dense_behavior_by_count.csv"],
        tables["dense_behavior_by_k.csv"],
        tables["dense_fixed_head_dynamics.csv"],
        tables["dense_manifold_geometry.csv"],
        tables["milestone_local_head_causality.csv"],
        exposure,
    )
    _atomic_csv(candidates, table_dir / "phase_transition_candidates.csv")
    plot_training_token_exposure(exposure, run_dir, trace_format=cfg.trace_format)
    plot_phase_diagnostics(
        tables["dense_behavior_by_count.csv"],
        tables["dense_fixed_head_dynamics.csv"],
        tables["dense_manifold_geometry.csv"],
        tables["milestone_local_head_causality.csv"],
        run_dir,
    )
    write_interactive_manifold_html(
        tables["milestone_manifold_cloud_3d.csv"],
        output_dir / "interactive_manifold_3d.html",
    )
    _atomic_json(
        {
            "version": cfg.version,
            "trace_format": cfg.trace_format,
            "checkpoint_cadence": cfg.checkpoint_every,
            "head_selection_split": "first heldout examples per count",
            "reporting_split": "subsequent disjoint heldout examples per count",
            "fixed_heads": {role: list(head) for role, head in fixed.items()},
            "same_layer_score_matched_controls": {
                role: list(head) for role, head in controls.items()
            },
            "definitions": {
                "targeted_retrieval_score": (
                    "mean attention mass from trace-query anchor k to matching prompt needle k; "
                    "the query anchor is the repeated <Sep> token in separator traces"
                    if cfg.trace_format == "separator"
                    else "mean attention mass from trace-index anchor k to matching prompt needle k"
                ),
                "marker_successor_score": (
                    "mean absolute attention mass from marker k to the immediately preceding "
                    "trace-query token group k"
                ),
                "adjacent_between_over_within": "mean squared adjacent-centroid distance divided by mean within-k squared scatter",
                "path_straightness": "distance from centroid k=1 to k=max divided by sum of adjacent-centroid distances",
                "token_exposure_k": (
                    "cumulative accepted training examples whose total count n is at least k, "
                    "multiplied by the trace-query width (one for separator traces)"
                ),
                "transition_candidate": "maximum adjacent-checkpoint slope; behavioral 80% onset additionally requires three consecutive snapshots at or above 0.8",
            },
            "storage_policy": "aggregate every snapshot; sample clouds and local causal interventions only at phase_cloud_steps; no raw attention tensor is written",
        },
        output_dir / "manifest.json",
    )
    return output_dir


__all__ = [
    "build_training_token_exposure",
    "plot_training_token_exposure",
    "plot_phase_diagnostics",
    "write_interactive_manifold_html",
    "run_phase_transition_analysis",
]
