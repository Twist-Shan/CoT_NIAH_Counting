from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from .config import V10Config
from .core import Example, Rendered, Spans, Vocab, balanced_examples, count_prediction, render
from .training import load_final_model


SITE_NONTHINKING_FINAL = "nonthinking_final_answer"
SITE_THINKING_FINAL = "thinking_final_answer"
SITE_THINKING_FIXED = "thinking_fixed_trace_answer"
SITE_THINKING_INDEX = "thinking_trace_index"
SITE_THINKING_MARKER = "thinking_trace_marker"


@dataclass(frozen=True)
class Direction:
    vector: np.ndarray
    step_size: float
    method: str
    site: str
    layer: int


def _r2(y: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(((y - y.mean()) ** 2).sum())
    if denominator <= 1e-12:
        return math.nan
    return 1.0 - float(((y - prediction) ** 2).sum()) / denominator


def _fit_pca(values: np.ndarray, n_components: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = values - values.mean(axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[: min(n_components, len(vh))]
    coordinates = centered @ components.T
    variance = singular**2
    ratio = variance / max(float(variance.sum()), 1e-12)
    return components, coordinates, ratio[: len(components)]


def _site_positions(item: Rendered, site: str) -> list[tuple[int, int]]:
    if site in {SITE_NONTHINKING_FINAL, SITE_THINKING_FINAL, SITE_THINKING_FIXED}:
        return [(item.spans.ans_pos, item.count)]
    if site == SITE_THINKING_INDEX:
        return [(position, k) for k, position in enumerate(item.spans.trace_index_positions, start=1)]
    if site == SITE_THINKING_MARKER:
        return [(position, k) for k, position in enumerate(item.spans.trace_marker_positions, start=1)]
    raise ValueError(site)


def _site_mode(site: str) -> str:
    return "nonthinking" if site == SITE_NONTHINKING_FINAL else "thinking"


def render_fixed_trace(example: Example, vocab: Vocab, trace_count: int) -> Rendered:
    trace: list[str] = []
    marker_template = vocab.markers
    for k in range(1, int(trace_count) + 1):
        trace.extend([vocab.number_token(k), marker_template[(k - 1) % len(marker_template)]])
    prompt_start = 1
    prompt_end = prompt_start + len(example.seq_tokens)
    think_pos = prompt_end
    trace_start = think_pos + 1
    trace_positions = list(range(trace_start, trace_start + len(trace)))
    close_pos = trace_start + len(trace)
    ans_pos = close_pos + 1
    count_pos = ans_pos + 1
    eos_pos = count_pos + 1
    tokens = [
        "<BOS>",
        *example.seq_tokens,
        "<Think>",
        *trace,
        "</Think>",
        "<Ans>",
        vocab.number_token(example.count),
        "<EOS>",
    ]
    spans = Spans(
        0,
        prompt_start,
        prompt_end,
        think_pos,
        trace_positions[0::2],
        trace_positions[1::2],
        close_pos,
        ans_pos,
        count_pos,
        eos_pos,
    )
    return Rendered(
        "thinking",
        tokens,
        vocab.encode(tokens),
        [-100] * len(tokens),
        spans,
        [prompt_start + pos for pos in example.needle_positions],
        example.count,
    )


def render_site(example: Example, vocab: Vocab, site: str, cfg: V10Config) -> Rendered:
    if site == SITE_THINKING_FIXED:
        return render_fixed_trace(example, vocab, cfg.fixed_trace_count)
    return render(example, vocab, _site_mode(site))


@torch.no_grad()
def collect_states(
    model,
    vocab: Vocab,
    examples: list[Example],
    site: str,
    device: str | torch.device,
    *,
    max_per_label: int,
    cfg: V10Config | None = None,
) -> tuple[list[np.ndarray], np.ndarray, pd.DataFrame]:
    layer_parts: list[list[np.ndarray]] = [[] for _ in range(int(model.config.n_layer))]
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    seen: dict[int, int] = {}
    mode = _site_mode(site)
    for example_idx, example in enumerate(examples):
        item = render_site(example, vocab, site, cfg) if cfg is not None else render(example, vocab, mode)
        positions = [
            (position, label)
            for position, label in _site_positions(item, site)
            if seen.get(label, 0) < max_per_label
        ]
        if not positions:
            continue
        ids = torch.tensor([item.input_ids], dtype=torch.long, device=device)
        output = model(input_ids=ids, output_hidden_states=True)
        hidden_states = output.hidden_states or ()
        for position, label in positions:
            if seen.get(label, 0) >= max_per_label:
                continue
            seen[label] = seen.get(label, 0) + 1
            labels.append(label)
            metadata.append(
                {
                    "example_idx": example_idx,
                    "site": site,
                    "mode": mode,
                    "gold_count": example.count,
                    "state_label": label,
                    "token_position": position,
                }
            )
            for layer in range(int(model.config.n_layer)):
                layer_parts[layer].append(hidden_states[layer + 1][0, position].detach().float().cpu().numpy())
        if len(seen) == len(range(1, max(ex.count for ex in examples) + 1)) and all(
            value >= max_per_label for value in seen.values()
        ):
            break
    return (
        [np.stack(parts) for parts in layer_parts],
        np.asarray(labels, dtype=int),
        pd.DataFrame(metadata),
    )


def fit_count_directions(
    train_states: dict[str, list[np.ndarray]],
    train_labels: dict[str, np.ndarray],
    eval_states: dict[str, list[np.ndarray]],
    eval_labels: dict[str, np.ndarray],
) -> tuple[dict[tuple[str, int, str], Direction], pd.DataFrame, dict[tuple[str, int, int], np.ndarray]]:
    directions: dict[tuple[str, int, str], Direction] = {}
    rows: list[dict[str, Any]] = []
    centroids: dict[tuple[str, int, int], np.ndarray] = {}
    for site, layers in train_states.items():
        y = train_labels[site]
        y_eval = eval_labels[site]
        labels = sorted(int(value) for value in np.unique(y))
        for layer, values in enumerate(layers):
            means = {label: values[y == label].mean(axis=0) for label in labels}
            for label, centroid in means.items():
                centroids[(site, layer, label)] = centroid
            deltas = [means[b] - means[a] for a, b in zip(labels[:-1], labels[1:]) if b == a + 1]
            if not deltas:
                continue
            delta_units = [delta / max(float(np.linalg.norm(delta)), 1e-12) for delta in deltas]
            pairwise_cosines = [
                float(delta_units[i] @ delta_units[j])
                for i in range(len(delta_units))
                for j in range(i + 1, len(delta_units))
            ]
            for method in ("adjacent_mean", "ridge"):
                if method == "adjacent_mean":
                    raw = np.mean(deltas, axis=0)
                else:
                    centered = values - values.mean(axis=0, keepdims=True)
                    target = y.astype(float) - y.mean()
                    raw = np.linalg.solve(
                        centered.T @ centered + np.eye(centered.shape[1]),
                        centered.T @ target,
                    )
                norm = float(np.linalg.norm(raw))
                unit = raw / max(norm, 1e-12)
                train_projection = values @ unit
                if np.corrcoef(train_projection, y)[0, 1] < 0:
                    unit = -unit
                    train_projection = -train_projection
                projected_steps = np.asarray([float(delta @ unit) for delta in deltas])
                step_size = float(projected_steps.mean())
                design = np.column_stack([np.ones(len(train_projection)), train_projection])
                beta, *_ = np.linalg.lstsq(design, y.astype(float), rcond=None)
                eval_projection = eval_states[site][layer] @ unit
                eval_prediction = np.column_stack([np.ones(len(eval_projection)), eval_projection]) @ beta
                direction = Direction(unit.astype(np.float32), step_size, method, site, layer)
                directions[(site, layer, method)] = direction
                rows.append(
                    {
                        "site": site,
                        "layer": layer,
                        "method": method,
                        "step_size": step_size,
                        "projection_r2_heldout": _r2(y_eval.astype(float), eval_prediction),
                        "projection_mae_heldout": float(np.mean(np.abs(eval_prediction - y_eval))),
                        "adjacent_delta_cosine_mean": float(np.mean(pairwise_cosines)) if pairwise_cosines else math.nan,
                        "adjacent_delta_cosine_min": float(np.min(pairwise_cosines)) if pairwise_cosines else math.nan,
                        "adjacent_projected_step_std": float(projected_steps.std()),
                    }
                )
    return directions, pd.DataFrame(rows), centroids


def make_manifold_tables(
    states: dict[str, list[np.ndarray]],
    labels: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    geometry_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    centroid_rows: list[dict[str, Any]] = []
    for site, layers in states.items():
        y = labels[site]
        for layer, values in enumerate(layers):
            _, coordinates, ratios = _fit_pca(values, 6)
            geometry_rows.append(
                {
                    "site": site,
                    "layer": layer,
                    **{f"pc{index + 1}_variance": float(ratios[index]) if index < len(ratios) else 0.0 for index in range(6)},
                    "pc2_cumulative": float(ratios[:2].sum()),
                    "pc3_cumulative": float(ratios[:3].sum()),
                    "pc6_cumulative": float(ratios[:6].sum()),
                }
            )
            for row_idx, (label, coordinate) in enumerate(zip(y, coordinates)):
                point_rows.append(
                    {
                        "site": site,
                        "layer": layer,
                        "row_idx": row_idx,
                        "state_label": int(label),
                        **{f"pc{index + 1}": float(coordinate[index]) if index < len(coordinate) else 0.0 for index in range(6)},
                    }
                )
            frame = pd.DataFrame(point_rows[-len(y) :])
            centroids = frame.groupby("state_label", as_index=False)[[f"pc{i}" for i in range(1, 7)]].mean()
            for row in centroids.itertuples(index=False):
                centroid_rows.append({"site": site, "layer": layer, **row._asdict()})
    return pd.DataFrame(geometry_rows), pd.DataFrame(point_rows), pd.DataFrame(centroid_rows)


def _capture_residuals(model, ids: torch.Tensor, position: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
    output = model(input_ids=ids, output_hidden_states=True)
    states = [hidden[:, position].detach() for hidden in (output.hidden_states or ())[1:]]
    return output.logits[0, position].detach(), states


def _forward_residual_patch(
    model,
    receiver_ids: torch.Tensor,
    layer: int,
    receiver_pos: int,
    donor_state: torch.Tensor,
    *,
    additive: bool = False,
) -> torch.Tensor:
    def hook(_module, _args, output):
        is_tuple = isinstance(output, tuple)
        hidden = (output[0] if is_tuple else output).clone()
        if additive:
            hidden[:, receiver_pos] += donor_state.to(hidden.device)
        else:
            hidden[:, receiver_pos] = donor_state.to(hidden.device)
        return (hidden, *output[1:]) if is_tuple else hidden

    handle = model.transformer.h[int(layer)].register_forward_hook(hook)
    try:
        return model(input_ids=receiver_ids).logits[0, receiver_pos].detach()
    finally:
        handle.remove()


def _example_for_count(cfg: V10Config, vocab: Vocab, count: int, seed: int) -> Example:
    return balanced_examples(cfg, vocab, 1, seed, count_min=count, count_max=count)[0]


@torch.no_grad()
def run_geometry_steering(
    models: dict[str, Any],
    cfg: V10Config,
    vocab: Vocab,
    directions: dict[tuple[str, int, str], Direction],
    examples: list[Example],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for site in (SITE_NONTHINKING_FINAL, SITE_THINKING_FINAL, SITE_THINKING_FIXED):
        mode = _site_mode(site)
        model = models[mode]
        for example_idx, example in enumerate(examples):
            item = render_site(example, vocab, site, cfg)
            ids = torch.tensor([item.input_ids], dtype=torch.long, device=cfg.device)
            query_pos = item.spans.ans_pos
            baseline_logits, _ = _capture_residuals(model, ids, query_pos)
            baseline_pred, baseline_expected, _ = count_prediction(baseline_logits, vocab)
            for layer in range(cfg.n_layer):
                direction = directions[(site, layer, "adjacent_mean")]
                for alpha in cfg.steering_alphas:
                    update = torch.tensor(
                        direction.vector * direction.step_size * float(alpha),
                        dtype=torch.float32,
                        device=cfg.device,
                    )
                    logits = _forward_residual_patch(
                        model, ids, layer, query_pos, update, additive=True
                    )
                    pred, expected, _ = count_prediction(logits, vocab)
                    rows.append(
                        {
                            "site": site,
                            "mode": mode,
                            "example_idx": example_idx,
                            "count": example.count,
                            "layer": layer,
                            "alpha": alpha,
                            "baseline_pred": baseline_pred,
                            "baseline_expected": baseline_expected,
                            "steered_pred": pred,
                            "steered_expected": expected,
                            "causal_expected_shift": expected - baseline_expected,
                            "target_shift": alpha,
                            "accuracy": float(pred == example.count),
                        }
                    )
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["site", "mode", "layer", "alpha"], as_index=False).mean(numeric_only=True)
    return detail, summary


@torch.no_grad()
def run_final_state_transplants(
    models: dict[str, Any],
    cfg: V10Config,
    vocab: Vocab,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for site in (SITE_NONTHINKING_FINAL, SITE_THINKING_FINAL, SITE_THINKING_FIXED):
        mode = _site_mode(site)
        model = models[mode]
        for receiver_count in range(cfg.count_min, cfg.count_max + 1):
            receiver = _example_for_count(cfg, vocab, receiver_count, cfg.seed + 410_000 + receiver_count)
            receiver_item = render_site(receiver, vocab, site, cfg)
            receiver_ids = torch.tensor([receiver_item.input_ids], dtype=torch.long, device=cfg.device)
            receiver_pos = receiver_item.spans.ans_pos
            baseline_logits, _ = _capture_residuals(model, receiver_ids, receiver_pos)
            baseline_pred, baseline_expected, _ = count_prediction(baseline_logits, vocab)
            for offset in cfg.patch_offsets:
                donor_count = receiver_count + int(offset)
                if not cfg.count_min <= donor_count <= cfg.count_max:
                    continue
                donor = _example_for_count(cfg, vocab, donor_count, cfg.seed + 420_000 + donor_count)
                donor_item = render_site(donor, vocab, site, cfg)
                donor_ids = torch.tensor([donor_item.input_ids], dtype=torch.long, device=cfg.device)
                donor_pos = donor_item.spans.ans_pos
                _, donor_states = _capture_residuals(model, donor_ids, donor_pos)
                for layer, donor_state in enumerate(donor_states):
                    logits = _forward_residual_patch(
                        model, receiver_ids, layer, receiver_pos, donor_state
                    )
                    pred, expected, _ = count_prediction(logits, vocab)
                    rows.append(
                        {
                            "site": site,
                            "mode": mode,
                            "receiver_count": receiver_count,
                            "donor_count": donor_count,
                            "donor_offset": offset,
                            "layer": layer,
                            "receiver_position": receiver_pos,
                            "donor_position": donor_pos,
                            "position_delta": donor_pos - receiver_pos,
                            "baseline_pred": baseline_pred,
                            "baseline_expected": baseline_expected,
                            "patched_pred": pred,
                            "patched_expected": expected,
                            "causal_expected_shift": expected - baseline_expected,
                            "follows_donor": float(pred == donor_count),
                            "follows_receiver": float(pred == receiver_count),
                        }
                    )
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["site", "mode", "donor_offset", "layer"], as_index=False).mean(numeric_only=True)
    return detail, summary


@torch.no_grad()
def run_trace_progress_transplants(
    model,
    cfg: V10Config,
    vocab: Vocab,
    examples: list[Example],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for example_idx, example in enumerate(examples):
        item = render(example, vocab, "thinking")
        ids = torch.tensor([item.input_ids], dtype=torch.long, device=cfg.device)
        output = model(input_ids=ids, output_hidden_states=True)
        hidden_states = output.hidden_states or ()
        for receiver_k, receiver_pos in enumerate(item.spans.trace_marker_positions, start=1):
            baseline_logits = output.logits[0, receiver_pos]
            baseline_target = (
                vocab.number_id(receiver_k + 1)
                if receiver_k < example.count
                else vocab.think_close_id
            )
            for offset in cfg.patch_offsets:
                donor_k = receiver_k + int(offset)
                if not 1 <= donor_k <= example.count:
                    continue
                donor_pos = item.spans.trace_marker_positions[donor_k - 1]
                donor_target = (
                    vocab.number_id(donor_k + 1)
                    if donor_k < example.count
                    else vocab.think_close_id
                )
                for layer in range(cfg.n_layer):
                    donor_state = hidden_states[layer + 1][:, donor_pos].detach()
                    logits = _forward_residual_patch(model, ids, layer, receiver_pos, donor_state)
                    pred_id = int(logits.argmax().item())
                    rows.append(
                        {
                            "example_idx": example_idx,
                            "gold_count": example.count,
                            "receiver_k": receiver_k,
                            "donor_k": donor_k,
                            "donor_offset": offset,
                            "layer": layer,
                            "receiver_position": receiver_pos,
                            "donor_position": donor_pos,
                            "position_delta": donor_pos - receiver_pos,
                            "baseline_target_id": baseline_target,
                            "donor_target_id": donor_target,
                            "patched_pred_id": pred_id,
                            "follows_donor_successor": float(pred_id == donor_target),
                            "keeps_receiver_successor": float(pred_id == baseline_target),
                            "donor_is_final": float(donor_k == example.count),
                            "receiver_is_final": float(receiver_k == example.count),
                            "early_close_induced": float(
                                donor_k == example.count
                                and receiver_k < example.count
                                and pred_id == vocab.think_close_id
                            ),
                            "continuation_induced": float(
                                donor_k < example.count
                                and receiver_k == example.count
                                and pred_id == donor_target
                            ),
                        }
                    )
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["donor_offset", "layer"], as_index=False).mean(numeric_only=True)
    return detail, summary


def run_state_causal(cfg: V10Config, vocab: Vocab, run_dir: Path) -> dict[str, pd.DataFrame]:
    out_dir = run_dir / "analysis" / "state_causal"
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    models = {mode: load_final_model(cfg, vocab, run_dir, mode) for mode in cfg.modes}
    train_examples = balanced_examples(cfg, vocab, cfg.state_train_examples_per_count, cfg.seed + 80_000)
    eval_examples = balanced_examples(cfg, vocab, cfg.state_eval_examples_per_count, cfg.seed + 81_000)
    sites = (
        SITE_NONTHINKING_FINAL,
        SITE_THINKING_FINAL,
        SITE_THINKING_FIXED,
        SITE_THINKING_INDEX,
        SITE_THINKING_MARKER,
    )
    train_states: dict[str, list[np.ndarray]] = {}
    eval_states: dict[str, list[np.ndarray]] = {}
    train_labels: dict[str, np.ndarray] = {}
    eval_labels: dict[str, np.ndarray] = {}
    metadata_frames: list[pd.DataFrame] = []
    for site in sites:
        model = models[_site_mode(site)]
        states, labels, metadata = collect_states(
            model,
            vocab,
            train_examples,
            site,
            cfg.device,
            max_per_label=cfg.state_train_examples_per_count,
            cfg=cfg,
        )
        metadata["split"] = "direction_train"
        train_states[site], train_labels[site] = states, labels
        metadata_frames.append(metadata)
        states, labels, metadata = collect_states(
            model,
            vocab,
            eval_examples,
            site,
            cfg.device,
            max_per_label=cfg.state_eval_examples_per_count,
            cfg=cfg,
        )
        metadata["split"] = "heldout_eval"
        eval_states[site], eval_labels[site] = states, labels
        metadata_frames.append(metadata)

    directions, geometry, centroids = fit_count_directions(
        train_states, train_labels, eval_states, eval_labels
    )
    manifold_geometry, manifold_points, manifold_centroids = make_manifold_tables(
        eval_states, eval_labels
    )
    direction_arrays = {
        f"{site}__L{layer + 1}__{method}": direction.vector
        for (site, layer, method), direction in directions.items()
    }
    np.savez(out_dir / "directions.npz", **direction_arrays)
    np.savez(
        out_dir / "centroids.npz",
        **{
            f"{site}__L{layer + 1}__C{count}": value
            for (site, layer, count), value in centroids.items()
        },
    )

    steering_rows, steering_summary = run_geometry_steering(
        models, cfg, vocab, directions, eval_examples
    )
    final_patch_rows, final_patch_summary = run_final_state_transplants(
        models, cfg, vocab
    )
    trace_examples = balanced_examples(
        cfg,
        vocab,
        cfg.state_causal_examples_per_count,
        cfg.seed + 82_000,
        count_min=max(2, cfg.count_max - 4),
        count_max=cfg.count_max,
    )
    trace_patch_rows, trace_patch_summary = run_trace_progress_transplants(
        models["thinking"], cfg, vocab, trace_examples
    )
    outputs = {
        "state_metadata": pd.concat(metadata_frames, ignore_index=True),
        "direction_geometry": geometry,
        "manifold_geometry": manifold_geometry,
        "manifold_points": manifold_points,
        "manifold_centroids": manifold_centroids,
        "steering_rows": steering_rows,
        "steering_summary": steering_summary,
        "final_state_transplant_rows": final_patch_rows,
        "final_state_transplant_summary": final_patch_summary,
        "trace_progress_transplant_rows": trace_patch_rows,
        "trace_progress_transplant_summary": trace_patch_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(tables / f"{name}.csv", index=False)
    for model in models.values():
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs
