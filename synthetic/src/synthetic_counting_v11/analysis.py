from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig, config_from_dict
from .data import Example, Rendered, Vocab, balanced_examples, collate, render
from .training import load_final_model


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _normalized_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    if len(values) <= 1 or total <= 1e-12:
        return 0.0
    probabilities = values / total
    entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())
    return entropy / math.log(len(values))


def _attention_categories(item: Rendered, weights: np.ndarray, query_position: int) -> dict[str, float]:
    spans = item.spans
    needles = list(item.prompt_needle_positions)
    needle_set = set(needles)
    prompt = list(range(spans.prompt_start, spans.prompt_end_exclusive))
    noise = [position for position in prompt if position not in needle_set]
    needle_weights = weights[needles] if needles else np.asarray([], dtype=np.float64)
    needle_mass = float(needle_weights.sum()) if len(needle_weights) else 0.0
    values = {
        "bos_mass": float(weights[spans.bos_pos]),
        "task_prefix_mass": (
            float(weights[list(spans.task_prefix_positions)].sum())
            if spans.task_prefix_positions
            else 0.0
        ),
        "prompt_needles_mass": needle_mass,
        "prompt_noise_mass": float(weights[noise].sum()) if noise else 0.0,
        "think_open_mass": float(weights[spans.think_pos]) if spans.think_pos is not None else 0.0,
        "trace_indices_mass": (
            float(weights[spans.trace_index_positions].sum()) if spans.trace_index_positions else 0.0
        ),
        "trace_markers_mass": (
            float(weights[spans.trace_marker_positions].sum()) if spans.trace_marker_positions else 0.0
        ),
        "think_close_mass": (
            float(weights[spans.think_close_pos]) if spans.think_close_pos is not None else 0.0
        ),
        "ans_mass": float(weights[spans.ans_pos]) if spans.ans_pos <= query_position else 0.0,
        "query_self_mass": float(weights[query_position]),
        "needle_entropy_normalized": _normalized_entropy(needle_weights),
    }
    values["broad_attention_score"] = (
        values["prompt_needles_mass"] * values["needle_entropy_normalized"]
    )
    # Categories overlap at query_self only when the query itself belongs to a named
    # set, so other_context is computed from the disjoint context partition instead.
    named_context = (
        values["bos_mass"]
        + values["task_prefix_mass"]
        + values["prompt_needles_mass"]
        + values["prompt_noise_mass"]
        + values["think_open_mass"]
        + values["trace_indices_mass"]
        + values["trace_markers_mass"]
        + values["think_close_mass"]
        + values["ans_mass"]
    )
    values["other_context_mass"] = max(0.0, 1.0 - named_context)
    return values


@torch.no_grad()
def collect_attention_for_variant(
    model,
    cfg: ExperimentConfig,
    vocab: Vocab,
    position_encoding: str,
    mode: str,
    examples: list[Example],
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    # Materializing L x B x H x T x T attention is the expensive part. A small
    # analysis batch is faster than per-example forwards but stays memory-safe at T=512.
    batch_size = min(int(cfg.analysis_batch_size), 16)
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        rendered = [render(example, vocab, mode) for example in chunk]
        ids, _, attention_mask = collate(rendered, vocab, cfg.device)
        output = model(input_ids=ids, attention_mask=attention_mask, output_attentions=True)
        attentions = output.attentions or ()
        for row_index, (example, item) in enumerate(zip(chunk, rendered)):
            queries: list[tuple[str, int, int | None]] = [("final_answer", item.spans.ans_pos, None)]
            if mode == "thinking":
                queries.extend(
                    ("trace_index", position, index)
                    for index, position in enumerate(item.spans.trace_index_positions, start=1)
                )
            for layer_index, layer_attention in enumerate(attentions, start=1):
                matrix = layer_attention[row_index].detach().float().cpu().numpy()
                for head_index in range(matrix.shape[0]):
                    for query_kind, query_position, query_k in queries:
                        weights = matrix[head_index, query_position]
                        categories = _attention_categories(item, weights, query_position)
                        correct_mass = math.nan
                        correct_top1 = math.nan
                        diagonal = math.nan
                        if query_k is not None:
                            needle_weights = weights[item.prompt_needle_positions]
                            correct_mass = float(needle_weights[query_k - 1])
                            correct_top1 = float(int(np.argmax(needle_weights) == query_k - 1))
                            diagonal = correct_mass / max(float(needle_weights.sum()), 1e-12)
                        rows.append(
                            {
                                "position_encoding": position_encoding,
                                "mode": mode,
                                "example_id": int(start + row_index),
                                "count": int(example.count),
                                "count_bin": cfg.count_bin(example.count),
                                "query_kind": query_kind,
                                "query_k": query_k,
                                "layer": int(layer_index),
                                "head": int(head_index),
                                "correct_prompt_needle_mass": correct_mass,
                                "correct_top1": correct_top1,
                                "diagonal_dominance": diagonal,
                                **categories,
                            }
                        )
    return pd.DataFrame(rows)


def summarize_attention(detail: pd.DataFrame) -> pd.DataFrame:
    keys = ["position_encoding", "mode", "query_kind", "count_bin", "layer", "head"]
    numeric = [
        column
        for column in detail.select_dtypes(include=[np.number]).columns
        if column not in {"example_id", "count", "query_k", "layer", "head"}
    ]
    by_bin = detail.groupby(keys, as_index=False)[numeric].mean()
    overall = detail.copy()
    overall["count_bin"] = "all"
    overall = overall.groupby(keys, as_index=False)[numeric].mean()
    return pd.concat((by_bin, overall), ignore_index=True)


def run_attention_analysis(cfg: ExperimentConfig, vocab: Vocab, run_dir: Path) -> None:
    examples = balanced_examples(
        cfg,
        vocab,
        cfg.attention_examples_per_count,
        cfg.seed + 110_000,
    )
    parts: list[pd.DataFrame] = []
    for position_encoding, mode in cfg.model_variants:
        print(f"[attention] {position_encoding}/{mode}", flush=True)
        _, _, model = load_final_model(run_dir, position_encoding, mode, cfg.device)
        parts.append(
            collect_attention_for_variant(
                model,
                cfg,
                vocab,
                position_encoding,
                mode,
                examples,
            )
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    detail = pd.concat(parts, ignore_index=True)
    _atomic_csv(detail, run_dir / "tables" / "attention_detail.csv")
    _atomic_csv(summarize_attention(detail), run_dir / "tables" / "attention_summary.csv")


def _site_positions(item: Rendered, site: str) -> list[tuple[int, int]]:
    if site == "final_answer":
        return [(item.spans.ans_pos, item.count)]
    if site == "trace_index":
        return [(position, k) for k, position in enumerate(item.spans.trace_index_positions, start=1)]
    if site == "trace_marker":
        return [(position, k) for k, position in enumerate(item.spans.trace_marker_positions, start=1)]
    raise ValueError(f"Unknown state site: {site}")


@torch.no_grad()
def collect_states_for_variant(
    model,
    cfg: ExperimentConfig,
    vocab: Vocab,
    position_encoding: str,
    mode: str,
    site: str,
    examples: list[Example],
    max_per_label: int,
) -> tuple[list[np.ndarray], np.ndarray, pd.DataFrame]:
    layer_parts: list[list[np.ndarray]] = [[] for _ in range(cfg.n_layer + 1)]
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    seen: dict[int, int] = {}
    model.eval()
    for example_index, example in enumerate(examples):
        item = render(example, vocab, mode)
        selected = [
            (position, label)
            for position, label in _site_positions(item, site)
            if seen.get(label, 0) < max_per_label
        ]
        if not selected:
            continue
        ids = torch.tensor([item.input_ids], dtype=torch.long, device=cfg.device)
        output = model(input_ids=ids, output_hidden_states=True)
        hidden_states = output.hidden_states or ()
        for position, label in selected:
            if seen.get(label, 0) >= max_per_label:
                continue
            seen[label] = seen.get(label, 0) + 1
            labels.append(int(label))
            metadata.append(
                {
                    "position_encoding": position_encoding,
                    "mode": mode,
                    "site": site,
                    "example_id": int(example_index),
                    "gold_count": int(example.count),
                    "state_label": int(label),
                    "token_position": int(position),
                }
            )
            for layer_index in range(cfg.n_layer + 1):
                vector = hidden_states[layer_index][0, position].detach().float().cpu().numpy()
                layer_parts[layer_index].append(vector)
    if not labels:
        raise RuntimeError(f"No hidden states collected for {position_encoding}/{mode}/{site}")
    return [np.stack(parts) for parts in layer_parts], np.asarray(labels), pd.DataFrame(metadata)


def _ridge_fit_predict(
    train: np.ndarray,
    train_y: np.ndarray,
    test: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    x_train = (train - mean) / scale
    x_test = (test - mean) / scale
    x_train = np.column_stack((np.ones(len(x_train)), x_train))
    x_test = np.column_stack((np.ones(len(x_test)), x_test))
    penalty = np.eye(x_train.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ train_y.astype(float))
    return x_test @ beta


def _r2(y: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(((y - y.mean()) ** 2).sum())
    if denominator <= 1e-12:
        return math.nan
    return 1.0 - float(((y - prediction) ** 2).sum()) / denominator


def _nearest_centroid_metrics(
    train: np.ndarray,
    train_y: np.ndarray,
    test: np.ndarray,
    test_y: np.ndarray,
) -> tuple[float, dict[int, np.ndarray]]:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    train_scaled = (train - mean) / scale
    test_scaled = (test - mean) / scale
    classes = sorted(int(value) for value in np.unique(train_y))
    centroids_scaled = np.stack([train_scaled[train_y == value].mean(axis=0) for value in classes])
    distances = ((test_scaled[:, None] - centroids_scaled[None]) ** 2).sum(axis=-1)
    predictions = np.asarray(classes)[distances.argmin(axis=1)]
    raw_centroids = {value: train[train_y == value].mean(axis=0) for value in classes}
    return float(np.mean(predictions == test_y)), raw_centroids


def _position_baseline(
    train_positions: np.ndarray,
    train_y: np.ndarray,
    test_positions: np.ndarray,
    test_y: np.ndarray,
) -> float:
    classes = sorted(int(value) for value in np.unique(train_y))
    means = np.asarray([train_positions[train_y == value].mean() for value in classes])
    predictions = np.asarray(classes)[np.abs(test_positions[:, None] - means[None]).argmin(axis=1)]
    return float(np.mean(predictions == test_y))


def _pca_centroids(
    centroids: dict[int, np.ndarray],
    n_components: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = np.asarray(sorted(centroids), dtype=int)
    values = np.stack([centroids[int(label)] for label in labels])
    centered = values - values.mean(axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    component_count = min(n_components, len(vh))
    coordinates = centered @ vh[:component_count].T
    variance = singular**2
    ratios = variance / max(float(variance.sum()), 1e-12)
    coordinate_rows = {"state_label": labels}
    for index in range(component_count):
        coordinate_rows[f"pc{index + 1}"] = coordinates[:, index]
    variance_frame = pd.DataFrame(
        {
            "component": np.arange(1, component_count + 1),
            "explained_variance_ratio": ratios[:component_count],
            "cumulative_explained_variance": np.cumsum(ratios[:component_count]),
        }
    )
    return pd.DataFrame(coordinate_rows), variance_frame


def analyze_state_pair(
    train_states: list[np.ndarray],
    train_labels: np.ndarray,
    train_meta: pd.DataFrame,
    eval_states: list[np.ndarray],
    eval_labels: np.ndarray,
    eval_meta: pd.DataFrame,
    *,
    position_encoding: str,
    mode: str,
    site: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    probe_rows: list[dict[str, Any]] = []
    centroid_parts: list[pd.DataFrame] = []
    variance_parts: list[pd.DataFrame] = []
    for layer_index, (train, test) in enumerate(zip(train_states, eval_states)):
        nearest_accuracy, centroids = _nearest_centroid_metrics(train, train_labels, test, eval_labels)
        ridge_prediction = _ridge_fit_predict(train, train_labels, test)
        position_accuracy = _position_baseline(
            train_meta.token_position.to_numpy(),
            train_labels,
            eval_meta.token_position.to_numpy(),
            eval_labels,
        )
        probe_rows.append(
            {
                "position_encoding": position_encoding,
                "mode": mode,
                "site": site,
                "layer": layer_index,
                "nearest_centroid_accuracy": nearest_accuracy,
                "position_only_accuracy": position_accuracy,
                "ridge_r2": _r2(eval_labels.astype(float), ridge_prediction),
                "ridge_mae": float(np.mean(np.abs(ridge_prediction - eval_labels))),
            }
        )
        coordinates, variance = _pca_centroids(centroids)
        for frame in (coordinates, variance):
            frame.insert(0, "layer", layer_index)
            frame.insert(0, "site", site)
            frame.insert(0, "mode", mode)
            frame.insert(0, "position_encoding", position_encoding)
        centroid_parts.append(coordinates)
        variance_parts.append(variance)
    return (
        pd.DataFrame(probe_rows),
        pd.concat(centroid_parts, ignore_index=True),
        pd.concat(variance_parts, ignore_index=True),
    )


def run_state_analysis(cfg: ExperimentConfig, vocab: Vocab, run_dir: Path) -> None:
    train_examples = balanced_examples(
        cfg,
        vocab,
        max(cfg.state_train_examples_per_count, 1),
        cfg.seed + 120_000,
    )
    eval_examples = balanced_examples(
        cfg,
        vocab,
        max(cfg.state_eval_examples_per_count, 1),
        cfg.seed + 130_000,
    )
    probe_parts: list[pd.DataFrame] = []
    centroid_parts: list[pd.DataFrame] = []
    variance_parts: list[pd.DataFrame] = []
    metadata_parts: list[pd.DataFrame] = []
    for position_encoding, mode in cfg.model_variants:
        _, _, model = load_final_model(run_dir, position_encoding, mode, cfg.device)
        sites = ("final_answer",) if mode == "nonthinking" else (
            "final_answer",
            "trace_index",
            "trace_marker",
        )
        for site in sites:
            print(f"[state] {position_encoding}/{mode}/{site}", flush=True)
            train_states, train_labels, train_meta = collect_states_for_variant(
                model,
                cfg,
                vocab,
                position_encoding,
                mode,
                site,
                train_examples,
                cfg.state_train_examples_per_count,
            )
            eval_states, eval_labels, eval_meta = collect_states_for_variant(
                model,
                cfg,
                vocab,
                position_encoding,
                mode,
                site,
                eval_examples,
                cfg.state_eval_examples_per_count,
            )
            eval_meta = eval_meta.copy()
            eval_meta["split"] = "eval"
            metadata_parts.append(eval_meta)
            probes, centroids, variance = analyze_state_pair(
                train_states,
                train_labels,
                train_meta,
                eval_states,
                eval_labels,
                eval_meta,
                position_encoding=position_encoding,
                mode=mode,
                site=site,
            )
            probe_parts.append(probes)
            centroid_parts.append(centroids)
            variance_parts.append(variance)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _atomic_csv(pd.concat(probe_parts, ignore_index=True), run_dir / "tables" / "state_probe_summary.csv")
    _atomic_csv(pd.concat(centroid_parts, ignore_index=True), run_dir / "tables" / "state_centroids_pca.csv")
    _atomic_csv(pd.concat(variance_parts, ignore_index=True), run_dir / "tables" / "state_pca_variance.csv")
    _atomic_csv(pd.concat(metadata_parts, ignore_index=True), run_dir / "tables" / "state_metadata.csv")


def load_run_config(run_dir: str | Path, device: str | None = None) -> tuple[ExperimentConfig, Vocab]:
    run_dir = Path(run_dir)
    cfg = config_from_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    if device is not None:
        cfg = ExperimentConfig(**{**cfg.__dict__, "device": device})
    return cfg, Vocab.load(run_dir / "vocab.json")
