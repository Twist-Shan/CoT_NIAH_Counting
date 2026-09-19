from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import V16_3Config
from .data import V16_3Example, V16_3Rendered, V16_3Vocab, collate_v16_3, render_v16_3
from .training import atomic_csv, load_final_v16_3_model


def _normalized_entropy(values: np.ndarray) -> float:
    total = float(values.sum())
    if len(values) <= 1 or total <= 1e-12:
        return 0.0
    probabilities = values / total
    return float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum() / math.log(len(values)))


def _attention_categories(item: V16_3Rendered, weights: np.ndarray) -> dict[str, float]:
    assert item.spans is not None
    spans = item.spans
    needles = list(item.prompt_needle_positions)
    needle_set = set(needles)
    prompt = list(range(spans.prompt_start, spans.prompt_end_exclusive))
    non_needles = [position for position in prompt if position not in needle_set]
    needle_weights = weights[needles] if needles else np.asarray([], dtype=np.float64)
    prompt_mass = float(weights[prompt].sum()) if prompt else 0.0
    needle_fraction = float(needle_weights.sum() / prompt_mass) if prompt_mass > 1e-12 else 0.0
    uniform_fraction = len(needles) / max(1, len(prompt))
    enrichment = needle_fraction / uniform_fraction if uniform_fraction > 0 else np.nan
    top_n_recall = np.nan
    top_n_precision = np.nan
    if needles and prompt:
        top_n = min(len(needles), len(prompt))
        prompt_weights = weights[prompt]
        selected = np.argpartition(prompt_weights, -top_n)[-top_n:]
        selected_positions = {prompt[int(index)] for index in selected}
        matches = len(selected_positions & needle_set)
        top_n_recall = matches / len(needles)
        top_n_precision = matches / top_n
    return {
        "bos_mass": float(weights[spans.bos_pos]),
        "task_query_mass": float(weights[list(spans.query_positions)].sum()),
        "prompt_needles_mass": float(needle_weights.sum()) if len(needle_weights) else 0.0,
        "prompt_non_needles_mass": float(weights[non_needles].sum()) if non_needles else 0.0,
        "trace_indices_mass": float(weights[list(spans.trace_index_positions)].sum()) if spans.trace_index_positions else 0.0,
        "trace_markers_mass": float(weights[list(spans.trace_marker_positions)].sum()) if spans.trace_marker_positions else 0.0,
        "needle_entropy_normalized": _normalized_entropy(needle_weights),
        "prompt_mass": prompt_mass,
        "needle_attention_enrichment": float(enrichment),
        "top_n_needle_recall": float(top_n_recall),
        "top_n_needle_precision": float(top_n_precision),
    }


@torch.no_grad()
def collect_v16_3_attention(
    model,
    cfg: V16_3Config,
    vocab: V16_3Vocab,
    examples: list[V16_3Example],
    *,
    position_encoding: str,
    mode: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    batch_size = min(cfg.analysis_batch_size, 16, len(examples))
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        rendered = [render_v16_3(example, vocab, mode) for example in chunk]
        ids, _, mask = collate_v16_3(rendered, vocab, cfg.device)
        attentions = model(input_ids=ids, attention_mask=mask, output_attentions=True).attentions or ()
        for row_index, (example, item) in enumerate(zip(chunk, rendered)):
            assert item.spans is not None
            queries: list[tuple[str, int, int | None]] = [("final_answer", item.spans.ans_pos, None)]
            if mode == "thinking":
                queries.extend(
                    ("trace_index", position, index)
                    for index, position in enumerate(item.spans.trace_index_positions, start=1)
                )
            for layer_index, layer_attention in enumerate(attentions, start=1):
                matrix = layer_attention[row_index].detach().float().cpu().numpy()
                for head in range(matrix.shape[0]):
                    for query_kind, query_position, query_k in queries:
                        weights = matrix[head, query_position]
                        categories = _attention_categories(item, weights)
                        correct_mass = np.nan
                        correct_top1 = np.nan
                        if query_k is not None:
                            needle_weights = weights[list(item.prompt_needle_positions)]
                            correct_mass = float(needle_weights[query_k - 1])
                            correct_top1 = float(int(needle_weights.argmax()) == query_k - 1)
                        rows.append(
                            {
                                "position_encoding": position_encoding,
                                "mode": mode,
                                "example_id": start + row_index,
                                "set_id": example.set_id,
                                "set_frequency_bin": example.set_frequency_bin,
                                "count": example.count,
                                "corpus_region": example.corpus_region,
                                "corpus_start": example.corpus_start,
                                "prompt_sha256": example.prompt_sha256,
                                "query_kind": query_kind,
                                "query_k": query_k,
                                "layer": layer_index,
                                "head": head,
                                "correct_prompt_needle_mass": correct_mass,
                                "correct_top1": correct_top1,
                                "chance_top1": 1.0 / float(example.count) if query_k is not None else np.nan,
                                "diagonal_dominance": (
                                    correct_mass / max(float(categories["prompt_needles_mass"]), 1e-12)
                                    if query_k is not None else np.nan
                                ),
                                "trace_readout_mass": (
                                    categories["trace_indices_mass"] + categories["trace_markers_mass"]
                                ),
                                **categories,
                            }
                        )
    return pd.DataFrame(rows)


def run_v16_3_attention_analysis(
    cfg: V16_3Config,
    vocab: V16_3Vocab,
    run_dir: Path,
    heldout_task_examples: list[V16_3Example],
) -> None:
    examples: list[V16_3Example] = []
    for count in range(1, cfg.count_max_threshold + 1):
        examples.extend(
            [item for item in heldout_task_examples if item.count == count][
                : cfg.attention_examples_per_count
            ]
        )
    parts = []
    for position_encoding, mode in cfg.model_variants:
        print(f"[attention] {position_encoding}/{mode}", flush=True)
        _, loaded_vocab, _, model = load_final_v16_3_model(
            run_dir, position_encoding, mode, cfg.device
        )
        parts.append(
            collect_v16_3_attention(
                model,
                cfg,
                loaded_vocab,
                examples,
                position_encoding=position_encoding,
                mode=mode,
            )
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    detail = pd.concat(parts, ignore_index=True)
    atomic_csv(detail, run_dir / "tables" / "attention_detail.csv")
    numeric = [
        column
        for column in detail.select_dtypes(include=[np.number]).columns
        if column not in {"example_id", "count", "query_k", "layer", "head", "set_frequency_bin"}
    ]
    summary = detail.groupby(
        ["position_encoding", "mode", "query_kind", "layer", "head"], as_index=False
    )[numeric].mean()
    atomic_csv(summary, run_dir / "tables" / "attention_summary.csv")


def _sites(item: V16_3Rendered, mode: str) -> list[tuple[str, int, int]]:
    assert item.spans is not None and item.count is not None
    result = [("final_answer", item.spans.ans_pos, item.count)]
    if mode == "thinking":
        result.extend(
            ("trace_index", position, index)
            for index, position in enumerate(item.spans.trace_index_positions, start=1)
        )
        result.extend(
            ("trace_marker", position, index)
            for index, position in enumerate(item.spans.trace_marker_positions, start=1)
        )
    return result


@torch.no_grad()
def collect_v16_3_states(
    model,
    cfg: V16_3Config,
    vocab: V16_3Vocab,
    examples: list[V16_3Example],
    mode: str,
    max_per_label: int,
    *,
    sites: set[str] | None = None,
) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    """Collect hidden states with an explicit per-site/per-label cap.

    Selection is decided before model execution, then evaluated in batches.  A
    caller can request only ``final_answer`` or only trace sites; this is used
    by checkpoint dynamics to keep trace progress independent of total count.
    """

    vectors: dict[tuple[str, int], list[np.ndarray]] = {}
    labels: dict[tuple[str, int], list[int]] = {}
    seen: dict[tuple[str, int], int] = {}
    planned: list[tuple[V16_3Rendered, list[tuple[str, int, int]]]] = []
    for example in examples:
        item = render_v16_3(example, vocab, mode)
        selected = [
            (site, position, label)
            for site, position, label in _sites(item, mode)
            if (sites is None or site in sites)
            and seen.get((site, label), 0) < max_per_label
        ]
        if not selected:
            continue
        for site, position, label in selected:
            seen[(site, label)] = seen.get((site, label), 0) + 1
        planned.append((item, selected))
    if not planned:
        return {}
    batch_size = min(cfg.analysis_batch_size, len(planned))
    for start in range(0, len(planned), batch_size):
        chunk = planned[start : start + batch_size]
        ids, _, attention_mask = collate_v16_3(
            [item for item, _ in chunk], vocab, cfg.device
        )
        hidden_states = model(
            input_ids=ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        ).hidden_states or ()
        for row, (_, selected) in enumerate(chunk):
            for site, position, label in selected:
                for layer, hidden in enumerate(hidden_states):
                    key = (site, layer)
                    vectors.setdefault(key, []).append(
                        hidden[row, position].detach().float().cpu().numpy()
                    )
                    labels.setdefault(key, []).append(label)
    return {
        key: (np.stack(values), np.asarray(labels[key], dtype=int))
        for key, values in vectors.items()
    }


def nearest_centroid(train: np.ndarray, train_y: np.ndarray, test: np.ndarray) -> np.ndarray:
    labels = np.unique(train_y)
    centers = np.stack([train[train_y == label].mean(axis=0) for label in labels])
    distances = ((test[:, None] - centers[None]) ** 2).sum(axis=-1)
    return labels[distances.argmin(axis=1)]


def fit_ridge(
    train: np.ndarray, train_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1
    x_train = np.column_stack((np.ones(len(train)), (train - mean) / scale))
    penalty = np.eye(x_train.shape[1])
    penalty[0, 0] = 0
    beta = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ train_y)
    return mean, scale, beta


def ridge_raw_direction(
    fitted: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Return the fitted ridge direction in the original hidden coordinates."""

    _, scale, beta = fitted
    return beta[1:] / scale.reshape(-1)


def apply_ridge(
    fitted: tuple[np.ndarray, np.ndarray, np.ndarray], test: np.ndarray
) -> np.ndarray:
    mean, scale, beta = fitted
    x_test = np.column_stack((np.ones(len(test)), (test - mean) / scale))
    return x_test @ beta


def ridge_prediction(train: np.ndarray, train_y: np.ndarray, test: np.ndarray) -> np.ndarray:
    return apply_ridge(fit_ridge(train, train_y), test)


# Private aliases preserve the existing final-only implementation and older imports.
_collect_states = collect_v16_3_states
_nearest_centroid = nearest_centroid
_ridge_prediction = ridge_prediction


def run_v16_3_state_analysis(
    cfg: V16_3Config,
    vocab: V16_3Vocab,
    run_dir: Path,
    train_task_examples: list[V16_3Example],
    heldout_task_examples: list[V16_3Example],
) -> None:
    rows: list[dict[str, Any]] = []
    for position_encoding, mode in cfg.model_variants:
        print(f"[state] {position_encoding}/{mode}", flush=True)
        _, loaded_vocab, _, model = load_final_v16_3_model(
            run_dir, position_encoding, mode, cfg.device
        )
        train = _collect_states(
            model, cfg, loaded_vocab, train_task_examples, mode, cfg.state_train_examples_per_count
        )
        heldout = _collect_states(
            model, cfg, loaded_vocab, heldout_task_examples, mode, cfg.state_eval_examples_per_count
        )
        for key in sorted(set(train) & set(heldout)):
            site, layer = key
            train_x, train_y = train[key]
            test_x, test_y = heldout[key]
            nearest = _nearest_centroid(train_x, train_y, test_x)
            ridge = _ridge_prediction(train_x, train_y.astype(float), test_x)
            denominator = float(((test_y - test_y.mean()) ** 2).sum())
            rows.append(
                {
                    "position_encoding": position_encoding,
                    "mode": mode,
                    "site": site,
                    "layer": layer,
                    "nearest_centroid_accuracy": float(np.mean(nearest == test_y)),
                    "ridge_mae": float(np.mean(np.abs(ridge - test_y))),
                    "ridge_r2": (
                        1 - float(((test_y - ridge) ** 2).sum()) / denominator
                        if denominator > 1e-12
                        else np.nan
                    ),
                    "train_states": len(train_y),
                    "heldout_states": len(test_y),
                }
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    atomic_csv(pd.DataFrame(rows), run_dir / "tables" / "state_probe_summary.csv")
