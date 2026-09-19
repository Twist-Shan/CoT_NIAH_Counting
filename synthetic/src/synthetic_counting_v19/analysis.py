from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import ReferenceConfig, RunSpec
from .data import (
    ANSWER,
    COUNT,
    END,
    START,
    ReferenceExample,
    digit_trace_layout,
    render,
    sample_example,
)
from .model import ReferenceTransformer
from .training import atomic_csv


def count_band(count: int) -> str:
    if count <= 32:
        return "count_1_32"
    if count <= 64:
        return "count_33_64"
    if count <= 96:
        return "count_65_96"
    return "count_97_128"


def balanced_examples(
    cfg: ReferenceConfig,
    spec: RunSpec,
    examples_per_count: int,
    seed: int,
) -> list[ReferenceExample]:
    rng = random.Random(seed)
    examples = [
        sample_example(cfg, spec, rng, count=count)
        for count in range(1, spec.eval_max_count + 1)
        for _ in range(examples_per_count)
    ]
    rng.shuffle(examples)
    return examples


def load_final_model(
    cfg: ReferenceConfig,
    spec: RunSpec,
    run_dir: Path,
) -> ReferenceTransformer:
    checkpoint = run_dir / "runs" / spec.name / "checkpoints" / "final.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing final checkpoint for {spec.name}: {checkpoint}")
    payload = torch.load(checkpoint, map_location=cfg.device, weights_only=False)
    model = ReferenceTransformer(cfg).to(cfg.device)
    model.load_state_dict(payload["model"])
    return model.eval()


def _normalized_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    if len(values) <= 1 or total <= 1e-12:
        return 0.0
    probabilities = values / total
    entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())
    return entropy / math.log(len(values))


def _trace_positions(context_length: int, count: int) -> tuple[list[int], list[int], int]:
    layout = digit_trace_layout(context_length, count)
    return (
        list(layout.index_query_positions),
        list(layout.marker_positions),
        layout.end_position,
    )


def _trace_number_positions(context_length: int, count: int) -> list[int]:
    layout = digit_trace_layout(context_length, count)
    return list(layout.index_role_positions) + [
        position
        for group in layout.index_digit_positions
        for position in group
    ]


def _attention_categories(
    example: ReferenceExample,
    spec: RunSpec,
    weights: np.ndarray,
    query_position: int,
) -> dict[str, float]:
    context_length = spec.context_length
    needle_positions = list(example.needle_positions)
    needle_set = set(needle_positions)
    noise_positions = [position for position in range(context_length) if position not in needle_set]
    needle_weights = weights[needle_positions]
    needle_mass = float(needle_weights.sum())
    index_positions: list[int] = []
    marker_positions: list[int] = []
    end_position: int | None = None
    if spec.mode == "cot":
        index_positions, marker_positions, end_position = _trace_positions(context_length, example.count)
    trace_number_positions = (
        _trace_number_positions(context_length, example.count)
        if spec.mode == "cot"
        else []
    )
    visible_indices = [position for position in trace_number_positions if position <= query_position]
    visible_markers = [position for position in marker_positions if position <= query_position]
    values = {
        "prompt_needles_mass": needle_mass,
        "prompt_noise_mass": float(weights[noise_positions].sum()),
        "needle_entropy_normalized": _normalized_entropy(needle_weights),
        "start_mass": float(weights[context_length]) if spec.mode == "cot" else 0.0,
        "answer_prompt_mass": float(weights[context_length]) if spec.mode == "direct" else 0.0,
        "trace_indices_mass": float(weights[visible_indices].sum()) if visible_indices else 0.0,
        "trace_markers_mass": float(weights[visible_markers].sum()) if visible_markers else 0.0,
        "end_mass": (
            float(weights[end_position])
            if end_position is not None and end_position <= query_position
            else 0.0
        ),
        "query_self_mass": float(weights[query_position]),
    }
    values["broad_attention_score"] = (
        values["prompt_needles_mass"] * values["needle_entropy_normalized"]
    )
    named = (
        values["prompt_needles_mass"]
        + values["prompt_noise_mass"]
        + values["start_mass"]
        + values["answer_prompt_mass"]
        + values["trace_indices_mass"]
        + values["trace_markers_mass"]
        + values["end_mass"]
    )
    values["other_context_mass"] = max(0.0, 1.0 - named)
    return values


@torch.no_grad()
def collect_attention_for_spec(
    model: ReferenceTransformer,
    cfg: ReferenceConfig,
    spec: RunSpec,
    examples: list[ReferenceExample],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for example_id, example in enumerate(examples):
        item = render(example, spec.mode)
        ids = torch.tensor([item.tokens], dtype=torch.long, device=cfg.device)
        output = model(ids, output_attentions=True)
        queries: list[tuple[str, int, int | None]]
        if spec.mode == "direct":
            queries = [("final_answer", spec.context_length + 1, None)]
        else:
            index_positions, marker_positions, end_position = _trace_positions(
                spec.context_length,
                example.count,
            )
            final_query = digit_trace_layout(
                spec.context_length,
                example.count,
            ).count_role_position
            queries = [("final_answer", final_query, None)]
            queries.extend(
                ("trace_index", position, k)
                for k, position in enumerate(index_positions, 1)
            )
            queries.extend(
                ("trace_marker", position, k)
                for k, position in enumerate(marker_positions, 1)
            )
        for layer, layer_attention in enumerate(output.attentions or (), 1):
            matrix = layer_attention[0].detach().float().cpu().numpy()
            for head in range(matrix.shape[0]):
                for query_kind, query_position, query_k in queries:
                    weights = matrix[head, query_position]
                    categories = _attention_categories(
                        example,
                        spec,
                        weights,
                        query_position,
                    )
                    correct_mass = math.nan
                    correct_top1 = math.nan
                    diagonal = math.nan
                    next_needle_mass = math.nan
                    if query_kind == "trace_index" and query_k is not None:
                        needle_weights = weights[list(example.needle_positions)]
                        correct_mass = float(needle_weights[query_k - 1])
                        correct_top1 = float(int(np.argmax(needle_weights) == query_k - 1))
                        diagonal = correct_mass / max(float(needle_weights.sum()), 1e-12)
                    if (
                        query_kind == "trace_marker"
                        and query_k is not None
                        and query_k < example.count
                    ):
                        next_needle_mass = float(weights[example.needle_positions[query_k]])
                    rows.append(
                        {
                            "run_name": spec.name,
                            "distribution": spec.distribution,
                            "mode": spec.mode,
                            "example_id": example_id,
                            "count": example.count,
                            "count_band": count_band(example.count),
                            "query_kind": query_kind,
                            "query_k": query_k,
                            "layer": layer,
                            "head": head,
                            "correct_prompt_needle_mass": correct_mass,
                            "correct_top1": correct_top1,
                            "diagonal_dominance": diagonal,
                            "next_prompt_needle_mass": next_needle_mass,
                            **categories,
                        }
                    )
        del output
    return pd.DataFrame(rows)


def summarize_attention(detail: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "run_name",
        "distribution",
        "mode",
        "query_kind",
        "count_band",
        "layer",
        "head",
    ]
    excluded = {"example_id", "count", "query_k", "layer", "head"}
    numeric = [
        column
        for column in detail.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    by_band = detail.groupby(keys, as_index=False)[numeric].mean()
    overall = detail.copy()
    overall["count_band"] = "all"
    overall = overall.groupby(keys, as_index=False)[numeric].mean()
    return pd.concat((by_band, overall), ignore_index=True)


def run_attention_analysis(
    cfg: ReferenceConfig,
    specs: tuple[RunSpec, ...],
    run_dir: Path,
) -> None:
    parts: list[pd.DataFrame] = []
    for run_index, spec in enumerate(specs, 1):
        print(f"[attention] {run_index}/{len(specs)} {spec.name}", flush=True)
        model = load_final_model(cfg, spec, run_dir)
        examples = balanced_examples(
            cfg,
            spec,
            cfg.attention_examples_per_count,
            cfg.seed + 110_000,
        )
        parts.append(collect_attention_for_spec(model, cfg, spec, examples))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    detail = pd.concat(parts, ignore_index=True)
    atomic_csv(detail, run_dir / "tables" / "attention_detail.csv")
    atomic_csv(summarize_attention(detail), run_dir / "tables" / "attention_summary.csv")


def _site_positions(
    example: ReferenceExample,
    spec: RunSpec,
    site: str,
) -> list[tuple[int, int]]:
    if site == "final_answer":
        if spec.mode == "direct":
            return [(spec.context_length + 1, example.count)]
        layout = digit_trace_layout(spec.context_length, example.count)
        return [(layout.count_role_position, example.count)]
    if spec.mode != "cot":
        return []
    index_positions, marker_positions, _ = _trace_positions(spec.context_length, example.count)
    if site == "trace_index":
        return [(position, k) for k, position in enumerate(index_positions, 1)]
    if site == "trace_marker":
        return [(position, k) for k, position in enumerate(marker_positions, 1)]
    raise ValueError(f"Unknown state site: {site}")


@torch.no_grad()
def collect_states_for_spec(
    model: ReferenceTransformer,
    cfg: ReferenceConfig,
    spec: RunSpec,
    site: str,
    examples: list[ReferenceExample],
    max_per_label: int,
) -> tuple[list[np.ndarray], np.ndarray, pd.DataFrame]:
    layer_parts: list[list[np.ndarray]] = [[] for _ in range(cfg.n_layer + 1)]
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    seen: dict[int, int] = {}
    for example_id, example in enumerate(examples):
        selected = [
            (position, label)
            for position, label in _site_positions(example, spec, site)
            if seen.get(label, 0) < max_per_label
        ]
        if not selected:
            continue
        item = render(example, spec.mode)
        ids = torch.tensor([item.tokens], dtype=torch.long, device=cfg.device)
        output = model(ids, output_hidden_states=True)
        hidden_states = output.hidden_states or ()
        for position, label in selected:
            if seen.get(label, 0) >= max_per_label:
                continue
            seen[label] = seen.get(label, 0) + 1
            labels.append(label)
            metadata.append(
                {
                    "run_name": spec.name,
                    "distribution": spec.distribution,
                    "mode": spec.mode,
                    "site": site,
                    "example_id": example_id,
                    "gold_count": example.count,
                    "state_label": label,
                    "token_position": position,
                }
            )
            for layer in range(cfg.n_layer + 1):
                layer_parts[layer].append(
                    hidden_states[layer][0, position].detach().float().cpu().numpy()
                )
        del output
    if not labels:
        raise RuntimeError(f"No states collected for {spec.name}/{site}")
    return [np.stack(values) for values in layer_parts], np.asarray(labels), pd.DataFrame(metadata)


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def _ridge_predict(
    train: np.ndarray,
    train_y: np.ndarray,
    test: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    train_scaled, test_scaled = _standardize(train, test)
    train_design = np.column_stack((np.ones(len(train_scaled)), train_scaled))
    test_design = np.column_stack((np.ones(len(test_scaled)), test_scaled))
    penalty = np.eye(train_design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(
        train_design.T @ train_design + penalty,
        train_design.T @ train_y.astype(float),
    )
    return test_design @ beta


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(((target - target.mean()) ** 2).sum())
    if denominator <= 1e-12:
        return math.nan
    return 1.0 - float(((target - prediction) ** 2).sum()) / denominator


def _nearest_centroid(
    train: np.ndarray,
    train_y: np.ndarray,
    test: np.ndarray,
    test_y: np.ndarray,
) -> tuple[float, dict[int, np.ndarray]]:
    train_scaled, test_scaled = _standardize(train, test)
    classes = sorted(int(value) for value in np.unique(train_y))
    scaled_centroids = np.stack([train_scaled[train_y == value].mean(0) for value in classes])
    distances = ((test_scaled[:, None] - scaled_centroids[None]) ** 2).sum(-1)
    predictions = np.asarray(classes)[distances.argmin(1)]
    centroids = {value: train[train_y == value].mean(0) for value in classes}
    return float(np.mean(predictions == test_y)), centroids


def _position_baseline(
    train_positions: np.ndarray,
    train_y: np.ndarray,
    test_positions: np.ndarray,
    test_y: np.ndarray,
) -> float:
    classes = sorted(int(value) for value in np.unique(train_y))
    means = np.asarray([train_positions[train_y == value].mean() for value in classes])
    predictions = np.asarray(classes)[np.abs(test_positions[:, None] - means[None]).argmin(1)]
    return float(np.mean(predictions == test_y))


def _pca_centroids(
    centroids: dict[int, np.ndarray],
    n_components: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = np.asarray(sorted(centroids), dtype=int)
    values = np.stack([centroids[int(label)] for label in labels])
    centered = values - values.mean(0, keepdims=True)
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    component_count = min(n_components, len(components))
    coordinates = centered @ components[:component_count].T
    variance = singular**2
    total_variance = float(variance.sum())
    # Centroids can be exactly equal at an uninformative site. Treat numerical
    # SVD residue as zero instead of reporting a spurious one-dimensional axis.
    scale = max(float(np.square(values).sum()), 1.0)
    if total_variance <= np.finfo(float).eps * scale * 100:
        ratios = np.zeros_like(variance)
        coordinates = np.zeros_like(coordinates)
        effective_dimension = 0.0
    else:
        ratios = variance / total_variance
        effective_dimension = float(total_variance**2 / float((variance**2).sum()))
    coordinate_values: dict[str, Any] = {"state_label": labels}
    for component in range(component_count):
        coordinate_values[f"pc{component + 1}"] = coordinates[:, component]
    variance_frame = pd.DataFrame(
        {
            "component": np.arange(1, component_count + 1),
            "explained_variance_ratio": ratios[:component_count],
            "cumulative_explained_variance": np.cumsum(ratios[:component_count]),
            "effective_dimension": effective_dimension,
        }
    )
    return pd.DataFrame(coordinate_values), variance_frame


def analyze_state_pair(
    train_states: list[np.ndarray],
    train_labels: np.ndarray,
    train_meta: pd.DataFrame,
    eval_states: list[np.ndarray],
    eval_labels: np.ndarray,
    eval_meta: pd.DataFrame,
    spec: RunSpec,
    site: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    probe_rows: list[dict[str, Any]] = []
    centroid_parts: list[pd.DataFrame] = []
    variance_parts: list[pd.DataFrame] = []
    for layer, (train, test) in enumerate(zip(train_states, eval_states)):
        nearest_accuracy, centroids = _nearest_centroid(train, train_labels, test, eval_labels)
        prediction = _ridge_predict(train, train_labels, test)
        probe_rows.append(
            {
                "run_name": spec.name,
                "distribution": spec.distribution,
                "mode": spec.mode,
                "site": site,
                "layer": layer,
                "nearest_centroid_accuracy": nearest_accuracy,
                "position_only_accuracy": _position_baseline(
                    train_meta.token_position.to_numpy(),
                    train_labels,
                    eval_meta.token_position.to_numpy(),
                    eval_labels,
                ),
                "ridge_r2": _r2(eval_labels.astype(float), prediction),
                "ridge_mae": float(np.mean(np.abs(prediction - eval_labels))),
            }
        )
        coordinates, variance = _pca_centroids(centroids)
        for frame in (coordinates, variance):
            frame.insert(0, "layer", layer)
            frame.insert(0, "site", site)
            frame.insert(0, "mode", spec.mode)
            frame.insert(0, "distribution", spec.distribution)
            frame.insert(0, "run_name", spec.name)
        centroid_parts.append(coordinates)
        variance_parts.append(variance)
    return (
        pd.DataFrame(probe_rows),
        pd.concat(centroid_parts, ignore_index=True),
        pd.concat(variance_parts, ignore_index=True),
    )


def run_state_analysis(
    cfg: ReferenceConfig,
    specs: tuple[RunSpec, ...],
    run_dir: Path,
) -> None:
    probe_parts: list[pd.DataFrame] = []
    centroid_parts: list[pd.DataFrame] = []
    variance_parts: list[pd.DataFrame] = []
    metadata_parts: list[pd.DataFrame] = []
    for run_index, spec in enumerate(specs, 1):
        model = load_final_model(cfg, spec, run_dir)
        sites = ("final_answer",) if spec.mode == "direct" else (
            "final_answer",
            "trace_index",
            "trace_marker",
        )
        train_examples = balanced_examples(
            cfg,
            spec,
            cfg.state_train_examples_per_count,
            cfg.seed + 120_000,
        )
        eval_examples = balanced_examples(
            cfg,
            spec,
            cfg.state_eval_examples_per_count,
            cfg.seed + 130_000,
        )
        for site in sites:
            print(
                f"[state] {run_index}/{len(specs)} {spec.name}/{site}",
                flush=True,
            )
            train_states, train_labels, train_meta = collect_states_for_spec(
                model,
                cfg,
                spec,
                site,
                train_examples,
                cfg.state_train_examples_per_count,
            )
            eval_states, eval_labels, eval_meta = collect_states_for_spec(
                model,
                cfg,
                spec,
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
                spec,
                site,
            )
            probe_parts.append(probes)
            centroid_parts.append(centroids)
            variance_parts.append(variance)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    atomic_csv(pd.concat(probe_parts, ignore_index=True), run_dir / "tables" / "state_probe_summary.csv")
    atomic_csv(pd.concat(centroid_parts, ignore_index=True), run_dir / "tables" / "state_centroids_pca.csv")
    atomic_csv(pd.concat(variance_parts, ignore_index=True), run_dir / "tables" / "state_pca_variance.csv")
    atomic_csv(pd.concat(metadata_parts, ignore_index=True), run_dir / "tables" / "state_metadata.csv")
