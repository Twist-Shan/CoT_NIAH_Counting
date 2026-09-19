#!/usr/bin/env python3
"""Anthropic-inspired count-manifold reanalysis in frozen probe-score space.

This script deliberately does *not* treat a scalar probe prediction as the
representation.  It uses the full ten-class frozen-probe score vector as a
linear view of the residual stream, estimates the natural count trajectory,
and asks whether textual marker toggles and causal K/V cache splices move in
the same directions as natural ``c -> c + 1`` transitions.

The analysis is a lightweight precursor to a transcoder attribution graph:
it can reveal multidimensional/curved structure and test whether cache-caused
residual changes are count-manifold aligned, but it cannot see residual
directions outside the probe row span or directly characterize K/V geometry.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCHEMA = "realistic_niah_v5_anthropic_count_manifold_v1"
DEFAULT_LAYERS = (15, 16, 24)
TARGET_LANDMARKS = ("target_marker", "target_boundary")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def gauge_scores(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Remove the softmax-invariant common logit offset."""

    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size < 2 or not np.all(np.isfinite(vector)):
        raise ValueError("Probe scores must be a finite vector of length >= 2")
    return vector - float(np.mean(vector))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return math.nan
    return float(np.dot(left, right) / denominator)


def _rank_for_fraction(ratios: np.ndarray, threshold: float) -> int:
    cumulative = np.cumsum(np.asarray(ratios, dtype=np.float64))
    indices = np.flatnonzero(cumulative >= float(threshold))
    return int(indices[0] + 1) if indices.size else int(ratios.size)


def _linear_trajectory_r2(centroids: np.ndarray) -> float:
    centroids = np.asarray(centroids, dtype=np.float64)
    counts = np.arange(1, centroids.shape[0] + 1, dtype=np.float64)
    design = np.column_stack([np.ones_like(counts), counts])
    coefficients = np.linalg.lstsq(design, centroids, rcond=None)[0]
    fitted = design @ coefficients
    centered = centroids - np.mean(centroids, axis=0, keepdims=True)
    denominator = float(np.sum(centered**2))
    if denominator <= 1e-12:
        return math.nan
    return float(1.0 - np.sum((centroids - fitted) ** 2) / denominator)


def _nearest_centroid_accuracy(
    train_panel: np.ndarray,
    test_panel: np.ndarray,
    rank: int,
) -> float:
    centroids = np.mean(train_panel, axis=0)
    centered = centroids - np.mean(centroids, axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    used_rank = min(int(rank), int(vt.shape[0]))
    basis = vt[:used_rank].T
    projected_centroids = centered @ basis
    test_centered = test_panel - np.mean(test_panel, axis=0, keepdims=True)
    projected_test = test_centered @ basis
    distances = np.sum(
        (projected_test[:, None, :] - projected_centroids[None, :, :]) ** 2,
        axis=-1,
    )
    predictions = np.argmin(distances, axis=1)
    labels = np.arange(test_panel.shape[0])
    return float(np.mean(predictions == labels))


@dataclass(frozen=True)
class NaturalReference:
    layer: int
    seeds: tuple[int, ...]
    panel: np.ndarray
    centroids: np.ndarray
    tangents: np.ndarray
    basis: np.ndarray
    explained_variance_ratio: np.ndarray
    rank90: int
    rank95: int
    summary: dict[str, Any]


def build_natural_reference(
    rows: Sequence[Mapping[str, Any]],
    *,
    layer: int,
    required_counts: Sequence[int] = tuple(range(1, 11)),
    final_count: int = 10,
) -> NaturalReference:
    """Build a context-centered natural count trajectory for one layer."""

    counts = tuple(int(value) for value in required_counts)
    grouped: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for row in rows:
        if int(row.get("layer", -1)) != int(layer):
            continue
        if int(row.get("gold_total", -1)) != int(final_count):
            continue
        count = int(row["raw_item_ordinal"])
        if count not in counts:
            continue
        grouped[(int(row["seed"]), count)].append(gauge_scores(row["probe_scores"]))

    candidate_seeds = sorted({seed for seed, _count in grouped})
    complete_seeds = tuple(
        seed
        for seed in candidate_seeds
        if all((seed, count) in grouped for count in counts)
    )
    if len(complete_seeds) < 3:
        raise ValueError(
            f"Layer {layer} has only {len(complete_seeds)} complete natural seeds"
        )

    panel = np.stack(
        [
            np.stack(
                [np.mean(grouped[(seed, count)], axis=0) for count in counts],
                axis=0,
            )
            for seed in complete_seeds
        ],
        axis=0,
    )
    # Context/seed centering removes prompt-specific score offsets while retaining
    # all within-trace count transitions.
    panel = panel - np.mean(panel, axis=1, keepdims=True)
    centroids = np.mean(panel, axis=0)
    centered = centroids - np.mean(centroids, axis=0, keepdims=True)
    _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values**2
    variance_ratio = variance / max(float(np.sum(variance)), 1e-30)
    tangents = np.diff(centroids, axis=0)

    sample_steps = np.diff(panel, axis=1)
    constant_step = np.mean(sample_steps, axis=(0, 1), keepdims=True)
    per_count_steps = np.mean(sample_steps, axis=0, keepdims=True)
    total_step_energy = float(np.sum(sample_steps**2))
    constant_residual = float(np.sum((sample_steps - constant_step) ** 2))
    count_specific_residual = float(np.sum((sample_steps - per_count_steps) ** 2))
    fixed_increment_r2 = (
        1.0 - constant_residual / total_step_energy
        if total_step_energy > 1e-12
        else math.nan
    )
    count_specific_gain = (
        1.0 - count_specific_residual / constant_residual
        if constant_residual > 1e-12
        else math.nan
    )

    leave_one_seed_out_cosines: list[float] = []
    relevant_cosines: list[float] = []
    for seed_index in range(panel.shape[0]):
        other = np.delete(sample_steps, seed_index, axis=0)
        other_means = np.mean(other, axis=0)
        for count_index in range(sample_steps.shape[1]):
            value = cosine(sample_steps[seed_index, count_index], other_means[count_index])
            if math.isfinite(value):
                leave_one_seed_out_cosines.append(value)
                # Counts 5->6 through 8->9 are used by the event-ledger endpoints.
                if 4 <= count_index <= 7:
                    relevant_cosines.append(value)

    loso_rank_accuracy: dict[str, float] = {}
    for rank in (1, 2, 3, 9):
        accuracies = []
        for seed_index in range(panel.shape[0]):
            train = np.delete(panel, seed_index, axis=0)
            test = panel[seed_index]
            accuracies.append(_nearest_centroid_accuracy(train, test, rank))
        loso_rank_accuracy[f"rank_{rank}"] = float(np.mean(accuracies))

    adjacent_centroid_cosines = [
        cosine(tangents[index], tangents[index + 1])
        for index in range(tangents.shape[0] - 1)
    ]
    summary = {
        "layer": int(layer),
        "complete_seed_count": len(complete_seeds),
        "complete_seeds": list(complete_seeds),
        "counts": list(counts),
        "probe_score_dimension_after_gauge": int(panel.shape[-1] - 1),
        "between_count_explained_variance_ratio": [
            float(value) for value in variance_ratio
        ],
        "rank90": _rank_for_fraction(variance_ratio, 0.90),
        "rank95": _rank_for_fraction(variance_ratio, 0.95),
        "pc1_variance_fraction": float(variance_ratio[0]),
        "linear_count_trajectory_r2": _linear_trajectory_r2(centroids),
        "fixed_increment_vector_r2_over_sample_steps": fixed_increment_r2,
        "gain_from_count_specific_over_fixed_increment_steps": count_specific_gain,
        "mean_adjacent_centroid_tangent_cosine": float(
            np.nanmean(adjacent_centroid_cosines)
        ),
        "mean_loso_same_count_tangent_cosine": float(
            np.nanmean(leave_one_seed_out_cosines)
        ),
        "mean_loso_relevant_count_tangent_cosine_c5_to_c9": float(
            np.nanmean(relevant_cosines)
        ),
        "loso_nearest_centroid_exact_accuracy": loso_rank_accuracy,
    }
    return NaturalReference(
        layer=int(layer),
        seeds=complete_seeds,
        panel=panel,
        centroids=centroids,
        tangents=tangents,
        basis=vt.T,
        explained_variance_ratio=variance_ratio,
        rank90=int(summary["rank90"]),
        rank95=int(summary["rank95"]),
        summary=summary,
    )


def transition_metrics(
    delta: np.ndarray,
    reference: np.ndarray,
    natural: NaturalReference,
) -> dict[str, float]:
    delta = np.asarray(delta, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    reference_norm_sq = float(np.dot(reference, reference))
    delta_norm = float(np.linalg.norm(delta))
    reference_norm = float(np.linalg.norm(reference))
    if reference_norm_sq <= 1e-12 or delta_norm <= 1e-12:
        return {
            "same_tangent_cosine": math.nan,
            "reference_scale": math.nan,
            "orthogonal_over_reference": math.nan,
            "delta_over_reference_norm": math.nan,
            "rank1_energy_fraction": math.nan,
            "rank3_energy_fraction": math.nan,
            "rank90_energy_fraction": math.nan,
        }
    scale = float(np.dot(delta, reference) / reference_norm_sq)
    orthogonal = delta - scale * reference

    def energy_fraction(rank: int) -> float:
        used = min(int(rank), int(natural.basis.shape[1]))
        basis = natural.basis[:, :used]
        projected = basis @ (basis.T @ delta)
        return float(np.dot(projected, projected) / np.dot(delta, delta))

    return {
        "same_tangent_cosine": cosine(delta, reference),
        "reference_scale": scale,
        "orthogonal_over_reference": float(
            np.linalg.norm(orthogonal) / reference_norm
        ),
        "delta_over_reference_norm": float(delta_norm / reference_norm),
        "rank1_energy_fraction": energy_fraction(1),
        "rank3_energy_fraction": energy_fraction(3),
        "rank90_energy_fraction": energy_fraction(natural.rank90),
    }


def vector_match_metrics(
    observed: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    """Compare an intervention displacement with its exact textual counterpart."""

    observed = np.asarray(observed, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    observed_norm = float(np.linalg.norm(observed))
    target_norm_sq = float(np.dot(target, target))
    target_norm = float(np.sqrt(target_norm_sq))
    if observed_norm <= 1e-12 or target_norm_sq <= 1e-12:
        return {
            "matched_textual_cosine": math.nan,
            "matched_textual_scale": 0.0 if target_norm_sq > 1e-12 else math.nan,
            "matched_textual_orthogonal_over_target": (
                0.0 if target_norm_sq > 1e-12 else math.nan
            ),
            "observed_over_textual_norm": (
                0.0 if target_norm_sq > 1e-12 else math.nan
            ),
        }
    scale = float(np.dot(observed, target) / target_norm_sq)
    orthogonal = observed - scale * target
    return {
        "matched_textual_cosine": cosine(observed, target),
        "matched_textual_scale": scale,
        "matched_textual_orthogonal_over_target": float(
            np.linalg.norm(orthogonal) / target_norm
        ),
        "observed_over_textual_norm": float(observed_norm / target_norm),
    }


def _bootstrap_seed_mean(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    draws: int = 10_000,
    seed: int = 20260826,
) -> dict[str, Any]:
    by_seed: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[metric])
        if math.isfinite(value):
            by_seed[int(row["seed"])].append(value)
    seed_values = np.asarray(
        [np.mean(values) for _seed, values in sorted(by_seed.items())],
        dtype=np.float64,
    )
    if seed_values.size == 0:
        return {"mean": math.nan, "ci95": [math.nan, math.nan], "seed_count": 0}
    rng = np.random.default_rng(int(seed))
    sampled = rng.choice(seed_values, size=(int(draws), seed_values.size), replace=True)
    bootstrap_means = np.mean(sampled, axis=1)
    return {
        "mean": float(np.mean(seed_values)),
        "ci95": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
        "seed_count": int(seed_values.size),
        "seed_values": [float(value) for value in seed_values],
    }


def summarize_edge_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = (
        "same_tangent_cosine",
        "reference_scale",
        "orthogonal_over_reference",
        "delta_over_reference_norm",
        "rank1_energy_fraction",
        "rank3_energy_fraction",
        "rank90_energy_fraction",
    )
    return {
        "edge_count": len(rows),
        "seed_count": len({int(row["seed"]) for row in rows}),
        "metrics": {metric: _bootstrap_seed_mean(rows, metric) for metric in metrics},
    }


def _bits(value: str | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, str):
        cleaned = value.removeprefix("markers_")
        result = tuple(int(character) for character in cleaned)
    else:
        result = tuple(int(character) for character in value)
    if len(result) != 3 or any(character not in (0, 1) for character in result):
        raise ValueError(f"Invalid three-marker bit pattern: {value!r}")
    return result  # type: ignore[return-value]


def _cube_edges() -> Iterable[tuple[tuple[int, int, int], tuple[int, int, int], int]]:
    for source in itertools.product((0, 1), repeat=3):
        for slot in range(3):
            if source[slot] != 0:
                continue
            target = list(source)
            target[slot] = 1
            yield source, tuple(target), slot  # type: ignore[arg-type]


def analyze_textual_factorial(
    rows: Sequence[Mapping[str, Any]],
    natural_by_layer: Mapping[int, NaturalReference],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    edge_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    selected = [row for row in rows if row.get("condition") == "textual_factorial"]
    for layer, natural in natural_by_layer.items():
        for landmark in TARGET_LANDMARKS:
            for seed in sorted({int(row["seed"]) for row in selected}):
                cells = {
                    _bits(row["marker_bits"]): row
                    for row in selected
                    if int(row["layer"] if "layer" in row else row["read_layer"])
                    == int(layer)
                    and str(row["landmark"]) == landmark
                    and int(row["seed"]) == seed
                }
                if len(cells) != 8:
                    raise ValueError(
                        f"Textual cube is incomplete for L{layer} {landmark} seed {seed}"
                    )
                for source_bits, target_bits, slot in _cube_edges():
                    source = cells[source_bits]
                    target = cells[target_bits]
                    source_count = int(source["expected_count"])
                    target_count = int(target["expected_count"])
                    if target_count != source_count + 1:
                        continue
                    delta = gauge_scores(target["probe_scores"]) - gauge_scores(
                        source["probe_scores"]
                    )
                    reference = natural.tangents[source_count - 1]
                    edge_rows.append(
                        {
                            "source": "textual_factorial",
                            "layer": int(layer),
                            "landmark": landmark,
                            "seed": seed,
                            "slot": int(slot),
                            "source_bits": "".join(map(str, source_bits)),
                            "target_bits": "".join(map(str, target_bits)),
                            "source_count": source_count,
                            "target_count": target_count,
                            **transition_metrics(delta, reference, natural),
                        }
                    )

                source = cells[(0, 0, 0)]
                target = cells[(1, 1, 1)]
                source_count = int(source["expected_count"])
                target_count = int(target["expected_count"])
                delta = gauge_scores(target["probe_scores"]) - gauge_scores(
                    source["probe_scores"]
                )
                reference = (
                    natural.centroids[target_count - 1]
                    - natural.centroids[source_count - 1]
                )
                endpoint_rows.append(
                    {
                        "source": "textual_factorial",
                        "layer": int(layer),
                        "landmark": landmark,
                        "seed": seed,
                        "source_count": source_count,
                        "target_count": target_count,
                        **transition_metrics(delta, reference, natural),
                    }
                )
    return edge_rows, endpoint_rows


def analyze_cache_subsets(
    rows: Sequence[Mapping[str, Any]],
    natural_by_layer: Mapping[int, NaturalReference],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    edge_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    selected = [row for row in rows if row.get("condition") == "cache_subset_splice"]
    families = sorted({str(row["family"]) for row in selected})
    for family in families:
        family_rows = [row for row in selected if str(row["family"]) == family]
        for layer, natural in natural_by_layer.items():
            for landmark in TARGET_LANDMARKS:
                base_count = 5 if landmark == "target_marker" else 6
                for seed in sorted({int(row["seed"]) for row in family_rows}):
                    cells = {
                        _bits(str(row["subset_id"])): row
                        for row in family_rows
                        if int(row["read_layer"]) == int(layer)
                        and str(row["landmark"]) == landmark
                        and int(row["seed"]) == seed
                    }
                    if len(cells) != 8:
                        raise ValueError(
                            f"Cache cube is incomplete for {family} L{layer} "
                            f"{landmark} seed {seed}"
                        )
                    for source_bits, target_bits, slot in _cube_edges():
                        source_count = base_count + sum(source_bits)
                        target_count = source_count + 1
                        delta = gauge_scores(cells[target_bits]["probe_scores"]) - gauge_scores(
                            cells[source_bits]["probe_scores"]
                        )
                        edge_rows.append(
                            {
                                "source": family,
                                "layer": int(layer),
                                "landmark": landmark,
                                "seed": seed,
                                "slot": int(slot),
                                "source_bits": "".join(map(str, source_bits)),
                                "target_bits": "".join(map(str, target_bits)),
                                "source_count": source_count,
                                "target_count": target_count,
                                **transition_metrics(
                                    delta, natural.tangents[source_count - 1], natural
                                ),
                            }
                        )
                    delta = gauge_scores(cells[(1, 1, 1)]["probe_scores"]) - gauge_scores(
                        cells[(0, 0, 0)]["probe_scores"]
                    )
                    endpoint_rows.append(
                        {
                            "source": family,
                            "layer": int(layer),
                            "landmark": landmark,
                            "seed": seed,
                            "source_count": base_count,
                            "target_count": base_count + 3,
                            **transition_metrics(
                                delta,
                                natural.centroids[base_count + 2]
                                - natural.centroids[base_count - 1],
                                natural,
                            ),
                        }
                    )
    return edge_rows, endpoint_rows


def analyze_cache_vs_textual(
    rows: Sequence[Mapping[str, Any]],
    *,
    layers: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Match every cache-subset displacement to the same textual cube edge."""

    textual = [row for row in rows if row.get("condition") == "textual_factorial"]
    cache = [row for row in rows if row.get("condition") == "cache_subset_splice"]
    families = sorted({str(row["family"]) for row in cache})
    edge_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    for layer in layers:
        for landmark in TARGET_LANDMARKS:
            for seed in sorted({int(row["seed"]) for row in textual}):
                textual_cells = {
                    _bits(row["marker_bits"]): row
                    for row in textual
                    if int(row["read_layer"]) == int(layer)
                    and str(row["landmark"]) == landmark
                    and int(row["seed"]) == seed
                }
                if len(textual_cells) != 8:
                    raise ValueError(
                        f"Missing matched textual cube for L{layer} {landmark} seed {seed}"
                    )
                for family in families:
                    cache_cells = {
                        _bits(str(row["subset_id"])): row
                        for row in cache
                        if str(row["family"]) == family
                        and int(row["read_layer"]) == int(layer)
                        and str(row["landmark"]) == landmark
                        and int(row["seed"]) == seed
                    }
                    if len(cache_cells) != 8:
                        raise ValueError(
                            f"Missing matched cache cube for {family} L{layer} "
                            f"{landmark} seed {seed}"
                        )
                    for source_bits, target_bits, slot in _cube_edges():
                        textual_delta = gauge_scores(
                            textual_cells[target_bits]["probe_scores"]
                        ) - gauge_scores(textual_cells[source_bits]["probe_scores"])
                        cache_delta = gauge_scores(
                            cache_cells[target_bits]["probe_scores"]
                        ) - gauge_scores(cache_cells[source_bits]["probe_scores"])
                        edge_rows.append(
                            {
                                "source": family,
                                "layer": int(layer),
                                "landmark": landmark,
                                "seed": seed,
                                "slot": int(slot),
                                "source_bits": "".join(map(str, source_bits)),
                                "target_bits": "".join(map(str, target_bits)),
                                **vector_match_metrics(cache_delta, textual_delta),
                            }
                        )
                    textual_endpoint = gauge_scores(
                        textual_cells[(1, 1, 1)]["probe_scores"]
                    ) - gauge_scores(textual_cells[(0, 0, 0)]["probe_scores"])
                    cache_endpoint = gauge_scores(
                        cache_cells[(1, 1, 1)]["probe_scores"]
                    ) - gauge_scores(cache_cells[(0, 0, 0)]["probe_scores"])
                    endpoint_rows.append(
                        {
                            "source": family,
                            "layer": int(layer),
                            "landmark": landmark,
                            "seed": seed,
                            **vector_match_metrics(cache_endpoint, textual_endpoint),
                        }
                    )
    return edge_rows, endpoint_rows


def _group_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source"]), int(row["layer"]), str(row["landmark"]))].append(row)
    return [
        {
            "source": source,
            "layer": layer,
            "landmark": landmark,
            **summarize_edge_rows(group),
        }
        for (source, layer, landmark), group in sorted(grouped.items())
    ]


def _group_match_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source"]), int(row["layer"]), str(row["landmark"]))].append(row)
    metrics = (
        "matched_textual_cosine",
        "matched_textual_scale",
        "matched_textual_orthogonal_over_target",
        "observed_over_textual_norm",
    )
    return [
        {
            "source": source,
            "layer": layer,
            "landmark": landmark,
            "edge_count": len(group),
            "seed_count": len({int(row["seed"]) for row in group}),
            "metrics": {
                metric: _bootstrap_seed_mean(group, metric) for metric in metrics
            },
        }
        for (source, layer, landmark), group in sorted(grouped.items())
    ]


def _plot_natural_manifold(
    natural_by_layer: Mapping[int, NaturalReference], output: Path
) -> None:
    layers = sorted(natural_by_layer)
    figure, axes = plt.subplots(1, len(layers), figsize=(5.2 * len(layers), 4.6))
    if len(layers) == 1:
        axes = [axes]
    for axis, layer in zip(axes, layers):
        natural = natural_by_layer[layer]
        centered = natural.centroids - np.mean(natural.centroids, axis=0, keepdims=True)
        coordinates = centered @ natural.basis[:, :2]
        axis.plot(coordinates[:, 0], coordinates[:, 1], "-o", color="#275d8c")
        for count, (x, y) in enumerate(coordinates, start=1):
            axis.annotate(str(count), (x, y), xytext=(5, 4), textcoords="offset points")
        axis.axhline(0.0, color="#bbbbbb", linewidth=0.7)
        axis.axvline(0.0, color="#bbbbbb", linewidth=0.7)
        axis.set_title(
            f"L{layer}: PC1={natural.summary['pc1_variance_fraction']:.2f}, "
            f"rank90={natural.rank90}"
        )
        axis.set_xlabel("count-trajectory PC1")
        axis.set_ylabel("count-trajectory PC2")
        axis.set_aspect("equal", adjustable="datalim")
    figure.suptitle("Natural N=10 count trajectory in frozen probe-score space")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_alignment(grouped: Sequence[Mapping[str, Any]], output: Path) -> None:
    selected = [
        row
        for row in grouped
        if int(row["layer"]) == 24 and str(row["landmark"]) == "target_boundary"
    ]
    labels = [str(row["source"]) for row in selected]
    cosine_means = [
        float(row["metrics"]["same_tangent_cosine"]["mean"]) for row in selected
    ]
    cosine_errors = np.asarray(
        [
            [
                mean - float(row["metrics"]["same_tangent_cosine"]["ci95"][0]),
                float(row["metrics"]["same_tangent_cosine"]["ci95"][1]) - mean,
            ]
            for row, mean in zip(selected, cosine_means)
        ]
    ).T
    rank3 = [
        float(row["metrics"]["rank3_energy_fraction"]["mean"]) for row in selected
    ]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 7.8), sharex=True)
    axes[0].bar(x, cosine_means, yerr=cosine_errors, color="#3978a8", capsize=3)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("cosine with natural c→c+1 tangent")
    axes[0].set_title("L24 target-boundary one-marker causal transitions")
    axes[1].bar(x, rank3, color="#d07a30")
    axes[1].set_ylabel("energy in top-3 natural count PCs")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _markdown_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Anthropic-inspired count-manifold reanalysis",
        "",
        "This is a retrospective, lightweight analysis in the full frozen-probe score space. "
        "It is not a raw-residual manifold analysis and not a transcoder attribution graph.",
        "",
        "## Natural count trajectory",
        "",
        "| Layer | PC1 variance | rank90 | linear-trajectory R² | fixed +1 vector R² | relevant tangent reliability |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["natural_references"]:
        lines.append(
            "| L{layer} | {pc1:.3f} | {rank90} | {linear:.3f} | {fixed:.3f} | {reliability:.3f} |".format(
                layer=row["layer"],
                pc1=row["pc1_variance_fraction"],
                rank90=row["rank90"],
                linear=row["linear_count_trajectory_r2"],
                fixed=row["fixed_increment_vector_r2_over_sample_steps"],
                reliability=row[
                    "mean_loso_relevant_count_tangent_cosine_c5_to_c9"
                ],
            )
        )

    lines.extend(
        [
            "",
            "Interpretation: high decodability is compatible with a multidimensional trajectory. "
            "A scalar-register geometry would predict PC1≈1, a nearly linear trajectory, and a "
            "stable fixed increment vector.",
            "",
            "## Causal transition alignment at L24 target boundary",
            "",
            "| Intervention | one-step cosine [95% CI] | scale along natural tangent | top-3 count-PC energy | 000→111 cosine |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    endpoints = {
        (row["source"], row["layer"], row["landmark"]): row
        for row in summary["endpoint_group_summaries"]
    }
    for row in summary["edge_group_summaries"]:
        if int(row["layer"]) != 24 or str(row["landmark"]) != "target_boundary":
            continue
        cosine_metric = row["metrics"]["same_tangent_cosine"]
        scale_metric = row["metrics"]["reference_scale"]
        rank3_metric = row["metrics"]["rank3_energy_fraction"]
        endpoint = endpoints[(row["source"], row["layer"], row["landmark"])]
        endpoint_cosine = endpoint["metrics"]["same_tangent_cosine"]["mean"]
        lines.append(
            "| {source} | {mean:.3f} [{low:.3f}, {high:.3f}] | {scale:.3f} | {rank3:.3f} | {endpoint:.3f} |".format(
                source=row["source"],
                mean=cosine_metric["mean"],
                low=cosine_metric["ci95"][0],
                high=cosine_metric["ci95"][1],
                scale=scale_metric["mean"],
                rank3=rank3_metric["mean"],
                endpoint=endpoint_cosine,
            )
        )

    lines.extend(
        [
            "",
            "## Cache displacement versus its exactly matched textual displacement",
            "",
            "This comparison does not depend on the cross-seed stability of the natural "
            "count tangent: each cache edge is compared with the same seed, marker slot, "
            "Hamming-cube edge, layer, and landmark under real textual toggling.",
            "",
            "| Cache intervention (L24 target boundary) | matched cosine [95% CI] | textual-direction scale | orthogonal / textual | 000→111 matched cosine |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    matched_endpoints = {
        (row["source"], row["layer"], row["landmark"]): row
        for row in summary["matched_endpoint_group_summaries"]
    }
    for row in summary["matched_edge_group_summaries"]:
        if int(row["layer"]) != 24 or str(row["landmark"]) != "target_boundary":
            continue
        cosine_metric = row["metrics"]["matched_textual_cosine"]
        scale_metric = row["metrics"]["matched_textual_scale"]
        orthogonal_metric = row["metrics"]["matched_textual_orthogonal_over_target"]
        endpoint = matched_endpoints[(row["source"], row["layer"], row["landmark"])]
        endpoint_cosine = endpoint["metrics"]["matched_textual_cosine"]["mean"]
        lines.append(
            "| {source} | {mean:.3f} [{low:.3f}, {high:.3f}] | {scale:.3f} | {orthogonal:.3f} | {endpoint:.3f} |".format(
                source=row["source"],
                mean=cosine_metric["mean"],
                low=cosine_metric["ci95"][0],
                high=cosine_metric["ci95"][1],
                scale=scale_metric["mean"],
                orthogonal=orthogonal_metric["mean"],
                endpoint=endpoint_cosine,
            )
        )

    lines.extend(
        [
            "",
            "## Scope and limitations",
            "",
            "- Natural references use held-out clean N=10 traces, the same frozen probe, "
            "within-seed centering, and within-seed transition vectors.",
            "- Probe-score geometry only covers the probe row span. A weak alignment cannot "
            "prove that the raw residual trajectory or K/V state lacks count information.",
            "- Cache interventions are observed through their downstream residual effect; this "
            "does not directly decompose QK versus OV computations head by head.",
            "- The three-event factorial is out of distribution and this analysis was designed "
            "after seeing earlier results, so all results are exploratory.",
            "",
        ]
    )
    return "\n".join(lines)


def _json_safe(value: Any) -> Any:
    """Convert numpy scalars and non-finite floats to strict-JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def analyze(
    natural_rows: Sequence[Mapping[str, Any]],
    ledger_rows: Sequence[Mapping[str, Any]],
    *,
    layers: Sequence[int] = DEFAULT_LAYERS,
) -> tuple[dict[str, Any], dict[int, NaturalReference]]:
    natural_by_layer = {
        int(layer): build_natural_reference(natural_rows, layer=int(layer))
        for layer in layers
    }
    textual_edges, textual_endpoints = analyze_textual_factorial(
        ledger_rows, natural_by_layer
    )
    cache_edges, cache_endpoints = analyze_cache_subsets(ledger_rows, natural_by_layer)
    matched_edges, matched_endpoints = analyze_cache_vs_textual(
        ledger_rows, layers=tuple(sorted(natural_by_layer))
    )
    edge_rows = textual_edges + cache_edges
    endpoint_rows = textual_endpoints + cache_endpoints
    summary = {
        "schema_version": SCHEMA,
        "analysis_status": "exploratory_retrospective",
        "representation_view": "full_10way_frozen_probe_scores_after_common_offset_removal",
        "natural_references": [
            natural_by_layer[layer].summary for layer in sorted(natural_by_layer)
        ],
        "edge_group_summaries": _group_summaries(edge_rows),
        "endpoint_group_summaries": _group_summaries(endpoint_rows),
        "matched_edge_group_summaries": _group_match_summaries(matched_edges),
        "matched_endpoint_group_summaries": _group_match_summaries(matched_endpoints),
        "edge_rows": edge_rows,
        "endpoint_rows": endpoint_rows,
        "matched_edge_rows": matched_edges,
        "matched_endpoint_rows": matched_endpoints,
        "limitations": [
            "probe row-span only; not raw residual geometry",
            "K/V is observed only through downstream residual probe scores",
            "no sparse-feature/transcoder attribution graph",
            "OOD three-event factorial",
            "retrospective analysis",
        ],
    }
    return summary, natural_by_layer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural-clean", type=Path, required=True)
    parser.add_argument("--event-ledger", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    natural_rows = read_jsonl(args.natural_clean)
    ledger_rows = read_jsonl(args.event_ledger)
    summary, natural_by_layer = analyze(
        natural_rows, ledger_rows, layers=tuple(sorted(set(args.layers)))
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    _plot_natural_manifold(natural_by_layer, args.output_dir / "natural_manifold_pca.png")
    _plot_alignment(
        summary["edge_group_summaries"], args.output_dir / "transition_alignment.png"
    )
    (args.output_dir / "REPORT.md").write_text(
        _markdown_report(summary), encoding="utf-8"
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
