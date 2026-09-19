from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .causal_generation import (
    _base_metadata,
    _baseline_metadata,
    _optional_int,
    _stable_seed,
    _validate_baseline_label,
    intervention_outcome,
    transport_fields,
)
from .modeling import (
    DecoderAdapter,
    capture_post_block_states,
    generate_with_residual_interventions,
)
from .prompts import PromptEncoding

GEOMETRIC_METHODS = (
    "centroid_transplant",
    "centroid_delta",
    "chord",
    "polyline",
)
STEERING_CONDITIONS = ("geometric", "orthogonal_norm_matched_random")


@dataclass(frozen=True)
class CountCentroidBundle:
    """Discovery-only answer-query count centroids.

    ``centroids`` has shape ``[variant, layer, count, hidden]`` and is stored
    in fp32.  The bundle deliberately contains no confirmation examples.
    """

    variants: tuple[str, ...]
    layers: tuple[int, ...]
    counts: tuple[int, ...]
    centroids: np.ndarray
    sample_counts: np.ndarray
    discovery_seeds: tuple[int, ...]

    def validate(self) -> None:
        expected = (
            len(self.variants),
            len(self.layers),
            len(self.counts),
        )
        if self.centroids.ndim != 4 or self.centroids.shape[:3] != expected:
            raise ValueError(
                f"Centroid shape {self.centroids.shape} does not begin with {expected}"
            )
        if self.sample_counts.shape != (len(self.variants), len(self.counts)):
            raise ValueError("Centroid sample-count shape is inconsistent")
        if not np.isfinite(self.centroids).all():
            raise ValueError("Centroid bundle contains non-finite values")
        if np.any(self.sample_counts <= 0):
            raise ValueError("Every centroid cell must contain discovery examples")
        if len(set(self.variants)) != len(self.variants):
            raise ValueError("Centroid variants are not unique")
        if tuple(sorted(set(self.layers))) != self.layers:
            raise ValueError("Centroid layers must be unique and sorted")
        if tuple(sorted(set(self.counts))) != self.counts:
            raise ValueError("Centroid counts must be unique and sorted")

    @property
    def hidden_size(self) -> int:
        return int(self.centroids.shape[-1])

    def state(self, variant: str, layer: int, count: int) -> torch.Tensor:
        try:
            variant_index = self.variants.index(str(variant))
            layer_index = self.layers.index(int(layer))
            count_index = self.counts.index(int(count))
        except ValueError as exc:
            raise KeyError(
                f"Missing centroid variant={variant} layer={layer} count={count}"
            ) from exc
        return torch.from_numpy(
            np.asarray(
                self.centroids[variant_index, layer_index, count_index],
                dtype=np.float32,
            ).copy()
        )


@dataclass(frozen=True)
class LayerSetSteeringPlan:
    """One registered answer-query centroid-delta intervention.

    A single-layer plan contains one post-block layer.  A multi-layer plan
    contains two or more layers and applies each layer's own discovery-fit
    count-centroid displacement during the same prompt-prefill forward pass.
    ``alpha`` scales every layer-specific displacement by the same registered
    dose; no PCA projection or hidden-dimension masking is performed.
    """

    layers: tuple[int, ...]
    alpha: float

    def validate(self, bundle: CountCentroidBundle) -> None:
        if not self.layers:
            raise ValueError("A layer-set steering plan must contain a layer")
        if tuple(sorted(set(int(layer) for layer in self.layers))) != self.layers:
            raise ValueError("Steering-plan layers must be unique and increasing")
        missing = sorted(set(self.layers) - set(bundle.layers))
        if missing:
            raise ValueError(f"Steering-plan layers missing from centroids: {missing}")
        if not 0.0 < float(self.alpha) <= 1.0:
            raise ValueError("Layer-set steering alpha must lie in (0, 1]")

    @property
    def protocol(self) -> str:
        return "single_layer" if len(self.layers) == 1 else "multi_layer"

    @property
    def label(self) -> str:
        return "+".join(str(int(layer)) for layer in self.layers)


def _save_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def capture_query_residual_shard(
    model: Any,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    layers: Sequence[int],
    path: str | Path,
    save_dtype: str = "float16",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Capture discovery answer-query post-block residuals in one shard."""

    if encoding.split != "discovery":
        raise ValueError("Steering centroids must be captured on discovery examples")
    dtype_map = {"float16": np.float16, "float32": np.float32}
    if save_dtype not in dtype_map:
        raise ValueError("Steering capture dtype must be float16 or float32")
    selected_layers = tuple(sorted({int(layer) for layer in layers}))
    if not selected_layers:
        raise ValueError("At least one steering layer is required")
    destination = Path(path)
    if destination.exists() and not overwrite:
        with np.load(destination, allow_pickle=False) as saved:
            observed_layers = tuple(int(value) for value in saved["layer_indices"])
            states = saved["query_states"]
        if (
            observed_layers != selected_layers
            or states.shape[0] != len(selected_layers)
        ):
            raise RuntimeError(f"Invalid existing steering capture: {destination}")
        return {
            "stimulus_id": encoding.stimulus_id,
            "design_variant": encoding.design_variant,
            "seed": int(encoding.seed),
            "split": encoding.split,
            "count": int(encoding.count),
            "rows": int(states.shape[0]),
            "hidden_size": int(states.shape[1]),
        }
    _logits, states = capture_post_block_states(
        model,
        adapter,
        encoding,
        [int(encoding.query_position)],
        layers=selected_layers,
    )
    query_states = np.stack(
        [states[layer][0].numpy() for layer in selected_layers], axis=0
    ).astype(dtype_map[save_dtype], copy=False)
    if query_states.ndim != 2 or not np.isfinite(query_states).all():
        raise RuntimeError("Invalid answer-query residual capture")
    _save_npz_atomic(
        destination,
        layer_indices=np.asarray(selected_layers, dtype=np.int64),
        query_states=query_states,
        query_position=np.asarray([encoding.query_position], dtype=np.int64),
        sequence_length=np.asarray([encoding.sequence_length], dtype=np.int64),
    )
    return {
        "stimulus_id": encoding.stimulus_id,
        "design_variant": encoding.design_variant,
        "seed": int(encoding.seed),
        "split": encoding.split,
        "count": int(encoding.count),
        "rows": int(query_states.shape[0]),
        "hidden_size": int(query_states.shape[1]),
    }


def fit_count_centroids(
    index_rows: Sequence[Mapping[str, Any]],
    *,
    capture_root: str | Path,
    variants: Sequence[str],
    layers: Sequence[int],
    counts: Sequence[int],
    discovery_seeds: Sequence[int],
) -> CountCentroidBundle:
    """Fit count centroids after verifying the complete discovery grid."""

    variants = tuple(str(value) for value in variants)
    layers = tuple(sorted({int(value) for value in layers}))
    counts = tuple(sorted({int(value) for value in counts}))
    discovery_seeds = tuple(sorted({int(value) for value in discovery_seeds}))
    if not index_rows:
        raise ValueError("No steering discovery captures were indexed")
    expected_keys = {
        (variant, seed, count)
        for variant in variants
        for seed in discovery_seeds
        for count in counts
    }
    observed: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for row in index_rows:
        if str(row.get("split")) != "discovery":
            raise ValueError("Centroid index contains non-discovery rows")
        key = (
            str(row["design_variant"]),
            int(row["seed"]),
            int(row["count"]),
        )
        if key in observed:
            raise ValueError(f"Duplicate steering discovery capture: {key}")
        observed[key] = row
    missing = sorted(expected_keys - set(observed))
    extra = sorted(set(observed) - expected_keys)
    if missing or extra:
        raise ValueError(
            "Steering discovery grid mismatch: "
            f"missing={missing[:5]} ({len(missing)} total), "
            f"extra={extra[:5]} ({len(extra)} total)"
        )
    root = Path(capture_root)
    sums: dict[tuple[str, int, int], np.ndarray] = {}
    cell_sizes: dict[tuple[str, int], int] = {}
    hidden_size: int | None = None
    for key in sorted(expected_keys):
        row = observed[key]
        path = root / str(row["shard_path"])
        with np.load(path, allow_pickle=False) as saved:
            observed_layers = tuple(int(value) for value in saved["layer_indices"])
            states = np.asarray(saved["query_states"], dtype=np.float32)
        if observed_layers != layers:
            raise ValueError(f"Steering layer mismatch in {path}")
        if states.ndim != 2 or states.shape[0] != len(layers):
            raise ValueError(f"Invalid steering state shape in {path}: {states.shape}")
        if hidden_size is None:
            hidden_size = int(states.shape[1])
        elif hidden_size != int(states.shape[1]):
            raise ValueError("Steering captures have inconsistent hidden sizes")
        if not np.isfinite(states).all():
            raise ValueError(f"Non-finite steering states in {path}")
        variant, _seed, count = key
        for layer_index, layer in enumerate(layers):
            cell = (variant, layer, count)
            sums.setdefault(cell, np.zeros(hidden_size, dtype=np.float64))
            sums[cell] += states[layer_index].astype(np.float64)
        count_cell = (variant, count)
        cell_sizes[count_cell] = cell_sizes.get(count_cell, 0) + 1
    assert hidden_size is not None
    centroid_array = np.empty(
        (len(variants), len(layers), len(counts), hidden_size), dtype=np.float32
    )
    sample_counts = np.empty((len(variants), len(counts)), dtype=np.int32)
    for variant_index, variant in enumerate(variants):
        for count_index, count in enumerate(counts):
            n = int(cell_sizes[(variant, count)])
            if n != len(discovery_seeds):
                raise ValueError(
                    f"Centroid {variant}/N{count} has {n} discovery seeds, "
                    f"expected {len(discovery_seeds)}"
                )
            sample_counts[variant_index, count_index] = n
            for layer_index, layer in enumerate(layers):
                centroid_array[variant_index, layer_index, count_index] = (
                    sums[(variant, layer, count)] / float(n)
                ).astype(np.float32)
    bundle = CountCentroidBundle(
        variants=variants,
        layers=layers,
        counts=counts,
        centroids=centroid_array,
        sample_counts=sample_counts,
        discovery_seeds=discovery_seeds,
    )
    bundle.validate()
    return bundle


def save_centroid_bundle(bundle: CountCentroidBundle, path: str | Path) -> Path:
    bundle.validate()
    destination = Path(path)
    _save_npz_atomic(
        destination,
        variants=np.asarray(bundle.variants, dtype=np.str_),
        layer_indices=np.asarray(bundle.layers, dtype=np.int64),
        counts=np.asarray(bundle.counts, dtype=np.int64),
        centroids=np.asarray(bundle.centroids, dtype=np.float32),
        sample_counts=np.asarray(bundle.sample_counts, dtype=np.int32),
        discovery_seeds=np.asarray(bundle.discovery_seeds, dtype=np.int64),
    )
    manifest = {
        "schema_version": "realistic_niah_v4_count_centroids_v1",
        "site": "answer_query_post_block_residual",
        "selection_split": "discovery",
        "variants": list(bundle.variants),
        "layers": list(bundle.layers),
        "counts": list(bundle.counts),
        "discovery_seeds": list(bundle.discovery_seeds),
        "hidden_size": bundle.hidden_size,
        "centroid_path": destination.name,
        "interpretation": (
            "Stimulus-count-conditioned discovery centroids; causal outcomes are "
            "evaluated only on held-out confirmation seeds with greedy generation."
        ),
    }
    manifest_path = destination.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def load_centroid_bundle(path: str | Path) -> CountCentroidBundle:
    with np.load(path, allow_pickle=False) as saved:
        bundle = CountCentroidBundle(
            variants=tuple(str(value) for value in saved["variants"].tolist()),
            layers=tuple(int(value) for value in saved["layer_indices"].tolist()),
            counts=tuple(int(value) for value in saved["counts"].tolist()),
            centroids=np.asarray(saved["centroids"], dtype=np.float32),
            sample_counts=np.asarray(saved["sample_counts"], dtype=np.int32),
            discovery_seeds=tuple(
                int(value) for value in saved["discovery_seeds"].tolist()
            ),
        )
    bundle.validate()
    return bundle


def chord_point(
    bundle: CountCentroidBundle,
    *,
    variant: str,
    layer: int,
    receiver_count: int,
    target_count: int,
    alpha: float,
) -> tuple[torch.Tensor, float]:
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Steering alpha must lie in [0, 1]")
    receiver = bundle.state(variant, layer, receiver_count)
    target = bundle.state(variant, layer, target_count)
    point = (1.0 - alpha) * receiver + alpha * target
    count_coordinate = (1.0 - alpha) * int(receiver_count) + alpha * int(target_count)
    return point, float(count_coordinate)


def polyline_point(
    bundle: CountCentroidBundle,
    *,
    variant: str,
    layer: int,
    receiver_count: int,
    target_count: int,
    alpha: float,
) -> tuple[torch.Tensor, float]:
    """Interpolate along adjacent centroids using normalized arc length."""

    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Steering alpha must lie in [0, 1]")
    receiver_count = int(receiver_count)
    target_count = int(target_count)
    if receiver_count == target_count:
        raise ValueError("Polyline steering requires distinct counts")
    step = 1 if target_count > receiver_count else -1
    ordered_counts = tuple(range(receiver_count, target_count + step, step))
    if any(count not in bundle.counts for count in ordered_counts):
        raise KeyError("Polyline path requires every intermediate count centroid")
    states = [bundle.state(variant, layer, count) for count in ordered_counts]
    lengths = torch.tensor(
        [
            float(torch.linalg.vector_norm(right - left))
            for left, right in pairwise(states)
        ],
        dtype=torch.float64,
    )
    total = float(lengths.sum())
    if not math.isfinite(total) or total <= 1e-12:
        return chord_point(
            bundle,
            variant=variant,
            layer=layer,
            receiver_count=receiver_count,
            target_count=target_count,
            alpha=alpha,
        )
    distance = alpha * total
    cumulative = 0.0
    for index, segment_length_tensor in enumerate(lengths):
        segment_length = float(segment_length_tensor)
        if distance <= cumulative + segment_length or index == len(lengths) - 1:
            local = min(max((distance - cumulative) / segment_length, 0.0), 1.0)
            point = (1.0 - local) * states[index] + local * states[index + 1]
            count_coordinate = (
                (1.0 - local) * ordered_counts[index]
                + local * ordered_counts[index + 1]
            )
            return point, float(count_coordinate)
        cumulative += segment_length
    raise AssertionError("Polyline interpolation failed to select a segment")


def centroid_geometry_tables(
    bundle: CountCentroidBundle,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return layer summaries and adjacent-step diagnostics."""

    bundle.validate()
    summary_rows: list[dict[str, Any]] = []
    adjacent_rows: list[dict[str, Any]] = []
    for variant in bundle.variants:
        for layer in bundle.layers:
            states = [bundle.state(variant, layer, count) for count in bundle.counts]
            deltas = [right - left for left, right in pairwise(states)]
            norms = np.asarray(
                [float(torch.linalg.vector_norm(delta)) for delta in deltas],
                dtype=float,
            )
            adjacent_cosines: list[float] = []
            for index, delta in enumerate(deltas):
                if index + 1 < len(deltas):
                    left_norm = float(torch.linalg.vector_norm(delta))
                    right_norm = float(torch.linalg.vector_norm(deltas[index + 1]))
                    cosine = (
                        float(torch.dot(delta, deltas[index + 1]))
                        / (left_norm * right_norm)
                        if left_norm > 1e-12 and right_norm > 1e-12
                        else math.nan
                    )
                    adjacent_cosines.append(cosine)
                adjacent_rows.append(
                    {
                        "design_variant": variant,
                        "layer": int(layer),
                        "lower_count": int(bundle.counts[index]),
                        "upper_count": int(bundle.counts[index + 1]),
                        "step_norm": float(norms[index]),
                    }
                )
            chord = states[-1] - states[0]
            chord_norm = float(torch.linalg.vector_norm(chord))
            path_length = float(norms.sum())
            if chord_norm > 1e-12:
                unit = chord / chord_norm
                coordinates = np.asarray(
                    [float(torch.dot(state - states[0], unit)) for state in states]
                )
                correlation = float(
                    np.corrcoef(
                        np.asarray(bundle.counts, dtype=float), coordinates
                    )[0, 1]
                )
                monotone_fraction = float(np.mean(np.diff(coordinates) > 0))
            else:
                correlation = math.nan
                monotone_fraction = math.nan
            summary_rows.append(
                {
                    "design_variant": variant,
                    "layer": int(layer),
                    "counts": len(bundle.counts),
                    "discovery_seeds_per_count": int(
                        bundle.sample_counts[bundle.variants.index(variant)].min()
                    ),
                    "mean_adjacent_step_norm": float(norms.mean()),
                    "sd_adjacent_step_norm": float(norms.std(ddof=1)),
                    "adjacent_step_cv": (
                        float(norms.std(ddof=1) / norms.mean())
                        if float(norms.mean()) > 1e-12
                        else math.nan
                    ),
                    "mean_successive_step_cosine": float(
                        np.nanmean(adjacent_cosines)
                    ),
                    "path_length": path_length,
                    "endpoint_chord_length": chord_norm,
                    "path_tortuosity": (
                        path_length / chord_norm if chord_norm > 1e-12 else math.nan
                    ),
                    "endpoint_projection_count_correlation": correlation,
                    "endpoint_projection_monotone_fraction": monotone_fraction,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(adjacent_rows)


def _orthogonal_norm_matched_delta(
    delta: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    norm = float(torch.linalg.vector_norm(delta))
    if norm <= 1e-12:
        return torch.zeros_like(delta)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    random = torch.randn(delta.shape, generator=generator, dtype=torch.float32)
    unit = delta.float() / norm
    random = random - torch.dot(random, unit) * unit
    random_norm = float(torch.linalg.vector_norm(random))
    if random_norm <= 1e-12:
        random = torch.roll(unit, shifts=1)
        random = random - torch.dot(random, unit) * unit
        random_norm = float(torch.linalg.vector_norm(random))
    if random_norm <= 1e-12:
        raise RuntimeError("Cannot form an orthogonal steering control")
    return random * (norm / random_norm)


def _planned_geometric_replacement(
    bundle: CountCentroidBundle,
    receiver_state: torch.Tensor,
    *,
    variant: str,
    layer: int,
    receiver_count: int,
    target_count: int,
    method: str,
    alpha: float,
) -> tuple[torch.Tensor, float]:
    receiver_centroid = bundle.state(variant, layer, receiver_count)
    target_centroid = bundle.state(variant, layer, target_count)
    if method == "centroid_transplant":
        if not math.isclose(float(alpha), 1.0):
            raise ValueError("Centroid transplant is defined only at alpha=1")
        return target_centroid, float(target_count)
    if method == "centroid_delta":
        if not math.isclose(float(alpha), 1.0):
            raise ValueError("Centroid delta is defined only at alpha=1")
        return receiver_state + target_centroid - receiver_centroid, float(target_count)
    if method == "chord":
        path_point, count_coordinate = chord_point(
            bundle,
            variant=variant,
            layer=layer,
            receiver_count=receiver_count,
            target_count=target_count,
            alpha=alpha,
        )
    elif method == "polyline":
        path_point, count_coordinate = polyline_point(
            bundle,
            variant=variant,
            layer=layer,
            receiver_count=receiver_count,
            target_count=target_count,
            alpha=alpha,
        )
    else:
        raise ValueError(f"Unknown geometric steering method: {method}")
    return receiver_state + path_point - receiver_centroid, count_coordinate


def run_generation_geometric_steering(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    encodings: Sequence[PromptEncoding],
    *,
    baseline_labels: Mapping[str, Mapping[str, Any]],
    centroids: CountCentroidBundle,
    count_pairs: Sequence[tuple[int, int]],
    layers: Sequence[int],
    methods: Sequence[str] = GEOMETRIC_METHODS,
    alphas: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    random_replicates: int = 1,
    valid_counts: Sequence[int] = tuple(range(1, 11)),
    max_new_tokens: int = 16,
) -> pd.DataFrame:
    """Steer held-out answer-query residuals and greedily generate answers."""

    methods = tuple(str(value) for value in methods)
    if not methods or any(method not in GEOMETRIC_METHODS for method in methods):
        raise ValueError(f"Invalid steering methods: {methods}")
    alphas = tuple(sorted({float(value) for value in alphas}))
    if not alphas or any(not 0.0 < alpha <= 1.0 for alpha in alphas):
        raise ValueError("Steering alphas must lie in (0, 1]")
    layers = tuple(sorted({int(value) for value in layers}))
    if not layers or any(layer not in centroids.layers for layer in layers):
        raise ValueError("Steering layers are missing from the centroid bundle")
    if int(random_replicates) < 0:
        raise ValueError("random_replicates must be nonnegative")
    by_key = {
        (item.design_variant, int(item.seed), int(item.count)): item
        for item in encodings
    }
    if len(by_key) != len(encodings):
        raise ValueError("Steering encodings are not unique by variant/seed/count")
    if any(item.split != "confirmation" for item in encodings):
        raise ValueError("Steering evaluation must use confirmation encodings")
    variants = sorted({item.design_variant for item in encodings})
    seeds = sorted({int(item.seed) for item in encodings})
    rows: list[dict[str, Any]] = []
    state_cache: dict[str, dict[int, torch.Tensor]] = {}
    for variant in variants:
        for seed in seeds:
            for receiver_count, target_count in count_pairs:
                receiver = by_key.get((variant, seed, int(receiver_count)))
                target = by_key.get((variant, seed, int(target_count)))
                if receiver is None or target is None:
                    raise KeyError(
                        f"Missing steering pair {variant} seed={seed} "
                        f"N={receiver_count}->{target_count}"
                    )
                receiver_label = baseline_labels[receiver.stimulus_id]
                target_label = baseline_labels[target.stimulus_id]
                _validate_baseline_label(receiver, receiver_label)
                _validate_baseline_label(target, target_label)
                if receiver.stimulus_id not in state_cache:
                    _logits, captured = capture_post_block_states(
                        model,
                        adapter,
                        receiver,
                        [int(receiver.query_position)],
                        layers=layers,
                    )
                    state_cache[receiver.stimulus_id] = {
                        layer: captured[layer][0] for layer in layers
                    }
                receiver_states = state_cache[receiver.stimulus_id]
                for layer in layers:
                    receiver_state = receiver_states[layer]
                    for method in methods:
                        method_alphas = (1.0,) if method in {
                            "centroid_transplant",
                            "centroid_delta",
                        } else alphas
                        for alpha in method_alphas:
                            replacement, count_coordinate = (
                                _planned_geometric_replacement(
                                    centroids,
                                    receiver_state,
                                    variant=variant,
                                    layer=layer,
                                    receiver_count=int(receiver_count),
                                    target_count=int(target_count),
                                    method=method,
                                    alpha=alpha,
                                )
                            )
                            geometric_delta = replacement - receiver_state
                            conditions: list[tuple[str, int, torch.Tensor]] = [
                                ("geometric", -1, replacement)
                            ]
                            for replicate in range(int(random_replicates)):
                                control_delta = _orthogonal_norm_matched_delta(
                                    geometric_delta,
                                    seed=_stable_seed(
                                        f"{variant}:{seed}:{receiver_count}:"
                                        f"{target_count}:{layer}:{method}:{alpha}:"
                                        f"random:{replicate}"
                                    ),
                                )
                                conditions.append(
                                    (
                                        "orthogonal_norm_matched_random",
                                        replicate,
                                        receiver_state + control_delta,
                                    )
                                )
                            for condition, replicate, condition_state in conditions:
                                completion = generate_with_residual_interventions(
                                    model,
                                    tokenizer,
                                    adapter,
                                    receiver,
                                    {
                                        int(layer): (
                                            (int(receiver.query_position),),
                                            condition_state.unsqueeze(0),
                                        )
                                    },
                                    max_new_tokens=max_new_tokens,
                                )
                                outcome = intervention_outcome(
                                    completion,
                                    receiver,
                                    receiver_label,
                                    valid_counts=valid_counts,
                                )
                                patched = _optional_int(
                                    outcome.get("patched_predicted_count")
                                )
                                baseline_prediction = _optional_int(
                                    receiver_label.get("parsed_count")
                                )
                                intended_shift = float(count_coordinate) - float(
                                    receiver_count
                                )
                                rows.append(
                                    {
                                        **_base_metadata(receiver),
                                        **_baseline_metadata(receiver_label),
                                        "receiver_stimulus_id": receiver.stimulus_id,
                                        "target_stimulus_id": target.stimulus_id,
                                        "receiver_count": int(receiver_count),
                                        "target_count": int(target_count),
                                        "target_baseline_outcome": str(
                                            target_label["outcome_group"]
                                        ),
                                        "target_baseline_predicted_count": (
                                            _optional_int(
                                                target_label.get("parsed_count")
                                            )
                                        ),
                                        "layer": int(layer),
                                        "site": "answer_query",
                                        "steering_method": method,
                                        "condition": condition,
                                        "random_replicate": int(replicate),
                                        "alpha": float(alpha),
                                        "target_direction": (
                                            "increase"
                                            if int(target_count) > int(receiver_count)
                                            else "decrease"
                                        ),
                                        "intended_path_count": float(count_coordinate),
                                        "intended_count_shift": intended_shift,
                                        "applied_delta_norm": float(
                                            torch.linalg.vector_norm(
                                                condition_state - receiver_state
                                            )
                                        ),
                                        "geometric_delta_norm": float(
                                            torch.linalg.vector_norm(geometric_delta)
                                        ),
                                        **outcome,
                                        **transport_fields(
                                            outcome,
                                            receiver_label=receiver_label,
                                            donor_label=target_label,
                                            receiver_count=int(receiver_count),
                                            donor_count=int(target_count),
                                        ),
                                        "nearest_path_count_hit": (
                                            bool(
                                                patched
                                                == math.floor(
                                                    count_coordinate + 0.5
                                                )
                                            )
                                            if patched is not None
                                            else math.nan
                                        ),
                                        "path_count_absolute_error": (
                                            abs(float(patched) - count_coordinate)
                                            if patched is not None
                                            else math.nan
                                        ),
                                        "moved_toward_path_count": (
                                            abs(float(patched) - count_coordinate)
                                            < abs(
                                                float(baseline_prediction)
                                                - count_coordinate
                                            )
                                            if patched is not None
                                            and baseline_prediction is not None
                                            else math.nan
                                        ),
                                        "intended_transport_fraction": (
                                            float(outcome["generated_count_shift"])
                                            / intended_shift
                                            if intended_shift != 0
                                            and np.isfinite(
                                                outcome["generated_count_shift"]
                                            )
                                            else math.nan
                                        ),
                                    }
                                )
                print(
                    "[v4 geometric steering] "
                    f"{variant} seed={seed} N={receiver_count}->{target_count}",
                    flush=True,
                )
    if not rows:
        raise ValueError("No geometric-steering rows were produced")
    return pd.DataFrame(rows)


def _normalize_layer_set_plans(
    plans: Sequence[LayerSetSteeringPlan],
    bundle: CountCentroidBundle,
) -> tuple[LayerSetSteeringPlan, ...]:
    normalized = tuple(
        LayerSetSteeringPlan(
            layers=tuple(int(layer) for layer in plan.layers),
            alpha=float(plan.alpha),
        )
        for plan in plans
    )
    if not normalized:
        raise ValueError("At least one layer-set steering plan is required")
    for plan in normalized:
        plan.validate(bundle)
    keys = [(plan.layers, float(plan.alpha)) for plan in normalized]
    if len(set(keys)) != len(keys):
        raise ValueError("Layer-set steering plans must be unique")
    return normalized


def _layer_set_centroid_delta_states(
    bundle: CountCentroidBundle,
    receiver_states: Mapping[int, torch.Tensor],
    *,
    variant: str,
    receiver_count: int,
    target_count: int,
    plan: LayerSetSteeringPlan,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Return full-width replacement states and their layer-specific deltas."""

    replacements: dict[int, torch.Tensor] = {}
    deltas: dict[int, torch.Tensor] = {}
    for layer in plan.layers:
        receiver_state = receiver_states[int(layer)].float()
        receiver_centroid = bundle.state(variant, int(layer), int(receiver_count))
        target_centroid = bundle.state(variant, int(layer), int(target_count))
        delta = float(plan.alpha) * (target_centroid - receiver_centroid)
        replacements[int(layer)] = receiver_state + delta
        deltas[int(layer)] = delta
    return replacements, deltas


def _combined_delta_norm(deltas: Mapping[int, torch.Tensor]) -> float:
    return float(
        math.sqrt(
            sum(float(torch.sum(delta.float() ** 2)) for delta in deltas.values())
        )
    )


def run_generation_layer_set_centroid_delta(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    encodings: Sequence[PromptEncoding],
    *,
    baseline_labels: Mapping[str, Mapping[str, Any]],
    centroids: CountCentroidBundle,
    count_pairs: Sequence[tuple[int, int]],
    plans: Sequence[LayerSetSteeringPlan],
    random_replicates: int = 1,
    valid_counts: Sequence[int] = tuple(range(1, 11)),
    max_new_tokens: int = 16,
) -> pd.DataFrame:
    """Run registered single- and multi-layer answer-query steering.

    For every selected layer ``l``, this applies

    ``h_l' = h_l + alpha * (mu_l,target - mu_l,receiver)``

    to the complete answer-query residual vector during prompt prefill.  A
    multi-layer plan supplies all replacements to one generation call, so the
    downstream state evolves under the cumulative interventions.  The matched
    random control is formed independently at every layer, is orthogonal to
    that layer's centroid delta, and has exactly the same per-layer norm.
    """

    centroids.validate()
    plans = _normalize_layer_set_plans(plans, centroids)
    if int(random_replicates) < 1:
        raise ValueError("Layer-set steering requires at least one random control")
    directed_pairs = tuple((int(left), int(right)) for left, right in count_pairs)
    if not directed_pairs or any(left == right for left, right in directed_pairs):
        raise ValueError("Layer-set steering count pairs must be directed and distinct")
    by_key = {
        (item.design_variant, int(item.seed), int(item.count)): item
        for item in encodings
    }
    if len(by_key) != len(encodings):
        raise ValueError("Steering encodings are not unique by variant/seed/count")
    splits = {str(item.split) for item in encodings}
    if len(splits) != 1 or next(iter(splits)) not in {"discovery", "confirmation"}:
        raise ValueError("Layer-set steering encodings must use one registered split")
    evaluation_split = next(iter(splits))
    variants = sorted({item.design_variant for item in encodings})
    seeds = sorted({int(item.seed) for item in encodings})
    capture_layers = tuple(sorted({layer for plan in plans for layer in plan.layers}))
    rows: list[dict[str, Any]] = []
    state_cache: dict[str, dict[int, torch.Tensor]] = {}
    for variant in variants:
        for seed in seeds:
            for receiver_count, target_count in directed_pairs:
                receiver = by_key.get((variant, seed, receiver_count))
                target = by_key.get((variant, seed, target_count))
                if receiver is None or target is None:
                    raise KeyError(
                        f"Missing layer-set pair {variant} seed={seed} "
                        f"N={receiver_count}->{target_count}"
                    )
                receiver_label = baseline_labels[receiver.stimulus_id]
                target_label = baseline_labels[target.stimulus_id]
                _validate_baseline_label(receiver, receiver_label)
                _validate_baseline_label(target, target_label)
                if receiver.stimulus_id not in state_cache:
                    _logits, captured = capture_post_block_states(
                        model,
                        adapter,
                        receiver,
                        [int(receiver.query_position)],
                        layers=capture_layers,
                    )
                    state_cache[receiver.stimulus_id] = {
                        layer: captured[layer][0] for layer in capture_layers
                    }
                receiver_states = state_cache[receiver.stimulus_id]
                for plan in plans:
                    geometric_states, geometric_deltas = (
                        _layer_set_centroid_delta_states(
                            centroids,
                            receiver_states,
                            variant=variant,
                            receiver_count=receiver_count,
                            target_count=target_count,
                            plan=plan,
                        )
                    )
                    conditions: list[
                        tuple[str, int, dict[int, torch.Tensor], dict[int, torch.Tensor]]
                    ] = [("geometric", -1, geometric_states, geometric_deltas)]
                    for replicate in range(int(random_replicates)):
                        random_deltas = {
                            layer: _orthogonal_norm_matched_delta(
                                delta,
                                seed=_stable_seed(
                                    f"layer-set:{variant}:{seed}:{receiver_count}:"
                                    f"{target_count}:{plan.label}:{plan.alpha}:"
                                    f"L{layer}:random:{replicate}"
                                ),
                            )
                            for layer, delta in geometric_deltas.items()
                        }
                        random_states = {
                            layer: receiver_states[layer].float() + delta
                            for layer, delta in random_deltas.items()
                        }
                        conditions.append(
                            (
                                "orthogonal_norm_matched_random",
                                int(replicate),
                                random_states,
                                random_deltas,
                            )
                        )
                    baseline_prediction = _optional_int(
                        receiver_label.get("parsed_count")
                    )
                    intended_shift = float(plan.alpha) * (
                        float(target_count) - float(receiver_count)
                    )
                    path_count = float(receiver_count) + intended_shift
                    direction_sign = 1 if target_count > receiver_count else -1
                    for condition, replicate, condition_states, condition_deltas in conditions:
                        completion = generate_with_residual_interventions(
                            model,
                            tokenizer,
                            adapter,
                            receiver,
                            {
                                int(layer): (
                                    (int(receiver.query_position),),
                                    state.unsqueeze(0),
                                )
                                for layer, state in condition_states.items()
                            },
                            max_new_tokens=max_new_tokens,
                        )
                        outcome = intervention_outcome(
                            completion,
                            receiver,
                            receiver_label,
                            valid_counts=valid_counts,
                        )
                        patched = _optional_int(
                            outcome.get("patched_predicted_count")
                        )
                        rows.append(
                            {
                                **_base_metadata(receiver),
                                **_baseline_metadata(receiver_label),
                                "evaluation_split": evaluation_split,
                                "receiver_stimulus_id": receiver.stimulus_id,
                                "target_stimulus_id": target.stimulus_id,
                                "receiver_count": receiver_count,
                                "target_count": target_count,
                                "target_baseline_outcome": str(
                                    target_label["outcome_group"]
                                ),
                                "target_baseline_predicted_count": _optional_int(
                                    target_label.get("parsed_count")
                                ),
                                "site": "answer_query",
                                "steering_method": "scaled_centroid_delta",
                                "steering_protocol": plan.protocol,
                                "layer_set": plan.label,
                                "layer_set_json": json.dumps(list(plan.layers)),
                                "intervened_layer_count": len(plan.layers),
                                "alpha": float(plan.alpha),
                                "condition": condition,
                                "random_replicate": int(replicate),
                                "target_direction": (
                                    "increase" if direction_sign > 0 else "decrease"
                                ),
                                "intended_path_count": path_count,
                                "intended_count_shift": intended_shift,
                                "combined_applied_delta_norm": _combined_delta_norm(
                                    condition_deltas
                                ),
                                "per_layer_applied_delta_norms": json.dumps(
                                    {
                                        str(layer): float(
                                            torch.linalg.vector_norm(delta.float())
                                        )
                                        for layer, delta in condition_deltas.items()
                                    },
                                    sort_keys=True,
                                ),
                                **outcome,
                                **transport_fields(
                                    outcome,
                                    receiver_label=receiver_label,
                                    donor_label=target_label,
                                    receiver_count=receiver_count,
                                    donor_count=target_count,
                                ),
                                "direction_aligned_generated_count_shift": (
                                    float(outcome["generated_count_shift"])
                                    * direction_sign
                                    if np.isfinite(outcome["generated_count_shift"])
                                    else math.nan
                                ),
                                "nearest_path_count_hit": (
                                    bool(patched == math.floor(path_count + 0.5))
                                    if patched is not None
                                    else False
                                ),
                                "moved_toward_path_count": (
                                    abs(float(patched) - path_count)
                                    < abs(float(baseline_prediction) - path_count)
                                    if patched is not None
                                    and baseline_prediction is not None
                                    else False
                                ),
                            }
                        )
                print(
                    "[v4 layer-set steering] "
                    f"split={evaluation_split} {variant} seed={seed} "
                    f"N={receiver_count}->{target_count}",
                    flush=True,
                )
    if not rows:
        raise ValueError("No layer-set steering rows were produced")
    return pd.DataFrame(rows)


def _paired_layer_set_effects(detail: pd.DataFrame) -> pd.DataFrame:
    required = {
        "model_label",
        "design_variant",
        "seed",
        "receiver_stimulus_id",
        "target_stimulus_id",
        "steering_protocol",
        "layer_set",
        "alpha",
        "condition",
        "patched_format_valid",
        "direction_aligned_generated_count_shift",
        "moved_toward_donor_gold",
        "follows_donor_gold",
    }
    missing = sorted(required - set(detail.columns))
    if missing:
        raise ValueError(f"Layer-set steering detail is missing columns: {missing}")
    identifiers = [
        "model_label",
        "design_variant",
        "seed",
        "receiver_stimulus_id",
        "target_stimulus_id",
        "receiver_count",
        "target_count",
        "target_direction",
        "steering_protocol",
        "layer_set",
        "alpha",
    ]
    metrics = [
        "patched_format_valid",
        "direction_aligned_generated_count_shift",
        "moved_toward_donor_gold",
        "follows_donor_gold",
    ]
    work = detail.copy()
    valid = work["patched_format_valid"].astype(bool)
    aligned = pd.to_numeric(
        work["direction_aligned_generated_count_shift"], errors="coerce"
    )
    work["strict_aligned_shift"] = np.where(valid & aligned.notna(), aligned, 0.0)
    work["strict_moved"] = np.where(
        valid, work["moved_toward_donor_gold"].fillna(False).astype(bool), False
    ).astype(float)
    work["strict_target_hit"] = np.where(
        valid, work["follows_donor_gold"].fillna(False).astype(bool), False
    ).astype(float)
    metrics = [
        "patched_format_valid",
        "strict_aligned_shift",
        "strict_moved",
        "strict_target_hit",
    ]
    geometric = work[work["condition"] == "geometric"][identifiers + metrics]
    if geometric.duplicated(identifiers).any():
        raise ValueError("Geometric layer-set rows are not unique")
    random = work[work["condition"] == "orthogonal_norm_matched_random"]
    if random.empty:
        raise ValueError("Layer-set steering requires matched random rows")
    random_mean = random.groupby(identifiers, as_index=False)[metrics].mean()
    random_mean = random_mean.rename(
        columns={metric: f"{metric}_random" for metric in metrics}
    )
    paired = geometric.merge(random_mean, on=identifiers, how="inner", validate="1:1")
    for metric in metrics:
        paired[f"{metric}_effect"] = (
            paired[metric] - paired[f"{metric}_random"]
        )
    return paired


def layer_set_steering_plan_scores(detail: pd.DataFrame) -> pd.DataFrame:
    """Discovery-only robust scores for locking one plan per protocol.

    The primary score is the worst V4-panel mean paired aligned-shift effect,
    penalized by twice the geometric strict-invalid rate.  This deliberately
    favors effects that survive every controlled-relaxation panel instead of a
    large average driven by one panel.  Mean effect and the lower alpha are
    deterministic tie-breakers.
    """

    paired = _paired_layer_set_effects(detail)
    block = (
        paired.groupby(
            [
                "steering_protocol",
                "layer_set",
                "alpha",
                "seed",
                "design_variant",
            ],
            as_index=False,
        )[
            [
                "strict_aligned_shift_effect",
                "strict_moved_effect",
                "strict_target_hit_effect",
                "patched_format_valid",
                "patched_format_valid_random",
            ]
        ]
        .mean()
    )
    rows: list[dict[str, Any]] = []
    for keys, frame in block.groupby(
        ["steering_protocol", "layer_set", "alpha"], sort=True
    ):
        protocol, layer_set, alpha = keys
        variant_effects = frame.groupby("design_variant")[
            "strict_aligned_shift_effect"
        ].mean()
        invalid_rate = 1.0 - float(frame["patched_format_valid"].mean())
        rows.append(
            {
                "steering_protocol": str(protocol),
                "layer_set": str(layer_set),
                "alpha": float(alpha),
                "screen_seeds": int(frame["seed"].nunique()),
                "screen_variants": int(frame["design_variant"].nunique()),
                "mean_aligned_shift_effect": float(
                    frame["strict_aligned_shift_effect"].mean()
                ),
                "worst_variant_aligned_shift_effect": float(variant_effects.min()),
                "positive_variant_count": int((variant_effects > 0).sum()),
                "mean_moved_effect": float(frame["strict_moved_effect"].mean()),
                "mean_target_hit_effect": float(
                    frame["strict_target_hit_effect"].mean()
                ),
                "geometric_valid_rate": float(
                    frame["patched_format_valid"].mean()
                ),
                "random_valid_rate": float(
                    frame["patched_format_valid_random"].mean()
                ),
                "robust_selection_score": float(variant_effects.min())
                - 2.0 * invalid_rate,
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "steering_protocol",
            "robust_selection_score",
            "mean_aligned_shift_effect",
            "alpha",
        ],
        ascending=[True, False, False, True],
        ignore_index=True,
    )


def select_layer_set_steering_plans(detail: pd.DataFrame) -> dict[str, Any]:
    scores = layer_set_steering_plan_scores(detail)
    selected: dict[str, Any] = {
        "selection_split": "discovery",
        "selection_rule": (
            "maximize worst-V4-panel paired strict aligned-shift effect minus "
            "2x geometric invalid rate; break ties by mean effect then lower alpha"
        ),
        "selected": {},
    }
    for protocol in ("single_layer", "multi_layer"):
        available = scores[scores["steering_protocol"] == protocol]
        if available.empty:
            raise ValueError(f"No discovery plans are available for {protocol}")
        row = available.iloc[0]
        selected["selected"][protocol] = {
            "layers": [int(value) for value in str(row["layer_set"]).split("+")],
            "layer_set": str(row["layer_set"]),
            "alpha": float(row["alpha"]),
            "robust_selection_score": float(row["robust_selection_score"]),
            "mean_aligned_shift_effect": float(row["mean_aligned_shift_effect"]),
            "worst_variant_aligned_shift_effect": float(
                row["worst_variant_aligned_shift_effect"]
            ),
            "positive_variant_count": int(row["positive_variant_count"]),
            "geometric_valid_rate": float(row["geometric_valid_rate"]),
        }
    return selected


def _bootstrap_seed_mean(
    values: np.ndarray,
    *,
    seed: int,
    repetitions: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, values.size, size=(int(repetitions), values.size))
    distribution = values[indices].mean(axis=1)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def summarize_layer_set_steering(
    detail: pd.DataFrame,
    *,
    bootstrap_repetitions: int = 10_000,
) -> pd.DataFrame:
    paired = _paired_layer_set_effects(detail)
    rows: list[dict[str, Any]] = []
    for keys, frame in paired.groupby(
        ["model_label", "steering_protocol", "layer_set", "alpha"], sort=True
    ):
        model_label, protocol, layer_set, alpha = keys
        seed_frame = frame.groupby("seed", as_index=False)[
            [
                "strict_aligned_shift_effect",
                "strict_moved_effect",
                "strict_target_hit_effect",
                "patched_format_valid",
                "patched_format_valid_random",
                "strict_aligned_shift",
                "strict_aligned_shift_random",
            ]
        ].mean()
        estimates: dict[str, tuple[float, float, float]] = {}
        for metric in (
            "strict_aligned_shift_effect",
            "strict_moved_effect",
            "strict_target_hit_effect",
        ):
            estimates[metric] = _bootstrap_seed_mean(
                seed_frame[metric].to_numpy(dtype=float),
                seed=_stable_seed(
                    f"layer-set-summary:{model_label}:{protocol}:{layer_set}:"
                    f"{alpha}:{metric}"
                ),
                repetitions=int(bootstrap_repetitions),
            )
        aligned = estimates["strict_aligned_shift_effect"]
        moved = estimates["strict_moved_effect"]
        hit = estimates["strict_target_hit_effect"]
        rows.append(
            {
                "model_label": str(model_label),
                "steering_protocol": str(protocol),
                "layer_set": str(layer_set),
                "alpha": float(alpha),
                "paired_rows": int(len(frame)),
                "seeds": int(seed_frame["seed"].nunique()),
                "variants": int(frame["design_variant"].nunique()),
                "geometric_valid_rate": float(
                    seed_frame["patched_format_valid"].mean()
                ),
                "random_valid_rate": float(
                    seed_frame["patched_format_valid_random"].mean()
                ),
                "geometric_mean_aligned_shift": float(
                    seed_frame["strict_aligned_shift"].mean()
                ),
                "random_mean_aligned_shift": float(
                    seed_frame["strict_aligned_shift_random"].mean()
                ),
                "aligned_shift_effect": aligned[0],
                "aligned_shift_ci95_low": aligned[1],
                "aligned_shift_ci95_high": aligned[2],
                "moved_rate_effect": moved[0],
                "moved_rate_ci95_low": moved[1],
                "moved_rate_ci95_high": moved[2],
                "target_hit_rate_effect": hit[0],
                "target_hit_rate_ci95_low": hit[1],
                "target_hit_rate_ci95_high": hit[2],
                "bootstrap_repetitions": int(bootstrap_repetitions),
            }
        )
    return pd.DataFrame(rows)


def _ols_slope(frame: pd.DataFrame, x: str, y: str) -> tuple[float, float, float]:
    finite = frame[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite) < 2 or finite[x].nunique() < 2:
        return math.nan, math.nan, math.nan
    xv = finite[x].to_numpy(dtype=float)
    yv = finite[y].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(xv)), xv])
    beta, *_ = np.linalg.lstsq(design, yv, rcond=None)
    fitted = design @ beta
    denominator = float(np.sum((yv - yv.mean()) ** 2))
    r2 = (
        1.0 - float(np.sum((yv - fitted) ** 2)) / denominator
        if denominator > 1e-12
        else math.nan
    )
    return float(beta[1]), float(beta[0]), float(r2)


def summarize_generation_geometric_steering(detail: pd.DataFrame) -> pd.DataFrame:
    groups = [
        "model_label",
        "design_variant",
        "layer",
        "steering_method",
        "condition",
        "alpha",
        "target_direction",
        "baseline_outcome",
    ]
    rows: list[dict[str, Any]] = []
    for keys, frame in detail.groupby(groups, sort=True, dropna=False):
        slope, intercept, r2 = _ols_slope(
            frame, "intended_count_shift", "generated_count_shift"
        )
        rows.append(
            {
                **dict(zip(groups, keys)),
                "examples": len(frame),
                "seeds": int(frame["seed"].nunique()),
                "patched_valid_rate": float(frame["patched_format_valid"].mean()),
                "prediction_changed_rate": float(frame["prediction_changed"].mean()),
                "target_hit_rate": float(frame["follows_donor_gold"].mean()),
                "nearest_path_count_hit_rate": float(
                    frame["nearest_path_count_hit"].mean()
                ),
                "moved_toward_target_rate": float(
                    frame["moved_toward_donor_gold"].mean()
                ),
                "moved_toward_path_rate": float(
                    frame["moved_toward_path_count"].mean()
                ),
                "mean_path_count_absolute_error": float(
                    frame["path_count_absolute_error"].mean()
                ),
                "mean_generated_count_shift": float(
                    frame["generated_count_shift"].mean()
                ),
                "mean_intended_count_shift": float(
                    frame["intended_count_shift"].mean()
                ),
                "transport_slope": slope,
                "transport_intercept": intercept,
                "transport_r2": r2,
            }
        )
    return pd.DataFrame(rows)


def compare_geometric_to_random(
    detail: pd.DataFrame,
    *,
    bootstrap_repetitions: int = 10_000,
) -> pd.DataFrame:
    identifiers = [
        "model_label",
        "design_variant",
        "seed",
        "receiver_stimulus_id",
        "target_stimulus_id",
        "layer",
        "steering_method",
        "alpha",
        "target_direction",
        "baseline_outcome",
    ]
    metrics = [
        "follows_donor_gold",
        "moved_toward_donor_gold",
        "path_count_absolute_error",
        "generated_count_shift",
    ]
    geometric = detail[detail["condition"] == "geometric"].copy()
    random = detail[
        detail["condition"] == "orthogonal_norm_matched_random"
    ].copy()
    if random.empty:
        return pd.DataFrame()
    random_mean = (
        random.groupby(identifiers, as_index=False)[metrics]
        .mean()
        .rename(columns={metric: f"{metric}_random_mean" for metric in metrics})
    )
    paired = geometric.merge(random_mean, on=identifiers, how="inner")
    group_columns = [
        "model_label",
        "design_variant",
        "layer",
        "steering_method",
        "alpha",
        "target_direction",
        "baseline_outcome",
    ]
    rows: list[dict[str, Any]] = []
    for keys, frame in paired.groupby(group_columns, sort=True, dropna=False):
        metadata = dict(zip(group_columns, keys))
        for metric in metrics:
            seed_means = (
                frame.assign(
                    difference=frame[metric] - frame[f"{metric}_random_mean"]
                )
                .groupby("seed")["difference"]
                .mean()
                .dropna()
                .to_numpy(dtype=float)
            )
            if len(seed_means) == 0:
                mean = low = high = math.nan
            else:
                rng = np.random.default_rng(
                    _stable_seed(
                        ":".join(str(metadata[name]) for name in group_columns)
                        + f":{metric}"
                    )
                )
                sampled = rng.integers(
                    0,
                    len(seed_means),
                    size=(int(bootstrap_repetitions), len(seed_means)),
                )
                distribution = seed_means[sampled].mean(axis=1)
                mean = float(seed_means.mean())
                low, high = np.quantile(distribution, [0.025, 0.975])
            rows.append(
                {
                    **metadata,
                    "metric": metric,
                    "geometric_minus_random_mean": float(mean),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "confirmation_seeds": len(seed_means),
                    "bootstrap_repetitions": int(bootstrap_repetitions),
                }
            )
    return pd.DataFrame(rows)
