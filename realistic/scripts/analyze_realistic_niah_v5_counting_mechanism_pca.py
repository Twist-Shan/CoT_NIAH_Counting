#!/usr/bin/env python3
"""Capture first-pass item states and compare PCA/NCC across layers and sites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v5.count_stream import build_answer_source_registry  # noqa: E402
from realistic_niah_v5.counting_mechanism_transfer import (  # noqa: E402
    build_first_pass_tstar_answer_source_registry,
)
from realistic_niah_v5.indexed_counter_patch import (  # noqa: E402
    capture_decoder_block_input_states,
)
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
    _model,
)
from scripts.run_realistic_niah_v5_counting_mechanism_transfer import (  # noqa: E402
    _load_config,
    _read_rows,
)


SITES = ("closing", "post_item")
VIEWS = ("raw", "within_seed_centered")


def _seed_sets(config: Mapping[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    contract = config.get("cohort_contract", {})
    discovery = tuple(
        int(value)
        for value in contract.get("discovery_seeds_used_for_confirmation_geometry_fit", ())
    )
    confirmation = tuple(
        int(value) for value in contract.get("confirmation_seeds_reserved", ())
    )
    if not discovery or not confirmation:
        raise ValueError("Config lacks frozen discovery/confirmation seed sets")
    if set(discovery) & set(confirmation):
        raise ValueError("Discovery and confirmation seeds overlap")
    return discovery, confirmation


def _site_positions(registry: Any, sequence_length: int) -> dict[str, tuple[int, ...]]:
    items = tuple((int(start), int(end)) for start, end in registry.trace_items)
    closing = tuple(end - 1 for _start, end in items)
    post_item = tuple(end for _start, end in items)
    if any(not 0 <= value < int(sequence_length) for value in closing + post_item):
        raise ValueError("PCA site falls outside the first-pass encoding")
    return {"closing": closing, "post_item": post_item}


def _capture_seed(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    row: Mapping[str, Any],
    *,
    candidate_counts: Sequence[int],
    registry_builder: Any,
) -> dict[str, np.ndarray]:
    encoding, registry = registry_builder(
        row, tokenizer, candidate_counts=tuple(int(value) for value in candidate_counts)
    )
    positions_by_site = _site_positions(registry, encoding.sequence_length)
    unique_positions = tuple(
        sorted({position for values in positions_by_site.values() for position in values})
    )
    layers = tuple(range(int(adapter.num_layers)))
    captured = capture_decoder_block_input_states(
        model, adapter, encoding, unique_positions, layers=layers
    )
    lookup = {position: index for index, position in enumerate(unique_positions)}
    result: dict[str, np.ndarray] = {}
    for site, positions in positions_by_site.items():
        indices = [lookup[position] for position in positions]
        result[site] = np.stack(
            [captured[layer][indices].detach().float().cpu().numpy() for layer in layers],
            axis=0,
        ).astype(np.float16)
    return result


def _nearest_centroid(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centroids = np.stack([train_x[train_y == label].mean(axis=0) for label in labels])
    distances = np.square(test_x[:, None, :] - centroids[None, :, :]).sum(axis=2)
    order = np.argsort(distances, axis=1)
    prediction = labels[order[:, 0]]
    margin = distances[np.arange(len(test_x)), order[:, 1]] - distances[
        np.arange(len(test_x)), order[:, 0]
    ]
    return prediction, margin


def _cosine_nearest_centroid(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    eps = np.finfo(np.float64).eps
    train = train_x / np.maximum(np.linalg.norm(train_x, axis=1, keepdims=True), eps)
    test = test_x / np.maximum(np.linalg.norm(test_x, axis=1, keepdims=True), eps)
    centroids = np.stack([train[train_y == label].mean(axis=0) for label in labels])
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), eps)
    return labels[np.argmax(test @ centroids.T, axis=1)]


def _orient(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    result = np.asarray(scores, dtype=np.float64).copy()
    if result.shape[1] and np.corrcoef(result[:, 0], labels)[0, 1] < 0:
        result[:, 0] *= -1.0
    return result


def _center_within_seed(states: np.ndarray) -> np.ndarray:
    return states - states.mean(axis=2, keepdims=True)


def _analyze(
    states: Mapping[str, np.ndarray],
    *,
    seeds: Sequence[int],
    splits: Sequence[str],
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seeds_array = np.asarray(seeds, dtype=np.int64)
    splits_array = np.asarray(splits, dtype=object)
    discovery_seed_mask = splits_array == "discovery"
    confirmation_seed_mask = splits_array == "confirmation"
    if not discovery_seed_mask.any() or not confirmation_seed_mask.any():
        raise ValueError("PCA requires both discovery and confirmation rows")
    metrics: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    for site in SITES:
        site_states = np.asarray(states[site], dtype=np.float32)
        if site_states.ndim != 4:
            raise ValueError("State tensor must have shape [seed, layer, k, hidden]")
        seed_count, layer_count, count, hidden = site_states.shape
        labels = np.arange(1, count + 1, dtype=np.int64)
        all_y = np.tile(labels, seed_count)
        discovery_y = np.tile(labels, int(discovery_seed_mask.sum()))
        confirmation_y = np.tile(labels, int(confirmation_seed_mask.sum()))
        for view in VIEWS:
            active = (
                site_states
                if view == "raw"
                else _center_within_seed(site_states)
            )
            for layer in range(layer_count):
                layer_states = active[:, layer]
                discovery_x = layer_states[discovery_seed_mask].reshape(-1, hidden)
                confirmation_x = layer_states[confirmation_seed_mask].reshape(-1, hidden)

                display_pca = PCA(
                    n_components=3,
                    svd_solver="randomized",
                    random_state=int(random_seed) + layer * 101,
                ).fit(discovery_x)
                display_all = _orient(
                    display_pca.transform(layer_states.reshape(-1, hidden)), all_y
                )

                scaler = StandardScaler().fit(discovery_x)
                discovery_scaled = scaler.transform(discovery_x)
                confirmation_scaled = scaler.transform(confirmation_x)
                components = min(16, discovery_scaled.shape[0] - 1, hidden)
                ncc_pca = PCA(
                    n_components=components,
                    whiten=True,
                    svd_solver="randomized",
                    random_state=int(random_seed) + 10000 + layer * 101,
                ).fit(discovery_scaled)
                discovery_z = ncc_pca.transform(discovery_scaled)
                confirmation_z = ncc_pca.transform(confirmation_scaled)
                prediction, margin = _nearest_centroid(
                    discovery_z, discovery_y, confirmation_z, labels
                )
                cosine_prediction = _cosine_nearest_centroid(
                    discovery_x, discovery_y, confirmation_x, labels
                )
                correct = prediction == confirmation_y
                per_k_accuracy = {
                    str(label): float(np.mean(correct[confirmation_y == label]))
                    for label in labels
                }
                per_k_margin = {
                    str(label): float(np.mean(margin[confirmation_y == label]))
                    for label in labels
                }
                centroids = np.stack(
                    [discovery_x[discovery_y == label].mean(axis=0) for label in labels]
                )
                adjacent = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
                residual = np.concatenate(
                    [
                        discovery_x[discovery_y == label] - centroids[index]
                        for index, label in enumerate(labels)
                    ],
                    axis=0,
                )
                within_radius = float(np.sqrt(np.mean(np.square(residual).sum(axis=1))))
                count_pc1_r = float(
                    np.corrcoef(
                        display_all.reshape(seed_count, count, 3)[
                            discovery_seed_mask, :, 0
                        ].reshape(-1),
                        discovery_y,
                    )[0, 1]
                )
                metrics.append(
                    {
                        "site": site,
                        "view": view,
                        "layer": layer,
                        "count": count,
                        "hidden_size": hidden,
                        "discovery_seed_count": int(discovery_seed_mask.sum()),
                        "confirmation_seed_count": int(confirmation_seed_mask.sum()),
                        "pca_explained_variance_ratio": [
                            float(value) for value in display_pca.explained_variance_ratio_
                        ],
                        "pca3_explained_variance_sum": float(
                            display_pca.explained_variance_ratio_.sum()
                        ),
                        "count_pc1_correlation": count_pc1_r,
                        "confirmation_pca16_whiten_ncc_accuracy": float(
                            np.mean(correct)
                        ),
                        "confirmation_raw_cosine_ncc_accuracy": float(
                            np.mean(cosine_prediction == confirmation_y)
                        ),
                        "confirmation_pca16_whiten_ncc_accuracy_by_k": per_k_accuracy,
                        "confirmation_pca16_whiten_ncc_margin_by_k": per_k_margin,
                        "mean_adjacent_centroid_distance": float(adjacent.mean()),
                        "adjacent_centroid_distance_by_transition": {
                            f"{left}->{left + 1}": float(adjacent[left - 1])
                            for left in labels[:-1]
                        },
                        "within_class_rms_radius": within_radius,
                        "adjacent_centroid_to_within_radius": float(
                            adjacent.mean() / max(within_radius, np.finfo(float).eps)
                        ),
                    }
                )
                reshaped = display_all.reshape(seed_count, count, 3)
                for seed_index, seed in enumerate(seeds_array):
                    for occurrence in labels:
                        coordinate = reshaped[seed_index, occurrence - 1]
                        points.append(
                            {
                                "site": site,
                                "view": view,
                                "layer": layer,
                                "seed": int(seed),
                                "split": str(splits_array[seed_index]),
                                "k": int(occurrence),
                                "pc1": float(coordinate[0]),
                                "pc2": float(coordinate[1]),
                                "pc3": float(coordinate[2]),
                            }
                        )
    return metrics, points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    shards = args.output / "state_shards"
    shards.mkdir(parents=True, exist_ok=True)
    config = _load_config(args.config)
    selection_mode = str(config.get("row_selection", "unique_seed"))
    registry_builder = (
        build_first_pass_tstar_answer_source_registry
        if selection_mode == "first_pass_noindex"
        else build_answer_source_registry
    )
    rows_by_seed = _read_rows(args.generations, selection_mode=selection_mode)
    discovery, confirmation = _seed_sets(config)
    seeds = discovery + confirmation
    missing = sorted(set(seeds) - set(rows_by_seed))
    if missing:
        raise ValueError(f"Generation file lacks PCA seeds: {missing}")
    candidate_counts = tuple(int(value) for value in config["candidate_counts"])
    model, tokenizer, adapter = _model(args)

    states_by_site: dict[str, list[np.ndarray]] = {site: [] for site in SITES}
    splits: list[str] = []
    for index, seed in enumerate(seeds, start=1):
        shard = shards / f"seed_{seed}.npz"
        if args.resume and shard.exists():
            saved = np.load(shard)
            captured = {site: saved[f"states_{site}"] for site in SITES}
        else:
            captured = _capture_seed(
                model,
                tokenizer,
                adapter,
                rows_by_seed[seed],
                candidate_counts=candidate_counts,
                registry_builder=registry_builder,
            )
            np.savez_compressed(
                shard,
                seed=np.asarray([seed], dtype=np.int64),
                **{f"states_{site}": captured[site] for site in SITES},
            )
        for site in SITES:
            states_by_site[site].append(np.asarray(captured[site], dtype=np.float16))
        splits.append("discovery" if seed in set(discovery) else "confirmation")
        print(f"[pca-capture] {index}/{len(seeds)} seed={seed}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    stacked = {site: np.stack(values, axis=0) for site, values in states_by_site.items()}
    np.savez_compressed(
        args.output / "states.npz",
        seeds=np.asarray(seeds, dtype=np.int64),
        splits=np.asarray(splits),
        **{f"states_{site}": value for site, value in stacked.items()},
    )
    metrics, points = _analyze(
        stacked,
        seeds=seeds,
        splits=splits,
        random_seed=int(config.get("random_seed", 20260827)),
    )
    _atomic_json(args.output / "layer_metrics.json", {"rows": metrics})
    _atomic_jsonl(args.output / "pca_points.jsonl", points)
    manifest = {
        "schema_version": "first_pass_counting_pca_v1",
        "status": "PASS",
        "model": str(args.model),
        "fixed_count": int(config["cohort_contract"]["fixed_count"]),
        "sites": list(SITES),
        "views": list(VIEWS),
        "layers": list(range(int(adapter.num_layers))),
        "discovery_seeds": list(discovery),
        "confirmation_seeds": list(confirmation),
        "pca_fit": "discovery only; confirmation projected without refit",
        "ncc": "discovery StandardScaler + whitened PCA16 + nearest centroid",
        "raw_cosine_ncc": "discovery centroids after rowwise L2 normalization",
        "elapsed_seconds": float(time.perf_counter() - started),
        "trial_point_count": len(points),
    }
    _atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
