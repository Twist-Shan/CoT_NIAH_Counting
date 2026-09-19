#!/usr/bin/env python3
"""Build seed-clustered layerwise statistics for V4.4.5 span restoration.

The primary unit is a confirmation seed.  Counts 1--10 are averaged within
each seed before uncertainty or sign tests are computed, so the ten counts are
not treated as independent replications.  P values are nominal two-sided exact
sign-flip values; no multiplicity correction is applied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


PRIMARY = "expected_absolute_error_reduction_specificity"
FULL_MINUS_ENDPOINT = "expected_absolute_error_reduction_full_minus_endpoint"
EXPECTED_MODELS = {"Qwen3-8B": list(range(36)), "Gemma4-E4B": list(range(42))}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_signflip_p(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    null_means = np.abs((signs * values[None, :]).mean(axis=1))
    return float(np.mean(null_means >= observed - 1e-12))


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    rng_seed: int,
    draws: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(rng_seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def contiguous_segments(layers: list[int]) -> list[list[int]]:
    if not layers:
        return []
    segments = [[layers[0]]]
    for layer in layers[1:]:
        if layer == segments[-1][-1] + 1:
            segments[-1].append(layer)
        else:
            segments.append([layer])
    return segments


def summarize_vector(
    values: np.ndarray,
    *,
    rng_seed: int,
    draws: int,
) -> dict[str, Any]:
    low, high = bootstrap_mean_ci(values, rng_seed=rng_seed, draws=draws)
    p_value = exact_signflip_p(values)
    mean = float(values.mean())
    return {
        "mean": mean,
        "ci95_low": low,
        "ci95_high": high,
        "exact_signflip_p": p_value,
        "nominal_p_lt_0_05": bool(p_value < 0.05),
        "positive_seeds": int(np.sum(values > 0)),
        "negative_seeds": int(np.sum(values < 0)),
        "zero_seeds": int(np.sum(values == 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=50_000)
    parser.add_argument("--rng-seed", type=int, default=20260814)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    primary_path = input_dir / "needle_minus_ordinary_specificity.csv"
    endpoint_path = input_dir / "full_minus_endpoint.csv"
    audit_path = input_dir / "analysis_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or int(audit.get("detail_rows", -1)) != 72_000:
        raise RuntimeError("The frozen span-restoration analysis audit is not PASS")
    confirmation_seeds = [int(value) for value in audit["confirmation_seeds"]]
    if confirmation_seeds != list(range(1254, 1264)):
        raise RuntimeError(f"Unexpected confirmation seeds: {confirmation_seeds}")

    primary_rows = read_csv(primary_path)
    endpoint_rows = read_csv(endpoint_path)
    if len(primary_rows) != 23_400 or len(endpoint_rows) != 23_400:
        raise RuntimeError("Expected 23,400 paired unit-layer rows in each source table")

    key_fields = ("model_label", "seed", "gold_count", "patch_layer")

    def key(row: dict[str, str]) -> tuple[str, int, int, int]:
        return (
            row["model_label"],
            int(row["seed"]),
            int(row["gold_count"]),
            int(row["patch_layer"]),
        )

    primary = {key(row): float(row[PRIMARY]) for row in primary_rows}
    full_minus_endpoint = {
        key(row): float(row[FULL_MINUS_ENDPOINT]) for row in endpoint_rows
    }
    if set(primary) != set(full_minus_endpoint):
        raise RuntimeError(f"Source key mismatch for {key_fields}")

    layer_rows: list[dict[str, Any]] = []
    adjacent_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    model_seed_vectors: dict[str, dict[int, np.ndarray]] = {}

    for model_index, (model, expected_layers) in enumerate(EXPECTED_MODELS.items()):
        seed_vectors: dict[int, np.ndarray] = {}
        endpoint_vectors: dict[int, np.ndarray] = {}
        for layer in expected_layers:
            primary_seed_means: list[float] = []
            endpoint_seed_means: list[float] = []
            for seed in confirmation_seeds:
                keys = [(model, seed, count, layer) for count in range(1, 11)]
                if any(item not in primary for item in keys):
                    raise RuntimeError(f"Missing confirmation key for {model} L{layer} seed {seed}")
                primary_seed_means.append(float(np.mean([primary[item] for item in keys])))
                endpoint_seed_means.append(
                    float(
                        np.mean(
                            [primary[item] - full_minus_endpoint[item] for item in keys]
                        )
                    )
                )
            seed_vector = np.asarray(primary_seed_means, dtype=float)
            endpoint_vector = np.asarray(endpoint_seed_means, dtype=float)
            seed_vectors[layer] = seed_vector
            endpoint_vectors[layer] = endpoint_vector
            primary_stats = summarize_vector(
                seed_vector,
                rng_seed=args.rng_seed + model_index * 10_000 + layer,
                draws=args.bootstrap_draws,
            )
            endpoint_stats = summarize_vector(
                endpoint_vector,
                rng_seed=args.rng_seed + 20_000 + model_index * 10_000 + layer,
                draws=args.bootstrap_draws,
            )
            layer_rows.append(
                {
                    "model_label": model,
                    "layer": layer,
                    "seed_units": len(confirmation_seeds),
                    "counts_per_seed": 10,
                    **primary_stats,
                    "endpoint_minus_ordinary_mean": endpoint_stats["mean"],
                    "endpoint_minus_ordinary_ci95_low": endpoint_stats["ci95_low"],
                    "endpoint_minus_ordinary_ci95_high": endpoint_stats["ci95_high"],
                    "endpoint_minus_ordinary_exact_signflip_p": endpoint_stats[
                        "exact_signflip_p"
                    ],
                }
            )

        model_seed_vectors[model] = seed_vectors
        for layer in expected_layers[:-1]:
            next_layer = layer + 1
            difference = seed_vectors[next_layer] - seed_vectors[layer]
            adjacent_rows.append(
                {
                    "model_label": model,
                    "from_layer": layer,
                    "to_layer": next_layer,
                    **summarize_vector(
                        difference,
                        rng_seed=args.rng_seed + 40_000 + model_index * 10_000 + layer,
                        draws=args.bootstrap_draws,
                    ),
                }
            )

        model_layers = [row for row in layer_rows if row["model_label"] == model]
        model_adjacent = [row for row in adjacent_rows if row["model_label"] == model]
        positive_nominal = [
            int(row["layer"])
            for row in model_layers
            if row["mean"] > 0 and row["nominal_p_lt_0_05"]
        ]
        negative_nominal = [
            int(row["layer"])
            for row in model_layers
            if row["mean"] < 0 and row["nominal_p_lt_0_05"]
        ]
        largest_drop = min(model_adjacent, key=lambda row: float(row["mean"]))
        summaries.append(
            {
                "model_label": model,
                "positive_nominal_layers": positive_nominal,
                "positive_nominal_segments": contiguous_segments(positive_nominal),
                "positive_nominal_layer_count": len(positive_nominal),
                "negative_nominal_layers": negative_nominal,
                "negative_nominal_segments": contiguous_segments(negative_nominal),
                "largest_adjacent_drop": largest_drop,
                "endpoint_minus_ordinary_min": min(
                    float(row["endpoint_minus_ordinary_mean"]) for row in model_layers
                ),
                "endpoint_minus_ordinary_max": max(
                    float(row["endpoint_minus_ordinary_mean"]) for row in model_layers
                ),
            }
        )

    output = {
        "schema_version": "realistic_niah_v4_4_5_span_layerwise_seed_statistics_v1",
        "status": "PASS",
        "definition": {
            "primary": (
                "full-needle expected-error repair minus equal-token-budget ordinary "
                "restoration repair"
            ),
            "unit": "mean over counts 1-10 within each confirmation seed",
            "mean": "equal-weight mean over ten confirmation seed units",
            "ci95": (
                f"percentile seed-cluster bootstrap, {args.bootstrap_draws} draws, "
                "counts kept together within seed"
            ),
            "p_value": (
                "two-sided exact sign flip over 2^10 seed effects; nominal p<0.05, "
                "no multiplicity correction"
            ),
            "adjacent_drop": "S_restore(to_layer) minus S_restore(from_layer), paired by seed",
            "endpoint_minus_ordinary": (
                "(full minus ordinary) minus (full minus endpoint) = endpoint minus ordinary"
            ),
        },
        "confirmation_seeds": confirmation_seeds,
        "bootstrap_draws": args.bootstrap_draws,
        "rng_seed": args.rng_seed,
        "source_sha256": {
            primary_path.name: sha256(primary_path),
            endpoint_path.name: sha256(endpoint_path),
            audit_path.name: sha256(audit_path),
        },
        "layer_rows": layer_rows,
        "adjacent_rows": adjacent_rows,
        "model_summaries": summaries,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": output["status"],
                "output": str(output_path),
                "layer_rows": len(layer_rows),
                "adjacent_rows": len(adjacent_rows),
                "summaries": summaries,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
