from __future__ import annotations

"""Fit frozen prompt counter geometry and extract scalar noise covariates."""

import argparse
import csv
import gzip
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from safetensors import safe_open


def metadata_rows(capture: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted((capture / "shards").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") == "completed":
            rows.append(row)
    return rows


def states_for(capture: Path, row: dict[str, object]) -> np.ndarray:
    with safe_open(capture / str(row["tensor_path"]), framework="np", device="cpu") as handle:
        states = np.asarray(handle.get_tensor("states"), dtype=np.float32)
        layers = np.asarray(handle.get_tensor("layer_indices"), dtype=np.int64)
    if not np.array_equal(layers, np.arange(states.shape[0])):
        raise RuntimeError(f"non-contiguous layer indices for {row['stimulus_id']}")
    return states


def request_covariates(path: Path) -> dict[str, dict[str, object]]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row.get("gold_count", 0)) != 10:
                continue
            evaluation = row.get("evaluation", {})
            design = row.get("design", {})
            haystack = row.get("haystack", {})
            result[str(row["stimulus_id"])] = {
                "baseline_correct": bool(evaluation.get("exact_count", False)),
                "absolute_deviation": evaluation.get("absolute_error", ""),
                "predicted_count": evaluation.get("predicted_count", ""),
                "realization_seed": row.get("realization_seed", ""),
                "content_seed": design.get("content_seed", ""),
                "haystack_source_mode": haystack.get("source_mode", ""),
                "haystack_source_files": "|".join(haystack.get("source_files", [])),
            }
    return result


FIELDS = [
    "model_label", "role", "stimulus_id", "seed", "realization_id", "split",
    "layer", "running_index", "count_progress", "city", "score", "token_start",
    "token_end", "token_span", "prompt_progress", "previous_endpoint_gap",
    "input_tokens", "passage_sha256", "realization_seed", "content_seed",
    "haystack_source_mode", "haystack_source_files", "baseline_correct",
    "absolute_deviation", "predicted_count", "noise_total", "noise_parallel",
    "noise_orthogonal", "noise_total_rms", "noise_parallel_rms",
    "noise_orthogonal_rms", "parallel_energy_fraction", "count_axis_deviation",
    "signal_rms", "noise_to_signal", "basis_rank", "centroid_variance_capture",
    "basis_fit_split",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=3)
    args = parser.parse_args()
    started = time.perf_counter()
    capture = args.capture.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = metadata_rows(capture)
    if not rows:
        raise RuntimeError(f"no completed prompt captures under {capture}")
    discovery = [row for row in rows if row["split"] == "discovery"]
    confirmation = [row for row in rows if row["split"] == "confirmation"]
    if not discovery or not confirmation:
        raise RuntimeError("both discovery and confirmation captures are required")
    first = states_for(capture, discovery[0])
    layers, labels, hidden = first.shape
    if labels != 10:
        raise RuntimeError(f"expected 10 prompt endpoints, got {first.shape}")
    sums = np.zeros((layers, labels, hidden), dtype=np.float64)
    counts = np.zeros(labels, dtype=np.int64)
    for index, row in enumerate(discovery, start=1):
        states = states_for(capture, row)
        sums += states
        counts += 1
        if index % 100 == 0:
            print(f"[prompt geometry fit] {index}/{len(discovery)}", flush=True)
    centroids = (sums / counts[None, :, None]).astype(np.float32)
    centers = centroids.mean(axis=1)
    bases = np.empty((layers, hidden, args.rank), dtype=np.float32)
    captures = np.empty(layers, dtype=np.float64)
    signal_rms = np.empty(layers, dtype=np.float64)
    for layer in range(layers):
        centered = centroids[layer].astype(np.float64) - centers[layer]
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        bases[layer] = vt[: args.rank].T
        captures[layer] = np.square(singular[: args.rank]).sum() / max(np.square(singular).sum(), 1e-12)
        signal_rms[layer] = math.sqrt(float(np.square(centered).sum(axis=1).mean()) / hidden)

    discovery_energy = np.zeros(layers, dtype=np.float64)
    discovery_observations = 0
    for row in discovery:
        residual = states_for(capture, row) - centroids
        discovery_energy += np.square(residual, dtype=np.float64).sum(axis=(1, 2))
        discovery_observations += labels
    discovery_noise_rms = np.sqrt(discovery_energy / (discovery_observations * hidden))
    discovery_nsr = discovery_noise_rms / np.maximum(signal_rms, 1e-12)
    primary_layer = int(np.nanargmin(discovery_nsr))

    covariates = request_covariates(args.requests.resolve())
    partial = output / "prompt_noise_rows.csv.gz.partial"
    final = output / "prompt_noise_rows.csv.gz"
    split_energy = {"discovery": np.zeros(layers), "confirmation": np.zeros(layers)}
    split_observations = {"discovery": 0, "confirmation": 0}
    scalar_rows = 0
    with gzip.open(partial, "wt", newline="", encoding="utf-8", compresslevel=5) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            states = states_for(capture, row)
            residual_all = states - centroids
            split = str(row["split"])
            split_energy[split] += np.square(residual_all, dtype=np.float64).sum(axis=(1, 2))
            split_observations[split] += labels
            states_primary = states[primary_layer]
            residual = states_primary - centroids[primary_layer]
            coordinates = residual @ bases[primary_layer]
            extra = covariates.get(str(row["stimulus_id"]), {})
            previous = None
            for event_index, event in enumerate(row["events"]):
                total = float(np.dot(residual[event_index], residual[event_index]))
                parallel = float(np.dot(coordinates[event_index], coordinates[event_index]))
                orthogonal = max(0.0, total - parallel)
                endpoint = int(event["token_end"])
                gap = "" if previous is None else endpoint - previous
                previous = endpoint
                writer.writerow({
                    "model_label": row["model_label"],
                    "role": "prompt_running",
                    "stimulus_id": row["stimulus_id"],
                    "seed": row["seed"],
                    "realization_id": row["realization_id"],
                    "split": split,
                    "layer": primary_layer,
                    "running_index": event["running_index"],
                    "count_progress": int(event["running_index"]) / 10.0,
                    "city": event["city"],
                    "score": event["score"],
                    "token_start": event["token_start"],
                    "token_end": endpoint,
                    "token_span": endpoint - int(event["token_start"]),
                    "prompt_progress": endpoint / int(row["input_tokens"]),
                    "previous_endpoint_gap": gap,
                    "input_tokens": row["input_tokens"],
                    "passage_sha256": row["passage_sha256"],
                    "realization_seed": extra.get("realization_seed", ""),
                    "content_seed": extra.get("content_seed", ""),
                    "haystack_source_mode": extra.get("haystack_source_mode", ""),
                    "haystack_source_files": extra.get("haystack_source_files", ""),
                    "baseline_correct": extra.get("baseline_correct", ""),
                    "absolute_deviation": extra.get("absolute_deviation", ""),
                    "predicted_count": extra.get("predicted_count", ""),
                    "noise_total": total,
                    "noise_parallel": parallel,
                    "noise_orthogonal": orthogonal,
                    "noise_total_rms": math.sqrt(total / hidden),
                    "noise_parallel_rms": math.sqrt(parallel / args.rank),
                    "noise_orthogonal_rms": math.sqrt(orthogonal / max(1, hidden - args.rank)),
                    "parallel_energy_fraction": parallel / max(total, 1e-12),
                    "count_axis_deviation": float(coordinates[event_index, 0]),
                    "signal_rms": signal_rms[primary_layer],
                    "noise_to_signal": math.sqrt(total / hidden) / max(signal_rms[primary_layer], 1e-12),
                    "basis_rank": args.rank,
                    "centroid_variance_capture": captures[primary_layer],
                    "basis_fit_split": "discovery",
                })
                scalar_rows += 1
            if index % 100 == 0:
                print(f"[prompt noise] {index}/{len(rows)} rows={scalar_rows}", flush=True)
    os.replace(partial, final)

    layer_rows = []
    for layer in range(layers):
        item = {
            "layer": layer,
            "signal_rms": signal_rms[layer],
            "centroid_variance_capture_rank3": captures[layer],
            "discovery_noise_rms": discovery_noise_rms[layer],
            "discovery_noise_to_signal": discovery_nsr[layer],
            "is_primary": layer == primary_layer,
        }
        for split in ("discovery", "confirmation"):
            noise = math.sqrt(split_energy[split][layer] / (split_observations[split] * hidden))
            item[f"{split}_noise_rms_all_pass"] = noise
            item[f"{split}_noise_to_signal"] = noise / max(signal_rms[layer], 1e-12)
        layer_rows.append(item)
    with (output / "prompt_noise_layer_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(layer_rows[0]))
        writer.writeheader()
        writer.writerows(layer_rows)
    np.savez_compressed(
        output / "frozen_prompt_counter_geometry.npz",
        centroids=centroids,
        centers=centers,
        bases=bases,
        signal_rms=signal_rms,
        centroid_variance_capture=captures,
        primary_layer=np.asarray(primary_layer),
        rank=np.asarray(args.rank),
    )
    audit = {
        "schema_version": "realistic_niah_v4_4_native_prompt_noise_v1",
        "source_capture": str(capture),
        "source_requests": str(args.requests.resolve()),
        "source_read_only": True,
        "discovery_samples": len(discovery),
        "confirmation_samples": len(confirmation),
        "layers": layers,
        "hidden_size": hidden,
        "rank": args.rank,
        "primary_layer_selected_on_discovery_nsr": primary_layer,
        "scalar_rows": scalar_rows,
        "elapsed_seconds": time.perf_counter() - started,
        "status": "PASS",
    }
    (output / "prompt_noise_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
