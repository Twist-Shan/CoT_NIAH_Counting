from __future__ import annotations

"""Stream frozen native-thinking geometry captures into scalar noise rows.

This extractor never modifies the source campaign and never duplicates hidden
states.  Count centroids and a rank-r counter basis are frozen from discovery
data; every state is then decomposed into count-subspace and orthogonal
residual energy.  The output contains only scalar covariates and diagnostics.
"""

import argparse
import csv
import gzip
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np
from safetensors import safe_open


def read_primary_layers(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = {
        row["site"]: int(row["layer"])
        for row in rows
        if row["estimand"] == "running_index" and row["stratum"] == "all"
    }
    if not selected:
        raise RuntimeError(f"no running_index/all primary layers in {path}")
    return selected


def load_frozen_geometry(path: Path, site_layers: dict[str, int], rank: int):
    result: dict[str, dict[str, np.ndarray | int]] = {}
    with np.load(path, allow_pickle=False) as payload:
        for site, layer in site_layers.items():
            prefix = f"running_index__{site}__all__layer_{layer:03d}__discovery_centroid_"
            keys = [prefix + str(label) for label in range(1, 11)]
            missing = [key for key in keys if key not in payload.files]
            if missing:
                raise KeyError(f"missing frozen centroid keys: {missing[:2]}")
            centroids = np.stack([np.asarray(payload[key], dtype=np.float32) for key in keys])
            centered = centroids.astype(np.float64) - centroids.mean(axis=0, keepdims=True)
            _, singular, vt = np.linalg.svd(centered, full_matrices=False)
            basis = vt[:rank].T.astype(np.float32)
            result[site] = {
                "layer": layer,
                "centroids": centroids,
                "center": centroids.mean(axis=0).astype(np.float32),
                "basis": basis,
                "centroid_variance_capture": float(
                    np.square(singular[:rank]).sum() / max(np.square(singular).sum(), 1e-12)
                ),
            }
    return result


def load_selection(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["request_id"]: row for row in csv.DictReader(handle)}


def prompt_group(stimulus_id: str) -> str:
    return re.sub(r"_realization\d+$", "", stimulus_id)


FIELDS = [
    "model_label", "request_id", "stimulus_id", "prompt_group", "seed",
    "realization_id", "split", "site", "layer", "state_kind", "marker_kind",
    "termination_kind", "trace_order_class", "trace_one_to_one", "gold_count",
    "item_count", "running_index", "count_progress", "position", "trace_position",
    "trace_progress", "previous_same_site_gap", "token_span", "input_tokens",
    "output_tokens", "sequence_length", "target_passage_tokens",
    "duplicate_gold_city_items", "city_occurrences_in_item", "baseline_correct",
    "cutoff_correct", "noise_total", "noise_parallel", "noise_orthogonal",
    "noise_total_rms", "noise_parallel_rms", "noise_orthogonal_rms",
    "parallel_energy_fraction", "count_axis_deviation", "basis_rank",
    "centroid_variance_capture", "basis_fit_split",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["Qwen3-8B", "Gemma4-E4B"])
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--sites", nargs="+", help="Optional site subset")
    parser.add_argument("--splits", nargs="+", help="Optional split subset")
    parser.add_argument("--limit", type=int, help="Smoke-test sample limit per model")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.rank <= 9:
        raise ValueError("rank must be in [1,9]")

    started = time.perf_counter()
    source = args.source_run.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    audits: list[dict[str, object]] = []

    for model in args.models:
        model_started = time.perf_counter()
        capture = source / "geometry" / model / "capture"
        analysis = source / "geometry" / model / "analysis"
        site_layers = read_primary_layers(analysis / "primary_layers.csv")
        if args.sites:
            wanted = set(args.sites)
            site_layers = {site: layer for site, layer in site_layers.items() if site in wanted}
            missing = wanted - set(site_layers)
            if missing:
                raise ValueError(f"unknown sites for {model}: {sorted(missing)}")
        frozen = load_frozen_geometry(
            analysis / "directions_and_centroids.npz", site_layers, args.rank
        )
        selection = load_selection(capture / "selection.csv")
        model_output = output / model
        model_output.mkdir(exist_ok=True)
        partial = model_output / "trace_noise_rows.csv.gz.partial"
        final = model_output / "trace_noise_rows.csv.gz"
        row_count = 0
        sample_count = 0
        split_counts: dict[str, int] = {}
        site_counts: dict[str, int] = {site: 0 for site in site_layers}
        with (capture / "capture_index.jsonl").open(encoding="utf-8") as index_handle, gzip.open(
            partial, "wt", newline="", encoding="utf-8", compresslevel=5
        ) as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=FIELDS)
            writer.writeheader()
            for line in index_handle:
                meta = json.loads(line)
                if meta.get("status") != "completed":
                    continue
                if args.splits and str(meta["split"]) not in set(args.splits):
                    continue
                if args.limit is not None and sample_count >= args.limit:
                    break
                tensor_path = capture / str(meta["tensor_path"])
                with safe_open(tensor_path, framework="np", device="cpu") as handle:
                    states = np.asarray(handle.get_tensor("states"), dtype=np.float32)
                    layer_indices = np.asarray(handle.get_tensor("layer_indices"), dtype=np.int64)
                layer_offsets = {int(layer): offset for offset, layer in enumerate(layer_indices)}
                if states.shape[1] != len(meta["events"]):
                    raise RuntimeError(f"state/event mismatch in {tensor_path}")
                selected = selection.get(str(meta["request_id"]), {})
                last_position: dict[str, float] = {}
                trace_length = max(1.0, float(meta["sequence_length"]) - float(meta["input_tokens"]))
                for event_offset, event in enumerate(meta["events"]):
                    site = str(event.get("site", ""))
                    if site not in frozen or event.get("estimand") != "running_index":
                        continue
                    label = int(event["running_index"])
                    if not 1 <= label <= 10:
                        continue
                    geometry = frozen[site]
                    layer = int(geometry["layer"])
                    state = states[layer_offsets[layer], event_offset]
                    centroid = geometry["centroids"][label - 1]
                    residual = state - centroid
                    basis = geometry["basis"]
                    coordinates = residual @ basis
                    parallel = float(np.dot(coordinates, coordinates))
                    total = float(np.dot(residual, residual))
                    orthogonal = max(0.0, total - parallel)
                    hidden = residual.size
                    position = float(event["position"])
                    prior = last_position.get(site)
                    last_position[site] = position
                    row = {
                        "model_label": model,
                        "request_id": meta["request_id"],
                        "stimulus_id": meta["stimulus_id"],
                        "prompt_group": prompt_group(str(meta["stimulus_id"])),
                        "seed": meta["seed"],
                        "realization_id": meta["realization_id"],
                        "split": meta["split"],
                        "site": site,
                        "layer": layer,
                        "state_kind": event.get("state_kind", ""),
                        "marker_kind": event.get("marker_kind", meta.get("marker_kind", "")),
                        "termination_kind": meta.get("termination_kind", ""),
                        "trace_order_class": meta.get("trace_order_class", ""),
                        "trace_one_to_one": int(bool(meta.get("trace_one_to_one"))),
                        "gold_count": meta["gold_count"],
                        "item_count": meta["item_count"],
                        "running_index": label,
                        "count_progress": label / max(1, int(meta["gold_count"])),
                        "position": position,
                        "trace_position": position - float(meta["input_tokens"]),
                        "trace_progress": (position - float(meta["input_tokens"])) / trace_length,
                        "previous_same_site_gap": "" if prior is None else position - prior,
                        "token_span": int(event.get("token_end", 0)) - int(event.get("token_start", 0)),
                        "input_tokens": meta["input_tokens"],
                        "output_tokens": meta["intervention_output_tokens"],
                        "sequence_length": meta["sequence_length"],
                        "target_passage_tokens": meta["target_passage_tokens"],
                        "duplicate_gold_city_items": meta.get("duplicate_gold_city_items", 0),
                        "city_occurrences_in_item": event.get("city_occurrences_in_item", ""),
                        "baseline_correct": selected.get("baseline_correct", ""),
                        "cutoff_correct": selected.get("cutoff_correct", ""),
                        "noise_total": total,
                        "noise_parallel": parallel,
                        "noise_orthogonal": orthogonal,
                        "noise_total_rms": math.sqrt(total / hidden),
                        "noise_parallel_rms": math.sqrt(parallel / args.rank),
                        "noise_orthogonal_rms": math.sqrt(orthogonal / max(1, hidden - args.rank)),
                        "parallel_energy_fraction": parallel / max(total, 1e-12),
                        "count_axis_deviation": float(coordinates[0]),
                        "basis_rank": args.rank,
                        "centroid_variance_capture": geometry["centroid_variance_capture"],
                        "basis_fit_split": "discovery",
                    }
                    writer.writerow(row)
                    row_count += 1
                    site_counts[site] += 1
                sample_count += 1
                split = str(meta["split"])
                split_counts[split] = split_counts.get(split, 0) + 1
                if sample_count % args.progress_every == 0:
                    print(f"[{model}] samples={sample_count} rows={row_count}", flush=True)
        os.replace(partial, final)
        audit = {
            "model_label": model,
            "source_capture": str(capture),
            "source_read_only": True,
            "rank": args.rank,
            "site_layers": site_layers,
            "samples": sample_count,
            "rows": row_count,
            "samples_by_split": split_counts,
            "rows_by_site": site_counts,
            "output": str(final),
            "elapsed_seconds": time.perf_counter() - model_started,
            "status": "PASS",
        }
        (model_output / "trace_noise_extract_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        audits.append(audit)
        print(json.dumps(audit, indent=2), flush=True)
    campaign_audit = {
        "schema_version": "realistic_niah_v4_4_native_noise_extract_v1",
        "source_run": str(source),
        "models": audits,
        "elapsed_seconds": time.perf_counter() - started,
        "status": "PASS",
    }
    (output / "extract_audit.json").write_text(json.dumps(campaign_audit, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
