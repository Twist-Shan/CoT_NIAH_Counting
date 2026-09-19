from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


CONDITIONS = ("cue_present", "cue_absent")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(path)


def occupied_bins(size: int, bins: int) -> np.ndarray:
    result = np.zeros(bins, dtype=bool)
    if size <= 0:
        return result
    positions = np.arange(size, dtype=np.int64)
    indices = np.minimum((positions * bins) // size, bins - 1)
    result[indices] = True
    return result


def safe_values(array: np.ndarray, digits: int = 7) -> list[Any]:
    rounded = np.round(array.astype(np.float64, copy=False), digits)
    return [
        None if not np.isfinite(value) else float(value)
        for value in rounded.reshape(-1)
    ]


def reshape_values(flat: list[Any], shape: tuple[int, ...]) -> Any:
    array = np.asarray(flat, dtype=object).reshape(shape)
    return array.tolist()


def block_reduce(
    sums: np.ndarray,
    counts: np.ndarray,
    *,
    query_bins: int,
    key_bins: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    source_query_bins = sums.shape[0]
    if source_query_bins % query_bins:
        raise RuntimeError("Query bins do not divide source bins")
    query_factor = source_query_bins // query_bins
    if key_bins is None:
        new_shape = (query_bins, query_factor, *sums.shape[1:])
        return sums.reshape(new_shape).sum(axis=1), counts.reshape(new_shape).sum(axis=1)
    source_key_bins = sums.shape[1]
    if source_key_bins % key_bins:
        raise RuntimeError("Key bins do not divide source bins")
    key_factor = source_key_bins // key_bins
    new_shape = (
        query_bins,
        query_factor,
        key_bins,
        key_factor,
        *sums.shape[2:],
    )
    return (
        sums.reshape(new_shape).sum(axis=(1, 3)),
        counts.reshape(new_shape).sum(axis=(1, 3)),
    )


def analyze(
    root: Path,
    output: Path,
    *,
    trace_bins: int,
    trace_to_trace_bins: int,
) -> dict[str, Any]:
    manifests = []
    pattern = "conditions/*/*/native_thinking/**/capture/capture_manifest.json"
    for path in sorted(root.glob(pattern)):
        manifest = read_json(path)
        if str(manifest["prompt_variant"]) not in CONDITIONS:
            continue
        manifests.append((path.parent, manifest))
    models = sorted({str(manifest["model_label"]) for _, manifest in manifests})
    payload: dict[str, Any] = {
        "schema_version": "realistic_niah_v4_4_2_attention_condition_maps_v1",
        "conditions": list(CONDITIONS),
        "trace_bins": trace_bins,
        "trace_to_trace_bins": trace_to_trace_bins,
        "models": {},
    }
    region_names: tuple[str, ...] | None = None

    for model in models:
        model_rows = [
            (capture_dir, manifest)
            for capture_dir, manifest in manifests
            if str(manifest["model_label"]) == model
        ]
        layers = sorted(
            {
                int(layer["layer"])
                for _, manifest in model_rows
                for layer in manifest["layers"]
            }
        )
        model_payload: dict[str, Any] = {"layers": layers, "conditions": {}}
        for condition in CONDITIONS:
            rows = [
                (capture_dir, manifest)
                for capture_dir, manifest in model_rows
                if str(manifest["prompt_variant"]) == condition
            ]
            answer_sum: dict[int, np.ndarray] = {}
            answer_count: dict[int, np.ndarray] = {}
            trace_sum: dict[int, np.ndarray] = {}
            trace_count: dict[int, np.ndarray] = {}
            ttt_sum: dict[int, np.ndarray] = {}
            ttt_count: dict[int, np.ndarray] = {}
            for capture_dir, manifest in rows:
                attention = torch.load(
                    capture_dir / "attention_summary.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                current_regions = tuple(str(value) for value in attention["region_names"])
                if region_names is None:
                    region_names = current_regions
                elif current_regions != region_names:
                    raise RuntimeError("Attention region schema changed")
                trace_length = int(
                    manifest["boundaries"]["trace_end"]
                    - manifest["boundaries"]["trace_start"]
                )
                for layer, layer_data in attention["layers"].items():
                    layer = int(layer)
                    answer = (
                        layer_data["answer_query_last_region"]
                        .float()
                        .mean(dim=0)
                        .numpy()
                        .astype(np.float64, copy=False)
                    )
                    if layer not in answer_sum:
                        answer_sum[layer] = np.zeros_like(answer)
                        answer_count[layer] = np.zeros_like(answer, dtype=np.int64)
                    valid_answer = np.isfinite(answer)
                    answer_sum[layer][valid_answer] += answer[valid_answer]
                    answer_count[layer][valid_answer] += 1

                    trace_map = (
                        layer_data["trace_region_map"]
                        .float()
                        .numpy()
                        .astype(np.float64, copy=False)
                    )
                    query_valid = (
                        layer_data["trace_query_bin_counts"].numpy() > 0
                    )
                    trace_valid = np.broadcast_to(
                        query_valid[:, None], trace_map.shape
                    ) & np.isfinite(trace_map)
                    if layer not in trace_sum:
                        trace_sum[layer] = np.zeros_like(trace_map)
                        trace_count[layer] = np.zeros_like(trace_map, dtype=np.int64)
                    trace_sum[layer][trace_valid] += trace_map[trace_valid]
                    trace_count[layer][trace_valid] += 1

                    trace_to_trace = (
                        layer_data["trace_to_trace_map"]
                        .float()
                        .numpy()
                        .astype(np.float64, copy=False)
                    )
                    key_valid = occupied_bins(trace_length, trace_to_trace.shape[1])
                    matrix_valid = (
                        query_valid[:, None] & key_valid[None, :]
                    ) & np.isfinite(trace_to_trace)
                    if layer not in ttt_sum:
                        ttt_sum[layer] = np.zeros_like(trace_to_trace)
                        ttt_count[layer] = np.zeros_like(
                            trace_to_trace, dtype=np.int64
                        )
                    ttt_sum[layer][matrix_valid] += trace_to_trace[matrix_valid]
                    ttt_count[layer][matrix_valid] += 1

            answer_maps: list[Any] = []
            trace_maps: list[Any] = []
            ttt_maps: list[Any] = []
            for layer in layers:
                answer_mean = np.divide(
                    answer_sum[layer],
                    answer_count[layer],
                    out=np.full_like(answer_sum[layer], np.nan),
                    where=answer_count[layer] > 0,
                )
                answer_maps.append(
                    reshape_values(
                        safe_values(answer_mean), answer_mean.shape
                    )
                )
                reduced_trace_sum, reduced_trace_count = block_reduce(
                    trace_sum[layer],
                    trace_count[layer],
                    query_bins=trace_bins,
                )
                trace_mean = np.divide(
                    reduced_trace_sum,
                    reduced_trace_count,
                    out=np.full_like(reduced_trace_sum, np.nan),
                    where=reduced_trace_count > 0,
                )
                trace_maps.append(
                    reshape_values(safe_values(trace_mean), trace_mean.shape)
                )
                reduced_ttt_sum, reduced_ttt_count = block_reduce(
                    ttt_sum[layer],
                    ttt_count[layer],
                    query_bins=trace_to_trace_bins,
                    key_bins=trace_to_trace_bins,
                )
                ttt_mean = np.divide(
                    reduced_ttt_sum,
                    reduced_ttt_count,
                    out=np.full_like(reduced_ttt_sum, np.nan),
                    where=reduced_ttt_count > 0,
                )
                ttt_maps.append(
                    reshape_values(safe_values(ttt_mean), ttt_mean.shape)
                )
            model_payload["conditions"][condition] = {
                "samples": len(rows),
                "answer_layer_region": answer_maps,
                "trace_time_region": trace_maps,
                "trace_to_trace": ttt_maps,
            }
        payload["models"][model] = model_payload
    payload["region_names"] = list(region_names or ())
    write_json(output, payload)
    summary = {
        "models": models,
        "regions": list(region_names or ()),
        "trace_bins": trace_bins,
        "trace_to_trace_bins": trace_to_trace_bins,
        "output": str(output),
        "bytes": output.stat().st_size,
    }
    write_json(output.with_name("attention_condition_maps_summary.json"), summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-bins", type=int, default=64)
    parser.add_argument("--trace-to-trace-bins", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = analyze(
        arguments.run_root,
        arguments.output,
        trace_bins=arguments.trace_bins,
        trace_to_trace_bins=arguments.trace_to_trace_bins,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
