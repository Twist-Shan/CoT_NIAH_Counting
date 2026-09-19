from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from realistic_niah_v4_4_2.aggregate import (
    _capture_lookup,
    _hidden_vector,
    _read_json,
)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(path)


def _load_attention(capture_dir: Path) -> dict[str, Any]:
    return torch.load(
        capture_dir / "attention_summary.pt",
        map_location="cpu",
        weights_only=True,
    )


def _update_array_stats(
    stats: dict[tuple[str, int], dict[str, np.ndarray]],
    key: tuple[str, int],
    value: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> None:
    array = value.detach().cpu().numpy().astype(np.float64, copy=False)
    if valid is None:
        mask = np.ones(array.shape, dtype=bool)
    else:
        valid_array = valid.detach().cpu().numpy().astype(bool, copy=False)
        while valid_array.ndim < array.ndim:
            valid_array = valid_array[..., None]
        mask = np.broadcast_to(valid_array, array.shape)
    slot = stats.get(key)
    if slot is None:
        slot = {
            "sum": np.zeros_like(array, dtype=np.float64),
            "sumsq": np.zeros_like(array, dtype=np.float64),
            "count": np.zeros_like(array, dtype=np.int64),
        }
        stats[key] = slot
    slot["sum"][mask] += array[mask]
    slot["sumsq"][mask] += array[mask] ** 2
    slot["count"][mask] += 1


def _finalize_stats(
    stats: dict[tuple[str, int], dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for (model, layer), slot in sorted(stats.items()):
        count = slot["count"]
        mean = np.divide(
            slot["sum"],
            count,
            out=np.full_like(slot["sum"], np.nan),
            where=count > 0,
        )
        variance = np.divide(
            slot["sumsq"],
            count,
            out=np.full_like(slot["sum"], np.nan),
            where=count > 0,
        ) - mean**2
        variance = np.maximum(variance, 0.0)
        std = np.sqrt(variance)
        prefix = f"{model}__layer_{layer:02d}"
        output[f"{prefix}__mean"] = mean.astype(np.float32)
        output[f"{prefix}__std"] = std.astype(np.float32)
        output[f"{prefix}__count"] = count.astype(np.int32)
    return output


def _occupied_bins(size: int, bins: int) -> torch.Tensor:
    result = torch.zeros(bins, dtype=torch.bool)
    if size <= 0:
        return result
    positions = torch.arange(size, dtype=torch.long)
    indices = torch.clamp((positions * bins) // size, max=bins - 1)
    result[indices] = True
    return result


def analyze(root: Path, output: Path) -> dict[str, Any]:
    lookup = _capture_lookup(root)
    identities = sorted({(key[0], key[1]) for key in lookup})
    modes = ("nonthinking", "native_thinking")
    hidden_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    head_accumulators: dict[
        tuple[str, str, int, int, str], dict[str, float]
    ] = defaultdict(lambda: {"sum": 0.0, "sumsq": 0.0, "positive": 0.0, "n": 0.0})
    trace_map_stats: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    trace_to_trace_stats: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    coverage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    region_names: tuple[str, ...] | None = None

    for model, stimulus_id in identities:
        for mode in modes:
            present_key = (model, stimulus_id, mode, "cue_present")
            absent_key = (model, stimulus_id, mode, "cue_absent")
            if present_key not in lookup or absent_key not in lookup:
                continue
            present_dir = lookup[present_key]
            absent_dir = lookup[absent_key]
            present_manifest = _read_json(present_dir / "capture_manifest.json")
            absent_manifest = _read_json(absent_dir / "capture_manifest.json")
            present_attention = _load_attention(present_dir)
            absent_attention = _load_attention(absent_dir)
            if tuple(present_attention["region_names"]) != tuple(
                absent_attention["region_names"]
            ):
                raise RuntimeError("Region schema changed across cue conditions")
            current_regions = tuple(str(value) for value in present_attention["region_names"])
            if region_names is None:
                region_names = current_regions
            elif region_names != current_regions:
                raise RuntimeError("Region schema changed across models")
            present_layers = {int(value["layer"]) for value in present_manifest["layers"]}
            absent_layers = {int(value["layer"]) for value in absent_manifest["layers"]}
            attention_layers = set(present_attention["layers"]) & set(
                absent_attention["layers"]
            )
            common_layers = sorted(present_layers & absent_layers & attention_layers)
            coverage[model][mode] += 1

            for layer in common_layers:
                present_hidden = _hidden_vector(
                    present_dir,
                    present_manifest,
                    layer=layer,
                    site="answer_query",
                )
                absent_hidden = _hidden_vector(
                    absent_dir,
                    absent_manifest,
                    layer=layer,
                    site="answer_query",
                )
                if present_hidden is None or absent_hidden is None:
                    continue
                hidden_delta = absent_hidden - present_hidden
                hidden_rows.append(
                    {
                        "model_label": model,
                        "mode": mode,
                        "stimulus_id": stimulus_id,
                        "seed": int(present_manifest["seed"]),
                        "gold_count": int(present_manifest["gold_count"]),
                        "layer": layer,
                        "site": "answer_query",
                        "cosine": float(
                            F.cosine_similarity(
                                present_hidden[None], absent_hidden[None]
                            )
                        ),
                        "relative_l2_delta": float(
                            hidden_delta.norm()
                            / present_hidden.norm().clamp_min(1e-12)
                        ),
                        "norm_delta": float(
                            absent_hidden.norm() - present_hidden.norm()
                        ),
                    }
                )

                present_layer = present_attention["layers"][layer]
                absent_layer = absent_attention["layers"][layer]
                answer_delta = (
                    absent_layer["answer_query_last_region"].float()
                    - present_layer["answer_query_last_region"].float()
                )
                if answer_delta.shape[1] != len(current_regions):
                    raise RuntimeError("Answer attention region dimension changed")
                for region_index, region in enumerate(current_regions):
                    values = answer_delta[:, region_index]
                    answer_rows.append(
                        {
                            "model_label": model,
                            "mode": mode,
                            "stimulus_id": stimulus_id,
                            "seed": int(present_manifest["seed"]),
                            "gold_count": int(present_manifest["gold_count"]),
                            "layer": layer,
                            "region": region,
                            "mean_head_mass_delta": float(values.mean()),
                            "std_head_mass_delta": float(values.std(unbiased=False)),
                        }
                    )
                    for head, value in enumerate(values.tolist()):
                        key = (model, mode, layer, head, region)
                        slot = head_accumulators[key]
                        slot["sum"] += float(value)
                        slot["sumsq"] += float(value) ** 2
                        slot["positive"] += float(value > 0)
                        slot["n"] += 1.0

                if mode != "native_thinking":
                    continue
                trace_head_delta = (
                    absent_layer["head_region_mean"].float()
                    - present_layer["head_region_mean"].float()
                )
                for region_index, region in enumerate(current_regions):
                    values = trace_head_delta[:, region_index]
                    trace_rows.append(
                        {
                            "model_label": model,
                            "stimulus_id": stimulus_id,
                            "seed": int(present_manifest["seed"]),
                            "gold_count": int(present_manifest["gold_count"]),
                            "layer": layer,
                            "region": region,
                            "mean_head_mass_delta": float(values.mean()),
                            "std_head_mass_delta": float(values.std(unbiased=False)),
                        }
                    )
                trace_map_delta = (
                    absent_layer["trace_region_map"].float()
                    - present_layer["trace_region_map"].float()
                )
                query_valid = (
                    absent_layer["trace_query_bin_counts"] > 0
                ) & (present_layer["trace_query_bin_counts"] > 0)
                _update_array_stats(
                    trace_map_stats,
                    (model, layer),
                    trace_map_delta,
                    query_valid,
                )
                trace_to_trace_delta = (
                    absent_layer["trace_to_trace_map"].float()
                    - present_layer["trace_to_trace_map"].float()
                )
                present_trace_length = int(
                    present_manifest["boundaries"]["trace_end"]
                    - present_manifest["boundaries"]["trace_start"]
                )
                absent_trace_length = int(
                    absent_manifest["boundaries"]["trace_end"]
                    - absent_manifest["boundaries"]["trace_start"]
                )
                bins = trace_to_trace_delta.shape[0]
                key_valid = _occupied_bins(
                    present_trace_length, bins
                ) & _occupied_bins(absent_trace_length, bins)
                matrix_valid = query_valid[:, None] & key_valid[None, :]
                _update_array_stats(
                    trace_to_trace_stats,
                    (model, layer),
                    trace_to_trace_delta,
                    matrix_valid,
                )

    head_rows: list[dict[str, Any]] = []
    for (model, mode, layer, head, region), slot in sorted(
        head_accumulators.items()
    ):
        n = slot["n"]
        mean = slot["sum"] / n
        variance = max(0.0, slot["sumsq"] / n - mean**2)
        head_rows.append(
            {
                "model_label": model,
                "mode": mode,
                "layer": layer,
                "head": head,
                "region": region,
                "stimuli": int(n),
                "mean_mass_delta": mean,
                "std_mass_delta": variance**0.5,
                "positive_fraction": slot["positive"] / n,
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    _atomic_csv(pd.DataFrame(hidden_rows), output / "hidden_cue_effect.csv.gz")
    _atomic_csv(
        pd.DataFrame(answer_rows), output / "answer_attention_cue_effect.csv.gz"
    )
    _atomic_csv(
        pd.DataFrame(trace_rows), output / "native_trace_attention_cue_effect.csv.gz"
    )
    _atomic_csv(
        pd.DataFrame(head_rows), output / "answer_attention_head_cue_effect.csv.gz"
    )
    np.savez_compressed(
        output / "native_trace_region_map_cue_effect.npz",
        **_finalize_stats(trace_map_stats),
    )
    np.savez_compressed(
        output / "native_trace_to_trace_cue_effect.npz",
        **_finalize_stats(trace_to_trace_stats),
    )
    metadata = {
        "schema_version": "realistic_niah_v4_4_2_cue_effect_posthoc_v1",
        "contrast": "cue_absent_minus_cue_present",
        "coverage": {model: dict(values) for model, values in coverage.items()},
        "region_names": list(region_names or ()),
        "hidden_rows": len(hidden_rows),
        "answer_attention_rows": len(answer_rows),
        "native_trace_attention_rows": len(trace_rows),
        "head_summary_rows": len(head_rows),
    }
    temporary = output / "metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary.replace(output / "metadata.json")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(Path(args.run_root), Path(args.output_dir)),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
