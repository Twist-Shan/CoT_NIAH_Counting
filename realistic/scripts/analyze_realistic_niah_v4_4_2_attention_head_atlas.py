from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


MODES = ("nonthinking", "native_thinking")
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


def rounded_nested(array: np.ndarray, digits: int = 7) -> list[Any]:
    rounded = np.round(array.astype(np.float64, copy=False), digits)
    flat = [
        None if not np.isfinite(value) else float(value)
        for value in rounded.reshape(-1)
    ]
    return np.asarray(flat, dtype=object).reshape(array.shape).tolist()


def analyze(root: Path, output: Path) -> dict[str, Any]:
    entries: list[tuple[str, str, str, Path, dict[str, Any]]] = []
    for mode in MODES:
        pattern = f"conditions/*/*/{mode}/**/capture/capture_manifest.json"
        for manifest_path in sorted(root.glob(pattern)):
            manifest = read_json(manifest_path)
            condition = str(manifest["prompt_variant"])
            if condition not in CONDITIONS:
                continue
            entries.append(
                (
                    str(manifest["model_label"]),
                    mode,
                    condition,
                    manifest_path.parent,
                    manifest,
                )
            )

    models = sorted({row[0] for row in entries})
    region_names: tuple[str, ...] | None = None
    payload: dict[str, Any] = {
        "schema_version": "realistic_niah_v4_4_2_attention_head_atlas_v1",
        "query": "last answer-query row (Total:)",
        "aggregation": "arithmetic mean over samples; head identity retained",
        "conditions": list(CONDITIONS),
        "modes": list(MODES),
        "models": {},
    }

    for model in models:
        model_payload: dict[str, Any] = {"modes": {}}
        for mode in MODES:
            mode_entries = [row for row in entries if row[0] == model and row[1] == mode]
            if not mode_entries:
                continue
            layers = sorted(
                {
                    int(layer["layer"])
                    for *_, manifest in mode_entries
                    for layer in manifest["layers"]
                }
            )
            mode_payload: dict[str, Any] = {
                "layers": layers,
                "conditions": {},
            }
            expected_heads: int | None = None
            for condition in CONDITIONS:
                rows = [row for row in mode_entries if row[2] == condition]
                sums: dict[int, np.ndarray] = {}
                counts: dict[int, np.ndarray] = {}
                for _, _, _, capture_dir, _ in rows:
                    attention = torch.load(
                        capture_dir / "attention_summary.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    current_regions = tuple(
                        str(value) for value in attention["region_names"]
                    )
                    if region_names is None:
                        region_names = current_regions
                    elif current_regions != region_names:
                        raise RuntimeError("Attention region schema changed")
                    for layer, layer_data in attention["layers"].items():
                        layer = int(layer)
                        values = (
                            layer_data["answer_query_last_region"]
                            .float()
                            .numpy()
                            .astype(np.float64, copy=False)
                        )
                        if expected_heads is None:
                            expected_heads = values.shape[0]
                        elif values.shape[0] != expected_heads:
                            raise RuntimeError(
                                f"Head count changed for {model}/{mode}"
                            )
                        if layer not in sums:
                            sums[layer] = np.zeros_like(values)
                            counts[layer] = np.zeros_like(values, dtype=np.int64)
                        valid = np.isfinite(values)
                        sums[layer][valid] += values[valid]
                        counts[layer][valid] += 1
                atlas = []
                for layer in layers:
                    mean = np.divide(
                        sums[layer],
                        counts[layer],
                        out=np.full_like(sums[layer], np.nan),
                        where=counts[layer] > 0,
                    )
                    atlas.append(rounded_nested(mean))
                mode_payload["conditions"][condition] = {
                    "samples": len(rows),
                    "layer_head_region": atlas,
                }
            mode_payload["heads"] = expected_heads or 0
            model_payload["modes"][mode] = mode_payload
            print(
                f"[attention-head-atlas] {model} {mode}: "
                f"{len(layers)} layers x {expected_heads or 0} heads",
                flush=True,
            )
        payload["models"][model] = model_payload

    payload["region_names"] = list(region_names or ())
    write_json(output, payload)
    summary = {
        "models": models,
        "modes": list(MODES),
        "regions": list(region_names or ()),
        "output": str(output),
        "bytes": output.stat().st_size,
    }
    write_json(output.with_name("attention_head_atlas_summary.json"), summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = analyze(arguments.run_root, arguments.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
