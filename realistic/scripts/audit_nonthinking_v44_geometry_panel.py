#!/usr/bin/env python3
"""Audit the exact V4.4 10-count x 30-seed non-thinking geometry panel."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np


MODELS = ("Qwen3-8B", "Gemma4-E4B")
COUNTS = tuple(range(1, 11))
DISCOVERY_SEEDS = tuple(range(1234, 1254))
CONFIRMATION_SEEDS = tuple(range(1254, 1264))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _expected_keys() -> set[tuple[str, int, int, str]]:
    keys: set[tuple[str, int, int, str]] = set()
    for split, seeds in (
        ("discovery", DISCOVERY_SEEDS),
        ("confirmation", CONFIRMATION_SEEDS),
    ):
        for seed in seeds:
            for count in COUNTS:
                stimulus_id = f"V4_4_T10000_N{count}_seed{seed}"
                keys.add((split, seed, count, stimulus_id))
    return keys


def _key(row: Mapping[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(row["split"]),
        int(row["seed"]),
        int(row["count"]),
        str(row["stimulus_id"]),
    )


def _validate_index(
    rows: list[dict[str, Any]], *, model: str, kind: str
) -> None:
    if len(rows) != 300:
        raise ValueError(f"{model}/{kind}: expected 300 rows, found {len(rows)}")
    keys = [_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{model}/{kind}: duplicate panel keys")
    if set(keys) != _expected_keys():
        missing = sorted(_expected_keys() - set(keys))[:5]
        extra = sorted(set(keys) - _expected_keys())[:5]
        raise ValueError(
            f"{model}/{kind}: key product mismatch; missing={missing}, extra={extra}"
        )
    for row in rows:
        if str(row["model_label"]) != model:
            raise ValueError(f"{model}/{kind}: mismatched model label")
        if str(row["design_variant"]) != "v4.4":
            raise ValueError(f"{model}/{kind}: non-v4.4 row")
        if str(row["answer_format"]) != "numeric":
            raise ValueError(f"{model}/{kind}: non-numeric answer format")


def _audit_running(
    index_path: Path, rows: list[dict[str, Any]], *, full_array_check: bool
) -> dict[str, Any]:
    layer_grid: tuple[int, ...] | None = None
    hidden_size: int | None = None
    bytes_total = 0
    for row in rows:
        shard = index_path.parent / str(row["shard_path"])
        if not shard.is_file():
            raise FileNotFoundError(shard)
        bytes_total += shard.stat().st_size
        with np.load(shard, allow_pickle=False) as saved:
            if set(saved.files) != {"layer_indices", "span_end", "span_mean"}:
                raise ValueError(f"Unexpected running shard fields: {shard}")
            layers = tuple(int(value) for value in saved["layer_indices"])
            span_end = np.asarray(saved["span_end"])
            span_mean = np.asarray(saved["span_mean"])
        expected_shape = tuple(int(value) for value in row["array_shape"])
        if span_end.shape != expected_shape or span_mean.shape != expected_shape:
            raise ValueError(f"Running shape mismatch: {shard}")
        if span_end.ndim != 3 or span_end.shape[1] != int(row["count"]):
            raise ValueError(f"Running count axis mismatch: {shard}")
        if layer_grid is None:
            layer_grid = layers
            hidden_size = int(span_end.shape[2])
        elif layers != layer_grid or int(span_end.shape[2]) != hidden_size:
            raise ValueError(f"Running decoder geometry mismatch: {shard}")
        if full_array_check and (
            not np.isfinite(span_end).all() or not np.isfinite(span_mean).all()
        ):
            raise ValueError(f"Non-finite running state: {shard}")
    return {
        "rows": len(rows),
        "layers": len(layer_grid or ()),
        "hidden_size": hidden_size,
        "shard_bytes": bytes_total,
    }


def _audit_final(
    index_path: Path, rows: list[dict[str, Any]], *, full_array_check: bool
) -> dict[str, Any]:
    layer_grid: tuple[int, ...] | None = None
    hidden_size: int | None = None
    bytes_total = 0
    for row in rows:
        shard = index_path.parent / str(row["shard_path"])
        if not shard.is_file():
            raise FileNotFoundError(shard)
        bytes_total += shard.stat().st_size
        with np.load(shard, allow_pickle=False) as saved:
            expected = {
                "layer_indices",
                "query_states",
                "query_position",
                "sequence_length",
            }
            if set(saved.files) != expected:
                raise ValueError(f"Unexpected final-count shard fields: {shard}")
            layers = tuple(int(value) for value in saved["layer_indices"])
            states = np.asarray(saved["query_states"])
            position = int(saved["query_position"][0])
            sequence_length = int(saved["sequence_length"][0])
        expected_shape = tuple(int(value) for value in row["array_shape"])
        if states.shape != expected_shape or states.ndim != 2:
            raise ValueError(f"Final-count shape mismatch: {shard}")
        if position != int(row["query_position"]):
            raise ValueError(f"Final-count query-position mismatch: {shard}")
        if sequence_length != int(row["sequence_length"]):
            raise ValueError(f"Final-count sequence-length mismatch: {shard}")
        if str(row["position"]) != "prompt_final_total_query":
            raise ValueError(f"Unexpected final-count position: {shard}")
        if layer_grid is None:
            layer_grid = layers
            hidden_size = int(states.shape[1])
        elif layers != layer_grid or int(states.shape[1]) != hidden_size:
            raise ValueError(f"Final-count decoder geometry mismatch: {shard}")
        if full_array_check and not np.isfinite(states).all():
            raise ValueError(f"Non-finite final-count state: {shard}")
    return {
        "rows": len(rows),
        "layers": len(layer_grid or ()),
        "hidden_size": hidden_size,
        "shard_bytes": bytes_total,
    }


def audit_panel(root: Path, *, full_array_check: bool) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model in MODELS:
        representation_root = root / model / "numeric" / "representation"
        running_index = representation_root / "capture" / "capture_index.jsonl"
        final_index = (
            representation_root
            / "answer_query_all_layers_v1"
            / "capture_index.jsonl"
        )
        running_rows = _read_jsonl(running_index)
        final_rows = _read_jsonl(final_index)
        _validate_index(running_rows, model=model, kind="running")
        _validate_index(final_rows, model=model, kind="final_count")
        if {_key(row) for row in running_rows} != {_key(row) for row in final_rows}:
            raise ValueError(f"{model}: running/final panels do not align")
        models[model] = {
            "running": _audit_running(
                running_index,
                running_rows,
                full_array_check=full_array_check,
            ),
            "final_count": _audit_final(
                final_index,
                final_rows,
                full_array_check=full_array_check,
            ),
            "split_counts": dict(
                sorted(Counter(str(row["split"]) for row in running_rows).items())
            ),
            "count_counts": dict(
                sorted(Counter(int(row["count"]) for row in running_rows).items())
            ),
            "key_alignment": "exact",
        }
    return {
        "schema_version": "realistic_niah_v44_geometry_panel_audit_v1",
        "root": str(root.resolve()),
        "expected_panel": "10 counts x 30 seeds = 300 trajectories per model",
        "full_array_check": bool(full_array_check),
        "models": models,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--full-array-check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = audit_panel(args.root, full_array_check=args.full_array_check)
    payload = json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(args.output)
    print(payload, end="")


if __name__ == "__main__":
    main()
