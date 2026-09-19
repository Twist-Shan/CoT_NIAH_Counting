#!/usr/bin/env python3
"""Capture paired native trace sites and non-thinking span_end states."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.modeling import (  # noqa: E402
    capture_span_states,
    load_registered_model,
)
from realistic_niah_v4.prompts import render_v4_prompt  # noqa: E402
from realistic_niah_v4.spec import V4Config, resolve_model_spec  # noqa: E402
from realistic_niah_v5.capture import capture_trace_shards  # noqa: E402
from realistic_niah_v5.pipeline import read_jsonl  # noqa: E402


SCHEMA = "realistic_niah_v5_pure_trace_n10_paired_capture_v2"


class SupplementCaptureConfig:
    primary_trace_site = "item_end"
    hidden_save_dtype = "float16"
    candidate_counts = tuple(range(1, 11))

    def __init__(self, site_kinds: Iterable[str]) -> None:
        registered = tuple(dict.fromkeys(map(str, site_kinds)))
        if "item_end" not in registered:
            raise ValueError("item_end is required for causal commit alignment")
        self.sensitivity_trace_sites = tuple(
            site for site in registered if site != self.primary_trace_site
        )

    @property
    def registered_sites(self) -> tuple[str, ...]:
        return (self.primary_trace_site,) + self.sensitivity_trace_sites

    def validate(self) -> None:
        return None


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_rows(path: Path, model: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            dict(row)
            for row in csv.DictReader(handle)
            if str(row.get("model_label")) == model
        ]
    rows.sort(key=lambda row: (str(row["split"]), int(row["seed"])))
    if not rows:
        raise ValueError(f"No selected supplement rows for {model}")
    if len({int(row["seed"]) for row in rows}) != len(rows):
        raise ValueError(f"Duplicate selected supplement seeds for {model}")
    return rows


def capture(args: argparse.Namespace) -> dict[str, Any]:
    selected = _selected_rows(args.selection, args.model)
    selected_seeds = {int(row["seed"]) for row in selected}
    split_by_seed = {int(row["seed"]): str(row["split"]) for row in selected}
    generations = [
        row
        for path in args.generations
        for row in read_jsonl(path)
        if int(row.get("seed", -1)) in selected_seeds
        and str(row.get("model_label", row.get("model"))) == args.model
    ]
    stimuli = [
        row
        for path in args.stimuli
        for row in read_jsonl(path)
        if int(row.get("seed", -1)) in selected_seeds
        and int(row.get("gold_count", -1)) == 10
    ]
    if len(generations) != len(selected) or len(stimuli) != len(selected):
        raise ValueError(
            f"Selected panel mismatch for {args.model}: selection={len(selected)} "
            f"generations={len(generations)} stimuli={len(stimuli)}"
        )
    for rows, label in ((generations, "generations"), (stimuli, "stimuli")):
        seeds = [int(row["seed"]) for row in rows]
        if len(set(seeds)) != len(seeds) or set(seeds) != selected_seeds:
            raise ValueError(f"{label} seed mismatch for {args.model}")
        for row in rows:
            row["split"] = split_by_seed[int(row["seed"])]

    model_spec = resolve_model_spec(args.model)
    model, tokenizer, adapter = load_registered_model(
        model_spec,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )
    model_root = args.output_root / args.model
    native_root = model_root / "native"
    non_root = model_root / "non_thinking"
    native_site_kinds = tuple(
        dict.fromkeys(("item_end", *map(str, args.native_site_kinds)))
    )
    print(
        f"[paired capture] {args.model} native sites={native_site_kinds} "
        f"rows={len(generations)}",
        flush=True,
    )
    native_index = capture_trace_shards(
        model,
        adapter,
        tokenizer,
        sorted(generations, key=lambda row: (str(row["split"]), int(row["seed"]))),
        config=SupplementCaptureConfig(native_site_kinds),
        output_dir=native_root,
        site_kinds=native_site_kinds,
        capture_span_pooling=False,
        overwrite=False,
    )

    discovery = tuple(sorted(seed for seed, split in split_by_seed.items() if split == "discovery"))
    confirmation = tuple(sorted(seed for seed, split in split_by_seed.items() if split == "confirmation"))
    v4_config = V4Config(
        seeds=discovery + confirmation,
        discovery_seeds=discovery,
        confirmation_seeds=confirmation,
    )
    v4_config.validate()
    non_index_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, stimulus in enumerate(
        sorted(stimuli, key=lambda row: (str(row["split"]), int(row["seed"]))),
        start=1,
    ):
        # The shared-backbone V5 export records its dataset protocol under
        # ``dataset_protocol_version``.  ``render_v4_prompt`` deliberately
        # requires the non-thinking prompt protocol under ``protocol_version``.
        # Adapt only that prompt-rendering field; the passage, slots, seed, and
        # split remain the exact paired V5 stimulus values.
        prompt_stimulus = dict(stimulus)
        prompt_stimulus["protocol_version"] = v4_config.protocol_version
        encoding = render_v4_prompt(
            prompt_stimulus,
            tokenizer=tokenizer,
            model_spec=model_spec,
            config=v4_config,
            answer_format="numeric",
        )
        if encoding.count != 10 or len(encoding.needle_spans) != 10:
            raise RuntimeError(f"Expected ten non-thinking spans for {encoding.stimulus_id}")
        relative = Path("shards") / f"seed_{encoding.seed}.npz"
        shard = non_root / relative
        if shard.is_file():
            with np.load(shard, allow_pickle=False) as archive:
                layer_indices = np.asarray(archive["layer_indices"])
                span_end = np.asarray(archive["span_end"])
            action = "reused"
        else:
            captured = capture_span_states(
                model,
                adapter,
                encoding,
                encoding.needle_spans,
            )
            layer_indices = captured["layer_indices"].numpy()
            span_end = captured["span_end"].numpy().astype(np.float16, copy=False)
            _save_npz(shard, layer_indices=layer_indices, span_end=span_end)
            action = "captured"
        expected_shape = (
            int(adapter.num_layers),
            10,
            int(model.get_input_embeddings().weight.shape[1]),
        )
        if tuple(span_end.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected non-thinking span_end {tuple(span_end.shape)}; "
                f"expected {expected_shape}"
            )
        non_index_rows.append(
            {
                "schema_version": SCHEMA,
                "model_label": args.model,
                "mode": "non_thinking",
                "design_variant": "v4.4",
                "stimulus_id": encoding.stimulus_id,
                "seed": int(encoding.seed),
                "split": str(encoding.split),
                "count": 10,
                "pooling": "span_end",
                "shard_path": relative.as_posix(),
            }
        )
        _atomic_jsonl(non_root / "capture_index.jsonl", non_index_rows)
        print(
            f"[paired capture] {args.model} non-thinking {index}/{len(stimuli)} "
            f"seed={encoding.seed} {action}",
            flush=True,
        )

    manifest = {
        "schema_version": SCHEMA,
        "model_label": args.model,
        "selection": str(args.selection),
        "selected_trajectories": len(selected),
        "discovery_trajectories": len(discovery),
        "confirmation_trajectories": len(confirmation),
        "native_site_kinds": list(native_site_kinds),
        "native_capture_index": str(native_index),
        "non_thinking_capture_index": str(non_root / "capture_index.jsonl"),
        "native_capture_index_sha256": _sha256(native_index),
        "non_thinking_capture_index_sha256": _sha256(non_root / "capture_index.jsonl"),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(model_root / "paired_capture_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--generations", type=Path, action="append", required=True)
    parser.add_argument("--stimuli", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend")
    parser.add_argument(
        "--native-site-kinds",
        nargs="+",
        default=("item_end",),
        help=(
            "Native sites to capture. item_end is always added so alternate "
            "sites can be audited against the causal progress commit."
        ),
    )
    args = parser.parse_args()
    capture(args)


if __name__ == "__main__":
    main()
