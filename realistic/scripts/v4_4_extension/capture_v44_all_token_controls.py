from __future__ import annotations

"""Capture selected non-needle and needle token states for the V4.4 N=10 prompts.

This intentionally samples tokens rather than materializing every layer-by-token
activation.  All needle and hard-negative tokens are retained; ordinary passage
tokens are sampled deterministically and stratified across prompt depth.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from realistic_niah_v4.modeling import capture_post_block_states, load_registered_model
from realistic_niah_v4.prompts import PromptEncoding, render_v4_prompt
from realistic_niah_v4.spec import V4Config, resolve_model_spec
from realistic_niah_v4.stimuli import load_stimuli


MODEL_LAYERS = {
    "Qwen3-8B": (0, 4, 8, 12, 16, 20, 24, 27, 28, 29, 32, 35),
    "Gemma4-E4B": (0, 5, 9, 11, 17, 22, 23, 29, 35, 36, 37, 41),
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def sample_positions(encoding: PromptEncoding, ordinary_count: int) -> dict[str, np.ndarray]:
    labels: dict[int, str] = {}
    occurrence: dict[int, int] = {}
    for index, span in enumerate(encoding.needle_spans, start=1):
        for position in range(int(span.start), int(span.end)):
            labels[position] = "needle_endpoint" if position == int(span.end) - 1 else "needle_interior"
            occurrence[position] = index
    for span in encoding.hard_negative_spans:
        for position in range(int(span.start), int(span.end)):
            labels.setdefault(position, "hard_negative")
            occurrence.setdefault(position, 0)

    occupied = set(labels)
    all_spans = tuple(encoding.needle_spans) + tuple(encoding.hard_negative_spans)
    passage_start = max(1, min(int(span.start) for span in all_spans) - 64)
    passage_end = min(int(encoding.query_position), max(int(span.end) for span in all_spans) + 64)
    candidates = np.asarray([p for p in range(passage_start, passage_end) if p not in occupied], dtype=np.int64)
    if len(candidates) < ordinary_count:
        raise RuntimeError("Not enough ordinary passage tokens for stratified sample")
    raw_indices = np.linspace(0, len(candidates) - 1, ordinary_count)
    selected_indices = np.unique(np.rint(raw_indices).astype(np.int64))
    if len(selected_indices) != ordinary_count:
        remaining = [i for i in range(len(candidates)) if i not in set(selected_indices.tolist())]
        selected_indices = np.sort(np.concatenate([selected_indices, np.asarray(remaining[: ordinary_count - len(selected_indices)])]))
    for position in candidates[selected_indices]:
        labels[int(position)] = "ordinary_passage"
        occurrence[int(position)] = 0

    positions = np.asarray(sorted(labels), dtype=np.int64)
    categories = np.asarray([labels[int(p)] for p in positions])
    occurrence_index = np.asarray([occurrence[int(p)] for p in positions], dtype=np.int16)
    prefix_count = np.asarray(
        [sum(int(span.end) - 1 <= int(p) for span in encoding.needle_spans) for p in positions],
        dtype=np.int16,
    )
    return {
        "positions": positions,
        "categories": categories,
        "occurrence_index": occurrence_index,
        "prefix_count": prefix_count,
        "token_ids": np.asarray([encoding.input_ids[int(p)] for p in positions], dtype=np.int64),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4-config", type=Path, required=True)
    parser.add_argument("--stimuli", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["Qwen3-8B", "Gemma4-E4B"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1234, 1264)))
    parser.add_argument("--ordinary-tokens", type=int, default=128)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = V4Config.from_json(args.v4_config)
    source = {
        (int(row["seed"]), int(row["gold_count"])): row
        for row in load_stimuli(args.stimuli)
        if str(row.get("design_variant")) == "v4.4" and int(row.get("gold_count", -1)) == 10
    }
    args.output.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    for model_label in args.models:
        spec = resolve_model_spec(model_label)
        model, tokenizer, adapter = load_registered_model(
            spec,
            cache_dir=args.cache_dir,
            device_map=args.device_map,
            torch_dtype=config.model_torch_dtype,
            attention_backend=config.attention_prefix_backend,
        )
        layers = MODEL_LAYERS[model_label]
        for seed in args.seeds:
            encoding = render_v4_prompt(
                source[(int(seed), 10)],
                tokenizer=tokenizer,
                model_spec=spec,
                config=config,
                answer_format="numeric",
            )
            meta = sample_positions(encoding, args.ordinary_tokens)
            relative = Path(model_label) / "shards" / f"seed_{seed}.npz"
            path = args.output / relative
            if path.exists() and not args.overwrite:
                with np.load(path, allow_pickle=False) as z:
                    states = np.asarray(z["states"])
                    if tuple(states.shape[:2]) != (len(layers), len(meta["positions"])):
                        raise RuntimeError(f"Existing shard shape mismatch: {path}")
            else:
                _logits, captured = capture_post_block_states(
                    model,
                    adapter,
                    encoding,
                    meta["positions"].tolist(),
                    layers=layers,
                )
                states = np.stack([captured[layer].numpy() for layer in layers]).astype(np.float16)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + ".tmp")
                with tmp.open("wb") as handle:
                    np.savez(
                        handle,
                        states=states,
                        layer_indices=np.asarray(layers, dtype=np.int16),
                        **meta,
                    )
                tmp.replace(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            index_rows.append(
                {
                    "model_label": model_label,
                    "seed": int(seed),
                    "split": "discovery" if seed in config.discovery_seeds else "confirmation",
                    "stimulus_id": encoding.stimulus_id,
                    "sequence_length": encoding.sequence_length,
                    "layers": list(layers),
                    "positions": len(meta["positions"]),
                    "category_counts": {key: int((meta["categories"] == key).sum()) for key in np.unique(meta["categories"])},
                    "path": relative.as_posix(),
                    "sha256": digest,
                }
            )
            print(f"[all-token] {model_label} seed={seed} layers={len(layers)} positions={len(meta['positions'])}", flush=True)
        del model, tokenizer, adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    index_path = args.output / "capture_index.jsonl"
    index_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in index_rows), encoding="utf-8")
    atomic_json(
        args.output / "capture_manifest.json",
        {
            "schema_version": "realistic_niah_v4_4_all_token_control_capture_v1",
            "models": args.models,
            "seeds": args.seeds,
            "ordinary_tokens_per_prompt": args.ordinary_tokens,
            "selection": "all needle/hard-negative tokens plus deterministic depth-stratified ordinary passage tokens",
            "rows": len(index_rows),
            "full_sequence_materialized": False,
            "save_dtype": "float16",
            "status": "PASS",
        },
    )


if __name__ == "__main__":
    main()
