from __future__ import annotations

"""Paired V4.4 token-corruption test with equal-token-count passage controls."""

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

FALLBACK_SRC = Path("runs/v4_4_counter_channel_20260806/code/src")
if FALLBACK_SRC.is_dir() and str(FALLBACK_SRC) not in sys.path:
    sys.path.insert(0, str(FALLBACK_SRC))

from realistic_niah_v4.modeling import (
    _is_prompt_prefill,
    _tensor_from_output,
    generate_answer_completion,
    load_registered_model,
)
from realistic_niah_v4.behavior import parse_numeric_completion
from realistic_niah_v4.prompts import PromptEncoding, render_v4_prompt
from realistic_niah_v4.spec import V4Config, resolve_model_spec
from realistic_niah_v4.stimuli import load_stimuli


CAPTURE_LAYERS = {
    "Qwen3-8B": (29, 35),
    "Gemma4-E4B": (37, 41),
}


def parsed_prediction(result: dict[str, Any]) -> int | None:
    """Apply the registered strict numeric parser to a generated continuation."""

    labels = parse_numeric_completion(str(result.get("completion_text", "")))
    value = labels["parsed_count"]
    return int(value) if value is not None else None


def ordinary_segments(encoding: PromptEncoding) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    lengths = [int(span.end) - int(span.start) for span in encoding.needle_spans]
    all_spans = tuple(encoding.slot_spans) + tuple(encoding.hard_negative_spans)
    forbidden: set[int] = set()
    for span in all_spans:
        forbidden.update(range(int(span.start), int(span.end)))
    start = max(1, min(int(span.start) for span in all_spans) - 64)
    end = min(int(encoding.query_position), max(int(span.end) for span in all_spans) + 64)
    used = set(forbidden)

    def allocate(length: int, phase: int) -> tuple[int, int]:
        candidates = range(start + phase, max(start + phase, end - length), max(1, length // 2))
        for candidate in candidates:
            positions = set(range(candidate, candidate + length))
            if candidate + length <= end and not positions.intersection(used):
                used.update(positions)
                return candidate, candidate + length
        for candidate in range(start, end - length):
            positions = set(range(candidate, candidate + length))
            if not positions.intersection(used):
                used.update(positions)
                return candidate, candidate + length
        raise RuntimeError("Could not allocate length-matched ordinary passage segment")

    needle_sources = [allocate(length, 0) for length in lengths]
    control_targets = [allocate(length, 1) for length in lengths]
    control_sources = [allocate(length, 2) for length in lengths]
    return needle_sources, control_targets, control_sources


def corrupt_encodings(encoding: PromptEncoding) -> tuple[PromptEncoding, PromptEncoding, dict[str, Any]]:
    needle_sources, control_targets, control_sources = ordinary_segments(encoding)
    clean = list(encoding.input_ids)
    needle_ids = clean.copy()
    control_ids = clean.copy()
    needle_changed = 0
    control_changed = 0
    for span, source in zip(encoding.needle_spans, needle_sources):
        length = int(span.end) - int(span.start)
        replacement_ids = clean[source[0] : source[1]]
        before = needle_ids[int(span.start) : int(span.end)]
        needle_ids[int(span.start) : int(span.end)] = replacement_ids
        needle_changed += sum(a != b for a, b in zip(before, replacement_ids))
        if len(replacement_ids) != length:
            raise RuntimeError("Needle replacement length changed")
    for target, source in zip(control_targets, control_sources):
        replacement_ids = clean[source[0] : source[1]]
        before = control_ids[target[0] : target[1]]
        control_ids[target[0] : target[1]] = replacement_ids
        control_changed += sum(a != b for a, b in zip(before, replacement_ids))
    token_budget = sum(int(span.end) - int(span.start) for span in encoding.needle_spans)
    return (
        replace(encoding, input_ids=tuple(needle_ids)),
        replace(encoding, input_ids=tuple(control_ids)),
        {
            "token_budget": token_budget,
            "needle_changed_tokens": needle_changed,
            "control_changed_tokens": control_changed,
            "needle_sources": needle_sources,
            "control_targets": control_targets,
            "control_sources": control_sources,
        },
    )


@torch.inference_mode()
def generate_and_capture(model: Any, tokenizer: Any, adapter: Any, encoding: PromptEncoding, layers: tuple[int, ...]) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    captured: dict[int, np.ndarray] = {}
    handles = []
    for layer in layers:
        def hook(_module, _args, output, *, layer=layer):
            hidden = _tensor_from_output(output)
            if _is_prompt_prefill(hidden, encoding):
                captured[layer] = hidden[0, int(encoding.query_position)].detach().float().cpu().numpy()
        handles.append(adapter.layers[layer].register_forward_hook(hook))
    try:
        result = generate_answer_completion(model, tokenizer, encoding, max_new_tokens=8)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(layers):
        raise RuntimeError(f"Missing answer state captures: {set(layers) - set(captured)}")
    return result, captured


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4-config", type=Path, required=True)
    parser.add_argument("--stimuli", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["Qwen3-8B", "Gemma4-E4B"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1254, 1264)))
    parser.add_argument("--counts", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    config = V4Config.from_json(args.v4_config)
    stimuli = {
        (int(row["seed"]), int(row["gold_count"])): row
        for row in load_stimuli(args.stimuli)
        if str(row.get("design_variant")) == "v4.4"
    }
    args.output.mkdir(parents=True, exist_ok=True)
    detail_path = args.output / "token_corruption_detail.csv"
    fieldnames = [
        "model_label", "seed", "gold_count", "condition", "prediction", "correct",
        "absolute_error", "signed_error", "completion", "token_budget",
        "changed_tokens", "clean_prediction", "clean_correct", "clean_absolute_error",
        "answer_state_paths",
    ]
    existing: set[tuple[str, int, int, str]] = set()
    if detail_path.exists():
        with detail_path.open(newline="", encoding="utf-8") as handle:
            existing = {(row["model_label"], int(row["seed"]), int(row["gold_count"]), row["condition"]) for row in csv.DictReader(handle)}
    write_header = not detail_path.exists()
    with detail_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for model_label in args.models:
            spec = resolve_model_spec(model_label)
            model, tokenizer, adapter = load_registered_model(
                spec,
                cache_dir=args.cache_dir,
                device_map=args.device_map,
                torch_dtype=config.model_torch_dtype,
                attention_backend=config.attention_prefix_backend,
            )
            layers = CAPTURE_LAYERS[model_label]
            for seed in args.seeds:
                for count in args.counts:
                    keys = [(model_label, seed, count, condition) for condition in ("clean", "needle_corrupt", "ordinary_control")]
                    if all(key in existing for key in keys):
                        continue
                    encoding = render_v4_prompt(
                        stimuli[(seed, count)], tokenizer=tokenizer, model_spec=spec,
                        config=config, answer_format="numeric",
                    )
                    needle_encoding, control_encoding, corruption = corrupt_encodings(encoding)
                    clean_result, clean_states = generate_and_capture(model, tokenizer, adapter, encoding, layers)
                    clean_prediction = parsed_prediction(clean_result)
                    clean_error = abs(int(clean_prediction) - count) if clean_prediction is not None else 10
                    for condition, current, changed in (
                        ("clean", encoding, 0),
                        ("needle_corrupt", needle_encoding, corruption["needle_changed_tokens"]),
                        ("ordinary_control", control_encoding, corruption["control_changed_tokens"]),
                    ):
                        if (model_label, seed, count, condition) in existing:
                            continue
                        if condition == "clean":
                            result, states = clean_result, clean_states
                        else:
                            result, states = generate_and_capture(model, tokenizer, adapter, current, layers)
                        parsed = parsed_prediction(result)
                        paths = {}
                        for layer, state in states.items():
                            relative = Path("states") / model_label / f"seed_{seed}_count_{count}_{condition}_L{layer}.npy"
                            state_path = args.output / relative
                            state_path.parent.mkdir(parents=True, exist_ok=True)
                            np.save(state_path, state.astype(np.float16))
                            paths[str(layer)] = relative.as_posix()
                        writer.writerow(
                            {
                                "model_label": model_label,
                                "seed": seed,
                                "gold_count": count,
                                "condition": condition,
                                "prediction": "" if parsed is None else parsed,
                                "correct": int(parsed == count) if parsed is not None else 0,
                                "absolute_error": abs(parsed - count) if parsed is not None else 10,
                                "signed_error": parsed - count if parsed is not None else "",
                                "completion": result.get("completion_text", ""),
                                "token_budget": corruption["token_budget"],
                                "changed_tokens": changed,
                                "clean_prediction": "" if clean_prediction is None else int(clean_prediction),
                                "clean_correct": int(clean_prediction == count) if clean_prediction is not None else 0,
                                "clean_absolute_error": clean_error,
                                "answer_state_paths": json.dumps(paths, sort_keys=True),
                            }
                        )
                        handle.flush()
                        print(f"[token-corruption] {model_label} seed={seed} N={count} {condition} prediction={parsed}", flush=True)
            del model, tokenizer, adapter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    audit = {
        "schema_version": "realistic_niah_v4_4_token_corruption_v1",
        "models": args.models,
        "seeds": args.seeds,
        "counts": args.counts,
        "conditions": ["clean", "needle_corrupt", "ordinary_control"],
        "inference_unit": "seed",
        "control": "equal token budget ordinary-passage target segments replaced by other ordinary-passage segments",
        "capture_layers": CAPTURE_LAYERS,
        "status": "PASS",
    }
    (args.output / "token_corruption_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
