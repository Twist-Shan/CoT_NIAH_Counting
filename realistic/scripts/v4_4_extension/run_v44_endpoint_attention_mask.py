from __future__ import annotations

"""Endpoint-query attention masking and matched earlier-span attention test."""

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch

FALLBACK_SRC = Path("runs/v4_4_counter_channel_20260806/code/src")
if FALLBACK_SRC.is_dir() and str(FALLBACK_SRC) not in sys.path:
    sys.path.insert(0, str(FALLBACK_SRC))

from realistic_niah_v4.modeling import (
    _accepts_keyword,
    _attention_tensor,
    _bounded_logits_kwargs,
    _encoding_tensors,
    _extract_attentions,
    _temporary_attention_backend,
    _tensor_from_output,
    load_registered_model,
)
from realistic_niah_v4.prompts import PromptEncoding, render_v4_prompt
from realistic_niah_v4.spec import V4Config, resolve_model_spec
from realistic_niah_v4.stimuli import load_stimuli


CONDITIONS = ("clean", "needle_only", "matched_nonneedle_only")
ATTENTION_COLUMNS = (
    "model_label", "seed", "split", "occurrence", "layer", "head",
    "attention_key_start", "prior_needle_span_mass",
    "prior_matched_nonneedle_mass", "prior_span_preference",
    "current_needle_span_mass",
)


def slice_mass(values: np.ndarray, start: int, end: int, key_start: int) -> np.ndarray:
    left = max(start, key_start) - key_start
    right = min(end, key_start + values.shape[1]) - key_start
    if right <= left:
        return np.zeros(values.shape[0], dtype=np.float64)
    return values[:, left:right].sum(axis=1, dtype=np.float64)


def matched_segments(encoding: PromptEncoding, occurrence: int) -> list[tuple[int, int]]:
    spans = encoding.needle_spans[:occurrence]
    forbidden: set[int] = set()
    for span in tuple(encoding.slot_spans) + tuple(encoding.hard_negative_spans):
        forbidden.update(range(int(span.start), int(span.end)))
    used = set(forbidden)
    matched = []
    for span in spans:
        length = int(span.end) - int(span.start)
        found = None
        for gap in range(8, 512):
            candidate = int(span.start) - gap - length
            positions = set(range(candidate, candidate + length))
            if candidate >= 1 and not positions.intersection(used):
                found = (candidate, candidate + length)
                used.update(positions)
                break
        if found is None:
            raise RuntimeError("Could not construct depth-matched non-needle segment")
        matched.append(found)
    return matched


def allowed_mask(encoding: PromptEncoding, occurrence: int, condition: str, matched: Sequence[tuple[int, int]]) -> torch.Tensor:
    query = int(encoding.needle_spans[occurrence - 1].end) - 1
    mask = torch.zeros((1, query + 1), dtype=torch.long)
    if condition == "clean":
        mask[:] = 1
    elif condition == "needle_only":
        for span in encoding.needle_spans[:occurrence]:
            mask[:, int(span.start) : min(int(span.end), query + 1)] = 1
        mask[:, query] = 1
    elif condition == "matched_nonneedle_only":
        budget = sum(int(span.end) - int(span.start) for span in encoding.needle_spans[:occurrence])
        positions = [p for start, end in matched for p in range(start, end)]
        positions = positions[: max(0, budget - 1)]
        if positions:
            mask[:, positions] = 1
        mask[:, query] = 1
    else:
        raise ValueError(condition)
    if int(mask.sum()) == 0:
        raise RuntimeError("Empty query key mask")
    return mask


@torch.inference_mode()
def query_once(model: Any, adapter: Any, encoding: PromptEncoding, query: int, mask: torch.Tensor, *, save_attention: bool) -> tuple[np.ndarray, list[torch.Tensor] | None, list[int] | None]:
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    prefix_output = model(
        input_ids=input_ids[:, :query], attention_mask=attention_mask[:, :query],
        use_cache=True, output_attentions=False, **_bounded_logits_kwargs(model),
    )
    past = getattr(prefix_output, "past_key_values", None)
    if past is None:
        raise RuntimeError("Prefix forward returned no cache")
    kwargs: dict[str, Any] = {
        "input_ids": input_ids[:, query : query + 1],
        "attention_mask": mask.to(input_ids.device),
        "past_key_values": past,
        "use_cache": False,
        "output_attentions": save_attention,
        **_bounded_logits_kwargs(model),
    }
    if _accepts_keyword(model, "position_ids"):
        kwargs["position_ids"] = torch.tensor([[query]], dtype=torch.long, device=input_ids.device)
    if _accepts_keyword(model, "cache_position"):
        kwargs["cache_position"] = torch.tensor([query], dtype=torch.long, device=input_ids.device)
    shared = getattr(prefix_output, "shared_kv_states", None)
    if shared is not None and _accepts_keyword(model, "shared_kv_states"):
        kwargs["shared_kv_states"] = shared
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer in range(int(adapter.num_layers)):
        def hook(_module, _args, output, *, layer=layer):
            hidden = _tensor_from_output(output)
            captured[layer] = hidden[0, -1].detach().float().cpu()
        handles.append(adapter.layers[layer].register_forward_hook(hook))
    try:
        context = _temporary_attention_backend(model, "eager") if save_attention else _temporary_attention_backend(model, "sdpa")
        with context:
            output = model(**kwargs)
    finally:
        for handle in handles:
            handle.remove()
    states = np.stack([captured[layer].numpy() for layer in range(int(adapter.num_layers))]).astype(np.float16)
    if not save_attention:
        return states, None, None
    attention = _extract_attentions(output)
    rows = []
    starts = []
    for value in attention:
        tensor = _attention_tensor(value)
        row = tensor[0, :, 0].detach().float().cpu()
        rows.append(row)
        starts.append(query + 1 - int(row.shape[-1]))
    return states, rows, starts


def attention_records(rows: Sequence[torch.Tensor], starts: Sequence[int], encoding: PromptEncoding, occurrence: int, matched: Sequence[tuple[int, int]]) -> list[dict[str, Any]]:
    records = []
    current = encoding.needle_spans[occurrence - 1]
    prior = encoding.needle_spans[: occurrence - 1]
    for layer, (tensor, key_start) in enumerate(zip(rows, starts)):
        values = tensor.numpy().astype(np.float64)
        prior_needle = sum((slice_mass(values, int(span.start), int(span.end), key_start) for span in prior), start=np.zeros(values.shape[0]))
        prior_matched = sum((slice_mass(values, start, end, key_start) for start, end in matched[: occurrence - 1]), start=np.zeros(values.shape[0]))
        current_mass = slice_mass(values, int(current.start), int(current.end), key_start)
        for head in range(values.shape[0]):
            records.append(
                {
                    "layer": layer,
                    "head": head,
                    "attention_key_start": int(key_start),
                    "prior_needle_span_mass": float(prior_needle[head]),
                    "prior_matched_nonneedle_mass": float(prior_matched[head]),
                    "prior_span_preference": float(prior_needle[head] - prior_matched[head]),
                    "current_needle_span_mass": float(current_mass[head]),
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4-config", type=Path, required=True)
    parser.add_argument("--stimuli", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["Qwen3-8B", "Gemma4-E4B"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1254, 1264)))
    parser.add_argument("--occurrences", nargs="+", type=int, default=[2, 4, 6, 8, 10])
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--save-attention-models",
        nargs="*",
        default=["Qwen3-8B"],
        help=(
            "Models for which clean endpoint attention rows are materialized. "
            "Gemma is omitted by default because its eager backend constructs a "
            "quadratic prefix tensor on this transformers build; all causal mask "
            "conditions still run with SDPA."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = V4Config.from_json(args.v4_config)
    stimuli = {int(row["seed"]): row for row in load_stimuli(args.stimuli) if str(row.get("design_variant")) == "v4.4" and int(row.get("gold_count", -1)) == 10}
    args.output.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for model_label in args.models:
        spec = resolve_model_spec(model_label)
        model, tokenizer, adapter = load_registered_model(
            spec, cache_dir=args.cache_dir, device_map=args.device_map,
            torch_dtype=config.model_torch_dtype, attention_backend=config.attention_prefix_backend,
        )
        for seed in args.seeds:
            encoding = render_v4_prompt(stimuli[seed], tokenizer=tokenizer, model_spec=spec, config=config, answer_format="numeric")
            shard = args.output / model_label / "states" / f"seed_{seed}.npz"
            metric_path = args.output / model_label / "attention" / f"seed_{seed}.csv.gz"
            if shard.exists() and metric_path.exists() and not args.overwrite:
                index_rows.append({"model_label": model_label, "seed": seed, "state_path": str(shard.relative_to(args.output).as_posix()), "attention_path": str(metric_path.relative_to(args.output).as_posix())})
                continue
            states = np.empty((len(CONDITIONS), len(args.occurrences), int(adapter.num_layers), int(model.get_input_embeddings().weight.shape[1])), dtype=np.float16)
            metrics = []
            masks = []
            for occurrence_index, occurrence in enumerate(args.occurrences):
                query = int(encoding.needle_spans[occurrence - 1].end) - 1
                matched = matched_segments(encoding, occurrence)
                for condition_index, condition in enumerate(CONDITIONS):
                    mask = allowed_mask(encoding, occurrence, condition, matched)
                    current_states, attention_rows, key_starts = query_once(
                        model, adapter, encoding, query, mask,
                        save_attention=(
                            condition == "clean"
                            and model_label in set(args.save_attention_models)
                        ),
                    )
                    states[condition_index, occurrence_index] = current_states
                    masks.append({"occurrence": occurrence, "condition": condition, "allowed_keys": int(mask.sum())})
                    if attention_rows is not None and key_starts is not None:
                        for row in attention_records(attention_rows, key_starts, encoding, occurrence, matched):
                            metrics.append({"model_label": model_label, "seed": seed, "split": "confirmation", "occurrence": occurrence, **row})
                    print(f"[endpoint-mask] {model_label} seed={seed} n={occurrence} {condition}", flush=True)
            shard.parent.mkdir(parents=True, exist_ok=True)
            tmp = shard.with_name(shard.name + ".tmp")
            with tmp.open("wb") as handle:
                np.savez(handle, states=states, conditions=np.asarray(CONDITIONS), occurrences=np.asarray(args.occurrences), layers=np.arange(int(adapter.num_layers)), mask_audit=np.asarray([json.dumps(row, sort_keys=True) for row in masks]))
            tmp.replace(shard)
            metric_path.parent.mkdir(parents=True, exist_ok=True)
            pd_frame = __import__("pandas").DataFrame(metrics, columns=ATTENTION_COLUMNS)
            pd_frame.to_csv(metric_path, index=False, compression="gzip")
            index_rows.append({"model_label": model_label, "seed": seed, "state_path": str(shard.relative_to(args.output).as_posix()), "attention_path": str(metric_path.relative_to(args.output).as_posix())})
        del model, tokenizer, adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (args.output / "capture_index.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in index_rows), encoding="utf-8")
    (args.output / "audit.json").write_text(json.dumps({
        "schema_version": "realistic_niah_v4_4_endpoint_attention_mask_v1",
        "models": args.models, "seeds": args.seeds, "occurrences": args.occurrences,
        "conditions": CONDITIONS,
        "needle_only": "endpoint query may attend only active needle-span keys plus itself",
        "matched_control": "same key budget from depth-matched non-needle spans plus query itself",
        "prefix_cache": "recomputed independently for every condition to prevent mutable-cache carryover",
        "attention_row_models": args.save_attention_models,
        "attention_row_boundary": (
            "Matched ordinary-span attention rows are materialized only for listed "
            "models; state-space masking is evaluated for every model."
        ),
        "status": "PASS",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
