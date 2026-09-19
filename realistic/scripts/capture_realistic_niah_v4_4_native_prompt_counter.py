from __future__ import annotations

"""Capture prompt running-index states for the 9000 native-thinking dataset.

Only N=10 prompts are needed: each contains all ten nested needle endpoints,
so 900 forwards cover the full 30 seed x 30 realization design.  Exact stored
input IDs are replayed and span-token alignment is audited against the stored
rendered prompt before any model forward.
"""

import argparse
import inspect
import json
import os
from pathlib import Path
import time
from typing import Any
from dataclasses import dataclass

import torch
from safetensors.torch import save_file

from realistic_niah_v4.spec import resolve_model_spec


@dataclass(frozen=True)
class MinimalAdapter:
    layers: tuple[torch.nn.Module, ...]

    @property
    def num_layers(self) -> int:
        return len(self.layers)


def discover_layers(model: torch.nn.Module) -> MinimalAdapter:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None) or config
    expected = getattr(text_config, "num_hidden_layers", None)
    candidates: list[tuple[int, str, torch.nn.ModuleList]] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or not module:
            continue
        if not all(
            any(isinstance(getattr(block, attr, None), torch.nn.Module) for attr in ("self_attn", "attn", "attention"))
            for block in module
        ):
            continue
        score = len(module) + (100 if expected is not None and len(module) == int(expected) else 0)
        lowered = name.lower()
        score += 30 if "language_model" in lowered or "text_model" in lowered else 0
        score += 20 if lowered.endswith("model.layers") else 0
        score -= 100 if "vision" in lowered or "encoder" in lowered else 0
        candidates.append((score, name, module))
    if not candidates:
        raise RuntimeError("could not discover decoder layer ModuleList")
    candidates.sort(key=lambda item: (item[0], len(item[2])), reverse=True)
    return MinimalAdapter(tuple(candidates[0][2]))


def load_model_and_tokenizer(model_spec: Any, cache_dir: Path | None):
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_spec.model_id,
        revision=model_spec.revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        trust_remote_code=False,
    )
    loader = getattr(transformers, model_spec.loader_class)
    model = loader.from_pretrained(
        model_spec.model_id,
        revision=model_spec.revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.eval()
    return model, tokenizer, discover_layers(model)


def tensor_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    hidden = getattr(output, "last_hidden_state", None)
    if isinstance(hidden, torch.Tensor):
        return hidden
    raise TypeError(f"cannot extract hidden tensor from {type(output).__name__}")


def bounded_logits_kwargs(model: Any) -> dict[str, int]:
    signature = inspect.signature(model.forward)
    accepts = "logits_to_keep" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    return {"logits_to_keep": 1} if accepts else {}


def locate_needles(row: dict[str, Any], tokenizer: Any) -> list[dict[str, int | str]]:
    rendered = str(row["rendered_prompt"])
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )
    input_ids = [int(value) for value in encoded["input_ids"]]
    stored_ids = [int(value) for value in row["input_ids"]]
    if input_ids != stored_ids:
        mismatch = next(
            (index for index, (left, right) in enumerate(zip(input_ids, stored_ids)) if left != right),
            min(len(input_ids), len(stored_ids)),
        )
        raise RuntimeError(
            f"stored/rendered token mismatch for {row['stimulus_id']} at {mismatch}; "
            f"lengths={len(input_ids)}/{len(stored_ids)}"
        )
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    events: list[dict[str, int | str]] = []
    for needle in sorted(row["active_needle_spans"], key=lambda item: int(item["slot_index"])):
        text = str(needle["text"])
        char_start = rendered.find(text)
        if char_start < 0 or rendered.find(text, char_start + 1) >= 0:
            raise RuntimeError(f"needle text is missing or non-unique: {row['stimulus_id']} N{needle['slot_index']}")
        char_end = char_start + len(text)
        token_offsets = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > start and start < char_end and end > char_start
        ]
        if not token_offsets:
            raise RuntimeError(f"no token overlaps needle span in {row['stimulus_id']}")
        start, end = min(token_offsets), max(token_offsets) + 1
        events.append(
            {
                "running_index": int(needle["slot_index"]),
                "city": str(needle["city"]),
                "score": int(needle["score"]),
                "token_start": start,
                "token_end": end,
                "char_start": char_start,
                "char_end": char_end,
            }
        )
    if [event["running_index"] for event in events] != list(range(1, 11)):
        raise RuntimeError(f"N=10 endpoint contract failed for {row['stimulus_id']}")
    return events


def capture(model: Any, adapter: Any, input_ids: list[int], events: list[dict[str, Any]]) -> torch.Tensor:
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            hidden = tensor_from_output(output)
            captured[layer] = torch.stack(
                [hidden[0, int(event["token_end"]) - 1].detach() for event in events]
            ).to(device="cpu", dtype=torch.float16).contiguous()
        return hook

    for layer, module in enumerate(adapter.layers):
        handles.append(module.register_forward_hook(make_hook(layer)))
    embeddings = model.get_input_embeddings()
    tokens = torch.tensor([input_ids], dtype=torch.long, device=embeddings.weight.device)
    try:
        with torch.inference_mode():
            model(
                input_ids=tokens,
                attention_mask=torch.ones_like(tokens),
                use_cache=False,
                **bounded_logits_kwargs(model),
            )
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != adapter.num_layers:
        raise RuntimeError(f"captured {len(captured)}/{adapter.num_layers} layers")
    return torch.stack([captured[layer] for layer in range(adapter.num_layers)])


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=["Qwen3-8B", "Gemma4-E4B"], required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve() / args.model / "capture"
    shards = output / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    model_spec = resolve_model_spec(args.model)
    # Audit-only still loads the pinned tokenizer, but not model weights.
    if args.audit_only:
        import transformers
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_spec.model_id,
            revision=model_spec.revision,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
            trust_remote_code=False,
        )
        model = adapter = None
    else:
        model, tokenizer, adapter = load_model_and_tokenizer(model_spec, args.cache_dir)

    started = time.perf_counter()
    completed = 0
    audited = 0
    split_counts: dict[str, int] = {}
    with args.requests.resolve().open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("model_label") != args.model or int(row.get("gold_count", 0)) != 10:
                continue
            if args.max_examples is not None and audited >= args.max_examples:
                break
            events = locate_needles(row, tokenizer)
            audited += 1
            if args.audit_only:
                continue
            stimulus_id = str(row["stimulus_id"])
            tensor_path = shards / f"{stimulus_id}.safetensors"
            metadata_path = shards / f"{stimulus_id}.json"
            if tensor_path.is_file() and metadata_path.is_file():
                existing = json.loads(metadata_path.read_text(encoding="utf-8"))
                if existing.get("status") == "completed":
                    completed += 1
                    continue
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            item_started = time.perf_counter()
            states = capture(model, adapter, [int(value) for value in row["input_ids"]], events)
            temporary = tensor_path.with_name(tensor_path.name + ".tmp")
            save_file(
                {
                    "states": states,
                    "layer_indices": torch.arange(states.shape[0], dtype=torch.int64),
                },
                str(temporary),
            )
            os.replace(temporary, tensor_path)
            metadata = {
                "schema_version": "realistic_niah_v4_4_native_prompt_counter_capture_v1",
                "status": "completed",
                "model_label": args.model,
                "model_id": row["model_id"],
                "model_revision": row["model_revision"],
                "request_id": row["request_id"],
                "stimulus_id": stimulus_id,
                "seed": int(row["seed"]),
                "realization_id": int(row["realization_id"]),
                "split": row["split"],
                "gold_count": 10,
                "input_tokens": len(row["input_ids"]),
                "passage_sha256": row["passage_sha256"],
                "events": events,
                "state_shape": list(states.shape),
                "tensor_path": str(tensor_path.relative_to(output).as_posix()),
                "capture_wall_time_seconds": time.perf_counter() - item_started,
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
            }
            atomic_json(metadata_path, metadata)
            completed += 1
            split_counts[str(row["split"])] = split_counts.get(str(row["split"]), 0) + 1
            print(
                f"[prompt counter] {args.model} {completed} {stimulus_id} "
                f"seconds={metadata['capture_wall_time_seconds']:.3f}",
                flush=True,
            )
            del states
            if torch.cuda.is_available() and completed % 50 == 0:
                torch.cuda.empty_cache()
    manifest = {
        "schema_version": "realistic_niah_v4_4_native_prompt_counter_manifest_v1",
        "status": "AUDIT_PASS" if args.audit_only else "PASS",
        "source_requests": str(args.requests.resolve()),
        "source_read_only": True,
        "model_label": args.model,
        "n10_rows_audited": audited,
        "completed": completed,
        "split_counts_newly_captured": split_counts,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output / ("alignment_audit.json" if args.audit_only else "capture_manifest.json"), manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
