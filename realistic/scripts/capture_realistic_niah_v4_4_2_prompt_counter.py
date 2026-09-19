from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from realistic_niah_v4.modeling import capture_span_states, load_registered_model
from realistic_niah_v4.spec import resolve_model_spec
from realistic_niah_v4_4_2.pipeline import shard_dir
from realistic_niah_v4_4_2.prompts import render_trace_prompt
from realistic_niah_v4_4_2.runtime import select_stimuli


PROMPT_VARIANTS = ("cue_present", "cue_absent")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def validate_reconstruction(
    *,
    encoding: Any,
    generation: dict[str, Any],
    capture_manifest: dict[str, Any],
    generation_path: Path,
) -> None:
    expected_hash = text_sha256(encoding.model_text)
    audit = generation.get("prompt_audit", {})
    checks = {
        "model_text_sha256": str(audit.get("model_text_sha256")) == expected_hash,
        "prompt_token_count": int(audit.get("prompt_token_count", -1))
        == int(encoding.prompt_token_count),
        "capture_prompt_token_count": int(
            capture_manifest.get("prompt_token_count", -1)
        )
        == int(encoding.prompt_token_count),
        "needle_end_positions": list(
            map(int, capture_manifest.get("needle_end_positions", []))
        )
        == [int(span.end - 1) for span in encoding.needle_spans],
        "stimulus_id": str(generation.get("stimulus_id"))
        == str(encoding.stimulus_id),
        "mode": str(generation.get("mode")) == "nonthinking",
        "prompt_variant": str(generation.get("prompt_variant"))
        == str(encoding.prompt_variant),
        "gold_count": int(generation.get("gold_count", -1)) == 10,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"Prompt reconstruction drift for {generation_path}: {', '.join(failed)}"
        )


def capture(
    *,
    run_root: Path,
    stimuli_path: Path,
    output_root: Path,
    model_label: str,
    seeds: tuple[int, ...],
    prompt_variants: tuple[str, ...],
    cache_dir: Path | None,
    device_map: str,
    torch_dtype: str,
    attention_backend: str,
    overwrite: bool,
) -> dict[str, Any]:
    selected = select_stimuli(
        stimuli_path,
        variants=("v4.4",),
        seeds=seeds,
        counts=(10,),
        split=None,
    )
    if len(selected) != len(seeds):
        raise RuntimeError(
            f"Expected one N=10 stimulus for each of {len(seeds)} seeds; "
            f"found {len(selected)}"
        )
    model_spec = resolve_model_spec(model_label)
    model, tokenizer, adapter = load_registered_model(
        model_spec,
        cache_dir=cache_dir,
        device_map=device_map,
        torch_dtype=torch_dtype,
        attention_backend=attention_backend,
    )
    model_output = output_root / model_label
    index_rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for prompt_variant in prompt_variants:
        if prompt_variant not in PROMPT_VARIANTS:
            raise ValueError(f"Unknown prompt variant: {prompt_variant}")
        for stimulus in selected:
            encoding = render_trace_prompt(
                stimulus,
                tokenizer=tokenizer,
                model_spec=model_spec,
                mode="nonthinking",
                prompt_variant=prompt_variant,
            )
            if encoding.count != 10 or len(encoding.needle_spans) != 10:
                raise RuntimeError(
                    f"Prompt counter requires exactly ten active spans: "
                    f"{encoding.stimulus_id}"
                )
            condition_dir = shard_dir(run_root, encoding)
            generation_path = condition_dir / "generation.json"
            capture_manifest_path = condition_dir / "capture" / "capture_manifest.json"
            if not generation_path.is_file() or not capture_manifest_path.is_file():
                raise FileNotFoundError(
                    f"Missing original V4.4.2 shard for {encoding.stimulus_id} "
                    f"({model_label}, {prompt_variant})"
                )
            generation = read_json(generation_path)
            capture_manifest = read_json(capture_manifest_path)
            validate_reconstruction(
                encoding=encoding,
                generation=generation,
                capture_manifest=capture_manifest,
                generation_path=generation_path,
            )

            relative = (
                Path(prompt_variant) / "shards" / f"{encoding.stimulus_id}.npz"
            )
            shard_path = model_output / relative
            if shard_path.exists() and not overwrite:
                with np.load(shard_path, allow_pickle=False) as payload:
                    if set(payload.files) != {"layer_indices", "span_end"}:
                        raise RuntimeError(f"Incomplete prompt-counter shard: {shard_path}")
                    layer_indices = np.asarray(payload["layer_indices"])
                    span_end = np.asarray(payload["span_end"])
            else:
                started = time.perf_counter()
                captured = capture_span_states(
                    model,
                    adapter,
                    encoding,
                    encoding.needle_spans,
                )
                elapsed = time.perf_counter() - started
                layer_indices = captured["layer_indices"].numpy()
                span_end = captured["span_end"].numpy().astype(np.float16, copy=False)
                expected_shape = (
                    int(adapter.num_layers),
                    10,
                    int(model.get_input_embeddings().weight.shape[1]),
                )
                if tuple(span_end.shape) != expected_shape:
                    raise RuntimeError(
                        f"Unexpected endpoint tensor {tuple(span_end.shape)}; "
                        f"expected {expected_shape}"
                    )
                shard_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = shard_path.with_name(shard_path.name + ".tmp")
                with temporary.open("wb") as handle:
                    np.savez(
                        handle,
                        layer_indices=layer_indices,
                        span_end=span_end,
                    )
                temporary.replace(shard_path)
                print(
                    "[prompt-counter] "
                    f"{model_label} {prompt_variant} seed={encoding.seed} "
                    f"shape={tuple(span_end.shape)} elapsed={elapsed:.3f}s",
                    flush=True,
                )
            index_rows.append(
                {
                    "schema_version": "realistic_niah_v4_4_2_prompt_counter_capture_v1",
                    "model_label": model_label,
                    "mode": "nonthinking",
                    "prompt_variant": prompt_variant,
                    "stimulus_id": encoding.stimulus_id,
                    "seed": int(encoding.seed),
                    "split": encoding.split,
                    "gold_count": 10,
                    "occurrence_count": 10,
                    "prompt_token_count": int(encoding.prompt_token_count),
                    "needle_end_positions": [
                        int(span.end - 1) for span in encoding.needle_spans
                    ],
                    "model_text_sha256": text_sha256(encoding.model_text),
                    "pooling": "span_end",
                    "position_semantics": (
                        "post-block residual at each of the ten active needle "
                        "span endpoints in the N=10 prompt"
                    ),
                    "array_shape": list(map(int, span_end.shape)),
                    "save_dtype": str(span_end.dtype),
                    "shard_path": relative.as_posix(),
                    "source_generation": str(generation_path.relative_to(run_root).as_posix()),
                    "source_capture_manifest": str(
                        capture_manifest_path.relative_to(run_root).as_posix()
                    ),
                }
            )

    index_rows.sort(key=lambda row: (row["prompt_variant"], row["seed"]))
    index_path = model_output / "capture_index.jsonl"
    atomic_jsonl(index_path, index_rows)
    manifest = {
        "schema_version": "realistic_niah_v4_4_2_prompt_counter_capture_v1",
        "model_label": model_label,
        "mode": "nonthinking",
        "prompt_variants": list(prompt_variants),
        "seeds": list(seeds),
        "rows": len(index_rows),
        "layers": int(adapter.num_layers),
        "occurrences_per_prompt": 10,
        "pooling": "span_end",
        "prompt_only_forward": True,
        "causal_equivalence_note": (
            "At a needle endpoint, a causal decoder state is unchanged by the "
            "later prompt suffix or generated continuation."
        ),
        "raw_shards_remain_in_filestream": True,
        "elapsed_seconds": time.perf_counter() - started_all,
        "capture_index": str(index_path),
    }
    atomic_json(model_output / "capture_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill all-layer non-thinking prompt-counter endpoint states for "
            "the V4.4.2 N=10 prompts."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stimuli", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True, choices=("Qwen3-8B", "Gemma4-E4B"))
    parser.add_argument("--seeds", default="1234,1235,1236,1237,1238,1239,1240,1241,1242,1243")
    parser.add_argument("--prompt-variants", default="cue_present,cue_absent")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = capture(
        run_root=args.run_root,
        stimuli_path=args.stimuli,
        output_root=args.output_dir,
        model_label=args.model,
        seeds=csv_ints(args.seeds),
        prompt_variants=tuple(
            item.strip() for item in args.prompt_variants.split(",") if item.strip()
        ),
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
