from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from realistic_niah_v4.modeling import load_registered_model
from realistic_niah_v4.spec import resolve_model_spec

from .attention import reconstruct_attention_summary, summarize_hidden_states
from .capture import capture_teacher_forced_trace, kv_source_layers
from .prompts import render_trace_prompt
from .runtime import EventLogger, select_stimuli
from .spec import V442Config
from .trace import generate_trace


LEGACY_CONDITION = ("nonthinking", "cue_present")


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def shard_dir(output_root: str | Path, encoding: Any) -> Path:
    return (
        Path(output_root)
        / "conditions"
        / encoding.model_label
        / encoding.prompt_variant
        / encoding.mode
        / encoding.split
        / encoding.stimulus_id
    )


def write_runtime_provenance(
    output_root: str | Path,
    *,
    config: V442Config,
    command: Sequence[str] | None = None,
) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "realistic_niah_v4_4_2_runtime_v1",
        "created_unix_time": time.time(),
        "command": list(sys.argv if command is None else command),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "config": config.to_dict(),
    }
    path = root / "runtime_provenance.json"
    _atomic_json(payload, path)
    _atomic_json(config.to_dict(), root / "config.resolved.json")
    return path


def attach_legacy_v44_baseline(
    legacy_run_root: str | Path,
    output_root: str | Path,
) -> Path:
    legacy = Path(legacy_run_root).resolve()
    if not legacy.exists():
        raise FileNotFoundError(f"Legacy V4.4 run root does not exist: {legacy}")
    references: list[dict[str, Any]] = []
    candidates = (
        "representation/capture/capture_index.jsonl",
        "representation/answer_query_all_layers_v1/capture_index.jsonl",
        "attention/capture/attention_capture_index.jsonl",
        "attention/analysis/attention_head_detail.csv",
        "attention/analysis/attention_head_summary.csv",
        "behavior/capture/generation_labels.csv",
    )
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        model_root = legacy / model / "numeric"
        found = 0
        for relative in candidates:
            path = model_root / relative
            if path.exists():
                references.append(
                    {
                        "model_label": model,
                        "artifact": relative,
                        "path": str(path.resolve()),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
                found += 1
        if found == 0:
            raise RuntimeError(
                f"No registered V4.4 baseline artifacts found for {model} under {model_root}"
            )
    payload = {
        "schema_version": "realistic_niah_v4_4_2_legacy_reference_v1",
        "condition": {"mode": "nonthinking", "prompt_variant": "cue_present"},
        "reuse_policy": "reference_only_no_rerun_no_copy",
        "legacy_run_root": str(legacy),
        "artifacts": references,
    }
    path = Path(output_root) / "legacy_v4_4_reference.json"
    _atomic_json(payload, path)
    return path


def _save_generation(
    payload: dict[str, Any],
    *,
    encoding: Any,
    path: Path,
) -> None:
    saved = dict(payload)
    saved["prompt_audit"] = {
        "user_text_sha256": _text_sha256(encoding.user_text),
        "model_text_sha256": _text_sha256(encoding.model_text),
        "prompt_token_count": encoding.prompt_token_count,
        "assistant_prefix_span": encoding.assistant_prefix_span,
    }
    _atomic_json(saved, path)


def run_condition(
    *,
    stimuli_path: str | Path,
    output_root: str | Path,
    config: V442Config,
    model_label: str,
    mode: str,
    prompt_variant: str,
    stage: str,
    cache_dir: str | Path | None = None,
    device_map: str | dict[str, Any] = "auto",
    torch_dtype: str = "bfloat16",
    attention_backend: str = "sdpa",
    analysis_device: str = "cpu",
    seeds: Sequence[int] | None = None,
    counts: Sequence[int] | None = None,
    split: str | None = None,
    overwrite: bool = False,
    reuse_legacy_baseline: bool = True,
) -> dict[str, Any]:
    config.validate()
    if stage not in {
        "preflight",
        "generate",
        "capture",
        "generate-capture",
        "analyze",
        "all",
    }:
        raise ValueError(f"Unknown V4.4.2 stage: {stage}")
    if (
        config.legacy_baseline_mode == "reference_only"
        and (mode, prompt_variant) == LEGACY_CONDITION
        and reuse_legacy_baseline
    ):
        return {
            "status": "legacy_reference",
            "message": "Completed V4.4 cue-present non-thinking artifacts are not rerun.",
            "reference": str(Path(output_root) / "legacy_v4_4_reference.json"),
        }
    selected = select_stimuli(
        stimuli_path,
        variants=("v4.4",),
        seeds=config.seeds if seeds is None else seeds,
        counts=config.counts if counts is None else counts,
        split=split,
    )
    model_spec = resolve_model_spec(model_label)
    logger = EventLogger(Path(output_root) / "events.jsonl")
    with logger.timer("v4_4_2_model_load", model=model_label, stage=stage):
        model, tokenizer, adapter = load_registered_model(
            model_spec,
            cache_dir=cache_dir,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attention_backend=attention_backend,
        )
    rows: list[dict[str, Any]] = []
    for stimulus in selected:
        encoding = render_trace_prompt(
            stimulus,
            tokenizer=tokenizer,
            model_spec=model_spec,
            mode=mode,
            prompt_variant=prompt_variant,
        )
        destination = shard_dir(output_root, encoding)
        generation_path = destination / "generation.json"
        capture_path = destination / "capture"
        if stage == "preflight":
            rows.append(
                {
                    "stimulus_id": encoding.stimulus_id,
                    "prompt_tokens": encoding.prompt_token_count,
                    "slots": len(encoding.slot_spans),
                    "needles": len(encoding.needle_spans),
                    "assistant_prefix_span": encoding.assistant_prefix_span,
                }
            )
            continue

        if stage in {"generate", "generate-capture", "all"}:
            if generation_path.exists() and not overwrite:
                generation = json.loads(generation_path.read_text(encoding="utf-8"))
            else:
                sampling_seed = config.sampling_seed(
                    model_label=model_label,
                    prompt_variant=prompt_variant,
                    seed=encoding.seed,
                    count=encoding.count,
                )
                with logger.timer(
                    "v4_4_2_generation",
                    stimulus_id=encoding.stimulus_id,
                    model=model_label,
                    mode=mode,
                    prompt_variant=prompt_variant,
                ):
                    generation = generate_trace(
                        model,
                        tokenizer,
                        encoding,
                        model_family=model_spec.family,
                        config=config,
                        sampling_seed=sampling_seed,
                    )
                _save_generation(
                    generation,
                    encoding=encoding,
                    path=generation_path,
                )
        else:
            if not generation_path.exists():
                raise FileNotFoundError(
                    f"Missing generation shard for {encoding.stimulus_id}: {generation_path}"
                )
            generation = json.loads(generation_path.read_text(encoding="utf-8"))
            if generation["prompt_audit"]["model_text_sha256"] != _text_sha256(
                encoding.model_text
            ):
                raise RuntimeError(
                    f"Rendered prompt drift for {encoding.stimulus_id}; refusing replay"
                )

        capture_manifest = None
        if stage in {"capture", "generate-capture", "all"}:
            with logger.timer(
                "v4_4_2_teacher_forced_capture",
                stimulus_id=encoding.stimulus_id,
                model=model_label,
                mode=mode,
                prompt_variant=prompt_variant,
            ):
                capture_manifest = capture_teacher_forced_trace(
                    model,
                    adapter,
                    encoding,
                    generation,
                    config=config,
                    output_dir=capture_path,
                    overwrite=overwrite,
                )
        if stage in {"analyze", "all"}:
            if not (capture_path / "capture_manifest.json").exists():
                raise FileNotFoundError(
                    f"Missing capture for {encoding.stimulus_id}: {capture_path}"
                )
            with logger.timer(
                "v4_4_2_attention_reconstruction",
                stimulus_id=encoding.stimulus_id,
                analysis_device=analysis_device,
            ):
                attention = reconstruct_attention_summary(
                    capture_path,
                    config=config,
                    device=analysis_device,
                    overwrite=overwrite,
                )
            hidden_rows = summarize_hidden_states(capture_path)
            _atomic_json(hidden_rows, capture_path / "hidden_summary.json")
        else:
            attention = None
        rows.append(
            {
                "stimulus_id": encoding.stimulus_id,
                "generation": str(generation_path),
                "capture": None if capture_manifest is None else str(capture_path),
                "attention": attention,
            }
        )
    return {
        "status": "ok",
        "stage": stage,
        "model_label": model_label,
        "mode": mode,
        "prompt_variant": prompt_variant,
        "rows": rows,
    }


def analyze_existing_captures(
    output_root: str | Path,
    *,
    config: V442Config,
    model_label: str | None = None,
    mode: str | None = None,
    prompt_variant: str | None = None,
    split: str | None = None,
    analysis_device: str = "cpu",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Reconstruct attention from saved Q/K without loading either model."""

    root = Path(output_root)
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(root.glob("conditions/**/capture/capture_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        filters = {
            "model_label": model_label,
            "mode": mode,
            "prompt_variant": prompt_variant,
            "split": split,
        }
        if any(
            expected is not None and str(manifest.get(key)) != str(expected)
            for key, expected in filters.items()
        ):
            continue
        capture_dir = manifest_path.parent
        attention = reconstruct_attention_summary(
            capture_dir,
            config=config,
            device=analysis_device,
            overwrite=overwrite,
        )
        hidden_rows = summarize_hidden_states(capture_dir)
        _atomic_json(hidden_rows, capture_dir / "hidden_summary.json")
        rows.append(
            {
                "stimulus_id": manifest["stimulus_id"],
                "model_label": manifest["model_label"],
                "mode": manifest["mode"],
                "prompt_variant": manifest["prompt_variant"],
                "attention": attention,
            }
        )
    if not rows:
        raise FileNotFoundError("No saved V4.4.2 captures matched the analysis filters")
    return {"status": "ok", "stage": "analyze-existing", "rows": rows}


def build_filestream_index(output_root: str | Path) -> Path:
    root = Path(output_root)
    rows: list[dict[str, Any]] = []
    for generation_path in sorted(root.glob("conditions/**/generation.json")):
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        capture_dir = generation_path.parent / "capture"
        row = {
            "source": "v4_4_2",
            "model_label": generation["model_label"],
            "mode": generation["mode"],
            "prompt_variant": generation["prompt_variant"],
            "stimulus_id": generation["stimulus_id"],
            "seed": generation["seed"],
            "split": generation["split"],
            "gold_count": generation["gold_count"],
            "generation_path": str(generation_path.relative_to(root).as_posix()),
            "capture_manifest_path": (
                str((capture_dir / "capture_manifest.json").relative_to(root).as_posix())
                if (capture_dir / "capture_manifest.json").exists()
                else None
            ),
            "attention_summary_path": (
                str((capture_dir / "attention_summary.pt").relative_to(root).as_posix())
                if (capture_dir / "attention_summary.pt").exists()
                else None
            ),
        }
        rows.append(row)
    legacy_path = root / "legacy_v4_4_reference.json"
    if legacy_path.exists():
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
        for artifact in legacy["artifacts"]:
            rows.append(
                {
                    "source": "legacy_v4_4_reference",
                    "model_label": artifact["model_label"],
                    "mode": "nonthinking",
                    "prompt_variant": "cue_present",
                    "artifact": artifact["artifact"],
                    "path": artifact["path"],
                    "sha256": artifact["sha256"],
                }
            )
    path = root / "filestream_index.jsonl"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)
    _atomic_json(
        {
            "schema_version": "realistic_niah_v4_4_2_filestream_index_v1",
            "rows": len(rows),
            "new_shards": sum(row["source"] == "v4_4_2" for row in rows),
            "legacy_references": sum(
                row["source"] == "legacy_v4_4_reference" for row in rows
            ),
            "index": str(path.resolve()),
        },
        root / "filestream_manifest.json",
    )
    return path


def _projection_width(module: Any) -> int:
    value = getattr(module, "out_features", None)
    if value is not None:
        return int(value)
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    raise RuntimeError(f"Cannot infer projection width for {type(module).__name__}")


def estimate_storage(
    model: Any,
    adapter: Any,
    *,
    config: V442Config,
    prompt_tokens: int = 11_000,
    trace_tokens: int = 4096,
    answer_tokens: int = 8,
) -> dict[str, Any]:
    layers = tuple(range(adapter.num_layers)) if not config.capture_layers else config.capture_layers
    sources = kv_source_layers(adapter, layers)
    unique_sources = sorted(set(sources.values()))
    bytes_per_value = torch.tensor([], dtype=getattr(torch, config.qk_save_dtype)).element_size()
    hidden_bytes_per_value = torch.tensor([], dtype=getattr(torch, config.hidden_save_dtype)).element_size()
    hidden_size = int(model.get_input_embeddings().weight.shape[1])

    def condition_bytes(condition_trace_tokens: int) -> dict[str, int]:
        query_tokens = int(condition_trace_tokens) + int(answer_tokens)
        sequence_tokens = int(prompt_tokens) + query_tokens
        q_bytes = sum(
            query_tokens
            * _projection_width(getattr(adapter.attentions[layer], "q_proj"))
            * bytes_per_value
            for layer in layers
        )
        k_bytes = sum(
            sequence_tokens
            * _projection_width(getattr(adapter.attentions[layer], "k_proj"))
            * bytes_per_value
            for layer in unique_sources
        )
        hidden_bytes = (
            len(layers) * query_tokens * hidden_size * hidden_bytes_per_value
        )
        rope_bytes = sum(
            sequence_tokens
            * 2
            * int(getattr(adapter.attentions[layer], "head_dim"))
            * bytes_per_value
            for layer in layers
        )
        return {
            "query_tokens": query_tokens,
            "sequence_tokens": sequence_tokens,
            "q_bytes": q_bytes,
            "k_bytes": k_bytes,
            "hidden_bytes": hidden_bytes,
            "rope_bytes": rope_bytes,
            "total_bytes": q_bytes + k_bytes + hidden_bytes + rope_bytes,
        }

    native = condition_bytes(int(trace_tokens))
    nonthinking = condition_bytes(0)
    native_shards = 2 * len(config.seeds) * len(config.counts)
    nonthinking_shards = 2 * len(config.seeds) * len(config.counts)
    per_model_total = (
        native["total_bytes"] * native_shards
        + nonthinking["total_bytes"] * nonthinking_shards
    )
    return {
        "model_label": getattr(model.config, "name_or_path", type(model).__name__),
        "prompt_tokens": prompt_tokens,
        "trace_tokens": trace_tokens,
        "answer_tokens": answer_tokens,
        "layers": list(layers),
        "unique_kv_source_layers": unique_sources,
        "native": {**native, "total_gib_per_shard": native["total_bytes"] / 2**30},
        "nonthinking": {
            **nonthinking,
            "total_gib_per_shard": nonthinking["total_bytes"] / 2**30,
        },
        "new_native_shards_per_model": native_shards,
        "new_nonthinking_shards_per_model": nonthinking_shards,
        "worst_case_tib_per_model": per_model_total / 2**40,
        "note": "Worst case assumes every native shard for this model reaches the requested trace length; size both models separately and calibrate with the two-model pilot.",
    }
