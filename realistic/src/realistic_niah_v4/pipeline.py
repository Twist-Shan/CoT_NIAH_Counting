from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .attention import (
    capture_attention_shards,
)
from .attention_outcomes import analyze_labeled_attention
from .behavior import capture_generation_labels
from .causal_generation import (
    NEEDLE_PATCH_PROTOCOLS,
    POOLINGS,
    RESIDUAL_PATCH_SITES,
    compare_ranked_ablation_to_random,
    load_broad_rankings,
    load_generation_labels,
    run_generation_head_ablation,
    run_generation_residual_patching,
    summarize_generation_head_ablation,
    summarize_generation_residual_patching,
)
from .geometric_steering import (
    capture_query_residual_shard,
    centroid_geometry_tables,
    compare_geometric_to_random,
    fit_count_centroids,
    run_generation_geometric_steering,
    save_centroid_bundle,
    summarize_generation_geometric_steering,
)
from .modeling import (
    DecoderAdapter,
    capture_post_block_states,
    capture_span_states,
    load_registered_model,
    load_registered_tokenizer,
    query_attention_outputs,
    run_last_logits,
    run_with_head_ablation,
    run_with_residual_patch,
)
from .prompts import PromptEncoding, render_v4_prompt
from .prompt_counter_attention import capture_prompt_counter_attention_shards
from .representation import (
    analyze_representation_captures,
    capture_answer_query_representation_shards,
    capture_representation_shards,
    label_representation_analysis_by_generation,
)
from .spec import V4Config, resolve_fractional_layers, resolve_model_spec
from .stimuli import load_stimuli


class EventLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **payload: Any) -> None:
        row = {
            "event": str(event),
            "unix_time": time.time(),
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(
            f"[v4] {event} "
            + " ".join(f"{key}={value}" for key, value in payload.items()),
            flush=True,
        )

    @contextmanager
    def timer(self, name: str, **payload: Any) -> Iterator[None]:
        start = time.perf_counter()
        self.write(f"{name}_start", **payload)
        try:
            yield
        except Exception as exc:
            self.write(
                f"{name}_failed",
                elapsed_seconds=time.perf_counter() - start,
                error_type=type(exc).__name__,
                error=str(exc),
                **payload,
            )
            raise
        self.write(
            f"{name}_complete",
            elapsed_seconds=time.perf_counter() - start,
            **payload,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)

    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
        "status_short": status,
    }


def write_runtime_provenance(
    *,
    output_dir: str | Path,
    config_path: str | Path,
    stimuli_path: str | Path,
    model_label: str,
    answer_format: str,
    repo_root: str | Path,
) -> Path:
    import scipy
    import sklearn
    import transformers

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_spec = resolve_model_spec(model_label)
    payload = {
        "schema_version": "realistic_niah_v4_runtime_provenance_v1",
        "model": {
            "label": model_spec.label,
            "model_id": model_spec.model_id,
            "revision": model_spec.revision,
            "loader_class": model_spec.loader_class,
        },
        "answer_format": str(answer_format),
        "inputs": {
            "config_path": str(Path(config_path).resolve()),
            "config_sha256": _sha256_file(Path(config_path)),
            "stimuli_path": str(Path(stimuli_path).resolve()),
            "stimuli_sha256": _sha256_file(Path(stimuli_path)),
        },
        "runtime": {
            "command": list(sys.argv),
            "working_directory": str(Path.cwd()),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
        "git": _git_state(repo_root),
    }
    path = output / "runtime_provenance.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    resolved_config = V4Config.from_json(config_path)
    (output / "config.resolved.json").write_text(
        json.dumps(
            resolved_config.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def select_stimuli(
    stimuli_path: str | Path,
    *,
    variants: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    counts: Sequence[int] | None = None,
    split: str | None = None,
) -> list[dict[str, Any]]:
    rows = load_stimuli(stimuli_path)
    variant_set = None if variants is None else {str(value) for value in variants}
    seed_set = None if seeds is None else {int(value) for value in seeds}
    count_set = None if counts is None else {int(value) for value in counts}
    selected = [
        row
        for row in rows
        if (
            (variant_set is None or str(row["design_variant"]) in variant_set)
            and (seed_set is None or int(row["seed"]) in seed_set)
            and (count_set is None or int(row["gold_count"]) in count_set)
            and (split is None or str(row["split"]) == str(split))
        )
    ]
    if not selected:
        raise ValueError("V4 stimulus filters selected no rows")
    return selected


def _write_csv_gzip_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(path)


def _write_shard_index(
    rows: Sequence[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_csv_gzip_shards(
    index_rows: Sequence[dict[str, Any]],
    *,
    root: Path,
) -> pd.DataFrame:
    if not index_rows:
        raise ValueError("No V4 causal shards were indexed")
    return pd.concat(
        [
            pd.read_csv(
                root / str(row["shard_path"]),
                compression="gzip",
            )
            for row in index_rows
        ],
        ignore_index=True,
    )


def render_encodings(
    rows: Iterable[dict[str, Any]],
    *,
    tokenizer: Any,
    model_label: str,
    config: V4Config,
    answer_format: str,
) -> Iterator[PromptEncoding]:
    model_spec = resolve_model_spec(model_label)
    for row in rows:
        yield render_v4_prompt(
            row,
            tokenizer=tokenizer,
            model_spec=model_spec,
            config=config,
            answer_format=answer_format,
        )


def preflight_report(
    *,
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    representative_rows: Sequence[dict[str, Any]],
    model_label: str,
    config: V4Config,
    answer_format: str,
    forward_smoke: bool = False,
) -> dict[str, Any]:
    encodings = list(
        render_encodings(
            representative_rows,
            tokenizer=tokenizer,
            model_label=model_label,
            config=config,
            answer_format=answer_format,
        )
    )
    sequence_lengths = [encoding.sequence_length for encoding in encodings]
    text_config = getattr(model.config, "text_config", model.config)
    max_positions = getattr(text_config, "max_position_embeddings", None)
    if max_positions is not None and max(sequence_lengths) > int(max_positions):
        raise RuntimeError(
            f"Rendered prompt length {max(sequence_lengths)} exceeds "
            f"max_position_embeddings={max_positions}"
        )
    forward = None
    if forward_smoke:
        primary = encodings[0]
        logits = run_last_logits(model, primary)
        first_candidate_ids = sorted(
            {
                int(token_ids[0])
                for _, token_ids in primary.count_candidate_answer_token_ids
            }
        )
        attention_rows, key_starts, attention_logits = query_attention_outputs(
            model,
            adapter,
            primary,
        )
        smoke_layers = sorted({0, adapter.num_layers - 1})
        first_span = primary.needle_spans[0]
        captured = capture_span_states(
            model,
            adapter,
            primary,
            spans=[first_span],
            layers=smoke_layers,
        )
        ablated_logits = run_with_head_ablation(
            model,
            adapter,
            primary,
            [(0, 0)],
            scope="answer_query",
        )
        donor_positions = [primary.query_position] + list(
            range(first_span.start, first_span.end)
        )
        _donor_logits, donor_states = capture_post_block_states(
            model,
            adapter,
            primary,
            donor_positions,
            layers=smoke_layers,
        )
        first_layer = smoke_layers[0]
        last_layer = smoke_layers[-1]
        query_patched = run_with_residual_patch(
            model,
            adapter,
            primary,
            layer=last_layer,
            receiver_positions=[primary.query_position],
            donor_states=donor_states[last_layer][0:1] + 0.01,
        )
        end_patched = run_with_residual_patch(
            model,
            adapter,
            primary,
            layer=first_layer,
            receiver_positions=[first_span.end - 1],
            donor_states=donor_states[first_layer][-1:] + 0.01,
        )
        span_patched = run_with_residual_patch(
            model,
            adapter,
            primary,
            layer=first_layer,
            receiver_positions=list(range(first_span.start, first_span.end)),
            donor_states=donor_states[first_layer][1:] + 0.01,
        )
        smoke_logits = {
            "baseline": logits,
            "head_ablation": ablated_logits,
            "query_patch": query_patched,
            "needle_end_patch": end_patched,
            "needle_span_patch": span_patched,
        }
        if not all(
            bool(torch.isfinite(values[first_candidate_ids]).all())
            for values in smoke_logits.values()
        ):
            raise RuntimeError("Preflight produced a non-finite first-token logit")
        forward = {
            "vocabulary_size": int(logits.numel()),
            "finite_candidate_logits": True,
            "query_attention_shapes": [
                [int(value) for value in row.shape] for row in attention_rows
            ],
            "query_attention_key_starts": [int(value) for value in key_starts],
            "query_attention_row_sum_range": [
                float(min(row.sum(dim=-1).min().item() for row in attention_rows)),
                float(max(row.sum(dim=-1).max().item() for row in attention_rows)),
            ],
            "query_cache_vs_full_candidate_logit_max_abs_delta": float(
                torch.max(
                    torch.abs(
                        attention_logits[first_candidate_ids]
                        - logits[first_candidate_ids]
                    )
                ).item()
            ),
            "span_capture_shapes": {
                key: [int(value) for value in tensor.shape]
                for key, tensor in captured.items()
            },
            "smoke_layers": smoke_layers,
            "head_ablation_max_abs_logit_delta": float(
                torch.max(torch.abs(ablated_logits - logits)).item()
            ),
            "query_patch_max_abs_logit_delta": float(
                torch.max(torch.abs(query_patched - logits)).item()
            ),
            "needle_end_patch_max_abs_logit_delta": float(
                torch.max(torch.abs(end_patched - logits)).item()
            ),
            "needle_span_patch_max_abs_logit_delta": float(
                torch.max(torch.abs(span_patched - logits)).item()
            ),
        }
    return {
        "schema_version": "realistic_niah_v4_preflight_v1",
        "model_label": model_label,
        "answer_format": str(answer_format),
        "decoder_layer_container": adapter.layer_container_name,
        "num_layers": adapter.num_layers,
        "num_heads_by_layer": list(adapter.num_heads),
        "head_dims_by_layer": list(adapter.head_dims),
        "layer_types": list(adapter.layer_types),
        "rendered_sequence_lengths": sequence_lengths,
        "max_position_embeddings": (
            None if max_positions is None else int(max_positions)
        ),
        "count_candidate_texts": {
            str(count): text for count, text in encodings[0].count_candidate_texts
        },
        "count_candidate_answer_token_ids": {
            str(count): list(token_ids)
            for count, token_ids in encodings[0].count_candidate_answer_token_ids
        },
        "count_candidate_scored_token_ids": {
            str(count): list(token_ids)
            for count, token_ids in encodings[0].count_candidate_token_ids
        },
        "query_is_last_token": all(
            encoding.query_position == encoding.sequence_length - 1
            for encoding in encodings
        ),
        "forward_smoke": forward,
    }


def _directed_count_pairs(
    pairs: Sequence[Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    directed: list[tuple[int, int]] = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"Count pair must have length two: {pair}")
        low, high = (int(pair[0]), int(pair[1]))
        if low >= high:
            raise ValueError(f"Registered count pairs must be low-to-high: {pair}")
        directed.extend(((low, high), (high, low)))
    if len(set(directed)) != len(directed):
        raise ValueError("Directed causal count pairs contain duplicates")
    return tuple(directed)


def _validate_generation_causal_shard(
    frame: pd.DataFrame,
    *,
    expected_variant: str | None = None,
    expected_seed: int | None = None,
    expected_stimulus_id: str | None = None,
) -> None:
    required = {
        "model_label",
        "design_variant",
        "baseline_outcome",
        "patched_outcome",
        "patched_completion_text",
        "patched_generated_token_ids",
        "behavior_metric",
    }
    missing = sorted(required - set(frame.columns))
    if frame.empty or missing:
        raise RuntimeError(
            f"Invalid complete-generation causal shard; missing={missing}"
        )
    if expected_variant is not None and set(frame["design_variant"].astype(str)) != {
        str(expected_variant)
    }:
        raise RuntimeError("Causal shard design variant mismatch")
    if expected_seed is not None and set(frame["seed"].astype(int)) != {
        int(expected_seed)
    }:
        raise RuntimeError("Causal shard seed mismatch")
    if expected_stimulus_id is not None and set(
        frame["stimulus_id"].astype(str)
    ) != {str(expected_stimulus_id)}:
        raise RuntimeError("Causal shard stimulus mismatch")


def _write_causal_tables(
    *,
    detail: pd.DataFrame,
    detail_path: Path,
    summary: pd.DataFrame,
    summary_path: Path,
) -> None:
    _write_csv_gzip_atomic(detail, detail_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_name(summary_path.name + ".tmp")
    summary.to_csv(temporary, index=False)
    temporary.replace(summary_path)


def _causal_design_root(
    causal_output: Path,
    family: str,
    settings: dict[str, Any],
) -> Path:
    payload = {
        "schema_version": "realistic_niah_v4_causal_design_v1",
        "family": str(family),
        **settings,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    design_id = hashlib.sha256(encoded).hexdigest()[:12]
    root = causal_output / str(family) / f"design_{design_id}"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "design.json"
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != payload:
            raise RuntimeError(f"Causal design hash collision at {path}")
    else:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    return root


def _ranking_design_payload(
    rankings: dict[tuple[str, str], Sequence[tuple[int, int]]],
) -> dict[str, list[list[int]]]:
    return {
        f"{variant}:{pooling}": [
            [int(layer), int(head)] for layer, head in heads
        ]
        for (variant, pooling), heads in sorted(rankings.items())
    }


def run_model_stage(
    *,
    stage: str,
    stimuli_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    model_label: str,
    answer_format: str,
    cache_dir: str | Path | None = None,
    device_map: str = "auto",
    variants: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    counts: Sequence[int] | None = None,
    representation_all_counts: bool = False,
    overwrite: bool = False,
    forward_smoke: bool = False,
    generation_max_new_tokens: int = 16,
    causal_layers: Sequence[int] | None = None,
    causal_top_ns: Sequence[int] | None = None,
    causal_random_replicates: int | None = None,
    causal_count_pairs: Sequence[Sequence[int]] | None = None,
    ablation_scopes: Sequence[str] | None = None,
    ablation_poolings: Sequence[str] | None = None,
    residual_patch_sites: Sequence[str] | None = None,
    residual_patch_protocols: Sequence[str] | None = None,
    steering_count_pairs: Sequence[Sequence[int]] | None = None,
    steering_methods: Sequence[str] | None = None,
    steering_alphas: Sequence[float] | None = None,
    steering_random_replicates: int | None = None,
) -> dict[str, Any]:
    """Run one GPU-facing V4 stage for one registered model."""

    allowed = {
        "preflight",
        "behavior",
        "representation-capture",
        "answer-query-representation-capture",
        "prompt-counter-attention-capture",
        "attention",
        "ablation",
        "patching",
        "geometric-steering",
    }
    if stage not in allowed:
        raise ValueError(f"Unknown model stage {stage!r}; choose from {allowed}")
    config = V4Config.from_json(config_path)
    answer_format = str(answer_format)
    if answer_format not in config.answer_formats:
        raise ValueError(f"Unregistered V4 answer format: {answer_format}")
    model_spec = resolve_model_spec(model_label)
    model_output = Path(output_dir) / model_spec.label / answer_format
    logger = EventLogger(model_output / "events.jsonl")
    with logger.timer("model_load", model=model_spec.label):
        model, tokenizer, adapter = load_registered_model(
            model_spec,
            cache_dir=cache_dir,
            device_map=device_map,
            torch_dtype=config.model_torch_dtype,
            attention_backend=config.attention_prefix_backend,
        )
    selected = select_stimuli(
        stimuli_path,
        variants=variants,
        seeds=seeds,
        counts=counts,
    )
    representative = []
    for variant in config.design_variants:
        match = next(
            (
                row
                for row in selected
                if row["design_variant"] == variant
                and int(row["gold_count"]) == max(config.needle_counts)
            ),
            None,
        )
        if match is not None:
            representative.append(match)
    if not representative:
        representative = [selected[0]]
    with logger.timer("preflight", model=model_spec.label):
        report = preflight_report(
            model=model,
            tokenizer=tokenizer,
            adapter=adapter,
            representative_rows=representative,
            model_label=model_spec.label,
            config=config,
            answer_format=answer_format,
            forward_smoke=forward_smoke,
        )
        preflight_path = model_output / "preflight.json"
        preflight_path.parent.mkdir(parents=True, exist_ok=True)
        preflight_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if stage == "preflight":
        return {"preflight": str(preflight_path)}

    if stage == "behavior":
        with logger.timer(
            "behavior_generation",
            rows=len(selected),
            decoding="greedy",
            max_new_tokens=int(generation_max_new_tokens),
        ):
            behavior_outputs = capture_generation_labels(
                model,
                tokenizer,
                render_encodings(
                    selected,
                    tokenizer=tokenizer,
                    model_label=model_spec.label,
                    config=config,
                    answer_format=answer_format,
                ),
                output_dir=model_output / "behavior" / "capture",
                valid_counts=tuple(int(value) for value in config.needle_counts),
                max_new_tokens=int(generation_max_new_tokens),
                overwrite=overwrite,
            )
        return {
            "preflight": str(preflight_path),
            **{key: str(value) for key, value in behavior_outputs.items()},
        }

    if stage == "representation-capture":
        representation_rows = [
            row
            for row in selected
            if representation_all_counts
            or int(row["gold_count"]) == config.representation_count
        ]
        with logger.timer(
            "representation_capture",
            rows=len(representation_rows),
        ):
            index_path = capture_representation_shards(
                model,
                adapter,
                render_encodings(
                    representation_rows,
                    tokenizer=tokenizer,
                    model_label=model_spec.label,
                    config=config,
                    answer_format=answer_format,
                ),
                output_dir=model_output / "representation" / "capture",
                save_dtype=config.hidden_save_dtype,
                overwrite=overwrite,
            )
        return {
            "preflight": str(preflight_path),
            "capture_index": str(index_path),
        }

    if stage == "answer-query-representation-capture":
        with logger.timer(
            "answer_query_representation_capture",
            rows=len(selected),
            layers=int(adapter.num_layers),
            position="prompt_final_total_query",
        ):
            index_path = capture_answer_query_representation_shards(
                model,
                adapter,
                render_encodings(
                    selected,
                    tokenizer=tokenizer,
                    model_label=model_spec.label,
                    config=config,
                    answer_format=answer_format,
                ),
                output_dir=(
                    model_output
                    / "representation"
                    / "answer_query_all_layers_v1"
                ),
                save_dtype=config.hidden_save_dtype,
                overwrite=overwrite,
            )
        return {
            "preflight": str(preflight_path),
            "capture_index": str(index_path),
        }

    if stage == "prompt-counter-attention-capture":
        prompt_counter_rows = [
            row
            for row in selected
            if int(row["gold_count"]) == int(config.representation_count)
        ]
        with logger.timer(
            "prompt_counter_attention_capture",
            rows=len(prompt_counter_rows),
            query_site="needle_end",
            query_occurrences=int(config.representation_count),
            key_poolings="needle_end,needle_span_sum",
        ):
            index_path = capture_prompt_counter_attention_shards(
                model,
                adapter,
                render_encodings(
                    prompt_counter_rows,
                    tokenizer=tokenizer,
                    model_label=model_spec.label,
                    config=config,
                    answer_format=answer_format,
                ),
                output_dir=(
                    model_output
                    / "representation"
                    / "prompt_counter_attention_v1"
                ),
                overwrite=overwrite,
            )
        return {
            "preflight": str(preflight_path),
            "capture_index": str(index_path),
        }

    if stage == "attention":
        with logger.timer("attention_capture", rows=len(selected)):
            attention_index = capture_attention_shards(
                model,
                adapter,
                render_encodings(
                    selected,
                    tokenizer=tokenizer,
                    model_label=model_spec.label,
                    config=config,
                    answer_format=answer_format,
                ),
                output_dir=model_output / "attention" / "capture",
                save_raw_rows=config.save_raw_attention_rows,
                save_dtype=config.attention_save_dtype,
                overwrite=overwrite,
            )
        return {
            "preflight": str(preflight_path),
            "capture_index": str(attention_index),
            "analysis_status": (
                "run the CPU-side attention-analyze stage after greedy behavior "
                "labels are complete"
            ),
        }

    if stage in {"patching", "geometric-steering"}:
        confirmation_rows = select_stimuli(
            stimuli_path,
            variants=variants,
            seeds=seeds,
            counts=None,
            split="confirmation",
        )
    else:
        confirmation_rows = [
            row for row in selected if row["split"] == "confirmation"
        ]
    if not confirmation_rows:
        raise ValueError("The causal stage selected no confirmation rows")
    confirmation_encodings = list(
        render_encodings(
            confirmation_rows,
            tokenizer=tokenizer,
            model_label=model_spec.label,
            config=config,
            answer_format=answer_format,
        )
    )
    baseline_labels = load_generation_labels(
        model_output / "behavior" / "capture" / "generation_labels.csv"
    )
    observed_variants = sorted(
        {encoding.design_variant for encoding in confirmation_encodings}
    )
    selected_layers = (
        tuple(sorted({int(layer) for layer in causal_layers}))
        if causal_layers is not None
        else resolve_fractional_layers(
            adapter.num_layers, config.causal_layer_fractions
        )
    )
    if (
        not selected_layers
        or selected_layers[0] < 0
        or selected_layers[-1] >= adapter.num_layers
    ):
        raise ValueError(
            f"Causal layers {selected_layers} are invalid for "
            f"{adapter.num_layers} layers"
        )
    resolved_top_ns = (
        tuple(sorted({int(value) for value in causal_top_ns}))
        if causal_top_ns is not None
        else config.ablation_top_ns
    )
    if not resolved_top_ns or resolved_top_ns[0] <= 0:
        raise ValueError("Causal top-N values must be positive")
    resolved_random_replicates = (
        int(causal_random_replicates)
        if causal_random_replicates is not None
        else int(config.ablation_random_replicates)
    )
    if resolved_random_replicates <= 0:
        raise ValueError("Causal head random replicates must be positive")
    resolved_ablation_scopes = (
        tuple(str(value) for value in ablation_scopes)
        if ablation_scopes is not None
        else config.ablation_scopes
    )
    if (
        not resolved_ablation_scopes
        or len(set(resolved_ablation_scopes)) != len(resolved_ablation_scopes)
        or any(
            scope not in {"answer_query", "global"}
            for scope in resolved_ablation_scopes
        )
    ):
        raise ValueError(f"Invalid ablation scopes: {resolved_ablation_scopes}")
    resolved_ablation_poolings = (
        tuple(str(value) for value in ablation_poolings)
        if ablation_poolings is not None
        else POOLINGS
    )
    if (
        not resolved_ablation_poolings
        or len(set(resolved_ablation_poolings))
        != len(resolved_ablation_poolings)
        or any(
            pooling not in POOLINGS for pooling in resolved_ablation_poolings
        )
    ):
        raise ValueError(
            f"Invalid ablation poolings: {resolved_ablation_poolings}"
        )
    resolved_patch_sites = (
        tuple(str(value) for value in residual_patch_sites)
        if residual_patch_sites is not None
        else tuple(config.patch_sites)
    )
    if (
        not resolved_patch_sites
        or len(set(resolved_patch_sites)) != len(resolved_patch_sites)
        or any(site not in RESIDUAL_PATCH_SITES for site in resolved_patch_sites)
    ):
        raise ValueError(f"Invalid residual patch sites: {resolved_patch_sites}")
    resolved_patch_protocols = (
        tuple(str(value) for value in residual_patch_protocols)
        if residual_patch_protocols is not None
        else tuple(config.residual_patch_protocols)
    )
    if (
        not resolved_patch_protocols
        or len(set(resolved_patch_protocols)) != len(resolved_patch_protocols)
        or any(
            protocol not in NEEDLE_PATCH_PROTOCOLS
            for protocol in resolved_patch_protocols
        )
    ):
        raise ValueError(
            f"Invalid residual patch protocols: {resolved_patch_protocols}"
        )
    resolved_patch_pairs = (
        tuple(tuple(int(item) for item in pair) for pair in causal_count_pairs)
        if causal_count_pairs is not None
        else config.patch_count_pairs
    )
    resolved_steering_pairs = (
        tuple(tuple(int(item) for item in pair) for pair in steering_count_pairs)
        if steering_count_pairs is not None
        else config.steering_count_pairs
    )
    valid_counts = {int(value) for value in config.needle_counts}
    for name, pairs in (
        ("causal_count_pairs", resolved_patch_pairs),
        ("steering_count_pairs", resolved_steering_pairs),
    ):
        if not pairs or any(
            len(pair) != 2
            or int(pair[0]) >= int(pair[1])
            or int(pair[0]) not in valid_counts
            or int(pair[1]) not in valid_counts
            for pair in pairs
        ):
            raise ValueError(f"Invalid {name}: {pairs}")
    resolved_steering_methods = (
        tuple(str(value) for value in steering_methods)
        if steering_methods is not None
        else config.steering_methods
    )
    invalid_methods = sorted(
        set(resolved_steering_methods) - set(config.steering_methods)
    )
    if (
        not resolved_steering_methods
        or len(set(resolved_steering_methods)) != len(resolved_steering_methods)
        or invalid_methods
    ):
        raise ValueError(f"Invalid steering methods: {invalid_methods}")
    resolved_steering_alphas = (
        tuple(sorted({float(value) for value in steering_alphas}))
        if steering_alphas is not None
        else config.steering_alphas
    )
    if not resolved_steering_alphas or any(
        not 0.0 < alpha <= 1.0 for alpha in resolved_steering_alphas
    ):
        raise ValueError("Steering alphas must lie in (0, 1]")
    resolved_steering_random_replicates = (
        int(steering_random_replicates)
        if steering_random_replicates is not None
        else int(config.steering_random_replicates)
    )
    if resolved_steering_random_replicates < 0:
        raise ValueError("Steering random replicates must be nonnegative")
    causal_output = model_output / "causal"
    causal_output.mkdir(parents=True, exist_ok=True)
    behavior_metric = "strict_greedy_complete_numeric_generation"
    selection_payload = {
        "stimuli_sha256": _sha256_file(Path(stimuli_path)),
        "model_label": model_spec.label,
        "answer_format": answer_format,
        "confirmation_variants": observed_variants,
        "confirmation_seeds": sorted(
            {int(encoding.seed) for encoding in confirmation_encodings}
        ),
        "confirmation_counts": sorted(
            {int(encoding.count) for encoding in confirmation_encodings}
        ),
        "generation_max_new_tokens": int(generation_max_new_tokens),
        "behavior_metric": behavior_metric,
    }

    if stage == "ablation":
        rankings = load_broad_rankings(
            model_output / "attention" / "analysis" / "rankings",
            variants=observed_variants,
            poolings=resolved_ablation_poolings,
        )

    if stage == "ablation":
        stage_root = _causal_design_root(
            causal_output,
            "generation_head_ablation_v1",
            {
                **selection_payload,
                "rankings": _ranking_design_payload(rankings),
                "poolings": list(resolved_ablation_poolings),
                "top_ns": list(resolved_top_ns),
                "random_replicates": resolved_random_replicates,
                "scopes": list(resolved_ablation_scopes),
            },
        )
        capture_root = stage_root / "capture"
        index_rows: list[dict[str, Any]] = []
        with logger.timer(
            "generation_head_ablation",
            rows=len(confirmation_encodings),
            poolings=list(resolved_ablation_poolings),
            scopes=list(resolved_ablation_scopes),
        ):
            for encoding in confirmation_encodings:
                relative = (
                    Path("shards")
                    / encoding.design_variant
                    / f"{encoding.stimulus_id}.csv.gz"
                )
                shard = capture_root / relative
                if shard.exists() and not overwrite:
                    frame = pd.read_csv(shard, compression="gzip")
                    _validate_generation_causal_shard(
                        frame, expected_stimulus_id=encoding.stimulus_id
                    )
                else:
                    frame = run_generation_head_ablation(
                        model,
                        tokenizer,
                        adapter,
                        [encoding],
                        baseline_labels=baseline_labels,
                        rankings=rankings,
                        poolings=resolved_ablation_poolings,
                        top_ns=resolved_top_ns,
                        random_replicates=resolved_random_replicates,
                        scopes=resolved_ablation_scopes,
                        max_new_tokens=generation_max_new_tokens,
                    )
                    frame["behavior_metric"] = behavior_metric
                    _write_csv_gzip_atomic(frame, shard)
                index_rows.append(
                    {
                        "stimulus_id": encoding.stimulus_id,
                        "design_variant": encoding.design_variant,
                        "seed": int(encoding.seed),
                        "count": int(encoding.count),
                        "rows": len(frame),
                        "shard_path": relative.as_posix(),
                    }
                )
        index_path = capture_root / "capture_index.jsonl"
        _write_shard_index(index_rows, index_path)
        detail = _load_csv_gzip_shards(index_rows, root=capture_root)
        detail_path = stage_root / "detail.csv.gz"
        summary_path = stage_root / "summary.csv"
        comparison_path = stage_root / "broad_vs_layer_matched_random.csv"
        _write_causal_tables(
            detail=detail,
            detail_path=detail_path,
            summary=summarize_generation_head_ablation(detail),
            summary_path=summary_path,
        )
        compare_ranked_ablation_to_random(
            detail,
            bootstrap_repetitions=config.causal_bootstrap_repetitions,
        ).to_csv(comparison_path, index=False)
        return {
            "preflight": str(preflight_path),
            "design": str(stage_root / "design.json"),
            "capture_index": str(index_path),
            "detail": str(detail_path),
            "summary": str(summary_path),
            "broad_vs_random": str(comparison_path),
        }

    grouped_encodings: dict[tuple[str, int], list[PromptEncoding]] = {}
    for encoding in confirmation_encodings:
        grouped_encodings.setdefault(
            (encoding.design_variant, int(encoding.seed)), []
        ).append(encoding)

    if stage == "patching":
        directed_pairs = _directed_count_pairs(resolved_patch_pairs)
        stage_root = _causal_design_root(
            causal_output,
            "generation_residual_patching_v1",
            {
                **selection_payload,
                "layers": list(selected_layers),
                "directed_count_pairs": [list(pair) for pair in directed_pairs],
                "sites": list(resolved_patch_sites),
                "needle_protocols": list(resolved_patch_protocols),
            },
        )
        capture_root = stage_root / "capture"
        index_rows = []
        with logger.timer(
            "generation_residual_patching",
            families=len(grouped_encodings),
            layers=list(selected_layers),
            sites=list(resolved_patch_sites),
            protocols=list(resolved_patch_protocols),
        ):
            for (variant, seed), family_encodings in sorted(
                grouped_encodings.items()
            ):
                relative = (
                    Path("shards")
                    / variant
                    / f"{variant.replace('.', '_')}_seed{seed}.csv.gz"
                )
                shard = capture_root / relative
                if shard.exists() and not overwrite:
                    frame = pd.read_csv(shard, compression="gzip")
                    _validate_generation_causal_shard(
                        frame[frame["status"] == "ok"],
                        expected_variant=variant,
                        expected_seed=seed,
                    )
                else:
                    frame = run_generation_residual_patching(
                        model,
                        tokenizer,
                        adapter,
                        family_encodings,
                        baseline_labels=baseline_labels,
                        count_pairs=directed_pairs,
                        start_layers=selected_layers,
                        sites=resolved_patch_sites,
                        needle_protocols=resolved_patch_protocols,
                        max_new_tokens=generation_max_new_tokens,
                    )
                    frame["behavior_metric"] = behavior_metric
                    _write_csv_gzip_atomic(frame, shard)
                index_rows.append(
                    {
                        "design_variant": variant,
                        "seed": int(seed),
                        "rows": len(frame),
                        "successful_rows": int((frame["status"] == "ok").sum()),
                        "skipped_rows": int((frame["status"] != "ok").sum()),
                        "shard_path": relative.as_posix(),
                    }
                )
        index_path = capture_root / "capture_index.jsonl"
        _write_shard_index(index_rows, index_path)
        detail = _load_csv_gzip_shards(index_rows, root=capture_root)
        detail_path = stage_root / "detail.csv.gz"
        summary_path = stage_root / "summary.csv"
        _write_causal_tables(
            detail=detail,
            detail_path=detail_path,
            summary=summarize_generation_residual_patching(detail),
            summary_path=summary_path,
        )
        return {
            "preflight": str(preflight_path),
            "design": str(stage_root / "design.json"),
            "capture_index": str(index_path),
            "detail": str(detail_path),
            "summary": str(summary_path),
        }

    if stage != "geometric-steering":
        raise AssertionError(f"Unhandled V4 stage: {stage}")
    discovery_rows = select_stimuli(
        stimuli_path,
        variants=observed_variants,
        seeds=config.discovery_seeds,
        counts=config.needle_counts,
        split="discovery",
    )
    discovery_encodings = render_encodings(
        discovery_rows,
        tokenizer=tokenizer,
        model_label=model_spec.label,
        config=config,
        answer_format=answer_format,
    )
    directed_pairs = _directed_count_pairs(resolved_steering_pairs)
    stage_root = _causal_design_root(
        causal_output,
        "geometric_steering_v1",
        {
            **selection_payload,
            "discovery_seeds": list(config.discovery_seeds),
            "layers": list(selected_layers),
            "directed_count_pairs": [list(pair) for pair in directed_pairs],
            "methods": list(resolved_steering_methods),
            "alphas": list(resolved_steering_alphas),
            "orthogonal_random_replicates": (
                resolved_steering_random_replicates
            ),
        },
    )
    discovery_capture_root = stage_root / "discovery_capture"
    discovery_index_rows: list[dict[str, Any]] = []
    with logger.timer(
        "geometric_steering_discovery_capture",
        rows=len(discovery_rows),
        layers=list(selected_layers),
    ):
        for encoding in discovery_encodings:
            relative = (
                Path("shards")
                / encoding.design_variant
                / f"{encoding.stimulus_id}.npz"
            )
            metadata = capture_query_residual_shard(
                model,
                adapter,
                encoding,
                layers=selected_layers,
                path=discovery_capture_root / relative,
                save_dtype=config.hidden_save_dtype,
                overwrite=overwrite,
            )
            discovery_index_rows.append(
                {**metadata, "shard_path": relative.as_posix()}
            )
    discovery_index_path = discovery_capture_root / "capture_index.jsonl"
    _write_shard_index(discovery_index_rows, discovery_index_path)
    centroids = fit_count_centroids(
        discovery_index_rows,
        capture_root=discovery_capture_root,
        variants=observed_variants,
        layers=selected_layers,
        counts=config.needle_counts,
        discovery_seeds=config.discovery_seeds,
    )
    centroid_path = save_centroid_bundle(centroids, stage_root / "centroids.npz")
    geometry_summary, geometry_adjacent = centroid_geometry_tables(centroids)
    geometry_summary_path = stage_root / "centroid_geometry_summary.csv"
    geometry_adjacent_path = stage_root / "centroid_adjacent_steps.csv"
    geometry_summary.to_csv(geometry_summary_path, index=False)
    geometry_adjacent.to_csv(geometry_adjacent_path, index=False)

    evaluation_root = stage_root / "confirmation_capture"
    evaluation_index_rows: list[dict[str, Any]] = []
    with logger.timer(
        "geometric_steering_confirmation_generation",
        families=len(grouped_encodings),
        layers=list(selected_layers),
        methods=list(resolved_steering_methods),
        alphas=list(resolved_steering_alphas),
    ):
        for (variant, seed), family_encodings in sorted(grouped_encodings.items()):
            relative = (
                Path("shards")
                / variant
                / f"{variant.replace('.', '_')}_seed{seed}.csv.gz"
            )
            shard = evaluation_root / relative
            if shard.exists() and not overwrite:
                frame = pd.read_csv(shard, compression="gzip")
                _validate_generation_causal_shard(
                    frame, expected_variant=variant, expected_seed=seed
                )
            else:
                frame = run_generation_geometric_steering(
                    model,
                    tokenizer,
                    adapter,
                    family_encodings,
                    baseline_labels=baseline_labels,
                    centroids=centroids,
                    count_pairs=directed_pairs,
                    layers=selected_layers,
                    methods=resolved_steering_methods,
                    alphas=resolved_steering_alphas,
                    random_replicates=resolved_steering_random_replicates,
                    max_new_tokens=generation_max_new_tokens,
                )
                frame["behavior_metric"] = behavior_metric
                _write_csv_gzip_atomic(frame, shard)
            evaluation_index_rows.append(
                {
                    "design_variant": variant,
                    "seed": int(seed),
                    "rows": len(frame),
                    "shard_path": relative.as_posix(),
                }
            )
    evaluation_index_path = evaluation_root / "capture_index.jsonl"
    _write_shard_index(evaluation_index_rows, evaluation_index_path)
    detail = _load_csv_gzip_shards(
        evaluation_index_rows, root=evaluation_root
    )
    detail_path = stage_root / "detail.csv.gz"
    summary_path = stage_root / "summary.csv"
    comparison_path = stage_root / "geometric_vs_random.csv"
    _write_causal_tables(
        detail=detail,
        detail_path=detail_path,
        summary=summarize_generation_geometric_steering(detail),
        summary_path=summary_path,
    )
    compare_geometric_to_random(
        detail,
        bootstrap_repetitions=config.causal_bootstrap_repetitions,
    ).to_csv(comparison_path, index=False)
    return {
        "preflight": str(preflight_path),
        "design": str(stage_root / "design.json"),
        "discovery_capture_index": str(discovery_index_path),
        "centroids": str(centroid_path),
        "centroid_geometry_summary": str(geometry_summary_path),
        "centroid_adjacent_steps": str(geometry_adjacent_path),
        "confirmation_capture_index": str(evaluation_index_path),
        "detail": str(detail_path),
        "summary": str(summary_path),
        "geometric_vs_random": str(comparison_path),
    }


def run_representation_analysis(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    model_label: str,
    answer_format: str,
) -> dict[str, str]:
    config = V4Config.from_json(config_path)
    model_spec = resolve_model_spec(model_label)
    answer_format = str(answer_format)
    if answer_format not in config.answer_formats:
        raise ValueError(f"Unregistered V4 answer format: {answer_format}")
    model_output = Path(output_dir) / model_spec.label / answer_format
    outputs = analyze_representation_captures(
        capture_index_path=(
            model_output / "representation" / "capture" / "capture_index.jsonl"
        ),
        output_dir=model_output / "representation" / "analysis",
        config=config,
    )
    return {key: str(value) for key, value in outputs.items()}


def run_labeled_attention_analysis(
    *,
    stimuli_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    model_label: str,
    answer_format: str,
    cache_dir: str | Path | None = None,
    variants: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    counts: Sequence[int] | None = None,
    overwrite_pooling_metrics: bool = False,
) -> dict[str, str]:
    """Run CPU-side behavior-stratified analysis of saved attention rows."""

    config = V4Config.from_json(config_path)
    model_spec = resolve_model_spec(model_label)
    answer_format = str(answer_format)
    if answer_format not in config.answer_formats:
        raise ValueError(f"Unregistered V4 answer format: {answer_format}")
    model_output = Path(output_dir) / model_spec.label / answer_format
    tokenizer = load_registered_tokenizer(model_spec, cache_dir=cache_dir)
    selected = select_stimuli(
        stimuli_path,
        variants=variants,
        seeds=seeds,
        counts=counts,
    )
    outputs = analyze_labeled_attention(
        attention_index_path=(
            model_output / "attention" / "capture" / "attention_capture_index.jsonl"
        ),
        generation_labels_path=(
            model_output / "behavior" / "capture" / "generation_labels.csv"
        ),
        encodings=render_encodings(
            selected,
            tokenizer=tokenizer,
            model_label=model_spec.label,
            config=config,
            answer_format=answer_format,
        ),
        output_dir=model_output / "attention" / "analysis",
        top_k=8,
        overwrite_pooling_metrics=overwrite_pooling_metrics,
    )
    representation_outputs = label_representation_analysis_by_generation(
        analysis_dir=model_output / "representation" / "analysis",
        generation_labels_path=(
            model_output / "behavior" / "capture" / "generation_labels.csv"
        ),
        output_dir=model_output / "representation" / "analysis" / "outcomes",
    )
    outputs.update(
        {
            f"representation_{key}": value
            for key, value in representation_outputs.items()
        }
    )
    return {key: str(value) for key, value in outputs.items()}
