from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from .modeling import DecoderAdapter, position_attention_outputs
from .prompts import PromptEncoding


SCHEMA_VERSION = "realistic_niah_v4_prompt_counter_attention_v1"
METRIC_COLUMNS = (
    "stimulus_id",
    "design_variant",
    "model_label",
    "seed",
    "split",
    "count",
    "sequence_length",
    "query_occurrence",
    "query_position",
    "query_site",
    "layer",
    "head",
    "layer_type",
    "attention_key_start",
    "attention_key_length",
    "attention_row_sum",
    "row_entropy_nats",
    "row_effective_tokens",
    "row_effective_fraction",
    "row_top1_mass",
    "row_top10_mass",
    "row_top100_mass",
    "visible_needle_endpoints",
    "visible_complete_needle_spans",
    "needle_end_total_mass",
    "needle_end_effective_number",
    "needle_end_relative_coverage",
    "needle_end_current_share",
    "needle_span_total_mass",
    "needle_span_effective_number",
    "needle_span_relative_coverage",
    "needle_span_current_share",
    "current_endpoint_self_mass",
    "current_span_mass",
    "current_span_without_endpoint_mass",
    "prior_needle_endpoint_mass",
    "prior_needle_span_mass",
    "non_needle_mass",
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_csv_gzip(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _slice_mass(
    rows: np.ndarray,
    *,
    start: int,
    end: int,
    key_start: int,
) -> np.ndarray:
    local_start = max(int(start), int(key_start)) - int(key_start)
    local_end = min(int(end), int(key_start) + rows.shape[1]) - int(key_start)
    if local_end <= local_start:
        return np.zeros(rows.shape[0], dtype=np.float64)
    return rows[:, local_start:local_end].sum(axis=1, dtype=np.float64)


def _profile_metrics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("Occurrence profiles must be [heads, occurrences]")
    total = values.sum(axis=1)
    probabilities = np.divide(
        values,
        total[:, None],
        out=np.zeros_like(values),
        where=total[:, None] > 0,
    )
    safe_probabilities = np.where(probabilities > 0, probabilities, 1.0)
    entropy = -np.sum(probabilities * np.log(safe_probabilities), axis=1)
    effective = np.where(total > 0, np.exp(entropy), 0.0)
    coverage = effective / float(values.shape[1])
    return total, effective, coverage


def _prompt_counter_attention_payload(
    attention_rows: Sequence[torch.Tensor | np.ndarray],
    key_starts: Sequence[int],
    *,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    occurrence_index: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Summarize one needle-end query row without saving full token maps.

    ``occurrence_index`` is zero based.  The attention query is the final token
    of that needle span.  Two key pools are kept separate: endpoint keys and
    every token in each complete needle span.  A span-mean hidden state has no
    unique native attention row, so it is intentionally not mislabeled as one.
    """

    spans = tuple(encoding.needle_spans)
    if not 0 <= int(occurrence_index) < len(spans):
        raise ValueError("Needle occurrence index is outside the prompt")
    if len(attention_rows) != int(adapter.num_layers):
        raise ValueError("Attention layer count does not match the decoder")
    if len(key_starts) != int(adapter.num_layers):
        raise ValueError("Attention key-start count does not match the decoder")
    active_spans = spans[: int(occurrence_index) + 1]
    current = active_spans[-1]
    query_position = int(current.end) - 1
    records: list[dict[str, Any]] = []
    endpoint_profiles: list[np.ndarray] = []
    span_profiles: list[np.ndarray] = []
    profile_layers: list[np.ndarray] = []
    profile_heads: list[np.ndarray] = []

    for layer, (tensor, key_start) in enumerate(zip(attention_rows, key_starts)):
        values = (
            tensor.detach().float().cpu().numpy()
            if isinstance(tensor, torch.Tensor)
            else np.asarray(tensor, dtype=np.float32)
        )
        if values.ndim != 2 or values.shape[0] != int(adapter.num_heads[layer]):
            raise ValueError(
                f"Layer {layer} attention rows have invalid shape {values.shape}"
            )
        values = np.asarray(values, dtype=np.float64)
        row_sum = values.sum(axis=1)
        probabilities = np.divide(
            values,
            row_sum[:, None],
            out=np.zeros_like(values),
            where=row_sum[:, None] > 0,
        )
        safe_probabilities = np.where(probabilities > 0, probabilities, 1.0)
        entropy = -np.sum(
            probabilities * np.log(safe_probabilities),
            axis=1,
        )
        effective_tokens = np.where(row_sum > 0, np.exp(entropy), 0.0)
        effective_fraction = effective_tokens / float(values.shape[1])

        def top_mass(k: int) -> np.ndarray:
            resolved = min(int(k), int(values.shape[1]))
            if resolved == values.shape[1]:
                return probabilities.sum(axis=1)
            partition = np.partition(
                probabilities,
                values.shape[1] - resolved,
                axis=1,
            )
            return partition[:, -resolved:].sum(axis=1)

        endpoint_profile = np.stack(
            [
                _slice_mass(
                    values,
                    start=int(span.end) - 1,
                    end=int(span.end),
                    key_start=int(key_start),
                )
                for span in active_spans
            ],
            axis=1,
        )
        span_profile = np.stack(
            [
                _slice_mass(
                    values,
                    start=int(span.start),
                    end=int(span.end),
                    key_start=int(key_start),
                )
                for span in active_spans
            ],
            axis=1,
        )
        endpoint_profiles.append(endpoint_profile)
        span_profiles.append(span_profile)
        profile_layers.append(
            np.full(values.shape[0], int(layer), dtype=np.int16)
        )
        profile_heads.append(np.arange(values.shape[0], dtype=np.int16))
        endpoint_total, endpoint_effective, endpoint_coverage = _profile_metrics(
            endpoint_profile
        )
        span_total, span_effective, span_coverage = _profile_metrics(span_profile)
        current_endpoint = endpoint_profile[:, -1]
        current_span = span_profile[:, -1]
        prior_endpoint = endpoint_profile[:, :-1].sum(axis=1)
        prior_span = span_profile[:, :-1].sum(axis=1)
        endpoint_share = np.divide(
            current_endpoint,
            endpoint_total,
            out=np.zeros_like(current_endpoint),
            where=endpoint_total > 0,
        )
        span_share = np.divide(
            current_span,
            span_total,
            out=np.zeros_like(current_span),
            where=span_total > 0,
        )
        top1 = top_mass(1)
        top10 = top_mass(10)
        top100 = top_mass(100)
        key_end = int(key_start) + int(values.shape[1])
        visible_endpoints = sum(
            int(key_start) <= int(span.end) - 1 < key_end
            for span in active_spans
        )
        visible_spans = sum(
            int(key_start) <= int(span.start) and int(span.end) <= key_end
            for span in active_spans
        )
        for head in range(values.shape[0]):
            records.append(
                {
                    "stimulus_id": encoding.stimulus_id,
                    "design_variant": encoding.design_variant,
                    "model_label": encoding.model_label,
                    "seed": int(encoding.seed),
                    "split": encoding.split,
                    "count": int(encoding.count),
                    "sequence_length": int(encoding.sequence_length),
                    "query_occurrence": int(occurrence_index) + 1,
                    "query_position": query_position,
                    "query_site": "needle_end",
                    "layer": int(layer),
                    "head": int(head),
                    "layer_type": str(adapter.layer_types[layer]),
                    "attention_key_start": int(key_start),
                    "attention_key_length": int(values.shape[1]),
                    "attention_row_sum": float(row_sum[head]),
                    "row_entropy_nats": float(entropy[head]),
                    "row_effective_tokens": float(effective_tokens[head]),
                    "row_effective_fraction": float(effective_fraction[head]),
                    "row_top1_mass": float(top1[head]),
                    "row_top10_mass": float(top10[head]),
                    "row_top100_mass": float(top100[head]),
                    "visible_needle_endpoints": int(visible_endpoints),
                    "visible_complete_needle_spans": int(visible_spans),
                    "needle_end_total_mass": float(endpoint_total[head]),
                    "needle_end_effective_number": float(endpoint_effective[head]),
                    "needle_end_relative_coverage": float(endpoint_coverage[head]),
                    "needle_end_current_share": float(endpoint_share[head]),
                    "needle_span_total_mass": float(span_total[head]),
                    "needle_span_effective_number": float(span_effective[head]),
                    "needle_span_relative_coverage": float(span_coverage[head]),
                    "needle_span_current_share": float(span_share[head]),
                    "current_endpoint_self_mass": float(current_endpoint[head]),
                    "current_span_mass": float(current_span[head]),
                    "current_span_without_endpoint_mass": float(
                        max(0.0, current_span[head] - current_endpoint[head])
                    ),
                    "prior_needle_endpoint_mass": float(prior_endpoint[head]),
                    "prior_needle_span_mass": float(prior_span[head]),
                    "non_needle_mass": float(
                        max(0.0, row_sum[head] - span_total[head])
                    ),
                }
            )
    return (
        pd.DataFrame.from_records(records, columns=METRIC_COLUMNS),
        np.concatenate(endpoint_profiles, axis=0),
        np.concatenate(span_profiles, axis=0),
        np.concatenate(profile_layers, axis=0),
        np.concatenate(profile_heads, axis=0),
    )


def prompt_counter_attention_metrics(
    attention_rows: Sequence[torch.Tensor | np.ndarray],
    key_starts: Sequence[int],
    *,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    occurrence_index: int,
) -> pd.DataFrame:
    """Return headwise metrics for one needle-end attention query."""

    frame, _endpoint, _span, _layers, _heads = (
        _prompt_counter_attention_payload(
            attention_rows,
            key_starts,
            adapter=adapter,
            encoding=encoding,
            occurrence_index=occurrence_index,
        )
    )
    return frame


@torch.inference_mode()
def capture_prompt_counter_attention_shards(
    model: Any,
    adapter: DecoderAdapter,
    encodings: Iterable[PromptEncoding],
    *,
    output_dir: str | Path,
    overwrite: bool = False,
) -> Path:
    """Capture compact write-side attention diagnostics for N=10 prompts."""

    output = Path(output_dir)
    index_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    expected_rows = int(sum(adapter.num_heads)) * 10
    for example_index, encoding in enumerate(encodings):
        if int(encoding.count) != 10 or len(encoding.needle_spans) != 10:
            raise ValueError("Prompt-counter attention capture requires N=10")
        if encoding.stimulus_id in seen:
            raise ValueError(
                f"Duplicate prompt-counter attention stimulus: {encoding.stimulus_id}"
            )
        seen.add(encoding.stimulus_id)
        relative = (
            Path("shards")
            / encoding.design_variant
            / f"{encoding.stimulus_id}.csv.gz"
        )
        shard = output / relative
        profile_relative = (
            Path("profiles")
            / encoding.design_variant
            / f"{encoding.stimulus_id}.npz"
        )
        profile_shard = output / profile_relative
        started = time.perf_counter()
        if shard.exists() and profile_shard.exists() and not overwrite:
            frame = pd.read_csv(shard)
            if tuple(frame.columns) != METRIC_COLUMNS:
                raise RuntimeError(f"Prompt-counter attention schema mismatch: {shard}")
            if len(frame) != expected_rows:
                raise RuntimeError(
                    f"Prompt-counter attention row mismatch in {shard}: "
                    f"{len(frame)} != {expected_rows}"
                )
            with np.load(profile_shard, allow_pickle=False) as payload:
                expected_profile_shape = (int(sum(adapter.num_heads)), 10, 10)
                for key in ("needle_end_mass", "needle_span_mass"):
                    if tuple(payload[key].shape) != expected_profile_shape:
                        raise RuntimeError(
                            f"Prompt-counter profile mismatch in {profile_shard}: "
                            f"{key}={payload[key].shape}"
                        )
        else:
            pieces: list[pd.DataFrame] = []
            endpoint_maps = np.zeros(
                (int(sum(adapter.num_heads)), 10, 10), dtype=np.float32
            )
            span_maps = np.zeros_like(endpoint_maps)
            profile_layers: np.ndarray | None = None
            profile_heads: np.ndarray | None = None
            for occurrence_index, span in enumerate(encoding.needle_spans):
                rows, key_starts, _logits = position_attention_outputs(
                    model,
                    adapter,
                    encoding,
                    int(span.end) - 1,
                )
                piece, endpoint_profile, span_profile, layers, heads = (
                    _prompt_counter_attention_payload(
                        rows,
                        key_starts,
                        adapter=adapter,
                        encoding=encoding,
                        occurrence_index=occurrence_index,
                    )
                )
                pieces.append(piece)
                endpoint_maps[:, occurrence_index, : occurrence_index + 1] = (
                    endpoint_profile
                )
                span_maps[:, occurrence_index, : occurrence_index + 1] = (
                    span_profile
                )
                if profile_layers is None:
                    profile_layers = layers
                    profile_heads = heads
                elif not np.array_equal(profile_layers, layers) or not np.array_equal(
                    profile_heads, heads
                ):
                    raise RuntimeError("Prompt-counter head grid changed by occurrence")
            frame = pd.concat(pieces, ignore_index=True)
            if len(frame) != expected_rows:
                raise RuntimeError(
                    f"Prompt-counter attention produced {len(frame)} rows; "
                    f"expected {expected_rows}"
                )
            _atomic_csv_gzip(frame, shard)
            assert profile_layers is not None and profile_heads is not None
            _atomic_npz(
                profile_shard,
                layer_indices=profile_layers,
                head_indices=profile_heads,
                query_occurrences=np.arange(1, 11, dtype=np.int16),
                key_occurrences=np.arange(1, 11, dtype=np.int16),
                needle_end_mass=endpoint_maps.astype(np.float16),
                needle_span_mass=span_maps.astype(np.float16),
            )
        index_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "example_index": int(example_index),
                "stimulus_id": encoding.stimulus_id,
                "design_variant": encoding.design_variant,
                "model_label": encoding.model_label,
                "seed": int(encoding.seed),
                "split": encoding.split,
                "count": int(encoding.count),
                "query_site": "needle_end",
                "query_occurrences": 10,
                "layers": int(adapter.num_layers),
                "heads_by_layer": [int(value) for value in adapter.num_heads],
                "rows": int(len(frame)),
                "shard_path": relative.as_posix(),
                "profile_path": profile_relative.as_posix(),
            }
        )
        print(
            "[v4 prompt-counter attention] "
            f"{example_index + 1} variant={encoding.design_variant} "
            f"seed={encoding.seed} rows={len(frame)} "
            f"elapsed_seconds={time.perf_counter() - started:.3f}",
            flush=True,
        )
    if not index_rows:
        raise ValueError("No prompt-counter attention encodings were supplied")
    index_path = output / "capture_index.jsonl"
    _atomic_jsonl(index_path, index_rows)
    _atomic_json(
        output / "capture_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "rows": len(index_rows),
            "stimulus_count": len(index_rows),
            "design_variants": sorted(
                {str(row["design_variant"]) for row in index_rows}
            ),
            "splits": sorted({str(row["split"]) for row in index_rows}),
            "query_site": "needle_end",
            "query_occurrences": 10,
            "key_poolings": ["needle_end", "needle_span_sum"],
            "full_attention_rows_saved": False,
            "per_occurrence_needle_profiles_saved": True,
            "profile_dtype": "float16",
            "restartable_shards": True,
            "interpretation": (
                "Write-side attention at each needle's final query token. "
                "The endpoint and full-span labels refer to key pooling; a "
                "span-mean hidden state has no single native attention row."
            ),
        },
    )
    return index_path
