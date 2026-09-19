#!/usr/bin/env python3
"""Capture one matched native-thinking seed for the main figure.

The script performs ten single-query forwards (the block-entry 0->1 state and
item endpoints 1->2 through 9->10), screens every Qwen3-8B attention head for
one fixed head that targets the next record consistently, and then reruns the
three high-mass item endpoints 2->3, 3->4, and 4->5 to save exact prompt-token
attention vectors for the paper demo.

No full T-by-T attention matrix is materialized.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2] / "realistic"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.modeling import (  # noqa: E402
    load_registered_model,
    position_attention_outputs,
)
from realistic_niah_v4.spec import resolve_model_spec  # noqa: E402
from realistic_niah_v5.causal_sites import compile_causal_site_plan  # noqa: E402
from realistic_niah_v5.encoding import (  # noqa: E402
    build_native_causal_encoding,
    build_native_trace_encoding,
)
from realistic_niah_v5.pipeline import read_jsonl  # noqa: E402


MODEL_LABEL = "Qwen3-8B"
TOKEN_VECTOR_SOURCES = (2, 3, 4)


def sorted_record_spans(encoding: Any) -> list[Any]:
    return sorted(
        encoding.prompt_record_spans,
        key=lambda value: (int(value.start), int(value.end)),
    )


def record_mass_cube(
    attention_rows: Any,
    key_starts: Any,
    prompt_record_spans: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return [layer, head, record] masses and [layer, head] row sums."""
    import torch

    spans = sorted(
        prompt_record_spans,
        key=lambda value: (int(value.start), int(value.end)),
    )
    layer_masses: list[np.ndarray] = []
    layer_sums: list[np.ndarray] = []
    for layer, attention in enumerate(attention_rows):
        key_start = int(key_starts[layer])
        key_end = key_start + int(attention.shape[-1])
        cumulative = torch.cat(
            [
                torch.zeros(
                    (int(attention.shape[0]), 1),
                    dtype=torch.float32,
                    device=attention.device,
                ),
                attention.float().cumsum(dim=-1),
            ],
            dim=-1,
        )
        masses = []
        for span in spans:
            overlap_start = max(int(span.start), key_start)
            overlap_end = min(int(span.end), key_end)
            if overlap_end <= overlap_start:
                masses.append(torch.zeros_like(cumulative[:, 0]))
            else:
                local_start = overlap_start - key_start
                local_end = overlap_end - key_start
                masses.append(cumulative[:, local_end] - cumulative[:, local_start])
        layer_masses.append(
            torch.stack(masses, dim=-1).detach().cpu().numpy().astype(np.float64)
        )
        layer_sums.append(
            attention.float().sum(dim=-1).detach().cpu().numpy().astype(np.float64)
        )
    return np.stack(layer_masses), np.stack(layer_sums)


def prompt_vector(
    attention: Any,
    *,
    key_start: int,
    prompt_tokens: int,
) -> np.ndarray:
    result = np.zeros(int(prompt_tokens), dtype=np.float32)
    source_start = max(0, -int(key_start))
    target_start = max(0, int(key_start))
    available = min(
        int(attention.shape[-1]) - source_start,
        int(prompt_tokens) - target_start,
    )
    if available > 0:
        result[target_start : target_start + available] = (
            attention[source_start : source_start + available]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
    return result


def event_encoding(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    source: int,
    causal_plan: Mapping[str, Any],
) -> tuple[Any, str, dict[str, Any]]:
    if source == 0:
        selected = dict(causal_plan["block_pre_d1"])
        if selected.get("status") != "ok":
            raise RuntimeError(f"block_pre_d1 is unavailable: {selected}")
        encoding = build_native_causal_encoding(
            row,
            tokenizer,
            query_output_token_index=int(selected["output_token_index"]),
            sequence_output_token_end=int(selected["output_prefix_token_count"]),
            selected_site=selected,
        )
        return encoding, "initial_block_pre_d1", selected

    site_id = f"item_end:{source}"
    encoding = build_native_trace_encoding(row, tokenizer, site_id=site_id)
    return encoding, site_id, dict(encoding.selected_site)


def rank_fixed_heads(mass_stack: np.ndarray) -> list[dict[str, Any]]:
    """Rank fixed heads on 1->2 through 9->10, excluding block initiation."""
    target_raw = np.stack(
        [mass_stack[source, :, :, source] for source in range(1, 10)],
        axis=0,
    )
    needle_total = mass_stack[1:].sum(axis=-1)
    target_share = target_raw / np.maximum(needle_total, 1e-12)
    geometric_raw = np.exp(np.mean(np.log(np.maximum(target_raw, 1e-10)), axis=0))
    mean_share = target_share.mean(axis=0)
    minimum_share = target_share.min(axis=0)
    score = geometric_raw * np.sqrt(np.maximum(mean_share, 0.0))

    ranking: list[dict[str, Any]] = []
    for flat_index in np.argsort(score, axis=None)[::-1][:25]:
        layer, head = np.unravel_index(flat_index, score.shape)
        ranking.append(
            {
                "layer": int(layer),
                "head": int(head),
                "selection_score": float(score[layer, head]),
                "geometric_mean_target_raw_mass": float(
                    geometric_raw[layer, head]
                ),
                "mean_target_share_of_needle_mass": float(mean_share[layer, head]),
                "minimum_target_share_of_needle_mass": float(
                    minimum_share[layer, head]
                ),
                "target_raw_masses": target_raw[:, layer, head].tolist(),
                "target_shares": target_share[:, layer, head].tolist(),
            }
        )
    return ranking


def records_payload(
    spans: list[Any],
    masses: np.ndarray,
    *,
    target_occurrence: int,
) -> list[dict[str, Any]]:
    return [
        {
            "source_index": index,
            "slot_index": int(getattr(span, "slot_index", index)),
            "city": str(span.city),
            "score": None if span.score is None else int(span.score),
            "token_start": int(span.start),
            "token_end": int(span.end),
            "visible_token_count": int(span.end) - int(span.start),
            "mass": float(masses[index - 1]),
            "is_target": index == int(target_occurrence),
        }
        for index, span in enumerate(spans, start=1)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    request_id = (
        f"{MODEL_LABEL}/native_thinking/v5/"
        f"V4_4_T10000_N10_seed{args.seed}"
    )
    row = next(
        (
            candidate
            for candidate in read_jsonl(args.generations)
            if str(candidate.get("request_id")) == request_id
        ),
        None,
    )
    if row is None:
        raise KeyError(f"Request not found: {request_id}")
    if int(row["gold_count"]) != 10:
        raise RuntimeError("The selected row is not N=10")

    spec = resolve_model_spec(MODEL_LABEL)
    model, tokenizer, adapter = load_registered_model(
        spec,
        cache_dir=args.cache_dir,
        device_map="auto",
        torch_dtype="bfloat16",
        attention_backend="sdpa",
    )
    causal_plan = compile_causal_site_plan(row, tokenizer)

    mass_cubes: list[np.ndarray] = []
    row_sum_cubes: list[np.ndarray] = []
    query_rows: list[dict[str, Any]] = []
    reference_prompt_ids: np.ndarray | None = None
    reference_spans: list[Any] | None = None

    for source in range(10):
        target = source + 1
        encoding, site_id, selected_site = event_encoding(
            row, tokenizer, source=source, causal_plan=causal_plan
        )
        prompt_ids = np.asarray(
            encoding.input_ids[: encoding.prompt_token_count], dtype=np.int32
        )
        spans = sorted_record_spans(encoding)
        if reference_prompt_ids is None:
            reference_prompt_ids = prompt_ids
            reference_spans = spans
        elif not np.array_equal(reference_prompt_ids, prompt_ids):
            raise RuntimeError("Prompt token IDs changed across endpoint queries")

        attention_rows, key_starts, _logits = position_attention_outputs(
            model, adapter, encoding, int(encoding.query_position)
        )
        mass_cube, row_sums = record_mass_cube(
            attention_rows, key_starts, encoding.prompt_record_spans
        )
        mass_cubes.append(mass_cube)
        row_sum_cubes.append(row_sums)
        query_rows.append(
            {
                "from_occurrence": source,
                "to_occurrence": target,
                "site_id": site_id,
                "query_output_token_index": int(
                    encoding.query_position - encoding.prompt_token_count
                ),
                "query_full_sequence_token": int(encoding.query_position),
                "query_token_text": tokenizer.decode(
                    [int(encoding.input_ids[encoding.query_position])],
                    skip_special_tokens=False,
                ),
                "key_starts": [int(value) for value in key_starts],
                "selected_site": selected_site,
            }
        )
        print(
            json.dumps(
                {
                    "captured": f"{source}->{target}",
                    "site_id": site_id,
                    "q": query_rows[-1]["query_output_token_index"],
                    "shape": list(mass_cube.shape),
                }
            ),
            flush=True,
        )
        del attention_rows, _logits
        import torch

        torch.cuda.empty_cache()

    assert reference_prompt_ids is not None
    assert reference_spans is not None
    mass_stack = np.stack(mass_cubes)
    row_sum_stack = np.stack(row_sum_cubes)
    ranking = rank_fixed_heads(mass_stack)
    selected_layer = int(ranking[0]["layer"])
    selected_head = int(ranking[0]["head"])
    print(json.dumps({"selected_fixed_head": ranking[0]}, indent=2), flush=True)

    event_payloads: list[dict[str, Any]] = []
    for event_index, query in enumerate(query_rows):
        source = int(query["from_occurrence"])
        target = int(query["to_occurrence"])
        masses = mass_stack[event_index, selected_layer, selected_head]
        needle_total = float(masses.sum())
        target_mass = float(masses[target - 1])
        total_mass = float(
            row_sum_stack[event_index, selected_layer, selected_head]
        )
        if abs(total_mass - 1.0) > 2e-3:
            raise RuntimeError(
                f"{source}->{target} selected attention is not normalized: "
                f"{total_mass:.8f}"
            )
        event_payloads.append(
            {
                **query,
                "layer": selected_layer,
                "head": selected_head,
                "records": records_payload(
                    reference_spans, masses, target_occurrence=target
                ),
                "needle_total_mass": needle_total,
                "target_mass": target_mass,
                "target_share_of_needle_mass": (
                    target_mass / needle_total if needle_total > 0 else None
                ),
                "non_needle_context_mass": total_mass - needle_total,
                "attention_total_mass": total_mass,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    transitions_path = args.output_dir / f"seed{args.seed}_native_transitions.json"
    transitions_payload = {
        "schema_version": "aurora_native_fixed_head_transitions_v1",
        "request_id": request_id,
        "model_label": MODEL_LABEL,
        "seed": int(args.seed),
        "gold_count": int(row["gold_count"]),
        "layer": selected_layer,
        "head": selected_head,
        "head_selection": (
            "maximum geometric mean next-record raw mass, multiplied by the "
            "square root of mean next-record share, over transitions 1->2...9->10"
        ),
        "head_ranking": ranking,
        "prompt_token_count": int(reference_prompt_ids.shape[0]),
        "record_spans": [
            {
                "slot_index": int(getattr(span, "slot_index", index)),
                "city": str(span.city),
                "score": None if span.score is None else int(span.score),
                "start": int(span.start),
                "end": int(span.end),
            }
            for index, span in enumerate(reference_spans, start=1)
        ],
        "events": event_payloads,
    }
    transitions_path.write_text(
        json.dumps(transitions_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    prompt_attention_rows: list[np.ndarray] = []
    token_metadata_rows: list[dict[str, Any]] = []
    for source in TOKEN_VECTOR_SOURCES:
        target = source + 1
        encoding, _site_id, _selected_site = event_encoding(
            row, tokenizer, source=source, causal_plan=causal_plan
        )
        attention_rows, key_starts, _logits = position_attention_outputs(
            model, adapter, encoding, int(encoding.query_position)
        )
        attention = attention_rows[selected_layer][selected_head]
        vector = prompt_vector(
            attention,
            key_start=int(key_starts[selected_layer]),
            prompt_tokens=int(encoding.prompt_token_count),
        )
        expected = event_payloads[source]
        span = reference_spans[target - 1]
        observed = float(vector[int(span.start) : int(span.end)].sum())
        if abs(observed - float(expected["target_mass"])) > 5e-5:
            raise RuntimeError(
                f"{source}->{target} token-vector target mismatch: "
                f"{observed:.8f} vs {expected['target_mass']:.8f}"
            )
        prompt_attention_rows.append(vector)
        token_metadata_rows.append(expected)
        print(
            json.dumps(
                {
                    "token_vector": f"{source}->{target}",
                    "target_mass": observed,
                }
            ),
            flush=True,
        )
        del attention_rows, _logits
        import torch

        torch.cuda.empty_cache()

    array_path = args.output_dir / f"seed{args.seed}_native_token_attention.npz"
    metadata_path = args.output_dir / f"seed{args.seed}_native_token_attention.json"
    np.savez_compressed(
        array_path,
        prompt_input_ids=reference_prompt_ids,
        prompt_attention=np.stack(prompt_attention_rows).astype(np.float32),
        query_full_sequence_tokens=np.asarray(
            [row["query_full_sequence_token"] for row in token_metadata_rows],
            dtype=np.int32,
        ),
        query_output_token_indices=np.asarray(
            [row["query_output_token_index"] for row in token_metadata_rows],
            dtype=np.int32,
        ),
        from_occurrences=np.asarray(TOKEN_VECTOR_SOURCES, dtype=np.int16),
        to_occurrences=np.asarray(
            [source + 1 for source in TOKEN_VECTOR_SOURCES], dtype=np.int16
        ),
        record_spans=np.asarray(
            [[int(span.start), int(span.end)] for span in reference_spans],
            dtype=np.int32,
        ),
    )
    token_payload = {
        "schema_version": "aurora_native_token_attention_v1",
        "request_id": request_id,
        "model_label": MODEL_LABEL,
        "seed": int(args.seed),
        "gold_count": int(row["gold_count"]),
        "layer": selected_layer,
        "head": selected_head,
        "prompt_token_count": int(reference_prompt_ids.shape[0]),
        "record_spans": transitions_payload["record_spans"],
        "events": token_metadata_rows,
        "array_file": array_path.name,
        "capture_method": (
            "cached long-prefix forward plus one-token eager attention query; "
            "no full T-by-T map"
        ),
    }
    metadata_path.write_text(
        json.dumps(token_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "transitions": str(transitions_path),
                "token_array": str(array_path),
                "token_metadata": str(metadata_path),
                "selected_head": f"L{selected_layer}H{selected_head}",
                "token_shape": list(np.stack(prompt_attention_rows).shape),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
