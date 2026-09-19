#!/usr/bin/env python3
"""Cross-seed final-query ablation of naturally broad-retrieval heads.

Heads are ranked on all *other* seeds by natural final-query broad score, then
tested on the held-out seed.  The intervention zeros selected head outputs
only for the single final answer-query token.  The complete native trace and
its natural KV cache remain intact.  Exact layer-matched random head banks are
the primary controls; prompt-broad heads are an independently ranked route.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.modeling import (  # noqa: E402
    _accepts_keyword,
    _bounded_logits_kwargs,
    _encoding_tensors,
    _extract_shared_kv_states,
    load_registered_model,
)
from realistic_niah_v4.spec import resolve_model_spec  # noqa: E402

# Reuse the audited frozen-item encoding and cohort gate from the descriptive
# assay rather than independently reimplementing parser/alignment policy.
from run_realistic_niah_v5_final_broad_retrieval import (  # noqa: E402
    eligible_rows,
    frozen_encoding,
    read_jsonl,
    write_json,
    write_jsonl,
)


SCHEMA = "realistic_niah_v5_final_broad_head_ablation_v1"


def _rank_heads(
    attention: Sequence[Mapping[str, Any]],
    *,
    held_out_seed: int,
    metric: str,
) -> list[tuple[int, int]]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in attention:
        if int(row["seed"]) == int(held_out_seed):
            continue
        grouped[(int(row["layer"]), int(row["head"]))].append(
            float(row[metric])
        )
    if not grouped:
        raise ValueError(f"No cross-seed attention rows for held-out {held_out_seed}")
    return sorted(
        grouped,
        key=lambda head: (
            -float(np.mean(grouped[head])),
            head[0],
            head[1],
        ),
    )


def _layer_matched_control(
    selected: Sequence[tuple[int, int]],
    *,
    adapter: Any,
    seed: int,
) -> list[tuple[int, int]]:
    by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, head in selected:
        by_layer[int(layer)].append(int(head))
    rng = random.Random(int(seed))
    controls: list[tuple[int, int]] = []
    selected_set = {(int(layer), int(head)) for layer, head in selected}
    for layer in sorted(by_layer):
        required = len(by_layer[layer])
        candidates = [
            (layer, head)
            for head in range(int(adapter.num_heads[layer]))
            if (layer, head) not in selected_set
        ]
        if len(candidates) < required:
            raise RuntimeError(
                f"Cannot construct disjoint layer-matched controls at L{layer}: "
                f"required={required}, available={len(candidates)}"
            )
        rng.shuffle(candidates)
        controls.extend(candidates[:required])
    if len(controls) != len(selected):
        raise RuntimeError("Layer-matched control bank changed bank size")
    return controls


def _control_constructable_bank(
    ranking: Sequence[tuple[int, int]],
    *,
    bank_size: int,
    adapter: Any,
) -> list[tuple[int, int]]:
    """Preserve discovery rank while reserving a disjoint matched bank."""

    selected: list[tuple[int, int]] = []
    occupancy: dict[int, int] = defaultdict(int)
    for raw_layer, raw_head in ranking:
        layer = int(raw_layer)
        head = int(raw_head)
        maximum = int(adapter.num_heads[layer]) // 2
        if maximum < 1 or occupancy[layer] >= maximum:
            continue
        selected.append((layer, head))
        occupancy[layer] += 1
        if len(selected) == int(bank_size):
            return selected
    raise RuntimeError(
        f"Could not construct a rank-preserving matched-control bank of {bank_size}"
    )


@torch.inference_mode()
def _prefix_cache(model: Any, adapter: Any, encoding: Any) -> tuple[Any, Any, Any, Any]:
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    query = int(encoding.query_position)
    uses_shared_kv = any(
        bool(getattr(attention, "is_kv_shared_layer", False))
        for attention in adapter.attentions
    )
    kwargs: dict[str, Any] = {
        "input_ids": input_ids[:, :query],
        "attention_mask": attention_mask[:, :query],
        "use_cache": True,
        "output_attentions": False,
        **_bounded_logits_kwargs(model),
    }
    if uses_shared_kv:
        kwargs["return_shared_kv_states"] = True
    output = model(**kwargs)
    past = getattr(output, "past_key_values", None)
    if past is None:
        raise RuntimeError("Natural prefix forward returned no KV cache")
    shared = _extract_shared_kv_states(output)
    if uses_shared_kv and shared is None:
        raise RuntimeError("Shared-KV model returned no shared prefix states")
    return input_ids, attention_mask, past, shared


@torch.inference_mode()
def _query_logits(
    model: Any,
    adapter: Any,
    encoding: Any,
    cache: tuple[Any, Any, Any, Any],
    *,
    heads: Sequence[tuple[int, int]],
) -> torch.Tensor:
    input_ids, attention_mask, past, shared = cache
    by_layer: dict[int, list[int]] = defaultdict(list)
    for raw_layer, raw_head in heads:
        layer = int(raw_layer)
        head = int(raw_head)
        if not 0 <= layer < int(adapter.num_layers):
            raise ValueError(f"Invalid layer {layer}")
        if not 0 <= head < int(adapter.num_heads[layer]):
            raise ValueError(f"Invalid head L{layer}H{head}")
        by_layer[layer].append(head)
    applications: dict[int, int] = {layer: 0 for layer in by_layer}
    handles = []
    for layer, layer_heads in by_layer.items():
        head_dim = int(adapter.head_dims[layer])

        def hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
            layer_heads: tuple[int, ...] = tuple(sorted(set(layer_heads))),
            head_dim: int = head_dim,
        ) -> tuple[Any, ...]:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Attention o_proj received no head tensor")
            value = args[0]
            if value.ndim != 3 or int(value.shape[1]) != 1:
                raise RuntimeError(
                    f"Final-query hook expected [batch,1,width], got {tuple(value.shape)}"
                )
            patched = value.clone()
            for head in layer_heads:
                left = int(head) * head_dim
                patched[:, 0, left : left + head_dim] = 0
            applications[layer] += 1
            return (patched, *args[1:])

        handles.append(
            adapter.output_projections[layer].register_forward_pre_hook(hook)
        )
    query = int(encoding.query_position)
    kwargs: dict[str, Any] = {
        "input_ids": input_ids[:, query : query + 1],
        "attention_mask": attention_mask[:, : query + 1],
        "past_key_values": copy.deepcopy(past),
        "use_cache": False,
        "output_attentions": False,
        **_bounded_logits_kwargs(model),
    }
    if _accepts_keyword(model, "position_ids"):
        kwargs["position_ids"] = torch.tensor(
            [[query]], dtype=torch.long, device=input_ids.device
        )
    if _accepts_keyword(model, "cache_position"):
        kwargs["cache_position"] = torch.tensor(
            [query], dtype=torch.long, device=input_ids.device
        )
    if shared is not None and _accepts_keyword(model, "shared_kv_states"):
        kwargs["shared_kv_states"] = copy.deepcopy(shared)
    try:
        output = model(**kwargs)
    finally:
        for handle in handles:
            handle.remove()
    bad = {layer: count for layer, count in applications.items() if count != 1}
    if bad:
        raise RuntimeError(f"Final-query head hooks did not apply exactly once: {bad}")
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise RuntimeError("Final-query forward returned no logits")
    return logits[0, -1].detach().float().cpu()


def _readout(
    logits: torch.Tensor,
    *,
    encoding: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    candidates: dict[int, int] = {}
    for count, ids in encoding.count_candidate_answer_token_ids:
        if len(ids) != 1:
            raise ValueError(f"Count {count} is not a one-token answer: {ids}")
        candidates[int(count)] = int(ids[0])
    counts = sorted(candidates)
    candidate_logits = torch.tensor(
        [float(logits[candidates[count]]) for count in counts]
    )
    probabilities = torch.softmax(candidate_logits, dim=0)
    candidate_prediction = int(counts[int(torch.argmax(candidate_logits))])
    gold_index = counts.index(int(encoding.count))
    other_logits = [
        float(candidate_logits[index])
        for index, count in enumerate(counts)
        if count != int(encoding.count)
    ]
    top_id = int(torch.argmax(logits))
    reverse = {token_id: count for count, token_id in candidates.items()}
    return {
        "candidate_prediction": candidate_prediction,
        "candidate_exact": candidate_prediction == int(encoding.count),
        "gold_candidate_probability": float(probabilities[gold_index]),
        "gold_vs_best_other_candidate_margin": float(
            candidate_logits[gold_index] - max(other_logits)
        ),
        "top_vocab_token_id": top_id,
        "top_vocab_token_text": tokenizer.decode([top_id], skip_special_tokens=False),
        "top_vocab_number": reverse.get(top_id),
        "top_vocab_exact": reverse.get(top_id) == int(encoding.count),
    }


def _trial_row(
    *,
    encoding: Any,
    tokenizer: Any,
    logits: torch.Tensor,
    condition: str,
    mechanism: str,
    bank_size: int,
    repeat: int,
    heads: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "request_id": encoding.request_id,
        "model_label": encoding.model_label,
        "seed": int(encoding.seed),
        "gold_count": int(encoding.count),
        "condition": condition,
        "mechanism": mechanism,
        "bank_size": int(bank_size),
        "repeat": int(repeat),
        "heads": [[int(layer), int(head)] for layer, head in heads],
        "intervention_scope": "one final answer-query token only",
        **_readout(logits, encoding=encoding, tokenizer=tokenizer),
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    clean_by_request = {
        str(row["request_id"]): row
        for row in rows
        if str(row["condition"]) == "clean"
    }
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["condition"]) == "clean":
            continue
        grouped[(str(row["mechanism"]), int(row["bank_size"]), str(row["condition"]))].append(row)
    condition_rows: list[dict[str, Any]] = []
    for (mechanism, bank_size, condition), frame in sorted(grouped.items()):
        clean_exact = [
            bool(clean_by_request[str(row["request_id"])]["top_vocab_exact"])
            for row in frame
        ]
        condition_rows.append(
            {
                "mechanism": mechanism,
                "bank_size": bank_size,
                "condition": condition,
                "trials": len(frame),
                "unique_requests": len({str(row["request_id"]) for row in frame}),
                "top_vocab_exact_rate": float(np.mean([row["top_vocab_exact"] for row in frame])),
                "candidate_exact_rate": float(np.mean([row["candidate_exact"] for row in frame])),
                "mean_gold_candidate_probability": float(np.mean([row["gold_candidate_probability"] for row in frame])),
                "mean_gold_margin": float(np.mean([row["gold_vs_best_other_candidate_margin"] for row in frame])),
                "mean_top_vocab_accuracy_damage_from_clean": float(
                    np.mean(clean_exact) - np.mean([row["top_vocab_exact"] for row in frame])
                ),
            }
        )
    return {
        "schema_version": SCHEMA,
        "clean_requests": len(clean_by_request),
        "clean_top_vocab_exact_rate": float(
            np.mean([row["top_vocab_exact"] for row in clean_by_request.values()])
        ),
        "condition_summary": condition_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--attention", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1238, 1241, 1243])
    parser.add_argument("--counts", type=int, nargs="+", default=[3, 6, 8])
    parser.add_argument("--bank-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--control-repeats", type=int, default=5)
    parser.add_argument("--site-id", default="answer_query_v3")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary_path = args.output.with_suffix(".summary.json")
    for path in (args.output, summary_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {path}")
    selected, gate_audit = eligible_rows(
        read_jsonl(args.generations),
        model_label=args.model,
        seeds=args.seeds,
        counts=args.counts,
    )
    attention = [
        row
        for row in read_jsonl(args.attention)
        if str(row.get("model_label")) == args.model
        and int(row.get("seed", -1)) in {int(value) for value in args.seeds}
    ]
    if not attention:
        raise ValueError("No matching natural-attention rows")

    spec = resolve_model_spec(args.model)
    model, tokenizer, adapter = load_registered_model(
        spec,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )
    output: list[dict[str, Any]] = []
    ranking_audit: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        encoding, anchor_audit = frozen_encoding(row, tokenizer, site_id=args.site_id)
        cache = _prefix_cache(model, adapter, encoding)
        clean_logits = _query_logits(model, adapter, encoding, cache, heads=[])
        clean = _trial_row(
            encoding=encoding,
            tokenizer=tokenizer,
            logits=clean_logits,
            condition="clean",
            mechanism="clean",
            bank_size=0,
            repeat=0,
            heads=[],
        )
        clean["trace_item_anchor_audit"] = anchor_audit
        output.append(clean)
        for mechanism, metric in (
            ("trace_broad", "trace_broad_score"),
            ("prompt_broad", "prompt_broad_score"),
        ):
            ranking = _rank_heads(
                attention,
                held_out_seed=int(encoding.seed),
                metric=metric,
            )
            for bank_size in [int(value) for value in args.bank_sizes]:
                heads = _control_constructable_bank(
                    ranking,
                    bank_size=bank_size,
                    adapter=adapter,
                )
                logits = _query_logits(model, adapter, encoding, cache, heads=heads)
                output.append(
                    _trial_row(
                        encoding=encoding,
                        tokenizer=tokenizer,
                        logits=logits,
                        condition="ranked",
                        mechanism=mechanism,
                        bank_size=bank_size,
                        repeat=0,
                        heads=heads,
                    )
                )
                ranking_audit.append(
                    {
                        "held_out_seed": int(encoding.seed),
                        "mechanism": mechanism,
                        "bank_size": bank_size,
                        "heads": [[layer, head] for layer, head in heads],
                        "selection_seeds": sorted(
                            {int(value) for value in args.seeds} - {int(encoding.seed)}
                        ),
                    }
                )
                stable_mechanism = sum((offset + 1) * ord(char) for offset, char in enumerate(mechanism))
                for repeat in range(int(args.control_repeats)):
                    control_seed = (
                        10_000_000 * int(encoding.seed)
                        + 10_000 * stable_mechanism
                        + 100 * bank_size
                        + repeat
                    )
                    control = _layer_matched_control(
                        heads, adapter=adapter, seed=control_seed
                    )
                    control_logits = _query_logits(
                        model, adapter, encoding, cache, heads=control
                    )
                    output.append(
                        _trial_row(
                            encoding=encoding,
                            tokenizer=tokenizer,
                            logits=control_logits,
                            condition="layer_matched_control",
                            mechanism=mechanism,
                            bank_size=bank_size,
                            repeat=repeat,
                            heads=control,
                        )
                    )
        clean_check = _readout(
            _query_logits(model, adapter, encoding, cache, heads=[]),
            encoding=encoding,
            tokenizer=tokenizer,
        )
        if clean_check != {key: clean[key] for key in clean_check}:
            raise RuntimeError("Cached clean query changed after interventions")
        print(
            f"[final-broad-head-ablation] {index}/{len(selected)} "
            f"{encoding.request_id} clean={clean['top_vocab_exact']}",
            flush=True,
        )
    write_jsonl(args.output, output)
    summary = {
        **summarize(output),
        "model_label": args.model,
        "generations": str(args.generations.resolve()),
        "attention": str(args.attention.resolve()),
        "output": str(args.output.resolve()),
        "site_id": args.site_id,
        "seeds": [int(value) for value in args.seeds],
        "counts": [int(value) for value in args.counts],
        "bank_sizes": [int(value) for value in args.bank_sizes],
        "control_repeats": int(args.control_repeats),
        "gate_audit": gate_audit,
        "ranking_audit": ranking_audit,
        "ranking_policy": "leave-one-seed-out mean natural broad score",
        "ranked_bank_constraint": "preserve rank subject to per-layer occupancy <= floor(layer_head_count/2), guaranteeing a disjoint exact layer-matched control bank",
        "control_policy": "disjoint exact layer-occupancy-matched heads, deterministic repeats",
        "interpretation_limit": "A ranked-head effect supports causal use of naturally trace-broad heads at final readout, but whole-head output ablation is not a source-edge-specific intervention and does not establish recurrence.",
    }
    write_json(summary_path, summary)
    print(f"[final-broad-head-ablation] wrote {args.output} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
