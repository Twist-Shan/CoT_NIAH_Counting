#!/usr/bin/env python3
"""Source-specific final-query masking of frozen trace endpoint keys.

The complete native trace is first encoded normally and cached without any
intervention.  Only when the single final answer-query token is evaluated are
selected key positions hidden from attention in every layer/head.  This tests
trace endpoint necessity without ablating whole heads or changing suffix
formation.  Last-only and all-except-last arms distinguish terminal lookup
from genuinely distributed trace retrieval.  Nearby non-endpoint keys provide
position- and token-count-matched controls.
"""

from __future__ import annotations

import argparse
import copy
import json
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
    load_registered_model,
)
from realistic_niah_v4.spec import resolve_model_spec  # noqa: E402
from run_realistic_niah_v5_final_broad_head_ablation import (  # noqa: E402
    _prefix_cache,
    _readout,
)
from run_realistic_niah_v5_final_broad_retrieval import (  # noqa: E402
    eligible_rows,
    frozen_encoding,
    read_jsonl,
    write_json,
    write_jsonl,
)


SCHEMA = "realistic_niah_v5_final_trace_edge_mask_v1"


def _nearby_control_positions(
    endpoints: Sequence[int],
    *,
    lower: int,
    upper: int,
    seed: int,
) -> list[int]:
    """Select unique nearby non-endpoint keys without consulting outcomes."""

    forbidden = {int(value) for value in endpoints}
    selected: list[int] = []
    rng = random.Random(int(seed))
    for endpoint in endpoints:
        candidates: list[int] = []
        for radius in (4, 8, 16, 32, 64, 128):
            candidates = [
                position
                for position in range(
                    max(int(lower), int(endpoint) - radius),
                    min(int(upper), int(endpoint) + radius + 1),
                )
                if position not in forbidden
                and position not in selected
                and position != int(endpoint)
            ]
            if candidates:
                break
        if not candidates:
            raise RuntimeError(
                f"No nearby matched key for endpoint {endpoint} in [{lower},{upper})"
            )
        # Prefer the same absolute distance profile; randomize only ties and
        # near-ties so repeated banks audit local-choice sensitivity.
        rng.shuffle(candidates)
        candidates.sort(key=lambda position: abs(position - int(endpoint)))
        pool = candidates[: min(4, len(candidates))]
        selected.append(int(rng.choice(pool)))
    if len(selected) != len(endpoints) or len(set(selected)) != len(selected):
        raise RuntimeError("Nearby control construction changed key count")
    return selected


@torch.inference_mode()
def _masked_query_logits(
    model: Any,
    encoding: Any,
    cache: tuple[Any, Any, Any, Any],
    *,
    masked_positions: Sequence[int],
) -> torch.Tensor:
    input_ids, attention_mask, past, shared = cache
    query = int(encoding.query_position)
    active_mask = attention_mask[:, : query + 1].clone()
    positions = sorted({int(value) for value in masked_positions})
    if any(position < 0 or position >= query for position in positions):
        raise ValueError(f"A masked key is outside the prior prefix: {positions}")
    if positions:
        active_mask[:, positions] = 0
    kwargs: dict[str, Any] = {
        "input_ids": input_ids[:, query : query + 1],
        "attention_mask": active_mask,
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
    output = model(**kwargs)
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise RuntimeError("Masked final-query forward returned no logits")
    return logits[0, -1].detach().float().cpu()


def _trial(
    *,
    encoding: Any,
    tokenizer: Any,
    logits: torch.Tensor,
    condition: str,
    family: str,
    repeat: int,
    masked_positions: Sequence[int],
    endpoint_positions: Sequence[int],
) -> dict[str, Any]:
    endpoint_set = {int(value) for value in endpoint_positions}
    positions = [int(value) for value in masked_positions]
    return {
        "schema_version": SCHEMA,
        "request_id": encoding.request_id,
        "model_label": encoding.model_label,
        "seed": int(encoding.seed),
        "gold_count": int(encoding.count),
        "condition": condition,
        "condition_family": family,
        "repeat": int(repeat),
        "masked_positions": positions,
        "masked_key_count": len(positions),
        "masked_endpoint_key_count": sum(
            position in endpoint_set for position in positions
        ),
        "query_position": int(encoding.query_position),
        "intervention_scope": "selected final-query attention keys across all layers and heads; natural prefix cache unchanged",
        **_readout(logits, encoding=encoding, tokenizer=tokenizer),
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    clean = {
        str(row["request_id"]): row
        for row in rows
        if str(row["condition"]) == "clean"
    }
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["condition"]) != "clean":
            grouped[(str(row["condition_family"]), str(row["condition"]))].append(row)
    output: list[dict[str, Any]] = []
    for (family, condition), frame in sorted(grouped.items()):
        output.append(
            {
                "condition_family": family,
                "condition": condition,
                "trials": len(frame),
                "unique_requests": len({str(row["request_id"]) for row in frame}),
                "mean_masked_key_count": float(np.mean([row["masked_key_count"] for row in frame])),
                "top_vocab_exact_rate": float(np.mean([row["top_vocab_exact"] for row in frame])),
                "candidate_exact_rate": float(np.mean([row["candidate_exact"] for row in frame])),
                "mean_gold_candidate_probability": float(np.mean([row["gold_candidate_probability"] for row in frame])),
                "mean_gold_margin": float(np.mean([row["gold_vs_best_other_candidate_margin"] for row in frame])),
            }
        )
    return {
        "schema_version": SCHEMA,
        "clean_requests": len(clean),
        "clean_top_vocab_exact_rate": float(np.mean([row["top_vocab_exact"] for row in clean.values()])),
        "clean_mean_gold_margin": float(np.mean([row["gold_vs_best_other_candidate_margin"] for row in clean.values()])),
        "condition_summary": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1238, 1241, 1243])
    parser.add_argument("--counts", type=int, nargs="+", default=[3, 6, 8])
    parser.add_argument("--control-repeats", type=int, default=10)
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
    spec = resolve_model_spec(args.model)
    model, tokenizer, adapter = load_registered_model(
        spec,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )

    output: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        encoding, anchor_audit = frozen_encoding(row, tokenizer, site_id=args.site_id)
        endpoints = [int(span.end) - 1 for span in encoding.trace_item_spans]
        if len(endpoints) != int(encoding.count) or len(set(endpoints)) != len(endpoints):
            raise RuntimeError("Trace endpoint keys are not one-to-one")
        cache = _prefix_cache(model, adapter, encoding)

        clean_logits = _masked_query_logits(
            model, encoding, cache, masked_positions=[]
        )
        clean = _trial(
            encoding=encoding,
            tokenizer=tokenizer,
            logits=clean_logits,
            condition="clean",
            family="clean",
            repeat=0,
            masked_positions=[],
            endpoint_positions=endpoints,
        )
        clean["trace_item_anchor_audit"] = anchor_audit
        output.append(clean)

        endpoint_conditions = {
            "last_endpoint": endpoints[-1:],
            "all_except_last_endpoints": endpoints[:-1],
            "all_endpoints": endpoints,
        }
        for condition, positions in endpoint_conditions.items():
            logits = _masked_query_logits(
                model, encoding, cache, masked_positions=positions
            )
            output.append(
                _trial(
                    encoding=encoding,
                    tokenizer=tokenizer,
                    logits=logits,
                    condition=condition,
                    family="trace_endpoint",
                    repeat=0,
                    masked_positions=positions,
                    endpoint_positions=endpoints,
                )
            )

        lower = int(encoding.prompt_token_count)
        upper = int(encoding.query_position)
        for repeat in range(int(args.control_repeats)):
            control_all = _nearby_control_positions(
                endpoints,
                lower=lower,
                upper=upper,
                seed=10_000_000 * int(encoding.seed) + 1000 * int(encoding.count) + repeat,
            )
            control_without_last = control_all[:-1]
            for condition, positions in (
                ("nearby_control_all", control_all),
                ("nearby_control_all_except_last", control_without_last),
            ):
                logits = _masked_query_logits(
                    model, encoding, cache, masked_positions=positions
                )
                output.append(
                    _trial(
                        encoding=encoding,
                        tokenizer=tokenizer,
                        logits=logits,
                        condition=condition,
                        family="nearby_nonendpoint_control",
                        repeat=repeat,
                        masked_positions=positions,
                        endpoint_positions=endpoints,
                    )
                )

        clean_check = _readout(
            _masked_query_logits(model, encoding, cache, masked_positions=[]),
            encoding=encoding,
            tokenizer=tokenizer,
        )
        if clean_check != {key: clean[key] for key in clean_check}:
            raise RuntimeError("Cached clean query changed after key masking")
        print(
            f"[final-trace-edge-mask] {index}/{len(selected)} "
            f"{encoding.request_id} clean={clean['top_vocab_exact']}",
            flush=True,
        )

    write_jsonl(args.output, output)
    summary = {
        **summarize(output),
        "model_label": args.model,
        "generations": str(args.generations.resolve()),
        "output": str(args.output.resolve()),
        "site_id": args.site_id,
        "seeds": [int(value) for value in args.seeds],
        "counts": [int(value) for value in args.counts],
        "control_repeats": int(args.control_repeats),
        "gate_audit": gate_audit,
        "endpoint_definition": "one frozen literal item-end baseline token per parser-registered trace item",
        "interpretation_limit": "All-layer final-query key masking is source-specific but not head- or layer-localized. These explicit trace endpoints may themselves contain count/index evidence; the assay tests answer-time trace retrieval, not a hidden recurrent counter.",
    }
    write_json(summary_path, summary)
    print(f"[final-trace-edge-mask] wrote {args.output} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
