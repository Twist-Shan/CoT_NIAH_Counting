#!/usr/bin/env python3
"""Source-specific final-query value replacement at frozen trace-item tails.

The complete native prefix is encoded normally.  At the single final answer
query, trace-tail value-cache entries are replaced by nearby non-tail values,
while their keys and therefore natural attention routing remain unchanged.
Matched controls replace the same number of nearby non-tail values.  This is
an exploratory source-necessity assay that does not select or remove heads.
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Mapping as TypingMapping, Sequence

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


SCHEMA = "realistic_niah_v5_final_trace_value_replace_v1"


def _tail_positions(
    spans: Sequence[Any], *, width: int, lower: int
) -> list[list[int]]:
    groups = []
    for span in spans:
        # ``trace_item_spans`` deliberately stores one frozen endpoint token
        # per item.  Expand backward from that endpoint for the tail sweep.
        start = max(int(lower), int(span.end) - int(width))
        groups.append(list(range(start, int(span.end))))
    return groups


def _nearby_bank(
    targets: Sequence[int],
    *,
    lower: int,
    upper: int,
    forbidden: Sequence[int],
    seed: int,
) -> list[int]:
    blocked = {int(value) for value in forbidden}
    selected: list[int] = []
    rng = random.Random(int(seed))
    for target in targets:
        candidates: list[int] = []
        for radius in (8, 16, 32, 64, 128, 256):
            candidates = [
                position
                for position in range(
                    max(int(lower), int(target) - radius),
                    min(int(upper), int(target) + radius + 1),
                )
                if position != int(target)
                and position not in blocked
                and position not in selected
            ]
            if candidates:
                break
        if not candidates:
            raise RuntimeError(f"No local control value for position {target}")
        rng.shuffle(candidates)
        candidates.sort(key=lambda position: abs(position - int(target)))
        selected.append(int(rng.choice(candidates[: min(8, len(candidates))])))
    if len(selected) != len(targets) or len(set(selected)) != len(selected):
        raise RuntimeError("Local value bank changed the requested token count")
    return selected


def _replace_tensor_values(
    values: torch.Tensor,
    *,
    targets: Sequence[int],
    donors: Sequence[int],
    query: int,
) -> int:
    if values.ndim != 4 or int(values.shape[0]) != 1:
        raise RuntimeError(f"Unsupported value-cache shape {tuple(values.shape)}")
    length = int(values.shape[2])
    absolute_start = int(query) - length
    replacements = 0
    original = values.clone()
    for target, donor in zip(targets, donors, strict=True):
        target_index = int(target) - absolute_start
        donor_index = int(donor) - absolute_start
        if 0 <= target_index < length and 0 <= donor_index < length:
            values[:, :, target_index, :] = original[:, :, donor_index, :]
            replacements += 1
    return replacements


def _replace_cache_values(
    past: Any,
    shared: Any,
    *,
    targets: Sequence[int],
    donors: Sequence[int],
    query: int,
) -> tuple[int, int]:
    past_replacements = 0
    for layer in getattr(past, "layers", []):
        values = getattr(layer, "values", None)
        if isinstance(values, torch.Tensor):
            past_replacements += _replace_tensor_values(
                values,
                targets=targets,
                donors=donors,
                query=query,
            )
    shared_replacements = 0
    if isinstance(shared, Mapping):
        for pair in shared.values():
            if isinstance(pair, (tuple, list)) and len(pair) == 2:
                values = pair[1]
                if isinstance(values, torch.Tensor):
                    shared_replacements += _replace_tensor_values(
                        values,
                        targets=targets,
                        donors=donors,
                        query=query,
                    )
    if past_replacements + shared_replacements == 0 and targets:
        raise RuntimeError("No cache value entries were replaced")
    return past_replacements, shared_replacements


@torch.inference_mode()
def _value_replaced_logits(
    model: Any,
    encoding: Any,
    cache: tuple[Any, Any, Any, Any],
    *,
    targets: Sequence[int],
    donors: Sequence[int],
) -> tuple[torch.Tensor, int, int]:
    if len(targets) != len(donors):
        raise ValueError("Target and donor value banks must have equal length")
    input_ids, attention_mask, original_past, original_shared = cache
    past = copy.deepcopy(original_past)
    shared = copy.deepcopy(original_shared)
    query = int(encoding.query_position)
    past_count, shared_count = _replace_cache_values(
        past,
        shared,
        targets=targets,
        donors=donors,
        query=query,
    )
    kwargs: dict[str, Any] = {
        "input_ids": input_ids[:, query : query + 1],
        "attention_mask": attention_mask[:, : query + 1],
        "past_key_values": past,
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
        kwargs["shared_kv_states"] = shared
    output = model(**kwargs)
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise RuntimeError("Value-replaced final-query forward returned no logits")
    return logits[0, -1].detach().float().cpu(), past_count, shared_count


def _trial(
    *,
    encoding: Any,
    tokenizer: Any,
    logits: torch.Tensor,
    condition: str,
    family: str,
    repeat: int,
    width: int,
    targets: Sequence[int],
    donors: Sequence[int],
    past_replacements: int,
    shared_replacements: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "request_id": encoding.request_id,
        "model_label": encoding.model_label,
        "seed": int(encoding.seed),
        "gold_count": int(encoding.count),
        "condition": condition,
        "condition_family": family,
        "repeat": int(repeat),
        "tail_width": int(width),
        "target_positions": [int(value) for value in targets],
        "donor_positions": [int(value) for value in donors],
        "replaced_token_count": len(targets),
        "past_layer_token_replacements": int(past_replacements),
        "shared_type_token_replacements": int(shared_replacements),
        "query_position": int(encoding.query_position),
        "intervention_scope": "selected cached values at the natural final query; keys, attention mask, prefix tokens, and all heads retained",
        **_readout(logits, encoding=encoding, tokenizer=tokenizer),
    }


def summarize(rows: Sequence[TypingMapping[str, Any]]) -> dict[str, Any]:
    clean = [row for row in rows if str(row["condition"]) == "clean"]
    grouped: dict[tuple[str, str, int], list[TypingMapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["condition"]) != "clean":
            grouped[(str(row["condition_family"]), str(row["condition"]), int(row["tail_width"]))].append(row)
    output = []
    for (family, condition, width), frame in sorted(grouped.items()):
        output.append(
            {
                "condition_family": family,
                "condition": condition,
                "tail_width": width,
                "trials": len(frame),
                "unique_requests": len({str(row["request_id"]) for row in frame}),
                "mean_replaced_token_count": float(np.mean([row["replaced_token_count"] for row in frame])),
                "top_vocab_exact_rate": float(np.mean([row["top_vocab_exact"] for row in frame])),
                "candidate_exact_rate": float(np.mean([row["candidate_exact"] for row in frame])),
                "mean_gold_candidate_probability": float(np.mean([row["gold_candidate_probability"] for row in frame])),
                "mean_gold_margin": float(np.mean([row["gold_vs_best_other_candidate_margin"] for row in frame])),
            }
        )
    return {
        "schema_version": SCHEMA,
        "clean_requests": len(clean),
        "clean_top_vocab_exact_rate": float(np.mean([row["top_vocab_exact"] for row in clean])),
        "clean_mean_gold_margin": float(np.mean([row["gold_vs_best_other_candidate_margin"] for row in clean])),
        "condition_summary": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1238, 1241, 1243])
    parser.add_argument("--counts", type=int, nargs="+", default=[3, 6, 8])
    parser.add_argument("--tail-widths", type=int, nargs="+", default=[1, 4, 8])
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
        cache = _prefix_cache(model, adapter, encoding)
        clean_logits, past_count, shared_count = _value_replaced_logits(
            model, encoding, cache, targets=[], donors=[]
        )
        clean = _trial(
            encoding=encoding,
            tokenizer=tokenizer,
            logits=clean_logits,
            condition="clean",
            family="clean",
            repeat=0,
            width=0,
            targets=[],
            donors=[],
            past_replacements=past_count,
            shared_replacements=shared_count,
        )
        clean["trace_item_anchor_audit"] = anchor_audit
        output.append(clean)

        for width in sorted({int(value) for value in args.tail_widths}):
            groups = _tail_positions(
                encoding.trace_item_spans,
                width=width,
                lower=int(encoding.prompt_token_count),
            )
            all_tail = [position for group in groups for position in group]
            conditions = {"all_tails": all_tail}
            if width == 4:
                conditions["last_tail"] = list(groups[-1])
                conditions["all_except_last_tails"] = [
                    position for group in groups[:-1] for position in group
                ]
            for condition, targets in conditions.items():
                if not targets:
                    continue
                for repeat in range(int(args.control_repeats)):
                    base_seed = (
                        int(encoding.seed) * 1_000_003
                        + int(encoding.count) * 10_007
                        + int(width) * 101
                        + repeat
                    )
                    donors = _nearby_bank(
                        targets,
                        lower=int(encoding.prompt_token_count),
                        upper=int(encoding.query_position),
                        forbidden=all_tail,
                        seed=base_seed,
                    )
                    logits, past_count, shared_count = _value_replaced_logits(
                        model,
                        encoding,
                        cache,
                        targets=targets,
                        donors=donors,
                    )
                    output.append(
                        _trial(
                            encoding=encoding,
                            tokenizer=tokenizer,
                            logits=logits,
                            condition=condition,
                            family="trace_tail_value_replace",
                            repeat=repeat,
                            width=width,
                            targets=targets,
                            donors=donors,
                            past_replacements=past_count,
                            shared_replacements=shared_count,
                        )
                    )

                    control_donors = _nearby_bank(
                        donors,
                        lower=int(encoding.prompt_token_count),
                        upper=int(encoding.query_position),
                        forbidden=[*all_tail, *donors],
                        seed=base_seed + 97_531,
                    )
                    control_logits, past_count, shared_count = _value_replaced_logits(
                        model,
                        encoding,
                        cache,
                        targets=donors,
                        donors=control_donors,
                    )
                    output.append(
                        _trial(
                            encoding=encoding,
                            tokenizer=tokenizer,
                            logits=control_logits,
                            condition=condition,
                            family="nearby_value_control",
                            repeat=repeat,
                            width=width,
                            targets=donors,
                            donors=control_donors,
                            past_replacements=past_count,
                            shared_replacements=shared_count,
                        )
                    )
        print(
            f"[final-trace-value-replace] {index}/{len(selected)} "
            f"{encoding.request_id} clean={clean['top_vocab_exact']}",
            flush=True,
        )

    write_jsonl(args.output, output)
    payload = summarize(output)
    payload.update(
        {
            "model_label": args.model,
            "generations": str(args.generations),
            "output": str(args.output),
            "seeds": [int(value) for value in args.seeds],
            "counts": [int(value) for value in args.counts],
            "tail_widths": [int(value) for value in args.tail_widths],
            "control_repeats": int(args.control_repeats),
            "site_id": args.site_id,
            "gate_audit": gate_audit,
            "interpretation_limit": "Exploratory source-necessity assay; explicit trace-tail tokens may carry lexical/index evidence and are not proof of a recurrent counter.",
        }
    )
    write_json(summary_path, payload)
    print(f"[final-trace-value-replace] wrote {args.output} and {summary_path}")


if __name__ == "__main__":
    main()
