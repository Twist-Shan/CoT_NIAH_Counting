#!/usr/bin/env python3
"""Extract exact P0 head maps and per-needle attention examples.

The grammar-specific head maps are read from the frozen causal-plan rankings.
The example matrices are recomputed at the same exact ``p0_item_end`` query
token used for ranking and ablation, so every cell is a raw attention mass to
one registered prompt-record span.  This script is intended to run on the
model host and writes a compact JSON artifact for the local HTML builder.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.modeling import (  # noqa: E402
    load_registered_model,
    position_attention_outputs,
)
from realistic_niah_v4.spec import resolve_model_spec  # noqa: E402
from realistic_niah_v5.causal import mechanism_continuations  # noqa: E402
from realistic_niah_v5.encoding import build_native_causal_encoding  # noqa: E402
from realistic_niah_v5.pipeline import (  # noqa: E402
    read_jsonl,
    registered_records,
)
from realistic_niah_v5.spec import V5Config  # noqa: E402


def _float_or_none(value: Any) -> float | None:
    text = str(value).strip()
    return float(text) if text else None


def _parse_head(value: str) -> tuple[str, int, int]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Head examples must be GRAMMAR:LAYER:HEAD"
        )
    grammar, layer, head = parts
    return grammar, int(layer), int(head)


def _parse_bank(value: str) -> tuple[str, int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "Bank examples must be GRAMMAR:K"
        )
    grammar, size = parts
    bank_size = int(size)
    if bank_size < 1:
        raise argparse.ArgumentTypeError("Bank size must be positive")
    return grammar, bank_size


def _grammar_from_pair(value: Any) -> str:
    return str(value).rsplit(" -> ", 1)[-1]


def _ranking_bundle(plan_root: Path, plan_k: int) -> dict[str, Any]:
    suffix = f"_p0_local_seed_event_k{int(plan_k)}_fullpanel_v1"
    rankings: dict[str, Any] = {}
    for plan_dir in sorted(plan_root.glob(f"causal_plan_*{suffix}")):
        grammar = plan_dir.name.removeprefix("causal_plan_").removesuffix(suffix)
        path = plan_dir / "crossfit_source_specific_head_ranking.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rankings[grammar] = {
            "plan_k": int(plan_k),
            "n_seeds": max((int(float(row["n_seeds"])) for row in rows), default=0),
            "rows": [
                {
                    "layer": int(row["layer"]),
                    "head": int(row["head"]),
                    "score": float(row["discovery_selection_value"]),
                    "rank": int(float(row["discovery_rank"])),
                    "n_seeds": int(float(row["n_seeds"])),
                    "mean_anchor_roles_per_seed": float(
                        row["mean_anchor_roles_per_seed"]
                    ),
                    "relative_attention_mass": _float_or_none(
                        row.get(
                            "discovery_target_source_relative_attention_mass",
                            "nan",
                        )
                    ),
                    "target_top1_rate": _float_or_none(
                        row.get("discovery_target_source_attention_top1", "nan")
                    ),
                }
                for row in rows
            ],
        }
    if not rankings:
        raise FileNotFoundError(
            f"No grammar-specific P0 K{plan_k} rankings under {plan_root}"
        )
    return rankings


def _eligible_tasks(
    rows: list[dict[str, Any]], tokenizer: Any
) -> dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    by_grammar: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        specifications, _excluded = mechanism_continuations(
            row,
            tokenizer,
            mechanism="retrieval_anchor_localization",
        )
        for specification in specifications:
            if "p0_item_end" not in {
                str(value) for value in specification.get("anchor_roles", [])
            }:
                continue
            if not bool(specification.get("event_specific")):
                continue
            if not bool(specification.get("local_anchor_eligible")):
                continue
            by_grammar[_grammar_from_pair(specification.get("grammar_pair"))].append(
                (row, specification)
            )
    return by_grammar


def _representative_request(
    tasks: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Choose deterministically: most events, then largest N, then lowest seed."""

    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row, specification in tasks:
        request_id = str(row.get("request_id", row.get("stimulus_id")))
        grouped[request_id].append((row, specification))
    if not grouped:
        raise ValueError("No eligible P0 tasks for requested grammar")

    def key(item: tuple[str, list[tuple[dict[str, Any], dict[str, Any]]]]) -> tuple[Any, ...]:
        request_id, values = item
        row = values[0][0]
        count = int(row.get("gold_count", len(row.get("gold_records", []))))
        seed = int(row["seed"])
        return (-len(values), -count, seed, request_id)

    _request_id, selected = min(grouped.items(), key=key)
    return sorted(
        selected,
        key=lambda value: (
            int(value[1]["from_occurrence"]),
            int(value[1]["query_output_token_index"]),
        ),
    )


def _record_masses(
    attention: Any,
    *,
    key_start: int,
    prompt_record_spans: Any,
) -> tuple[list[dict[str, Any]], float]:
    key_end = int(key_start) + int(attention.shape[-1])
    records: list[dict[str, Any]] = []
    needle_total = 0.0
    for source_index, span in enumerate(
        sorted(prompt_record_spans, key=lambda value: (int(value.start), int(value.end))),
        start=1,
    ):
        overlap_start = max(int(span.start), int(key_start))
        overlap_end = min(int(span.end), key_end)
        mass = (
            float(
                attention[
                    overlap_start - int(key_start) : overlap_end - int(key_start)
                ].sum().item()
            )
            if overlap_end > overlap_start
            else 0.0
        )
        needle_total += mass
        records.append(
            {
                "source_index": int(source_index),
                "city": str(span.city),
                "token_start": int(span.start),
                "token_end": int(span.end),
                "visible_token_count": max(0, overlap_end - overlap_start),
                "mass": mass,
            }
        )
    total_mass = float(attention.sum().item())
    return records, max(0.0, total_mass - needle_total)


def _capture_examples(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    *,
    tasks_by_grammar: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]],
    examples: list[tuple[str, int, int]],
    max_queries: int,
) -> list[dict[str, Any]]:
    requested: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for grammar, layer, head in examples:
        if (layer, head) not in requested[grammar]:
            requested[grammar].append((layer, head))

    output: list[dict[str, Any]] = []
    for grammar, heads in requested.items():
        selected = _representative_request(tasks_by_grammar.get(grammar, []))
        if len(selected) > max_queries:
            indices = [
                round(index * (len(selected) - 1) / (max_queries - 1))
                for index in range(max_queries)
            ] if max_queries > 1 else [0]
            selected = [selected[index] for index in dict.fromkeys(indices)]
        row = selected[0][0]
        panels = {
            (layer, head): {
                "grammar": grammar,
                "layer": int(layer),
                "head": int(head),
                "request_id": str(row.get("request_id", row.get("stimulus_id"))),
                "seed": int(row["seed"]),
                "gold_count": int(
                    row.get("gold_count", len(row.get("gold_records", [])))
                ),
                "query_site": "p0_item_end",
                "events": [],
            }
            for layer, head in heads
        }
        for task_index, (_row, specification) in enumerate(selected, start=1):
            encoding = build_native_causal_encoding(
                _row,
                tokenizer,
                query_output_token_index=int(specification["query_output_token_index"]),
                sequence_output_token_end=int(specification["target_output_token_end"]),
                selected_site=specification,
            )
            rows, key_starts, _logits = position_attention_outputs(
                model,
                adapter,
                encoding,
                int(encoding.query_position),
            )
            for layer, head in heads:
                attention = rows[layer][head]
                record_rows, context_mass = _record_masses(
                    attention,
                    key_start=int(key_starts[layer]),
                    prompt_record_spans=encoding.prompt_record_spans,
                )
                target_city = str(specification["target_city"])
                for record in record_rows:
                    record["is_target"] = (
                        str(record["city"]).casefold() == target_city.casefold()
                    )
                panels[(layer, head)]["events"].append(
                    {
                        "event_index": int(task_index),
                        "from_occurrence": int(specification["from_occurrence"]),
                        "to_occurrence": int(specification["to_occurrence"]),
                        "query_output_token_index": int(
                            specification["query_output_token_index"]
                        ),
                        "query_full_sequence_token": int(encoding.query_position),
                        "query_token_text": str(specification.get("token_text", "")),
                        "target_city": target_city,
                        "records": record_rows,
                        "non_needle_context_mass": float(context_mass),
                        "attention_total_mass": float(attention.sum().item()),
                    }
                )
        output.extend(panels.values())
    return output


def _ranked_bank_heads(
    rankings: dict[str, Any], grammar: str, bank_size: int
) -> list[tuple[int, int]]:
    if grammar not in rankings:
        raise KeyError(f"No frozen ranking for grammar {grammar!r}")
    ordered = sorted(
        rankings[grammar]["rows"],
        key=lambda row: (int(row["rank"]), int(row["layer"]), int(row["head"])),
    )
    if len(ordered) < bank_size:
        raise ValueError(
            f"Grammar {grammar!r} has only {len(ordered)} ranked heads, "
            f"cannot build Top-{bank_size}"
        )
    selected = ordered[:bank_size]
    observed_ranks = [int(row["rank"]) for row in selected]
    if observed_ranks != list(range(1, bank_size + 1)):
        raise ValueError(
            f"Grammar {grammar!r} does not contain a contiguous Top-{bank_size}"
        )
    return [(int(row["layer"]), int(row["head"])) for row in selected]


def _capture_bank_examples(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    *,
    tasks_by_grammar: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]],
    rankings: dict[str, Any],
    banks: list[tuple[str, int]],
    max_queries: int,
) -> list[dict[str, Any]]:
    """Sum exact-P0 prompt-region attention over each frozen Top-K bank.

    The output deliberately preserves the same event-by-record layout as a
    single-head example.  The only semantic change is that every ``mass`` and
    ``attention_total_mass`` is a sum over the selected layer-head identities.
    Consequently each event's total mass is K (up to floating-point error), not
    one.  This is the artifact needed for a city-labelled bank-summed heatmap;
    ordinal aggregates cannot recover the off-target cells.
    """

    output: list[dict[str, Any]] = []
    for grammar, bank_size in banks:
        selected = _representative_request(tasks_by_grammar.get(grammar, []))
        if len(selected) > max_queries:
            indices = (
                [
                    round(index * (len(selected) - 1) / (max_queries - 1))
                    for index in range(max_queries)
                ]
                if max_queries > 1
                else [0]
            )
            selected = [selected[index] for index in dict.fromkeys(indices)]
        row = selected[0][0]
        heads = _ranked_bank_heads(rankings, grammar, bank_size)
        panel: dict[str, Any] = {
            "grammar": grammar,
            "bank_size": int(bank_size),
            "bank_heads": [
                {"layer": int(layer), "head": int(head)}
                for layer, head in heads
            ],
            "request_id": str(row.get("request_id", row.get("stimulus_id"))),
            "seed": int(row["seed"]),
            "gold_count": int(
                row.get("gold_count", len(row.get("gold_records", [])))
            ),
            "query_site": "p0_item_end",
            "aggregation": "sum_over_frozen_bank_heads",
            "events": [],
        }
        for task_index, (_row, specification) in enumerate(selected, start=1):
            encoding = build_native_causal_encoding(
                _row,
                tokenizer,
                query_output_token_index=int(specification["query_output_token_index"]),
                sequence_output_token_end=int(specification["target_output_token_end"]),
                selected_site=specification,
            )
            attention_rows, key_starts, _logits = position_attention_outputs(
                model,
                adapter,
                encoding,
                int(encoding.query_position),
            )
            summed_records: list[dict[str, Any]] | None = None
            context_mass = 0.0
            total_mass = 0.0
            for layer, head in heads:
                attention = attention_rows[layer][head]
                record_rows, head_context_mass = _record_masses(
                    attention,
                    key_start=int(key_starts[layer]),
                    prompt_record_spans=encoding.prompt_record_spans,
                )
                if summed_records is None:
                    summed_records = [
                        {**record, "mass": 0.0}
                        for record in record_rows
                    ]
                if len(summed_records) != len(record_rows):
                    raise RuntimeError("Prompt-record span count changed across layers")
                for accumulator, record in zip(summed_records, record_rows):
                    if (
                        int(accumulator["source_index"]) != int(record["source_index"])
                        or str(accumulator["city"]) != str(record["city"])
                    ):
                        raise RuntimeError("Prompt-record ordering changed across layers")
                    accumulator["mass"] += float(record["mass"])
                context_mass += float(head_context_mass)
                total_mass += float(attention.sum().item())
            if summed_records is None:
                raise RuntimeError("Frozen bank unexpectedly contained no heads")
            target_city = str(specification["target_city"])
            for record in summed_records:
                record["is_target"] = (
                    str(record["city"]).casefold() == target_city.casefold()
                )
            # Attention rows are materialized from bfloat16 model outputs.  A
            # per-head normalization error of a few 1e-4 can accumulate over a
            # wide bank, so validate to 0.1% relative error rather than using a
            # float32-scale absolute tolerance.
            if abs(total_mass - float(bank_size)) > max(1e-3, bank_size * 1e-3):
                raise RuntimeError(
                    f"Bank-summed attention total {total_mass:.6f} does not equal "
                    f"K={bank_size}"
                )
            panel["events"].append(
                {
                    "event_index": int(task_index),
                    "from_occurrence": int(specification["from_occurrence"]),
                    "to_occurrence": int(specification["to_occurrence"]),
                    "query_output_token_index": int(
                        specification["query_output_token_index"]
                    ),
                    "query_full_sequence_token": int(encoding.query_position),
                    "query_token_text": str(specification.get("token_text", "")),
                    "target_city": target_city,
                    "records": summed_records,
                    "non_needle_context_mass": float(context_mass),
                    "attention_total_mass": float(total_mass),
                }
            )
        output.append(panel)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path)
    parser.add_argument("--plan-k", type=int)
    parser.add_argument(
        "--atlas-template",
        type=Path,
        help="Reuse frozen rankings (and existing single-head examples) from an atlas JSON",
    )
    parser.add_argument("--model", required=True, choices=["Qwen3-8B", "Gemma4-E4B"])
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--head-example", action="append", type=_parse_head, default=[])
    parser.add_argument("--bank-example", action="append", type=_parse_bank, default=[])
    parser.add_argument("--max-queries", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.head_example and not args.bank_example:
        raise ValueError("At least one --head-example or --bank-example is required")
    if args.max_queries < 1:
        raise ValueError("--max-queries must be positive")

    config = V5Config.load(args.config)
    spec = resolve_model_spec(args.model)
    model, tokenizer, adapter = load_registered_model(
        spec,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )
    development = set(config.causal_development_seeds)
    generation_rows = [
        row
        for row in registered_records(
            read_jsonl(args.generations), config, model_label=args.model
        )
        if int(row["seed"]) in development
    ]
    tasks_by_grammar = _eligible_tasks(generation_rows, tokenizer)
    if args.atlas_template is not None:
        template = json.loads(args.atlas_template.read_text(encoding="utf-8"))
        if str(template.get("model_label")) != args.model:
            raise ValueError("Atlas template model does not match --model")
        rankings = template["rankings"]
        existing_examples = list(template.get("examples", []))
    else:
        if args.plan_root is None or args.plan_k is None:
            raise ValueError("--plan-root and --plan-k are required without --atlas-template")
        rankings = _ranking_bundle(args.plan_root, args.plan_k)
        existing_examples = []

    bundle = {
        "schema_version": "realistic_niah_v5_p0_head_atlas_v2",
        "model_label": args.model,
        "query_site": "p0_item_end",
        "selection_split": "discovery",
        "selection_aggregation": "seed_event_mean",
        "selection_metric": "target_source_attention_mass",
        "development_seeds": sorted(development),
        "rankings": rankings,
        "examples": existing_examples + (
            _capture_examples(
                model,
                tokenizer,
                adapter,
                tasks_by_grammar=tasks_by_grammar,
                examples=args.head_example,
                max_queries=int(args.max_queries),
            )
            if args.head_example
            else []
        ),
        "bank_examples": _capture_bank_examples(
            model,
            tokenizer,
            adapter,
            tasks_by_grammar=tasks_by_grammar,
            rankings=rankings,
            banks=args.bank_example,
            max_queries=int(args.max_queries),
        ) if args.bank_example else [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
