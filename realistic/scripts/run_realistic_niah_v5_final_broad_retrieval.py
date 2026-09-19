#!/usr/bin/env python3
"""Natural final-query broad retrieval on complete native-thinking traces.

This is the realistic-NiaH analogue of the synthetic thinking-mode
``trace_readout`` attention assay.  At the literal query token immediately
before the final numeric answer, it measures, per layer and head:

* total attention mass to the frozen parser-registered trace item endpoints;
* entropy-normalized coverage across those endpoints; and
* ``mass * coverage`` (the frozen V4.4 broad score).

Prompt needle records are measured independently with the same definition.
No token, attention edge, hidden state, or KV-cache entry is changed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.modeling import (  # noqa: E402
    load_registered_model,
    position_attention_outputs,
)
from realistic_niah_v4.prompts import TokenSpan  # noqa: E402
from realistic_niah_v4.spec import resolve_model_spec  # noqa: E402
from realistic_niah_v5.encoding import (  # noqa: E402
    NativeTraceEncoding,
    build_native_trace_encoding,
)
from realistic_niah_v5.parsing import (  # noqa: E402
    TraceCharSite,
    align_trace_sites,
    output_token_ids,
    raw_output_text,
)


SCHEMA = "realistic_niah_v5_final_broad_retrieval_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )
    temporary.replace(path)


def _stored_parse(row: Mapping[str, Any]) -> Mapping[str, Any]:
    parsed = row.get("trace_parse")
    if not isinstance(parsed, Mapping) or not isinstance(
        parsed.get("parser"), Mapping
    ):
        raise ValueError("The broad-retrieval assay requires frozen trace_parse")
    return parsed


def eligible_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    model_label: str,
    seeds: Iterable[int],
    counts: Iterable[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    allowed_seeds = {int(value) for value in seeds}
    allowed_counts = {int(value) for value in counts}
    selected: dict[tuple[int, int], dict[str, Any]] = {}
    audit: list[dict[str, Any]] = []
    for raw_row in rows:
        row = dict(raw_row)
        if str(row.get("model_label", row.get("model"))) != model_label:
            continue
        seed = int(row.get("seed", -1))
        count = len(row.get("gold_records", row.get("gold_pairs", [])))
        if seed not in allowed_seeds or count not in allowed_counts:
            continue
        parsed = _stored_parse(row)
        parser = parsed["parser"]
        accepted = bool(
            parser.get("trace_one_to_one")
            and row.get("exact_count", parsed.get("exact_count"))
            and int(parser.get("item_count", -1)) == count
        )
        audit.append(
            {
                "request_id": str(row.get("request_id", row.get("stimulus_id"))),
                "seed": seed,
                "gold_count": count,
                "trace_one_to_one": bool(parser.get("trace_one_to_one")),
                "frozen_item_count": int(parser.get("item_count", -1)),
                "stored_final_answer_exact": bool(
                    row.get("exact_count", parsed.get("exact_count"))
                ),
                "accepted": accepted,
            }
        )
        if not accepted:
            continue
        key = (seed, count)
        if key in selected:
            raise ValueError(f"Duplicate eligible row {key}")
        selected[key] = row
    missing = sorted(
        (seed, count)
        for seed in allowed_seeds
        for count in allowed_counts
        if (seed, count) not in selected
    )
    if missing:
        raise ValueError(f"Missing frozen one-to-one exact rows: {missing}")
    ordered = [selected[(seed, count)] for seed in sorted(allowed_seeds) for count in sorted(allowed_counts)]
    return ordered, audit


def _frozen_item_spans(
    row: Mapping[str, Any],
    tokenizer: Any,
    encoding: NativeTraceEncoding,
) -> tuple[tuple[TokenSpan, ...], list[dict[str, Any]]]:
    parsed = _stored_parse(row)
    raw_sites = parsed.get("char_sites")
    if not isinstance(raw_sites, list):
        raise ValueError("Frozen trace_parse has no char_sites")
    char_sites = [
        TraceCharSite(**dict(value))
        for value in raw_sites
        if isinstance(value, Mapping) and value.get("site_kind") == "item_end"
    ]
    aligned = align_trace_sites(
        tokenizer,
        raw_text=raw_output_text(row),
        baseline_output_token_ids=output_token_ids(row),
        sites=char_sites,
    )
    spans: list[TokenSpan] = []
    anchor_audit: list[dict[str, Any]] = []
    for site in aligned:
        end = site.literal_token_end
        anchor_policy = "literal_item_endpoint"
        if end is None and site.alignment_eligible:
            # Boundary-only retokenization can replace the last baseline token
            # when a partial item prefix is encoded.  Use the final token that
            # is still byte-for-byte shared with the frozen generated output;
            # never introduce the newly retokenized suffix token as a key.
            end = int(site.shared_baseline_prefix_tokens)
            anchor_policy = "last_shared_baseline_token_before_retokenized_boundary"
        if not site.alignment_eligible or end is None or end < 1:
            raise RuntimeError(
                f"Frozen item site failed exact token alignment: {site.to_dict()}"
            )
        # Use one exact endpoint key per item.  Besides avoiding a tokenizer
        # boundary ambiguity when a city shares its first token with leading
        # whitespace, this is the direct realistic analogue of synthetic
        # ``trace_markers_mass`` (one marker/endpoint key per trace item).
        absolute_end = int(encoding.prompt_token_count) + int(end)
        absolute_start = absolute_end - 1
        if absolute_end > int(encoding.query_position):
            raise RuntimeError("Frozen item span lies after the final answer query")
        occurrence = int(site.char_site.occurrence or 0)
        anchor_audit.append(
            {
                "occurrence": occurrence,
                "anchor_policy": anchor_policy,
                "output_token_start": int(end) - 1,
                "output_token_end": int(end),
                "alignment_strategy": site.alignment_strategy,
                "retokenized_suffix_tokens": int(site.retokenized_suffix_tokens),
            }
        )
        spans.append(
            TokenSpan(
                slot_index=occurrence,
                start=absolute_start,
                end=absolute_end,
                active=True,
                kind="frozen_native_trace_item_endpoint",
                canonical_length=1,
                model_token_length=1,
            )
        )
    spans.sort(key=lambda span: int(span.slot_index))
    expected = int(_stored_parse(row)["parser"]["item_count"])
    if len(spans) != expected or [int(span.slot_index) for span in spans] != list(
        range(1, expected + 1)
    ):
        raise RuntimeError(
            f"Frozen item registry is not exactly 1..{expected}: "
            f"{[span.slot_index for span in spans]}"
        )
    return tuple(spans), anchor_audit


def frozen_encoding(
    row: Mapping[str, Any], tokenizer: Any, *, site_id: str
) -> tuple[NativeTraceEncoding, list[dict[str, Any]]]:
    encoding = build_native_trace_encoding(
        row,
        tokenizer,
        site_id=site_id,
        candidate_counts=tuple(range(1, 10)),
    )
    spans, anchor_audit = _frozen_item_spans(row, tokenizer, encoding)
    return (
        replace(
            encoding,
            trace_item_spans=spans,
            slot_spans=spans,
            needle_spans=spans,
        ),
        anchor_audit,
    )


def _mass(
    row: torch.Tensor, *, key_start: int, spans: Sequence[Any]
) -> float:
    total = 0.0
    key_end = int(key_start) + int(row.shape[-1])
    for span in spans:
        left = max(int(span.start), int(key_start))
        right = min(int(span.end), key_end)
        if right > left:
            total += float(row[left - key_start : right - key_start].sum().item())
    return total


def _span_masses(
    row: torch.Tensor, *, key_start: int, spans: Sequence[Any]
) -> list[float]:
    return [_mass(row, key_start=key_start, spans=[span]) for span in spans]


def _broad(masses: Sequence[float], epsilon: float = 1e-12) -> dict[str, float]:
    values = np.asarray([float(value) for value in masses], dtype=float)
    total = float(values.sum())
    if values.size == 0 or total <= epsilon:
        return {"mass": total, "coverage": 0.0, "effective_spans": 0.0, "score": 0.0}
    probabilities = values / (total + epsilon)
    entropy = float(-np.sum(probabilities * np.log(probabilities + epsilon)))
    effective = float(math.exp(entropy))
    coverage = float(effective / values.size)
    return {
        "mass": total,
        "coverage": coverage,
        "effective_spans": effective,
        "score": float(total * coverage),
    }


def _next_token_readout(
    logits: torch.Tensor, encoding: NativeTraceEncoding, tokenizer: Any
) -> dict[str, Any]:
    values = logits.detach().float().cpu().reshape(-1)
    candidate_ids: dict[int, int] = {}
    for count, ids in encoding.count_candidate_answer_token_ids:
        if len(ids) != 1:
            raise ValueError(f"Count {count} is not a one-token answer: {ids}")
        candidate_ids[int(count)] = int(ids[0])
    counts = sorted(candidate_ids)
    candidate_logits = torch.tensor([float(values[candidate_ids[count]]) for count in counts])
    probabilities = torch.softmax(candidate_logits, dim=0)
    prediction = int(counts[int(torch.argmax(candidate_logits))])
    top_id = int(torch.argmax(values))
    reverse = {token_id: count for count, token_id in candidate_ids.items()}
    return {
        "candidate_prediction": prediction,
        "candidate_exact": prediction == int(encoding.count),
        "candidate_probabilities": {
            str(count): float(probabilities[index])
            for index, count in enumerate(counts)
        },
        "top_vocab_token_id": top_id,
        "top_vocab_token_text": tokenizer.decode([top_id], skip_special_tokens=False),
        "top_vocab_number": reverse.get(top_id),
        "top_vocab_exact": reverse.get(top_id) == int(encoding.count),
    }


def capture_row(
    model: Any,
    adapter: Any,
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    site_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    encoding, anchor_audit = frozen_encoding(row, tokenizer, site_id=site_id)
    attention_rows, key_starts, logits = position_attention_outputs(
        model, adapter, encoding, encoding.query_position
    )
    output: list[dict[str, Any]] = []
    for layer, (attention, key_start) in enumerate(zip(attention_rows, key_starts)):
        for head in range(int(attention.shape[0])):
            head_row = attention[head]
            trace_masses = _span_masses(
                head_row, key_start=int(key_start), spans=encoding.trace_item_spans
            )
            prompt_masses = _span_masses(
                head_row, key_start=int(key_start), spans=encoding.prompt_record_spans
            )
            trace = _broad(trace_masses)
            prompt = _broad(prompt_masses)
            generated_left = max(
                0, int(encoding.prompt_token_count) - int(key_start)
            )
            generated_right = min(
                int(encoding.query_position) - int(key_start),
                int(head_row.shape[-1]),
            )
            prior_trace_mass = (
                float(head_row[generated_left:max(generated_left, generated_right)].sum().item())
                if generated_right > generated_left
                else 0.0
            )
            prompt_right = min(
                int(encoding.prompt_token_count) - int(key_start),
                int(head_row.shape[-1]),
            )
            full_prompt_mass = (
                float(head_row[:max(0, prompt_right)].sum().item())
                if prompt_right > 0
                else 0.0
            )
            output.append(
                {
                    "schema_version": SCHEMA,
                    "request_id": encoding.request_id,
                    "model_label": encoding.model_label,
                    "seed": int(encoding.seed),
                    "gold_count": int(encoding.count),
                    "site_id": site_id,
                    "query_position": int(encoding.query_position),
                    "layer": int(layer),
                    "head": int(head),
                    "key_start": int(key_start),
                    "registered_trace_item_spans": len(trace_masses),
                    "trace_span_masses": trace_masses,
                    "trace_item_raw_mass": trace["mass"],
                    "trace_broad_coverage": trace["coverage"],
                    "trace_broad_effective_spans": trace["effective_spans"],
                    "trace_broad_score": trace["score"],
                    "all_prior_generated_trace_attention_mass": prior_trace_mass,
                    "trace_item_relative_mass": (
                        trace["mass"] / prior_trace_mass
                        if prior_trace_mass > 0
                        else None
                    ),
                    "active_prompt_needle_spans": len(prompt_masses),
                    "prompt_span_masses": prompt_masses,
                    "prompt_needle_raw_mass": prompt["mass"],
                    "prompt_broad_coverage": prompt["coverage"],
                    "prompt_broad_effective_spans": prompt["effective_spans"],
                    "prompt_broad_score": prompt["score"],
                    "full_prompt_mass": full_prompt_mass,
                    "prompt_needle_relative_mass": (
                        prompt["mass"] / full_prompt_mass
                        if full_prompt_mass > 0
                        else None
                    ),
                    "row_sum": float(head_row.sum().item()),
                }
            )
    reference = {
        "request_id": encoding.request_id,
        "model_label": encoding.model_label,
        "seed": int(encoding.seed),
        "gold_count": int(encoding.count),
        "query_position": int(encoding.query_position),
        "trace_item_token_spans": [
            [int(span.start), int(span.end)] for span in encoding.trace_item_spans
        ],
        "trace_item_anchor_audit": anchor_audit,
        "retokenized_boundary_anchor_count": sum(
            value["anchor_policy"]
            != "literal_item_endpoint"
            for value in anchor_audit
        ),
        "prompt_needle_token_spans": [
            [int(span.start), int(span.end)] for span in encoding.prompt_record_spans
        ],
        **_next_token_readout(logits, encoding, tokenizer),
    }
    return output, reference


def summarize(
    rows: Sequence[Mapping[str, Any]], references: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    by_layer: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["layer"]), int(row["head"]))].append(row)
        by_layer[int(row["layer"])].append(row)

    head_rows: list[dict[str, Any]] = []
    for (layer, head), frame in grouped.items():
        head_rows.append(
            {
                "layer": layer,
                "head": head,
                "samples": len(frame),
                "mean_trace_raw_mass": float(np.mean([r["trace_item_raw_mass"] for r in frame])),
                "mean_trace_coverage": float(np.mean([r["trace_broad_coverage"] for r in frame])),
                "mean_trace_effective_spans": float(np.mean([r["trace_broad_effective_spans"] for r in frame])),
                "mean_trace_broad_score": float(np.mean([r["trace_broad_score"] for r in frame])),
                "mean_trace_relative_mass": float(np.mean([r["trace_item_relative_mass"] for r in frame if r["trace_item_relative_mass"] is not None])),
                "mean_prompt_raw_mass": float(np.mean([r["prompt_needle_raw_mass"] for r in frame])),
                "mean_prompt_coverage": float(np.mean([r["prompt_broad_coverage"] for r in frame])),
                "mean_prompt_broad_score": float(np.mean([r["prompt_broad_score"] for r in frame])),
            }
        )
    trace_rank = sorted(
        head_rows,
        key=lambda row: (-row["mean_trace_broad_score"], -row["mean_trace_raw_mass"], row["layer"], row["head"]),
    )
    prompt_rank = sorted(
        head_rows,
        key=lambda row: (-row["mean_prompt_broad_score"], -row["mean_prompt_raw_mass"], row["layer"], row["head"]),
    )

    layer_rows: list[dict[str, Any]] = []
    for layer, frame in sorted(by_layer.items()):
        head_summary = [row for row in head_rows if int(row["layer"]) == layer]
        best_trace = max(head_summary, key=lambda row: row["mean_trace_broad_score"])
        best_prompt = max(head_summary, key=lambda row: row["mean_prompt_broad_score"])
        layer_rows.append(
            {
                "layer": layer,
                "heads": len(head_summary),
                "mean_over_heads_trace_raw_mass": float(np.mean([r["trace_item_raw_mass"] for r in frame])),
                "mean_over_heads_trace_coverage": float(np.mean([r["trace_broad_coverage"] for r in frame])),
                "mean_over_heads_trace_broad_score": float(np.mean([r["trace_broad_score"] for r in frame])),
                "best_trace_head": [int(best_trace["layer"]), int(best_trace["head"])],
                "best_trace_head_mean_raw_mass": best_trace["mean_trace_raw_mass"],
                "best_trace_head_mean_coverage": best_trace["mean_trace_coverage"],
                "best_trace_head_mean_effective_spans": best_trace["mean_trace_effective_spans"],
                "best_trace_head_mean_broad_score": best_trace["mean_trace_broad_score"],
                "best_prompt_head": [int(best_prompt["layer"]), int(best_prompt["head"])],
                "best_prompt_head_mean_broad_score": best_prompt["mean_prompt_broad_score"],
            }
        )
    return {
        "schema_version": SCHEMA,
        "samples": len(references),
        "clean_candidate_exact_rate": float(np.mean([r["candidate_exact"] for r in references])),
        "clean_top_vocab_exact_rate": float(np.mean([r["top_vocab_exact"] for r in references])),
        "top_trace_heads": trace_rank[:32],
        "top_prompt_heads": prompt_rank[:32],
        "layer_summary": layer_rows,
        "references": list(references),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1238, 1241, 1243])
    parser.add_argument("--counts", type=int, nargs="+", default=[3, 6, 8])
    parser.add_argument("--site-id", default="answer_query_v3")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")
    summary_path = args.output.with_suffix(".summary.json")
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"Summary already exists: {summary_path}")

    selected, audit = eligible_rows(
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
    rows: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        captured, reference = capture_row(
            model, adapter, tokenizer, row, site_id=args.site_id
        )
        rows.extend(captured)
        references.append(reference)
        print(
            f"[final-broad-retrieval] {index}/{len(selected)} "
            f"{reference['request_id']} clean={reference['top_vocab_exact']}",
            flush=True,
        )
    write_jsonl(args.output, rows)
    summary = {
        **summarize(rows, references),
        "model_label": args.model,
        "generations": str(args.generations.resolve()),
        "output": str(args.output.resolve()),
        "site_id": args.site_id,
        "seeds": [int(value) for value in args.seeds],
        "counts": [int(value) for value in args.counts],
        "gate_audit": audit,
        "metric_definition": "total span attention mass * exp(entropy(normalized span masses)) / item_count",
        "trace_key_definition": "one exact frozen item-end endpoint token per registered trace item, directly analogous to synthetic trace marker keys",
        "interpretation_limit": "Natural attention is descriptive routing evidence, not causal necessity or recurrent-counter evidence.",
    }
    write_json(summary_path, summary)
    print(f"[final-broad-retrieval] wrote {args.output} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
