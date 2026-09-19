#!/usr/bin/env python3
"""Layerwise answer-query state transport on complete native-thinking traces.

This is the realistic-NiaH analogue of the synthetic counting experiment's
``<Ans>`` residual transplant.  A donor and receiver use complete, naturally
generated traces from the same seed but have different gold counts.  At the
literal token immediately before the first answer-number token, one decoder
block's post-block residual is replaced once; all later blocks recompute.

The script deliberately does not truncate a trace or alter the task.  Its
primary endpoint is whether the greedy next token (and the restricted numeric
candidate argmax) adopts the donor count.  Self patches and same-count,
different-seed patches are matched controls.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.modeling import (  # noqa: E402
    capture_post_block_states,
    generate_with_residual_interventions,
    load_registered_model,
    run_with_residual_patch,
)
from realistic_niah_v4.spec import resolve_model_spec  # noqa: E402
from realistic_niah.parsing import parse_total  # noqa: E402
from realistic_niah_v5.encoding import (  # noqa: E402
    NativeTraceEncoding,
    build_native_trace_encoding,
)
from realistic_niah_v5.parsing import parse_trace_record  # noqa: E402


SCHEMA = "realistic_niah_v5_final_readout_transplant_v1"


def completion_metrics(
    result: Mapping[str, Any], *, gold_count: int
) -> dict[str, Any]:
    """Minimal generation readout, kept local to avoid causal-analysis deps."""

    full = str(result.get("full_answer_text", ""))
    prediction = parse_total(full)
    return {
        "prediction": prediction,
        "exact_count": prediction == int(gold_count),
        "signed_error": (
            prediction - int(gold_count) if prediction is not None else None
        ),
        "absolute_error": (
            abs(prediction - int(gold_count)) if prediction is not None else None
        ),
        "completion_text_raw": result.get("completion_text_raw"),
        "generated_token_count": result.get("generated_token_count"),
        "generation_truncated": result.get("generation_truncated"),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("request_id", row.get("stimulus_id")))


def eligible_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    model_label: str,
    seeds: Iterable[int],
    counts: Iterable[int],
) -> tuple[dict[tuple[int, int], dict[str, Any]], list[dict[str, Any]]]:
    allowed_seeds = {int(value) for value in seeds}
    allowed_counts = {int(value) for value in counts}
    selected: dict[tuple[int, int], dict[str, Any]] = {}
    audit: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        if str(row.get("model_label", row.get("model"))) != str(model_label):
            continue
        seed = int(row["seed"])
        count = int(row["gold_count"])
        if seed not in allowed_seeds or count not in allowed_counts:
            continue
        # ``*_generations_reparsed.jsonl`` is itself the frozen parser artifact.
        # Prefer its stored decision so a stale GPU-runtime parser cannot
        # silently redefine cohort membership.  Legacy rows without the stored
        # payload retain the runtime-parser fallback.
        stored_parse = row.get("trace_parse")
        if isinstance(stored_parse, Mapping) and isinstance(
            stored_parse.get("parser"), Mapping
        ):
            parsed = dict(stored_parse)
            parser_source = "stored_frozen_trace_parse"
        else:
            parsed = parse_trace_record(row)
            parser_source = "runtime_parser_fallback"
        one_to_one = bool(parsed["parser"].get("trace_one_to_one"))
        exact = bool(row.get("exact_count", parsed.get("exact_count")))
        accepted = bool(one_to_one and exact)
        audit.append(
            {
                "request_id": _row_id(row),
                "seed": seed,
                "gold_count": count,
                "trace_one_to_one": one_to_one,
                "stored_final_answer_exact": exact,
                "accepted": accepted,
                "trace_category": parsed["parser"].get("trace_category"),
                "parser_source": parser_source,
            }
        )
        if not accepted:
            continue
        key = (seed, count)
        if key in selected:
            raise ValueError(f"Duplicate eligible seed/count row: {key}")
        selected[key] = row
    missing = sorted(
        (seed, count)
        for seed in allowed_seeds
        for count in allowed_counts
        if (seed, count) not in selected
    )
    if missing:
        raise ValueError(
            "Every requested seed/count must have a clean-correct, strict "
            f"one-to-one native trace; missing={missing}"
        )
    return selected, audit


def cyclic_successor(values: Iterable[int]) -> dict[int, int]:
    ordered = tuple(int(value) for value in values)
    if len(ordered) < 2 or len(set(ordered)) != len(ordered):
        raise ValueError("Cyclic controls require at least two unique values")
    return {
        value: ordered[(index + 1) % len(ordered)]
        for index, value in enumerate(ordered)
    }


def _single_token_candidates(
    encoding: NativeTraceEncoding,
    candidate_counts: Iterable[int],
) -> dict[int, int]:
    answer_ids = dict(encoding.count_candidate_answer_token_ids)
    result: dict[int, int] = {}
    for count in candidate_counts:
        tokens = tuple(int(value) for value in answer_ids[int(count)])
        if len(tokens) != 1:
            raise ValueError(
                f"Count {count} is not a single-token answer at "
                f"{encoding.request_id}: {tokens}. Restrict --candidate-counts "
                "to single-token numbers for this next-token assay."
            )
        result[int(count)] = tokens[0]
    if len(set(result.values())) != len(result):
        raise ValueError("Numeric candidates do not have unique next-token ids")
    return result


def logit_readout(
    logits: torch.Tensor,
    *,
    tokenizer: Any,
    candidates: Mapping[int, int],
) -> dict[str, Any]:
    values = logits.detach().float().cpu().reshape(-1)
    counts = tuple(sorted(int(value) for value in candidates))
    token_ids = torch.tensor([int(candidates[count]) for count in counts])
    candidate_logits = values[token_ids]
    probabilities = torch.softmax(candidate_logits, dim=0)
    best_index = int(torch.argmax(candidate_logits))
    candidate_prediction = int(counts[best_index])
    expected_count = float(
        sum(float(count) * float(probabilities[index]) for index, count in enumerate(counts))
    )
    top_token_id = int(torch.argmax(values))
    reverse = {int(token): int(count) for count, token in candidates.items()}
    top_token_number = reverse.get(top_token_id)
    return {
        "candidate_prediction": candidate_prediction,
        "candidate_expected_count": expected_count,
        "candidate_probabilities": {
            str(count): float(probabilities[index])
            for index, count in enumerate(counts)
        },
        "candidate_logits": {
            str(count): float(candidate_logits[index])
            for index, count in enumerate(counts)
        },
        "top_vocab_token_id": top_token_id,
        "top_vocab_token_text": tokenizer.decode(
            [top_token_id], skip_special_tokens=False
        ),
        "top_vocab_number": top_token_number,
        "top_vocab_is_numeric_candidate": top_token_number is not None,
    }


def summarize(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    clean = [row for row in rows if str(row.get("condition")) == "clean_reference"]
    trials = [row for row in rows if str(row.get("condition")) != "clean_reference"]
    for row in trials:
        groups[(int(row["layer"]), str(row["condition"]))].append(row)

    def rate(values: Iterable[bool]) -> float | None:
        materialized = [bool(value) for value in values]
        return (
            float(sum(materialized) / len(materialized)) if materialized else None
        )

    layer_rows = []
    for (layer, condition), frame in sorted(groups.items()):
        transport = [
            float(row["normalized_expected_transport"])
            for row in frame
            if row.get("normalized_expected_transport") is not None
            and math.isfinite(float(row["normalized_expected_transport"]))
        ]
        generated = [row for row in frame if row.get("generated_prediction") is not None]
        layer_rows.append(
            {
                "layer": layer,
                "condition": condition,
                "trials": len(frame),
                "seeds": len({int(row["receiver_seed"]) for row in frame}),
                "candidate_target_hit_rate": rate(
                    int(row["candidate_prediction"]) == int(row["target_count"])
                    for row in frame
                ),
                "top_vocab_target_hit_rate": rate(
                    row.get("top_vocab_number") is not None
                    and int(row["top_vocab_number"]) == int(row["target_count"])
                    for row in frame
                ),
                "candidate_receiver_retention_rate": rate(
                    int(row["candidate_prediction"])
                    == int(row["receiver_count"])
                    for row in frame
                ),
                "mean_normalized_expected_transport": (
                    float(sum(transport) / len(transport)) if transport else None
                ),
                "generation_validation_trials": len(generated),
                "generated_target_hit_rate": rate(
                    int(row["generated_prediction"]) == int(row["target_count"])
                    for row in generated
                ),
            }
        )
    return {
        "schema_version": SCHEMA,
        "clean_references": len(clean),
        "clean_candidate_exact_rate": rate(
            int(row["candidate_prediction"]) == int(row["gold_count"])
            for row in clean
        ),
        "clean_top_vocab_exact_rate": rate(
            row.get("top_vocab_number") is not None
            and int(row["top_vocab_number"]) == int(row["gold_count"])
            for row in clean
        ),
        "layer_condition_summary": layer_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1238, 1241, 1243])
    parser.add_argument("--counts", type=int, nargs="+", default=[3, 6, 8])
    parser.add_argument(
        "--candidate-counts", type=int, nargs="+", default=list(range(1, 10))
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[0, 4, 8, 12, 16, 20, 24, 28, 32, 35]
    )
    parser.add_argument("--generation-validation-layers", type=int, nargs="*", default=[35])
    parser.add_argument("--site-id", default="answer_query_v3")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    seeds = tuple(int(value) for value in args.seeds)
    counts = tuple(int(value) for value in args.counts)
    layers = tuple(sorted({int(value) for value in args.layers}))
    validation_layers = {int(value) for value in args.generation_validation_layers}
    if not validation_layers.issubset(layers):
        raise ValueError("Generation-validation layers must be included in --layers")
    count_donor = cyclic_successor(counts)
    same_count_seed_donor = cyclic_successor(seeds)

    selected, gate_audit = eligible_rows(
        read_jsonl(args.generations),
        model_label=args.model,
        seeds=seeds,
        counts=counts,
    )
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {args.output}")
        args.output.unlink()

    spec = resolve_model_spec(args.model)
    model, tokenizer, adapter = load_registered_model(
        spec,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )
    if any(layer < 0 or layer >= int(adapter.num_layers) for layer in layers):
        raise ValueError(
            f"Invalid layers for {args.model}; num_layers={adapter.num_layers}: {layers}"
        )

    encodings: dict[tuple[int, int], NativeTraceEncoding] = {}
    states: dict[tuple[int, int], dict[int, torch.Tensor]] = {}
    clean_readouts: dict[tuple[int, int], dict[str, Any]] = {}
    candidate_maps: dict[tuple[int, int], dict[int, int]] = {}
    for index, key in enumerate(sorted(selected), start=1):
        encoding = build_native_trace_encoding(
            selected[key], tokenizer, site_id=args.site_id
        )
        candidates = _single_token_candidates(encoding, args.candidate_counts)
        logits, captured = capture_post_block_states(
            model,
            adapter,
            encoding,
            [encoding.query_position],
            layers=layers,
        )
        encodings[key] = encoding
        candidate_maps[key] = candidates
        states[key] = {layer: captured[layer][0] for layer in layers}
        clean_readouts[key] = logit_readout(
            logits, tokenizer=tokenizer, candidates=candidates
        )
        clean_row = {
            "schema_version": SCHEMA,
            "condition": "clean_reference",
            "request_id": encoding.request_id,
            "seed": key[0],
            "gold_count": key[1],
            "site_id": args.site_id,
            "query_position": int(encoding.query_position),
            "sequence_length": int(encoding.sequence_length),
            "stored_final_answer_exact": True,
            **clean_readouts[key],
        }
        append_jsonl(args.output, clean_row)
        print(
            f"[capture] {index}/{len(selected)} seed={key[0]} N={key[1]} "
            f"greedy={clean_row['top_vocab_number']} "
            f"candidate={clean_row['candidate_prediction']}",
            flush=True,
        )

    trial_total = len(layers) * len(seeds) * len(counts) * 3
    completed = 0
    for layer in layers:
        for receiver_seed in seeds:
            for receiver_count in counts:
                receiver_key = (receiver_seed, receiver_count)
                receiver = encodings[receiver_key]
                receiver_clean = clean_readouts[receiver_key]
                conditions = (
                    ("self_patch", receiver_seed, receiver_count),
                    (
                        "same_count_different_seed_control",
                        same_count_seed_donor[receiver_seed],
                        receiver_count,
                    ),
                    (
                        "different_count_same_seed_donor",
                        receiver_seed,
                        count_donor[receiver_count],
                    ),
                )
                for condition, donor_seed, donor_count in conditions:
                    donor_key = (donor_seed, donor_count)
                    donor = encodings[donor_key]
                    patched_logits = run_with_residual_patch(
                        model,
                        adapter,
                        receiver,
                        layer=layer,
                        receiver_positions=[receiver.query_position],
                        donor_states=states[donor_key][layer],
                    )
                    readout = logit_readout(
                        patched_logits,
                        tokenizer=tokenizer,
                        candidates=candidate_maps[receiver_key],
                    )
                    target_count = (
                        donor_count
                        if condition == "different_count_same_seed_donor"
                        else receiver_count
                    )
                    receiver_expected = float(
                        receiver_clean["candidate_expected_count"]
                    )
                    donor_expected = float(
                        clean_readouts[donor_key]["candidate_expected_count"]
                    )
                    denominator = donor_expected - receiver_expected
                    normalized_transport = (
                        (float(readout["candidate_expected_count"]) - receiver_expected)
                        / denominator
                        if condition == "different_count_same_seed_donor"
                        and abs(denominator) > 1e-8
                        else None
                    )
                    generated_prediction = None
                    generated_payload: dict[str, Any] = {}
                    if layer in validation_layers:
                        generated = generate_with_residual_interventions(
                            model,
                            tokenizer,
                            adapter,
                            receiver,
                            {layer: ([receiver.query_position], states[donor_key][layer])},
                            max_new_tokens=args.max_new_tokens,
                        )
                        metrics = completion_metrics(
                            generated, gold_count=receiver_count
                        )
                        generated_prediction = metrics["prediction"]
                        generated_payload = {
                            "generated_prediction": generated_prediction,
                            "generated_completion_text_raw": metrics[
                                "completion_text_raw"
                            ],
                            "generated_token_count": metrics[
                                "generated_token_count"
                            ],
                            "generated_target_hit": (
                                generated_prediction == int(target_count)
                            ),
                        }
                    trial_row = {
                        "schema_version": SCHEMA,
                        "condition": condition,
                        "layer": layer,
                        "site_id": args.site_id,
                        "receiver_request_id": receiver.request_id,
                        "donor_request_id": donor.request_id,
                        "receiver_seed": receiver_seed,
                        "donor_seed": donor_seed,
                        "receiver_count": receiver_count,
                        "donor_count": donor_count,
                        "target_count": target_count,
                        "receiver_query_position": int(receiver.query_position),
                        "donor_query_position": int(donor.query_position),
                        "query_position_delta": int(
                            donor.query_position - receiver.query_position
                        ),
                        "receiver_clean_candidate_prediction": int(
                            receiver_clean["candidate_prediction"]
                        ),
                        "receiver_clean_top_vocab_number": receiver_clean[
                            "top_vocab_number"
                        ],
                        "donor_clean_candidate_prediction": int(
                            clean_readouts[donor_key]["candidate_prediction"]
                        ),
                        "donor_clean_top_vocab_number": clean_readouts[donor_key][
                            "top_vocab_number"
                        ],
                        "normalized_expected_transport": normalized_transport,
                        "generated_prediction": generated_prediction,
                        **readout,
                        **generated_payload,
                    }
                    append_jsonl(args.output, trial_row)
                    completed += 1
                    print(
                        f"[patch] {completed}/{trial_total} L{layer} {condition} "
                        f"S{receiver_seed}:N{receiver_count} <- "
                        f"S{donor_seed}:N{donor_count} pred="
                        f"{trial_row['top_vocab_number']}/"
                        f"{trial_row['candidate_prediction']}",
                        flush=True,
                    )

    all_rows = read_jsonl(args.output)
    summary = summarize(all_rows)
    summary.update(
        {
            "model_label": args.model,
            "generations": str(args.generations.resolve()),
            "output": str(args.output.resolve()),
            "site_id": args.site_id,
            "seeds": list(seeds),
            "counts": list(counts),
            "candidate_counts": [int(value) for value in args.candidate_counts],
            "layers": list(layers),
            "generation_validation_layers": sorted(validation_layers),
            "pair_policy": (
                "within-seed cyclic different-count donor; cyclic different-seed "
                "same-count control"
            ),
            "eligibility": (
                "stored native final answer exact and strict trace one-to-one"
            ),
            "gate_audit": gate_audit,
            "interpretation_limit": (
                "This tests a sufficient final answer-query count readout. It does "
                "not by itself establish a recurrent trace-local counter."
            ),
        }
    )
    summary_path = args.output.with_suffix(".summary.json")
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"[done] detail={args.output} summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
