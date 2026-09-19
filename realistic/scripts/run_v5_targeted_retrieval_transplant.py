#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import inspect
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.modeling import load_registered_model
from realistic_niah_v4.spec import resolve_model_spec
from realistic_niah_v5.encoding import NativeTraceEncoding
from realistic_niah_v5.parsing import output_token_ids, raw_output_text
from realistic_niah_v5.pre_city import (
    PreCityQuery,
    baseline_prefix_encoding,
    pre_city_token_queries,
)


SCHEMA = "realistic_niah_v5_targeted_retrieval_transplant_smoke_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temporary.replace(path)


def strict_discovery_n10(row: Mapping[str, Any], model_label: str) -> bool:
    parser = ((row.get("trace_parse") or {}).get("parser") or {})
    return bool(
        str(row.get("model_label")) == model_label
        and str(row.get("split")) == "discovery"
        and len(row.get("gold_records") or ()) == 10
        and parser.get("trace_one_to_one") is True
        and (row.get("trace_parse") or {}).get("exact_count") is True
    )


def parse_heads(value: str) -> list[tuple[int, int]]:
    parsed = json.loads(value)
    return [(int(layer), int(head)) for layer, head in parsed]


def frozen_heads(
    path: Path, model_label: str, query_variant: str
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["model_label"] == model_label
            and row["query_variant"] == query_variant
            and int(row["bank_size"]) == 1
        ]
    ranked = next(
        row for row in rows if row["condition"] == "pre_city_targeted_retrieval_ranked"
    )
    random = next(
        row
        for row in rows
        if row["condition"] == "layer_matched_random" and int(row["repeat"]) == 1
    )
    selected_heads = parse_heads(ranked["heads"])
    random_heads = parse_heads(random["heads"])
    if {layer for layer, _ in selected_heads} != {layer for layer, _ in random_heads}:
        raise RuntimeError("Random control is not exactly layer matched")
    if set(selected_heads) & set(random_heads):
        raise RuntimeError("Random control overlaps the selected bank")
    audit = {
        "selected_heads": selected_heads,
        "random_heads": random_heads,
        "selected_discovery_raw_mass": float(ranked["target_needle_raw_mass"]),
        "selected_discovery_relative_mass": float(
            ranked["target_needle_relative_mass"]
        ),
        "random_discovery_raw_mass": float(random["target_needle_raw_mass"]),
        "random_discovery_relative_mass": float(
            random["target_needle_relative_mass"]
        ),
        "attention_mass_split": ranked["attention_mass_split"],
    }
    if audit["attention_mass_split"] != "discovery":
        raise RuntimeError("Target head bank was not frozen on discovery")
    return selected_heads, random_heads, audit


def _encode(tokenizer: Any, text: str) -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in tokenizer.encode(text, add_special_tokens=False)
    )


def _score_char_span(
    row: Mapping[str, Any], occurrence: int, city: str, score: int
) -> tuple[int, int]:
    raw = raw_output_text(row)
    sites = ((row.get("trace_parse") or {}).get("char_sites") or ())
    item = next(
        site
        for site in sites
        if site.get("site_id") == f"item_end:{int(occurrence)}"
    )
    left, right = int(item["char_start"]), int(item["char_end"])
    segment = raw[left:right]
    match = re.search(
        rf"(?<!\w){re.escape(city)}(?!\w).*?score\s+of\s+({score})(?!\d)",
        segment,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        raise ValueError("Could not isolate the target score in its trace item")
    return left + match.start(1), left + match.end(1)


def _score_token_start(
    row: Mapping[str, Any], tokenizer: Any, occurrence: int, city: str, score: int
) -> tuple[int, int, int]:
    raw = raw_output_text(row)
    char_left, char_right = _score_char_span(row, occurrence, city, score)
    encoded = tokenizer(
        raw, add_special_tokens=False, return_offsets_mapping=True
    )
    ids = tuple(int(value) for value in encoded["input_ids"])
    offsets = tuple((int(a), int(b)) for a, b in encoded["offset_mapping"])
    baseline = output_token_ids(row)
    if baseline[: len(ids)] != ids:
        raise RuntimeError("Raw trace no longer retokenizes to the baseline prefix")
    indices = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > char_left and left < char_right and right > left
    ]
    if not indices:
        raise RuntimeError("No output token overlaps the target score")
    return min(indices), char_left, char_right


def make_score_counterfactual(
    row: Mapping[str, Any], tokenizer: Any, query: PreCityQuery
) -> tuple[dict[str, Any], dict[str, Any]]:
    records = list(row.get("gold_records") or ())
    record = records[int(query.occurrence) - 1]
    city = str(record["city"])
    original_score = int(record["score"])
    if city.casefold() != query.city.casefold():
        raise RuntimeError("Query occurrence and gold record disagree")
    rendered = str(row["rendered_prompt"])
    clean_prompt_ids = tuple(int(value) for value in row["input_ids"])
    if _encode(tokenizer, rendered) != clean_prompt_ids:
        raise RuntimeError("Rendered prompt does not reproduce stored prompt IDs")
    literal = f"{city} received a score of {original_score}."
    if rendered.count(literal) != 1:
        raise RuntimeError(f"Prompt target literal count is {rendered.count(literal)}")
    span = (row.get("prompt_record_spans") or ())[int(query.occurrence) - 1]
    score_token_start, char_left, char_right = _score_token_start(
        row, tokenizer, query.occurrence, city, original_score
    )
    raw = raw_output_text(row)
    candidates: list[tuple[int, int, tuple[int, ...], tuple[int, ...], str]] = []
    for candidate in range(10, 100):
        if candidate == original_score:
            continue
        replacement = f"{city} received a score of {candidate}."
        candidate_prompt = rendered.replace(literal, replacement, 1)
        candidate_ids = _encode(tokenizer, candidate_prompt)
        if len(candidate_ids) != len(clean_prompt_ids):
            continue
        changed = tuple(
            index
            for index, (clean, altered) in enumerate(
                zip(clean_prompt_ids, candidate_ids)
            )
            if clean != altered
        )
        if not changed or not all(
            int(span["start"]) <= index < int(span["end"]) for index in changed
        ):
            continue
        hypothetical_raw = raw[:char_left] + str(candidate) + raw[char_right:]
        hypothetical_ids = _encode(tokenizer, hypothetical_raw)
        if hypothetical_ids[:score_token_start] != output_token_ids(row)[:score_token_start]:
            continue
        if score_token_start >= len(hypothetical_ids):
            continue
        if hypothetical_ids[score_token_start] == output_token_ids(row)[score_token_start]:
            continue
        candidates.append(
            (
                abs(candidate - original_score),
                candidate,
                candidate_ids,
                changed,
                candidate_prompt,
            )
        )
    if not candidates:
        raise RuntimeError("No token-length-matched score counterfactual exists")
    _, altered_score, altered_prompt_ids, changed, altered_prompt = max(candidates)
    cf = copy.deepcopy(dict(row))
    cf["input_ids"] = list(altered_prompt_ids)
    cf["rendered_prompt"] = altered_prompt
    cf_records = copy.deepcopy(records)
    cf_records[int(query.occurrence) - 1]["score"] = int(altered_score)
    cf["gold_records"] = cf_records
    cf_spans = copy.deepcopy(list(row.get("prompt_record_spans") or ()))
    cf_spans[int(query.occurrence) - 1]["score"] = int(altered_score)
    cf["prompt_record_spans"] = cf_spans
    hypothetical_raw = raw[:char_left] + str(altered_score) + raw[char_right:]
    hypothetical_ids = _encode(tokenizer, hypothetical_raw)
    audit = {
        "occurrence": int(query.occurrence),
        "city": city,
        "original_score": original_score,
        "counterfactual_score": int(altered_score),
        "prompt_token_count": len(clean_prompt_ids),
        "prompt_changed_token_indices": list(changed),
        "prompt_changed_token_count": len(changed),
        "prompt_change_inside_target_record_only": True,
        "prompt_length_matched": True,
        "score_output_token_start": int(score_token_start),
        "original_score_first_token_id": int(output_token_ids(row)[score_token_start]),
        "counterfactual_score_first_token_id": int(
            hypothetical_ids[score_token_start]
        ),
    }
    return cf, audit


def _input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


def _tensors(model: Any, encoding: NativeTraceEncoding) -> tuple[torch.Tensor, torch.Tensor]:
    device = _input_device(model)
    ids = torch.tensor([encoding.input_ids], dtype=torch.long, device=device)
    mask = torch.tensor([encoding.attention_mask], dtype=torch.long, device=device)
    return ids, mask


def _accepts(model: Any, name: str) -> bool:
    signature = inspect.signature(model.forward)
    return name in signature.parameters or any(
        value.kind == inspect.Parameter.VAR_KEYWORD
        for value in signature.parameters.values()
    )


@torch.inference_mode()
def capture_head_outputs(
    model: Any,
    adapter: Any,
    encoding: NativeTraceEncoding,
    heads: Sequence[tuple[int, int]],
    hook_position: int,
) -> tuple[dict[tuple[int, int], torch.Tensor], dict[str, Any]]:
    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(int(layer), []).append(int(head))
    captured: dict[tuple[int, int], torch.Tensor] = {}
    calls: dict[int, int] = {layer: 0 for layer in by_layer}
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
        ) -> None:
            value = args[0]
            if value.ndim != 3 or hook_position >= int(value.shape[1]):
                raise RuntimeError("Capture query is outside the prefill tensor")
            calls[layer] += 1
            for head in layer_heads:
                left = head * head_dim
                captured[(layer, head)] = value[
                    0, hook_position, left : left + head_dim
                ].detach().clone()

        handles.append(adapter.output_projections[layer].register_forward_pre_hook(hook))
    ids, mask = _tensors(model, encoding)
    kwargs: dict[str, Any] = {
        "input_ids": ids,
        "attention_mask": mask,
        "use_cache": False,
        "return_dict": True,
    }
    if _accepts(model, "logits_to_keep"):
        kwargs["logits_to_keep"] = 1
    try:
        model(**kwargs)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(heads):
        raise RuntimeError("Not every requested head output was captured")
    return captured, {"capture_calls_by_layer": calls, "capture_audit": "PASS"}


def _patch_hooks(
    adapter: Any,
    donor: Mapping[tuple[int, int], torch.Tensor],
    heads: Sequence[tuple[int, int]],
    hook_position: int,
) -> tuple[list[Any], dict[str, int]]:
    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(int(layer), []).append(int(head))
    applied = {f"L{layer}": 0 for layer in by_layer}
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
        ) -> tuple[Any, ...] | None:
            value = args[0]
            if value.ndim != 3:
                raise RuntimeError("Attention projection input is not rank three")
            if hook_position >= int(value.shape[1]):
                return None
            patched = value.clone()
            for head in layer_heads:
                left = head * head_dim
                patched[:, hook_position, left : left + head_dim] = donor[
                    (layer, head)
                ].to(device=value.device, dtype=value.dtype)
            applied[f"L{layer}"] += 1
            return (patched, *args[1:])

        handles.append(adapter.output_projections[layer].register_forward_pre_hook(hook))
    return handles, applied


@torch.inference_mode()
def score_logits(
    model: Any,
    adapter: Any,
    encoding: NativeTraceEncoding,
    *,
    patch_position: int,
    donor: Mapping[tuple[int, int], torch.Tensor] | None,
    heads: Sequence[tuple[int, int]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    handles: list[Any] = []
    applied: dict[str, int] = {}
    if heads:
        if donor is None:
            raise ValueError("Patched logits require donor states")
        handles, applied = _patch_hooks(
            adapter, donor, heads, hook_position=patch_position
        )
    ids, mask = _tensors(model, encoding)
    kwargs: dict[str, Any] = {
        "input_ids": ids,
        "attention_mask": mask,
        "use_cache": False,
        "return_dict": True,
    }
    if _accepts(model, "logits_to_keep"):
        kwargs["logits_to_keep"] = 1
    try:
        output = model(**kwargs)
    finally:
        for handle in handles:
            handle.remove()
    logits = output.logits[0, -1].detach().float().cpu()
    if heads and any(value != 1 for value in applied.values()):
        raise RuntimeError(f"Patch application audit failed: {applied}")
    return logits, {"patch_applied_by_layer": applied, "patch_audit": "PASS"}


@torch.inference_mode()
def generate_marker(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    encoding: NativeTraceEncoding,
    *,
    patch_position: int,
    donor: Mapping[tuple[int, int], torch.Tensor] | None,
    heads: Sequence[tuple[int, int]],
    max_new_tokens: int,
) -> tuple[str, tuple[int, ...], dict[str, Any]]:
    handles: list[Any] = []
    applied: dict[str, int] = {}
    if heads:
        if donor is None:
            raise ValueError("Patched generation requires donor states")
        handles, applied = _patch_hooks(
            adapter, donor, heads, hook_position=patch_position
        )
    ids, mask = _tensors(model, encoding)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        eos = tokenizer.eos_token_id
        pad_id = int(eos[0] if isinstance(eos, (list, tuple)) else eos)
    try:
        generated = model.generate(
            input_ids=ids,
            attention_mask=mask,
            do_sample=False,
            use_cache=True,
            max_new_tokens=int(max_new_tokens),
            pad_token_id=int(pad_id),
        )
    finally:
        for handle in handles:
            handle.remove()
    continuation = tuple(int(value) for value in generated[0, ids.shape[1] :].tolist())
    text = tokenizer.decode(
        list(continuation),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if heads and any(value != 1 for value in applied.values()):
        raise RuntimeError(f"Generation patch application audit failed: {applied}")
    return text, continuation, {
        "patch_applied_by_layer": applied,
        "patch_audit": "PASS",
    }


def _head_delta(
    clean: Mapping[tuple[int, int], torch.Tensor],
    counterfactual: Mapping[tuple[int, int], torch.Tensor],
    heads: Sequence[tuple[int, int]],
) -> dict[str, float]:
    left = torch.cat([clean[head].detach().float().cpu() for head in heads])
    right = torch.cat(
        [counterfactual[head].detach().float().cpu() for head in heads]
    )
    cosine = torch.nn.functional.cosine_similarity(left[None], right[None]).item()
    return {
        "head_output_clean_norm": float(left.norm().item()),
        "head_output_counterfactual_norm": float(right.norm().item()),
        "head_output_delta_norm": float((right - left).norm().item()),
        "head_output_cosine": float(cosine),
    }


def _mentions(text: str, value: int) -> bool:
    return re.search(rf"(?<!\d){int(value)}(?!\d)", text) is not None


def run(args: argparse.Namespace) -> None:
    selected_heads, random_heads, head_audit = frozen_heads(
        args.plan, args.model, args.query_variant
    )
    rows = [
        row
        for row in read_jsonl(args.generations)
        if strict_discovery_n10(row, args.model)
    ]
    if not rows:
        raise RuntimeError("No correct strict one-to-one discovery N10 row")
    spec = resolve_model_spec(args.model)
    model, tokenizer, adapter = load_registered_model(
        spec,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )
    row = rows[0]
    queries, exclusions = pre_city_token_queries(row, tokenizer)
    candidates = [q for q in queries if q.query_variant == args.query_variant]
    if args.occurrence is not None:
        candidates = [q for q in candidates if q.occurrence == args.occurrence]
    chosen: tuple[PreCityQuery, dict[str, Any], dict[str, Any]] | None = None
    failures: list[dict[str, Any]] = []
    for query in candidates:
        try:
            cf, cf_audit = make_score_counterfactual(row, tokenizer, query)
            chosen = query, cf, cf_audit
            break
        except Exception as error:
            failures.append(
                {"occurrence": query.occurrence, "error": repr(error)}
            )
    if chosen is None:
        raise RuntimeError(f"No constructable query: {failures}")
    query, counterfactual, cf_audit = chosen
    clean_query_encoding = baseline_prefix_encoding(row, tokenizer, query)
    cf_query_encoding = baseline_prefix_encoding(counterfactual, tokenizer, query)
    if clean_query_encoding.sequence_length != cf_query_encoding.sequence_length:
        raise RuntimeError("Counterfactual shifted the mechanism query")
    patch_position = int(clean_query_encoding.query_position)
    union_heads = sorted(set(selected_heads) | set(random_heads))
    clean_states, clean_capture_audit = capture_head_outputs(
        model, adapter, clean_query_encoding, union_heads, patch_position
    )
    cf_states, cf_capture_audit = capture_head_outputs(
        model, adapter, cf_query_encoding, union_heads, patch_position
    )

    score_start = int(cf_audit["score_output_token_start"])
    clean_score_encoding = baseline_prefix_encoding(
        row, tokenizer, query, prefix_output_token_count=score_start
    )
    cf_score_encoding = baseline_prefix_encoding(
        counterfactual,
        tokenizer,
        query,
        prefix_output_token_count=score_start,
    )
    original_token = int(cf_audit["original_score_first_token_id"])
    cf_token = int(cf_audit["counterfactual_score_first_token_id"])
    conditions = [
        ("clean_unpatched", clean_query_encoding, clean_score_encoding, None, []),
        ("counterfactual_unpatched", cf_query_encoding, cf_score_encoding, None, []),
        (
            "clean_from_counterfactual_selected",
            clean_query_encoding,
            clean_score_encoding,
            cf_states,
            selected_heads,
        ),
        (
            "counterfactual_from_clean_selected",
            cf_query_encoding,
            cf_score_encoding,
            clean_states,
            selected_heads,
        ),
        (
            "clean_from_counterfactual_random",
            clean_query_encoding,
            clean_score_encoding,
            cf_states,
            random_heads,
        ),
        (
            "counterfactual_from_clean_random",
            cf_query_encoding,
            cf_score_encoding,
            clean_states,
            random_heads,
        ),
    ]
    output_rows: list[dict[str, Any]] = []
    for index, (condition, marker_encoding, score_encoding, donor, heads) in enumerate(
        conditions, start=1
    ):
        logits, logit_audit = score_logits(
            model,
            adapter,
            score_encoding,
            patch_position=patch_position,
            donor=donor,
            heads=heads,
        )
        marker_text, marker_ids, generation_audit = generate_marker(
            model,
            tokenizer,
            adapter,
            marker_encoding,
            patch_position=patch_position,
            donor=donor,
            heads=heads,
            max_new_tokens=args.max_new_tokens,
        )
        original_logit = float(logits[original_token].item())
        counterfactual_logit = float(logits[cf_token].item())
        output_rows.append(
            {
                "schema_version": SCHEMA,
                "model_label": args.model,
                "request_id": row["request_id"],
                "seed": int(row["seed"]),
                "split": row["split"],
                "query_variant": query.query_variant,
                "occurrence": int(query.occurrence),
                "target_city": query.city,
                "condition": condition,
                "heads": [list(head) for head in heads],
                "original_score": int(cf_audit["original_score"]),
                "counterfactual_score": int(cf_audit["counterfactual_score"]),
                "original_score_first_token_logit": original_logit,
                "counterfactual_score_first_token_logit": counterfactual_logit,
                "counterfactual_minus_original_logit": (
                    counterfactual_logit - original_logit
                ),
                "actual_marker_continuation": marker_text,
                "actual_marker_token_ids": list(marker_ids),
                "actual_marker_mentions_city": query.city.casefold()
                in marker_text.casefold(),
                "actual_marker_mentions_original_score": _mentions(
                    marker_text, int(cf_audit["original_score"])
                ),
                "actual_marker_mentions_counterfactual_score": _mentions(
                    marker_text, int(cf_audit["counterfactual_score"])
                ),
                **logit_audit,
                "generation_patch_audit": generation_audit,
            }
        )
        print(
            f"[{args.model}] {index}/{len(conditions)} {condition} "
            f"margin={counterfactual_logit - original_logit:.4f} "
            f"marker={marker_text!r}",
            flush=True,
        )
    by_condition = {row["condition"]: row for row in output_rows}
    clean_margin = by_condition["clean_unpatched"][
        "counterfactual_minus_original_logit"
    ]
    cf_margin = by_condition["counterfactual_unpatched"][
        "counterfactual_minus_original_logit"
    ]
    clean_selected = by_condition["clean_from_counterfactual_selected"][
        "counterfactual_minus_original_logit"
    ]
    cf_selected = by_condition["counterfactual_from_clean_selected"][
        "counterfactual_minus_original_logit"
    ]
    clean_random = by_condition["clean_from_counterfactual_random"][
        "counterfactual_minus_original_logit"
    ]
    cf_random = by_condition["counterfactual_from_clean_random"][
        "counterfactual_minus_original_logit"
    ]
    summary = {
        "schema_version": SCHEMA,
        "model_label": args.model,
        "request_id": row["request_id"],
        "query_variant": query.query_variant,
        "occurrence": int(query.occurrence),
        "counterfactual_audit": cf_audit,
        "frozen_head_audit": head_audit,
        "clean_capture_audit": clean_capture_audit,
        "counterfactual_capture_audit": cf_capture_audit,
        "selected_head_content_delta": _head_delta(
            clean_states, cf_states, selected_heads
        ),
        "random_head_content_delta": _head_delta(
            clean_states, cf_states, random_heads
        ),
        "unpatched_prompt_edit_effect_on_score_margin": cf_margin - clean_margin,
        "selected_clean_receiver_margin_shift": clean_selected - clean_margin,
        "selected_counterfactual_receiver_margin_shift": cf_selected - cf_margin,
        "random_clean_receiver_margin_shift": clean_random - clean_margin,
        "random_counterfactual_receiver_margin_shift": cf_random - cf_margin,
        "bidirectional_selected_directional": bool(
            clean_selected > clean_margin and cf_selected < cf_margin
        ),
        "bidirectional_random_directional": bool(
            clean_random > clean_margin and cf_random < cf_margin
        ),
        "construction_failures_before_selected_query": failures,
        "pre_city_query_exclusions": exclusions,
    }
    write_jsonl(args.output / "trials.jsonl", output_rows)
    write_json(args.output / "summary.json", summary)
    write_json(
        args.output / "audit.json",
        {
            "passed": True,
            "schema_version": SCHEMA,
            "discovery_only": True,
            "strict_one_to_one_n10_correct": True,
            "counterfactual_prompt_length_matched": True,
            "counterfactual_change_inside_one_target_record": True,
            "head_bank_frozen_on_discovery": True,
            "selected_and_control_disjoint_layer_matched": True,
            "actual_marker_generation_not_argmax_proxy": True,
            "outputs_isolated_from_original_v5": True,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["Qwen3-8B", "Gemma4-E4B"])
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-variant", default="pre_city_d1")
    parser.add_argument("--occurrence", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
