"""Counterfactual running-count restoration from marker-scrubbed lists.

The experiment deliberately makes a narrow sufficiency claim.  Source traces
were generated naturally under the frozen V5 prompt.  A trace is eligible when
its raw output contains one complete ten-record bullet or indexed-list episode.
At test time, prompt records, non-item reasoning, and explicit within-item
progress markers are replaced without changing token positions.  A source
item-k block-input state is then patched once into a receiver in which all
visible list items are also replaced.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from realistic_niah_v4.modeling import DecoderAdapter, _encoding_tensors
from realistic_niah_v4.prompts import TokenSpan

from .causal_sites import build_output_token_map
from .count_stream import (
    AnswerSourceRegistry,
    _prefix_forward,
    _score_and_generate_prefill,
    _sha256_json,
    build_answer_source_registry,
)
from .encoding import NativeTraceEncoding
from .indexed_counter_patch import (
    build_minimal_item_early_stop_encoding,
    capture_decoder_block_input_states,
    minimal_terminal_suffix_token_ids,
    prefill_with_single_decoder_block_input_replacement,
)


BULLET_COUNTER_RESTORE_SCHEMA_VERSION = (
    "realistic_niah_v5_marker_scrubbed_list_counterfactual_restore_v2"
)
_TOTAL_QUERY_RE = re.compile(r"Total\s*:\s*", re.IGNORECASE)
_LEADING_ITEM_INDEX_RE = re.compile(
    r"^\s*-\s*(?:#\s*)?(?:[0-9]+|one|two|three|four|five|six|seven|"
    r"eight|nine|ten)\s*[.)\]:-]",
    re.IGNORECASE,
)
_ORDINAL_ITEM_RE = re.compile(
    r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|"
    r"tenth)\s+(?:record|match|item|entry|city)\b",
    re.IGNORECASE,
)
_SAFE_TOKEN_EXCLUSION_RE = re.compile(
    r"[0-9]|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"count|total|subtotal|record|records|match|matches|item|items|city|"
    r"cities|score|scores|audit|number|ordinal)\b",
    re.IGNORECASE,
)
_COUNT_NEUTRAL_BANNED_SUBSTRINGS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "count",
    "total",
    "subtotal",
    "record",
    "match",
    "item",
    "city",
    "cities",
    "score",
    "audit",
    "number",
    "ordinal",
)
_PREBULLET_EXPLICIT_COUNT_RE = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|[0-9]+)\b|"
    r"\b(?:counts?|totals?|subtotals?|records?|matches?|items?|cities|city)\b",
    re.IGNORECASE,
)
_LIST_ITEM_PREFIX_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?:(?P<bullet>[-+*])(?:[ \t]+|$)|"
    r"(?P<index>10|[1-9])[.)](?:[ \t]+|$))"
)
_RUNNING_VALUE_RE = re.compile(
    r"(?<![0-9])(?:10|[1-9])(?![0-9])|"
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|"
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b",
    re.IGNORECASE,
)
_EXPLICIT_PROGRESS_PHRASE_RE = re.compile(
    r"\b(?:count|record|scan|excerpt|item|entry|match)\s*"
    r"(?:[:=#.()\-]+\s*)?"
    r"(?:10|[1-9]|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b|"
    r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)"
    r"\s+(?:record|scan|excerpt|item|entry|match)\b",
    re.IGNORECASE,
)


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _span_positions(spans: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    return tuple(
        position
        for start, end in spans
        for position in range(int(start), int(end))
    )


def _literal_present(text: str, literal: str) -> bool:
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9]){re.escape(str(literal))}(?![A-Za-z0-9])",
            text,
            flags=re.IGNORECASE,
        )
    )


def _raw_lines(raw: str) -> list[tuple[int, int, str]]:
    lines: list[tuple[int, int, str]] = []
    cursor = 0
    for value in raw.splitlines(keepends=True):
        content = value.rstrip("\r\n")
        lines.append((cursor, cursor + len(content), content))
        cursor += len(value)
    if cursor < len(raw):
        lines.append((cursor, len(raw), raw[cursor:]))
    return lines


def _marker_char_spans(text: str, *, offset: int) -> list[list[int]]:
    spans = {
        (int(offset) + int(match.start()), int(offset) + int(match.end()))
        for pattern in (_EXPLICIT_PROGRESS_PHRASE_RE, _RUNNING_VALUE_RE)
        for match in pattern.finditer(text)
    }
    return [list(value) for value in sorted(spans)]


def audit_complete_marker_scrubbable_list(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Outcome-blind raw-text gate for one complete N=10 list episode.

    The gate accepts bullet and indexed list lines.  Each of exactly ten item
    lines must mention one distinct gold city and its registered score.  It
    never reads the model's final numeric answer.  Explicit item indices are
    registered for same-position token replacement rather than treated as an
    exclusion criterion.
    """

    raw = str(row.get("raw_output_text", ""))
    parser = row.get("trace_parse", {}).get("parser", {})
    gold = tuple(row.get("gold_records", ()))
    reasoning_start = int(parser.get("reasoning_start_char", 0))
    reasoning_end = int(parser.get("reasoning_end_char", -1))
    reasons: list[str] = []
    candidates: list[dict[str, Any]] = []
    all_lines = _raw_lines(raw)

    if int(row.get("gold_count", -1)) != 10 or len(gold) != 10:
        reasons.append("source_is_not_N10")
    if bool(row.get("generation_truncated", False)):
        reasons.append("generation_truncated")
    if not 0 <= reasoning_start < reasoning_end <= len(raw):
        reasons.append("reasoning_bounds_unresolved")

    for line_number, (start, end, text) in enumerate(all_lines, start=1):
        if start < reasoning_start or end > reasoning_end or end <= start:
            continue
        prefix = _LIST_ITEM_PREFIX_RE.match(text)
        if prefix is None:
            continue
        matched: list[tuple[int, Mapping[str, Any]]] = []
        for gold_index, record in enumerate(gold, start=1):
            city = str(record.get("city", ""))
            score = str(int(record.get("score", -1)))
            if _literal_present(text, city) and _literal_present(text, score):
                matched.append((gold_index, record))
        if len(matched) != 1:
            continue
        gold_index, record = matched[0]
        candidates.append(
            {
                "occurrence": len(candidates) + 1,
                "gold_index": int(gold_index),
                "city": str(record.get("city", "")),
                "score": int(record.get("score", -1)),
                "style": "bullet" if prefix.group("bullet") else "indexed",
                "line_number": int(line_number),
                "char_start": int(start),
                "char_end": int(end),
                "text_sha256": _hash_text(text),
                "marker_char_spans": _marker_char_spans(text, offset=start),
            }
        )

    if len(candidates) != 10:
        reasons.append(f"list_item_candidate_count_not_10:{len(candidates)}")
    gold_indices = [int(value["gold_index"]) for value in candidates]
    if len(candidates) == 10 and set(gold_indices) != set(range(1, 11)):
        reasons.append("list_gold_coverage_not_exact")
    styles = {str(value["style"]) for value in candidates}
    if len(candidates) == 10 and len(styles) != 1:
        reasons.append("mixed_list_item_styles")

    if candidates:
        first_line = int(candidates[0]["line_number"])
        last_line = int(candidates[-1]["line_number"])
        item_lines = {int(value["line_number"]) for value in candidates}
        for line_number in range(first_line, last_line + 1):
            if line_number in item_lines:
                continue
            bridge = all_lines[line_number - 1][2].strip()
            if bridge and _LIST_ITEM_PREFIX_RE.match(bridge) is None:
                reasons.append(f"nonstructural_interitem_bridge:{line_number}")

    starts = [int(value["char_start"]) for value in candidates]
    ends = [int(value["char_end"]) for value in candidates]
    if any(right_start < left_end for left_end, right_start in zip(ends, starts[1:])):
        reasons.append("list_item_char_spans_overlap")
    first_start = starts[0] if starts else reasoning_start
    last_end = ends[-1] if ends else reasoning_start
    prelist = raw[reasoning_start:first_start] if 0 <= reasoning_start <= first_start else ""
    tail = raw[last_end:] if 0 <= last_end <= len(raw) else ""
    if reasoning_end < last_end or reasoning_end > len(raw):
        reasons.append("reasoning_close_after_list_unresolved")
    if reasoning_end >= 0 and not raw.startswith(("</think>", "<channel|>"), reasoning_end):
        reasons.append("native_reasoning_close_missing")
    if _TOTAL_QUERY_RE.search(raw, max(reasoning_end, last_end)) is None:
        reasons.append("native_Total_query_missing")

    return {
        "schema_version": "marker_scrubbable_list_raw_audit_v2",
        "status": "PASS" if not reasons else "FAIL",
        "eligible": not reasons,
        "reasons": reasons,
        "gold_count": int(row.get("gold_count", -1)),
        "parsed_item_count": len(candidates),
        "marker_kind": next(iter(styles)) if len(styles) == 1 else "mixed",
        "original_parser_marker_kind": str(parser.get("marker_kind", "")),
        "trace_one_to_one": bool(
            len(candidates) == 10 and set(gold_indices) == set(range(1, 11))
        ),
        "list_episode_contiguous": not any(
            value.startswith("nonstructural_interitem_bridge:") for value in reasons
        ),
        "item_char_spans": [[start, end] for start, end in zip(starts, ends)],
        "item_marker_char_spans": [
            list(value["marker_char_spans"]) for value in candidates
        ],
        "item_line_numbers": [int(value["line_number"]) for value in candidates],
        "item_gold_indices": gold_indices,
        "item_gold_cities": [str(value["city"]) for value in candidates],
        "item_text_sha256": [str(value["text_sha256"]) for value in candidates],
        "prebullet_text_sha256": _hash_text(prelist),
        "prebullet_contains_explicit_count_or_record_language": bool(
            _PREBULLET_EXPLICIT_COUNT_RE.search(prelist)
        ),
        "postbullet_tail_sha256": _hash_text(tail),
        "eligibility_uses_final_answer": False,
        "final_answer_correctness_accessed": False,
        "prompt_modified_during_generation": False,
    }


def audit_complete_contiguous_unnumbered_bullets(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Compatibility alias for the v2 marker-scrubbable list gate."""

    return audit_complete_marker_scrubbable_list(row)


def _positions_to_spans(positions: Sequence[int]) -> tuple[tuple[int, int], ...]:
    values = sorted({int(value) for value in positions})
    if not values:
        return ()
    result: list[tuple[int, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        result.append((start, previous + 1))
        start = previous = value
    result.append((start, previous + 1))
    return tuple(result)


def build_marker_scrubbed_list_registry(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    answer_site_id: str = "answer_query_v3",
    trace_audit: Mapping[str, Any] | None = None,
) -> tuple[NativeTraceEncoding, AnswerSourceRegistry, dict[str, Any]]:
    """Replace the legacy parser spans with the frozen v2 list-item spans."""

    audit = dict(trace_audit or audit_complete_marker_scrubbable_list(row))
    if not audit.get("eligible"):
        raise ValueError(f"Marker-scrubbable list audit failed: {audit.get('reasons')}")
    encoding, legacy = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    token_map = build_output_token_map(row, tokenizer)
    prompt_count = int(encoding.prompt_token_count)
    item_spans: list[tuple[int, int]] = []
    mapping_audit: list[dict[str, Any]] = []
    for occurrence, (char_start, char_end) in enumerate(
        audit["item_char_spans"], start=1
    ):
        mapped = token_map.span(
            f"marker_scrubbed_list_item:{occurrence}", int(char_start), int(char_end)
        )
        if mapped.get("status") != "ok":
            raise ValueError(f"Cannot token-map list item {occurrence}: {mapped}")
        start = prompt_count + int(mapped["output_token_start"])
        end = prompt_count + int(mapped["output_token_end"])
        if not prompt_count <= start < end <= int(encoding.query_position):
            raise ValueError(f"List item {occurrence} crosses prompt/query bounds")
        if item_spans and start < item_spans[-1][1]:
            raise ValueError("Token-mapped list items overlap")
        item_spans.append((start, end))
        mapping_audit.append(
            {
                "occurrence": occurrence,
                "token_start": start,
                "token_end": end,
                "left_spill_chars": int(mapped.get("left_spill_chars", 0)),
                "right_spill_chars": int(mapped.get("right_spill_chars", 0)),
            }
        )

    item_positions = set(_span_positions(item_spans))
    marker_positions: set[int] = set()
    for occurrence, spans in enumerate(audit["item_marker_char_spans"], start=1):
        allowed = set(range(*item_spans[occurrence - 1]))
        for marker_index, (char_start, char_end) in enumerate(spans, start=1):
            mapped = token_map.span(
                f"list_item_marker:{occurrence}:{marker_index}",
                int(char_start),
                int(char_end),
            )
            if mapped.get("status") != "ok":
                raise ValueError(
                    f"Cannot token-map item marker {occurrence}:{marker_index}"
                )
            positions = set(
                range(
                    prompt_count + int(mapped["output_token_start"]),
                    prompt_count + int(mapped["output_token_end"]),
                )
            )
            marker_positions.update(positions & allowed)

    trace_context_positions = set(
        range(int(legacy.prompt_token_count), int(legacy.query_position))
    )
    trace_other_positions = trace_context_positions - item_positions
    nonmarker_positions = item_positions - marker_positions
    spans = tuple(
        TokenSpan(
            slot_index=occurrence,
            start=start,
            end=end,
            active=True,
            kind="marker_scrubbed_list_item",
            canonical_length=end - start,
            model_token_length=end - start,
        )
        for occurrence, (start, end) in enumerate(item_spans, start=1)
    )
    encoding = replace(
        encoding,
        trace_item_spans=spans,
        slot_spans=spans,
        needle_spans=spans,
    )
    registry = replace(
        legacy,
        trace_items=tuple(item_spans),
        trace_other=_positions_to_spans(tuple(trace_other_positions)),
        trace_markers=_positions_to_spans(tuple(marker_positions)),
        trace_nonmarkers=_positions_to_spans(tuple(nonmarker_positions)),
        earlier_trace_items=tuple(item_spans[:-1]),
        terminal_trace_item=(tuple(item_spans[-1]),),
    )
    registry.validate()
    return encoding, registry, {
        "list_item_token_mapping": mapping_audit,
        "list_item_token_count": len(item_positions),
        "explicit_item_marker_token_count": len(marker_positions),
        "legacy_parser_trace_item_count": len(legacy.trace_items),
        "v2_registry_replaces_legacy_parser_items": True,
    }


def _safe_ordinary_prompt_token_pool(
    encoding: NativeTraceEncoding,
    registry: Any,
    tokenizer: Any,
) -> tuple[int, ...]:
    original = tuple(int(value) for value in encoding.input_ids)
    forbidden_positions = set(_span_positions(registry.prompt_records))
    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", ())}
    pool: list[int] = []
    for position in range(1, int(registry.prompt_token_count)):
        if position in forbidden_positions:
            continue
        token_id = original[position]
        if token_id in special_ids:
            continue
        text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        lowered = text.lower()
        if (
            not text
            or _SAFE_TOKEN_EXCLUSION_RE.search(text)
            or any(value in lowered for value in _COUNT_NEUTRAL_BANNED_SUBSTRINGS)
        ):
            continue
        if not any(character.isalpha() for character in text):
            continue
        pool.append(token_id)
    if len(pool) < 64:
        raise ValueError(
            "Fewer than 64 count-neutral ordinary prompt tokens are available"
        )
    return tuple(pool)


def _replace_positions_from_pool(
    token_ids: Sequence[int],
    positions: Sequence[int],
    pool: Sequence[int],
    *,
    salt: str,
    tokenizer: Any,
) -> tuple[tuple[int, ...], int]:
    output = [int(value) for value in token_ids]
    changed = 0
    previous_position: int | None = None
    replacement_tail = ""
    for order, position in enumerate(int(value) for value in positions):
        if previous_position is None or position != previous_position + 1:
            replacement_tail = ""
        digest = hashlib.sha256(f"{salt}|{order}|{position}".encode("utf-8")).digest()
        initial = int.from_bytes(digest[:8], "big") % len(pool)
        replacement = -1
        replacement_text = ""
        for offset in range(len(pool)):
            candidate = int(pool[(initial + offset) % len(pool)])
            if candidate == output[position]:
                continue
            candidate_text = tokenizer.decode(
                [candidate],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            joined = (replacement_tail + candidate_text).lower()
            if any(
                value in joined for value in _COUNT_NEUTRAL_BANNED_SUBSTRINGS
            ) or any(character.isdigit() for character in joined):
                continue
            replacement = candidate
            replacement_text = candidate_text
            break
        if replacement < 0:
            raise RuntimeError(
                "Safe token pool cannot construct a count-neutral replacement"
            )
        output[position] = replacement
        replacement_tail = (replacement_tail + replacement_text)[-32:]
        previous_position = position
        changed += 1
    return tuple(output), changed


def build_scrubbed_source_and_blank(
    encoding: NativeTraceEncoding,
    registry: Any,
    tokenizer: Any,
    *,
    random_seed: int,
) -> tuple[NativeTraceEncoding, NativeTraceEncoding, dict[str, Any]]:
    """Build same-length Source and Blank prefixes without retokenization."""

    original = tuple(int(value) for value in encoding.input_ids)
    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", ())}
    prompt_positions = _span_positions(registry.prompt_records)
    first_bullet_start = int(registry.trace_items[0][0])
    prebullet_positions = tuple(
        position
        for position in range(int(registry.prompt_token_count), first_bullet_start)
        if original[position] not in special_ids
    )
    trace_other_spans = getattr(registry, "trace_other", ())
    trace_other_positions = tuple(
        position
        for position in _span_positions(trace_other_spans)
        if original[position] not in special_ids
    )
    marker_positions = tuple(
        position
        for position in _span_positions(getattr(registry, "trace_markers", ()))
        if original[position] not in special_ids
    )
    base_positions = tuple(
        sorted(
            set(prompt_positions)
            | set(prebullet_positions)
            | set(trace_other_positions)
            | set(marker_positions)
        )
    )
    bullet_positions = _span_positions(registry.trace_items)
    if not base_positions or not bullet_positions:
        raise ValueError("Source/Blank scrub positions are empty")
    if (set(base_positions) & set(bullet_positions)) != set(marker_positions):
        raise RuntimeError("Only registered item markers may overlap the base scrub")
    if any(original[position] in special_ids for position in prompt_positions):
        raise ValueError("A registered prompt record contains a special token")
    if any(original[position] in special_ids for position in bullet_positions):
        raise ValueError("A registered bullet span contains a special token")

    pool = _safe_ordinary_prompt_token_pool(encoding, registry, tokenizer)
    source_ids, source_changed = _replace_positions_from_pool(
        original,
        base_positions,
        pool,
        salt=f"{encoding.request_id}|source-base|{int(random_seed)}",
        tokenizer=tokenizer,
    )
    blank_ids, blank_changed = _replace_positions_from_pool(
        source_ids,
        bullet_positions,
        pool,
        salt=f"{encoding.request_id}|blank-bullets|{int(random_seed)}",
        tokenizer=tokenizer,
    )
    source = replace(encoding, input_ids=source_ids)
    blank = replace(encoding, input_ids=blank_ids)
    if not (
        len(source.input_ids) == len(blank.input_ids) == len(original)
        and source.query_position == blank.query_position == encoding.query_position
    ):
        raise RuntimeError("Source/Blank token geometry changed")
    unchanged_positions = set(range(len(original))) - set(base_positions) - set(
        bullet_positions
    )
    if any(
        source.input_ids[position] != original[position]
        or blank.input_ids[position] != original[position]
        for position in unchanged_positions
    ):
        raise RuntimeError("Source/Blank changed an unregistered position")
    nonmarker_positions = set(bullet_positions) - set(marker_positions)
    if any(source.input_ids[position] != original[position] for position in nonmarker_positions):
        raise RuntimeError("Source changed non-marker list-item content")
    if any(source.input_ids[position] == original[position] for position in marker_positions):
        raise RuntimeError("Source failed to replace an explicit item marker")
    nonitem_base_positions = set(base_positions) - set(marker_positions)
    if any(
        source.input_ids[position] != blank.input_ids[position]
        for position in nonitem_base_positions
    ):
        raise RuntimeError("Source/Blank do not share the identical non-item base scrub")
    decoded_source_items: list[str] = []
    for start, end in registry.trace_items:
        decoded = tokenizer.decode(
            list(source.input_ids[int(start) : int(end)]),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        decoded_source_items.append(str(decoded))
    residual_markers = [
        occurrence
        for occurrence, text in enumerate(decoded_source_items, start=1)
        if _RUNNING_VALUE_RE.search(text)
    ]
    if residual_markers:
        raise RuntimeError(
            "Source item retains an explicit running value after marker scrub: "
            f"items={residual_markers}"
        )

    audit = {
        "scrub_method": "same_position_count_neutral_ordinary_prompt_token_replacement",
        "retokenization_used": False,
        "token_deletion_used": False,
        "sequence_length_preserved": True,
        "special_tokens_preserved": True,
        "prompt_record_token_count": len(prompt_positions),
        "prebullet_reasoning_token_count": len(prebullet_positions),
        "trace_other_scrub_token_count": len(trace_other_positions),
        "explicit_item_marker_scrub_token_count": len(marker_positions),
        "base_scrub_token_count": len(base_positions),
        "blank_bullet_token_count": len(bullet_positions),
        "source_changed_token_count": source_changed,
        "blank_additional_changed_token_count": blank_changed,
        "ordinary_safe_token_pool_size": len(pool),
        "replacement_contains_candidate_digit": False,
        "replacement_contains_count_or_record_substring": False,
        "base_scrub_positions_sha256": _sha256_json(base_positions),
        "blank_bullet_positions_sha256": _sha256_json(bullet_positions),
        "source_input_ids_sha256": _sha256_json(source.input_ids),
        "blank_input_ids_sha256": _sha256_json(blank.input_ids),
        "source_item_text_sha256": [
            _hash_text(value) for value in decoded_source_items
        ],
        "source_items_contain_explicit_running_value": False,
        "source_item_nonmarkers_preserved": True,
        "source_explicit_item_markers_scrubbed": True,
        "all_nonitem_reasoning_scrubbed": bool(trace_other_spans),
        "source_blank_base_scrub_identical": True,
    }
    return source, blank, audit


def running_target_metrics(outcomes: Mapping[str, Any], *, target_k: int) -> dict[str, Any]:
    """Re-express candidate scores relative to the early-stop target k."""

    target = int(target_k)
    if not 1 <= target <= 10:
        raise ValueError("target_k must lie in 1..10")
    counts = tuple(int(value) for value in str(outcomes["candidate_counts"]).split(","))
    scores = np.asarray(
        [float(value) for value in str(outcomes["candidate_log_scores"]).split(",")],
        dtype=float,
    )
    probabilities = np.asarray(
        [float(value) for value in str(outcomes["candidate_probabilities"]).split(",")],
        dtype=float,
    )
    if counts != tuple(range(1, 11)) or scores.shape != (10,) or probabilities.shape != (10,):
        raise ValueError("Candidate outcomes do not contain aligned counts 1..10")
    index = target - 1
    rival = np.delete(scores, index)
    predicted = int(counts[int(np.argmax(scores))])
    return {
        "running_target_k": target,
        "running_target_log_score": float(scores[index]),
        "running_target_margin": float(scores[index] - np.max(rival)),
        "running_target_probability": float(probabilities[index]),
        "predicted_running_count": predicted,
        "running_target_exact": bool(predicted == target),
    }


@torch.inference_mode()
def _score_encoding(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    target_k: int,
) -> dict[str, Any]:
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    prefill = _prefix_forward(model, adapter, input_ids, attention_mask)
    outcomes = _score_and_generate_prefill(
        model,
        tokenizer,
        encoding,
        prefill,
        run_greedy=False,
        max_new_tokens=1,
    )
    return {**outcomes, **running_target_metrics(outcomes, target_k=int(target_k))}


@torch.inference_mode()
def run_bullet_counterfactual_restore_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    source_layers: Sequence[int],
    target_occurrences: Sequence[int] = tuple(range(1, 11)),
    random_seed: int,
    answer_site_id: str = "answer_query_v3",
) -> list[dict[str, Any]]:
    """Run Source, Blank, and bullet-k-to-Blank restoration conditions."""

    trace_audit = audit_complete_marker_scrubbable_list(row)
    if not trace_audit["eligible"]:
        raise ValueError(f"Marker-scrubbable list audit failed: {trace_audit['reasons']}")
    clean_full, registry, registry_audit = build_marker_scrubbed_list_registry(
        row,
        tokenizer,
        answer_site_id=answer_site_id,
        trace_audit=trace_audit,
    )
    if len(registry.trace_items) != 10:
        raise ValueError("List-counter experiment requires ten token-mapped items")
    source_full, blank_full, scrub_audit = build_scrubbed_source_and_blank(
        clean_full,
        registry,
        tokenizer,
        random_seed=int(random_seed),
    )
    suffix_ids, suffix_audit = minimal_terminal_suffix_token_ids(row, tokenizer)
    if int(suffix_ids[-1]) != int(clean_full.input_ids[registry.query_position]):
        raise ValueError("Minimal native suffix changed the Total query token")

    layers = tuple(sorted({int(value) for value in source_layers}))
    targets = tuple(sorted({int(value) for value in target_occurrences}))
    if not layers or layers[0] < 0 or layers[-1] >= int(adapter.num_layers):
        raise ValueError("Source layer registry is invalid")
    if not targets or min(targets) < 1 or max(targets) > 10:
        raise ValueError("Target occurrences must be a nonempty subset of 1..10")

    results: list[dict[str, Any]] = []
    for occurrence in targets:
        source, source_early_audit = build_minimal_item_early_stop_encoding(
            source_full,
            registry,
            target_occurrence=occurrence,
            terminal_suffix_token_ids=suffix_ids,
        )
        blank, blank_early_audit = build_minimal_item_early_stop_encoding(
            blank_full,
            registry,
            target_occurrence=occurrence,
            terminal_suffix_token_ids=suffix_ids,
        )
        if source_early_audit != blank_early_audit:
            raise RuntimeError("Source/Blank early-stop geometry differs")
        if source.query_position != blank.query_position or source.input_ids[
            source.query_position
        ] != blank.input_ids[blank.query_position]:
            raise RuntimeError("Source/Blank Total queries are misaligned")
        start, end = registry.trace_items[occurrence - 1]
        positions = tuple(range(int(start), int(end)))
        source_capture = capture_decoder_block_input_states(
            model,
            adapter,
            source,
            positions,
            layers=layers,
        )
        source_outcomes = _score_encoding(
            model, tokenizer, adapter, source, target_k=occurrence
        )
        blank_outcomes = _score_encoding(
            model, tokenizer, adapter, blank, target_k=occurrence
        )
        common = {
            "schema_version": BULLET_COUNTER_RESTORE_SCHEMA_VERSION,
            "experiment_id": "marker_scrubbed_list_counterfactual_sufficiency",
            "request_id": str(source.request_id),
            "model_label": str(source.model_label),
            "seed": int(source.seed),
            "dataset_split": str(source.split),
            "source_gold_count": int(source.count),
            "target_occurrence": int(occurrence),
            "answer_site_id": answer_site_id,
            "registered_target_occurrences": list(targets),
            "patch_geometry": "complete_source_list_item_k_same_token_positions",
            "patch_site": "decoder_block_input",
            "patch_layer_mode": "single_decoder_block_input_once",
            "upper_layers_recomputed_after_patch": True,
            "source_base_scrubbed_before_state_capture": True,
            "blank_visible_list_items_replaced": int(occurrence),
            "blank_earlier_list_items_remain_blank_during_restoration": True,
            "future_list_items_removed": 10 - int(occurrence),
            "readout_mode": "immediate_list_item_k_minimal_native_Total_query",
            "diagnostic_suffix_used": False,
            "candidate_scoring": "joint_sequence_log_probability_counts_1_through_10",
            "selection_uses_final_answer": False,
            "claim_scope": "counterfactual_sufficiency_after_all_explicit_progress_cues_scrubbed",
            "natural_no_index_counter_formation_claim_allowed": False,
            "necessity_claim_allowed": False,
            "outcome_blind": True,
            "registry_sha256": registry.to_dict()["registry_sha256"],
            **trace_audit,
            **registry_audit,
            **scrub_audit,
            **suffix_audit,
            **source_early_audit,
        }
        results.extend(
            [
                {
                    **common,
                    "condition": "source_reference",
                    "source_layer": -1,
                    "patch_token_count": 0,
                    **source_outcomes,
                },
                {
                    **common,
                    "condition": "blank_reference",
                    "source_layer": -1,
                    "patch_token_count": 0,
                    **blank_outcomes,
                },
            ]
        )
        for layer in layers:
            prefill, applications, realized_norm = (
                prefill_with_single_decoder_block_input_replacement(
                    model,
                    adapter,
                    blank,
                    positions=positions,
                    layer=int(layer),
                    replacement_states=source_capture[int(layer)],
                )
            )
            outcomes = _score_and_generate_prefill(
                model,
                tokenizer,
                blank,
                prefill,
                run_greedy=False,
                max_new_tokens=1,
            )
            outcomes.update(running_target_metrics(outcomes, target_k=occurrence))
            results.append(
                {
                    **common,
                    "condition": "source_list_item_k_to_blank_restoration",
                    "source_layer": int(layer),
                    "patch_layers": [int(layer)],
                    "patch_layer_count": 1,
                    "patch_token_count": len(positions),
                    "patch_positions_sha256": _sha256_json(positions),
                    "donor_receiver_positions_identical": True,
                    "donor_receiver_span_lengths_equal": True,
                    "patch_hook_applications": {str(layer): int(applications)},
                    "patch_realized_fro_norm_by_layer": {
                        str(layer): float(realized_norm)
                    },
                    **outcomes,
                }
            )
    return results
