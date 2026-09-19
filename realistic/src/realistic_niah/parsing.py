from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .spec import ENUMERATION_PROMPT_MODES, THINKING_PROMPT_MODES

OPTIONAL_END_TOKEN = (
    r"(?:<\|im_end\|>|<turn\|>|<\|endoftext\|>|"
    r"<｜end▁of▁sentence｜>)?"
)
TOTAL_RE = re.compile(
    rf"(?im)^\s*Total\s*:\s*(-?\d+)\s*{OPTIONAL_END_TOKEN}\s*$"
)
TOTAL_ONLY_RE = re.compile(
    rf"(?is)^\s*Total\s*:\s*(-?\d+)\s*{OPTIONAL_END_TOKEN}\s*$"
)
INDEXED_ENUMERATION_RE = re.compile(
    r"(?m)^\s*(\d+)\.\s*(.+?)\s*:\s*(-?\d+)\s*$"
)
BULLET_ENUMERATION_RE = re.compile(
    r"(?m)^\s*-\s*(.+?)\s*:\s*(-?\d+)\s*$"
)
REASONING_INDEX_RE = re.compile(r"(?m)^\s*(\d+)[.)]\s+")
QWEN_THINK_OPEN = "<think>"
QWEN_THINK_CLOSE = "</think>"
GEMMA_THINK_OPEN = "<|channel>thought\n"
GEMMA_THINK_CLOSE = "<channel|>"
MISTRAL_THINK_OPEN = "[THINK]"
MISTRAL_THINK_CLOSE = "[/THINK]"
MISTRAL_SPECIAL_THINK_OPEN = "<SPECIAL_34>"
MISTRAL_SPECIAL_THINK_CLOSE = "<SPECIAL_35>"
RESPONSE_WRAPPERS = (
    ("<response>", "</response>"),
    ("[RESPONSE]", "[/RESPONSE]"),
)


def _strip_response_wrapper(text: str) -> str:
    stripped = text.strip()
    for opening, closing in RESPONSE_WRAPPERS:
        if stripped.startswith(opening) and stripped.endswith(closing):
            return stripped[len(opening) : -len(closing)].strip()
    return stripped


def split_reasoning_and_final(
    raw_text: str,
    *,
    prompt_mode: str,
    reasoning_expected: bool | None = None,
) -> tuple[str, str]:
    expects_reasoning = (
        prompt_mode in THINKING_PROMPT_MODES
        if reasoning_expected is None
        else reasoning_expected
    )
    has_reasoning_delimiter = any(
        delimiter in raw_text
        for delimiter in (
            QWEN_THINK_OPEN,
            QWEN_THINK_CLOSE,
            GEMMA_THINK_OPEN,
            GEMMA_THINK_CLOSE,
            MISTRAL_THINK_OPEN,
            MISTRAL_THINK_CLOSE,
            MISTRAL_SPECIAL_THINK_OPEN,
            MISTRAL_SPECIAL_THINK_CLOSE,
        )
    )
    if not expects_reasoning and not has_reasoning_delimiter:
        return "", _strip_response_wrapper(raw_text)

    if QWEN_THINK_OPEN in raw_text:
        after_open = raw_text.split(QWEN_THINK_OPEN, 1)[1]
        if QWEN_THINK_CLOSE not in after_open:
            return after_open.strip(), ""
        reasoning, final = after_open.rsplit(QWEN_THINK_CLOSE, 1)
        return reasoning.strip(), _strip_response_wrapper(final)
    if QWEN_THINK_CLOSE in raw_text:
        # Qwen's chat template normally places the opening <think> token in
        # the prompt. Offline generation therefore returns only the reasoning
        # body, the closing token, and the final answer.
        reasoning, final = raw_text.rsplit(QWEN_THINK_CLOSE, 1)
        return reasoning.strip(), _strip_response_wrapper(final)

    if GEMMA_THINK_OPEN in raw_text:
        after_open = raw_text.split(GEMMA_THINK_OPEN, 1)[1]
        if GEMMA_THINK_CLOSE not in after_open:
            return after_open.strip(), ""
        reasoning, final = after_open.split(GEMMA_THINK_CLOSE, 1)
        return reasoning.strip(), _strip_response_wrapper(final)
    if GEMMA_THINK_CLOSE in raw_text:
        reasoning, final = raw_text.split(GEMMA_THINK_CLOSE, 1)
        return reasoning.strip(), _strip_response_wrapper(final)

    for opening, closing in (
        (MISTRAL_THINK_OPEN, MISTRAL_THINK_CLOSE),
        (MISTRAL_SPECIAL_THINK_OPEN, MISTRAL_SPECIAL_THINK_CLOSE),
    ):
        if opening in raw_text:
            after_open = raw_text.split(opening, 1)[1]
            if closing not in after_open:
                return after_open.strip(), ""
            reasoning, final = after_open.rsplit(closing, 1)
            return reasoning.strip(), _strip_response_wrapper(final)
        if closing in raw_text:
            reasoning, final = raw_text.rsplit(closing, 1)
            return reasoning.strip(), _strip_response_wrapper(final)

    # Some inference backends return an already-parsed final response with
    # the reasoning channel removed. Treat that as a final answer rather than
    # incorrectly classifying it as an unterminated thought.
    return "", _strip_response_wrapper(raw_text)


def parse_total(final_text: str) -> int | None:
    matches = TOTAL_RE.findall(final_text)
    return int(matches[-1]) if matches else None


def parse_indexed_records(final_text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, city, score in INDEXED_ENUMERATION_RE.findall(final_text):
        records.append(
            {
                "index": int(index),
                "city": city.strip(),
                "score": int(score),
                "marker": "index",
            }
        )
    return records


def parse_bullet_records(final_text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for city, score in BULLET_ENUMERATION_RE.findall(final_text):
        records.append(
            {
                "index": None,
                "city": city.strip(),
                "score": int(score),
                "marker": "bullet",
            }
        )
    return records


def parse_semantic_records(text: str) -> list[dict[str, Any]]:
    """Parse either registered enumeration marker while preserving line order."""

    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        indexed = INDEXED_ENUMERATION_RE.fullmatch(line)
        if indexed:
            index, city, score = indexed.groups()
            records.append(
                {
                    "index": int(index),
                    "city": city.strip(),
                    "score": int(score),
                    "marker": "index",
                }
            )
            continue
        bullet = BULLET_ENUMERATION_RE.fullmatch(line)
        if bullet:
            city, score = bullet.groups()
            records.append(
                {
                    "index": None,
                    "city": city.strip(),
                    "score": int(score),
                    "marker": "bullet",
                }
            )
    return records


def _enumeration_audit(
    final_text: str,
    *,
    prompt_mode: str,
) -> dict[str, Any]:
    if prompt_mode not in ENUMERATION_PROMPT_MODES:
        return {
            "listed_records": [],
            "strict_listed_records": [],
            "enumeration_format_status": None,
            "enumeration_format_compliant": None,
        }

    indexed = parse_indexed_records(final_text)
    bullets = parse_bullet_records(final_text)
    semantic = parse_semantic_records(final_text)
    expected = indexed if prompt_mode == "enumeration_index" else bullets
    opposite = bullets if prompt_mode == "enumeration_index" else indexed

    if indexed and bullets:
        status = "mixed_markers"
    elif opposite and not expected:
        status = "wrong_marker"
    elif not expected:
        status = "no_records"
    elif prompt_mode == "enumeration_index" and [
        row["index"] for row in indexed
    ] != list(range(1, len(indexed) + 1)):
        status = "index_sequence_error"
    else:
        status = "ok"
    return {
        "listed_records": semantic,
        "strict_listed_records": expected,
        "enumeration_format_status": status,
        "enumeration_format_compliant": status == "ok",
    }


def _reasoning_repetition_audit(reasoning: str) -> dict[str, Any]:
    normalized_lines = [
        " ".join(line.lower().split())
        for line in reasoning.splitlines()
        if line.strip()
    ]
    counts = Counter(normalized_lines)
    duplicate_lines = sum(count - 1 for count in counts.values() if count > 1)

    indices = [int(value) for value in REASONING_INDEX_RE.findall(reasoning)]
    restart_count = sum(
        current <= previous
        for previous, current in zip(indices, indices[1:])
    )

    records = parse_semantic_records(reasoning)
    pairs = [(row["city"], row["score"]) for row in records]
    duplicate_record_mentions = len(pairs) - len(set(pairs))
    return {
        "reasoning_characters": len(reasoning),
        "reasoning_nonempty_lines": len(normalized_lines),
        "reasoning_duplicate_lines": duplicate_lines,
        "reasoning_line_repetition_fraction": (
            duplicate_lines / len(normalized_lines)
            if normalized_lines
            else 0.0
        ),
        "reasoning_enumeration_restart_count": restart_count,
        "reasoning_record_mentions": len(records),
        "reasoning_duplicate_record_mentions": duplicate_record_mentions,
    }


def _response_format_compliant(
    final_text: str,
    *,
    prompt_mode: str,
    enumeration: dict[str, Any],
) -> bool:
    if prompt_mode not in ENUMERATION_PROMPT_MODES:
        return TOTAL_ONLY_RE.fullmatch(final_text) is not None

    lines = [line for line in final_text.splitlines() if line.strip()]
    if not lines or enumeration["enumeration_format_status"] != "ok":
        return False
    total_lines = [
        index
        for index, line in enumerate(lines)
        if TOTAL_RE.fullmatch(line) is not None
    ]
    if total_lines != [len(lines) - 1]:
        return False
    record_pattern = (
        INDEXED_ENUMERATION_RE
        if prompt_mode == "enumeration_index"
        else BULLET_ENUMERATION_RE
    )
    return all(record_pattern.fullmatch(line) is not None for line in lines[:-1])


def evaluate_generation(
    raw_text: str,
    *,
    prompt_mode: str,
    reasoning_expected: bool | None = None,
    gold_pairs: list[dict[str, Any]],
    finish_reason: str | None,
    output_tokens: int | None = None,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    reasoning, final = split_reasoning_and_final(
        raw_text,
        prompt_mode=prompt_mode,
        reasoning_expected=reasoning_expected,
    )
    predicted_count = parse_total(final)
    enumeration = _enumeration_audit(final, prompt_mode=prompt_mode)
    listed_records = enumeration["listed_records"]
    strict_listed_records = enumeration["strict_listed_records"]
    gold = [(str(pair["city"]), int(pair["score"])) for pair in gold_pairs]
    predicted = [
        (str(pair["city"]), int(pair["score"])) for pair in listed_records
    ]
    gold_set = set(gold)
    predicted_set = set(predicted)
    true_positive = len(gold_set & predicted_set)
    precision = true_positive / len(predicted_set) if predicted_set else None
    recall = true_positive / len(gold_set) if gold_set else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall > 0
        else None
    )
    repetition = _reasoning_repetition_audit(reasoning)
    overthinking_signals: list[str] = []
    if finish_reason == "length":
        overthinking_signals.append("truncated")
    if repetition["reasoning_enumeration_restart_count"] > 0:
        overthinking_signals.append("enumeration_restart")
    if repetition["reasoning_duplicate_record_mentions"] > 0:
        overthinking_signals.append("duplicate_record_mentions")
    if repetition["reasoning_duplicate_lines"] > 0:
        overthinking_signals.append("duplicate_reasoning_lines")
    response_format_compliant = _response_format_compliant(
        final,
        prompt_mode=prompt_mode,
        enumeration=enumeration,
    )
    exact_count = predicted_count == len(gold)
    registered_success = bool(
        exact_count
        and response_format_compliant
        and finish_reason != "length"
    )
    return {
        "reasoning_text": reasoning,
        "final_text": final,
        "predicted_count": predicted_count,
        "parse_status": "ok" if predicted_count is not None else "parse_fail",
        "exact_count": exact_count,
        "response_format_compliant": response_format_compliant,
        "registered_success": registered_success,
        "signed_error": (
            predicted_count - len(gold) if predicted_count is not None else None
        ),
        "absolute_error": (
            abs(predicted_count - len(gold))
            if predicted_count is not None
            else None
        ),
        "normalized_absolute_error": (
            abs(predicted_count - len(gold)) / max(1, len(gold))
            if predicted_count is not None
            else None
        ),
        "listed_records": listed_records,
        "strict_listed_records": strict_listed_records,
        "enumeration_format_status": enumeration[
            "enumeration_format_status"
        ],
        "enumeration_format_compliant": enumeration[
            "enumeration_format_compliant"
        ],
        "listed_total_matches_length": (
            predicted_count == len(listed_records)
            if prompt_mode in ENUMERATION_PROMPT_MODES
            and predicted_count is not None
            else None
        ),
        "strict_listed_total_matches_length": (
            predicted_count == len(strict_listed_records)
            if prompt_mode in ENUMERATION_PROMPT_MODES
            and predicted_count is not None
            else None
        ),
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f1": f1,
        "duplicate_listed_pairs": len(predicted) - len(predicted_set),
        "hallucinated_pairs": sorted(predicted_set - gold_set),
        "missing_pairs": sorted(gold_set - predicted_set),
        "finish_reason": finish_reason,
        "truncated": finish_reason == "length",
        "output_tokens": output_tokens,
        "max_output_tokens": max_output_tokens,
        "output_budget_fraction": (
            output_tokens / max_output_tokens
            if output_tokens is not None
            and max_output_tokens is not None
            and max_output_tokens > 0
            else None
        ),
        **repetition,
        "overthinking_signals": overthinking_signals,
        "overthinking_signal_count": len(overthinking_signals),
        "overthinking_flag": bool(overthinking_signals),
    }
