from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from realistic_niah.parsing import (
    GEMMA_THINK_CLOSE,
    GEMMA_THINK_OPEN,
    QWEN_THINK_CLOSE,
    QWEN_THINK_OPEN,
)


INDEXED_ITEM_RE = re.compile(r"^[ \t]*(\d+)[.)][ \t]+\S.*$")
BULLET_ITEM_RE = re.compile(r"^[ \t]*-[ \t]+\S.*$")


@dataclass(frozen=True)
class ReasoningSpan:
    start: int
    end: int
    opening_delimiter_present: bool
    closing_delimiter_present: bool
    closing_delimiter: str


@dataclass(frozen=True)
class FirstListCut:
    detected: bool
    status: str
    marker_kind: str | None = None
    item_count: int = 0
    item_markers: tuple[int | str, ...] = ()
    reasoning_start_char: int | None = None
    reasoning_end_char: int | None = None
    list_start_char: int | None = None
    cut_char: int | None = None
    boundary_kind: str | None = None
    closing_delimiter_present: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TokenPrefixAlignment:
    eligible: bool
    status: str
    strategy: str | None = None
    token_ids: tuple[int, ...] = ()
    shared_baseline_prefix_tokens: int = 0
    retokenized_suffix_tokens: int = 0

    def to_dict(self, *, include_token_ids: bool = False) -> dict[str, Any]:
        value = asdict(self)
        if not include_token_ids:
            value.pop("token_ids")
        return value


@dataclass(frozen=True)
class _Line:
    text: str
    start: int
    end: int
    has_newline: bool


def thinking_delimiters(model_family: str) -> tuple[str, str]:
    if model_family == "qwen3":
        return QWEN_THINK_OPEN, QWEN_THINK_CLOSE
    if model_family == "gemma4":
        return GEMMA_THINK_OPEN, GEMMA_THINK_CLOSE
    raise ValueError(
        "First-list cutoff is registered only for qwen3 and gemma4, got "
        f"{model_family!r}"
    )


def locate_reasoning_span(raw_text: str, *, model_family: str) -> ReasoningSpan:
    opening, closing = thinking_delimiters(model_family)
    opening_index = raw_text.find(opening)
    if opening_index >= 0:
        start = opening_index + len(opening)
        opening_present = True
    else:
        start = 0
        opening_present = False

    if model_family == "qwen3":
        closing_index = raw_text.rfind(closing, start)
    else:
        closing_index = raw_text.find(closing, start)
    closing_present = closing_index >= 0
    end = closing_index if closing_present else len(raw_text)
    if end < start:
        raise ValueError("Reasoning delimiters are out of order")
    return ReasoningSpan(
        start=start,
        end=end,
        opening_delimiter_present=opening_present,
        closing_delimiter_present=closing_present,
        closing_delimiter=closing,
    )


def _lines_with_offsets(text: str) -> list[_Line]:
    lines: list[_Line] = []
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        end = offset + len(raw_line)
        has_newline = raw_line.endswith(("\n", "\r"))
        lines.append(
            _Line(
                text=raw_line.rstrip("\r\n"),
                start=offset,
                end=end,
                has_newline=has_newline,
            )
        )
        offset = end
    return lines


def _index_marker(line: _Line) -> int | None:
    match = INDEXED_ITEM_RE.fullmatch(line.text)
    return int(match.group(1)) if match else None


def _is_bullet(line: _Line) -> bool:
    return BULLET_ITEM_RE.fullmatch(line.text) is not None


def _boundary_kind(
    lines: list[_Line],
    next_index: int,
    *,
    reasoning_closed: bool,
) -> str | None:
    if next_index < len(lines):
        return "blank_line" if not lines[next_index].text.strip() else "next_line"
    if reasoning_closed:
        return "thinking_close"
    return None


def find_first_completed_list(
    raw_text: str,
    *,
    model_family: str,
) -> FirstListCut:
    """Find the first completed visible numbered or hyphen-list block.

    This parser deliberately does not receive the gold count or gold records.
    A numbered block must start at 1 and increment strictly. A bullet block is
    an uninterrupted run of ``- `` lines. Blank lines terminate a block. The
    cut is the newline ending the last item line. An output that simply ends
    after an item without a thinking-close delimiter is not proof that the
    list was completed and is therefore a no-hit.
    """

    span = locate_reasoning_span(raw_text, model_family=model_family)
    reasoning = raw_text[span.start : span.end]
    lines = _lines_with_offsets(reasoning)

    for start_index, line in enumerate(lines):
        marker = _index_marker(line)
        if marker == 1:
            markers: list[int | str] = []
            expected = 1
            cursor = start_index
            unterminated_matching_item = False
            while cursor < len(lines):
                current = _index_marker(lines[cursor])
                if current != expected:
                    break
                if not lines[cursor].has_newline:
                    unterminated_matching_item = True
                    break
                markers.append(current)
                expected += 1
                cursor += 1
            if markers and not unterminated_matching_item:
                boundary = _boundary_kind(
                    lines,
                    cursor,
                    reasoning_closed=span.closing_delimiter_present,
                )
                if boundary is not None:
                    last = lines[cursor - 1]
                    return FirstListCut(
                        detected=True,
                        status="ok",
                        marker_kind="indexed",
                        item_count=len(markers),
                        item_markers=tuple(markers),
                        reasoning_start_char=span.start,
                        reasoning_end_char=span.end,
                        list_start_char=span.start + line.start,
                        cut_char=span.start + last.end,
                        boundary_kind=boundary,
                        closing_delimiter_present=(
                            span.closing_delimiter_present
                        ),
                    )

        if _is_bullet(line):
            markers = []
            cursor = start_index
            unterminated_matching_item = False
            while cursor < len(lines) and _is_bullet(lines[cursor]):
                if not lines[cursor].has_newline:
                    unterminated_matching_item = True
                    break
                markers.append("-")
                cursor += 1
            if markers and not unterminated_matching_item:
                boundary = _boundary_kind(
                    lines,
                    cursor,
                    reasoning_closed=span.closing_delimiter_present,
                )
                if boundary is not None:
                    last = lines[cursor - 1]
                    return FirstListCut(
                        detected=True,
                        status="ok",
                        marker_kind="bullet",
                        item_count=len(markers),
                        item_markers=tuple(markers),
                        reasoning_start_char=span.start,
                        reasoning_end_char=span.end,
                        list_start_char=span.start + line.start,
                        cut_char=span.start + last.end,
                        boundary_kind=boundary,
                        closing_delimiter_present=(
                            span.closing_delimiter_present
                        ),
                    )

    return FirstListCut(
        detected=False,
        status="no_completed_list",
        reasoning_start_char=span.start,
        reasoning_end_char=span.end,
        closing_delimiter_present=span.closing_delimiter_present,
    )


def exact_token_prefix_length(
    tokenizer: Any,
    *,
    raw_text: str,
    output_token_ids: Iterable[int],
    cut_char: int,
) -> int | None:
    """Return the exact token boundary for ``raw_text[:cut_char]``.

    No nearest-token approximation is allowed. The intervention is eligible
    only when re-encoding and decoding reproduce the exact textual prefix and
    the resulting IDs are literally the baseline output prefix.
    """

    token_ids = list(output_token_ids)
    target = raw_text[:cut_char]
    encoded = list(tokenizer.encode(target, add_special_tokens=False))
    if token_ids[: len(encoded)] != encoded:
        return None
    decoded = tokenizer.decode(
        encoded,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return len(encoded) if decoded == target else None


def align_text_exact_token_prefix(
    tokenizer: Any,
    *,
    raw_text: str,
    output_token_ids: Iterable[int],
    cut_char: int,
) -> TokenPrefixAlignment:
    """Encode the exact visible prefix and audit boundary retokenization.

    A newline cut often falls inside a token encoding two consecutive
    newlines. In that case a literal baseline-ID prefix cannot represent the
    requested text. We preserve the exact characters and retokenize only the
    suffix beginning at the first differing token. This is not a nearest-token
    approximation: decoding must reproduce ``raw_text[:cut_char]`` exactly.
    """

    baseline_ids = list(output_token_ids)
    target = raw_text[:cut_char]
    encoded = list(tokenizer.encode(target, add_special_tokens=False))
    decoded = tokenizer.decode(
        encoded,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != target:
        return TokenPrefixAlignment(
            eligible=False,
            status="text_prefix_did_not_round_trip",
        )

    shared = 0
    for baseline_id, encoded_id in zip(baseline_ids, encoded):
        if baseline_id != encoded_id:
            break
        shared += 1
    exact_boundary = baseline_ids[: len(encoded)] == encoded
    return TokenPrefixAlignment(
        eligible=True,
        status="ok",
        strategy=(
            "literal_baseline_token_prefix"
            if exact_boundary
            else "text_exact_boundary_retokenization"
        ),
        token_ids=tuple(encoded),
        shared_baseline_prefix_tokens=shared,
        retokenized_suffix_tokens=(
            0 if exact_boundary else len(encoded) - shared
        ),
    )


def intervention_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    detected = [row for row in values if row["parser"]["detected"]]
    eligible = [row for row in values if row["intervention_eligible"]]
    completed = [row for row in eligible if row.get("intervention")]

    def accuracy(source: Iterable[dict[str, Any]], key: str) -> float | None:
        subset = list(source)
        if not subset:
            return None
        return sum(bool(row[key]["evaluation"]["exact_count"]) for row in subset) / len(subset)

    return {
        "rows": len(values),
        "parser_detected": len(detected),
        "parser_hit_rate": len(detected) / len(values) if values else None,
        "intervention_eligible": len(eligible),
        "intervention_completed": len(completed),
        "baseline_exact_count_accuracy_all": accuracy(values, "baseline"),
        "baseline_exact_count_accuracy_eligible": accuracy(
            completed, "baseline"
        ),
        "intervention_exact_count_accuracy_eligible": accuracy(
            completed, "intervention"
        ),
        "policy_exact_count_accuracy_all": accuracy(values, "policy"),
    }
