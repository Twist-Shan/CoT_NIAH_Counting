from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Sequence

from .first_list_cutoff import locate_reasoning_span
from .gold_city_cutoff import _GoldCityMatcher, _gold_cities


INDEXED_ITEM_RE = re.compile(r"^[ \t]*(\d+)[.)][ \t]*\S.*$")
BULLET_ITEM_RE = re.compile(
    r"^[ \t]*(?P<bullet>[-\u2022]|\*(?!\*))[ \t]*(?!>)\S.*$"
)

_ORDINAL_BASE = (
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
    "eleventh",
    "twelfth",
    "thirteenth",
    "fourteenth",
    "fifteenth",
    "sixteenth",
    "seventeenth",
    "eighteenth",
    "nineteenth",
    "twentieth",
)
ORDINAL_VALUES = {
    spelling: index for index, spelling in enumerate(_ORDINAL_BASE, start=1)
}
ORDINAL_VALUES |= {
    spelling + "ly": index
    for index, spelling in enumerate(_ORDINAL_BASE, start=1)
}
ORDINAL_ITEM_RE = re.compile(
    r"^[ \t]*(?:\*{1,2}|`)?(?P<ordinal>"
    + "|".join(sorted(ORDINAL_VALUES, key=len, reverse=True))
    + r")(?:\*{1,2}|`)?(?:"
    + r"(?:[.):,-](?:\*{1,2}|`)?)[ \t]*|[ \t]+)\S.*$",
    flags=re.IGNORECASE,
)
TRANSITION_ITEM_RE = re.compile(
    r"^[ \t]*(?:\*{1,2}|`)?(?P<transition>then|finally)"
    r"(?:\*{1,2}|`)?(?:"
    r"(?:[.):,-](?:\*{1,2}|`)?)[ \t]*|[ \t]+)\S.*$",
    flags=re.IGNORECASE,
)
_BARE_PERIOD_MARKER_RE = re.compile(
    r"^[ \t]*(?:\d+|(?:\*{1,2}|`)?(?:"
    + "|".join(sorted(ORDINAL_VALUES, key=len, reverse=True))
    + r"|then|finally)(?:\*{1,2}|`)?)$",
    flags=re.IGNORECASE,
)

# Qwen often performs the same structured count without putting an explicit
# marker in front of every city.  These two fallbacks are deliberately narrow:
# a late, explicitly count-like recap, or the exact synthetic audit-record
# sentence used by V4.4.  They are tried only after the explicit-marker parser.
_RECAP_CUE_RE = re.compile(
    r"\b(?:"
    r"counting(?:\s+them)?(?:\s+again)?|"
    r"(?:let\s+me|let(?:'|\u2019)s|I(?:'|\u2019)ll|we(?:'|\u2019)ll)\s+"
    r"(?:count|recount)(?:\s+them)?(?:\s+again)?|"
    r"tally(?:ing)?(?:\s+up)?|"
    r"recount(?:ing|ed)?|"
    r"(?:cities|entries|records|instances|mentions)\s+"
    r"(?:listed|found|identified|counted|are)|"
    r"(?:cities|entries|records|instances|mentions)\s+"
    r"(?:mentioned|seen)\s+(?:are|were)|"
    r"(?:cities|entries|records|instances|mentions)\s+"
    r"(?:I|we)\s+(?:found|identified|counted|listed)(?:\s+are)?|"
    r"(?:found|identified|listed)\s+(?:the following|these)|"
    r"all\s+(?:the\s+)?(?:cities|entries|records|instances|mentions)|"
    r"the\s+(?:cities|entries|records|instances|mentions)\s+listed|"
    r"(?:in\s+total|total\s+of)|"
    r"(?:there\s+(?:are|were)\s+)?(?:exactly\s+)?"
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:explicit\s+)?(?:city[- ]score\s+audit\s+)?"
    r"(?:cities|entries|records|instances|mentions)|"
    r"(?:only|single)\s+(?:city[- ]score\s+audit\s+)?record\s+is"
    r")\b",
    flags=re.IGNORECASE,
)
_N1_RECAP_COUNT_RE = re.compile(
    r"\b(?:one|1|only|single)\b", flags=re.IGNORECASE
)
_CARDINAL_COUNTS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_RECAP_COUNT_SUMMARY_RE = re.compile(
    r"\b(?P<count>\d+|" + "|".join(_CARDINAL_COUNTS) + r")\s+"
    r"(?:cities|entries|records|instances|mentions)\b",
    flags=re.IGNORECASE,
)
_CLOSING_SENTENCE_CHARS = frozenset("\"'\u201d\u2019)]")


@dataclass(frozen=True)
class _Line:
    text: str
    start: int
    end: int
    content_end: int
    has_newline: bool
    boundary_kind: str | None
    physical_line_number: int

    @property
    def has_boundary(self) -> bool:
        return self.boundary_kind is not None


@dataclass(frozen=True)
class _Marker:
    kind: str
    value: int | str


@dataclass(frozen=True)
class _CityOccurrence:
    city: str
    start: int
    end: int


@dataclass(frozen=True)
class _Candidate:
    terminated: bool
    rejection_reason: str | None
    termination_kind: str | None
    marker_kind: str
    item_markers: tuple[int | str, ...]
    item_gold_cities: tuple[str, ...]
    item_line_numbers: tuple[int, ...]
    item_start_chars: tuple[int, ...]
    item_end_chars: tuple[int, ...]
    item_boundary_kinds: tuple[str, ...]
    matched_gold_cities: tuple[str, ...]
    duplicate_gold_city_items: int
    bridge_line_count: int
    trailing_line_count: int
    start_char: int
    cut_char: int | None

    @property
    def coverage_count(self) -> int:
        return len(self.matched_gold_cities)

    @property
    def item_count(self) -> int:
        return len(self.item_markers)


@dataclass(frozen=True)
class CityListTerminationCut:
    detected: bool
    status: str
    all_items_gold_city: bool
    coverage_complete: bool
    gold_city_count: int
    gold_cities: tuple[str, ...]
    coverage_count: int = 0
    coverage_fraction: float = 0.0
    matched_gold_cities: tuple[str, ...] = ()
    missing_gold_cities: tuple[str, ...] = ()
    marker_kind: str | None = None
    item_count: int = 0
    item_markers: tuple[int | str, ...] = ()
    item_gold_cities: tuple[str, ...] = ()
    item_line_numbers: tuple[int, ...] = ()
    item_start_chars: tuple[int, ...] = ()
    item_end_chars: tuple[int, ...] = ()
    item_boundary_kinds: tuple[str, ...] = ()
    duplicate_gold_city_items: int = 0
    trace_one_to_one: bool = False
    trace_order_class: str | None = None
    trace_category: str | None = None
    bridge_line_count: int = 0
    trailing_line_count: int = 0
    candidates_considered: int = 0
    rejected_candidates: int = 0
    termination_kind: str | None = None
    reasoning_start_char: int | None = None
    reasoning_end_char: int | None = None
    list_start_char: int | None = None
    cut_char: int | None = None
    boundary_kind: str | None = None
    closing_delimiter_present: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _marker_text(text: str) -> _Marker | None:
    indexed = INDEXED_ITEM_RE.fullmatch(text)
    if indexed:
        return _Marker("indexed", int(indexed.group(1)))
    bullet = BULLET_ITEM_RE.fullmatch(text)
    if bullet:
        return _Marker("bullet", bullet.group("bullet"))
    ordinal = ORDINAL_ITEM_RE.fullmatch(text)
    if ordinal:
        spelling = ordinal.group("ordinal").casefold()
        return _Marker("ordinal", ORDINAL_VALUES[spelling])
    transition = TRANSITION_ITEM_RE.fullmatch(text)
    if transition:
        return _Marker(
            "transition", transition.group("transition").casefold()
        )
    return None


def _lines_with_offsets(
    text: str,
    matcher: _GoldCityMatcher | None = None,
) -> list[_Line]:
    """Return logical item units while preserving exact character offsets.

    Without a matcher this is the legacy physical-line view used by the
    no-hit diagnostics.  With a matcher, a sentence-ending period can be a
    logical boundary when the segment before it is a supported marker plus
    exactly one gold city, or the following segment begins with a supported
    marker.  A final period immediately before the reasoning close is also a
    boundary.  Marker punctuation such as ``1. Paris`` and ``First. Paris``
    is not split because the prefix is not itself a complete city item.
    """

    lines: list[_Line] = []
    offset = 0
    for physical_line_number, raw_line in enumerate(
        text.splitlines(keepends=True), start=1
    ):
        raw_end = offset + len(raw_line)
        content = raw_line.rstrip("\r\n")
        has_newline = len(content) != len(raw_line)
        if matcher is None:
            lines.append(
                _Line(
                    text=content,
                    start=offset,
                    end=raw_end,
                    content_end=offset + len(content),
                    has_newline=has_newline,
                    boundary_kind="newline" if has_newline else None,
                    physical_line_number=physical_line_number,
                )
            )
            offset = raw_end
            continue

        segment_start = 0
        for period_index, character in enumerate(content):
            if character != ".":
                continue
            next_index = period_index + 1
            if next_index < len(content) and not content[next_index].isspace():
                continue
            before = content[segment_start:period_index]
            after = content[next_index:]
            # Preserve legacy newline coordinates when a sentence period is
            # immediately followed by the physical line ending.  Period
            # boundaries are only needed inside a line or at an un-newlined
            # reasoning end.
            if not after.strip() and has_newline:
                continue
            # In ``2. Then later ...`` or ``First. Chicago``, this period is
            # marker punctuation, even though the following text can itself
            # begin with a transition word.  It must never become a sentence
            # boundary.
            if _BARE_PERIOD_MARKER_RE.fullmatch(before):
                continue
            before_is_city_item = (
                _marker_text(before) is not None
                and len(matcher.cities_in(before)) == 1
            )
            after_begins_item = _marker_text(after.lstrip()) is not None
            if not (before_is_city_item or after_begins_item):
                continue
            lines.append(
                _Line(
                    text=before,
                    start=offset + segment_start,
                    end=offset + next_index,
                    content_end=offset + period_index,
                    has_newline=False,
                    boundary_kind="period",
                    physical_line_number=physical_line_number,
                )
            )
            segment_start = next_index

        remainder = content[segment_start:]
        if remainder or has_newline:
            lines.append(
                _Line(
                    text=remainder,
                    start=offset + segment_start,
                    end=raw_end,
                    content_end=offset + len(content),
                    has_newline=has_newline,
                    boundary_kind="newline" if has_newline else None,
                    physical_line_number=physical_line_number,
                )
            )
        offset = raw_end
    return lines


def _marker(line: _Line) -> _Marker | None:
    return _marker_text(line.text)


def _gold_city_occurrences(
    text: str,
    gold_cities: Sequence[str],
) -> list[_CityOccurrence]:
    """Return every non-overlapping gold-city mention in textual order.

    ``_GoldCityMatcher.cities_in`` intentionally deduplicates cities, which is
    right for coverage but wrong for trace classification.  The fallback
    formats need to preserve repeats, so they use exact occurrence spans.
    Longer city names are placed first to keep names such as ``Mexico City``
    intact if the catalogue also contains a shorter overlapping name.
    """

    if not gold_cities:
        return []
    canonical = {city.casefold(): city for city in gold_cities}
    alternatives = "|".join(
        re.escape(city)
        for city in sorted(gold_cities, key=len, reverse=True)
    )
    pattern = re.compile(
        rf"(?<!\w)(?P<city>{alternatives})(?!\w)",
        flags=re.IGNORECASE,
    )
    return [
        _CityOccurrence(
            city=canonical[match.group("city").casefold()],
            start=match.start("city"),
            end=match.end("city"),
        )
        for match in pattern.finditer(text)
    ]


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split prose at conservative sentence punctuation with exact offsets."""

    spans: list[tuple[int, int]] = []
    start = 0
    cursor = 0
    while cursor < len(text):
        if text[cursor] not in ".!?":
            cursor += 1
            continue
        end = cursor + 1
        while end < len(text) and text[end] in _CLOSING_SENTENCE_CHARS:
            end += 1
        if end < len(text) and not text[end].isspace():
            cursor += 1
            continue
        left = start
        while left < end and text[left].isspace():
            left += 1
        if left < end:
            spans.append((left, end))
        start = end
        cursor = end
    left = start
    while left < len(text) and text[left].isspace():
        left += 1
    if left < len(text):
        spans.append((left, len(text)))
    return spans


def _physical_line_number(text: str, char_index: int) -> int:
    return text.count("\n", 0, char_index) + 1


def _candidate_from_occurrences(
    *,
    reasoning: str,
    marker_kind: str,
    termination_kind: str,
    occurrences: Sequence[_CityOccurrence],
    item_spans: Sequence[tuple[int, int]],
    start_char: int,
    cut_char: int,
    boundary_kind: str,
) -> _Candidate:
    if not occurrences or len(occurrences) != len(item_spans):
        raise ValueError("Fallback candidate occurrence/span mismatch")
    item_cities = [occurrence.city for occurrence in occurrences]
    matched: list[str] = []
    seen: set[str] = set()
    duplicates = 0
    for city in item_cities:
        if not _append_city(city, seen=seen, matched=matched):
            duplicates += 1
    return _candidate_result(
        terminated=True,
        rejection_reason=None,
        termination_kind=termination_kind,
        marker_kind=marker_kind,
        markers=list(range(1, len(occurrences) + 1)),
        item_cities=item_cities,
        item_lines=[
            _physical_line_number(reasoning, start) for start, _end in item_spans
        ],
        item_starts=[start for start, _end in item_spans],
        item_ends=[end for _start, end in item_spans],
        item_boundaries=[boundary_kind] * len(item_spans),
        matched=matched,
        duplicate_items=duplicates,
        bridges=0,
        trailing_lines=0,
        start_char=start_char,
        cut_char=cut_char,
    )


def _late_completion_recap_candidate(
    reasoning: str,
    *,
    gold_cities: Sequence[str],
) -> _Candidate | None:
    """Find a compact, count-signalled city recap in the reasoning tail.

    A cue can occur in the city sentence itself or in the immediately previous
    sentence (``Let me count again. Paris, Rome, Oslo.``).  Requiring the tail,
    a strong cue, and a compact city span prevents ordinary scan prose from
    becoming a cutoff.  Coverage is not an eligibility gate.
    """

    sentence_spans = _sentence_spans(reasoning)
    tail_start = int(len(reasoning) * 0.45)
    for index, (start, end) in enumerate(sentence_spans):
        # Use the sentence end so a recap that straddles the tail boundary is
        # not discarded merely because its introductory words began earlier.
        if end < tail_start:
            continue
        sentence = reasoning[start:end]
        local_occurrences = _gold_city_occurrences(sentence, gold_cities)
        if not local_occurrences:
            continue
        direct_cue = bool(_RECAP_CUE_RE.search(sentence))
        previous_cue = False
        if index > 0:
            previous_start, previous_end = sentence_spans[index - 1]
            previous_cue = bool(
                _RECAP_CUE_RE.search(reasoning[previous_start:previous_end])
            )
        next_count_confirms_items = False
        if index + 1 < len(sentence_spans):
            next_start, next_end = sentence_spans[index + 1]
            next_sentence = reasoning[next_start:next_end]
            summary = _RECAP_COUNT_SUMMARY_RE.search(next_sentence)
            if summary is not None:
                token = summary.group("count").casefold()
                stated_count = (
                    int(token) if token.isdigit() else _CARDINAL_COUNTS[token]
                )
                next_count_confirms_items = stated_count == len(local_occurrences)
        if not (direct_cue or previous_cue or next_count_confirms_items):
            continue
        first_city = local_occurrences[0].start
        last_city = local_occurrences[-1].end
        compact_limit = max(220, 55 * len(local_occurrences))
        if last_city - first_city > compact_limit:
            continue
        if len(local_occurrences) == 1:
            # A single city is too easy to mention incidentally.  It is only a
            # recap when N=1 and the same sentence explicitly says one/only.
            if len(gold_cities) != 1 or not (
                direct_cue and _N1_RECAP_COUNT_RE.search(sentence)
            ):
                continue
        occurrences = [
            _CityOccurrence(
                city=occurrence.city,
                start=start + occurrence.start,
                end=start + occurrence.end,
            )
            for occurrence in local_occurrences
        ]
        return _candidate_from_occurrences(
            reasoning=reasoning,
            marker_kind="completion_recap",
            termination_kind="completion_recap_period",
            occurrences=occurrences,
            item_spans=[
                (occurrence.start, occurrence.end) for occurrence in occurrences
            ],
            start_char=start,
            cut_char=end,
            boundary_kind="recap_period",
        )
    return None


def _audit_sentence_chain_candidate(
    reasoning: str,
    *,
    gold_cities: Sequence[str],
) -> _Candidate | None:
    """Parse exact V4.4 audit-record sentences as an implicit item chain."""

    if not gold_cities:
        return None
    canonical = {city.casefold(): city for city in gold_cities}
    city_alternatives = "|".join(
        re.escape(city)
        for city in sorted(gold_cities, key=len, reverse=True)
    )
    emphasis = r"(?:\*{1,2}|`)?"
    pattern = re.compile(
        r"In\s+the\s+2024\s+city\s+score\s+audit,\s+"
        + emphasis
        + rf"(?P<city>{city_alternatives})"
        + emphasis
        + r"\s+received\s+a\s+score\s+of\s+"
        + emphasis
        + r"\d+"
        + emphasis
        + r"\s*\.",
        flags=re.IGNORECASE,
    )
    matches = list(pattern.finditer(reasoning))
    if not matches:
        return None
    occurrences = [
        _CityOccurrence(
            city=canonical[match.group("city").casefold()],
            start=match.start("city"),
            end=match.end("city"),
        )
        for match in matches
    ]
    return _candidate_from_occurrences(
        reasoning=reasoning,
        marker_kind="audit_sentence",
        termination_kind="audit_sentence_chain_period",
        occurrences=occurrences,
        item_spans=[match.span() for match in matches],
        start_char=matches[0].start(),
        cut_char=matches[-1].end(),
        boundary_kind="audit_sentence_period",
    )


def _trace_classification(
    item_cities: Sequence[str],
    gold_cities: Sequence[str],
) -> tuple[bool, str, str]:
    """Classify the accepted city sequence without making it an eligibility gate.

    ``trace_one_to_one`` means that every gold city occurs exactly once and no
    other item is present.  Ordering is deliberately secondary: passage order,
    exact reverse order, and another permutation are recorded separately.
    Partial and repeated traces remain valid parser hits.
    """

    observed = tuple(city.casefold() for city in item_cities)
    gold = tuple(city.casefold() for city in gold_cities)
    one_to_one = len(observed) == len(gold) and Counter(observed) == Counter(gold)
    if one_to_one:
        if observed == gold:
            order_class = "forward"
        elif len(gold) > 1 and observed == tuple(reversed(gold)):
            order_class = "reverse"
        else:
            order_class = "other_permutation"
        return True, order_class, "one_to_one"

    unique_observed = set(observed)
    coverage_complete = set(gold).issubset(unique_observed)
    has_duplicates = len(observed) != len(unique_observed)
    if coverage_complete and has_duplicates:
        category = "full_coverage_with_duplicates"
    elif has_duplicates:
        category = "partial_with_duplicates"
    else:
        category = "partial_unique"
    return False, "not_one_to_one", category


def _append_city(
    city: str,
    *,
    seen: set[str],
    matched: list[str],
) -> bool:
    folded = city.casefold()
    if folded in seen:
        return False
    seen.add(folded)
    matched.append(city)
    return True


def _candidate_result(
    *,
    terminated: bool,
    rejection_reason: str | None,
    termination_kind: str | None,
    marker_kind: str,
    markers: list[int | str],
    item_cities: list[str],
    item_lines: list[int],
    item_starts: list[int],
    item_ends: list[int],
    item_boundaries: list[str],
    matched: list[str],
    duplicate_items: int,
    bridges: int,
    trailing_lines: int,
    start_char: int,
    cut_char: int | None,
) -> _Candidate:
    return _Candidate(
        terminated=terminated,
        rejection_reason=rejection_reason,
        termination_kind=termination_kind,
        marker_kind=marker_kind,
        item_markers=tuple(markers),
        item_gold_cities=tuple(item_cities),
        item_line_numbers=tuple(item_lines),
        item_start_chars=tuple(item_starts),
        item_end_chars=tuple(item_ends),
        item_boundary_kinds=tuple(item_boundaries),
        matched_gold_cities=tuple(matched),
        duplicate_gold_city_items=duplicate_items,
        bridge_line_count=bridges,
        trailing_line_count=trailing_lines,
        start_char=start_char,
        cut_char=cut_char,
    )


def _scan_sequenced_candidate(
    lines: Sequence[_Line],
    *,
    start_index: int,
    marker_kind: str,
    matcher: _GoldCityMatcher,
    reasoning_closed: bool,
    gold_city_count: int,
) -> _Candidate:
    markers: list[int | str] = []
    item_cities: list[str] = []
    item_lines: list[int] = []
    item_starts: list[int] = []
    item_ends: list[int] = []
    item_boundaries: list[str] = []
    matched: list[str] = []
    seen: set[str] = set()
    duplicate_items = 0
    bridges = 0
    pending_bridges = 0
    pending_nonblank = False
    expected = 1
    finally_seen = False
    last_end: int | None = None

    for cursor in range(start_index, len(lines)):
        line = lines[cursor]
        marker = _marker(line)
        cities = matcher.cities_in(line.text)
        same_sequence_family = marker is not None and (
            marker.kind == marker_kind
            or (
                marker_kind == "ordinal"
                and marker.kind == "transition"
            )
        )
        if same_sequence_family:
            actual_marker: int | str = marker.value
            if marker_kind == "ordinal" and marker.value == "then":
                actual_marker = 2
            elif marker_kind == "ordinal" and marker.value == "finally":
                actual_marker = expected
            if finally_seen or actual_marker != expected:
                return _candidate_result(
                    terminated=True,
                    rejection_reason=None,
                    termination_kind=(
                        "marker_restart"
                        if actual_marker == 1
                        else "marker_sequence_break"
                    ),
                    marker_kind=marker_kind,
                    markers=markers,
                    item_cities=item_cities,
                    item_lines=item_lines,
                    item_starts=item_starts,
                    item_ends=item_ends,
                    item_boundaries=item_boundaries,
                    matched=matched,
                    duplicate_items=duplicate_items,
                    bridges=bridges,
                    trailing_lines=pending_bridges,
                    start_char=lines[start_index].start,
                    cut_char=last_end,
                )
            if len(cities) != 1:
                if markers and (len(markers) >= 2 or gold_city_count == 1):
                    return _candidate_result(
                        terminated=True,
                        rejection_reason=None,
                        termination_kind="expected_marker_not_gold_city",
                        marker_kind=marker_kind,
                        markers=markers,
                        item_cities=item_cities,
                        item_lines=item_lines,
                        item_starts=item_starts,
                        item_ends=item_ends,
                        item_boundaries=item_boundaries,
                        matched=matched,
                        duplicate_items=duplicate_items,
                        bridges=bridges,
                        trailing_lines=pending_bridges,
                        start_char=lines[start_index].start,
                        cut_char=last_end,
                    )
                return _candidate_result(
                    terminated=False,
                    rejection_reason="expected_marker_not_single_gold_city",
                    termination_kind=None,
                    marker_kind=marker_kind,
                    markers=markers,
                    item_cities=item_cities,
                    item_lines=item_lines,
                    item_starts=item_starts,
                    item_ends=item_ends,
                    item_boundaries=item_boundaries,
                    matched=matched,
                    duplicate_items=duplicate_items,
                    bridges=bridges,
                    trailing_lines=pending_bridges,
                    start_char=lines[start_index].start,
                    cut_char=None,
                )
            if not line.has_boundary:
                return _candidate_result(
                    terminated=False,
                    rejection_reason="city_item_has_no_ending_boundary",
                    termination_kind=None,
                    marker_kind=marker_kind,
                    markers=markers,
                    item_cities=item_cities,
                    item_lines=item_lines,
                    item_starts=item_starts,
                    item_ends=item_ends,
                    item_boundaries=item_boundaries,
                    matched=matched,
                    duplicate_items=duplicate_items,
                    bridges=bridges,
                    trailing_lines=pending_bridges,
                    start_char=lines[start_index].start,
                    cut_char=None,
                )
            bridges += pending_bridges
            pending_bridges = 0
            pending_nonblank = False
            city = cities[0]
            markers.append(actual_marker)
            item_cities.append(city)
            item_lines.append(line.physical_line_number)
            item_starts.append(line.start)
            item_ends.append(line.content_end)
            item_boundaries.append(str(line.boundary_kind))
            if not _append_city(city, seen=seen, matched=matched):
                duplicate_items += 1
            expected += 1
            finally_seen = marker.value == "finally"
            last_end = line.end
            continue

        if (
            marker is not None
            and marker.kind != "transition"
            and cities
        ):
            return _candidate_result(
                terminated=True,
                rejection_reason=None,
                termination_kind="gold_city_marker_change",
                marker_kind=marker_kind,
                markers=markers,
                item_cities=item_cities,
                item_lines=item_lines,
                item_starts=item_starts,
                item_ends=item_ends,
                item_boundaries=item_boundaries,
                matched=matched,
                duplicate_items=duplicate_items,
                bridges=bridges,
                trailing_lines=pending_bridges,
                start_char=lines[start_index].start,
                cut_char=last_end,
            )

        pending_bridges += 1
        pending_nonblank = pending_nonblank or bool(line.text.strip())

    terminated = bool(markers) and (
        reasoning_closed or pending_nonblank or finally_seen
    )
    return _candidate_result(
        terminated=terminated,
        rejection_reason=(None if terminated else "no_list_termination_evidence"),
        termination_kind=(
            "thinking_close"
            if terminated and reasoning_closed
            else "final_transition"
            if terminated and finally_seen
            else "trailing_prose"
            if terminated
            else None
        ),
        marker_kind=marker_kind,
        markers=markers,
        item_cities=item_cities,
        item_lines=item_lines,
        item_starts=item_starts,
        item_ends=item_ends,
        item_boundaries=item_boundaries,
        matched=matched,
        duplicate_items=duplicate_items,
        bridges=bridges,
        trailing_lines=pending_bridges,
        start_char=lines[start_index].start,
        cut_char=last_end if terminated else None,
    )


def _scan_bullet_candidate(
    lines: Sequence[_Line],
    *,
    start_index: int,
    matcher: _GoldCityMatcher,
    reasoning_closed: bool,
    gold_city_count: int,
) -> _Candidate:
    markers: list[int | str] = []
    item_cities: list[str] = []
    item_lines: list[int] = []
    item_starts: list[int] = []
    item_ends: list[int] = []
    item_boundaries: list[str] = []
    matched: list[str] = []
    seen: set[str] = set()
    duplicate_items = 0
    bridges = 0
    pending_bridges = 0
    pending_nonblank = False
    last_end: int | None = None

    for cursor in range(start_index, len(lines)):
        line = lines[cursor]
        marker = _marker(line)
        cities = matcher.cities_in(line.text)
        if marker is not None and marker.kind == "bullet":
            if len(cities) != 1:
                if len(markers) < 2 and gold_city_count > 1:
                    return _candidate_result(
                        terminated=False,
                        rejection_reason=(
                            "single_city_item_before_non_gold_bullet"
                        ),
                        termination_kind=None,
                        marker_kind="bullet",
                        markers=markers,
                        item_cities=item_cities,
                        item_lines=item_lines,
                        item_starts=item_starts,
                        item_ends=item_ends,
                        item_boundaries=item_boundaries,
                        matched=matched,
                        duplicate_items=duplicate_items,
                        bridges=bridges,
                        trailing_lines=pending_bridges,
                        start_char=lines[start_index].start,
                        cut_char=None,
                    )
                return _candidate_result(
                    terminated=True,
                    rejection_reason=None,
                    termination_kind="next_non_gold_bullet",
                    marker_kind="bullet",
                    markers=markers,
                    item_cities=item_cities,
                    item_lines=item_lines,
                    item_starts=item_starts,
                    item_ends=item_ends,
                    item_boundaries=item_boundaries,
                    matched=matched,
                    duplicate_items=duplicate_items,
                    bridges=bridges,
                    trailing_lines=pending_bridges,
                    start_char=lines[start_index].start,
                    cut_char=last_end,
                )
            if not line.has_boundary:
                return _candidate_result(
                    terminated=False,
                    rejection_reason="city_item_has_no_ending_boundary",
                    termination_kind=None,
                    marker_kind="bullet",
                    markers=markers,
                    item_cities=item_cities,
                    item_lines=item_lines,
                    item_starts=item_starts,
                    item_ends=item_ends,
                    item_boundaries=item_boundaries,
                    matched=matched,
                    duplicate_items=duplicate_items,
                    bridges=bridges,
                    trailing_lines=pending_bridges,
                    start_char=lines[start_index].start,
                    cut_char=None,
                )
            bridges += pending_bridges
            pending_bridges = 0
            pending_nonblank = False
            city = cities[0]
            markers.append(marker.value)
            item_cities.append(city)
            item_lines.append(line.physical_line_number)
            item_starts.append(line.start)
            item_ends.append(line.content_end)
            item_boundaries.append(str(line.boundary_kind))
            if not _append_city(city, seen=seen, matched=matched):
                duplicate_items += 1
            last_end = line.end
            continue

        if (
            marker is not None
            and marker.kind != "transition"
            and cities
        ):
            return _candidate_result(
                terminated=True,
                rejection_reason=None,
                termination_kind="gold_city_marker_change",
                marker_kind="bullet",
                markers=markers,
                item_cities=item_cities,
                item_lines=item_lines,
                item_starts=item_starts,
                item_ends=item_ends,
                item_boundaries=item_boundaries,
                matched=matched,
                duplicate_items=duplicate_items,
                bridges=bridges,
                trailing_lines=pending_bridges,
                start_char=lines[start_index].start,
                cut_char=last_end,
            )

        pending_bridges += 1
        pending_nonblank = pending_nonblank or bool(line.text.strip())

    terminated = bool(markers) and (reasoning_closed or pending_nonblank)
    return _candidate_result(
        terminated=terminated,
        rejection_reason=(None if terminated else "no_list_termination_evidence"),
        termination_kind=(
            "thinking_close"
            if terminated and reasoning_closed
            else "trailing_prose"
            if terminated
            else None
        ),
        marker_kind="bullet",
        markers=markers,
        item_cities=item_cities,
        item_lines=item_lines,
        item_starts=item_starts,
        item_ends=item_ends,
        item_boundaries=item_boundaries,
        matched=matched,
        duplicate_items=duplicate_items,
        bridges=bridges,
        trailing_lines=pending_bridges,
        start_char=lines[start_index].start,
        cut_char=last_end if terminated else None,
    )


def _detected_cut(
    *,
    candidate: _Candidate,
    candidates: Sequence[_Candidate],
    gold_cities: tuple[str, ...],
    reasoning_start: int,
    reasoning_end: int,
    closing_delimiter_present: bool,
) -> CityListTerminationCut:
    if not candidate.terminated or candidate.cut_char is None:
        raise ValueError("Detected result requires a terminated candidate")
    matched_folded = {
        city.casefold() for city in candidate.matched_gold_cities
    }
    missing = tuple(
        city for city in gold_cities if city.casefold() not in matched_folded
    )
    trace_one_to_one, trace_order_class, trace_category = (
        _trace_classification(candidate.item_gold_cities, gold_cities)
    )
    return CityListTerminationCut(
        detected=True,
        status="ok_gold_city_list_terminated",
        all_items_gold_city=True,
        coverage_complete=not missing,
        gold_city_count=len(gold_cities),
        gold_cities=gold_cities,
        coverage_count=candidate.coverage_count,
        coverage_fraction=candidate.coverage_count / len(gold_cities),
        matched_gold_cities=candidate.matched_gold_cities,
        missing_gold_cities=missing,
        marker_kind=candidate.marker_kind,
        item_count=candidate.item_count,
        item_markers=candidate.item_markers,
        item_gold_cities=candidate.item_gold_cities,
        item_line_numbers=candidate.item_line_numbers,
        item_start_chars=tuple(
            reasoning_start + value for value in candidate.item_start_chars
        ),
        item_end_chars=tuple(
            reasoning_start + value for value in candidate.item_end_chars
        ),
        item_boundary_kinds=candidate.item_boundary_kinds,
        duplicate_gold_city_items=candidate.duplicate_gold_city_items,
        trace_one_to_one=trace_one_to_one,
        trace_order_class=trace_order_class,
        trace_category=trace_category,
        bridge_line_count=candidate.bridge_line_count,
        trailing_line_count=candidate.trailing_line_count,
        candidates_considered=len(candidates),
        rejected_candidates=sum(
            value.rejection_reason is not None for value in candidates
        ),
        termination_kind=candidate.termination_kind,
        reasoning_start_char=reasoning_start,
        reasoning_end_char=reasoning_end,
        list_start_char=reasoning_start + candidate.start_char,
        cut_char=reasoning_start + candidate.cut_char,
        boundary_kind="gold_city_list_termination",
        closing_delimiter_present=closing_delimiter_present,
    )


def find_first_terminated_gold_city_list(
    raw_text: str,
    *,
    model_family: str,
    gold_records: Iterable[dict[str, Any] | str],
) -> CityListTerminationCut:
    """Find the first terminated list whose accepted items are gold cities.

    Unlike V16, distinct-city coverage may be smaller than the registered N.
    Numeric and English-ordinal streams must start at 1/First and increment
    strictly. ``Then`` is accepted only as item 2 and ``Finally`` as the last
    sequential item. ``-``, ``*``, and ``•`` are unnumbered bullets. Prose,
    blank lines, and non-gold section headings may bridge chunks. Each item
    must end at a physical newline or a conservative sentence-period boundary;
    the cut retains that exact boundary after the last accepted city item.

    If no explicit-marker list terminates, two conservative implicit formats
    are allowed: a compact, count-signalled recap in the final 55% of the
    reasoning, or exact V4.4 audit-record template sentences.  These fallbacks
    still accept only registered gold-city items, preserve repeats, and do not
    require full coverage.
    """

    gold_cities = _gold_cities(gold_records)
    matcher = _GoldCityMatcher(gold_cities)
    span = locate_reasoning_span(raw_text, model_family=model_family)
    reasoning = raw_text[span.start : span.end]
    lines = _lines_with_offsets(reasoning, matcher=matcher)
    candidates: list[_Candidate] = []

    for start_index, line in enumerate(lines):
        marker = _marker(line)
        cities = matcher.cities_in(line.text)
        if marker is None or len(cities) != 1 or not line.has_boundary:
            continue
        candidate: _Candidate | None = None
        if marker.kind in {"indexed", "ordinal"} and marker.value == 1:
            candidate = _scan_sequenced_candidate(
                lines,
                start_index=start_index,
                marker_kind=marker.kind,
                matcher=matcher,
                reasoning_closed=span.closing_delimiter_present,
                gold_city_count=len(gold_cities),
            )
        elif marker.kind == "bullet":
            candidate = _scan_bullet_candidate(
                lines,
                start_index=start_index,
                matcher=matcher,
                reasoning_closed=span.closing_delimiter_present,
                gold_city_count=len(gold_cities),
            )
        if candidate is None:
            continue
        if (
            candidate.marker_kind == "ordinal"
            and candidate.item_count < 2
            and len(gold_cities) > 1
        ):
            candidate = replace(
                candidate,
                terminated=False,
                rejection_reason="ordinal_sequence_requires_two_items",
                termination_kind=None,
                cut_char=None,
            )
        candidates.append(candidate)
        if candidate.terminated and candidate.cut_char is not None:
            return _detected_cut(
                candidate=candidate,
                candidates=candidates,
                gold_cities=gold_cities,
                reasoning_start=span.start,
                reasoning_end=span.end,
                closing_delimiter_present=span.closing_delimiter_present,
            )

    # Preserve the first explicit numbered/bulleted/ordinal list whenever one
    # exists.  Only no-hit traces reach the two implicit structured formats.
    for fallback in (
        _late_completion_recap_candidate(
            reasoning,
            gold_cities=gold_cities,
        ),
        _audit_sentence_chain_candidate(
            reasoning,
            gold_cities=gold_cities,
        ),
    ):
        if fallback is None:
            continue
        candidates.append(fallback)
        return _detected_cut(
            candidate=fallback,
            candidates=candidates,
            gold_cities=gold_cities,
            reasoning_start=span.start,
            reasoning_end=span.end,
            closing_delimiter_present=span.closing_delimiter_present,
        )

    best = max(
        candidates,
        key=lambda candidate: (
            candidate.coverage_count,
            candidate.item_count,
            -candidate.start_char,
        ),
        default=None,
    )
    matched = best.matched_gold_cities if best is not None else ()
    matched_folded = {city.casefold() for city in matched}
    missing = tuple(
        city for city in gold_cities if city.casefold() not in matched_folded
    )
    coverage_count = len(matched)
    trace_one_to_one, trace_order_class, trace_category = _trace_classification(
        best.item_gold_cities if best is not None else (),
        gold_cities,
    )
    return CityListTerminationCut(
        detected=False,
        status="no_terminated_gold_city_list",
        all_items_gold_city=False,
        coverage_complete=False,
        gold_city_count=len(gold_cities),
        gold_cities=gold_cities,
        coverage_count=coverage_count,
        coverage_fraction=coverage_count / len(gold_cities),
        matched_gold_cities=matched,
        missing_gold_cities=missing,
        marker_kind=best.marker_kind if best is not None else None,
        item_count=best.item_count if best is not None else 0,
        item_markers=best.item_markers if best is not None else (),
        item_gold_cities=best.item_gold_cities if best is not None else (),
        item_line_numbers=best.item_line_numbers if best is not None else (),
        item_start_chars=(
            tuple(span.start + value for value in best.item_start_chars)
            if best is not None
            else ()
        ),
        item_end_chars=(
            tuple(span.start + value for value in best.item_end_chars)
            if best is not None
            else ()
        ),
        item_boundary_kinds=(
            best.item_boundary_kinds if best is not None else ()
        ),
        duplicate_gold_city_items=(
            best.duplicate_gold_city_items if best is not None else 0
        ),
        trace_one_to_one=trace_one_to_one,
        trace_order_class=trace_order_class,
        trace_category=trace_category,
        bridge_line_count=best.bridge_line_count if best is not None else 0,
        trailing_line_count=best.trailing_line_count if best is not None else 0,
        candidates_considered=len(candidates),
        rejected_candidates=sum(
            candidate.rejection_reason is not None for candidate in candidates
        ),
        reasoning_start_char=span.start,
        reasoning_end_char=span.end,
        list_start_char=(
            span.start + best.start_char if best is not None else None
        ),
        closing_delimiter_present=span.closing_delimiter_present,
    )
