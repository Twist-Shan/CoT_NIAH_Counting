"""Rank-supported native-thinking episode parsing.

This module extracts local ``gold city + running-rank evidence`` events and
segments them into contiguous 1..M episodes.  It deliberately does not inspect
the registered gold count or the final ``Total`` when building or selecting an
episode.  Those values remain downstream audit fields only.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from realistic_niah_v3.first_list_cutoff import locate_reasoning_span


EPISODE_SCHEMA_VERSION = "realistic_niah_v5_rank_episode_v1"
EPISODE_SELECTION_POLICY = (
    "segment every contiguous rank-supported 1..M sequence at each new rank-1; "
    "select the greatest terminal M and break ties by earliest semantic start; "
    "never use registered gold N or final Total for construction, padding, or "
    "selection"
)

_CARDINAL_VALUES = {
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
_ORDINAL_VALUES = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
}
_CARDINAL_WORDS = "|".join(
    sorted(_CARDINAL_VALUES, key=len, reverse=True)
)
_ORDINAL_WORDS = "|".join(
    sorted(_ORDINAL_VALUES, key=len, reverse=True)
)

_PREFIX_WRAPPER = r"^[ \t]*(?:>[ \t]*)?(?:[-*\u2022][ \t]+)?(?:[*_`]{0,2})"
_PREFIX_INDEX_RE = re.compile(
    _PREFIX_WRAPPER + r"(?P<value>[1-9]\d?)[.)](?=[ \t*`_]|$)",
    re.IGNORECASE,
)
_PREFIX_ORDINAL_RE = re.compile(
    _PREFIX_WRAPPER
    + r"(?P<value>"
    + _ORDINAL_WORDS
    + r")(?:ly)?(?:[*_`]{0,2})(?=[ \t.,:;)-]|$)",
    re.IGNORECASE,
)
_LABELED_NUMERIC_RE = re.compile(
    r"(?<!\w)(?P<label>count|record|entry|instance|item|found)"
    r"(?:[ \t]+(?:number|no\.?))?[ \t]*(?:[:=#-][ \t]*)?"
    r"(?P<value>[1-9]\d?)(?!\d)",
    re.IGNORECASE,
)
_NUMERIC_SUMMARY_RE = re.compile(
    r"(?<!\w)(?:count|total(?:[ \t]+count)?)[ \t]+"
    r"(?:is|was|would[ \t]+be|should[ \t]+be|comes?[ \t]+to)[ \t]+"
    r"(?P<value>[1-9]\d?)(?!\d)",
    re.IGNORECASE,
)
_TRAILING_ORDINAL_RE = re.compile(
    r"(?<!\w)(?:that(?:'|\u2019)s|that[ \t]+is|this[ \t]+is)"
    r"[ \t]+(?:the[ \t]+)?(?P<value>"
    + _ORDINAL_WORDS
    + r")(?:ly)?(?:[ \t]+(?:record|entry|item|city|instance))?\b",
    re.IGNORECASE,
)
_THATS_CARDINAL_RE = re.compile(
    r"(?<!\w)(?:that(?:'|\u2019)s|that[ \t]+is)[ \t]+"
    r"(?P<value>"
    + _CARDINAL_WORDS
    + r")(?:[ \t]+(?:record|entry|item|city|instance)s?)?\b",
    re.IGNORECASE,
)
_STANDALONE_ORDINAL_RE = re.compile(
    _PREFIX_WRAPPER
    + r"(?P<value>"
    + _ORDINAL_WORDS
    + r")(?:ly)?(?:[*_`]{0,2})(?:[ \t]+(?:record|entry|item|city|instance))?"
    r"[ \t]*[.!?:,)]?[ \t*`_]*$",
    re.IGNORECASE,
)
_STANDALONE_CARDINAL_RE = re.compile(
    _PREFIX_WRAPPER
    + r"(?P<value>"
    + _CARDINAL_WORDS
    + r")(?:[*_`]{0,2})(?:[ \t]+(?:record|entry|item|city|instance))?"
    r"[ \t]*[.!?:,)]?[ \t*`_]*$",
    re.IGNORECASE,
)
_SENTENCE_END_RE = re.compile(
    r"[.!?](?:[\"'\u2019)\]}`*_]+)?(?=[ \t]|$)"
)


@dataclass(frozen=True)
class TextUnit:
    start: int
    end: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RankEvidence:
    rank: int
    start: int
    end: int
    kind: str
    family: str
    priority: int
    surface: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CityOccurrence:
    city: str
    start: int
    end: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunningCountEvent:
    rank: int
    city: str
    city_start_char: int
    city_end_char: int
    city_unit_start_char: int
    city_unit_end_char: int
    rank_evidence_start_char: int
    rank_evidence_end_char: int
    semantic_start_char: int
    semantic_end_char: int
    evidence_kind: str
    evidence_family: str
    evidence_surface: str
    evidence_priority: int
    association: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunningCountSequence:
    sequence_index: int
    events: tuple[RunningCountEvent, ...]

    @property
    def terminal_rank(self) -> int:
        return int(self.events[-1].rank)

    @property
    def start_char(self) -> int:
        return min(event.semantic_start_char for event in self.events)

    @property
    def end_char(self) -> int:
        return max(event.semantic_end_char for event in self.events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_index": self.sequence_index,
            "terminal_rank": self.terminal_rank,
            "event_count": len(self.events),
            "start_char": self.start_char,
            "end_char": self.end_char,
            "cities": [event.city for event in self.events],
            "evidence_kinds": [event.evidence_kind for event in self.events],
            "events": [event.to_dict() for event in self.events],
        }


@dataclass(frozen=True)
class RankEpisodeParse:
    model_family: str
    reasoning_start_char: int
    reasoning_end_char: int
    closing_delimiter_present: bool
    events: tuple[RunningCountEvent, ...]
    sequences: tuple[RunningCountSequence, ...]
    selected_sequence_index: int | None

    @property
    def selected_sequence(self) -> RunningCountSequence | None:
        if self.selected_sequence_index is None:
            return None
        return self.sequences[self.selected_sequence_index]

    def to_dict(self) -> dict[str, Any]:
        selected = self.selected_sequence
        return {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "selection_policy": EPISODE_SELECTION_POLICY,
            "model_family": self.model_family,
            "reasoning_start_char": self.reasoning_start_char,
            "reasoning_end_char": self.reasoning_end_char,
            "closing_delimiter_present": self.closing_delimiter_present,
            "rank_supported_event_count": len(self.events),
            "raw_sequence_count": len(self.sequences),
            "selected_sequence_index": self.selected_sequence_index,
            "selected_terminal_rank": (
                selected.terminal_rank if selected is not None else None
            ),
            "selected_event_count": len(selected.events) if selected else 0,
            "events": [event.to_dict() for event in self.events],
            "sequences": [sequence.to_dict() for sequence in self.sequences],
        }


def _trimmed_unit(text: str, start: int, end: int) -> TextUnit | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return TextUnit(start=start, end=end, text=text[start:end])


def text_units(text: str, *, offset: int = 0) -> tuple[TextUnit, ...]:
    """Split physical lines and conservative sentence endpoints with offsets."""

    units: list[TextUnit] = []
    line_start = 0
    for newline in re.finditer(r"\r\n|\r|\n", text):
        line_end = newline.start()
        units.extend(_line_units(text, line_start, line_end, offset=offset))
        line_start = newline.end()
    units.extend(_line_units(text, line_start, len(text), offset=offset))
    return tuple(units)


def _line_units(
    text: str, line_start: int, line_end: int, *, offset: int
) -> list[TextUnit]:
    result: list[TextUnit] = []
    cursor = line_start
    line = text[line_start:line_end]
    for match in _SENTENCE_END_RE.finditer(line):
        end = line_start + match.end()
        unit = _trimmed_unit(text, cursor, end)
        if unit is not None:
            result.append(
                TextUnit(
                    start=offset + unit.start,
                    end=offset + unit.end,
                    text=unit.text,
                )
            )
        cursor = end
    unit = _trimmed_unit(text, cursor, line_end)
    if unit is not None:
        result.append(
            TextUnit(
                start=offset + unit.start,
                end=offset + unit.end,
                text=unit.text,
            )
        )
    return result


def _value(match: re.Match[str], words: dict[str, int] | None) -> int:
    spelling = match.group("value").casefold()
    return int(spelling) if words is None else int(words[spelling])


def _extended_end(unit_text: str, end: int) -> int:
    while end < len(unit_text) and unit_text[end] in " \t*`_\"'\u2019":
        end += 1
    while end < len(unit_text) and unit_text[end] in ".,:;)]":
        end += 1
    return end


def rank_evidence(unit: TextUnit) -> tuple[RankEvidence, ...]:
    specs: tuple[
        tuple[re.Pattern[str], dict[str, int] | None, str, str, int], ...
    ] = (
        (_PREFIX_INDEX_RE, None, "indexed_prefix", "indexed", 7),
        (_LABELED_NUMERIC_RE, None, "labeled_rank", "inline_count", 8),
        (_TRAILING_ORDINAL_RE, _ORDINAL_VALUES, "trailing_ordinal", "ordinal", 7),
        (_THATS_CARDINAL_RE, _CARDINAL_VALUES, "cardinal_commit", "inline_count", 6),
        (_NUMERIC_SUMMARY_RE, None, "numeric_commit", "inline_count", 5),
        (_PREFIX_ORDINAL_RE, _ORDINAL_VALUES, "ordinal_prefix", "ordinal", 5),
        (
            _STANDALONE_ORDINAL_RE,
            _ORDINAL_VALUES,
            "standalone_ordinal",
            "ordinal",
            4,
        ),
        (
            _STANDALONE_CARDINAL_RE,
            _CARDINAL_VALUES,
            "standalone_cardinal",
            "inline_count",
            3,
        ),
    )
    hits: list[RankEvidence] = []
    for pattern, words, kind, family, priority in specs:
        for match in pattern.finditer(unit.text):
            rank = _value(match, words)
            if not 1 <= rank <= 20:
                continue
            local_end = _extended_end(unit.text, match.end())
            hits.append(
                RankEvidence(
                    rank=rank,
                    start=unit.start + match.start(),
                    end=unit.start + local_end,
                    kind=kind,
                    family=family,
                    priority=priority,
                    surface=unit.text[match.start():local_end],
                )
            )
    unique: dict[tuple[int, int, int, str], RankEvidence] = {}
    for hit in hits:
        unique[(hit.rank, hit.start, hit.end, hit.kind)] = hit
    return tuple(
        sorted(
            unique.values(),
            key=lambda hit: (hit.start, hit.end, -hit.priority, hit.kind),
        )
    )


def _city_occurrences(
    unit: TextUnit, gold_cities: Sequence[str]
) -> tuple[CityOccurrence, ...]:
    canonical = {city.casefold(): city for city in gold_cities}
    if not canonical:
        return ()
    alternatives = "|".join(
        re.escape(city) for city in sorted(canonical.values(), key=len, reverse=True)
    )
    pattern = re.compile(
        rf"(?<!\w)(?P<city>{alternatives})(?!\w)", re.IGNORECASE
    )
    return tuple(
        CityOccurrence(
            city=canonical[match.group("city").casefold()],
            start=unit.start + match.start("city"),
            end=unit.start + match.end("city"),
        )
        for match in pattern.finditer(unit.text)
    )


def _unique_rank(evidence: Sequence[RankEvidence]) -> RankEvidence | None:
    if not evidence:
        return None
    ranks = {item.rank for item in evidence}
    if len(ranks) != 1:
        return None
    return max(evidence, key=lambda item: (item.priority, item.end, -item.start))


def _single_city(
    occurrences: Sequence[CityOccurrence],
) -> CityOccurrence | None:
    """Collapse repeated mentions of one city inside a single semantic unit."""

    if not occurrences:
        return None
    if len({item.city.casefold() for item in occurrences}) != 1:
        return None
    return max(occurrences, key=lambda item: (item.end, item.start))


_PREFIX_EVIDENCE_KINDS = frozenset({"indexed_prefix", "ordinal_prefix"})
_POSTFIX_EVIDENCE_KINDS = frozenset(
    {
        "trailing_ordinal",
        "cardinal_commit",
        "numeric_commit",
        "standalone_ordinal",
        "standalone_cardinal",
    }
)


def _association_rank(
    evidence: Sequence[RankEvidence], *, association: str
) -> RankEvidence | None:
    """Choose one unambiguous rank with direction-aware surface preference.

    An isolated ``Count: 2`` can sit on either side of its city and is treated
    as neutral. Prefix numbering is preferred before a city; standalone or
    commitment language is preferred after it. This prevents a trailing count
    for item k from being reassigned as the prefix for item k+1.
    """

    if not evidence:
        return None
    ranks = {item.rank for item in evidence}
    if len(ranks) != 1:
        return None

    return max(
        evidence,
        key=lambda item: (
            _orientation_score(item, association=association),
            item.priority,
            item.end,
            -item.start,
        ),
    )


def _evidence_key(evidence: RankEvidence) -> tuple[int, int, int, str]:
    return evidence.start, evidence.end, evidence.rank, evidence.kind


def _orientation_score(evidence: RankEvidence, *, association: str) -> int:
    if association == "rank_before_city":
        if evidence.kind in _PREFIX_EVIDENCE_KINDS:
            return 2
        return 1 if evidence.kind == "labeled_rank" else 0
    if association == "rank_after_city":
        if evidence.kind in _POSTFIX_EVIDENCE_KINDS:
            return 2
        return 1 if evidence.kind == "labeled_rank" else 0
    return 0


def _adjacent_evidence_eligible(
    evidence: RankEvidence,
    unit: TextUnit,
    *,
    association: str,
    bridge_units: Sequence[TextUnit] = (),
) -> bool:
    if _orientation_score(evidence, association=association) == 0:
        return False
    if (
        association == "rank_before_city"
        and evidence.kind in _PREFIX_EVIDENCE_KINDS
    ):
        # ``4. Count the Records:`` is a procedure heading, not the fourth
        # city. A cross-unit prefix may bridge to a city only when the source
        # unit is effectively marker-only.
        tail = unit.text[evidence.end - unit.start :]
        if re.fullmatch(r"[ \t*`_\"'\u2019():,.;-]*", tail) is None:
            return False
        # Sentence splitting can turn ``4. Count the Records:`` into a
        # marker-only unit followed by an alphabetic heading. Do not bridge
        # over that heading to the first recap city.
        return not any(re.search(r"[A-Za-z]", item.text) for item in bridge_units)
    return True


def _nearby_rank_units(
    *,
    city_index: int,
    units: Sequence[TextUnit],
    cities_by_unit: Sequence[Sequence[CityOccurrence]],
    ranks_by_unit: Sequence[Sequence[RankEvidence]],
    adjacent_gap_chars: int,
) -> tuple[tuple[int, str, int], ...]:
    """Return city-free rank units before/after the city, stopping at a city.

    A short bridge such as ``End excerpt.`` is allowed. Crossing another city
    is not: this keeps a marker local to one candidate update.
    """

    nearby: list[tuple[int, str, int]] = []
    for step, association in (
        (-1, "rank_before_city"),
        (1, "rank_after_city"),
    ):
        neighbor = city_index + step
        while 0 <= neighbor < len(units):
            if cities_by_unit[neighbor]:
                break
            gap = (
                units[city_index].start - units[neighbor].end
                if neighbor < city_index
                else units[neighbor].start - units[city_index].end
            )
            if gap > adjacent_gap_chars:
                break
            if ranks_by_unit[neighbor]:
                nearby.append((neighbor, association, gap))
            neighbor += step
    return tuple(nearby)


def _event(
    city: CityOccurrence,
    city_unit: TextUnit,
    evidence: RankEvidence,
    *,
    association: str,
) -> RunningCountEvent:
    return RunningCountEvent(
        rank=evidence.rank,
        city=city.city,
        city_start_char=city.start,
        city_end_char=city.end,
        city_unit_start_char=city_unit.start,
        city_unit_end_char=city_unit.end,
        rank_evidence_start_char=evidence.start,
        rank_evidence_end_char=evidence.end,
        semantic_start_char=min(city_unit.start, evidence.start),
        semantic_end_char=max(city_unit.end, evidence.end),
        evidence_kind=evidence.kind,
        evidence_family=evidence.family,
        evidence_surface=evidence.surface,
        evidence_priority=evidence.priority,
        association=association,
    )


def extract_rank_supported_events(
    reasoning: str,
    *,
    reasoning_offset: int,
    gold_cities: Sequence[str],
    adjacent_gap_chars: int = 100,
) -> tuple[RunningCountEvent, ...]:
    units = text_units(reasoning, offset=reasoning_offset)
    cities_by_unit = [_city_occurrences(unit, gold_cities) for unit in units]
    ranks_by_unit = [rank_evidence(unit) for unit in units]
    events: list[RunningCountEvent] = []
    consumed: set[tuple[int, int, int, str]] = set()

    # Resolve same-unit events first. They are the strongest associations and
    # must reserve their evidence before any neighboring city can borrow it.
    for index, (unit, cities, evidence) in enumerate(
        zip(units, cities_by_unit, ranks_by_unit)
    ):
        city = _single_city(cities)
        # Multiple distinct cities in one unit are ambiguous. Repeating the
        # same city around a quotation is a single supported update.
        if city is None:
            continue
        direct = _unique_rank(evidence)
        if direct is not None:
            events.append(_event(city, unit, direct, association="same_unit"))
            consumed.update(_evidence_key(item) for item in evidence)

            # A second, city-free sentence often restates the same update
            # (``Record 1: Athens. (Count = 1)`` or ``... That's one``).
            # Reserve that redundant evidence so it cannot become the marker
            # for the following city.
            for neighbor, association, gap in _nearby_rank_units(
                city_index=index,
                units=units,
                cities_by_unit=cities_by_unit,
                ranks_by_unit=ranks_by_unit,
                adjacent_gap_chars=adjacent_gap_chars,
            ):
                eligible = [
                    item
                    for item in ranks_by_unit[neighbor]
                    if _adjacent_evidence_eligible(
                        item,
                        units[neighbor],
                        association=association,
                        bridge_units=units[
                            min(index, neighbor) + 1 : max(index, neighbor)
                        ],
                    )
                ]
                candidate = _association_rank(eligible, association=association)
                if (
                    candidate is None
                    or candidate.rank != direct.rank
                ):
                    continue
                if 0 <= gap <= adjacent_gap_chars:
                    consumed.update(
                        _evidence_key(item)
                        for item in ranks_by_unit[neighbor]
                        if item.rank == direct.rank
                    )

    # Then match city-only units to unused adjacent evidence in textual order.
    # Evidence is single-use, which is what stops ``Count: k`` from labeling
    # both the preceding and following city.
    for index, (unit, cities, evidence) in enumerate(
        zip(units, cities_by_unit, ranks_by_unit)
    ):
        city = _single_city(cities)
        if city is None:
            continue
        direct = _unique_rank(evidence)
        if direct is not None:
            continue
        # Conflicting ranks in the city unit invalidate it; adjacent evidence
        # must not be used to paper over the conflict.
        if evidence:
            continue
        adjacent: list[tuple[int, int, int, RankEvidence, str]] = []
        for neighbor, association, gap in _nearby_rank_units(
            city_index=index,
            units=units,
            cities_by_unit=cities_by_unit,
            ranks_by_unit=ranks_by_unit,
            adjacent_gap_chars=adjacent_gap_chars,
        ):
            available = [
                item
                for item in ranks_by_unit[neighbor]
                if _evidence_key(item) not in consumed
                and _adjacent_evidence_eligible(
                    item,
                    units[neighbor],
                    association=association,
                    bridge_units=units[
                        min(index, neighbor) + 1 : max(index, neighbor)
                    ],
                )
            ]
            candidate = _association_rank(available, association=association)
            if candidate is None:
                continue
            orientation = _orientation_score(candidate, association=association)
            if orientation > 0 and 0 <= gap <= adjacent_gap_chars:
                adjacent.append(
                    (orientation, gap, neighbor, candidate, association)
                )
        if adjacent:
            _orientation, _gap, evidence_unit, candidate, association = min(
                adjacent,
                key=lambda item: (
                    item[1],
                    -item[0],
                    -item[3].priority,
                    item[3].start,
                ),
            )
            events.append(
                _event(city, unit, candidate, association=association)
            )
            consumed.update(
                _evidence_key(item)
                for item in ranks_by_unit[evidence_unit]
                if item.rank == candidate.rank
            )
    deduplicated: dict[
        tuple[int, str, int, int, int, int], RunningCountEvent
    ] = {}
    for event in events:
        key = (
            event.rank,
            event.city.casefold(),
            event.city_start_char,
            event.city_end_char,
            event.rank_evidence_start_char,
            event.rank_evidence_end_char,
        )
        deduplicated[key] = event
    return tuple(
        sorted(
            deduplicated.values(),
            key=lambda event: (
                event.semantic_end_char,
                event.semantic_start_char,
                event.rank,
            ),
        )
    )


def segment_rank_sequences(
    events: Sequence[RunningCountEvent],
) -> tuple[RunningCountSequence, ...]:
    raw: list[list[RunningCountEvent]] = []
    current: list[RunningCountEvent] = []
    for event in events:
        if event.rank == 1:
            if current and current[-1].rank == 1 and (
                current[-1].city.casefold() == event.city.casefold()
            ):
                # Equivalent repeated evidence for the same update: keep the
                # later, usually more explicit commitment site.
                current[-1] = event
                continue
            if current:
                raw.append(current)
            current = [event]
            continue
        if not current:
            continue
        previous = current[-1]
        if event.rank == previous.rank and (
            event.city.casefold() == previous.city.casefold()
        ):
            current[-1] = event
        elif event.rank == previous.rank + 1:
            current.append(event)
        # A jump, stale rank, or conflicting repeated rank is ignored. A later
        # expected event may still continue this sequence, as in the senior
        # episode parser.
    if current:
        raw.append(current)
    return tuple(
        RunningCountSequence(sequence_index=index, events=tuple(sequence))
        for index, sequence in enumerate(raw)
    )


def parse_rank_episodes(
    raw_text: str,
    *,
    model_family: str,
    gold_cities: Sequence[str],
) -> RankEpisodeParse:
    span = locate_reasoning_span(raw_text, model_family=model_family)
    reasoning = raw_text[span.start:span.end]
    events = extract_rank_supported_events(
        reasoning,
        reasoning_offset=span.start,
        gold_cities=gold_cities,
    )
    sequences = segment_rank_sequences(events)
    selected_index: int | None = None
    if sequences:
        selected = max(
            sequences,
            key=lambda sequence: (
                sequence.terminal_rank,
                -sequence.start_char,
            ),
        )
        selected_index = selected.sequence_index
    return RankEpisodeParse(
        model_family=model_family,
        reasoning_start_char=span.start,
        reasoning_end_char=span.end,
        closing_delimiter_present=span.closing_delimiter_present,
        events=events,
        sequences=sequences,
        selected_sequence_index=selected_index,
    )


def find_rank_evidence_span(
    text: str, *, expected_rank: int
) -> RankEvidence | None:
    """Find the latest strongest rank evidence inside one selected item span."""

    candidates: list[RankEvidence] = []
    for unit in text_units(text):
        candidates.extend(
            hit for hit in rank_evidence(unit) if hit.rank == expected_rank
        )
    if not candidates:
        return None
    return max(candidates, key=lambda hit: (hit.end, hit.priority, -hit.start))


def find_city_unit_span(
    text: str,
    *,
    city_start: int,
    city_end: int,
) -> tuple[int, int] | None:
    for unit in text_units(text):
        if unit.start <= city_start and city_end <= unit.end:
            return unit.start, unit.end
    return None
