from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from .first_list_cutoff import locate_reasoning_span


INDEXED_ITEM_RE = re.compile(r"^[ \t]*(\d+)[.)][ \t]+\S.*$")
BULLET_ITEM_RE = re.compile(r"^[ \t]*-[ \t]+\S.*$")


@dataclass(frozen=True)
class _Line:
    text: str
    start: int
    end: int
    has_newline: bool


@dataclass(frozen=True)
class _Candidate:
    marker_kind: str
    item_markers: tuple[int | str, ...]
    item_gold_cities: tuple[str, ...]
    item_line_numbers: tuple[int, ...]
    matched_gold_cities: tuple[str, ...]
    duplicate_gold_city_items: int
    bridge_line_count: int
    start_char: int
    cut_char: int | None

    @property
    def coverage_count(self) -> int:
        return len(self.matched_gold_cities)

    @property
    def item_count(self) -> int:
        return len(self.item_markers)


@dataclass(frozen=True)
class GoldCityListCut:
    detected: bool
    status: str
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
    duplicate_gold_city_items: int = 0
    bridge_line_count: int = 0
    candidates_considered: int = 0
    reasoning_start_char: int | None = None
    reasoning_end_char: int | None = None
    list_start_char: int | None = None
    cut_char: int | None = None
    boundary_kind: str | None = None
    closing_delimiter_present: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _lines_with_offsets(text: str) -> list[_Line]:
    lines: list[_Line] = []
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        end = offset + len(raw_line)
        lines.append(
            _Line(
                text=raw_line.rstrip("\r\n"),
                start=offset,
                end=end,
                has_newline=raw_line.endswith(("\n", "\r")),
            )
        )
        offset = end
    return lines


def _index_marker(line: _Line) -> int | None:
    match = INDEXED_ITEM_RE.fullmatch(line.text)
    return int(match.group(1)) if match else None


def _is_bullet(line: _Line) -> bool:
    return BULLET_ITEM_RE.fullmatch(line.text) is not None


def _gold_cities(
    gold_records: Iterable[dict[str, Any] | str],
) -> tuple[str, ...]:
    cities: list[str] = []
    for record in gold_records:
        city = record if isinstance(record, str) else record.get("city")
        city = str(city).strip() if city is not None else ""
        if not city:
            raise ValueError("Every gold record must have a non-empty city")
        cities.append(city)
    if not cities:
        raise ValueError("At least one gold record is required")
    folded = [city.casefold() for city in cities]
    if len(folded) != len(set(folded)):
        raise ValueError("Gold city names must be unique, case-insensitively")
    return tuple(cities)


class _GoldCityMatcher:
    def __init__(self, cities: Sequence[str]) -> None:
        # Longer names win when one registered city is a substring of another.
        self._patterns = [
            (
                city,
                re.compile(
                    rf"(?<!\w){re.escape(city)}(?!\w)",
                    flags=re.IGNORECASE,
                ),
            )
            for city in sorted(cities, key=lambda value: (-len(value), value))
        ]

    def cities_in(self, text: str) -> tuple[str, ...]:
        accepted: list[tuple[int, int, str]] = []
        for city, pattern in self._patterns:
            for match in pattern.finditer(text):
                span = (match.start(), match.end())
                if any(
                    span[0] < existing_end and existing_start < span[1]
                    for existing_start, existing_end, _ in accepted
                ):
                    continue
                accepted.append((span[0], span[1], city))
        accepted.sort(key=lambda value: (value[0], value[1]))
        ordered: list[str] = []
        seen: set[str] = set()
        for _, _, city in accepted:
            folded = city.casefold()
            if folded not in seen:
                seen.add(folded)
                ordered.append(city)
        return tuple(ordered)


def _add_city(
    city: str,
    *,
    seen_folded: set[str],
    matched: list[str],
) -> bool:
    folded = city.casefold()
    if folded in seen_folded:
        return False
    seen_folded.add(folded)
    matched.append(city)
    return True


def _scan_indexed_candidate(
    lines: Sequence[_Line],
    *,
    start_index: int,
    matcher: _GoldCityMatcher,
    gold_city_count: int,
) -> _Candidate:
    markers: list[int | str] = []
    item_cities: list[str] = []
    item_lines: list[int] = []
    matched: list[str] = []
    seen: set[str] = set()
    duplicate_items = 0
    bridges = 0
    expected = 1
    cursor = start_index
    cut_char: int | None = None

    while cursor < len(lines):
        line = lines[cursor]
        marker = _index_marker(line)
        cities = matcher.cities_in(line.text)
        if marker is not None:
            if marker != expected or len(cities) != 1 or not line.has_newline:
                break
            city = cities[0]
            markers.append(marker)
            item_cities.append(city)
            item_lines.append(cursor + 1)
            if not _add_city(city, seen_folded=seen, matched=matched):
                duplicate_items += 1
            expected += 1
            if len(matched) == gold_city_count:
                cut_char = line.end
                break
        elif _is_bullet(line) and cities:
            # A gold-bearing item with another marker starts another list.
            break
        else:
            # Blank text, prose, and non-gold headings may bridge list chunks.
            bridges += 1
        cursor += 1

    return _Candidate(
        marker_kind="indexed",
        item_markers=tuple(markers),
        item_gold_cities=tuple(item_cities),
        item_line_numbers=tuple(item_lines),
        matched_gold_cities=tuple(matched),
        duplicate_gold_city_items=duplicate_items,
        bridge_line_count=bridges,
        start_char=lines[start_index].start,
        cut_char=cut_char,
    )


def _scan_bullet_candidate(
    lines: Sequence[_Line],
    *,
    start_index: int,
    matcher: _GoldCityMatcher,
    gold_city_count: int,
) -> _Candidate:
    markers: list[int | str] = []
    item_cities: list[str] = []
    item_lines: list[int] = []
    matched: list[str] = []
    seen: set[str] = set()
    duplicate_items = 0
    bridges = 0
    cursor = start_index
    cut_char: int | None = None

    while cursor < len(lines):
        line = lines[cursor]
        cities = matcher.cities_in(line.text)
        if _is_bullet(line):
            if len(cities) != 1 or not line.has_newline:
                break
            city = cities[0]
            markers.append("-")
            item_cities.append(city)
            item_lines.append(cursor + 1)
            if not _add_city(city, seen_folded=seen, matched=matched):
                duplicate_items += 1
            if len(matched) == gold_city_count:
                cut_char = line.end
                break
        elif _index_marker(line) is not None and cities:
            break
        else:
            bridges += 1
        cursor += 1

    return _Candidate(
        marker_kind="bullet",
        item_markers=tuple(markers),
        item_gold_cities=tuple(item_cities),
        item_line_numbers=tuple(item_lines),
        matched_gold_cities=tuple(matched),
        duplicate_gold_city_items=duplicate_items,
        bridge_line_count=bridges,
        start_char=lines[start_index].start,
        cut_char=cut_char,
    )


def find_first_gold_city_complete_list(
    raw_text: str,
    *,
    model_family: str,
    gold_records: Iterable[dict[str, Any] | str],
) -> GoldCityListCut:
    """Find the first visible list that covers every registered gold city.

    This is intentionally an oracle/gold-assisted parser. A numbered or
    hyphen item is a counting item only when its text contains exactly one
    registered city. Numbered candidates start at 1 and increment strictly.
    Blank/prose/non-gold heading lines may bridge chunks, allowing a list such
    as 1--4, explanatory prose, then 5--9 to remain one enumeration. The cut
    is the newline after the item that first completes full distinct-city
    coverage. Partial coverage is never intervention-eligible.
    """

    cities = _gold_cities(gold_records)
    matcher = _GoldCityMatcher(cities)
    span = locate_reasoning_span(raw_text, model_family=model_family)
    reasoning = raw_text[span.start : span.end]
    lines = _lines_with_offsets(reasoning)
    candidates: list[_Candidate] = []

    for start_index, line in enumerate(lines):
        line_cities = matcher.cities_in(line.text)
        marker = _index_marker(line)
        candidate: _Candidate | None = None
        if marker == 1 and len(line_cities) == 1 and line.has_newline:
            candidate = _scan_indexed_candidate(
                lines,
                start_index=start_index,
                matcher=matcher,
                gold_city_count=len(cities),
            )
        elif _is_bullet(line) and len(line_cities) == 1 and line.has_newline:
            candidate = _scan_bullet_candidate(
                lines,
                start_index=start_index,
                matcher=matcher,
                gold_city_count=len(cities),
            )
        if candidate is None:
            continue
        candidates.append(candidate)
        if candidate.cut_char is not None:
            missing: tuple[str, ...] = ()
            return GoldCityListCut(
                detected=True,
                status="ok_gold_city_coverage_complete",
                coverage_complete=True,
                gold_city_count=len(cities),
                gold_cities=cities,
                coverage_count=candidate.coverage_count,
                coverage_fraction=1.0,
                matched_gold_cities=candidate.matched_gold_cities,
                missing_gold_cities=missing,
                marker_kind=candidate.marker_kind,
                item_count=candidate.item_count,
                item_markers=candidate.item_markers,
                item_gold_cities=candidate.item_gold_cities,
                item_line_numbers=candidate.item_line_numbers,
                duplicate_gold_city_items=(
                    candidate.duplicate_gold_city_items
                ),
                bridge_line_count=candidate.bridge_line_count,
                candidates_considered=len(candidates),
                reasoning_start_char=span.start,
                reasoning_end_char=span.end,
                list_start_char=span.start + candidate.start_char,
                cut_char=span.start + candidate.cut_char,
                boundary_kind="gold_city_coverage_complete",
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
        city for city in cities if city.casefold() not in matched_folded
    )
    coverage_count = len(matched)
    return GoldCityListCut(
        detected=False,
        status="no_complete_gold_city_list",
        coverage_complete=False,
        gold_city_count=len(cities),
        gold_cities=cities,
        coverage_count=coverage_count,
        coverage_fraction=coverage_count / len(cities),
        matched_gold_cities=matched,
        missing_gold_cities=missing,
        marker_kind=best.marker_kind if best is not None else None,
        item_count=best.item_count if best is not None else 0,
        item_markers=best.item_markers if best is not None else (),
        item_gold_cities=best.item_gold_cities if best is not None else (),
        item_line_numbers=(best.item_line_numbers if best is not None else ()),
        duplicate_gold_city_items=(
            best.duplicate_gold_city_items if best is not None else 0
        ),
        bridge_line_count=best.bridge_line_count if best is not None else 0,
        candidates_considered=len(candidates),
        reasoning_start_char=span.start,
        reasoning_end_char=span.end,
        list_start_char=(
            span.start + best.start_char if best is not None else None
        ),
        closing_delimiter_present=span.closing_delimiter_present,
    )
