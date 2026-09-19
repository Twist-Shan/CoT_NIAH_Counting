from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from realistic_niah.parsing import parse_total, split_reasoning_and_final
from realistic_niah_v3.city_list_termination import (
    CityListTerminationCut,
    find_first_terminated_gold_city_list as _find_first_terminated_gold_city_list,
)
from realistic_niah_v3.first_list_cutoff import (
    align_text_exact_token_prefix,
    exact_token_prefix_length,
    locate_reasoning_span,
)
from realistic_niah_v5.hybrid_trace_parser import (
    EPISODE_SCHEMA_VERSION,
    EPISODE_SELECTION_POLICY,
    RankEpisodeParse,
    find_city_unit_span,
    find_rank_evidence_span,
    parse_rank_episodes,
)


PARSER_UPSTREAM_REPOSITORY = "anonymous-source"
PARSER_UPSTREAM_COMMIT = "anonymous-snapshot"
PARSER_IMPLEMENTATION = "realistic_niah_v5.parse_hybrid_trace"
_PARSER_DIRECTORY = Path(__file__).resolve().parent
PARSER_FILE_SHA256 = {
    "parsing.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "city_list_termination.py": "bb2cd01275a4dbfd339a388a3a830c4c2aa762ec14ad89ca04fd98bbc1b64728",
    "first_list_cutoff.py": "72781f9060d21fd6c693da4c0b0c0ad58831a031d37bc50fed21ee860ded66b7",
    "gold_city_cutoff.py": "bc5c37f410b96008023724f3f88895f82fdd39d9a0a05163427f1d3e017c03a9",
    "hybrid_trace_parser.py": hashlib.sha256(
        (_PARSER_DIRECTORY / "hybrid_trace_parser.py").read_bytes()
    ).hexdigest(),
}
PARSER_SCHEMA_VERSION = "realistic_niah_v5_hybrid_rank_trace_v3"
SITE_SCHEMA_VERSION = "realistic_niah_v5_trace_sites_v4"
PARSER_SELECTION_RULE = (
    "Extract gold-city observations paired with local explicit rank evidence, "
    "segment a new episode at every rank-1 restart, and select the longest "
    "contiguous 1..M episode (earliest on ties). Preserve rank-advancing city "
    "duplicates. Compare that episode with the frozen conservative structural "
    "span. The explicit episode is primary unless the structural sequence has "
    "the episode as an exact city prefix and adds at least one new city; that "
    "case is retained as a span-supported unmarked continuation. Then use an "
    "explicitly "
    "synthetic score-supported order only when "
    "neither span parser fires. Registered gold N and final Total never "
    "construct, pad, or select an episode."
)

_TOTAL_RE = re.compile(r"(?im)^\s*Total\s*:")
_TOTAL_ANYWHERE_RE = re.compile(r"(?i)(?<!\w)Total\s*:")
_TOTAL_VALUE_RE = re.compile(
    r"(?i)(?P<label>\bTotal\s*:\s*)(?P<value>[+-]?\d+)\b"
)
_INDEXED_MARKER_RE = re.compile(r"^[ \t]*\d+[.)][ \t]*")
_BULLET_MARKER_RE = re.compile(r"^[ \t]*(?:[-\u2022]|\*(?!\*))[ \t]*")
_ORDINAL_MARKER_RE = re.compile(
    r"^[ \t]*(?:\*{1,2}|`)?(?:first|second|third|fourth|fifth|sixth|"
    r"seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|"
    r"fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|"
    r"then|finally)(?:\*{1,2}|`)?(?:[.):,-](?:\*{1,2}|`)?[ \t]*|[ \t]+)",
    flags=re.IGNORECASE,
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
_NUMBER_WORDS = "|".join(
    sorted(_CARDINAL_VALUES, key=len, reverse=True)
)
_ORDINAL_WORDS = "|".join(
    sorted(_ORDINAL_VALUES, key=len, reverse=True)
)
_NUMERIC_COUNT_MARKER_RE = re.compile(
    r"\b(?:count|record)\s*(?:(?:[:=])\s*|\s+)(?P<value>\d{1,2})\b",
    flags=re.IGNORECASE,
)
_NUMERIC_SUMMARY_MARKER_RE = re.compile(
    r"\b(?:total|count)\s+(?:is|was|would\s+be|should\s+be|comes?\s+to)\s+"
    r"(?P<value>\d{1,2})\b",
    flags=re.IGNORECASE,
)
_CARDINAL_SUMMARY_MARKER_RE = re.compile(
    r"\b(?:(?:total|count)\s+(?:is|was|would\s+be|should\s+be|comes?\s+to)|"
    r"there\s+(?:are|were))\s+(?P<value>"
    + _NUMBER_WORDS
    + r")(?:\s+(?:records|entries|items|cities))?\b",
    flags=re.IGNORECASE,
)
_THATS_COUNT_MARKER_RE = re.compile(
    r"\b(?:that(?:'|\u2019)s|that\s+is)\s+(?P<value>"
    + _NUMBER_WORDS
    + r")(?:\s+(?:record|entry|item))?\b",
    flags=re.IGNORECASE,
)
_CARDINAL_SENTENCE_MARKER_RE = re.compile(
    r"(?<!\w)(?P<value>"
    + _NUMBER_WORDS
    + r")(?:\s+(?:record|entry|item))?(?=\s*[.!])",
    flags=re.IGNORECASE,
)
_ORDINAL_SENTENCE_MARKER_RE = re.compile(
    r"(?<!\w)(?P<value>"
    + _ORDINAL_WORDS
    + r")(?:\s+(?:record|entry|item))?(?=\s*[.!])",
    flags=re.IGNORECASE,
)
_INLINE_CITY_TO_MARKER_MAX_CHARS = 320


@dataclass(frozen=True)
class _MarkerHit:
    value: int
    start: int
    end: int
    priority: int
    surface_kind: str


@dataclass(frozen=True)
class _CityHit:
    city: str
    start: int
    end: int


@dataclass(frozen=True)
class _CountEvent:
    marker: _MarkerHit
    city: _CityHit


@dataclass(frozen=True)
class _EvidenceEvent:
    city: _CityHit
    end: int


@dataclass(frozen=True)
class TraceCharSite:
    site_id: str
    site_kind: str
    occurrence: int | None
    city: str | None
    marker: int | str | None
    boundary_kind: str | None
    char_start: int
    char_end: int
    primary: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TraceTokenSite:
    char_site: TraceCharSite
    alignment_eligible: bool
    alignment_status: str
    alignment_strategy: str | None
    prefix_token_count: int | None
    shared_baseline_prefix_tokens: int
    retokenized_suffix_tokens: int
    literal_token_start: int | None
    literal_token_end: int | None
    prefix_token_ids_sha256: str | None
    prefix_token_ids: tuple[int, ...] = ()

    @property
    def endpoint_token(self) -> int | None:
        if self.prefix_token_count is None or self.prefix_token_count < 1:
            return None
        return self.prefix_token_count - 1

    def to_dict(self, *, include_token_ids: bool = False) -> dict[str, Any]:
        payload = {
            **self.char_site.to_dict(),
            "alignment_eligible": self.alignment_eligible,
            "alignment_status": self.alignment_status,
            "alignment_strategy": self.alignment_strategy,
            "prefix_token_count": self.prefix_token_count,
            "endpoint_token": self.endpoint_token,
            "shared_baseline_prefix_tokens": self.shared_baseline_prefix_tokens,
            "retokenized_suffix_tokens": self.retokenized_suffix_tokens,
            "literal_token_start": self.literal_token_start,
            "literal_token_end": self.literal_token_end,
            "prefix_token_ids_sha256": self.prefix_token_ids_sha256,
        }
        if include_token_ids:
            payload["prefix_token_ids"] = list(self.prefix_token_ids)
        return payload


def infer_model_family(row: Mapping[str, Any], override: str | None = None) -> str:
    if override:
        family = str(override).lower()
    elif row.get("model_family"):
        family = str(row["model_family"]).lower()
    else:
        label = " ".join(
            str(row.get(key, ""))
            for key in ("model_label", "model", "model_id")
        ).lower()
        if "qwen3" in label:
            family = "qwen3"
        elif "gemma" in label:
            family = "gemma4"
        else:
            raise ValueError("Cannot infer qwen3/gemma4 model family")
    aliases = {"qwen": "qwen3", "gemma": "gemma4", "gemma-4": "gemma4"}
    family = aliases.get(family, family)
    if family not in {"qwen3", "gemma4"}:
        raise ValueError(f"Unsupported V5 parser family: {family}")
    return family


def raw_output_text(row: Mapping[str, Any]) -> str:
    for key in ("raw_output_text", "completion_text_raw", "output_text"):
        if row.get(key) is not None:
            return str(row[key])
    baseline = row.get("baseline")
    if isinstance(baseline, Mapping) and baseline.get("raw_output_text") is not None:
        return str(baseline["raw_output_text"])
    raise ValueError("Record has no raw native-thinking output text")


def output_token_ids(row: Mapping[str, Any]) -> tuple[int, ...]:
    for key in ("output_token_ids", "generated_token_ids"):
        value = row.get(key)
        if value is not None:
            return tuple(int(token) for token in value)
    raise ValueError("Record has no output/generated token IDs")


def prompt_token_ids(row: Mapping[str, Any]) -> tuple[int, ...]:
    for key in ("input_ids", "prompt_token_ids"):
        value = row.get(key)
        if value is not None:
            return tuple(int(token) for token in value)
    raise ValueError("Record has no prompt input IDs")


def gold_records(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("gold_records", "gold_pairs", "relevant_records"):
        value = row.get(key)
        if value is None:
            continue
        records: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, Mapping):
                if "city" not in item:
                    raise ValueError(f"Gold record lacks city: {item}")
                records.append(dict(item))
            else:
                records.append({"city": str(item)})
        return records
    raise ValueError("Record has no oracle city registry")


def _marker_hits(text: str) -> list[_MarkerHit]:
    hits: list[_MarkerHit] = []
    for pattern, values, priority, surface_kind in (
        (_NUMERIC_COUNT_MARKER_RE, None, 4, "numeric_count"),
        (_NUMERIC_SUMMARY_MARKER_RE, None, 2, "numeric_summary"),
        (_THATS_COUNT_MARKER_RE, _CARDINAL_VALUES, 3, "thats_cardinal"),
        (_ORDINAL_SENTENCE_MARKER_RE, _ORDINAL_VALUES, 2, "ordinal_sentence"),
        (_CARDINAL_SUMMARY_MARKER_RE, _CARDINAL_VALUES, 2, "cardinal_summary"),
        (_CARDINAL_SENTENCE_MARKER_RE, _CARDINAL_VALUES, 1, "cardinal_sentence"),
    ):
        for match in pattern.finditer(text):
            spelling = match.group("value").casefold()
            value = int(spelling) if values is None else int(values[spelling])
            hits.append(
                _MarkerHit(
                    value=value,
                    start=match.start(),
                    end=match.end(),
                    priority=priority,
                    surface_kind=surface_kind,
                )
            )
    return sorted(
        set(hits),
        key=lambda hit: (hit.start, hit.end, -hit.priority, hit.value),
    )


def _city_hits(text: str, cities: Sequence[str]) -> list[_CityHit]:
    if not cities:
        return []
    canonical = {str(city).casefold(): str(city) for city in cities}
    alternatives = "|".join(
        re.escape(city) for city in sorted(canonical.values(), key=len, reverse=True)
    )
    pattern = re.compile(
        rf"(?<!\w)(?P<city>{alternatives})(?!\w)", flags=re.IGNORECASE
    )
    return [
        _CityHit(
            city=canonical[match.group("city").casefold()],
            start=match.start("city"),
            end=match.end("city"),
        )
        for match in pattern.finditer(text)
    ]


def _record_evidence_events(
    reasoning: str,
    *,
    registry: Sequence[dict[str, Any] | str],
) -> tuple[_EvidenceEvent, ...]:
    """Find first score-supported mention of every actually cited gold record."""

    scores: dict[str, int | None] = {}
    canonical: dict[str, str] = {}
    for record in registry:
        if isinstance(record, Mapping):
            city = str(record["city"])
            score_value = record.get("score")
            score = int(score_value) if score_value is not None else None
        else:
            city = str(record)
            score = None
        canonical[city.casefold()] = city
        scores[city.casefold()] = score
    hits = _city_hits(reasoning, list(canonical.values()))
    supported: list[_EvidenceEvent] = []
    for hit in hits:
        score = scores[hit.city.casefold()]
        right = min(len(reasoning), hit.end + 96)
        window = reasoning[hit.end:right]
        if score is None:
            match = re.search(
                r"\breceived\s+a\s+(?:numeric\s+)?score\b", window, re.IGNORECASE
            )
        else:
            match = re.search(rf"(?<!\d){int(score)}(?!\d)", window)
        if match is None:
            continue
        end = hit.end + match.end()
        while end < len(reasoning) and reasoning[end] in " \t*`)]\"'":
            end += 1
        if end < len(reasoning) and reasoning[end] in ".;":
            end += 1
        supported.append(_EvidenceEvent(city=hit, end=end))

    # Re-reading or verification does not create another implicit running-index
    # event. Explicit 1..M markers are handled separately and may preserve a
    # deliberate repeated-city count.
    unique: list[_EvidenceEvent] = []
    seen: set[str] = set()
    for event in sorted(supported, key=lambda value: value.city.start):
        folded = event.city.city.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        unique.append(event)
    return tuple(unique)


def _continuous_inline_count_events(
    reasoning: str,
    *,
    gold_cities: Sequence[str],
) -> tuple[_CountEvent, ...]:
    """Return the longest trace-local 1..M inline count-marker sequence.

    A marker is eligible only when a registered city occurs shortly before it.
    The rule never looks at the final answer or at the registered target count.
    This recovers nested ``Scan``/``Excerpt`` bullets whose actual update is the
    trailing ``Count k``/``Record k`` marker, as well as Qwen prose such as
    ``... score of 81. Three.``.
    """

    cities = _city_hits(reasoning, gold_cities)
    if not cities:
        return ()
    events: list[_CountEvent] = []
    for marker in _marker_hits(reasoning):
        preceding = [city for city in cities if city.end <= marker.start]
        if not preceding:
            continue
        city = preceding[-1]
        if marker.start - city.end > _INLINE_CITY_TO_MARKER_MAX_CHARS:
            continue
        events.append(_CountEvent(marker=marker, city=city))
    if not events:
        return ()

    sequences: list[tuple[_CountEvent, ...]] = []
    for first in (event for event in events if event.marker.value == 1):
        sequence = [first]
        expected = 2
        while True:
            candidates = [
                event
                for event in events
                if event.marker.value == expected
                and event.marker.start >= sequence[-1].marker.end
                and event.city.start > sequence[-1].city.start
            ]
            if not candidates:
                break
            sequence.append(
                min(
                    candidates,
                    key=lambda event: (
                        event.marker.start,
                        -event.marker.priority,
                        event.marker.end,
                    ),
                )
            )
            expected += 1
        sequences.append(tuple(sequence))
    if not sequences:
        return ()
    return max(
        sequences,
        key=lambda sequence: (
            len(sequence),
            sum(event.marker.priority for event in sequence),
            -sequence[0].marker.start,
        ),
    )


def _trace_classification(
    item_cities: Sequence[str], gold_cities: Sequence[str]
) -> tuple[bool, str, str]:
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
    unique = set(observed)
    complete = set(gold).issubset(unique)
    duplicates = len(observed) != len(unique)
    if complete and duplicates:
        category = "full_coverage_with_duplicates"
    elif duplicates:
        category = "partial_with_duplicates"
    else:
        category = "partial_unique"
    return False, "not_one_to_one", category


def _inline_count_cut(
    raw_text: str,
    *,
    model_family: str,
    base: CityListTerminationCut,
) -> CityListTerminationCut | None:
    span = locate_reasoning_span(raw_text, model_family=model_family)
    reasoning = raw_text[span.start : span.end]
    events = _continuous_inline_count_events(
        reasoning, gold_cities=base.gold_cities
    )
    if not events:
        return None
    item_cities = tuple(event.city.city for event in events)
    matched: list[str] = []
    seen: set[str] = set()
    duplicate_items = 0
    for city in item_cities:
        folded = city.casefold()
        if folded in seen:
            duplicate_items += 1
            continue
        seen.add(folded)
        matched.append(city)
    missing = tuple(
        city for city in base.gold_cities if city.casefold() not in seen
    )
    one_to_one, order_class, category = _trace_classification(
        item_cities, base.gold_cities
    )
    starts = tuple(span.start + event.city.start for event in events)
    ends = tuple(span.start + event.marker.end for event in events)
    return CityListTerminationCut(
        detected=True,
        status="ok_continuous_inline_count_sequence",
        all_items_gold_city=True,
        coverage_complete=not missing,
        gold_city_count=len(base.gold_cities),
        gold_cities=base.gold_cities,
        coverage_count=len(matched),
        coverage_fraction=len(matched) / len(base.gold_cities),
        matched_gold_cities=tuple(matched),
        missing_gold_cities=missing,
        marker_kind="inline_count",
        item_count=len(events),
        item_markers=tuple(event.marker.value for event in events),
        item_gold_cities=item_cities,
        item_line_numbers=tuple(
            reasoning.count("\n", 0, event.city.start) + 1 for event in events
        ),
        item_start_chars=starts,
        item_end_chars=ends,
        item_boundary_kinds=("inline_count_marker",) * len(events),
        duplicate_gold_city_items=duplicate_items,
        trace_one_to_one=one_to_one,
        trace_order_class=order_class,
        trace_category=category,
        bridge_line_count=0,
        trailing_line_count=0,
        candidates_considered=int(base.candidates_considered) + 1,
        rejected_candidates=int(base.rejected_candidates),
        termination_kind="continuous_inline_count_markers",
        reasoning_start_char=span.start,
        reasoning_end_char=span.end,
        list_start_char=starts[0],
        cut_char=ends[-1],
        boundary_kind="inline_count_marker_sequence",
        closing_delimiter_present=span.closing_delimiter_present,
    )


def _evidence_sequence_cut(
    raw_text: str,
    *,
    model_family: str,
    registry: Sequence[dict[str, Any] | str],
    base: CityListTerminationCut,
) -> CityListTerminationCut | None:
    span = locate_reasoning_span(raw_text, model_family=model_family)
    reasoning = raw_text[span.start : span.end]
    events = _record_evidence_events(reasoning, registry=registry)
    if not events:
        return None
    item_cities = tuple(event.city.city for event in events)
    matched = tuple(item_cities)
    seen = {city.casefold() for city in matched}
    missing = tuple(
        city for city in base.gold_cities if city.casefold() not in seen
    )
    one_to_one, order_class, category = _trace_classification(
        item_cities, base.gold_cities
    )
    starts = tuple(span.start + event.city.start for event in events)
    ends = tuple(span.start + event.end for event in events)
    return CityListTerminationCut(
        detected=True,
        status="ok_score_supported_evidence_sequence",
        all_items_gold_city=True,
        coverage_complete=not missing,
        gold_city_count=len(base.gold_cities),
        gold_cities=base.gold_cities,
        coverage_count=len(matched),
        coverage_fraction=len(matched) / len(base.gold_cities),
        matched_gold_cities=matched,
        missing_gold_cities=missing,
        marker_kind="evidence_sequence",
        item_count=len(events),
        item_markers=tuple(range(1, len(events) + 1)),
        item_gold_cities=item_cities,
        item_line_numbers=tuple(
            reasoning.count("\n", 0, event.city.start) + 1 for event in events
        ),
        item_start_chars=starts,
        item_end_chars=ends,
        item_boundary_kinds=("score_supported_record",) * len(events),
        duplicate_gold_city_items=0,
        trace_one_to_one=one_to_one,
        trace_order_class=order_class,
        trace_category=category,
        bridge_line_count=0,
        trailing_line_count=0,
        candidates_considered=int(base.candidates_considered) + 1,
        rejected_candidates=int(base.rejected_candidates),
        termination_kind="score_supported_evidence_sequence",
        reasoning_start_char=span.start,
        reasoning_end_char=span.end,
        list_start_char=starts[0],
        cut_char=ends[-1],
        boundary_kind="score_supported_evidence_sequence",
        closing_delimiter_present=span.closing_delimiter_present,
    )


def _rank_episode_cut(
    raw_text: str,
    *,
    base: CityListTerminationCut,
    episodes: RankEpisodeParse,
) -> CityListTerminationCut:
    selected = episodes.selected_sequence
    if selected is None:
        raise ValueError("Rank-episode cut requires a selected sequence")
    events = selected.events
    item_cities = tuple(event.city for event in events)
    seen: set[str] = set()
    matched: list[str] = []
    duplicate_items = 0
    for city in item_cities:
        folded = city.casefold()
        if folded in seen:
            duplicate_items += 1
            continue
        seen.add(folded)
        matched.append(city)
    missing = tuple(
        city for city in base.gold_cities if city.casefold() not in seen
    )
    one_to_one, order_class, category = _trace_classification(
        item_cities, base.gold_cities
    )
    evidence_families = {event.evidence_family for event in events}
    marker_kind = (
        next(iter(evidence_families))
        if len(evidence_families) == 1
        else "inline_count"
    )
    return CityListTerminationCut(
        detected=True,
        status="ok_rank_supported_episode",
        all_items_gold_city=True,
        coverage_complete=not missing,
        gold_city_count=len(base.gold_cities),
        gold_cities=base.gold_cities,
        coverage_count=len(matched),
        coverage_fraction=len(matched) / len(base.gold_cities),
        matched_gold_cities=tuple(matched),
        missing_gold_cities=missing,
        marker_kind=marker_kind,
        item_count=len(events),
        item_markers=tuple(event.rank for event in events),
        item_gold_cities=item_cities,
        item_line_numbers=tuple(
            raw_text.count("\n", 0, event.semantic_start_char) + 1
            for event in events
        ),
        item_start_chars=tuple(event.semantic_start_char for event in events),
        item_end_chars=tuple(event.semantic_end_char for event in events),
        item_boundary_kinds=tuple(
            f"semantic_end:{event.evidence_kind}" for event in events
        ),
        duplicate_gold_city_items=duplicate_items,
        trace_one_to_one=one_to_one,
        trace_order_class=order_class,
        trace_category=category,
        bridge_line_count=0,
        trailing_line_count=0,
        candidates_considered=(
            int(base.candidates_considered) + len(episodes.sequences)
        ),
        rejected_candidates=int(base.rejected_candidates),
        termination_kind="selected_rank_supported_episode",
        reasoning_start_char=episodes.reasoning_start_char,
        reasoning_end_char=episodes.reasoning_end_char,
        list_start_char=selected.start_char,
        cut_char=selected.end_char,
        boundary_kind="rank_supported_semantic_end",
        closing_delimiter_present=episodes.closing_delimiter_present,
    )


def parse_hybrid_trace(
    raw_text: str,
    *,
    model_family: str,
    gold_records: Iterable[dict[str, Any] | str],
) -> tuple[CityListTerminationCut, dict[str, Any]]:
    """Parse one trajectory with explicit-rank, structural, and synthetic tiers."""

    registry = list(gold_records)
    base = _find_first_terminated_gold_city_list(
        raw_text,
        model_family=model_family,
        gold_records=registry,
    )
    episodes = parse_rank_episodes(
        raw_text,
        model_family=model_family,
        gold_cities=base.gold_cities,
    )
    selected = episodes.selected_sequence
    selected_cities = (
        tuple(event.city.casefold() for event in selected.events)
        if selected is not None
        else ()
    )
    structural_cities = tuple(
        city.casefold() for city in base.item_gold_cities
    )
    structural_extension = bool(
        selected is not None
        and base.detected
        and len(structural_cities) > len(selected_cities)
        and structural_cities[: len(selected_cities)] == selected_cities
        and any(
            city not in set(selected_cities)
            for city in structural_cities[len(selected_cities) :]
        )
    )
    if selected is not None and not structural_extension:
        cut = _rank_episode_cut(raw_text, base=base, episodes=episodes)
        source = "rank_supported_episode"
        selection_reason = (
            "no_structural_span"
            if not base.detected
            else "rank_episode_not_strictly_extended_by_structural_span"
        )
    elif structural_extension:
        cut = base
        source = "structural_extension"
        selection_reason = "structural_span_adds_new_cities_after_rank_prefix"
    elif base.detected:
        cut = base
        source = "structural_fallback"
        selection_reason = "no_rank_supported_episode"
    else:
        synthetic = _evidence_sequence_cut(
            raw_text,
            model_family=model_family,
            registry=registry,
            base=base,
        )
        if synthetic is None:
            cut = base
            source = "no_parser_hit"
            selection_reason = "no_span_or_synthetic_sequence"
        else:
            cut = replace(
                synthetic,
                status="ok_synthetic_score_supported_order",
                marker_kind="evidence_sequence",
                trace_one_to_one=False,
                trace_order_class="synthetic_unverified",
                trace_category="synthetic_unverified",
                termination_kind="synthetic_score_supported_order",
                boundary_kind="synthetic_score_supported_order",
            )
            source = "synthetic_evidence_fallback"
            selection_reason = "no_span_sequence"
    audit = {
        **episodes.to_dict(),
        "sequence_source": source,
        "selection_reason": selection_reason,
        "structural_base": {
            "detected": bool(base.detected),
            "status": base.status,
            "marker_kind": base.marker_kind,
            "item_count": int(base.item_count),
            "trace_category": base.trace_category,
            "trace_one_to_one": bool(base.trace_one_to_one),
        },
        "selected_parser": {
            "status": cut.status,
            "marker_kind": cut.marker_kind,
            "item_count": int(cut.item_count),
            "trace_category": cut.trace_category,
            "trace_one_to_one": bool(cut.trace_one_to_one),
        },
    }
    return cut, audit


def find_trace_count_sequence(
    raw_text: str,
    *,
    model_family: str,
    gold_records: Iterable[dict[str, Any] | str],
) -> CityListTerminationCut:
    """Return the selected 1..M trace while retaining full audit via record API."""

    cut, _audit = parse_hybrid_trace(
        raw_text,
        model_family=model_family,
        gold_records=gold_records,
    )
    return cut


def _marker_span(
    text: str,
    start: int,
    end: int,
    marker_kind: str | None,
    marker: int | str | None,
) -> tuple[int, int] | None:
    item = text[start:end]
    if (
        isinstance(marker, int)
        and marker_kind in {"indexed", "ordinal", "inline_count"}
    ):
        evidence = find_rank_evidence_span(item, expected_rank=int(marker))
        if evidence is not None:
            return start + evidence.start, start + evidence.end
    if marker_kind == "inline_count" and marker is not None:
        # Sentence-style markers (``Three.``) use punctuation as a right-side
        # assertion.  The parser item itself ends at ``Three``, so expose a
        # small read-only lookahead while requiring the returned span to remain
        # inside the registered item boundary.
        probe = text[start : min(len(text), end + 8)]
        matches = [
            hit
            for hit in _marker_hits(probe)
            if int(hit.value) == int(marker) and hit.end <= end - start
        ]
        if matches:
            match = max(matches, key=lambda hit: (hit.end, hit.priority, -hit.start))
            return start + match.start, start + match.end
    patterns: tuple[re.Pattern[str], ...]
    if marker_kind == "indexed":
        patterns = (_INDEXED_MARKER_RE,)
    elif marker_kind == "bullet":
        patterns = (_BULLET_MARKER_RE,)
    elif marker_kind == "ordinal":
        patterns = (_ORDINAL_MARKER_RE,)
    else:
        patterns = (_INDEXED_MARKER_RE, _BULLET_MARKER_RE, _ORDINAL_MARKER_RE)
    for pattern in patterns:
        match = pattern.match(item)
        if match and match.end() > 0:
            return start + match.start(), start + match.end()
    return None


def _city_span(text: str, start: int, end: int, city: str) -> tuple[int, int] | None:
    item = text[start:end]
    pattern = re.compile(r"(?<!\w)" + re.escape(city) + r"(?!\w)", re.IGNORECASE)
    matches = list(pattern.finditer(item))
    if not matches:
        return None
    match = matches[-1]
    return start + match.start(), start + match.end()


def _post_boundary_end(text: str, item_end: int) -> int:
    end = int(item_end)
    if text.startswith("\r\n", end):
        return end + 2
    if end < len(text) and text[end] in "\r\n":
        return end + 1
    return end


def _answer_query_span(text: str, reasoning_end: int | None) -> tuple[int, int] | None:
    start = int(reasoning_end or 0)
    matches = list(_TOTAL_RE.finditer(text, pos=start))
    if not matches:
        matches = list(_TOTAL_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    colon = text.find(":", match.start(), match.end())
    if colon < 0:
        return None
    return match.start(), colon + 1


def _answer_query_v2_span(
    text: str, reasoning_end: int | None
) -> tuple[int, int] | None:
    """Locate a possibly channel-prefixed literal ``Total:`` answer query."""

    start = int(reasoning_end or 0)
    matches = list(_TOTAL_ANYWHERE_RE.finditer(text, pos=start))
    if not matches:
        matches = list(_TOTAL_ANYWHERE_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    colon = text.find(":", match.start(), match.end())
    if colon < 0:
        return None
    return match.start(), colon + 1


def _answer_query_v3_span(
    text: str, reasoning_end: int | None
) -> tuple[int, int] | None:
    """Return the literal response prefix ending immediately before the answer.

    Unlike ``answer_query`` (which ends at the colon), this boundary includes
    any literal whitespace after ``Total:`` and stops at the first digit of the
    final integer. It therefore matches the historical V3 final-count capture
    while remaining an ordinary registered site in the unified V5 shard.
    """

    start = int(reasoning_end or 0)
    matches = list(_TOTAL_VALUE_RE.finditer(text, pos=start))
    if not matches:
        matches = list(_TOTAL_VALUE_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    return match.start("label"), match.start("value")


def _answer_query_char_sites(
    raw_text: str, reasoning_end: int | None
) -> list[TraceCharSite]:
    """Extract final-answer sites independently of running-trace parsing.

    A response can contain a perfectly valid ``Total: <integer>`` boundary
    even when no running-count episode is recoverable.  Keeping these sites
    independent prevents a running-parser miss from silently deleting a
    final-count representation sample.
    """

    sites: list[TraceCharSite] = []
    answer = _answer_query_span(raw_text, reasoning_end)
    if answer is not None:
        sites.append(
            TraceCharSite(
                site_id="answer_query",
                site_kind="answer_query",
                occurrence=None,
                city=None,
                marker=None,
                boundary_kind="total_colon",
                char_start=answer[0],
                char_end=answer[1],
                primary=False,
            )
        )
    answer_v2 = _answer_query_v2_span(raw_text, reasoning_end)
    if answer_v2 is not None:
        sites.append(
            TraceCharSite(
                site_id="answer_query_v2",
                site_kind="answer_query_v2",
                occurrence=None,
                city=None,
                marker=None,
                boundary_kind="total_colon_relaxed_anchor_v2",
                char_start=answer_v2[0],
                char_end=answer_v2[1],
                primary=False,
            )
        )
    answer_v3 = _answer_query_v3_span(raw_text, reasoning_end)
    if answer_v3 is not None:
        sites.append(
            TraceCharSite(
                site_id="answer_query_v3",
                site_kind="answer_query_v3",
                occurrence=None,
                city=None,
                marker=None,
                boundary_kind="literal_token_before_numeric_answer_v3",
                char_start=answer_v3[0],
                char_end=answer_v3[1],
                primary=False,
            )
        )
    return sites


def trace_char_sites(raw_text: str, parser: CityListTerminationCut) -> list[TraceCharSite]:
    answer_sites = _answer_query_char_sites(raw_text, parser.reasoning_end_char)
    if not parser.detected:
        return answer_sites
    lengths = {
        len(parser.item_markers),
        len(parser.item_gold_cities),
        len(parser.item_start_chars),
        len(parser.item_end_chars),
        len(parser.item_boundary_kinds),
    }
    if len(lengths) != 1:
        raise RuntimeError("Parser item arrays are not index-aligned")
    sites: list[TraceCharSite] = []
    for offset, (marker, city, start, end, boundary) in enumerate(
        zip(
            parser.item_markers,
            parser.item_gold_cities,
            parser.item_start_chars,
            parser.item_end_chars,
            parser.item_boundary_kinds,
        ),
        start=1,
    ):
        if not 0 <= start < end <= len(raw_text):
            raise RuntimeError(f"Parser item span is invalid: [{start}, {end})")
        common = {
            "occurrence": offset,
            "city": str(city),
            "marker": marker,
            "boundary_kind": str(boundary),
        }
        marker_span = _marker_span(
            raw_text, start, end, parser.marker_kind, marker
        )
        if marker_span is not None:
            if marker_span[0] > 0:
                sites.append(
                    TraceCharSite(
                        site_id=f"pre_marker:{offset}",
                        site_kind="pre_marker",
                        char_start=marker_span[0] - 1,
                        char_end=marker_span[0],
                        primary=False,
                        **common,
                    )
                )
            sites.append(
                TraceCharSite(
                    site_id=f"marker_end:{offset}",
                    site_kind="marker_end",
                    char_start=marker_span[0],
                    char_end=marker_span[1],
                    primary=False,
                    **common,
                )
            )
        city_span = _city_span(raw_text, start, end, str(city))
        if city_span is not None:
            if city_span[0] > 0:
                sites.append(
                    TraceCharSite(
                        site_id=f"pre_city:{offset}",
                        site_kind="pre_city",
                        char_start=city_span[0] - 1,
                        char_end=city_span[0],
                        primary=False,
                        **common,
                    )
                )
            sites.append(
                TraceCharSite(
                    site_id=f"city_end:{offset}",
                    site_kind="city_end",
                    char_start=city_span[0],
                    char_end=city_span[1],
                    primary=False,
                    **common,
                )
            )
            city_unit = find_city_unit_span(
                raw_text,
                city_start=city_span[0],
                city_end=city_span[1],
            )
            if city_unit is not None:
                sites.append(
                    TraceCharSite(
                        site_id=f"city_unit_end:{offset}",
                        site_kind="city_unit_end",
                        char_start=city_unit[0],
                        char_end=city_unit[1],
                        primary=False,
                        **common,
                    )
                )
        sites.append(
            TraceCharSite(
                site_id=f"item_end:{offset}",
                site_kind="item_end",
                char_start=start,
                char_end=end,
                primary=True,
                **common,
            )
        )
        post_end = _post_boundary_end(raw_text, end)
        sites.append(
            TraceCharSite(
                site_id=f"post_boundary:{offset}",
                site_kind="post_boundary",
                char_start=start,
                char_end=post_end,
                primary=False,
                **common,
            )
        )
    if parser.cut_char is not None and parser.list_start_char is not None:
        sites.append(
            TraceCharSite(
                site_id="list_cut",
                site_kind="list_cut",
                occurrence=parser.item_count,
                city=(parser.item_gold_cities[-1] if parser.item_gold_cities else None),
                marker=(parser.item_markers[-1] if parser.item_markers else None),
                boundary_kind=parser.boundary_kind,
                char_start=int(parser.list_start_char),
                char_end=int(parser.cut_char),
                primary=False,
            )
        )
    sites.extend(answer_sites)
    return sites


def parse_trace_record(
    row: Mapping[str, Any], *, model_family: str | None = None
) -> dict[str, Any]:
    family = infer_model_family(row, model_family)
    raw = raw_output_text(row)
    gold = gold_records(row)
    parser, episode_parse = parse_hybrid_trace(
        raw,
        model_family=family,
        gold_records=gold,
    )
    reasoning, final = split_reasoning_and_final(
        raw,
        prompt_mode="native_thinking",
        reasoning_expected=True,
    )
    parsed_count = parse_total(final)
    exact_count = parsed_count == len(gold) if parsed_count is not None else False
    return {
        "schema_version": PARSER_SCHEMA_VERSION,
        "parser_upstream_repository": PARSER_UPSTREAM_REPOSITORY,
        "parser_upstream_commit": PARSER_UPSTREAM_COMMIT,
        "parser_implementation": PARSER_IMPLEMENTATION,
        "parser_selection_rule": PARSER_SELECTION_RULE,
        "rank_episode_schema_version": EPISODE_SCHEMA_VERSION,
        "rank_episode_selection_policy": EPISODE_SELECTION_POLICY,
        "parser_file_sha256": dict(PARSER_FILE_SHA256),
        "request_id": row.get("request_id", row.get("stimulus_id")),
        "stimulus_id": row.get("stimulus_id"),
        "model_label": row.get("model_label", row.get("model")),
        "model_family": family,
        "seed": row.get("seed"),
        "split": row.get("split"),
        "gold_count": len(gold),
        "parsed_count": parsed_count,
        "exact_count": exact_count,
        "reasoning_text": reasoning,
        "final_text": final,
        "sequence_source": episode_parse["sequence_source"],
        "episode_parse": episode_parse,
        "parser": parser.to_dict(),
        "char_sites": [site.to_dict() for site in trace_char_sites(raw, parser)],
    }


def _token_ids_sha256(values: Sequence[int]) -> str:
    payload = ",".join(str(int(value)) for value in values).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def align_trace_sites(
    tokenizer: Any,
    *,
    raw_text: str,
    baseline_output_token_ids: Iterable[int],
    sites: Iterable[TraceCharSite],
) -> list[TraceTokenSite]:
    baseline = tuple(int(value) for value in baseline_output_token_ids)
    aligned: list[TraceTokenSite] = []
    for site in sites:
        start = exact_token_prefix_length(
            tokenizer,
            raw_text=raw_text,
            output_token_ids=baseline,
            cut_char=site.char_start,
        )
        end = exact_token_prefix_length(
            tokenizer,
            raw_text=raw_text,
            output_token_ids=baseline,
            cut_char=site.char_end,
        )
        prefix = align_text_exact_token_prefix(
            tokenizer,
            raw_text=raw_text,
            output_token_ids=baseline,
            cut_char=site.char_end,
        )
        token_ids = tuple(prefix.token_ids) if prefix.eligible else ()
        aligned.append(
            TraceTokenSite(
                char_site=site,
                alignment_eligible=bool(prefix.eligible),
                alignment_status=str(prefix.status),
                alignment_strategy=prefix.strategy,
                prefix_token_count=(len(token_ids) if prefix.eligible else None),
                shared_baseline_prefix_tokens=int(prefix.shared_baseline_prefix_tokens),
                retokenized_suffix_tokens=int(prefix.retokenized_suffix_tokens),
                literal_token_start=start,
                literal_token_end=end,
                prefix_token_ids_sha256=(
                    _token_ids_sha256(token_ids) if token_ids else None
                ),
                prefix_token_ids=token_ids,
            )
        )
    return aligned


def parse_and_align_record(
    row: Mapping[str, Any], tokenizer: Any, *, model_family: str | None = None
) -> dict[str, Any]:
    parsed = parse_trace_record(row, model_family=model_family)
    parser = CityListTerminationCut(**parsed["parser"])
    sites = trace_char_sites(raw_output_text(row), parser)
    aligned = align_trace_sites(
        tokenizer,
        raw_text=raw_output_text(row),
        baseline_output_token_ids=output_token_ids(row),
        sites=sites,
    )
    parsed["site_schema_version"] = SITE_SCHEMA_VERSION
    parsed["token_sites"] = [site.to_dict() for site in aligned]
    parsed["alignment_summary"] = {
        "sites": len(aligned),
        "eligible": sum(site.alignment_eligible for site in aligned),
        "literal_baseline": sum(
            site.alignment_strategy == "literal_baseline_token_prefix"
            for site in aligned
        ),
        "retokenized": sum(
            site.alignment_strategy == "text_exact_boundary_retokenization"
            for site in aligned
        ),
    }
    return parsed
