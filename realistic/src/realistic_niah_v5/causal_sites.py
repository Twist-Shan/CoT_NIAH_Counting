"""Grammar-aware causal-site plans for native-thinking traces.

The representation pipeline intentionally uses one common observation site
(``item_end``).  Causal experiments need a stricter object: the token queried
immediately before a city, the token at which one trace event is committed,
and the continuation that leads to the next city or final answer boundary.
Those roles depend on the event grammar and must not be inferred from a fixed
``marker_end -> city_end`` template.

This module does not run interventions.  It compiles the existing hybrid trace
parse into an auditable, event-level plan.  Character parsing and token
alignment are kept separate so that a parser hit is never silently promoted to
a causal-eligible token site.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .parsing import (
    gold_records,
    output_token_ids,
    parse_trace_record,
    prompt_token_ids,
    raw_output_text,
)


SCHEMA_VERSION = "realistic_niah_v5_causal_site_plan_v2"
TOKEN_ALIGNMENT_VERSION = "exact_reencode_with_monotone_covering_offset_spans_v1"
GRAMMAR_POLICY_VERSION = "transition_anchor_surface_grammar_v3"
COHORT_POLICY_VERSION = "rank_structural_partial_and_evidence_secondary_v2"

RECORD_CLAUSE = "In the 2024 city score audit"
OPENING_DELIMITERS = frozenset("([{（［｛")
CLOSING_TO_OPENING = {")": "(", "]": "[", "}": "{", "）": "（", "］": "［", "｝": "｛"}
BULLET_DELIMITERS = frozenset({"-", "*", "•", "–", "—", "·"})

# Lower values win only when two semantic roles resolve to the same model token.
# Every alias is retained, so this order never erases the parser interpretation.
ANCHOR_ROLE_PRIORITY = {
    "p0_item_end": 0,
    "unit_pre_d1": 1,
    "pre_marker_d1": 2,
    "post_open_delimiter": 3,
    "post_marker": 4,
    "record_clause_pre_d1": 5,
    "city_pre_d1": 6,
    "block_pre_d1": 7,
}

COHORT_ALLOWED_ESTIMANDS: dict[str, tuple[str, ...]] = {
    "primary_rank_resolved_full_chain": (
        "source_to_city_retrieval",
        "rank_core_and_format_controls",
        "continue_to_next_city",
        "terminal_stop",
        "serial_mediation",
    ),
    "secondary_structural_marker_neutral": (
        "structural_source_to_city_retrieval",
        "invariant_or_structural_marker_controls",
        "structural_continue",
        "structural_terminal_stop",
    ),
    "secondary_structural_recap": (
        "recap_source_to_city_retrieval",
        "recap_marker_controls",
        "recap_continue",
        "recap_terminal_stop",
    ),
    "secondary_local_partial_unique": (
        "local_source_to_city_retrieval",
        "local_rank_core_and_format_controls",
        "local_continue_to_next_observed_city",
    ),
    "occurrence_retrieval_only_duplicates": (
        "occurrence_source_to_city_retrieval",
    ),
    "secondary_evidence_sequence_exploratory": (
        "score_supported_source_to_city_retrieval",
        "evidence_sequence_continue",
        "evidence_sequence_terminal_stop",
    ),
    "secondary_evidence_sequence_partial_exploratory": (
        "local_score_supported_source_to_city_retrieval",
        "local_evidence_sequence_continue",
    ),
    "audit_only_unresolved": (),
}
CONTINUE_ELIGIBLE_COHORTS = frozenset(
    {
        "primary_rank_resolved_full_chain",
        "secondary_structural_marker_neutral",
        "secondary_structural_recap",
        "secondary_local_partial_unique",
        "secondary_evidence_sequence_exploratory",
        "secondary_evidence_sequence_partial_exploratory",
    }
)
STOP_ELIGIBLE_COHORTS = frozenset(
    {
        "primary_rank_resolved_full_chain",
        "secondary_structural_marker_neutral",
        "secondary_structural_recap",
        "secondary_evidence_sequence_exploratory",
    }
)
MARKER_CONTROL_ELIGIBLE_COHORTS = CONTINUE_ELIGIBLE_COHORTS
INVARIANT_SURFACE_ELIGIBLE_COHORTS = frozenset(
    {
        "secondary_structural_marker_neutral",
        "secondary_structural_recap",
    }
)


class CausalSiteError(RuntimeError):
    """Raised when parser output cannot be compiled without guessing."""


@dataclass(frozen=True)
class OutputTokenMap:
    """Exact tokenization of one stored assistant output."""

    raw_text: str
    token_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    tokenizer: Any

    def decode(self, values: Sequence[int]) -> str:
        ids = [int(value) for value in values]
        try:
            return str(
                self.tokenizer.decode(
                    ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
        except TypeError:
            return str(self.tokenizer.decode(ids, skip_special_tokens=False))

    def span(self, role: str, char_start: int, char_end: int) -> dict[str, Any]:
        """Map a character span to the smallest overlapping token interval.

        A leading-space tokenizer token may begin before ``char_start``.  That
        is not treated as an alignment failure: it is the actual model token
        that contains the first semantic character.  Spill is recorded so the
        choice remains inspectable.
        """

        start = int(char_start)
        end = int(char_end)
        if not 0 <= start < end <= len(self.raw_text):
            return {
                "role": role,
                "status": "invalid_char_span",
                "char_start": start,
                "char_end": end,
            }
        hits = [
            index
            for index, (left, right) in enumerate(self.offsets)
            if right > left and right > start and left < end
        ]
        if not hits:
            return {
                "role": role,
                "status": "no_overlapping_tokens",
                "char_start": start,
                "char_end": end,
                "char_text": self.raw_text[start:end],
            }
        token_start = hits[0]
        token_end = hits[-1] + 1
        offset_start = int(self.offsets[token_start][0])
        offset_end = int(self.offsets[token_end - 1][1])
        token_ids = self.token_ids[token_start:token_end]
        return {
            "role": role,
            "status": "ok",
            "char_start": start,
            "char_end": end,
            "char_text": self.raw_text[start:end],
            "output_token_start": token_start,
            "output_token_end": token_end,
            "token_ids": list(token_ids),
            "token_text": self.decode(token_ids),
            "offset_char_start": offset_start,
            "offset_char_end": offset_end,
            "exact_char_start": offset_start == start,
            "exact_char_end": offset_end == end,
            "left_spill_chars": max(0, start - offset_start),
            "right_spill_chars": max(0, offset_end - end),
        }


def _normalize_encoding_payload(payload: Any) -> tuple[list[int], list[tuple[int, int]]]:
    if isinstance(payload, Mapping):
        ids = payload.get("input_ids")
        offsets = payload.get("offset_mapping")
    else:
        ids = getattr(payload, "ids", None)
        offsets = getattr(payload, "offsets", None)
    if ids is None or offsets is None:
        raise CausalSiteError("Tokenizer supplied no IDs/offset mapping")
    if ids and isinstance(ids[0], (list, tuple)):
        if len(ids) != 1:
            raise CausalSiteError("Batched tokenization is not supported")
        ids = ids[0]
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(
        offsets[0][0], (list, tuple)
    ):
        if len(offsets) != 1:
            raise CausalSiteError("Batched offset mapping is not supported")
        offsets = offsets[0]
    normalized_ids = [int(value) for value in ids]
    normalized_offsets = [(int(left), int(right)) for left, right in offsets]
    return normalized_ids, normalized_offsets


def build_output_token_map(row: Mapping[str, Any], tokenizer: Any) -> OutputTokenMap:
    """Re-tokenize one output and require exact equality with stored IDs."""

    raw = raw_output_text(row)
    payload: Any | None = None
    if callable(tokenizer):
        try:
            payload = tokenizer(
                raw,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
        except TypeError:
            payload = None
    if payload is None:
        encoded = tokenizer.encode(raw, add_special_tokens=False)
        payload = encoded
    ids, offsets = _normalize_encoding_payload(payload)
    expected = list(output_token_ids(row))
    if ids != expected:
        mismatch = next(
            (
                index
                for index, (actual, frozen) in enumerate(zip(ids, expected))
                if actual != frozen
            ),
            min(len(ids), len(expected)),
        )
        raise CausalSiteError(
            "Re-tokenized output differs from frozen output IDs at "
            f"token {mismatch}: actual_len={len(ids)} frozen_len={len(expected)}"
        )
    if len(offsets) != len(ids):
        raise CausalSiteError("Token IDs and offsets have different lengths")
    previous_left = 0
    previous_right = 0
    for index, (left, right) in enumerate(offsets):
        if left < 0 or right < left or right > len(raw):
            raise CausalSiteError(f"Invalid tokenizer offset at token {index}")
        # Byte-level tokenizers can split one multi-byte Unicode character
        # across multiple tokens.  Those tokens legitimately share the same
        # character offset (for example both cover U+2029), so intervals need
        # only be monotone in their left and right boundaries; they need not be
        # disjoint.
        if right > left and (
            left < previous_left or right < previous_right
        ):
            raise CausalSiteError(f"Non-monotone tokenizer offset at token {index}")
        if right > left:
            previous_left = left
            previous_right = right
    return OutputTokenMap(
        raw_text=raw,
        token_ids=tuple(ids),
        offsets=tuple(offsets),
        tokenizer=tokenizer,
    )


def _selected_rank_events(episode_parse: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_index = episode_parse.get("selected_sequence_index")
    if raw_index is None:
        return []
    index = int(raw_index)
    sequences = list(episode_parse.get("sequences") or [])
    if not 0 <= index < len(sequences):
        raise CausalSiteError("Selected rank sequence index is out of bounds")
    return [dict(value) for value in sequences[index].get("events", [])]


def _char_site_index(parsed: Mapping[str, Any]) -> dict[tuple[int | None, str], dict[str, Any]]:
    output: dict[tuple[int | None, str], dict[str, Any]] = {}
    for raw in parsed.get("char_sites", []):
        site = dict(raw)
        occurrence = site.get("occurrence")
        key = (None if occurrence is None else int(occurrence), str(site["site_kind"]))
        if key in output:
            raise CausalSiteError(f"Duplicate character site: {key}")
        output[key] = site
    return output


def _surface_order(
    city_start: int,
    city_end: int,
    evidence_start: int | None,
    evidence_end: int | None,
) -> str:
    if evidence_start is None or evidence_end is None:
        return "no_explicit_rank_evidence"
    if int(evidence_end) <= int(city_start):
        return "rank_before_city"
    if int(city_end) <= int(evidence_start):
        return "rank_after_city"
    return "rank_city_overlap"


def _trim_semantic_whitespace(
    raw_text: str, char_start: int, char_end: int, *, role: str
) -> tuple[int, int, dict[str, int]]:
    """Remove parser-edge whitespace before selecting overlapping tokens.

    Byte-level BPEs commonly attach a separator space to the following word.
    If a parser evidence span is ``"First "``, mapping the untrimmed span would
    therefore select both ``First`` and `` excerpt``.  The whitespace carries
    no semantic marker content, so semantic item/rank/unit spans are normalized
    here.  City surfaces and the frozen ``Total: `` answer boundary are not
    passed through this helper.
    """

    parser_start = int(char_start)
    parser_end = int(char_end)
    if not 0 <= parser_start < parser_end <= len(raw_text):
        raise CausalSiteError(f"Invalid parser character span for {role}")
    start = parser_start
    end = parser_end
    while start < end and raw_text[start].isspace():
        start += 1
    while end > start and raw_text[end - 1].isspace():
        end -= 1
    if start >= end:
        raise CausalSiteError(f"Whitespace-only parser character span for {role}")
    return start, end, {
        "parser_char_start": parser_start,
        "parser_char_end": parser_end,
        "selected_char_start": start,
        "selected_char_end": end,
        "left_trimmed_chars": start - parser_start,
        "right_trimmed_chars": parser_end - end,
    }


def _rank_core_bounds(
    raw_text: str, surface_start: int, surface_end: int
) -> tuple[int, int, dict[str, int | bool]]:
    """Remove edge punctuation when an alphanumeric rank core remains.

    This separates semantic rank content (``Third`` or ``Count: 3``) from a
    sentence/list delimiter whose tokenizer token can also contain following
    newlines.  Punctuation-only invariant markers such as bullets are retained
    unchanged so structural controls remain representable.
    """

    start = int(surface_start)
    end = int(surface_end)

    def punctuation(value: str) -> bool:
        return unicodedata.category(value).startswith("P")

    candidate_start = start
    candidate_end = end
    while candidate_start < candidate_end and punctuation(raw_text[candidate_start]):
        candidate_start += 1
    while candidate_end > candidate_start and punctuation(raw_text[candidate_end - 1]):
        candidate_end -= 1
    use_candidate = bool(
        candidate_start < candidate_end
        and any(
            value.isalnum()
            for value in raw_text[candidate_start:candidate_end]
        )
    )
    core_start = candidate_start if use_candidate else start
    core_end = candidate_end if use_candidate else end
    return core_start, core_end, {
        "surface_char_start": start,
        "surface_char_end": end,
        "selected_char_start": core_start,
        "selected_char_end": core_end,
        "left_trimmed_punctuation_chars": core_start - start,
        "right_trimmed_punctuation_chars": end - core_end,
        "punctuation_only_surface_retained": not use_candidate,
    }


def _grammar_class(
    *,
    sequence_source: str,
    association: str,
    surface_order: str,
    marker_kind: str,
) -> str:
    if sequence_source == "synthetic_evidence_fallback":
        return "evidence_sequence_unranked"
    if sequence_source == "rank_supported_episode":
        if association == "same_unit":
            if surface_order == "rank_before_city":
                return "same_unit_rank_before_city"
            if surface_order == "rank_after_city":
                return "same_unit_rank_after_city"
            return "same_unit_rank_city_overlap"
        if association == "rank_before_city":
            return "adjacent_rank_before_city"
        if association == "rank_after_city":
            return "adjacent_rank_after_city"
        return "rank_supported_unknown_association"
    if marker_kind in {"indexed", "ordinal"}:
        return f"structural_explicit_{surface_order}"
    if marker_kind == "bullet":
        return "structural_invariant_bullet"
    return "structural_unmarked"


def _marker_semantics(
    *, sequence_source: str, marker_kind: str, has_evidence: bool
) -> str:
    if sequence_source == "synthetic_evidence_fallback":
        return "evidence_sequence_no_rank_marker"
    if sequence_source == "rank_supported_episode" and has_evidence:
        return "explicit_ordinal_or_count"
    if marker_kind in {"indexed", "ordinal"} and has_evidence:
        return "explicit_structural_ordinal"
    if marker_kind == "bullet" and has_evidence:
        return "invariant_marker"
    return "no_explicit_marker"


def _retrieval_surface_variant(
    *,
    grammar_class: str,
    raw_text: str,
    rank_end_char: int | None,
    city_start_char: int,
    has_record_clause: bool,
) -> tuple[str, str | None, bool | None]:
    """Split rank-before-city syntax by the material between rank and city.

    ``adjacent_rank_before_city`` is an association label, not a literal
    adjacency guarantee.  It covers both compact ``k. City`` items and long
    ``k. ... In the 2024 ... City`` clauses.  Causal query routing must not
    pool those surfaces because their candidate retrieval positions differ.
    """

    if grammar_class != "adjacent_rank_before_city" or rank_end_char is None:
        return f"other:{grammar_class}", None, None
    if int(rank_end_char) > int(city_start_char):
        return "rank_before_city_overlap_or_reversed", None, None
    interstitial = str(raw_text)[int(rank_end_char) : int(city_start_char)]
    lexical = any(value.isalnum() for value in interstitial)
    if has_record_clause:
        variant = "rank_before_city_record_clause"
    elif not lexical:
        variant = "rank_before_city_compact"
    else:
        variant = "rank_before_city_extended"
    return variant, interstitial, lexical


def classify_causal_cohort(
    *,
    sequence_source: str,
    trace_category: str,
    trace_one_to_one: bool,
    coverage_complete: bool,
    observed_item_count: int,
    gold_count: int,
) -> str:
    """Assign a scientific estimand without inspecting causal outcomes."""

    if sequence_source == "synthetic_evidence_fallback":
        if coverage_complete and int(observed_item_count) == int(gold_count):
            return "secondary_evidence_sequence_exploratory"
        return "secondary_evidence_sequence_partial_exploratory"
    if sequence_source == "structural_extension":
        return "secondary_structural_recap"
    if trace_one_to_one and trace_category == "one_to_one":
        if sequence_source == "rank_supported_episode":
            return "primary_rank_resolved_full_chain"
        return "secondary_structural_marker_neutral"
    if trace_category == "partial_unique":
        return "secondary_local_partial_unique"
    if trace_category in {"full_coverage_with_duplicates", "partial_with_duplicates"}:
        return "occurrence_retrieval_only_duplicates"
    return "audit_only_unresolved"


def _rank_event_rows(
    *, parsed: Mapping[str, Any], parser: Mapping[str, Any]
) -> list[dict[str, Any]]:
    events = _selected_rank_events(parsed["episode_parse"])
    item_count = int(parser["item_count"])
    if len(events) != item_count:
        raise CausalSiteError(
            "Rank-supported event count differs from selected parser item count"
        )
    cities = [str(value) for value in parser["item_gold_cities"]]
    markers = [int(value) for value in parser["item_markers"]]
    rows: list[dict[str, Any]] = []
    for occurrence, (event, city, marker) in enumerate(
        zip(events, cities, markers), start=1
    ):
        if str(event["city"]).casefold() != city.casefold():
            raise CausalSiteError("Rank event city differs from selected parser city")
        if int(event["rank"]) != marker or marker != occurrence:
            raise CausalSiteError("Rank-supported labels are not trace-local 1..M")
        rows.append(
            {
                "occurrence": occurrence,
                "rank": marker,
                "rank_basis": "observed_rank_evidence",
                "city": city,
                "event_source": "rank_supported",
                "association": str(event["association"]),
                "evidence_kind": str(event["evidence_kind"]),
                "evidence_family": str(event["evidence_family"]),
                "evidence_surface": str(event["evidence_surface"]),
                "city_start_char": int(event["city_start_char"]),
                "city_end_char": int(event["city_end_char"]),
                "city_unit_start_char": int(event["city_unit_start_char"]),
                "city_unit_end_char": int(event["city_unit_end_char"]),
                "rank_evidence_start_char": int(event["rank_evidence_start_char"]),
                "rank_evidence_end_char": int(event["rank_evidence_end_char"]),
                "semantic_start_char": int(event["semantic_start_char"]),
                "semantic_end_char": int(event["semantic_end_char"]),
            }
        )
    return rows


def _structural_event_rows(
    *, parsed: Mapping[str, Any], parser: Mapping[str, Any]
) -> list[dict[str, Any]]:
    sites = _char_site_index(parsed)
    rows: list[dict[str, Any]] = []
    item_count = int(parser["item_count"])
    arrays = (
        list(parser["item_markers"]),
        list(parser["item_gold_cities"]),
        list(parser["item_start_chars"]),
        list(parser["item_end_chars"]),
    )
    if any(len(values) != item_count for values in arrays):
        raise CausalSiteError("Structural parser item arrays are not aligned")
    for occurrence, (marker, city, item_start, item_end) in enumerate(
        zip(*arrays), start=1
    ):
        evidence_sequence = (
            parsed["sequence_source"] == "synthetic_evidence_fallback"
        )
        city_site = sites.get((occurrence, "city_end"))
        if city_site is None:
            raise CausalSiteError(f"Structural item {occurrence} lacks a city span")
        unit_site = sites.get((occurrence, "city_unit_end"))
        marker_site = sites.get((occurrence, "marker_end"))
        rows.append(
            {
                "occurrence": occurrence,
                "rank": int(marker) if isinstance(marker, int) else occurrence,
                "rank_basis": (
                    "compiler_occurrence_index_only"
                    if evidence_sequence
                    else "structural_marker_or_occurrence_index"
                ),
                "city": str(city),
                "event_source": (
                    "score_supported_evidence_sequence"
                    if evidence_sequence
                    else "structural"
                ),
                "association": "not_rank_supported",
                "evidence_kind": (
                    "score_supported_evidence_mention"
                    if evidence_sequence
                    else f"structural_{parser['marker_kind']}"
                ),
                "evidence_family": str(parser["marker_kind"]),
                "evidence_surface": "",
                "city_start_char": int(city_site["char_start"]),
                "city_end_char": int(city_site["char_end"]),
                "city_unit_start_char": int(
                    unit_site["char_start"] if unit_site is not None else item_start
                ),
                "city_unit_end_char": int(
                    unit_site["char_end"] if unit_site is not None else item_end
                ),
                "rank_evidence_start_char": (
                    None if marker_site is None else int(marker_site["char_start"])
                ),
                "rank_evidence_end_char": (
                    None if marker_site is None else int(marker_site["char_end"])
                ),
                "semantic_start_char": int(item_start),
                "semantic_end_char": int(item_end),
            }
        )
    return rows


def _with_absolute_tokens(site: dict[str, Any], prompt_count: int) -> dict[str, Any]:
    output = dict(site)
    if output.get("status") != "ok":
        return output
    output["full_sequence_token_start"] = int(prompt_count) + int(
        output["output_token_start"]
    )
    output["full_sequence_token_end"] = int(prompt_count) + int(
        output["output_token_end"]
    )
    return output


def _query_before(
    *,
    role: str,
    target: Mapping[str, Any],
    prompt_count: int,
    token_map: OutputTokenMap,
) -> dict[str, Any]:
    if target.get("status") != "ok":
        return {"role": role, "status": "target_token_span_unavailable"}
    token = int(target["output_token_start"]) - 1
    if token < 0:
        return {"role": role, "status": "no_preceding_output_token"}
    return {
        "role": role,
        "status": "ok",
        "output_token_index": token,
        "output_prefix_token_count": token + 1,
        "full_sequence_token_index": int(prompt_count) + token,
        "token_id": int(token_map.token_ids[token]),
        "token_text": token_map.decode([token_map.token_ids[token]]),
        "selection_rule": "token_immediately_before_first_token_overlapping_target",
    }


def _state_after(
    *,
    role: str,
    span: Mapping[str, Any],
    prompt_count: int,
    token_map: OutputTokenMap,
) -> dict[str, Any]:
    if span.get("status") != "ok":
        return {"role": role, "status": "span_token_interval_unavailable"}
    token = int(span["output_token_end"]) - 1
    return {
        "role": role,
        "status": "ok",
        "output_token_index": token,
        "output_prefix_token_count": token + 1,
        "full_sequence_token_index": int(prompt_count) + token,
        "token_id": int(token_map.token_ids[token]),
        "token_text": token_map.decode([token_map.token_ids[token]]),
        "selection_rule": "last_token_overlapping_semantic_span",
        "right_spill_chars": int(span.get("right_spill_chars", 0)),
    }


def _not_applicable(role: str, reason: str) -> dict[str, Any]:
    return {
        "role": role,
        "status": "not_applicable",
        "not_applicable_reason": reason,
    }


def _state_after_exact_boundary(
    *,
    role: str,
    span: Mapping[str, Any] | None,
    prompt_count: int,
    token_map: OutputTokenMap,
) -> dict[str, Any]:
    """Select the post-boundary token only when the delimiter is separable.

    A tokenizer token that overlaps both ``(`` and the following semantic text
    cannot distinguish a delimiter state from a label/city state.  Such cases
    are explicit N/A outcomes rather than silent fallbacks.
    """

    if span is None:
        return _not_applicable(role, "delimiter_not_present")
    if span.get("status") != "ok":
        return _not_applicable(role, "delimiter_token_span_unavailable")
    if int(span.get("right_spill_chars", 0)) > 0:
        return _not_applicable(role, "delimiter_fused_with_following_token")
    return _state_after(
        role=role,
        span=span,
        prompt_count=prompt_count,
        token_map=token_map,
    )


def _record_clause_span(
    *,
    raw_text: str,
    item_start: int,
    item_end: int,
    city_start: int,
    token_map: OutputTokenMap,
    prompt_count: int,
) -> dict[str, Any] | None:
    """Return the literal canonical audit clause associated with this item."""

    haystack = raw_text[int(item_start) : min(int(item_end), int(city_start))]
    relative = haystack.casefold().rfind(RECORD_CLAUSE.casefold())
    if relative < 0:
        return None
    start = int(item_start) + relative
    end = start + len(RECORD_CLAUSE)
    return _with_absolute_tokens(
        token_map.span("record_clause_span", start, end), prompt_count
    )


def _opening_delimiter_span(
    *,
    raw_text: str,
    item_start: int,
    city_start: int,
    marker_kind: str,
    token_map: OutputTokenMap,
    prompt_count: int,
) -> dict[str, Any] | None:
    """Resolve an opening bracket or invariant leading bullet before the city."""

    left = int(item_start)
    right = int(city_start)
    if not 0 <= left < right <= len(raw_text):
        return None
    prefix = raw_text[left:right]
    stripped = len(prefix) - len(prefix.lstrip())
    first = left + stripped
    if (
        marker_kind == "bullet"
        and first < right
        and raw_text[first] in BULLET_DELIMITERS
    ):
        return _with_absolute_tokens(
            token_map.span("opening_delimiter_span", first, first + 1),
            prompt_count,
        )

    stack: list[tuple[str, int]] = []
    for index in range(left, right):
        value = raw_text[index]
        if value in OPENING_DELIMITERS:
            stack.append((value, index))
        elif value in CLOSING_TO_OPENING:
            expected = CLOSING_TO_OPENING[value]
            for stack_index in range(len(stack) - 1, -1, -1):
                if stack[stack_index][0] == expected:
                    del stack[stack_index:]
                    break
    if not stack:
        return None
    _value, index = stack[-1]
    return _with_absolute_tokens(
        token_map.span("opening_delimiter_span", index, index + 1), prompt_count
    )


def _token_shell(
    *,
    role: str,
    surface: Mapping[str, Any],
    core: Mapping[str, Any],
    token_map: OutputTokenMap,
    prompt_count: int,
) -> dict[str, Any]:
    """Return surface tokens outside a nested semantic core interval."""

    if surface.get("status") != "ok" or core.get("status") != "ok":
        return {"role": role, "status": "source_span_unavailable"}
    surface_start = int(surface["output_token_start"])
    surface_end = int(surface["output_token_end"])
    core_start = int(core["output_token_start"])
    core_end = int(core["output_token_end"])
    if not surface_start <= core_start < core_end <= surface_end:
        return {"role": role, "status": "core_not_nested_in_surface"}
    intervals = [
        (start, end)
        for start, end in (
            (surface_start, core_start),
            (core_end, surface_end),
        )
        if start < end
    ]
    indices = [index for start, end in intervals for index in range(start, end)]
    ids = [int(token_map.token_ids[index]) for index in indices]
    return {
        "role": role,
        "status": "ok" if indices else "empty_shell",
        "output_token_intervals": [
            {"start": start, "end": end} for start, end in intervals
        ],
        "output_token_indices": indices,
        "full_sequence_token_indices": [
            int(prompt_count) + index for index in indices
        ],
        "token_ids": ids,
        "token_text": token_map.decode(ids) if ids else "",
        "token_count": len(ids),
    }


def _prompt_source_records(row: Mapping[str, Any], token_map: OutputTokenMap) -> dict[str, dict[str, Any]]:
    prompt_ids = tuple(prompt_token_ids(row))
    output: dict[str, dict[str, Any]] = {}
    for raw in row.get("prompt_record_spans") or []:
        city = str(raw["city"])
        key = city.casefold()
        if key in output:
            raise CausalSiteError(f"Duplicate prompt source record for city {city}")
        start = int(raw["start"])
        end = int(raw["end"])
        if not 0 <= start < end <= len(prompt_ids):
            raise CausalSiteError(f"Prompt source record is out of bounds: {city}")
        ids = prompt_ids[start:end]
        token_text = token_map.decode(ids)
        if city.casefold() not in token_text.casefold():
            raise CausalSiteError(
                f"Prompt source record span does not decode to its city: {city}"
            )
        output[key] = {
            "status": "ok",
            "slot_index": int(raw["slot_index"]),
            "city": city,
            "score": None if raw.get("score") is None else int(raw["score"]),
            "prompt_token_start": start,
            "prompt_token_end": end,
            "full_sequence_token_start": start,
            "full_sequence_token_end": end,
            "token_count": end - start,
            "token_text": token_text,
        }
    expected = {str(value["city"]).casefold() for value in gold_records(row)}
    if set(output) != expected:
        raise CausalSiteError("Prompt source registry differs from gold city registry")
    return output


def _event_plan(
    raw_event: Mapping[str, Any],
    *,
    sequence_source: str,
    marker_kind: str,
    cohort: str,
    token_map: OutputTokenMap,
    prompt_count: int,
    source_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    city_start = int(raw_event["city_start_char"])
    city_end = int(raw_event["city_end_char"])
    item_start, item_end, item_normalization = _trim_semantic_whitespace(
        token_map.raw_text,
        int(raw_event["semantic_start_char"]),
        int(raw_event["semantic_end_char"]),
        role="semantic_item_span",
    )
    unit_start, unit_end, unit_normalization = _trim_semantic_whitespace(
        token_map.raw_text,
        int(raw_event["city_unit_start_char"]),
        int(raw_event["city_unit_end_char"]),
        role="city_unit_span",
    )
    raw_rank_start = raw_event.get("rank_evidence_start_char")
    raw_rank_end = raw_event.get("rank_evidence_end_char")
    rank_surface_start = None
    rank_surface_end = None
    rank_start = None
    rank_end = None
    rank_surface_normalization = None
    rank_core_normalization = None
    if raw_rank_start is not None and raw_rank_end is not None:
        (
            rank_surface_start,
            rank_surface_end,
            rank_surface_normalization,
        ) = _trim_semantic_whitespace(
            token_map.raw_text,
            int(raw_rank_start),
            int(raw_rank_end),
            role="rank_evidence_surface_span",
        )
        rank_start, rank_end, rank_core_normalization = _rank_core_bounds(
            token_map.raw_text,
            rank_surface_start,
            rank_surface_end,
        )
    order = _surface_order(city_start, city_end, rank_start, rank_end)
    grammar = _grammar_class(
        sequence_source=sequence_source,
        association=str(raw_event["association"]),
        surface_order=order,
        marker_kind=marker_kind,
    )
    item = _with_absolute_tokens(
        token_map.span(
            "semantic_item_span",
            item_start,
            item_end,
        ),
        prompt_count,
    )
    city_unit = _with_absolute_tokens(
        token_map.span(
            "city_unit_span",
            unit_start,
            unit_end,
        ),
        prompt_count,
    )
    city = _with_absolute_tokens(
        token_map.span("city_target_span", city_start, city_end), prompt_count
    )
    rank = None
    rank_surface = None
    rank_shell = None
    if rank_start is not None and rank_end is not None:
        rank = _with_absolute_tokens(
            token_map.span("rank_evidence_core_span", rank_start, rank_end),
            prompt_count,
        )
        rank_surface = _with_absolute_tokens(
            token_map.span(
                "rank_evidence_surface_span",
                int(rank_surface_start),
                int(rank_surface_end),
            ),
            prompt_count,
        )
        rank_shell = _token_shell(
            role="rank_visible_format_shell_tokens",
            surface=rank_surface,
            core=rank,
            token_map=token_map,
            prompt_count=prompt_count,
        )
    city_pre = _query_before(
        role="city_pre_d1",
        target=city,
        prompt_count=prompt_count,
        token_map=token_map,
    )
    unit_pre = _query_before(
        role="unit_pre_d1",
        target=item,
        prompt_count=prompt_count,
        token_map=token_map,
    )
    clause_span = _record_clause_span(
        raw_text=token_map.raw_text,
        item_start=item_start,
        item_end=item_end,
        city_start=city_start,
        token_map=token_map,
        prompt_count=prompt_count,
    )
    clause_pre = (
        _not_applicable("record_clause_pre_d1", "canonical_clause_not_present")
        if clause_span is None
        else _query_before(
            role="record_clause_pre_d1",
            target=clause_span,
            prompt_count=prompt_count,
            token_map=token_map,
        )
    )
    (
        retrieval_surface_variant,
        rank_to_city_interstitial_text,
        rank_to_city_has_lexical_content,
    ) = _retrieval_surface_variant(
        grammar_class=grammar,
        raw_text=token_map.raw_text,
        rank_end_char=rank_end,
        city_start_char=city_start,
        has_record_clause=clause_span is not None,
    )
    rank_to_city_interstitial_token_count = (
        None
        if rank is None
        or rank.get("output_token_end") is None
        or city.get("output_token_start") is None
        else int(city["output_token_start"])
        - int(rank["output_token_end"])
    )
    delimiter_span = _opening_delimiter_span(
        raw_text=token_map.raw_text,
        item_start=item_start,
        city_start=city_start,
        marker_kind=marker_kind,
        token_map=token_map,
        prompt_count=prompt_count,
    )
    delimiter_post = _state_after_exact_boundary(
        role="post_open_delimiter",
        span=delimiter_span,
        prompt_count=prompt_count,
        token_map=token_map,
    )
    city_end_state = _state_after(
        role="city_end_state",
        span=city,
        prompt_count=prompt_count,
        token_map=token_map,
    )
    commit = _state_after(
        role="post_update_commit_state",
        span=item,
        prompt_count=prompt_count,
        token_map=token_map,
    )
    marker_pre = None
    marker_post = None
    if rank is not None:
        marker_pre = _query_before(
            role="pre_marker_state",
            target=rank,
            prompt_count=prompt_count,
            token_map=token_map,
        )
        marker_post = _state_after(
            role="post_marker_state",
            span=rank,
            prompt_count=prompt_count,
            token_map=token_map,
        )
    marker_semantics = _marker_semantics(
        sequence_source=sequence_source,
        marker_kind=marker_kind,
        has_evidence=rank is not None,
    )
    exclusions: list[str] = []
    if city_pre["status"] != "ok" or city["status"] != "ok":
        exclusions.append("retrieval_site_unresolved")
    if commit["status"] != "ok":
        exclusions.append("commit_site_unresolved")
    if grammar in {"same_unit_rank_city_overlap", "rank_supported_unknown_association"}:
        exclusions.append("ambiguous_rank_city_order")
    if rank is not None and int(rank.get("right_spill_chars", 0)) > 0:
        exclusions.append("rank_core_token_right_spill")
    source = source_records.get(str(raw_event["city"]).casefold())
    if source is None:
        exclusions.append("prompt_source_record_missing")
    retrieval_eligible = not any(
        reason in exclusions
        for reason in (
            "retrieval_site_unresolved",
            "ambiguous_rank_city_order",
            "prompt_source_record_missing",
        )
    )
    progress_site_resolved = not any(
        reason in exclusions
        for reason in (
            "commit_site_unresolved",
            "ambiguous_rank_city_order",
        )
    )
    progress_eligible = (
        progress_site_resolved and cohort in CONTINUE_ELIGIBLE_COHORTS
    )
    primary = cohort == "primary_rank_resolved_full_chain"
    return {
        "occurrence": int(raw_event["occurrence"]),
        "rank": int(raw_event["rank"]),
        "rank_basis": str(raw_event["rank_basis"]),
        "city": str(raw_event["city"]),
        "event_source": str(raw_event["event_source"]),
        "association": str(raw_event["association"]),
        "surface_order": order,
        "grammar_class": grammar,
        "retrieval_surface_variant": retrieval_surface_variant,
        "rank_to_city_interstitial_char_count": (
            None
            if rank_to_city_interstitial_text is None
            else len(rank_to_city_interstitial_text)
        ),
        "rank_to_city_interstitial_token_count": (
            rank_to_city_interstitial_token_count
        ),
        "rank_to_city_has_lexical_content": (
            rank_to_city_has_lexical_content
        ),
        "evidence_kind": str(raw_event["evidence_kind"]),
        "evidence_family": str(raw_event["evidence_family"]),
        "evidence_surface": str(raw_event.get("evidence_surface") or ""),
        "marker_semantics": marker_semantics,
        "semantic_span_normalization": {
            "semantic_item_span": item_normalization,
            "city_unit_span": unit_normalization,
            "rank_evidence_surface_span": rank_surface_normalization,
            "rank_evidence_core_span": rank_core_normalization,
        },
        "prompt_source_record": None if source is None else dict(source),
        "sites": {
            "semantic_item_span": item,
            "city_unit_span": city_unit,
            "city_target_span": city,
            "rank_evidence_core_span": rank,
            "rank_evidence_surface_span": rank_surface,
            "rank_visible_format_shell_tokens": rank_shell,
            # ``retrieve_query_state`` is a compatibility alias. New causal
            # runners consume the explicit transition anchor registry below.
            "retrieve_query_state": city_pre,
            "city_pre_d1": city_pre,
            "unit_pre_d1": unit_pre,
            "record_clause_span": clause_span,
            "record_clause_pre_d1": clause_pre,
            "opening_delimiter_span": delimiter_span,
            "post_open_delimiter": delimiter_post,
            "city_end_state": city_end_state,
            "pre_marker_state": marker_pre,
            "post_marker_state": marker_post,
            "post_update_commit_state": commit,
        },
        "eligibility": {
            "retrieval": retrieval_eligible,
            "marker_control": (
                marker_semantics
                in {"explicit_ordinal_or_count", "explicit_structural_ordinal"}
                and marker_pre is not None
                and marker_post is not None
                and marker_pre["status"] == "ok"
                and marker_post["status"] == "ok"
                and int(rank.get("right_spill_chars", 0)) == 0
                and cohort in MARKER_CONTROL_ELIGIBLE_COHORTS
            ),
            "format_shell_control": (
                rank_shell is not None
                and rank_shell["status"] == "ok"
                and cohort in MARKER_CONTROL_ELIGIBLE_COHORTS
            ),
            "invariant_marker_surface_control": (
                marker_semantics == "invariant_marker"
                and rank_surface is not None
                and rank_surface["status"] == "ok"
                and cohort in INVARIANT_SURFACE_ELIGIBLE_COHORTS
            ),
            "progress_commit_site_resolved": progress_site_resolved,
            "progress_commit": progress_eligible,
            "primary_full_chain_event": primary
            and retrieval_eligible
            and progress_eligible,
            "structural_marker_neutral_secondary": cohort
            == "secondary_structural_marker_neutral",
        },
        "exclusion_reasons": exclusions,
    }


def _continuation_text(
    token_map: OutputTokenMap, start: int, end: int
) -> tuple[list[int], str]:
    ids = list(token_map.token_ids[int(start) : int(end)])
    return ids, token_map.decode(ids)


def _anchor_candidate(
    role: str,
    site: Mapping[str, Any] | None,
    *,
    target_city_token_start: int | None,
    event_specific: bool,
    timing_stage: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "anchor_role": role,
        "timing_stage": timing_stage,
        "event_specific": bool(event_specific),
    }
    if site is None:
        return {
            **payload,
            "status": "not_applicable",
            "not_applicable_reason": "semantic_site_not_present",
        }
    status = str(site.get("status", "unresolved"))
    if status != "ok":
        return {
            **payload,
            "status": status,
            "not_applicable_reason": site.get(
                "not_applicable_reason", "semantic_site_unresolved"
            ),
        }
    query = site.get("output_token_index")
    if query is None:
        return {
            **payload,
            "status": "unresolved",
            "not_applicable_reason": "query_token_missing",
        }
    query = int(query)
    if target_city_token_start is None or query >= int(target_city_token_start):
        return {
            **payload,
            "status": "not_applicable",
            "not_applicable_reason": "anchor_not_strictly_before_target_city",
            "output_token_index": query,
        }
    return {
        **payload,
        "status": "ok",
        "output_token_index": query,
        "full_sequence_token_index": site.get("full_sequence_token_index"),
        "token_id": site.get("token_id"),
        "token_text": site.get("token_text", ""),
        "selection_rule": site.get("selection_rule"),
    }


def _deduplicate_anchor_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    from_occurrence: int,
    to_occurrence: int,
) -> list[dict[str, Any]]:
    by_token: dict[int, list[Mapping[str, Any]]] = {}
    for candidate in candidates:
        if candidate.get("status") != "ok":
            continue
        by_token.setdefault(int(candidate["output_token_index"]), []).append(candidate)
    anchors: list[dict[str, Any]] = []
    for query, equivalent in sorted(by_token.items()):
        ordered = sorted(
            equivalent,
            key=lambda value: (
                ANCHOR_ROLE_PRIORITY.get(str(value["anchor_role"]), 999),
                str(value["anchor_role"]),
            ),
        )
        canonical = dict(ordered[0])
        roles = [str(value["anchor_role"]) for value in ordered]
        canonical.update(
            {
                "anchor_role": roles[0],
                "anchor_roles": roles,
                "anchor_equivalence_id": (
                    f"{int(from_occurrence)}->{int(to_occurrence)}@q{query}"
                ),
                "event_specific": any(
                    bool(value.get("event_specific")) for value in ordered
                ),
            }
        )
        anchors.append(canonical)
    return anchors


def _transition_plan(
    current: Mapping[str, Any],
    following: Mapping[str, Any],
    *,
    cohort: str,
    token_map: OutputTokenMap,
    prompt_count: int,
    block_pre: Mapping[str, Any] | None,
) -> dict[str, Any]:
    query = current["sites"]["post_update_commit_state"]
    city = following["sites"]["city_target_span"]
    exclusions: list[str] = []
    if query.get("status") != "ok":
        exclusions.append("current_commit_unresolved")
    if city.get("status") != "ok":
        exclusions.append("next_city_target_unresolved")
    full_start = None
    full_end = None
    full_ids: list[int] = []
    full_text = ""
    if not exclusions:
        full_start = int(query["output_token_index"]) + 1
        full_end = int(city["output_token_end"])
        if full_start >= full_end:
            exclusions.append("empty_or_reversed_next_city_continuation")
        else:
            full_ids, full_text = _continuation_text(
                token_map, full_start, full_end
            )
    local_eligible = not exclusions and bool(
        current["eligibility"]["progress_commit"]
        and following["eligibility"]["retrieval"]
    )
    following_sites = following["sites"]
    marker_before_city = following.get("surface_order") == "rank_before_city"
    marker_pre = (
        following_sites.get("pre_marker_state")
        if marker_before_city
        else _not_applicable(
            "pre_marker_d1", "rank_marker_not_before_target_city"
        )
    )
    marker_post = (
        following_sites.get("post_marker_state")
        if marker_before_city
        else _not_applicable(
            "post_marker", "rank_marker_not_before_target_city"
        )
    )
    city_start = city.get("output_token_start")
    anchor_candidates = [
        _anchor_candidate(
            "p0_item_end",
            query,
            target_city_token_start=city_start,
            event_specific=True,
            timing_stage="P0_previous_item_commit",
        ),
        _anchor_candidate(
            "block_pre_d1",
            block_pre,
            target_city_token_start=city_start,
            event_specific=False,
            timing_stage="exploratory_block_entry",
        ),
        _anchor_candidate(
            "unit_pre_d1",
            following_sites.get("unit_pre_d1"),
            target_city_token_start=city_start,
            event_specific=True,
            timing_stage="P1_target_unit_entry",
        ),
        _anchor_candidate(
            "pre_marker_d1",
            marker_pre,
            target_city_token_start=city_start,
            event_specific=True,
            timing_stage="P1_pre_marker",
        ),
        _anchor_candidate(
            "post_open_delimiter",
            following_sites.get("post_open_delimiter"),
            target_city_token_start=city_start,
            event_specific=True,
            timing_stage="P2_post_delimiter",
        ),
        _anchor_candidate(
            "post_marker",
            marker_post,
            target_city_token_start=city_start,
            event_specific=True,
            timing_stage="P2_post_marker",
        ),
        _anchor_candidate(
            "record_clause_pre_d1",
            following_sites.get("record_clause_pre_d1"),
            target_city_token_start=city_start,
            event_specific=True,
            timing_stage="P1_record_clause_entry",
        ),
        _anchor_candidate(
            "city_pre_d1",
            following_sites.get("city_pre_d1"),
            target_city_token_start=city_start,
            event_specific=True,
            timing_stage="P3_city_emission",
        ),
    ]
    anchors = _deduplicate_anchor_candidates(
        anchor_candidates,
        from_occurrence=int(current["occurrence"]),
        to_occurrence=int(following["occurrence"]),
    )
    for anchor in anchors:
        anchor["local_anchor_eligible"] = bool(local_eligible)
        anchor["primary_anchor_eligible"] = bool(
            local_eligible
            and cohort == "primary_rank_resolved_full_chain"
            and anchor.get("event_specific")
        )
    if local_eligible and not anchors:
        exclusions.append("no_resolved_retrieval_anchor")
        local_eligible = False
    return {
        "transition_kind": "continue_to_next_city",
        "from_occurrence": int(current["occurrence"]),
        "to_occurrence": int(following["occurrence"]),
        "current_city": str(current["city"]),
        "target_city": str(following["city"]),
        "grammar_pair": f"{current['grammar_class']} -> {following['grammar_class']}",
        "target_retrieval_surface_variant": str(
            following["retrieval_surface_variant"]
        ),
        "target_rank_to_city_interstitial_char_count": following.get(
            "rank_to_city_interstitial_char_count"
        ),
        "target_rank_to_city_interstitial_token_count": following.get(
            "rank_to_city_interstitial_token_count"
        ),
        "target_rank_to_city_has_lexical_content": following.get(
            "rank_to_city_has_lexical_content"
        ),
        "query_output_token_index": query.get("output_token_index"),
        "query_full_sequence_token_index": query.get("full_sequence_token_index"),
        "full_continuation_output_token_start": full_start,
        "full_continuation_output_token_end": full_end,
        "full_continuation_full_sequence_token_start": (
            None if full_start is None else int(prompt_count) + full_start
        ),
        "full_continuation_full_sequence_token_end": (
            None if full_end is None else int(prompt_count) + full_end
        ),
        "full_continuation_token_ids": full_ids,
        "full_continuation_token_text": full_text,
        "next_city_output_token_start": city.get("output_token_start"),
        "next_city_output_token_end": city.get("output_token_end"),
        "next_city_token_ids": city.get("token_ids", []),
        "next_city_token_text": city.get("token_text", ""),
        "anchor_candidates": anchor_candidates,
        "anchors": anchors,
        "local_transition_eligible": local_eligible,
        "primary_transition_eligible": local_eligible
        and cohort == "primary_rank_resolved_full_chain",
        "exclusion_reasons": exclusions,
    }


def _terminal_plan(
    last_event: Mapping[str, Any],
    *,
    answer_query: Mapping[str, Any] | None,
    cohort: str,
    token_map: OutputTokenMap,
    prompt_count: int,
) -> dict[str, Any]:
    query = last_event["sites"]["post_update_commit_state"]
    exclusions: list[str] = []
    if query.get("status") != "ok":
        exclusions.append("terminal_commit_unresolved")
    if answer_query is None or answer_query.get("status") != "ok":
        exclusions.append("answer_query_v3_unresolved")
    start = None
    end = None
    ids: list[int] = []
    text = ""
    if not exclusions:
        start = int(query["output_token_index"]) + 1
        end = int(answer_query["output_token_end"])
        if start >= end:
            exclusions.append("empty_or_reversed_stop_continuation")
        else:
            ids, text = _continuation_text(token_map, start, end)
    answer_ids = [] if answer_query is None else list(answer_query.get("token_ids", []))
    local_eligible = not exclusions and cohort in STOP_ELIGIBLE_COHORTS and bool(
        last_event["eligibility"]["progress_commit"]
    )
    return {
        "transition_kind": "stop_to_answer_query_v3",
        "from_occurrence": int(last_event["occurrence"]),
        "query_output_token_index": query.get("output_token_index"),
        "query_full_sequence_token_index": query.get("full_sequence_token_index"),
        "full_continuation_output_token_start": start,
        "full_continuation_output_token_end": end,
        "full_continuation_full_sequence_token_start": (
            None if start is None else int(prompt_count) + start
        ),
        "full_continuation_full_sequence_token_end": (
            None if end is None else int(prompt_count) + end
        ),
        "full_continuation_token_ids": ids,
        "full_continuation_token_text": text,
        "answer_query_output_token_start": (
            None if answer_query is None else answer_query.get("output_token_start")
        ),
        "answer_query_output_token_end": (
            None if answer_query is None else answer_query.get("output_token_end")
        ),
        "answer_query_token_ids": answer_ids,
        "answer_query_token_text": (
            "" if answer_query is None else answer_query.get("token_text", "")
        ),
        "observed_stop_eligible": local_eligible,
        "primary_terminal_eligible": local_eligible
        and cohort == "primary_rank_resolved_full_chain",
        "exclusion_reasons": exclusions,
    }


def compile_causal_site_plan(
    row: Mapping[str, Any], tokenizer: Any
) -> dict[str, Any]:
    """Compile one stored native-thinking trajectory into causal roles."""

    parsed = parse_trace_record(row)
    parser = dict(parsed["parser"])
    if not parser.get("detected"):
        raise CausalSiteError("Running trace parser did not recover an event sequence")
    token_map = build_output_token_map(row, tokenizer)
    prompt_count = len(prompt_token_ids(row))
    source_records = _prompt_source_records(row, token_map)
    sequence_source = str(parsed["sequence_source"])
    trace_category = str(parser["trace_category"])
    trace_one_to_one = bool(parser["trace_one_to_one"])
    cohort = classify_causal_cohort(
        sequence_source=sequence_source,
        trace_category=trace_category,
        trace_one_to_one=trace_one_to_one,
        coverage_complete=bool(parser["coverage_complete"]),
        observed_item_count=int(parser["item_count"]),
        gold_count=len(gold_records(row)),
    )
    if sequence_source == "rank_supported_episode":
        raw_events = _rank_event_rows(parsed=parsed, parser=parser)
    else:
        raw_events = _structural_event_rows(parsed=parsed, parser=parser)
    events = [
        _event_plan(
            event,
            sequence_source=sequence_source,
            marker_kind=str(parser["marker_kind"]),
            cohort=cohort,
            token_map=token_map,
            prompt_count=prompt_count,
            source_records=source_records,
        )
        for event in raw_events
    ]
    block_pre = _query_before(
        role="block_pre_d1",
        target=events[0]["sites"]["semantic_item_span"],
        prompt_count=prompt_count,
        token_map=token_map,
    )
    transitions = [
        _transition_plan(
            current,
            following,
            cohort=cohort,
            token_map=token_map,
            prompt_count=prompt_count,
            block_pre=block_pre,
        )
        for current, following in zip(events, events[1:])
    ]
    char_sites = _char_site_index(parsed)
    answer_char = char_sites.get((None, "answer_query_v3"))
    answer_query = None
    if answer_char is not None:
        answer_query = _with_absolute_tokens(
            token_map.span(
                "answer_query_v3_span",
                int(answer_char["char_start"]),
                int(answer_char["char_end"]),
            ),
            prompt_count,
        )
    terminal = _terminal_plan(
        events[-1],
        answer_query=answer_query,
        cohort=cohort,
        token_map=token_map,
        prompt_count=prompt_count,
    )
    grammar_classes = sorted({str(event["grammar_class"]) for event in events})
    plan_exclusions = sorted(
        {
            reason
            for event in events
            for reason in event["exclusion_reasons"]
        }
        | {
            reason
            for transition in transitions
            for reason in transition["exclusion_reasons"]
        }
        | set(terminal["exclusion_reasons"])
    )
    primary_complete = bool(
        cohort == "primary_rank_resolved_full_chain"
        and all(event["eligibility"]["primary_full_chain_event"] for event in events)
        and all(transition["primary_transition_eligible"] for transition in transitions)
        and terminal["primary_terminal_eligible"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "token_alignment_version": TOKEN_ALIGNMENT_VERSION,
        "grammar_policy_version": GRAMMAR_POLICY_VERSION,
        "cohort_policy_version": COHORT_POLICY_VERSION,
        "request_id": str(row.get("request_id", row.get("stimulus_id"))),
        "stimulus_id": str(row.get("stimulus_id", "")),
        "model_label": str(row.get("model_label", "")),
        "model_family": str(row.get("model_family", "")),
        "seed": int(row.get("seed", -1)),
        "split": str(row.get("split", "")),
        "gold_count": len(gold_records(row)),
        "parsed_count": parsed.get("parsed_count"),
        "exact_count": bool(parsed.get("exact_count")),
        "prompt_token_count": prompt_count,
        "output_token_count": len(token_map.token_ids),
        "token_reencode_exact": True,
        "prompt_source_registry_exact": True,
        "sequence_source": sequence_source,
        "marker_kind": str(parser["marker_kind"]),
        "trace_category": trace_category,
        "trace_one_to_one": trace_one_to_one,
        "trace_order_class": str(parser["trace_order_class"]),
        "observed_item_count": int(parser["item_count"]),
        "coverage_complete": bool(parser["coverage_complete"]),
        "causal_cohort": cohort,
        "allowed_estimands": list(COHORT_ALLOWED_ESTIMANDS[cohort]),
        "grammar_signature": " + ".join(grammar_classes),
        "mixed_event_grammar": len(grammar_classes) > 1,
        "answer_query_v3_span": answer_query,
        "block_pre_d1": block_pre,
        "events": events,
        "transitions": transitions,
        "terminal_transition": terminal,
        "primary_full_chain_site_complete": primary_complete,
        "plan_exclusion_reasons": plan_exclusions,
    }


def flatten_event_rows(plan: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield compact event rows for CSV review."""

    base = {
        key: plan.get(key)
        for key in (
            "request_id",
            "model_label",
            "seed",
            "split",
            "gold_count",
            "parsed_count",
            "exact_count",
            "sequence_source",
            "marker_kind",
            "trace_category",
            "causal_cohort",
            "grammar_signature",
        )
    }
    base["allowed_estimands"] = "|".join(plan.get("allowed_estimands", []))
    for event in plan.get("events", []):
        city = event["sites"]["city_target_span"]
        rank = event["sites"].get("rank_evidence_core_span")
        rank_surface = event["sites"].get("rank_evidence_surface_span")
        rank_shell = event["sites"].get("rank_visible_format_shell_tokens")
        query = event["sites"]["retrieve_query_state"]
        commit = event["sites"]["post_update_commit_state"]
        item = event["sites"]["semantic_item_span"]
        source = event.get("prompt_source_record") or {}
        yield {
            **base,
            "occurrence": event["occurrence"],
            "rank": event["rank"],
            "rank_basis": event["rank_basis"],
            "city": event["city"],
            "event_source": event["event_source"],
            "association": event["association"],
            "surface_order": event["surface_order"],
            "grammar_class": event["grammar_class"],
            "retrieval_surface_variant": event[
                "retrieval_surface_variant"
            ],
            "rank_to_city_interstitial_char_count": event[
                "rank_to_city_interstitial_char_count"
            ],
            "rank_to_city_interstitial_token_count": event[
                "rank_to_city_interstitial_token_count"
            ],
            "rank_to_city_has_lexical_content": event[
                "rank_to_city_has_lexical_content"
            ],
            "evidence_kind": event["evidence_kind"],
            "marker_semantics": event["marker_semantics"],
            "source_record_token_start": source.get("prompt_token_start"),
            "source_record_token_end": source.get("prompt_token_end"),
            "source_record_token_text": source.get("token_text", ""),
            "retrieve_query_output_token": query.get("output_token_index"),
            "retrieve_query_token_id": query.get("token_id"),
            "retrieve_query_token_text": query.get("token_text", ""),
            "city_output_token_start": city.get("output_token_start"),
            "city_output_token_end": city.get("output_token_end"),
            "city_token_ids": json.dumps(city.get("token_ids", [])),
            "city_token_text": city.get("token_text", ""),
            "city_left_spill_chars": city.get("left_spill_chars"),
            "city_right_spill_chars": city.get("right_spill_chars"),
            "rank_output_token_start": None if rank is None else rank.get("output_token_start"),
            "rank_output_token_end": None if rank is None else rank.get("output_token_end"),
            "rank_token_ids": "" if rank is None else json.dumps(rank.get("token_ids", [])),
            "rank_token_text": "" if rank is None else rank.get("token_text", ""),
            "rank_left_spill_chars": None if rank is None else rank.get("left_spill_chars"),
            "rank_right_spill_chars": None if rank is None else rank.get("right_spill_chars"),
            "rank_surface_output_token_start": (
                None
                if rank_surface is None
                else rank_surface.get("output_token_start")
            ),
            "rank_surface_output_token_end": (
                None
                if rank_surface is None
                else rank_surface.get("output_token_end")
            ),
            "rank_surface_token_ids": (
                ""
                if rank_surface is None
                else json.dumps(rank_surface.get("token_ids", []))
            ),
            "rank_surface_token_text": (
                "" if rank_surface is None else rank_surface.get("token_text", "")
            ),
            "rank_format_shell_token_indices": (
                ""
                if rank_shell is None
                else json.dumps(rank_shell.get("output_token_indices", []))
            ),
            "rank_format_shell_token_ids": (
                ""
                if rank_shell is None
                else json.dumps(rank_shell.get("token_ids", []))
            ),
            "rank_format_shell_token_text": (
                "" if rank_shell is None else rank_shell.get("token_text", "")
            ),
            "rank_format_shell_token_count": (
                None if rank_shell is None else rank_shell.get("token_count")
            ),
            "commit_output_token": commit.get("output_token_index"),
            "commit_token_id": commit.get("token_id"),
            "commit_token_text": commit.get("token_text", ""),
            "item_output_token_start": item.get("output_token_start"),
            "item_output_token_end": item.get("output_token_end"),
            "item_left_spill_chars": item.get("left_spill_chars"),
            "item_right_spill_chars": item.get("right_spill_chars"),
            "item_left_trimmed_chars": event["semantic_span_normalization"][
                "semantic_item_span"
            ]["left_trimmed_chars"],
            "item_right_trimmed_chars": event["semantic_span_normalization"][
                "semantic_item_span"
            ]["right_trimmed_chars"],
            "rank_left_trimmed_chars": (
                None
                if event["semantic_span_normalization"][
                    "rank_evidence_surface_span"
                ]
                is None
                else event["semantic_span_normalization"][
                    "rank_evidence_surface_span"
                ]["left_trimmed_chars"]
            ),
            "rank_right_trimmed_chars": (
                None
                if event["semantic_span_normalization"][
                    "rank_evidence_surface_span"
                ]
                is None
                else event["semantic_span_normalization"][
                    "rank_evidence_surface_span"
                ]["right_trimmed_chars"]
            ),
            "rank_left_trimmed_punctuation_chars": (
                None
                if event["semantic_span_normalization"]["rank_evidence_core_span"]
                is None
                else event["semantic_span_normalization"][
                    "rank_evidence_core_span"
                ]["left_trimmed_punctuation_chars"]
            ),
            "rank_right_trimmed_punctuation_chars": (
                None
                if event["semantic_span_normalization"]["rank_evidence_core_span"]
                is None
                else event["semantic_span_normalization"][
                    "rank_evidence_core_span"
                ]["right_trimmed_punctuation_chars"]
            ),
            "city_text": city.get("char_text", ""),
            "rank_text": "" if rank is None else rank.get("char_text", ""),
            "rank_surface_text": (
                "" if rank_surface is None else rank_surface.get("char_text", "")
            ),
            "item_text": item.get("char_text", ""),
            "retrieval_eligible": event["eligibility"]["retrieval"],
            "marker_control_eligible": event["eligibility"]["marker_control"],
            "format_shell_control_eligible": event["eligibility"][
                "format_shell_control"
            ],
            "invariant_marker_surface_control_eligible": event["eligibility"][
                "invariant_marker_surface_control"
            ],
            "progress_commit_eligible": event["eligibility"]["progress_commit"],
            "progress_commit_site_resolved": event["eligibility"][
                "progress_commit_site_resolved"
            ],
            "primary_full_chain_event": event["eligibility"]["primary_full_chain_event"],
            "exclusion_reasons": "|".join(event["exclusion_reasons"]),
        }


def flatten_anchor_rows(plan: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield one row per semantic candidate, including explicit N/A outcomes."""

    base = {
        key: plan.get(key)
        for key in (
            "request_id",
            "model_label",
            "seed",
            "split",
            "gold_count",
            "sequence_source",
            "trace_category",
            "causal_cohort",
            "grammar_signature",
        )
    }
    base["allowed_estimands"] = "|".join(plan.get("allowed_estimands", []))
    for transition in plan.get("transitions", []):
        resolved = {
            role: anchor
            for anchor in transition.get("anchors", [])
            for role in anchor.get("anchor_roles", [anchor.get("anchor_role")])
        }
        for candidate in transition.get("anchor_candidates", []):
            role = str(candidate["anchor_role"])
            anchor = resolved.get(role)
            yield {
                **base,
                "transition_kind": transition["transition_kind"],
                "from_occurrence": transition["from_occurrence"],
                "to_occurrence": transition.get("to_occurrence"),
                "target_city": transition.get("target_city"),
                "grammar_pair": transition.get("grammar_pair"),
                "target_grammar_class": str(transition.get("grammar_pair", "")).split(
                    " -> "
                )[-1],
                "target_retrieval_surface_variant": transition.get(
                    "target_retrieval_surface_variant"
                ),
                "target_rank_to_city_interstitial_char_count": transition.get(
                    "target_rank_to_city_interstitial_char_count"
                ),
                "target_rank_to_city_interstitial_token_count": transition.get(
                    "target_rank_to_city_interstitial_token_count"
                ),
                "target_rank_to_city_has_lexical_content": transition.get(
                    "target_rank_to_city_has_lexical_content"
                ),
                "anchor_role": role,
                "anchor_roles": (
                    "|".join(anchor.get("anchor_roles", [])) if anchor else ""
                ),
                "anchor_equivalence_id": (
                    None if anchor is None else anchor.get("anchor_equivalence_id")
                ),
                "timing_stage": candidate.get("timing_stage"),
                "event_specific": candidate.get("event_specific"),
                "status": candidate.get("status"),
                "not_applicable_reason": candidate.get("not_applicable_reason"),
                "query_output_token": candidate.get("output_token_index"),
                "query_full_sequence_token": candidate.get(
                    "full_sequence_token_index"
                ),
                "query_token_id": candidate.get("token_id"),
                "query_token_text": candidate.get("token_text", ""),
                "target_city_output_token_start": transition.get(
                    "next_city_output_token_start"
                ),
                "target_city_output_token_end": transition.get(
                    "next_city_output_token_end"
                ),
                "target_city_token_ids": json.dumps(
                    transition.get("next_city_token_ids", [])
                ),
                "target_city_token_text": transition.get(
                    "next_city_token_text", ""
                ),
                "local_anchor_eligible": (
                    False if anchor is None else anchor.get("local_anchor_eligible")
                ),
                "primary_anchor_eligible": (
                    False if anchor is None else anchor.get("primary_anchor_eligible")
                ),
            }


def flatten_transition_rows(plan: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    base = {
        key: plan.get(key)
        for key in (
            "request_id",
            "model_label",
            "seed",
            "split",
            "gold_count",
            "sequence_source",
            "trace_category",
            "causal_cohort",
            "grammar_signature",
        )
    }
    base["allowed_estimands"] = "|".join(plan.get("allowed_estimands", []))
    for transition in [*plan.get("transitions", []), plan.get("terminal_transition")]:
        if not transition:
            continue
        is_terminal = transition["transition_kind"] == "stop_to_answer_query_v3"
        target_ids = (
            transition.get("answer_query_token_ids", [])
            if is_terminal
            else transition.get("next_city_token_ids", [])
        )
        continuation_ids = transition.get("full_continuation_token_ids", [])
        yield {
            **base,
            "transition_kind": transition["transition_kind"],
            "from_occurrence": transition["from_occurrence"],
            "to_occurrence": transition.get("to_occurrence"),
            "target_city": transition.get("target_city"),
            "grammar_pair": transition.get("grammar_pair"),
            "target_retrieval_surface_variant": transition.get(
                "target_retrieval_surface_variant"
            ),
            "target_rank_to_city_interstitial_char_count": transition.get(
                "target_rank_to_city_interstitial_char_count"
            ),
            "target_rank_to_city_interstitial_token_count": transition.get(
                "target_rank_to_city_interstitial_token_count"
            ),
            "resolved_anchor_count": len(transition.get("anchors", [])),
            "primary_anchor_count": sum(
                int(bool(value.get("primary_anchor_eligible")))
                for value in transition.get("anchors", [])
            ),
            "query_output_token": transition.get("query_output_token_index"),
            "continuation_output_token_start": transition.get(
                "full_continuation_output_token_start"
            ),
            "continuation_output_token_end": transition.get(
                "full_continuation_output_token_end"
            ),
            "continuation_token_count": len(
                continuation_ids
            ),
            "target_only_output_token_start": transition.get(
                "answer_query_output_token_start"
                if is_terminal
                else "next_city_output_token_start"
            ),
            "target_only_output_token_end": transition.get(
                "answer_query_output_token_end"
                if is_terminal
                else "next_city_output_token_end"
            ),
            "target_only_token_count": len(target_ids),
            "interstitial_token_count": max(
                0, len(continuation_ids) - len(target_ids)
            ),
            "next_city_output_token_start": transition.get(
                "next_city_output_token_start"
            ),
            "next_city_output_token_end": transition.get(
                "next_city_output_token_end"
            ),
            "continuation_token_text": transition.get(
                "full_continuation_token_text", ""
            ),
            "target_only_token_text": transition.get(
                "answer_query_token_text"
                if is_terminal
                else "next_city_token_text",
                "",
            ),
            "local_or_observed_stop_eligible": transition.get(
                "local_transition_eligible",
                transition.get("observed_stop_eligible", False),
            ),
            "primary_eligible": transition.get(
                "primary_transition_eligible",
                transition.get("primary_terminal_eligible", False),
            ),
            "exclusion_reasons": "|".join(transition["exclusion_reasons"]),
        }


def causal_site_rules() -> dict[str, Any]:
    """Machine-readable policy embedded in review manifests."""

    return {
        "schema_version": SCHEMA_VERSION,
        "token_alignment_version": TOKEN_ALIGNMENT_VERSION,
        "grammar_policy_version": GRAMMAR_POLICY_VERSION,
        "cohort_policy_version": COHORT_POLICY_VERSION,
        "semantic_roles": {
            "city_pre_d1": (
                "output token immediately before the first token overlapping "
                "the parser-registered city span"
            ),
            "p0_item_end": (
                "the committed endpoint of item k, used as the earliest query "
                "for target city k+1"
            ),
            "unit_pre_d1": (
                "output token immediately before the target city-bearing semantic "
                "unit, including its marker or opening delimiter"
            ),
            "post_open_delimiter": (
                "last token overlapping a separable opening bracket or invariant "
                "bullet; fused delimiter/content tokens are explicit N/A"
            ),
            "record_clause_pre_d1": (
                f"output token immediately before the literal clause {RECORD_CLAUSE!r} "
                "when it occurs inside the target event"
            ),
            "block_pre_d1": (
                "output token before the first compiled event; exploratory and not "
                "event-specific"
            ),
            "city_target_span": "all output tokens overlapping the city characters",
            "rank_evidence_core_span": (
                "semantic ordinal/count content after edge whitespace and removable "
                "edge punctuation normalization"
            ),
            "rank_evidence_surface_span": (
                "whitespace-trimmed parser evidence surface retained for visible-"
                "surface controls"
            ),
            "rank_visible_format_shell_tokens": (
                "tokenized evidence surface minus its nested semantic core; an "
                "explicit syntax/delimiter control, possibly empty"
            ),
            "pre_marker_state": (
                "output token immediately before the first token overlapping "
                "rank_evidence_core_span"
            ),
            "post_marker_state": (
                "last output token overlapping rank_evidence_core_span"
            ),
            "post_update_commit_state": (
                "last output token overlapping the selected event semantic span"
            ),
            "continue_to_next_city": (
                "tokens after current commit through the end of the next city; "
                "next-city-only tokens are retained separately"
            ),
            "stop_to_answer_query_v3": (
                "tokens after the final observed commit through answer_query_v3"
            ),
            "semantic_target_only": (
                "next-city tokens for continue, or answer_query_v3 tokens for stop; "
                "interstitial tokens are retained separately"
            ),
            "anchor_equivalence": (
                "candidate roles resolving to one output token are executed once and "
                "retain all semantic aliases"
            ),
        },
        "surface_grammar_axis": (
            "event association plus actual rank/city character order; trajectory "
            "marker_kind alone never selects a causal token"
        ),
        "semantic_span_normalization": (
            "strip leading/trailing Unicode whitespace from parser item, city-unit, "
            "and rank-evidence surface; strip edge punctuation from rank evidence "
            "only when an alphanumeric core remains; preserve parser/surface/core "
            "boundaries and trim counts; do not trim city or answer_query_v3"
        ),
        "cohort_axis": {
            "primary_rank_resolved_full_chain": (
                "rank-supported, one-to-one observed trace"
            ),
            "secondary_structural_marker_neutral": (
                "one-to-one structural fallback without a rank-supported episode"
            ),
            "secondary_structural_recap": (
                "structural sequence that extends an earlier rank episode; recap "
                "sites are not pooled with the primary running trace"
            ),
            "secondary_local_partial_unique": (
                "observed local transitions only; no correct terminal-count claim"
            ),
            "occurrence_retrieval_only_duplicates": (
                "occurrence retrieval only; emitted rank is not unique-count state"
            ),
            "secondary_evidence_sequence_exploratory": (
                "exact score-supported first-mention sequence; retrieval, continue, "
                "and terminal stop are exploratory, with no rank-marker claim"
            ),
            "secondary_evidence_sequence_partial_exploratory": (
                "partial score-supported first-mention sequence; local retrieval and "
                "continue only, with no terminal-count or rank-marker claim"
            ),
        },
        "cohort_allowed_estimands": {
            cohort: list(values)
            for cohort, values in COHORT_ALLOWED_ESTIMANDS.items()
        },
        "non_use_guarantee": (
            "gold N and final Total are not used to construct events, choose grammar, "
            "or place running causal sites"
        ),
    }
