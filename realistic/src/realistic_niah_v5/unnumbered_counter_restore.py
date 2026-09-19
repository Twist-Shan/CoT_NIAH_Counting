"""No-explicit-index traces and old-HTML-style internal-counter restoration."""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from typing import Any, Mapping, Sequence

import torch

from realistic_niah_v4.modeling import DecoderAdapter, capture_post_block_states

from .count_stream import (
    _prefill_with_layerwise_state_replacements,
    _score_and_generate_prefill,
    _sha256_json,
    build_answer_source_registry,
)
from .encoding import NativeTraceEncoding


UNNUMBERED_STYLE_INSTRUCTIONS = (
    "During reasoning, list each matching record as an unnumbered dash bullet. "
    "Never write a running item number, an ordinal word, or a phrase such as "
    "'records so far'. City names and their scores may be copied. Do not state "
    "the total anywhere before the final Total line.",
    "During reasoning, use exactly one unnumbered line '- <city>: <score>' for "
    "each matching record. Do not write Record 1, Match 1, first/second, a "
    "running subtotal, or the final count before the required Total line.",
    "Reason only with unnumbered dash bullets of the form '- <city>: <score>'. "
    "No prose between bullets; no item labels, ordinal words, running subtotal, "
    "or total before the final required Total line.",
)

UNNUMBERED_END_PRIORITY_INSTRUCTIONS = (
    "IMPORTANT REASONING FORMAT: Begin the reasoning channel immediately with "
    "an unnumbered dash bullet for the first matching record. The reasoning "
    "channel must contain only one '- <city>: <score>' line per match and no "
    "other prose. Never label a bullet with its position or state a subtotal. "
    "After the last bullet, close reasoning and give the required Total line.",
    "Your hidden reasoning must be only an unnumbered dash-bullet list of the "
    "matching city-score records. Start with '-' immediately: no plan, no "
    "preamble, no first/second labels, no Record 1, no running count, and no "
    "verification recap. Then give Total only in the final answer channel.",
    "FORMAT OVERRIDE: In the reasoning channel, copy every match exactly once "
    "using only lines shaped '- <city>: <score>'. Do not write anything before, "
    "between, or after those bullets in reasoning; in particular do not number "
    "them or say how many have been found. Put the count only in the final "
    "Total line after reasoning ends.",
    "STRICT CHANNEL CONTRACT: The thinking section must contain exactly one "
    "unnumbered dash bullet per matching city-score record and nothing else. "
    "Do not write Total, a subtotal, a count, or any prose inside thinking. "
    "Immediately after the final bullet, end the thinking section. In the final "
    "answer section, output exactly one line 'Total: <integer>' and nothing else. "
    "Never repeat a city, score, bullet, or explanation after thinking ends.",
    "Use two strictly separated response sections. THINKING: only lines shaped "
    "'- <city>: <score>', one per match, with no count or Total line. End thinking "
    "immediately after the last bullet. FINAL: exactly 'Total: <integer>' on one "
    "line. The final section is answer-only: repeating any evidence or bullet "
    "there makes the response invalid.",
    "QWEN OUTPUT RULE: Before the thinking section closes, emit only unnumbered "
    "dash bullets for the matching records; never emit Total or the number of "
    "matches there. After thinking closes, the entire remaining response must "
    "match the regular expression ^Total: [0-9]+$ . Do not restate, summarize, "
    "or copy any city-score record in the final answer.",
    "QWEN BOUNDARY RULE: Immediately after the final bullet's newline, emit the "
    "literal closing token </think>. The substring 'Total:' must not occur before "
    "that closing token and must occur exactly once in the entire response. After "
    "</think>, emit only 'Total: <integer>' and end the response; never repeat the "
    "bullets or any city-score evidence.",
    "Treat the response as a two-state machine. In the thinking state, the only "
    "legal lines are unnumbered '- <city>: <score>' bullets. On the final match, "
    "transition directly by writing </think>; do not prepare, calculate, state, "
    "or summarize the total before that transition. In the answer state, write "
    "exactly one line 'Total: <integer>' and then stop, with no repeated evidence.",
    "REQUIRED RESPONSE SHAPE (the assistant is already inside thinking):\n"
    "- <matching city>: <score>\n"
    "...\n"
    "- <last matching city>: <score>\n"
    "</think>\n\n"
    "Total: <integer>\n"
    "There is no Total line before </think>, and there are no bullets after it. "
    "Follow this channel boundary exactly and end after the integer.",
)

_QWEN_FINAL_TOTAL_ONLY_RE = re.compile(
    r"^\s*Total:\s*([0-9]+)\s*(?:<\|im_end\|>)?\s*$",
    re.IGNORECASE,
)

_ORDINAL_RE = re.compile(
    r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b",
    re.IGNORECASE,
)
_LABELED_INDEX_RE = re.compile(
    r"\b(?:record|match|excerpt|item|number|count)\s*#?\s*(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
    re.IGNORECASE,
)
_RUNNING_PHRASE_RE = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
    r"(?:records?|items?|matches?|needles?)\s+(?:so far|found|identified|total)\b",
    re.IGNORECASE,
)
_LEADING_INDEX_RE = re.compile(r"^\s*(?:[-*•]\s*)?\d+\s*[.)\]:-]", re.MULTILINE)
_TAIL_NUMBER_RE = re.compile(
    r"(?:\d+|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b)",
    re.IGNORECASE,
)

# This audit is deliberately narrower than ``audit_unnumbered_trace``.  A dash
# bullet is only a delimiter; it does not reveal the running count.  What would
# invalidate a state at occurrence k is a *causally earlier* record number,
# ordinal, running subtotal, or completed total.  Generic plan numbering such
# as ``1. Analyze the request`` is not a record counter and is therefore not
# matched here.
_EXPLICIT_RECORD_ENUM_RE = re.compile(
    r"\b(?:record|match|item|entry|city|found)\s*#?\s*"
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b"
    r"|\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+"
    r"(?:record|match|item|entry|city|excerpt)\b",
    re.IGNORECASE,
)
_EXPLICIT_PROGRESS_TOTAL_RE = re.compile(
    r"\b(?:count|total|subtotal)\s*(?:is|=|:|of)?\s*"
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b"
    r"|\b(?:there\s+(?:are|is)|that(?:'s|\s+is)|found)\s+"
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:records?|matches?|items?|entries|cities)\b"
    r"|\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:records?|matches?|items?|entries|cities)\s+(?:so\s+far|found|identified|total)\b",
    re.IGNORECASE,
)
_NUMBERED_EVIDENCE_LINE_RE = re.compile(
    r"^\s*\d+\s*[.)\]:-]\s*.*(?:received\s+a\s+score|city\s*[:=]|score\s*[:=])",
    re.IGNORECASE | re.MULTILINE,
)

_COUNT_WORD = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d{1,2})"
_INDEX_NOUN = r"(?:record|match|item|entry|city|excerpt|mention|instance)"
_PREFIX_LABELED_INDEX_RE = re.compile(
    rf"\b{_INDEX_NOUN}s?\s*#?\s*{_COUNT_WORD}\b"
    rf"|\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+"
    rf"{_INDEX_NOUN}\b"
    r"|\b(?:that(?:'|\u2019)s|that\s+is)\s+(?:a\s+|the\s+)?"
    r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b",
    re.IGNORECASE,
)
_PREFIX_RUNNING_PROGRESS_RE = re.compile(
    rf"\b(?:count|subtotal)\s*(?:is|was|=|:|of)?\s*{_COUNT_WORD}\b"
    rf"|\b(?:that(?:'|\u2019)s|that\s+is)\s+(?:a\s+|the\s+)?{_COUNT_WORD}"
    rf"(?:\s+(?:records?|matches?|items?|entries|cities))?(?:\s+so\s+far)?\b"
    rf"|\b(?:now|currently)\s+(?:at\s+)?{_COUNT_WORD}\b"
    rf"|\b{_COUNT_WORD}\s+(?:records?|matches?|items?|entries|cities)?\s*so\s+far\b",
    re.IGNORECASE,
)
_BARE_CARDINAL_SENTENCE_RE = re.compile(
    r"(?:^|[.!?]\s+)(?:one|two|three|four|five|six|seven|eight|nine|ten)\s*[.!?]",
    re.IGNORECASE,
)
_SMALL_NUMBER_VALUES = {
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
}


def _reasoning_bounds(row: Mapping[str, Any], raw: str) -> tuple[int, int]:
    parser = row.get("trace_parse", {}).get("parser", {})
    start = int(parser.get("reasoning_start_char", -1))
    end = int(parser.get("reasoning_end_char", -1))
    if 0 <= start <= end <= len(raw):
        return start, end
    think_start = raw.find("<think>")
    start = think_start + len("<think>") if think_start >= 0 else 0
    think_end = raw.find("</think>", start)
    end = think_end if think_end >= 0 else len(raw)
    return start, end


def _score_supported_gold_mentions(
    row: Mapping[str, Any],
) -> tuple[int, int, list[dict[str, Any]]]:
    """Locate every locally score-supported mention of a gold city."""

    raw = str(row.get("raw_output_text", ""))
    reasoning_start, reasoning_end = _reasoning_bounds(row, raw)
    reasoning = raw[reasoning_start:reasoning_end]
    registry: list[tuple[str, int | None]] = []
    for record in row.get("gold_records", ()):
        if not isinstance(record, Mapping) or not record.get("city"):
            continue
        score_value = record.get("score")
        registry.append(
            (
                str(record["city"]),
                int(score_value) if score_value is not None else None,
            )
        )
    if not registry:
        return reasoning_start, reasoning_end, []
    canonical = {city.casefold(): city for city, _score in registry}
    scores = {city.casefold(): score for city, score in registry}
    alternatives = "|".join(
        re.escape(city) for city in sorted(canonical.values(), key=len, reverse=True)
    )
    city_re = re.compile(
        rf"(?<!\w)(?P<city>{alternatives})(?!\w)", re.IGNORECASE
    )
    city_hits = list(city_re.finditer(reasoning))
    supported: list[dict[str, Any]] = []
    for index, match in enumerate(city_hits):
        city = canonical[match.group("city").casefold()]
        score = scores[city.casefold()]
        next_city_start = (
            city_hits[index + 1].start() if index + 1 < len(city_hits) else len(reasoning)
        )
        window_end = min(len(reasoning), match.end() + 96, next_city_start)
        window = reasoning[match.end():window_end]
        if score is None:
            score_match = re.search(
                r"\breceived\s+a\s+(?:numeric\s+)?score\b", window, re.IGNORECASE
            )
        else:
            score_match = re.search(rf"(?<!\d){score}(?!\d)", window)
        if score_match is None:
            continue
        end = reasoning_start + match.end() + score_match.end()
        while end < reasoning_end and raw[end] in " \t*`)]\"'":
            end += 1
        if end < reasoning_end and raw[end] in ".;":
            end += 1
        supported.append(
            {
                "city": city,
                "char_start": reasoning_start + match.start(),
                "char_end": end,
            }
        )
    return reasoning_start, reasoning_end, supported


def _first_score_supported_gold_mentions(
    row: Mapping[str, Any],
) -> tuple[int, int, list[dict[str, Any]]]:
    """Locate the first locally score-supported mention of every gold city."""

    reasoning_start, reasoning_end, supported = _score_supported_gold_mentions(row)
    first: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in sorted(supported, key=lambda value: int(value["char_start"])):
        folded = str(event["city"]).casefold()
        if folded in seen:
            continue
        seen.add(folded)
        first.append(event)
    return reasoning_start, reasoning_end, first


def _gold_city_leading_indices(
    text: str,
    *,
    cities: Sequence[str],
    max_index: int,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Find shorthand numbered evidence lines such as ``1. Paris - 61``."""

    if not cities:
        return []
    alternatives = "|".join(
        re.escape(city) for city in sorted(set(cities), key=len, reverse=True)
    )
    city_re = re.compile(
        rf"(?<!\w)(?P<city>{alternatives})(?!\w)", re.IGNORECASE
    )
    hits: list[dict[str, Any]] = []
    for line_match in re.finditer(r"[^\r\n]*", text):
        line = line_match.group(0)
        index_match = re.match(
            r"\s*(?P<value>\d{1,2})\s*[.)\]:-]\s*", line
        )
        if index_match is None:
            continue
        value = int(index_match.group("value"))
        if not 1 <= value <= int(max_index):
            continue
        city_match = city_re.search(line, index_match.end())
        if city_match is None:
            continue
        hits.append(
            {
                "city": city_match.group("city"),
                "value": value,
                "char_start": int(offset) + line_match.start() + index_match.start(),
                "char_end": int(offset) + line_match.start() + city_match.end(),
            }
        )
    return hits


def _city_parenthetical_indices(
    text: str,
    *,
    cities: Sequence[str],
    max_index: int,
    offset: int = 0,
) -> list[dict[str, Any]]:
    if not cities:
        return []
    alternatives = "|".join(
        re.escape(city) for city in sorted(set(cities), key=len, reverse=True)
    )
    city_re = re.compile(
        rf"(?<!\w)(?P<city>{alternatives})(?!\w)", re.IGNORECASE
    )
    hits: list[dict[str, Any]] = []
    for city_match in city_re.finditer(text):
        suffix = text[city_match.end(): min(len(text), city_match.end() + 16)]
        index_match = re.match(
            r"\s*\(\s*(?P<value>\d{1,2}|one|two|three|four|five|six|seven|"
            r"eight|nine|ten)\s*\)",
            suffix,
            re.IGNORECASE,
        )
        if index_match is None:
            continue
        surface = index_match.group("value").casefold()
        value = int(surface) if surface.isdigit() else _SMALL_NUMBER_VALUES[surface]
        if not 1 <= value <= int(max_index):
            continue
        hits.append(
            {
                "city": city_match.group("city"),
                "value": value,
                "char_start": offset + city_match.end() + index_match.start(),
                "char_end": offset + city_match.end() + index_match.end(),
            }
        )
    return hits


def explicit_count_cues(
    text: str,
    *,
    cities: Sequence[str] = (),
    max_index: int = 10,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Return explicit running-count/index cues under the prefix-clean grammar.

    This public helper lets post-evidence boundary scans apply exactly the same
    lexical exclusion family as the cohort audit.  ``offset`` maps local spans
    back to character positions in the archived assistant output.
    """

    value = str(text)
    cues: list[dict[str, Any]] = []
    for kind, pattern in (
        ("labeled_record_index", _PREFIX_LABELED_INDEX_RE),
        ("running_progress", _PREFIX_RUNNING_PROGRESS_RE),
        ("bare_cardinal_sentence", _BARE_CARDINAL_SENTENCE_RE),
        ("numbered_evidence_line", _NUMBERED_EVIDENCE_LINE_RE),
    ):
        for match in pattern.finditer(value):
            cues.append(
                {
                    "kind": kind,
                    "char_start": int(offset) + match.start(),
                    "char_end": int(offset) + match.end(),
                }
            )
    cues.extend(
        {**hit, "kind": "city_parenthetical_index"}
        for hit in _city_parenthetical_indices(
            value,
            cities=tuple(str(city) for city in cities),
            max_index=int(max_index),
            offset=int(offset),
        )
    )
    cues.extend(
        {**hit, "kind": "gold_city_leading_index"}
        for hit in _gold_city_leading_indices(
            value,
            cities=tuple(str(city) for city in cities),
            max_index=int(max_index),
            offset=int(offset),
        )
    )
    cues.sort(key=lambda cue: (int(cue["char_start"]), str(cue["kind"])))
    return cues


def audit_first_occurrence_prefix_clean(row: Mapping[str, Any]) -> dict[str, Any]:
    """Audit an early-stop cohort at the first complete evidence pass.

    ``t_star_char`` is the end of the first locally score-supported mention of
    the K-th unique gold record. Only text at or before that endpoint is
    causally available to the selected states. A secondary global audit rejects
    later per-record numbering while allowing a terminal aggregate total.
    """

    raw = str(row.get("raw_output_text", ""))
    gold_count = int(row.get("gold_count", 0))
    reasoning_start, reasoning_end, supported_events = _score_supported_gold_mentions(row)
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in sorted(supported_events, key=lambda value: int(value["char_start"])):
        folded = str(event["city"]).casefold()
        if folded in seen:
            continue
        seen.add(folded)
        events.append(event)
    coverage_complete = len(events) == gold_count and gold_count > 0
    reasons: list[str] = []
    if not coverage_complete:
        reasons.append("first_score_supported_gold_coverage_mismatch")
    t_star = int(events[-1]["char_end"]) if coverage_complete else None
    prefix_text = raw[reasoning_start:t_star] if t_star is not None else ""
    cities = [str(record.get("city", "")) for record in row.get("gold_records", ())]
    prefix_cues = explicit_count_cues(
        prefix_text,
        cities=cities,
        max_index=gold_count,
        offset=reasoning_start,
    )
    if prefix_cues:
        reasons.extend(
            f"explicit_{cue['kind']}_before_t_star" for cue in prefix_cues
        )
    repeated_evidence: list[dict[str, Any]] = []
    if t_star is not None:
        visible_seen: set[str] = set()
        for event in sorted(
            supported_events, key=lambda value: int(value["char_start"])
        ):
            if int(event["char_end"]) > t_star:
                continue
            folded = str(event["city"]).casefold()
            if folded in visible_seen:
                repeated_evidence.append(
                    {
                        "city": str(event["city"]),
                        "char_start": int(event["char_start"]),
                        "char_end": int(event["char_end"]),
                    }
                )
            else:
                visible_seen.add(folded)
    if repeated_evidence:
        reasons.append("repeated_gold_evidence_before_t_star")

    occurrence_audits: list[dict[str, Any]] = []
    for occurrence, event in enumerate(events, start=1):
        event_end = int(event["char_end"])
        visible_cues = [
            cue for cue in prefix_cues if int(cue["char_end"]) <= event_end
        ]
        occurrence_audits.append(
            {
                "occurrence": occurrence,
                "city": str(event["city"]),
                "char_start": int(event["char_start"]),
                "char_end": event_end,
                "eligible": not visible_cues,
                "causally_prior_cue_kinds": [str(cue["kind"]) for cue in visible_cues],
            }
        )

    reasoning_text = raw[reasoning_start:reasoning_end]
    global_reasons = list(reasons)
    parser_marker_kind = str(
        row.get("trace_parse", {}).get("parser", {}).get("marker_kind", "")
    )
    if parser_marker_kind in {"indexed", "ordinal", "inline_count"}:
        global_reasons.append(
            f"global_explicit_parser_marker_kind:{parser_marker_kind}"
        )
    if _PREFIX_LABELED_INDEX_RE.search(reasoning_text):
        global_reasons.append("global_labeled_record_index")
    if _NUMBERED_EVIDENCE_LINE_RE.search(reasoning_text):
        global_reasons.append("global_numbered_evidence_line")
    if _BARE_CARDINAL_SENTENCE_RE.search(reasoning_text):
        global_reasons.append("global_bare_cardinal_sentence")
    if _city_parenthetical_indices(
        reasoning_text,
        cities=cities,
        max_index=gold_count,
        offset=reasoning_start,
    ):
        global_reasons.append("global_city_parenthetical_index")

    return {
        "status": "PASS" if not reasons else "FAIL",
        "eligible": not reasons,
        "prefix_clean_eligible": not reasons,
        "global_clean_eligible": not global_reasons,
        "reasons": reasons,
        "global_reasons": list(dict.fromkeys(global_reasons)),
        "eligibility_definition": (
            "first complete score-supported gold-record pass is free of explicit "
            "running indices through t_star; later recap text is causally downstream"
        ),
        "global_sensitivity_definition": (
            "full reasoning contains no per-record labeled, ordinal, numbered-line, "
            "or city-parenthetical index; terminal aggregate total remains allowed"
        ),
        "selection_site_kind": "first_score_supported_gold_mentions",
        "parser_marker_kind": parser_marker_kind,
        "first_occurrence_item_count": len(events),
        "gold_count": gold_count,
        "coverage_complete": coverage_complete,
        "first_pass_complete": coverage_complete and not repeated_evidence,
        "pre_tstar_score_supported_event_count": sum(
            int(event["char_end"]) <= int(t_star)
            for event in supported_events
        )
        if t_star is not None
        else 0,
        "pre_tstar_repeated_gold_evidence": repeated_evidence,
        "reasoning_start_char": reasoning_start,
        "reasoning_end_char": reasoning_end,
        "t_star_char": t_star,
        "first_occurrences": occurrence_audits,
        "prefix_cues": prefix_cues,
        "future_text_after_t_star_not_a_primary_exclusion": True,
        "outcome_fields_accessed": False,
    }


def inject_unnumbered_instruction(user_text: str, *, attempt: int) -> str:
    """Insert a registered style instruction after the passage, not inside it."""

    index = int(attempt) - 1
    if 0 <= index < len(UNNUMBERED_STYLE_INSTRUCTIONS):
        marker = "</passage>"
        if str(user_text).count(marker) != 1:
            raise ValueError("V5 user prompt does not contain one passage terminator")
        return str(user_text).replace(
            marker,
            marker + "\n\n" + UNNUMBERED_STYLE_INSTRUCTIONS[index],
            1,
        )
    end_index = int(attempt) - 4
    if not 0 <= end_index < len(UNNUMBERED_END_PRIORITY_INSTRUCTIONS):
        raise ValueError("Unnumbered generation attempt is outside the registry")
    return str(user_text).rstrip() + "\n\n" + UNNUMBERED_END_PRIORITY_INSTRUCTIONS[end_index]


def audit_unnumbered_trace(row: Mapping[str, Any]) -> dict[str, Any]:
    """Verify that trace items expose no running ordinal/index label.

    Score values and the phrase ``2024 city score audit`` are allowed.  The
    exclusion concerns only an explicit *running index*; this distinction is
    recorded rather than ambiguously claiming that the trace contains no digit.
    """

    trace_parse = row.get("trace_parse", {})
    parser = trace_parse.get("parser", {})
    raw = str(row.get("raw_output_text", ""))
    starts = [int(value) for value in parser.get("item_start_chars", ())]
    ends = [int(value) for value in parser.get("item_end_chars", ())]
    item_count = int(parser.get("item_count", 0))
    gold_count = int(row.get("gold_count", 0))
    reasons: list[str] = []
    if not bool(parser.get("trace_one_to_one")):
        reasons.append("not_one_to_one")
    if str(parser.get("marker_kind")) != "bullet":
        reasons.append("not_unnumbered_bullet")
    if item_count != gold_count or len(starts) != item_count or len(ends) != item_count:
        reasons.append("item_count_mismatch")
    item_hashes: list[str] = []
    for occurrence, (start, end) in enumerate(zip(starts, ends), start=1):
        text = raw[start:end]
        item_hashes.append(hashlib.sha256(text.encode("utf-8")).hexdigest())
        if _LEADING_INDEX_RE.search(text):
            reasons.append(f"leading_index:{occurrence}")
        if _ORDINAL_RE.search(text):
            reasons.append(f"ordinal_word:{occurrence}")
        if _LABELED_INDEX_RE.search(text):
            reasons.append(f"labeled_index:{occurrence}")
        if _RUNNING_PHRASE_RE.search(text):
            reasons.append(f"running_phrase:{occurrence}")
    answer_sites = {
        str(site.get("site_kind")): site
        for site in trace_parse.get("char_sites", ())
        if str(site.get("site_kind", "")).startswith("answer_query")
    }
    answer_site = next(
        (
            answer_sites[kind]
            for kind in ("answer_query", "answer_query_v3", "answer_query_v2")
            if kind in answer_sites
        ),
        None,
    )
    if answer_site is None or not ends:
        reasons.append("answer_query_or_tail_unresolved")
        tail = ""
    else:
        tail = raw[ends[-1] : int(answer_site["char_start"])]
        if _TAIL_NUMBER_RE.search(tail):
            reasons.append("pre_answer_tail_count_leak")
    return {
        "status": "PASS" if not reasons else "FAIL",
        "eligible": not reasons,
        "reasons": reasons,
        "running_index_definition": (
            "no numbered item label, ordinal word, labeled index, running subtotal, "
            "or pre-answer total; city score digits remain allowed"
        ),
        "marker_kind": str(parser.get("marker_kind", "")),
        "item_count": item_count,
        "gold_count": gold_count,
        "item_text_sha256": item_hashes,
        "pre_answer_tail_sha256": hashlib.sha256(tail.encode("utf-8")).hexdigest(),
        "outcome_fields_accessed": False,
    }


def audit_qwen_thinking_bullets_final_total(row: Mapping[str, Any]) -> dict[str, Any]:
    """Audit a free Qwen completion for strict thinking/final channel separation.

    Eligibility is format-only: final-count correctness is reported but never
    used to choose a prompt variant.
    """

    base = audit_no_count_enumeration_trace(row)
    raw = str(row.get("raw_output_text", ""))
    reasons = list(base["reasons"])
    if raw.count("</think>") != 1:
        reasons.append("thinking_close_count")
        thinking = raw
        final = ""
    else:
        thinking, final = raw.split("</think>", 1)
    thinking_lines = [line.strip() for line in thinking.splitlines() if line.strip()]
    bullet_lines = [line for line in thinking_lines if line.startswith("- ")]
    if not thinking_lines or len(bullet_lines) != len(thinking_lines):
        reasons.append("thinking_not_bullets_only")
    if re.search(r"\bTotal\s*:", thinking, re.IGNORECASE):
        reasons.append("total_inside_thinking")
    final_match = _QWEN_FINAL_TOTAL_ONLY_RE.fullmatch(final)
    if final_match is None:
        reasons.append("final_not_total_only")
        final_total = None
    else:
        final_total = int(final_match.group(1))
    if re.search(r"(?m)^\s*-\s+", final):
        reasons.append("bullet_repeated_in_final")
    gold_cities = [str(value.get("city", "")) for value in row.get("gold_records", ())]
    if any(city and re.search(rf"\b{re.escape(city)}\b", final, re.IGNORECASE) for city in gold_cities):
        reasons.append("city_repeated_in_final")
    return {
        **base,
        "status": "PASS" if not reasons else "FAIL",
        "eligible": not reasons,
        "reasons": reasons,
        "thinking_bullet_line_count": len(bullet_lines),
        "thinking_nonempty_line_count": len(thinking_lines),
        "thinking_contains_total": bool(
            re.search(r"\bTotal\s*:", thinking, re.IGNORECASE)
        ),
        "final_total_only": final_match is not None,
        "final_total_value": final_total,
        "final_count_correct": (
            final_total == int(row.get("gold_count", -1))
            if final_total is not None
            else False
        ),
        "prompt_selection_uses_final_count_correctness": False,
    }


def audit_no_count_enumeration_trace(row: Mapping[str, Any]) -> dict[str, Any]:
    """Audit whether every registered item state is count-label-free when formed.

    The audit is causal-prefix based: text generated *after* item k cannot have
    influenced item-k hidden states and is not grounds for exclusion.  Plain
    bullets and unnumbered quoted audit sentences are allowed.  Explicit
    ``Record 1``, ``first record``, numbered evidence lines, running subtotals,
    and a total stated before the registered item span are excluded.
    """

    trace_parse = row.get("trace_parse", {})
    parser = trace_parse.get("parser", {})
    raw = str(row.get("raw_output_text", ""))
    starts = [int(value) for value in parser.get("item_start_chars", ())]
    ends = [int(value) for value in parser.get("item_end_chars", ())]
    item_count = int(parser.get("item_count", 0))
    gold_count = int(row.get("gold_count", 0))
    reasoning_start = int(parser.get("reasoning_start_char", 0))
    marker_kind = str(parser.get("marker_kind", ""))
    reasons: list[str] = []
    if not bool(parser.get("trace_one_to_one")):
        reasons.append("not_one_to_one")
    if item_count != gold_count or len(starts) != item_count or len(ends) != item_count:
        reasons.append("item_count_mismatch")
    if marker_kind in {"indexed", "inline_count", "ordinal"}:
        reasons.append(f"explicit_counter_marker_kind:{marker_kind}")
    if not 0 <= reasoning_start <= len(raw):
        reasons.append("reasoning_start_unresolved")
        reasoning_start = 0

    occurrence_audits: list[dict[str, Any]] = []
    previous_end = reasoning_start
    for occurrence, (start, end) in enumerate(zip(starts, ends), start=1):
        if not reasoning_start <= start < end <= len(raw) or start < previous_end:
            occurrence_reasons = ["invalid_or_nonmonotone_item_span"]
            prefix = ""
            item_text = ""
        else:
            prefix = raw[reasoning_start:end]
            item_text = raw[start:end]
            occurrence_reasons = []
            if _EXPLICIT_RECORD_ENUM_RE.search(prefix):
                occurrence_reasons.append("explicit_record_enumeration_in_causal_prefix")
            if _EXPLICIT_PROGRESS_TOTAL_RE.search(prefix):
                occurrence_reasons.append("explicit_progress_total_in_causal_prefix")
            if _NUMBERED_EVIDENCE_LINE_RE.search(prefix):
                occurrence_reasons.append("numbered_evidence_line_in_causal_prefix")
        previous_end = end
        if occurrence_reasons:
            reasons.extend(
                f"occurrence_{occurrence}:{value}" for value in occurrence_reasons
            )
        occurrence_audits.append(
            {
                "occurrence": occurrence,
                "eligible": not occurrence_reasons,
                "reasons": occurrence_reasons,
                "item_text_sha256": hashlib.sha256(
                    item_text.encode("utf-8")
                ).hexdigest(),
                "causal_prefix_sha256": hashlib.sha256(
                    prefix.encode("utf-8")
                ).hexdigest(),
            }
        )

    return {
        "status": "PASS" if not reasons else "FAIL",
        "eligible": not reasons,
        "reasons": reasons,
        "eligibility_definition": (
            "plain bullets are allowed; each item-k causal prefix must contain no "
            "record-number enumeration, ordinal record label, numbered evidence line, "
            "running subtotal, or already-stated total"
        ),
        "marker_kind": marker_kind,
        "item_count": item_count,
        "gold_count": gold_count,
        "occurrences": occurrence_audits,
        "future_text_after_each_item_not_a_causal_exclusion": True,
        "outcome_fields_accessed": False,
    }


def build_fully_uninformative_encoding(
    encoding: NativeTraceEncoding,
    registry: Any,
    tokenizer: Any,
    *,
    random_seed: int,
) -> tuple[NativeTraceEncoding, dict[str, Any]]:
    """Replace prompt needles and every trace item with same-length background."""

    original = tuple(int(value) for value in encoding.input_ids)
    spans = tuple(registry.prompt_records) + tuple(registry.trace_items)
    if not spans:
        raise ValueError("No prompt/trace needle spans are registered")
    forbidden = {position for start, end in spans for position in range(start, end)}
    special = {int(value) for value in getattr(tokenizer, "all_special_ids", ())}
    max_width = max(int(end) - int(start) for start, end in spans)
    windows: list[tuple[int, tuple[int, ...]]] = []
    for start in range(1, int(registry.prompt_token_count) - max_width + 1):
        positions = tuple(range(start, start + max_width))
        if set(positions) & forbidden:
            continue
        values = tuple(original[position] for position in positions)
        if any(value in special for value in values):
            continue
        windows.append((start, values))
    if not windows:
        raise ValueError("No ordinary prompt window can build the no-needle receiver")

    receiver = list(original)
    starts: list[int] = []
    changed = 0
    for span_index, (span_start, span_end) in enumerate(spans):
        width = int(span_end) - int(span_start)
        source = original[int(span_start) : int(span_end)]
        digest = hashlib.sha256(
            f"{encoding.request_id}|{random_seed}|{span_index}|{span_start}|{span_end}".encode()
        ).digest()
        initial = int.from_bytes(digest[:8], "big") % len(windows)
        chosen_start = -1
        replacement: tuple[int, ...] = ()
        for offset in range(len(windows)):
            candidate_start, candidate = windows[(initial + offset) % len(windows)]
            active = candidate[:width]
            if active != source:
                chosen_start = int(candidate_start)
                replacement = active
                break
        if chosen_start < 0:
            raise ValueError("Every ordinary window equals a needle span")
        receiver[int(span_start) : int(span_end)] = replacement
        starts.append(chosen_start)
        changed += sum(left != right for left, right in zip(source, replacement))
    control = replace(encoding, input_ids=tuple(receiver))
    if len(control.input_ids) != len(original) or changed <= 0:
        raise RuntimeError("No-needle receiver length/change audit failed")
    return control, {
        "control_construction": "same_length_prompt_background_windows",
        "prompt_and_trace_needles_replaced": True,
        "control_retokenized": False,
        "control_sequence_length_equal": True,
        "control_attention_mask_equal": tuple(control.attention_mask) == tuple(encoding.attention_mask),
        "replaced_span_count": len(spans),
        "replaced_prompt_record_count": len(registry.prompt_records),
        "replaced_trace_item_count": len(registry.trace_items),
        "changed_token_count": int(changed),
        "control_window_starts_sha256": _sha256_json(starts),
        "control_input_ids_sha256": _sha256_json(control.input_ids),
        "outcome_fields_accessed": False,
    }


def build_item_early_stop_encoding(
    encoding: NativeTraceEncoding,
    registry: Any,
    *,
    target_occurrence: int,
) -> tuple[NativeTraceEncoding, dict[str, Any]]:
    """Stop immediately after item k and splice only the terminal answer suffix.

    The suffix is copied from the clean transition between the final registered
    item and ``answer_query_v3``.  It contains the grammar needed to close the
    reasoning channel and reach ``Total:``, but no future trace item.  Because
    item-k positions precede the splice, their absolute token indices are
    unchanged.
    """

    occurrence = int(target_occurrence)
    items = tuple(registry.trace_items)
    if not 1 <= occurrence <= len(items):
        raise ValueError("Early-stop target occurrence is outside the trace")
    item_start, item_end = items[occurrence - 1]
    terminal_end = int(items[-1][1])
    query = int(registry.query_position)
    if not (
        int(encoding.prompt_token_count)
        <= int(item_start)
        < int(item_end)
        <= terminal_end
        <= query
        < int(encoding.sequence_length)
    ):
        raise ValueError("Early-stop item/terminal/query ordering is invalid")
    original_ids = tuple(int(value) for value in encoding.input_ids)
    original_mask = tuple(int(value) for value in encoding.attention_mask)
    suffix_start = terminal_end
    suffix_end = query + 1
    early_ids = original_ids[: int(item_end)] + original_ids[suffix_start:suffix_end]
    early_mask = original_mask[: int(item_end)] + original_mask[suffix_start:suffix_end]
    new_query = len(early_ids) - 1
    if new_query != int(item_end) + (suffix_end - suffix_start) - 1:
        raise RuntimeError("Early-stop answer query moved unexpectedly")
    if early_ids[: int(item_end)] != original_ids[: int(item_end)]:
        raise RuntimeError("Early-stop construction changed the causal item prefix")
    visible_items = tuple(encoding.trace_item_spans[:occurrence])
    result = replace(
        encoding,
        input_ids=early_ids,
        attention_mask=early_mask,
        query_position=new_query,
        trace_item_spans=visible_items,
        slot_spans=visible_items,
        needle_spans=visible_items,
    )
    return result, {
        "early_stop_target_occurrence": occurrence,
        "early_stop_item_start": int(item_start),
        "early_stop_item_end": int(item_end),
        "early_stop_original_terminal_end": terminal_end,
        "early_stop_original_query_position": query,
        "early_stop_query_position": new_query,
        "early_stop_suffix_token_count": suffix_end - suffix_start,
        "future_trace_token_count_removed": terminal_end - int(item_end),
        "future_trace_items_removed": len(items) - occurrence,
        "future_trace_tokens_present": False,
        "item_positions_unchanged": True,
        "terminal_suffix_contains_candidate_digit": False,
        "outcome_fields_accessed": False,
    }


@torch.inference_mode()
def run_unnumbered_counter_restore_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    source_layers: Sequence[int],
    target_occurrences: Sequence[int] = tuple(range(2, 10)),
    random_seed: int,
    answer_site_id: str = "answer_query_v3",
) -> list[dict[str, Any]]:
    """Restore occurrence-k clean states into a same-position no-needle receiver."""

    natural_panel = bool(row.get("natural_unnumbered_teacher_forced") is False)
    trace_audit = (
        audit_no_count_enumeration_trace(row)
        if natural_panel
        else audit_unnumbered_trace(row)
    )
    if not trace_audit["eligible"]:
        raise ValueError(f"Unnumbered trace audit failed: {trace_audit['reasons']}")
    clean, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    receiver, receiver_audit = build_fully_uninformative_encoding(
        clean, registry, tokenizer, random_seed=int(random_seed)
    )
    layers = tuple(sorted({int(value) for value in source_layers}))
    if not layers or layers[0] < 0 or layers[-1] >= int(adapter.num_layers) - 1:
        raise ValueError("Unnumbered restore source-layer registry is invalid")
    targets = tuple(int(value) for value in target_occurrences)
    if not targets or min(targets) < 1 or max(targets) > len(registry.trace_items):
        raise ValueError("Unnumbered restore target occurrences are invalid")
    patch_layers = tuple(range(min(layers), int(adapter.num_layers)))
    all_positions = tuple(registry.positions("trace_items"))
    _unused, clean_capture = capture_post_block_states(
        model, adapter, clean, all_positions, layers=patch_layers
    )
    position_index = {position: index for index, position in enumerate(all_positions)}

    baseline_prefill, _captures, _applications, _norms = (
        _prefill_with_layerwise_state_replacements(
            model,
            adapter,
            receiver,
            positions=tuple(range(*registry.trace_items[0])),
            replacements=None,
            readout_layers=(int(adapter.num_layers) - 1,),
            readout_positions=(int(registry.query_position),),
        )
    )
    baseline_outcomes = _score_and_generate_prefill(
        model, tokenizer, receiver, baseline_prefill, run_greedy=False, max_new_tokens=1
    )
    clean_prefill, _clean_readout, _clean_apps, _clean_norms = (
        _prefill_with_layerwise_state_replacements(
            model,
            adapter,
            clean,
            positions=tuple(range(*registry.trace_items[0])),
            replacements=None,
            readout_layers=(int(adapter.num_layers) - 1,),
            readout_positions=(int(registry.query_position),),
        )
    )
    clean_outcomes = _score_and_generate_prefill(
        model, tokenizer, clean, clean_prefill, run_greedy=False, max_new_tokens=1
    )

    common = {
        "schema_version": "realistic_niah_v5_unnumbered_counter_restore_v1",
        "experiment_id": "unnumbered_old_html_counter_restore",
        "request_id": str(clean.request_id),
        "model_label": str(clean.model_label),
        "seed": int(clean.seed),
        "dataset_split": str(clean.split),
        "gold_count": int(clean.count),
        "answer_site_id": answer_site_id,
        "target_occurrences": list(targets),
        "patch_geometry": "full_trace_item_same_position",
        "patch_layer_mode": "cumulative_clamp_source_through_last",
        "receiver_has_no_prompt_or_trace_needles": True,
        "trace_has_no_explicit_running_index": True,
        "score_digits_are_not_running_indices": True,
        "trace_panel_kind": (
            "model_generated_no_count_enumeration"
            if natural_panel
            else str(row.get("counterfactual_trace_kind", ""))
        ),
        "natural_generation_claim_allowed": natural_panel,
        "trace_tokens_teacher_forced": not natural_panel,
        "controlled_hidden_state_sufficiency_claim_allowed": True,
        "outcome_blind": True,
        "selection_rank_used": False,
        "causal_claim_scope": (
            "model_generated_no_count_enumeration_state_sufficiency_for_early_stop_count"
            if natural_panel
            else "controlled_teacher_forced_no_running_index_state_sufficiency_for_early_stop_count"
        ),
        "registry_sha256": registry.to_dict()["registry_sha256"],
        **trace_audit,
        **receiver_audit,
    }
    rows: list[dict[str, Any]] = [
        {
            **common,
            "condition": "clean_natural_reference",
            "source_layer": -1,
            "target_occurrence": 0,
            "patch_token_count": 0,
            **clean_outcomes,
        },
        {
            **common,
            "condition": "fully_uninformative",
            "source_layer": -1,
            "target_occurrence": 0,
            "patch_token_count": 0,
            **baseline_outcomes,
        },
    ]
    for source_layer in layers:
        active_layers = tuple(range(int(source_layer), int(adapter.num_layers)))
        for occurrence in targets:
            start, end = registry.trace_items[int(occurrence) - 1]
            positions = tuple(range(int(start), int(end)))
            indices = [position_index[position] for position in positions]
            replacements = {
                layer: clean_capture[layer][indices].clone() for layer in active_layers
            }
            prefill, _readout, applications, realized = (
                _prefill_with_layerwise_state_replacements(
                    model,
                    adapter,
                    receiver,
                    positions=positions,
                    replacements=replacements,
                    readout_layers=(int(adapter.num_layers) - 1,),
                    readout_positions=(int(registry.query_position),),
                )
            )
            outcomes = _score_and_generate_prefill(
                model, tokenizer, receiver, prefill, run_greedy=False, max_new_tokens=1
            )
            rows.append(
                {
                    **common,
                    "condition": "occurrence_state_restore",
                    "source_layer": int(source_layer),
                    "patch_layers": list(active_layers),
                    "target_occurrence": int(occurrence),
                    "patch_token_count": len(positions),
                    "receiver_positions_sha256": _sha256_json(positions),
                    "donor_positions_sha256": _sha256_json(positions),
                    "donor_receiver_positions_identical": True,
                    "donor_receiver_span_lengths_equal": True,
                    "patch_hook_applications": {
                        str(key): int(value) for key, value in applications.items()
                    },
                    "patch_realized_fro_norm_by_layer": {
                        str(key): float(value) for key, value in realized.items()
                    },
                    **outcomes,
                }
            )
    return rows


@torch.inference_mode()
def run_unnumbered_counter_early_stop_restore_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    source_layers: Sequence[int],
    target_occurrences: Sequence[int] = tuple(range(2, 10)),
    random_seed: int,
    answer_site_id: str = "answer_query_v3",
) -> list[dict[str, Any]]:
    """Restore item-k states, delete all future items, and score immediately."""

    natural_panel = bool(row.get("natural_unnumbered_teacher_forced") is False)
    trace_audit = (
        audit_no_count_enumeration_trace(row)
        if natural_panel
        else audit_unnumbered_trace(row)
    )
    if not trace_audit["eligible"]:
        raise ValueError(f"Unnumbered trace audit failed: {trace_audit['reasons']}")
    clean_full, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    receiver_full, receiver_audit = build_fully_uninformative_encoding(
        clean_full, registry, tokenizer, random_seed=int(random_seed)
    )
    layers = tuple(sorted({int(value) for value in source_layers}))
    if not layers or layers[0] < 0 or layers[-1] >= int(adapter.num_layers) - 1:
        raise ValueError("Early-stop restore source-layer registry is invalid")
    targets = tuple(int(value) for value in target_occurrences)
    if not targets or min(targets) < 1 or max(targets) > len(registry.trace_items):
        raise ValueError("Early-stop restore target occurrences are invalid")
    capture_layers = tuple(range(min(layers), int(adapter.num_layers)))
    rows: list[dict[str, Any]] = []
    for occurrence in targets:
        clean, early_audit = build_item_early_stop_encoding(
            clean_full, registry, target_occurrence=occurrence
        )
        receiver, receiver_early_audit = build_item_early_stop_encoding(
            receiver_full, registry, target_occurrence=occurrence
        )
        if clean.query_position != receiver.query_position or clean.input_ids[
            clean.query_position
        ] != receiver.input_ids[receiver.query_position]:
            raise RuntimeError("Clean/receiver early-stop answer queries are not aligned")
        start, end = registry.trace_items[int(occurrence) - 1]
        positions = tuple(range(int(start), int(end)))
        _unused, clean_capture = capture_post_block_states(
            model, adapter, clean, positions, layers=capture_layers
        )
        baseline_prefill, _baseline_readout, _baseline_apps, _baseline_norms = (
            _prefill_with_layerwise_state_replacements(
                model,
                adapter,
                receiver,
                positions=positions,
                replacements=None,
                readout_layers=(int(adapter.num_layers) - 1,),
                readout_positions=(int(receiver.query_position),),
            )
        )
        baseline_outcomes = _score_and_generate_prefill(
            model,
            tokenizer,
            receiver,
            baseline_prefill,
            run_greedy=False,
            max_new_tokens=1,
        )
        clean_prefill, _clean_readout, _clean_apps, _clean_norms = (
            _prefill_with_layerwise_state_replacements(
                model,
                adapter,
                clean,
                positions=positions,
                replacements=None,
                readout_layers=(int(adapter.num_layers) - 1,),
                readout_positions=(int(clean.query_position),),
            )
        )
        clean_outcomes = _score_and_generate_prefill(
            model,
            tokenizer,
            clean,
            clean_prefill,
            run_greedy=False,
            max_new_tokens=1,
        )
        common = {
            "schema_version": "realistic_niah_v5_unnumbered_counter_early_stop_restore_v1",
            "experiment_id": "unnumbered_old_html_counter_early_stop_restore",
            "request_id": str(clean.request_id),
            "model_label": str(clean.model_label),
            "seed": int(clean.seed),
            "dataset_split": str(clean.split),
            "gold_count": int(clean.count),
            "answer_site_id": answer_site_id,
            "target_occurrences": list(targets),
            "patch_geometry": "full_trace_item_same_position",
            "patch_layer_mode": "cumulative_clamp_source_through_last",
            "readout_mode": "immediate_item_k_early_stop_minimal_terminal_suffix",
            "receiver_has_no_prompt_or_trace_needles": True,
            "trace_has_no_explicit_running_index": True,
            "score_digits_are_not_running_indices": True,
            "trace_panel_kind": "model_generated_no_count_enumeration",
            "natural_generation_claim_allowed": natural_panel,
            "trace_tokens_teacher_forced": not natural_panel,
            "controlled_hidden_state_sufficiency_claim_allowed": True,
            "outcome_blind": True,
            "selection_rank_used": False,
            "registry_sha256": registry.to_dict()["registry_sha256"],
            **trace_audit,
            **receiver_audit,
            **early_audit,
        }
        if early_audit != receiver_early_audit:
            raise RuntimeError("Clean/receiver early-stop geometry audits disagree")
        rows.extend(
            [
                {
                    **common,
                    "condition": "clean_early_stop_reference",
                    "source_layer": -1,
                    "target_occurrence": int(occurrence),
                    "patch_token_count": 0,
                    **clean_outcomes,
                },
                {
                    **common,
                    "condition": "early_stop_uninformative",
                    "source_layer": -1,
                    "target_occurrence": int(occurrence),
                    "patch_token_count": 0,
                    **baseline_outcomes,
                },
            ]
        )
        for source_layer in layers:
            active_layers = tuple(range(int(source_layer), int(adapter.num_layers)))
            replacements = {
                layer: clean_capture[layer].clone() for layer in active_layers
            }
            prefill, _readout, applications, realized = (
                _prefill_with_layerwise_state_replacements(
                    model,
                    adapter,
                    receiver,
                    positions=positions,
                    replacements=replacements,
                    readout_layers=(int(adapter.num_layers) - 1,),
                    readout_positions=(int(receiver.query_position),),
                )
            )
            outcomes = _score_and_generate_prefill(
                model,
                tokenizer,
                receiver,
                prefill,
                run_greedy=False,
                max_new_tokens=1,
            )
            rows.append(
                {
                    **common,
                    "condition": "occurrence_state_restore_early_stop",
                    "source_layer": int(source_layer),
                    "patch_layers": list(active_layers),
                    "target_occurrence": int(occurrence),
                    "patch_token_count": len(positions),
                    "receiver_positions_sha256": _sha256_json(positions),
                    "donor_positions_sha256": _sha256_json(positions),
                    "donor_receiver_positions_identical": True,
                    "donor_receiver_span_lengths_equal": True,
                    "patch_hook_applications": {
                        str(key): int(value) for key, value in applications.items()
                    },
                    "patch_realized_fro_norm_by_layer": {
                        str(key): float(value) for key, value in realized.items()
                    },
                    **outcomes,
                }
            )
    return rows
