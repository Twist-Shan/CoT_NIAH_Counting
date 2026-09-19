"""Pure helpers for same-site progress-state transplant experiments.

The experiment creates equal-length marker factorial branches.  A branch with
``m`` valid markers has logical progress ``base_count + m`` while every branch
shares the same physical commit position and the same surface token there.
These helpers deliberately contain no model hooks so their cohort and outcome
logic can be unit tested independently of GPU inference.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


_BULLET_LINE_RE = re.compile(r"^\s*(?:[-*\u2022])\s+(.+?)\s*$")


def canonical_marker_bits(
    factor_count: int,
    valid_count: int,
    *,
    side: str = "right",
) -> tuple[int, ...]:
    """Return a deterministic factorial cell with ``valid_count`` active bits."""

    width = int(factor_count)
    active = int(valid_count)
    if not 1 <= width <= 4:
        raise ValueError("factor_count must lie in [1, 4]")
    if not 0 <= active <= width:
        raise ValueError("valid_count lies outside the factorial")
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    zeros = (0,) * (width - active)
    ones = (1,) * active
    return ones + zeros if side == "left" else zeros + ones


def select_count_cells(
    variants: Sequence[Mapping[str, Any]],
    *,
    factor_count: int,
    donor_valid_counts: Sequence[int],
) -> dict[int, dict[str, Mapping[str, Any] | None]]:
    """Select primary and marker-position-alternative cells for each count.

    The primary cell activates the rightmost markers.  When a different cell
    with the same number of valid markers exists, the alternative activates
    the leftmost markers.  Agreement between them is a count-invariance check,
    not a negative control.
    """

    by_bits: dict[tuple[int, ...], Mapping[str, Any]] = {}
    for variant in variants:
        bits = tuple(int(value) for value in variant["marker_bits"])
        if len(bits) != int(factor_count):
            raise ValueError("A factorial cell has the wrong bit width")
        if bits in by_bits:
            raise ValueError(f"Duplicate factorial cell {bits}")
        by_bits[bits] = variant
    expected = 2 ** int(factor_count)
    if len(by_bits) != expected:
        raise ValueError(
            f"Expected {expected} factorial cells, observed {len(by_bits)}"
        )

    selected: dict[int, dict[str, Mapping[str, Any] | None]] = {}
    for raw_count in donor_valid_counts:
        active = int(raw_count)
        if not 1 <= active <= int(factor_count):
            raise ValueError("Every donor must add at least one valid event")
        primary_bits = canonical_marker_bits(
            factor_count, active, side="right"
        )
        alternative_bits = canonical_marker_bits(
            factor_count, active, side="left"
        )
        selected[active] = {
            "primary": by_bits[primary_bits],
            "alternative": (
                None
                if alternative_bits == primary_bits
                else by_bits[alternative_bits]
            ),
        }
    return selected


def native_item_candidates(
    encoding: Any,
    trace_items: Sequence[Sequence[int]],
) -> dict[int, tuple[int, ...]]:
    """Return the native surface string of every trace item as token ids."""

    candidates: dict[int, tuple[int, ...]] = {}
    for occurrence, raw_span in enumerate(trace_items, start=1):
        start, end = (int(raw_span[0]), int(raw_span[1]))
        if not 0 <= start < end <= int(encoding.sequence_length):
            raise ValueError(f"Trace item {occurrence} has invalid geometry")
        candidates[occurrence] = tuple(
            int(value) for value in encoding.input_ids[start:end]
        )
    if not candidates or any(not value for value in candidates.values()):
        raise ValueError("Native item candidates may not be empty")
    if len(set(candidates.values())) != len(candidates):
        raise ValueError("Native item candidates are not token-unique")
    return candidates


def first_subsequence(
    haystack: Sequence[int], needle: Sequence[int]
) -> int | None:
    """Return the first exact token-subsequence offset, if present."""

    values = tuple(int(value) for value in haystack)
    target = tuple(int(value) for value in needle)
    if not target or len(target) > len(values):
        return None
    for start in range(len(values) - len(target) + 1):
        if values[start : start + len(target)] == target:
            return start
    return None


def query_prefix_before_city(
    candidate_tokens: Sequence[int], city_token_ids: Sequence[int]
) -> tuple[int, ...]:
    """Return the shared teacher-forced path ending just before a city name."""

    candidate = tuple(int(value) for value in candidate_tokens)
    city = tuple(int(value) for value in city_token_ids)
    offset = first_subsequence(candidate, city)
    if offset is None:
        raise ValueError("The successor city is absent from its native item tokens")
    if offset == 0:
        raise ValueError("The native item has no common query token before its city")
    return candidate[:offset]


def generated_bullet_city_ordinals(
    completion_text: str,
    ordered_cities: Sequence[str],
) -> dict[str, Any]:
    """Extract known-city ordinals from bullet lines before the reasoning close.

    Restricting to bullet lines avoids counting cities repeated in a later
    prose recap.  The raw completion remains the authoritative audit artifact;
    this parser is an intentionally narrow behavioral endpoint.
    """

    text = str(completion_text)
    close_candidates = [
        position
        for marker in ("</think>", "<|im_end|>", "<end_of_turn>")
        if (position := text.find(marker)) >= 0
    ]
    close_position = min(close_candidates) if close_candidates else None
    active = text if close_position is None else text[:close_position]
    cities = tuple(str(value) for value in ordered_cities)
    any_surface_hits: list[tuple[int, int]] = []
    for ordinal, city in enumerate(cities, start=1):
        for match in re.finditer(
            rf"(?<![A-Za-z0-9]){re.escape(city)}(?![A-Za-z0-9])",
            active,
            flags=re.IGNORECASE,
        ):
            any_surface_hits.append((int(match.start()), int(ordinal)))
    any_surface_hits.sort()
    any_surface_ordinals = [ordinal for _position, ordinal in any_surface_hits]
    ordinals: list[int] = []
    matched_lines: list[dict[str, Any]] = []
    ambiguous_lines: list[str] = []
    for line_number, line in enumerate(active.splitlines(), start=1):
        match = _BULLET_LINE_RE.match(line)
        if match is None:
            continue
        payload = match.group(1)
        hits = [
            index
            for index, city in enumerate(cities, start=1)
            if re.search(
                rf"(?<![A-Za-z0-9]){re.escape(city)}(?![A-Za-z0-9])",
                payload,
                flags=re.IGNORECASE,
            )
        ]
        if len(hits) == 1:
            ordinal = int(hits[0])
            ordinals.append(ordinal)
            matched_lines.append(
                {"line_number": line_number, "ordinal": ordinal, "line": line}
            )
        elif len(hits) > 1:
            ambiguous_lines.append(line)
    return {
        "generated_known_city_ordinals_any_surface": any_surface_ordinals,
        "generated_known_city_count_any_surface": len(any_surface_ordinals),
        "first_generated_known_city_ordinal": (
            None if not any_surface_ordinals else int(any_surface_ordinals[0])
        ),
        "last_generated_known_city_ordinal": (
            None if not any_surface_ordinals else int(any_surface_ordinals[-1])
        ),
        "generated_bullet_city_ordinals": ordinals,
        "generated_known_city_bullet_count": len(ordinals),
        "first_generated_bullet_city_ordinal": (
            None if not ordinals else int(ordinals[0])
        ),
        "last_generated_bullet_city_ordinal": (
            None if not ordinals else int(ordinals[-1])
        ),
        "generated_bullet_city_lines": matched_lines,
        "ambiguous_known_city_bullet_lines": ambiguous_lines,
        "reasoning_close_observed": close_position is not None,
        "reasoning_close_char_position": close_position,
    }


def donor_receiver_logodds(
    scores: Sequence[float],
    *,
    donor_successor: int,
    receiver_successor: int,
) -> float:
    """Return donor-successor minus receiver-successor candidate score."""

    values = tuple(float(value) for value in scores)
    donor_index = int(donor_successor) - 1
    receiver_index = int(receiver_successor) - 1
    if not 0 <= donor_index < len(values) or not 0 <= receiver_index < len(values):
        raise ValueError("A successor ordinal is outside the score vector")
    return float(values[donor_index] - values[receiver_index])
