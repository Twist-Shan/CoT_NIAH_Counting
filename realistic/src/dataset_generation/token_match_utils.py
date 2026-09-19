from __future__ import annotations

from typing import Any


def find_subsequence_positions(sequence: list[Any], pattern: list[Any]) -> list[tuple[int, int]]:
    """Return (start, end_exclusive) spans where pattern occurs in sequence."""
    if not pattern:
        return []
    spans: list[tuple[int, int]] = []
    m = len(pattern)
    for i in range(0, len(sequence) - m + 1):
        if sequence[i : i + m] == pattern:
            spans.append((i, i + m))
    return spans


def find_first_subsequence_position(sequence: list[Any], pattern: list[Any]) -> tuple[int, int] | None:
    spans = find_subsequence_positions(sequence, pattern)
    return spans[0] if spans else None
