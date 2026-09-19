from __future__ import annotations

"""Classify the visible counting device used in native-thinking traces.

The classes are deliberately based on observable text structure rather than
an interpretation of the model's latent strategy.  They are mutually
exclusive and are frozen before the V3 formal run so that the report cannot
redefine a favorable category after seeing the results.
"""

import re
from typing import Any


NATIVE_THINKING_STYLE_ORDER = (
    "indexed_list",
    "bullet_list",
    "mixed_structured_list",
    "ordinal_word_enumeration",
    "inline_tally_or_arithmetic",
    "prose_reasoning",
    "no_visible_reasoning",
)

# A line-leading ordinary number followed by a period or closing parenthesis
# is an indexed list marker.  Requiring whitespace after the marker avoids
# treating decimal numbers such as 3.14 as list items.
_INDEX_LINE_RE = re.compile(r"(?m)^\s*\d{1,4}[.)]\s+\S")
_BULLET_LINE_RE = re.compile(r"(?m)^\s*(?:[-*\u2022])\s+\S")

# These patterns capture a visible running tally without requiring a list.
# They are intentionally narrow; any non-empty trace that does not match a
# structured device is retained as prose rather than forced into this class.
_ARITHMETIC_RE = re.compile(
    r"(?<!\w)\d+\s*(?:\+|=|->|\u2192)\s*\d+(?!\w)",
    flags=re.IGNORECASE,
)
_TALLY_WORD_RE = re.compile(
    r"\b(?:running\s+(?:count|total)|total\s+so\s+far|"
    r"count\s+so\s+far|tally|increment(?:ing|ed)?\s+the\s+count)\b",
    flags=re.IGNORECASE,
)
_ORDINAL_WORD_RE = re.compile(
    r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|"
    r"tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|"
    r"seventeenth|eighteenth|nineteenth|twentieth)\b",
    flags=re.IGNORECASE,
)


def native_thinking_style_flags(reasoning_text: Any) -> dict[str, bool]:
    """Return the registered observable markers for one reasoning trace."""

    text = "" if reasoning_text is None else str(reasoning_text)
    visible = bool(text.strip())
    ordinal_terms = {
        match.group(0).lower() for match in _ORDINAL_WORD_RE.finditer(text)
    }
    return {
        "native_has_visible_reasoning": visible,
        "native_has_index_marker": bool(_INDEX_LINE_RE.search(text)),
        "native_has_bullet_marker": bool(_BULLET_LINE_RE.search(text)),
        "native_has_inline_tally": bool(
            _ARITHMETIC_RE.search(text) or _TALLY_WORD_RE.search(text)
        ),
        "native_has_ordinal_word_enumeration": len(ordinal_terms) >= 2,
    }


def classify_native_thinking_style(reasoning_text: Any) -> str:
    """Assign one preregistered, mutually exclusive counting-style class.

    Precedence is structural: a trace containing both numbered and bulleted
    list markers is ``mixed_structured_list``; otherwise a single structured
marker wins over an enumeration using two or more distinct ordinal words,
which wins over inline tally/arithmetic and then prose. Empty or
whitespace-only reasoning is ``no_visible_reasoning``.
    """

    flags = native_thinking_style_flags(reasoning_text)
    if not flags["native_has_visible_reasoning"]:
        return "no_visible_reasoning"
    if (
        flags["native_has_index_marker"]
        and flags["native_has_bullet_marker"]
    ):
        return "mixed_structured_list"
    if flags["native_has_index_marker"]:
        return "indexed_list"
    if flags["native_has_bullet_marker"]:
        return "bullet_list"
    if flags["native_has_ordinal_word_enumeration"]:
        return "ordinal_word_enumeration"
    if flags["native_has_inline_tally"]:
        return "inline_tally_or_arithmetic"
    return "prose_reasoning"
