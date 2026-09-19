from __future__ import annotations

from dataclasses import dataclass

from .vocab import MARKER_TOKENS, NUMBER_TOKENS


@dataclass(frozen=True)
class ParsedGeneration:
    trace_tokens: list[str]
    final_count: int | None
    invalid: bool
    has_think_close: bool
    has_ans: bool
    reason: str


def token_to_number(token: str) -> int | None:
    if token in NUMBER_TOKENS:
        return int(token.strip("<>"))
    return None


def parse_thinking_generation(generated_tokens: list[str]) -> ParsedGeneration:
    if "</Think>" not in generated_tokens:
        return ParsedGeneration(generated_tokens, None, True, False, False, "missing_think_close")
    close_idx = generated_tokens.index("</Think>")
    trace_tokens = generated_tokens[:close_idx]
    rest = generated_tokens[close_idx + 1 :]
    if "<Ans>" not in rest:
        return ParsedGeneration(trace_tokens, None, True, True, False, "missing_ans")
    ans_idx = rest.index("<Ans>")
    suffix = rest[ans_idx + 1 :]
    final_count = None
    for token in suffix:
        value = token_to_number(token)
        if value is not None:
            final_count = value
            break
    if final_count is None:
        return ParsedGeneration(trace_tokens, None, True, True, True, "missing_numeric_answer")
    return ParsedGeneration(trace_tokens, final_count, False, True, True, "ok")


def expected_trace_tokens(markers: list[str]) -> list[str]:
    tokens: list[str] = []
    for idx, marker in enumerate(markers, start=1):
        tokens.extend([f"<{idx}>", marker])
    return tokens


def trace_metrics(trace_tokens: list[str], gold_markers: list[str]) -> dict[str, float]:
    expected = expected_trace_tokens(gold_markers)
    generated_markers = [token for token in trace_tokens if token in MARKER_TOKENS]
    generated_indices = [token for token in trace_tokens if token in NUMBER_TOKENS]
    marker_overlap = sum((generated_markers.count(marker) > 0) for marker in gold_markers)
    marker_recall = marker_overlap / max(1, len(gold_markers))
    marker_precision = marker_overlap / max(1, len(generated_markers))
    expected_indices = [f"<{idx}>" for idx in range(1, len(gold_markers) + 1)]
    index_hits = sum(1 for got, exp in zip(generated_indices, expected_indices) if got == exp)
    index_accuracy = index_hits / max(1, len(expected_indices))
    duplicate_rate = max(0, len(generated_markers) - len(set(generated_markers))) / max(1, len(generated_markers))
    missing_items = max(0, len(gold_markers) - len(generated_markers)) / max(1, len(gold_markers))
    extra_items = max(0, len(generated_markers) - len(gold_markers)) / max(1, len(gold_markers))
    return {
        "trace_exact_rate": float(trace_tokens == expected),
        "trace_marker_recall": float(marker_recall),
        "trace_marker_precision": float(marker_precision),
        "trace_index_accuracy": float(index_accuracy),
        "duplicate_marker_position_rate": float(duplicate_rate),
        "missing_trace_item_rate": float(missing_items),
        "extra_trace_item_rate": float(extra_items),
    }
