from __future__ import annotations

import torch

from .render import RenderSpans


def build_training_weights(tokens: list[int], spans: RenderSpans, model_type: str) -> torch.Tensor:
    """Return one CE weight per position.

    With causal LM shifting, CE at position t predicts tokens[t + 1]. v3 uses
    exactly one fixed objective per model type; there is no loss-policy sweep.
    """

    weights = torch.zeros(len(tokens), dtype=torch.float32)
    if model_type == "non_thinking":
        weights[spans.ans_pos] = 1.0
        weights[spans.final_count_pos] = 1.0
        return weights
    if model_type == "thinking":
        if spans.think_open_pos is None or spans.think_close_pos is None:
            raise ValueError("Thinking spans are missing trace delimiters.")
        weights[spans.think_open_pos] = 1.0
        for pos in spans.trace_token_positions:
            weights[pos] = 1.0
        weights[spans.think_close_pos] = 1.0
        weights[spans.ans_pos] = 1.0
        weights[spans.final_count_pos] = 1.0
        return weights
    raise ValueError(f"Unknown model_type={model_type}")


def diagnostic_masks(spans: RenderSpans, model_type: str, length: int) -> dict[str, torch.Tensor]:
    masks = {
        "final_count": torch.zeros(length, dtype=torch.bool),
        "eos": torch.zeros(length, dtype=torch.bool),
        "trace": torch.zeros(length, dtype=torch.bool),
        "trace_index": torch.zeros(length, dtype=torch.bool),
        "trace_marker": torch.zeros(length, dtype=torch.bool),
        "think_close": torch.zeros(length, dtype=torch.bool),
        "ans_token": torch.zeros(length, dtype=torch.bool),
    }
    masks["final_count"][spans.ans_pos] = True
    masks["eos"][spans.final_count_pos] = True
    if model_type == "thinking":
        if spans.think_open_pos is not None:
            masks["trace"][spans.think_open_pos] = True
        for pos in spans.trace_token_positions:
            masks["trace"][pos] = True
        for pos in spans.trace_index_positions:
            masks["trace_index"][pos - 1 if pos > 0 else pos] = True
        for pos in spans.trace_marker_positions:
            masks["trace_marker"][pos - 1 if pos > 0 else pos] = True
        if spans.think_close_pos is not None:
            masks["think_close"][spans.think_close_pos - 1] = True
            masks["ans_token"][spans.think_close_pos] = True
    return masks
