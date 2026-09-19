from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .data import BaseExample
from .render import trace_tokens_for_example
from .vocab import MARKER_TOKENS, NUMBER_TOKENS, Vocab


@dataclass(frozen=True)
class ParsedGeneration:
    pred_count: int | None
    has_ans: bool
    has_close: bool
    trace_tokens: list[str]
    trace_markers: list[str]
    sep_count: int
    premature_close: bool
    missing_close: bool


def pad_sequences(sequences: list[list[int]], pad_id: int, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) for seq in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long, device=device)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long, device=device)
    for row_idx, seq in enumerate(sequences):
        input_ids[row_idx, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return input_ids, lengths


@torch.no_grad()
def greedy_generate(
    model,
    prefixes: list[list[int]],
    vocab: Vocab,
    device: str | torch.device,
    max_new_tokens: int,
    batch_size: int = 64,
) -> list[list[int]]:
    model.eval()
    outputs: list[list[int]] = []
    for start in range(0, len(prefixes), batch_size):
        chunk = prefixes[start : start + batch_size]
        input_ids, lengths = pad_sequences(chunk, vocab.pad_id, device)
        generated = [[] for _ in chunk]
        active = torch.ones(len(chunk), dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            attention_mask = (input_ids != vocab.pad_id).long()
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = out.logits[torch.arange(input_ids.size(0), device=device), lengths - 1]
            next_ids = logits.argmax(dim=-1)
            for row_idx, token_id in enumerate(next_ids.detach().cpu().tolist()):
                if active[row_idx]:
                    generated[row_idx].append(int(token_id))
            active = active & (next_ids != vocab.eos_id)
            input_ids = torch.cat([input_ids, next_ids[:, None]], dim=1)
            lengths = lengths + 1
            if not bool(active.any()):
                break
        outputs.extend(generated)
    return outputs


def _lcs_len(a: list[str], b: list[str]) -> int:
    dp = [0] * (len(b) + 1)
    for token_a in a:
        prev = 0
        for j, token_b in enumerate(b, start=1):
            old = dp[j]
            if token_a == token_b:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = old
    return dp[-1]


def parse_generated_suffix(tokens: list[str], example: BaseExample) -> ParsedGeneration:
    close_idx = tokens.index("</Think>") if "</Think>" in tokens else None
    ans_idx = tokens.index("<Ans>") if "<Ans>" in tokens else None
    trace_end = close_idx if close_idx is not None else (ans_idx if ans_idx is not None else len(tokens))
    trace_tokens = tokens[:trace_end]
    trace_markers = [token for token in trace_tokens if token in MARKER_TOKENS]
    sep_count = sum(token == "<Sep>" for token in trace_tokens)
    pred_count = None
    if ans_idx is not None and ans_idx + 1 < len(tokens):
        token = tokens[ans_idx + 1]
        if token in NUMBER_TOKENS:
            pred_count = int(token[1:-1])
    return ParsedGeneration(
        pred_count=pred_count,
        has_ans=ans_idx is not None,
        has_close=close_idx is not None,
        trace_tokens=trace_tokens,
        trace_markers=trace_markers,
        sep_count=sep_count,
        premature_close=bool(close_idx is not None and len(trace_markers) < example.count),
        missing_close=close_idx is None,
    )


def trace_metrics(parsed: ParsedGeneration, example: BaseExample) -> dict[str, float]:
    expected_trace = trace_tokens_for_example(example)
    lcs = _lcs_len(parsed.trace_markers, example.needle_markers)
    return {
        "trace_exact_match_rate": float(parsed.trace_tokens == expected_trace),
        "trace_marker_precision": float(lcs / max(1, len(parsed.trace_markers))),
        "trace_marker_recall": float(lcs / max(1, len(example.needle_markers))),
        "trace_delimiter_count_accuracy": float(parsed.sep_count == example.count),
        "premature_close_rate": float(parsed.premature_close),
        "missing_close_rate": float(parsed.missing_close),
        "ans_generated_rate": float(parsed.has_ans),
        "think_close_generated_rate": float(parsed.has_close),
    }


def generation_row(tokens: list[str], example: BaseExample) -> dict[str, Any]:
    parsed = parse_generated_suffix(tokens, example)
    pred = parsed.pred_count
    return {
        "pred_count": pred if pred is not None else -1,
        "accuracy": float(pred == example.count),
        "mae": abs(pred - example.count) if pred is not None else float("nan"),
        "under_rate": float(pred is not None and pred < example.count),
        "over_rate": float(pred is not None and pred > example.count),
        "invalid_rate": float(pred is None),
        **trace_metrics(parsed, example),
    }

