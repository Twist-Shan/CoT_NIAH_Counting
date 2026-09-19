from __future__ import annotations

from dataclasses import dataclass

from .data import BaseExample
from .vocab import Vocab, number_token


@dataclass(frozen=True)
class RenderSpans:
    bos_pos: int
    seq_start: int
    seq_end_exclusive: int
    think_open_pos: int | None
    trace_token_positions: list[int]
    trace_index_positions: list[int]
    trace_marker_positions: list[int]
    think_close_pos: int | None
    ans_pos: int
    final_count_pos: int
    eos_pos: int


@dataclass(frozen=True)
class Rendered:
    tokens: list[int]
    token_strs: list[str]
    spans: RenderSpans
    prompt_needle_token_positions: list[int]


def trace_tokens_for_example(example: BaseExample) -> list[str]:
    tokens: list[str] = []
    for idx, marker in enumerate(example.needle_markers, start=1):
        tokens.extend([number_token(idx), marker])
    return tokens


def render_non_thinking(example: BaseExample, vocab: Vocab) -> Rendered:
    token_strs = ["<BOS>"] + example.seq_tokens + ["<Ans>", number_token(example.count), "<EOS>"]
    ans_pos = 1 + example.seq_len
    spans = RenderSpans(
        bos_pos=0,
        seq_start=1,
        seq_end_exclusive=1 + example.seq_len,
        think_open_pos=None,
        trace_token_positions=[],
        trace_index_positions=[],
        trace_marker_positions=[],
        think_close_pos=None,
        ans_pos=ans_pos,
        final_count_pos=ans_pos + 1,
        eos_pos=ans_pos + 2,
    )
    needle_token_positions = [spans.seq_start + pos for pos in example.needle_positions]
    return Rendered(vocab.encode(token_strs), token_strs, spans, needle_token_positions)


def render_thinking(example: BaseExample, vocab: Vocab) -> Rendered:
    trace = trace_tokens_for_example(example)
    token_strs = (
        ["<BOS>"]
        + example.seq_tokens
        + ["<Think/>"]
        + trace
        + ["</Think>", "<Ans>", number_token(example.count), "<EOS>"]
    )
    think_open_pos = 1 + example.seq_len
    trace_start = think_open_pos + 1
    trace_token_positions = list(range(trace_start, trace_start + len(trace)))
    think_close_pos = trace_start + len(trace)
    ans_pos = think_close_pos + 1
    spans = RenderSpans(
        bos_pos=0,
        seq_start=1,
        seq_end_exclusive=1 + example.seq_len,
        think_open_pos=think_open_pos,
        trace_token_positions=trace_token_positions,
        trace_index_positions=trace_token_positions[0::2],
        trace_marker_positions=trace_token_positions[1::2],
        think_close_pos=think_close_pos,
        ans_pos=ans_pos,
        final_count_pos=ans_pos + 1,
        eos_pos=ans_pos + 2,
    )
    needle_token_positions = [spans.seq_start + pos for pos in example.needle_positions]
    return Rendered(vocab.encode(token_strs), token_strs, spans, needle_token_positions)


def render_for_model(example: BaseExample, vocab: Vocab, model_type: str) -> Rendered:
    if model_type == "non_thinking":
        return render_non_thinking(example, vocab)
    if model_type == "thinking":
        return render_thinking(example, vocab)
    raise ValueError(f"Unknown model_type={model_type}")


def non_thinking_eval_prefix(example: BaseExample, vocab: Vocab) -> list[int]:
    return vocab.encode(["<BOS>"] + example.seq_tokens + ["<Ans>"])


def thinking_generation_prefix(example: BaseExample, vocab: Vocab) -> list[int]:
    return vocab.encode(["<BOS>"] + example.seq_tokens + ["<Think/>"])


def thinking_oracle_trace_prefix(example: BaseExample, vocab: Vocab) -> list[int]:
    return vocab.encode(
        ["<BOS>"]
        + example.seq_tokens
        + ["<Think/>"]
        + trace_tokens_for_example(example)
        + ["</Think>", "<Ans>"]
    )


def thinking_corrupted_trace_prefix(example: BaseExample, corrupted_trace: list[str], vocab: Vocab) -> list[int]:
    return vocab.encode(["<BOS>"] + example.seq_tokens + ["<Think/>"] + corrupted_trace + ["</Think>", "<Ans>"])
