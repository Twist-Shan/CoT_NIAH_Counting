from __future__ import annotations

import random
from dataclasses import dataclass

from .vocab import MARKER_TOKENS, NOISE_TOKENS, Vocab, count_token, index_token


IGNORE_INDEX = -100


@dataclass(frozen=True)
class BaseExample:
    seq_len: int
    count: int
    needle_positions: list[int]
    needle_markers: list[str]
    seq_tokens: list[str]
    seed: int | None = None


@dataclass(frozen=True)
class RenderSpans:
    bos_pos: int
    mode_pos: int
    seq_start: int
    seq_end_exclusive: int
    think_open_pos: int
    trace_token_positions: list[int]
    trace_index_positions: list[int]
    trace_marker_positions: list[int]
    think_close_pos: int
    pre_count_pos: int
    count_pos: int
    eos_pos: int


@dataclass(frozen=True)
class RenderedExample:
    variant: str
    token_strs: list[str]
    input_ids: list[int]
    labels: list[int]
    spans: RenderSpans
    prompt_needle_token_positions: list[int]
    gold_trace_markers: list[str]


def trace_prediction_queries(rendered: RenderedExample) -> list[dict[str, int | str]]:
    """Return the causal query position that predicts every trace marker."""
    marker_positions = list(rendered.spans.trace_marker_positions)
    index_positions = list(rendered.spans.trace_index_positions)
    if index_positions:
        if len(index_positions) != len(marker_positions):
            raise AssertionError("Indexed traces must pair every index with one marker.")
        query_positions = index_positions
        query_kind = "index_token_k_predicts_marker_k"
    else:
        query_positions = [rendered.spans.think_open_pos, *marker_positions[:-1]]
        query_kind = "previous_token_predicts_marker_k"
    if len(query_positions) != len(marker_positions):
        raise AssertionError("Each trace marker must have exactly one causal prediction query.")
    return [
        {
            "k": k + 1,
            "prediction_query_pos": query_pos,
            "target_marker_pos": marker_positions[k],
            "post_marker_query_pos": marker_positions[k],
            "query_kind": query_kind,
        }
        for k, query_pos in enumerate(query_positions)
    ]


def count_bin(count: int) -> str:
    if count <= 3:
        return "low"
    if count <= 6:
        return "mid"
    return "high"


def validate_example(example: BaseExample) -> None:
    if len(example.seq_tokens) != example.seq_len:
        raise AssertionError("Generated sequence length is wrong.")
    if example.count != len(example.needle_positions) or example.count != len(example.needle_markers):
        raise AssertionError("Count and needle metadata disagree.")
    if example.needle_positions != sorted(example.needle_positions):
        raise AssertionError("Needle positions must be sorted.")
    if len(set(example.needle_positions)) != len(example.needle_positions):
        raise AssertionError("Needle positions must be unique.")
    for pos, marker in zip(example.needle_positions, example.needle_markers):
        if example.seq_tokens[pos] != marker:
            raise AssertionError("Needle metadata does not match sequence token.")
    for token in example.seq_tokens:
        if token not in NOISE_TOKENS and token not in MARKER_TOKENS:
            raise AssertionError(f"Unexpected prompt token: {token}")


def make_example(
    seq_len: int,
    count: int,
    rng: random.Random,
    seed: int | None = None,
) -> BaseExample:
    if not 1 <= int(count) <= 10:
        raise ValueError(f"Count must be in 1..10, got {count}.")
    positions = sorted(rng.sample(range(int(seq_len)), int(count)))
    markers = [rng.choice(MARKER_TOKENS) for _ in positions]
    seq_tokens = [rng.choice(NOISE_TOKENS) for _ in range(int(seq_len))]
    for pos, marker in zip(positions, markers):
        seq_tokens[pos] = marker
    example = BaseExample(int(seq_len), int(count), positions, markers, seq_tokens, seed)
    validate_example(example)
    return example


def sample_example(
    seq_len: int,
    rng: random.Random,
    count_min: int = 1,
    count_max: int = 10,
    seed: int | None = None,
) -> BaseExample:
    return make_example(seq_len, rng.randint(int(count_min), int(count_max)), rng, seed=seed)


def balanced_examples(
    seq_len: int,
    examples_per_count: int,
    seed: int,
    count_min: int = 1,
    count_max: int = 10,
) -> list[BaseExample]:
    rng = random.Random(seed)
    examples: list[BaseExample] = []
    for count in range(int(count_min), int(count_max) + 1):
        for idx in range(int(examples_per_count)):
            ex_seed = seed * 1_000_000 + seq_len * 10_000 + count * 100 + idx
            examples.append(make_example(seq_len, count, rng, seed=ex_seed))
    rng.shuffle(examples)
    return examples


def trace_tokens_for_example(example: BaseExample, trace_indices: bool = False) -> list[str]:
    tokens: list[str] = []
    for idx, marker in enumerate(example.needle_markers, start=1):
        if trace_indices:
            tokens.append(index_token(idx))
        tokens.append(marker)
    return tokens


def _build_labels(
    token_strs: list[str],
    supervised_positions: list[int],
    vocab: Vocab,
) -> list[int]:
    labels = [IGNORE_INDEX for _ in token_strs]
    for pos in supervised_positions:
        labels[pos] = vocab.token_to_id[token_strs[pos]]
    return labels


def render_thinking(example: BaseExample, vocab: Vocab, trace_indices: bool = False) -> RenderedExample:
    trace = trace_tokens_for_example(example, trace_indices=trace_indices)
    token_strs = ["<BOS>", "<THINK_ON>"] + example.seq_tokens + ["<Think/>"] + trace + ["</Think>", count_token(example.count), "<EOS>"]
    think_open_pos = 2 + example.seq_len
    trace_start = think_open_pos + 1
    trace_positions = list(range(trace_start, trace_start + len(trace)))
    if trace_indices:
        trace_index_positions = trace_positions[0::2]
        trace_marker_positions = trace_positions[1::2]
    else:
        trace_index_positions = []
        trace_marker_positions = trace_positions
    think_close_pos = trace_start + len(trace)
    count_pos = think_close_pos + 1
    eos_pos = count_pos + 1
    spans = RenderSpans(
        bos_pos=0,
        mode_pos=1,
        seq_start=2,
        seq_end_exclusive=2 + example.seq_len,
        think_open_pos=think_open_pos,
        trace_token_positions=trace_positions,
        trace_index_positions=trace_index_positions,
        trace_marker_positions=trace_marker_positions,
        think_close_pos=think_close_pos,
        pre_count_pos=think_close_pos,
        count_pos=count_pos,
        eos_pos=eos_pos,
    )
    supervised = trace_positions + [think_close_pos, count_pos, eos_pos]
    needle_positions = [spans.seq_start + pos for pos in example.needle_positions]
    return RenderedExample(
        "thinking",
        token_strs,
        vocab.encode(token_strs),
        _build_labels(token_strs, supervised, vocab),
        spans,
        needle_positions,
        list(example.needle_markers),
    )


def render_nonthinking(
    example: BaseExample,
    vocab: Vocab,
) -> RenderedExample:
    token_strs = ["<BOS>", "<THINK_OFF>"] + example.seq_tokens + ["<Think/>", "</Think>", count_token(example.count), "<EOS>"]
    think_open_pos = 2 + example.seq_len
    think_close_pos = think_open_pos + 1
    count_pos = think_close_pos + 1
    eos_pos = count_pos + 1
    spans = RenderSpans(
        bos_pos=0,
        mode_pos=1,
        seq_start=2,
        seq_end_exclusive=2 + example.seq_len,
        think_open_pos=think_open_pos,
        trace_token_positions=[],
        trace_index_positions=[],
        trace_marker_positions=[],
        think_close_pos=think_close_pos,
        pre_count_pos=think_close_pos,
        count_pos=count_pos,
        eos_pos=eos_pos,
    )
    supervised = [think_close_pos, count_pos, eos_pos]
    needle_positions = [spans.seq_start + pos for pos in example.needle_positions]
    return RenderedExample(
        "nonthinking",
        token_strs,
        vocab.encode(token_strs),
        _build_labels(token_strs, supervised, vocab),
        spans,
        needle_positions,
        list(example.needle_markers),
    )


def render_example(
    example: BaseExample,
    variant: str,
    vocab: Vocab,
    trace_indices: bool = False,
) -> RenderedExample:
    if variant == "thinking":
        return render_thinking(example, vocab, trace_indices=trace_indices)
    if variant in {"nonthinking", "non_thinking"}:
        return render_nonthinking(example, vocab)
    raise ValueError(f"Unknown variant={variant}")


def thinking_query(example: BaseExample, vocab: Vocab) -> list[int]:
    return vocab.encode(["<BOS>", "<THINK_ON>"] + example.seq_tokens + ["<Think/>"])


def nonthinking_query(example: BaseExample, vocab: Vocab) -> list[int]:
    return vocab.encode(["<BOS>", "<THINK_OFF>"] + example.seq_tokens + ["<Think/>"])


def thinking_oracle_prefix(example: BaseExample, vocab: Vocab, trace_indices: bool = False) -> list[int]:
    return vocab.encode(
        ["<BOS>", "<THINK_ON>"]
        + example.seq_tokens
        + ["<Think/>"]
        + trace_tokens_for_example(example, trace_indices)
        + ["</Think>"]
    )
