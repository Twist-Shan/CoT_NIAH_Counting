from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from .config import ReferenceConfig, RunSpec


# v19 keeps the v18 prompt vocabulary but renders every integer with shared
# decimal digit tokens. INDEX and COUNT identify the number's semantic role.
NOISE_OFFSET = 0
NOISE_VOCAB_SIZE = 256
MARKER_OFFSET = 256
MARKER_VOCAB_SIZE = 10
ANSWER = 266
START = 267
END = 268
PAD = 269
INDEX = 270
COUNT = 271
NUM_END = 272
DIGIT_OFFSET = 273
DIGIT_VOCAB_SIZE = 10
MAXIMUM_COUNT = 128
IGNORE_INDEX = -100


@dataclass(frozen=True)
class ReferenceExample:
    context: tuple[int, ...]
    count: int
    needle_positions: tuple[int, ...]
    needle_markers: tuple[int, ...]


@dataclass(frozen=True)
class RenderedExample:
    tokens: tuple[int, ...]
    labels: tuple[int, ...]
    prefix_length: int
    count: int


@dataclass(frozen=True)
class DigitTraceLayout:
    index_role_positions: tuple[int, ...]
    index_digit_positions: tuple[tuple[int, ...], ...]
    index_query_positions: tuple[int, ...]
    marker_positions: tuple[int, ...]
    end_position: int
    count_role_position: int
    count_digit_positions: tuple[int, ...]
    num_end_position: int


def digit_token(digit: int) -> int:
    if not 0 <= int(digit) <= 9:
        raise ValueError(f"digit must be in 0..9: {digit}")
    return DIGIT_OFFSET + int(digit)


def digit_from_token(token: int) -> int | None:
    value = int(token) - DIGIT_OFFSET
    return value if 0 <= value <= 9 else None


def encode_number(value: int, maximum: int = MAXIMUM_COUNT) -> tuple[int, ...]:
    if not 1 <= int(value) <= int(maximum):
        raise ValueError(f"number must be in 1..{maximum}: {value}")
    return tuple(digit_token(int(char)) for char in str(int(value)))


def decode_number(tokens: list[int] | tuple[int, ...], maximum: int = MAXIMUM_COUNT) -> int | None:
    digits = [digit_from_token(token) for token in tokens]
    if not digits or any(digit is None for digit in digits):
        return None
    values = [int(digit) for digit in digits if digit is not None]
    if len(values) > 1 and values[0] == 0:
        return None
    value = int("".join(str(digit) for digit in values))
    return value if 1 <= value <= int(maximum) else None


def number_width(maximum: int) -> int:
    return len(str(int(maximum)))


def digit_trace_layout(context_length: int, count: int) -> DigitTraceLayout:
    """Return exact v19 positions for a gold CoT trace.

    `context_length` excludes START. Every k-to-k query is the final digit of
    the rendered index because that position predicts the following marker.
    """

    cursor = int(context_length) + 1
    roles: list[int] = []
    digit_groups: list[tuple[int, ...]] = []
    queries: list[int] = []
    markers: list[int] = []
    for index in range(1, int(count) + 1):
        roles.append(cursor)
        cursor += 1
        digits = tuple(range(cursor, cursor + len(encode_number(index))))
        digit_groups.append(digits)
        queries.append(digits[-1])
        cursor += len(digits)
        markers.append(cursor)
        cursor += 1
    end_position = cursor
    count_role_position = end_position + 1
    count_digit_positions = tuple(
        range(count_role_position + 1, count_role_position + 1 + len(encode_number(count)))
    )
    num_end_position = count_digit_positions[-1] + 1
    return DigitTraceLayout(
        tuple(roles),
        tuple(digit_groups),
        tuple(queries),
        tuple(markers),
        end_position,
        count_role_position,
        count_digit_positions,
        num_end_position,
    )


def is_marker_token(token: int) -> bool:
    return MARKER_OFFSET <= int(token) < MARKER_OFFSET + MARKER_VOCAB_SIZE


def count_probabilities(spec: RunSpec) -> np.ndarray:
    counts = np.arange(1, spec.train_max_count + 1, dtype=np.float64)
    if spec.distribution == "uniform":
        weights = np.ones_like(counts)
    else:
        weights = counts ** (-float(spec.alpha))
    return weights / weights.sum()


def sample_count(spec: RunSpec, rng: random.Random) -> int:
    draw = rng.random()
    cumulative = 0.0
    for count, probability in zip(range(1, spec.train_max_count + 1), count_probabilities(spec)):
        cumulative += float(probability)
        if draw <= cumulative:
            return count
    return spec.train_max_count


def sample_example(
    cfg: ReferenceConfig,
    spec: RunSpec,
    rng: random.Random,
    *,
    count: int | None = None,
) -> ReferenceExample:
    n = sample_count(spec, rng) if count is None else int(count)
    if not 1 <= n <= min(spec.context_length, cfg.maximum_count):
        raise ValueError(f"count={n} is invalid for L={spec.context_length}")
    context = [rng.randrange(NOISE_OFFSET, NOISE_OFFSET + cfg.noise_vocab_size) for _ in range(spec.context_length)]
    positions = sorted(rng.sample(range(spec.context_length), n))
    markers = [MARKER_OFFSET + rng.randrange(cfg.needle_vocab_size) for _ in positions]
    for position, marker in zip(positions, markers):
        context[position] = marker
    return ReferenceExample(tuple(context), n, tuple(positions), tuple(markers))


def render(example: ReferenceExample, mode: str) -> RenderedExample:
    if len(example.needle_markers) != example.count:
        raise ValueError("needle_markers must contain one identity per needle")
    prefix = list(example.context)
    if mode == "direct":
        prefix.append(ANSWER)
        completion = [COUNT, *encode_number(example.count), NUM_END]
    elif mode == "cot":
        prefix.append(START)
        completion: list[int] = []
        for index, marker in enumerate(example.needle_markers, 1):
            completion.extend((INDEX, *encode_number(index), marker))
        completion.extend((END, COUNT, *encode_number(example.count), NUM_END))
    else:
        raise ValueError(mode)
    tokens = tuple(prefix + completion)
    labels = tuple([IGNORE_INDEX] * len(prefix) + completion)
    return RenderedExample(tokens, labels, len(prefix), example.count)


def collate(
    examples: list[RenderedExample],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(item.tokens) for item in examples)
    ids = torch.full((len(examples), width), PAD, dtype=torch.long)
    labels = torch.full((len(examples), width), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros((len(examples), width), dtype=torch.long)
    for row, item in enumerate(examples):
        length = len(item.tokens)
        ids[row, :length] = torch.tensor(item.tokens)
        labels[row, :length] = torch.tensor(item.labels)
        mask[row, :length] = 1
    return ids.to(device), labels.to(device), mask.to(device)
