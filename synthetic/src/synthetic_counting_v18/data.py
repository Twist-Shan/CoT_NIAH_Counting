from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from .config import ReferenceConfig, RunSpec


# Disjoint token families make marker identity, ordinal progress, and scalar
# count independently inspectable.
NOISE_OFFSET = 0
NOISE_VOCAB_SIZE = 256
MARKER_OFFSET = 256
MARKER_VOCAB_SIZE = 10
ANSWER = 266
START = 267
END = 268
PAD = 269
INDEX_OFFSET = 270
MAXIMUM_COUNT = 128
COUNT_OFFSET = INDEX_OFFSET + MAXIMUM_COUNT
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


def index_token(index: int) -> int:
    if not 1 <= int(index) <= MAXIMUM_COUNT:
        raise ValueError(f"index must be in 1..{MAXIMUM_COUNT}: {index}")
    return INDEX_OFFSET + int(index) - 1


def index_from_token(token: int, maximum: int = MAXIMUM_COUNT) -> int | None:
    value = int(token) - INDEX_OFFSET + 1
    return value if 1 <= value <= int(maximum) else None


def count_token(count: int) -> int:
    if not 1 <= int(count) <= MAXIMUM_COUNT:
        raise ValueError(f"count must be in 1..{MAXIMUM_COUNT}: {count}")
    return COUNT_OFFSET + int(count) - 1


def count_from_token(token: int, maximum: int) -> int | None:
    value = int(token) - COUNT_OFFSET + 1
    return value if 1 <= value <= int(maximum) else None


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
        completion = [count_token(example.count)]
    elif mode == "cot":
        prefix.append(START)
        completion: list[int] = []
        for index, marker in enumerate(example.needle_markers, 1):
            completion.extend((index_token(index), marker))
        completion.extend((END, count_token(example.count)))
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
