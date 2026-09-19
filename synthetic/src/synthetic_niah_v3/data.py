from __future__ import annotations

import random
from dataclasses import dataclass

from .vocab import MARKER_TOKENS, NOISE_TOKENS


@dataclass(frozen=True)
class BaseExample:
    seq_len: int
    seq_tokens: list[str]
    count: int
    needle_positions: list[int]
    needle_markers: list[str]
    seed: int | None = None


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
            raise AssertionError("Needle marker metadata does not match sequence token.")
    for token in example.seq_tokens:
        if token not in NOISE_TOKENS and token not in MARKER_TOKENS:
            raise AssertionError(f"Unexpected prompt token: {token}")


def make_example(seq_len: int, count: int, rng: random.Random, seed: int | None = None) -> BaseExample:
    if not 1 <= count <= 10:
        raise ValueError(f"Count must be in 1..10, got {count}.")
    positions = sorted(rng.sample(range(seq_len), count))
    markers = [rng.choice(MARKER_TOKENS) for _ in range(count)]
    seq_tokens = [rng.choice(NOISE_TOKENS) for _ in range(seq_len)]
    for pos, marker in zip(positions, markers):
        seq_tokens[pos] = marker
    example = BaseExample(seq_len, seq_tokens, count, positions, markers, seed)
    validate_example(example)
    return example


def sample_example(seq_len: int, rng: random.Random, seed: int | None = None) -> BaseExample:
    return make_example(seq_len, rng.randint(1, 10), rng, seed=seed)


def balanced_examples(
    seq_len: int,
    examples_per_count: int,
    seed: int,
    counts: list[int] | None = None,
) -> list[BaseExample]:
    counts = counts or list(range(1, 11))
    rng = random.Random(seed)
    examples: list[BaseExample] = []
    for count in counts:
        for idx in range(examples_per_count):
            ex_seed = seed * 1_000_000 + seq_len * 10_000 + count * 100 + idx
            examples.append(make_example(seq_len, count, rng, seed=ex_seed))
    rng.shuffle(examples)
    return examples
