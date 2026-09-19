from __future__ import annotations

import random
from dataclasses import dataclass, replace

from .vocab import MARKER_TOKENS, NOISE_TOKENS


@dataclass(frozen=True)
class BaseExample:
    example_id: str
    seq_len: int
    count: int
    needle_positions: list[int]
    needle_markers: list[str]
    seq_tokens: list[str]
    seed: int | None = None


def count_bin(count: int) -> str:
    if count <= 3:
        return "low"
    if count <= 6:
        return "mid"
    return "high"


def validate_example(example: BaseExample) -> None:
    if len(example.seq_tokens) != example.seq_len:
        raise AssertionError("seq_tokens length does not match seq_len.")
    if example.count != len(example.needle_positions) or example.count != len(example.needle_markers):
        raise AssertionError("count, needle_positions, and needle_markers disagree.")
    if example.needle_positions != sorted(example.needle_positions):
        raise AssertionError("needle_positions must be sorted.")
    if len(set(example.needle_positions)) != len(example.needle_positions):
        raise AssertionError("needle_positions must be unique.")
    for pos, marker in zip(example.needle_positions, example.needle_markers):
        if example.seq_tokens[pos] != marker:
            raise AssertionError("needle metadata does not match seq_tokens.")
    for token in example.seq_tokens:
        if token not in NOISE_TOKENS and token not in MARKER_TOKENS:
            raise AssertionError(f"unexpected prompt token: {token}")


def make_example(
    seq_len: int = 256,
    count: int = 1,
    rng: random.Random | None = None,
    seed: int | None = None,
    example_id: str | None = None,
) -> BaseExample:
    if not 1 <= int(count) <= 10:
        raise ValueError(f"count must be in 1..10, got {count}.")
    rng = rng or random.Random(seed)
    positions = sorted(rng.sample(range(int(seq_len)), int(count)))
    markers = [rng.choice(MARKER_TOKENS) for _ in range(int(count))]
    seq_tokens = [rng.choice(NOISE_TOKENS) for _ in range(int(seq_len))]
    for pos, marker in zip(positions, markers):
        seq_tokens[pos] = marker
    if example_id is None:
        base = seed if seed is not None else rng.randrange(1_000_000_000)
        example_id = f"ex_{base}_{seq_len}_{count}"
    example = BaseExample(str(example_id), int(seq_len), int(count), positions, markers, seq_tokens, seed)
    validate_example(example)
    return example


def sample_example(
    seq_len: int,
    rng: random.Random,
    count_min: int = 1,
    count_max: int = 10,
    seed: int | None = None,
) -> BaseExample:
    return make_example(seq_len, rng.randint(count_min, count_max), rng, seed=seed)


def balanced_examples(
    seq_len: int,
    examples_per_count: int,
    seed: int,
    counts: list[int] | None = None,
) -> list[BaseExample]:
    rng = random.Random(seed)
    counts = counts or list(range(1, 11))
    examples: list[BaseExample] = []
    for count in counts:
        for idx in range(int(examples_per_count)):
            ex_seed = seed * 1_000_000 + seq_len * 10_000 + count * 100 + idx
            examples.append(
                make_example(
                    seq_len=seq_len,
                    count=count,
                    rng=rng,
                    seed=ex_seed,
                    example_id=f"s{seed}_l{seq_len}_c{count}_{idx}",
                )
            )
    rng.shuffle(examples)
    return examples


def add_one_needle_pair(
    seq_len: int,
    count: int,
    rng: random.Random,
    example_id: str,
) -> tuple[BaseExample, BaseExample]:
    if not 1 <= count < 10:
        raise ValueError("count must be in 1..9 for add-one pairs.")
    minus = make_example(seq_len, count, rng, example_id=f"{example_id}_minus")
    available = [idx for idx in range(seq_len) if idx not in set(minus.needle_positions)]
    add_pos = rng.choice(available)
    add_marker = rng.choice(MARKER_TOKENS)
    seq_tokens = list(minus.seq_tokens)
    seq_tokens[add_pos] = add_marker
    insert_at = 0
    while insert_at < len(minus.needle_positions) and minus.needle_positions[insert_at] < add_pos:
        insert_at += 1
    positions = list(minus.needle_positions)
    markers = list(minus.needle_markers)
    positions.insert(insert_at, add_pos)
    markers.insert(insert_at, add_marker)
    plus = BaseExample(
        example_id=f"{example_id}_plus",
        seq_len=seq_len,
        count=count + 1,
        needle_positions=positions,
        needle_markers=markers,
        seq_tokens=seq_tokens,
    )
    validate_example(plus)
    return minus, plus


def delete_one_needle(example: BaseExample, rng: random.Random, example_id: str | None = None) -> BaseExample:
    if example.count <= 1:
        raise ValueError("cannot delete a needle from count=1 example.")
    remove_idx = rng.randrange(example.count)
    seq_tokens = list(example.seq_tokens)
    seq_tokens[example.needle_positions[remove_idx]] = rng.choice(NOISE_TOKENS)
    out = replace(
        example,
        example_id=example_id or f"{example.example_id}_delete",
        count=example.count - 1,
        needle_positions=[pos for idx, pos in enumerate(example.needle_positions) if idx != remove_idx],
        needle_markers=[marker for idx, marker in enumerate(example.needle_markers) if idx != remove_idx],
        seq_tokens=seq_tokens,
    )
    validate_example(out)
    return out


def replace_irrelevant_noise(example: BaseExample, rng: random.Random, example_id: str | None = None) -> BaseExample:
    noise_positions = [idx for idx in range(example.seq_len) if idx not in set(example.needle_positions)]
    if not noise_positions:
        return example
    seq_tokens = list(example.seq_tokens)
    for pos in rng.sample(noise_positions, k=min(3, len(noise_positions))):
        seq_tokens[pos] = rng.choice(NOISE_TOKENS)
    out = replace(example, example_id=example_id or f"{example.example_id}_noise", seq_tokens=seq_tokens)
    validate_example(out)
    return out


def permute_needle_markers(example: BaseExample, rng: random.Random, example_id: str | None = None) -> BaseExample:
    markers = list(example.needle_markers)
    rng.shuffle(markers)
    seq_tokens = list(example.seq_tokens)
    for pos, marker in zip(example.needle_positions, markers):
        seq_tokens[pos] = marker
    out = replace(example, example_id=example_id or f"{example.example_id}_perm_markers", needle_markers=markers, seq_tokens=seq_tokens)
    validate_example(out)
    return out
