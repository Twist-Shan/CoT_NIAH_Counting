from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import count_bin
from .vocab import MARKER_TOKENS, NOISE_TOKENS


@dataclass(frozen=True)
class BaseExample:
    seq_tokens: list[str]
    count: int
    needle_positions: list[int]
    needle_markers: list[str]
    seed: int | None = None
    example_id: str = ""

    @property
    def seq_len(self) -> int:
        return len(self.seq_tokens)


def validate_example(example: BaseExample) -> None:
    marker_vocab = set(MARKER_TOKENS)
    if example.count != len(example.needle_positions) or example.count != len(example.needle_markers):
        raise AssertionError("count, needle positions, and needle markers disagree.")
    if example.needle_positions != sorted(example.needle_positions):
        raise AssertionError("needle positions must be sorted.")
    if len(set(example.needle_positions)) != len(example.needle_positions):
        raise AssertionError("needle positions must be unique.")
    if not 1 <= example.count <= 10:
        raise AssertionError("count must be in 1..10.")
    for pos, marker in zip(example.needle_positions, example.needle_markers):
        if example.seq_tokens[pos] != marker:
            raise AssertionError("needle metadata does not match seq_tokens.")
    observed_markers = sum(token in marker_vocab for token in example.seq_tokens)
    if observed_markers != example.count:
        raise AssertionError(f"prompt contains {observed_markers} marker tokens but count={example.count}.")
    for token in example.seq_tokens:
        if token not in marker_vocab and token not in NOISE_TOKENS:
            raise AssertionError(f"unexpected prompt token: {token}")


def make_example(
    seq_len: int,
    count: int,
    rng: random.Random,
    seed: int | None = None,
    example_id: str = "",
) -> BaseExample:
    if not 1 <= int(count) <= 10:
        raise ValueError(f"count must be in 1..10, got {count}.")
    positions = sorted(rng.sample(range(int(seq_len)), int(count)))
    markers = [rng.choice(MARKER_TOKENS) for _ in range(int(count))]
    seq_tokens = [rng.choice(NOISE_TOKENS) for _ in range(int(seq_len))]
    for pos, marker in zip(positions, markers):
        seq_tokens[pos] = marker
    example = BaseExample(seq_tokens, int(count), positions, markers, seed=seed, example_id=example_id)
    validate_example(example)
    return example


def sample_example(seq_len: int, rng: random.Random, min_count: int = 1, max_count: int = 10, seed: int | None = None) -> BaseExample:
    count = rng.randint(int(min_count), int(max_count))
    return make_example(seq_len, count, rng, seed=seed, example_id=f"s{seed}_c{count}" if seed is not None else "")


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
            ex_seed = seed * 1_000_000 + count * 10_000 + idx
            examples.append(
                make_example(
                    seq_len=seq_len,
                    count=count,
                    rng=rng,
                    seed=ex_seed,
                    example_id=f"s{seed}_c{count}_{idx}",
                )
            )
    rng.shuffle(examples)
    return examples


def example_to_dict(example: BaseExample) -> dict:
    data = asdict(example)
    data["count_bin"] = count_bin(example.count)
    return data


def save_examples(path: str | Path, examples: list[BaseExample]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example_to_dict(example), ensure_ascii=False) + "\n")


def load_examples(path: str | Path) -> list[BaseExample]:
    examples: list[BaseExample] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            examples.append(
                BaseExample(
                    seq_tokens=list(obj["seq_tokens"]),
                    count=int(obj["count"]),
                    needle_positions=[int(x) for x in obj["needle_positions"]],
                    needle_markers=list(obj["needle_markers"]),
                    seed=obj.get("seed"),
                    example_id=str(obj.get("example_id", "")),
                )
            )
    for example in examples:
        validate_example(example)
    return examples


def has_repeated_markers(example: BaseExample) -> bool:
    return len(set(example.needle_markers)) != len(example.needle_markers)

