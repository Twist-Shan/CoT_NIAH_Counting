from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn.functional as F

from .config import ExperimentConfig


IGNORE_INDEX = -100
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
TINY_SHAKESPEARE_ENV_VAR = "SYNTHETIC_COUNTING_TINY_SHAKESPEARE_PATH"
TINY_SHAKESPEARE_MIN_BYTES = 1_000_000


def tiny_shakespeare_path() -> Path:
    override = os.environ.get(TINY_SHAKESPEARE_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return Path(__file__).with_name("resources") / "tiny_shakespeare" / "input.txt"


def ensure_tiny_shakespeare_corpus(path: str | Path | None = None, *, allow_download: bool = True) -> Path:
    corpus_path = Path(path).expanduser() if path is not None else tiny_shakespeare_path()
    if corpus_path.exists() and corpus_path.stat().st_size >= TINY_SHAKESPEARE_MIN_BYTES:
        return corpus_path
    if not allow_download:
        raise FileNotFoundError(
            f"Standard Tiny Shakespeare corpus not found at {corpus_path}. "
            f"Run `python scripts/fetch_tiny_shakespeare.py` or set {TINY_SHAKESPEARE_ENV_VAR}."
        )
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urlretrieve(TINY_SHAKESPEARE_URL, corpus_path)
    except (OSError, URLError) as exc:
        raise FileNotFoundError(
            f"Could not download standard Tiny Shakespeare from {TINY_SHAKESPEARE_URL}. "
            f"Run `python scripts/fetch_tiny_shakespeare.py` when network is available, "
            f"or set {TINY_SHAKESPEARE_ENV_VAR} to a local input.txt."
        ) from exc
    if corpus_path.stat().st_size < TINY_SHAKESPEARE_MIN_BYTES:
        raise RuntimeError(
            f"Downloaded Tiny Shakespeare file is unexpectedly small: {corpus_path} "
            f"({corpus_path.stat().st_size} bytes)."
        )
    return corpus_path


@lru_cache(maxsize=1)
def shakespeare_text() -> str:
    return ensure_tiny_shakespeare_corpus().read_text(encoding="utf-8")


def _char_token(character: str) -> str:
    return f"<CH_{ord(character):04X}>"


@lru_cache(maxsize=1)
def shakespeare_char_tokens() -> tuple[str, ...]:
    return tuple(_char_token(char) for char in shakespeare_text())


@lru_cache(maxsize=1)
def shakespeare_vocab_tokens() -> tuple[str, ...]:
    return tuple(sorted(set(shakespeare_char_tokens())))


def corpus_split_bounds(cfg: ExperimentConfig) -> dict[str, tuple[int, int]]:
    """Split the corpus before any sliding-window index is constructed."""

    size = len(shakespeare_char_tokens())
    train_end = int(size * cfg.corpus_train_fraction)
    validation_end = train_end + int(size * cfg.corpus_validation_fraction)
    return {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, size),
    }


def corpus_split_tokens(cfg: ExperimentConfig, split: str) -> tuple[str, ...]:
    bounds = corpus_split_bounds(cfg)
    if split not in bounds:
        raise ValueError(f"Unknown corpus split {split!r}; expected one of {tuple(bounds)}")
    start, end = bounds[split]
    return shakespeare_char_tokens()[start:end]


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    numbers: list[str]
    markers: list[str]
    noise: list[str]
    noise_source: str
    task_type: str = "inserted_marker"
    loss_scope: str = "completion"
    target_characters: tuple[str, ...] = ()

    @classmethod
    def build(cls, cfg: ExperimentConfig) -> "Vocab":
        special = ["<PAD>", "<BOS>", "<EOS>", "<Think>", "</Think>", "<Ans>"]
        if cfg.task_type == "target_character":
            special.extend(["<CountChar>", "<Sep>"])
        if cfg.noise_source == "shakespeare_char":
            noise = list(shakespeare_vocab_tokens())
        else:
            noise = [f"<N{i}>" for i in range(cfg.noise_vocab_size)]
        markers = [f"<M{i}>" for i in range(cfg.marker_vocab_size)]
        numbers = [f"<{i}>" for i in range(1, cfg.count_max + 1)]
        tokens = special + noise + markers + numbers
        if len(tokens) != len(set(tokens)):
            raise ValueError("Synthetic counting vocabulary has duplicate tokens")
        return cls(
            {token: idx for idx, token in enumerate(tokens)},
            tokens,
            numbers,
            markers,
            noise,
            cfg.noise_source,
            cfg.task_type,
            cfg.loss_scope,
            tuple(cfg.target_characters),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            dict(obj["token_to_id"]),
            list(obj["id_to_token"]),
            list(obj["numbers"]),
            list(obj["markers"]),
            list(obj["noise"]),
            str(obj.get("noise_source", "uniform")),
            str(obj.get("task_type", "inserted_marker")),
            str(obj.get("loss_scope", "completion")),
            tuple(obj.get("target_characters", ())),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "token_to_id": self.token_to_id,
                    "id_to_token": self.id_to_token,
                    "numbers": self.numbers,
                    "markers": self.markers,
                    "noise": self.noise,
                    "noise_source": self.noise_source,
                    "task_type": self.task_type,
                    "loss_scope": self.loss_scope,
                    "target_characters": list(self.target_characters),
                    "shared_trace_and_answer_numbers": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def encode(self, tokens: Iterable[str]) -> list[int]:
        return [self.token_to_id[token] for token in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.id_to_token[int(idx)] for idx in ids]

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    @property
    def ans_id(self) -> int:
        return self.token_to_id["<Ans>"]

    @property
    def number_ids(self) -> list[int]:
        return [self.token_to_id[token] for token in self.numbers]

    @property
    def marker_ids(self) -> list[int]:
        return [self.token_to_id[token] for token in self.markers]

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.id_to_token, ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def number_token(self, value: int) -> str:
        value = int(value)
        if not 1 <= value <= len(self.numbers):
            raise ValueError(f"numeric token must be in 1..{len(self.numbers)}, got {value}")
        return self.numbers[value - 1]

    def number_id(self, value: int) -> int:
        return self.token_to_id[self.number_token(value)]

    def number_from_id(self, token_id: int) -> int | None:
        token = self.id_to_token[int(token_id)]
        return self.numbers.index(token) + 1 if token in self.numbers else None


@dataclass(frozen=True)
class Example:
    seq_tokens: list[str]
    count: int
    needle_positions: list[int]
    needle_markers: list[str]
    seed: int | None = None
    target_token: str | None = None
    target_character: str | None = None


@dataclass(frozen=True)
class Spans:
    bos_pos: int
    prompt_start: int
    prompt_end_exclusive: int
    think_pos: int | None
    trace_index_positions: list[int]
    trace_marker_positions: list[int]
    think_close_pos: int | None
    ans_pos: int
    count_pos: int
    eos_pos: int
    task_prefix_positions: tuple[int, ...] = ()


@dataclass(frozen=True)
class Rendered:
    mode: str
    tokens: list[str]
    input_ids: list[int]
    labels: list[int]
    spans: Spans
    prompt_needle_positions: list[int]
    count: int


def _noise_sequence(cfg: ExperimentConfig, vocab: Vocab, rng: random.Random) -> list[str]:
    if cfg.noise_source == "uniform":
        return [rng.choice(vocab.noise) for _ in range(cfg.seq_len)]
    source = shakespeare_char_tokens()
    if len(source) < cfg.seq_len:
        repeats = (cfg.seq_len + len(source) - 1) // len(source)
        source = (source * repeats)[: cfg.seq_len]
        return list(source)
    start = rng.randrange(0, len(source) - cfg.seq_len + 1)
    return list(source[start : start + cfg.seq_len])


def count_sampling_probabilities(cfg: ExperimentConfig) -> np.ndarray:
    """Return the normalized training distribution over exact counts."""

    counts = np.arange(cfg.count_min, cfg.count_max + 1, dtype=np.float64)
    if cfg.count_sampling == "uniform":
        weights = np.ones_like(counts)
    elif cfg.count_sampling == "power":
        weights = counts ** (-float(cfg.power_alpha))
    elif cfg.count_sampling == "exponential":
        weights = np.exp(-float(cfg.exponential_beta) * (counts - cfg.count_min))
    else:
        raise ValueError(f"Unknown count sampler: {cfg.count_sampling}")
    return weights / weights.sum()


def sample_training_count(cfg: ExperimentConfig, rng: random.Random) -> int:
    probabilities = count_sampling_probabilities(cfg)
    draw = rng.random()
    cumulative = 0.0
    for count, probability in zip(range(cfg.count_min, cfg.count_max + 1), probabilities):
        cumulative += float(probability)
        if draw <= cumulative:
            return int(count)
    return int(cfg.count_max)


@lru_cache(maxsize=16)
def _target_window_starts(
    seq_len: int,
    count_max: int,
    target_tokens: tuple[str, ...],
) -> dict[tuple[str, int], np.ndarray]:
    """Index Tiny Shakespeare windows by target token and exact occurrence count."""

    source = np.asarray(shakespeare_char_tokens(), dtype=object)
    result: dict[tuple[str, int], np.ndarray] = {}
    for target in target_tokens:
        matches = (source == target).astype(np.int32)
        prefix = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(matches)))
        window_counts = prefix[seq_len:] - prefix[:-seq_len]
        for count in range(1, int(count_max) + 1):
            result[(target, count)] = np.flatnonzero(window_counts == count)
    return result


@lru_cache(maxsize=32)
def _split_target_window_starts(
    seq_len: int,
    count_max: int,
    target_tokens: tuple[str, ...],
    train_fraction: float,
    validation_fraction: float,
    split: str,
) -> dict[tuple[str, int], np.ndarray]:
    """Index windows after the raw corpus has been partitioned."""

    size = len(shakespeare_char_tokens())
    train_end = int(size * train_fraction)
    validation_end = train_end + int(size * validation_fraction)
    bounds = {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, size),
    }
    if split not in bounds:
        raise ValueError(f"Unknown corpus split {split!r}")
    lo, hi = bounds[split]
    source = np.asarray(shakespeare_char_tokens()[lo:hi], dtype=object)
    if len(source) < seq_len:
        raise ValueError(f"Corpus split {split!r} is shorter than seq_len={seq_len}")
    result: dict[tuple[str, int], np.ndarray] = {}
    for target in target_tokens:
        matches = (source == target).astype(np.int32)
        prefix = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(matches)))
        window_counts = prefix[seq_len:] - prefix[:-seq_len]
        for count in range(1, int(count_max) + 1):
            result[(target, count)] = np.flatnonzero(window_counts == count)
    return result


def split_target_window_starts(
    cfg: ExperimentConfig,
    split: str,
) -> dict[tuple[str, int], np.ndarray]:
    target_tokens = tuple(_char_token(character) for character in cfg.target_characters)
    return _split_target_window_starts(
        cfg.seq_len,
        cfg.count_max,
        target_tokens,
        cfg.corpus_train_fraction,
        cfg.corpus_validation_fraction,
        split,
    )


class WindowWithoutReplacementSampler:
    """Uniformly sample eligible indexed windows once per sampler epoch.

    Strata are not balanced. Selecting in proportion to the number of remaining
    windows makes every eligible `(target, count, start)` window equally likely.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        vocab: Vocab,
        *,
        split: str,
        seed: int,
        minimum_candidates: int | None = None,
    ) -> None:
        if cfg.task_type != "target_character":
            raise ValueError("window sampler is only defined for target-character counting")
        self.cfg = cfg
        self.vocab = vocab
        self.split = split
        # Materialize the immutable corpus split once per sampler.  Slicing the
        # full Shakespeare token tuple creates a new tuple, so doing this in
        # sample_example() would copy roughly the entire training split for
        # every example in every batch.
        self.split_tokens = corpus_split_tokens(cfg, split)
        self.split_start = corpus_split_bounds(cfg)[split][0]
        self.seed = int(seed)
        self.minimum_candidates = int(
            cfg.min_candidate_windows if minimum_candidates is None else minimum_candidates
        )
        self.index = split_target_window_starts(cfg, split)
        self.token_to_character = {
            _char_token(character): character for character in cfg.target_characters
        }
        self.strata = tuple(
            sorted(
                key
                for key, starts in self.index.items()
                if len(starts) >= self.minimum_candidates
            )
        )
        if not self.strata:
            raise ValueError(
                f"No {split} (target,count) stratum has at least "
                f"{self.minimum_candidates} candidate windows"
            )
        self.epoch = 0
        self.cursors = {key: 0 for key in self.strata}
        self.selection_rng = random.Random(self.seed)
        self._orders: dict[tuple[str, int], np.ndarray] = {}
        self._make_orders()

    @property
    def epoch_size(self) -> int:
        return int(sum(len(self.index[key]) for key in self.strata))

    def _order_seed(self, key: tuple[str, int]) -> int:
        token, count = key
        digest = hashlib.sha256(
            f"{self.seed}|{self.split}|{self.epoch}|{token}|{count}".encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "little", signed=False)

    def _make_orders(self) -> None:
        self._orders = {}
        for key in self.strata:
            order = np.arange(len(self.index[key]), dtype=np.int64)
            np.random.default_rng(self._order_seed(key)).shuffle(order)
            self._orders[key] = order

    def _start_next_epoch(self) -> None:
        self.epoch += 1
        self.cursors = {key: 0 for key in self.strata}
        self._make_orders()

    def state_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "seed": self.seed,
            "minimum_candidates": self.minimum_candidates,
            "epoch": self.epoch,
            "cursors": {f"{token}\t{count}": value for (token, count), value in self.cursors.items()},
            "selection_rng_state": self.selection_rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("split") != self.split or int(state.get("seed", -1)) != self.seed:
            raise ValueError("window sampler state does not match split/seed")
        if int(state.get("minimum_candidates", -1)) != self.minimum_candidates:
            raise ValueError("window sampler state uses a different candidate threshold")
        self.epoch = int(state["epoch"])
        raw_cursors = dict(state["cursors"])
        restored = {}
        for key in self.strata:
            restored[key] = int(raw_cursors.get(f"{key[0]}\t{key[1]}", 0))
        self.cursors = restored
        self.selection_rng.setstate(state["selection_rng_state"])
        self._make_orders()

    def _remaining(self, key: tuple[str, int]) -> int:
        return len(self._orders[key]) - self.cursors[key]

    def sample_example(self) -> Example:
        remaining = [(key, self._remaining(key)) for key in self.strata if self._remaining(key)]
        if not remaining:
            self._start_next_epoch()
            remaining = [(key, self._remaining(key)) for key in self.strata]
        draw = self.selection_rng.randrange(sum(value for _, value in remaining))
        chosen = remaining[-1][0]
        cumulative = 0
        for key, value in remaining:
            cumulative += value
            if draw < cumulative:
                chosen = key
                break
        cursor = self.cursors[chosen]
        local_start = int(self.index[chosen][self._orders[chosen][cursor]])
        self.cursors[chosen] = cursor + 1
        target_token, count = chosen
        sequence = list(self.split_tokens[local_start : local_start + self.cfg.seq_len])
        positions = [idx for idx, token in enumerate(sequence) if token == target_token]
        return Example(
            sequence,
            int(count),
            positions,
            [target_token] * len(positions),
            self.split_start + local_start,
            target_token,
            self.token_to_character[target_token],
        )

    def sample(self, batch_size: int) -> list[Example]:
        return [self.sample_example() for _ in range(int(batch_size))]


def split_window_examples(
    cfg: ExperimentConfig,
    vocab: Vocab,
    examples_per_count: int,
    seed: int,
    *,
    split: str,
    count_min: int | None = None,
    count_max: int | None = None,
) -> list[Example]:
    """Balanced evaluation drawn without replacement from a held-out split."""

    index = split_target_window_starts(cfg, split)
    tokens = corpus_split_tokens(cfg, split)
    split_start = corpus_split_bounds(cfg)[split][0]
    token_to_character = {_char_token(char): char for char in cfg.target_characters}
    rng = random.Random(seed)
    result: list[Example] = []
    lo = cfg.count_min if count_min is None else int(count_min)
    hi = cfg.count_max if count_max is None else int(count_max)
    for count in range(lo, hi + 1):
        candidates = [
            (target, int(start))
            for (target, indexed_count), starts in index.items()
            if indexed_count == count
            for start in starts
        ]
        if len(candidates) < examples_per_count:
            raise ValueError(
                f"Held-out split {split!r} has {len(candidates)} windows for count={count}, "
                f"but {examples_per_count} were requested"
            )
        rng.shuffle(candidates)
        for target_token, local_start in candidates[:examples_per_count]:
            sequence = list(tokens[local_start : local_start + cfg.seq_len])
            positions = [idx for idx, token in enumerate(sequence) if token == target_token]
            result.append(
                Example(
                    sequence,
                    count,
                    positions,
                    [target_token] * count,
                    split_start + local_start,
                    target_token,
                    token_to_character[target_token],
                )
            )
    rng.shuffle(result)
    return result


def _target_character_example(
    cfg: ExperimentConfig,
    rng: random.Random,
    count: int,
    seed: int | None,
) -> Example:
    target_pairs = tuple((character, _char_token(character)) for character in cfg.target_characters)
    starts = _target_window_starts(
        cfg.seq_len,
        cfg.count_max,
        tuple(token for _, token in target_pairs),
    )
    viable = [pair for pair in target_pairs if len(starts[(pair[1], count)])]
    if not viable:
        raise ValueError(
            f"Tiny Shakespeare has no length-{cfg.seq_len} target-character window "
            f"with exactly {count} occurrences for targets {cfg.target_characters}"
        )
    character, target_token = rng.choice(viable)
    candidate_starts = starts[(target_token, count)]
    start = int(candidate_starts[rng.randrange(len(candidate_starts))])
    sequence = list(shakespeare_char_tokens()[start : start + cfg.seq_len])
    positions = [index for index, token in enumerate(sequence) if token == target_token]
    return Example(
        sequence,
        int(count),
        positions,
        [target_token] * len(positions),
        seed,
        target_token,
        character,
    )


def make_example(
    cfg: ExperimentConfig,
    vocab: Vocab,
    rng: random.Random,
    count: int | None = None,
    seed: int | None = None,
) -> Example:
    n = sample_training_count(cfg, rng) if count is None else int(count)
    if not cfg.count_min <= n <= cfg.count_max:
        raise ValueError(f"count must be in {cfg.count_min}..{cfg.count_max}")
    if cfg.task_type == "target_character":
        return _target_character_example(cfg, rng, n, seed)
    positions = sorted(rng.sample(range(cfg.seq_len), n))
    markers = [rng.choice(vocab.markers) for _ in positions]
    sequence = _noise_sequence(cfg, vocab, rng)
    for position, marker in zip(positions, markers):
        sequence[position] = marker
    return Example(sequence, n, positions, markers, seed)


def balanced_examples(
    cfg: ExperimentConfig,
    vocab: Vocab,
    examples_per_count: int,
    seed: int,
    *,
    count_min: int | None = None,
    count_max: int | None = None,
) -> list[Example]:
    lo = cfg.count_min if count_min is None else int(count_min)
    hi = cfg.count_max if count_max is None else int(count_max)
    if cfg.training_data_mode == "split_window_index":
        return split_window_examples(
            cfg,
            vocab,
            examples_per_count,
            seed,
            split="validation",
            count_min=lo,
            count_max=hi,
        )
    rng = random.Random(seed)
    result: list[Example] = []
    for count in range(lo, hi + 1):
        for index in range(int(examples_per_count)):
            example_seed = seed * 1_000_000 + count * 10_000 + index
            result.append(make_example(cfg, vocab, rng, count=count, seed=example_seed))
    rng.shuffle(result)
    return result


def render(example: Example, vocab: Vocab, mode: str) -> Rendered:
    task_prefix = (
        ["<CountChar>", str(example.target_token), "<Sep>"]
        if vocab.task_type == "target_character"
        else []
    )
    if vocab.task_type == "target_character" and example.target_token is None:
        raise ValueError("target-character examples require target_token")
    prompt_start = 1 + len(task_prefix)
    prompt_end = prompt_start + len(example.seq_tokens)
    prompt_needles = [prompt_start + position for position in example.needle_positions]
    if mode == "nonthinking":
        tokens = [
            "<BOS>",
            *task_prefix,
            *example.seq_tokens,
            "<Ans>",
            vocab.number_token(example.count),
            "<EOS>",
        ]
        ans_pos = prompt_end
        count_pos = ans_pos + 1
        eos_pos = count_pos + 1
        labels = [IGNORE_INDEX] * len(tokens)
        labels[count_pos] = vocab.number_id(example.count)
        labels[eos_pos] = vocab.eos_id
        spans = Spans(
            0,
            prompt_start,
            prompt_end,
            None,
            [],
            [],
            None,
            ans_pos,
            count_pos,
            eos_pos,
            tuple(range(1, prompt_start)),
        )
    elif mode == "thinking":
        trace: list[str] = []
        for k, marker in enumerate(example.needle_markers, start=1):
            trace.extend([vocab.number_token(k), marker])
        think_pos = prompt_end
        trace_start = think_pos + 1
        trace_positions = list(range(trace_start, trace_start + len(trace)))
        index_positions = trace_positions[0::2]
        marker_positions = trace_positions[1::2]
        close_pos = trace_start + len(trace)
        ans_pos = close_pos + 1
        count_pos = ans_pos + 1
        eos_pos = count_pos + 1
        tokens = [
            "<BOS>",
            *task_prefix,
            *example.seq_tokens,
            "<Think>",
            *trace,
            "</Think>",
            "<Ans>",
            vocab.number_token(example.count),
            "<EOS>",
        ]
        labels = [IGNORE_INDEX] * len(tokens)
        for target_pos in range(trace_start, len(tokens)):
            labels[target_pos] = vocab.token_to_id[tokens[target_pos]]
        spans = Spans(
            0,
            prompt_start,
            prompt_end,
            think_pos,
            index_positions,
            marker_positions,
            close_pos,
            ans_pos,
            count_pos,
            eos_pos,
            tuple(range(1, prompt_start)),
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    input_ids = vocab.encode(tokens)
    if vocab.loss_scope == "all_sequence":
        labels = list(input_ids)
    return Rendered(mode, tokens, input_ids, labels, spans, prompt_needles, example.count)


def collate(
    rendered: list[Rendered],
    vocab: Vocab,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(item.input_ids) for item in rendered)
    ids = torch.full((len(rendered), max_len), vocab.pad_id, dtype=torch.long)
    labels = torch.full((len(rendered), max_len), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros((len(rendered), max_len), dtype=torch.long)
    for row, item in enumerate(rendered):
        length = len(item.input_ids)
        ids[row, :length] = torch.tensor(item.input_ids, dtype=torch.long)
        labels[row, :length] = torch.tensor(item.labels, dtype=torch.long)
        mask[row, :length] = 1
    return ids.to(device), labels.to(device), mask.to(device)


def shifted_token_losses(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    ).view_as(shift_labels)
    active = shift_labels.ne(IGNORE_INDEX)
    total = (losses * active).sum() / active.sum().clamp_min(1)
    return total, losses


def component_target_positions(item: Rendered) -> dict[str, list[int]]:
    if item.mode == "nonthinking":
        components = {
            "ans_token": [item.spans.ans_pos],
            "final_count": [item.spans.count_pos],
            "eos": [item.spans.eos_pos],
        }
    else:
        components = {
            "trace_index": list(item.spans.trace_index_positions),
            "trace_marker": list(item.spans.trace_marker_positions),
            "think_close": (
                [item.spans.think_close_pos]
                if item.spans.think_close_pos is not None
                else []
            ),
            "ans_token": [item.spans.ans_pos],
            "final_count": [item.spans.count_pos],
            "eos": [item.spans.eos_pos],
        }
    active_positions = {
        position for position, label in enumerate(item.labels) if label != IGNORE_INDEX and position > 0
    }
    if item.spans.task_prefix_positions:
        components["task_prefix"] = list(item.spans.task_prefix_positions)
    components["prompt_body"] = list(
        range(item.spans.prompt_start, item.spans.prompt_end_exclusive)
    )
    if item.spans.think_pos is not None:
        components["think_open"] = [item.spans.think_pos]
    return {
        name: [position for position in positions if position in active_positions]
        for name, positions in components.items()
    }


def component_loss_values(losses: torch.Tensor, rendered: list[Rendered]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row, item in enumerate(rendered):
        for name, positions in component_target_positions(item).items():
            for target_position in positions:
                if target_position > 0:
                    values.setdefault(name, []).append(float(losses[row, target_position - 1].detach().cpu()))
    return {name: float(np.mean(parts)) for name, parts in values.items() if parts}


def count_prediction(logits: torch.Tensor, vocab: Vocab) -> tuple[int, int, float]:
    number_ids = torch.tensor(vocab.number_ids, device=logits.device)
    number_logits = logits[number_ids]
    local_index = int(number_logits.argmax())
    prediction = local_index + 1
    return prediction, vocab.number_id(prediction), float(number_logits[local_index].detach().cpu())


@dataclass
class FixedExamplePool:
    prompt_ids: np.ndarray
    counts: np.ndarray
    seeds: np.ndarray
    vocab_fingerprint: str

    def __len__(self) -> int:
        return int(len(self.counts))

    def example(self, index: int, vocab: Vocab) -> Example:
        prompt = [int(value) for value in self.prompt_ids[int(index)]]
        marker_ids = set(vocab.marker_ids)
        positions = [position for position, token_id in enumerate(prompt) if token_id in marker_ids]
        markers = [vocab.id_to_token[prompt[position]] for position in positions]
        return Example(vocab.decode(prompt), int(self.counts[index]), positions, markers, int(self.seeds[index]))

    def sample(self, rng: random.Random, batch_size: int, vocab: Vocab) -> list[Example]:
        return [self.example(rng.randrange(len(self)), vocab) for _ in range(int(batch_size))]


def load_or_create_fixed_pool(
    path: str | Path,
    cfg: ExperimentConfig,
    vocab: Vocab,
) -> FixedExamplePool:
    path = Path(path)
    metadata_path = path.with_suffix(".json")
    if path.exists() and metadata_path.exists():
        arrays = np.load(path, allow_pickle=False)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        pool = FixedExamplePool(arrays["prompt_ids"], arrays["counts"], arrays["seeds"], metadata["vocab_fingerprint"])
        if pool.vocab_fingerprint != vocab.fingerprint:
            raise ValueError("fixed dataset vocabulary fingerprint does not match this run")
        if pool.prompt_ids.shape[1] != cfg.seq_len:
            raise ValueError("fixed dataset prompt length does not match this run")
        return pool

    examples = balanced_examples(
        cfg,
        vocab,
        cfg.fixed_train_examples_per_count,
        cfg.seed + 13_000,
    )
    prompt_ids = np.asarray([vocab.encode(example.seq_tokens) for example in examples], dtype=np.int16)
    counts = np.asarray([example.count for example in examples], dtype=np.int16)
    seeds = np.asarray([example.seed for example in examples], dtype=np.int64)
    temporary = path.with_suffix(".tmp.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(temporary, prompt_ids=prompt_ids, counts=counts, seeds=seeds)
    temporary.replace(path)
    dataset_hash = hashlib.sha256()
    dataset_hash.update(prompt_ids.tobytes())
    dataset_hash.update(counts.tobytes())
    metadata_path.write_text(
        json.dumps(
            {
                "num_examples": int(len(examples)),
                "examples_per_count": int(cfg.fixed_train_examples_per_count),
                "seq_len": int(cfg.seq_len),
                "count_min": int(cfg.count_min),
                "count_max": int(cfg.count_max),
                "vocab_fingerprint": vocab.fingerprint,
                "dataset_sha256": dataset_hash.hexdigest(),
                "training_semantics": "finite pool sampled with replacement; no newly generated prompts",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return FixedExamplePool(prompt_ids, counts, seeds, vocab.fingerprint)
