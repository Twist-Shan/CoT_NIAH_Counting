from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from .config import V16_3Config
from .needle_pool import NeedlePool, NeedleSet


IGNORE_INDEX = -100
TINY_SHAKESPEARE_ENV_VAR = "SYNTHETIC_COUNTING_TINY_SHAKESPEARE_PATH"


def character_token(character: str) -> str:
    if len(character) != 1:
        raise ValueError("character tokens require exactly one Unicode character")
    return f"<CH_{ord(character):04X}>"


def default_corpus_path() -> Path:
    return Path(__file__).parents[1] / "synthetic_counting_v11" / "resources" / "tiny_shakespeare" / "input.txt"


def load_corpus_text(path: str | Path | None = None) -> str:
    selected = Path(path) if path is not None else Path(os.environ.get(TINY_SHAKESPEARE_ENV_VAR, default_corpus_path()))
    if not selected.exists():
        raise FileNotFoundError(
            f"Tiny Shakespeare is missing at {selected}; run scripts/fetch_tiny_shakespeare.py "
            f"or set {TINY_SHAKESPEARE_ENV_VAR}"
        )
    return selected.read_text(encoding="utf-8")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CorpusRegion:
    name: str
    start: int
    end: int
    sha256: str

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class CorpusSplit:
    corpus_sha256: str
    corpus_length: int
    seq_len: int
    train: CorpusRegion
    validation: CorpusRegion
    test: CorpusRegion
    guards: tuple[tuple[int, int], tuple[int, int]]
    split_fingerprint: str

    def region(self, name: str) -> CorpusRegion:
        if name == "heldout":
            name = "validation"
        if name not in {"train", "validation", "test"}:
            raise ValueError(f"unknown corpus region: {name}")
        return getattr(self, name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "v16_3",
            "corpus_sha256": self.corpus_sha256,
            "corpus_length": self.corpus_length,
            "seq_len": self.seq_len,
            "train": asdict(self.train),
            "validation": asdict(self.validation),
            "test": asdict(self.test),
            "guards": [list(pair) for pair in self.guards],
            "split_fingerprint": self.split_fingerprint,
        }


def build_corpus_split(cfg: V16_3Config, text: str) -> CorpusSplit:
    guard = cfg.seq_len - 1
    usable = len(text) - 2 * guard
    if usable < 3 * cfg.seq_len:
        raise ValueError("corpus is too short for guarded train/validation/test splits")
    train_length = int(usable * cfg.corpus_train_fraction)
    validation_length = int(usable * cfg.corpus_validation_fraction)
    test_length = usable - train_length - validation_length
    lengths = (train_length, validation_length, test_length)
    if min(lengths) < cfg.seq_len:
        raise ValueError("each corpus region must contain at least one complete window")
    train_start, train_end = 0, train_length
    guard1 = (train_end, train_end + guard)
    validation_start = guard1[1]
    validation_end = validation_start + validation_length
    guard2 = (validation_end, validation_end + guard)
    test_start, test_end = guard2[1], len(text)
    regions = {
        "train": CorpusRegion("train", train_start, train_end, _sha256_text(text[train_start:train_end])),
        "validation": CorpusRegion(
            "validation", validation_start, validation_end, _sha256_text(text[validation_start:validation_end])
        ),
        "test": CorpusRegion("test", test_start, test_end, _sha256_text(text[test_start:test_end])),
    }
    payload = {
        "corpus_sha256": _sha256_text(text),
        "corpus_length": len(text),
        "seq_len": cfg.seq_len,
        "fractions": [cfg.corpus_train_fraction, cfg.corpus_validation_fraction, cfg.corpus_test_fraction],
        "regions": {key: asdict(value) for key, value in regions.items()},
        "guards": [guard1, guard2],
    }
    return CorpusSplit(
        corpus_sha256=payload["corpus_sha256"],
        corpus_length=len(text),
        seq_len=cfg.seq_len,
        train=regions["train"],
        validation=regions["validation"],
        test=regions["test"],
        guards=(guard1, guard2),
        split_fingerprint=_sha256_json(payload),
    )


def save_corpus_split(split: CorpusSplit, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(split.to_dict(), indent=2), encoding="utf-8")
    temporary.replace(path)


def load_corpus_split(path: str | Path, cfg: V16_3Config, text: str) -> CorpusSplit:
    saved = json.loads(Path(path).read_text(encoding="utf-8"))
    current = build_corpus_split(cfg, text)
    if saved != current.to_dict():
        raise ValueError("saved corpus split does not match the current corpus/config")
    return current


@dataclass(frozen=True)
class V16_3Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    numbers: list[str]
    character_tokens: list[str]
    task_type: str = "target_character_set"
    loss_scope: str = "all_sequence"
    sequence_layout: str = "data_then_query"

    @classmethod
    def build(cls, cfg: V16_3Config, corpus_text: str) -> "V16_3Vocab":
        special = ["<PAD>", "<BOS>", "<EOS>", "<Think>", "</Think>", "<Ans>", "<CountChar>", "<Sep>"]
        characters = [character_token(char) for char in sorted(set(corpus_text), key=ord)]
        numbers = [f"<{index}>" for index in range(1, cfg.count_max_threshold + 1)]
        tokens = special + characters + numbers
        if len(tokens) != len(set(tokens)):
            raise ValueError("v16_3 vocabulary contains duplicate tokens")
        return cls(
            {token: index for index, token in enumerate(tokens)},
            tokens,
            numbers,
            characters,
            sequence_layout=cfg.sequence_layout,
        )

    def encode(self, tokens: Iterable[str]) -> list[int]:
        return [self.token_to_id[token] for token in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.id_to_token[int(index)] for index in ids]

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    @property
    def number_ids(self) -> list[int]:
        return [self.token_to_id[token] for token in self.numbers]

    @property
    def fingerprint(self) -> str:
        return _sha256_json(
            {
                "id_to_token": self.id_to_token,
                "task_type": self.task_type,
                "loss_scope": self.loss_scope,
                "sequence_layout": self.sequence_layout,
            }
        )

    def number_token(self, value: int) -> str:
        if not 1 <= int(value) <= len(self.numbers):
            raise ValueError(f"number must be in 1..{len(self.numbers)}")
        return self.numbers[int(value) - 1]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "token_to_id": self.token_to_id,
                    "id_to_token": self.id_to_token,
                    "numbers": self.numbers,
                    "character_tokens": self.character_tokens,
                    "task_type": self.task_type,
                    "loss_scope": self.loss_scope,
                    "sequence_layout": self.sequence_layout,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "V16_3Vocab":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            dict(obj["token_to_id"]),
            list(obj["id_to_token"]),
            list(obj["numbers"]),
            list(obj["character_tokens"]),
            str(obj.get("task_type", "target_character_set")),
            str(obj.get("loss_scope", "all_sequence")),
            str(obj.get("sequence_layout", "data_then_query")),
        )


@dataclass(frozen=True)
class V16_3Example:
    example_kind: str
    seq_tokens: list[str]
    corpus_region: str
    corpus_start: int
    corpus_end: int
    prompt_sha256: str
    seed: int | None = None
    set_id: str | None = None
    needle_characters: tuple[str, str, str] | None = None
    rendered_set_order: tuple[str, str, str] | None = None
    needle_positions: tuple[int, ...] = ()
    needle_markers: tuple[str, ...] = ()
    count: int | None = None
    set_frequency_sum: float | None = None
    set_frequency_bin: int | None = None
    per_character_counts: tuple[int, int, int] | None = None
    filter_attempts: int = 0
    rejected_zero: int = 0
    rejected_over_threshold: int = 0
    proposed_count: int | None = None


@dataclass(frozen=True)
class V16_3Spans:
    bos_pos: int
    prompt_start: int
    prompt_end_exclusive: int
    query_start: int
    query_end_exclusive: int
    think_pos: int | None
    trace_index_positions: tuple[int, ...]
    trace_marker_positions: tuple[int, ...]
    think_close_pos: int | None
    ans_pos: int
    count_pos: int
    eos_pos: int
    query_positions: tuple[int, ...]


@dataclass(frozen=True)
class V16_3Rendered:
    example_kind: str
    mode: str
    tokens: list[str]
    input_ids: list[int]
    labels: list[int]
    spans: V16_3Spans | None
    prompt_needle_positions: tuple[int, ...]
    count: int | None


def _window_start(region: CorpusRegion, seq_len: int, rng: random.Random) -> int:
    return rng.randrange(region.start, region.end - seq_len + 1)


def make_raw_example(
    cfg: V16_3Config,
    vocab: V16_3Vocab,
    text: str,
    split: CorpusSplit,
    region_name: str,
    rng: random.Random,
    *,
    start: int | None = None,
    seed: int | None = None,
) -> V16_3Example:
    region = split.region(region_name)
    selected_start = _window_start(region, cfg.seq_len, rng) if start is None else int(start)
    if selected_start < region.start or selected_start + cfg.seq_len > region.end:
        raise ValueError("raw window lies outside its declared corpus region")
    window = text[selected_start : selected_start + cfg.seq_len]
    return V16_3Example(
        example_kind="raw_lm",
        seq_tokens=[character_token(char) for char in window],
        corpus_region=region.name,
        corpus_start=selected_start,
        corpus_end=selected_start + cfg.seq_len,
        prompt_sha256=_sha256_text(window),
        seed=seed,
    )


def make_v16_3_example(
    cfg: V16_3Config,
    vocab: V16_3Vocab,
    text: str,
    split: CorpusSplit,
    pool: NeedlePool,
    rng: random.Random,
    *,
    region_name: str = "train",
    initial_start: int | None = None,
    needle_set: NeedleSet | None = None,
    seed: int | None = None,
) -> V16_3Example:
    region = split.region(region_name)
    selected_set = pool.sets[rng.randrange(len(pool.sets))] if needle_set is None else needle_set
    region_text = text[region.start : region.end]
    if not any(character in region_text for character in selected_set.characters):
        raise RuntimeError(
            f"{selected_set.set_id} has no occurrences in corpus region {region.name}; "
            "sample from that region's viable pool subset"
        )
    start = _window_start(region, cfg.seq_len, rng) if initial_start is None else int(initial_start)
    rejected_zero = 0
    rejected_over = 0
    proposed_count: int | None = None
    character_set = set(selected_set.characters)
    for attempt in range(1, cfg.candidate_filter_max_attempts + 1):
        if start < region.start or start + cfg.seq_len > region.end:
            raise ValueError("candidate window lies outside its declared corpus region")
        window = text[start : start + cfg.seq_len]
        positions = tuple(index for index, char in enumerate(window) if char in character_set)
        count = len(positions)
        if proposed_count is None:
            proposed_count = count
        if 1 <= count <= cfg.count_max_threshold:
            markers = tuple(character_token(window[position]) for position in positions)
            canonical = selected_set.characters
            order = tuple(rng.sample(list(canonical), len(canonical))) if cfg.shuffle_needle_set_order else canonical
            per_counts = tuple(window.count(character) for character in canonical)
            return V16_3Example(
                example_kind="counting_task",
                seq_tokens=[character_token(char) for char in window],
                corpus_region=region.name,
                corpus_start=start,
                corpus_end=start + cfg.seq_len,
                prompt_sha256=_sha256_text(window),
                seed=seed,
                set_id=selected_set.set_id,
                needle_characters=canonical,
                rendered_set_order=order,  # type: ignore[arg-type]
                needle_positions=positions,
                needle_markers=markers,
                count=count,
                set_frequency_sum=selected_set.frequency_sum,
                set_frequency_bin=selected_set.frequency_bin,
                per_character_counts=per_counts,  # type: ignore[arg-type]
                filter_attempts=attempt,
                rejected_zero=rejected_zero,
                rejected_over_threshold=rejected_over,
                proposed_count=proposed_count,
            )
        if count == 0:
            rejected_zero += 1
        else:
            rejected_over += 1
        start = _window_start(region, cfg.seq_len, rng)
    raise RuntimeError(
        f"failed to find a valid {region.name} window for {selected_set.set_id} after "
        f"{cfg.candidate_filter_max_attempts} attempts (zero={rejected_zero}, over={rejected_over})"
    )


def make_training_example(
    cfg: V16_3Config,
    vocab: V16_3Vocab,
    text: str,
    split: CorpusSplit,
    pool: NeedlePool,
    rng: random.Random,
) -> V16_3Example:
    initial_start = _window_start(split.train, cfg.seq_len, rng)
    if cfg.task_occurrence_ratio <= 0:
        return make_raw_example(cfg, vocab, text, split, "train", rng, start=initial_start)
    is_task = cfg.task_occurrence_ratio >= 1 or rng.random() < cfg.task_occurrence_ratio
    if not is_task:
        return make_raw_example(cfg, vocab, text, split, "train", rng, start=initial_start)
    return make_v16_3_example(
        cfg,
        vocab,
        text,
        split,
        pool,
        rng,
        region_name="train",
        initial_start=initial_start,
    )


def balanced_v16_3_examples(
    cfg: V16_3Config,
    vocab: V16_3Vocab,
    text: str,
    split: CorpusSplit,
    pool: NeedlePool,
    examples_per_count: int,
    seed: int,
    *,
    region_name: str,
) -> list[V16_3Example]:
    rng = random.Random(seed)
    region = split.region(region_name)
    region_text = text[region.start : region.end]
    viable_sets = tuple(
        item for item in pool.sets if any(character in region_text for character in item.characters)
    )
    if not viable_sets:
        raise RuntimeError(f"no needle-pool set has an occurrence in corpus region {region.name}")
    buckets: dict[int, list[V16_3Example]] = {
        count: [] for count in range(1, cfg.count_max_threshold + 1)
    }
    seen: set[tuple[str | None, int]] = set()
    accepted_draws = 0
    max_draws = cfg.candidate_filter_max_attempts
    while accepted_draws < max_draws and any(len(values) < examples_per_count for values in buckets.values()):
        accepted_draws += 1
        example = make_v16_3_example(
            cfg,
            vocab,
            text,
            split,
            pool,
            rng,
            region_name=region_name,
            needle_set=viable_sets[rng.randrange(len(viable_sets))],
            seed=seed * 1_000_000 + accepted_draws,
        )
        key = (example.set_id, example.corpus_start)
        if key in seen or len(buckets[int(example.count)]) >= examples_per_count:
            continue
        seen.add(key)
        buckets[int(example.count)].append(example)
    unfilled = {count: examples_per_count - len(values) for count, values in buckets.items() if len(values) < examples_per_count}
    if unfilled:
        raise RuntimeError(
            f"could not fill balanced {region_name} count buckets after {accepted_draws} draws: {unfilled}"
        )
    result = [example for count in sorted(buckets) for example in buckets[count]]
    rng.shuffle(result)
    return result


def raw_v16_3_examples(
    cfg: V16_3Config,
    vocab: V16_3Vocab,
    text: str,
    split: CorpusSplit,
    count: int,
    seed: int,
    *,
    region_name: str,
) -> list[V16_3Example]:
    rng = random.Random(seed)
    region = split.region(region_name)
    available = region.length - cfg.seq_len + 1
    if count > available:
        raise ValueError(f"requested {count} unique raw windows from only {available} starts")
    starts = rng.sample(range(region.start, region.end - cfg.seq_len + 1), count)
    return [
        make_raw_example(cfg, vocab, text, split, region_name, rng, start=start, seed=seed * 1_000_000 + index)
        for index, start in enumerate(starts)
    ]


def render_v16_3(example: V16_3Example, vocab: V16_3Vocab, mode: str) -> V16_3Rendered:
    if mode not in {"nonthinking", "thinking"}:
        raise ValueError(f"unknown mode: {mode}")
    if example.example_kind == "raw_lm":
        tokens = list(example.seq_tokens)
        ids = vocab.encode(tokens)
        return V16_3Rendered("raw_lm", mode, tokens, ids, list(ids), None, (), None)
    if (
        example.count is None
        or example.rendered_set_order is None
        or example.needle_characters is None
        or example.set_id is None
    ):
        raise ValueError("counting-task examples require complete set/count metadata")
    query_tokens = [
        "<CountChar>",
        *(character_token(character) for character in example.rendered_set_order),
        "<Sep>",
    ]
    prompt_start = 1
    prompt_end = prompt_start + len(example.seq_tokens)
    query_start = prompt_end
    query_end = query_start + len(query_tokens)
    prompt_needles = tuple(prompt_start + position for position in example.needle_positions)
    if mode == "nonthinking":
        tokens = [
            "<BOS>",
            *example.seq_tokens,
            *query_tokens,
            "<Ans>",
            vocab.number_token(example.count),
            "<EOS>",
        ]
        ans_pos = query_end
        count_pos = ans_pos + 1
        eos_pos = count_pos + 1
        spans = V16_3Spans(
            bos_pos=0,
            prompt_start=prompt_start,
            prompt_end_exclusive=prompt_end,
            query_start=query_start,
            query_end_exclusive=query_end,
            think_pos=None,
            trace_index_positions=(),
            trace_marker_positions=(),
            think_close_pos=None,
            ans_pos=ans_pos,
            count_pos=count_pos,
            eos_pos=eos_pos,
            query_positions=tuple(range(query_start, query_end)),
        )
    else:
        trace = [
            token
            for index, marker in enumerate(example.needle_markers, start=1)
            for token in (vocab.number_token(index), marker)
        ]
        think_pos = query_end
        trace_start = think_pos + 1
        trace_positions = tuple(range(trace_start, trace_start + len(trace)))
        index_positions = trace_positions[0::2]
        marker_positions = trace_positions[1::2]
        close_pos = trace_start + len(trace)
        ans_pos = close_pos + 1
        count_pos = ans_pos + 1
        eos_pos = count_pos + 1
        tokens = [
            "<BOS>",
            *example.seq_tokens,
            *query_tokens,
            "<Think>",
            *trace,
            "</Think>",
            "<Ans>",
            vocab.number_token(example.count),
            "<EOS>",
        ]
        spans = V16_3Spans(
            bos_pos=0,
            prompt_start=prompt_start,
            prompt_end_exclusive=prompt_end,
            query_start=query_start,
            query_end_exclusive=query_end,
            think_pos=think_pos,
            trace_index_positions=index_positions,
            trace_marker_positions=marker_positions,
            think_close_pos=close_pos,
            ans_pos=ans_pos,
            count_pos=count_pos,
            eos_pos=eos_pos,
            query_positions=tuple(range(query_start, query_end)),
        )
    ids = vocab.encode(tokens)
    return V16_3Rendered("counting_task", mode, tokens, ids, list(ids), spans, prompt_needles, example.count)


def render_v16_3_shortened_trace(
    example: V16_3Example, vocab: V16_3Vocab
) -> V16_3Rendered:
    """Render an analysis-only thinking trace with its final index/marker pair removed."""

    if example.example_kind != "counting_task" or example.count is None or example.count < 2:
        raise ValueError("shortened-trace analysis requires a counting example with count >= 2")
    gold = render_v16_3(example, vocab, "thinking")
    assert gold.spans is not None and gold.spans.think_pos is not None
    shortened_markers = example.needle_markers[:-1]
    trace = [
        token
        for index, marker in enumerate(shortened_markers, start=1)
        for token in (vocab.number_token(index), marker)
    ]
    think_pos = gold.spans.think_pos
    trace_start = think_pos + 1
    trace_positions = tuple(range(trace_start, trace_start + len(trace)))
    close_pos = trace_start + len(trace)
    ans_pos = close_pos + 1
    count_pos = ans_pos + 1
    eos_pos = count_pos + 1
    tokens = [
        *gold.tokens[: think_pos + 1],
        *trace,
        "</Think>",
        "<Ans>",
        vocab.number_token(example.count),
        "<EOS>",
    ]
    spans = V16_3Spans(
        bos_pos=gold.spans.bos_pos,
        prompt_start=gold.spans.prompt_start,
        prompt_end_exclusive=gold.spans.prompt_end_exclusive,
        query_start=gold.spans.query_start,
        query_end_exclusive=gold.spans.query_end_exclusive,
        think_pos=think_pos,
        trace_index_positions=trace_positions[0::2],
        trace_marker_positions=trace_positions[1::2],
        think_close_pos=close_pos,
        ans_pos=ans_pos,
        count_pos=count_pos,
        eos_pos=eos_pos,
        query_positions=gold.spans.query_positions,
    )
    ids = vocab.encode(tokens)
    return V16_3Rendered(
        "counting_task",
        "thinking",
        tokens,
        ids,
        list(ids),
        spans,
        gold.prompt_needle_positions,
        example.count,
    )


def collate_v16_3(
    rendered: list[V16_3Rendered],
    vocab: V16_3Vocab,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not rendered:
        raise ValueError("cannot collate an empty list")
    max_len = max(len(item.input_ids) for item in rendered)
    ids = torch.full((len(rendered), max_len), vocab.pad_id, dtype=torch.long)
    labels = torch.full((len(rendered), max_len), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros((len(rendered), max_len), dtype=torch.long)
    for row, item in enumerate(rendered):
        length = len(item.input_ids)
        ids[row, :length] = torch.tensor(item.input_ids)
        labels[row, :length] = torch.tensor(item.labels)
        mask[row, :length] = 1
    return ids.to(device), labels.to(device), mask.to(device)


def collate_v16_3_loss_weights(
    rendered: list[V16_3Rendered],
    cfg: V16_3Config,
    device: str | torch.device,
    *,
    step: int | None = None,
) -> torch.Tensor:
    """Return step-specific unshifted weights aligned with rendered target positions."""

    if not rendered:
        raise ValueError("cannot build loss weights for an empty list")
    if step is not None and (type(step) is not int or step < 0):
        raise ValueError("step must be a nonnegative integer or None")
    task_output_only = step is not None and step > cfg.max_steps_for_language_pred
    max_len = max(len(item.labels) for item in rendered)
    weights = torch.zeros((len(rendered), max_len), dtype=torch.float32)
    for row, item in enumerate(rendered):
        if task_output_only:
            if item.spans is None:
                continue
            if item.mode == "nonthinking":
                start = item.spans.ans_pos
            elif item.mode == "thinking":
                if item.spans.think_pos is None:
                    raise ValueError("thinking task output requires a <Think> position")
                start = item.spans.think_pos
            else:
                raise ValueError(f"unknown rendered mode: {item.mode}")
            weights[row, start : len(item.labels)] = 1.0
        else:
            weights[row, : len(item.labels)] = 1.0
        if item.spans is None:
            continue
        if weights[row, item.spans.count_pos] > 0:
            weights[row, item.spans.count_pos] = float(cfg.final_count_loss_weight)
        for position in (
            *item.spans.trace_index_positions,
            *item.spans.trace_marker_positions,
        ):
            if weights[row, position] > 0:
                weights[row, position] = float(cfg.cot_trace_loss_weight)
    return weights.to(device)


def shifted_v16_3_token_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shifted_logits = logits[:, :-1].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    losses = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    ).view_as(shifted_labels)
    active = shifted_labels.ne(IGNORE_INDEX)
    if loss_weights is None:
        shifted_weights = active.to(losses.dtype)
    else:
        if loss_weights.shape != labels.shape:
            raise ValueError("loss_weights must have the same shape as labels")
        shifted_weights = loss_weights[:, 1:].to(device=losses.device, dtype=losses.dtype)
        shifted_weights = shifted_weights * active
    total = (losses * shifted_weights).sum() / shifted_weights.sum().clamp_min(1)
    return total, losses, active


def component_target_positions(item: V16_3Rendered) -> dict[str, tuple[int, ...]]:
    if item.example_kind == "raw_lm":
        return {"raw_language": tuple(range(1, len(item.tokens)))}
    spans = item.spans
    assert spans is not None
    components: dict[str, tuple[int, ...]] = {
        "task_data": tuple(range(spans.prompt_start, spans.prompt_end_exclusive)),
        "task_query": spans.query_positions,
        "final_count": (spans.count_pos,),
        "eos": (spans.eos_pos,),
        "ans_token": (spans.ans_pos,),
    }
    if spans.think_pos is not None:
        components.update(
            {
                "think_open": (spans.think_pos,),
                "trace_index": spans.trace_index_positions,
                "trace_marker": spans.trace_marker_positions,
                "think_close": (spans.think_close_pos,) if spans.think_close_pos is not None else (),
            }
        )
    return components


def example_to_dict(example: V16_3Example) -> dict[str, Any]:
    value = asdict(example)
    return value


def example_from_dict(value: dict[str, Any]) -> V16_3Example:
    data = dict(value)
    for name in ("needle_characters", "rendered_set_order", "needle_positions", "needle_markers", "per_character_counts"):
        if data.get(name) is not None:
            data[name] = tuple(data[name])
    return V16_3Example(**data)


def _balanced_prefix(examples: list[V16_3Example], count: int, seed: int) -> list[V16_3Example]:
    by_count: dict[int, list[V16_3Example]] = {}
    for example in examples:
        by_count.setdefault(int(example.count), []).append(example)
    rng = random.Random(seed)
    for values in by_count.values():
        rng.shuffle(values)
    result: list[V16_3Example] = []
    labels = sorted(by_count)
    cursor = 0
    while len(result) < count:
        label = labels[cursor % len(labels)]
        values = by_count[label]
        index = cursor // len(labels)
        if index < len(values):
            result.append(values[index])
        cursor += 1
        if cursor > count * len(labels) * 2:
            raise RuntimeError("not enough task examples for balanced mixture selection")
    return result


def build_loss_suite_manifests(
    cfg: V16_3Config,
    vocab: V16_3Vocab,
    text: str,
    split: CorpusSplit,
    pool: NeedlePool,
) -> dict[str, dict[str, list[V16_3Example]]]:
    suite_size = cfg.eval_examples_per_count * cfg.count_max_threshold
    result: dict[str, dict[str, list[V16_3Example]]] = {}
    for source, region in (("train", "train"), ("heldout", "validation")):
        raw = raw_v16_3_examples(
            cfg, vocab, text, split, suite_size, cfg.seed + (60_000 if source == "train" else 70_000),
            region_name=region,
        )
        task = balanced_v16_3_examples(
            cfg, vocab, text, split, pool, cfg.eval_examples_per_count,
            cfg.seed + (61_000 if source == "train" else 71_000), region_name=region,
        )
        task_count = round(cfg.task_occurrence_ratio * suite_size)
        selected_task = _balanced_prefix(task, task_count, cfg.seed + 72_000) if task_count else []
        mixture = [*selected_task, *raw[: suite_size - task_count]]
        random.Random(cfg.seed + (62_000 if source == "train" else 72_000)).shuffle(mixture)
        result[source] = {"raw": raw, "task": task, "mixture": mixture}
    return result


def build_test_suite_manifests(
    cfg: V16_3Config,
    vocab: V16_3Vocab,
    text: str,
    split: CorpusSplit,
    pool: NeedlePool,
) -> dict[str, list[V16_3Example]]:
    suite_size = cfg.eval_examples_per_count * cfg.count_max_threshold
    raw = raw_v16_3_examples(cfg, vocab, text, split, suite_size, cfg.seed + 80_000, region_name="test")
    task = balanced_v16_3_examples(
        cfg, vocab, text, split, pool, cfg.eval_examples_per_count, cfg.seed + 81_000,
        region_name="test",
    )
    task_count = round(cfg.task_occurrence_ratio * suite_size)
    mixture = [*_balanced_prefix(task, task_count, cfg.seed + 82_000), *raw[: suite_size - task_count]]
    random.Random(cfg.seed + 83_000).shuffle(mixture)
    return {"raw": raw, "task": task, "mixture": mixture}


def save_suite_manifests(
    curve_suites: dict[str, dict[str, list[V16_3Example]]],
    test_suites: dict[str, list[V16_3Example]],
    path: str | Path,
    *,
    split_fingerprint: str,
    pool_fingerprint: str,
) -> None:
    obj = {
        "version": "v16_3",
        "split_fingerprint": split_fingerprint,
        "pool_fingerprint": pool_fingerprint,
        "curve_suites": {
            source: {suite: [example_to_dict(item) for item in examples] for suite, examples in suites.items()}
            for source, suites in curve_suites.items()
        },
        "test_suites": {
            suite: [example_to_dict(item) for item in examples] for suite, examples in test_suites.items()
        },
    }
    obj["manifest_fingerprint"] = _sha256_json(obj)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(path)


def load_suite_manifests(
    path: str | Path, *, split_fingerprint: str, pool_fingerprint: str
) -> tuple[dict[str, dict[str, list[V16_3Example]]], dict[str, list[V16_3Example]]]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    fingerprint = obj.pop("manifest_fingerprint")
    if _sha256_json(obj) != fingerprint:
        raise ValueError("loss-suite manifest fingerprint is invalid")
    if obj["split_fingerprint"] != split_fingerprint or obj["pool_fingerprint"] != pool_fingerprint:
        raise ValueError("loss-suite manifests do not match the current split/pool")
    curve = {
        source: {suite: [example_from_dict(item) for item in values] for suite, values in suites.items()}
        for source, suites in obj["curve_suites"].items()
    }
    test = {
        suite: [example_from_dict(item) for item in values] for suite, values in obj["test_suites"].items()
    }
    return curve, test
