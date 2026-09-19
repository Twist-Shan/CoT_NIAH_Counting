from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from synthetic_niah_v5.model import make_model

from .config import V10Config


IGNORE_INDEX = -100


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    numbers: list[str]
    markers: list[str]
    noise: list[str]

    @classmethod
    def build(cls, cfg: V10Config) -> "Vocab":
        special = ["<PAD>", "<BOS>", "<EOS>", "<Think>", "</Think>", "<Ans>"]
        noise = [f"<N{i}>" for i in range(cfg.noise_vocab_size)]
        markers = [f"<M{i}>" for i in range(cfg.marker_vocab_size)]
        numbers = [f"<{i}>" for i in range(1, cfg.count_max + 1)]
        tokens = special + noise + markers + numbers
        if len(tokens) != len(set(tokens)):
            raise ValueError("v10 vocabulary has duplicate tokens")
        return cls({token: idx for idx, token in enumerate(tokens)}, tokens, numbers, markers, noise)

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            dict(obj["token_to_id"]),
            list(obj["id_to_token"]),
            list(obj["numbers"]),
            list(obj["markers"]),
            list(obj["noise"]),
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
    def think_id(self) -> int:
        return self.token_to_id["<Think>"]

    @property
    def think_close_id(self) -> int:
        return self.token_to_id["</Think>"]

    @property
    def ans_id(self) -> int:
        return self.token_to_id["<Ans>"]

    @property
    def number_ids(self) -> list[int]:
        return [self.token_to_id[token] for token in self.numbers]

    @property
    def marker_ids(self) -> list[int]:
        return [self.token_to_id[token] for token in self.markers]

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


@dataclass(frozen=True)
class Rendered:
    mode: str
    tokens: list[str]
    input_ids: list[int]
    labels: list[int]
    spans: Spans
    prompt_needle_positions: list[int]
    count: int


def make_example(
    cfg: V10Config,
    vocab: Vocab,
    rng: random.Random,
    count: int | None = None,
    seed: int | None = None,
) -> Example:
    n = rng.randint(cfg.count_min, cfg.count_max) if count is None else int(count)
    if not cfg.count_min <= n <= cfg.count_max:
        raise ValueError(f"count must be in {cfg.count_min}..{cfg.count_max}")
    positions = sorted(rng.sample(range(cfg.seq_len), n))
    markers = [rng.choice(vocab.markers) for _ in positions]
    seq = [rng.choice(vocab.noise) for _ in range(cfg.seq_len)]
    for pos, marker in zip(positions, markers):
        seq[pos] = marker
    return Example(seq, n, positions, markers, seed)


def balanced_examples(
    cfg: V10Config,
    vocab: Vocab,
    examples_per_count: int,
    seed: int,
    *,
    count_min: int | None = None,
    count_max: int | None = None,
) -> list[Example]:
    lo = cfg.count_min if count_min is None else int(count_min)
    hi = cfg.count_max if count_max is None else int(count_max)
    rng = random.Random(seed)
    examples: list[Example] = []
    for count in range(lo, hi + 1):
        for index in range(int(examples_per_count)):
            ex_seed = seed * 1_000_000 + count * 10_000 + index
            examples.append(make_example(cfg, vocab, rng, count=count, seed=ex_seed))
    rng.shuffle(examples)
    return examples


def count_bin(count: int) -> str:
    count = int(count)
    if count <= 10:
        return "1-10"
    if count <= 20:
        return "11-20"
    return "21-30"


def render(example: Example, vocab: Vocab, mode: str) -> Rendered:
    prompt_start = 1
    prompt_end = prompt_start + len(example.seq_tokens)
    needle_positions = [prompt_start + pos for pos in example.needle_positions]
    if mode == "nonthinking":
        tokens = ["<BOS>", *example.seq_tokens, "<Ans>", vocab.number_token(example.count), "<EOS>"]
        ans_pos = prompt_end
        count_pos = ans_pos + 1
        eos_pos = count_pos + 1
        labels = [IGNORE_INDEX] * len(tokens)
        labels[count_pos] = vocab.number_id(example.count)
        labels[eos_pos] = vocab.eos_id
        spans = Spans(0, prompt_start, prompt_end, None, [], [], None, ans_pos, count_pos, eos_pos)
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
        )
    else:
        raise ValueError(f"Unknown v10 mode: {mode}")
    return Rendered(mode, tokens, vocab.encode(tokens), labels, spans, needle_positions, example.count)


def component_target_positions(rendered: Rendered) -> dict[str, list[int]]:
    if rendered.mode == "nonthinking":
        return {
            "final_count": [rendered.spans.count_pos],
            "eos": [rendered.spans.eos_pos],
        }
    return {
        "trace_index": list(rendered.spans.trace_index_positions),
        "trace_marker": list(rendered.spans.trace_marker_positions),
        "think_close": [rendered.spans.think_close_pos] if rendered.spans.think_close_pos is not None else [],
        "ans_token": [rendered.spans.ans_pos],
        "final_count": [rendered.spans.count_pos],
        "eos": [rendered.spans.eos_pos],
    }


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


def model_config(cfg: V10Config, vocab: Vocab) -> dict[str, Any]:
    return {
        "vocab_size": len(vocab.id_to_token),
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_embd": cfg.n_embd,
        "n_inner": cfg.n_inner,
        "n_positions": cfg.n_positions,
        "n_ctx": cfg.n_positions,
        "activation_function": "gelu_new",
        "resid_pdrop": 0.0,
        "embd_pdrop": 0.0,
        "attn_pdrop": 0.0,
        "use_cache": False,
        "bos_token_id": vocab.bos_id,
        "eos_token_id": vocab.eos_id,
        "pad_token_id": vocab.pad_id,
    }


def build_model(cfg: V10Config, vocab: Vocab, device: str | torch.device | None = None):
    return make_model(model_config(cfg, vocab), device or cfg.device)


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


def component_loss_values(losses: torch.Tensor, rendered: list[Rendered]) -> dict[str, float]:
    values: dict[str, list[torch.Tensor]] = {}
    for row, item in enumerate(rendered):
        for name, target_positions in component_target_positions(item).items():
            for target_pos in target_positions:
                ce_pos = int(target_pos) - 1
                if 0 <= ce_pos < losses.shape[1]:
                    values.setdefault(name, []).append(losses[row, ce_pos])
    return {
        name: float(torch.stack(items).mean().detach().cpu())
        for name, items in values.items()
        if items
    }


def learning_rate(step: int, cfg: V10Config) -> float:
    if step <= cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.train_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def margin(logits: torch.Tensor, target_id: int, competitor_ids: Iterable[int]) -> float:
    ids = [int(idx) for idx in competitor_ids if int(idx) != int(target_id)]
    if not ids:
        return math.nan
    values = logits.detach().float().cpu()
    return float(values[int(target_id)] - values[ids].max())


def count_prediction(logits: torch.Tensor, vocab: Vocab) -> tuple[int, float, np.ndarray]:
    subset = logits.detach().float().cpu()[vocab.number_ids]
    probabilities = torch.softmax(subset, dim=-1).numpy()
    pred = int(np.argmax(probabilities)) + 1
    expected = float(probabilities @ np.arange(1, len(probabilities) + 1, dtype=float))
    return pred, expected, probabilities


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def torch_load(path: str | Path, device: str | torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)
