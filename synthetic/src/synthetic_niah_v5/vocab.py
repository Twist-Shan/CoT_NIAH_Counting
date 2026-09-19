from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


SPECIAL_TOKENS = ["<BOS>", "<EOS>", "<THINK_ON>", "<THINK_OFF>", "<Think/>", "</Think>"]
NOISE_TOKENS = [f"<N{i}>" for i in range(64)]
MARKER_TOKENS = [f"<{chr(ord('A') + i)}>" for i in range(10)]
COUNT_TOKENS = [f"<C{i}>" for i in range(1, 11)]
TRACE_INDEX_TOKENS = [f"<I{i}>" for i in range(1, 11)]


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    include_trace_indices: bool = False

    @classmethod
    def build(cls, include_trace_indices: bool = False) -> "Vocab":
        tokens = SPECIAL_TOKENS + NOISE_TOKENS + MARKER_TOKENS + COUNT_TOKENS
        if include_trace_indices:
            tokens = tokens + TRACE_INDEX_TOKENS
        if len(tokens) != len(set(tokens)):
            raise ValueError("Vocabulary contains duplicate tokens.")
        return cls({tok: idx for idx, tok in enumerate(tokens)}, tokens, include_trace_indices)

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = cls(dict(obj["token_to_id"]), list(obj["id_to_token"]), bool(obj.get("include_trace_indices", False)))
        missing = [token for token in ("<THINK_ON>", "<THINK_OFF>") if token not in vocab.token_to_id]
        if missing:
            raise ValueError(
                "This is a legacy v5 vocabulary without explicit thinking-mode tokens. "
                "Retrain v5 with the current soft-switch format before loading it. "
                f"Missing tokens: {missing}"
            )
        return vocab

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "token_to_id": self.token_to_id,
                    "id_to_token": self.id_to_token,
                    "include_trace_indices": self.include_trace_indices,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id[token] for token in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token[int(idx)] for idx in ids]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    @property
    def pad_id(self) -> int:
        return self.eos_id

    @property
    def think_id(self) -> int:
        return self.token_to_id["<Think/>"]

    @property
    def think_on_id(self) -> int:
        return self.token_to_id["<THINK_ON>"]

    @property
    def think_off_id(self) -> int:
        return self.token_to_id["<THINK_OFF>"]

    @property
    def think_close_id(self) -> int:
        return self.token_to_id["</Think>"]

    @property
    def marker_ids(self) -> list[int]:
        return [self.token_to_id[token] for token in MARKER_TOKENS]

    @property
    def count_ids(self) -> list[int]:
        return [self.token_to_id[token] for token in COUNT_TOKENS]

    def count_id(self, value: int) -> int:
        return self.token_to_id[count_token(value)]

    def count_from_id(self, idx: int) -> int | None:
        return token_to_count(self.id_to_token[int(idx)])


def count_token(value: int) -> str:
    value = int(value)
    if not 1 <= value <= 10:
        raise ValueError(f"Count token must be in 1..10, got {value}.")
    return f"<C{value}>"


def index_token(value: int) -> str:
    value = int(value)
    if not 1 <= value <= 10:
        raise ValueError(f"Trace index token must be in 1..10, got {value}.")
    return f"<I{value}>"


def token_to_count(token: str) -> int | None:
    if token in COUNT_TOKENS:
        return int(token[2:-1])
    return None
