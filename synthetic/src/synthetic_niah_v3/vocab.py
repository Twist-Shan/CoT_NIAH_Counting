from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<Ans>", "<Think/>", "</Think>"]
NOISE_TOKENS = [f"<N{i}>" for i in range(64)]
MARKER_TOKENS = [f"<{chr(ord('A') + i)}>" for i in range(10)]
NUMBER_TOKENS = [f"<{i}>" for i in range(1, 11)]


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]

    @classmethod
    def build(cls) -> "Vocab":
        tokens = SPECIAL_TOKENS + NOISE_TOKENS + MARKER_TOKENS + NUMBER_TOKENS
        if len(tokens) != 90 or len(set(tokens)) != 90:
            raise ValueError(f"Expected exactly 90 unique tokens, got {len(set(tokens))}.")
        return cls({tok: idx for idx, tok in enumerate(tokens)}, tokens)

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(dict(obj["token_to_id"]), list(obj["id_to_token"]))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"token_to_id": self.token_to_id, "id_to_token": self.id_to_token}, indent=2),
            encoding="utf-8",
        )

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id[token] for token in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token[int(idx)] for idx in ids]

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    @property
    def ans_id(self) -> int:
        return self.token_to_id["<Ans>"]

    @property
    def think_id(self) -> int:
        return self.token_to_id["<Think/>"]

    @property
    def think_close_id(self) -> int:
        return self.token_to_id["</Think>"]

    @property
    def number_ids(self) -> list[int]:
        return [self.token_to_id[token] for token in NUMBER_TOKENS]

    def number_id(self, value: int) -> int:
        return self.token_to_id[f"<{value}>"]

    def number_from_id(self, idx: int) -> int | None:
        tok = self.id_to_token[int(idx)]
        if tok in NUMBER_TOKENS:
            return int(tok.strip("<>"))
        return None


def number_token(value: int) -> str:
    if not 1 <= int(value) <= 10:
        raise ValueError(f"Count token must be in 1..10, got {value}.")
    return f"<{int(value)}>"
