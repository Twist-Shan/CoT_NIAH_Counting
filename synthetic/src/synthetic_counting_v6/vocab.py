from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<Ans>", "<Think/>", "</Think>", "<Sep>"]
NOISE_TOKENS = [f"<N{i}>" for i in range(64)]
MARKER_TOKENS = [f"<{chr(ord('A') + i)}>" for i in range(10)]
NUMBER_TOKENS = [f"<{i}>" for i in range(1, 11)]
ALL_TOKENS = SPECIAL_TOKENS + NOISE_TOKENS + MARKER_TOKENS + NUMBER_TOKENS


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]

    @classmethod
    def build(cls) -> "Vocab":
        if len(ALL_TOKENS) != 91 or len(set(ALL_TOKENS)) != 91:
            raise ValueError("v6 vocabulary must contain exactly 91 unique tokens.")
        return cls({token: idx for idx, token in enumerate(ALL_TOKENS)}, list(ALL_TOKENS))

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(dict(obj["token_to_id"]), list(obj["id_to_token"]))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps({"token_to_id": self.token_to_id, "id_to_token": self.id_to_token}, indent=2),
            encoding="utf-8",
        )

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id[token] for token in tokens]

    def decode(self, ids: list[int], skip_pad: bool = False) -> list[str]:
        tokens = [self.id_to_token[int(idx)] for idx in ids]
        return [token for token in tokens if token != "<PAD>"] if skip_pad else tokens

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
    def think_id(self) -> int:
        return self.token_to_id["<Think/>"]

    @property
    def think_close_id(self) -> int:
        return self.token_to_id["</Think>"]

    @property
    def sep_id(self) -> int:
        return self.token_to_id["<Sep>"]

    @property
    def marker_ids(self) -> list[int]:
        return [self.token_to_id[token] for token in MARKER_TOKENS]

    @property
    def noise_ids(self) -> list[int]:
        return [self.token_to_id[token] for token in NOISE_TOKENS]

    @property
    def count_ids(self) -> list[int]:
        return [self.token_to_id[token] for token in NUMBER_TOKENS]

    def count_id(self, value: int) -> int:
        return self.token_to_id[count_token(value)]

    def count_from_id(self, idx: int) -> int | None:
        token = self.id_to_token[int(idx)]
        if token.startswith("<") and token.endswith(">") and token[1:-1].isdigit():
            value = int(token[1:-1])
            if 1 <= value <= 10:
                return value
        return None


def count_token(value: int) -> str:
    if not 1 <= int(value) <= 10:
        raise ValueError(f"count token must be in 1..10, got {value}.")
    return f"<{int(value)}>"

