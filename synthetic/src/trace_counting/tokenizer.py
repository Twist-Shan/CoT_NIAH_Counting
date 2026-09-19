from __future__ import annotations

from dataclasses import dataclass, field
import re
from pathlib import Path

from .io_utils import load_json, save_json

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<Think>", "<ANS>", "<CNT>", "<TICK>"]
DEFAULT_POSITIVE_VOCAB = ["X", "Y", "Z"]


@dataclass
class VocabTokenizer:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.id_to_token) != len(self.token_to_id):
            raise ValueError("id_to_token and token_to_id sizes do not match.")
        for idx, token in enumerate(self.id_to_token):
            if self.token_to_id.get(token) != idx:
                raise ValueError(f"Vocabulary mismatch at id {idx}: {token!r}")

    def __len__(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id["<BOS>"]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id["<EOS>"]

    @property
    def count_token_ids(self) -> list[int]:
        max_count = int(self.metadata.get("max_count", self._infer_max_count()))
        return [self.token_to_id[f"<C{k}>"] for k in range(max_count + 1)]

    @property
    def count_tokens(self) -> list[str]:
        max_count = int(self.metadata.get("max_count", self._infer_max_count()))
        return [f"<C{k}>" for k in range(max_count + 1)]

    @property
    def count_unit_token_id(self) -> int:
        return self.token_to_id["<CNT>"]

    @property
    def trace_tick_token_id(self) -> int:
        return self.token_to_id["<TICK>"]

    def encode(self, tokens: list[str]) -> list[int]:
        try:
            return [self.token_to_id[token] for token in tokens]
        except KeyError as exc:
            raise KeyError(f"Unknown token {exc.args[0]!r}") from exc

    def decode(self, ids: list[int], *, skip_pad: bool = False) -> list[str]:
        tokens = [self.id_to_token[int(idx)] for idx in ids]
        if skip_pad:
            tokens = [token for token in tokens if token != "<PAD>"]
        return tokens

    def save(self, path: str | Path) -> None:
        save_json(
            {
                "token_to_id": self.token_to_id,
                "id_to_token": self.id_to_token,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "VocabTokenizer":
        obj = load_json(path)
        return cls(
            token_to_id={str(k): int(v) for k, v in obj["token_to_id"].items()},
            id_to_token=list(obj["id_to_token"]),
            metadata=dict(obj.get("metadata", {})),
        )

    def _infer_max_count(self) -> int:
        values = []
        for token in self.id_to_token:
            match = re.fullmatch(r"<C(\d+)>", token)
            if match:
                values.append(int(match.group(1)))
        if not values:
            raise ValueError("No count tokens found in vocabulary.")
        return max(values)


def build_default_tokenizer(
    *,
    max_count: int = 64,
    noise_vocab_size: int = 64,
    positive_vocab: list[str] | tuple[str, ...] = tuple(DEFAULT_POSITIVE_VOCAB),
) -> VocabTokenizer:
    tokens: list[str] = []
    tokens.extend(SPECIAL_TOKENS)
    tokens.extend(list(positive_vocab))
    tokens.extend([f"N{i}" for i in range(noise_vocab_size)])
    tokens.extend([f"<I{i}>" for i in range(1, max_count + 1)])
    tokens.extend([f"<C{i}>" for i in range(max_count + 1)])
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return VocabTokenizer(
        token_to_id=token_to_id,
        id_to_token=tokens,
        metadata={
            "max_count": max_count,
            "noise_vocab_size": noise_vocab_size,
            "positive_vocab": list(positive_vocab),
        },
    )


def token_is_count(token: str) -> bool:
    return re.fullmatch(r"<C\d+>", token) is not None


def parse_count_token(token: str) -> int | None:
    match = re.fullmatch(r"<C(\d+)>", token)
    return int(match.group(1)) if match else None
