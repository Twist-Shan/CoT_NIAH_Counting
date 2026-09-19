from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import V20Config


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _display_character(character: str) -> str:
    return character.encode("unicode_escape").decode("ascii") or "<empty>"


@dataclass(frozen=True)
class NeedleSet:
    set_id: str
    characters: tuple[str, str, str]
    frequencies: tuple[float, float, float]
    frequency_sum: float
    frequency_bin: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "set_id": self.set_id,
            "characters": list(self.characters),
            "frequencies": list(self.frequencies),
            "frequency_sum": self.frequency_sum,
            "frequency_bin": self.frequency_bin,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NeedleSet":
        characters = tuple(value["characters"])
        frequencies = tuple(float(item) for item in value["frequencies"])
        if len(characters) != 3 or len(frequencies) != 3:
            raise ValueError("needle set artifacts must contain three members")
        return cls(
            str(value["set_id"]),
            characters,  # type: ignore[arg-type]
            frequencies,  # type: ignore[arg-type]
            float(value["frequency_sum"]),
            int(value["frequency_bin"]),
        )


@dataclass(frozen=True)
class NeedlePool:
    sets: tuple[NeedleSet, ...]
    character_counts: dict[str, int]
    character_frequencies: dict[str, float]
    pool_fingerprint: str
    corpus_sha256: str
    split_fingerprint: str
    vocab_fingerprint: str
    pool_seed: int
    threshold: float
    num_bins: int
    redistribution: tuple[dict[str, int], ...] = ()

    def __len__(self) -> int:
        return len(self.sets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "v20_v21_needle_pool_v1",
            "pool_fingerprint": self.pool_fingerprint,
            "corpus_sha256": self.corpus_sha256,
            "split_fingerprint": self.split_fingerprint,
            "vocab_fingerprint": self.vocab_fingerprint,
            "pool_seed": self.pool_seed,
            "threshold": self.threshold,
            "num_bins": self.num_bins,
            "character_counts": self.character_counts,
            "character_frequencies": self.character_frequencies,
            "redistribution": list(self.redistribution),
            "sets": [item.to_dict() for item in self.sets],
        }

    def validate(self, cfg: V20Config, *, split_fingerprint: str, vocab_fingerprint: str) -> None:
        if len(self.sets) != cfg.needle_pool_size:
            raise ValueError("needle pool size does not match config")
        if self.split_fingerprint != split_fingerprint:
            raise ValueError("needle pool split fingerprint does not match this run")
        if self.vocab_fingerprint != vocab_fingerprint:
            raise ValueError("needle pool vocabulary fingerprint does not match this run")
        canonical: set[tuple[str, ...]] = set()
        for item in self.sets:
            if len(item.characters) != 3 or len(set(item.characters)) != 3:
                raise ValueError(f"{item.set_id} does not contain three distinct characters")
            if tuple(sorted(item.characters, key=ord)) != item.characters:
                raise ValueError(f"{item.set_id} is not canonically ordered")
            if item.characters in canonical:
                raise ValueError("needle pool contains duplicate unordered sets")
            canonical.add(item.characters)
            expected = sum(self.character_frequencies[character] for character in item.characters)
            if abs(expected - item.frequency_sum) > 1e-12:
                raise ValueError(f"{item.set_id} has inconsistent frequencies")
            if item.frequency_sum > cfg.needle_pool_frequency_threshold + 1e-12:
                raise ValueError(f"{item.set_id} exceeds the pool frequency threshold")
        fingerprint = _pool_fingerprint(
            self.sets,
            self.corpus_sha256,
            self.split_fingerprint,
            self.vocab_fingerprint,
            self.pool_seed,
            self.threshold,
            self.num_bins,
        )
        if fingerprint != self.pool_fingerprint:
            raise ValueError("needle pool fingerprint is invalid")


def _pool_fingerprint(
    sets: Iterable[NeedleSet],
    corpus_sha256: str,
    split_fingerprint: str,
    vocab_fingerprint: str,
    pool_seed: int,
    threshold: float,
    num_bins: int,
) -> str:
    return _sha256_json(
        {
            "sets": [item.to_dict() for item in sets],
            "corpus_sha256": corpus_sha256,
            "split_fingerprint": split_fingerprint,
            "vocab_fingerprint": vocab_fingerprint,
            "pool_seed": pool_seed,
            "threshold": threshold,
            "num_bins": num_bins,
        }
    )


def build_needle_pool(
    cfg: V20Config,
    corpus_text: str,
    split: Any,
    vocab_fingerprint: str,
) -> NeedlePool:
    train_text = corpus_text[split.train.start : split.train.end]
    counts = {character: train_text.count(character) for character in sorted(set(train_text), key=ord)}
    frequencies = {character: count / len(train_text) for character, count in counts.items()}
    width = cfg.needle_pool_frequency_threshold / cfg.needle_pool_frequency_bins
    candidates: list[list[tuple[tuple[str, str, str], tuple[float, float, float], float]]] = [
        [] for _ in range(cfg.needle_pool_frequency_bins)
    ]
    for characters in itertools.combinations(sorted(frequencies, key=ord), cfg.needle_set_size):
        member_frequencies = tuple(frequencies[character] for character in characters)
        total = float(sum(member_frequencies))
        if total <= cfg.needle_pool_frequency_threshold + 1e-15:
            bin_index = min(cfg.needle_pool_frequency_bins - 1, int(total / width))
            candidates[bin_index].append((characters, member_frequencies, total))
    if sum(map(len, candidates)) < cfg.needle_pool_size:
        raise ValueError(
            f"only {sum(map(len, candidates))} distinct triples satisfy the pool threshold; "
            f"need {cfg.needle_pool_size}"
        )

    rng = random.Random(cfg.effective_needle_pool_seed)
    for bin_candidates in candidates:
        rng.shuffle(bin_candidates)
    base, remainder = divmod(cfg.needle_pool_size, cfg.needle_pool_frequency_bins)
    quotas = [base + int(index < remainder) for index in range(cfg.needle_pool_frequency_bins)]
    selected_by_bin: list[list[tuple[tuple[str, str, str], tuple[float, float, float], float]]] = []
    deficits: list[tuple[int, int]] = []
    for bin_index, (bin_candidates, quota) in enumerate(zip(candidates, quotas)):
        take = min(quota, len(bin_candidates))
        selected_by_bin.append(bin_candidates[:take])
        del bin_candidates[:take]
        if take < quota:
            deficits.append((bin_index, quota - take))

    redistribution: list[dict[str, int]] = []
    for requested_bin, deficit in deficits:
        for _ in range(deficit):
            donor_bins = [index for index, values in enumerate(candidates) if values]
            if not donor_bins:
                raise RuntimeError("needle pool quota redistribution exhausted all candidates")
            distance = min(abs(index - requested_bin) for index in donor_bins)
            nearest = [index for index in donor_bins if abs(index - requested_bin) == distance]
            donor = nearest[rng.randrange(len(nearest))]
            selected_by_bin[donor].append(candidates[donor].pop())
            redistribution.append({"requested_bin": requested_bin, "donor_bin": donor, "count": 1})

    raw_selected = [
        (bin_index, candidate)
        for bin_index, values in enumerate(selected_by_bin)
        for candidate in values
    ]
    raw_selected.sort(key=lambda value: (value[0], value[1][2], value[1][0]))
    sets = tuple(
        NeedleSet(
            set_id=f"set_{index:03d}",
            characters=candidate[0],
            frequencies=candidate[1],
            frequency_sum=candidate[2],
            frequency_bin=bin_index,
        )
        for index, (bin_index, candidate) in enumerate(raw_selected)
    )
    fingerprint = _pool_fingerprint(
        sets,
        split.corpus_sha256,
        split.split_fingerprint,
        vocab_fingerprint,
        cfg.effective_needle_pool_seed,
        cfg.needle_pool_frequency_threshold,
        cfg.needle_pool_frequency_bins,
    )
    pool = NeedlePool(
        sets=sets,
        character_counts=counts,
        character_frequencies=frequencies,
        pool_fingerprint=fingerprint,
        corpus_sha256=split.corpus_sha256,
        split_fingerprint=split.split_fingerprint,
        vocab_fingerprint=vocab_fingerprint,
        pool_seed=cfg.effective_needle_pool_seed,
        threshold=cfg.needle_pool_frequency_threshold,
        num_bins=cfg.needle_pool_frequency_bins,
        redistribution=tuple(redistribution),
    )
    pool.validate(cfg, split_fingerprint=split.split_fingerprint, vocab_fingerprint=vocab_fingerprint)
    return pool


def save_needle_pool(pool: NeedlePool, run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    path = run_dir / "data" / "needle_pool.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(pool.to_dict(), indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(path)
    rows = []
    for item in pool.sets:
        row: dict[str, Any] = {
            "set_id": item.set_id,
            "frequency_sum": item.frequency_sum,
            "frequency_bin": item.frequency_bin,
        }
        for index, (character, frequency) in enumerate(zip(item.characters, item.frequencies), start=1):
            row[f"character_{index}"] = _display_character(character)
            row[f"codepoint_{index}"] = ord(character)
            row[f"frequency_{index}"] = frequency
        rows.append(row)
    pd.DataFrame(rows).to_csv(run_dir / "tables" / "needle_pool.csv", index=False)


def load_needle_pool(
    path: str | Path,
    cfg: V20Config,
    *,
    split_fingerprint: str,
    vocab_fingerprint: str,
) -> NeedlePool:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    pool = NeedlePool(
        sets=tuple(NeedleSet.from_dict(item) for item in obj["sets"]),
        character_counts={str(key): int(value) for key, value in obj["character_counts"].items()},
        character_frequencies={
            str(key): float(value) for key, value in obj["character_frequencies"].items()
        },
        pool_fingerprint=str(obj["pool_fingerprint"]),
        corpus_sha256=str(obj["corpus_sha256"]),
        split_fingerprint=str(obj["split_fingerprint"]),
        vocab_fingerprint=str(obj["vocab_fingerprint"]),
        pool_seed=int(obj["pool_seed"]),
        threshold=float(obj["threshold"]),
        num_bins=int(obj["num_bins"]),
        redistribution=tuple(dict(item) for item in obj.get("redistribution", [])),
    )
    pool.validate(cfg, split_fingerprint=split_fingerprint, vocab_fingerprint=vocab_fingerprint)
    return pool


def plot_needle_pool(pool: NeedlePool, output_path: str | Path) -> None:
    values = [item.frequency_sum for item in pool.sets]
    figure, axis = plt.subplots(figsize=(8.5, 4.4))
    axis.hist(values, bins=pool.num_bins, range=(0, pool.threshold), color="#2f6fed", edgecolor="white")
    axis.set_xlabel("sum of training-region character frequencies")
    axis.set_ylabel("number of needle sets")
    axis.set_title("v20/v21 shared needle-pool frequency distribution")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
