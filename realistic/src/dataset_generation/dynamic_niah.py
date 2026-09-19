from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DynamicNiahConfig:
    tokenizer_name: str
    max_haystack_tokens: int
    base_seed: int
    haystack_seed: int
    needle_global_seed: int
    insertion_positions: list[int]
    num_needles: int
    task_type: str = "argmax"
    haystack_dir: str = "data/haystacks/paul_graham"
    entities_path: str = "data/entities/cities.csv"
    fact_templates_path: str = "data/templates/niah_fact_templates.txt"
    temp_output_dir: str = "generated/dynamic_niah_debug"
    needle_seeds: dict[int, int] | None = None
    year: int = 2024


class SimpleTokenizer:
    """Fallback tokenizer used when external tokenizers are unavailable.

    Tokenization is whitespace-based to keep behavior deterministic in tests.
    """

    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


class TokenizerAdapter:
    def __init__(
        self,
        tokenizer_name: str,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
    ):
        self.tokenizer_name = tokenizer_name
        self.revision = revision
        self.backend = "simple"
        self.load_error: str | None = None
        self._tok: Any = SimpleTokenizer()

        if tokenizer_name != "simple":
            try:
                from transformers import AutoTokenizer

                self._tok = AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    revision=revision,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                )
                self.backend = "huggingface"
            except Exception as exc:
                # Keep fallback deterministic behavior.
                self.load_error = f"{type(exc).__name__}: {exc}"
                self._tok = SimpleTokenizer()
                self.backend = "simple"

    def encode(self, text: str) -> list[Any]:
        if self.backend == "huggingface":
            return list(self._tok.encode(text, add_special_tokens=False))
        return self._tok.encode(text)

    def encode_with_offsets(
        self, text: str
    ) -> tuple[list[Any], list[tuple[int, int]]]:
        """Encode text and return exact character offsets for every token."""

        if self.backend == "huggingface":
            encoded = self._tok(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            if "offset_mapping" not in encoded:
                raise RuntimeError(
                    f"Tokenizer {self.tokenizer_name!r} did not return offset mappings"
                )
            token_ids = list(encoded["input_ids"])
            offsets = [
                (int(start), int(end))
                for start, end in encoded["offset_mapping"]
            ]
            if len(token_ids) != len(offsets):
                raise RuntimeError(
                    "Tokenizer returned different input-id and offset-map lengths"
                )
            return token_ids, offsets

        matches = list(re.finditer(r"\S+", text))
        return (
            [match.group(0) for match in matches],
            [(match.start(), match.end()) for match in matches],
        )

    def decode(self, tokens: list[Any]) -> str:
        if self.backend == "huggingface":
            return self._tok.decode(tokens)
        return self._tok.decode(tokens)


@dataclass
class NeedleRecord:
    needle_id: str
    city: str
    score: int
    text: str
    tokens: list[Any]


@dataclass
class DynamicNiahInstance:
    tokenizer_name: str
    tokenizer_backend: str
    haystack_source_file: str
    haystack_window_start: int
    haystack_window_end: int
    base_haystack_tokens: list[Any]
    final_sequence_tokens: list[Any]
    decoded_text: str
    insertion_positions_original: list[int]
    realized_insertions: list[dict[str, Any]]
    relevant_records: list[dict[str, Any]]
    query: str
    gold_answer: dict[str, Any]
    seeds: dict[str, Any]



def _load_haystack_files(haystack_dir: str) -> list[Path]:
    files = sorted(Path(haystack_dir).glob("*.txt"))
    if not files:
        raise ValueError(f"No haystack text files found in {haystack_dir}")
    return files


def _choose_haystack_window(cfg: DynamicNiahConfig, tok: TokenizerAdapter) -> tuple[list[Any], str, int, int]:
    rng = random.Random(cfg.base_seed * 1009 + cfg.haystack_seed)
    files = _load_haystack_files(cfg.haystack_dir)
    chosen = rng.choice(files)
    raw_text = chosen.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", raw_text).strip()
    all_tokens = tok.encode(normalized)
    window_len = min(cfg.max_haystack_tokens, len(all_tokens))
    if window_len <= 0:
        raise ValueError("Haystack token length must be > 0")
    start = 0 if len(all_tokens) == window_len else rng.randint(0, len(all_tokens) - window_len)
    end = start + window_len
    return list(all_tokens[start:end]), chosen.name, start, end


def _read_cities(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_fact_templates(path: str) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _needle_seed(cfg: DynamicNiahConfig, idx: int) -> int:
    if cfg.needle_seeds and idx in cfg.needle_seeds:
        return cfg.needle_seeds[idx]
    return cfg.needle_global_seed + idx * 7919


def _generate_needles(cfg: DynamicNiahConfig, tok: TokenizerAdapter) -> list[NeedleRecord]:
    if cfg.task_type != "argmax":
        raise ValueError(f"Unsupported task_type for v1 dynamic flow: {cfg.task_type}")

    cities = _read_cities(cfg.entities_path)
    fact_templates = _read_fact_templates(cfg.fact_templates_path)
    if not fact_templates:
        raise ValueError("fact templates file must contain at least one non-empty template line")
    if cfg.num_needles > len(cities):
        raise ValueError("num_needles exceeds available entities")

    picked: list[NeedleRecord] = []
    for i in range(cfg.num_needles):
        rng = random.Random(cfg.base_seed * 3253 + _needle_seed(cfg, i))
        row = cities[rng.randrange(len(cities))]
        score = 50 + (rng.randrange(51))
        tpl = fact_templates[i % len(fact_templates)]
        text = tpl.format(
            year=cfg.year,
            entity=row["entity"],
            region=row.get("region", "unknown"),
            category=row.get("category", "unknown"),
            score=score,
        )
        picked.append(NeedleRecord(needle_id=f"N{i+1}", city=row["entity"], score=score, text=text, tokens=tok.encode(text)))
    return picked


def _insert_tokens(base_tokens: list[Any], needles: list[NeedleRecord], positions_original: list[int]) -> tuple[list[Any], list[dict[str, Any]]]:
    if len(needles) != len(positions_original):
        raise ValueError("Number of insertion_positions must equal num_needles")

    base_len = len(base_tokens)
    for pos in positions_original:
        if pos < 0 or pos > base_len:
            raise ValueError(f"Insertion position {pos} out of range for base haystack length {base_len}")

    # Positions are defined on original haystack token index. We therefore sort by
    # original index and account for right-shift as we insert.
    combined = sorted(zip(positions_original, needles), key=lambda x: x[0])
    output = list(base_tokens)
    realized: list[dict[str, Any]] = []
    shift = 0
    for original_pos, needle in combined:
        final_pos = original_pos + shift
        output[final_pos:final_pos] = needle.tokens
        realized.append(
            {
                "needle_id": needle.needle_id,
                "start_position_original": original_pos,
                "start_position_final": final_pos,
                "token_length": len(needle.tokens),
                "needle_text": needle.text,
                "needle_tokens": needle.tokens,
            }
        )
        shift += len(needle.tokens)
    return output, realized


def generate_dynamic_niah_instance(cfg: DynamicNiahConfig) -> DynamicNiahInstance:
    tok = TokenizerAdapter(cfg.tokenizer_name)
    base_tokens, source_name, window_start, window_end = _choose_haystack_window(cfg, tok)
    needles = _generate_needles(cfg, tok)
    final_tokens, realized = _insert_tokens(base_tokens, needles, cfg.insertion_positions)

    relevant_records = [{"needle_id": n.needle_id, "city": n.city, "score": n.score} for n in needles]
    winner = max(relevant_records, key=lambda r: (r["score"], r["city"]))
    query = "Which city has the highest score? Respond with city and score."

    return DynamicNiahInstance(
        tokenizer_name=cfg.tokenizer_name,
        tokenizer_backend=tok.backend,
        haystack_source_file=source_name,
        haystack_window_start=window_start,
        haystack_window_end=window_end,
        base_haystack_tokens=base_tokens,
        final_sequence_tokens=final_tokens,
        decoded_text=tok.decode(final_tokens),
        insertion_positions_original=cfg.insertion_positions,
        realized_insertions=realized,
        relevant_records=relevant_records,
        query=query,
        gold_answer={"city": winner["city"], "score": winner["score"]},
        seeds={
            "base_seed": cfg.base_seed,
            "haystack_seed": cfg.haystack_seed,
            "needle_global_seed": cfg.needle_global_seed,
            "needle_seeds": cfg.needle_seeds or {},
        },
    )


def write_dynamic_artifacts(instance: DynamicNiahInstance, cfg: DynamicNiahConfig) -> dict[str, str]:
    out_dir = Path(cfg.temp_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inst_path = out_dir / "instance.json"
    inst_path.write_text(json.dumps(asdict(instance), indent=2, ensure_ascii=False), encoding="utf-8")

    text_path = out_dir / "instance.txt"
    text_path.write_text(instance.decoded_text, encoding="utf-8")

    cfg_path = out_dir / "config.used.json"
    cfg_path.write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")

    return {"instance_json": str(inst_path), "instance_text": str(text_path), "config_used": str(cfg_path)}
