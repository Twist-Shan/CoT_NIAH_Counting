from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import shutil
import warnings
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

from dataset_generation.dynamic_niah import TokenizerAdapter
from dataset_generation.niah_prompt_utils import (
    build_messages_easier,
    build_messages_vanilla,
    response_schema_for_task,
)
from dataset_generation.response_eval import (
    build_control_gold_answer,
    build_gold_answer,
    build_task_query,
    canonical_task_type,
)

DATASET_SCHEMA_VERSION = "dynamic_niah_v2_dataset_v1"
DATASET_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "dataset.schema.json"


@dataclass(frozen=True)
class DynamicNiahV2Config:
    task_type: str = "argmax"
    tokenizer_name: str = "Qwen/Qwen3-8B"
    num_examples: int = 100
    target_haystack_tokens: int = 1000
    num_needles: int = 3
    num_max_needles: int | None = None
    insertion_positions: tuple[int | None, ...] = (100, 200, 400)
    randomize_needle_insertion: bool = False
    randomize_needle_seed: int = 42
    randomize_needle_margin: int = 50
    randomize_needle_min_separation: int = 50
    sentence_level_insertion: bool = False
    word_level_insertion: bool = False
    text_insertion_min_fraction: float = 0.0
    text_insertion_max_fraction: float = 1.0
    prompt_style: str = "easier"
    thinking_mode: bool = False
    output_dir: str | None = None
    data_save_path: str | None = None
    output_pred_jsonl: str | None = None
    output_metrics_json: str | None = None
    results_root: str = "results"
    run_dir: str | None = None
    run_name: str | None = None
    global_random_seed: int = 42
    haystack_dir: str = "data/haystacks/paul_graham"
    haystack_source_mode: str = "single_file_repeat"
    entities_path: str = "data/entities/cities.csv"
    fact_templates_path: str = "data/templates/niah_fact_templates.txt"
    counting_needle_kind: str = "city_score"
    marker_text: str = "[dolphin]"
    uid_token_length: int = 4
    haystack_seed: int | None = None
    needle_seed: int | None = None
    needle_seeds: dict[int, int | None] | None = None
    control_switch: list[bool] | None = None
    save_data: bool = False
    max_new_tokens: int | None = None
    max_new_tokens_for_cot: int = 64
    analyze_reasoning_tokens: bool = False
    temperature: float = 0.0
    trust_remote_code: bool = True
    device_map: str = "auto"
    torch_dtype: str = "bfloat16_if_cuda_else_float32"
    cache_dir: str = "/content/huggingface_models"

    @classmethod
    def from_config_file(cls, path: str | Path) -> "DynamicNiahV2Config":
        return cls(**dynamic_niah_v2_config_kwargs(load_config_file(path)))


def dynamic_niah_v2_config_kwargs(
    payload: dict[str, Any], *, warn_unknown: bool = False
) -> dict[str, Any]:
    """Return only keys accepted by DynamicNiahV2Config.

    Some notebooks share one JSON config across generation, evaluation,
    hidden-state analysis, and ablation steps. Those shared configs may include
    analysis-only keys such as ``layers`` that are intentionally not part of the
    dataset-generation config dataclass. Filter them before constructing the
    dataclass so generation/evaluation scripts can consume the shared config.
    """

    valid_fields = {field.name for field in fields(DynamicNiahV2Config)}
    unknown_keys = sorted(set(payload) - valid_fields)
    if warn_unknown and unknown_keys:
        warnings.warn(
            "Ignoring config keys that are not DynamicNiahV2Config fields: "
            + ", ".join(unknown_keys),
            stacklevel=2,
        )
    return {key: value for key, value in payload.items() if key in valid_fields}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)


def parse_insertion_position(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"null", "none", ""}:
        return None
    return int(value)


def load_config_file(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    normalized: dict[str, Any] = dict(payload)

    if "insertion_positions" in normalized and isinstance(
        normalized["insertion_positions"], list
    ):
        normalized["insertion_positions"] = tuple(
            parse_insertion_position(x) for x in normalized["insertion_positions"]
        )

    if "needle_seeds" in normalized and isinstance(normalized["needle_seeds"], dict):
        normalized["needle_seeds"] = {
            int(k): (None if v is None else int(v))
            for k, v in normalized["needle_seeds"].items()
        }

    if "control_switch" in normalized and isinstance(
        normalized["control_switch"], list
    ):
        normalized["control_switch"] = [
            _coerce_bool(v) for v in normalized["control_switch"]
        ]

    return normalized


def _load_haystack_files_filtered(
    haystack_dir: str, min_bytes: int = 5 * 1024
) -> list[Path]:
    files = [
        p
        for p in sorted(Path(haystack_dir).glob("*.txt"))
        if p.stat().st_size >= min_bytes
    ]
    if not files:
        raise ValueError(
            f"No haystack text files >= {min_bytes} bytes found in {haystack_dir}"
        )
    return files


def _read_cities(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_templates(path: str) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sample_haystack_window(
    cfg: DynamicNiahV2Config, tok: TokenizerAdapter, ex_idx: int
) -> tuple[list[Any], dict[str, Any]]:
    if cfg.haystack_seed is None:
        seed = cfg.global_random_seed * 1_000_003 + ex_idx
    else:
        seed = cfg.haystack_seed + ex_idx
    rng = random.Random(seed)
    candidates = _load_haystack_files_filtered(cfg.haystack_dir)
    if cfg.haystack_source_mode == "multi_file_no_repeat":
        ordered = list(candidates)
        rng.shuffle(ordered)
        source_files: list[str] = []
        duplicate_source_files_skipped: list[str] = []
        normalized_parts: list[str] = []
        seen_content: set[str] = set()
        tokens: list[Any] = []
        for source in ordered:
            normalized = re.sub(
                r"\s+", " ", source.read_text(encoding="utf-8")
            ).strip()
            if not normalized:
                continue
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if digest in seen_content:
                duplicate_source_files_skipped.append(source.name)
                continue
            seen_content.add(digest)
            source_files.append(source.name)
            normalized_parts.append(normalized)
            tokens = tok.encode("\n\n".join(normalized_parts))
            if len(tokens) >= cfg.target_haystack_tokens:
                break
        if not normalized_parts:
            raise ValueError(
                "No nonempty, content-unique haystack sources were found"
            )
        if len(tokens) < cfg.target_haystack_tokens:
            raise ValueError(
                "The deduplicated multi-file haystack is too short: "
                f"{len(tokens)} < {cfg.target_haystack_tokens}"
            )
        # Each seed defines one shuffled composite corpus. Prefix windows make
        # passage lengths nested within seed, matching the common NIAH/RULER
        # construction and avoiding a different filler sample at every T.
        start = 0
        end = start + cfg.target_haystack_tokens
        return list(tokens[start:end]), {
            "source_file": "__multi_file_no_repeat__",
            "source_files": source_files,
            "source_file_count": len(source_files),
            "duplicate_source_files_skipped": duplicate_source_files_skipped,
            "source_mode": cfg.haystack_source_mode,
            "window_strategy": "seed_shuffled_nested_prefix",
            "window_start": start,
            "window_end": end,
            "seed": seed,
            "original_token_count": len(tokens),
            "expanded_token_count": len(tokens),
            "source_repeat_count": 1,
            "source_repeated_to_target": False,
        }
    if cfg.haystack_source_mode != "single_file_repeat":
        raise ValueError(
            f"Unsupported haystack_source_mode: {cfg.haystack_source_mode}"
        )
    source = rng.choice(candidates)
    normalized = re.sub(r"\s+", " ", source.read_text(encoding="utf-8")).strip()
    if not normalized:
        raise ValueError(f"Haystack file {source.name} is empty after normalization")
    tokens = tok.encode(normalized)
    original_token_count = len(tokens)
    repeat_count = 1
    if len(tokens) < cfg.target_haystack_tokens:
        if not tokens:
            raise ValueError(f"Haystack file {source.name} produced no tokens")
        while len(tokens) < cfg.target_haystack_tokens:
            repeat_count += 1
            tokens = tok.encode(" ".join([normalized] * repeat_count))
    start = rng.randint(0, len(tokens) - cfg.target_haystack_tokens)
    end = start + cfg.target_haystack_tokens
    return list(tokens[start:end]), {
        "source_file": source.name,
        "source_files": [source.name],
        "source_file_count": 1,
        "duplicate_source_files_skipped": [],
        "source_mode": cfg.haystack_source_mode,
        "window_strategy": "random_single_source_window",
        "window_start": start,
        "window_end": end,
        "seed": seed,
        "original_token_count": original_token_count,
        "expanded_token_count": len(tokens),
        "source_repeat_count": repeat_count,
        "source_repeated_to_target": repeat_count > 1,
    }


def _effective_needle_seed(
    cfg: DynamicNiahV2Config, ex_idx: int, needle_idx: int
) -> int:
    mapping = cfg.needle_seeds or {}
    if needle_idx in mapping and mapping[needle_idx] is not None:
        return int(mapping[needle_idx]) + ex_idx
    if cfg.needle_seed is not None:
        return int(cfg.needle_seed) + ex_idx * 1_009 + needle_idx
    return cfg.global_random_seed * 10_000_019 + ex_idx * 1_009 + needle_idx


def _needle_seed(cfg: DynamicNiahV2Config, ex_idx: int, needle_idx: int) -> int:
    return _effective_needle_seed(cfg, ex_idx, needle_idx)


def _normalize_counting_needle_kind(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"city", "city_score", "city_scores", "score", "scores"}:
        return "city_score"
    if normalized in {"marker", "exact_marker", "simple_marker"}:
        return "marker"
    raise ValueError(
        "counting_needle_kind must be 'city_score' or 'marker', "
        f"got {value!r}"
    )


def _generate_city_score_needles(
    cfg: DynamicNiahV2Config, tok: TokenizerAdapter, ex_idx: int
) -> list[dict[str, Any]]:
    cities = _read_cities(cfg.entities_path)
    templates = _read_templates(cfg.fact_templates_path)
    if cfg.num_needles > len(cities):
        raise ValueError("num_needles exceeds available entities")

    score_pool = list(range(50, 101))
    if cfg.num_needles > len(score_pool):
        raise ValueError("num_needles exceeds unique score pool size (51)")

    # Ensure score uniqueness per example.
    needle_seed_mix = sum(
        _effective_needle_seed(cfg, ex_idx, i) for i in range(cfg.num_needles)
    )
    score_rng_seed = needle_seed_mix + ex_idx * 10007 + 17
    score_rng = random.Random(score_rng_seed)
    sampled_scores = score_rng.sample(score_pool, k=cfg.num_needles)
    entity_rng = random.Random(needle_seed_mix + ex_idx * 10007 + 31)
    sampled_city_rows = entity_rng.sample(cities, k=cfg.num_needles)

    needles: list[dict[str, Any]] = []
    for i in range(cfg.num_needles):
        row = sampled_city_rows[i]
        score = sampled_scores[i]
        text = templates[i % len(templates)].format(
            year=2024,
            entity=row["entity"],
            region=row.get("region", "unknown"),
            category=row.get("category", "unknown"),
            score=score,
        )
        needle_tokens = tok.encode(text)
        needles.append(
            {
                "needle_id": f"N{i+1}",
                "decoded_text": text,
                "tokens": needle_tokens,
                "token_length": len(needle_tokens),
                "record": {"city": row["entity"], "score": score},
            }
        )
    return needles


def _generate_marker_needles(
    cfg: DynamicNiahV2Config, tok: TokenizerAdapter, ex_idx: int
) -> list[dict[str, Any]]:
    marker = str(cfg.marker_text)
    if not marker:
        raise ValueError("marker_text must be non-empty when counting_needle_kind='marker'")
    needle_tokens = tok.encode(marker)
    if not needle_tokens:
        raise ValueError(f"marker_text tokenization produced no tokens: {marker!r}")
    needles: list[dict[str, Any]] = []
    for i in range(cfg.num_needles):
        needles.append(
            {
                "needle_id": f"N{i+1}",
                "decoded_text": marker,
                "tokens": list(needle_tokens),
                "token_length": len(needle_tokens),
                "record": {
                    "marker": marker,
                    "copy_index": i + 1,
                    "needle_kind": "marker",
                },
            }
        )
    return needles


def _random_canary(rng: random.Random) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    length = rng.randint(20, 40)
    return "".join(rng.choice(alphabet) for _ in range(length))


def _random_uid_candidate(
    rng: random.Random, *, token_length: int, tokenizer_backend: str
) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    if tokenizer_backend == "simple" and token_length > 1:
        chunk_len = max(4, min(8, 24 // token_length))
        return " ".join(
            "".join(rng.choice(alphabet) for _ in range(chunk_len))
            for _ in range(token_length)
        )
    length = rng.randint(max(1, token_length), max(40, token_length * 12))
    return "".join(rng.choice(alphabet) for _ in range(length))


def _literal_uid_tokens(
    tok: TokenizerAdapter, uid: str, *, text_level_insertion: bool
) -> list[Any]:
    if not text_level_insertion:
        return list(tok.encode(uid))

    # Text-level insertion places the UID between words.  Tokenizers such as
    # Qwen's often attach the preceding whitespace to the first UID token, so
    # bare ``tok.encode(uid)`` is not the length that the final context span
    # will recover.  Measure the UID in a neutral between-word context instead.
    prefix = "alpha "
    sample = f"{prefix}{uid} omega"
    char_start = len(prefix)
    char_end = char_start + len(uid)
    span_start, span_end = _token_span_for_text_slice(tok, sample, char_start, char_end)
    return list(tok.encode(sample[:char_end]))[span_start:span_end]


def _generate_literal_uid(
    cfg: DynamicNiahV2Config, tok: TokenizerAdapter, rng: random.Random
) -> tuple[str, list[Any]]:
    target_len = int(cfg.uid_token_length)
    if target_len <= 0:
        raise ValueError(f"uid_token_length must be positive, got {target_len}")

    for _ in range(10_000):
        candidate = _random_uid_candidate(
            rng, token_length=target_len, tokenizer_backend=tok.backend
        )
        tokens = _literal_uid_tokens(
            tok,
            candidate,
            text_level_insertion=bool(
                cfg.sentence_level_insertion or cfg.word_level_insertion
            ),
        )
        if len(tokens) == target_len:
            return candidate, list(tokens)

    raise RuntimeError(
        "Unable to generate literal_count UID with requested token length "
        f"{target_len} for tokenizer {cfg.tokenizer_name!r}"
    )


def _generate_literal_needles(
    cfg: DynamicNiahV2Config, tok: TokenizerAdapter, ex_idx: int
) -> list[dict[str, Any]]:
    rng = random.Random(_effective_needle_seed(cfg, ex_idx, 0) + 97_531)
    canary, needle_tokens = _generate_literal_uid(cfg, tok, rng)
    if len(needle_tokens) != int(cfg.uid_token_length):
        raise ValueError(
            "literal_count UID tokenization length mismatch: expected "
            f"{cfg.uid_token_length}, got {len(needle_tokens)} for {canary!r}"
        )

    needles: list[dict[str, Any]] = []
    for i in range(cfg.num_needles):
        needles.append(
            {
                "needle_id": f"N{i+1}",
                "decoded_text": canary,
                "tokens": list(needle_tokens),
                "token_length": len(needle_tokens),
                "record": {
                    "canary": canary,
                    "literal": canary,
                    "copy_index": i + 1,
                    "uid_token_length": int(cfg.uid_token_length),
                    "delimited_text": canary,
                },
            }
        )
    return needles


def _generate_needles(
    cfg: DynamicNiahV2Config, tok: TokenizerAdapter, ex_idx: int
) -> list[dict[str, Any]]:
    task_type = canonical_task_type(cfg.task_type)
    if task_type == "literal_count":
        return _generate_literal_needles(cfg, tok, ex_idx)
    if task_type == "match_count" and _normalize_counting_needle_kind(
        cfg.counting_needle_kind
    ) == "marker":
        return _generate_marker_needles(cfg, tok, ex_idx)
    return _generate_city_score_needles(cfg, tok, ex_idx)


def _normalize_control_switch(
    control_switch: list[bool] | None, num_needles: int
) -> list[bool]:
    if control_switch is None:
        return [False] * num_needles

    normalized = [_coerce_bool(v) for v in control_switch]
    if len(normalized) < num_needles:
        missing = num_needles - len(normalized)
        warnings.warn(
            "control_switch has fewer values than num_needles; padding missing "
            f"needle controls with False (got {len(normalized)} control values "
            f"for num_needles={num_needles}, padded {missing}).",
            stacklevel=2,
        )
        normalized.extend([False] * missing)
    elif len(normalized) > num_needles:
        raise ValueError(
            "control_switch length cannot exceed num_needles when provided "
            f"(got {len(normalized)} control values for num_needles={num_needles}). "
            "When running gen_responses.py from the notebook, pass matching "
            "--num-needles and --positions values or update CONFIG_PATH to match "
            "CONTROL_SWITCH."
        )
    return normalized


def _normalize_insertion_positions(
    insertion_positions: tuple[int | None, ...], num_needles: int
) -> tuple[int | None, ...]:
    if len(insertion_positions) != num_needles:
        raise ValueError(
            "Number of insertion_positions must equal num_needles "
            f"(got {len(insertion_positions)} positions for num_needles={num_needles})."
        )
    return tuple(None if pos is None else int(pos) for pos in insertion_positions)


def _validate_num_max_needles(value: int | None) -> int | None:
    if value is None:
        return None
    out = int(value)
    if out <= 0:
        raise ValueError(f"num_max_needles must be positive when set, got {value}")
    return out


def _sample_num_needles_for_example(
    cfg: DynamicNiahV2Config, ex_idx: int
) -> int:
    """Return the per-example needle count for fixed or variable-count runs."""

    max_needles = _validate_num_max_needles(cfg.num_max_needles)
    if max_needles is None:
        return int(cfg.num_needles)
    rng = random.Random(int(cfg.global_random_seed) * 1_000_003 + int(ex_idx) + 7919)
    return int(rng.randint(1, max_needles))


def _per_example_config(
    cfg: DynamicNiahV2Config, *, ex_idx: int
) -> tuple[DynamicNiahV2Config, tuple[int | None, ...], list[bool]]:
    """Resolve count, positions, and controls for one generated example."""

    target_num_needles = _sample_num_needles_for_example(cfg, ex_idx)
    if len(cfg.insertion_positions) < target_num_needles:
        raise ValueError(
            "insertion_positions must contain at least the sampled per-example "
            f"needle count (got {len(cfg.insertion_positions)} positions for "
            f"sampled count {target_num_needles}; num_max_needles={cfg.num_max_needles})."
        )
    insertion_position_pattern = _normalize_insertion_positions(
        tuple(cfg.insertion_positions[:target_num_needles]), target_num_needles
    )
    raw_control_switch = (
        None if cfg.control_switch is None else list(cfg.control_switch[:target_num_needles])
    )
    control_switch = _normalize_control_switch(raw_control_switch, target_num_needles)
    ex_cfg = replace(
        cfg,
        num_needles=target_num_needles,
        insertion_positions=insertion_position_pattern,
        control_switch=control_switch,
    )
    return ex_cfg, insertion_position_pattern, control_switch


def sample_random_insertion_positions(
    *,
    target_haystack_tokens: int,
    num_needles: int,
    seed: int,
    margin: int = 50,
    min_separation: int = 50,
) -> tuple[int, ...]:
    """Sample sorted per-example insertion positions with a spacing constraint."""

    if num_needles < 0:
        raise ValueError(f"num_needles must be non-negative, got {num_needles}")
    if num_needles == 0:
        return ()
    if margin < 0:
        raise ValueError(f"margin must be non-negative, got {margin}")
    if min_separation < 0:
        raise ValueError(
            f"min_separation must be non-negative, got {min_separation}"
        )
    low = int(margin)
    high_exclusive = int(target_haystack_tokens) - int(margin)
    candidates = list(range(low, high_exclusive))
    if not candidates:
        raise ValueError(
            "Cannot randomize needle insertions: no candidate positions in "
            f"range({low}, {high_exclusive}) for target_haystack_tokens={target_haystack_tokens}"
        )
    max_possible = 0
    last = candidates[0] - min_separation
    for pos in candidates:
        if pos - last >= min_separation:
            max_possible += 1
            last = pos
    if num_needles > max_possible:
        raise ValueError(
            "Cannot randomize needle insertions: requested "
            f"{num_needles} positions but at most {max_possible} fit in "
            f"range({low}, {high_exclusive}) with min_separation={min_separation}"
        )

    rng = random.Random(int(seed))
    for _ in range(10_000):
        sampled = sorted(rng.sample(candidates, k=num_needles))
        if all(b - a >= min_separation for a, b in zip(sampled, sampled[1:])):
            return tuple(int(x) for x in sampled)

    shuffled = candidates[:]
    rng.shuffle(shuffled)
    chosen: list[int] = []
    for pos in shuffled:
        if all(abs(pos - existing) >= min_separation for existing in chosen):
            chosen.append(pos)
            if len(chosen) == num_needles:
                return tuple(sorted(chosen))
    raise RuntimeError("Failed to sample feasible randomized insertion positions")


def insertion_positions_for_example(
    cfg: DynamicNiahV2Config, fixed_pattern: tuple[int | None, ...], ex_idx: int
) -> tuple[int | None, ...]:
    """Return the effective insertion positions for one generated example."""

    if not cfg.randomize_needle_insertion:
        return fixed_pattern
    sampled = sample_random_insertion_positions(
        target_haystack_tokens=cfg.target_haystack_tokens,
        num_needles=cfg.num_needles,
        seed=int(cfg.randomize_needle_seed) + int(ex_idx),
        margin=int(cfg.randomize_needle_margin),
        min_separation=int(cfg.randomize_needle_min_separation),
    )
    return tuple(
        None if fixed_pattern[i] is None else int(sampled[i])
        for i in range(cfg.num_needles)
    )


def _control_seed(cfg: DynamicNiahV2Config, ex_idx: int, needle_idx: int) -> int:
    return (
        cfg.global_random_seed * 100_000_007 + ex_idx * 10_007 + needle_idx * 101 + 53
    )


def _sample_control_segment(
    cfg: DynamicNiahV2Config,
    tok: TokenizerAdapter,
    ex_idx: int,
    needle_idx: int,
    target_len: int,
) -> dict[str, Any]:
    if target_len <= 0:
        raise ValueError("Control segment target length must be > 0")
    seed = _control_seed(cfg, ex_idx, needle_idx)
    rng = random.Random(seed)
    candidates = _load_haystack_files_filtered(cfg.haystack_dir)
    for _ in range(max(10, len(candidates) * 2)):
        source = rng.choice(candidates)
        normalized = re.sub(r"\s+", " ", source.read_text(encoding="utf-8")).strip()
        tokens = tok.encode(normalized)
        if len(tokens) < target_len:
            continue
        start = (
            0 if len(tokens) == target_len else rng.randint(0, len(tokens) - target_len)
        )
        end = start + target_len
        sampled = list(tokens[start:end])
        return {
            "seed": seed,
            "source_file": source.name,
            "segment_start_token": start,
            "segment_end_token": end,
            "tokens": sampled,
            "decoded_text": tok.decode(sampled),
            "token_length": len(sampled),
        }
    raise RuntimeError(f"Unable to sample control segment of length {target_len}")


def _insert(
    base_tokens: list[Any],
    needles: list[dict[str, Any]],
    positions: tuple[int | None, ...],
) -> tuple[list[Any], list[dict[str, Any]]]:
    if len(needles) != len(positions):
        raise ValueError("Number of insertion positions must equal number of needles")
    out = list(base_tokens)
    shift = 0
    realized: list[dict[str, Any]] = []
    scheduled = [
        (pos, needle) for pos, needle in zip(positions, needles) if pos is not None
    ]
    for pos, needle in sorted(scheduled, key=lambda x: int(x[0])):
        pos = int(pos)
        if pos < 0 or pos > len(base_tokens):
            raise ValueError(
                f"Insertion position {pos} out of range 0..{len(base_tokens)}"
            )
        final_pos = pos + shift
        inserted_tokens = needle["inserted_tokens"]
        out[final_pos:final_pos] = inserted_tokens
        realized.append(
            {
                "needle_id": needle["needle_id"],
                "requested_position": pos,
                "final_position": final_pos,
                "token_length": len(inserted_tokens),
                "tokens": inserted_tokens,
                "decoded_text": needle["inserted_decoded_text"],
                "is_control": needle["is_control"],
                "inserted_from": "control" if needle["is_control"] else "needle",
                "needle_token_length": needle["token_length"],
                "control": needle.get("control"),
            }
        )
        shift += len(inserted_tokens)
    return out, realized


_COMMON_ABBREVIATIONS = {
    "adm",
    "capt",
    "col",
    "corp",
    "dr",
    "e",
    "eg",
    "etc",
    "fig",
    "gen",
    "gov",
    "i",
    "ie",
    "inc",
    "jr",
    "ltd",
    "mr",
    "mrs",
    "ms",
    "no",
    "prof",
    "rev",
    "sen",
    "sr",
    "st",
    "vs",
}


def _nonspace_token_around(text: str, index: int) -> str:
    start = index
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    end = index + 1
    while end < len(text) and not text[end].isspace():
        end += 1
    return text[start:end]


def _looks_like_url_or_path_context(text: str, index: int) -> bool:
    token = _nonspace_token_around(text, index)
    stripped = token.strip("\"'()[]{}<>")
    if any(marker in stripped for marker in ("://", "/", "\\", "@")):
        return True
    if re.search(r"\b(?:https?|www)\.", stripped, flags=re.IGNORECASE):
        return True
    if "." in stripped.rstrip(".!?") and re.search(r"[A-Za-z0-9]", stripped):
        return True
    return False


def _looks_like_abbreviation_context(text: str, index: int) -> bool:
    if text[index] != ".":
        return False
    prefix = text[max(0, index - 24) : index]
    match = re.search(r"([A-Za-z]+)$", prefix)
    if not match:
        return False
    word = match.group(1).lower()
    if word in _COMMON_ABBREVIATIONS:
        return True
    if len(word) == 1:
        return True
    return False


def _looks_like_numeric_fragment(text: str, index: int) -> bool:
    prev_char = text[index - 1] if index > 0 else ""
    next_char = text[index + 1] if index + 1 < len(text) else ""
    return bool(prev_char.isdigit() and next_char.isdigit())


def _valid_sentence_following_context(text: str, index: int) -> bool:
    pos = index + 1
    while pos < len(text) and text[pos] in "\"'”’)]}":
        pos += 1
    if pos >= len(text):
        return True
    if not text[pos].isspace():
        return False
    while pos < len(text) and text[pos].isspace():
        pos += 1
    while pos < len(text) and text[pos] in "\"'“‘([{":
        pos += 1
    if pos >= len(text):
        return True
    next_char = text[pos]
    if text[index] == "." and next_char.islower():
        return False
    return True


def _is_valid_sentence_end(text: str, index: int) -> bool:
    delimiter = text[index]
    if delimiter not in ".!?":
        return False
    if _looks_like_numeric_fragment(text, index):
        return False
    if _looks_like_url_or_path_context(text, index):
        return False
    if _looks_like_abbreviation_context(text, index):
        return False
    if not _valid_sentence_following_context(text, index):
        return False
    return True


def _sentence_end_offsets(text: str) -> list[int]:
    """Return conservative char offsets after likely sentence-ending delimiters."""

    return [
        match.end()
        for match in re.finditer(r"[.!?]", text)
        if _is_valid_sentence_end(text, match.start())
    ]


def _word_boundary_offsets(text: str) -> list[int]:
    """Return char offsets at whitespace runs between non-whitespace chunks."""

    offsets: list[int] = []
    for match in re.finditer(r"\s+", text):
        if match.start() <= 0 or match.end() >= len(text):
            continue
        if text[match.start() - 1].isspace() or text[match.end()].isspace():
            continue
        offsets.append(match.start())
    return offsets


def _token_span_for_text_slice(
    tok: TokenizerAdapter, text: str, start: int, end: int
) -> tuple[int, int]:
    encode_with_offsets = getattr(tok, "encode_with_offsets", None)
    if callable(encode_with_offsets):
        _, offsets = encode_with_offsets(text)
        overlapping = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > token_start
            and token_start < end
            and token_end > start
        ]
        if not overlapping:
            raise ValueError(
                "No tokenizer offset overlaps inserted text at "
                f"chars {start}:{end}"
            )
        return overlapping[0], overlapping[-1] + 1

    # Compatibility fallback for lightweight tokenizer doubles used by callers.
    # Production Hugging Face and simple adapters always take the exact
    # offset-mapping path above.
    start_token = len(tok.encode(text[:start]))
    end_token = len(tok.encode(text[:end]))
    if end_token < start_token:
        raise ValueError("Computed invalid token span for inserted needle text")
    return start_token, end_token


def _verified_text_insertion_metadata(
    *,
    tok: TokenizerAdapter,
    text: str,
    inserted_text: str,
    char_start: int,
    char_end: int,
    expected_token_length: int | None = None,
) -> dict[str, Any]:
    if char_start < 0 or char_end > len(text) or char_end < char_start:
        raise ValueError(
            "Inserted text character span is out of range: "
            f"{char_start}:{char_end} for text length {len(text)}"
        )
    observed = text[char_start:char_end]
    if observed != inserted_text:
        raise ValueError(
            "Inserted text span verification failed: expected "
            f"{inserted_text!r}, found {observed!r}"
        )
    span_start, span_end = _token_span_for_text_slice(tok, text, char_start, char_end)
    if span_end <= span_start:
        raise ValueError(
            "Inserted text token span is empty for "
            f"{inserted_text!r} at chars {char_start}:{char_end}"
        )
    observed_token_length = span_end - span_start
    if expected_token_length is not None and observed_token_length != int(
        expected_token_length
    ):
        raise ValueError(
            "Inserted text token length verification failed for "
            f"{inserted_text!r}: expected {expected_token_length}, "
            f"found {observed_token_length}"
        )
    return {
        "char_start": char_start,
        "char_end": char_end,
        "context_span_start": span_start,
        "context_span_end": span_end,
        "observed_token_length": observed_token_length,
        "token_span_verified": True,
    }


def _format_text_insertion(
    out: str, final_insert_offset: int, inserted_text: str
) -> tuple[str, int]:
    prefix = (
        ""
        if final_insert_offset > 0 and out[final_insert_offset - 1].isspace()
        else " "
    )
    suffix = (
        ""
        if final_insert_offset < len(out) and out[final_insert_offset].isspace()
        else " "
    )
    return f"{prefix}{inserted_text}{suffix}", len(prefix)


def _sample_text_offsets(
    *,
    candidates: list[int],
    active_indices: list[int],
    seed: int,
    mode_label: str,
    candidate_label: str,
) -> tuple[dict[int, int], bool]:
    if not candidates:
        raise ValueError(
            f"Cannot perform {mode_label} needle insertion: requested "
            f"{len(active_indices)} inserted needles but found no {candidate_label}."
        )
    rng = random.Random(int(seed))
    sample_with_replacement = len(candidates) < len(active_indices)
    if sample_with_replacement:
        print(
            f"{mode_label.upper()}=True: requested "
            f"{len(active_indices)} inserted needles but found only "
            f"{len(candidates)} {candidate_label}; sampling insertion sites "
            "with replacement."
        )
        selected_offsets = [rng.choice(candidates) for _ in range(len(active_indices))]
    else:
        selected_offsets = sorted(rng.sample(candidates, k=len(active_indices)))
    return dict(zip(active_indices, selected_offsets)), sample_with_replacement


def _validated_text_insertion_fraction_bounds(
    cfg: DynamicNiahV2Config,
) -> tuple[float, float]:
    minimum = float(cfg.text_insertion_min_fraction)
    maximum = float(cfg.text_insertion_max_fraction)
    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or minimum < 0.0
        or maximum > 1.0
        or minimum >= maximum
    ):
        raise ValueError(
            "text insertion fractions must be finite and satisfy "
            "0 <= min < max <= 1"
        )
    return minimum, maximum


def _depth_filtered_text_offsets(
    *,
    base_text: str,
    candidates: list[int],
    needles: list[dict[str, Any]],
    active_indices: list[int],
    cfg: DynamicNiahV2Config,
) -> tuple[list[int], dict[str, Any]]:
    """Restrict candidate offsets so final character depths stay in bounds.

    The bound is conservative. Each active insertion may add at most two
    whitespace characters in addition to its decoded text. The lower bound
    assumes all inserted characters enlarge the final denominator, while the
    upper bound assumes all inserted characters occur before the candidate.
    The completed passage is checked again after insertion.
    """

    minimum, maximum = _validated_text_insertion_fraction_bounds(cfg)
    minimum_inserted_characters = sum(
        len(str(needles[index]["inserted_decoded_text"]))
        for index in active_indices
    )
    maximum_inserted_characters = sum(
        len(str(needles[index]["inserted_decoded_text"])) + 2
        for index in active_indices
    )
    base_characters = len(base_text)
    if minimum == 0.0 and maximum == 1.0:
        # Preserve the exact V2 behavior: the default interval is not an
        # insertion-site filter, so candidate counts and seeded samples remain
        # bit-for-bit unchanged.
        return list(candidates), {}
    maximum_final_characters = base_characters + maximum_inserted_characters
    minimum_final_characters = base_characters + minimum_inserted_characters
    minimum_offset = math.ceil(minimum * maximum_final_characters)
    maximum_offset = math.floor(
        maximum * minimum_final_characters
        - maximum_inserted_characters
    )
    eligible = [
        int(offset)
        for offset in candidates
        if minimum_offset <= int(offset) <= maximum_offset
    ]
    metadata = {
        "text_insertion_min_fraction": minimum,
        "text_insertion_max_fraction": maximum,
        "text_insertion_unfiltered_candidate_count": len(candidates),
        "text_insertion_eligible_candidate_count": len(eligible),
        "text_insertion_conservative_min_offset": minimum_offset,
        "text_insertion_conservative_max_offset": maximum_offset,
        "text_insertion_minimum_added_characters": minimum_inserted_characters,
        "text_insertion_maximum_added_characters": maximum_inserted_characters,
    }
    if active_indices and not eligible:
        raise ValueError(
            "No text insertion candidates remain within the registered "
            f"fraction bounds [{minimum}, {maximum}]; "
            f"base_characters={base_characters}, candidates={len(candidates)}"
        )
    return eligible, metadata


def _insert_at_text_offsets(
    base_text: str,
    needles: list[dict[str, Any]],
    active_slots: tuple[bool, ...],
    *,
    cfg: DynamicNiahV2Config,
    tok: TokenizerAdapter,
    ex_idx: int,
    mode: str,
    candidates: list[int],
    candidate_label: str,
) -> tuple[str, list[dict[str, Any]]]:
    if len(needles) != len(active_slots):
        raise ValueError("Number of active insertion slots must equal number of needles")
    active_indices = [idx for idx, active in enumerate(active_slots) if active]
    if not active_indices:
        return base_text, []

    candidates, depth_filter_metadata = _depth_filtered_text_offsets(
        base_text=base_text,
        candidates=candidates,
        needles=needles,
        active_indices=active_indices,
        cfg=cfg,
    )
    offset_by_needle_idx, sample_with_replacement = _sample_text_offsets(
        candidates=candidates,
        active_indices=active_indices,
        seed=int(cfg.randomize_needle_seed) + int(ex_idx),
        mode_label=f"{mode}-level",
        candidate_label=candidate_label,
    )
    insertion_order = sorted(
        active_indices,
        key=lambda needle_idx: (int(offset_by_needle_idx[needle_idx]), int(needle_idx)),
    )

    out = base_text
    shift = 0
    realized: list[dict[str, Any]] = []
    for needle_idx in insertion_order:
        needle = needles[needle_idx]
        offset = int(offset_by_needle_idx[needle_idx])
        final_insert_offset = offset + shift
        inserted_text = str(needle["inserted_decoded_text"])
        text_to_insert, prefix_len = _format_text_insertion(
            out, final_insert_offset, inserted_text
        )
        out = out[:final_insert_offset] + text_to_insert + out[final_insert_offset:]
        needle_char_start = final_insert_offset + prefix_len
        needle_char_end = needle_char_start + len(inserted_text)
        strict_expected_token_length = (
            needle["token_length"]
            if canonical_task_type(cfg.task_type) == "literal_count"
            else None
        )
        span_metadata = _verified_text_insertion_metadata(
            tok=tok,
            text=out,
            inserted_text=inserted_text,
            char_start=needle_char_start,
            char_end=needle_char_end,
            expected_token_length=strict_expected_token_length,
        )
        span_start = int(span_metadata["context_span_start"])
        observed_token_length = int(span_metadata["observed_token_length"])
        insertion_metadata: dict[str, Any] = {
            "needle_id": needle["needle_id"],
            "requested_position": span_start,
            "final_position": span_start,
            "token_length": observed_token_length,
            "tokens": needle["inserted_tokens"],
            "decoded_text": inserted_text,
            "is_control": needle["is_control"],
            "inserted_from": "control" if needle["is_control"] else "needle",
            "needle_token_length": needle["token_length"],
            "observed_token_length": observed_token_length,
            "control": needle.get("control"),
            "text_level_insertion": True,
            "text_insertion_mode": mode,
            "text_insertion_offset": offset,
            "text_insertion_candidate_count": len(candidates),
            "text_insertion_sampled_with_replacement": sample_with_replacement,
            **depth_filter_metadata,
            **span_metadata,
            "token_span_verified": True,
        }
        if mode == "sentence":
            insertion_metadata.update(
                {
                    "sentence_level_insertion": True,
                    "sentence_delimiter_offset": offset,
                    "sentence_delimiter_candidate_count": len(candidates),
                    "sentence_delimiter_sampled_with_replacement": sample_with_replacement,
                    "token_span_source": "sentence_text_offsets",
                }
            )
        elif mode == "word":
            insertion_metadata.update(
                {
                    "word_level_insertion": True,
                    "word_boundary_offset": offset,
                    "word_boundary_candidate_count": len(candidates),
                    "word_boundary_sampled_with_replacement": sample_with_replacement,
                    "token_span_source": "word_text_offsets",
                }
            )
        else:
            raise ValueError(f"Unsupported text insertion mode: {mode}")
        realized.append(insertion_metadata)
        shift += len(text_to_insert)

    # Recompute every span against the final passage. This matters when several
    # needles share a sampled insertion site and a later insertion changes the
    # local tokenization boundary of an earlier one.
    refreshed: list[dict[str, Any]] = []
    for insertion in realized:
        strict_expected_token_length = (
            insertion["needle_token_length"]
            if canonical_task_type(cfg.task_type) == "literal_count"
            else None
        )
        span_metadata = _verified_text_insertion_metadata(
            tok=tok,
            text=out,
            inserted_text=str(insertion["decoded_text"]),
            char_start=int(insertion["char_start"]),
            char_end=int(insertion["char_end"]),
            expected_token_length=strict_expected_token_length,
        )
        updated = dict(insertion)
        updated.update(span_metadata)
        updated.update(
            {
                "requested_position": int(span_metadata["context_span_start"]),
                "final_position": int(span_metadata["context_span_start"]),
                "token_length": int(span_metadata["observed_token_length"]),
                "token_span_alignment": (
                    "offset_mapping_overlap"
                    if callable(getattr(tok, "encode_with_offsets", None))
                    else "prefix_encoding_fallback"
                ),
            }
        )
        minimum, maximum = _validated_text_insertion_fraction_bounds(cfg)
        if minimum != 0.0 or maximum != 1.0:
            normalized_text_depth = float(updated["char_start"]) / max(
                1,
                len(out),
            )
            if not minimum <= normalized_text_depth <= maximum:
                raise RuntimeError(
                    "Final text insertion depth fell outside the registered "
                    f"bounds: {normalized_text_depth} not in "
                    f"[{minimum}, {maximum}]"
                )
            updated["normalized_text_depth"] = normalized_text_depth
        refreshed.append(updated)
    return out, refreshed


def _insert_at_sentence_ends(
    base_text: str,
    needles: list[dict[str, Any]],
    active_slots: tuple[bool, ...],
    *,
    cfg: DynamicNiahV2Config,
    tok: TokenizerAdapter,
    ex_idx: int,
) -> tuple[str, list[dict[str, Any]]]:
    candidates = _sentence_end_offsets(base_text)
    if candidates:
        return _insert_at_text_offsets(
            base_text,
            needles,
            active_slots,
            cfg=cfg,
            tok=tok,
            ex_idx=ex_idx,
            mode="sentence",
            candidates=candidates,
            candidate_label="conservative sentence-ending delimiters",
        )

    fallback_candidates = _word_boundary_offsets(base_text)
    if fallback_candidates:
        print(
            "SENTENCE_LEVEL_INSERTION=True: found no conservative sentence-ending "
            "delimiters for this example; falling back to word-boundary insertion "
            "to avoid URL/path/abbreviation delimiter artifacts."
        )
        out, realized = _insert_at_text_offsets(
            base_text,
            needles,
            active_slots,
            cfg=cfg,
            tok=tok,
            ex_idx=ex_idx,
            mode="sentence",
            candidates=fallback_candidates,
            candidate_label="word-boundary fallback offsets",
        )
        for item in realized:
            item.update(
                {
                    "sentence_delimiter_filter_fallback": "word_boundary",
                    "sentence_delimiter_conservative_candidate_count": 0,
                    "sentence_delimiter_fallback_candidate_count": len(fallback_candidates),
                    "token_span_source": "sentence_text_offsets_word_boundary_fallback",
                }
            )
        return out, realized

    return _insert_at_text_offsets(
        base_text,
        needles,
        active_slots,
        cfg=cfg,
        tok=tok,
        ex_idx=ex_idx,
        mode="sentence",
        candidates=candidates,
        candidate_label="conservative sentence-ending delimiters or word-boundary fallback offsets",
    )


def _insert_at_word_boundaries(
    base_text: str,
    needles: list[dict[str, Any]],
    active_slots: tuple[bool, ...],
    *,
    cfg: DynamicNiahV2Config,
    tok: TokenizerAdapter,
    ex_idx: int,
) -> tuple[str, list[dict[str, Any]]]:
    return _insert_at_text_offsets(
        base_text,
        needles,
        active_slots,
        cfg=cfg,
        tok=tok,
        ex_idx=ex_idx,
        mode="word",
        candidates=_word_boundary_offsets(base_text),
        candidate_label="word-boundary whitespace runs",
    )


def _annotate_token_insertions_with_context_spans(
    realized: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for insertion in realized:
        updated = dict(insertion)
        start = int(updated["final_position"])
        end = start + int(updated["token_length"])
        updated.update(
            {
                "context_span_start": start,
                "context_span_end": end,
                "token_span_verified": True,
                "token_span_source": "token_insertion",
            }
        )
        annotated.append(updated)
    return annotated


def _count_subsequence(tokens: list[Any], needle: list[Any]) -> int:
    if not needle or len(needle) > len(tokens):
        return 0
    count = 0
    width = len(needle)
    for start in range(0, len(tokens) - width + 1):
        if tokens[start : start + width] == needle:
            count += 1
    return count


def _literal_validation_metadata(
    tok: TokenizerAdapter,
    context: str,
    needles: list[dict[str, Any]],
    inserted_count: int,
) -> dict[str, Any]:
    if not needles:
        return {"checked": False, "reason": "no_needles"}

    first_record = needles[0].get("record", {})
    literal = str(first_record.get("literal", ""))
    if not literal:
        raise ValueError("literal_count validation requires a non-empty literal")

    expected_tokens = list(needles[0]["tokens"])
    retokenized_context = tok.encode(context)
    token_occurrences = _count_subsequence(retokenized_context, expected_tokens)
    literal_text_occurrences = context.count(literal)
    delimited_text = str(first_record.get("delimited_text", needles[0]["decoded_text"]))
    delimited_text_occurrences = context.count(delimited_text)

    # The prediction target for literal_count is the literal canary string, not
    # the full delimiter token sequence.  Some Hugging Face tokenizers are not
    # decode/encode idempotent at arbitrary insertion boundaries, so a full
    # needle token subsequence such as ``<<<NIAH_CANARY ... >>>`` can fail to
    # remain contiguous after the decoded context is retokenized even when the
    # exact literal canary text appears the correct number of times.  Validate
    # the task target strictly, and keep the full-token count as audit metadata.
    if literal_text_occurrences != inserted_count:
        raise ValueError(
            "literal_count text validation failed: expected "
            f"{inserted_count} occurrences of literal {literal!r}, "
            f"found {literal_text_occurrences}"
        )

    return {
        "checked": True,
        "validation_basis": "literal_text",
        "validated_text": literal,
        "validated_delimited_text": delimited_text,
        "validated_token_length": len(expected_tokens),
        "expected_occurrences": inserted_count,
        "observed_occurrences": literal_text_occurrences,
        "observed_literal_text_occurrences": literal_text_occurrences,
        "observed_delimited_text_occurrences": delimited_text_occurrences,
        "observed_delimited_token_occurrences": token_occurrences,
        "delimited_token_occurrences_match_expected": token_occurrences
        == inserted_count,
    }


def _literal_text_from_row(row: dict[str, Any]) -> str | None:
    if row.get("task_type") != "literal_count":
        return None
    needles = row.get("needles") or []
    if not needles:
        return None
    record = needles[0].get("record") or {}
    literal = record.get("literal")
    return str(literal) if literal is not None else None


def _build_inputs(
    cfg: DynamicNiahV2Config,
    context: str,
    query: str,
    *,
    literal_text: str | None = None,
) -> list[dict[str, str]]:
    response_schema = response_schema_for_task(canonical_task_type(cfg.task_type))
    counting_needle_kind = _normalize_counting_needle_kind(cfg.counting_needle_kind)
    if cfg.prompt_style == "easier":
        return build_messages_easier(
            context,
            query,
            thinking_mode=cfg.thinking_mode,
            response_schema=response_schema,
            task_type=canonical_task_type(cfg.task_type),
            counting_needle_kind=counting_needle_kind,
            marker_text=cfg.marker_text,
            literal_text=literal_text,
        )
    if cfg.prompt_style == "vanilla_no_cue":
        return build_messages_vanilla(
            context,
            query,
            thinking_mode=cfg.thinking_mode,
            response_schema=response_schema,
            task_type=canonical_task_type(cfg.task_type),
            counting_needle_kind=counting_needle_kind,
            marker_text=cfg.marker_text,
            literal_text=literal_text,
            include_reasoning_ban=False,
            include_memorization_instruction=False,
        )
    return build_messages_vanilla(
        context,
        query,
        thinking_mode=cfg.thinking_mode,
        response_schema=response_schema,
        task_type=canonical_task_type(cfg.task_type),
        counting_needle_kind=counting_needle_kind,
        marker_text=cfg.marker_text,
        literal_text=literal_text,
    )


def build_uncontrolled_context(row: dict[str, Any]) -> str:
    """Return the all-needle context for a generated row.

    Rows generated with ``control_switch`` keep their primary ``context`` aligned
    with the control condition so hidden-state analysis can compare it directly.
    Response evaluation, however, is scored against ``gold_answer`` over all
    generated needles. This helper restores each controlled span to the original
    needle text before prompting the model.
    """

    if row.get("uncontrolled_context") is not None:
        return str(row["uncontrolled_context"])

    context = str(row.get("context", ""))
    for needle in row.get("needles", []):
        if not needle.get("is_control"):
            continue
        if needle.get("is_inserted") is False:
            continue
        control_text = str(needle.get("inserted_decoded_text", ""))
        needle_text = str(needle.get("decoded_text", ""))
        if not control_text:
            continue
        if control_text not in context:
            raise ValueError(
                f"Cannot restore uncontrolled context for {row.get('id')}: "
                f"control text for {needle.get('needle_id')} was not found"
            )
        context = context.replace(control_text, needle_text, 1)
    return context


def build_uncontrolled_messages(
    cfg: DynamicNiahV2Config, row: dict[str, Any]
) -> list[dict[str, str]]:
    """Build saved all-needle messages for compatibility/inspection."""

    if row.get("uncontrolled_messages") is not None:
        return row["uncontrolled_messages"]
    return _build_inputs(
        cfg,
        build_uncontrolled_context(row),
        row["query"],
        literal_text=_literal_text_from_row(row),
    )


def build_prediction_messages(
    cfg: DynamicNiahV2Config, row: dict[str, Any]
) -> list[dict[str, str]]:
    """Build response-evaluation messages directly from uncontrolled_context.

    Prediction must be based on the all-needle context so the ideal answer stays
    aligned with ``gold_answer``. In particular, a ``count_avg`` row with three
    generated needles and one control replacement still has ideal count 3, not
    the controlled-context count 2.
    """

    return _build_inputs(
        cfg,
        build_uncontrolled_context(row),
        row["query"],
        literal_text=_literal_text_from_row(row),
    )


def generate_dynamic_niah_dataset_v2(
    cfg: DynamicNiahV2Config,
    *,
    tokenizer_adapter: TokenizerAdapter | None = None,
) -> list[dict[str, Any]]:
    if cfg.sentence_level_insertion and cfg.word_level_insertion:
        raise ValueError(
            "sentence_level_insertion and word_level_insertion are mutually exclusive"
        )
    _validated_text_insertion_fraction_bounds(cfg)
    _validate_num_max_needles(cfg.num_max_needles)
    counting_needle_kind = _normalize_counting_needle_kind(cfg.counting_needle_kind)
    if cfg.num_max_needles is None:
        insertion_position_pattern = _normalize_insertion_positions(
            cfg.insertion_positions, cfg.num_needles
        )
        control_switch = _normalize_control_switch(cfg.control_switch, cfg.num_needles)
    else:
        if len(cfg.insertion_positions) < int(cfg.num_max_needles):
            raise ValueError(
                "insertion_positions must contain at least num_max_needles values "
                f"(got {len(cfg.insertion_positions)} positions for "
                f"num_max_needles={cfg.num_max_needles})."
            )
        insertion_position_pattern = tuple(cfg.insertion_positions)
        control_switch = _normalize_control_switch(
            None if cfg.control_switch is None else list(cfg.control_switch),
            len(insertion_position_pattern),
        )
    tok = tokenizer_adapter or TokenizerAdapter(cfg.tokenizer_name)
    if tok.tokenizer_name != cfg.tokenizer_name:
        raise ValueError(
            "tokenizer_adapter does not match cfg.tokenizer_name "
            f"({tok.tokenizer_name!r} != {cfg.tokenizer_name!r})"
        )
    if cfg.sentence_level_insertion:
        print(
            "SENTENCE_LEVEL_INSERTION=True: needles will be inserted randomly "
            "at sentence ends after '.', '!', or '?'. Fixed numeric "
            "INSERTION_POSITIONS are ignored except that None slots are not inserted."
        )
    if cfg.word_level_insertion:
        print(
            "WORD_LEVEL_INSERTION=True: needles will be inserted randomly "
            "at whitespace boundaries between non-whitespace text chunks. Fixed "
            "numeric INSERTION_POSITIONS are ignored except that None slots are "
            "not inserted."
        )
    rows: list[dict[str, Any]] = []
    ex_idx = 0
    attempts = 0
    while len(rows) < cfg.num_examples:
        attempts += 1
        if attempts > cfg.num_examples * 20:
            raise RuntimeError(
                "Too many failed sampling attempts; check haystack/tokenizer settings"
            )
        ex_cfg, ex_insertion_position_pattern, ex_control_switch = _per_example_config(
            cfg, ex_idx=ex_idx
        )
        base_tokens, hay_meta = _sample_haystack_window(ex_cfg, tok, ex_idx)
        needles = _generate_needles(ex_cfg, tok, ex_idx)
        task_type = canonical_task_type(cfg.task_type)
        needles_for_insertion: list[dict[str, Any]] = []
        relevant_records: list[dict[str, Any]] = []
        control_relevant_records: list[dict[str, Any]] = []
        insertion_positions = (
            ex_insertion_position_pattern
            if ex_cfg.sentence_level_insertion or ex_cfg.word_level_insertion
            else insertion_positions_for_example(
                ex_cfg, ex_insertion_position_pattern, ex_idx
            )
        )
        for i, needle in enumerate(needles):
            is_control = ex_control_switch[i]
            requested_position = insertion_positions[i]
            is_inserted = requested_position is not None
            enriched = dict(needle)
            enriched["is_control"] = is_control
            enriched["requested_position"] = requested_position
            enriched["is_inserted"] = is_inserted
            if is_control:
                sampled = _sample_control_segment(
                    ex_cfg, tok, ex_idx, i, needle["token_length"]
                )
                enriched["inserted_tokens"] = sampled["tokens"]
                enriched["inserted_decoded_text"] = sampled["decoded_text"]
                enriched["control"] = {
                    k: v for k, v in sampled.items() if k != "tokens"
                }
            else:
                enriched["inserted_tokens"] = needle["tokens"]
                enriched["inserted_decoded_text"] = needle["decoded_text"]
                enriched["control"] = None

            if is_inserted:
                record = needle["record"] | {"needle_id": needle["needle_id"]}
                relevant_records.append(record)
                if not is_control:
                    control_relevant_records.append(record)
            needles_for_insertion.append(enriched)

        normal_needles_for_insertion = [
            dict(
                needle,
                is_control=False,
                requested_position=insertion_positions[i],
                is_inserted=insertion_positions[i] is not None,
                inserted_tokens=needle["tokens"],
                inserted_decoded_text=needle["decoded_text"],
                control=None,
            )
            for i, needle in enumerate(needles)
        ]
        if ex_cfg.sentence_level_insertion or ex_cfg.word_level_insertion:
            active_slots = tuple(pos is not None for pos in insertion_positions)
            base_text = tok.decode(base_tokens)
            text_insert_fn = (
                _insert_at_word_boundaries
                if ex_cfg.word_level_insertion
                else _insert_at_sentence_ends
            )
            decoded_context, realized = text_insert_fn(
                base_text,
                needles_for_insertion,
                active_slots,
                cfg=ex_cfg,
                tok=tok,
                ex_idx=ex_idx,
            )
            decoded_uncontrolled_context, uncontrolled_realized = text_insert_fn(
                base_text,
                normal_needles_for_insertion,
                active_slots,
                cfg=ex_cfg,
                tok=tok,
                ex_idx=ex_idx,
            )
            effective_insertion_positions: list[int | None] = [None] * ex_cfg.num_needles
            realized_by_id = {item["needle_id"]: item for item in realized}
            for needle in needles_for_insertion:
                insertion = realized_by_id.get(str(needle["needle_id"]))
                if insertion is None:
                    continue
                start = int(insertion["context_span_start"])
                effective_insertion_positions[int(needle["needle_id"][1:]) - 1] = start
                needle.update(
                    {
                        "requested_position": start,
                        "char_start": insertion["char_start"],
                        "char_end": insertion["char_end"],
                        "context_span_start": insertion["context_span_start"],
                        "context_span_end": insertion["context_span_end"],
                        "token_span_verified": insertion["token_span_verified"],
                        "token_span_source": insertion["token_span_source"],
                        "token_span_alignment": insertion.get(
                            "token_span_alignment"
                        ),
                        "text_level_insertion": insertion.get(
                            "text_level_insertion", False
                        ),
                        "text_insertion_mode": insertion.get("text_insertion_mode"),
                        "text_insertion_offset": insertion.get(
                            "text_insertion_offset"
                        ),
                        "text_insertion_candidate_count": insertion.get(
                            "text_insertion_candidate_count"
                        ),
                        "text_insertion_sampled_with_replacement": insertion.get(
                            "text_insertion_sampled_with_replacement"
                        ),
                    }
                )
                if ex_cfg.word_level_insertion:
                    needle.update(
                        {
                            "word_boundary_offset": insertion[
                                "word_boundary_offset"
                            ],
                            "word_boundary_candidate_count": insertion[
                                "word_boundary_candidate_count"
                            ],
                            "word_boundary_sampled_with_replacement": insertion[
                                "word_boundary_sampled_with_replacement"
                            ],
                        }
                    )
                if ex_cfg.sentence_level_insertion:
                    needle.update(
                        {
                            "sentence_delimiter_offset": insertion[
                                "sentence_delimiter_offset"
                            ],
                            "sentence_delimiter_candidate_count": insertion[
                                "sentence_delimiter_candidate_count"
                            ],
                            "sentence_delimiter_sampled_with_replacement": insertion[
                                "sentence_delimiter_sampled_with_replacement"
                            ],
                        }
                    )
        else:
            final_tokens, realized = _insert(
                base_tokens, needles_for_insertion, insertion_positions
            )
            normal_tokens, uncontrolled_realized = _insert(
                base_tokens, normal_needles_for_insertion, insertion_positions
            )
            realized = _annotate_token_insertions_with_context_spans(realized)
            uncontrolled_realized = _annotate_token_insertions_with_context_spans(
                uncontrolled_realized
            )
            decoded_context = tok.decode(final_tokens)
            decoded_uncontrolled_context = tok.decode(normal_tokens)
            effective_insertion_positions = list(insertion_positions)
        query_literal = (
            needles[0]["record"]["literal"] if task_type == "literal_count" else None
        )
        query = build_task_query(
            task_type,
            literal=query_literal,
            counting_needle_kind=counting_needle_kind,
            marker_text=cfg.marker_text,
        )
        gold_answer = build_gold_answer(relevant_records, task_type)
        control_gold_answer = build_control_gold_answer(
            control_relevant_records, task_type
        )
        literal_validation = None
        if task_type == "literal_count":
            literal_validation = _literal_validation_metadata(
                tok,
                decoded_uncontrolled_context,
                needles,
                inserted_count=len(relevant_records),
            )
        messages = _build_inputs(cfg, decoded_context, query, literal_text=query_literal)
        uncontrolled_messages = _build_inputs(
            cfg, decoded_uncontrolled_context, query, literal_text=query_literal
        )

        rows.append(
            {
                "schema_version": DATASET_SCHEMA_VERSION,
                "id": f"dynamic_niah_v2_{len(rows)+1}",
                "task_type": task_type,
                "counting_needle_kind": counting_needle_kind,
                "marker_text": cfg.marker_text if counting_needle_kind == "marker" else None,
                "tokenizer": {"name": cfg.tokenizer_name, "backend": tok.backend},
                "haystack": {
                    **hay_meta,
                    "target_haystack_tokens": cfg.target_haystack_tokens,
                    "base_tokens": base_tokens,
                },
                "needles": needles_for_insertion,
                "realized_insertions": realized,
                "uncontrolled_realized_insertions": uncontrolled_realized,
                "context": decoded_context,
                "uncontrolled_context": decoded_uncontrolled_context,
                "query": query,
                "messages": messages,
                "uncontrolled_messages": uncontrolled_messages,
                "gold_answer": gold_answer,
                "control_gold_answer": control_gold_answer,
                "relevant_records": relevant_records,
                "control_relevant_records": control_relevant_records,
                "literal_validation": literal_validation,
                "controls": {
                    "num_needles_configured": int(cfg.num_needles),
                    "num_max_needles": cfg.num_max_needles,
                    "target_num_needles": int(ex_cfg.num_needles),
                    "variable_num_needles": cfg.num_max_needles is not None,
                    "insertion_position_pattern": list(ex_insertion_position_pattern),
                    "insertion_positions": list(effective_insertion_positions),
                    "randomize_needle_insertion": bool(
                        cfg.randomize_needle_insertion
                    ),
                    "randomize_needle_seed": int(cfg.randomize_needle_seed),
                    "randomize_needle_margin": int(cfg.randomize_needle_margin),
                    "randomize_needle_min_separation": int(
                        cfg.randomize_needle_min_separation
                    ),
                    "control_switch": ex_control_switch,
                    "sentence_level_insertion": bool(ex_cfg.sentence_level_insertion),
                    "word_level_insertion": bool(ex_cfg.word_level_insertion),
                    **(
                        {
                            "text_insertion_min_fraction": float(
                                ex_cfg.text_insertion_min_fraction
                            ),
                            "text_insertion_max_fraction": float(
                                ex_cfg.text_insertion_max_fraction
                            ),
                        }
                        if (
                            ex_cfg.text_insertion_min_fraction != 0.0
                            or ex_cfg.text_insertion_max_fraction != 1.0
                        )
                        else {}
                    ),
                    "prompt_style": ex_cfg.prompt_style,
                    "counting_needle_kind": counting_needle_kind,
                    "marker_text": cfg.marker_text
                    if counting_needle_kind == "marker"
                    else None,
                    "uid_token_length": int(cfg.uid_token_length)
                    if task_type == "literal_count"
                    else None,
                    "thinking_mode": cfg.thinking_mode,
                    "seeds": {
                        "global_random_seed": cfg.global_random_seed,
                        "haystack_seed": cfg.haystack_seed,
                        "needle_seed": cfg.needle_seed,
                        "needle_seeds": cfg.needle_seeds or {},
                    },
                    "control_seed_derivation": "global_random_seed*100000007 + ex_idx*10007 + needle_idx*101 + 53",
                },
            }
        )
        ex_idx += 1
    return rows


def write_dynamic_niah_v2(
    rows: list[dict[str, Any]], cfg: DynamicNiahV2Config
) -> dict[str, str]:
    if cfg.output_dir is None or cfg.data_save_path is None:
        raise ValueError(
            "output_dir and data_save_path must be resolved before writing dynamic NIAH v2 data"
        )
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = Path(cfg.data_save_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    config_path = out / "config.used.json"
    config_path.write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    written = {"jsonl": str(jsonl_path), "config": str(config_path)}
    if DATASET_SCHEMA_PATH.exists():
        schema_path = out / "dataset.schema.json"
        shutil.copyfile(DATASET_SCHEMA_PATH, schema_path)
        written["schema"] = str(schema_path)
    return written
