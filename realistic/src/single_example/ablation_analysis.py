from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from dataset_generation.response_eval import build_response_result
from dataset_generation.dynamic_niah_v2 import DynamicNiahV2Config
from .single_example_analysis import SingleExamplePaths, _flat_input_ids


DEFAULT_ABLATION_CONFIG_PATH = Path("configs/ablation.json")


@dataclass(frozen=True)
class AblationConfig:
    """Configuration for single-example critical-token ablations."""

    num_critical_tokens: int = 10
    critical_token_calc_layer: int = 24
    ablation_random_seed: int = 12345
    haystack_dir: str = "data/haystacks/paul_graham"
    irrelevant_token_pool_size: int = 5000
    patterns: tuple[str, ...] = (
        "massive_activation",
        "attention_sink",
        "needle_sensitive",
        "massive_activation_all",
        "attention_sink_all",
        "needle_sensitive_all",
        "needle_span",
        "needle_tail",
    )
    attention_sink_score: str = "received_uniform_ratio"
    max_new_tokens: int | None = None
    temperature: float = 0.0
    replacement_sample_with_replacement: bool = False
    edge_exclusion_tokens: int = 5

    def with_overrides(
        self,
        *,
        num_critical_tokens: int | None = None,
        ablation_random_seed: int | None = None,
        critical_token_calc_layer: int | None = None,
    ) -> "AblationConfig":
        kwargs: dict[str, Any] = {}
        if num_critical_tokens is not None:
            kwargs["num_critical_tokens"] = int(num_critical_tokens)
        if ablation_random_seed is not None:
            kwargs["ablation_random_seed"] = int(ablation_random_seed)
        if critical_token_calc_layer is not None:
            kwargs["critical_token_calc_layer"] = int(critical_token_calc_layer)
        return replace(self, **kwargs)


def _tuple_patterns(value: Any) -> tuple[str, ...]:
    if value is None:
        return AblationConfig.patterns
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def load_ablation_config(
    path: str | Path = DEFAULT_ABLATION_CONFIG_PATH,
    *,
    num_critical_tokens: int | None = None,
    ablation_random_seed: int | None = None,
    critical_token_calc_layer: int | None = None,
) -> AblationConfig:
    """Load ablation defaults and apply optional notebook/runtime overrides."""

    config_path = Path(path)
    payload: dict[str, Any] = {}
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    if "patterns" in payload:
        payload["patterns"] = _tuple_patterns(payload["patterns"])
    cfg = AblationConfig(**payload)
    return cfg.with_overrides(
        num_critical_tokens=num_critical_tokens,
        ablation_random_seed=ablation_random_seed,
        critical_token_calc_layer=critical_token_calc_layer,
    )


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return path


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _warn_missing_selection_table(path: Path, pattern: str) -> None:
    print(
        f"WARNING: Missing selection table for {pattern}: {path}. "
        "Returning no critical tokens for this pattern. Run Q/K outlier analysis "
        "first if you need massive-activation or attention-sink ablations.",
        flush=True,
    )


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
        if parsed != parsed:
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_input_metadata(run_dir: str | Path, example_id: int) -> dict[str, Any]:
    path = Path(run_dir) / "generate_data" / f"inputs_{int(example_id)}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing single-example input metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def needle_positions_from_segments(needle_segments: Sequence[dict[str, Any]]) -> set[int]:
    positions: set[int] = set()
    for segment in needle_segments:
        if "positions" in segment and isinstance(segment["positions"], list):
            positions.update(int(pos) for pos in segment["positions"])
            continue
        start = int(segment["start"])
        end = int(segment["end"])
        positions.update(range(start, end))
    return positions


def _is_selectable_critical_position(
    position: int,
    needle_positions: set[int],
    seq_len: int,
    *,
    edge_exclusion_tokens: int = 5,
) -> bool:
    """Return whether a critical-token candidate is outside needles and sequence edges."""

    pos = int(position)
    seq_len = int(seq_len)
    edge = max(0, int(edge_exclusion_tokens))
    return edge <= pos < seq_len - edge and pos not in needle_positions


def _is_outside_needles(position: int, needle_positions: set[int], seq_len: int) -> bool:
    """Backward-compatible needle-only predicate; prefer _is_selectable_critical_position."""

    return 0 <= int(position) < int(seq_len) and int(position) not in needle_positions


def _token_text(tokenizer: Any, token_id: int) -> str:
    if tokenizer is None:
        return f"<id:{int(token_id)}>"
    try:
        return str(tokenizer.convert_ids_to_tokens(int(token_id)))
    except Exception:
        try:
            return str(tokenizer.decode([int(token_id)]))
        except Exception:
            return f"<id:{int(token_id)}>"


def _record_token(
    *,
    pattern: str,
    rank: int,
    position: int,
    input_ids: Sequence[int],
    tokenizer: Any,
    score_name: str | None = None,
    score: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    token_id = int(input_ids[int(position)])
    row = {
        "pattern": pattern,
        "rank": int(rank),
        "position": int(position),
        "token_id": token_id,
        "token": _token_text(tokenizer, token_id),
        "score_name": score_name or "",
        "score": "" if score is None else float(score),
    }
    if extra:
        row.update(extra)
    return row


def _dedup_ranked_positions(rows: Iterable[tuple[int, float | None, dict[str, Any]]]) -> list[tuple[int, float | None, dict[str, Any]]]:
    seen: set[int] = set()
    out: list[tuple[int, float | None, dict[str, Any]]] = []
    for pos, score, extra in rows:
        pos = int(pos)
        if pos in seen:
            continue
        seen.add(pos)
        out.append((pos, score, extra))
    return out


def select_massive_activation_tokens(
    *,
    run_dir: str | Path,
    example_id: int,
    cfg: AblationConfig,
    input_ids: Sequence[int],
    needle_positions: set[int],
    tokenizer: Any = None,
) -> list[dict[str, Any]]:
    table_path = Path(run_dir) / "tables" / "massive_tokens_outside_needles_all.csv"
    if not table_path.exists():
        _warn_missing_selection_table(table_path, "massive_activation")
        return []
    rows = _read_csv_dicts(table_path)
    candidates: list[tuple[int, float | None, dict[str, Any]]] = []
    for row in rows:
        if _int_or_none(row.get("example_idx")) != int(example_id):
            continue
        if _int_or_none(row.get("layer")) != int(cfg.critical_token_calc_layer):
            continue
        pos = _int_or_none(row.get("position"))
        score = _float_or_none(row.get("norm_ratio_to_median"))
        if pos is None or score is None:
            continue
        if not _is_selectable_critical_position(
            pos,
            needle_positions,
            len(input_ids),
            edge_exclusion_tokens=cfg.edge_exclusion_tokens,
        ):
            continue
        candidates.append((pos, score, {"layer": cfg.critical_token_calc_layer}))
    candidates.sort(key=lambda item: (-(item[1] if item[1] is not None else float("-inf")), item[0]))
    return [
        _record_token(
            pattern="massive_activation",
            rank=rank,
            position=pos,
            input_ids=input_ids,
            tokenizer=tokenizer,
            score_name="norm_ratio_to_median",
            score=score,
            extra=extra,
        )
        for rank, (pos, score, extra) in enumerate(
            _dedup_ranked_positions(candidates)[: cfg.num_critical_tokens], start=1
        )
    ]


def select_attention_sink_tokens(
    *,
    run_dir: str | Path,
    example_id: int,
    cfg: AblationConfig,
    input_ids: Sequence[int],
    needle_positions: set[int],
    tokenizer: Any = None,
) -> list[dict[str, Any]]:
    table_path = Path(run_dir) / "tables" / "attention_sinks_topk.csv"
    rows = _read_csv_dicts(table_path)
    best_by_pos: dict[int, tuple[float, dict[str, Any]]] = {}
    score_col = cfg.attention_sink_score
    for row in rows:
        if _int_or_none(row.get("example_idx")) != int(example_id):
            continue
        if _int_or_none(row.get("layer")) != int(cfg.critical_token_calc_layer):
            continue
        pos = _int_or_none(row.get("position"))
        score = _float_or_none(row.get(score_col))
        if pos is None or score is None:
            continue
        if not _is_selectable_critical_position(
            pos,
            needle_positions,
            len(input_ids),
            edge_exclusion_tokens=cfg.edge_exclusion_tokens,
        ):
            continue
        head = _int_or_none(row.get("head"))
        current = best_by_pos.get(pos)
        if current is None or score > current[0]:
            best_by_pos[pos] = (score, {"layer": cfg.critical_token_calc_layer, "head": head})
    candidates = [(pos, score, extra) for pos, (score, extra) in best_by_pos.items()]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return [
        _record_token(
            pattern="attention_sink",
            rank=rank,
            position=pos,
            input_ids=input_ids,
            tokenizer=tokenizer,
            score_name=score_col,
            score=score,
            extra=extra,
        )
        for rank, (pos, score, extra) in enumerate(candidates[: cfg.num_critical_tokens], start=1)
    ]


def select_needle_sensitive_tokens(
    *,
    run_dir: str | Path,
    example_id: int,
    cfg: AblationConfig,
    input_ids: Sequence[int],
    needle_positions: set[int],
    tokenizer: Any = None,
) -> list[dict[str, Any]]:
    table_path = Path(run_dir) / "tables" / f"inputs_{int(example_id)}_measurements.csv"
    rows = _read_csv_dicts(table_path)
    all_rows: list[dict[str, Any]] = []
    candidates: list[tuple[int, float | None, dict[str, Any]]] = []
    for row in rows:
        if _int_or_none(row.get("layer")) != int(cfg.critical_token_calc_layer):
            continue
        pos = _int_or_none(row.get("position"))
        score = _float_or_none(row.get("cosine_similarity"))
        if pos is None or score is None:
            continue
        if not _is_selectable_critical_position(
            pos,
            needle_positions,
            len(input_ids),
            edge_exclusion_tokens=cfg.edge_exclusion_tokens,
        ):
            continue
        record = _record_token(
            pattern="needle_sensitive",
            rank=0,
            position=pos,
            input_ids=input_ids,
            tokenizer=tokenizer,
            score_name="cosine_similarity",
            score=score,
            extra={"layer": cfg.critical_token_calc_layer},
        )
        all_rows.append(record)
        candidates.append((pos, score, {"layer": cfg.critical_token_calc_layer}))
    ablation_dir = Path(run_dir) / "tables" / "ablation"
    _write_csv(
        ablation_dir / "needle_sensitive_tokens_outside_needles_all.csv",
        all_rows,
        fieldnames=["pattern", "rank", "position", "token_id", "token", "score_name", "score", "layer"],
    )
    candidates.sort(key=lambda item: ((item[1] if item[1] is not None else float("inf")), item[0]))
    return [
        _record_token(
            pattern="needle_sensitive",
            rank=rank,
            position=pos,
            input_ids=input_ids,
            tokenizer=tokenizer,
            score_name="cosine_similarity",
            score=score,
            extra=extra,
        )
        for rank, (pos, score, extra) in enumerate(
            _dedup_ranked_positions(candidates)[: cfg.num_critical_tokens], start=1
        )
    ]


def select_massive_activation_tokens_all(
    *,
    run_dir: str | Path,
    example_id: int,
    cfg: AblationConfig,
    input_ids: Sequence[int],
    tokenizer: Any = None,
) -> list[dict[str, Any]]:
    """Rank massive-activation tokens over all in-range sequence positions."""

    table_path = Path(run_dir) / "tables" / "massive_tokens_all.csv"
    if not table_path.exists():
        _warn_missing_selection_table(table_path, "massive_activation_all")
        return []
    rows = _read_csv_dicts(table_path)
    candidates: list[tuple[int, float | None, dict[str, Any]]] = []
    for row in rows:
        if _int_or_none(row.get("example_idx")) != int(example_id):
            continue
        if _int_or_none(row.get("layer")) != int(cfg.critical_token_calc_layer):
            continue
        pos = _int_or_none(row.get("position"))
        score = _float_or_none(row.get("norm_ratio_to_median"))
        if pos is None or score is None or not (0 <= pos < len(input_ids)):
            continue
        candidates.append((pos, score, {"layer": cfg.critical_token_calc_layer}))
    candidates.sort(key=lambda item: (-(item[1] if item[1] is not None else float("-inf")), item[0]))
    return [
        _record_token(
            pattern="massive_activation_all",
            rank=rank,
            position=pos,
            input_ids=input_ids,
            tokenizer=tokenizer,
            score_name="norm_ratio_to_median",
            score=score,
            extra=extra,
        )
        for rank, (pos, score, extra) in enumerate(
            _dedup_ranked_positions(candidates)[: cfg.num_critical_tokens], start=1
        )
    ]


def select_attention_sink_tokens_all(
    *,
    run_dir: str | Path,
    example_id: int,
    cfg: AblationConfig,
    input_ids: Sequence[int],
    tokenizer: Any = None,
) -> list[dict[str, Any]]:
    """Rank attention-sink tokens over all in-range sequence positions."""

    table_path = Path(run_dir) / "tables" / "attention_sinks_topk.csv"
    rows = _read_csv_dicts(table_path)
    best_by_pos: dict[int, tuple[float, dict[str, Any]]] = {}
    score_col = cfg.attention_sink_score
    for row in rows:
        if _int_or_none(row.get("example_idx")) != int(example_id):
            continue
        if _int_or_none(row.get("layer")) != int(cfg.critical_token_calc_layer):
            continue
        pos = _int_or_none(row.get("position"))
        score = _float_or_none(row.get(score_col))
        if pos is None or score is None or not (0 <= pos < len(input_ids)):
            continue
        head = _int_or_none(row.get("head"))
        current = best_by_pos.get(pos)
        if current is None or score > current[0]:
            best_by_pos[pos] = (score, {"layer": cfg.critical_token_calc_layer, "head": head})
    candidates = [(pos, score, extra) for pos, (score, extra) in best_by_pos.items()]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return [
        _record_token(
            pattern="attention_sink_all",
            rank=rank,
            position=pos,
            input_ids=input_ids,
            tokenizer=tokenizer,
            score_name=score_col,
            score=score,
            extra=extra,
        )
        for rank, (pos, score, extra) in enumerate(candidates[: cfg.num_critical_tokens], start=1)
    ]


def select_needle_sensitive_tokens_all(
    *,
    run_dir: str | Path,
    example_id: int,
    cfg: AblationConfig,
    input_ids: Sequence[int],
    tokenizer: Any = None,
) -> list[dict[str, Any]]:
    """Rank needle-sensitive tokens over all in-range sequence positions."""

    table_path = Path(run_dir) / "tables" / f"inputs_{int(example_id)}_measurements.csv"
    rows = _read_csv_dicts(table_path)
    candidates: list[tuple[int, float | None, dict[str, Any]]] = []
    for row in rows:
        if _int_or_none(row.get("layer")) != int(cfg.critical_token_calc_layer):
            continue
        pos = _int_or_none(row.get("position"))
        score = _float_or_none(row.get("cosine_similarity"))
        if pos is None or score is None or not (0 <= pos < len(input_ids)):
            continue
        candidates.append((pos, score, {"layer": cfg.critical_token_calc_layer}))
    candidates.sort(key=lambda item: ((item[1] if item[1] is not None else float("inf")), item[0]))
    return [
        _record_token(
            pattern="needle_sensitive_all",
            rank=rank,
            position=pos,
            input_ids=input_ids,
            tokenizer=tokenizer,
            score_name="cosine_similarity",
            score=score,
            extra=extra,
        )
        for rank, (pos, score, extra) in enumerate(
            _dedup_ranked_positions(candidates)[: cfg.num_critical_tokens], start=1
        )
    ]


def select_needle_span_tokens(
    *,
    needle_segments: Sequence[dict[str, Any]],
    input_ids: Sequence[int],
    tokenizer: Any = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return one uncapped critical-token pattern for each needle span."""

    by_pattern: dict[str, list[dict[str, Any]]] = {}
    for needle_idx, segment in enumerate(sorted(needle_segments, key=lambda item: int(item["start"]))):
        pattern = f"needle_span_{needle_idx}"
        positions = segment.get("positions")
        if not isinstance(positions, list):
            positions = list(range(int(segment["start"]), int(segment["end"])))
        rows = []
        for rank, pos in enumerate([int(p) for p in positions if 0 <= int(p) < len(input_ids)], start=1):
            rows.append(
                _record_token(
                    pattern=pattern,
                    rank=rank,
                    position=pos,
                    input_ids=input_ids,
                    tokenizer=tokenizer,
                    score_name="needle_span_position",
                    score=rank,
                    extra={"needle_id": segment.get("needle_id", needle_idx)},
                )
            )
        by_pattern[pattern] = rows
    return by_pattern


def select_needle_tail_tokens(
    *,
    needle_segments: Sequence[dict[str, Any]],
    cfg: AblationConfig,
    input_ids: Sequence[int],
    needle_positions: set[int] | None = None,
    tokenizer: Any = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return one K-token adjacent-tail pattern after each needle span."""

    by_pattern: dict[str, list[dict[str, Any]]] = {}
    for needle_idx, segment in enumerate(sorted(needle_segments, key=lambda item: int(item["start"]))):
        pattern = f"needle_tail_{needle_idx}"
        positions = segment.get("positions")
        if isinstance(positions, list) and positions:
            tail_start = max(int(pos) for pos in positions) + 1
        else:
            tail_start = int(segment["end"])
        rows = []
        for rank, pos in enumerate(range(tail_start, tail_start + int(cfg.num_critical_tokens)), start=1):
            if not 0 <= int(pos) < len(input_ids):
                continue
            rows.append(
                _record_token(
                    pattern=pattern,
                    rank=rank,
                    position=pos,
                    input_ids=input_ids,
                    tokenizer=tokenizer,
                    score_name="needle_tail_position",
                    score=rank,
                    extra={"needle_id": segment.get("needle_id", needle_idx)},
                )
            )
        by_pattern[pattern] = rows
    return by_pattern


def select_critical_tokens(
    *,
    run_dir: str | Path,
    example_id: int,
    cfg: AblationConfig,
    input_ids: Sequence[int],
    needle_segments: Sequence[dict[str, Any]],
    tokenizer: Any = None,
) -> dict[str, list[dict[str, Any]]]:
    needle_positions = needle_positions_from_segments(needle_segments)
    selected: dict[str, list[dict[str, Any]]] = {}
    patterns = set(cfg.patterns)
    if "massive_activation" in patterns:
        selected["massive_activation"] = select_massive_activation_tokens(
            run_dir=run_dir,
            example_id=example_id,
            cfg=cfg,
            input_ids=input_ids,
            needle_positions=needle_positions,
            tokenizer=tokenizer,
        )
    if "attention_sink" in patterns:
        selected["attention_sink"] = select_attention_sink_tokens(
            run_dir=run_dir,
            example_id=example_id,
            cfg=cfg,
            input_ids=input_ids,
            needle_positions=needle_positions,
            tokenizer=tokenizer,
        )
    if "needle_sensitive" in patterns:
        selected["needle_sensitive"] = select_needle_sensitive_tokens(
            run_dir=run_dir,
            example_id=example_id,
            cfg=cfg,
            input_ids=input_ids,
            needle_positions=needle_positions,
            tokenizer=tokenizer,
        )
    if "massive_activation_all" in patterns:
        selected["massive_activation_all"] = select_massive_activation_tokens_all(
            run_dir=run_dir,
            example_id=example_id,
            cfg=cfg,
            input_ids=input_ids,
            tokenizer=tokenizer,
        )
    if "attention_sink_all" in patterns:
        selected["attention_sink_all"] = select_attention_sink_tokens_all(
            run_dir=run_dir,
            example_id=example_id,
            cfg=cfg,
            input_ids=input_ids,
            tokenizer=tokenizer,
        )
    if "needle_sensitive_all" in patterns:
        selected["needle_sensitive_all"] = select_needle_sensitive_tokens_all(
            run_dir=run_dir,
            example_id=example_id,
            cfg=cfg,
            input_ids=input_ids,
            tokenizer=tokenizer,
        )
    if "needle_span" in patterns:
        selected.update(
            select_needle_span_tokens(
                needle_segments=needle_segments,
                input_ids=input_ids,
                tokenizer=tokenizer,
            )
        )
    if "needle_tail" in patterns:
        selected.update(
            select_needle_tail_tokens(
                needle_segments=needle_segments,
                cfg=cfg,
                input_ids=input_ids,
                needle_positions=needle_positions,
                tokenizer=tokenizer,
            )
        )
    return selected


def save_critical_tokens(
    *,
    run_dir: str | Path,
    selected: dict[str, list[dict[str, Any]]],
    cfg: AblationConfig,
) -> dict[str, Path]:
    ablation_dir = Path(run_dir) / "tables" / "ablation"
    flat_rows = [row for rows in selected.values() for row in rows]
    fields = [
        "pattern",
        "rank",
        "position",
        "token_id",
        "token",
        "score_name",
        "score",
        "layer",
        "head",
        "needle_id",
    ]
    return {
        "json": _write_json(
            ablation_dir / "critical_tokens.json",
            {
                "config": asdict(cfg),
                "patterns": selected,
            },
        ),
        "csv": _write_csv(ablation_dir / "critical_tokens.csv", flat_rows, fields),
    }


def build_irrelevant_token_pool(
    *,
    tokenizer: Any,
    haystack_dir: str | Path,
    pool_size: int,
    seed: int,
) -> list[int]:
    """Build a deterministic pool of replacement token IDs from haystack text."""

    root = Path(haystack_dir)
    if not root.exists():
        raise FileNotFoundError(f"Haystack directory does not exist: {root}")
    token_ids: list[int] = []
    special_ids = set(int(x) for x in getattr(tokenizer, "all_special_ids", []) or [])
    for path in sorted(root.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        encoded = tokenizer(text, add_special_tokens=False)
        ids = encoded.get("input_ids", encoded) if hasattr(encoded, "get") else encoded
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        token_ids.extend(int(tok) for tok in ids if int(tok) not in special_ids)
        if len(token_ids) >= max(int(pool_size) * 3, int(pool_size)):
            break
    if not token_ids:
        raise ValueError(f"No replacement tokens found under {root}")
    rng = random.Random(int(seed))
    rng.shuffle(token_ids)
    return token_ids[: min(int(pool_size), len(token_ids))]


def replacement_tokens_for_k(
    *,
    pool: Sequence[int],
    k: int,
    seed: int,
    pattern: str,
    sample_with_replacement: bool = False,
) -> list[int]:
    if k <= 0:
        return []
    if not pool:
        raise ValueError("Replacement pool is empty")
    rng = random.Random(f"{int(seed)}:{pattern}:{int(k)}")
    if sample_with_replacement or k > len(pool):
        return [int(rng.choice(pool)) for _ in range(k)]
    return [int(x) for x in rng.sample(list(pool), k)]


def make_ablated_input_ids(
    input_ids: torch.Tensor | Sequence[int],
    positions: Sequence[int],
    replacement_token_ids: Sequence[int],
) -> torch.Tensor:
    ablated = torch.as_tensor(input_ids, dtype=torch.long).clone()
    if ablated.ndim == 1:
        ablated = ablated.unsqueeze(0)
    if len(positions) != len(replacement_token_ids):
        raise ValueError(
            f"positions and replacement_token_ids must have same length; got {len(positions)} and {len(replacement_token_ids)}"
        )
    seq_len = int(ablated.shape[1])
    for pos, token_id in zip(positions, replacement_token_ids, strict=True):
        pos = int(pos)
        if pos < 0 or pos >= seq_len:
            raise IndexError(f"Ablation position {pos} outside sequence length {seq_len}")
        ablated[0, pos] = int(token_id)
    return ablated


def _resolve_max_new_tokens(ablation_cfg: AblationConfig, dynamic_cfg: DynamicNiahV2Config | None) -> int:
    if ablation_cfg.max_new_tokens is not None:
        return int(ablation_cfg.max_new_tokens)
    if dynamic_cfg is not None and dynamic_cfg.max_new_tokens is not None:
        return int(dynamic_cfg.max_new_tokens)
    if dynamic_cfg is not None and dynamic_cfg.thinking_mode:
        return 1024
    return 64


def generate_from_input_ids(
    *,
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    ablation_cfg: AblationConfig,
    dynamic_cfg: DynamicNiahV2Config | None = None,
) -> str:
    input_device = model.get_input_embeddings().weight.device
    model_inputs = {
        "input_ids": input_ids.to(input_device),
        "attention_mask": torch.ones_like(input_ids, dtype=torch.long).to(input_device),
    }
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": _resolve_max_new_tokens(ablation_cfg, dynamic_cfg),
        "do_sample": float(ablation_cfg.temperature) > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if float(ablation_cfg.temperature) > 0:
        generation_kwargs["temperature"] = float(ablation_cfg.temperature)
    with torch.no_grad():
        out = model.generate(**model_inputs, **generation_kwargs)
    prompt_len = int(model_inputs["input_ids"].shape[1])
    gen_ids = out[0][prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def _score_row(row: dict[str, Any], model_output: str) -> dict[str, Any]:
    result = build_response_result(row, model_output)
    return result | {"accuracy": 1.0 if result.get("exact_match") else 0.0}


def run_ablation_generation(
    *,
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    example_id: int,
    input_ids: torch.Tensor | Sequence[int],
    selected: dict[str, list[dict[str, Any]]],
    replacement_pool: Sequence[int],
    cfg: AblationConfig,
    out_dir: str | Path,
    dynamic_cfg: DynamicNiahV2Config | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = Path(out_dir)
    generation_dir = out_dir / "generations"
    generation_dir.mkdir(parents=True, exist_ok=True)
    base_input = torch.as_tensor(input_ids, dtype=torch.long)
    if base_input.ndim == 1:
        base_input = base_input.unsqueeze(0)

    prediction_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []

    baseline_output = generate_from_input_ids(
        model=model,
        tokenizer=tokenizer,
        input_ids=base_input,
        ablation_cfg=cfg,
        dynamic_cfg=dynamic_cfg,
    )
    baseline_score = _score_row(row, baseline_output)
    (generation_dir / "baseline.txt").write_text(baseline_output, encoding="utf-8")
    baseline_result = {
        "example_id": int(example_id),
        "row_id": row.get("id"),
        "pattern": "baseline",
        "k": 0,
        "ablated_positions": json.dumps([]),
        "original_token_ids": json.dumps([]),
        "replacement_token_ids": json.dumps([]),
        "replacement_tokens": json.dumps([]),
        "model_output_text": baseline_output,
        "parse_mode": baseline_score.get("parse_mode"),
        "exact_match": bool(baseline_score.get("exact_match")),
        "accuracy": float(baseline_score["accuracy"]),
    }
    result_rows.append(baseline_result)
    prediction_rows.append({**baseline_score, **baseline_result})

    for pattern, tokens in selected.items():
        max_k = min(int(cfg.num_critical_tokens), len(tokens))
        for k in range(1, max_k + 1):
            chosen = tokens[:k]
            positions = [int(item["position"]) for item in chosen]
            original_token_ids = [int(item["token_id"]) for item in chosen]
            replacement_ids = replacement_tokens_for_k(
                pool=replacement_pool,
                k=k,
                seed=cfg.ablation_random_seed,
                pattern=pattern,
                sample_with_replacement=cfg.replacement_sample_with_replacement,
            )
            ablated_input = make_ablated_input_ids(base_input, positions, replacement_ids)
            output = generate_from_input_ids(
                model=model,
                tokenizer=tokenizer,
                input_ids=ablated_input,
                ablation_cfg=cfg,
                dynamic_cfg=dynamic_cfg,
            )
            score = _score_row(row, output)
            replacement_tokens = [_token_text(tokenizer, tok) for tok in replacement_ids]
            generation_path = generation_dir / f"{pattern}_k{k}.txt"
            generation_path.write_text(output, encoding="utf-8")
            result = {
                "example_id": int(example_id),
                "row_id": row.get("id"),
                "pattern": pattern,
                "k": int(k),
                "ablated_positions": json.dumps(positions),
                "original_token_ids": json.dumps(original_token_ids),
                "replacement_token_ids": json.dumps(replacement_ids),
                "replacement_tokens": json.dumps(replacement_tokens, ensure_ascii=False),
                "model_output_text": output,
                "parse_mode": score.get("parse_mode"),
                "exact_match": bool(score.get("exact_match")),
                "accuracy": float(score["accuracy"]),
            }
            result_rows.append(result)
            prediction_rows.append({**score, **result, "generation_path": str(generation_path)})
    return prediction_rows, result_rows


def run_single_example_ablation(
    *,
    paths: SingleExamplePaths,
    row: dict[str, Any],
    example_id: int,
    model: Any,
    tokenizer: Any,
    uncontrolled_input_ids: torch.Tensor | Sequence[int] | None = None,
    needle_segments: Sequence[dict[str, Any]] | None = None,
    config_path: str | Path = DEFAULT_ABLATION_CONFIG_PATH,
    num_critical_tokens: int | None = None,
    ablation_random_seed: int | None = None,
    critical_token_calc_layer: int | None = None,
    dynamic_cfg: DynamicNiahV2Config | None = None,
) -> dict[str, Any]:
    """Select critical tokens, run unablated and ablated generations, and save tables."""

    cfg = load_ablation_config(
        config_path,
        num_critical_tokens=num_critical_tokens,
        ablation_random_seed=ablation_random_seed,
        critical_token_calc_layer=critical_token_calc_layer,
    )
    if uncontrolled_input_ids is None or needle_segments is None:
        metadata = load_input_metadata(paths.run_dir, example_id)
        if uncontrolled_input_ids is None:
            uncontrolled_input_ids = metadata["uncontrolled_input_ids"]
        if needle_segments is None:
            needle_segments = metadata["needle_segments"]
    input_ids_list = _flat_input_ids(uncontrolled_input_ids)
    selected = select_critical_tokens(
        run_dir=paths.run_dir,
        example_id=example_id,
        cfg=cfg,
        input_ids=input_ids_list,
        needle_segments=list(needle_segments),
        tokenizer=tokenizer,
    )
    critical_paths = save_critical_tokens(run_dir=paths.run_dir, selected=selected, cfg=cfg)
    replacement_pool = build_irrelevant_token_pool(
        tokenizer=tokenizer,
        haystack_dir=cfg.haystack_dir,
        pool_size=cfg.irrelevant_token_pool_size,
        seed=cfg.ablation_random_seed,
    )
    ablation_dir = paths.tables_dir / "ablation"
    prediction_rows, result_rows = run_ablation_generation(
        model=model,
        tokenizer=tokenizer,
        row=row,
        example_id=example_id,
        input_ids=torch.tensor([input_ids_list], dtype=torch.long),
        selected=selected,
        replacement_pool=replacement_pool,
        cfg=cfg,
        out_dir=ablation_dir,
        dynamic_cfg=dynamic_cfg,
    )
    predictions_path = _write_jsonl(ablation_dir / "ablation_predictions.jsonl", prediction_rows)
    result_fields = [
        "example_id",
        "row_id",
        "pattern",
        "k",
        "ablated_positions",
        "original_token_ids",
        "replacement_token_ids",
        "replacement_tokens",
        "model_output_text",
        "parse_mode",
        "exact_match",
        "accuracy",
    ]
    results_path = _write_csv(ablation_dir / "ablation_results.csv", result_rows, result_fields)
    summary = {
        "config": asdict(cfg),
        "example_id": int(example_id),
        "row_id": row.get("id"),
        "baseline": next((r for r in result_rows if r["pattern"] == "baseline"), None),
        "num_patterns": len(selected),
        "pattern_lengths": {pattern: len(rows) for pattern, rows in selected.items()},
        "num_result_rows": len(result_rows),
        "critical_tokens_json": str(critical_paths["json"]),
        "critical_tokens_csv": str(critical_paths["csv"]),
        "predictions_path": str(predictions_path),
        "results_path": str(results_path),
    }
    summary_path = _write_json(ablation_dir / "ablation_summary.json", summary)
    summary["summary_path"] = str(summary_path)
    return summary


def summarize_ablation_results_all(
    *,
    run_dir: str | Path,
    example_dirs: Sequence[str | Path] | None = None,
    output_name: str = "ablation_results_all.csv",
) -> Path:
    """Average per-example ablation accuracy by pattern and k for ALL_EXAMPLES runs."""

    root = Path(run_dir)
    if example_dirs is None:
        candidates = sorted(root.glob("example_id_*"), key=lambda path: path.name)
    else:
        candidates = [Path(path) for path in example_dirs]
    grouped: dict[tuple[str, int], list[float]] = {}
    for example_dir in candidates:
        results_path = example_dir / "tables" / "ablation" / "ablation_results.csv"
        for row in _read_csv_dicts(results_path):
            pattern = str(row.get("pattern", ""))
            k = _int_or_none(row.get("k"))
            accuracy = _float_or_none(row.get("accuracy"))
            if not pattern or k is None or accuracy is None:
                continue
            grouped.setdefault((pattern, k), []).append(float(accuracy))
    rows = [
        {
            "Pattern": pattern,
            "k": k,
            "accuracy": sum(values) / len(values),
        }
        for (pattern, k), values in grouped.items()
        if values
    ]
    rows.sort(key=lambda row: (row["Pattern"] != "baseline", row["Pattern"], int(row["k"])))
    return _write_csv(root / output_name, rows, ["Pattern", "k", "accuracy"])
