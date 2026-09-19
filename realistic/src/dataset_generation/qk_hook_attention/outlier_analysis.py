"""Notebook-friendly Q/K-cache outlier analysis utilities for Qwen3 runs.

The functions in this module are intended to be called from
``notebooks/analysis_hidden_states_v4.ipynb`` after the Dynamic NIAH hidden-state
run has written its canonical ``RUN_DIR``.  They keep heavyweight work out of
notebook cells while reusing the offline Q/K reconstruction code in
``analyze_qk_qwen3.py``.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset_generation.chat_templates import apply_generation_chat_template
from dataset_generation.dynamic_niah_v2 import (
    DynamicNiahV2Config,
    build_uncontrolled_context,
    dynamic_niah_v2_config_kwargs,
)
from dataset_generation.hidden_state_analysis import (
    build_outside_segments_mask,
    expand_needle_segments,
    load_hidden_state_records,
    stored_layer_index,
)
from dataset_generation.niah_prompt_utils import (
    build_messages_easier,
    build_messages_vanilla,
    response_schema_for_task,
)
from dataset_generation.qk_hook_attention.analyze_qk_qwen3 import (
    COMPUTE_DTYPES,
    average_window_received,
    compute_query_block_stats,
    load_cache_analysis_spec,
    load_cache_metadata,
    load_cache_tokens,
    load_json,
    load_tensor,
    reconstruct_single_head_qk,
)
from dataset_generation.qk_hook_attention.capture_qk_qwen3 import (
    DTYPE_MAP,
    SAVE_DTYPE_MAP,
    capture_qk_cache,
)
from dataset_generation.response_eval import canonical_task_type


@dataclass(frozen=True)
class QKOutlierAnalysisConfig:
    """Configuration for the run-level Q/K outlier analysis stage."""

    analysis_name: str = "qk_outlier_analysis"
    notebook: str = "notebooks/analysis_hidden_states_v4.ipynb"
    model: str = "Qwen/Qwen3-8B"
    run_dir: str = ""
    layers: tuple[int, ...] = ()
    heads: tuple[int, ...] | None = None
    uncontrolled_prompts_only: bool = True
    massive_norm_ratio_threshold: float = 10.0
    massive_top_k_per_layer: int = 50
    n_critical_edge_tokens: int = 10
    n_after_needle: int = 10
    qk_cache_dir: str = "tensors/qk_cache"
    attention_stats_dir: str = "tensors/attention_stats"
    massive_activations_dir: str = "tensors/massive_activations"
    tables_dir: str = "tables"
    figures_dir: str = "figures"
    compute_dtype: str = "fp32"
    device: str = "cuda"
    key_block_size: int = 8192
    query_block_size: int = 64
    topk: int = 32
    local_windows: tuple[int, ...] = (32, 128)
    capture_attn_implementation: str = "sdpa"
    capture_model_dtype: str = "bf16"
    capture_save_dtype: str = "bf16"

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["layers"] = list(self.layers)
        payload["heads"] = None if self.heads is None else list(self.heads)
        payload["local_windows"] = list(self.local_windows)
        return payload


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, payload: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return out


def _write_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str] | None = None
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return path


def _dynamic_cfg_from_run(run_dir: Path) -> DynamicNiahV2Config:
    for candidate in [
        run_dir / "run_metadata.json",
        run_dir / "analyze_hidden_states_config.json",
    ]:
        if not candidate.exists():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        resolved = payload.get("resolved_config")
        if isinstance(resolved, dict):
            return DynamicNiahV2Config(**dynamic_niah_v2_config_kwargs(resolved))
    raise FileNotFoundError(
        f"Could not find resolved_config in {run_dir / 'run_metadata.json'} or "
        f"{run_dir / 'analyze_hidden_states_config.json'}"
    )


def _model_name_from_run(run_dir: Path, fallback: str) -> str:
    for candidate in [
        run_dir / "analyze_hidden_states_config.json",
        run_dir / "run_metadata.json",
    ]:
        if candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            model = payload.get("model")
            if model:
                return str(model)
    return fallback


def _literal_text_from_row(row: dict[str, Any]) -> str | None:
    if row.get("task_type") != "literal_count":
        return None
    needles = row.get("needles") or []
    if not needles:
        return None
    record = needles[0].get("record") or {}
    literal = record.get("literal")
    return str(literal) if literal is not None else None


def build_uncontrolled_prompt_text(
    cfg: DynamicNiahV2Config, row: dict[str, Any], tokenizer=None
) -> str:
    """Render the exact all-needle prompt text used for uncontrolled analysis."""

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer_name,
            trust_remote_code=cfg.trust_remote_code,
            cache_dir=cfg.cache_dir,
        )
    response_schema = response_schema_for_task(canonical_task_type(cfg.task_type))
    context = build_uncontrolled_context(row)
    literal_text = _literal_text_from_row(row)
    if cfg.prompt_style == "easier":
        messages = build_messages_easier(
            context,
            row["query"],
            thinking_mode=cfg.thinking_mode,
            response_schema=response_schema,
            task_type=canonical_task_type(cfg.task_type),
            counting_needle_kind=cfg.counting_needle_kind,
            marker_text=cfg.marker_text,
            literal_text=literal_text,
        )
    else:
        messages = build_messages_vanilla(
            context,
            row["query"],
            thinking_mode=cfg.thinking_mode,
            response_schema=response_schema,
            task_type=canonical_task_type(cfg.task_type),
            counting_needle_kind=cfg.counting_needle_kind,
            marker_text=cfg.marker_text,
            literal_text=literal_text,
        )
    return apply_generation_chat_template(
        tokenizer, messages, thinking_mode=cfg.thinking_mode
    )


def needle_markers(row: dict[str, Any]) -> dict[str, str]:
    markers: dict[str, str] = {}
    for idx, needle in enumerate(row.get("needles", [])):
        text = (
            needle.get("decoded_text")
            or needle.get("text")
            or needle.get("inserted_text")
        )
        if text:
            markers[f"needle_{idx}"] = str(text)
    return markers


def _qk_root(run_dir: Path, analysis_cfg: QKOutlierAnalysisConfig) -> Path:
    return run_dir / analysis_cfg.qk_cache_dir


def _qk_manifest_path(qk_root: Path) -> Path:
    return qk_root / "qk_cache_metadata.json"


def _load_qk_manifest(qk_root: Path) -> dict[str, Any]:
    path = _qk_manifest_path(qk_root)
    if not path.exists():
        return {"examples": {}}
    payload = load_json(path)
    examples = payload.get("examples")
    if not isinstance(examples, dict):
        payload["examples"] = {}
    return payload


def _write_qk_manifest(qk_root: Path, payload: dict[str, Any]) -> Path:
    qk_root.mkdir(parents=True, exist_ok=True)
    return write_json(_qk_manifest_path(qk_root), payload)


def _qk_manifest_example(cache_dir: Path) -> dict[str, Any]:
    manifest = _load_qk_manifest(cache_dir.parent)
    examples = manifest.get("examples", {})
    item = examples.get(cache_dir.name, {}) if isinstance(examples, dict) else {}
    return item if isinstance(item, dict) else {}


def _cot_tensor_candidates(run_dir: Path, example_idx: int) -> list[Path]:
    """Return likely CoT tensor locations for a root or per-example run directory."""

    name = f"inputs_cot_{int(example_idx)}.pt"
    candidates = [run_dir / "tensors" / name]
    # Counting ablation examples are nested under
    # <run>/ablation_examples/example_id_N, while the CoT tensors are produced
    # in the parent run's tensors directory. Include that parent-run candidate so
    # Q/K capture uses the same extended input as hidden-state analysis.
    if run_dir.parent.name == "ablation_examples":
        candidates.append(run_dir.parent.parent / "tensors" / name)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def _load_cot_input_ids_for_qk(
    run_dir: Path, example_idx: int
) -> tuple[torch.Tensor | None, Path | None]:
    for cot_path in _cot_tensor_candidates(run_dir, example_idx):
        if not cot_path.exists():
            continue
        cot_payload = torch.load(cot_path, map_location="cpu")
        input_ids = (
            cot_payload.get("input_ids") if isinstance(cot_payload, dict) else None
        )
        if input_ids is None:
            raise ValueError(
                f"CoT tensor payload at {cot_path} does not contain input_ids"
            )
        return input_ids, cot_path
    return None, None


def _cached_input_ids_match(cache_dir: Path, expected_input_ids: torch.Tensor) -> bool:
    path = cache_dir / "input_ids.pt"
    if not path.exists():
        return False
    try:
        cached = torch.load(path, map_location="cpu")
    except Exception:
        return False
    return torch.equal(cached.cpu(), expected_input_ids.cpu())


def qk_cache_complete(cache_dir: Path, layers: Sequence[int]) -> bool:
    required = ["input_ids.pt", "attention_mask.pt", "position_ids.pt"]
    if any(not (cache_dir / name).exists() for name in required):
        return False
    example_meta = _qk_manifest_example(cache_dir)
    has_metadata = (cache_dir / "metadata.json").exists() or isinstance(
        example_meta.get("metadata"), dict
    )
    has_spec = (cache_dir / "analysis_spec.json").exists() or isinstance(
        example_meta.get("analysis_spec"), dict
    )
    if not has_metadata or not has_spec:
        return False
    for layer in layers:
        if not (cache_dir / f"layer_{layer:02d}_q_raw.pt").exists():
            return False
        if not (cache_dir / f"layer_{layer:02d}_k_raw.pt").exists():
            return False
        if not (cache_dir / f"layer_{layer:02d}_qk_norms.pt").exists():
            return False
    return True


def ensure_qk_cache(
    *,
    run_dir: str | Path,
    example_idx: int,
    row: dict[str, Any],
    cfg: DynamicNiahV2Config,
    analysis_cfg: QKOutlierAnalysisConfig,
    repo_root: str | Path = ".",
    model=None,
    tokenizer=None,
    force: bool = False,
    timing_records: list[dict[str, Any]] | None = None,
) -> Path:
    """Generate a Q/K cache for one uncontrolled example if it is missing."""

    del repo_root  # Kept for API compatibility with older notebook calls.
    run_dir = Path(run_dir)
    qk_root = _qk_root(run_dir, analysis_cfg)
    cache_dir = qk_root / f"input_{example_idx}"
    input_ids_override = None
    cot_path = None
    if getattr(cfg, "analyze_reasoning_tokens", False):
        input_ids_override, cot_path = _load_cot_input_ids_for_qk(run_dir, example_idx)
    cache_is_complete = qk_cache_complete(cache_dir, analysis_cfg.layers)
    if (
        cache_is_complete
        and input_ids_override is not None
        and not _cached_input_ids_match(cache_dir, input_ids_override)
    ):
        print(
            f"[qk-outlier] cached input ids for example {example_idx} do not match "
            "CoT extended input ids; recapturing Q/K cache",
            flush=True,
        )
        cache_is_complete = False
    if not force and cache_is_complete:
        if timing_records is not None:
            timing_records.append(
                {
                    "example_idx": int(example_idx),
                    "cache_dir": str(cache_dir),
                    "skipped": True,
                    "qk_cache_elapsed_seconds": 0.0,
                    "model_inference_elapsed_seconds": 0.0,
                }
            )
        return cache_dir
    if model is None or tokenizer is None:
        raise ValueError(
            "ensure_qk_cache requires an already-loaded model and tokenizer"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    qk_cache_start = time.perf_counter()
    prompt = build_uncontrolled_prompt_text(cfg, row, tokenizer=tokenizer)
    if getattr(cfg, "analyze_reasoning_tokens", False):
        if input_ids_override is not None:
            print(
                f"[qk-outlier] using CoT extended input ids for example {example_idx} "
                f"from {cot_path}",
                flush=True,
            )
        else:
            print(
                f"[qk-outlier] warning: CoT input ids missing for example {example_idx}; "
                "falling back to prompt text",
                flush=True,
            )
    markers = needle_markers(row)
    capture_payload = capture_qk_cache(
        model=model,
        tokenizer=tokenizer,
        model_name=analysis_cfg.model,
        prompt=prompt,
        out_dir=cache_dir,
        target_layers=analysis_cfg.layers,
        markers=markers,
        chat_template=False,
        enable_thinking=None,
        attn_implementation_requested=analysis_cfg.capture_attn_implementation,
        save_dtype=SAVE_DTYPE_MAP[analysis_cfg.capture_save_dtype],
        save_token_strings=True,
        write_metadata_files=False,
        input_ids_override=input_ids_override,
    )

    manifest = _load_qk_manifest(qk_root)
    manifest.update(
        {
            "schema_version": "qk_cache_metadata_v1",
            "model": analysis_cfg.model,
            "layers": [int(x) for x in analysis_cfg.layers],
            "qk_cache_dir": analysis_cfg.qk_cache_dir,
        }
    )
    examples = manifest.setdefault("examples", {})
    examples[cache_dir.name] = {
        "example_idx": int(example_idx),
        "cache_dir": cache_dir.name,
        "prompt": capture_payload["prompt"],
        "model_text": capture_payload["model_text"],
        "markers": capture_payload["markers"],
        "analysis_spec": capture_payload["analysis_spec"],
        "tokens": capture_payload["tokens"],
        "metadata": capture_payload["metadata"],
    }
    _write_qk_manifest(qk_root, manifest)
    qk_cache_elapsed_seconds = time.perf_counter() - qk_cache_start
    if timing_records is not None:
        timing_records.append(
            {
                "example_idx": int(example_idx),
                "cache_dir": str(cache_dir),
                "skipped": False,
                "qk_cache_elapsed_seconds": qk_cache_elapsed_seconds,
                "model_inference_elapsed_seconds": float(
                    capture_payload.get("metadata", {}).get(
                        "model_inference_elapsed_seconds", 0.0
                    )
                ),
            }
        )
    return cache_dir


def print_qk_shape_checks(cache_dir: str | Path, layers: Sequence[int]) -> None:
    cache_dir = Path(cache_dir)
    meta = load_cache_metadata(cache_dir)
    print(f"Q/K cache: {cache_dir} seq_len={meta.get('seq_len')}")
    for layer in layers:
        q_raw = load_tensor(cache_dir / f"layer_{layer:02d}_q_raw.pt")
        k_raw = load_tensor(cache_dir / f"layer_{layer:02d}_k_raw.pt")
        print(
            f"  layer {layer:02d}: q_raw={tuple(q_raw.shape)} {q_raw.dtype}; "
            f"k_raw={tuple(k_raw.shape)} {k_raw.dtype}"
        )


def resolve_heads(cache_dir: str | Path, heads: Sequence[int] | None) -> list[int]:
    meta = load_cache_metadata(Path(cache_dir))
    n_heads = int(meta["model_config"]["num_attention_heads"])
    selected = list(range(n_heads)) if heads is None else [int(h) for h in heads]
    bad = [h for h in selected if h < 0 or h >= n_heads]
    if bad:
        raise ValueError(f"Invalid heads {bad}; model has {n_heads} attention heads")
    return selected


def _load_tokens_and_ids(cache_dir: Path) -> tuple[list[str] | None, list[int]]:
    input_ids = load_tensor(cache_dir / "input_ids.pt")[0].tolist()
    tokens = load_cache_tokens(cache_dir)
    return tokens, [int(x) for x in input_ids]


def _flatten_input_ids(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], list):
            return [int(x) for x in value[0]]
        return [int(x) for x in value]
    return None


def _input_ids_from_model_input_table(
    run_dir: Path, example_idx: int
) -> list[int] | None:
    path = run_dir / "tables" / "model_input_ids.txt"
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    marker = f"Example ID {int(example_idx)}"
    for idx, line in enumerate(lines):
        if line.strip() != marker:
            continue
        for next_idx in range(idx + 1, len(lines) - 1):
            if lines[next_idx].strip() == "uncontrolled input ids":
                values = lines[next_idx + 1].strip()
                if not values:
                    return []
                return [int(x) for x in values.split()]
            if lines[next_idx].startswith("Example ID "):
                break
    return None


def _token_source_for_massive_record(
    *,
    run_dir: Path,
    record: dict[str, Any],
    example_idx: int,
    hidden_seq_len: int,
    cache_dir: Path,
) -> tuple[list[str] | None, list[int], set[int], list[tuple[str, int, int]]]:
    """Return token metadata aligned to the hidden-state sequence axis.

    Massive-activation positions are selected from ``hidden`` rows, so token text
    must be indexed with the exact uncontrolled input ids used for that hidden
    forward pass.  A stale/mismatched Q/K cache can have a valid token list but a
    different sequence; using it would print plausible but incorrect tokens.
    """

    authoritative_input_ids = _flatten_input_ids(record.get("input_ids"))
    if authoritative_input_ids is None:
        authoritative_input_ids = _flatten_input_ids(
            record.get("uncontrolled_input_ids")
        )
    if authoritative_input_ids is None:
        authoritative_input_ids = _input_ids_from_model_input_table(
            run_dir, example_idx
        )

    input_ids = (
        authoritative_input_ids[:hidden_seq_len]
        if authoritative_input_ids is not None
        else list(range(hidden_seq_len))
    )
    tokens: list[str] | None = None
    special_ids: set[int] = set()
    needle_intervals: list[tuple[str, int, int]] = []

    if cache_dir.exists() and (cache_dir / "input_ids.pt").exists():
        try:
            meta = load_cache_metadata(cache_dir)
            cache_tokens, cache_input_ids = _load_tokens_and_ids(cache_dir)
            if authoritative_input_ids is None:
                input_ids = cache_input_ids[:hidden_seq_len]
                tokens = cache_tokens
                special_ids = set(int(x) for x in meta.get("special_token_ids", []))
                needle_intervals = _needle_intervals_from_spec(cache_dir)
            else:
                cache_matches_hidden_input = (
                    len(cache_input_ids) >= len(input_ids)
                    and cache_input_ids[: len(input_ids)] == input_ids
                )
                if cache_matches_hidden_input:
                    tokens = cache_tokens
                    special_ids = set(int(x) for x in meta.get("special_token_ids", []))
                    needle_intervals = _needle_intervals_from_spec(cache_dir)
                else:
                    print(
                        "[qk-outlier] warning: ignoring mismatched Q/K cache tokens "
                        f"for input={example_idx}; hidden_input_len={len(input_ids)} "
                        f"cache_input_len={len(cache_input_ids)}"
                    )
        except (FileNotFoundError, ValueError, TypeError) as exc:
            print(
                "[qk-outlier] warning: could not load Q/K cache token metadata "
                f"for input={example_idx}: {exc}"
            )

    return tokens, input_ids, special_ids, needle_intervals


def _token_string(tokens: list[str] | None, input_ids: list[int], pos: int) -> str:
    if tokens is not None and 0 <= pos < len(tokens):
        return str(tokens[pos])
    if 0 <= pos < len(input_ids):
        return f"<id:{input_ids[pos]}>"
    return ""


def _context_string(
    tokens: list[str] | None, input_ids: list[int], pos: int, radius: int = 5
) -> str:
    start = max(0, pos - radius)
    end = min(len(input_ids), pos + radius + 1)
    return " ".join(_token_string(tokens, input_ids, i) for i in range(start, end))


def _needle_intervals_from_spec(cache_dir: Path) -> list[tuple[str, int, int]]:
    spec = load_cache_analysis_spec(cache_dir)
    out: list[tuple[str, int, int]] = []
    spans = spec.get("spans") or {}
    for name, intervals in spans.items():
        if not str(name).startswith("needle"):
            continue
        for interval in intervals:
            if len(interval) == 2:
                out.append((str(name), int(interval[0]), int(interval[1])))
    return out


def token_flags(
    *,
    pos: int,
    seq_len: int,
    token_id: int,
    special_ids: set[int],
    needle_intervals: Sequence[tuple[str, int, int]],
    n_edge: int,
    n_after_needle: int,
) -> dict[str, Any]:
    needle_name = ""
    is_after_needle = False
    for name, start, end in needle_intervals:
        if start <= pos < end:
            needle_name = name
        if end <= pos < end + n_after_needle:
            is_after_needle = True
    return {
        "is_special": token_id in special_ids,
        "is_early": pos < n_edge,
        "is_late": pos >= max(0, seq_len - n_edge),
        "is_needle": bool(needle_name),
        "needle_name": needle_name,
        "is_after_needle": is_after_needle,
    }


@torch.no_grad()
def compute_received_attention_stats(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    info: dict[str, Any],
    key_padding_mask: torch.Tensor | None,
    key_block_size: int = 8192,
    query_block_size: int = 64,
) -> dict[str, torch.Tensor]:
    """Compute received-attention sink statistics without full matrix materialization."""

    device = q.device
    T = int(info["seq_len"])
    scaling = float(info["scaling"])
    received_sum = torch.zeros(T, dtype=torch.float64, device=device)
    received_count = torch.zeros(T, dtype=torch.float64, device=device)
    uniform_sum = torch.zeros(T, dtype=torch.float64, device=device)
    all_queries = torch.arange(T, dtype=torch.long, device=device)
    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.to(device=device, dtype=torch.bool)
        all_queries = all_queries[key_padding_mask[all_queries]]

    if key_block_size <= 0 or query_block_size <= 0:
        raise ValueError("key_block_size and query_block_size must be positive")

    for qb in range(0, int(all_queries.numel()), query_block_size):
        qpos = all_queries[qb : qb + query_block_size]
        if qpos.numel() == 0:
            continue
        q_block = q[qpos]
        row_max = torch.full(
            (qpos.numel(),), -torch.inf, dtype=torch.float32, device=device
        )
        for ks in range(0, T, key_block_size):
            ke = min(ks + key_block_size, T)
            key_positions = torch.arange(ks, ke, dtype=torch.long, device=device)
            scores = (q_block.float() @ k[ks:ke].float().T) * scaling
            allowed = key_positions[None, :] <= qpos[:, None]
            if key_padding_mask is not None:
                allowed = allowed & key_padding_mask[ks:ke][None, :]
            scores = scores.masked_fill(~allowed, -torch.inf)
            row_max = torch.maximum(row_max, scores.max(dim=-1).values)

        has_valid_key = torch.isfinite(row_max)
        row_max_safe = torch.where(has_valid_key, row_max, torch.zeros_like(row_max))
        denom = torch.zeros((qpos.numel(),), dtype=torch.float32, device=device)
        block_weights: list[tuple[int, int, torch.Tensor, torch.Tensor]] = []
        for ks in range(0, T, key_block_size):
            ke = min(ks + key_block_size, T)
            key_positions = torch.arange(ks, ke, dtype=torch.long, device=device)
            scores = (q_block.float() @ k[ks:ke].float().T) * scaling
            allowed = key_positions[None, :] <= qpos[:, None]
            if key_padding_mask is not None:
                allowed = allowed & key_padding_mask[ks:ke][None, :]
            scores = scores.masked_fill(~allowed, -torch.inf)
            weights = torch.exp(scores - row_max_safe[:, None]).masked_fill(
                ~allowed, 0.0
            )
            denom += weights.sum(dim=-1)
            block_weights.append((ks, ke, key_positions, weights))

        denom_safe = denom.clamp_min(1e-30)
        valid_queries = has_valid_key[:, None]
        inv_uniform = (1.0 / (qpos.float() + 1.0)).to(dtype=torch.float64)
        for ks, ke, key_positions, weights in block_weights:
            later = (qpos[:, None] > key_positions[None, :]) & valid_queries
            probs = (
                (weights / denom_safe[:, None])
                .to(dtype=torch.float64)
                .masked_fill(~later, 0.0)
            )
            received_sum[ks:ke] += probs.sum(dim=0)
            received_count[ks:ke] += later.to(dtype=torch.float64).sum(dim=0)
            uniform_sum[ks:ke] += (
                later.to(dtype=torch.float64) * inv_uniform[:, None]
            ).sum(dim=0)

    received_mean = received_sum / received_count.clamp_min(1.0)
    received_mean = torch.where(
        received_count > 0, received_mean, torch.full_like(received_mean, torch.nan)
    )
    uniform_baseline = uniform_sum / received_count.clamp_min(1.0)
    uniform_baseline = torch.where(
        received_count > 0,
        uniform_baseline,
        torch.full_like(uniform_baseline, torch.nan),
    )
    received_uniform_ratio = received_mean / uniform_baseline
    return {
        "received_sum": received_sum.detach().cpu().float(),
        "received_count": received_count.detach().cpu().long(),
        "received_mean": received_mean.detach().cpu().float(),
        "uniform_baseline": uniform_baseline.detach().cpu().float(),
        "received_uniform_ratio": received_uniform_ratio.detach().cpu().float(),
    }


def _critical_positions(
    *,
    cache_dir: Path,
    massive_positions: Sequence[int] = (),
    n_edge: int,
    n_after_needle: int,
) -> dict[int, list[str]]:
    meta = load_cache_metadata(cache_dir)
    tokens, input_ids = _load_tokens_and_ids(cache_dir)
    T = len(input_ids)
    special_ids = set(int(x) for x in meta.get("special_token_ids", []))
    labels: dict[int, list[str]] = {}

    def add(pos: int, label: str) -> None:
        if 0 <= pos < T:
            labels.setdefault(int(pos), []).append(label)

    add(0, "bos")
    for pos, tok_id in enumerate(input_ids):
        tok = _token_string(tokens, input_ids, pos).lower()
        if tok_id in special_ids:
            add(pos, "special")
        if any(
            marker in tok
            for marker in ["<|im_start|>", "<|im_end|>", "think", "assistant", "user"]
        ):
            add(pos, "chat_or_thinking_marker")
    for pos in range(min(n_edge, T)):
        add(pos, "first_edge")
    for pos in range(max(0, T - n_edge), T):
        add(pos, "last_edge")
    for name, _start, end in _needle_intervals_from_spec(cache_dir):
        for pos in range(end, min(T, end + n_after_needle)):
            add(pos, f"after_{name}")
    for pos in massive_positions:
        add(int(pos), "massive_activation")
    return {pos: sorted(set(vals)) for pos, vals in sorted(labels.items())}


@torch.no_grad()
def compute_attention_columns(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    info: dict[str, Any],
    positions: Sequence[int],
    key_padding_mask: torch.Tensor | None,
    query_block_size: int,
    key_block_size: int,
) -> torch.Tensor:
    """Return selected received-attention columns as ``[len(positions), T]``.

    Invalid entries with ``query <= key`` are stored as NaN.  This computes only
    the requested columns, avoiding full ``[T, T]`` attention materialization.
    """

    device = q.device
    T = int(info["seq_len"])
    scaling = float(info["scaling"])
    selected = torch.tensor(
        [int(p) for p in positions], dtype=torch.long, device=device
    )
    out = torch.full((len(positions), T), torch.nan, dtype=torch.float32)
    all_queries = torch.arange(T, dtype=torch.long, device=device)
    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.to(device=device, dtype=torch.bool)
        all_queries = all_queries[key_padding_mask[all_queries]]

    for qb in range(0, int(all_queries.numel()), query_block_size):
        qpos = all_queries[qb : qb + query_block_size]
        q_block = q[qpos]
        row_max = torch.full(
            (qpos.numel(),), -torch.inf, dtype=torch.float32, device=device
        )
        for ks in range(0, T, key_block_size):
            ke = min(ks + key_block_size, T)
            key_positions = torch.arange(ks, ke, dtype=torch.long, device=device)
            scores = (q_block.float() @ k[ks:ke].float().T) * scaling
            allowed = key_positions[None, :] <= qpos[:, None]
            if key_padding_mask is not None:
                allowed = allowed & key_padding_mask[ks:ke][None, :]
            scores = scores.masked_fill(~allowed, -torch.inf)
            row_max = torch.maximum(row_max, scores.max(dim=-1).values)
        has_valid_key = torch.isfinite(row_max)
        row_max_safe = torch.where(has_valid_key, row_max, torch.zeros_like(row_max))
        denom = torch.zeros((qpos.numel(),), dtype=torch.float32, device=device)
        selected_scores = torch.full(
            (qpos.numel(), len(positions)),
            -torch.inf,
            dtype=torch.float32,
            device=device,
        )
        selected_allowed = torch.zeros(
            (qpos.numel(), len(positions)), dtype=torch.bool, device=device
        )
        for ks in range(0, T, key_block_size):
            ke = min(ks + key_block_size, T)
            key_positions = torch.arange(ks, ke, dtype=torch.long, device=device)
            scores = (q_block.float() @ k[ks:ke].float().T) * scaling
            allowed = key_positions[None, :] <= qpos[:, None]
            if key_padding_mask is not None:
                allowed = allowed & key_padding_mask[ks:ke][None, :]
            scores = scores.masked_fill(~allowed, -torch.inf)
            denom += (
                torch.exp(scores - row_max_safe[:, None])
                .masked_fill(~allowed, 0.0)
                .sum(dim=-1)
            )
            in_block = (selected >= ks) & (selected < ke)
            if in_block.any():
                selected_idx = torch.nonzero(in_block, as_tuple=False).flatten()
                block_cols = selected[selected_idx] - ks
                selected_scores[:, selected_idx] = scores[:, block_cols]
                selected_allowed[:, selected_idx] = allowed[:, block_cols]
        denom_safe = denom.clamp_min(1e-30)
        probs = torch.exp(selected_scores - row_max_safe[:, None]) / denom_safe[:, None]
        later = qpos[:, None] > selected[None, :]
        probs = probs.masked_fill(
            ~(selected_allowed & later & has_valid_key[:, None]), float("nan")
        )
        out[:, qpos.detach().cpu()] = probs.detach().cpu().T.float()
    return out


@torch.no_grad()
def save_critical_attention(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    info: dict[str, Any],
    cache_dir: Path,
    layer: int,
    head: int,
    out_dir: Path,
    critical_positions: dict[int, list[str]],
    key_padding_mask: torch.Tensor | None,
    key_block_size: int,
    query_block_size: int,
) -> Path:
    positions = sorted(critical_positions)
    if not positions:
        payload = {
            "positions": [],
            "labels": {},
            "rows": torch.empty(0),
            "columns": torch.empty(0),
        }
    else:
        qpos = torch.tensor(positions, dtype=torch.long, device=q.device)
        row_stats = compute_query_block_stats(
            q_block=q[qpos],
            k=k,
            query_positions=qpos,
            spans={},
            key_padding_mask=key_padding_mask,
            scaling=float(info["scaling"]),
            key_block_size=key_block_size,
            topk=0,
            local_windows=(),
            return_rows=True,
        )
        rows = row_stats["rows"]
        cols = compute_attention_columns(
            q=q,
            k=k,
            info=info,
            positions=positions,
            key_padding_mask=key_padding_mask,
            query_block_size=query_block_size,
            key_block_size=key_block_size,
        )
        payload = {
            "positions": positions,
            "labels": {str(k): v for k, v in critical_positions.items()},
            "rows": rows,
            "columns": cols,
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"critical_attention_layer_{layer:02d}_head_{head:02d}.pt"
    torch.save(payload, path)
    return path


def _expanded_needle_segments_for_massive_record(
    *,
    run_dir: Path,
    example_idx: int,
    record: dict[str, Any],
    sequence_length: int,
    expansion: int,
) -> list[dict[str, Any]]:
    """Return expanded needle segments on the uncontrolled final-input axis."""

    expanded = record.get("expanded_needle_segments")
    if isinstance(expanded, list) and expanded:
        return [dict(segment) for segment in expanded]

    needle_spans = record.get("needle_spans")
    if isinstance(needle_spans, list) and needle_spans:
        return expand_needle_segments(
            [dict(span) for span in needle_spans],
            sequence_length=sequence_length,
            expansion=expansion,
        )

    cache_dir = run_dir / "tensors/qk_cache" / f"input_{example_idx}"
    if cache_dir.exists():
        intervals = _needle_intervals_from_spec(cache_dir)
        if intervals:
            spans = [
                {"needle_id": name, "start": start, "end": end, "length": end - start}
                for name, start, end in intervals
            ]
            return expand_needle_segments(
                spans, sequence_length=sequence_length, expansion=expansion
            )

    return []


def _event_sort_key_desc_ratio(row: dict[str, Any]) -> tuple[float, int, int, int]:
    return (
        -float(row.get("norm_ratio_to_median", 0.0)),
        int(row.get("example_idx", -1)),
        int(row.get("layer", -1)),
        int(row.get("position", -1)),
    )


def _massive_tokens_text(rows: Sequence[dict[str, Any]], *, limit: int = 1000) -> str:
    return "\n".join(
        f"input={r['example_idx']} layer={r['layer']} pos={r['position']} "
        f"ratio={r['norm_ratio_to_median']:.3g} token={r['token']!r}"
        for r in rows[:limit]
    )


def analyze_massive_activations(
    *,
    run_dir: str | Path,
    layers: Sequence[int],
    threshold: float,
    top_k: int,
    n_edge: int,
    n_after_needle: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    run_dir = Path(run_dir)
    records = load_hidden_state_records(run_dir / "tensors")
    all_events: list[dict[str, Any]] = []
    outside_needle_events: list[dict[str, Any]] = []
    scalar_events: list[dict[str, Any]] = []
    dim_count: dict[tuple[int, int], int] = {}

    for record in records:
        example_idx = int(record["sample_idx"])
        hidden = record["hidden"]
        cache_dir = run_dir / "tensors/qk_cache" / f"input_{example_idx}"
        tokens, input_ids, special_ids, needle_intervals = (
            _token_source_for_massive_record(
                run_dir=run_dir,
                record=record,
                example_idx=example_idx,
                hidden_seq_len=int(hidden.shape[1]),
                cache_dir=cache_dir,
            )
        )
        seq_len = min(hidden.shape[1], len(input_ids))
        expanded_needle_segments = _expanded_needle_segments_for_massive_record(
            run_dir=run_dir,
            example_idx=example_idx,
            record=record,
            sequence_length=seq_len,
            expansion=n_after_needle,
        )
        outside_needle_mask = build_outside_segments_mask(
            seq_len, expanded_needle_segments
        )
        for layer in layers:
            try:
                hidden_layer_idx = stored_layer_index(record, int(layer))
            except ValueError:
                continue
            h = hidden[hidden_layer_idx, :seq_len].to(dtype=torch.float32)
            norms = torch.linalg.vector_norm(h, ord=2, dim=-1)
            median = norms.median()
            mad = (norms - median).abs().median().clamp_min(1e-12)
            ratios = norms / median.clamp_min(1e-12)
            robust_z = (norms - median) / mad
            max_abs, argmax_dim = h.abs().max(dim=-1)
            scalar_median = max_abs.median().clamp_min(1e-12)
            scalar_ratio = max_abs / scalar_median

            out_dir = run_dir / "tensors/massive_activations" / f"input_{example_idx}"
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "hidden_norm": norms.cpu(),
                    "norm_ratio_to_median": ratios.cpu(),
                    "robust_z": robust_z.cpu(),
                    "median_norm": float(median.item()),
                    "mad_norm": float(mad.item()),
                    "max_abs_activation": max_abs.cpu(),
                    "argmax_abs_dim": argmax_dim.cpu(),
                    "max_abs_activation_ratio_to_median": scalar_ratio.cpu(),
                    "linf_norm_ratio_to_median": scalar_ratio.cpu(),
                    "scalar_median": float(scalar_median.item()),
                },
                out_dir / f"hidden_norms_layer_{layer:02d}.pt",
            )

            top_norm_positions = (
                set(
                    torch.topk(norms, k=min(top_k, int(norms.numel()))).indices.tolist()
                )
                if top_k > 0
                else set()
            )
            selected: set[int] = set(
                torch.nonzero(ratios >= threshold, as_tuple=False).flatten().tolist()
            )
            selected.update(top_norm_positions)
            scalar_selected: set[int] = (
                set(
                    torch.topk(
                        max_abs, k=min(top_k, int(max_abs.numel()))
                    ).indices.tolist()
                )
                if top_k > 0
                else set()
            )

            outside_positions = torch.nonzero(
                outside_needle_mask, as_tuple=False
            ).flatten()
            outside_selected: set[int] = set()
            outside_top_positions: set[int] = set()
            outside_ratios = torch.full_like(ratios, torch.nan)
            outside_robust_z = torch.full_like(robust_z, torch.nan)
            outside_median = torch.tensor(float("nan"), dtype=torch.float32)
            outside_mad = torch.tensor(float("nan"), dtype=torch.float32)
            if int(outside_positions.numel()) > 0:
                outside_norms = norms[outside_positions]
                outside_median = outside_norms.median().clamp_min(1e-12)
                outside_mad = (
                    (outside_norms - outside_median).abs().median().clamp_min(1e-12)
                )
                outside_ratios[outside_positions] = outside_norms / outside_median
                outside_robust_z[outside_positions] = (
                    outside_norms - outside_median
                ) / outside_mad
                outside_selected.update(
                    outside_positions[
                        torch.nonzero(
                            outside_ratios[outside_positions] >= threshold,
                            as_tuple=False,
                        ).flatten()
                    ].tolist()
                )
                if top_k > 0:
                    outside_top_local = torch.topk(
                        outside_norms, k=min(top_k, int(outside_norms.numel()))
                    ).indices
                    outside_top_positions = set(
                        outside_positions[outside_top_local].tolist()
                    )
                    outside_selected.update(outside_top_positions)

            for pos in sorted(selected):
                flags = token_flags(
                    pos=pos,
                    seq_len=seq_len,
                    token_id=int(input_ids[pos]) if pos < len(input_ids) else -1,
                    special_ids=special_ids,
                    needle_intervals=needle_intervals,
                    n_edge=n_edge,
                    n_after_needle=n_after_needle,
                )
                trigger = []
                if float(ratios[pos]) >= threshold:
                    trigger.append("threshold")
                if pos in top_norm_positions:
                    trigger.append("top_k")
                all_events.append(
                    {
                        "example_idx": example_idx,
                        "layer": int(layer),
                        "position": int(pos),
                        "token_id": (
                            int(input_ids[pos]) if pos < len(input_ids) else None
                        ),
                        "token": _token_string(tokens, input_ids, pos),
                        "hidden_norm": float(norms[pos].item()),
                        "median_norm": float(median.item()),
                        "norm_ratio_to_median": float(ratios[pos].item()),
                        "robust_z": float(robust_z[pos].item()),
                        "max_abs_activation": float(max_abs[pos].item()),
                        "argmax_abs_dim": int(argmax_dim[pos].item()),
                        "max_abs_activation_ratio_to_median": float(
                            scalar_ratio[pos].item()
                        ),
                        "trigger_rule": "+".join(trigger),
                        "left_right_context": _context_string(tokens, input_ids, pos),
                        **flags,
                    }
                )
            for pos in sorted(outside_selected):
                flags = token_flags(
                    pos=int(pos),
                    seq_len=seq_len,
                    token_id=int(input_ids[pos]) if pos < len(input_ids) else -1,
                    special_ids=special_ids,
                    needle_intervals=needle_intervals,
                    n_edge=n_edge,
                    n_after_needle=n_after_needle,
                )
                trigger = []
                if (
                    torch.isfinite(outside_ratios[pos])
                    and float(outside_ratios[pos]) >= threshold
                ):
                    trigger.append("threshold_outside_needles")
                if pos in outside_top_positions:
                    trigger.append("top_k_outside_needles")
                outside_needle_events.append(
                    {
                        "example_idx": example_idx,
                        "layer": int(layer),
                        "position": int(pos),
                        "token_id": (
                            int(input_ids[pos]) if pos < len(input_ids) else None
                        ),
                        "token": _token_string(tokens, input_ids, int(pos)),
                        "hidden_norm": float(norms[pos].item()),
                        "median_norm_outside_needles": float(outside_median.item()),
                        "norm_ratio_to_median": float(outside_ratios[pos].item()),
                        "robust_z_outside_needles": float(outside_robust_z[pos].item()),
                        "max_abs_activation": float(max_abs[pos].item()),
                        "argmax_abs_dim": int(argmax_dim[pos].item()),
                        "max_abs_activation_ratio_to_median": float(
                            scalar_ratio[pos].item()
                        ),
                        "trigger_rule": "+".join(trigger),
                        "left_right_context": _context_string(
                            tokens, input_ids, int(pos)
                        ),
                        "excluded_expanded_needle_segment_count": len(
                            expanded_needle_segments
                        ),
                        **flags,
                    }
                )
            for pos in sorted(scalar_selected):
                dim = int(argmax_dim[pos].item())
                dim_count[(int(layer), dim)] = dim_count.get((int(layer), dim), 0) + 1
                scalar_events.append(
                    {
                        "example_idx": example_idx,
                        "layer": int(layer),
                        "position": int(pos),
                        "token_id": (
                            int(input_ids[pos]) if pos < len(input_ids) else None
                        ),
                        "token": _token_string(tokens, input_ids, pos),
                        "max_abs_activation": float(max_abs[pos].item()),
                        "argmax_abs_dim": dim,
                        "max_abs_activation_ratio_to_median": float(
                            scalar_ratio[pos].item()
                        ),
                        "hidden_norm": float(norms[pos].item()),
                        "norm_ratio_to_median": float(ratios[pos].item()),
                    }
                )

    tables_dir = run_dir / "tables"
    outside_needle_events = sorted(
        outside_needle_events, key=_event_sort_key_desc_ratio
    )
    _write_csv(tables_dir / "massive_tokens_all.csv", all_events)
    _write_jsonl(tables_dir / "massive_tokens_all.jsonl", all_events)
    _write_csv(
        tables_dir / "massive_tokens_outside_needles_all.csv", outside_needle_events
    )
    _write_jsonl(
        tables_dir / "massive_tokens_outside_needles_all.jsonl", outside_needle_events
    )
    _write_csv(tables_dir / "massive_scalar_activations_all.csv", scalar_events)
    dim_rows = [
        {"layer": layer, "argmax_abs_dim": dim, "count": count}
        for (layer, dim), count in sorted(
            dim_count.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        )
    ]
    _write_csv(tables_dir / "massive_activation_dim_counts.csv", dim_rows)
    txt_path = tables_dir / "massive_tokens.txt"
    txt_path.write_text(_massive_tokens_text(all_events), encoding="utf-8")
    outside_txt_path = tables_dir / "massive_tokens_outside_needles.txt"
    outside_txt_path.write_text(
        _massive_tokens_text(outside_needle_events), encoding="utf-8"
    )
    return all_events, scalar_events, dim_rows


def _attention_mask(cache_dir: Path, device: torch.device) -> torch.Tensor | None:
    path = cache_dir / "attention_mask.pt"
    if not path.exists():
        return None
    return load_tensor(path)[0].to(device=device).bool()


def analyze_attention_for_cache(
    *,
    run_dir: str | Path,
    example_idx: int,
    cache_dir: str | Path,
    analysis_cfg: QKOutlierAnalysisConfig,
    massive_events: Sequence[dict[str, Any]] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_dir = Path(run_dir)
    cache_dir = Path(cache_dir)
    device = torch.device(
        analysis_cfg.device
        if torch.cuda.is_available() or analysis_cfg.device == "cpu"
        else "cpu"
    )
    compute_dtype = COMPUTE_DTYPES[analysis_cfg.compute_dtype]
    heads = resolve_heads(cache_dir, analysis_cfg.heads)
    tokens, input_ids = _load_tokens_and_ids(cache_dir)
    meta = load_cache_metadata(cache_dir)
    special_ids = set(int(x) for x in meta.get("special_token_ids", []))
    needle_intervals = _needle_intervals_from_spec(cache_dir)
    massive_by_layer: dict[int, set[int]] = {}
    for event in massive_events:
        if int(event.get("example_idx", -1)) == int(example_idx):
            massive_by_layer.setdefault(int(event["layer"]), set()).add(
                int(event["position"])
            )

    sink_rows: list[dict[str, Any]] = []
    needle_rows: list[dict[str, Any]] = []
    stats_dir = run_dir / analysis_cfg.attention_stats_dir / f"input_{example_idx}"
    stats_dir.mkdir(parents=True, exist_ok=True)

    spec = load_cache_analysis_spec(cache_dir)
    for layer in analysis_cfg.layers:
        critical = _critical_positions(
            cache_dir=cache_dir,
            massive_positions=sorted(massive_by_layer.get(int(layer), set())),
            n_edge=analysis_cfg.n_critical_edge_tokens,
            n_after_needle=analysis_cfg.n_after_needle,
        )
        for head in heads:
            q, k, info = reconstruct_single_head_qk(
                cache_dir=cache_dir,
                layer=int(layer),
                head=int(head),
                device=device,
                compute_dtype=compute_dtype,
            )
            key_padding_mask = _attention_mask(cache_dir, q.device)
            stats = compute_received_attention_stats(
                q=q,
                k=k,
                info=info,
                key_padding_mask=key_padding_mask,
                key_block_size=analysis_cfg.key_block_size,
                query_block_size=analysis_cfg.query_block_size,
            )
            torch.save(
                {
                    **stats,
                    "layer": int(layer),
                    "head": int(head),
                    "example_idx": int(example_idx),
                },
                stats_dir / f"attention_stats_layer_{layer:02d}_head_{head:02d}.pt",
            )
            topk = min(analysis_cfg.topk, int(stats["received_uniform_ratio"].numel()))
            values = torch.nan_to_num(stats["received_uniform_ratio"], nan=-torch.inf)
            top_positions = (
                torch.topk(values, k=topk).indices.tolist() if topk > 0 else []
            )
            for rank, pos in enumerate(top_positions, start=1):
                flags = token_flags(
                    pos=int(pos),
                    seq_len=len(input_ids),
                    token_id=int(input_ids[pos]),
                    special_ids=special_ids,
                    needle_intervals=needle_intervals,
                    n_edge=analysis_cfg.n_critical_edge_tokens,
                    n_after_needle=analysis_cfg.n_after_needle,
                )
                sink_rows.append(
                    {
                        "example_idx": int(example_idx),
                        "layer": int(layer),
                        "head": int(head),
                        "rank": rank,
                        "position": int(pos),
                        "token_id": int(input_ids[pos]),
                        "token": _token_string(tokens, input_ids, int(pos)),
                        "received_mean": float(stats["received_mean"][pos].item()),
                        "received_sum": float(stats["received_sum"][pos].item()),
                        "received_count": int(stats["received_count"][pos].item()),
                        "uniform_baseline": float(
                            stats["uniform_baseline"][pos].item()
                        ),
                        "received_uniform_ratio": float(
                            stats["received_uniform_ratio"][pos].item()
                        ),
                        **flags,
                    }
                )

            critical_path = save_critical_attention(
                q=q,
                k=k,
                info=info,
                cache_dir=cache_dir,
                layer=int(layer),
                head=int(head),
                out_dir=stats_dir,
                critical_positions=critical,
                key_padding_mask=key_padding_mask,
                key_block_size=analysis_cfg.key_block_size,
                query_block_size=analysis_cfg.query_block_size,
            )

            if needle_intervals:
                received = average_window_received(
                    q=q,
                    k=k,
                    info=info,
                    cache_dir=cache_dir,
                    spec=spec,
                    key_block_size=analysis_cfg.key_block_size,
                    query_block_size=analysis_cfg.query_block_size,
                )
                json_path = (
                    stats_dir
                    / f"needle_attention_mass_layer_{layer:02d}_head_{head:02d}.json"
                )
                write_json(json_path, received)
                for name, item in received.items():
                    needle_rows.append(
                        {
                            "example_idx": int(example_idx),
                            "layer": int(layer),
                            "head": int(head),
                            "name": name,
                            "intervals": json.dumps(item.get("intervals", [])),
                            "query_range": json.dumps(item.get("query_range", [])),
                            "num_queries": item.get("num_queries"),
                            "mean_mass": item.get("mean_mass"),
                            "sum_mass": item.get("sum_mass"),
                            "mean_entropy_nats": item.get("mean_entropy_nats"),
                            "critical_attention_path": str(critical_path),
                        }
                    )
            del q, k
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return sink_rows, needle_rows


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.sum().item()) < 2:
        return None
    x = x[mask]
    y = y[mask]
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom.item()) == 0.0:
        return None
    return float((x @ y / denom).item())


def join_outlier_attention(
    *,
    run_dir: str | Path,
    massive_events: Sequence[dict[str, Any]],
    sink_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Join massive activation events to per-head received-attention stats."""

    run_dir = Path(run_dir)
    topk_sink_keys = {
        (
            int(row["example_idx"]),
            int(row["layer"]),
            int(row["head"]),
            int(row["position"]),
        )
        for row in sink_rows
    }
    stats_cache: dict[tuple[int, int], list[tuple[int, dict[str, torch.Tensor]]]] = {}

    def stats_for(
        example_idx: int, layer: int
    ) -> list[tuple[int, dict[str, torch.Tensor]]]:
        key = (example_idx, layer)
        if key in stats_cache:
            return stats_cache[key]
        stats_dir = run_dir / "tensors/attention_stats" / f"input_{example_idx}"
        loaded: list[tuple[int, dict[str, torch.Tensor]]] = []
        for path in sorted(
            stats_dir.glob(f"attention_stats_layer_{layer:02d}_head_*.pt")
        ):
            payload = torch.load(path, map_location="cpu")
            loaded.append((int(payload["head"]), payload))
        stats_cache[key] = loaded
        return loaded

    joined: list[dict[str, Any]] = []
    for event in massive_events:
        example_idx = int(event["example_idx"])
        layer = int(event["layer"])
        position = int(event["position"])
        per_head = stats_for(example_idx, layer)
        if not per_head:
            joined.append({**event, "is_topk_sink": False})
            continue
        joined_any_head = False
        skipped_lengths: list[int] = []
        for head, payload in per_head:
            received_mean = payload.get("received_mean")
            if received_mean is None or position >= int(received_mean.shape[0]):
                if received_mean is not None:
                    skipped_lengths.append(int(received_mean.shape[0]))
                continue
            joined_any_head = True
            joined.append(
                {
                    **event,
                    "head": head,
                    "received_mean": float(payload["received_mean"][position].item()),
                    "received_sum": float(payload["received_sum"][position].item()),
                    "received_count": int(payload["received_count"][position].item()),
                    "uniform_baseline": float(
                        payload["uniform_baseline"][position].item()
                    ),
                    "received_uniform_ratio": float(
                        payload["received_uniform_ratio"][position].item()
                    ),
                    "is_topk_sink": (example_idx, layer, head, position)
                    in topk_sink_keys,
                }
            )
        if not joined_any_head:
            reason = (
                "position_outside_attention_stats"
                if skipped_lengths
                else "missing_attention_stats"
            )
            joined.append(
                {
                    **event,
                    "is_topk_sink": False,
                    "attention_stats_missing_reason": reason,
                    "attention_stats_seq_len": (
                        min(skipped_lengths) if skipped_lengths else None
                    ),
                }
            )

    summary: list[dict[str, Any]] = []
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for row in joined:
        by_layer.setdefault(int(row["layer"]), []).append(row)
    for layer, rows in sorted(by_layer.items()):
        massive_count = len({(r["example_idx"], r["position"]) for r in rows})
        joined_count = len(rows)
        overlap_count = sum(1 for r in rows if r.get("is_topk_sink"))
        xs = [
            float(r["norm_ratio_to_median"])
            for r in rows
            if "received_uniform_ratio" in r
        ]
        ys = [
            float(r["received_uniform_ratio"])
            for r in rows
            if "received_uniform_ratio" in r
        ]
        summary.append(
            {
                "layer": layer,
                "massive_event_count": massive_count,
                "massive_head_join_count": joined_count,
                "massive_topk_sink_overlap_count": overlap_count,
                "massive_topk_sink_overlap_fraction": (
                    (overlap_count / joined_count) if joined_count else 0.0
                ),
                "norm_ratio_received_uniform_ratio_pearson": _pearson(xs, ys),
            }
        )
    _write_csv(run_dir / "tables/outlier_attention_join.csv", joined)
    _write_csv(run_dir / "tables/outlier_overlap_summary.csv", summary)
    return joined, summary


def _read_measurement_plot_rows(
    run_dir: Path, example_idx: int, *, tables_dir: str | Path = "tables"
) -> list[dict[str, Any]]:
    path = run_dir / tables_dir / f"inputs_{example_idx}_measurements.csv"
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    tensor_path = run_dir / "tensors" / f"inputs_{example_idx}.pt"
    if not tensor_path.exists():
        return []
    measurements = torch.load(tensor_path, map_location="cpu")
    metric_keys = [
        key
        for key in measurements.keys()
        if key in {"relative_norm_diff", "cosine_similarity"}
    ]
    layer_ids = [int(x) for x in measurements["layers"].tolist()]
    max_positions = max(
        (int(measurements[key].shape[1]) for key in metric_keys), default=0
    )
    if "positions" in measurements:
        positions = [int(x) for x in measurements["positions"].detach().cpu().tolist()]
    else:
        positions = list(range(max_positions))
    rows: list[dict[str, Any]] = []
    for layer_i, layer_id in enumerate(layer_ids):
        for value_idx, position in enumerate(positions):
            row: dict[str, Any] = {"position": str(position), "layer": str(layer_id)}
            for key in metric_keys:
                arr = measurements[key]
                if layer_i < arr.shape[0] and value_idx < arr.shape[1]:
                    row[key] = str(
                        float(arr[layer_i, value_idx].to(torch.float32).item())
                    )
            rows.append(row)
    return rows


def _layer_colors_from_measurement_rows(
    rows: Sequence[dict[str, Any]], layers: Sequence[int], color_cycle: Sequence[Any]
) -> dict[int, Any]:
    color_by_layer: dict[int, Any] = {}
    for row in rows:
        color = row.get("layer_color")
        if color:
            color_by_layer.setdefault(int(row["layer"]), color)
    for idx, layer in enumerate(layers):
        if int(layer) not in color_by_layer:
            color_by_layer[int(layer)] = (
                color_cycle[idx % len(color_cycle)] if color_cycle else None
            )
    return color_by_layer


def _plot_measurement_rows(
    ax,
    rows: Sequence[dict[str, Any]],
    *,
    metric: str,
    layers: Sequence[int],
    color_by_layer: dict[int, Any],
) -> bool:
    plotted = False
    for layer in layers:
        layer_rows = sorted(
            (
                row
                for row in rows
                if int(row.get("layer", -1)) == int(layer)
                and row.get(metric) not in {None, ""}
            ),
            key=lambda row: int(row["position"]),
        )
        if not layer_rows:
            continue
        ax.plot(
            [int(row["position"]) for row in layer_rows],
            [float(row[metric]) for row in layer_rows],
            label=f"layer {int(layer)}",
            color=color_by_layer[int(layer)],
            linewidth=1.0,
        )
        plotted = True
    return plotted


def _low_cosine_positions(
    rows: Sequence[dict[str, Any]], *, threshold: float = 0.8
) -> list[int]:
    """Return token positions where any plotted layer has low cosine similarity."""

    positions: set[int] = set()
    for row in rows:
        value = row.get("cosine_similarity")
        if value in {None, ""} or row.get("position") in {None, ""}:
            continue
        try:
            cosine = float(value)
            position = int(row["position"])
        except (TypeError, ValueError):
            continue
        if cosine < threshold:
            positions.add(position)
    return sorted(positions)


def _mark_low_cosine_positions(
    axes: Sequence[Any], positions: Sequence[int], *, threshold: float = 0.8
) -> None:
    """Draw dashed vertical guides at low-cosine token positions on every subplot."""

    if not positions:
        return
    for ax in axes:
        for position in positions:
            ax.axvline(
                int(position),
                color="tab:red",
                linestyle="--",
                linewidth=0.8,
                alpha=0.35,
                label=f"_cosine_similarity_below_{threshold:g}",
                zorder=1,
            )


def _positive_log_values(tensor: torch.Tensor) -> torch.Tensor:
    """Return float values with non-positive/non-finite entries hidden for log plots."""

    values = tensor.to(torch.float32).cpu()
    finite_positive = torch.isfinite(values) & (values > 0)
    return torch.where(finite_positive, values, torch.full_like(values, torch.nan))


def _needle_spans_for_example(run_dir: Path, example_idx: int) -> list[dict[str, Any]]:
    """Load prompt-level needle spans for an example when available."""

    for tensor_path in [
        run_dir / "tensors" / f"inputs_{example_idx}.pt",
        run_dir / "tensors" / f"hidden_inputs_{example_idx}.pt",
    ]:
        if not tensor_path.exists():
            continue
        payload = torch.load(tensor_path, map_location="cpu")
        if isinstance(payload, dict) and isinstance(payload.get("needle_spans"), list):
            return payload["needle_spans"]
    return []


def _shade_needle_spans(ax, needle_spans: Sequence[dict[str, Any]]) -> None:
    labeled_any = False
    labeled_control = False
    for span in needle_spans:
        start = int(span["start"])
        end = int(span["end"])
        is_control = bool(span.get("is_control", False))
        label = None
        if is_control and not labeled_control:
            label = "controlled needle span"
            labeled_control = True
        elif not is_control and not labeled_any:
            label = "needle span"
            labeled_any = True
        ax.axvspan(
            start,
            end,
            color="tab:red" if is_control else "tab:orange",
            alpha=0.16 if is_control else 0.12,
            label=label,
            zorder=0,
        )


def _measurement_x_range(rows: Sequence[dict[str, Any]]) -> list[int] | None:
    positions = sorted(
        {int(row["position"]) for row in rows if row.get("position") not in {None, ""}}
    )
    return positions or None


def _reference_model_input_x(
    run_dir: Path, example_idx: int, measurement_rows: Sequence[dict[str, Any]]
) -> list[int] | None:
    """Return the uncontrolled final model-input token axis for a plotted example."""

    input_ids_path = (
        run_dir / "tensors/qk_cache" / f"input_{example_idx}" / "input_ids.pt"
    )
    if input_ids_path.exists():
        input_ids = load_tensor(input_ids_path)[0]
        return list(range(int(input_ids.numel())))

    for tensor_path in [
        run_dir / "tensors" / f"hidden_inputs_{example_idx}.pt",
        run_dir / "tensors" / f"inputs_{example_idx}.pt",
    ]:
        if not tensor_path.exists():
            continue
        payload = torch.load(tensor_path, map_location="cpu")
        if isinstance(payload, dict):
            hidden = payload.get("hidden")
            if isinstance(hidden, torch.Tensor) and hidden.ndim >= 2:
                return list(range(int(hidden.shape[1])))
            positions = payload.get("positions")
            if isinstance(positions, torch.Tensor) and int(positions.numel()) > 0:
                return [int(x) for x in positions.detach().cpu().tolist()]

    norm_dir = run_dir / "tensors/massive_activations" / f"input_{example_idx}"
    for norm_path in sorted(norm_dir.glob("hidden_norms_layer_*.pt")):
        payload = torch.load(norm_path, map_location="cpu")
        if isinstance(payload, dict):
            for key in [
                "hidden_norm",
                "norm_ratio_to_median",
                "linf_norm_ratio_to_median",
            ]:
                value = payload.get(key)
                if isinstance(value, torch.Tensor):
                    return list(range(int(value.numel())))

    return _measurement_x_range(measurement_rows)


def _aligned_metric_for_measurement_x(
    tensor: torch.Tensor,
    *,
    measurement_x: Sequence[int] | None,
) -> tuple[list[int], torch.Tensor]:
    """Select full-token metric values on the model-input x-axis."""

    values = tensor.to(torch.float32).cpu().flatten()
    if measurement_x is None:
        return list(range(int(values.numel()))), values

    xs: list[int] = []
    ys: list[torch.Tensor] = []
    for x in measurement_x:
        source_pos = int(x)
        if 0 <= source_pos < int(values.numel()):
            xs.append(int(x))
            ys.append(values[source_pos])
    if not ys:
        return [], torch.empty(0, dtype=torch.float32)
    return xs, torch.stack(ys).to(torch.float32)


def _save_qk_outlier_plot_table(
    run_dir: Path,
    example_idx: int,
    layer_values: dict[int, dict[str, tuple[list[int], torch.Tensor]]],
    *,
    tables_dir: str | Path = "tables",
) -> Path:
    path = run_dir / tables_dir / f"inputs_{example_idx}_qk_outliers.csv"
    rows: list[dict[str, Any]] = []
    for layer, values in sorted(layer_values.items()):
        positions = sorted(
            {pos for metric_values in values.values() for pos in metric_values[0]}
        )
        by_metric = {
            key: {int(pos): tensor[idx] for idx, pos in enumerate(xs)}
            for key, (xs, tensor) in values.items()
        }
        for position in positions:
            row: dict[str, Any] = {"position": position, "layer": int(layer)}
            for key, value_by_pos in by_metric.items():
                if position in value_by_pos:
                    row[key] = float(value_by_pos[position].to(torch.float32).item())
            rows.append(row)
    return _write_csv(
        path,
        rows,
        fieldnames=[
            "position",
            "layer",
            "hidden_norm_ratio_to_median",
            "hidden_linf_norm_ratio_to_median",
            "max_received_uniform_ratio",
        ],
    )


def plot_qk_outlier_figures(
    *,
    run_dir: str | Path,
    example_indices: Sequence[int],
    layers: Sequence[int],
    figures_dir: str | Path = "figures",
    tables_dir: str | Path = "tables",
) -> list[Path]:
    """Save per-example figures combining hidden-difference and Q/K outlier plots.

    ``figures_dir`` and ``tables_dir`` are interpreted relative to ``run_dir`` so
    notebook/runtime configuration can redirect the generated
    ``inputs_*_qk_outliers.png`` figures and matching CSV tables together with
    the rest of the run artifacts.
    """

    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    layer_ids = [int(layer) for layer in layers]
    paths: list[Path] = []
    for example_idx in example_indices:
        fig, axes = plt.subplots(5, 1, figsize=(14, 17), sharex=True)
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        plotted_any = False

        measurement_rows = _read_measurement_plot_rows(
            run_dir, int(example_idx), tables_dir=tables_dir
        )
        reference_x = _reference_model_input_x(
            run_dir, int(example_idx), measurement_rows
        )
        needle_spans = _needle_spans_for_example(run_dir, int(example_idx))
        color_by_layer = _layer_colors_from_measurement_rows(
            measurement_rows, layer_ids, color_cycle
        )
        for ax in axes:
            _shade_needle_spans(ax, needle_spans)
        plotted_any = (
            _plot_measurement_rows(
                axes[0],
                measurement_rows,
                metric="relative_norm_diff",
                layers=layer_ids,
                color_by_layer=color_by_layer,
            )
            or plotted_any
        )
        axes[0].set_title("Relative norm difference")
        axes[0].set_ylabel("relative norm diff")
        if axes[0].lines:
            axes[0].legend(loc="best", ncols=4, fontsize="small")

        plotted_any = (
            _plot_measurement_rows(
                axes[1],
                measurement_rows,
                metric="cosine_similarity",
                layers=layer_ids,
                color_by_layer=color_by_layer,
            )
            or plotted_any
        )
        axes[1].set_title("Cosine similarity")
        axes[1].set_ylabel("cosine similarity")
        low_cosine_positions = _low_cosine_positions(measurement_rows, threshold=0.8)
        if axes[1].lines:
            axes[1].legend(loc="best", ncols=4, fontsize="small")

        layer_values: dict[int, dict[str, tuple[list[int], torch.Tensor]]] = {}
        for layer in layer_ids:
            norm_path = (
                run_dir
                / "tensors/massive_activations"
                / f"input_{example_idx}"
                / f"hidden_norms_layer_{layer:02d}.pt"
            )
            if norm_path.exists():
                payload = torch.load(norm_path, map_location="cpu")
                ratio = payload["norm_ratio_to_median"].to(torch.float32).cpu()
                xs, aligned_ratio = _aligned_metric_for_measurement_x(
                    ratio, measurement_x=reference_x
                )
                layer_values.setdefault(layer, {})["hidden_norm_ratio_to_median"] = (
                    xs,
                    aligned_ratio,
                )
                if xs:
                    axes[2].plot(
                        xs,
                        _positive_log_values(aligned_ratio).numpy(),
                        label=f"layer {layer}",
                        color=color_by_layer[layer],
                        linewidth=1.0,
                    )
                    plotted_any = True
                linf_ratio = payload.get(
                    "linf_norm_ratio_to_median",
                    payload.get("max_abs_activation_ratio_to_median"),
                )
                if linf_ratio is not None:
                    linf_tensor = linf_ratio.to(torch.float32).cpu()
                    linf_xs, aligned_linf = _aligned_metric_for_measurement_x(
                        linf_tensor, measurement_x=reference_x
                    )
                    layer_values.setdefault(layer, {})[
                        "hidden_linf_norm_ratio_to_median"
                    ] = (
                        linf_xs,
                        aligned_linf,
                    )
                    if linf_xs:
                        axes[3].plot(
                            linf_xs,
                            _positive_log_values(aligned_linf).numpy(),
                            label=f"layer {layer}",
                            color=color_by_layer[layer],
                            linewidth=1.0,
                        )
                        plotted_any = True
        axes[2].set_title("Hidden-state L2 norm ratio to median")
        axes[2].set_ylabel("L2 norm / median")
        axes[2].set_yscale("log")
        if axes[2].lines:
            axes[2].legend(loc="best", ncols=4, fontsize="small")

        axes[3].set_title("Hidden-state L∞ norm ratio to median")
        axes[3].set_ylabel("L∞ norm / median")
        axes[3].set_yscale("log")
        if axes[3].lines:
            axes[3].legend(loc="best", ncols=4, fontsize="small")

        stats_dir = run_dir / "tensors/attention_stats" / f"input_{example_idx}"
        for layer in layer_ids:
            ratio_max: torch.Tensor | None = None
            for stats_path in sorted(
                stats_dir.glob(f"attention_stats_layer_{layer:02d}_head_*.pt")
            ):
                payload = torch.load(stats_path, map_location="cpu")
                ratio = payload["received_uniform_ratio"].to(torch.float32).cpu()
                ratio = torch.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
                ratio_max = (
                    ratio if ratio_max is None else torch.maximum(ratio_max, ratio)
                )
            if ratio_max is not None:
                xs, aligned_ratio_max = _aligned_metric_for_measurement_x(
                    ratio_max, measurement_x=reference_x
                )
                layer_values.setdefault(layer, {})["max_received_uniform_ratio"] = (
                    xs,
                    aligned_ratio_max,
                )
                if xs:
                    axes[4].plot(
                        xs,
                        _positive_log_values(aligned_ratio_max).numpy(),
                        label=f"layer {layer}",
                        color=color_by_layer[layer],
                        linewidth=1.0,
                    )
                    plotted_any = True
        axes[4].set_title("Max received-attention uniform ratio across heads")
        axes[4].set_xlabel("model input token position")
        axes[4].set_ylabel("received / uniform")
        axes[4].set_yscale("log")
        if axes[4].lines:
            axes[4].legend(loc="best", ncols=4, fontsize="small")

        _mark_low_cosine_positions(axes, low_cosine_positions, threshold=0.8)

        if reference_x is not None:
            min_x = min(reference_x)
            max_x = max(reference_x)
            if min_x == max_x:
                axes[-1].set_xlim(min_x - 0.5, max_x + 0.5)
            else:
                axes[-1].set_xlim(min_x, max_x)
        if layer_values:
            _save_qk_outlier_plot_table(
                run_dir, int(example_idx), layer_values, tables_dir=tables_dir
            )
        fig.tight_layout()
        out_path = run_dir / figures_dir / f"inputs_{example_idx}_qk_outliers.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if plotted_any:
            fig.savefig(out_path)
            print(f"[qk-outlier] saved figure={out_path}")
            paths.append(out_path)
        plt.close(fig)
    return paths


def _runtime_stage(name: str, elapsed_seconds: float, **extra: Any) -> dict[str, Any]:
    payload = {"stage": name, "elapsed_seconds": float(elapsed_seconds)}
    payload.update(extra)
    return payload


def _write_qk_runtime_log(
    run_dir: Path,
    *,
    started_unix_time: float,
    finished_unix_time: float,
    stages: Sequence[dict[str, Any]],
    qk_cache_examples: Sequence[dict[str, Any]],
) -> Path:
    model_inference_total = sum(
        float(row.get("model_inference_elapsed_seconds", 0.0))
        for row in qk_cache_examples
    )
    qk_cache_total = sum(
        float(row.get("qk_cache_elapsed_seconds", 0.0)) for row in qk_cache_examples
    )
    payload = {
        "schema_version": "qk_outlier_runtime_v1",
        "started_unix_time": float(started_unix_time),
        "finished_unix_time": float(finished_unix_time),
        "total_elapsed_seconds": float(finished_unix_time - started_unix_time),
        "stage_totals": {
            "model_inference_elapsed_seconds": model_inference_total,
            "qk_cache_elapsed_seconds": qk_cache_total,
            "analysis_visualization_elapsed_seconds": sum(
                float(stage.get("elapsed_seconds", 0.0))
                for stage in stages
                if stage.get("stage")
                in {
                    "massive_activation_analysis",
                    "attention_analysis",
                    "join_analysis",
                    "visualization",
                }
            ),
        },
        "stages": list(stages),
        "qk_cache_examples": list(qk_cache_examples),
    }
    return write_json(run_dir / "qk_outlier_runtime_log.json", payload)


def save_qk_outlier_config(run_dir: str | Path, cfg: QKOutlierAnalysisConfig) -> Path:
    return write_json(
        Path(run_dir) / "qk_outlier_analysis_config.json", cfg.to_json_dict()
    )


def run_qk_outlier_analysis(
    *,
    run_dir: str | Path,
    layers: Sequence[int],
    heads: Sequence[int] | None = None,
    model: str | None = None,
    repo_root: str | Path = ".",
    force_capture: bool = False,
    massive_norm_ratio_threshold: float = 10.0,
    massive_top_k_per_layer: int = 50,
    n_critical_edge_tokens: int = 10,
    n_after_needle: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    compute_dtype: str = "fp32",
    key_block_size: int = 8192,
    query_block_size: int = 64,
    topk: int = 32,
    capture_attn_implementation: str = "sdpa",
    capture_model_dtype: str = "bf16",
    capture_save_dtype: str = "bf16",
    figures_dir: str | Path = "figures",
    tables_dir: str | Path = "tables",
    example_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """End-to-end uncontrolled-prompt Q/K outlier analysis for one run directory.

    ``example_indices`` maps each row in the generated JSONL to the externally
    visible example id used in artifact names. By default, rows keep their
    historical contiguous zero-based ids.
    """

    run_started_unix_time = time.time()
    run_dir = Path(run_dir)
    repo_root = Path(repo_root)
    runtime_stages: list[dict[str, Any]] = []
    qk_cache_timing_records: list[dict[str, Any]] = []
    stage_start = time.perf_counter()
    dyn_cfg = _dynamic_cfg_from_run(run_dir)
    model_name = model or _model_name_from_run(run_dir, dyn_cfg.tokenizer_name)
    analysis_cfg = QKOutlierAnalysisConfig(
        model=model_name,
        run_dir=str(run_dir),
        layers=tuple(int(x) for x in layers),
        heads=None if heads is None else tuple(int(x) for x in heads),
        massive_norm_ratio_threshold=float(massive_norm_ratio_threshold),
        massive_top_k_per_layer=int(massive_top_k_per_layer),
        n_critical_edge_tokens=int(n_critical_edge_tokens),
        n_after_needle=int(n_after_needle),
        device=device,
        compute_dtype=compute_dtype,
        key_block_size=int(key_block_size),
        query_block_size=int(query_block_size),
        topk=int(topk),
        capture_attn_implementation=capture_attn_implementation,
        capture_model_dtype=capture_model_dtype,
        capture_save_dtype=capture_save_dtype,
        figures_dir=str(figures_dir),
        tables_dir=str(tables_dir),
    )
    config_path = save_qk_outlier_config(run_dir, analysis_cfg)
    runtime_stages.append(_runtime_stage("setup", time.perf_counter() - stage_start))

    dataset_path = run_dir / "generate_data/dynamic_niah_v2.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing generated dataset: {dataset_path}")
    rows = load_jsonl(dataset_path)
    resolved_example_indices = (
        [int(idx) for idx in range(len(rows))]
        if example_indices is None
        else [int(idx) for idx in example_indices]
    )
    if len(resolved_example_indices) != len(rows):
        raise ValueError(
            "example_indices must contain exactly one id per generated dataset row; "
            f"got {len(resolved_example_indices)} id(s) for {len(rows)} row(s)"
        )
    if len(set(resolved_example_indices)) != len(resolved_example_indices):
        raise ValueError(f"example_indices must be unique: {resolved_example_indices}")

    initial_cache_dirs = [
        _qk_root(run_dir, analysis_cfg) / f"input_{idx}"
        for idx in resolved_example_indices
    ]
    needs_capture = force_capture or any(
        not qk_cache_complete(path, analysis_cfg.layers) for path in initial_cache_dirs
    )
    tokenizer = None
    model_obj = None
    if needs_capture:
        stage_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            dyn_cfg.tokenizer_name,
            trust_remote_code=dyn_cfg.trust_remote_code,
            cache_dir=dyn_cfg.cache_dir,
        )
        model_dtype = DTYPE_MAP[analysis_cfg.capture_model_dtype]
        try:
            model_obj = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=model_dtype,
                device_map=dyn_cfg.device_map,
                attn_implementation=analysis_cfg.capture_attn_implementation,
                trust_remote_code=dyn_cfg.trust_remote_code,
                cache_dir=dyn_cfg.cache_dir,
            )
        except Exception as exc:
            if analysis_cfg.capture_attn_implementation != "flash_attention_2":
                raise
            print(
                "[warn] flash_attention_2 model load failed; retrying Q/K capture model load with attn_implementation='sdpa'. "
                f"Original error: {exc}"
            )
            model_obj = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=model_dtype,
                device_map=dyn_cfg.device_map,
                attn_implementation="sdpa",
                trust_remote_code=dyn_cfg.trust_remote_code,
                cache_dir=dyn_cfg.cache_dir,
            )
        model_obj.eval()
        runtime_stages.append(
            _runtime_stage("model_load", time.perf_counter() - stage_start)
        )

    cache_dirs: list[Path] = []
    stage_start = time.perf_counter()
    for example_idx, row in zip(resolved_example_indices, rows, strict=True):
        cache_dir = ensure_qk_cache(
            run_dir=run_dir,
            example_idx=example_idx,
            row=row,
            cfg=dyn_cfg,
            analysis_cfg=analysis_cfg,
            repo_root=repo_root,
            model=model_obj,
            tokenizer=tokenizer,
            force=force_capture,
            timing_records=qk_cache_timing_records,
        )
        print_qk_shape_checks(cache_dir, analysis_cfg.layers)
        cache_dirs.append(cache_dir)
    if model_obj is not None:
        del model_obj
    runtime_stages.append(
        _runtime_stage(
            "qk_cache",
            time.perf_counter() - stage_start,
            skipped=not needs_capture,
            num_examples=len(cache_dirs),
        )
    )
    if torch.cuda.is_available() and needs_capture:
        torch.cuda.empty_cache()

    stage_start = time.perf_counter()
    massive_events, scalar_events, dim_rows = analyze_massive_activations(
        run_dir=run_dir,
        layers=analysis_cfg.layers,
        threshold=analysis_cfg.massive_norm_ratio_threshold,
        top_k=analysis_cfg.massive_top_k_per_layer,
        n_edge=analysis_cfg.n_critical_edge_tokens,
        n_after_needle=analysis_cfg.n_after_needle,
    )
    runtime_stages.append(
        _runtime_stage("massive_activation_analysis", time.perf_counter() - stage_start)
    )

    sink_rows: list[dict[str, Any]] = []
    stage_start = time.perf_counter()
    needle_rows: list[dict[str, Any]] = []
    for example_idx, cache_dir in zip(
        resolved_example_indices, cache_dirs, strict=True
    ):
        per_sink, per_needle = analyze_attention_for_cache(
            run_dir=run_dir,
            example_idx=example_idx,
            cache_dir=cache_dir,
            analysis_cfg=analysis_cfg,
            massive_events=massive_events,
        )
        sink_rows.extend(per_sink)
        needle_rows.extend(per_needle)

    _write_csv(run_dir / "tables/attention_sinks_topk.csv", sink_rows)
    _write_csv(run_dir / "tables/needle_attention_mass.csv", needle_rows)
    runtime_stages.append(
        _runtime_stage("attention_analysis", time.perf_counter() - stage_start)
    )

    stage_start = time.perf_counter()
    joined, overlap = join_outlier_attention(
        run_dir=run_dir, massive_events=massive_events, sink_rows=sink_rows
    )
    runtime_stages.append(
        _runtime_stage("join_analysis", time.perf_counter() - stage_start)
    )

    stage_start = time.perf_counter()
    figure_paths = plot_qk_outlier_figures(
        run_dir=run_dir,
        example_indices=resolved_example_indices,
        layers=analysis_cfg.layers,
        figures_dir=analysis_cfg.figures_dir,
        tables_dir=analysis_cfg.tables_dir,
    )
    runtime_stages.append(
        _runtime_stage("visualization", time.perf_counter() - stage_start)
    )
    runtime_log_path = _write_qk_runtime_log(
        run_dir,
        started_unix_time=run_started_unix_time,
        finished_unix_time=time.time(),
        stages=runtime_stages,
        qk_cache_examples=qk_cache_timing_records,
    )

    summary = {
        "config_path": str(config_path),
        "num_examples": len(rows),
        "example_indices": resolved_example_indices,
        "num_massive_events": len(massive_events),
        "num_scalar_events": len(scalar_events),
        "num_sink_rows": len(sink_rows),
        "num_needle_attention_rows": len(needle_rows),
        "num_joined_rows": len(joined),
        "num_dim_count_rows": len(dim_rows),
        "num_overlap_rows": len(overlap),
        "figure_paths": [str(path) for path in figure_paths],
        "runtime_log_path": str(runtime_log_path),
    }
    write_json(run_dir / "tables/qk_outlier_analysis_summary.json", summary)
    return summary
