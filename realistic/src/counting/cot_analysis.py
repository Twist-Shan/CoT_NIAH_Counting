from __future__ import annotations

import csv
import json
import math
import shutil
import time
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from counting.cot import split_generated_cot
from counting.analysis import (
    cleanup_counting_archive_artifacts,
    build_counting_setting_name,
    load_counting_dataset,
    save_counting_dataset_cache,
    validate_counting_dataset_cache,
)
from counting.feature_analysis import (
    StageTimer,
    print_gpu_memory_snapshot,
    release_torch_memory,
)
from dataset_generation.chat_templates import apply_generation_chat_template
from dataset_generation.dynamic_niah_v2 import (
    DynamicNiahV2Config,
    build_prediction_messages,
    dynamic_niah_v2_config_kwargs,
    generate_dynamic_niah_dataset_v2,
    write_dynamic_niah_v2,
)
from dataset_generation.hidden_state_analysis import build_prompt_needle_spans
from dataset_generation.qk_hook_attention.analyze_qk_qwen3 import (
    COMPUTE_DTYPES,
    compute_query_block_stats,
    load_cache_analysis_spec,
    load_cache_metadata,
    reconstruct_single_head_qk,
)
from dataset_generation.qk_hook_attention.outlier_analysis import run_qk_outlier_analysis
from dataset_generation.response_eval import (
    build_response_result,
    summarize_results,
    write_jsonl,
)
from dataset_generation.run_utils import archive_directory, slugify


SUPPORTED_THINKING_MODES = {"nonthinking", "thinking"}


COT_ANALYSIS_CONFIG_DEFAULTS: dict[str, Any] = {
    # Dataset settings.
    "TASK_TYPE": "match_count",
    "MODEL_NAME": "Qwen/Qwen3-8B",
    "TOKENIZER_NAME": None,
    "NUM_EXAMPLES": 100,
    "TARGET_HAYSTACK_TOKENS": 1000,
    "NUM_NEEDLES": 3,
    "NUM_MAX_NEEDLES": None,
    "INSERTION_POSITIONS": [0, 0, 0],
    "RANDOMIZE_NEEDLE_INSERTION": True,
    "RANDOMIZE_NEEDLE_SEED": 42,
    "SENTENCE_LEVEL_INSERTION": False,
    "WORD_LEVEL_INSERTION": True,
    "GLOBAL_RANDOM_SEED": 42,
    "HAYSTACK_SEED": 123,
    "NEEDLE_SEED": 456,
    "FACT_TEMPLATES_PATH": "data/templates/niah_fact_single_template.txt",
    "COUNTING_NEEDLE_KIND": "city_score",
    "MARKER_TEXT": "[dolphin]",
    "UID_TOKEN_LENGTH": 4,
    "PROMPT_STYLE": "vanilla",
    # Generation settings.
    "THINKING_MODES": ["nonthinking", "thinking"],
    "MAX_NEW_TOKENS_NONTHINKING": 64,
    "MAX_NEW_TOKENS_THINKING": 1024,
    "GENERATION_TEMPERATURE": 0.0,
    "GENERATION_DO_SAMPLE": None,
    "GENERATION_TOP_P": None,
    "USE_KV_CACHE_FOR_NONTHINKING": True,
    "USE_KV_CACHE_FOR_THINKING": True,
    # Analysis settings.
    "LAYERS": [4, 8, 12, 16, 20, 24, 28],
    "K": 32,
    "OUTLIER_RATIO_THRESHOLD": 5.0,
    "MAX_ANALYSIS_EXAMPLES": 10,
    "ANALYSIS_EXAMPLE_SELECTION": "balanced_success_failure",
    "ATTENTION_HEAD_AGG": "max_across_heads",
    "SAVE_PER_HEAD_ATTENTION_TABLES": True,
    "SAVE_QK_CACHE": False,
    "SAVE_FULL_HIDDEN_STATES": False,
    "ANALYSIS_DTYPE": "bfloat16_if_cuda_else_float32",
    "PLOT_SPECIAL_TOKEN_LINES": True,
    "THINKING_MARKER_STRINGS": ["</think>"],
    "FINAL_ANSWER_MARKER_STRINGS": [],
    "CAPTURE_ATTN_IMPLEMENTATION": "sdpa",
    "CAPTURE_MODEL_DTYPE": "bf16",
    "CAPTURE_SAVE_DTYPE": "bf16",
    "QK_COMPUTE_DTYPE": "fp32",
    "QK_KEY_BLOCK_SIZE": 8192,
    "QK_QUERY_BLOCK_SIZE": 64,
    "RUN_DATASET_GENERATION": True,
    "RUN_RESPONSE_GENERATION": True,
    "RUN_OUTLIER_ANALYSIS": True,
    "RUN_ATTENTION_ANALYSIS": True,
    "RUN_PLOTTING": True,
    # I/O settings.
    "USER_RUN_NAME": None,
    "RUN_ROOT": "/content",
    "DATA_CACHE_ROOT": "data/niah-example",
    "RESULTS_PATH": None,
    "FORCE_REGENERATE_DATASET": False,
    "FORCE_REGENERATE_RESPONSES": False,
    "ZIP_RESULTS_AFTER_RESPONSE_STAGE": True,
    "ZIP_RESULTS_AFTER_ANALYSIS_STAGE": True,
    "LOG_GPU_MEMORY": True,
    "LOG_CPU_MEMORY": True,
}


@dataclass(frozen=True)
class CotRunPaths:
    setting_name: str
    cache_dir: Path
    run_name: str
    run_dir: Path
    config_path: Path
    logs_dir: Path
    timing_json_path: Path
    timing_csv_path: Path
    dataset_path: Path
    response_checkpoint_zip: Path
    archives_dir: Path

    def mode_dir(self, mode: str) -> Path:
        return self.run_dir / "modes" / mode

    def response_dir(self, mode: str) -> Path:
        return self.run_dir / "responses" / mode

    def predictions_path(self, mode: str) -> Path:
        return self.response_dir(mode) / "predictions.jsonl"

    def metrics_path(self, mode: str) -> Path:
        return self.response_dir(mode) / "metrics.json"


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return out


def _write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")
    return out


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(str(key))
                seen.add(str(key))
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return out


def normalize_thinking_modes(modes: Sequence[str]) -> list[str]:
    out: list[str] = []
    for raw in modes:
        mode = str(raw).strip().lower().replace("-", "_")
        if mode in {"non_thinking", "no_thinking", "false", "0"}:
            mode = "nonthinking"
        if mode in {"true", "1"}:
            mode = "thinking"
        if mode not in SUPPORTED_THINKING_MODES:
            raise ValueError(
                f"THINKING_MODES entries must be one of {sorted(SUPPORTED_THINKING_MODES)}, got {raw!r}"
            )
        if mode not in out:
            out.append(mode)
    if not out:
        raise ValueError("THINKING_MODES must contain at least one mode")
    return out


def thinking_mode_to_bool(mode: str) -> bool:
    mode = normalize_thinking_modes([mode])[0]
    return mode == "thinking"


def build_cot_analysis_run_config(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(COT_ANALYSIS_CONFIG_DEFAULTS)
    unknown = sorted(set(overrides or {}) - set(cfg))
    if unknown:
        raise ValueError(f"Unknown CoT-analysis config override(s): {unknown}")
    cfg.update(dict(overrides or {}))
    cfg["THINKING_MODES"] = normalize_thinking_modes(cfg["THINKING_MODES"])
    if cfg["TOKENIZER_NAME"] is None:
        cfg["TOKENIZER_NAME"] = cfg["MODEL_NAME"]
    cfg["LAYERS"] = [int(layer) for layer in cfg["LAYERS"]]
    cfg["K"] = int(cfg["K"])
    if cfg["K"] <= 0:
        raise ValueError(f"K must be positive, got {cfg['K']}")
    cfg["OUTLIER_RATIO_THRESHOLD"] = float(cfg["OUTLIER_RATIO_THRESHOLD"])
    if cfg["OUTLIER_RATIO_THRESHOLD"] < 0:
        raise ValueError(
            "OUTLIER_RATIO_THRESHOLD must be non-negative, "
            f"got {cfg['OUTLIER_RATIO_THRESHOLD']}"
        )
    cfg["MAX_ANALYSIS_EXAMPLES"] = int(cfg["MAX_ANALYSIS_EXAMPLES"])
    if cfg["MAX_ANALYSIS_EXAMPLES"] <= 0:
        raise ValueError(
            "MAX_ANALYSIS_EXAMPLES must be positive, "
            f"got {cfg['MAX_ANALYSIS_EXAMPLES']}"
        )
    if cfg["NUM_MAX_NEEDLES"] is not None:
        cfg["NUM_MAX_NEEDLES"] = int(cfg["NUM_MAX_NEEDLES"])
        if cfg["NUM_MAX_NEEDLES"] <= 0:
            raise ValueError("NUM_MAX_NEEDLES must be None or a positive integer")
    cfg["INSERTION_POSITIONS"] = list(cfg["INSERTION_POSITIONS"])
    cfg["RUN_ROOT"] = str(cfg["RUN_ROOT"])
    cfg["DATA_CACHE_ROOT"] = str(cfg["DATA_CACHE_ROOT"])
    if cfg["RESULTS_PATH"] is not None:
        cfg["RESULTS_PATH"] = str(cfg["RESULTS_PATH"])
    if cfg["ATTENTION_HEAD_AGG"] != "max_across_heads":
        raise ValueError("Only ATTENTION_HEAD_AGG='max_across_heads' is implemented")
    return cfg


def cot_run_name(cfg: Mapping[str, Any]) -> str:
    if cfg.get("USER_RUN_NAME"):
        return str(cfg["USER_RUN_NAME"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = slugify(str(cfg["MODEL_NAME"]).replace("/", "_"), max_length=80)
    modes = "-".join(cfg["THINKING_MODES"])
    max_needles = cfg.get("NUM_MAX_NEEDLES")
    needle_part = (
        f"num_max_needles_{max_needles}"
        if max_needles is not None
        else f"needles_{cfg['NUM_NEEDLES']}"
    )
    return (
        f"run_{timestamp}_{model}_{cfg['TASK_TYPE']}_cot_{modes}_"
        f"{cfg['TARGET_HAYSTACK_TOKENS']}_{needle_part}"
    )


def cot_setting_name(cfg: Mapping[str, Any]) -> str:
    """Return a timestamp-free cache key for reusable CoT datasets/responses."""

    base = build_counting_setting_name(
        model_name=str(cfg["MODEL_NAME"]),
        task_type=str(cfg["TASK_TYPE"]),
        prompt_style=str(cfg["PROMPT_STYLE"]),
        target_haystack_tokens=int(cfg["TARGET_HAYSTACK_TOKENS"]),
        num_examples=int(cfg["NUM_EXAMPLES"]),
        insertion_positions=list(cfg["INSERTION_POSITIONS"]),
        global_random_seed=int(cfg["GLOBAL_RANDOM_SEED"]),
        haystack_seed=cfg["HAYSTACK_SEED"],
        needle_seed=cfg["NEEDLE_SEED"],
        # Dataset rows are shared across thinking/non-thinking in this workflow.
        thinking_mode=False,
        num_max_needles=cfg["NUM_MAX_NEEDLES"],
        randomize_needle_insertion=bool(cfg["RANDOMIZE_NEEDLE_INSERTION"]),
        randomize_needle_seed=int(cfg["RANDOMIZE_NEEDLE_SEED"]),
        sentence_level_insertion=bool(cfg["SENTENCE_LEVEL_INSERTION"]),
        word_level_insertion=bool(cfg["WORD_LEVEL_INSERTION"]),
        fact_templates_path=str(cfg["FACT_TEMPLATES_PATH"]),
        counting_needle_kind=str(cfg["COUNTING_NEEDLE_KIND"]),
        uid_token_length=int(cfg["UID_TOKEN_LENGTH"]),
    )
    return "cot_fullseq_" + base


def cot_expected_dataset_cache_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_type": cfg["TASK_TYPE"],
        "tokenizer_name": cfg["TOKENIZER_NAME"],
        "num_examples": int(cfg["NUM_EXAMPLES"]),
        "target_haystack_tokens": int(cfg["TARGET_HAYSTACK_TOKENS"]),
        "num_needles": int(cfg["NUM_NEEDLES"]),
        "num_max_needles": cfg["NUM_MAX_NEEDLES"],
        "insertion_positions": list(cfg["INSERTION_POSITIONS"]),
        "randomize_needle_insertion": bool(cfg["RANDOMIZE_NEEDLE_INSERTION"]),
        "randomize_needle_seed": int(cfg["RANDOMIZE_NEEDLE_SEED"]),
        "sentence_level_insertion": bool(cfg["SENTENCE_LEVEL_INSERTION"]),
        "word_level_insertion": bool(cfg["WORD_LEVEL_INSERTION"]),
        "prompt_style": cfg["PROMPT_STYLE"],
        "counting_needle_kind": cfg["COUNTING_NEEDLE_KIND"],
        "marker_text": cfg["MARKER_TEXT"],
        "uid_token_length": int(cfg["UID_TOKEN_LENGTH"]),
        "thinking_mode": False,
        "global_random_seed": int(cfg["GLOBAL_RANDOM_SEED"]),
        "haystack_seed": cfg["HAYSTACK_SEED"],
        "needle_seed": cfg["NEEDLE_SEED"],
        "fact_templates_path": str(cfg["FACT_TEMPLATES_PATH"]),
    }


def create_cot_run_paths(cfg: Mapping[str, Any]) -> CotRunPaths:
    run_name = cot_run_name(cfg)
    setting_name = cot_setting_name(cfg)
    run_dir = Path(cfg["RUN_ROOT"]) / run_name
    cache_dir = Path(cfg["DATA_CACHE_ROOT"]) / setting_name
    logs_dir = run_dir / "logs"
    paths = CotRunPaths(
        setting_name=setting_name,
        cache_dir=cache_dir,
        run_name=run_name,
        run_dir=run_dir,
        config_path=run_dir / "config.json",
        logs_dir=logs_dir,
        timing_json_path=logs_dir / "timing_summary.json",
        timing_csv_path=logs_dir / "timing_summary.csv",
        dataset_path=run_dir / "generate_data" / "dynamic_niah_v2.jsonl",
        response_checkpoint_zip=run_dir / f"{run_name}_response_generation_checkpoint.zip",
        archives_dir=run_dir / "archives",
    )
    paths.dataset_path.parent.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    paths.archives_dir.mkdir(parents=True, exist_ok=True)
    for mode in cfg["THINKING_MODES"]:
        paths.response_dir(mode).mkdir(parents=True, exist_ok=True)
        paths.mode_dir(mode).mkdir(parents=True, exist_ok=True)
    _write_json(
        paths.config_path,
        dict(cfg)
        | {
            "setting_name": setting_name,
            "cache_dir": str(cache_dir),
            "run_name": run_name,
            "run_dir": str(run_dir),
        },
    )
    return paths


def dynamic_config_from_cot_config(
    cfg: Mapping[str, Any],
    *,
    run_dir: str | Path,
    dataset_path: str | Path,
    mode: str = "nonthinking",
) -> DynamicNiahV2Config:
    use_thinking = thinking_mode_to_bool(mode)
    max_new_tokens = (
        int(cfg["MAX_NEW_TOKENS_THINKING"])
        if use_thinking
        else int(cfg["MAX_NEW_TOKENS_NONTHINKING"])
    )
    do_sample = cfg.get("GENERATION_DO_SAMPLE")
    temperature = float(cfg["GENERATION_TEMPERATURE"])
    payload = {
        "task_type": cfg["TASK_TYPE"],
        "tokenizer_name": cfg["TOKENIZER_NAME"],
        "num_examples": int(cfg["NUM_EXAMPLES"]),
        "target_haystack_tokens": int(cfg["TARGET_HAYSTACK_TOKENS"]),
        "num_needles": int(cfg["NUM_NEEDLES"]),
        "num_max_needles": cfg["NUM_MAX_NEEDLES"],
        "insertion_positions": tuple(cfg["INSERTION_POSITIONS"]),
        "randomize_needle_insertion": bool(cfg["RANDOMIZE_NEEDLE_INSERTION"]),
        "randomize_needle_seed": int(cfg["RANDOMIZE_NEEDLE_SEED"]),
        "sentence_level_insertion": bool(cfg["SENTENCE_LEVEL_INSERTION"]),
        "word_level_insertion": bool(cfg["WORD_LEVEL_INSERTION"]),
        "global_random_seed": int(cfg["GLOBAL_RANDOM_SEED"]),
        "haystack_seed": int(cfg["HAYSTACK_SEED"]),
        "needle_seed": int(cfg["NEEDLE_SEED"]),
        "fact_templates_path": str(cfg["FACT_TEMPLATES_PATH"]),
        "counting_needle_kind": str(cfg["COUNTING_NEEDLE_KIND"]),
        "marker_text": str(cfg["MARKER_TEXT"]),
        "uid_token_length": int(cfg["UID_TOKEN_LENGTH"]),
        "prompt_style": str(cfg["PROMPT_STYLE"]),
        "thinking_mode": use_thinking,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "output_dir": str(Path(dataset_path).parent),
        "data_save_path": str(dataset_path),
        "output_pred_jsonl": str(Path(run_dir) / "predictions.jsonl"),
        "output_metrics_json": str(Path(run_dir) / "metrics.json"),
        "run_dir": str(run_dir),
        "run_name": Path(run_dir).name,
        "analyze_reasoning_tokens": True,
    }
    if do_sample is not None and not bool(do_sample):
        payload["temperature"] = 0.0
    return DynamicNiahV2Config(**dynamic_niah_v2_config_kwargs(payload))


def generate_or_load_dataset(
    cfg: Mapping[str, Any],
    paths: CotRunPaths,
    *,
    force: bool = False,
) -> list[dict[str, Any]]:
    if paths.dataset_path.exists() and not force:
        print(f"[cot-analysis] using existing dataset: {paths.dataset_path}", flush=True)
        return load_counting_dataset(paths.dataset_path)
    if paths.cache_dir.exists() and not force:
        try:
            cache_info = validate_counting_dataset_cache(
                paths.cache_dir,
                cot_expected_dataset_cache_config(cfg),
            )
            print(f"[cot-analysis] validated reusable dataset cache: {paths.cache_dir}", flush=True)
            shutil.copyfile(cache_info["dataset_path"], paths.dataset_path)
            shutil.copyfile(cache_info["config_path"], paths.dataset_path.parent / "config.used.json")
            return load_counting_dataset(paths.dataset_path)
        except Exception as exc:
            print(
                "[cot-analysis] existing reusable dataset cache did not validate; "
                f"regenerating. Reason: {exc!r}",
                flush=True,
            )
    print(f"[cot-analysis] generating dataset: {paths.dataset_path}", flush=True)
    dyn_cfg = dynamic_config_from_cot_config(
        cfg,
        run_dir=paths.run_dir,
        dataset_path=paths.dataset_path,
        mode="nonthinking",
    )
    rows = generate_dynamic_niah_dataset_v2(dyn_cfg)
    write_dynamic_niah_v2(rows, dyn_cfg)
    try:
        save_counting_dataset_cache(
            cache_dir=paths.cache_dir,
            dataset_path=paths.dataset_path,
            config_path=paths.dataset_path.parent / "config.used.json",
            overwrite=True,
        )
        print(f"[cot-analysis] saved reusable dataset cache: {paths.cache_dir}", flush=True)
    except Exception as exc:
        print(f"[cot-analysis] warning: could not save reusable dataset cache: {exc!r}", flush=True)
    print(f"[cot-analysis] generated {len(rows)} row(s)", flush=True)
    return rows


def _resolve_torch_dtype(value: str) -> torch.dtype:
    if value == "bfloat16_if_cuda_else_float32":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return getattr(torch, value)


def _model_input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


def _needle_count(row: Mapping[str, Any]) -> int:
    gold = row.get("gold_answer")
    if isinstance(gold, Mapping) and gold.get("count") is not None:
        try:
            return int(gold["count"])
        except Exception:
            pass
    return len(row.get("needles", []) or [])


def _count_bin(count: int, unique_counts: Sequence[int]) -> str:
    unique = sorted({int(value) for value in unique_counts})
    if len(unique) <= 1:
        return "all"
    rank = unique.index(int(count))
    if len(unique) == 2:
        return "low" if rank == 0 else "high"
    frac = rank / max(1, len(unique) - 1)
    if frac <= 1.0 / 3.0:
        return "low"
    if frac <= 2.0 / 3.0:
        return "medium"
    return "high"


def _success_state_for_index(
    idx: int,
    results_by_mode: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[str, dict[str, bool | None]]:
    by_mode: dict[str, bool | None] = {}
    for mode, results in results_by_mode.items():
        if idx < len(results) and results[idx].get("exact_match") is not None:
            by_mode[str(mode)] = bool(results[idx].get("exact_match"))
        else:
            by_mode[str(mode)] = None
    known = [value for value in by_mode.values() if value is not None]
    if not known:
        return "unknown", by_mode
    if any(value is False for value in known):
        return "failed", by_mode
    if any(value is None for value in by_mode.values()):
        return "unknown", by_mode
    return "success", by_mode


def select_analysis_example_ids(
    *,
    rows: Sequence[Mapping[str, Any]],
    results_by_mode: Mapping[str, Sequence[Mapping[str, Any]]],
    max_examples: int,
    analysis_eligibility_by_idx: Mapping[int, Mapping[str, Any]] | None = None,
) -> list[int]:
    """Select common example ids across modes, balanced by success/failure/count."""

    n = len(rows)
    if n == 0:
        return []
    requested_max_examples = int(max_examples)
    eligible_indices = list(range(n))
    excluded: list[dict[str, Any]] = []
    if analysis_eligibility_by_idx is not None:
        eligible_indices = []
        for idx in range(n):
            status = analysis_eligibility_by_idx.get(idx, {"eligible": True})
            if bool(status.get("eligible", True)):
                eligible_indices.append(idx)
            else:
                excluded.append(
                    {
                        "example_id": idx,
                        "missing_needle_ids": status.get("missing_needle_ids", []),
                        "missing_by_mode": status.get("missing_by_mode", {}),
                    }
                )
    if excluded:
        print(
            "[cot-analysis] excluding examples from analysis after needle-span "
            "verification: "
            + json.dumps(excluded, ensure_ascii=False, default=str),
            flush=True,
        )
    if len(eligible_indices) < requested_max_examples:
        print(
            "WARNING: only "
            f"{len(eligible_indices)} eligible examples remain after "
            "needle-span verification; fewer than "
            f"MAX_ANALYSIS_EXAMPLES={requested_max_examples} will be analyzed.",
            flush=True,
        )
    if not eligible_indices:
        return []
    max_examples = min(requested_max_examples, len(eligible_indices))
    counts = [_needle_count(row) for row in rows]
    unique_counts = sorted(set(counts))
    metadata_by_idx: dict[int, dict[str, Any]] = {}
    groups: dict[tuple[str, str], list[int]] = {}
    for idx in eligible_indices:
        row = rows[idx]
        count = _needle_count(row)
        bin_name = _count_bin(count, unique_counts)
        success_state, by_mode = _success_state_for_index(idx, results_by_mode)
        metadata_by_idx[idx] = {
            "count": count,
            "count_bin": bin_name,
            "success_state": success_state,
            "by_mode": by_mode,
        }
        groups.setdefault((bin_name, success_state), []).append(idx)

    selected: list[int] = []
    bin_order = {"low": 0, "medium": 1, "high": 2, "all": 3}
    state_order = {"failed": 0, "success": 1, "unknown": 2}
    group_keys = sorted(
        groups,
        key=lambda key: (state_order.get(key[1], 99), bin_order.get(key[0], 99)),
    )
    before_counts = {f"{key[0]}:{key[1]}": len(value) for key, value in groups.items()}
    while len(selected) < max_examples and any(groups.values()):
        for key in group_keys:
            bucket = groups[key]
            if bucket and len(selected) < max_examples:
                selected.append(bucket.pop(0))
    selected = sorted(selected)
    after_counts = {f"{key[0]}:{key[1]}": len(value) for key, value in groups.items()}
    print(
        "[cot-analysis] selected analysis examples: "
        + json.dumps(
            {
                "max_examples": max_examples,
                "requested_max_examples": requested_max_examples,
                "eligible_examples": len(eligible_indices),
                "bucket_counts_before": before_counts,
                "bucket_counts_after": after_counts,
                "selected": [
                    {"example_id": idx, **metadata_by_idx[idx]} for idx in selected
                ],
            },
            ensure_ascii=False,
            default=str,
        ),
        flush=True,
    )
    return selected


def _row_realized_insertions(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(
        row.get("uncontrolled_realized_insertions")
        or row.get("realized_insertions")
        or []
    )


def _needle_span_verification_status(
    *,
    row: Mapping[str, Any],
    prompt_input_ids: torch.Tensor,
    prompt_text: str,
) -> dict[str, Any]:
    insertions = _row_realized_insertions(row)
    expected_ids = [str(item.get("needle_id")) for item in insertions]
    spans = build_prompt_needle_spans(
        prompt_input_ids,
        insertions,
        prompt_text=prompt_text,
    )
    found_ids = {str(span.get("needle_id")) for span in spans}
    missing_ids = [needle_id for needle_id in expected_ids if needle_id not in found_ids]
    return {
        "eligible": not missing_ids,
        "expected_needle_ids": expected_ids,
        "found_needle_ids": sorted(found_ids),
        "missing_needle_ids": missing_ids,
        "needle_spans_verified_for_analysis": not missing_ids,
        "needle_span_count": len(spans),
        "needle_spans": spans,
    }


def collect_analysis_needle_span_eligibility(
    *,
    cfg: Mapping[str, Any],
    paths: CotRunPaths,
    rows: Sequence[dict[str, Any]],
    modes: Sequence[str],
    tokenizer: Any,
) -> dict[int, dict[str, Any]]:
    """Check whether each example's inserted needles can be located in prompts."""

    combined: dict[int, dict[str, Any]] = {
        idx: {
            "eligible": True,
            "missing_needle_ids": [],
            "missing_by_mode": {},
            "mode_status": {},
        }
        for idx in range(len(rows))
    }
    for mode in normalize_thinking_modes(modes):
        dyn_cfg = dynamic_config_from_cot_config(
            cfg,
            run_dir=paths.mode_dir(mode),
            dataset_path=paths.dataset_path,
            mode=mode,
        )
        use_thinking = thinking_mode_to_bool(mode)
        for idx, row in enumerate(rows):
            eval_messages = build_prediction_messages(dyn_cfg, row)
            prompt_text = apply_generation_chat_template(
                tokenizer,
                eval_messages,
                thinking_mode=use_thinking,
            )
            prompt_input_ids = tokenizer(prompt_text, return_tensors="pt")[
                "input_ids"
            ].detach().cpu()
            status = _needle_span_verification_status(
                row=row,
                prompt_input_ids=prompt_input_ids,
                prompt_text=prompt_text,
            )
            combined[idx]["mode_status"][mode] = status
            if not bool(status["eligible"]):
                combined[idx]["eligible"] = False
                missing = [str(x) for x in status.get("missing_needle_ids", [])]
                combined[idx]["missing_by_mode"][mode] = missing
                merged = list(combined[idx]["missing_needle_ids"])
                for needle_id in missing:
                    if needle_id not in merged:
                        merged.append(needle_id)
                combined[idx]["missing_needle_ids"] = merged
            del prompt_input_ids
    excluded = {
        idx: {
            "missing_needle_ids": status["missing_needle_ids"],
            "missing_by_mode": status["missing_by_mode"],
        }
        for idx, status in combined.items()
        if not bool(status["eligible"])
    }
    print(
        "[cot-analysis] needle-span analysis eligibility: "
        + json.dumps(
            {
                "num_examples": len(rows),
                "eligible": sum(1 for status in combined.values() if status["eligible"]),
                "excluded": excluded,
            },
            ensure_ascii=False,
            default=str,
        ),
        flush=True,
    )
    return combined


def _full_sequence_tensor_path(tensors_dir: str | Path, example_id: int) -> Path:
    return Path(tensors_dir) / f"inputs_cot_{int(example_id)}.pt"


def save_full_sequence_artifact(
    *,
    tensors_dir: str | Path,
    tables_dir: str | Path,
    example_id: int,
    row: Mapping[str, Any],
    prompt_input_ids: torch.Tensor,
    generated_ids: torch.Tensor,
    tokenizer: Any,
    task_type: str,
    max_new_tokens: int,
    prompt_text: str,
    mode: str,
) -> dict[str, Any]:
    tensors = Path(tensors_dir)
    tables = Path(tables_dir)
    tensors.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    prompt_ids = [int(x) for x in prompt_input_ids.detach().cpu().reshape(-1).tolist()]
    gen_ids_all = [int(x) for x in generated_ids.detach().cpu().reshape(-1).tolist()]
    split = split_generated_cot(
        gen_ids_all,
        tokenizer,
        task_type=task_type,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        max_new_tokens=max_new_tokens,
    )
    eos_index = split.get("eos_index")
    decode_limit = len(gen_ids_all) if eos_index is None else int(eos_index)
    gen_ids = gen_ids_all[:decode_limit]
    full_ids = prompt_ids + gen_ids
    prompt_tokens = len(prompt_ids)
    needle_status = _needle_span_verification_status(
        row=row,
        prompt_input_ids=torch.tensor([prompt_ids], dtype=torch.long),
        prompt_text=prompt_text,
    )
    needle_spans = list(needle_status["needle_spans"])
    answer_start = int(split["answer_start_generated_index"])
    think_end = int(split.get("thinking_end_marker_token_end") or answer_start)
    boundary_rows = [
        {
            "example_id": int(example_id),
            "mode": mode,
            "boundary_type": "prompt_generation_boundary",
            "full_position": prompt_tokens,
            "generated_position": 0,
        }
    ]
    if split.get("thinking_end_marker_found"):
        boundary_rows.append(
            {
                "example_id": int(example_id),
                "mode": mode,
                "boundary_type": "thinking_end",
                "full_position": prompt_tokens + think_end,
                "generated_position": think_end,
            }
        )
    if answer_start < len(gen_ids):
        boundary_rows.append(
            {
                "example_id": int(example_id),
                "mode": mode,
                "boundary_type": "final_answer_start",
                "full_position": prompt_tokens + answer_start,
                "generated_position": answer_start,
            }
        )
    tensor_path = _full_sequence_tensor_path(tensors, example_id)
    payload = {
        "schema_version": "counting_cot_full_sequence_v1",
        "example_id": int(example_id),
        "row_id": row.get("id"),
        "task_type": task_type,
        "mode": mode,
        "input_ids": torch.tensor([full_ids], dtype=torch.long),
        "prompt_input_ids": torch.tensor([prompt_ids], dtype=torch.long),
        "generated_ids": torch.tensor([gen_ids], dtype=torch.long),
        "prompt_tokens": prompt_tokens,
        "generation_tokens": len(gen_ids),
        "full_sequence_tokens": len(full_ids),
        "needle_spans": needle_spans,
        "needle_spans_verified_for_analysis": bool(needle_status["eligible"]),
        "missing_needle_ids_for_analysis": list(needle_status["missing_needle_ids"]),
        "expected_needle_ids_for_analysis": list(needle_status["expected_needle_ids"]),
        "answer_start_generated_index": answer_start,
        "answer_start_full_index": prompt_tokens + answer_start,
        "thinking_end_full_index": (
            prompt_tokens + think_end if split.get("thinking_end_marker_found") else None
        ),
        **{k: v for k, v in split.items() if not k.endswith("_ids")},
    }
    torch.save(payload, tensor_path)
    _write_csv(tables / f"full_sequence_boundaries_{int(example_id)}.csv", boundary_rows)
    (tables / f"full_sequence_{int(example_id)}.txt").write_text(
        "PROMPT\n======\n"
        + prompt_text
        + "\n\nGENERATED\n=========\n"
        + tokenizer.decode(gen_ids, skip_special_tokens=False)
        + "\n",
        encoding="utf-8",
    )
    return {
        k: v
        for k, v in payload.items()
        if not isinstance(v, torch.Tensor)
    } | {"tensor_path": str(tensor_path)}


def _layer_modules(model: Any) -> Sequence[Any]:
    for attr in ("model", "transformer", "gpt_neox"):
        root = getattr(model, attr, None)
        layers = getattr(root, "layers", None)
        if layers is not None:
            return layers
        layers = getattr(root, "h", None)
        if layers is not None:
            return layers
    layers = getattr(model, "layers", None)
    if layers is not None:
        return layers
    raise ValueError("Could not locate transformer block list on model")


@torch.no_grad()
def capture_selected_hidden_states(
    *,
    model: Any,
    input_ids: torch.Tensor,
    layers: Sequence[int],
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    modules = _layer_modules(model)
    layer_ids = [int(layer) for layer in layers]
    bad = [layer for layer in layer_ids if layer < 0 or layer >= len(modules)]
    if bad:
        raise ValueError(f"Invalid layer ids {bad}; model has {len(modules)} layers")
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_id: int):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            tensor = value.detach()[0].to("cpu")
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            captured[layer_id] = tensor

        return hook

    for layer in layer_ids:
        handles.append(modules[layer].register_forward_hook(make_hook(layer)))
    input_device = _model_input_device(model)
    try:
        _ = model(
            input_ids=input_ids.to(input_device),
            attention_mask=torch.ones_like(input_ids).to(input_device),
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
            logits_to_keep=1,
        )
    finally:
        for handle in handles:
            handle.remove()
    missing = [layer for layer in layer_ids if layer not in captured]
    if missing:
        raise RuntimeError(f"Failed to capture hidden states for layers {missing}")
    return torch.stack([captured[layer] for layer in layer_ids], dim=0)


def save_selected_hidden_states_for_examples(
    *,
    model: Any,
    mode_dir: str | Path,
    example_ids: Sequence[int],
    layers: Sequence[int],
    hidden_state_dtype: torch.dtype | None = None,
) -> list[Path]:
    mode_root = Path(mode_dir)
    out_paths: list[Path] = []
    total = len(example_ids)
    for pos, example_id in enumerate(example_ids, start=1):
        print(
            f"[cot-analysis] hidden states {pos}/{total}: mode_dir={mode_root.name} example={int(example_id)}",
            flush=True,
        )
        payload = torch.load(
            _full_sequence_tensor_path(mode_root / "tensors", int(example_id)),
            map_location="cpu",
        )
        input_ids = payload["input_ids"].to(dtype=torch.long)
        hidden = capture_selected_hidden_states(
            model=model,
            input_ids=input_ids,
            layers=layers,
            dtype=hidden_state_dtype,
        )
        path = mode_root / "tensors" / f"hidden_inputs_{int(example_id)}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        layer_ids = [int(layer) for layer in layers]
        torch.save(
            {
                "hidden": hidden,
                "hidden_control": hidden,
                "sample_idx": int(example_id),
                "layers": layer_ids,
                "stored_layers": layer_ids,
                "input_ids": [int(x) for x in input_ids.reshape(-1).tolist()],
                "input_ids_control": [int(x) for x in input_ids.reshape(-1).tolist()],
                "insertion_position": None,
                "offset": None,
                "pca_start_position": None,
                "needle_spans": list(payload.get("needle_spans") or []),
                "expanded_needle_segments": list(payload.get("needle_spans") or []),
            },
            path,
        )
        out_paths.append(path)
        print(f"[cot-analysis] saved hidden states: {path}", flush=True)
        del hidden
        release_torch_memory(collect_garbage=True)
    return out_paths


def run_generation_for_mode(
    *,
    cfg: Mapping[str, Any],
    paths: CotRunPaths,
    mode: str,
    rows: Sequence[dict[str, Any]],
    model: Any,
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mode = normalize_thinking_modes([mode])[0]
    use_thinking = thinking_mode_to_bool(mode)
    max_new_tokens = (
        int(cfg["MAX_NEW_TOKENS_THINKING"])
        if use_thinking
        else int(cfg["MAX_NEW_TOKENS_NONTHINKING"])
    )
    response_dir = paths.response_dir(mode)
    tensors_dir = paths.mode_dir(mode) / "tensors"
    tables_dir = paths.mode_dir(mode) / "tables"
    results: list[dict[str, Any]] = []
    full_infos: list[dict[str, Any]] = []
    input_device = _model_input_device(model)
    total = len(rows)
    for idx, row in enumerate(rows):
        print(
            f"[cot-analysis] generating mode={mode} example {idx + 1}/{total} id={row.get('id')}",
            flush=True,
        )
        dyn_cfg = dynamic_config_from_cot_config(
            cfg,
            run_dir=paths.mode_dir(mode),
            dataset_path=paths.dataset_path,
            mode=mode,
        )
        eval_messages = build_prediction_messages(dyn_cfg, row)
        prompt_text = apply_generation_chat_template(
            tokenizer, eval_messages, thinking_mode=use_thinking
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(input_device)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": bool(cfg["GENERATION_TEMPERATURE"] > 0),
            "pad_token_id": tokenizer.eos_token_id,
            "use_cache": bool(
                cfg["USE_KV_CACHE_FOR_THINKING"]
                if use_thinking
                else cfg["USE_KV_CACHE_FOR_NONTHINKING"]
            ),
        }
        if float(cfg["GENERATION_TEMPERATURE"]) > 0:
            generation_kwargs["temperature"] = float(cfg["GENERATION_TEMPERATURE"])
        if cfg.get("GENERATION_TOP_P") is not None:
            generation_kwargs["top_p"] = float(cfg["GENERATION_TOP_P"])
        with torch.inference_mode():
            output = model.generate(**inputs, **generation_kwargs)
        prompt_len = int(inputs["input_ids"].shape[1])
        gen_ids = output[0][prompt_len:].detach().cpu()
        model_output = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        result = build_response_result(row, model_output)
        result.update({"example_id": idx, "mode": mode})
        results.append(result)
        print(
            f"[cot-analysis] scored mode={mode} example={idx} exact={result.get('exact_match')} "
            f"parse_mode={result.get('parse_mode')}",
            flush=True,
        )
        full_infos.append(
            save_full_sequence_artifact(
                tensors_dir=tensors_dir,
                tables_dir=tables_dir,
                example_id=idx,
                row=row,
                prompt_input_ids=inputs["input_ids"].detach().cpu(),
                generated_ids=gen_ids,
                tokenizer=tokenizer,
                task_type=str(cfg["TASK_TYPE"]),
                max_new_tokens=max_new_tokens,
                prompt_text=prompt_text,
                mode=mode,
            )
        )
        (tables_dir / f"input_generate_{idx}.txt").write_text(
            prompt_text + "\n" + model_output,
            encoding="utf-8",
        )
        del inputs, output, gen_ids
        release_torch_memory(collect_garbage=True)
    write_jsonl(results, response_dir / "predictions.jsonl")
    metrics = summarize_results(results) | {
        "mode": mode,
        "thinking_mode": use_thinking,
        "model_name": cfg["MODEL_NAME"],
        "tokenizer_name": cfg["TOKENIZER_NAME"],
        "max_new_tokens": max_new_tokens,
        "temperature": cfg["GENERATION_TEMPERATURE"],
        "dataset_path": str(paths.dataset_path),
    }
    _write_json(response_dir / "metrics.json", metrics)
    _write_json(tables_dir / "full_sequence_info.json", {"examples": full_infos})
    return results, metrics


def cache_mode_outputs(
    *,
    cfg: Mapping[str, Any],
    paths: CotRunPaths,
    mode: str,
) -> Path:
    cache_root = paths.cache_dir / "responses" / mode
    cache_root.mkdir(parents=True, exist_ok=True)
    for source, name in [
        (paths.dataset_path, "dynamic_niah_v2.jsonl"),
        (paths.dataset_path.parent / "config.used.json", "config.used.json"),
        (paths.config_path, "cot_config.json"),
        (paths.predictions_path(mode), "predictions.jsonl"),
        (paths.metrics_path(mode), "metrics.json"),
    ]:
        if Path(source).exists():
            shutil.copyfile(source, cache_root / name)
    mode_dir = paths.mode_dir(mode)
    for tensor_path in sorted((mode_dir / "tensors").glob("inputs_cot_*.pt")):
        dst = cache_root / "full_sequence" / "tensors" / tensor_path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tensor_path, dst)
    for table_path in sorted((mode_dir / "tables").glob("full_sequence*")):
        if table_path.is_file():
            dst = cache_root / "full_sequence" / "tables" / table_path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(table_path, dst)
    _write_json(
        cache_root / "response_cache_metadata.json",
        {
            "setting_name": paths.setting_name,
            "mode": mode,
            "model_name": cfg["MODEL_NAME"],
            "tokenizer_name": cfg["TOKENIZER_NAME"],
            "max_new_tokens": (
                cfg["MAX_NEW_TOKENS_THINKING"]
                if mode == "thinking"
                else cfg["MAX_NEW_TOKENS_NONTHINKING"]
            ),
            "temperature": cfg["GENERATION_TEMPERATURE"],
            "generation_top_p": cfg["GENERATION_TOP_P"],
            "use_kv_cache": (
                cfg["USE_KV_CACHE_FOR_THINKING"]
                if mode == "thinking"
                else cfg["USE_KV_CACHE_FOR_NONTHINKING"]
            ),
        },
    )
    return cache_root


def restore_mode_outputs_from_cache(
    *,
    cfg: Mapping[str, Any],
    paths: CotRunPaths,
    mode: str,
) -> bool:
    cache_root = paths.cache_dir / "responses" / mode
    cached_predictions = cache_root / "predictions.jsonl"
    cached_metrics = cache_root / "metrics.json"
    if not cached_predictions.exists() or not cached_metrics.exists():
        return False
    metadata_path = cache_root / "response_cache_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_max_new = (
            cfg["MAX_NEW_TOKENS_THINKING"]
            if mode == "thinking"
            else cfg["MAX_NEW_TOKENS_NONTHINKING"]
        )
        mismatches = {
            key: {"expected": expected, "actual": metadata.get(key)}
            for key, expected in {
                "setting_name": paths.setting_name,
                "mode": mode,
                "model_name": cfg["MODEL_NAME"],
                "tokenizer_name": cfg["TOKENIZER_NAME"],
                "max_new_tokens": expected_max_new,
                "temperature": cfg["GENERATION_TEMPERATURE"],
                "generation_top_p": cfg["GENERATION_TOP_P"],
            }.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            print(
                "[cot-analysis] response cache metadata mismatch for "
                f"mode={mode}: {json.dumps(mismatches, ensure_ascii=False)}",
                flush=True,
            )
            return False
    paths.response_dir(mode).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached_predictions, paths.predictions_path(mode))
    shutil.copyfile(cached_metrics, paths.metrics_path(mode))
    cached_dataset = cache_root / "dynamic_niah_v2.jsonl"
    if cached_dataset.exists() and not paths.dataset_path.exists():
        paths.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached_dataset, paths.dataset_path)
    cached_full_sequence = cache_root / "full_sequence"
    if cached_full_sequence.exists():
        mode_dir = paths.mode_dir(mode)
        for source_root_name in ["tensors", "tables"]:
            source_root = cached_full_sequence / source_root_name
            if not source_root.exists():
                continue
            target_root = mode_dir / source_root_name
            target_root.mkdir(parents=True, exist_ok=True)
            for src in sorted(source_root.iterdir()):
                if src.is_file():
                    dst = target_root / src.name
                    if not dst.exists():
                        shutil.copyfile(src, dst)
    return True


def write_selected_mode_run(
    *,
    cfg: Mapping[str, Any],
    paths: CotRunPaths,
    mode: str,
    rows: Sequence[dict[str, Any]],
    selected_ids: Sequence[int],
) -> Path:
    mode_dir = paths.mode_dir(mode)
    selected_rows = [rows[int(idx)] for idx in selected_ids]
    dataset_path = mode_dir / "generate_data" / "dynamic_niah_v2.jsonl"
    _write_jsonl(dataset_path, selected_rows)
    dyn_cfg = dynamic_config_from_cot_config(
        cfg,
        run_dir=mode_dir,
        dataset_path=dataset_path,
        mode=mode,
    )
    metadata = {
        "run_dir": str(mode_dir),
        "model": cfg["MODEL_NAME"],
        "mode": mode,
        "selected_example_ids": [int(idx) for idx in selected_ids],
        "resolved_config": asdict(dyn_cfg),
    }
    _write_json(mode_dir / "run_metadata.json", metadata)
    for subdir in ["tables", "figures", "tensors"]:
        (mode_dir / subdir).mkdir(parents=True, exist_ok=True)
    return dataset_path


def copy_selected_full_sequence_artifacts(
    *,
    source_mode_dir: str | Path,
    target_mode_dir: str | Path,
    selected_ids: Sequence[int],
) -> None:
    source = Path(source_mode_dir)
    target = Path(target_mode_dir)
    if source.resolve() == target.resolve():
        return
    for example_id in selected_ids:
        src = _full_sequence_tensor_path(source / "tensors", int(example_id))
        if not src.exists():
            raise FileNotFoundError(f"Missing full-sequence tensor for selected example: {src}")
        dst = _full_sequence_tensor_path(target / "tensors", int(example_id))
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)


def ensure_selected_full_sequence_artifacts(
    *,
    cfg: Mapping[str, Any],
    paths: CotRunPaths,
    mode: str,
    rows: Sequence[dict[str, Any]],
    selected_ids: Sequence[int],
    tokenizer: Any,
) -> list[Path]:
    """Ensure selected prompt+generation tensors exist for downstream CoT analysis.

    Response caches are allowed to restore only decoded predictions/metrics.  In
    that case, rebuild the compact full-sequence tensors by retokenizing the
    saved decoded output and the deterministic prompt for the same row.
    """

    mode_dir = paths.mode_dir(mode)
    tensors_dir = mode_dir / "tensors"
    tables_dir = mode_dir / "tables"
    predictions_path = paths.predictions_path(mode)
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Cannot prepare full-sequence artifacts without predictions: {predictions_path}"
        )
    predictions = _read_jsonl(predictions_path)
    dyn_cfg = dynamic_config_from_cot_config(
        cfg,
        run_dir=mode_dir,
        dataset_path=paths.dataset_path,
        mode=mode,
    )
    use_thinking = thinking_mode_to_bool(mode)
    max_new_tokens = (
        int(cfg["MAX_NEW_TOKENS_THINKING"])
        if use_thinking
        else int(cfg["MAX_NEW_TOKENS_NONTHINKING"])
    )
    written: list[Path] = []
    for example_id in selected_ids:
        example_idx = int(example_id)
        tensor_path = _full_sequence_tensor_path(tensors_dir, example_idx)
        if tensor_path.exists():
            written.append(tensor_path)
            continue
        if example_idx >= len(rows):
            raise IndexError(f"Selected example id {example_idx} is outside dataset rows")
        if example_idx >= len(predictions):
            raise IndexError(
                f"Selected example id {example_idx} is outside predictions rows at {predictions_path}"
            )
        row = rows[example_idx]
        prediction = predictions[example_idx]
        if "model_output_text" not in prediction:
            raise ValueError(
                f"Prediction row {example_idx} lacks model_output_text; cannot rebuild full-sequence artifact"
            )
        eval_messages = build_prediction_messages(dyn_cfg, row)
        prompt_text = apply_generation_chat_template(
            tokenizer,
            eval_messages,
            thinking_mode=use_thinking,
        )
        prompt_input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].detach().cpu()
        generated_ids = tokenizer(
            str(prediction.get("model_output_text", "")),
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0].detach().cpu()
        save_full_sequence_artifact(
            tensors_dir=tensors_dir,
            tables_dir=tables_dir,
            example_id=example_idx,
            row=row,
            prompt_input_ids=prompt_input_ids,
            generated_ids=generated_ids,
            tokenizer=tokenizer,
            task_type=str(cfg["TASK_TYPE"]),
            max_new_tokens=max_new_tokens,
            prompt_text=prompt_text,
            mode=mode,
        )
        (tables_dir / f"input_generate_{example_idx}.txt").write_text(
            prompt_text + "\n" + str(prediction.get("model_output_text", "")),
            encoding="utf-8",
        )
        written.append(tensor_path)
    return written


def run_mode_qk_outlier_analysis(
    *,
    cfg: Mapping[str, Any],
    paths: CotRunPaths,
    mode: str,
    selected_ids: Sequence[int],
) -> dict[str, Any]:
    mode_dir = paths.mode_dir(mode)
    requested_ids = [int(idx) for idx in selected_ids]
    if not requested_ids:
        summary = {
            "status": "skipped",
            "reason": "no_selected_examples",
            "mode": mode,
            "num_examples": 0,
            "example_indices": [],
        }
        print(
            "[cot-analysis] skipping Q/K outlier analysis because no examples "
            f"were selected for mode={mode}",
            flush=True,
        )
        return summary

    tensors_dir = mode_dir / "tensors"
    available_ids: list[int] = []
    missing_ids: list[int] = []
    for idx in requested_ids:
        path = tensors_dir / f"hidden_inputs_{idx}.pt"
        if path.exists():
            available_ids.append(idx)
        else:
            missing_ids.append(idx)
    if missing_ids:
        print(
            "[cot-analysis] warning: skipping selected examples without hidden-state "
            f"files for mode={mode}: {missing_ids}",
            flush=True,
        )
    if not available_ids:
        summary = {
            "status": "skipped",
            "reason": "no_hidden_state_files",
            "mode": mode,
            "num_examples": 0,
            "example_indices": [],
            "requested_example_indices": requested_ids,
            "missing_hidden_state_example_indices": missing_ids,
            "tensors_dir": str(tensors_dir),
        }
        print(
            "[cot-analysis] skipping Q/K outlier analysis because no "
            f"hidden_inputs_*.pt files exist for selected examples in {tensors_dir}",
            flush=True,
        )
        return summary

    summary = run_qk_outlier_analysis(
        run_dir=mode_dir,
        layers=cfg["LAYERS"],
        model=cfg["MODEL_NAME"],
        force_capture=True,
        massive_top_k_per_layer=cfg["K"],
        topk=cfg["K"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_dtype=str(cfg["QK_COMPUTE_DTYPE"]),
        key_block_size=int(cfg["QK_KEY_BLOCK_SIZE"]),
        query_block_size=int(cfg["QK_QUERY_BLOCK_SIZE"]),
        capture_attn_implementation=str(cfg["CAPTURE_ATTN_IMPLEMENTATION"]),
        capture_model_dtype=str(cfg["CAPTURE_MODEL_DTYPE"]),
        capture_save_dtype=str(cfg["CAPTURE_SAVE_DTYPE"]),
        figures_dir="figures/qk_outliers",
        tables_dir="tables",
        example_indices=available_ids,
    )
    if missing_ids:
        summary = dict(summary)
        summary["requested_example_indices"] = requested_ids
        summary["missing_hidden_state_example_indices"] = missing_ids
    return summary


def _load_boundary_rows(mode_dir: Path, example_id: int) -> list[dict[str, Any]]:
    path = mode_dir / "tables" / f"full_sequence_boundaries_{int(example_id)}.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_cot_payload(mode_dir: Path, example_id: int) -> dict[str, Any]:
    path = _full_sequence_tensor_path(mode_dir / "tensors", int(example_id))
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def _needle_spans_from_qk_spec(cache_dir: Path) -> list[dict[str, Any]]:
    if not cache_dir.exists():
        return []
    try:
        spec = load_cache_analysis_spec(cache_dir)
    except Exception:
        return []
    spans = spec.get("spans") or {}
    out: list[dict[str, Any]] = []
    for name, intervals in spans.items():
        if not str(name).startswith("needle"):
            continue
        for interval in intervals:
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                continue
            start = int(interval[0])
            end = int(interval[1])
            if end <= start:
                continue
            out.append(
                {
                    "needle_id": str(name).replace("needle_", ""),
                    "start": start,
                    "end": end,
                    "length": end - start,
                    "is_control": False,
                    "source": "qk_analysis_spec",
                }
            )
    return sorted(out, key=lambda span: (int(span["start"]), int(span["end"])))


def _load_cot_needle_spans(
    *,
    mode_dir: Path,
    payload: Mapping[str, Any],
    example_id: int,
) -> list[dict[str, Any]]:
    """Return prompt-level needle spans, with fallbacks for older partial runs."""

    spans = payload.get("needle_spans")
    if isinstance(spans, list) and spans:
        return [dict(span) for span in spans]

    hidden_path = mode_dir / "tensors" / f"hidden_inputs_{int(example_id)}.pt"
    if hidden_path.exists():
        try:
            hidden_payload = torch.load(hidden_path, map_location="cpu")
        except Exception:
            hidden_payload = None
        hidden_spans = (
            hidden_payload.get("needle_spans")
            if isinstance(hidden_payload, dict)
            else None
        )
        if isinstance(hidden_spans, list) and hidden_spans:
            return [dict(span) for span in hidden_spans]

    cache_dir = mode_dir / "tensors" / "qk_cache" / f"input_{int(example_id)}"
    return _needle_spans_from_qk_spec(cache_dir)


def _read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _needle_patterns_from_spans(needle_spans: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    for rank, span in enumerate(needle_spans, start=1):
        try:
            start = int(span["start"])
            end = int(span["end"])
        except Exception:
            continue
        if end <= start:
            continue
        needle_id = span.get("needle_id", rank - 1)
        patterns.append(
            {
                "pattern_name": f"needle_span_{needle_id}",
                "pattern_display": f"needle_span_{needle_id} [{start}, {end})",
                "pattern_type": "needle_span",
                "pattern_rank": rank,
                "positions": list(range(start, end)),
                "score": "",
                "token": "",
            }
        )
    return patterns


def _ranked_single_position_patterns(
    *,
    rows: Sequence[Mapping[str, Any]],
    example_id: int,
    layer: int,
    pattern_type: str,
    score_field: str,
    name_prefix: str,
    k: int,
    threshold: float,
) -> list[dict[str, Any]]:
    best_by_pos: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            if int(row.get("example_idx", row.get("example_id", -1))) != int(example_id):
                continue
            if int(row.get("layer", -1)) != int(layer):
                continue
            pos = int(float(row["position"]))
            score = float(row.get(score_field, "nan"))
        except Exception:
            continue
        previous = best_by_pos.get(pos)
        if previous is None or score > float(previous["score"]):
            best_by_pos[pos] = {
                "position": pos,
                "score": score,
                "token": row.get("token", ""),
            }
    ranked = [
        item
        for item in sorted(
            best_by_pos.values(),
            key=lambda item: (-float(item["score"]), int(item["position"])),
        )
        if float(item["score"]) >= float(threshold)
    ]
    patterns: list[dict[str, Any]] = []
    for rank, item in enumerate(ranked[: max(0, int(k))], start=1):
        token = str(item.get("token", ""))
        display = token if token else f"{name_prefix}_{rank}"
        patterns.append(
            {
                "pattern_name": f"{name_prefix}_{rank}",
                "pattern_display": display,
                "pattern_type": pattern_type,
                "pattern_rank": rank,
                "positions": [int(item["position"])],
                "score": float(item["score"]),
                "token": token,
            }
        )
    return patterns


def _prompt_span_patterns(prompt_tokens: int) -> list[dict[str, Any]]:
    prompt_tokens = int(prompt_tokens)
    patterns: list[dict[str, Any]] = []
    if prompt_tokens > 0:
        patterns.append(
            {
                "pattern_name": "prompt_span",
                "pattern_display": "prompt_span",
                "pattern_type": "prompt_span",
                "pattern_rank": 1,
                "positions": list(range(prompt_tokens)),
                "score": "",
                "token": "",
            }
        )
    if prompt_tokens > 1:
        patterns.append(
            {
                "pattern_name": "prompt_span_no_first",
                "pattern_display": "prompt_span_no_first",
                "pattern_type": "prompt_span",
                "pattern_rank": 2,
                "positions": list(range(1, prompt_tokens)),
                "score": "",
                "token": "",
            }
        )
    else:
        print(
            "[cot-attention] warning: skipping prompt_span_no_first because prompt has <= 1 token",
            flush=True,
        )
    return patterns


def _cot_attention_patterns(
    *,
    mode_dir: Path,
    payload: Mapping[str, Any],
    example_id: int,
    layer: int,
    k: int,
    needle_spans: Sequence[Mapping[str, Any]],
    outlier_ratio_threshold: float,
) -> list[dict[str, Any]]:
    sink_rows = _read_csv_rows(mode_dir / "tables" / "attention_sinks_topk.csv")
    massive_rows = _read_csv_rows(mode_dir / "tables" / "massive_tokens_all.csv")
    prompt_tokens = int(payload.get("prompt_tokens", 0))
    patterns = _prompt_span_patterns(prompt_tokens)
    patterns.extend(_needle_patterns_from_spans(needle_spans))
    patterns.extend(
        _ranked_single_position_patterns(
            rows=sink_rows,
            example_id=int(example_id),
            layer=int(layer),
            pattern_type="attention_sink",
            score_field="received_uniform_ratio",
            name_prefix="attention_sink_rank",
            k=int(k),
            threshold=float(outlier_ratio_threshold),
        )
    )
    patterns.extend(
        _ranked_single_position_patterns(
            rows=massive_rows,
            example_id=int(example_id),
            layer=int(layer),
            pattern_type="massive_activation",
            score_field="norm_ratio_to_median",
            name_prefix="massive_activation_rank",
            k=int(k),
            threshold=float(outlier_ratio_threshold),
        )
    )
    seq_len = int(payload["full_sequence_tokens"])
    valid: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pattern in patterns:
        positions = sorted({int(pos) for pos in pattern["positions"] if 0 <= int(pos) < seq_len})
        if not positions:
            continue
        name = str(pattern["pattern_name"])
        if name in seen:
            continue
        seen.add(name)
        valid.append({**pattern, "positions": positions})
    return valid


def _load_qk_attention_mask(cache_dir: Path, device: torch.device) -> torch.Tensor | None:
    path = cache_dir / "attention_mask.pt"
    if not path.exists():
        return None
    mask = torch.load(path, map_location="cpu")
    if isinstance(mask, torch.Tensor):
        return mask.reshape(-1).to(device=device, dtype=torch.bool)
    return None


def _pattern_intervals(patterns: Sequence[Mapping[str, Any]]) -> dict[str, list[tuple[int, int]]]:
    intervals: dict[str, list[tuple[int, int]]] = {}
    for pattern in patterns:
        positions = sorted(int(pos) for pos in pattern["positions"])
        merged: list[tuple[int, int]] = []
        start = prev = positions[0]
        for pos in positions[1:]:
            if pos == prev + 1:
                prev = pos
                continue
            merged.append((start, prev + 1))
            start = prev = pos
        merged.append((start, prev + 1))
        intervals[str(pattern["pattern_name"])] = merged
    return intervals


def _pattern_baseline(
    *,
    qpos: torch.Tensor,
    pattern_positions: Sequence[int],
    key_padding_mask: torch.Tensor | None,
    seq_len: int,
) -> torch.Tensor:
    qpos_cpu = qpos.detach().cpu().to(dtype=torch.long)
    if key_padding_mask is None:
        valid_key_count = qpos_cpu.to(dtype=torch.float32) + 1.0
        valid_pattern_positions = torch.tensor(pattern_positions, dtype=torch.long)
    else:
        mask_cpu = key_padding_mask.detach().cpu().to(dtype=torch.bool)
        cumulative = torch.cumsum(mask_cpu.to(dtype=torch.long), dim=0)
        valid_key_count = cumulative[qpos_cpu.clamp(min=0, max=seq_len - 1)].to(dtype=torch.float32)
        valid_pattern_positions = torch.tensor(
            [pos for pos in pattern_positions if bool(mask_cpu[int(pos)].item())],
            dtype=torch.long,
        )
    if valid_pattern_positions.numel() == 0:
        return torch.zeros_like(valid_key_count, dtype=torch.float32)
    pattern_count = (valid_pattern_positions[None, :] <= qpos_cpu[:, None]).sum(dim=1).to(dtype=torch.float32)
    return pattern_count / valid_key_count.clamp_min(1.0)


def _compute_pattern_attention_for_head(
    *,
    q: torch.Tensor,
    k_tensor: torch.Tensor,
    info: Mapping[str, Any],
    patterns: Sequence[Mapping[str, Any]],
    prompt_tokens: int,
    key_padding_mask: torch.Tensor | None,
    key_block_size: int,
    query_block_size: int,
) -> dict[str, dict[str, torch.Tensor]]:
    seq_len = int(q.shape[0])
    pattern_spans = _pattern_intervals(patterns)
    query_positions_all = torch.arange(seq_len, dtype=torch.long, device=q.device)
    by_name = {
        str(pattern["pattern_name"]): {
            "attention_mass": torch.full((seq_len,), torch.nan, dtype=torch.float32),
            "uniform_baseline": torch.full((seq_len,), torch.nan, dtype=torch.float32),
            "attention_ratio": torch.full((seq_len,), torch.nan, dtype=torch.float32),
        }
        for pattern in patterns
    }
    positions_by_name = {
        str(pattern["pattern_name"]): [int(pos) for pos in pattern["positions"]]
        for pattern in patterns
    }
    max_pos_by_name = {
        name: max(positions) for name, positions in positions_by_name.items()
    }
    for qb in range(0, seq_len, int(query_block_size)):
        qpos = query_positions_all[qb : qb + int(query_block_size)]
        stats = compute_query_block_stats(
            q_block=q[qpos],
            k=k_tensor,
            query_positions=qpos,
            spans=pattern_spans,
            key_padding_mask=key_padding_mask,
            scaling=float(info["scaling"]),
            key_block_size=int(key_block_size),
            topk=0,
            local_windows=(),
            return_rows=False,
        )
        qpos_cpu = qpos.detach().cpu()
        for name, mass in stats["span_mass"].items():
            baseline = _pattern_baseline(
                qpos=qpos_cpu,
                pattern_positions=positions_by_name[name],
                key_padding_mask=key_padding_mask,
                seq_len=seq_len,
            )
            later_mask = qpos_cpu > int(max_pos_by_name[name])
            ratio = mass.to(dtype=torch.float32) / baseline.clamp_min(1e-30)
            ratio = torch.where(
                later_mask & (baseline > 0),
                ratio,
                torch.full_like(ratio, torch.nan),
            )
            clean_mass = torch.where(
                later_mask,
                mass.to(dtype=torch.float32),
                torch.full_like(mass.to(dtype=torch.float32), torch.nan),
            )
            clean_baseline = torch.where(
                later_mask,
                baseline,
                torch.full_like(baseline, torch.nan),
            )
            by_name[name]["attention_mass"][qpos_cpu] = clean_mass
            by_name[name]["uniform_baseline"][qpos_cpu] = clean_baseline
            by_name[name]["attention_ratio"][qpos_cpu] = ratio
    return by_name


def _pattern_position_summary(pattern: Mapping[str, Any]) -> tuple[int, int, int]:
    positions = [int(pos) for pos in pattern["positions"]]
    return min(positions), max(positions) + 1, len(positions)


def _shade_prompt_needle_spans(ax: Any, needle_spans: Sequence[Mapping[str, Any]]) -> None:
    labeled_any = False
    labeled_control = False
    for span in needle_spans:
        try:
            start = int(span["start"])
            end = int(span["end"])
        except Exception:
            continue
        if end <= start:
            continue
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


def _prediction_summary_for_example(
    *,
    paths: CotRunPaths,
    mode: str,
    example_id: int,
) -> dict[str, Any]:
    path = paths.predictions_path(mode)
    if not path.exists():
        return {}
    rows = _read_jsonl(path)
    if 0 <= int(example_id) < len(rows):
        row = rows[int(example_id)]
        return {
            "gold_answer": row.get("gold_answer"),
            "prediction": row.get("prediction"),
            "exact_match": row.get("exact_match"),
            "parse_success": row.get("parse_success"),
        }
    return {}


def write_cot_attention_projection_tables(
    *,
    cfg: Mapping[str, Any],
    paths: CotRunPaths,
    mode: str,
    selected_ids: Sequence[int],
) -> list[Path]:
    """Write per-pattern attention-flow tables for prompt/generation query positions."""

    mode_dir = paths.mode_dir(mode)
    output_paths: list[Path] = []
    for example_id in selected_ids:
        payload = _load_cot_payload(mode_dir, int(example_id))
        prompt_tokens = int(payload["prompt_tokens"])
        full_tokens = int(payload["full_sequence_tokens"])
        needle_spans = _load_cot_needle_spans(
            mode_dir=mode_dir,
            payload=payload,
            example_id=int(example_id),
        )
        boundaries = _load_boundary_rows(mode_dir, int(example_id))
        cache_dir = mode_dir / "tensors" / "qk_cache" / f"input_{int(example_id)}"
        if not cache_dir.exists():
            print(
                f"[cot-attention] warning: missing Q/K cache for mode={mode} example={example_id}: {cache_dir}",
                flush=True,
            )
            continue
        device = torch.device(
            "cuda"
            if torch.cuda.is_available() and str(cfg.get("QK_COMPUTE_DTYPE", "fp32"))
            else "cpu"
        )
        compute_dtype = COMPUTE_DTYPES[str(cfg["QK_COMPUTE_DTYPE"])]
        meta = load_cache_metadata(cache_dir)
        head_count_total = int(meta["model_config"]["num_attention_heads"])
        key_padding_mask = _load_qk_attention_mask(cache_dir, device)
        for layer in cfg["LAYERS"]:
            patterns = _cot_attention_patterns(
                mode_dir=mode_dir,
                payload=payload,
                example_id=int(example_id),
                layer=int(layer),
                k=int(cfg["K"]),
                needle_spans=needle_spans,
                outlier_ratio_threshold=float(cfg["OUTLIER_RATIO_THRESHOLD"]),
            )
            if not patterns:
                print(
                    f"[cot-attention] warning: no attention patterns for mode={mode} example={example_id} layer={layer}",
                    flush=True,
                )
                continue
            aggregate: dict[str, dict[str, torch.Tensor]] = {}
            best_head: dict[str, torch.Tensor] = {}
            per_head_summary_rows: list[dict[str, Any]] = []
            for head in range(head_count_total):
                q, k_tensor, info = reconstruct_single_head_qk(
                    cache_dir=cache_dir,
                    layer=int(layer),
                    head=int(head),
                    device=device,
                    compute_dtype=compute_dtype,
                )
                per_head = _compute_pattern_attention_for_head(
                    q=q,
                    k_tensor=k_tensor,
                    info=info,
                    patterns=patterns,
                    prompt_tokens=prompt_tokens,
                    key_padding_mask=key_padding_mask,
                    key_block_size=int(cfg["QK_KEY_BLOCK_SIZE"]),
                    query_block_size=int(cfg["QK_QUERY_BLOCK_SIZE"]),
                )
                for pattern in patterns:
                    name = str(pattern["pattern_name"])
                    current = per_head[name]
                    ratio = current["attention_ratio"].to(dtype=torch.float32)
                    mass = current["attention_mass"].to(dtype=torch.float32)
                    baseline = current["uniform_baseline"].to(dtype=torch.float32)
                    if name not in aggregate:
                        aggregate[name] = {
                            "attention_ratio": ratio,
                            "attention_mass": mass,
                            "uniform_baseline": baseline,
                        }
                        best_head[name] = torch.full_like(ratio, int(head), dtype=torch.int64)
                    else:
                        old_ratio = aggregate[name]["attention_ratio"]
                        choose_current = torch.nan_to_num(
                            ratio, nan=-torch.inf, posinf=torch.inf, neginf=-torch.inf
                        ) > torch.nan_to_num(
                            old_ratio, nan=-torch.inf, posinf=torch.inf, neginf=-torch.inf
                        )
                        aggregate[name]["attention_ratio"] = torch.where(
                            choose_current, ratio, old_ratio
                        )
                        aggregate[name]["attention_mass"] = torch.where(
                            choose_current, mass, aggregate[name]["attention_mass"]
                        )
                        aggregate[name]["uniform_baseline"] = baseline
                        best_head[name] = torch.where(
                            choose_current,
                            torch.full_like(best_head[name], int(head)),
                            best_head[name],
                        )
                    if cfg.get("SAVE_PER_HEAD_ATTENTION_TABLES", True):
                        finite = torch.isfinite(ratio)
                        prompt_mask = finite & (torch.arange(ratio.numel()) < prompt_tokens)
                        gen_mask = finite & (torch.arange(ratio.numel()) >= prompt_tokens)
                        start, end, length = _pattern_position_summary(pattern)
                        per_head_summary_rows.append(
                            {
                                "mode": mode,
                                "example_id": int(example_id),
                                "layer": int(layer),
                                "head": int(head),
                                "pattern_name": name,
                                "pattern_display": pattern.get("pattern_display", name),
                                "pattern_type": pattern["pattern_type"],
                                "pattern_rank": pattern["pattern_rank"],
                                "pattern_start": start,
                                "pattern_end": end,
                                "pattern_length": length,
                                "prompt_mean_attention_ratio": (
                                    float(ratio[prompt_mask].mean().item())
                                    if bool(prompt_mask.any().item())
                                    else ""
                                ),
                                "prompt_max_attention_ratio": (
                                    float(ratio[prompt_mask].max().item())
                                    if bool(prompt_mask.any().item())
                                    else ""
                                ),
                                "generation_mean_attention_ratio": (
                                    float(ratio[gen_mask].mean().item())
                                    if bool(gen_mask.any().item())
                                    else ""
                                ),
                                "generation_max_attention_ratio": (
                                    float(ratio[gen_mask].max().item())
                                    if bool(gen_mask.any().item())
                                    else ""
                                ),
                            }
                        )
                del q, k_tensor
                release_torch_memory(collect_garbage=False)
            if cfg.get("SAVE_PER_HEAD_ATTENTION_TABLES", True) and per_head_summary_rows:
                summary_path = (
                    mode_dir
                    / "tables"
                    / "cot_attention"
                    / "per_head_pattern_summary"
                    / f"example_{int(example_id)}_layer_{int(layer):02d}.csv"
                )
                output_paths.append(_write_csv(summary_path, per_head_summary_rows))
            boundary_by_pos = {
                int(float(row["full_position"])): row.get("boundary_type", "")
                for row in boundaries
                if row.get("full_position") not in {"", None}
            }
            rows: list[dict[str, Any]] = []
            for pattern in patterns:
                name = str(pattern["pattern_name"])
                start, end, length = _pattern_position_summary(pattern)
                ratio = aggregate[name]["attention_ratio"]
                mass = aggregate[name]["attention_mass"]
                baseline = aggregate[name]["uniform_baseline"]
                heads = best_head[name]
                for pos in range(min(full_tokens, int(ratio.numel()))):
                    if not torch.isfinite(ratio[pos]):
                        continue
                    rows.append(
                        {
                            "mode": mode,
                            "example_id": int(example_id),
                            "layer": int(layer),
                            "pattern_name": name,
                            "pattern_display": pattern.get("pattern_display", name),
                            "pattern_type": pattern["pattern_type"],
                            "pattern_rank": pattern["pattern_rank"],
                            "pattern_start": start,
                            "pattern_end": end,
                            "pattern_length": length,
                            "pattern_score": pattern.get("score", ""),
                            "pattern_token": pattern.get("token", ""),
                            "position": pos,
                            "segment": "prompt" if pos < prompt_tokens else "generation",
                            "relative_generation_position": (
                                "" if pos < prompt_tokens else pos - prompt_tokens
                            ),
                            "max_attention_mass": float(mass[pos].item()),
                            "uniform_baseline": float(baseline[pos].item()),
                            "max_attention_ratio": float(ratio[pos].item()),
                            "best_head": int(heads[pos].item()),
                            "head_count": head_count_total,
                            "boundary_type": boundary_by_pos.get(pos, ""),
                        }
                    )
            out = (
                mode_dir
                / "tables"
                / "cot_attention"
                / "patterns"
                / f"example_{int(example_id)}_layer_{int(layer):02d}.csv"
            )
            output_paths.append(_write_csv(out, rows))
            release_torch_memory(collect_garbage=True)
    return output_paths


def plot_cot_attention_tables(
    *,
    cfg: Mapping[str, Any],
    paths: CotRunPaths,
    mode: str,
    selected_ids: Sequence[int],
) -> list[Path]:
    import matplotlib.pyplot as plt

    mode_dir = paths.mode_dir(mode)
    figure_paths: list[Path] = []
    for example_id in selected_ids:
        payload = _load_cot_payload(mode_dir, int(example_id))
        prompt_tokens = int(payload["prompt_tokens"])
        needle_spans = _load_cot_needle_spans(
            mode_dir=mode_dir,
            payload=payload,
            example_id=int(example_id),
        )
        prediction_summary = _prediction_summary_for_example(
            paths=paths,
            mode=mode,
            example_id=int(example_id),
        )
        boundaries = _load_boundary_rows(mode_dir, int(example_id))
        for layer in cfg["LAYERS"]:
            table_path = (
                mode_dir
                / "tables"
                / "cot_attention"
                / "patterns"
                / f"example_{int(example_id)}_layer_{int(layer):02d}.csv"
            )
            if not table_path.exists():
                continue
            with table_path.open(encoding="utf-8", newline="") as handle:
                rows = [dict(row) for row in csv.DictReader(handle)]
            rows_by_pattern: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                rows_by_pattern.setdefault(str(row["pattern_name"]), []).append(row)
            for pattern_name, pattern_rows in sorted(rows_by_pattern.items()):
                prompt_rows = [r for r in pattern_rows if r["segment"] == "prompt"]
                gen_rows = [r for r in pattern_rows if r["segment"] == "generation"]
                if not prompt_rows and not gen_rows:
                    continue
                meta = pattern_rows[0]
                pattern_start = int(float(meta["pattern_start"]))
                pattern_end = int(float(meta["pattern_end"]))
                pattern_display = str(meta.get("pattern_display") or pattern_name)
                pattern_type = str(meta.get("pattern_type") or "")
                fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
                _shade_prompt_needle_spans(axes[0], needle_spans)
                if pattern_start < prompt_tokens:
                    axes[0].axvspan(
                        pattern_start,
                        min(pattern_end, prompt_tokens),
                        color="tab:green",
                        alpha=0.14 if pattern_type == "prompt_span" else 0.22,
                        zorder=1,
                    )
                elif pattern_start >= prompt_tokens:
                    axes[1].axvspan(
                        pattern_start - prompt_tokens,
                        pattern_end - prompt_tokens,
                        color="tab:green",
                        alpha=0.22,
                        zorder=1,
                    )
                if prompt_rows:
                    axes[0].plot(
                        [int(float(r["position"])) for r in prompt_rows],
                        [float(r["max_attention_ratio"]) for r in prompt_rows],
                        linewidth=1.0,
                    )
                if gen_rows:
                    axes[1].plot(
                        [int(float(r["relative_generation_position"])) for r in gen_rows],
                        [float(r["max_attention_ratio"]) for r in gen_rows],
                        linewidth=1.0,
                    )
                for ax in axes:
                    ax.set_yscale("symlog", linthresh=1.0)
                    ax.set_ylabel("attention to pattern / uniform")
                axes[0].set_title("Prompt query positions")
                axes[0].set_xlabel("full token position")
                axes[1].set_title("Generated query positions")
                axes[1].set_xlabel("generation token position")
                handles, labels = axes[0].get_legend_handles_labels()
                if handles:
                    axes[0].legend(loc="upper right", fontsize=7)
                for row in boundaries:
                    try:
                        full_pos = int(float(row["full_position"]))
                    except Exception:
                        continue
                    label = str(row.get("boundary_type", ""))
                    if full_pos < prompt_tokens:
                        axes[0].axvline(full_pos, linestyle="--", color="tab:red", alpha=0.7)
                    else:
                        rel = full_pos - prompt_tokens
                        axes[1].axvline(rel, linestyle="--", color="tab:red", alpha=0.7)
                        axes[1].text(
                            rel,
                            0.95,
                            label,
                            rotation=90,
                            transform=axes[1].get_xaxis_transform(),
                            fontsize=7,
                            va="top",
                        )
                correctness = prediction_summary.get("exact_match", "")
                gold = prediction_summary.get("gold_answer", "")
                pred = prediction_summary.get("prediction", "")
                fig.suptitle(
                    f"{mode} example={example_id} layer={layer} pattern={pattern_display} "
                    f"({pattern_name}) "
                    f"correct={correctness} gold={gold} pred={pred}",
                    fontsize=10,
                )
                fig.tight_layout()
                out = (
                    mode_dir
                    / "figures"
                    / "cot_attention"
                    / f"layer_{int(layer):02d}"
                    / f"example_{int(example_id)}"
                    / f"{slugify(pattern_name, max_length=80)}.png"
                )
                out.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out)
                plt.close(fig)
                figure_paths.append(out)
    return figure_paths


def cleanup_cot_mode_artifacts(
    run_dir: str | Path,
    *,
    keep_hidden_states: bool = False,
    keep_full_sequence_tensors: bool = False,
) -> dict[str, list[Path]]:
    removed = cleanup_counting_archive_artifacts(run_dir, delete_large_pt=True)
    root = Path(run_dir)
    removed.setdefault("hidden_inputs", [])
    removed.setdefault("full_sequence_inputs", [])
    removed.setdefault("massive_activation_tensors", [])
    if not keep_hidden_states:
        for path in sorted(root.rglob("hidden_inputs_*.pt")):
            if path.is_file():
                path.unlink()
                removed["hidden_inputs"].append(path)
        for path in sorted(root.rglob("tensors/massive_activations")):
            if path.is_dir():
                shutil.rmtree(path)
                removed["massive_activation_tensors"].append(path)
    if not keep_full_sequence_tensors:
        for path in sorted(root.rglob("inputs_cot_*.pt")):
            if path.is_file():
                path.unlink()
                removed["full_sequence_inputs"].append(path)
    return removed


def archive_cot_results(paths: CotRunPaths, *, results_path: str | Path | None) -> Path:
    destination = Path(results_path) if results_path is not None else paths.archives_dir
    return archive_directory(paths.run_dir, destination, archive_name=paths.run_name)


def write_cot_response_checkpoint_archive(
    *,
    paths: CotRunPaths,
    modes: Sequence[str],
) -> Path:
    required = [paths.dataset_path, paths.config_path]
    for mode in modes:
        required.extend([paths.predictions_path(mode), paths.metrics_path(mode)])
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot write CoT response checkpoint; missing file(s): "
            + "; ".join(missing)
        )
    paths.response_checkpoint_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        paths.response_checkpoint_zip, "w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        for file_path in required:
            path = Path(file_path)
            try:
                arcname = path.relative_to(paths.run_dir)
            except ValueError:
                arcname = path.name
            zf.write(path, arcname.as_posix())
    return paths.response_checkpoint_zip
