#!/usr/bin/env python3
"""Prepare audited raw-arm and attention evidence for the HTML report.

This script does not run a model.  It derives two compact, reviewable report
artifacts from immutable Filestream captures:

1. raw ranked and layer-matched-random top-K ablation curves; and
2. one deterministically selected Qwen answer-query attention example; and
3. a fixed-grid Qwen attention gallery spanning four frozen heads and three
   confirmation prompts, with both whole-document bins and token-level rows.

The attention example is display-only.  Selection freezes the rank-1 V4.4
span-sum head from the discovery atlas, then maximizes that head's registered
``broad_primary`` metric over the 100 confirmation seed-count prompts.  The
selection therefore cannot be used as a statistical test.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realistic_niah.prompts import COMMON_COUNTING_CUE  # noqa: E402
from realistic_niah_v4.prompts import V4_NUMERIC_QUERY_BLOCK  # noqa: E402


MODELS = ("Qwen3-8B", "Gemma4-E4B")
TOP_NS = (1, 2, 4, 8, 16, 32)
EXPECTED_SEEDS = tuple(range(1316, 1336))
EXPECTED_COUNTS = tuple(range(1, 6))
BOOTSTRAP_REPETITIONS = 10_000
TOKENIZER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
RAW_FLOAT16_MASS_TOLERANCE = 5e-6
ATTENTION_MODEL = "Qwen3-8B"
ATTENTION_VARIANT = "v4.4"
ATTENTION_SPLIT = "confirmation"
ATTENTION_CANDIDATE_SEEDS = tuple(range(1254, 1264))
ATTENTION_CANDIDATE_COUNTS = tuple(range(1, 11))
ATTENTION_GALLERY_HEADS = ((27, 18), (28, 19), (23, 29), (23, 13))
ATTENTION_GALLERY_PROMPTS = ((1254, 3), (1254, 6), (1254, 9))
ATTENTION_GALLERY_BIN_WIDTH = 64


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Not a registered boolean value: {value!r}")


def stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "little")


def bootstrap_ci(values: Iterable[float], *, label: str) -> tuple[float, float]:
    vector = np.asarray(list(values), dtype=np.float64)
    vector = vector[np.isfinite(vector)]
    if len(vector) != len(EXPECTED_SEEDS):
        raise RuntimeError(f"Expected 20 finite seed values for {label}; got {len(vector)}")
    rng = np.random.default_rng(stable_seed(label))
    indices = rng.integers(
        0,
        len(vector),
        size=(BOOTSTRAP_REPETITIONS, len(vector)),
    )
    distribution = vector[indices].mean(axis=1)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return float(low), float(high)


def detail_path(stage1_root: Path, model: str) -> Path:
    name = "qwen_topk_detail.csv.gz" if model == "Qwen3-8B" else "gemma_topk_detail.csv.gz"
    path = stage1_root / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def eligible_head_counts(atlas_rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for model in MODELS:
        heads = {
            (int(row["layer"]), int(row["head"]))
            for row in atlas_rows
            if row["model"] == model
            and row["variant"] == "v4.4"
            and row["pooling"] == "span_sum"
        }
        if not heads:
            raise RuntimeError(f"No V4.4 span-sum atlas heads for {model}")
        counts[model] = len(heads)
    return counts


def raw_topk_statistics(
    stage1_root: Path,
    *,
    atlas_rows: list[dict[str, str]],
    formal_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    denominators = eligible_head_counts(atlas_rows)
    output: list[dict[str, object]] = []
    source_hashes: dict[str, str] = {}
    maximum_contrast_difference = 0.0

    for model in MODELS:
        path = detail_path(stage1_root, model)
        source_hashes[model] = sha256_file(path)
        rows = read_csv_gz(path)
        if len(rows) != 2400:
            raise RuntimeError(f"{model} top-K detail must contain 2400 rows")
        seeds = tuple(sorted({int(row["seed"]) for row in rows}))
        counts = tuple(sorted({int(row["gold_count"]) for row in rows}))
        top_ns = tuple(sorted({int(row["top_n"]) for row in rows}))
        if seeds != EXPECTED_SEEDS or counts != EXPECTED_COUNTS or top_ns != TOP_NS:
            raise RuntimeError(f"Unexpected frozen top-K coverage for {model}")

        for top_n in TOP_NS:
            dose = [row for row in rows if int(row["top_n"]) == top_n]
            seed_arm_values: dict[str, dict[str, list[float]]] = {
                "absolute_count_shift": defaultdict(list),
                "clean_correct_to_wrong_rate": defaultdict(list),
            }
            arm_row_counts: dict[str, int] = {}
            clean_row_counts: dict[str, int] = {}

            for seed in EXPECTED_SEEDS:
                seed_rows = [row for row in dose if int(row["seed"]) == seed]
                ranked = [row for row in seed_rows if row["condition"] == "ranked"]
                random_rows = [
                    row for row in seed_rows if row["condition"] == "layer_matched_random"
                ]
                if len(ranked) != 5 or len(random_rows) != 15:
                    raise RuntimeError(
                        f"Unexpected {model} K={top_n} seed={seed} arm coverage"
                    )
                arms = {"ranked": ranked, "layer_matched_random": random_rows}
                for condition, arm in arms.items():
                    seed_arm_values["absolute_count_shift"][condition].append(
                        float(np.mean([abs(float(row["generated_count_shift"])) for row in arm]))
                    )
                    clean = [
                        row
                        for row in arm
                        if as_bool(row["baseline_is_correct"])
                        and as_bool(row["baseline_format_valid"])
                    ]
                    if not clean:
                        raise RuntimeError(
                            f"No clean-correct rows for {model} K={top_n} seed={seed}"
                        )
                    seed_arm_values["clean_correct_to_wrong_rate"][condition].append(
                        float(np.mean([not as_bool(row["patched_is_correct"]) for row in clean]))
                    )
                    arm_row_counts[condition] = arm_row_counts.get(condition, 0) + len(arm)
                    clean_row_counts[condition] = clean_row_counts.get(condition, 0) + len(clean)

                ranked_ids = {
                    row["stimulus_id"]
                    for row in ranked
                    if as_bool(row["baseline_is_correct"])
                    and as_bool(row["baseline_format_valid"])
                }
                random_ids = {
                    row["stimulus_id"]
                    for row in random_rows
                    if as_bool(row["baseline_is_correct"])
                    and as_bool(row["baseline_format_valid"])
                }
                if ranked_ids != random_ids:
                    raise RuntimeError("Ranked/random clean-correct populations are misaligned")

            metric_to_population = {
                "absolute_count_shift": "all_examples_signed",
                "clean_correct_to_wrong_rate": "clean_correct_only",
            }
            for metric, arm_values in seed_arm_values.items():
                for condition in ("ranked", "layer_matched_random"):
                    values = arm_values[condition]
                    label = f"{model}|K{top_n}|{metric}|{condition}|raw-arm"
                    low, high = bootstrap_ci(values, label=label)
                    output.append(
                        {
                            "model_label": model,
                            "top_n": top_n,
                            "eligible_head_count": denominators[model],
                            "head_proportion": top_n / denominators[model],
                            "metric": metric,
                            "condition": condition,
                            "seed_clusters": len(values),
                            "rows": (
                                arm_row_counts[condition]
                                if metric == "absolute_count_shift"
                                else clean_row_counts[condition]
                            ),
                            "mean": float(np.mean(values)),
                            "ci95_low": low,
                            "ci95_high": high,
                            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                        }
                    )

                raw_difference = float(
                    np.mean(arm_values["ranked"])
                    - np.mean(arm_values["layer_matched_random"])
                )
                formal = [
                    row
                    for row in formal_rows
                    if row["model_label"] == model
                    and int(row["top_n"]) == top_n
                    and row["analysis_population"] == metric_to_population[metric]
                ]
                if len(formal) != 1:
                    raise RuntimeError(
                        f"Missing formal contrast for {model} K={top_n} {metric}"
                    )
                difference = abs(raw_difference - float(formal[0]["primary_effect"]))
                maximum_contrast_difference = max(maximum_contrast_difference, difference)
                if difference > 1e-12:
                    raise RuntimeError(
                        f"Raw arms do not reproduce formal contrast for {model} "
                        f"K={top_n} {metric}: difference={difference}"
                    )

    return output, {
        "source_detail_sha256": source_hashes,
        "models": list(MODELS),
        "top_ns": list(TOP_NS),
        "seeds": list(EXPECTED_SEEDS),
        "counts": list(EXPECTED_COUNTS),
        "maximum_raw_minus_formal_contrast_difference": maximum_contrast_difference,
        "status": "PASS",
    }


def span_for_char_interval(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int, int]:
    indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and start < int(char_end) and end > int(char_start)
    ]
    if not indices:
        raise RuntimeError(f"No tokens overlap char interval [{char_start}, {char_end})")
    return indices[0], indices[-1] + 1


def visible_surface(text: str, start: int, end: int, raw_token: str) -> str:
    surface = text[start:end] if end > start else raw_token
    if not surface:
        surface = raw_token
    return (
        surface.replace(" ", "␠")
        .replace("\n", "↵")
        .replace("\t", "⇥")
        .replace("\r", "")
    )


def render_qwen_prompt(stimulus: dict[str, Any]) -> str:
    passage = str(stimulus["passage"])
    content = (
        COMMON_COUNTING_CUE
        + "\n\n<passage>\n"
        + passage
        + "\n</passage>\n\n"
        + V4_NUMERIC_QUERY_BLOCK
    )
    # Exact frozen Qwen chat template with one user message, no tools, no
    # system prompt, add_generation_prompt=True, enable_thinking=False.
    return (
        "<|im_start|>user\n"
        + content
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nTotal:"
    )


def attention_example(
    stage1_root: Path,
    stage2_root: Path,
    *,
    atlas_rows: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, object]]:
    ranked = [
        row
        for row in atlas_rows
        if row["model"] == ATTENTION_MODEL
        and row["variant"] == ATTENTION_VARIANT
        and row["pooling"] == "span_sum"
        and row["candidate_rank"]
        and int(float(row["candidate_rank"])) == 1
    ]
    if len(ranked) != 1:
        raise RuntimeError("Expected one rank-1 Qwen V4.4 span-sum head")
    layer = int(ranked[0]["layer"])
    head = int(ranked[0]["head"])

    metrics_dir = (
        stage1_root
        / "runs"
        / "run_20260731_v4_numeric_presentation_v3"
        / ATTENTION_MODEL
        / "numeric"
        / "attention"
        / "capture"
        / "shards"
        / ATTENTION_VARIANT
    )
    metric_files = sorted(metrics_dir.glob("*.csv.gz"))
    if len(metric_files) != 100:
        raise RuntimeError(f"Expected 100 confirmation metric shards; got {len(metric_files)}")
    candidates: list[tuple[float, str, dict[str, str], Path]] = []
    for path in metric_files:
        rows = read_csv_gz(path)
        selected = [
            row for row in rows if int(row["layer"]) == layer and int(row["head"]) == head
        ]
        if len(selected) != 1:
            raise RuntimeError(f"Expected one L{layer}H{head} row in {path.name}")
        row = selected[0]
        if row["split"] != ATTENTION_SPLIT or row["design_variant"] != ATTENTION_VARIANT:
            raise RuntimeError("Attention display candidate left the frozen panel")
        candidates.append((float(row["broad_primary"]), row["stimulus_id"], row, path))
    coverage = {
        (int(row[2]["seed"]), int(row[2]["count"])) for row in candidates
    }
    expected_coverage = {
        (seed, count)
        for seed in ATTENTION_CANDIDATE_SEEDS
        for count in ATTENTION_CANDIDATE_COUNTS
    }
    if coverage != expected_coverage:
        raise RuntimeError("Attention display candidates do not cover 10×10 confirmation cells")
    _, stimulus_id, metric_row, metric_path = min(
        candidates,
        key=lambda item: (-item[0], item[1]),
    )

    stimulus_path = stage2_root / f"{stimulus_id}.stimulus.v2.jsonl"
    index_path = stage2_root / f"{stimulus_id}.index.v2.jsonl"
    tokenizer_path = stage2_root / "qwen_tokenizer_b968826" / "tokenizer.json"
    raw_candidates = list(stage2_root.rglob(f"{stimulus_id}.npz"))
    selected_metric_candidates = list(stage2_root.rglob(f"{stimulus_id}.csv.gz"))
    if not stimulus_path.is_file() or not index_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError("Selected attention evidence is incomplete")
    if len(raw_candidates) != 1 or len(selected_metric_candidates) != 1:
        raise RuntimeError("Expected one selected raw NPZ and one selected metric shard")
    raw_path = raw_candidates[0]
    selected_metric_path = selected_metric_candidates[0]
    if sha256_file(selected_metric_path) != sha256_file(metric_path):
        raise RuntimeError("Stage-1 and stage-2 selected metric shards differ")

    stimulus = json.loads(stimulus_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if stimulus["stimulus_id"] != stimulus_id or index["stimulus_id"] != stimulus_id:
        raise RuntimeError("Selected stimulus/index identity mismatch")
    if str(stimulus["canonical_tokenizer_revision"]) != TOKENIZER_REVISION:
        raise RuntimeError("Selected stimulus tokenizer revision changed")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model_text = render_qwen_prompt(stimulus)
    encoding = tokenizer.encode(model_text, add_special_tokens=False)
    ids = [int(value) for value in encoding.ids]
    offsets = [(int(start), int(end)) for start, end in encoding.offsets]
    raw_tokens = list(encoding.tokens)
    if len(ids) != len(offsets) or len(ids) != len(raw_tokens):
        raise RuntimeError("Tokenizer IDs/offsets/tokens length mismatch")

    raw = np.load(raw_path, allow_pickle=False)
    sequence_length = int(raw["sequence_length"][0])
    query_position = int(raw["query_position"][0])
    key_start = int(raw["key_starts"][layer])
    attention = np.asarray(raw[f"layer_{layer:03d}"][head], dtype=np.float64)
    if len(ids) != sequence_length or query_position != sequence_length - 1:
        raise RuntimeError(
            f"Retokenized prompt mismatch: ids={len(ids)} raw={sequence_length} "
            f"query={query_position}"
        )
    if key_start != 0 or len(attention) != sequence_length:
        raise RuntimeError("Selected Qwen attention row must cover the full prompt")
    if not np.isfinite(attention).all():
        raise RuntimeError("Selected attention row contains non-finite values")

    passage = str(stimulus["passage"])
    passage_char_start = model_text.find(passage)
    if passage_char_start < 0 or model_text.find(passage, passage_char_start + 1) >= 0:
        raise RuntimeError("Rendered prompt must contain the passage exactly once")
    passage_char_end = passage_char_start + len(passage)
    passage_token_start, passage_token_end = span_for_char_interval(
        offsets, passage_char_start, passage_char_end
    )

    active_spans: list[dict[str, Any]] = []
    active_positions: dict[int, int] = {}
    for item in stimulus["active_needle_spans"]:
        start, end = span_for_char_interval(
            offsets,
            passage_char_start + int(item["char_start"]),
            passage_char_start + int(item["char_end"]),
        )
        row = dict(item)
        row["token_start"] = start
        row["token_end"] = end
        active_spans.append(row)
        for position in range(start, end):
            if position in active_positions:
                raise RuntimeError("Active needle token spans overlap")
            active_positions[position] = int(item["slot_index"])

    negative_spans: list[dict[str, Any]] = []
    negative_positions: dict[int, int] = {}
    for item in stimulus["hard_negative_spans"]:
        start, end = span_for_char_interval(
            offsets,
            passage_char_start + int(item["char_start"]),
            passage_char_start + int(item["char_end"]),
        )
        row = dict(item)
        row["token_start"] = start
        row["token_end"] = end
        negative_spans.append(row)
        for position in range(start, end):
            if position in active_positions or position in negative_positions:
                raise RuntimeError("Registered active/negative token spans overlap")
            negative_positions[position] = int(item["slot_index"])

    stored_needle_masses = np.asarray(
        json.loads(metric_row["needle_span_masses"]), dtype=np.float64
    )
    derived_needle_masses = np.asarray(
        [float(attention[item["token_start"] : item["token_end"]].sum()) for item in active_spans],
        dtype=np.float64,
    )
    if len(stored_needle_masses) != len(derived_needle_masses):
        raise RuntimeError("Stored/derived needle mass length mismatch")
    mass_max_abs_difference = float(
        np.max(np.abs(stored_needle_masses - derived_needle_masses))
    )
    if mass_max_abs_difference > RAW_FLOAT16_MASS_TOLERANCE:
        raise RuntimeError(
            f"Derived needle masses do not reproduce metric shard: {mass_max_abs_difference}"
        )

    def region(position: int) -> tuple[str, int | None]:
        if position in active_positions:
            return "active_needle", active_positions[position]
        if position in negative_positions:
            return "hard_negative", negative_positions[position]
        if passage_token_start <= position < passage_token_end:
            return "ordinary_passage", None
        if position == query_position:
            return "answer_query", None
        return "prompt_wrapper_or_instruction", None

    def token_record(position: int) -> dict[str, object]:
        start, end = offsets[position]
        token_region, slot_index = region(position)
        return {
            "position": position,
            "token_id": ids[position],
            "display": visible_surface(model_text, start, end, raw_tokens[position]),
            "raw_token": raw_tokens[position],
            "char_start": start,
            "char_end": end,
            "attention": float(attention[position]),
            "region": token_region,
            "slot_index": slot_index,
        }

    needle_rows: list[dict[str, object]] = []
    for item, span_mass in zip(active_spans, derived_needle_masses):
        positions = range(int(item["token_start"]), int(item["token_end"]))
        needle_rows.append(
            {
                "slot_index": int(item["slot_index"]),
                "needle_id": str(item["needle_id"]),
                "city": str(item["city"]),
                "score": int(item["score"]),
                "token_start": int(item["token_start"]),
                "token_end": int(item["token_end"]),
                "attention_mass": float(span_mass),
                "tokens": [token_record(position) for position in positions],
            }
        )

    ordered_positions = sorted(
        range(sequence_length), key=lambda position: (-attention[position], position)
    )
    top_tokens = [token_record(position) for position in ordered_positions[:36]]
    top_non_needle_tokens = [
        token_record(position)
        for position in ordered_positions
        if position not in active_positions
    ][:18]

    category_mass = defaultdict(float)
    for position, value in enumerate(attention):
        category_mass[region(position)[0]] += float(value)
    attention_sum = float(attention.sum())
    stored_broad_mass = float(metric_row["broad_mass"])
    if (
        abs(category_mass["active_needle"] - stored_broad_mass)
        > RAW_FLOAT16_MASS_TOLERANCE * len(active_spans)
    ):
        raise RuntimeError("Full token categorization does not reproduce broad mass")

    payload: dict[str, Any] = {
        "schema_version": "niah_nonthinking_attention_example_v1",
        "selection": {
            "purpose": "display_only",
            "model_label": ATTENTION_MODEL,
            "design_variant": ATTENTION_VARIANT,
            "split": ATTENTION_SPLIT,
            "frozen_head_rank": 1,
            "layer": layer,
            "head": head,
            "candidate_prompts": len(candidates),
            "candidate_seeds": list(ATTENTION_CANDIDATE_SEEDS),
            "candidate_counts": list(ATTENTION_CANDIDATE_COUNTS),
            "criterion": "maximum registered broad_primary for the frozen rank-1 head",
            "stimulus_id": stimulus_id,
            "seed": int(metric_row["seed"]),
            "gold_count": int(metric_row["count"]),
            "broad_primary": float(metric_row["broad_primary"]),
            "broad_mass": stored_broad_mass,
            "broad_coverage": float(metric_row["broad_coverage"]),
        },
        "prompt": {
            "sequence_length": sequence_length,
            "query_position": query_position,
            "passage_token_start": passage_token_start,
            "passage_token_end": passage_token_end,
            "active_needle_count": len(active_spans),
            "hard_negative_count": len(negative_spans),
        },
        "attention": {
            "sum": attention_sum,
            "category_mass": dict(category_mass),
            "max_token_attention": float(attention.max()),
            "needle_rows": needle_rows,
            "top_tokens": top_tokens,
            "top_non_needle_tokens": top_non_needle_tokens,
        },
        "interpretation_boundary": (
            "The figure visualizes one natural answer-query attention row. "
            "Attention weights are not OV/logit attribution and do not by themselves "
            "establish causal necessity or a unique route."
        ),
    }
    audit: dict[str, object] = {
        "status": "PASS",
        "selection_rule": payload["selection"],
        "source_sha256": {
            "raw_npz": sha256_file(raw_path),
            "metric_shard": sha256_file(metric_path),
            "stimulus_jsonl": sha256_file(stimulus_path),
            "attention_index_row": sha256_file(index_path),
            "tokenizer_json": sha256_file(tokenizer_path),
        },
        "tokenizer_revision": TOKENIZER_REVISION,
        "retokenized_sequence_length": len(ids),
        "raw_sequence_length": sequence_length,
        "retokenized_query_position": len(ids) - 1,
        "raw_query_position": query_position,
        "attention_row_sum": attention_sum,
        "stored_broad_mass": stored_broad_mass,
        "derived_broad_mass": category_mass["active_needle"],
        "needle_mass_max_abs_difference": mass_max_abs_difference,
        "raw_float16_mass_tolerance": RAW_FLOAT16_MASS_TOLERANCE,
    }
    return payload, audit


def read_selected_jsonl(
    path: Path,
    stimulus_ids: set[str],
) -> dict[str, dict[str, Any]]:
    """Read only the requested stimulus rows from a large JSONL artifact."""

    selected: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            stimulus_id = str(row.get("stimulus_id", ""))
            if stimulus_id in stimulus_ids:
                if stimulus_id in selected:
                    raise RuntimeError(f"Duplicate JSONL row for {stimulus_id} in {path}")
                selected[stimulus_id] = row
    if set(selected) != stimulus_ids:
        missing = sorted(stimulus_ids - set(selected))
        raise RuntimeError(f"Missing selected JSONL rows in {path}: {missing}")
    return selected


def attention_gallery(
    stage2_root: Path,
    stage3_root: Path,
    *,
    atlas_rows: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, object]]:
    """Build a non-posthoc multi-head, multi-prompt attention gallery.

    Heads are the discovery-frozen top four of the formal Qwen V4.4 span-sum
    ranking.  Prompts are fixed without reading their attention outcomes: the
    smallest confirmation seed and low/middle/high counts 3, 6, and 9.
    """

    frozen_rank = {
        (int(row["layer"]), int(row["head"])): int(float(row["candidate_rank"]))
        for row in atlas_rows
        if row["model"] == ATTENTION_MODEL
        and row["variant"] == ATTENTION_VARIANT
        and row["pooling"] == "span_sum"
        and row["candidate_rank"]
    }
    expected_ranks = {
        head: rank for rank, head in enumerate(ATTENTION_GALLERY_HEADS, start=1)
    }
    observed_ranks = {head: frozen_rank.get(head) for head in ATTENTION_GALLERY_HEADS}
    if observed_ranks != expected_ranks:
        raise RuntimeError(
            "Attention gallery heads are not the discovery-frozen Qwen top four: "
            f"{observed_ranks}"
        )

    stimulus_ids = {
        f"V4_4_T10000_N{count}_seed{seed}"
        for seed, count in ATTENTION_GALLERY_PROMPTS
    }
    stimuli_path = stage3_root / "stimuli.jsonl"
    index_path = stage3_root / "attention_capture_index.jsonl"
    tokenizer_path = stage2_root / "qwen_tokenizer_b968826" / "tokenizer.json"
    if not stimuli_path.is_file() or not index_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError("Attention gallery source artifacts are incomplete")
    stimuli = read_selected_jsonl(stimuli_path, stimulus_ids)
    indices = read_selected_jsonl(index_path, stimulus_ids)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    records: list[dict[str, Any]] = []
    prompt_audits: list[dict[str, Any]] = []
    global_bin_max = 0.0
    global_token_max = 0.0

    for seed, count in ATTENTION_GALLERY_PROMPTS:
        stimulus_id = f"V4_4_T10000_N{count}_seed{seed}"
        stimulus = stimuli[stimulus_id]
        index = indices[stimulus_id]
        raw_path = stage3_root / f"{stimulus_id}.npz"
        metric_path = stage3_root / f"{stimulus_id}.csv.gz"
        if not raw_path.is_file() or not metric_path.is_file():
            raise FileNotFoundError(f"Missing raw/metric attention source for {stimulus_id}")
        if stimulus["stimulus_id"] != stimulus_id or index["stimulus_id"] != stimulus_id:
            raise RuntimeError(f"Stimulus/index identity mismatch for {stimulus_id}")
        if str(stimulus["canonical_tokenizer_revision"]) != TOKENIZER_REVISION:
            raise RuntimeError(f"Tokenizer revision mismatch for {stimulus_id}")
        if int(index["seed"]) != seed or int(index["count"]) != count:
            raise RuntimeError(f"Index seed/count mismatch for {stimulus_id}")
        if index["split"] != ATTENTION_SPLIT or index["design_variant"] != ATTENTION_VARIANT:
            raise RuntimeError(f"Index row left frozen panel for {stimulus_id}")

        model_text = render_qwen_prompt(stimulus)
        encoding = tokenizer.encode(model_text, add_special_tokens=False)
        ids = [int(value) for value in encoding.ids]
        offsets = [(int(start), int(end)) for start, end in encoding.offsets]
        raw_tokens = list(encoding.tokens)
        if len(ids) != len(offsets) or len(ids) != len(raw_tokens):
            raise RuntimeError(f"Tokenizer output mismatch for {stimulus_id}")

        passage = str(stimulus["passage"])
        passage_char_start = model_text.find(passage)
        if passage_char_start < 0 or model_text.find(passage, passage_char_start + 1) >= 0:
            raise RuntimeError(f"Rendered prompt must contain passage once: {stimulus_id}")
        passage_char_end = passage_char_start + len(passage)
        passage_token_start, passage_token_end = span_for_char_interval(
            offsets, passage_char_start, passage_char_end
        )

        active_spans: list[dict[str, Any]] = []
        active_positions: dict[int, int] = {}
        for item in stimulus["active_needle_spans"]:
            start, end = span_for_char_interval(
                offsets,
                passage_char_start + int(item["char_start"]),
                passage_char_start + int(item["char_end"]),
            )
            row = dict(item)
            row["token_start"] = start
            row["token_end"] = end
            active_spans.append(row)
            for position in range(start, end):
                if position in active_positions:
                    raise RuntimeError(f"Overlapping active spans in {stimulus_id}")
                active_positions[position] = int(item["slot_index"])

        negative_spans: list[dict[str, Any]] = []
        negative_positions: dict[int, int] = {}
        for item in stimulus["hard_negative_spans"]:
            start, end = span_for_char_interval(
                offsets,
                passage_char_start + int(item["char_start"]),
                passage_char_start + int(item["char_end"]),
            )
            row = dict(item)
            row["token_start"] = start
            row["token_end"] = end
            negative_spans.append(row)
            for position in range(start, end):
                if position in active_positions or position in negative_positions:
                    raise RuntimeError(f"Overlapping registered spans in {stimulus_id}")
                negative_positions[position] = int(item["slot_index"])

        metric_lookup = {
            (int(row["layer"]), int(row["head"])): row
            for row in read_csv_gz(metric_path)
        }
        with np.load(raw_path, allow_pickle=False) as raw:
            sequence_length = int(raw["sequence_length"][0])
            query_position = int(raw["query_position"][0])
            if len(ids) != sequence_length or query_position != sequence_length - 1:
                raise RuntimeError(
                    f"Retokenized prompt mismatch for {stimulus_id}: "
                    f"ids={len(ids)} raw={sequence_length} query={query_position}"
                )

            def region(position: int) -> tuple[str, int | None]:
                if position in active_positions:
                    return "active_needle", active_positions[position]
                if position in negative_positions:
                    return "hard_negative", negative_positions[position]
                if passage_token_start <= position < passage_token_end:
                    return "ordinary_passage", None
                if position == query_position:
                    return "answer_query", None
                return "prompt_wrapper_or_instruction", None

            for layer, head in ATTENTION_GALLERY_HEADS:
                metric_row = metric_lookup.get((layer, head))
                if metric_row is None:
                    raise RuntimeError(f"Missing L{layer}H{head} metric for {stimulus_id}")
                if metric_row["split"] != ATTENTION_SPLIT:
                    raise RuntimeError(f"Metric split mismatch for {stimulus_id} L{layer}H{head}")
                key_start = int(raw["key_starts"][layer])
                attention = np.asarray(raw[f"layer_{layer:03d}"][head], dtype=np.float64)
                if key_start != 0 or len(attention) != sequence_length:
                    raise RuntimeError(f"Attention row is not full-prefix for {stimulus_id}")
                if not np.isfinite(attention).all():
                    raise RuntimeError(f"Non-finite attention for {stimulus_id} L{layer}H{head}")

                def token_record(position: int) -> dict[str, object]:
                    start, end = offsets[position]
                    token_region, slot_index = region(position)
                    return {
                        "position": position,
                        "token_id": ids[position],
                        "display": visible_surface(
                            model_text, start, end, raw_tokens[position]
                        ),
                        "attention": float(attention[position]),
                        "region": token_region,
                        "slot_index": slot_index,
                    }

                stored_needle_masses = np.asarray(
                    json.loads(metric_row["needle_span_masses"]), dtype=np.float64
                )
                derived_needle_masses = np.asarray(
                    [
                        float(attention[item["token_start"] : item["token_end"]].sum())
                        for item in active_spans
                    ],
                    dtype=np.float64,
                )
                if len(stored_needle_masses) != len(derived_needle_masses):
                    raise RuntimeError(f"Needle mass length mismatch for {stimulus_id}")
                mass_difference = float(
                    np.max(np.abs(stored_needle_masses - derived_needle_masses))
                )
                if mass_difference > RAW_FLOAT16_MASS_TOLERANCE:
                    raise RuntimeError(
                        f"Needle masses do not reproduce metric for {stimulus_id} "
                        f"L{layer}H{head}: {mass_difference}"
                    )

                category_mass = defaultdict(float)
                for position, value in enumerate(attention):
                    category_mass[region(position)[0]] += float(value)
                stored_broad_mass = float(metric_row["broad_mass"])
                if (
                    abs(category_mass["active_needle"] - stored_broad_mass)
                    > RAW_FLOAT16_MASS_TOLERANCE * len(active_spans)
                ):
                    raise RuntimeError(f"Broad mass mismatch for {stimulus_id} L{layer}H{head}")

                needle_rows: list[dict[str, object]] = []
                for item, span_mass in zip(active_spans, derived_needle_masses):
                    positions = range(int(item["token_start"]), int(item["token_end"]))
                    needle_rows.append(
                        {
                            "slot_index": int(item["slot_index"]),
                            "needle_id": str(item["needle_id"]),
                            "city": str(item["city"]),
                            "score": int(item["score"]),
                            "text": str(item["text"]),
                            "token_start": int(item["token_start"]),
                            "token_end": int(item["token_end"]),
                            "attention_mass": float(span_mass),
                            "tokens": [token_record(position) for position in positions],
                        }
                    )
                hard_negative_rows = [
                    {
                        "slot_index": int(item["slot_index"]),
                        "text": str(item["text"]),
                        "token_start": int(item["token_start"]),
                        "token_end": int(item["token_end"]),
                        "attention_mass": float(
                            attention[item["token_start"] : item["token_end"]].sum()
                        ),
                    }
                    for item in negative_spans
                ]
                bins = [
                    {
                        "token_start": start,
                        "token_end": min(sequence_length, start + ATTENTION_GALLERY_BIN_WIDTH),
                        "attention_mass": float(
                            attention[start : min(sequence_length, start + ATTENTION_GALLERY_BIN_WIDTH)].sum()
                        ),
                    }
                    for start in range(0, sequence_length, ATTENTION_GALLERY_BIN_WIDTH)
                ]
                ordered_positions = sorted(
                    range(sequence_length),
                    key=lambda position: (-attention[position], position),
                )
                top_non_needle_tokens = [
                    token_record(position)
                    for position in ordered_positions
                    if position not in active_positions
                ][:12]
                attention_sum = float(attention.sum())
                if abs(sum(float(row["attention_mass"]) for row in bins) - attention_sum) > 1e-10:
                    raise RuntimeError(f"Binned attention mismatch for {stimulus_id}")
                global_bin_max = max(
                    global_bin_max,
                    max(float(row["attention_mass"]) for row in bins),
                )
                global_token_max = max(global_token_max, float(attention.max()))
                records.append(
                    {
                        "record_id": f"{stimulus_id}__L{layer}H{head}",
                        "selection": {
                            "stimulus_id": stimulus_id,
                            "seed": seed,
                            "gold_count": count,
                            "layer": layer,
                            "head": head,
                            "frozen_head_rank": expected_ranks[(layer, head)],
                        },
                        "prompt": {
                            "sequence_length": sequence_length,
                            "query_position": query_position,
                            "passage_token_start": passage_token_start,
                            "passage_token_end": passage_token_end,
                            "active_needle_count": len(active_spans),
                            "hard_negative_count": len(negative_spans),
                        },
                        "attention": {
                            "sum": attention_sum,
                            "category_mass": dict(category_mass),
                            "max_token_attention": float(attention.max()),
                            "broad_primary": float(metric_row["broad_primary"]),
                            "broad_mass": stored_broad_mass,
                            "broad_coverage": float(metric_row["broad_coverage"]),
                            "bins": bins,
                            "needle_rows": needle_rows,
                            "hard_negative_rows": hard_negative_rows,
                            "top_non_needle_tokens": top_non_needle_tokens,
                        },
                    }
                )
                prompt_audits.append(
                    {
                        "stimulus_id": stimulus_id,
                        "layer": layer,
                        "head": head,
                        "sequence_length": sequence_length,
                        "query_position": query_position,
                        "attention_sum": attention_sum,
                        "stored_broad_mass": stored_broad_mass,
                        "derived_broad_mass": category_mass["active_needle"],
                        "needle_mass_max_abs_difference": mass_difference,
                    }
                )

    expected_coverage = {
        (seed, count, layer, head)
        for seed, count in ATTENTION_GALLERY_PROMPTS
        for layer, head in ATTENTION_GALLERY_HEADS
    }
    observed_coverage = {
        (
            int(row["selection"]["seed"]),
            int(row["selection"]["gold_count"]),
            int(row["selection"]["layer"]),
            int(row["selection"]["head"]),
        )
        for row in records
    }
    if observed_coverage != expected_coverage:
        raise RuntimeError("Attention gallery Cartesian coverage mismatch")

    payload: dict[str, Any] = {
        "schema_version": "niah_nonthinking_attention_gallery_v1",
        "selection": {
            "purpose": "display_only",
            "model_label": ATTENTION_MODEL,
            "design_variant": ATTENTION_VARIANT,
            "split": ATTENTION_SPLIT,
            "heads": [
                {"layer": layer, "head": head, "frozen_rank": expected_ranks[(layer, head)]}
                for layer, head in ATTENTION_GALLERY_HEADS
            ],
            "head_rule": "discovery-frozen formal V4.4 span-sum broad ranking top four",
            "prompts": [
                {"seed": seed, "gold_count": count}
                for seed, count in ATTENTION_GALLERY_PROMPTS
            ],
            "prompt_rule": (
                "smallest confirmation seed (1254) crossed with fixed low/middle/high "
                "counts 3, 6, and 9; no raw-attention outcome used for prompt selection"
            ),
        },
        "display": {
            "bin_width_tokens": ATTENTION_GALLERY_BIN_WIDTH,
            "global_bin_mass_max": global_bin_max,
            "global_token_attention_max": global_token_max,
        },
        "records": records,
        "interpretation_boundary": (
            "The gallery visualizes natural answer-query routing for fixed heads and prompts. "
            "It is not OV/logit attribution and does not establish causal necessity or a unique route."
        ),
    }
    audit: dict[str, object] = {
        "status": "PASS",
        "selection_rule": payload["selection"],
        "expected_records": len(expected_coverage),
        "observed_records": len(records),
        "record_audits": prompt_audits,
        "raw_float16_mass_tolerance": RAW_FLOAT16_MASS_TOLERANCE,
        "source_sha256": {
            "stimuli_jsonl": sha256_file(stimuli_path),
            "attention_capture_index_jsonl": sha256_file(index_path),
            "tokenizer_json": sha256_file(tokenizer_path),
            "raw_npz": {
                stimulus_id: sha256_file(stage3_root / f"{stimulus_id}.npz")
                for stimulus_id in sorted(stimulus_ids)
            },
            "metric_shards": {
                stimulus_id: sha256_file(stage3_root / f"{stimulus_id}.csv.gz")
                for stimulus_id in sorted(stimulus_ids)
            },
        },
    }
    return payload, audit


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage1-root",
        type=Path,
        default=ROOT / "work" / "nonthinking_report_filestream_stage1",
    )
    parser.add_argument(
        "--stage2-root",
        type=Path,
        default=ROOT / "work" / "nonthinking_report_filestream_stage2",
    )
    parser.add_argument(
        "--stage3-root",
        type=Path,
        default=ROOT / "work" / "nonthinking_report_filestream_stage3",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "reports"
            / "v4_non-thinking_causal"
            / "v4_4_report_additions"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    atlas_path = (
        ROOT
        / "reports"
        / "v4_non-thinking_causal"
        / "v4_4"
        / "realistic_niah_v4_head_atlas.csv"
    )
    formal_path = (
        ROOT
        / "reports"
        / "v4_non-thinking_causal"
        / "v4_4_causal_v2"
        / "full_span_topk"
        / "full_span_topk_primary_statistics.csv"
    )
    atlas_rows = read_csv(atlas_path)
    formal_rows = read_csv(formal_path)
    topk_rows, topk_audit = raw_topk_statistics(
        args.stage1_root,
        atlas_rows=atlas_rows,
        formal_rows=formal_rows,
    )
    attention_payload, attention_audit = attention_example(
        args.stage1_root,
        args.stage2_root,
        atlas_rows=atlas_rows,
    )
    gallery_payload, gallery_audit = attention_gallery(
        args.stage2_root,
        args.stage3_root,
        atlas_rows=atlas_rows,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    topk_output = args.output_dir / "full_span_topk_raw_arms.csv"
    attention_output = args.output_dir / "qwen_attention_example_l27h18.json"
    gallery_output = args.output_dir / "qwen_attention_gallery.json"
    audit_output = args.output_dir / "report_additions_audit.json"
    write_csv(topk_output, topk_rows)
    attention_output.write_text(
        json.dumps(attention_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    gallery_output.write_text(
        json.dumps(gallery_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    audit_payload = {
        "schema_version": "niah_nonthinking_report_additions_audit_v1",
        "status": "PASS",
        "topk": topk_audit,
        "attention": attention_audit,
        "attention_gallery": gallery_audit,
        "outputs": {
            "full_span_topk_raw_arms.csv": sha256_file(topk_output),
            "qwen_attention_example_l27h18.json": sha256_file(attention_output),
            "qwen_attention_gallery.json": sha256_file(gallery_output),
        },
    }
    audit_output.write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "topk_rows": len(topk_rows),
                "attention_stimulus": attention_payload["selection"]["stimulus_id"],
                "attention_gallery_records": len(gallery_payload["records"]),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
