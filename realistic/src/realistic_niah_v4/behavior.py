from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .modeling import generate_answer_completion
from .prompts import PromptEncoding


_STRICT_NUMERIC_RE = re.compile(r"[0-9]+")
_ISOLATED_INTEGER_RE = re.compile(r"(?<![0-9])[0-9]+(?![0-9])")


def count_logit_metrics(
    logits: torch.Tensor | np.ndarray,
    encoding: PromptEncoding,
) -> dict[str, Any]:
    """Reduce one answer-query vocabulary vector to the registered count outcomes."""

    values = (
        logits.detach().float().cpu().numpy()
        if isinstance(logits, torch.Tensor)
        else np.asarray(logits, dtype=float)
    )
    if values.ndim != 1:
        raise ValueError("count_logit_metrics expects one vocabulary vector")
    candidates: list[tuple[int, int]] = []
    for count, token_ids in encoding.count_candidate_answer_token_ids:
        if len(token_ids) != 1:
            raise ValueError(
                "count_logit_metrics is only valid for distinct single-token "
                "answers; use joint sequence log-probabilities for numeric V4"
            )
        candidates.append((int(count), int(token_ids[0])))
    candidates.sort()
    if len({token_id for _, token_id in candidates}) != len(candidates):
        raise ValueError(
            "count_logit_metrics cannot distinguish candidates that share an "
            "initial token; use joint sequence log-probabilities"
        )
    counts = np.asarray([count for count, _ in candidates], dtype=float)
    token_ids = np.asarray([token_id for _, token_id in candidates], dtype=int)
    if int(token_ids.max()) >= len(values):
        raise ValueError("A count candidate token is outside the vocabulary")
    candidate_logits = values[token_ids].astype(float)
    shifted = candidate_logits - float(candidate_logits.max())
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    correct_index = int(np.flatnonzero(counts == encoding.count)[0])
    other = np.delete(candidate_logits, correct_index)
    return {
        "gold_count": int(encoding.count),
        "predicted_count_among_candidates": int(counts[int(candidate_logits.argmax())]),
        "correct_count_logit": float(candidate_logits[correct_index]),
        "correct_count_margin": float(candidate_logits[correct_index] - other.max()),
        "correct_count_probability": float(probabilities[correct_index]),
        "expected_count": float(np.sum(probabilities * counts)),
        "candidate_counts": ",".join(str(int(value)) for value in counts),
        "candidate_logits": ",".join(
            f"{float(value):.9g}" for value in candidate_logits
        ),
        "candidate_probabilities": ",".join(
            f"{float(value):.9g}" for value in probabilities
        ),
    }


def parse_numeric_completion(
    completion_text: str,
    *,
    valid_counts: tuple[int, ...] = tuple(range(1, 11)),
) -> dict[str, Any]:
    """Parse a generated continuation after the registered ``Total:`` prefix.

    ``parsed_count`` is deliberately strict: the special-token-stripped
    continuation must contain only one in-range decimal integer (surrounding
    whitespace is ignored). ``extracted_count`` is a diagnostic-only, lenient
    field populated when exactly one isolated in-range integer occurs anywhere
    in the continuation. The latter must never be used as the primary accuracy
    label.
    """

    text = str(completion_text)
    stripped = text.strip()
    allowed = {int(value) for value in valid_counts}
    strict_match = _STRICT_NUMERIC_RE.fullmatch(stripped)
    parsed = int(stripped) if strict_match is not None else None
    format_valid = parsed in allowed if parsed is not None else False
    isolated = [int(value) for value in _ISOLATED_INTEGER_RE.findall(stripped)]
    extracted = isolated[0] if len(isolated) == 1 and isolated[0] in allowed else None
    return {
        "completion_text_stripped": stripped,
        "format_valid": bool(format_valid),
        "parsed_count": int(parsed) if format_valid else None,
        "extracted_count": int(extracted) if extracted is not None else None,
        "isolated_integer_count": len(isolated),
    }


def label_generated_completion(
    completion_text: str,
    *,
    gold_count: int,
    valid_counts: tuple[int, ...] = tuple(range(1, 11)),
) -> dict[str, Any]:
    """Assign strict correct/wrong/invalid labels from the actual generation."""

    parsed = parse_numeric_completion(completion_text, valid_counts=valid_counts)
    predicted = parsed["parsed_count"]
    if not parsed["format_valid"]:
        outcome = "invalid"
        is_correct = False
        count_error = None
    else:
        is_correct = int(predicted) == int(gold_count)
        outcome = "correct" if is_correct else "wrong"
        count_error = int(predicted) - int(gold_count)
    return {
        **parsed,
        "gold_count": int(gold_count),
        "is_correct": bool(is_correct),
        "outcome_group": outcome,
        "count_error": count_error,
        "error_direction": (
            "invalid"
            if count_error is None
            else "undercount"
            if count_error < 0
            else "overcount"
            if count_error > 0
            else "none"
        ),
        "omission_count": max(0, -int(count_error)) if count_error is not None else None,
        "extra_count": max(0, int(count_error)) if count_error is not None else None,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_generation_record(
    payload: dict[str, Any],
    *,
    encoding: PromptEncoding,
    max_new_tokens: int,
) -> None:
    expected = {
        "stimulus_id": encoding.stimulus_id,
        "model_label": encoding.model_label,
        "design_variant": encoding.design_variant,
        "seed": int(encoding.seed),
        "gold_count": int(encoding.count),
        "max_new_tokens": int(max_new_tokens),
        "decoding": "greedy",
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Incompatible V4 generation-label shard for {encoding.stimulus_id}: "
            f"{mismatches}"
        )
    if payload.get("outcome_group") not in {"correct", "wrong", "invalid"}:
        raise RuntimeError(f"Invalid outcome label for {encoding.stimulus_id}")


@torch.inference_mode()
def capture_generation_labels(
    model: Any,
    tokenizer: Any,
    encodings: list[PromptEncoding] | tuple[PromptEncoding, ...] | Any,
    *,
    output_dir: str | Path,
    valid_counts: tuple[int, ...] = tuple(range(1, 11)),
    max_new_tokens: int = 16,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Greedily generate, strictly label, and checkpoint every V4 prompt.

    The attention and representation captures remain immutable. These behavior
    rows are joined later by ``stimulus_id`` so labels always retain explicit
    decoding provenance.
    """

    if int(max_new_tokens) < 2:
        raise ValueError("max_new_tokens must allow the two-token answer 10")
    output = Path(output_dir)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for example_index, encoding in enumerate(encodings):
        if encoding.stimulus_id in seen:
            raise ValueError(f"Duplicate V4 behavior stimulus: {encoding.stimulus_id}")
        seen.add(encoding.stimulus_id)
        relative = (
            Path("shards")
            / encoding.design_variant
            / f"{encoding.stimulus_id}.json"
        )
        shard = output / relative
        if shard.exists() and not overwrite:
            payload = json.loads(shard.read_text(encoding="utf-8"))
            _validate_generation_record(
                payload,
                encoding=encoding,
                max_new_tokens=max_new_tokens,
            )
        else:
            start = time.perf_counter()
            generated = generate_answer_completion(
                model,
                tokenizer,
                encoding,
                max_new_tokens=int(max_new_tokens),
            )
            labels = label_generated_completion(
                generated["completion_text"],
                gold_count=encoding.count,
                valid_counts=valid_counts,
            )
            payload = {
                "schema_version": "realistic_niah_v4_generation_label_v1",
                "example_index": int(example_index),
                "stimulus_id": encoding.stimulus_id,
                "design_variant": encoding.design_variant,
                "model_label": encoding.model_label,
                "seed": int(encoding.seed),
                "split": encoding.split,
                "gold_count": int(encoding.count),
                "sequence_length": int(encoding.sequence_length),
                "query_position": int(encoding.query_position),
                "answer_prefix": "Total:",
                "decoding": "greedy",
                "do_sample": False,
                "max_new_tokens": int(max_new_tokens),
                "elapsed_seconds": float(time.perf_counter() - start),
                **generated,
                **labels,
            }
            _atomic_json(shard, payload)
        records.append({**payload, "shard_path": relative.as_posix()})
        print(
            "[v4 behavior] "
            f"{example_index + 1} {encoding.design_variant} "
            f"seed={encoding.seed} N={encoding.count} "
            f"outcome={payload['outcome_group']} "
            f"prediction={payload.get('parsed_count')}",
            flush=True,
        )
    if not records:
        raise ValueError("No V4 behavior encodings were supplied")

    index_path = output / "generation_label_index.jsonl"
    temporary = index_path.with_name(index_path.name + ".tmp")
    output.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        for row in records:
            compact = {
                key: row.get(key)
                for key in (
                    "stimulus_id",
                    "design_variant",
                    "model_label",
                    "seed",
                    "split",
                    "gold_count",
                    "outcome_group",
                    "is_correct",
                    "format_valid",
                    "parsed_count",
                    "count_error",
                    "shard_path",
                )
            }
            handle.write(json.dumps(compact, sort_keys=True) + "\n")
    temporary.replace(index_path)

    table = pd.DataFrame(records)
    labels_path = output / "generation_labels.csv"
    table.to_csv(labels_path, index=False)
    summary = (
        table.groupby(
            ["model_label", "design_variant", "split", "gold_count"],
            as_index=False,
            dropna=False,
        )
        .agg(
            examples=("stimulus_id", "count"),
            correct=("is_correct", "sum"),
            accuracy=("is_correct", "mean"),
            format_valid_rate=("format_valid", "mean"),
            mean_elapsed_seconds=("elapsed_seconds", "mean"),
        )
    )
    summary_path = output / "generation_accuracy_by_count.csv"
    summary.to_csv(summary_path, index=False)
    manifest_path = output / "generation_label_manifest.json"
    _atomic_json(
        manifest_path,
        {
            "schema_version": "realistic_niah_v4_generation_label_manifest_v1",
            "decoding": "greedy",
            "do_sample": False,
            "max_new_tokens": int(max_new_tokens),
            "valid_counts": [int(value) for value in valid_counts],
            "examples": len(table),
            "outcome_counts": {
                str(key): int(value)
                for key, value in table["outcome_group"].value_counts().items()
            },
            "strict_label_rule": (
                "After removing special tokens, the continuation following the "
                "already-present Total: prefix must be exactly one in-range "
                "decimal integer, allowing only surrounding whitespace."
            ),
            "labels_csv": str(labels_path),
            "accuracy_by_count_csv": str(summary_path),
            "index_jsonl": str(index_path),
        },
    )
    if not math.isclose(float(table["is_correct"].notna().mean()), 1.0):
        raise RuntimeError("Generation label table contains missing correctness labels")
    return {
        "index": index_path,
        "labels": labels_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }
