#!/usr/bin/env python3
"""Build one surface-matched explicit-index control panel for both models.

This builder is deliberately outcome blind.  It selects the canonical N=10
stimulus for seeds 1234--1263 in both model registries, verifies that the
stimulus and ordered gold records are identical, and teacher-forces the same
numbered list body in both models.  Only the native channel wrappers differ.

The resulting rows are a controlled explicit-index positive control.  They do
not provide evidence for spontaneous no-index counting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.modeling import load_registered_tokenizer  # noqa: E402
from realistic_niah_v4.spec import resolve_model_spec  # noqa: E402
from realistic_niah_v5.parsing import parse_trace_record  # noqa: E402


SCHEMA_VERSION = "realistic_niah_v5_cross_model_aligned_indexed_progress_v1"
MODEL_LABELS = ("Qwen3-8B", "Gemma4-E4B")
DEFAULT_SEEDS = tuple(range(1234, 1264))
SURFACE_TEMPLATE = "k. City - score"
GRAMMAR_CLASS = "cross_model_surface_matched_index_before_city"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class _TokenizerJsonAdapter:
    """Small adapter for a bare ``tokenizers.Tokenizer`` instance."""

    def __init__(self, path: Path) -> None:
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(path))

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(
            self._tokenizer.encode(
                str(text), add_special_tokens=bool(add_special_tokens)
            ).ids
        )

    def decode(
        self,
        ids: Sequence[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del clean_up_tokenization_spaces
        return self._tokenizer.decode(
            [int(value) for value in ids],
            skip_special_tokens=bool(skip_special_tokens),
        )


def _load_tokenizer(
    model_label: str,
    *,
    cache_dir: Path | None,
    tokenizer_json: Path | None,
) -> Any:
    if tokenizer_json is not None:
        return _TokenizerJsonAdapter(tokenizer_json)
    return load_registered_tokenizer(
        resolve_model_spec(model_label),
        cache_dir=cache_dir,
    )


def _records(row: Mapping[str, Any]) -> tuple[tuple[str, int, int], ...]:
    values: list[tuple[str, int, int]] = []
    for record in row.get("gold_records", ()):  # ordered passage registry
        values.append(
            (
                str(record["city"]),
                int(record["score"]),
                int(record.get("slot_index", len(values) + 1)),
            )
        )
    return tuple(values)


def _source_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_label: str,
    seeds: Sequence[int],
    gold_count: int,
) -> dict[int, dict[str, Any]]:
    requested = {int(seed) for seed in seeds}
    selected: dict[int, dict[str, Any]] = {}
    for value in rows:
        row = dict(value)
        if str(row.get("model_label", row.get("model", ""))) != model_label:
            continue
        if int(row.get("gold_count", -1)) != int(gold_count):
            continue
        seed = int(row.get("seed", -1))
        if seed not in requested:
            continue
        if seed in selected:
            raise ValueError(f"Duplicate {model_label} seed={seed}, N={gold_count}")
        selected[seed] = row
    missing = sorted(requested - set(selected))
    if missing:
        raise ValueError(f"Missing {model_label} N={gold_count} seeds: {missing}")
    return selected


def _body(row: Mapping[str, Any]) -> str:
    return "\n".join(
        f"{index}. {city} - {score}"
        for index, (city, score, _slot) in enumerate(_records(row), start=1)
    )


def _wrapped_text(model_label: str, body: str, *, gold_count: int) -> str:
    if model_label == "Qwen3-8B":
        return (
            f"<think>\n{body}\n</think>\n\n"
            f"Total: {int(gold_count)}<|im_end|>"
        )
    if model_label == "Gemma4-E4B":
        return (
            f"<|channel>thought\n{body}\n<channel|>"
            f"Total: {int(gold_count)}<turn|>"
        )
    raise KeyError(model_label)


def _parsed_items(row: Mapping[str, Any]) -> tuple[str, ...]:
    parser = dict(row.get("trace_parse", {}).get("parser", {}))
    starts = tuple(int(value) for value in parser.get("item_start_chars", ()))
    ends = tuple(int(value) for value in parser.get("item_end_chars", ()))
    raw = str(row.get("raw_output_text", ""))
    if len(starts) != len(ends):
        return ()
    return tuple(raw[start:end].strip() for start, end in zip(starts, ends))


def _audit_row(
    row: Mapping[str, Any],
    *,
    expected_body: str,
    gold_count: int,
) -> dict[str, Any]:
    parser = dict(row.get("trace_parse", {}).get("parser", {}))
    records = _records(row)
    expected_items = tuple(expected_body.splitlines())
    observed_items = _parsed_items(row)
    reasons: list[str] = []
    if int(row.get("gold_count", -1)) != int(gold_count):
        reasons.append("gold_count_mismatch")
    if len(records) != int(gold_count):
        reasons.append("ordered_gold_registry_not_complete")
    if int(parser.get("item_count", -1)) != int(gold_count):
        reasons.append("parser_item_count_mismatch")
    if not bool(parser.get("trace_one_to_one")):
        reasons.append("trace_not_one_to_one")
    if tuple(int(value) for value in parser.get("item_markers", ())) != tuple(
        range(1, int(gold_count) + 1)
    ):
        reasons.append("visible_indices_not_exact")
    if tuple(str(value) for value in parser.get("item_gold_cities", ())) != tuple(
        city for city, _score, _slot in records
    ):
        reasons.append("parsed_city_order_mismatch")
    if observed_items != expected_items:
        reasons.append("surface_items_mismatch")
    exact_count = bool(row.get("trace_parse", {}).get("exact_count"))
    if not exact_count:
        reasons.append("final_count_not_exact")
    return {
        "status": "PASS" if not reasons else "FAIL",
        "primary_eligible_indexed_positive_control": not reasons,
        "reasons": reasons,
        "gold_count": int(gold_count),
        "grammar_class": GRAMMAR_CLASS,
        "marker_kind": "indexed",
        "surface_template": SURFACE_TEMPLATE,
        "inner_enumeration_surface_matched_across_models": True,
        "model_native_channel_wrapper_retained": True,
        "controlled_teacher_forced_trace": True,
        "visible_indices": list(range(1, int(gold_count) + 1)),
        "expected_surface_items_sha256": _json_sha256(expected_items),
        "observed_surface_items_sha256": _json_sha256(observed_items),
        "selection_uses_hidden_states": False,
        "selection_uses_patch_outcomes": False,
        "internal_counter_without_visible_index_claim_allowed": False,
    }


def _build_row(
    source: Mapping[str, Any],
    *,
    model_label: str,
    tokenizer: Any,
    gold_count: int,
) -> dict[str, Any]:
    body = _body(source)
    intended = _wrapped_text(model_label, body, gold_count=gold_count)
    output_ids = tuple(
        int(value)
        for value in tokenizer.encode(intended, add_special_tokens=False)
    )
    raw = tokenizer.decode(
        output_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if raw != intended:
        raise ValueError(f"{model_label} tokenizer does not exactly round-trip panel")
    clean = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    seed = int(source["seed"])
    row = dict(source)
    row.update(
        {
            "schema_version": SCHEMA_VERSION,
            "request_id": (
                f"{model_label}/native_thinking/v5/"
                f"aligned_indexed/{source['stimulus_id']}"
            ),
            "model_label": model_label,
            "model_family": "qwen3" if model_label == "Qwen3-8B" else "gemma4",
            "prompt_mode": "native_thinking",
            "split": "discovery" if seed <= 1253 else "confirmation",
            "gold_count": int(gold_count),
            "output_token_ids": list(output_ids),
            "output_tokens": len(output_ids),
            "raw_output_text": raw,
            "clean_output_text": clean,
            "sampling_seed": seed,
            "decoding": {
                "mode": "controlled_teacher_forcing",
                "outcome_blind": True,
            },
            "stopped_on_eos": bool(
                source.get("generation_eos_token_ids")
                and int(output_ids[-1])
                in {
                    int(value)
                    for value in source.get("generation_eos_token_ids", ())
                }
            ),
            "generation_truncated": False,
            "elapsed_seconds": 0.0,
            "controlled_indexed_surface_text": body,
        }
    )
    row["trace_parse"] = parse_trace_record(row)
    audit = _audit_row(row, expected_body=body, gold_count=gold_count)
    if not audit["primary_eligible_indexed_positive_control"]:
        raise ValueError(
            f"Aligned indexed row failed seed={seed}, model={model_label}: "
            f"{audit['reasons']}"
        )
    row["indexed_progress_control_format_audit"] = audit
    row["indexed_progress_control_cohort"] = {
        "schema_version": SCHEMA_VERSION,
        "selection_population": "indexed_positive_control",
        "model_label": model_label,
        "split": str(row["split"]),
        "alignment_key": {
            "phase": str(row["split"]),
            "seed": seed,
            "gold_count": int(gold_count),
        },
        "grammar_class": GRAMMAR_CLASS,
        "surface_template": SURFACE_TEMPLATE,
        "controlled_teacher_forced_trace": True,
        "visible_progress_confound_allowed": True,
        "internal_counter_without_visible_index_claim_allowed": False,
        "selection_independent_of_hidden_states": True,
        "selection_independent_of_patch_outcomes": True,
    }
    return row


def build_panel(
    *,
    qwen_rows: Sequence[Mapping[str, Any]],
    gemma_rows: Sequence[Mapping[str, Any]],
    qwen_tokenizer: Any,
    gemma_tokenizer: Any,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    gold_count: int = 10,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    indexes = {
        "Qwen3-8B": _source_index(
            qwen_rows,
            model_label="Qwen3-8B",
            seeds=seeds,
            gold_count=gold_count,
        ),
        "Gemma4-E4B": _source_index(
            gemma_rows,
            model_label="Gemma4-E4B",
            seeds=seeds,
            gold_count=gold_count,
        ),
    }
    tokenizers = {
        "Qwen3-8B": qwen_tokenizer,
        "Gemma4-E4B": gemma_tokenizer,
    }
    shared_keys: list[dict[str, Any]] = []
    output = {model: [] for model in MODEL_LABELS}
    for seed in sorted(int(value) for value in seeds):
        left = indexes["Qwen3-8B"][seed]
        right = indexes["Gemma4-E4B"][seed]
        left_key = (
            str(left.get("stimulus_id", "")),
            _records(left),
        )
        right_key = (
            str(right.get("stimulus_id", "")),
            _records(right),
        )
        if left_key != right_key:
            raise ValueError(f"Cross-model stimulus mismatch for seed={seed}")
        body = _body(left)
        shared_keys.append(
            {
                "phase": "discovery" if seed <= 1253 else "confirmation",
                "seed": seed,
                "gold_count": int(gold_count),
                "stimulus_id": str(left["stimulus_id"]),
                "ordered_gold_records_sha256": _json_sha256(_records(left)),
                "enumeration_body_sha256": _json_sha256(body),
            }
        )
        for model_label in MODEL_LABELS:
            output[model_label].append(
                _build_row(
                    indexes[model_label][seed],
                    model_label=model_label,
                    tokenizer=tokenizers[model_label],
                    gold_count=gold_count,
                )
            )
    for model_label in MODEL_LABELS:
        model_keys = [
            (
                row["split"],
                int(row["seed"]),
                int(row["gold_count"]),
            )
            for row in output[model_label]
        ]
        expected_keys = [
            (row["phase"], int(row["seed"]), int(row["gold_count"]))
            for row in shared_keys
        ]
        if model_keys != expected_keys:
            raise RuntimeError(f"Internal alignment failure for {model_label}")
    return output, shared_keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-generations", type=Path, required=True)
    parser.add_argument("--gemma-generations", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--qwen-tokenizer-json", type=Path)
    parser.add_argument("--gemma-tokenizer-json", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--gold-count", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    qwen_tokenizer = _load_tokenizer(
        "Qwen3-8B",
        cache_dir=args.cache_dir,
        tokenizer_json=args.qwen_tokenizer_json,
    )
    gemma_tokenizer = _load_tokenizer(
        "Gemma4-E4B",
        cache_dir=args.cache_dir,
        tokenizer_json=args.gemma_tokenizer_json,
    )
    output, shared_keys = build_panel(
        qwen_rows=_read_jsonl(args.qwen_generations),
        gemma_rows=_read_jsonl(args.gemma_generations),
        qwen_tokenizer=qwen_tokenizer,
        gemma_tokenizer=gemma_tokenizer,
        seeds=tuple(args.seeds),
        gold_count=int(args.gold_count),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        model_label: args.output / f"{model_label}.jsonl"
        for model_label in MODEL_LABELS
    }
    for model_label in MODEL_LABELS:
        _write_jsonl(paths[model_label], output[model_label])
    shared_path = args.output / "shared_alignment_keys.jsonl"
    _write_jsonl(shared_path, shared_keys)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "FROZEN_OUTCOME_BLIND_EXACT_ALIGNMENT",
        "claim_role": "controlled explicit-index positive control only",
        "gold_count": int(args.gold_count),
        "surface_template": SURFACE_TEMPLATE,
        "grammar_class": GRAMMAR_CLASS,
        "inner_enumeration_surface_matched_across_models": True,
        "model_native_channel_wrapper_retained": True,
        "controlled_teacher_forced_trace": True,
        "internal_counter_without_visible_index_claim_allowed": False,
        "selection_independent_of_hidden_states": True,
        "selection_independent_of_patch_outcomes": True,
        "alignment_key_fields": ["phase", "seed", "gold_count"],
        "exact_cross_model_key_equality": True,
        "sample_count_per_model": len(shared_keys),
        "discovery_seeds": [
            int(row["seed"]) for row in shared_keys if row["phase"] == "discovery"
        ],
        "confirmation_seeds": [
            int(row["seed"])
            for row in shared_keys
            if row["phase"] == "confirmation"
        ],
        "shared_alignment_keys": str(shared_path),
        "shared_alignment_keys_sha256": _sha256(shared_path),
        "models": {
            model_label: {
                "output": str(paths[model_label]),
                "output_sha256": _sha256(paths[model_label]),
                "sample_count": len(output[model_label]),
            }
            for model_label in MODEL_LABELS
        },
        "sources": {
            "Qwen3-8B": str(args.qwen_generations),
            "Gemma4-E4B": str(args.gemma_generations),
        },
    }
    _write_json(args.output / "alignment_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
