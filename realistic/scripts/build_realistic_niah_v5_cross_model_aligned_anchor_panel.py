#!/usr/bin/env python3
"""Freeze one outcome-blind, exactly paired terminal transition per seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


DEVELOPMENT_SEEDS = tuple(range(1234, 1254))
CONFIRMATION_SEEDS = tuple(range(1254, 1264))
MODEL_LABELS = ("Qwen3-8B", "Gemma4-E4B")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _transition_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row["seed"]),
        int(row["gold_count"]),
        int(row["from_occurrence"]),
        int(row["to_occurrence"]),
    )


def _index_rows(
    rows: Iterable[dict[str, Any]], *, source_name: str
) -> dict[tuple[int, int, int, int], dict[str, Any]]:
    indexed: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for row in rows:
        if "selection_rank" in row:
            raise ValueError(f"{source_name} contains forbidden selection_rank")
        key = _transition_key(row)
        if key in indexed:
            raise ValueError(f"{source_name} duplicates transition key {key}")
        indexed[key] = dict(row)
    return indexed


def _phase(seed: int) -> str:
    if seed in DEVELOPMENT_SEEDS:
        return "development"
    if seed in CONFIRMATION_SEEDS:
        return "confirmation"
    raise ValueError(f"Seed {seed} is outside the frozen 20+10 contract")


def build_aligned_panel(
    qwen_rows: Iterable[dict[str, Any]],
    gemma_rows: Iterable[dict[str, Any]],
    *,
    require_explicit_rank: bool = False,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Return shared keys, model panels, and an exact-alignment manifest."""

    qwen = _index_rows(qwen_rows, source_name="Qwen registry")
    gemma = _index_rows(gemma_rows, source_name="Gemma registry")
    def explicit_rank(row: dict[str, Any]) -> bool:
        grammar = str(row.get("target_grammar_class", ""))
        return "rank_before_city" in grammar or "rank_after_city" in grammar

    common = {
        key
        for key in set(qwen) & set(gemma)
        if int(key[3]) == int(key[1]) and int(key[2]) == int(key[1]) - 1
        and (
            not require_explicit_rank
            or (explicit_rank(qwen[key]) and explicit_rank(gemma[key]))
        )
    }
    selected_keys: list[tuple[int, int, int, int]] = []
    for seed in DEVELOPMENT_SEEDS + CONFIRMATION_SEEDS:
        candidates = sorted(
            (key for key in common if int(key[0]) == int(seed)),
            key=lambda key: (-int(key[1]), int(key[2]), int(key[3])),
        )
        if not candidates:
            raise ValueError(f"Seed {seed} has no common terminal transition")
        selected_keys.append(candidates[0])

    shared: list[dict[str, Any]] = []
    model_panels = {label: [] for label in MODEL_LABELS}
    for panel_index, key in enumerate(selected_keys):
        seed, gold_count, from_occurrence, to_occurrence = key
        pair_id = f"seed{seed}_N{gold_count}_{from_occurrence}to{to_occurrence}"
        shared_row = {
            "alignment_pair_id": pair_id,
            "panel_index": panel_index,
            "phase": _phase(seed),
            "seed": seed,
            "gold_count": gold_count,
            "from_occurrence": from_occurrence,
            "to_occurrence": to_occurrence,
            "qwen_request_id": str(qwen[key]["request_id"]),
            "gemma_request_id": str(gemma[key]["request_id"]),
            "qwen_grammar_class": str(qwen[key].get("target_grammar_class", "")),
            "gemma_grammar_class": str(gemma[key].get("target_grammar_class", "")),
            "selection_rule": (
                ("common_explicit_rank_terminal_transition" if require_explicit_rank else "common_terminal_transition")
                + "_then_highest_gold_count_then_"
                "lexicographic_transition"
            ),
            "outcome_blind": True,
            "selection_rank_used": False,
        }
        shared.append(shared_row)
        for label, source in (("Qwen3-8B", qwen), ("Gemma4-E4B", gemma)):
            row = dict(source[key])
            row.update(
                {
                    "alignment_pair_id": pair_id,
                    "alignment_panel_index": panel_index,
                    "alignment_phase": _phase(seed),
                    "alignment_key": list(key),
                    "alignment_outcome_blind": True,
                    "alignment_selection_rank_used": False,
                    "alignment_selection_rule": shared_row["selection_rule"],
                }
            )
            model_panels[label].append(row)

    expected_seeds = list(DEVELOPMENT_SEEDS + CONFIRMATION_SEEDS)
    if [int(row["seed"]) for row in shared] != expected_seeds:
        raise RuntimeError("Aligned panel changed the frozen seed order")
    qwen_keys = [_transition_key(row) for row in model_panels["Qwen3-8B"]]
    gemma_keys = [_transition_key(row) for row in model_panels["Gemma4-E4B"]]
    if qwen_keys != gemma_keys or qwen_keys != selected_keys:
        raise RuntimeError("Cross-model transition keys are not exactly aligned")

    core = {
        "schema_version": "realistic_niah_v5_cross_model_aligned_anchor_panel_v1",
        "status": "FROZEN_OUTCOME_BLIND_EXACT_ALIGNMENT",
        "alignment_key_fields": [
            "seed",
            "gold_count",
            "from_occurrence",
            "to_occurrence",
        ],
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "development_sample_count": len(DEVELOPMENT_SEEDS),
        "confirmation_sample_count": len(CONFIRMATION_SEEDS),
        "total_sample_count_per_model": len(selected_keys),
        "common_registry_transition_count": len(common),
        "require_explicit_rank_in_both_models": bool(require_explicit_rank),
        "outcome_blind": True,
        "selection_rank_used": False,
        "selection_rule": shared[0]["selection_rule"],
        "exact_cross_model_key_equality": True,
        "selected_alignment_keys": [list(key) for key in selected_keys],
    }
    manifest = {
        **core,
        "shared_panel_sha256": _sha256_json(shared),
        "qwen_panel_sha256": _sha256_json(model_panels["Qwen3-8B"]),
        "gemma_panel_sha256": _sha256_json(model_panels["Gemma4-E4B"]),
        "manifest_sha256": _sha256_json(core),
    }
    return shared, model_panels["Qwen3-8B"], model_panels["Gemma4-E4B"], manifest


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def _freeze(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Existing frozen output changed: {path}")
    _atomic_text(path, text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-registry", type=Path, required=True)
    parser.add_argument("--gemma-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-explicit-rank", action="store_true")
    args = parser.parse_args()

    shared, qwen, gemma, manifest = build_aligned_panel(
        _read_jsonl(args.qwen_registry),
        _read_jsonl(args.gemma_registry),
        require_explicit_rank=bool(args.require_explicit_rank),
    )
    outputs = {
        args.output_dir / "shared_alignment_keys.jsonl": _jsonl_text(shared),
        args.output_dir / "Qwen3-8B_anchor_panel.jsonl": _jsonl_text(qwen),
        args.output_dir / "Gemma4-E4B_anchor_panel.jsonl": _jsonl_text(gemma),
        args.output_dir / "alignment_manifest.json": (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ),
    }
    for path, text in outputs.items():
        _freeze(path, text)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
