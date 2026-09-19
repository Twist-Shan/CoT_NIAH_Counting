#!/usr/bin/env python3
"""Freeze exact common-support terminal grammar panels for both models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_realistic_niah_v5_cross_model_aligned_anchor_panel import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    _index_rows,
    _transition_key,
)


def _timing(grammar_class: str) -> str | None:
    if "rank_after_city" in grammar_class:
        return "rank_after_city"
    if "rank_before_city" in grammar_class:
        return "rank_before_city"
    return None


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_aligned_grammar_panels(
    qwen_rows: Iterable[dict[str, Any]],
    gemma_rows: Iterable[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    """Choose highest-count common transitions with identical timing strata."""

    qwen = _index_rows(qwen_rows, source_name="Qwen grammar registry")
    gemma = _index_rows(gemma_rows, source_name="Gemma grammar registry")
    common = set(qwen) & set(gemma)
    selected: list[tuple[int, int, int, int]] = []
    excluded_seeds: list[int] = []
    for seed in DEVELOPMENT_SEEDS + CONFIRMATION_SEEDS:
        candidates = []
        for key in common:
            if int(key[0]) != int(seed):
                continue
            if int(key[3]) != int(key[1]) or int(key[2]) != int(key[1]) - 1:
                continue
            q_timing = _timing(str(qwen[key].get("target_grammar_class", "")))
            g_timing = _timing(str(gemma[key].get("target_grammar_class", "")))
            if q_timing is not None and q_timing == g_timing:
                candidates.append(key)
        candidates.sort(key=lambda key: (-int(key[1]), int(key[2]), int(key[3])))
        if candidates:
            selected.append(candidates[0])
        else:
            excluded_seeds.append(int(seed))

    if not selected:
        raise ValueError("No same-timing common grammar transitions")
    shared: list[dict[str, Any]] = []
    panels: dict[str, list[dict[str, Any]]] = {"Qwen3-8B": [], "Gemma4-E4B": []}
    for panel_index, key in enumerate(selected):
        seed, gold_count, from_occurrence, to_occurrence = key
        phase = "development" if seed in DEVELOPMENT_SEEDS else "confirmation"
        timing = _timing(str(qwen[key]["target_grammar_class"]))
        assert timing is not None
        pair_id = f"grammar_seed{seed}_N{gold_count}_{from_occurrence}to{to_occurrence}"
        shared.append(
            {
                "alignment_pair_id": pair_id,
                "panel_index": panel_index,
                "phase": phase,
                "seed": seed,
                "gold_count": gold_count,
                "from_occurrence": from_occurrence,
                "to_occurrence": to_occurrence,
                "timing_stratum": timing,
                "qwen_request_id": str(qwen[key]["request_id"]),
                "gemma_request_id": str(gemma[key]["request_id"]),
                "qwen_grammar_class": str(qwen[key]["target_grammar_class"]),
                "gemma_grammar_class": str(gemma[key]["target_grammar_class"]),
                "outcome_blind": True,
                "selection_rank_used": False,
            }
        )
        for label, source in (("Qwen3-8B", qwen), ("Gemma4-E4B", gemma)):
            row = dict(source[key])
            row.update(
                {
                    "alignment_pair_id": pair_id,
                    "alignment_phase": phase,
                    "alignment_key": list(key),
                    "grammar_span_outcome_blind": True,
                    "grammar_span_selection_rank_used": False,
                    "grammar_span_timing_stratum": timing,
                    "grammar_span_selection_rule": (
                        "same_timing_common_terminal_transition_then_highest_count"
                    ),
                    "stratified_ncc_outcome_blind": True,
                    "stratified_ncc_selection_rank_used": False,
                    "stratified_ncc_seed_role": phase,
                    "cross_model_exact_sample_alignment": True,
                }
            )
            panels[label].append(row)

    q_keys = [_transition_key(row) for row in panels["Qwen3-8B"]]
    g_keys = [_transition_key(row) for row in panels["Gemma4-E4B"]]
    if q_keys != g_keys or q_keys != selected:
        raise RuntimeError("Grammar panels are not exactly key aligned")

    phase_seeds = {
        "development": [int(key[0]) for key in selected if key[0] in DEVELOPMENT_SEEDS],
        "confirmation": [int(key[0]) for key in selected if key[0] in CONFIRMATION_SEEDS],
    }
    timing_counts = {
        phase: {
            timing: sum(
                row["phase"] == phase and row["timing_stratum"] == timing
                for row in shared
            )
            for timing in ("rank_after_city", "rank_before_city")
        }
        for phase in ("development", "confirmation")
    }
    shared_core = {
        "schema_version": "realistic_niah_v5_cross_model_aligned_grammar_panel_v1",
        "status": "FROZEN_OUTCOME_BLIND_COMMON_SAME_TIMING_SUPPORT",
        "alignment_key_fields": [
            "seed",
            "gold_count",
            "from_occurrence",
            "to_occurrence",
            "timing_stratum",
        ],
        "exact_cross_model_key_equality": True,
        "phase_seeds": phase_seeds,
        "excluded_no_common_timing_seeds": excluded_seeds,
        "timing_counts_by_phase": timing_counts,
        "selection_rule": (
            "same_timing_common_terminal_transition_then_highest_gold_count_"
            "then_lexicographic_transition"
        ),
        "outcome_blind": True,
        "selection_rank_used": False,
        "selected_alignment_keys": [list(key) for key in selected],
        "shared_panel_sha256": _sha256_json(shared),
    }
    manifests: dict[str, dict[str, Any]] = {}
    for label in panels:
        core = {
            **shared_core,
            "model_label": label,
            "panel_sha256": _sha256_json(panels[label]),
        }
        manifests[label] = {**core, "manifest_sha256": _sha256_json(core)}
    return (
        shared,
        panels["Qwen3-8B"],
        panels["Gemma4-E4B"],
        manifests["Qwen3-8B"],
        manifests["Gemma4-E4B"],
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Existing frozen output changed: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _jsonl(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-registry", type=Path, required=True)
    parser.add_argument("--gemma-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    shared, qwen, gemma, q_manifest, g_manifest = build_aligned_grammar_panels(
        _read_jsonl(args.qwen_registry), _read_jsonl(args.gemma_registry)
    )
    outputs = {
        "shared_alignment_keys.jsonl": _jsonl(shared),
        "Qwen3-8B_anchor_panel.jsonl": _jsonl(qwen),
        "Gemma4-E4B_anchor_panel.jsonl": _jsonl(gemma),
        "Qwen3-8B_manifest.json": json.dumps(q_manifest, indent=2, sort_keys=True) + "\n",
        "Gemma4-E4B_manifest.json": json.dumps(g_manifest, indent=2, sort_keys=True) + "\n",
    }
    for label, rows in (("Qwen3-8B", qwen), ("Gemma4-E4B", gemma)):
        for timing in ("rank_after_city", "rank_before_city"):
            outputs[f"{label}_{timing}_panel.jsonl"] = _jsonl(
                [
                    row
                    for row in rows
                    if str(row["grammar_span_timing_stratum"]) == timing
                ]
            )
    for name, text in outputs.items():
        _atomic(args.output_dir / name, text)
    print(
        json.dumps(
            {
                "status": q_manifest["status"],
                "phase_seeds": q_manifest["phase_seeds"],
                "excluded_no_common_timing_seeds": q_manifest[
                    "excluded_no_common_timing_seeds"
                ],
                "shared_panel_sha256": q_manifest["shared_panel_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
