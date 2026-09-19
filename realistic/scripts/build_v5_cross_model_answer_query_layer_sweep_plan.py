#!/usr/bin/env python3
"""Freeze exactly aligned Qwen/Gemma answer-query donor pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.parsing import parse_trace_record  # noqa: E402


SCHEMA = "realistic_niah_v5_cross_model_answer_query_layer_sweep_plan_v1"
CONFIRMATION_SEEDS = tuple(range(1254, 1264))
MODEL_LAYERS = {
    "Qwen3-8B": [0, 5, 10, 15, 20, 25, 30, 35],
    "Gemma4-E4B": [0, 6, 12, 18, 23, 29, 35, 41],
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _eligible_by_seed_count(
    rows: list[dict[str, Any]], model_label: str
) -> dict[int, dict[int, dict[str, Any]]]:
    indexed: dict[int, dict[int, dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("model_label", row.get("model"))) != model_label:
            continue
        if str(row.get("split")) != "confirmation":
            continue
        parsed = parse_trace_record(row)
        if not bool(parsed["parser"].get("trace_one_to_one")):
            continue
        if not bool(parsed.get("exact_count")):
            continue
        seed = int(row["seed"])
        count = int(parsed["gold_count"])
        if count in indexed.setdefault(seed, {}):
            raise ValueError(f"Duplicate clean-correct row for {model_label} {seed}/N{count}")
        indexed[seed][count] = row
    return indexed


def _selected_edges(common_counts: list[int]) -> list[tuple[int, int]]:
    edges = list(zip(common_counts[:-1], common_counts[1:]))
    if len(edges) < 2:
        raise ValueError(f"Need at least two common edges, got counts={common_counts}")
    selected: list[tuple[int, int]] = []
    for target in (1.5, 9.5):
        remaining = [edge for edge in edges if edge not in selected]
        selected.append(
            min(
                remaining,
                key=lambda edge: (
                    abs(((edge[0] + edge[1]) / 2.0) - target),
                    edge[1] - edge[0],
                    edge[0],
                ),
            )
        )
    return sorted(selected)


def build_cross_model_pairs(
    qwen_rows: list[dict[str, Any]], gemma_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return model-specific pair files with an identical alignment-key multiset."""

    indexed = {
        "Qwen3-8B": _eligible_by_seed_count(qwen_rows, "Qwen3-8B"),
        "Gemma4-E4B": _eligible_by_seed_count(gemma_rows, "Gemma4-E4B"),
    }
    panels: dict[str, list[dict[str, Any]]] = {label: [] for label in indexed}
    alignment_keys: list[list[int]] = []
    common_counts_by_seed: dict[str, list[int]] = {}
    for seed in CONFIRMATION_SEEDS:
        common_counts = sorted(
            set(indexed["Qwen3-8B"].get(seed, {}))
            & set(indexed["Gemma4-E4B"].get(seed, {}))
        )
        common_counts_by_seed[str(seed)] = common_counts
        for lower, higher in _selected_edges(common_counts):
            for receiver, donor, donor_role in (
                (lower, higher, "common_grid_higher"),
                (higher, lower, "common_grid_lower"),
            ):
                alignment_key = [seed, receiver, donor]
                alignment_keys.append(alignment_key)
                alignment_id = f"seed{seed}_R{receiver}_D{donor}"
                for label in panels:
                    receiver_row = indexed[label][seed][receiver]
                    donor_row = indexed[label][seed][donor]
                    panels[label].append(
                        {
                            "schema_version": SCHEMA,
                            "pair_id": f"{label}__{alignment_id}",
                            "alignment_pair_id": alignment_id,
                            "alignment_key": alignment_key,
                            "model_label": label,
                            "seed": seed,
                            "split": "confirmation",
                            "receiver_request_id": str(receiver_row["request_id"]),
                            "donor_request_id": str(donor_row["request_id"]),
                            "receiver_count": receiver,
                            "donor_count": donor,
                            "receiver_site_id": "answer_query_v3",
                            "donor_site_id": "answer_query_v3",
                            "donor_role": donor_role,
                            "pair_direction": (
                                "higher_to_lower" if donor > receiver else "lower_to_higher"
                            ),
                            "receiver_exact_count": True,
                            "donor_exact_count": True,
                            "pair_eligibility": (
                                "cross_model_common_strict_one_to_one_and_both_"
                                "baseline_final_answers_exact"
                            ),
                            "pair_selection_uses_patch_outcome": False,
                            "cross_model_exact_sample_alignment": True,
                        }
                    )
    q_keys = [row["alignment_key"] for row in panels["Qwen3-8B"]]
    g_keys = [row["alignment_key"] for row in panels["Gemma4-E4B"]]
    if q_keys != g_keys or q_keys != alignment_keys:
        raise RuntimeError("Answer-query panels are not exactly aligned")
    if len(alignment_keys) != 40:
        raise RuntimeError(f"Expected 40 directed pairs, got {len(alignment_keys)}")

    core = {
        "schema_version": SCHEMA,
        "status": "FROZEN_OUTCOME_BLIND_EXACT_ALIGNMENT",
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "alignment_key_fields": ["seed", "receiver_count", "donor_count"],
        "alignment_keys": alignment_keys,
        "directed_pairs_per_model": len(alignment_keys),
        "directed_pairs_per_seed": 4,
        "common_counts_by_seed": common_counts_by_seed,
        "pair_policy": (
            "within each seed intersect strict one-to-one clean-correct counts; "
            "select two adjacent-in-common-grid edges nearest low/high anchors; "
            "run both directions"
        ),
        "pair_selection_uses_patch_outcome": False,
        "exact_cross_model_key_equality": True,
        "layers": MODEL_LAYERS,
        "normalized_depth_policy": (
            "eight architecture-specific approximately uniform post-block layers"
        ),
    }
    manifest = {
        **core,
        "qwen_pairs_sha256": _sha256_json(panels["Qwen3-8B"]),
        "gemma_pairs_sha256": _sha256_json(panels["Gemma4-E4B"]),
        "manifest_sha256": _sha256_json(core),
    }
    return panels["Qwen3-8B"], panels["Gemma4-E4B"], manifest


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Existing frozen output changed: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-generations", type=Path, required=True)
    parser.add_argument("--gemma-generations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    qwen, gemma, manifest = build_cross_model_pairs(
        _read_jsonl(args.qwen_generations), _read_jsonl(args.gemma_generations)
    )
    outputs = {
        "Qwen3-8B_pairs.jsonl": _jsonl(qwen),
        "Gemma4-E4B_pairs.jsonl": _jsonl(gemma),
        "alignment_manifest.json": json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    }
    for name, text in outputs.items():
        _atomic(args.output_dir / name, text)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
