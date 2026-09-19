#!/usr/bin/env python3
"""Run same-position old-HTML restoration on no-index native traces."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
    _model,
    _prepare_shards,
    _runtime_manifest,
    _safe_stem,
)
from realistic_niah_v5.unnumbered_counter_restore import (  # noqa: E402
    audit_no_count_enumeration_trace,
    audit_unnumbered_trace,
    run_unnumbered_counter_restore_trials,
)
from realistic_niah_v5.pipeline import read_jsonl  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    # The archived passages can contain U+2028 inside a valid JSON string;
    # str.splitlines() would incorrectly treat it as a JSONL record boundary.
    return read_jsonl(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--phase", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--expected-seeds", type=int, nargs="+")
    parser.add_argument("--source-layers", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.command = "unnumbered-counter-restore"

    default_seeds = (
        tuple(range(1234, 1254))
        if args.phase == "discovery"
        else tuple(range(1254, 1264))
    )
    expected_seed_order = (
        tuple(int(value) for value in args.expected_seeds)
        if args.expected_seeds
        else default_seeds
    )
    expected_count = 20 if args.phase == "discovery" else 10
    if (
        len(expected_seed_order) != expected_count
        or len(set(expected_seed_order)) != expected_count
    ):
        raise ValueError(f"{args.phase} unnumbered restore requires {expected_count} unique seeds")
    expected_seeds = set(expected_seed_order)
    rows = [
        row
        for row in _read_jsonl(args.generations)
        if str(row.get("model_label")) == str(args.model)
        and int(row.get("seed", -1)) in expected_seeds
    ]
    if len(rows) != len(expected_seeds) or {int(row["seed"]) for row in rows} != expected_seeds:
        raise ValueError("Unnumbered restore seed contract changed")
    natural_panel = all(
        row.get("natural_unnumbered_teacher_forced") is False for row in rows
    )
    teacher_forced_panel = all(
        str(row.get("counterfactual_trace_kind"))
        == "teacher_forced_unnumbered_gold_bullets"
        for row in rows
    )
    if natural_panel == teacher_forced_panel:
        raise ValueError("Unnumbered restore requires one homogeneous registered panel")
    for row in rows:
        audit = (
            audit_no_count_enumeration_trace(row)
            if natural_panel
            else audit_unnumbered_trace(row)
        )
        if not audit["eligible"]:
            raise ValueError(f"Unnumbered row became ineligible: {audit['reasons']}")
    rows.sort(key=lambda row: int(row["seed"]))
    layers = tuple(sorted({int(value) for value in args.source_layers}))
    expected_layers = {
        "Qwen3-8B": (18, 22, 26, 30),
        "Gemma4-E4B": (16, 20, 24, 28, 32, 36),
    }
    if args.phase == "discovery" and layers != expected_layers[str(args.model)]:
        raise ValueError("Unnumbered discovery layer ladder changed")
    if args.phase == "confirmation" and len(layers) != 1:
        raise ValueError("Unnumbered confirmation must use one discovery-frozen layer")
    plan = {
        "schema_version": "realistic_niah_v5_unnumbered_restore_plan_v3",
        "model_label": str(args.model),
        "phase": str(args.phase),
        "seeds": list(expected_seed_order),
        "seed_count": len(expected_seeds),
        "source_layers": list(layers),
        "target_occurrences": list(range(2, 10)),
        "patch_geometry": "full_trace_item_same_position",
        "receiver": "all_prompt_and_trace_needles_same_length_background_replaced",
        "trace_running_index": "absent_in_each_item_causal_prefix_by_format_audit",
        "trace_panel_kind": (
            "model_generated_no_count_enumeration"
            if natural_panel
            else "teacher_forced_unnumbered_gold_bullets"
        ),
        "natural_generation_claim_allowed": natural_panel,
        "trace_tokens_teacher_forced": not natural_panel,
        "controlled_hidden_state_sufficiency_claim_allowed": True,
        "outcome_blind": True,
        "selection_rank_used": False,
    }
    plan_path = args.output / "frozen_row_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing unnumbered restore plan changed")
    else:
        _atomic_json(plan_path, plan)

    model, tokenizer, adapter = _model(args)
    shards = _prepare_shards(args.output, resume=bool(args.resume), suffix="jsonl")
    started = time.perf_counter()
    completed = skipped = 0
    for index, row in enumerate(rows, start=1):
        shard = shards / f"{_safe_stem(row['request_id'], 'unnumbered_restore')}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            continue
        results = run_unnumbered_counter_restore_trials(
            model,
            tokenizer,
            adapter,
            row,
            source_layers=layers,
            target_occurrences=tuple(range(2, 10)),
            random_seed=20260823 + int(row["seed"]),
        )
        for result in results:
            result.update(
                {
                    "phase": str(args.phase),
                    "row_plan": str(plan_path.resolve()),
                }
            )
        _atomic_jsonl(shard, results)
        completed += 1
        print(f"[unnumbered-restore] {index}/{len(rows)}", flush=True)
    _atomic_json(
        args.output / "manifest.json",
        {
            "schema_version": "realistic_niah_v5_unnumbered_restore_manifest_v1",
            "status": "PASS",
            "model_label": str(args.model),
            "phase": str(args.phase),
            "seed_count": len(expected_seeds),
            "source_layers": list(layers),
            "target_occurrences": list(range(2, 10)),
            "completed_shards": len(list(shards.glob("*.jsonl"))),
            "newly_completed": completed,
            "resume_skipped": skipped,
            "elapsed_seconds": time.perf_counter() - started,
            "outcome_blind": True,
            "selection_rank_used": False,
        },
    )


if __name__ == "__main__":
    main()
