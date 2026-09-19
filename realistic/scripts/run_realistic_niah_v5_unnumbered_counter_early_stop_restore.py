#!/usr/bin/env python3
"""Run old-HTML cumulative full-item restoration with immediate item-k stop."""

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
    _safe_stem,
)
from realistic_niah_v5.pipeline import read_jsonl  # noqa: E402
from realistic_niah_v5.unnumbered_counter_restore import (  # noqa: E402
    audit_no_count_enumeration_trace,
    run_unnumbered_counter_early_stop_restore_trials,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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
    parser.add_argument("--expected-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--source-layers", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-prompt-conditioned-a7-auxiliary", action="store_true")
    args = parser.parse_args()
    args.command = "unnumbered-counter-early-stop-restore"

    expected_seed_order = tuple(int(value) for value in args.expected_seeds)
    expected_count = 20 if args.phase == "discovery" else 10
    if (
        len(expected_seed_order) != expected_count
        or len(set(expected_seed_order)) != expected_count
    ):
        raise ValueError(f"{args.phase} early-stop restore requires {expected_count} unique seeds")
    expected_seeds = set(expected_seed_order)
    rows = [
        row
        for row in _read_jsonl(args.generations)
        if str(row.get("model_label")) == str(args.model)
        and int(row.get("seed", -1)) in expected_seeds
    ]
    if len(rows) != expected_count or {int(row["seed"]) for row in rows} != expected_seeds:
        raise ValueError("Early-stop restore seed contract changed")
    prompt_conditioned = any(
        row.get("natural_unnumbered_attempt") is not None
        or str(row.get("natural_unnumbered_fixed_reasoning_prefix", ""))
        or "_NATURAL_UNNUMBERED_" in str(row.get("stimulus_id", ""))
        for row in rows
    )
    if prompt_conditioned != bool(args.allow_prompt_conditioned_a7_auxiliary):
        raise ValueError(
            "A7 auxiliary provenance must match --allow-prompt-conditioned-a7-auxiliary"
        )
    for row in rows:
        if prompt_conditioned:
            if (
                int(row.get("natural_unnumbered_attempt", -1)) != 7
                or str(row.get("natural_unnumbered_fixed_reasoning_prefix", "")) != "- "
                or not str(row.get("stimulus_id", "")).endswith("_NATURAL_UNNUMBERED_A7")
            ):
                raise ValueError("A7 auxiliary cohort contains a non-A7 trace")
        elif (
            row.get("natural_unnumbered_attempt") is not None
            or str(row.get("natural_unnumbered_fixed_reasoning_prefix", ""))
            or "_NATURAL_UNNUMBERED_" in str(row.get("stimulus_id", ""))
        ):
            raise ValueError(
                "PROMPT_INTEGRITY_FAILURE: prompt-conditioned no-enumeration "
                "banks are forbidden; use byte-identical frozen prompts"
            )
        if row.get("natural_unnumbered_teacher_forced") is not False:
            raise ValueError("Early-stop restore requires model-generated traces")
        audit = audit_no_count_enumeration_trace(row)
        if not audit["eligible"] or str(audit["marker_kind"]) != "bullet":
            raise ValueError(f"Early-stop row became ineligible: {audit['reasons']}")
    rows.sort(key=lambda row: expected_seed_order.index(int(row["seed"])))
    layers = tuple(sorted({int(value) for value in args.source_layers}))
    expected_layers = {
        "Qwen3-8B": (18, 22, 26, 30),
        "Gemma4-E4B": (16, 20, 24, 28, 32, 36),
    }
    if args.phase == "discovery" and layers != expected_layers[str(args.model)]:
        raise ValueError("Early-stop discovery layer ladder changed")
    if args.phase == "confirmation" and len(layers) != 1:
        raise ValueError("Early-stop confirmation must use one discovery-frozen layer")
    plan = {
        "schema_version": "realistic_niah_v5_unnumbered_early_stop_restore_plan_v1",
        "model_label": str(args.model),
        "phase": str(args.phase),
        "seeds": list(expected_seed_order),
        "seed_count": expected_count,
        "source_layers": list(layers),
        "target_occurrences": list(range(2, 10)),
        "patch_geometry": "full_trace_item_same_position",
        "patch_layer_mode": "cumulative_clamp_source_through_last",
        "readout_mode": "immediate_item_k_early_stop_minimal_terminal_suffix",
        "future_trace_items_removed": True,
        "terminal_suffix_contains_count_information": False,
        "trace_panel_kind": "model_generated_no_count_enumeration",
        "outcome_blind": True,
        "selection_rank_used": False,
        "prompt_conditioned_a7_auxiliary": prompt_conditioned,
        "formal_frozen_prompt_claim_allowed": not prompt_conditioned,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "frozen_row_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing early-stop restore plan changed")
    else:
        _atomic_json(plan_path, plan)

    model, tokenizer, adapter = _model(args)
    shards = _prepare_shards(args.output, resume=bool(args.resume), suffix="jsonl")
    started = time.perf_counter()
    completed = skipped = 0
    for index, row in enumerate(rows, start=1):
        shard = shards / f"{_safe_stem(row['request_id'], 'early_stop_restore')}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            continue
        results = run_unnumbered_counter_early_stop_restore_trials(
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
                    "prompt_conditioned_a7_auxiliary": prompt_conditioned,
                    "formal_frozen_prompt_claim_allowed": not prompt_conditioned,
                    "natural_generation_claim_allowed": not prompt_conditioned,
                }
            )
        _atomic_jsonl(shard, results)
        completed += 1
        print(f"[early-stop-restore] {index}/{len(rows)}", flush=True)
    _atomic_json(
        args.output / "manifest.json",
        {
            "schema_version": "realistic_niah_v5_unnumbered_early_stop_restore_manifest_v1",
            "status": "PASS",
            "model_label": str(args.model),
            "phase": str(args.phase),
            "seed_count": expected_count,
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
