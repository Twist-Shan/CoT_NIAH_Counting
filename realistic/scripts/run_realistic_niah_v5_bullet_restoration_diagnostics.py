#!/usr/bin/env python3
"""Run small-sample marker-scrubbed list restoration diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v5.bullet_restoration_diagnostics import (  # noqa: E402
    run_bullet_restoration_diagnostics,
)
from scripts.run_realistic_niah_v5_bullet_counterfactual_restore import (  # noqa: E402
    _selected_rows,
)
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
    _model,
    _safe_stem,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--source-layer", type=int, required=True)
    parser.add_argument("--target-occurrences", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.command = "bullet-restoration-diagnostics"

    confirmation_rows, _registered, _cohort = _selected_rows(
        generations=args.generations,
        cohort_manifest=args.cohort_manifest,
        model_label=str(args.model),
        phase="confirmation",
    )
    requested_seeds = tuple(int(value) for value in args.seeds)
    selected = [row for row in confirmation_rows if int(row["seed"]) in requested_seeds]
    if {int(row["seed"]) for row in selected} != set(requested_seeds):
        raise ValueError("Every diagnostic seed must belong to frozen confirmation")
    targets = tuple(sorted({int(value) for value in args.target_occurrences}))
    plan = {
        "schema_version": "realistic_niah_v5_bullet_restore_diagnostic_plan_v1",
        "status": "DIAGNOSTIC_ONLY",
        "model_label": str(args.model),
        "seeds": requested_seeds,
        "source_layer": int(args.source_layer),
        "target_occurrences": targets,
        "conditions": [
            "source_reference",
            "blank_reference",
            "single_item_once",
            "single_item_once_cache_aligned",
            "all_items_once",
            "single_item_cumulative_clamp",
            "full_prefix_identity",
        ],
        "formal_confirmation_unchanged": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output / "diagnostic_plan.json", plan)
    shards = args.output / "shards"
    shards.mkdir(parents=True, exist_ok=True)

    model, tokenizer, adapter = _model(args)
    started = time.perf_counter()
    completed = skipped = 0
    for index, row in enumerate(selected, start=1):
        shard = shards / f"{_safe_stem(row['request_id'], 'bullet_diagnostic')}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            print(
                f"[bullet-diagnostic] {index}/{len(selected)} seed={row['seed']} resume-skip",
                flush=True,
            )
            continue
        rows = run_bullet_restoration_diagnostics(
            model,
            tokenizer,
            adapter,
            row,
            source_layer=int(args.source_layer),
            target_occurrences=targets,
            random_seed=20260824 + int(row["seed"]),
        )
        _atomic_jsonl(shard, rows)
        completed += 1
        print(
            f"[bullet-diagnostic] {index}/{len(selected)} seed={row['seed']} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    observed = len(list(shards.glob("*.jsonl")))
    if observed != len(selected):
        raise RuntimeError("Diagnostic did not produce one shard per selected seed")
    _atomic_json(
        args.output / "manifest.json",
        {
            **plan,
            "status": "PASS",
            "completed_shards": observed,
            "newly_completed": completed,
            "resume_skipped": skipped,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    print(json.dumps(json.loads((args.output / "manifest.json").read_text())), flush=True)


if __name__ == "__main__":
    main()
