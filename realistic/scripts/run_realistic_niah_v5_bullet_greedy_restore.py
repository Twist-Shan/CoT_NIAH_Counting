#!/usr/bin/env python3
"""Run marker-scrubbed list restoration with free greedy integer readout."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v5.bullet_greedy_restore import (  # noqa: E402
    run_bullet_greedy_restore_trials,
)
from scripts.run_realistic_niah_v5_bullet_counterfactual_restore import (  # noqa: E402
    DISCOVERY_LAYERS,
    PHASE_SIZES,
    _selected_rows,
    _sha256_json,
)
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
    _model,
    _prepare_shards,
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
    parser.add_argument("--phase", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--source-layers", type=int, nargs="+", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.command = "bullet-greedy-restore"

    rows, registered_rows, cohort = _selected_rows(
        generations=args.generations,
        cohort_manifest=args.cohort_manifest,
        model_label=str(args.model),
        phase=str(args.phase),
    )
    layers = tuple(sorted({int(value) for value in args.source_layers}))
    if args.phase == "discovery":
        if layers != DISCOVERY_LAYERS[str(args.model)]:
            raise ValueError("Greedy discovery layer ladder changed")
    elif len(layers) != 3:
        raise ValueError("Greedy confirmation must use three frozen layers")
    if len(rows) != PHASE_SIZES[str(args.phase)]:
        raise ValueError("Greedy phase seed count changed")

    plan_rows = [
        {
            "selection_rank": int(registered["selection_rank"]),
            "seed": int(row["seed"]),
            "request_id": str(row["request_id"]),
            "request_id_sha256": hashlib.sha256(
                str(row["request_id"]).encode("utf-8")
            ).hexdigest(),
        }
        for row, registered in zip(rows, registered_rows)
    ]
    plan = {
        "schema_version": "realistic_niah_v5_marker_scrubbed_greedy_plan_v1",
        "model_label": str(args.model),
        "phase": str(args.phase),
        "seed_count": len(rows),
        "source_layers": list(layers),
        "target_occurrences": list(range(1, 11)),
        "conditions": [
            "source_reference",
            "blank_reference",
            "source_list_item_k_to_blank_restoration",
        ],
        "primary_outcome": "free_greedy_generated_integer_equals_target_k",
        "layer_selection": (
            "descending_seed_equal_mean_restored_exact_minus_blank_exact; "
            "ties descending restored exact then ascending layer"
        ),
        "max_new_tokens": int(args.max_new_tokens),
        "candidate_scoring_used": False,
        "diagnostic_suffix_used": False,
        "formal_status": "corrected_readout_rerun_same_frozen_cohort",
        "independent_new_confirmation_claim_allowed": False,
        "cohort_last_selected_seed": int(cohort["last_selected_seed"]),
        "outcome_blind_format_selection": True,
        "rows": plan_rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "frozen_trial_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing greedy trial plan changed")
    else:
        _atomic_json(plan_path, plan)

    model, tokenizer, adapter = _model(args)
    shards = _prepare_shards(args.output, resume=bool(args.resume), suffix="jsonl")
    started = time.perf_counter()
    completed = skipped = 0
    for index, row in enumerate(rows, start=1):
        shard = shards / f"{_safe_stem(row['request_id'], 'greedy_restore')}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            print(f"[greedy-restore] {index}/{len(rows)} seed={row['seed']} resume-skip", flush=True)
            continue
        results = run_bullet_greedy_restore_trials(
            model,
            tokenizer,
            adapter,
            row,
            source_layers=layers,
            target_occurrences=tuple(range(1, 11)),
            random_seed=20260824 + int(row["seed"]),
            max_new_tokens=int(args.max_new_tokens),
        )
        for result in results:
            result.update(
                {
                    "phase": str(args.phase),
                    "trial_plan_sha256": _sha256_json(plan),
                    "cohort_selection_rank": int(
                        next(
                            value["selection_rank"]
                            for value in plan_rows
                            if value["request_id"] == row["request_id"]
                        )
                    ),
                }
            )
        _atomic_jsonl(shard, results)
        completed += 1
        print(
            f"[greedy-restore] {index}/{len(rows)} seed={row['seed']} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    observed = len(list(shards.glob("*.jsonl")))
    if observed != len(rows):
        raise RuntimeError("Greedy restoration did not seal one shard per seed")
    _atomic_json(
        args.output / "manifest.json",
        {
            "schema_version": "realistic_niah_v5_marker_scrubbed_greedy_manifest_v1",
            "status": "PASS",
            "model_label": str(args.model),
            "phase": str(args.phase),
            "seed_count": len(rows),
            "source_layers": list(layers),
            "max_new_tokens": int(args.max_new_tokens),
            "completed_shards": observed,
            "newly_completed": completed,
            "resume_skipped": skipped,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
