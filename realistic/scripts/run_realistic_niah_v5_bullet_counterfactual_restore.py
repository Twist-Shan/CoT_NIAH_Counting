#!/usr/bin/env python3
"""Run discovery or three-layer marker-scrubbed list restoration trials."""

from __future__ import annotations

import argparse
import hashlib
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

from realistic_niah_v5.bullet_counterfactual_restore import (  # noqa: E402
    audit_complete_marker_scrubbable_list,
    run_bullet_counterfactual_restore_trials,
)
from realistic_niah_v5.pipeline import read_jsonl  # noqa: E402
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
    _model,
    _prepare_shards,
    _safe_stem,
)


DISCOVERY_LAYERS = {
    "Qwen3-8B": tuple(range(0, 33, 4)),
    "Gemma4-E4B": tuple(range(0, 41, 4)),
}
PHASE_SIZES = {"discovery": 20, "confirmation": 10}


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _selected_rows(
    *,
    generations: Path,
    cohort_manifest: Path,
    model_label: str,
    phase: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cohort = json.loads(cohort_manifest.read_text(encoding="utf-8"))
    if (
        cohort.get("status") != "FROZEN"
        or str(cohort.get("model_label")) != str(model_label)
        or int(cohort.get("selected_seed_count", -1)) != 30
        or bool(cohort.get("selection_uses_final_answer", True))
        or bool(cohort.get("patch_outcomes_accessed", True))
    ):
        raise ValueError("Cohort manifest is not a frozen outcome-blind 30-seed registry")
    selected = [
        dict(value)
        for value in cohort.get("rows", ())
        if str(value.get("cohort_role")) == str(phase)
    ]
    expected = PHASE_SIZES[str(phase)]
    if len(selected) != expected:
        raise ValueError(f"Frozen {phase} registry does not contain {expected} rows")
    all_rows = {
        str(row.get("request_id")): row
        for row in read_jsonl(generations)
        if str(row.get("model_label")) == str(model_label)
        and int(row.get("gold_count", -1)) == 10
    }
    rows: list[dict[str, Any]] = []
    for registered in selected:
        request_id = str(registered["request_id"])
        row = all_rows.get(request_id)
        if row is None:
            raise ValueError(f"Frozen generation is missing: {request_id}")
        identity = {
            "input_ids_sha256": _sha256_json(row.get("input_ids", ())),
            "output_token_ids_sha256": _sha256_json(row.get("output_token_ids", ())),
            "raw_output_text_sha256": hashlib.sha256(
                str(row.get("raw_output_text", "")).encode("utf-8")
            ).hexdigest(),
        }
        expected_identity = {
            key: str(registered[key]) for key in identity
        }
        if identity != expected_identity:
            raise ValueError(f"Frozen generation identity changed: {request_id}")
        audit = audit_complete_marker_scrubbable_list(row)
        if not audit["eligible"]:
            raise ValueError(
                f"Frozen cohort row no longer passes list audit: {request_id}: "
                f"{audit['reasons']}"
            )
        rows.append(row)
    return rows, selected, cohort


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.command = "bullet-counterfactual-restore"

    rows, registered_rows, cohort = _selected_rows(
        generations=args.generations,
        cohort_manifest=args.cohort_manifest,
        model_label=str(args.model),
        phase=str(args.phase),
    )
    layers = tuple(sorted({int(value) for value in args.source_layers}))
    if args.phase == "discovery":
        if layers != DISCOVERY_LAYERS[str(args.model)]:
            raise ValueError("Discovery layer ladder changed")
    elif len(layers) != 3:
        raise ValueError("Confirmation must retain exactly three discovery-frozen layers")

    plan_rows = []
    for row, registered in zip(rows, registered_rows):
        plan_rows.append(
            {
                "selection_rank": int(registered["selection_rank"]),
                "cohort_role": str(registered["cohort_role"]),
                "seed": int(row["seed"]),
                "request_id": str(row["request_id"]),
                "request_id_sha256": hashlib.sha256(
                    str(row["request_id"]).encode("utf-8")
                ).hexdigest(),
                "input_ids_sha256": _sha256_json(row.get("input_ids", ())),
                "output_token_ids_sha256": _sha256_json(
                    row.get("output_token_ids", ())
                ),
            }
        )
    plan = {
        "schema_version": "realistic_niah_v5_marker_scrubbed_list_restore_plan_v2",
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
        "base_scrub": [
            "prompt_needle_records",
            "all_nonitem_assistant_reasoning_tokens",
            "explicit_within_item_progress_markers",
        ],
        "blank_additional_scrub": "visible_list_items_1_through_k",
        "scrub_geometry": "same_position_equal_token_count_replacement",
        "patch_geometry": "complete_source_list_item_k_same_token_positions",
        "patch_layer_mode": "one_decoder_block_input_once",
        "upper_layers_recomputed": True,
        "readout": "immediate_minimal_native_Total_query",
        "diagnostic_suffix_used": False,
        "candidate_counts": list(range(1, 11)),
        "candidate_score": "joint_sequence_log_probability_with_termination",
        "cohort_manifest_sha256": hashlib.sha256(
            args.cohort_manifest.read_bytes()
        ).hexdigest(),
        "cohort_last_selected_seed": int(cohort["last_selected_seed"]),
        "outcome_blind": True,
        "rows": plan_rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "frozen_trial_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing bullet restoration trial plan changed")
    else:
        _atomic_json(plan_path, plan)

    model, tokenizer, adapter = _model(args)
    if layers[-1] >= int(adapter.num_layers):
        raise ValueError("Registered source layer exceeds the loaded model")
    shards = _prepare_shards(args.output, resume=bool(args.resume), suffix="jsonl")
    started = time.perf_counter()
    completed = skipped = 0
    for index, row in enumerate(rows, start=1):
        shard = shards / f"{_safe_stem(row['request_id'], 'bullet_restore')}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            print(
                f"[bullet-restore] {index}/{len(rows)} seed={row['seed']} resume-skip",
                flush=True,
            )
            continue
        results = run_bullet_counterfactual_restore_trials(
            model,
            tokenizer,
            adapter,
            row,
            source_layers=layers,
            target_occurrences=tuple(range(1, 11)),
            random_seed=20260824 + int(row["seed"]),
        )
        for result in results:
            result.update(
                {
                    "phase": str(args.phase),
                    "cohort_role": str(args.phase),
                    "source_stimulus_original_split": str(row.get("split", "")),
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
            f"[bullet-restore] {index}/{len(rows)} seed={row['seed']} "
            f"layers={list(layers)} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    observed_shards = len(list(shards.glob("*.jsonl")))
    if observed_shards != len(rows):
        raise RuntimeError("Bullet restoration did not seal one shard per frozen seed")
    _atomic_json(
        args.output / "manifest.json",
        {
            "schema_version": "realistic_niah_v5_bullet_counter_restore_manifest_v1",
            "status": "PASS",
            "model_label": str(args.model),
            "phase": str(args.phase),
            "seed_count": len(rows),
            "source_layers": list(layers),
            "target_occurrences": list(range(1, 11)),
            "conditions": plan["conditions"],
            "completed_shards": observed_shards,
            "newly_completed": completed,
            "resume_skipped": skipped,
            "elapsed_seconds": time.perf_counter() - started,
            "prompt_modified": False,
            "outcome_blind": True,
        },
    )


if __name__ == "__main__":
    main()
