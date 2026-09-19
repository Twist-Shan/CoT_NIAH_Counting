#!/usr/bin/env python3
"""Run the frozen 30-seed old-HTML explicit-progress trace patch panel."""

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

from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
    _model,
    _prepare_shards,
    _safe_stem,
)
from realistic_niah_v5.indexed_counter_patch import (  # noqa: E402
    audit_original_explicit_progress_row,
    minimal_terminal_suffix_token_ids,
    run_indexed_counter_early_stop_patch_trials,
)
from realistic_niah_v5.pipeline import read_jsonl  # noqa: E402


DISCOVERY_SEEDS = tuple(range(1234, 1254))
CONFIRMATION_SEEDS = tuple(range(1254, 1264))
DISCOVERY_LAYERS = {
    "Qwen3-8B": tuple(range(0, 33, 4)),
    "Gemma4-E4B": tuple(range(0, 41, 4)),
}


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    args = parser.parse_args()
    args.command = "indexed-counter-early-stop-patch"

    expected_seed_order = tuple(int(value) for value in args.expected_seeds)
    registered_seeds = (
        DISCOVERY_SEEDS if args.phase == "discovery" else CONFIRMATION_SEEDS
    )
    if expected_seed_order != registered_seeds:
        raise ValueError(f"{args.phase} seed contract changed")
    rows = [
        row
        for row in read_jsonl(args.generations)
        if str(row.get("model_label")) == str(args.model)
        and int(row.get("seed", -1)) in set(expected_seed_order)
        and int(row.get("gold_count", -1)) == 10
    ]
    if len(rows) != len(expected_seed_order) or {
        int(row["seed"]) for row in rows
    } != set(expected_seed_order):
        raise ValueError("Indexed patch needs one frozen N=10 row per formal seed")
    rows.sort(key=lambda row: expected_seed_order.index(int(row["seed"])))

    layers = tuple(sorted({int(value) for value in args.source_layers}))
    if args.phase == "discovery" and layers != DISCOVERY_LAYERS[str(args.model)]:
        raise ValueError("Indexed patch discovery layer ladder changed")
    if args.phase == "confirmation" and len(layers) != 1:
        raise ValueError("Indexed patch confirmation needs one discovery-frozen layer")

    row_plan: list[dict[str, Any]] = []
    for row in rows:
        audit = audit_original_explicit_progress_row(row)
        if not audit["eligible"]:
            raise ValueError(f"Frozen indexed row failed geometry audit: {audit['reasons']}")
        target_occurrences = list(range(1, min(10, int(audit["parsed_item_count"])) + 1))
        row_plan.append(
            {
                "seed": int(row["seed"]),
                "request_id": str(row["request_id"]),
                "request_id_sha256": hashlib.sha256(
                    str(row["request_id"]).encode("utf-8")
                ).hexdigest(),
                "raw_output_text_sha256": hashlib.sha256(
                    str(row["raw_output_text"]).encode("utf-8")
                ).hexdigest(),
                "prompt_token_ids_sha256": _sha256_json(row.get("input_ids", ())),
                "marker_kind": str(audit["marker_kind"]),
                "trace_one_to_one": bool(audit["trace_one_to_one"]),
                "parsed_item_count": int(audit["parsed_item_count"]),
                "target_occurrences": target_occurrences,
            }
        )
    plan = {
        "schema_version": "realistic_niah_v5_indexed_counter_patch_plan_v2",
        "model_label": str(args.model),
        "phase": str(args.phase),
        "seeds": list(expected_seed_order),
        "seed_count": len(expected_seed_order),
        "source_gold_count": 10,
        "source_layers": list(layers),
        "registered_target_occurrences": list(range(1, 11)),
        "available_span_policy": "use contiguous parser-registered 1..M episode",
        "patch_geometry": "full_trace_item_same_position",
        "patch_layer_mode": "single_decoder_block_input",
        "upper_layers_recomputed_after_patch": True,
        "conditions": [
            "clean_early_stop_reference",
            "corrupt_early_stop_reference",
            "clean_item_restore_into_corrupt",
            "corrupt_item_ablate_into_clean",
        ],
        "readout_mode": "immediate_item_k_minimal_native_terminal_suffix",
        "natural_recap_removed": True,
        "prompt_modified": False,
        "source_prompt_is_frozen_original": True,
        "visible_progress_confound_allowed": True,
        "internal_counter_without_visible_index_claim_allowed": False,
        "outcome_blind": True,
        "selection_rank_used": False,
        "rows": row_plan,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "frozen_row_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing indexed patch plan changed")
    else:
        _atomic_json(plan_path, plan)

    model, tokenizer, adapter = _model(args)
    if layers[-1] >= int(adapter.num_layers):
        raise ValueError("Indexed patch layer ladder exceeds the loaded model")
    # Preflight every frozen row before any new shard is written.  This catches
    # model-family-specific final-channel recaps (notably Gemma's occasional
    # prose between <channel|> and Total:) at phase start instead of mid-run.
    for row in rows:
        minimal_terminal_suffix_token_ids(row, tokenizer)
    shards = _prepare_shards(args.output, resume=bool(args.resume), suffix="jsonl")
    started = time.perf_counter()
    completed = skipped = 0
    for index, (row, registered) in enumerate(zip(rows, row_plan), start=1):
        shard = shards / f"{_safe_stem(row['request_id'], 'indexed_patch')}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            continue
        results = run_indexed_counter_early_stop_patch_trials(
            model,
            tokenizer,
            adapter,
            row,
            source_layers=layers,
            target_occurrences=tuple(registered["target_occurrences"]),
            random_seed=20260823 + int(row["seed"]),
        )
        for result in results:
            result.update(
                {
                    "phase": str(args.phase),
                    "row_plan": str(plan_path.resolve()),
                    "row_plan_sha256": _sha256_json(registered),
                }
            )
        _atomic_jsonl(shard, results)
        completed += 1
        print(
            f"[indexed-counter-patch] {index}/{len(rows)} "
            f"seed={row['seed']} targets={registered['target_occurrences']}",
            flush=True,
        )
    observed_shards = len(list(shards.glob("*.jsonl")))
    if observed_shards != len(expected_seed_order):
        raise RuntimeError("Indexed patch did not seal one shard per frozen seed")
    _atomic_json(
        args.output / "manifest.json",
        {
            "schema_version": "realistic_niah_v5_indexed_counter_patch_manifest_v2",
            "status": "PASS",
            "model_label": str(args.model),
            "phase": str(args.phase),
            "seed_count": len(expected_seed_order),
            "source_layers": list(layers),
            "patch_layer_mode": "single_decoder_block_input",
            "upper_layers_recomputed_after_patch": True,
            "completed_shards": observed_shards,
            "newly_completed": completed,
            "resume_skipped": skipped,
            "elapsed_seconds": time.perf_counter() - started,
            "prompt_modified": False,
            "outcome_blind": True,
            "selection_rank_used": False,
        },
    )


if __name__ == "__main__":
    main()
