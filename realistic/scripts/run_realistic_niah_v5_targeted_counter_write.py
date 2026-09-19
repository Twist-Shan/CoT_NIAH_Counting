#!/usr/bin/env python3
"""Run the frozen teacher-forced targeted-query -> counter-state write assay."""

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
    _registered_rows,
    _runtime_manifest,
    _safe_stem,
    _spec,
)
from scripts.run_realistic_niah_v5_generated_suffix_state_bridge import (  # noqa: E402
    MODEL_CONTRACTS,
    _load_banks,
)
from realistic_niah_v5.targeted_counter_write import (  # noqa: E402
    run_targeted_counter_write_trials,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism-config", type=Path, required=True)
    parser.add_argument("--v5-config", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODEL_CONTRACTS), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument(
        "--seed-role", choices=("development", "confirmation"), required=True
    )
    parser.add_argument("--anchor-registry", type=Path, required=True)
    parser.add_argument("--targeted-registry", type=Path, required=True)
    parser.add_argument("--bank-plan", type=Path, required=True)
    parser.add_argument("--source-layer", type=int, required=True)
    parser.add_argument(
        "--head-ablation-scope",
        choices=("query_local", "query_through_carrier"),
        default="query_local",
    )
    parser.add_argument("--answer-site-id", default="answer_query_v3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.command = "targeted-counter-write"
    args.cohort = "one_to_one"
    args.row_panel = "trace_patch"
    args.limit = None

    contract = MODEL_CONTRACTS[str(args.model)]
    if int(args.source_layer) != int(contract["source_layer"]):
        raise ValueError(f"{args.model} source-layer contract changed")
    mechanism = _spec(args)
    registered = _registered_rows(args, mechanism)
    anchors = _read_jsonl(args.anchor_registry)
    targeted_rows = _read_jsonl(args.targeted_registry)
    if any("selection_rank" in row for row in anchors + targeted_rows):
        raise ValueError("Counter write registries forbid selection_rank")
    anchor_ids = {str(row["request_id"]) for row in anchors}
    targeted = {str(row["request_id"]): row for row in targeted_rows}
    if len(targeted) != len(targeted_rows) or not anchor_ids <= set(targeted):
        raise ValueError("Counter write targeted registry is incomplete or duplicated")
    rows = [row for row in registered if str(row["request_id"]) in anchor_ids]
    expected = 20 if args.seed_role == "development" else 10
    if len(rows) != expected or len({int(row["seed"]) for row in rows}) != expected:
        raise ValueError("Counter write seed contract changed")
    rows.sort(key=lambda row: int(row["seed"]))
    banks = _load_banks(args.bank_plan, model_label=str(args.model))
    selected = next(bank for bank in banks if bank["condition"] == "selected_bank")

    plan_core = {
        "schema_version": "realistic_niah_v5_targeted_counter_write_plan_v1",
        "model_label": str(args.model),
        "seed_role": str(args.seed_role),
        "seeds": [int(row["seed"]) for row in rows],
        "seed_count": expected,
        "selected_request_by_seed": {
            str(row["seed"]): str(row["request_id"]) for row in rows
        },
        "selected_count_by_seed": {
            str(row["seed"]): int(row["gold_count"]) for row in rows
        },
        "anchor_registry_sha256": _sha256(args.anchor_registry),
        "targeted_registry_sha256": _sha256(args.targeted_registry),
        "bank_plan_sha256": _sha256(args.bank_plan),
        "selected_bank_size": len(selected["heads"]),
        "selected_bank_sha256": str(selected["bank_sha256"]),
        "source_layer": int(args.source_layer),
        "teacher_forced_trace_tokens": True,
        "query_local_head_mask": str(args.head_ablation_scope) == "query_local",
        "head_ablation_scope": str(args.head_ablation_scope),
        "carrier_rule": "rank_after_marker_core_else_rank_before_city_to_commit_tail",
        "carrier_clamp": "clean_cumulative_source_through_penultimate_layer",
        "matched_position_control": "equal_token_near_depth_nonitem_clean_state",
        "outcome_blind": True,
        "selection_rank_used": False,
    }
    plan = {**plan_core, "plan_sha256": _sha256_json(plan_core)}
    plan_path = args.output / "frozen_row_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing counter write row plan changed")
    else:
        _atomic_json(plan_path, plan)

    started = time.perf_counter()
    model, tokenizer, adapter = _model(args)
    shards = _prepare_shards(args.output, resume=bool(args.resume), suffix="jsonl")
    completed = skipped = 0
    for index, row in enumerate(rows, start=1):
        shard = shards / f"{_safe_stem(row['request_id'], 'targeted_counter_write')}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            continue
        request_id = str(row["request_id"])
        results = run_targeted_counter_write_trials(
            model,
            tokenizer,
            adapter,
            row,
            banks=banks,
            targeted_site=targeted[request_id],
            source_layer=int(args.source_layer),
            head_ablation_scope=str(args.head_ablation_scope),
            answer_site_id=str(args.answer_site_id),
        )
        for result in results:
            result.update(
                {
                    "mechanism_split": str(args.seed_role),
                    "row_plan": str(plan_path.resolve()),
                    "row_plan_sha256": str(plan["plan_sha256"]),
                }
            )
        _atomic_jsonl(shard, results)
        completed += 1
        print(f"[targeted-counter-write] {index}/{len(rows)}", flush=True)
    _atomic_json(
        args.output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shards.glob("*.jsonl"))),
            extra={
                "newly_completed": completed,
                "resume_skipped": skipped,
                "seed_role": str(args.seed_role),
                "seed_count": expected,
                "rows_per_sample": 7,
                "row_plan_sha256": str(plan["plan_sha256"]),
                "selected_bank_size": len(selected["heads"]),
                "selected_bank_sha256": str(selected["bank_sha256"]),
                "source_layer": int(args.source_layer),
                "head_ablation_scope": str(args.head_ablation_scope),
                "teacher_forced_trace_tokens": True,
                "selection_rank_used": False,
                "outcome_blind": True,
            },
        ),
    )


if __name__ == "__main__":
    main()
