#!/usr/bin/env python3
"""Run the V6 terminal-token -> state -> answer bridge for either model."""

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
from realistic_niah_v5.terminal_token_state import (  # noqa: E402
    REGISTERED_TERMINAL_TOKEN_STATE_CONDITIONS,
    run_terminal_token_state_bridge_trials,
)
from realistic_niah_v6.kernel import MODE_TIMING_STRATA  # noqa: E402
from realistic_niah_v6.spec import V6Config  # noqa: E402


LAYER_CONTRACTS = {"Qwen3-8B": 19, "Gemma4-E4B": 16}


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism-config", type=Path, required=True)
    parser.add_argument("--v5-config", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(LAYER_CONTRACTS), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument(
        "--seed-role", choices=("development", "confirmation"), required=True
    )
    parser.add_argument("--anchor-registry", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--answer-site-id", default="answer_query_v3")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.command = "terminal-token-state-bridge"
    args.cohort = "one_to_one"
    args.row_panel = "trace_patch"
    args.limit = None

    if int(args.layer) != LAYER_CONTRACTS[str(args.model)]:
        raise ValueError(
            f"V6 terminal bridge layer contract for {args.model} is "
            f"L{LAYER_CONTRACTS[str(args.model)]}"
        )
    config = V6Config.load(args.v5_config)
    timing = MODE_TIMING_STRATA[config.prompt_mode]
    started = time.perf_counter()
    mechanism = _spec(args)
    rows = _registered_rows(args, mechanism)
    anchors = [
        json.loads(line)
        for line in args.anchor_registry.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any("selection_rank" in row for row in anchors):
        raise ValueError("V6 terminal bridge forbids selection_rank")
    if any(
        str(row.get("mode_timing_stratum", timing)) != timing for row in anchors
    ):
        raise ValueError("V6 terminal bridge registry mixes timing strata")
    anchor_ids = {str(row["request_id"]) for row in anchors}
    if len(anchor_ids) != len(anchors):
        raise ValueError("V6 terminal bridge registry duplicates requests")
    rows = [row for row in rows if str(row["request_id"]) in anchor_ids]
    expected = 20 if args.seed_role == "development" else 10
    if len(rows) != expected or len({int(row["seed"]) for row in rows}) != expected:
        raise ValueError("V6 terminal bridge requires one row per registered seed")
    rows.sort(key=lambda row: int(row["seed"]))
    plan_core = {
        "schema_version": "realistic_niah_v6_terminal_token_state_row_plan_v1",
        "model_label": str(args.model),
        "prompt_mode": config.prompt_mode,
        "mode_timing_stratum": timing,
        "seed_role": str(args.seed_role),
        "selection_rule": "frozen_outcome_blind_highest_strict_count_per_seed",
        "selection_rank_used": False,
        "outcome_blind": True,
        "seed_count": expected,
        "seeds": [int(row["seed"]) for row in rows],
        "selected_count_by_seed": {
            str(row["seed"]): int(row["gold_count"]) for row in rows
        },
        "selected_request_by_seed": {
            str(row["seed"]): str(row["request_id"]) for row in rows
        },
        "anchor_registry": str(args.anchor_registry.resolve()),
        "anchor_registry_sha256": hashlib.sha256(
            args.anchor_registry.read_bytes()
        ).hexdigest(),
        "layer": int(args.layer),
        "conditions": list(REGISTERED_TERMINAL_TOKEN_STATE_CONDITIONS),
    }
    plan = {**plan_core, "plan_sha256": _sha256_json(plan_core)}
    plan_path = args.output / "frozen_row_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing V6 terminal bridge row plan changed")
    else:
        _atomic_json(plan_path, plan)

    model, tokenizer, adapter = _model(args)
    shards = _prepare_shards(args.output, resume=bool(args.resume), suffix="jsonl")
    completed = skipped = 0
    for index, row in enumerate(rows, start=1):
        shard = shards / f"{_safe_stem(row['request_id'], 'v6_terminal_token_state')}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            continue
        results = run_terminal_token_state_bridge_trials(
            model,
            tokenizer,
            adapter,
            row,
            layer=int(args.layer),
            random_seed=20260828 + int(row["seed"]),
            answer_site_id=str(args.answer_site_id),
            run_greedy=True,
            max_new_tokens=int(args.max_new_tokens),
        )
        for result in results:
            result.update(
                {
                    "v6_prompt_mode": config.prompt_mode,
                    "v6_mode_timing_stratum": timing,
                    "mechanism_split": str(args.seed_role),
                    "row_plan": str(plan_path.resolve()),
                    "row_plan_sha256": str(plan["plan_sha256"]),
                    "selection_rank_used": False,
                    "outcome_blind": True,
                }
            )
        _atomic_jsonl(shard, results)
        completed += 1
        print(f"[v6 terminal-token-state] {index}/{len(rows)}", flush=True)
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
                "rows_per_sample": len(REGISTERED_TERMINAL_TOKEN_STATE_CONDITIONS),
                "prompt_mode": config.prompt_mode,
                "mode_timing_stratum": timing,
                "row_plan_sha256": str(plan["plan_sha256"]),
                "selection_rank_used": False,
                "outcome_blind": True,
            },
        ),
    )


if __name__ == "__main__":
    main()
