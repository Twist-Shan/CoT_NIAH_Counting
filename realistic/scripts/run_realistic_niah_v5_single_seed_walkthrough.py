#!/usr/bin/env python3
"""Run one frozen Native-thinking count-restoration trajectory."""

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

from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
    _model,
    _registered_rows,
    _runtime_manifest,
    _spec,
)
from realistic_niah_v5.single_seed_walkthrough import (  # noqa: E402
    run_single_seed_walkthrough_trials,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism-config", type=Path, required=True)
    parser.add_argument("--walkthrough-config", type=Path, required=True)
    parser.add_argument("--v5-config", type=Path, required=True)
    parser.add_argument("--model", choices=["Qwen3-8B", "Gemma4-E4B"], required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--source-layer", type=int, required=True)
    parser.add_argument("--answer-site-id", default="answer_query_v3")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.command = "single-seed-counter-walkthrough"
    args.seed_role = "confirmation"
    args.cohort = "one_to_one"
    args.row_panel = "all"
    args.limit = None

    started = time.perf_counter()
    mechanism = _spec(args)
    walkthrough_protocol = json.loads(
        args.walkthrough_config.read_text(encoding="utf-8")
    )
    registered_case = walkthrough_protocol["models"][str(args.model)]
    if (
        int(registered_case["seed"]) != int(args.seed)
        or int(registered_case["gold_count"]) != int(args.expected_count)
        or int(registered_case["source_layer"]) != int(args.source_layer)
        or str(registered_case["request_id"]) != str(args.request_id)
    ):
        raise ValueError("Walkthrough CLI disagrees with the frozen case protocol")
    rows = [
        row
        for row in _registered_rows(args, mechanism)
        if int(row["seed"]) == int(args.seed)
        and str(row["request_id"]) == str(args.request_id)
    ]
    if len(rows) != 1:
        raise ValueError(
            "Frozen walkthrough request must resolve to exactly one registered row: "
            f"observed={len(rows)}"
        )
    row = rows[0]
    if int(row.get("gold_count", 0)) != int(args.expected_count):
        raise ValueError("Walkthrough count metadata changed")
    if int(args.seed) not in tuple(int(value) for value in mechanism.confirmation_seeds):
        raise ValueError("Walkthrough seed is outside the frozen confirmation registry")

    results = run_single_seed_walkthrough_trials(
        *_model(args),
        row,
        source_layer=int(args.source_layer),
        random_seed=20260822 + int(args.seed),
        answer_site_id=str(args.answer_site_id),
        max_new_tokens=int(args.max_new_tokens),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path = args.output / "walkthrough_rows.jsonl"
    _atomic_jsonl(rows_path, results)
    plan = {
        "schema_version": "realistic_niah_v5_single_seed_walkthrough_plan_v1",
        "status": "FROZEN_CASE_STUDY",
        "model_label": str(args.model),
        "request_id": str(args.request_id),
        "seed": int(args.seed),
        "gold_count": int(args.expected_count),
        "source_layer": int(args.source_layer),
        "selection_rule": (
            "metadata-only confirmation case with a complete ten-item trace; "
            "never used for formal estimation or head/site selection"
        ),
        "case_selected_by_outcome": False,
        "case_study_not_inferential": True,
        "conditions": [
            "clean",
            "uninformative",
            "full_item_restore",
            "counter_carrier_restore",
            "counter_carrier_matched_control",
        ],
        "answer_query_patched": False,
    }
    _atomic_json(args.output / "frozen_case_plan.json", plan)
    _atomic_json(
        args.output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=1,
            extra={
                "walkthrough_rows": len(results),
                "walkthrough_rows_path": str(rows_path.resolve()),
                "walkthrough_config": str(args.walkthrough_config.resolve()),
                "walkthrough_config_sha256": hashlib.sha256(
                    args.walkthrough_config.read_bytes()
                ).hexdigest(),
                "walkthrough_protocol_schema_version": str(
                    walkthrough_protocol["schema_version"]
                ),
                "frozen_case_plan": plan,
                "case_study_not_inferential": True,
            },
        ),
    )
    print(json.dumps(plan, sort_keys=True))


if __name__ == "__main__":
    main()
