#!/usr/bin/env python3
"""Run formal grammar-timed terminal-span state decomposition trials."""

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
    REGISTERED_GRAMMAR_SPAN_DECOMPOSITION_CONDITIONS,
    run_grammar_span_decomposition_trials,
)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism-config", type=Path, required=True)
    parser.add_argument("--v5-config", type=Path, required=True)
    parser.add_argument("--model", choices=["Qwen3-8B", "Gemma4-E4B"], required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument(
        "--seed-role", choices=["development", "confirmation"], required=True
    )
    parser.add_argument("--anchor-panel", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--answer-site-id", default="answer_query_v3")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.command = "grammar-span-decomposition"
    args.cohort = "one_to_one"
    args.row_panel = "trace_patch"
    args.limit = None

    started = time.perf_counter()
    mechanism = _spec(args)
    rows = _registered_rows(args, mechanism)
    anchors = [
        json.loads(line)
        for line in args.anchor_panel.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads(args.anchor_manifest.read_text(encoding="utf-8"))
    if _sha256_json(anchors) != str(manifest["panel_sha256"]):
        raise ValueError("Grammar-span anchor panel hash changed")
    if any(
        "selection_rank" in anchor
        or not bool(anchor.get("grammar_span_outcome_blind"))
        or bool(anchor.get("grammar_span_selection_rank_used"))
        for anchor in anchors
    ):
        raise ValueError("Grammar-span anchor panel violates the frozen contract")
    anchor_by_request = {str(anchor["request_id"]): anchor for anchor in anchors}
    rows = [row for row in rows if str(row["request_id"]) in anchor_by_request]
    is_cross_model_aligned = bool(manifest.get("exact_cross_model_key_equality"))
    if is_cross_model_aligned:
        expected_seeds = [
            int(seed) for seed in manifest["phase_seeds"][str(args.seed_role)]
        ]
        expected = len(expected_seeds)
    else:
        expected = 20 if args.seed_role == "development" else 10
        expected_seeds = list(
            range(1234, 1254)
            if args.seed_role == "development"
            else range(1254, 1264)
        )
    if len(rows) != expected or len({int(row["seed"]) for row in rows}) != expected:
        raise ValueError("Grammar-span trials changed the fixed seed contract")
    rows.sort(key=lambda row: int(row["seed"]))
    if [int(row["seed"]) for row in rows] != expected_seeds:
        raise ValueError("Grammar-span rows do not match the frozen phase seeds")
    timing_counts = {
        timing: sum(
            str(anchor_by_request[str(row["request_id"])]["grammar_span_timing_stratum"])
            == timing
            for row in rows
        )
        for timing in ("rank_after_city", "rank_before_city")
    }
    if is_cross_model_aligned:
        expected_timing_counts = {
            str(key): int(value)
            for key, value in manifest["timing_counts_by_phase"][
                str(args.seed_role)
            ].items()
        }
        if timing_counts != expected_timing_counts:
            raise ValueError("Grammar-span timing counts changed from aligned manifest")
    else:
        expected_per_timing = 10 if args.seed_role == "development" else 5
        if set(timing_counts.values()) != {expected_per_timing}:
            raise ValueError("Grammar-span phase is not exactly timing-balanced")
    plan_core = {
        "schema_version": "realistic_niah_v5_grammar_span_row_plan_v1",
        "model_label": str(args.model),
        "seed_role": str(args.seed_role),
        "selection_rank_used": False,
        "outcome_blind": True,
        "seed_count": expected,
        "seeds": [int(row["seed"]) for row in rows],
        "request_ids": [str(row["request_id"]) for row in rows],
        "timing_counts": timing_counts,
        "anchor_panel_sha256": str(manifest["panel_sha256"]),
        "anchor_manifest_sha256": str(manifest["manifest_sha256"]),
        "cross_model_exact_sample_alignment": is_cross_model_aligned,
        "layer": int(args.layer),
        "layer_selection_rule": "historical_full_span_confirmation_layer_frozen",
        "conditions": list(REGISTERED_GRAMMAR_SPAN_DECOMPOSITION_CONDITIONS),
    }
    plan = {**plan_core, "plan_sha256": _sha256_json(plan_core)}
    plan_path = args.output / "frozen_row_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing grammar-span row plan changed")
    else:
        _atomic_json(plan_path, plan)

    model, tokenizer, adapter = _model(args)
    shards = _prepare_shards(args.output, resume=bool(args.resume), suffix="jsonl")
    completed = skipped = 0
    for index, row in enumerate(rows, start=1):
        anchor = anchor_by_request[str(row["request_id"])]
        shard = shards / f"{_safe_stem(row['request_id'], 'grammar_span')}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            continue
        results = run_grammar_span_decomposition_trials(
            model,
            tokenizer,
            adapter,
            row,
            layer=int(args.layer),
            random_seed=20260821 + int(row["seed"]),
            registered_grammar_class=str(anchor["target_grammar_class"]),
            registered_timing_stratum=str(anchor["grammar_span_timing_stratum"]),
            answer_site_id=str(args.answer_site_id),
            run_greedy=True,
            max_new_tokens=int(args.max_new_tokens),
        )
        for result in results:
            result.update(
                {
                    "mechanism_split": str(args.seed_role),
                    "row_plan": str(plan_path.resolve()),
                    "row_plan_sha256": str(plan["plan_sha256"]),
                    "anchor_panel_sha256": str(manifest["panel_sha256"]),
                    "selection_rank_used": False,
                    "outcome_blind": True,
                }
            )
        _atomic_jsonl(shard, results)
        completed += 1
        print(f"[grammar-span] {index}/{len(rows)}", flush=True)
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
                "timing_counts": timing_counts,
                "rows_per_sample": len(
                    REGISTERED_GRAMMAR_SPAN_DECOMPOSITION_CONDITIONS
                ),
                "row_plan": str(plan_path.resolve()),
                "row_plan_sha256": str(plan["plan_sha256"]),
                "selection_rank_used": False,
                "outcome_blind": True,
            },
        ),
    )


if __name__ == "__main__":
    main()
