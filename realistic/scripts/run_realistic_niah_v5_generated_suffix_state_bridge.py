#!/usr/bin/env python3
"""Run a frozen free-running targeted suffix -> state -> answer bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


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
from realistic_niah_v5.generated_suffix_bridge import (  # noqa: E402
    run_generated_suffix_state_bridge_trials,
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


MODEL_CONTRACTS = {
    "Gemma4-E4B": {"bank_size": 6, "source_layer": 16},
    "Qwen3-8B": {"bank_size": 128, "source_layer": 19},
}


def _load_banks(path: Path, *, model_label: str) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    if "selection_rank" in frame.columns:
        raise ValueError("Generated-suffix bridge forbids selection_rank")
    required = {
        "model_label",
        "condition",
        "repeat",
        "bank_size",
        "bank_sha256",
        "heads",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Generated-suffix bank plan lacks {missing}")
    frame = frame.loc[frame["model_label"].astype(str).eq(model_label)].copy()
    counts = frame["condition"].astype(str).value_counts().to_dict()
    if counts != {"layer_matched_random": 3, "selected_bank": 1}:
        raise ValueError(f"Generated-suffix bank plan changed: {counts}")
    selected = frame.loc[frame["condition"].astype(str).eq("selected_bank")]
    expected_size = int(MODEL_CONTRACTS[model_label]["bank_size"])
    if len(selected) != 1 or int(selected.iloc[0]["bank_size"]) != expected_size:
        raise ValueError(
            f"Generated-suffix bridge is frozen to {model_label} Top-{expected_size}"
        )
    banks: list[dict[str, Any]] = [
        {"condition": "clean", "repeat": 0, "heads": [], "bank_sha256": "clean"}
    ]
    for row in frame.sort_values(["condition", "repeat"]).itertuples(index=False):
        heads = [[int(a), int(b)] for a, b in json.loads(str(row.heads))]
        if len(heads) != int(row.bank_size) or len({tuple(x) for x in heads}) != len(
            heads
        ):
            raise ValueError("Generated-suffix bank head count changed")
        banks.append(
            {
                "condition": str(row.condition),
                "repeat": int(row.repeat),
                "heads": heads,
                "bank_sha256": str(row.bank_sha256),
            }
        )
    return banks


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
        "--seed-role", choices=["development", "confirmation"], required=True
    )
    parser.add_argument("--anchor-registry", type=Path, required=True)
    parser.add_argument("--targeted-registry", type=Path, required=True)
    parser.add_argument("--bank-plan", type=Path, required=True)
    parser.add_argument("--source-layer", type=int, default=16)
    parser.add_argument(
        "--state-patch-geometry",
        choices=[
            "terminal_span",
            "generated_suffix_span",
            "terminal_prefix_span",
            "grammar_counter_carrier",
            "grammar_counter_tail",
            "terminal_last4",
            "terminal_last8",
        ],
        default="terminal_span",
    )
    parser.add_argument("--include-matched-position-control", action="store_true")
    parser.add_argument(
        "--selection-rule",
        default="frozen_outcome_blind_highest_count_anchor_per_seed",
    )
    parser.add_argument("--panel-id", default="highest_count")
    parser.add_argument("--answer-site-id", default="answer_query_v3")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.command = "generated-suffix-state-bridge"
    args.cohort = "one_to_one"
    args.row_panel = "trace_patch"
    args.limit = None

    contract = MODEL_CONTRACTS[str(args.model)]
    if int(args.source_layer) != int(contract["source_layer"]):
        raise ValueError(
            f"{args.model} requires source layer {contract['source_layer']}"
        )

    started = time.perf_counter()
    mechanism = _spec(args)
    rows = _registered_rows(args, mechanism)
    anchors = _read_jsonl(args.anchor_registry)
    targeted_rows = _read_jsonl(args.targeted_registry)
    if any("selection_rank" in row for row in anchors + targeted_rows):
        raise ValueError("Generated-suffix registries forbid selection_rank")
    anchor_ids = {str(row["request_id"]) for row in anchors}
    targeted = {str(row["request_id"]): row for row in targeted_rows}
    if len(targeted) != len(targeted_rows):
        raise ValueError("Generated-suffix targeted registry has duplicate requests")
    missing_targeted = sorted(anchor_ids - set(targeted))
    if missing_targeted:
        raise ValueError(
            f"Generated-suffix anchors lack targeted sites: {missing_targeted}"
        )
    rows = [row for row in rows if str(row["request_id"]) in anchor_ids]
    expected = 20 if args.seed_role == "development" else 10
    if len(rows) != expected or len({int(row["seed"]) for row in rows}) != expected:
        raise ValueError("Generated-suffix bridge seed contract changed")
    rows.sort(key=lambda row: int(row["seed"]))
    banks = _load_banks(args.bank_plan, model_label=str(args.model))
    selected = next(bank for bank in banks if bank["condition"] == "selected_bank")
    plan_core = {
        "schema_version": "realistic_niah_v5_generated_suffix_state_plan_v2",
        "model_label": str(args.model),
        "seed_role": str(args.seed_role),
        "panel_id": str(args.panel_id),
        "selection_rule": str(args.selection_rule),
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
        "anchor_registry_sha256": _sha256(args.anchor_registry),
        "targeted_registry": str(args.targeted_registry.resolve()),
        "targeted_registry_sha256": _sha256(args.targeted_registry),
        "bank_plan": str(args.bank_plan.resolve()),
        "bank_plan_sha256": _sha256(args.bank_plan),
        "selected_bank_size": len(selected["heads"]),
        "selected_bank_sha256": str(selected["bank_sha256"]),
        "source_layer": int(args.source_layer),
        "state_patch_geometry": str(args.state_patch_geometry),
        "free_running_policy": "fixed_token_budget_through_terminal_item_end",
        "post_terminal_suffix_teacher_forced": True,
    }
    if args.include_matched_position_control:
        plan_core["include_matched_position_control"] = True
    plan = {**plan_core, "plan_sha256": _sha256_json(plan_core)}
    plan_path = args.output / "frozen_row_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing generated-suffix row plan changed")
    else:
        _atomic_json(plan_path, plan)

    model, tokenizer, adapter = _model(args)
    shards = _prepare_shards(args.output, resume=bool(args.resume), suffix="jsonl")
    completed = skipped = 0
    for index, row in enumerate(rows, start=1):
        shard = shards / f"{_safe_stem(row['request_id'], 'generated_suffix_state')}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            continue
        request_id = str(row["request_id"])
        results = run_generated_suffix_state_bridge_trials(
            model,
            tokenizer,
            adapter,
            row,
            banks=banks,
            targeted_site=targeted[request_id],
            source_layer=int(args.source_layer),
            state_patch_geometry=str(args.state_patch_geometry),
            include_matched_position_control=bool(
                args.include_matched_position_control
            ),
            answer_site_id=str(args.answer_site_id),
            run_greedy_answer=True,
            max_new_tokens=int(args.max_new_tokens),
        )
        for result in results:
            result.update(
                {
                    "mechanism_split": str(args.seed_role),
                    "row_plan": str(plan_path.resolve()),
                    "row_plan_sha256": str(plan["plan_sha256"]),
                    "selection_rank_used": False,
                    "outcome_blind": True,
                }
            )
        _atomic_jsonl(shard, results)
        completed += 1
        print(f"[generated-suffix-state] {index}/{len(rows)}", flush=True)
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
                "rows_per_sample": (
                    11 if args.include_matched_position_control else 10
                ),
                "row_plan": str(plan_path.resolve()),
                "row_plan_sha256": str(plan["plan_sha256"]),
                "selected_bank_size": len(selected["heads"]),
                "selected_bank_sha256": str(selected["bank_sha256"]),
                "source_layer": int(args.source_layer),
                "state_patch_geometry": str(args.state_patch_geometry),
                "panel_id": str(args.panel_id),
                "selection_rank_used": False,
                "outcome_blind": True,
                "include_matched_position_control": bool(
                    args.include_matched_position_control
                ),
            },
        ),
    )


if __name__ == "__main__":
    main()
