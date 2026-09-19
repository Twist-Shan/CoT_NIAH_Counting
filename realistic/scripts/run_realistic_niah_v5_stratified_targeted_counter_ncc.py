#!/usr/bin/env python3
"""Capture one frozen timing branch of the section 5.1 NCC assay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
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
from realistic_niah_v5.stratified_targeted_counter_ncc import (  # noqa: E402
    STRATIFIED_NCC_ENDPOINTS,
    capture_stratified_targeted_counter_ncc,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_npz(
    path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(path)


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
    parser.add_argument(
        "--timing", choices=tuple(STRATIFIED_NCC_ENDPOINTS), required=True
    )
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--bank-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.command = "stratified-targeted-counter-ncc"
    args.cohort = "one_to_one"
    args.row_panel = "trace_patch"
    args.limit = None

    mechanism = _spec(args)
    registered = _registered_rows(args, mechanism)
    panel_rows = _read_jsonl(args.panel)
    if not panel_rows:
        raise ValueError("Stratified NCC panel is empty")
    if any("selection_rank" in row for row in panel_rows):
        raise ValueError("Stratified NCC panel forbids selection_rank")
    if len(panel_rows) != len({str(row["request_id"]) for row in panel_rows}):
        raise ValueError("Stratified NCC panel contains duplicate requests")
    if any(
        str(row.get("grammar_span_timing_stratum")) != str(args.timing)
        for row in panel_rows
    ):
        raise ValueError("Stratified NCC panel mixes timing branches")
    if any(
        not bool(row.get("stratified_ncc_outcome_blind"))
        or bool(row.get("stratified_ncc_selection_rank_used"))
        for row in panel_rows
    ):
        raise ValueError("Stratified NCC panel outcome-blind contract changed")
    phase_panel = [
        row
        for row in panel_rows
        if str(row.get("stratified_ncc_seed_role")) == str(args.seed_role)
    ]
    panel_by_id = {str(row["request_id"]): row for row in phase_panel}
    expected_seeds = {int(row["seed"]) for row in phase_panel}
    if len(expected_seeds) != len(phase_panel):
        raise ValueError("Stratified NCC phase panel contains duplicate seeds")
    exact_aligned = all(
        bool(row.get("cross_model_exact_sample_alignment")) for row in panel_rows
    )
    minimum = (
        (8 if args.seed_role == "development" else 4)
        if exact_aligned
        else (15 if args.seed_role == "development" else 8)
    )
    if len(expected_seeds) < minimum:
        raise ValueError(
            f"Stratified NCC {args.seed_role} branch has only {len(expected_seeds)} seeds"
        )
    rows = [row for row in registered if str(row["request_id"]) in panel_by_id]
    observed_ids = {str(row["request_id"]) for row in rows}
    if observed_ids != set(panel_by_id) or len(rows) != len(observed_ids):
        raise ValueError("Registered generations do not reproduce the frozen branch panel")
    if {int(row["seed"]) for row in rows} != expected_seeds:
        raise ValueError("Stratified NCC seed set changed")
    rows.sort(key=lambda row: int(row["seed"]))

    contract = MODEL_CONTRACTS[str(args.model)]
    banks = _load_banks(args.bank_plan, model_label=str(args.model))
    selected = next(bank for bank in banks if bank["condition"] == "selected_bank")
    maximum_head_layer = max(int(layer) for layer, _head in selected["heads"])
    capture_start_layer = maximum_head_layer + 1

    plan = {
        "schema_version": "realistic_niah_v5_stratified_targeted_counter_ncc_plan_v1",
        "model_label": str(args.model),
        "timing_branch": str(args.timing),
        "seed_role": str(args.seed_role),
        "seeds": [int(row["seed"]) for row in rows],
        "seed_count": len(rows),
        "request_ids": [str(row["request_id"]) for row in rows],
        "panel_sha256": _sha256(args.panel),
        "bank_plan_sha256": _sha256(args.bank_plan),
        "selected_bank_sha256": str(selected["bank_sha256"]),
        "selected_bank_size": len(selected["heads"]),
        "maximum_selected_head_layer": maximum_head_layer,
        "capture_start_layer": capture_start_layer,
        "capture_layer_rule": "strictly_above_maximum_ablated_head_layer",
        "endpoint_names": list(STRATIFIED_NCC_ENDPOINTS[str(args.timing)]),
        "primary_endpoint": STRATIFIED_NCC_ENDPOINTS[str(args.timing)][0],
        "fit_policy": "branch_discovery_clean_matching_events_only",
        "confirmation_used_for_fit_or_layer_selection": False,
        "outcome_blind": True,
        "selection_rank_used": False,
        "cross_model_exact_sample_alignment": exact_aligned,
    }
    plan_path = args.output / "frozen_row_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing stratified NCC row plan changed")
    else:
        _atomic_json(plan_path, plan)

    started = time.perf_counter()
    model, tokenizer, adapter = _model(args)
    if capture_start_layer >= int(adapter.num_layers):
        raise ValueError("Selected head bank leaves no reachable capture layer")
    shards = _prepare_shards(args.output, resume=bool(args.resume), suffix="npz")
    completed = skipped = 0
    for index, row in enumerate(rows, start=1):
        shard = shards / f"{_safe_stem(row['request_id'], 'stratified_ncc')}.npz"
        if args.resume and shard.exists():
            skipped += 1
            continue
        targeted_site = panel_by_id[str(row["request_id"])]
        arrays, metadata = capture_stratified_targeted_counter_ncc(
            model,
            tokenizer,
            adapter,
            row,
            banks=banks,
            targeted_site=targeted_site,
            timing=str(args.timing),
            capture_start_layer=capture_start_layer,
            selected_bank_size=int(contract["bank_size"]),
        )
        metadata.update(
            {
                "mechanism_split": str(args.seed_role),
                "row_plan": str(plan_path.resolve()),
                "panel_sha256": _sha256(args.panel),
                "bank_plan_sha256": _sha256(args.bank_plan),
            }
        )
        _atomic_npz(shard, arrays, metadata)
        completed += 1
        print(
            f"[stratified-ncc:{args.timing}] {index}/{len(rows)}",
            flush=True,
        )
    _atomic_json(
        args.output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shards.glob("*.npz"))),
            extra={
                "newly_completed": completed,
                "resume_skipped": skipped,
                "seed_count": len(rows),
                "seeds": sorted(expected_seeds),
                "conditions": 5,
                "timing_branch": str(args.timing),
                "endpoint_names": list(STRATIFIED_NCC_ENDPOINTS[str(args.timing)]),
                "capture_start_layer": capture_start_layer,
                "maximum_selected_head_layer": maximum_head_layer,
                "fit_policy": "branch_discovery_clean_matching_events_only",
                "outcome_blind": True,
                "selection_rank_used": False,
                "cross_model_exact_sample_alignment": exact_aligned,
            },
        ),
    )


if __name__ == "__main__":
    main()
