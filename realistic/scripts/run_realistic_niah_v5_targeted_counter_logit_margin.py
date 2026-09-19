#!/usr/bin/env python3
"""Run one frozen timing branch of the direct count-logit margin assay."""

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
from realistic_niah_v5.targeted_counter_logit_margin import (  # noqa: E402
    LOGIT_MARGIN_ENDPOINTS,
    run_targeted_counter_logit_margin,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        "--timing", choices=tuple(LOGIT_MARGIN_ENDPOINTS), required=True
    )
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--bank-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.command = "targeted-counter-logit-margin"
    args.cohort = "one_to_one"
    args.row_panel = "trace_patch"
    args.limit = None

    mechanism = _spec(args)
    registered = _registered_rows(args, mechanism)
    panel_rows = _read_jsonl(args.panel)
    if not panel_rows or any("selection_rank" in row for row in panel_rows):
        raise ValueError("Logit-margin panel is empty or contains selection_rank")
    if len(panel_rows) != len({str(row["request_id"]) for row in panel_rows}):
        raise ValueError("Logit-margin panel contains duplicate requests")
    if any(
        str(row.get("grammar_span_timing_stratum")) != str(args.timing)
        for row in panel_rows
    ):
        raise ValueError("Logit-margin panel mixes timing branches")
    if any(
        not bool(row.get("stratified_ncc_outcome_blind"))
        or bool(row.get("stratified_ncc_selection_rank_used"))
        for row in panel_rows
    ):
        raise ValueError("Logit-margin panel outcome-blind contract changed")
    phase_panel = [
        row
        for row in panel_rows
        if str(row.get("stratified_ncc_seed_role")) == str(args.seed_role)
    ]
    panel_by_id = {str(row["request_id"]): row for row in phase_panel}
    expected_seeds = {int(row["seed"]) for row in phase_panel}
    if len(expected_seeds) != len(phase_panel):
        raise ValueError("Logit-margin phase panel contains duplicate seeds")
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
            f"Logit-margin {args.seed_role} branch has only {len(expected_seeds)} seeds"
        )
    rows = [row for row in registered if str(row["request_id"]) in panel_by_id]
    observed_ids = {str(row["request_id"]) for row in rows}
    if observed_ids != set(panel_by_id) or len(rows) != len(observed_ids):
        raise ValueError("Registered generations do not reproduce the margin panel")
    if {int(row["seed"]) for row in rows} != expected_seeds:
        raise ValueError("Logit-margin seed set changed")
    rows.sort(key=lambda row: int(row["seed"]))

    contract = MODEL_CONTRACTS[str(args.model)]
    banks = _load_banks(args.bank_plan, model_label=str(args.model))
    selected = next(bank for bank in banks if bank["condition"] == "selected_bank")
    plan = {
        "schema_version": "realistic_niah_v5_targeted_counter_logit_margin_plan_v2",
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
        "endpoint_names": list(LOGIT_MARGIN_ENDPOINTS[str(args.timing)]),
        "primary_endpoint": "final_answer_sequence_margin",
        "primary_estimand": "clean_margin_minus_mask_margin",
        "specificity_estimand": "selected_loss_minus_three_random_mean_loss",
        "readout_validity_gate": (
            "clean mean margin > 0 and clean candidate accuracy > chance"
        ),
        "directional_gate": "selected loss > 0 and specificity > 0",
        "interval_gate": "both registered 95% seed-bootstrap CI lower bounds > 0",
        "local_rank_secondary_requires_exact_adjacent_same_grammar": True,
        "local_rank_secondary_scoring": (
            "full_autoregressive_N_vs_N_minus_1_marker_sequence_log_probability"
        ),
        "no_decoder_fit_or_layer_selection": True,
        "confirmation_status": "registered_existing_split_after_ncc_inspection",
        "confirmation_used_for_registration": False,
        "outcome_blind_panel": True,
        "selection_rank_used": False,
        "cross_model_exact_sample_alignment": exact_aligned,
    }
    plan_path = args.output / "frozen_row_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing logit-margin row plan changed")
    else:
        _atomic_json(plan_path, plan)

    started = time.perf_counter()
    model, tokenizer, adapter = _model(args)
    shards = _prepare_shards(args.output, resume=bool(args.resume), suffix="json")
    completed = skipped = 0
    for index, row in enumerate(rows, start=1):
        shard = shards / f"{_safe_stem(row['request_id'], 'logit_margin')}.json"
        if args.resume and shard.exists():
            skipped += 1
            continue
        result = run_targeted_counter_logit_margin(
            model,
            tokenizer,
            adapter,
            row,
            banks=banks,
            targeted_site=panel_by_id[str(row["request_id"])],
            timing=str(args.timing),
            selected_bank_size=int(contract["bank_size"]),
        )
        result.update(
            {
                "mechanism_split": str(args.seed_role),
                "row_plan": str(plan_path.resolve()),
                "panel_sha256": _sha256(args.panel),
                "bank_plan_sha256": _sha256(args.bank_plan),
            }
        )
        _atomic_json(shard, result)
        completed += 1
        print(f"[logit-margin:{args.timing}] {index}/{len(rows)}", flush=True)
    _atomic_json(
        args.output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shards.glob("*.json"))),
            extra={
                "newly_completed": completed,
                "resume_skipped": skipped,
                "seed_count": len(rows),
                "seeds": sorted(expected_seeds),
                "conditions": 5,
                "timing_branch": str(args.timing),
                "endpoint_names": list(LOGIT_MARGIN_ENDPOINTS[str(args.timing)]),
                "primary_endpoint": "final_answer_sequence_margin",
                "no_decoder_fit_or_layer_selection": True,
                "outcome_blind_panel": True,
                "selection_rank_used": False,
                "cross_model_exact_sample_alignment": exact_aligned,
            },
        ),
    )


if __name__ == "__main__":
    main()
