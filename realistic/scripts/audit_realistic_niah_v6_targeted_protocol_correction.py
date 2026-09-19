#!/usr/bin/env python3
"""Audit the Qwen K125 structural preflight and report-grid correction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from analyze_realistic_niah_v6_targeted_retrieval import (
    DEFAULT_REPORT_CONTRACT,
    _sha256,
    _validate_frozen_plan,
    load_report_contract,
    model_report_contract,
)


SCHEMA_VERSION = "realistic_niah_v6_targeted_protocol_correction_audit_v1"


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-root", type=Path, required=True)
    parser.add_argument(
        "--prompt-mode",
        choices=("enumeration_index", "enumeration_bullet"),
        required=True,
    )
    parser.add_argument(
        "--report-contract", type=Path, default=DEFAULT_REPORT_CONTRACT
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = load_report_contract(args.report_contract)
    registered = model_report_contract(contract, "Qwen3-8B")
    if 125 in registered["bank_grid"] or registered["bank_grid"] != (
        32,
        64,
        80,
        96,
        112,
        128,
    ):
        raise ValueError("Corrected Qwen report grid is not exact")
    correction = contract["protocol_correction"]
    if correction.get("trigger") != (
        "qwen_k125_layer_matched_control_structural_infeasibility"
    ):
        raise ValueError("Protocol-correction trigger changed")

    failed_log = args.causal_root / "logs/plan_k125.log"
    if not failed_log.is_file():
        raise FileNotFoundError(f"Missing K125 structural preflight log: {failed_log}")
    log_text = failed_log.read_text(encoding="utf-8", errors="replace")
    required_log_fragments = (
        "Could not construct every registered bank",
        "Not enough non-selected heads",
        "layer-matched",
    )
    if any(fragment not in log_text for fragment in required_log_fragments):
        raise ValueError("K125 log does not prove the registered structural failure")

    behavior_shards = sorted((args.causal_root / "behavior").glob("k*/shards/*.jsonl"))
    if behavior_shards:
        raise ValueError(
            "Targeted behavior outcomes existed before the protocol correction"
        )
    k125 = args.causal_root / "plans/k125"
    forbidden_complete = (
        k125 / "retrieval_anchor_bank_plan.csv",
        k125 / "causal_plan_audit.json",
    )
    if any(path.is_file() for path in forbidden_complete):
        raise ValueError("K125 unexpectedly produced a completed scientific plan")

    reused: dict[str, object] = {}
    for bank_size in (32, 64, 80, 96, 112):
        directory = args.causal_root / f"plans/k{bank_size}"
        reused[str(bank_size)] = _validate_frozen_plan(
            directory / "retrieval_anchor_bank_plan.csv",
            directory / "causal_plan_audit.json",
            model="Qwen3-8B",
            bank_size=bank_size,
            random_condition="layer_matched_random",
        )

    retained_partial = sorted(
        {
            str(path.resolve()): _sha256(path)
            for path in k125.glob("*")
            if path.is_file()
        }.items()
    )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_OUTCOME_BLIND_STRUCTURAL_PROTOCOL_CORRECTION",
        "model_label": "Qwen3-8B",
        "prompt_mode": args.prompt_mode,
        "failed_candidate_k": 125,
        "failed_candidate_registered_in_final_report_grid": False,
        "failure_class": "structural_control_infeasibility_before_behavior",
        "sample_failure": False,
        "seed_replacement_triggered": False,
        "targeted_behavior_outcomes_observed": False,
        "confirmation_outcomes_observed": False,
        "model_outputs_recomputed": False,
        "deletion_performed": False,
        "partial_k125_artifacts_retained": dict(retained_partial),
        "scientific_artifacts_reused": True,
        "reused_completed_plans": reused,
        "corrected_grid": list(registered["bank_grid"]),
        "k128_random_condition": registered["random_condition_by_k"][128],
        "failed_log": str(failed_log.resolve()),
        "failed_log_sha256": _sha256(failed_log),
        "report_contract": str(Path(contract["_path"])),
        "report_contract_sha256": str(contract["_sha256"]),
        "base_config_files_mutated": False,
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
