#!/usr/bin/env python3
"""Fail-closed audit for reusing one completed V6 targeted bank plan."""

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


SCHEMA_VERSION = "realistic_niah_v6_targeted_plan_resume_audit_v1"


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument("--bank-size", type=int, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument(
        "--report-contract", type=Path, default=DEFAULT_REPORT_CONTRACT
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = load_report_contract(args.report_contract)
    registered = model_report_contract(contract, args.model)
    bank_size = int(args.bank_size)
    if bank_size not in registered["bank_grid"]:
        raise ValueError(
            f"K={bank_size} is outside the report-matched {args.model} grid"
        )
    random_condition = registered["random_condition_by_k"][bank_size]
    plan = args.plan_dir / "retrieval_anchor_bank_plan.csv"
    plan_audit = args.plan_dir / "causal_plan_audit.json"
    validated = _validate_frozen_plan(
        plan,
        plan_audit,
        model=args.model,
        bank_size=bank_size,
        random_condition=random_condition,
    )
    adapter = args.plan_dir / "v6_adapter_manifest.json"
    if not adapter.is_file():
        raise FileNotFoundError(f"Missing V6 plan adapter manifest: {adapter}")
    adapter_value = json.loads(adapter.read_text(encoding="utf-8"))
    if adapter_value.get("run_status") not in {"DISPATCHED", "COMPLETE"}:
        raise ValueError("V6 plan adapter manifest has an invalid run status")
    if adapter_value.get("model_label") != args.model:
        raise ValueError("V6 plan adapter model changed")
    legacy_argv = [str(value) for value in adapter_value.get("legacy_argv", ())]
    if adapter_value.get("command") != "causal-plan" or "--bank-size" not in legacy_argv:
        raise ValueError("V6 plan adapter is not a causal-plan dispatch")
    observed_bank_size = int(legacy_argv[legacy_argv.index("--bank-size") + 1])
    if observed_bank_size != bank_size:
        raise ValueError("V6 plan adapter bank size changed")
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_REUSE_WITHOUT_RECOMPUTATION",
        "model_label": args.model,
        "bank_size": bank_size,
        "random_condition": random_condition,
        "plan": validated,
        "adapter_manifest": str(adapter.resolve()),
        "adapter_manifest_sha256": _sha256(adapter),
        "report_contract": str(Path(contract["_path"])),
        "report_contract_sha256": str(contract["_sha256"]),
        "sample_failure": False,
        "model_outputs_recomputed": False,
        "scientific_artifacts_reused": True,
        "deletion_performed": False,
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
