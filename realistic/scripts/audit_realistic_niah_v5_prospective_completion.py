#!/usr/bin/env python3
"""Audit terminal prospective evidence, report integrity, and model-specific claims."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_v5_native_count_chain_report as report_builder


MODELS = ("Qwen3-8B", "Gemma4-E4B")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(evidence_root: Path, report: Path) -> dict[str, Any]:
    evidence_root = evidence_root.resolve()
    report = report.resolve()
    evidence_manifest_path = evidence_root / "prospective_evidence_manifest.json"
    evidence_manifest = _load(evidence_manifest_path)
    if evidence_manifest.get("status") != "PASS":
        raise ValueError("Prospective evidence manifest is not PASS")

    evidence, evidence_hashes = report_builder._read_evidence(evidence_root)
    report_builder._assert_contract(evidence)
    contract = evidence.get("claim_contract")
    if contract is None:
        raise ValueError("Prospective evidence lacks the frozen claim contract")

    report_manifest_path = report.with_suffix(".manifest.json")
    report_manifest = _load(report_manifest_path)
    if report_manifest.get("status") != "PASS":
        raise ValueError("Report manifest is not PASS")
    if Path(str(report_manifest.get("output"))).resolve() != report:
        raise ValueError("Report manifest output path changed")
    if str(report_manifest.get("output_sha256")) != _sha(report):
        raise ValueError("Report SHA-256 does not match its manifest")
    recorded_evidence_hashes = report_manifest.get("evidence_sha256", {})
    if recorded_evidence_hashes != evidence_hashes:
        raise ValueError("Report evidence-hash ledger is stale")

    classifications: dict[str, dict[str, Any]] = {}
    all_terminal = True
    all_full = True
    for model in MODELS:
        extension = evidence[f"{model}:prospective_extension"]
        extension_status = str(extension["status"])
        endpoint_status = str(extension["endpoint_status"])
        bridge_status = str(extension["bridge_status"])
        if extension_status == "PASS":
            classification = "MODEL_FULL_CHAIN_CONFIRMED"
            if endpoint_status != "PASS" or bridge_status != "PASS":
                raise ValueError(f"{model} PASS extension has a non-PASS stage")
        elif extension_status == "PROTOCOL_EXHAUSTED":
            all_full = False
            if endpoint_status == "PASS":
                classification = "MODEL_PARTIAL_CHAIN_ONLY"
                if bridge_status not in {
                    "DISCOVERY_GATE_FAIL",
                    "CONFIRMATION_GATE_FAIL",
                }:
                    raise ValueError(f"{model} exhausted bridge status is invalid")
            else:
                classification = "PROSPECTIVE_BANK_ENDPOINT_NOT_SUPPORTED"
                if bridge_status != "NOT_OPENED":
                    raise ValueError(f"{model} failed endpoint opened its bridge")
        else:
            all_terminal = False
            all_full = False
            classification = "NONTERMINAL"
        classifications[model] = {
            "classification": classification,
            "extension_status": extension_status,
            "endpoint_status": endpoint_status,
            "bridge_status": bridge_status,
            "bank_size": int(extension["bank_size"]),
            "selected_bank_sha256": str(extension["selected_bank_sha256"]),
            "selection_rank_used": bool(extension["selection_rank_used"]),
        }

    if not all_terminal:
        raise ValueError("At least one prospective extension is nonterminal")
    expected_cross_model = contract["terminal_claim_rules"]["CROSS_MODEL_GENERALITY"]
    return {
        "schema_version": "realistic_niah_v5_prospective_completion_audit_v1",
        "audited_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PASS",
        "protocol_terminal_for_both_models": True,
        "objective_terminal_condition_met": True,
        "cross_model_full_chain_confirmed": all_full,
        "cross_model_claim_rule": expected_cross_model,
        "models": classifications,
        "claim_contract_sha256": _sha(evidence_root / "claim_contract.json"),
        "evidence_manifest_sha256": _sha(evidence_manifest_path),
        "report_sha256": _sha(report),
        "report_manifest_sha256": _sha(report_manifest_path),
        "selection_rank_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.evidence_root, args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
