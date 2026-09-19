#!/usr/bin/env python3
"""Audit and finalize one prospectively frozen targeted-bank chain extension."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ENDPOINT_TERMINAL = {"PASS", "DISCOVERY_NEGATIVE", "CONFIRMATION_NEGATIVE"}
BRIDGE_TERMINAL = {"PASS", "DISCOVERY_GATE_FAIL", "CONFIRMATION_GATE_FAIL"}
PLAN_AUDIT_TERMINAL = {"PASS", "FROZEN_NEW", "REUSED_IDENTICAL"}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _audit_seed_contract(
    audit: dict[str, Any], *, phase: str, expected: int, applicable: bool
) -> None:
    if audit.get("status") != "PASS":
        raise ValueError(f"{phase} audit is not PASS")
    if int(audit.get("seed_count", -1)) != expected:
        raise ValueError(f"{phase} seed contract changed")
    if audit.get("selection_rank_used") is not False:
        raise ValueError(f"{phase} used selection_rank")
    if applicable and int(audit.get("applicable_seed_count", -1)) != expected:
        raise ValueError(f"{phase} effective seed contract changed")


def _plan_signature(root: Path) -> tuple[int, str, dict[str, Any]]:
    plan = root / "frozen_targeted_count_plan.csv"
    with plan.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        rows = list(reader)
    if "selection_rank" in fields:
        raise ValueError("Frozen targeted plan contains selection_rank")
    selected = [row for row in rows if row["condition"] == "selected_bank"]
    if len(selected) != 1:
        raise ValueError("Frozen targeted plan must contain one selected bank")
    audit = _load(plan.with_suffix(".audit.json"))
    return int(selected[0]["bank_size"]), str(selected[0]["bank_sha256"]), audit


def finalize(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    root = root.resolve()
    model = str(protocol["model_label"])
    bank = protocol["bank"]
    expected_k = int(bank["size"])
    expected_sha = str(bank["selected_bank_sha256"])
    stages = protocol["stages"]
    if [stage["name"] for stage in stages] != [
        "targeted_retrieval_to_final_count",
        "targeted_retrieval_to_terminal_state_to_readout",
    ]:
        raise ValueError("Bank-extension stage order changed")

    endpoint_root = _resolve(root, str(stages[0]["root"]))
    endpoint_complete_path = endpoint_root / "targeted_count_complete.json"
    endpoint_complete = _load(endpoint_complete_path)
    endpoint_status = str(endpoint_complete.get("status"))
    if endpoint_status not in ENDPOINT_TERMINAL:
        raise ValueError(f"Endpoint is not terminal: {endpoint_status}")
    observed_k, observed_sha, plan_audit = _plan_signature(endpoint_root)
    if (observed_k, observed_sha) != (expected_k, expected_sha):
        raise ValueError("Endpoint plan does not match the frozen bank")
    if plan_audit.get("selection_rank_used") is not False:
        raise ValueError("Endpoint plan audit used selection_rank")
    plan_audit_status = str(plan_audit.get("status"))
    if plan_audit_status not in PLAN_AUDIT_TERMINAL:
        raise ValueError("Endpoint plan audit is not terminal")
    if plan_audit_status != "PASS":
        if plan_audit.get("outcome_blind") is not True:
            raise ValueError("Endpoint plan audit is not outcome-blind")
        if plan_audit.get("historical_artifacts_modified") is not False:
            raise ValueError("Endpoint plan audit modified historical artifacts")
        if str(plan_audit.get("model_label")) != model:
            raise ValueError("Endpoint plan audit model mismatch")
        if int(plan_audit.get("bank_size", -1)) != expected_k:
            raise ValueError("Endpoint plan audit bank size mismatch")
        if str(plan_audit.get("selected_bank_sha256")) != expected_sha:
            raise ValueError("Endpoint plan audit bank hash mismatch")
    endpoint_discovery_path = endpoint_root / "targeted_count_analysis_discovery/audit.json"
    endpoint_discovery = _load(endpoint_discovery_path)
    _audit_seed_contract(
        endpoint_discovery, phase="endpoint discovery", expected=20, applicable=False
    )
    endpoint_confirmation_path = endpoint_root / "targeted_count_analysis_confirmation/audit.json"
    endpoint_confirmation = None
    if endpoint_status in {"PASS", "CONFIRMATION_NEGATIVE"}:
        endpoint_confirmation = _load(endpoint_confirmation_path)
        _audit_seed_contract(
            endpoint_confirmation,
            phase="endpoint confirmation",
            expected=10,
            applicable=False,
        )
    elif endpoint_confirmation_path.exists():
        raise ValueError("Endpoint confirmation opened after discovery failure")

    bridge_root = _resolve(root, str(stages[1]["root"]))
    bridge_complete_path = bridge_root / "restoration_complete.json"
    if endpoint_status != "PASS":
        if bridge_complete_path.exists() or (
            bridge_root / "restoration_discovery"
        ).exists():
            raise ValueError("Bridge opened before endpoint PASS")
        early_paths = [endpoint_complete_path, endpoint_discovery_path]
        if endpoint_confirmation is not None:
            early_paths.append(endpoint_confirmation_path)
        return {
            "schema_version": "realistic_niah_v5_bank_extension_complete_v1",
            "model_label": model,
            "bank_size": expected_k,
            "selected_bank_sha256": expected_sha,
            "status": "PROTOCOL_EXHAUSTED",
            "endpoint_status": endpoint_status,
            "bridge_status": "NOT_OPENED",
            "selection_rank_used": False,
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_sha256": {
                str(path.relative_to(root)): _sha(path) for path in early_paths
            },
        }

    bridge_complete = _load(bridge_complete_path)
    bridge_status = str(bridge_complete.get("status"))
    if bridge_status not in BRIDGE_TERMINAL:
        raise ValueError(f"Bridge is not terminal: {bridge_status}")
    if str(bridge_complete.get("model_label")) != model:
        raise ValueError("Bridge model mismatch")
    if int(bridge_complete.get("targeted_bank_size", -1)) != expected_k:
        raise ValueError("Bridge bank size does not match endpoint")
    if str(bridge_complete.get("targeted_bank_sha256")) != expected_sha:
        raise ValueError("Bridge bank hash does not match endpoint")
    bridge_discovery_path = bridge_root / "restoration_analysis_discovery/audit.json"
    bridge_discovery = _load(bridge_discovery_path)
    _audit_seed_contract(
        bridge_discovery, phase="bridge discovery", expected=20, applicable=True
    )
    bridge_confirmation_path = bridge_root / "restoration_analysis_confirmation/audit.json"
    bridge_confirmation = None
    if bridge_status in {"PASS", "CONFIRMATION_GATE_FAIL"}:
        bridge_confirmation = _load(bridge_confirmation_path)
        _audit_seed_contract(
            bridge_confirmation,
            phase="bridge confirmation",
            expected=10,
            applicable=True,
        )
    elif bridge_confirmation_path.exists():
        raise ValueError("Bridge confirmation opened after discovery failure")

    source_paths = [
        endpoint_complete_path,
        endpoint_discovery_path,
        endpoint_confirmation_path,
        bridge_complete_path,
        bridge_discovery_path,
    ]
    if bridge_confirmation is not None:
        source_paths.append(bridge_confirmation_path)
    passed = endpoint_status == "PASS" and bridge_status == "PASS"
    return {
        "schema_version": "realistic_niah_v5_bank_extension_complete_v1",
        "model_label": model,
        "bank_size": expected_k,
        "selected_bank_sha256": expected_sha,
        "status": "PASS" if passed else "PROTOCOL_EXHAUSTED",
        "endpoint_status": endpoint_status,
        "bridge_status": bridge_status,
        "selection_rank_used": False,
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_sha256": {
            str(path.relative_to(root)): _sha(path) for path in source_paths
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = _load(args.protocol.resolve())
    result = finalize(args.root, protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
