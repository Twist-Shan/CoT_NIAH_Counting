#!/usr/bin/env python3
"""Audit a terminal prospective endpoint before opening its same-bank bridge."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


TERMINAL = {"PASS", "DISCOVERY_NEGATIVE", "CONFIRMATION_NEGATIVE"}
PLAN_AUDIT_TERMINAL = {"PASS", "FROZEN_NEW", "REUSED_IDENTICAL"}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _audit_phase(path: Path, *, expected: int, label: str) -> dict[str, Any]:
    audit = _load(path)
    if audit.get("status") != "PASS":
        raise ValueError(f"{label} audit is not PASS")
    if int(audit.get("seed_count", -1)) != expected:
        raise ValueError(f"{label} seed contract changed")
    if audit.get("selection_rank_used") is not False:
        raise ValueError(f"{label} used selection_rank")
    return audit


def audit(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    root = root.resolve()
    model = str(protocol["model_label"])
    stages = protocol["stages"]
    if [stage["name"] for stage in stages] != [
        "targeted_retrieval_to_final_count",
        "targeted_retrieval_to_terminal_state_to_readout",
    ]:
        raise ValueError("Prospective stage order changed")
    expected_discovery = list(range(1234, 1254))
    expected_confirmation = list(range(1254, 1264))
    if list(stages[0].get("discovery_seeds", [])) != expected_discovery:
        raise ValueError("Prospective discovery seeds changed")
    if list(stages[0].get("confirmation_seeds", [])) != expected_confirmation:
        raise ValueError("Prospective confirmation seeds changed")

    expected_k = int(protocol["bank"]["size"])
    expected_sha = str(protocol["bank"]["selected_bank_sha256"])
    endpoint_root = _resolve(root, str(stages[0]["root"]))
    complete_path = endpoint_root / "targeted_count_complete.json"
    complete = _load(complete_path)
    endpoint_status = str(complete.get("status"))
    if endpoint_status not in TERMINAL:
        raise ValueError(f"Endpoint is not terminal: {endpoint_status}")

    plan = endpoint_root / "frozen_targeted_count_plan.csv"
    with plan.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        rows = list(reader)
    if "selection_rank" in fields:
        raise ValueError("Endpoint plan contains selection_rank")
    selected = [row for row in rows if row.get("condition") == "selected_bank"]
    if len(selected) != 1:
        raise ValueError("Endpoint plan must contain one selected_bank row")
    observed = (int(selected[0]["bank_size"]), str(selected[0]["bank_sha256"]))
    if observed != (expected_k, expected_sha):
        raise ValueError("Endpoint plan bank signature changed")
    plan_audit_path = plan.with_suffix(".audit.json")
    plan_audit = _load(plan_audit_path)
    plan_audit_status = str(plan_audit.get("status"))
    if plan_audit_status not in PLAN_AUDIT_TERMINAL:
        raise ValueError("Endpoint plan audit is not terminal")
    if plan_audit.get("selection_rank_used") is not False:
        raise ValueError("Endpoint plan audit used selection_rank")
    if plan_audit_status != "PASS":
        if plan_audit.get("outcome_blind") is not True:
            raise ValueError("Endpoint plan audit is not outcome-blind")
        if plan_audit.get("historical_artifacts_modified") is not False:
            raise ValueError("Endpoint plan audit modified historical artifacts")
        if str(plan_audit.get("model_label")) != model:
            raise ValueError("Endpoint plan audit model changed")
        if int(plan_audit.get("bank_size", -1)) != expected_k:
            raise ValueError("Endpoint plan audit bank size changed")
        if str(plan_audit.get("selected_bank_sha256")) != expected_sha:
            raise ValueError("Endpoint plan audit bank hash changed")

    discovery_path = endpoint_root / "targeted_count_analysis_discovery/audit.json"
    _audit_phase(discovery_path, expected=20, label="endpoint discovery")
    confirmation_path = endpoint_root / "targeted_count_analysis_confirmation/audit.json"
    sources = [complete_path, plan, plan_audit_path, discovery_path]
    if endpoint_status in {"PASS", "CONFIRMATION_NEGATIVE"}:
        _audit_phase(confirmation_path, expected=10, label="endpoint confirmation")
        sources.append(confirmation_path)
    elif confirmation_path.exists():
        raise ValueError("Endpoint confirmation opened after discovery failure")

    if endpoint_status == "PASS":
        discovery_claims = complete.get("discovery", {})
        confirmation_claims = complete.get("confirmation", {})
        if discovery_claims.get("targeted_to_count_pass") is not True:
            raise ValueError("PASS endpoint lacks a passing discovery gate")
        if confirmation_claims.get("targeted_to_count_pass") is not True:
            raise ValueError("PASS endpoint lacks a passing confirmation gate")

    return {
        "schema_version": "realistic_niah_v5_prospective_endpoint_gate_v1",
        "model_label": model,
        "bank_size": expected_k,
        "selected_bank_sha256": expected_sha,
        "endpoint_status": endpoint_status,
        "status": (
            "BRIDGE_ELIGIBLE" if endpoint_status == "PASS" else "PROTOCOL_EXHAUSTED"
        ),
        "discovery_seed_count": 20,
        "confirmation_seed_count": 10 if endpoint_status != "DISCOVERY_NEGATIVE" else 0,
        "selection_rank_used": False,
        "audited_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_sha256": {
            str(path.relative_to(root)): _sha(path) for path in sources
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, _load(args.protocol.resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
