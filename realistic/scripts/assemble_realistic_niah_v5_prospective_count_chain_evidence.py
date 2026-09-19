#!/usr/bin/env python3
"""Assemble terminal prospective bank extensions over immutable historical evidence."""

from __future__ import annotations

import argparse
import csv
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

import finalize_realistic_niah_v5_bank_extension as bank_extension


MODELS = ("Qwen3-8B", "Gemma4-E4B")
PLAN_AUDIT_TERMINAL = {"PASS", "FROZEN_NEW", "REUSED_IDENTICAL"}
CLAIM_CONTRACT = Path(
    "configs/realistic_niah_v5_prospective_count_chain_claim_contract_v1.json"
)
METADATA_FIX_LEDGER = Path(
    "configs/realistic_niah_v5_integrated_bridge_metadata_fix_v1.json"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(
        path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def _copy(source: Path, target: Path) -> None:
    _atomic_bytes(target, source.read_bytes())


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _repo_key(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _copy_historical_evidence(source: Path, output: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    if output == source or source in output.parents:
        raise ValueError("Output must not be inside the historical evidence root")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Prospective evidence output must be new or empty")
    for path in source.rglob("*"):
        if path.is_file():
            _copy(path, output / path.relative_to(source))


def _validate_claim_contract(
    contract: dict[str, Any], protocols: dict[str, tuple[Path, dict[str, Any]]]
) -> None:
    if not str(contract.get("status", "")).startswith("FROZEN_"):
        raise ValueError("Prospective claim contract is not frozen")
    shared = contract["shared_protocol"]
    if list(shared.get("discovery_seeds", [])) != list(range(1234, 1254)):
        raise ValueError("Claim-contract discovery seeds changed")
    if list(shared.get("confirmation_seeds", [])) != list(range(1254, 1264)):
        raise ValueError("Claim-contract confirmation seeds changed")
    if shared.get("outcome_blind") is not True:
        raise ValueError("Claim contract is not outcome-blind")
    if shared.get("selection_rank_used") is not False:
        raise ValueError("Claim contract used selection_rank")
    for model in MODELS:
        protocol = protocols[model][1]
        contract_model = contract["models"][model]
        observed = (
            int(contract_model["prospective_bank_size"]),
            str(contract_model["selected_bank_sha256"]),
        )
        expected = (
            int(protocol["bank"]["size"]),
            str(protocol["bank"]["selected_bank_sha256"]),
        )
        if observed != expected:
            raise ValueError(f"{model} claim-contract bank signature changed")


def _validate_metadata_fix_ledger(
    repo_root: Path, ledger: dict[str, Any]
) -> Path:
    expected_status = (
        "RESULT_INDEPENDENT_METADATA_ONLY_FIX_BEFORE_ANY_PROSPECTIVE_BRIDGE_RUN"
    )
    if ledger.get("status") != expected_status:
        raise ValueError("Integrated-bridge metadata fix is not frozen")
    if ledger.get("fix") != {
        "Qwen3-8B": "post_marker",
        "Gemma4-E4B": "p0_item_end",
    }:
        raise ValueError("Integrated-bridge targeted anchor-role mapping changed")
    timing = ledger.get("timing_and_blinding", {})
    false_fields = (
        "Qwen3-8B_prospective_bridge_started",
        "Gemma4-E4B_prospective_bridge_started",
        "prospective_endpoint_aggregate_effects_read",
        "prospective_endpoint_claim_gates_read",
    )
    if any(timing.get(field) is not False for field in false_fields):
        raise ValueError("Metadata fix was not recorded before prospective results")
    source = _resolve(repo_root, str(ledger["file"])).resolve()
    if not source.is_relative_to(repo_root):
        raise ValueError("Metadata-fix source is outside the repository")
    if _sha(source) != str(ledger.get("new_sha256")):
        raise ValueError("Integrated-bridge source does not match metadata-fix ledger")
    if str(ledger.get("old_sha256")) == str(ledger.get("new_sha256")):
        raise ValueError("Metadata-fix ledger does not record a source transition")
    return source


def _plan_metadata(
    endpoint_root: Path,
    *,
    model: str,
    expected_k: int,
    expected_sha: str,
) -> tuple[Path, Path, dict[str, Any]]:
    plan = endpoint_root / "frozen_targeted_count_plan.csv"
    with plan.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        rows = list(reader)
    if "selection_rank" in fields:
        raise ValueError(f"{model} prospective plan contains selection_rank")
    selected = [row for row in rows if row.get("condition") == "selected_bank"]
    if len(selected) != 1:
        raise ValueError(f"{model} prospective plan lacks one selected_bank row")
    observed = (int(selected[0]["bank_size"]), str(selected[0]["bank_sha256"]))
    if observed != (expected_k, expected_sha):
        raise ValueError(f"{model} prospective plan bank signature changed")
    audit_path = plan.with_suffix(".audit.json")
    audit = _load(audit_path)
    audit_status = str(audit.get("status"))
    if audit_status not in PLAN_AUDIT_TERMINAL:
        raise ValueError(f"{model} prospective plan audit is not terminal")
    if audit.get("selection_rank_used") is not False:
        raise ValueError(f"{model} prospective plan audit used selection_rank")
    if audit_status != "PASS":
        if audit.get("outcome_blind") is not True:
            raise ValueError(f"{model} prospective plan is not outcome-blind")
        if audit.get("historical_artifacts_modified") is not False:
            raise ValueError(f"{model} prospective plan modified historical artifacts")
        if str(audit.get("model_label")) != model:
            raise ValueError(f"{model} prospective plan audit model changed")
        if int(audit.get("bank_size", -1)) != expected_k:
            raise ValueError(f"{model} prospective plan audit bank size changed")
        if str(audit.get("selected_bank_sha256")) != expected_sha:
            raise ValueError(f"{model} prospective plan audit bank hash changed")
    return plan, audit_path, {
        "schema_version": "realistic_niah_v5_targeted_plan_metadata_v2",
        "model_label": model,
        "bank_size": expected_k,
        "selected_bank_sha256": expected_sha,
        "plan_row_count": len(rows),
        "selection_rank_used": False,
        "frozen_targeted_count_plan_sha256": _sha(plan),
        "frozen_targeted_count_plan_audit_sha256": _sha(audit_path),
    }


def _overlay_confirmed_extension(
    *,
    repo_root: Path,
    output: Path,
    protocol: dict[str, Any],
    result: dict[str, Any],
    source_hashes: dict[str, str],
) -> None:
    model = str(protocol["model_label"])
    target = output / model
    endpoint_root = _resolve(repo_root, str(protocol["stages"][0]["root"]))
    bridge_root = _resolve(repo_root, str(protocol["stages"][1]["root"]))
    expected_k = int(result["bank_size"])
    expected_sha = str(result["selected_bank_sha256"])
    plan, plan_audit, plan_meta = _plan_metadata(
        endpoint_root,
        model=model,
        expected_k=expected_k,
        expected_sha=expected_sha,
    )

    replacements = {
        endpoint_root / "targeted_count_complete.json": target
        / "targeted_complete.json",
        endpoint_root / "targeted_count_analysis_discovery/audit.json": target
        / "targeted_discovery_audit.json",
        endpoint_root / "targeted_count_analysis_confirmation/audit.json": target
        / "targeted_confirmation_audit.json",
        bridge_root / "restoration_complete.json": target / "integrated_complete.json",
        bridge_root / "restoration_analysis_discovery/audit.json": target
        / "integrated_discovery_audit.json",
        bridge_root / "restoration_analysis_confirmation/audit.json": target
        / "integrated_confirmation_audit.json",
    }
    bridge_complete = _load(bridge_root / "restoration_complete.json")
    if bridge_complete.get("status") != "PASS":
        raise ValueError(f"{model} confirmed extension bridge is not PASS")
    if bridge_complete.get("integrated_mediator_restoration_pass") is not True:
        raise ValueError(f"{model} PASS bridge lacks mediator-restoration pass")
    for source, destination in replacements.items():
        _copy(source, destination)
        source_hashes[_repo_key(repo_root, source)] = _sha(source)
    _atomic_json(target / "targeted_plan_meta.json", plan_meta)
    source_hashes[_repo_key(repo_root, plan)] = _sha(plan)
    source_hashes[_repo_key(repo_root, plan_audit)] = _sha(plan_audit)
    _atomic_json(
        target / "branch_ledger.json",
        {
            "schema_version": "realistic_niah_v5_prospective_branch_ledger_v1",
            "model_label": model,
            "status": "PASS",
            "branch_outcomes": [
                {
                    "name": "prospective_same_bank_fullspan_restoration",
                    "status": "PASS",
                    "bank_size": expected_k,
                    "selected_bank_sha256": expected_sha,
                }
            ],
        },
    )


def assemble(
    *,
    repo_root: Path,
    historical_evidence_root: Path,
    protocol_paths: list[Path],
    output: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    historical_evidence_root = historical_evidence_root.resolve()
    output = output.resolve()
    historical_manifest = _load(historical_evidence_root / "evidence_manifest.json")
    if historical_manifest.get("status") != "PASS":
        raise ValueError("Historical evidence manifest is not PASS")

    protocols: dict[str, tuple[Path, dict[str, Any]]] = {}
    for supplied in protocol_paths:
        path = supplied if supplied.is_absolute() else repo_root / supplied
        path = path.resolve()
        protocol = _load(path)
        model = str(protocol.get("model_label"))
        if model not in MODELS:
            raise ValueError(f"Unknown extension model: {model}")
        if model in protocols:
            raise ValueError(f"Duplicate extension protocol for {model}")
        protocols[model] = (path, protocol)
    if set(protocols) != set(MODELS):
        raise ValueError("Prospective evidence requires one terminal protocol per model")

    claim_contract_path = (repo_root / CLAIM_CONTRACT).resolve()
    claim_contract = _load(claim_contract_path)
    _validate_claim_contract(claim_contract, protocols)
    metadata_fix_path = (repo_root / METADATA_FIX_LEDGER).resolve()
    metadata_fix = _load(metadata_fix_path)
    metadata_fix_source = _validate_metadata_fix_ledger(repo_root, metadata_fix)

    terminal_results: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        _protocol_path, protocol = protocols[model]
        terminal_results[model] = bank_extension.finalize(repo_root, protocol)

    _copy_historical_evidence(historical_evidence_root, output)
    _copy(claim_contract_path, output / "claim_contract.json")
    _copy(metadata_fix_path, output / "metadata_fix_ledger.json")
    source_hashes: dict[str, str] = {
        str((historical_evidence_root / "evidence_manifest.json")):
        _sha(historical_evidence_root / "evidence_manifest.json"),
        _repo_key(repo_root, claim_contract_path): _sha(claim_contract_path),
        _repo_key(repo_root, metadata_fix_path): _sha(metadata_fix_path),
        _repo_key(repo_root, metadata_fix_source): _sha(metadata_fix_source),
    }
    outcomes: dict[str, str] = {}
    for model in MODELS:
        protocol_path, protocol = protocols[model]
        result = terminal_results[model]
        target = output / model
        _atomic_json(target / "prospective_extension_complete.json", result)
        _copy(protocol_path, target / "prospective_extension_protocol.json")
        source_hashes[_repo_key(repo_root, protocol_path)] = _sha(protocol_path)
        outcomes[model] = str(result["status"])
        if result["status"] == "PASS":
            _overlay_confirmed_extension(
                repo_root=repo_root,
                output=output,
                protocol=protocol,
                result=result,
                source_hashes=source_hashes,
            )

    manifest = {
        "schema_version": "realistic_niah_v5_prospective_count_chain_evidence_v1",
        "assembled_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "historical_evidence_root": str(historical_evidence_root),
        "output_root": str(output),
        "prospective_extensions": outcomes,
        "source_sha256": dict(sorted(source_hashes.items())),
        "status": "PASS",
    }
    _atomic_json(output / "prospective_evidence_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--historical-evidence-root", type=Path, required=True)
    parser.add_argument(
        "--extension-protocol", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assemble(
        repo_root=args.repo_root,
        historical_evidence_root=args.historical_evidence_root,
        protocol_paths=args.extension_protocol,
        output=args.output,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
