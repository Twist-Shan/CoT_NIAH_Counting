"""Validate frozen causal-chain evidence and export machine-readable snapshots."""

from __future__ import annotations

import hashlib

import json

from pathlib import Path

from typing import Any

MODELS = ("Qwen3-8B", "Gemma4-E4B")

def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _read_evidence(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    values: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    claim_contract = root / "claim_contract.json"
    if claim_contract.exists():
        values["claim_contract"] = _load(claim_contract)
        hashes[str(claim_contract.relative_to(root))] = _sha(claim_contract)
    prospective_manifest = root / "prospective_evidence_manifest.json"
    if prospective_manifest.exists():
        values["prospective_evidence_manifest"] = _load(prospective_manifest)
        hashes[str(prospective_manifest.relative_to(root))] = _sha(
            prospective_manifest
        )
    metadata_fix = root / "metadata_fix_ledger.json"
    if metadata_fix.exists():
        values["metadata_fix_ledger"] = _load(metadata_fix)
        hashes[str(metadata_fix.relative_to(root))] = _sha(metadata_fix)
    for model in MODELS:
        for kind in ("targeted", "readout", "integrated"):
            path = root / model / f"{kind}_complete.json"
            if not path.exists():
                raise FileNotFoundError(path)
            values[f"{model}:{kind}"] = _load(path)
            hashes[str(path.relative_to(root))] = _sha(path)
        plan_meta = root / model / "targeted_plan_meta.json"
        if not plan_meta.exists():
            raise FileNotFoundError(plan_meta)
        values[f"{model}:targeted_plan_meta"] = _load(plan_meta)
        hashes[str(plan_meta.relative_to(root))] = _sha(plan_meta)
        extension = root / model / "prospective_extension_complete.json"
        if extension.exists():
            values[f"{model}:prospective_extension"] = _load(extension)
            hashes[str(extension.relative_to(root))] = _sha(extension)
        extension_protocol = root / model / "prospective_extension_protocol.json"
        if extension_protocol.exists():
            hashes[str(extension_protocol.relative_to(root))] = _sha(
                extension_protocol
            )
        for phase in ("discovery", "confirmation"):
            path = root / model / f"integrated_{phase}_audit.json"
            if path.exists():
                values[f"{model}:integrated_{phase}_audit"] = _load(path)
                hashes[str(path.relative_to(root))] = _sha(path)
    return values, hashes

def _assert_contract(evidence: dict[str, Any]) -> None:
    claim_contract = evidence.get("claim_contract")
    if claim_contract is not None:
        if not str(claim_contract.get("status", "")).startswith("FROZEN_"):
            raise ValueError("Prospective claim contract is not frozen")
        shared = claim_contract["shared_protocol"]
        if list(shared.get("discovery_seeds", [])) != list(range(1234, 1254)):
            raise ValueError("Claim-contract discovery seeds changed")
        if list(shared.get("confirmation_seeds", [])) != list(range(1254, 1264)):
            raise ValueError("Claim-contract confirmation seeds changed")
        if shared.get("outcome_blind") is not True:
            raise ValueError("Claim contract is not outcome-blind")
        if shared.get("selection_rank_used") is not False:
            raise ValueError("Claim contract used selection_rank")
        metadata_fix = evidence.get("metadata_fix_ledger")
        prospective_manifest = evidence.get("prospective_evidence_manifest")
        if metadata_fix is None or prospective_manifest is None:
            raise ValueError("Prospective evidence lacks metadata-fix provenance")
        expected_status = (
            "RESULT_INDEPENDENT_METADATA_ONLY_FIX_BEFORE_ANY_PROSPECTIVE_BRIDGE_RUN"
        )
        if metadata_fix.get("status") != expected_status:
            raise ValueError("Metadata-fix provenance is not frozen")
        if metadata_fix.get("fix") != {
            "Qwen3-8B": "post_marker",
            "Gemma4-E4B": "p0_item_end",
        }:
            raise ValueError("Metadata-fix anchor-role mapping changed")
        source_hashes = prospective_manifest.get("source_sha256", {})
        source_path = str(metadata_fix.get("file"))
        if source_hashes.get(source_path) != metadata_fix.get("new_sha256"):
            raise ValueError("Metadata-fix source hash is absent or stale")
        ledger_path = (
            "configs/realistic_niah_v5_integrated_bridge_metadata_fix_v1.json"
        )
        if ledger_path not in source_hashes:
            raise ValueError("Metadata-fix ledger source hash is absent")
    for model in MODELS:
        targeted = evidence[f"{model}:targeted"]
        targeted_plan = evidence[f"{model}:targeted_plan_meta"]
        readout = evidence[f"{model}:readout"]
        integrated = evidence[f"{model}:integrated"]
        if targeted.get("status") != "PASS":
            raise ValueError(f"{model} targeted endpoint is not PASS")
        if str(targeted_plan.get("model_label")) != model:
            raise ValueError(f"{model} targeted plan metadata has a model mismatch")
        if int(targeted_plan.get("bank_size", 0)) <= 0:
            raise ValueError(f"{model} targeted plan metadata lacks a valid bank size")
        if targeted_plan.get("selection_rank_used") is not False:
            raise ValueError(f"{model} targeted plan metadata used selection_rank")
        if readout.get("status") != "PASS":
            raise ValueError(f"{model} readout is not PASS")
        extension = evidence.get(f"{model}:prospective_extension")
        if extension is not None:
            if claim_contract is None:
                raise ValueError(f"{model} prospective extension lacks claim contract")
            contract_model = claim_contract["models"][model]
            contract_signature = (
                int(contract_model["prospective_bank_size"]),
                str(contract_model["selected_bank_sha256"]),
            )
            extension_signature = (
                int(extension.get("bank_size", -1)),
                str(extension.get("selected_bank_sha256")),
            )
            if contract_signature != extension_signature:
                raise ValueError(f"{model} prospective extension violates claim contract")
            if str(extension.get("model_label")) != model:
                raise ValueError(f"{model} prospective extension model mismatch")
            if extension.get("selection_rank_used") is not False:
                raise ValueError(f"{model} prospective extension used selection_rank")
            extension_status = str(extension.get("status"))
            if extension_status not in {"PASS", "PROTOCOL_EXHAUSTED"}:
                raise ValueError(
                    f"{model} prospective extension is not terminal: {extension_status}"
                )
            if extension_status == "PASS":
                if extension.get("endpoint_status") != "PASS":
                    raise ValueError(f"{model} PASS extension endpoint is not PASS")
                if extension.get("bridge_status") != "PASS":
                    raise ValueError(f"{model} PASS extension bridge is not PASS")
                if int(extension.get("bank_size", -1)) != int(
                    targeted_plan["bank_size"]
                ):
                    raise ValueError(f"{model} prospective bank size is not primary")
                expected_sha = extension.get("selected_bank_sha256")
                observed_sha = targeted_plan.get("selected_bank_sha256")
                if observed_sha != expected_sha:
                    raise ValueError(f"{model} prospective bank hash is not primary")
        bridge_flags = (
            bool(integrated.get("integrated_serial_bridge_pass")),
            bool(integrated.get("integrated_mediator_restoration_pass")),
        )
        status = str(integrated.get("status"))
        if status == "PASS" and sum(bridge_flags) != 1:
            raise ValueError(
                f"{model} must pass exactly one registered integrated bridge"
            )
        if status == "PRE_REGISTERED_BRANCHES_EXHAUSTED":
            if sum(bridge_flags) != 0:
                raise ValueError(f"{model} exhausted ledger cannot contain a pass")
            if integrated.get("pre_registered_branches_exhausted") is not True:
                raise ValueError(f"{model} lacks an exhausted-branch audit")
            outcomes = integrated.get("branch_outcomes")
            expected_names = [
                "exact_query_transfer",
                "persistent_transfer",
                "suffix8_restoration",
                "fullspan_restoration",
            ]
            if not isinstance(outcomes, list) or [
                str(value.get("name")) for value in outcomes
            ] != expected_names:
                raise ValueError(f"{model} branch ledger is incomplete")
            if any(str(value.get("status")) == "PASS" for value in outcomes):
                raise ValueError(f"{model} exhausted ledger contains a PASS branch")
        elif status != "PASS":
            raise ValueError(f"{model} integrated outcome is not terminal: {status}")
        required_phases = [("discovery", 20)]
        if status == "PASS":
            required_phases.append(("confirmation", 10))
        for phase, expected in required_phases:
            key = f"{model}:integrated_{phase}_audit"
            if key not in evidence:
                raise ValueError(f"{model} lacks integrated {phase} audit")
            audit = evidence[key]
            if int(audit["seed_count"]) != expected:
                raise ValueError(f"{model} {phase} seed contract changed")
            if int(audit.get("applicable_seed_count", -1)) != expected:
                raise ValueError(f"{model} {phase} effective seed contract changed")
            if audit.get("selection_rank_used") is not False:
                raise ValueError(f"{model} {phase} used selection_rank")


def build(evidence: dict, evidence_hashes: dict) -> str:
    """Serialize audited evidence without rendering a narrative report."""
    _assert_contract(evidence)
    return json.dumps({"schema_version": "native_count_chain_evidence_snapshot_v1",
                       "evidence": evidence, "evidence_sha256": evidence_hashes},
                      indent=2, ensure_ascii=True) + "\n"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Audited evidence snapshot (.json)")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".json":
        parser.error("--output must have a .json suffix")
    evidence, hashes = _read_evidence(args.evidence_root)
    document = build(evidence, hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(args.output)
    manifest = {"schema_version": "native_count_chain_evidence_manifest_v1",
                "output": str(args.output), "output_sha256": _sha(args.output),
                "evidence_sha256": hashes, "status": "PASS"}
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
