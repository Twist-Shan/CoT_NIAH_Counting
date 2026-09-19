#!/usr/bin/env python3
"""Assemble and audit the final native-thinking count-chain evidence root."""

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

import finalize_realistic_niah_v5_integrated_branch_ledger as ledger


MODELS = ("Qwen3-8B", "Gemma4-E4B")


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


def _audit_component(
    *,
    model: str,
    label: str,
    complete_path: Path,
    discovery_path: Path,
    confirmation_path: Path,
) -> None:
    complete = _load(complete_path)
    if complete.get("status") != "PASS":
        raise ValueError(f"{model} {label} complete is not PASS")
    model_label = complete.get("model_label")
    if model_label is not None and str(model_label) != model:
        raise ValueError(f"{model} {label} complete has model {model_label}")
    discovery = _load(discovery_path)
    confirmation = _load(confirmation_path)
    discovery = _add_supplemental_rank_proof(
        discovery, discovery_path, expected_seeds=20, label=f"{model} {label} discovery"
    )
    confirmation = _add_supplemental_rank_proof(
        confirmation,
        confirmation_path,
        expected_seeds=10,
        label=f"{model} {label} confirmation",
    )
    ledger._validate_audit(
        discovery, expected_seeds=20, branch=f"{model}:{label}", phase="discovery"
    )
    ledger._validate_audit(
        confirmation,
        expected_seeds=10,
        branch=f"{model}:{label}",
        phase="confirmation",
    )


def _add_supplemental_rank_proof(
    audit: dict[str, Any], audit_path: Path, *, expected_seeds: int, label: str
) -> dict[str, Any]:
    if audit.get("selection_rank_used") is False:
        return audit
    if audit.get("selection_rank_used") is True:
        raise ValueError(f"{label} used selection_rank")
    proof_path = audit_path.parent / "selection_rank_audit.json"
    if not proof_path.exists():
        raise ValueError(f"{label} lacks selection-rank proof")
    proof = _load(proof_path)
    if proof.get("status") != "PASS":
        raise ValueError(f"{label} supplemental selection-rank audit is not PASS")
    if proof.get("selection_rank_used") is not False:
        raise ValueError(f"{label} supplemental audit found selection_rank")
    if int(proof.get("seed_count", -1)) != int(expected_seeds):
        raise ValueError(f"{label} supplemental audit seed contract changed")
    supplemented = dict(audit)
    supplemented["selection_rank_used"] = False
    supplemented["selection_rank_proof"] = str(proof_path)
    supplemented["selection_rank_proof_sha256"] = _sha(proof_path)
    return supplemented


def _component_paths(root: Path, model: str) -> dict[str, tuple[Path, Path, Path]]:
    targeted = (
        root
        / "work/v5_native_count_stream/targeted_count_chain_20d10c_20260821"
        / model
    )
    if model == "Qwen3-8B":
        readout = (
            root
            / "work/v5_native_count_stream/complementary_readout_20d10c_20260821"
            / model
        )
        readout_paths = (
            readout / "complementary_complete.json",
            readout / "complementary_analysis_discovery/audit.json",
            readout / "complementary_analysis_confirmation/audit.json",
        )
    else:
        readout = (
            root
            / "work/v5_native_count_stream/serial_source_persistent_20d10c_20260821"
            / model
        )
        readout_paths = (
            readout / "serial_source_complete.json",
            readout / "serial_source_discovery_analysis/audit.json",
            readout / "serial_source_confirmation_analysis/audit.json",
        )
    return {
        "targeted": (
            targeted / "targeted_count_complete.json",
            targeted / "targeted_count_analysis_discovery/audit.json",
            targeted / "targeted_count_analysis_confirmation/audit.json",
        ),
        "readout": readout_paths,
    }


def _targeted_plan_metadata(root: Path, model: str) -> tuple[Path, dict[str, Any]]:
    path = (
        root
        / "work/v5_native_count_stream/targeted_count_chain_20d10c_20260821"
        / model
        / "frozen_targeted_count_plan.csv"
    )
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        rows = list(reader)
    if not rows:
        raise ValueError(f"{model} targeted plan is empty")
    if "selection_rank" in fields:
        raise ValueError(f"{model} targeted plan contains selection_rank")
    bank_sizes = sorted({int(row["bank_size"]) for row in rows})
    if len(bank_sizes) != 1 or bank_sizes[0] <= 0:
        raise ValueError(f"{model} targeted plan has ambiguous bank sizes: {bank_sizes}")
    return path, {
        "schema_version": "realistic_niah_v5_targeted_plan_metadata_v1",
        "model_label": model,
        "bank_size": bank_sizes[0],
        "plan_row_count": len(rows),
        "selection_rank_used": False,
        "frozen_targeted_count_plan_sha256": _sha(path),
    }


def _branch_roots(root: Path, model: str) -> list[tuple[str, Path, str | None]]:
    exact_parent = (
        "integrated_serial_bridge_20d10c_20260821_rerun1"
        if model == "Qwen3-8B"
        else "integrated_serial_bridge_20d10c_20260821"
    )
    stream = root / "work/v5_native_count_stream"
    return [
        ("exact_query_transfer", stream / exact_parent / model, "suffix8"),
        (
            "persistent_transfer",
            stream / "integrated_serial_bridge_persistent_20d10c_20260821" / model,
            "suffix8",
        ),
        (
            "suffix8_restoration",
            stream / "integrated_mediator_restoration_20d10c_20260821_rerun1" / model,
            "suffix8",
        ),
        (
            "fullspan_restoration",
            stream / "integrated_mediator_restoration_fullspan_20d10c_20260821" / model,
            "full_span",
        ),
    ]


def _branch_spec(root: Path, model: str) -> dict[str, Any]:
    branches: list[dict[str, Any]] = []
    for name, branch_root, geometry in _branch_roots(root, model):
        restoration = "restoration" in name
        prefix = "restoration" if restoration else "integrated_bridge"
        complete = branch_root / f"{prefix}_complete.json"
        discovery = branch_root / f"{prefix}_analysis_discovery/audit.json"
        confirmation = branch_root / f"{prefix}_analysis_confirmation/audit.json"
        value: dict[str, Any] = {
            "name": name,
            "complete": str(complete),
            "discovery_audit": str(discovery),
            "confirmation_audit": str(confirmation) if confirmation.exists() else None,
        }
        if geometry is not None:
            value["mediator_geometry"] = geometry
        branches.append(value)
    return {"model_label": model, "branches": branches}


def assemble(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    source_hashes: dict[str, str] = {}
    outcomes: dict[str, str] = {}
    for model in MODELS:
        target = output / model
        components = _component_paths(root, model)
        for label, paths in components.items():
            complete, discovery, confirmation = paths
            _audit_component(
                model=model,
                label=label,
                complete_path=complete,
                discovery_path=discovery,
                confirmation_path=confirmation,
            )
            _copy(complete, target / f"{label}_complete.json")
            _copy(discovery, target / f"{label}_discovery_audit.json")
            _copy(confirmation, target / f"{label}_confirmation_audit.json")
            for path in paths:
                source_hashes[str(path.relative_to(root))] = _sha(path)
            for phase, audit_path in (
                ("discovery", discovery),
                ("confirmation", confirmation),
            ):
                proof = audit_path.parent / "selection_rank_audit.json"
                if proof.exists():
                    _copy(
                        proof,
                        target / f"{label}_{phase}_selection_rank_audit.json",
                    )
                    source_hashes[str(proof.relative_to(root))] = _sha(proof)

        targeted_plan, targeted_plan_meta = _targeted_plan_metadata(root, model)
        _atomic_json(target / "targeted_plan_meta.json", targeted_plan_meta)
        source_hashes[str(targeted_plan.relative_to(root))] = _sha(targeted_plan)

        spec = _branch_spec(root, model)
        integrated, discovery, confirmation = ledger.finalize(spec, base=root)
        _atomic_json(target / "integrated_complete.json", integrated)
        _atomic_json(target / "integrated_discovery_audit.json", discovery)
        if confirmation is not None:
            _atomic_json(target / "integrated_confirmation_audit.json", confirmation)
        _atomic_json(
            target / "branch_ledger.json",
            {
                "schema_version": integrated["schema_version"],
                "model_label": model,
                "status": integrated["status"],
                "branch_outcomes": integrated["branch_outcomes"],
            },
        )
        _atomic_json(target / "branch_spec.json", spec)
        outcomes[model] = str(integrated["status"])
        for branch in spec["branches"]:
            for key in ("complete", "discovery_audit", "confirmation_audit"):
                if branch[key] is not None:
                    path = Path(branch[key])
                    source_hashes[str(path.relative_to(root))] = _sha(path)

    manifest = {
        "schema_version": "realistic_niah_v5_count_chain_evidence_v1",
        "assembled_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_root": str(root),
        "output_root": str(output),
        "models": outcomes,
        "source_sha256": dict(sorted(source_hashes.items())),
        "status": "PASS",
    }
    _atomic_json(output / "evidence_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.root, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
