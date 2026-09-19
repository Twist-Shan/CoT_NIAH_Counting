#!/usr/bin/env python3
"""Finalize the pre-registered integrated-bridge branch ledger."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


EXPECTED_BRANCHES = (
    "exact_query_transfer",
    "persistent_transfer",
    "suffix8_restoration",
    "fullspan_restoration",
)
TERMINAL_FAILURES = {"DISCOVERY_GATE_FAIL", "CONFIRMATION_GATE_FAIL"}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _resolve(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def _validate_audit(
    audit: dict[str, Any], *, expected_seeds: int, branch: str, phase: str
) -> None:
    if audit.get("status") != "PASS":
        raise ValueError(f"{branch} {phase} audit is not PASS")
    if int(audit.get("seed_count", -1)) != expected_seeds:
        raise ValueError(f"{branch} {phase} seed contract changed")
    if audit.get("selection_rank_used") is not False:
        raise ValueError(f"{branch} {phase} used selection_rank")


def finalize(spec: dict[str, Any], *, base: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    model = str(spec["model_label"])
    raw_branches = spec.get("branches")
    if not isinstance(raw_branches, list) or not raw_branches:
        raise ValueError("Branch ledger spec is empty")
    names = [str(branch.get("name")) for branch in raw_branches]
    if names != list(EXPECTED_BRANCHES[: len(names)]):
        raise ValueError(f"Branches are not the frozen prefix: {names}")

    outcomes: list[dict[str, Any]] = []
    passed_indices: list[int] = []
    loaded: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]] = []
    for index, branch in enumerate(raw_branches):
        name = names[index]
        complete_path = _resolve(base, str(branch["complete"]))
        discovery_path = _resolve(base, str(branch["discovery_audit"]))
        confirmation_path = _resolve(base, branch.get("confirmation_audit"))
        assert complete_path is not None and discovery_path is not None
        complete = _load(complete_path)
        discovery = _load(discovery_path)
        confirmation = _load(confirmation_path) if confirmation_path else None
        if str(complete.get("model_label")) != model:
            raise ValueError(f"{name} model mismatch")
        status = str(complete.get("status"))
        if status not in TERMINAL_FAILURES | {"PASS"}:
            raise ValueError(f"{name} is not terminal: {status}")
        _validate_audit(discovery, expected_seeds=20, branch=name, phase="discovery")
        if name == "fullspan_restoration" and int(
            discovery.get("applicable_seed_count", -1)
        ) != 20:
            raise ValueError(f"{name} discovery effective seed contract changed")
        if status == "DISCOVERY_GATE_FAIL" and confirmation is not None:
            raise ValueError(f"{name} opened confirmation after discovery failure")
        if status in {"PASS", "CONFIRMATION_GATE_FAIL"}:
            if confirmation is None:
                raise ValueError(f"{name} lacks its confirmation audit")
            _validate_audit(
                confirmation, expected_seeds=10, branch=name, phase="confirmation"
            )
            if name == "fullspan_restoration" and int(
                confirmation.get("applicable_seed_count", -1)
            ) != 10:
                raise ValueError(f"{name} confirmation effective seed contract changed")
        if status == "PASS":
            passed_indices.append(index)
        outcome = {
            "name": name,
            "status": status,
            "discovery_seed_count": 20,
            "confirmation_seed_count": 10 if confirmation is not None else None,
            "selection_rank_used": False,
            "complete_sha256": _sha(complete_path),
            "discovery_audit_sha256": _sha(discovery_path),
            "confirmation_audit_sha256": (
                _sha(confirmation_path) if confirmation_path is not None else None
            ),
        }
        geometry = branch.get("mediator_geometry")
        if geometry is not None:
            if str(geometry) not in {"suffix8", "full_span"}:
                raise ValueError(f"{name} has unsupported geometry: {geometry}")
            outcome["mediator_geometry"] = str(geometry)
        outcomes.append(outcome)
        loaded.append((complete, discovery, confirmation))

    if len(passed_indices) > 1:
        raise ValueError("More than one integrated branch passed")
    if passed_indices and passed_indices[0] != len(outcomes) - 1:
        raise ValueError("A later branch ran after an integrated PASS")
    if not passed_indices and names != list(EXPECTED_BRANCHES):
        raise ValueError("Cannot declare exhaustion before all four branches terminate")

    final_complete, final_discovery, final_confirmation = loaded[-1]
    passed = bool(passed_indices)
    passed_name = names[passed_indices[0]] if passed else None
    serial_pass = passed_name in {"exact_query_transfer", "persistent_transfer"}
    restoration_pass = passed_name in {"suffix8_restoration", "fullspan_restoration"}
    integrated: dict[str, Any] = {
        "schema_version": "realistic_niah_v5_integrated_branch_ledger_v1",
        "model_label": model,
        "status": "PASS" if passed else "PRE_REGISTERED_BRANCHES_EXHAUSTED",
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection_rank_used": False,
        "branch_outcomes": outcomes,
        "passed_branch": passed_name,
        "pre_registered_branches_exhausted": not passed,
        "integrated_serial_bridge_pass": serial_pass,
        "integrated_mediator_restoration_pass": restoration_pass,
        "discovery_claim_gates": final_complete["discovery_claim_gates"],
    }
    geometry = raw_branches[-1].get("mediator_geometry")
    if geometry is not None:
        integrated["mediator_geometry"] = str(geometry)
    if passed:
        integrated["confirmation_claim_gates"] = final_complete[
            "confirmation_claim_gates"
        ]
    else:
        if any(outcome["status"] not in TERMINAL_FAILURES for outcome in outcomes):
            raise ValueError("Exhausted ledger contains a non-failure branch")
        integrated["final_branch_claim_gates"] = final_complete[
            "discovery_claim_gates"
        ]
    return integrated, final_discovery, final_confirmation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec_path = args.spec.resolve()
    spec = _load(spec_path)
    integrated, discovery, confirmation = finalize(spec, base=spec_path.parent)
    output = args.output
    _atomic_json(output / "integrated_complete.json", integrated)
    _atomic_json(output / "integrated_discovery_audit.json", discovery)
    if confirmation is not None:
        _atomic_json(output / "integrated_confirmation_audit.json", confirmation)
    _atomic_json(
        output / "branch_ledger.json",
        {
            "schema_version": "realistic_niah_v5_integrated_branch_ledger_v1",
            "model_label": integrated["model_label"],
            "status": integrated["status"],
            "branch_outcomes": integrated["branch_outcomes"],
        },
    )
    print(json.dumps(integrated, sort_keys=True))


if __name__ == "__main__":
    main()
