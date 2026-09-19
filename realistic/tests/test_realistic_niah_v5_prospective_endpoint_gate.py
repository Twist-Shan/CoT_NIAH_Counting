from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.audit_realistic_niah_v5_prospective_endpoint_gate import audit


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _protocol() -> dict:
    return {
        "model_label": "Qwen3-8B",
        "bank": {"size": 128, "selected_bank_sha256": "bank128"},
        "stages": [
            {
                "name": "targeted_retrieval_to_final_count",
                "root": "endpoint",
                "discovery_seeds": list(range(1234, 1254)),
                "confirmation_seeds": list(range(1254, 1264)),
            },
            {
                "name": "targeted_retrieval_to_terminal_state_to_readout",
                "root": "bridge",
            },
        ],
    }


def _endpoint(root: Path, status: str) -> None:
    plan = root / "endpoint/frozen_targeted_count_plan.csv"
    plan.parent.mkdir(parents=True, exist_ok=True)
    with plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["condition", "bank_size", "bank_sha256"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "condition": "selected_bank",
                "bank_size": 128,
                "bank_sha256": "bank128",
            }
        )
    _write(
        plan.with_suffix(".audit.json"),
        {
            "status": "FROZEN_NEW",
            "model_label": "Qwen3-8B",
            "bank_size": 128,
            "selected_bank_sha256": "bank128",
            "outcome_blind": True,
            "selection_rank_used": False,
            "historical_artifacts_modified": False,
        },
    )
    complete = {"status": status}
    if status == "PASS":
        complete.update(
            {
                "discovery": {"targeted_to_count_pass": True},
                "confirmation": {"targeted_to_count_pass": True},
            }
        )
    _write(root / "endpoint/targeted_count_complete.json", complete)
    _write(
        root / "endpoint/targeted_count_analysis_discovery/audit.json",
        {"status": "PASS", "seed_count": 20, "selection_rank_used": False},
    )
    if status in {"PASS", "CONFIRMATION_NEGATIVE"}:
        _write(
            root / "endpoint/targeted_count_analysis_confirmation/audit.json",
            {"status": "PASS", "seed_count": 10, "selection_rank_used": False},
        )


def test_endpoint_gate_opens_bridge_only_after_confirmed_pass(tmp_path: Path) -> None:
    _endpoint(tmp_path, "PASS")
    result = audit(tmp_path, _protocol())
    assert result["status"] == "BRIDGE_ELIGIBLE"
    assert result["endpoint_status"] == "PASS"
    assert result["discovery_seed_count"] == 20
    assert result["confirmation_seed_count"] == 10
    assert result["selection_rank_used"] is False


def test_endpoint_gate_seals_bridge_after_discovery_negative(tmp_path: Path) -> None:
    _endpoint(tmp_path, "DISCOVERY_NEGATIVE")
    result = audit(tmp_path, _protocol())
    assert result["status"] == "PROTOCOL_EXHAUSTED"
    assert result["confirmation_seed_count"] == 0


def test_endpoint_gate_rejects_seed_contract_change(tmp_path: Path) -> None:
    _endpoint(tmp_path, "PASS")
    path = tmp_path / "endpoint/targeted_count_analysis_confirmation/audit.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["seed_count"] = 9
    _write(path, value)
    with pytest.raises(ValueError, match="seed contract"):
        audit(tmp_path, _protocol())


def test_endpoint_gate_rejects_non_blind_frozen_plan(tmp_path: Path) -> None:
    _endpoint(tmp_path, "PASS")
    path = tmp_path / "endpoint/frozen_targeted_count_plan.audit.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["outcome_blind"] = False
    _write(path, value)
    with pytest.raises(ValueError, match="outcome-blind"):
        audit(tmp_path, _protocol())
