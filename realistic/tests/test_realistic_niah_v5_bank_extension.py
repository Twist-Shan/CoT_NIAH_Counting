from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.finalize_realistic_niah_v5_bank_extension import finalize


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _protocol() -> dict:
    return {
        "model_label": "Gemma4-E4B",
        "bank": {"size": 6, "selected_bank_sha256": "bank6"},
        "stages": [
            {
                "name": "targeted_retrieval_to_final_count",
                "root": "endpoint",
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
            {"condition": "selected_bank", "bank_size": 6, "bank_sha256": "bank6"}
        )
    _write(
        plan.with_suffix(".audit.json"),
        {"status": "PASS", "selection_rank_used": False},
    )
    _write(root / "endpoint/targeted_count_complete.json", {"status": status})
    _write(
        root / "endpoint/targeted_count_analysis_discovery/audit.json",
        {"status": "PASS", "seed_count": 20, "selection_rank_used": False},
    )
    if status in {"PASS", "CONFIRMATION_NEGATIVE"}:
        _write(
            root / "endpoint/targeted_count_analysis_confirmation/audit.json",
            {"status": "PASS", "seed_count": 10, "selection_rank_used": False},
        )


def _bridge(root: Path, status: str) -> None:
    _write(
        root / "bridge/restoration_complete.json",
        {
            "status": status,
            "model_label": "Gemma4-E4B",
            "targeted_bank_size": 6,
            "targeted_bank_sha256": "bank6",
        },
    )
    _write(
        root / "bridge/restoration_analysis_discovery/audit.json",
        {
            "status": "PASS",
            "seed_count": 20,
            "applicable_seed_count": 20,
            "selection_rank_used": False,
        },
    )
    if status in {"PASS", "CONFIRMATION_GATE_FAIL"}:
        _write(
            root / "bridge/restoration_analysis_confirmation/audit.json",
            {
                "status": "PASS",
                "seed_count": 10,
                "applicable_seed_count": 10,
                "selection_rank_used": False,
            },
        )


def test_bank_extension_passes_only_when_endpoint_and_bridge_pass(tmp_path: Path) -> None:
    _endpoint(tmp_path, "PASS")
    _bridge(tmp_path, "PASS")
    result = finalize(tmp_path, _protocol())
    assert result["status"] == "PASS"
    assert result["endpoint_status"] == "PASS"
    assert result["bridge_status"] == "PASS"


def test_bank_extension_seals_bridge_after_endpoint_discovery_failure(
    tmp_path: Path,
) -> None:
    _endpoint(tmp_path, "DISCOVERY_NEGATIVE")
    result = finalize(tmp_path, _protocol())
    assert result["status"] == "PROTOCOL_EXHAUSTED"
    assert result["bridge_status"] == "NOT_OPENED"


def test_bank_extension_rejects_bridge_opened_before_endpoint_pass(
    tmp_path: Path,
) -> None:
    _endpoint(tmp_path, "DISCOVERY_NEGATIVE")
    _bridge(tmp_path, "DISCOVERY_GATE_FAIL")
    with pytest.raises(ValueError, match="before endpoint PASS"):
        finalize(tmp_path, _protocol())


def test_bank_extension_rejects_effective_seed_loss(tmp_path: Path) -> None:
    _endpoint(tmp_path, "PASS")
    _bridge(tmp_path, "DISCOVERY_GATE_FAIL")
    audit = tmp_path / "bridge/restoration_analysis_discovery/audit.json"
    value = json.loads(audit.read_text(encoding="utf-8"))
    value["applicable_seed_count"] = 19
    _write(audit, value)
    with pytest.raises(ValueError, match="effective seed"):
        finalize(tmp_path, _protocol())
