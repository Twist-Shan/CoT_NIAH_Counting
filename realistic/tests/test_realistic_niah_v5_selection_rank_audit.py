from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_realistic_niah_v5_selection_rank_absence import audit
from scripts.assemble_realistic_niah_v5_count_chain_evidence import (
    _add_supplemental_rank_proof,
)


def test_selection_rank_absence_audit_hashes_all_rows(tmp_path: Path) -> None:
    shards = tmp_path / "trials/shards"
    shards.mkdir(parents=True)
    for seed in (1, 2):
        (shards / f"seed-{seed}.jsonl").write_text(
            json.dumps({"seed": seed, "status": "ok"}) + "\n",
            encoding="utf-8",
        )
    result = audit(tmp_path / "trials", expected_seeds=2)
    assert result["status"] == "PASS"
    assert result["selection_rank_used"] is False
    assert result["shard_count"] == 2
    assert result["row_count"] == 2


def test_selection_rank_absence_audit_rejects_key(tmp_path: Path) -> None:
    shard = tmp_path / "trials/shards/a.jsonl"
    shard.parent.mkdir(parents=True)
    shard.write_text(
        json.dumps({"seed": 1, "selection_rank": 1}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="selection_rank found"):
        audit(tmp_path / "trials", expected_seeds=1)


def test_assembler_accepts_independent_supplemental_rank_proof(
    tmp_path: Path,
) -> None:
    audit_path = tmp_path / "analysis/audit.json"
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text("{}", encoding="utf-8")
    proof = {
        "status": "PASS",
        "selection_rank_used": False,
        "seed_count": 20,
    }
    (audit_path.parent / "selection_rank_audit.json").write_text(
        json.dumps(proof), encoding="utf-8"
    )
    supplemented = _add_supplemental_rank_proof(
        {"status": "PASS", "seed_count": 20},
        audit_path,
        expected_seeds=20,
        label="test",
    )
    assert supplemented["selection_rank_used"] is False
    assert "selection_rank_proof_sha256" in supplemented


def test_assembler_rejects_missing_supplemental_rank_proof(
    tmp_path: Path,
) -> None:
    audit_path = tmp_path / "analysis/audit.json"
    with pytest.raises(ValueError, match="lacks selection-rank proof"):
        _add_supplemental_rank_proof(
            {"status": "PASS", "seed_count": 20},
            audit_path,
            expected_seeds=20,
            label="test",
        )
