"""Provenance guards for a fresh-input run, without loading a tokenizer/model."""
import hashlib
import json

import pytest

import prepare_realistic_niah_v3_1 as preparation
import realistic_niah_v3_1.stimuli as stimuli


@pytest.fixture
def fresh_input(tmp_path, monkeypatch):
    data = b'{"new_input": true}\n'
    (tmp_path / "stimuli.jsonl").write_bytes(data)
    report = {"passed": True, "protocol_version": preparation.PROTOCOL_VERSION,
              "rows_checked": preparation.EXPECTED_STIMULI,
              "stimuli_sha256": hashlib.sha256(data).hexdigest()}
    (tmp_path / "audit_report.json").write_text(json.dumps(report))
    calls = []

    def audit(**kwargs):
        calls.append(kwargs)
        return dict(report)

    monkeypatch.setattr(stimuli, "audit_v31_grid", audit)
    return tmp_path, report, calls


def test_fresh_identity_is_the_new_hash_and_requires_real_tokenizer(fresh_input):
    root, report, calls = fresh_input
    result = preparation.validate_fresh_dataset(root, cache_dir="tokenizer-cache")
    assert result["dataset_id"].startswith("fresh-replication/")
    assert result["revision"] == report["stimuli_sha256"]
    assert calls[0]["require_huggingface_tokenizer"] is True
    assert calls[0]["cache_dir"] == "tokenizer-cache"


@pytest.mark.parametrize("field,value", [("passed", False), ("protocol_version", "other"),
                                         ("rows_checked", 1), ("stimuli_sha256", "stale")])
def test_reject_failed_or_stale_saved_audit(fresh_input, field, value):
    root, report, _ = fresh_input
    (root / "audit_report.json").write_text(json.dumps({**report, field: value}))
    with pytest.raises(RuntimeError, match="grid/content audit"):
        preparation.validate_fresh_dataset(root)


def test_reject_failed_recomputed_audit(fresh_input):
    root, report, _ = fresh_input
    report["passed"] = False
    with pytest.raises(RuntimeError, match="grid/content audit"):
        preparation.validate_fresh_dataset(root)
