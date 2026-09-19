"""Freezing must accept relocated prerequisites and fail before partial output."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from freeze import freeze
from test_protocol import source as source_row


@pytest.fixture
def inputs(tmp_path):
    config = Path(__file__).parent / "configs/pilot_v1.json"
    source = tmp_path / "controlled.jsonl"
    rows = [dict(source_row(n, seed), design_variant="v4.4")
            for seed in (1234, 1254) for n in (2, 8, 10)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    membership = tmp_path / "relocated_heads.csv"
    with membership.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_label", "top_n", "rank", "layer", "head"])
        writer.writeheader()
        for model, top_n, count in (("Qwen3-8B", 32, 32), ("Gemma4-E4B", 8, 6)):
            writer.writerows(dict(model_label=model, top_n=top_n, rank=i + 1, layer=i // 8, head=i % 8)
                             for i in range(count))
    basis_root = tmp_path / "relocated_bases"
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        folder = basis_root / model
        folder.mkdir(parents=True)
        (folder / "item_end_discovery_basis.json").write_text('{"fixture": true}', encoding="utf-8")
        np.savez(folder / "item_end_discovery_basis.npz", basis=np.eye(2))
    return config, source, membership, basis_root, tmp_path / "frozen"


def test_relocated_inputs_preserve_cases_and_basis_bytes(inputs):
    config, source, membership, basis_root, output = inputs
    before = source.read_bytes()
    audit = freeze(config, source, output, smoke=True,
                   head_membership=membership, native_basis_root=basis_root)
    assert audit["case_count"] == 12
    assert source.read_bytes() == before
    assert audit["source_sha256"] == hashlib.sha256(before).hexdigest()
    assert len(audit["legacy_native_bases"]) == 4
    for item in audit["legacy_native_bases"]:
        assert (output / item["path"]).read_bytes() == Path(item["source"]).read_bytes()
    frozen = json.loads((output / "frozen_banks.json").read_text(encoding="utf-8"))
    assert frozen["source_sha256"] == hashlib.sha256(membership.read_bytes()).hexdigest()
    assert {name: len(bank) for name, bank in frozen["banks"].items()} == {"Qwen3-8B": 32, "Gemma4-E4B": 6}


@pytest.mark.parametrize("missing", ["membership", "basis_json", "basis_npz"])
def test_missing_prerequisite_leaves_no_output(inputs, missing):
    config, source, membership, basis_root, output = inputs
    path = membership if missing == "membership" else basis_root / "Gemma4-E4B" / (
        "item_end_discovery_basis.json" if missing == "basis_json" else "item_end_discovery_basis.npz")
    path.unlink()
    with pytest.raises(FileNotFoundError, match="no output created"):
        freeze(config, source, output, smoke=True,
               head_membership=membership, native_basis_root=basis_root)
    assert not output.exists()
