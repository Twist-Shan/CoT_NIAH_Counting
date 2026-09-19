from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_restructured_report_is_concise_audited_and_nonthinking_ordered(tmp_path: Path) -> None:
    output = tmp_path / "native.html"
    manifest = tmp_path / "manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_v5_native_thinking_report_restructured.py"),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    text = output.read_text(encoding="utf-8")
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    section_ids = (
        "summary",
        "design",
        "representation",
        "retrieval",
        "write",
        "answer",
        "walkthrough",
        "comparison",
        "audit",
    )
    offsets = [text.index(f'<section id="{section_id}">') for section_id in section_ids]
    assert offsets == sorted(offsets)
    assert "Qwen3-8B · frozen Top-128" in text
    assert "Gemma4-E4B · frozen Top-6" in text
    assert "20 discovery / 10 confirmation" in text
    assert "descriptive null" in text
    assert "does not walk 1→10" in text
    assert "natural end-to-end sufficiency 仍未证明" in text
    assert "Gemma4-E4B · frozen Top-8" not in text
    assert "完整串行链获得confirmation" not in text
    assert text.count('<svg class=') == text.count("</svg>")
    assert len(text.encode("utf-8")) < 150_000

    assert payload["status"] == "PASS"
    assert payload["scientific_contract"] == {
        "discovery_seed_count": 20,
        "confirmation_seed_count": 10,
        "outcome_blind": True,
        "selection_rank_used": False,
        "qwen_targeted_bank": 128,
        "gemma_targeted_bank": 6,
    }
    assert payload["claim_scope"]["recurrent_pathway_supported"] is True
    assert payload["claim_scope"]["exclusive_circuit_claimed"] is False
    assert payload["claim_scope"]["natural_end_to_end_single_state_sufficiency"] is False
    assert payload["claim_scope"]["single_seed_walkthrough_inferential"] is False
