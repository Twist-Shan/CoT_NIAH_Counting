from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_shipped_report_keeps_natural_noindex_claim_qwen_only() -> None:
    report = REPO_ROOT / "reports" / "NiaH_Native-Thinking_report.html"
    manifest_path = (
        REPO_ROOT
        / "reports"
        / "v5_native_final_localizers"
        / "report_manifest_restructured.json"
    )
    text = report.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "本文主张（仅限 Qwen3-8B 的自然 no-index trace）" in text
    assert "Gemma 尚无对应的自然 no-index 因果结果" in text
    assert "显式 index positive control" in text
    assert "Pre-confirmation amendment" in text
    assert "Qwen L19、Gemma L16" in text
    assert (
        "Appendix K · Gemma prompt-conditioned no-visible-index forward transplant"
        in text
    )
    assert "Qwen forward 为 25/30，Gemma 为 22/30" in text
    assert "12/30 vs Qwen 29/30" in text
    assert "自然 no-index 主张仍严格限于 Qwen" in text
    assert "simulatively confirmed†" in text
    assert "可诱发的机制能力，不是自然使用" in text
    assert "实验前置 · Parser 与因果设计合同" in text
    assert "strict_eligible_no_explicit_count_cue" in text
    assert manifest["schema_version"] == (
        "realistic_niah_v5_native_thinking_restructured_v14"
    )
    assert manifest["claim_scope"][
        "qwen_no_index_scope_result_extrapolated_to_gemma"
    ] is False
    assert manifest["claim_scope"][
        "indexed_positive_control_supports_no_index_internal_counter"
    ] is False
    assert manifest["claim_scope"][
        "gemma_natural_no_index_causal_result_available"
    ] is False
    assert manifest["claim_scope"]["gemma_next_item_routing_status"] == (
        "simulatively confirmed under auxiliary settings"
    )
    assert manifest["claim_scope"]["gemma_simulative_support_sources"] == [
        "prompt_conditioned_no_visible_index",
        "explicit_index",
    ]
    assert manifest["claim_scope"][
        "gemma_prompt_conditioned_no_index_auxiliary_complete"
    ] is True
    assert manifest["claim_scope"][
        "gemma_prompt_conditioned_supports_natural_no_index_claim"
    ] is False
    assert manifest["claim_scope"][
        "gemma_prompt_conditioned_no_index_first_city_transfer"
    ] == "22/30"
    assert manifest["claim_scope"][
        "gemma_prompt_conditioned_no_index_attention_positive"
    ] == "12/30"
    assert manifest["claim_scope"][
        "qwen_indexed_positive_control_first_city_transfer"
    ] == "59/60"
    assert manifest["claim_scope"][
        "qwen_indexed_positive_control_first_city_paired_gain"
    ] == 59 / 60
    assert manifest["claim_scope"][
        "gemma_indexed_positive_control_first_city_transfer"
    ] == "42/60"
    assert manifest["claim_scope"][
        "gemma_indexed_positive_control_first_city_paired_gain"
    ] == 37 / 60
    assert manifest["scientific_contract"][
        "indexed_positive_control_active_confirmation_layer"
    ] == {"Qwen3-8B": 19, "Gemma4-E4B": 16}


def test_indexed_confirmation_freeze_precedes_confirmation() -> None:
    root = REPO_ROOT / "work" / "v5_native_sample_aligned_20260829"
    freeze = json.loads(
        (root / "indexed_progress_control_freeze.json").read_text(encoding="utf-8")
    )
    assert freeze["status"] == "FROZEN_BEFORE_CONFIRMATION"
    assert freeze["confirmation_outcomes_observed"] is False
    assert freeze["active_confirmation_layers"] == {
        "Qwen3-8B": 19,
        "Gemma4-E4B": 16,
    }
    assert _sha256(root / "indexed_progress_control_panel" / "alignment_manifest.json") == (
        freeze["aligned_cohort_manifest_sha256"]
    )
    assert _sha256(REPO_ROOT / freeze["representation_alignment_audit"]) == (
        freeze["representation_alignment_audit_sha256"]
    )
