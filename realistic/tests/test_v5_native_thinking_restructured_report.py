from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_EVIDENCE_SENTINEL = (
    REPO_ROOT
    / "reports"
    / "v5_native_final_localizers"
    / "analysis"
    / "qwen_final_merged_dose_grid.json"
)
SECTION_IDS = (
    "summary",
    "baseline",
    "representation",
    "formation",
    "retrieval",
    "write",
    "answer",
    "integrated-chain",
    "ledger",
    "extension-audit",
    "limitations",
    "appendix",
)


def _assert_current_report_contract(text: str, payload: dict) -> None:
    offsets = [text.index(f'<section id="{section_id}">') for section_id in SECTION_IDS]
    assert offsets == sorted(offsets)
    assert "本文主张（仅限 Qwen3-8B 的自然 no-index trace）" in text
    assert "Gemma 尚无对应的自然 no-index 因果结果" in text
    assert "simulatively confirmed†" in text
    assert "可诱发的机制能力，不是自然使用" in text
    assert "实验前置 · Parser 与因果设计合同" in text
    assert text.count('class="parser-disclosure"') == 4
    assert "strict_eligible_no_explicit_count_cue" in text
    assert "first_generated_known_city_ordinal" in text
    assert "20 discovery / 10 confirmation" in text
    assert "J.1 显式 index positive control" in text
    assert "Appendix K · Gemma prompt-conditioned no-visible-index" in text
    assert "图 6d · Answer-query full-state patch 的逐层 donor-count adoption" in text
    assert "registered existing-split extension" in text
    assert "answer_query_v3" in text
    assert "A · Teacher-forced next-city exact" in text
    assert "为什么 Gemma 的 commit→next-query 现在标 confirmed" in text
    assert "均值为正但跨 seed 不够稳定" not in text
    assert 'id="native-running-canvas"' in text
    assert 'id="native-final-canvas"' in text
    assert '<img ' not in text
    assert len(re.findall(r"<svg(?:\s|>)", text)) == text.count("</svg>")
    assert len(text.encode("utf-8")) < 4_500_000

    assert payload["status"] == "PASS"
    assert payload["schema_version"] == (
        "realistic_niah_v5_native_thinking_restructured_v14"
    )
    contract = payload["scientific_contract"]
    assert contract["discovery_seed_count"] == 20
    assert contract["confirmation_seed_count"] == 10
    assert contract["outcome_blind"] is True
    assert contract["parser_design_contract_in_main_text"] is True
    assert contract["parser_design_disclosure_count"] == 4
    assert contract["generation_primary_endpoint"] == (
        "first_generated_known_city_ordinal"
    )
    assert contract["indexed_positive_control_active_confirmation_layer"] == {
        "Qwen3-8B": 19,
        "Gemma4-E4B": 16,
    }
    assert contract["answer_query_layer_sweep_seed_count"] == 10
    assert contract["cross_model_sample_alignment_audit_status"] == "PASS"
    assert contract["cross_model_output_alignment_audit_status"] == "PASS"
    assert contract["token_source_primary_retrieval_endpoint"] == {
        "Qwen3-8B": "target_city_teacher_forced_exact",
        "Gemma4-E4B": "target_city_teacher_forced_exact",
    }
    assert contract["token_source_endpoint_is_teacher_forced_not_free_generation"] is True
    assert contract["answer_query_layer_sweep_pair_counts"] == {
        "Qwen3-8B": 40,
        "Gemma4-E4B": 40,
    }
    assert contract["answer_query_layer_sweep_is_pristine_new_confirmation"] is False
    assert contract["answer_query_site_absolute_position_match_required"] is False

    scope = payload["claim_scope"]
    assert scope["distributed_content_bound_event_progress_state_supported"] is True
    assert scope["memoryless_arithmetic_plus_one_recurrence_supported"] is False
    assert scope["qwen_no_index_scope_result_extrapolated_to_gemma"] is False
    assert scope["gemma_natural_no_index_causal_result_available"] is False
    assert scope["gemma_commit_to_query_direct_effect_confirmed"] is True
    assert scope["gemma_commit_to_query_orthogonal_interval_positive"] is True
    assert scope[
        "commit_query_orthogonal_control_matches_count_projection_norm_not_full_delta_norm"
    ] is True
    assert scope["gemma_next_item_routing_status"] == (
        "simulatively confirmed under auxiliary settings"
    )
    assert scope["gemma_prompt_conditioned_supports_natural_no_index_claim"] is False
    assert scope["qwen_answer_query_full_state_executable"] is True
    assert scope["gemma_answer_query_full_state_executable"] is True
    assert scope["answer_query_full_state_implies_low_dimensional_counter"] is False
    assert scope["same_trial_trace_to_answer_serial_mediation_complete"] is False
    assert scope["trace_to_answer_ordered_partial_path_supported"] is True
    assert scope["trace_to_answer_same_trial_mediation_common_support"] is True


@pytest.mark.skipif(
    not EXTERNAL_EVIDENCE_SENTINEL.exists(),
    reason="external Native-thinking evidence bundle is not installed",
)
def test_restructured_report_is_audited_and_nonthinking_ordered(
    tmp_path: Path,
) -> None:
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
    _assert_current_report_contract(text, payload)

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "validate_v5_native_thinking_report.py"),
            str(output),
            str(manifest),
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def test_shipped_restructured_report_is_self_contained() -> None:
    report = REPO_ROOT / "reports" / "NiaH_Native-Thinking_report.html"
    manifest = (
        REPO_ROOT
        / "reports"
        / "v5_native_final_localizers"
        / "report_manifest_restructured.json"
    )
    text = report.read_text(encoding="utf-8")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    _assert_current_report_contract(text, payload)
