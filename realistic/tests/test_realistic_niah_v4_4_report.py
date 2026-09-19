from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_realistic_niah_v4_4_report.py"
SOURCE = SCRIPT.read_text(encoding="utf-8")
INTEGRATED_SOURCE = (
    ROOT / "scripts" / "build_realistic_niah_v4_4_integrated_report.py"
).read_text(encoding="utf-8")


def _literal_assignment(name: str):
    tree = ast.parse(SOURCE)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"missing assignment: {name}")


def test_v4_4_report_has_the_requested_evidence_order_and_controls() -> None:
    template = _literal_assignment("REPORT_TEMPLATE")

    assert _literal_assignment("FOCUS_VARIANT") == "v4.4"
    assert template.index('<section id="prompt">') < template.index(
        '<section id="answer">'
    )
    assert template.index('<section id="answer">') < template.index(
        '<section id="attention">'
    )
    assert template.index('<section id="attention">') < template.index(
        '<section id="causal">'
    )
    assert "Head ablation" in template
    assert "Needle-end patching" in template
    assert "Answer-query patching" in template
    assert "Steering v1" in template
    assert "Steering v2" in template

    assert "discovery-ranked span-end top-4 / top-8" in SOURCE
    assert "layer-matched" in SOURCE
    assert "没有 matched-random residual control" in SOURCE
    assert "h_receiver(q,l) ← h_donor(q,l)" in SOURCE
    assert "等范数且正交" in SOURCE
    assert "invalid-as-failure" in SOURCE


def test_v4_4_report_exposes_all_layer_interactions_and_both_attention_poolings() -> None:
    template = _literal_assignment("REPORT_TEMPLATE")

    assert 'id="{prefix}-layer"' in SOURCE
    assert '_projector_controls("prompt", answer=False)' in SOURCE
    assert '_projector_controls("answer", answer=True)' in SOURCE
    assert 'id="joint-layer"' in SOURCE
    assert 'id="{prefix}-canvas"' in SOURCE
    assert 'id="joint-canvas"' in SOURCE
    assert 'data-atlas="span_end"' in SOURCE
    assert 'data-atlas="span_sum"' in SOURCE
    assert "完整 needle span 的 literal sum" in template
    assert "correct-only basis" in template


def test_v4_4_report_moves_complete_span_locator_evidence_to_appendix() -> None:
    template = _literal_assignment("REPORT_TEMPLATE")

    assert "@@FIRST_LOCATOR_APPENDIX@@" in template
    assert "@@FIRST_SPAN_SECTION@@" not in template
    assert "@@PHENOTYPE_TABLE@@" not in template
    assert template.index('<section id="limits">') < template.index(
        "@@FIRST_LOCATOR_APPENDIX@@"
    )
    assert "Appendix A · Complete-first-span first-locator" in SOURCE
    assert "All-head first-span absolute mass" in SOURCE
    assert "All-head first-span share of ten-span mass" in SOURCE
    assert "Exact rank-1 ten-span masses" in SOURCE
    assert "First-span-ranked top-k ablation curves" in SOURCE
    assert "Layer + M₁₀-nearest" in SOURCE
    assert "相对 gold 的绝对误差增加量之差" in SOURCE
    assert "不是原始的 <code>y(ranked) − y(control)</code>" in SOURCE
    assert "不支持“first-span locator 是独特必要 circuit”" in SOURCE
    assert "V4.4 endpoint phenotype counts and representatives" not in SOURCE
    assert '<h3>8.3 Complete-first-span attention phenotype</h3>' not in INTEGRATED_SOURCE
    assert "ensure_first_locator_appendix" in INTEGRATED_SOURCE
    assert "base = ensure_first_locator_appendix(base, repo_root)" in INTEGRATED_SOURCE
    assert "remove_first_locator_body_claims" in INTEGRATED_SOURCE
    assert "remove_legacy_endpoint_phenotype_table" in INTEGRATED_SOURCE
    assert "base = remove_legacy_endpoint_phenotype_table(base)" in INTEGRATED_SOURCE


def test_v4_4_causal_subsections_state_results_and_inference_limits() -> None:
    template = _literal_assignment("REPORT_TEMPLATE")

    for placeholder in (
        "@@ABLATION_RESULT@@",
        "@@ENDPOINT_PATCH_RESULT@@",
        "@@ANSWER_PATCH_RESULT@@",
        "@@STEERING_V1_RESULT@@",
        "@@STEERING_V2_RESULT@@",
    ):
        assert placeholder in template
    assert "bank-level、mixed-phenotype necessity screen" in SOURCE
    assert "不等同于变成 donor gold" in SOURCE
    assert "不否定 full-span、多 token" in SOURCE
    assert "multi-layer 没有稳定超过 single-layer" in SOURCE


def test_integrated_report_defines_layerwise_map_rotation_without_gauge_overclaim() -> None:
    assert "build_layerwise_subspace_section" in INTEGRATED_SOURCE
    assert "5.4C · 实验 A 的跨层扫描" in INTEGRATED_SOURCE
    assert "5.4D · 实验 B 的跨层扫描" in INTEGRATED_SOURCE
    assert "A<sub>ℓ</sub>=R<sub>ℓ</sub>S<sub>ℓ</sub>" in INTEGRATED_SOURCE
    assert "PCA basis 可各自右乘任意正交矩阵" in INTEGRATED_SOURCE
    assert "T<sub>ℓ</sub>=U<sub>ℓ</sub>A<sub>ℓ</sub>U<sub>ℓ+1</sub><sup>T</sup>" in INTEGRATED_SOURCE
    assert "不是同一个参数化" in INTEGRATED_SOURCE
    assert "不是把相关性误写成对 rotation matrix 的直接干预" in INTEGRATED_SOURCE
    assert "outcome-blind stability rule" in INTEGRATED_SOURCE
    assert "full_operator_cosine_to_next" in INTEGRATED_SOURCE
    assert "full_operator_relative_drift_to_next" in INTEGRATED_SOURCE


def test_integrated_report_keeps_removal_positions_as_separate_estimands() -> None:
    assert "5.4C-1 · Needle-end prompt removal" in INTEGRATED_SOURCE
    assert "5.4C-2 · Answer-query removal" in INTEGRATED_SOURCE
    assert "candidate damage vs clean" in INTEGRATED_SOURCE
    assert "orthogonal-control damage vs clean" in INTEGRATED_SOURCE
    assert "不搜索或优化 control 方向" in INTEGRATED_SOURCE
    assert "layerwise_answer_query_removal_damage_statistics.csv" in INTEGRATED_SOURCE
    assert '("layerwise answer-query removal", layerwise_answer_audit)' in INTEGRATED_SOURCE


def test_variant_filter_rejects_cross_panel_leakage() -> None:
    assert 'frame["design_variant"].astype(str) == FOCUS_VARIANT' in SOURCE
    assert 'raise RuntimeError(f"{model}/{stage}: no {FOCUS_VARIANT} rows")' in SOURCE
    assert "_nested_v44_rows" in SOURCE
    assert 'answer_patch["design_variant"].astype(str) == FOCUS_VARIANT' in SOURCE
