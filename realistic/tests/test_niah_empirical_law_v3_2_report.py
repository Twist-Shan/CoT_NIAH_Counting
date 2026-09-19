from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "NiaH_Empirical-law_report.html"
BUILDER = ROOT / "scripts" / "build_niah_empirical_law_v3_2_report.py"
TABLES = (
    ROOT
    / "outputs"
    / "anvil_realistic_niah_v3_1_20260819_formal"
    / "analysis"
    / "v3_2_inverse_n_candidate_extension"
    / "tables"
)


def test_v3_2_report_is_complete_and_uses_frozen_laws() -> None:
    text = REPORT.read_text(encoding="utf-8")
    # Later supplements retain their own checks and are outside the original
    # 18-figure scope of these historical assertions.
    import re
    for start, end in [
        ('PIECEWISE_N3_ADDENDUM_START', 'PIECEWISE_N3_ADDENDUM_END'),
        ('LENGTH_FINDINGS_FRONT_START', 'LENGTH_FINDINGS_FRONT_END'),
        ('V3_2_N_FIXED_START', 'V3_2_N_FIXED_END'),
        ('V3_3_LONG_CONTEXT_START', 'V3_3_LONG_CONTEXT_END'),
    ]:
        text = re.sub(f'<!-- {start} -->.*?<!-- {end} -->', '', text, flags=re.S)
    required = [
        "161,280",
        "V3.2 · 10% trimmed MAE",
        "基础设定、estimands 与计算方法",
        "Exactness follows mode-specific odds laws",
        "Mechanism hypothesis: a mode-dependent bottleneck exchange",
        "近似标量化的 numerosity noise",
        "一次检索路由加逐项目标聚合",
        "Three falsifiable follow-ups for the mechanism hypothesis",
        "Central error magnitude follows a shared N–L interaction law",
        "Mechanism hypothesis: intrinsic numerosity noise versus retrieval–aggregation coupling",
        "Falsifiable MAE checks",
        "10% symmetrically trimmed conditional MAE identity regression",
        "LOMO",
        "leave-one-model-out",
        "HC3 sandwich covariance",
        "Benjamini–Hochberg",
        "IQR",
        "symlog",
        "A shared interaction topology, with distinct strength",
        "Bias regression and outlier control",
        "Enumeration controls repeat the Accuracy and MAE views",
        "Three estimands on the joint N–L grid",
        "FIGURE D1",
        "Drag to rotate",
        "Reset view",
        "appendix-d-interactive",
        "Plotly.react",
        "Trimmed conditional MAE",
        "这是双侧 10% 截尾",
        "10% trimmed conditional MAE 汇总图",
        "Accuracy 汇总图",
        "10% trimmed signed bias 汇总图",
        "Identity-link 限制",
        "不作静默裁剪",
        "Cook's D",
        "L-horizontal",
        "12 个模型的 Accuracy 方程",
        "12 个模型的 trimmed-MAE 方程",
        "MathJax",
    ]
    assert all(item in text for item in required)
    aurora = [
        "#23165C",
        "#6750E8",
        "#00C2FF",
        "#F6E36A",
        "#00D4B4",
        "#39E58C",
        "#C04DFF",
        "#FF5FA2",
    ]
    assert all(color in text for color in aurora)
    assert text.count("<figure>") == 18
    assert text.count("结论：") == 11
    assert text.count('src="data:image/png;base64,') == 18
    assert text.count('data-d3-metric=') == 3
    assert '"schema_version":"niah_appendix_d_plotly_v1"' in text
    assert "plotly.js v3.6.0" in text.lower()
    assert "�" not in text
    assert "2,000 次 bootstrap" not in text
    assert "2000 次 bootstrap" not in text

    builder = BUILDER.read_text(encoding="utf-8")
    scaling_law_palette = ["#F0F921", "#ED7953", "#9C179E", "#2C115F"]
    assert all(color in builder for color in scaling_law_palette)
    required_english_plot_text = [
        "Target count N (log2 positions)",
        "Passage length L (tokens; log positions)",
        "10% trimmed signed bias (count units)",
        "10% trimmed conditional MAE laws across all models",
        "Enumeration accuracy laws",
        "Enumeration 10% trimmed conditional MAE laws",
        "Three estimands on the registered N–L grid",
    ]
    assert all(item in builder for item in required_english_plot_text)
    forbidden_plot_text = [
        "真实针数量 N（log2 scale）",
        "上下文长度 L（token）",
        "10% 截尾有符号偏差\"",
        "点面积随该槽准确率增加",
    ]
    assert all(item not in builder for item in forbidden_plot_text)
    assert "preferred_lengths" not in builder
    assert builder.count("apply_all_n_ticks(ax, n_levels") >= 4
    assert builder.count('l_levels = sorted(int(x) for x in cells["L"].unique())') >= 4

    selected = pd.read_csv(TABLES / "selected_mode_laws.csv")
    actual = {
        (row.outcome_family, row.prompt_mode): row.selected_candidate
        for row in selected.itertuples(index=False)
    }
    expected = {
        ("accuracy_bernoulli_logit", "direct"): "logN__L_k",
        ("accuracy_bernoulli_logit", "native_thinking"): "N__logL",
        ("trimmed_signed_bias_10", "direct"): "N__L_k__N_x_L_k",
        ("trimmed_signed_bias_10", "native_thinking"): "N__L_k__N_x_L_k",
    }
    for key, value in expected.items():
        assert actual[key] == value


def test_piecewise_additive_extension_covers_both_versions() -> None:
    text = REPORT.read_text(encoding="utf-8")
    assert text.count('id="piecewise-n3"') == 1
    assert '<a href="#piecewise-n3">' in text
    extension = text.split('<!-- PIECEWISE_N3_ADDENDUM_START -->')[1].split('<!-- PIECEWISE_N3_ADDENDUM_END -->')[0]
    assert extension.count('<figure>') == 10
    assert extension.count('alt="Piecewise main figure:') == 8
    assert '原版 · 无交叉项选式' in extension
    assert '分段 · 无交叉项选式' in extension
    summary = pd.read_csv(TABLES.parent.parent / 'v3_2_piecewise_n3_20260909/tables/report_version_summary.csv')
    additive = summary.loc[summary.variant.isin(['full_additive','tail_additive'])]
    assert len(additive) == 32  # 2 outcomes x 4 modes x 2 versions x 2 domains
    assert not additive.candidate.str.contains('_x_').any()
    assert additive.median_prediction_mae.notna().all()


def test_v3_2_build_manifest_records_audited_scope() -> None:
    import json

    manifest = json.loads(
        (
            ROOT
            / "reports"
            / "niah_empirical_law_v3_2_assets"
            / "report_build_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["analysis_state"] == "complete"
    assert manifest["requests"] == manifest["unique_request_ids"] == 161_280
    assert manifest["physical_model_revisions"] == 14
    assert manifest["comparison_slots"] == 12
    assert manifest["bootstrap_repetitions"] == 0
    assert len(manifest["figures"]) == 18
    assert manifest["figures"]["accuracy_aggregate_n"]["path"].endswith(
        "fig00a_accuracy_aggregate_by_N.png"
    )
    assert manifest["figures"]["accuracy_aggregate_l"]["path"].endswith(
        "fig00b_accuracy_aggregate_by_L.png"
    )
    assert manifest["figures"]["accuracy_n"]["path"].endswith("fig01_accuracy_by_N_all_models.png")
    assert manifest["figures"]["accuracy_l"]["path"].endswith("fig02_accuracy_by_L_all_models.png")
    assert manifest["figures"]["mae_n"]["path"].endswith("fig03_trimmed_mae_by_N_all_models.png")
    assert manifest["figures"]["mae_l"]["path"].endswith("fig04_trimmed_mae_by_L_all_models.png")
    assert manifest["figures"]["mae_aggregate_n"]["path"].endswith(
        "fig02a_trimmed_mae_aggregate_by_N.png"
    )
    assert manifest["figures"]["bias_aggregate_l"]["path"].endswith(
        "figB0b_bias_aggregate_by_L.png"
    )
    assert manifest["figures"]["interaction"]["path"].endswith("figA1_shared_interaction_strength.png")
    assert manifest["figures"]["enum_mae_l"]["path"].endswith("figC4_enumeration_mae_by_L.png")
    assert manifest["figures"]["appendix_d_3d"]["path"].endswith(
        "figD1_three_estimands_3d.png"
    )
    interactive = manifest["interactive_appendix_d"]
    assert interactive["engine"] == "Plotly.js 3.6.0"
    assert interactive["payload_schema"] == "niah_appendix_d_plotly_v1"
    assert interactive["registered_points_per_mode"] == 14 * 8
    assert interactive["surface_grid_points"] == 48 * 48
    assert interactive["refit"] is False
    assert len(interactive["bundle_sha256"]) == 64
    assert len(interactive["payload_sha256"]) == 64
    assert "untrimmed_bias_sensitivity_tables" in manifest["sources"]
    assert len(manifest["sources"]["untrimmed_bias_sensitivity_manifest_sha256"]) == 64
    assert len(manifest["sources"]["trimmed_count_error_extension_manifest_sha256"]) == 64
    assert len(manifest["sources"]["inverse_n_candidate_extension_manifest_sha256"]) == 64
    assert manifest["analysis_version"] == "V3.2 + inverse-count candidate extension"
    assert manifest["headline_trimmed_conditional_mae_laws"] == {
        "direct": "N__L_k__N_x_L_k",
        "native_thinking": "N__L_k__N_x_L_k",
    }
