from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_realistic_niah_v4_representation_report.py"


def _load_report_module():
    spec = importlib.util.spec_from_file_location("v4_report_builder", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_uses_the_registered_aurora_palette() -> None:
    report = _load_report_module()
    assert report.AURORA == {
        "midnight_indigo": "#23165C",
        "polar_violet": "#6750E8",
        "ice_cyan": "#00C2FF",
        "aurora_yellow": "#F6E36A",
        "aurora_teal": "#00D4B4",
        "aurora_green": "#39E58C",
        "polar_magenta": "#C04DFF",
        "sunset_pink": "#FF5FA2",
        "night_black": "#161923",
        "snow_white": "#F8FBFF",
        "frost_gray": "#8190A5",
        "warm_brown": "#765347",
    }
    old_viridis = {
        "#482878",
        "#3e4989",
        "#31688e",
        "#26828e",
        "#1f9e89",
        "#35b779",
        "#6ece58",
        "#b5de2b",
        "#fde725",
    }
    assert not any(color in report.REPORT_TEMPLATE.lower() for color in old_viridis)


def test_report_template_covers_the_full_mechanistic_argument() -> None:
    report = _load_report_module()
    template = report.REPORT_TEMPLATE
    section_ids = (
        "behavior",
        "counter-representation",
        "attention-representation",
        "head-ablation",
        "geometry-steering",
    )
    assert template.count('<section id="') == 5
    for section_id in section_ids:
        assert f'<section id="{section_id}">' in template
    for block in range(1, 6):
        assert f"Block {block} / 5" in template
    assert template.count('class="section-conclusion"') >= 5
    assert "distributed retrieval/aggregation" in template
    assert "late executable query state" in template
    assert "@@BEHAVIOR_ACCURACY_SVG@@" in template
    assert "@@REPRESENTATION_R2_SVG@@" in template
    assert "@@LAYER_SWEEP_SVG@@" in template
    assert "@@ANSWER_QUERY_LAYER_SWEEP_SVG@@" in template
    assert "@@ANSWER_QUERY_COUNTER_SVG@@" in template
    assert "@@ANSWER_QUERY_DATA@@" in template
    assert "@@ATTENTION_HEAD_ATLAS_HTML@@" in template
    assert "@@ATTENTION_ATLAS_ROW_COUNT@@" in template
    assert "@@ATTENTION_PHENOTYPE_ROW_COUNT@@" in template
    assert "@@ATTENTION_POOLING_ALIGNMENT_CONCLUSION@@" in template
    assert "@@ATTENTION_HEAD_PROFILE_SVG@@" in template
    assert "@@ATTENTION_OUTCOME_EFFECT_SVG@@" in template
    assert "@@ATTENTION_OUTCOME_CONCLUSION@@" in template
    assert "@@ATTENTION_OMISSION_POOLING_CONCLUSION@@" in template
    assert "@@ATTENTION_BREADTH_SVG@@" in template
    assert "@@CAUSAL_ABLATION_SVG@@" in template
    assert "@@ANSWER_QUERY_ADOPTION_SVG@@" in template
    assert "@@CAUSAL_STEERING_SVG@@" in template
    assert "@@STEERING_V2_SVG@@" in template
    assert "@@STEERING_V2_SELECTION_ROWS@@" in template
    assert "@@STEERING_V2_SUMMARY_ROWS@@" in template
    assert "@@STEERING_V2_PANEL_ROWS@@" in template
    assert "realistic_niah_v4_steering_v2_selection.csv" in template
    assert "realistic_niah_v4_steering_v2_confirmation.csv" in template
    assert "realistic_niah_v4_steering_v2_panels.csv" in template
    assert 'id="layer-select"' in template
    assert 'id="answer-counter3d"' in template
    assert 'id="aq-layer-select"' in template
    assert 'id="aq-fit-select"' in template
    assert 'id="aq-outcome-select"' in template
    assert 'id="joint-counter3d"' in template
    assert 'id="joint-layer-select"' in template
    assert 'id="joint-mode-select"' in template
    assert "@@JOINT_COUNTER_DATA@@" in template
    assert 'id="prompt-counter-attention-map"' in template
    assert "@@PROMPT_COUNTER_PROFILE_DATA@@" in template
    assert "@@PROMPT_COUNTER_ASSOCIATION_SVG@@" in template
    assert "@@PROMPT_COUNTER_DYNAMICS_ROWS@@" in template
    assert "all-heads-control" in template
    assert "@@ANSWER_QUERY_PCA_SENSITIVITY_ROWS@@" in template
    assert "@@ANSWER_QUERY_PCA_CONCLUSION@@" in template
    assert "function aqDraw()" in template
    assert "function drawCountLabels" in template
    assert "自动避让" in template
    assert "Discovery / confirmation" in template
    assert 'id="definitions"' not in template
    assert template.index("Discovery / confirmation") < template.index('<section id="behavior">')
    assert template.index("Ridge count probe 与 held-out 拟合度") > template.index(
        '<section id="counter-representation">'
    )
    assert template.index("每个 occurrence 的三种 attention pooling") > template.index(
        '<section id="attention-representation">'
    )
    assert template.index("Ablation 的 paired necessity estimand") > template.index(
        '<section id="head-ablation">'
    )
    assert template.index("三种 full-dimensional residual intervention") > template.index(
        '<section id="geometry-steering">'
    )
    assert "Centroid transplant" in template
    assert "Donor-state replacement" in template
    assert "N<sub>eff,H</sub>" in template
    assert "N<sub>eff,2</sub>" in template
    assert "Full-attention visibility" in template
    assert "Count-adjusted wrong−correct" in template
    assert "controls.layer.value=String(defaultLayer??layers[0])" in template
    assert "previous!==null" not in template
    assert "background:linear-gradient" not in template
    assert "--paper:#F3EEE4" in template
    assert "--surface:#FFFDF8" in template
    assert 'class="equation-row"' in template
    assert 'class="command-block"' in template
    assert template.count('class="figure-intro"') >= 12
    assert "function makeTablesCollapsible()" in template
    assert "展开数据表" in template
    assert "<details open>" not in template


def test_steering_v2_strict_pairing_keeps_invalid_outputs_as_failures() -> None:
    report = _load_report_module()
    shared = {
        "model_label": "Qwen3-8B",
        "design_variant": "v4.1",
        "seed": 1254,
        "receiver_stimulus_id": "receiver",
        "target_stimulus_id": "target",
        "receiver_count": 5,
        "target_count": 10,
        "target_direction": "up",
        "steering_protocol": "single_layer",
        "layer_set": "26",
        "alpha": 1.0,
        "moved_toward_donor_gold": True,
        "follows_donor_gold": True,
    }
    detail = report.pd.DataFrame(
        [
            {
                **shared,
                "condition": "geometric",
                "patched_format_valid": False,
                "direction_aligned_generated_count_shift": 5.0,
            },
            {
                **shared,
                "condition": "orthogonal_norm_matched_random",
                "patched_format_valid": True,
                "direction_aligned_generated_count_shift": 1.0,
            },
        ]
    )
    paired = report._steering_v2_paired_effects(detail)
    assert len(paired) == 1
    assert paired.iloc[0]["strict_aligned_shift"] == 0.0
    assert paired.iloc[0]["strict_aligned_shift_effect"] == -1.0
    assert paired.iloc[0]["strict_moved"] == 0.0
    assert paired.iloc[0]["strict_target_hit"] == 0.0


def test_attention_phenotype_rules_are_ordered_and_explicit() -> None:
    report = _load_report_module()
    shared = {
        "dominant_share": 0.2,
        "winner_frequency": 0.4,
        "winner_mode": 4,
        "first_share": 0.1,
        "winner_is_first": 0.1,
        "local_count": 1.0,
        "local_effective_fraction": 0.2,
        "dominant_quarter_mass": 0.3,
        "span_mean_effective_number": 2.0,
        "span_mean_dominant_share": 0.5,
    }
    assert report._classify_attention_phenotype(
        effective_number=6.0, **shared
    ) == "global_endpoint_aggregator"
    assert report._classify_attention_phenotype(
        effective_number=3.0,
        **{
            **shared,
            "local_count": 2.0,
            "local_effective_fraction": 0.8,
            "dominant_quarter_mass": 0.5,
        },
    ) == "partition_local_endpoint_aggregator"
    assert report._classify_attention_phenotype(
        effective_number=2.0,
        **{
            **shared,
            "dominant_share": 0.9,
            "winner_frequency": 0.8,
            "winner_mode": 1,
            "first_share": 0.8,
            "winner_is_first": 0.9,
        },
    ) == "first_needle_locator"
    assert report._classify_attention_phenotype(
        effective_number=2.0,
        **{
            **shared,
            "dominant_share": 0.8,
            "winner_frequency": 0.8,
            "winner_mode": 7,
        },
    ) == "targeted_occurrence_retriever"
    assert report._classify_attention_phenotype(
        effective_number=3.0,
        **{
            **shared,
            "span_mean_effective_number": 6.0,
            "span_mean_dominant_share": 0.25,
        },
    ) == "broad_span_mean_only"


def test_manifold_layer_selection_applies_the_decodability_gate() -> None:
    report = _load_report_module()
    rows = [
        {
            "layer": 0,
            "full_space_discovery_cv_r2": 0.99,
            "manifold_fidelity_m3": 0.30,
        },
        {
            "layer": 1,
            "full_space_discovery_cv_r2": 0.98,
            "manifold_fidelity_m3": 0.50,
        },
        {
            "layer": 2,
            "full_space_discovery_cv_r2": 0.80,
            "manifold_fidelity_m3": 0.95,
        },
    ]
    assert report._select_manifold_layer(rows) == 1


def test_answer_query_counter_is_a_separate_layered_figure() -> None:
    report = _load_report_module()
    projections = {}
    layers = {
        "Qwen3-8B": (9, 18, 26),
        "Gemma4-E4B": (10, 20, 31),
    }
    for model, model_layers in layers.items():
        for layer in model_layers:
            rows = []
            for variant in ("v4.1", "v4.4"):
                for seed in (1234, 1235):
                    for count in range(1, 11):
                        rows.append(
                            [
                                variant,
                                seed,
                                count,
                                "correct" if seed == 1234 else "wrong",
                                count if seed == 1234 else max(1, count - 1),
                                0 if seed == 1234 else -1,
                                float(count),
                                float(count * count + seed % 2),
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                            ]
                        )
            projections[f"{model}|{layer}|all"] = {
                "model": model,
                "layer": layer,
                "fit_cohort": "all",
                "explained_variance_ratio": [0.4, 0.3, 0.1, 0.1, 0.05, 0.05],
                "pca3_discovery_cv_r2": 0.7 + 0.01 * model_layers.index(layer),
                "count_signal_capture_pc1_3": 0.8,
                "discovery_compactness": 0.6,
                "probe_optimal": layer == model_layers[1],
                "manifold_display": layer == model_layers[2],
                "rows": rows,
            }
    svg = report._answer_query_counter_svg(projections)
    assert "Answer-query count manifolds" in svg
    assert "PC1 score" in svg and "PC2 score" in svg
    for model, model_layers in layers.items():
        assert model in svg
        for layer in model_layers:
            assert f"L{layer}" in svg
    assert "<title" in svg and "<desc" in svg
    layer_sweep_svg = report._answer_query_layer_sweep_svg(projections)
    assert "All-layer answer-query manifold diagnostics" in layer_sweep_svg
    assert "PCA3 grouped-seed CV R²" in layer_sweep_svg
    assert "seed compactness" in layer_sweep_svg
    assert "P:L" in layer_sweep_svg and "M:L" in layer_sweep_svg
    template = report.REPORT_TEMPLATE
    assert "Interactive answer-query counter manifold" in template
    assert "Answer-query 逐层 discovery diagnostics" in template
    assert "Static all-fit answer-query" in template
    assert "AQ_DATA" in template
    assert "JOINT_DATA" in template


def test_joint_counter_geometry_metrics_are_coordinate_invariant() -> None:
    report = _load_report_module()
    left = report.np.stack(
        [
            report.np.asarray([float(count), float(count**2), 1.0])
            for count in range(1, 11)
        ]
    )
    rotation = report.np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    right = left @ rotation
    assert report._trajectory_linear_cka(left, right) == pytest.approx(1.0)
    assert report._trajectory_distance_correlation(left, right) == pytest.approx(
        1.0
    )
    assert report._successive_step_alignment(left, left) == pytest.approx(1.0)


def test_answer_query_pca_sensitivity_uses_common_v41_evaluation() -> None:
    report = _load_report_module()
    projections = {}
    for cohort, offset in (("all", 0.0), ("correct_only", 0.2)):
        rows = []
        for seed in (1, 2):
            for count in range(1, 11):
                rows.append(
                    [
                        "v4.1",
                        seed,
                        count,
                        "correct" if seed == 1 else "wrong",
                        count,
                        0,
                        count + offset,
                        count * 0.5,
                        seed * 0.1,
                        0.0,
                        0.0,
                        0.0,
                    ]
                )
        projections[f"Qwen3-8B|9|{cohort}"] = {
            "model": "Qwen3-8B",
            "layer": 9,
            "fit_cohort": cohort,
            "fit_rows": 20 if cohort == "all" else 10,
            "fit_count_support": {
                str(count): 2 if cohort == "all" else 1
                for count in range(1, 11)
            },
            "explained_variance_ratio": [0.5, 0.2, 0.1, 0.05, 0.03, 0.02],
            "common_v41_variance_capture": [0.5, 0.7, 0.8, 0.85, 0.88, 0.9],
            "rows": rows,
        }
    summary = report._answer_query_pca_sensitivity_rows(projections)
    assert len(summary) == 2
    assert {row["fit_cohort"] for row in summary} == {"all", "correct_only"}
    assert all(row["fit_count_support_min"] > 0 for row in summary)
    assert summary[1]["centroid_distance_corr_to_all"] == pytest.approx(1.0)
    conclusion = report._answer_query_pca_conclusion_html(summary)
    assert "common PC1–3 capture" in conclusion
    assert "noise/signal" in conclusion
    assert "centroid-distance" in conclusion


def test_core_figures_define_axes_and_accessibility_text() -> None:
    report = _load_report_module()
    behavior_rows = [
        {
            "model": model,
            "variant": variant,
            "count": count,
            "accuracy": count / 10,
        }
        for model in report.MODELS
        for variant in report.VARIANTS
        for count in range(1, 11)
    ]
    behavior_svg = report._behavior_accuracy_svg(behavior_rows)
    assert "true needle count N" in behavior_svg
    assert "greedy exact-match accuracy" in behavior_svg
    assert "<title" in behavior_svg and "<desc" in behavior_svg

    metric_rows = [
        {
            "model_label": model,
            "design_variant": variant,
            "pooling": pooling,
            "confirmation_r2": 0.5,
        }
        for model in report.MODELS
        for variant in report.VARIANTS
        for pooling in report.POOLINGS
    ]
    representation_svg = report._representation_r2_svg(metric_rows)
    assert "confirmation R²" in representation_svg
    assert "R²=0" in representation_svg
    assert "<title" in representation_svg and "<desc" in representation_svg


def test_every_static_figure_caption_is_self_contained() -> None:
    report = _load_report_module()
    captions = re.findall(
        r"<figcaption>(.*?)</figcaption>", report.REPORT_TEMPLATE, flags=re.DOTALL
    )
    # The four B2-F3b cards are emitted by _static_figure_html rather than
    # stored literally in REPORT_TEMPLATE; the template itself has 18 figures
    # after adding the joint-counter, prompt-dynamics, and answer-layer views.
    assert len(captions) == 18
    for caption in captions:
        assert any(marker in caption for marker in ("横轴", "横向"))
        assert any(marker in caption for marker in ("纵轴", "纵向", "上排"))
    template = report.REPORT_TEMPLATE
    assert "靛蓝、紫、青、粉四条线依次表示 V4.1、V4.2、V4.3、V4.4" in template
    assert "粉线是 seed compactness C=1/(1+R<sub>LOO</sub>)" in template
    assert "灰色虚线连接 V4.1" in template
    assert "粉色为负值" in template and "绿色为正值" in template
    assert "灰线只连接 chance 与 observed" in template
    assert "右侧文字重复 estimate [CI]" in template
    assert "图 B2-F3a · Interactive prompt-reading counter trajectory" in template


def test_attention_pooling_sensitivity_conclusions_require_paired_rows() -> None:
    report = _load_report_module()
    alignment = [
        {
            "model": model,
            "design_variant": variant,
            "left_pooling": "span_end",
            "right_pooling": right,
            "spearman_primary": 0.75,
            "top_k_intersection": 4,
            "top_k": 8,
        }
        for model in report.MODELS
        for variant in report.VARIANTS
        for right in ("span_mean", "span_sum")
    ]
    alignment_html = report._attention_pooling_alignment_conclusion_html(alignment)
    assert "end↔sum Spearman" in alignment_html
    assert "top-8 overlap" in alignment_html

    omission = [
        {
            "model": model,
            "pooling": pooling,
            "prompts": 20,
            "delta": 0.1 if pooling == "span_end" else 0.08,
            "tail_prefix_ratio": 0.4 if pooling == "span_end" else 0.5,
        }
        for model in report.MODELS
        for pooling in ("span_end", "span_sum")
    ]
    omission_html = report._attention_omission_pooling_conclusion_html(omission)
    assert "span-end→span-sum" in omission_html
    assert "同一 20 条 undercount prompts" in omission_html


def test_layer_sweep_and_forest_layouts_reserve_annotation_space() -> None:
    report = _load_report_module()
    sweep_rows = []
    for model in report.MODELS:
        for pooling in report.POOLINGS:
            for layer in (0, 1):
                sweep_rows.append(
                    {
                        "model": model,
                        "pooling": pooling,
                        "layer": layer,
                        "full_space_discovery_cv_r2": 0.8 + 0.1 * layer,
                        "pca_evr_pc1_3": 0.5 + 0.1 * layer,
                        "count_signal_capture_pc1_3": 0.6 + 0.1 * layer,
                        "discovery_compactness": 0.7 + 0.1 * layer,
                        "probe_optimal": layer == 1,
                        "manifold_display": layer == 0,
                    }
                )
    sweep = report._layer_sweep_svg(sweep_rows)
    assert 'viewBox="0 0 1180 850"' in sweep
    assert "seed compactness C" in sweep
    assert "P · probe-optimal" in sweep and "M · manifold-display" in sweep

    forest = report._forest_svg(
        [
            {
                "model": "Qwen3-8B",
                "estimate": 0.1,
                "low": -0.1,
                "high": 0.3,
            },
            {
                "model": "Gemma4-E4B",
                "estimate": 0.2,
                "low": 0.0,
                "high": 0.4,
            },
        ],
        estimate_key="estimate",
        low_key="low",
        high_key="high",
        title="Layout audit",
        axis_label="paired effect",
        label=lambda row: row["model"],
    )
    assert 'viewBox="0 0 1280 ' in forest
    assert 'x="1262"' in forest and 'text-anchor="end"' in forest
    assert "Qwen3-8B" in forest and "Gemma4-E4B" in forest
