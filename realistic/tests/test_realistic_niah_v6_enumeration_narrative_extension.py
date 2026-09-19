from __future__ import annotations

import json
from pathlib import Path

from scripts.build_realistic_niah_v6_enumeration_narrative_report import (
    read_style_block,
    render_report,
)
from scripts.rebuild_realistic_niah_v6_enumeration_report_from_embedded import (
    index_item_end_sensitivity_payload,
)
from scripts.validate_realistic_niah_v6_enumeration_narrative_report import (
    ReportParser,
    sha256,
    validate,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "NiaH_Enumeration_report.html"
SMOKE_REPORT = ROOT / "reports" / "_NiaH_Enumeration_report_smoke.html"
NATIVE_REPORT = ROOT / "reports" / "NiaH_Native-Thinking_report.html"


def _synthetic_gate(estimate: float, *, passed: bool = True) -> dict:
    return {
        "pass": passed,
        "estimate": estimate,
        "low": estimate - 0.05,
        "high": estimate + 0.05,
    }


def _extension_cell(prompt_mode: str, model: str) -> dict:
    layers = (
        [0, 5, 10, 15, 20, 25, 30, 35]
        if model == "Qwen3-8B"
        else [0, 6, 12, 18, 23, 29, 35, 41]
    )
    effects = []
    for index, layer in enumerate(layers):
        estimate = index / (len(layers) - 1)
        effects.append(
            {
                "layer": layer,
                "seed_clusters": 10,
                "pairs": 40,
                "full_donor_adoption": estimate,
                "full_donor_adoption_ci95_low": max(0.0, estimate - 0.05),
                "full_donor_adoption_ci95_high": min(1.0, estimate + 0.05),
                "adoption_specificity": estimate,
                "adoption_specificity_ci95_low": max(0.0, estimate - 0.05),
                "adoption_specificity_ci95_high": min(1.0, estimate + 0.05),
                "registered_numeric_valid": 1.0,
            }
        )
    relay_gates = {
        "terminal_state_patch_effect": _synthetic_gate(0.8),
        "post_terminal_suffix_specific_mediation": _synthetic_gate(0.5),
        "post_terminal_suffix_residual_equivalence": _synthetic_gate(0.4),
        "self_reset_is_nondamaging": _synthetic_gate(0.1),
        "answer_query_only_mediation": _synthetic_gate(0.3),
    }
    return {
        "prompt_mode": prompt_mode,
        "model_label": model,
        "answer_registered_pairs": 40,
        "answer_seed_clusters": 10,
        "answer_terminal_layer": layers[-1],
        "answer_terminal_adoption": 1.0,
        "answer_terminal_ci_low": 0.95,
        "answer_terminal_ci_high": 1.0,
        "answer_descriptive_onset": layers[3],
        "answer_layer_effects": effects,
        "relay_gates": relay_gates,
        "terminal_patch": relay_gates["terminal_state_patch_effect"],
        "suffix_mediation": relay_gates[
            "post_terminal_suffix_specific_mediation"
        ],
        "suffix_residual_ratio": relay_gates[
            "post_terminal_suffix_residual_equivalence"
        ],
        "self_reset_ratio": relay_gates["self_reset_is_nondamaging"],
        "query_mediation": relay_gates["answer_query_only_mediation"],
        "relay_planned_seed_count": 10,
        "relay_eligible_seed_count": 9,
        "relay_estimable": True,
        "relay_geometry": "suffix8",
        "relay_original_geometry": "suffix8",
        "relay_evidence_label": "original_registered_suffix8",
        "relay_geometry_amendment_sha256": None,
        "original_suffix8_relay": {
            "geometry": "suffix8",
            "estimable": True,
            "planned_seed_count": 10,
            "eligible_seed_count": 9,
            "geometry_not_applicable_full_seed_count": 1,
            "geometry_not_applicable_full_seeds": [1259],
            "not_estimable_reason": None,
        },
        "relay_not_estimable_reason": None,
        "relay_geometry_not_applicable_full_seed_count": 1,
        "relay_geometry_not_applicable_full_seeds": [1259],
        "partial_mediation_pass": True,
        "complete_mediation_not_claimed": True,
        "seed_aliasing": False,
        "artifact_hashes": {
            name: str(index) * 64
            for index, name in enumerate(
                (
                    "completion",
                    "answer_audit",
                    "layer_effects",
                    "relay_audit",
                    "claim_gates",
                ),
                start=1,
            )
        },
    }


def _bootstrap(estimate: float) -> dict:
    return {
        "estimate": estimate,
        "ci95": [estimate - 0.05, estimate + 0.05],
        "n_analysis_slot_seeds": 20,
        "bootstrap_samples": 10000,
        "bootstrap_random_seed": 20260829,
    }


def _sensitivity_analysis(model: str) -> dict:
    cells = {}
    roles = {
        "p2bank_at_p2": ("post_marker", "post_marker"),
        "p2bank_at_p0": ("post_marker", "p0_item_end"),
        "p0bank_at_p2": ("p0_item_end", "post_marker"),
        "p0bank_at_p0": ("p0_item_end", "p0_item_end"),
    }
    for index, (name, (bank_role, start_role)) in enumerate(roles.items()):
        cells[name] = {
            "cell": name,
            "bank_selection_anchor_role": bank_role,
            "intervention_start_anchor_role": start_role,
            "selected_failure_rate": 0.2 + index * 0.1,
            "registered_random_failure_rate": 0.1,
            "clean_failure_rate": 0.0 if index == 0 else None,
            "selected_minus_random_failure": _bootstrap(0.1 + index * 0.1),
        }
    contrast_specs = {
        "overall_item_end_minus_primary": ("p0bank_at_p0", "p2bank_at_p2"),
        "site_effect_for_p2_bank": ("p2bank_at_p0", "p2bank_at_p2"),
        "site_effect_for_p0_bank": ("p0bank_at_p0", "p0bank_at_p2"),
        "bank_effect_at_p0": ("p0bank_at_p0", "p2bank_at_p0"),
        "bank_effect_at_p2": ("p0bank_at_p2", "p2bank_at_p2"),
    }
    contrasts = {
        name: {"left": left, "right": right, **_bootstrap(0.2)}
        for name, (left, right) in contrast_specs.items()
    }
    return {
        "schema_version": "realistic_niah_v6_index_item_end_anchor_sensitivity_analysis_v1",
        "status": "PASS",
        "scientific_scope": "post_hoc_motivated_prospectively_frozen_sensitivity",
        "model_label": model,
        "prompt_mode": "enumeration_index",
        "fixed_k": 128 if model == "Qwen3-8B" else 8,
        "decision": "SUPPORTS_ANCHOR_SENSITIVITY",
        "decision_is_exploratory": True,
        "may_replace_primary_result": False,
        "may_reselect_k": False,
        "confirmation_authorized": False,
        "contract_sha256": "c" * 64,
        "cells": cells,
        "contrasts": contrasts,
        "generation_container_audit": {
            "status": (
                "PASS_AMENDED_APPENDABLE_CONTAINER_WITH_PRE_FREEZE_ROW_IDENTITY"
                if model == "Gemma4-E4B"
                else "PASS_EXACT_CONTAINER_HASH"
            )
        },
    }


def _synthetic_manifold() -> dict:
    payload = {
        "schema_version": "synthetic_render_only",
        "status": "PASS_DISCOVERY_FIT_CONFIRMATION_PROJECTION",
        "qualification": "DESCRIPTIVE_DISCOVERY_FIT_CONFIRMATION_PROJECTION",
        "fit_split": "discovery",
        "display_split": "confirmation",
        "running": {},
        "final": {},
    }
    for endpoint in ("running", "final"):
        for mode in ("enumeration_index", "enumeration_bullet"):
            for model in ("Qwen3-8B", "Gemma4-E4B"):
                payload[endpoint][f"{mode}|{model}"] = {
                    "default_layer": 0,
                    "layers": {
                        "0": {
                            "evr": [0.4, 0.2, 0.1],
                            "rows": [
                                [1253 + label, label, label / 10, label / 20, label / 30]
                                for label in range(1, 11)
                            ],
                        }
                    },
                    "discovery_rows": 20,
                    "confirmation_rows": 10,
                    "prompt_mode": mode,
                    "model_label": model,
                    "token_site": "item_end" if endpoint == "running" else "answer_query_v3",
                }
    return payload


def test_exact_answer_trace_extension_renders_native_mirrored_topology(
    tmp_path: Path,
) -> None:
    parser = ReportParser()
    source_report = SMOKE_REPORT if SMOKE_REPORT.exists() else REPORT
    parser.feed(source_report.read_text(encoding="utf-8"))
    data = json.loads(parser.manifest_text)
    data["native_template_css"] = read_style_block(NATIVE_REPORT)
    data["representation_manifold"] = _synthetic_manifold()
    data["answer_trace_extension"] = {
        "schema_version": "synthetic_render_only",
        "status": "PASS_COMPLETE",
        "extension_contract_sha256": "a" * 64,
        "cells": [
            _extension_cell(mode, model)
            for mode in ("enumeration_index", "enumeration_bullet")
            for model in ("Qwen3-8B", "Gemma4-E4B")
        ],
    }
    data["answer_trace_extension_summary_sha256"] = "b" * 64
    data["index_item_end_anchor_sensitivity"] = {
        "schema_version": "realistic_niah_v6_index_item_end_anchor_sensitivity_report_payload_v1",
        "status": "PASS_COMPLETE",
        "scientific_scope": "post_hoc_motivated_prospectively_frozen_discovery_only",
        "contract_sha256": "c" * 64,
        "analyses": {
            model: _sensitivity_analysis(model)
            for model in ("Qwen3-8B", "Gemma4-E4B")
        },
        "analysis_source_sha256": {
            "Qwen3-8B": "d" * 64,
            "Gemma4-E4B": "e" * 64,
        },
        "primary_confirmation_replaced": False,
        "k_reselected": False,
    }

    document = render_report(data)
    rendered = ReportParser()
    rendered.feed(document)

    assert rendered.figure_count == 21
    assert rendered.figure_title_count == 21
    assert rendered.figcaption_count == 21
    assert rendered.figure_primer_count == 21
    assert rendered.experiment_frame_count == 16
    assert rendered.svg_count >= 18
    assert rendered.svg_viewbox_count == rendered.svg_count
    assert rendered.svg_role_img_count == rendered.svg_count
    assert rendered.svg_titled_count == rendered.svg_count
    assert rendered.canvas_aria_count == 2
    assert (
        ".paper-figure{margin:24px 0;overflow-x:auto;"
        "overscroll-behavior-inline:contain}" in document
    )
    assert (
        ".paper-figure>.paper-chart,.paper-figure>.figure-stack .paper-chart"
        "{width:980px;min-width:980px;max-width:none}" in document
    )
    assert document.count('class="bar-value bar-value-inverse"') == 4
    assert document.count('class="scope-label"') == 4
    assert document.count('data-series-offset="-6"') == 2
    assert document.count('data-series-offset="6"') == 2
    assert "§6.3/§6.4 已完成 exact assay 对齐" in document
    assert "Index item-end anchor 的 2×2 discovery-only 敏感性" in document
    assert "尚未逐层复制 Native exact assay" not in document
    assert "Scientific gate FAIL 只适用于有数值支持但区间未通过的格" in document
    assert "A.9 Native-thinking ↔ Enumeration：mechanism-claim 对照" in document
    assert "Mechanistic fidelity 的操作化检查" in document
    assert "mechanistically faithful controlled proxy" in document
    assert "相同 recurrent behavioral consequence" in document

    rendered_report = tmp_path / "report.html"
    rendered_report.write_text(document, encoding="utf-8")
    external = json.loads(
        (ROOT / "reports" / "v6_enumeration_narrative_report_manifest_v3.json")
        .read_text(encoding="utf-8")
    )
    external.update(
        {
            "status": "PASS",
            "format_revision": "native_mirrored_v6_index_anchor_and_answer_trace_exact",
            "output": str(rendered_report),
            "output_sha256": sha256(rendered_report),
            "answer_trace_extension_complete": True,
            "answer_trace_extension_summary_sha256": "b" * 64,
            "index_item_end_anchor_sensitivity_complete": True,
            "index_item_end_anchor_sensitivity_source_sha256": {
                "Qwen3-8B": "d" * 64,
                "Gemma4-E4B": "e" * 64,
            },
        }
    )
    external_manifest = tmp_path / "manifest.json"
    external_manifest.write_text(json.dumps(external), encoding="utf-8")
    validation = validate(rendered_report, external_manifest, NATIVE_REPORT)
    assert validation["status"] == "PASS", validation["errors"]
    assert validation["visual_layout_contract_pass"] is True


def test_zero_support_relay_cell_is_rendered_and_validated_as_not_estimable(
    tmp_path: Path,
) -> None:
    parser = ReportParser()
    source_report = SMOKE_REPORT if SMOKE_REPORT.exists() else REPORT
    parser.feed(source_report.read_text(encoding="utf-8"))
    data = json.loads(parser.manifest_text)
    data["native_template_css"] = read_style_block(NATIVE_REPORT)
    data["representation_manifold"] = _synthetic_manifold()
    cells = [
        _extension_cell(mode, model)
        for mode in ("enumeration_index", "enumeration_bullet")
        for model in ("Qwen3-8B", "Gemma4-E4B")
    ]
    cell = cells[-1]
    cell["relay_estimable"] = False
    cell["relay_not_estimable_reason"] = (
        "not applicable: a trace item is shorter than the requested suffix8 geometry"
    )
    cell["relay_eligible_seed_count"] = 0
    cell["relay_geometry_not_applicable_full_seed_count"] = 10
    cell["relay_geometry_not_applicable_full_seeds"] = list(range(1254, 1264))
    cell["original_suffix8_relay"].update(
        {
            "estimable": False,
            "eligible_seed_count": 0,
            "geometry_not_applicable_full_seed_count": 10,
            "geometry_not_applicable_full_seeds": list(range(1254, 1264)),
            "not_estimable_reason": cell["relay_not_estimable_reason"],
        }
    )
    cell["partial_mediation_pass"] = False
    for gate in cell["relay_gates"].values():
        gate.update(
            {
                "pass": False,
                "estimate": None,
                "low": None,
                "high": None,
                "estimable": False,
            }
        )
    data["answer_trace_extension"] = {
        "schema_version": "synthetic_render_only",
        "status": "PASS_COMPLETE",
        "extension_contract_sha256": "a" * 64,
        "cells": cells,
    }
    data["answer_trace_extension_summary_sha256"] = "b" * 64
    data["index_item_end_anchor_sensitivity"] = {
        "schema_version": "realistic_niah_v6_index_item_end_anchor_sensitivity_report_payload_v1",
        "status": "PASS_COMPLETE",
        "scientific_scope": "post_hoc_motivated_prospectively_frozen_discovery_only",
        "contract_sha256": "c" * 64,
        "analyses": {
            model: _sensitivity_analysis(model)
            for model in ("Qwen3-8B", "Gemma4-E4B")
        },
        "analysis_source_sha256": {
            "Qwen3-8B": "d" * 64,
            "Gemma4-E4B": "e" * 64,
        },
        "primary_confirmation_replaced": False,
        "k_reselected": False,
    }

    document = render_report(data)

    assert "3/4 格在各自明确标注的 geometry 下具有数值支持" in document
    assert "其余 1/4 格为 geometry 不可估计" in document
    assert "0/10; full-NA=1254" in document
    assert "不可估计：not applicable" in document
    assert "全格 0/10 eligible 的 cell 不画成零效应" in document

    rendered_report = tmp_path / "report_zero_support.html"
    rendered_report.write_text(document, encoding="utf-8")
    external = json.loads(
        (ROOT / "reports" / "v6_enumeration_narrative_report_manifest_v3.json")
        .read_text(encoding="utf-8")
    )
    external.update(
        {
            "status": "PASS",
            "format_revision": "native_mirrored_v6_index_anchor_and_answer_trace_exact",
            "output": str(rendered_report),
            "output_sha256": sha256(rendered_report),
            "answer_trace_extension_complete": True,
            "answer_trace_extension_summary_sha256": "b" * 64,
            "index_item_end_anchor_sensitivity_complete": True,
            "index_item_end_anchor_sensitivity_source_sha256": {
                "Qwen3-8B": "d" * 64,
                "Gemma4-E4B": "e" * 64,
            },
        }
    )
    external_manifest = tmp_path / "manifest_zero_support.json"
    external_manifest.write_text(json.dumps(external), encoding="utf-8")
    validation = validate(rendered_report, external_manifest, NATIVE_REPORT)
    assert validation["status"] == "PASS", validation["errors"]


def test_bullet_suffix4_replication_is_labeled_without_hiding_suffix8(
    tmp_path: Path,
) -> None:
    parser = ReportParser()
    source_report = SMOKE_REPORT if SMOKE_REPORT.exists() else REPORT
    parser.feed(source_report.read_text(encoding="utf-8"))
    data = json.loads(parser.manifest_text)
    data["native_template_css"] = read_style_block(NATIVE_REPORT)
    data["representation_manifold"] = _synthetic_manifold()
    cells = [
        _extension_cell(mode, model)
        for mode in ("enumeration_index", "enumeration_bullet")
        for model in ("Qwen3-8B", "Gemma4-E4B")
    ]
    amendment_hash = "f" * 64
    for cell in cells[2:]:
        cell.update(
            {
                "relay_geometry": "suffix4",
                "relay_evidence_label": (
                    "post_hoc_task_adapted_bullet_relay_replication"
                ),
                "relay_geometry_amendment_sha256": amendment_hash,
            }
        )
    cells[-1]["original_suffix8_relay"].update(
        {
            "estimable": False,
            "eligible_seed_count": 0,
            "geometry_not_applicable_full_seed_count": 10,
            "geometry_not_applicable_full_seeds": list(range(1254, 1264)),
            "not_estimable_reason": (
                "not applicable: a trace item is shorter than the requested "
                "suffix8 geometry"
            ),
        }
    )
    data["answer_trace_extension"] = {
        "schema_version": "synthetic_render_only",
        "status": "PASS_COMPLETE",
        "extension_contract_sha256": "a" * 64,
        "relay_geometry_amendment_sha256": amendment_hash,
        "cells": cells,
    }
    data["answer_trace_extension_summary_sha256"] = "b" * 64
    data["index_item_end_anchor_sensitivity"] = {
        "schema_version": "realistic_niah_v6_index_item_end_anchor_sensitivity_report_payload_v1",
        "status": "PASS_COMPLETE",
        "scientific_scope": "post_hoc_motivated_prospectively_frozen_discovery_only",
        "contract_sha256": "c" * 64,
        "analyses": {
            model: _sensitivity_analysis(model)
            for model in ("Qwen3-8B", "Gemma4-E4B")
        },
        "analysis_source_sha256": {
            "Qwen3-8B": "d" * 64,
            "Gemma4-E4B": "e" * 64,
        },
        "primary_confirmation_replaced": False,
        "k_reselected": False,
    }

    document = render_report(data)

    assert "4/4 格在各自明确标注的 geometry 下具有数值支持" in document
    assert "Bullet 两格统一使用在 suffix4" in document
    assert "原 suffix8 审计仍保留，其中不可估计格为：Bullet·Gemma4-E4B" in document
    assert "post_hoc_task_adapted_bullet_relay_replication" in document
    assert "不把 task adaptation 冒充原合同" in document

    rendered_report = tmp_path / "report_suffix4.html"
    rendered_report.write_text(document, encoding="utf-8")
    external = json.loads(
        (ROOT / "reports" / "v6_enumeration_narrative_report_manifest_v3.json")
        .read_text(encoding="utf-8")
    )
    external.update(
        {
            "status": "PASS",
            "format_revision": "native_mirrored_v6_index_anchor_and_answer_trace_exact",
            "output": str(rendered_report),
            "output_sha256": sha256(rendered_report),
            "answer_trace_extension_complete": True,
            "answer_trace_extension_summary_sha256": "b" * 64,
            "index_item_end_anchor_sensitivity_complete": True,
            "index_item_end_anchor_sensitivity_source_sha256": {
                "Qwen3-8B": "d" * 64,
                "Gemma4-E4B": "e" * 64,
            },
        }
    )
    external_manifest = tmp_path / "manifest_suffix4.json"
    external_manifest.write_text(json.dumps(external), encoding="utf-8")
    validation = validate(rendered_report, external_manifest, NATIVE_REPORT)
    assert validation["status"] == "PASS", validation["errors"]


def test_index_item_end_payload_requires_both_frozen_20_seed_analyses(
    tmp_path: Path,
) -> None:
    qwen = tmp_path / "qwen.json"
    gemma = tmp_path / "gemma.json"
    qwen.write_text(json.dumps(_sensitivity_analysis("Qwen3-8B")), encoding="utf-8")
    gemma.write_text(json.dumps(_sensitivity_analysis("Gemma4-E4B")), encoding="utf-8")

    payload = index_item_end_sensitivity_payload(qwen, gemma)

    assert payload["status"] == "PASS_COMPLETE"
    assert payload["contract_sha256"] == "c" * 64
    assert payload["primary_confirmation_replaced"] is False
    assert payload["k_reselected"] is False
