#!/usr/bin/env python3
"""Validate the self-contained V6 Enumeration narrative mechanism report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


REQUIRED_SECTIONS = {
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
}
EXPECTED_PARSER_SITES = {
    "pre_marker",
    "marker_end",
    "pre_city",
    "city_end",
    "city_unit_end",
    "item_end",
    "post_boundary",
    "list_cut",
    "answer_query",
    "answer_query_v3",
}
EXPECTED_PARSER_GATES = {
    "registered_success",
    "enumeration_format_compliant",
    "strict_listed_total_matches_length",
    "exact_ordered_gold_pairs",
    "marker_kind_compliant",
    "parser_forward_one_to_one",
    "item_count_matches_gold",
}
EXPECTED_K = {
    "enumeration_index|Qwen3-8B": 128,
    "enumeration_index|Gemma4-E4B": 8,
    "enumeration_bullet|Qwen3-8B": 96,
    "enumeration_bullet|Gemma4-E4B": 2,
}
EXPECTED_ANSWER_LAYERS = {
    "Qwen3-8B": [0, 5, 10, 15, 20, 25, 30, 35],
    "Gemma4-E4B": [0, 6, 12, 18, 23, 29, 35, 41],
}
EXPECTED_RELAY_GATES = {
    "terminal_state_patch_effect",
    "post_terminal_suffix_specific_mediation",
    "post_terminal_suffix_residual_equivalence",
    "self_reset_is_nondamaging",
    "answer_query_only_mediation",
}
EXPECTED_SENSITIVITY_CELLS = {
    "p2bank_at_p2",
    "p2bank_at_p0",
    "p0bank_at_p2",
    "p0bank_at_p0",
}
EXPECTED_SENSITIVITY_CONTRASTS = {
    "overall_item_end_minus_primary",
    "site_effect_for_p2_bank",
    "site_effect_for_p0_bank",
    "bank_effect_at_p0",
    "bank_effect_at_p2",
}
EXPECTED_RAW_STRICT = {
    "enumeration_index|Qwen3-8B": {
        "discovery": (198, 200),
        "confirmation": (99, 100),
        "pooled": (297, 300),
    },
    "enumeration_index|Gemma4-E4B": {
        "discovery": (180, 200),
        "confirmation": (73, 100),
        "pooled": (253, 300),
    },
    "enumeration_bullet|Qwen3-8B": {
        "discovery": (197, 200),
        "confirmation": (100, 100),
        "pooled": (297, 300),
    },
    "enumeration_bullet|Gemma4-E4B": {
        "discovery": (162, 200),
        "confirmation": (67, 100),
        "pooled": (229, 300),
    },
}
EXPECTED_MODE_RAW_STRICT = {
    "enumeration_index": (550, 600),
    "enumeration_bullet": (526, 600),
}
VISUAL_LAYOUT_CSS_MARKERS = {
    "figure horizontal overflow containment": (
        ".paper-figure{margin:24px 0;overflow-x:auto;"
        "overscroll-behavior-inline:contain}"
    ),
    "figure title spacing": (
        ".paper-figure>.figure-title{margin:0 0 14px;line-height:1.38}"
    ),
    "caption separation": (
        ".paper-figure>figcaption{margin:14px 0 0;padding-top:12px;"
        "border-top:1px solid #e5e9ef"
    ),
    "canvas overflow containment": ".cloud-panel{min-width:0;overflow:hidden;",
    "canvas toolbar wrapping": ".cloud-head{display:flex;flex-wrap:wrap;",
    "mobile chart legibility": (
        ".paper-figure>.paper-chart,.paper-figure>.figure-stack .paper-chart"
        "{width:980px;min-width:980px;max-width:none}"
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style_block(path: Path) -> str:
    match = re.search(
        r"<style(?:\s[^>]*)?>(.*?)</style>",
        path.read_text(encoding="utf-8"),
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"Missing inline style block: {path}")
    return match.group(1).strip()


class ReportParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.external: list[str] = []
        self.cell_keys: list[str] = []
        self.table_count = 0
        self.figure_count = 0
        self.figcaption_count = 0
        self.figure_title_count = 0
        self.experiment_frame_count = 0
        self.figure_primer_count = 0
        self.section_conclusion_count = 0
        self.subsection_conclusion_count = 0
        self.canvas_count = 0
        self.canvas_aria_count = 0
        self.svg_count = 0
        self.svg_viewbox_count = 0
        self.svg_role_img_count = 0
        self.svg_titled_count = 0
        self._svg_depth = 0
        self._manifest_depth = 0
        self._manifest_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set(str(values.get("class", "")).split())
        if values.get("id"):
            self.ids.append(str(values["id"]))
        if values.get("data-cell"):
            self.cell_keys.append(str(values["data-cell"]))
        if tag == "table":
            self.table_count += 1
        if tag == "figure":
            self.figure_count += 1
        if tag == "figcaption":
            self.figcaption_count += 1
        if "figure-title" in classes:
            self.figure_title_count += 1
        if "experiment-frame" in classes:
            self.experiment_frame_count += 1
        if "figure-primer" in classes:
            self.figure_primer_count += 1
        if "section-conclusion" in classes:
            self.section_conclusion_count += 1
        if "subsection-conclusion" in classes:
            self.subsection_conclusion_count += 1
        if tag == "svg":
            self.svg_count += 1
            if values.get("viewbox"):
                self.svg_viewbox_count += 1
            if values.get("role") == "img":
                self.svg_role_img_count += 1
            self._svg_depth = 1
        elif self._svg_depth:
            if tag == "title" and self._svg_depth == 1:
                self.svg_titled_count += 1
            self._svg_depth += 1
        if tag == "canvas":
            self.canvas_count += 1
            if values.get("aria-label"):
                self.canvas_aria_count += 1
        for key in ("src", "href"):
            target = values.get(key)
            if target and not str(target).startswith(("#", "data:")):
                self.external.append(str(target))
        if tag == "script" and values.get("id") == "report-manifest":
            self._manifest_depth = 1
        elif self._manifest_depth:
            self._manifest_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._svg_depth:
            self._svg_depth -= 1
        if self._manifest_depth:
            self._manifest_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._manifest_depth:
            self._manifest_parts.append(data)

    @property
    def manifest_text(self) -> str:
        return "".join(self._manifest_parts).strip()


def validate(
    report: Path,
    manifest: Path,
    native_template_report: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    visual_errors: list[str] = []
    text = report.read_text(encoding="utf-8")
    parser = ReportParser()
    parser.feed(text)
    if not (
        parser.svg_count >= 18
        and parser.svg_count
        == parser.svg_viewbox_count
        == parser.svg_role_img_count
        == parser.svg_titled_count
    ):
        visual_errors.append(
            "expected at least 18 accessible inline SVG charts with complete metadata, found "
            f"svg={parser.svg_count}, viewBox={parser.svg_viewbox_count}, "
            f"role=img={parser.svg_role_img_count}, root titles={parser.svg_titled_count}"
        )
    if parser.canvas_aria_count != 2:
        visual_errors.append(
            "both interactive canvases must have accessible labels: "
            f"found {parser.canvas_aria_count}/2"
        )
    for label, marker in VISUAL_LAYOUT_CSS_MARKERS.items():
        if marker not in text:
            visual_errors.append(f"missing visual CSS contract: {label}")

    svg_blocks = re.findall(
        r"<svg\b([^>]*)>(.*?)</svg>", text, flags=re.DOTALL | re.IGNORECASE
    )
    if len(svg_blocks) != parser.svg_count:
        visual_errors.append(
            "inline SVG roots could not be paired for boundary validation: "
            f"paired={len(svg_blocks)}, parsed={parser.svg_count}"
        )
    for svg_index, (attributes, body) in enumerate(svg_blocks, start=1):
        viewbox = re.search(
            r'\bviewBox="([+\-]?[0-9.]+)\s+([+\-]?[0-9.]+)\s+'
            r'([0-9.]+)\s+([0-9.]+)"',
            attributes,
            flags=re.IGNORECASE,
        )
        if viewbox is None:
            continue
        height = float(viewbox.group(4))
        for text_tag in re.finditer(r"<text\b([^>]*)>", body, flags=re.IGNORECASE):
            y_match = re.search(
                r'\by="([+\-]?[0-9.]+)"', text_tag.group(1), flags=re.IGNORECASE
            )
            if y_match is None:
                continue
            y = float(y_match.group(1))
            if y < 8 or y > height - 8:
                visual_errors.append(
                    f"SVG {svg_index} text baseline y={y:g} is too close to "
                    f"the viewBox boundary [0,{height:g}]"
                )

    visible_svg_text = [
        re.sub(r"<[^>]+>", "", match.group(1))
        for match in re.finditer(
            r"<text\b[^>]*>(.*?)</text>", text, flags=re.DOTALL | re.IGNORECASE
        )
    ]
    for machine_note in (
        "decode_head_ablation_steps",
        "query_local;",
        "registered_query_through_city_prefix",
        "query_through_carrier",
    ):
        if any(machine_note in label for label in visible_svg_text):
            visual_errors.append(
                f"machine-length scope note remains visible inside a timeline bar: {machine_note}"
            )
    if text.count('class="scope-label"') != 4:
        visual_errors.append("expected four short, bar-contained scope labels")
    if text.count('class="bar-value bar-value-inverse"') != 4:
        visual_errors.append("behavioral-accuracy values are not all anchored inside bars")
    for offset in (-6, 6):
        if text.count(
            f'class="multihop-series" data-series-offset="{offset}"'
        ) != 2:
            visual_errors.append(
                f"multihop model series offset {offset:+d}px is not present in both panels"
            )
    errors.extend(visual_errors)
    duplicate_ids = sorted({value for value in parser.ids if parser.ids.count(value) > 1})
    missing_sections = sorted(REQUIRED_SECTIONS - set(parser.ids))
    if duplicate_ids:
        errors.append(f"duplicate ids: {duplicate_ids}")
    if missing_sections:
        errors.append(f"missing sections: {missing_sections}")
    if parser.external:
        errors.append(f"external assets: {parser.external}")
    if parser.table_count < 30:
        errors.append(f"expected at least 30 evidence tables, found {parser.table_count}")
    if not (
        parser.figure_count
        == parser.figcaption_count
        == parser.figure_title_count
        == 21
    ):
        errors.append(
            "expected 21 figures with one title and one caption each, found "
            f"figures={parser.figure_count}, titles={parser.figure_title_count}, "
            f"captions={parser.figcaption_count}"
        )
    if parser.experiment_frame_count != 16:
        errors.append(
            f"expected 16 complete experiment frames, found {parser.experiment_frame_count}"
        )
    if parser.figure_primer_count != 21:
        errors.append(
            f"expected 21 figure reading primers, found {parser.figure_primer_count}"
        )
    if text.count("<strong>坐标怎么读</strong>") != parser.figure_count:
        errors.append(
            "every figure must have an explicit coordinate/axis reading contract"
        )
    if parser.section_conclusion_count != len(REQUIRED_SECTIONS):
        errors.append(
            "every required section must end in a section conclusion: "
            f"found {parser.section_conclusion_count}/{len(REQUIRED_SECTIONS)}"
        )
    if parser.subsection_conclusion_count < 13:
        errors.append(
            "expected at least 13 local experiment/appendix conclusions, found "
            f"{parser.subsection_conclusion_count}"
        )
    for label in ("实验目的", "实验设定", "计算方法", "结果", "分析", "简单例子"):
        count = text.count(f'class="experiment-label">{label}')
        if count != parser.experiment_frame_count:
            errors.append(
                f"experiment label {label} appears {count} times; "
                f"expected {parser.experiment_frame_count}"
            )
    if parser.canvas_count != 2:
        errors.append(f"expected two interactive 3D canvases, found {parser.canvas_count}")
    geometry_match = re.search(
        r"const ENUM_GEOMETRY=(.*?);\s*const ENUM_COLORS=", text, flags=re.DOTALL
    )
    if geometry_match is None:
        errors.append("interactive 3D coordinate payload is absent")
    else:
        try:
            geometry = json.loads(geometry_match.group(1))
        except json.JSONDecodeError as error:
            geometry = {}
            errors.append(f"interactive 3D coordinate payload is invalid: {error}")
        expected_cells = set(EXPECTED_K)
        if geometry:
            if geometry.get("status") != (
                "PASS_DISCOVERY_FIT_CONFIRMATION_PROJECTION"
            ):
                errors.append("interactive 3D payload lost its PASS status")
            for endpoint in ("running", "final"):
                endpoint_cells = geometry.get(endpoint, {})
                if not isinstance(endpoint_cells, dict) or set(endpoint_cells) != expected_cells:
                    errors.append(f"interactive 3D payload lost {endpoint} cells")
                    continue
                for cell_key, cell in endpoint_cells.items():
                    layers = cell.get("layers", {}) if isinstance(cell, dict) else {}
                    default_layer = (
                        str(cell.get("default_layer"))
                        if isinstance(cell, dict)
                        else ""
                    )
                    if not layers or default_layer not in layers:
                        errors.append(
                            f"interactive 3D payload lost layers for "
                            f"{endpoint}/{cell_key}"
                        )
                        continue
                    for layer, layer_data in layers.items():
                        rows = (
                            layer_data.get("rows", [])
                            if isinstance(layer_data, dict)
                            else []
                        )
                        evr = (
                            layer_data.get("evr", [])
                            if isinstance(layer_data, dict)
                            else []
                        )
                        if (
                            len(evr) != 3
                            or not rows
                            or any(
                                not isinstance(row, list) or len(row) != 5
                                for row in rows
                            )
                        ):
                            errors.append(
                                f"interactive 3D coordinates are invalid for "
                                f"{endpoint}/{cell_key}/L{layer}"
                            )
                            break
    if set(parser.cell_keys) != set(EXPECTED_K) or len(parser.cell_keys) != 4:
        errors.append(f"four-cell chain changed: {parser.cell_keys}")
    if native_template_report is not None:
        try:
            native_css = style_block(native_template_report)
            report_css = style_block(report)
        except ValueError as error:
            errors.append(str(error))
        else:
            if not report_css.startswith(native_css):
                errors.append("report does not inherit the exact Native-thinking CSS block")
        native_text = native_template_report.read_text(encoding="utf-8")
        section_pattern = re.compile(
            r'<section\b[^>]*\bid="([^"]+)"', flags=re.IGNORECASE
        )
        native_section_order = section_pattern.findall(native_text)
        report_section_order = section_pattern.findall(text)
        if report_section_order != native_section_order:
            errors.append(
                "report section order differs from Native-thinking: "
                f"{report_section_order} != {native_section_order}"
            )
        nav_pattern = re.compile(r"<nav>(.*?)</nav>", flags=re.DOTALL | re.IGNORECASE)
        native_nav_match = nav_pattern.search(native_text)
        report_nav_match = nav_pattern.search(text)
        native_nav = (
            re.findall(r'href="#([^"]+)"', native_nav_match.group(1))
            if native_nav_match
            else []
        )
        report_nav = (
            re.findall(r'href="#([^"]+)"', report_nav_match.group(1))
            if report_nav_match
            else []
        )
        if report_nav != native_nav:
            errors.append(
                f"report navigation topology differs from Native-thinking: "
                f"{report_nav} != {native_nav}"
            )
        for wrapper in ('<article class="page">', "<header>", "<main>"):
            if text.count(wrapper) != 1:
                errors.append(f"report must contain exactly one Native wrapper {wrapper}")

    for phrase in (
        "本报告严格复刻 Native-thinking 报告的叙事与视觉拓扑",
        "StrictExactAccuracy",
        "BalancedAccuracy",
        "full-state teacher-forced commit→query direct edge",
        "Δattn",
        "selected damage",
        "deformation",
        "depth = max d",
        "necessity = utility",
        "low-dimensional count component",
        "margin(h)=",
        "NO_DIRECTIONAL_SPECIFIC_SUPPORT",
        "事后机制诊断不是 fresh confirmation",
        "显式 ordinal 只提供 address/progress cue，不提供 city content",
        "every position from the registered query through the final grammar-carrier token, inclusive",
        "overall_item_end_minus_primary",
        "p0_item_end",
        "post_hoc_motivated_prospectively_frozen_discovery_only",
        "full item span",
        "FRESH_CAUSAL_OUTCOME_REPLICATION_COMPLETE",
        "POSTHOC_AGGREGATE_REPARSE_AFTER_ONE_SCHEMA_SMOKE_ROW",
        "Full-state、content-bound 的多步 continuation",
        "discovery states fit StandardScaler/PCA3",
        "低维 loop、update 与 stop 为 0/4",
        "strict_causal_eligible",
        "Gold N 和 final Total 不会构造、补齐或选择 item sequence",
        "literal_baseline_token_prefix",
        "text_exact_boundary_retokenization",
        "generated_known_city_ordinals_any_surface",
        "skip/reorder/deduplicate/repair forbidden",
        "Index 为 550/600=91.7%",
        "Bullet 为 526/600=87.7%",
        "replacement-filtered cohort",
        "Native-thinking CSS/topology mirrored",
    ):
        if phrase not in text:
            errors.append(f"missing required qualification: {phrase}")

    try:
        embedded = json.loads(parser.manifest_text)
    except (json.JSONDecodeError, TypeError) as error:
        embedded = {}
        errors.append(f"invalid embedded report manifest: {error}")
    if embedded:
        cells = embedded.get("cells", {})
        if set(cells) != set(EXPECTED_K):
            errors.append("embedded manifest lost a model×grammar cell")
        for key, expected_k in EXPECTED_K.items():
            if int(cells.get(key, {}).get("selected_k", -1)) != expected_k:
                errors.append(f"frozen K changed for {key}")
            full = (
                cells.get(key, {})
                .get("full_commit_to_query", {})
                .get("confirmation", {})
            )
            if not bool(full.get("strong_direct_gate_pass", False)):
                errors.append(f"full commit direct edge not sealed for {key}")
            if cells.get(key, {}).get("ncc", {}).get("ncc_effect_status") != (
                "NO_DIRECTIONAL_SPECIFIC_SUPPORT"
            ):
                errors.append(f"NCC null changed for {key}")
            narrow = cells.get(key, {}).get("narrow_loop", {})
            if (
                narrow.get("native_loop_pass") is not False
                or narrow.get("update_pass") is not False
                or narrow.get("stop_pass") is not False
                or narrow.get("group_gates")
                != {"commit_to_retrieval": False, "stop": False, "update": False}
            ):
                errors.append(f"low-dimensional update/stop null changed for {key}")
            multihop = cells.get(key, {}).get("full_item_multihop", {})
            if (
                not bool(multihop.get("primary_depth4_strong_gate_pass", False))
                or "depth_4" not in multihop
            ):
                errors.append(f"V3 multihop cell gate changed for {key}")
            for phase in ("discovery", "confirmation"):
                diagnostic = (
                    cells.get(key, {})
                    .get("targeted_retrieval", {})
                    .get("continuous", {})
                    .get(phase, {})
                )
                if diagnostic.get("analysis_status") != (
                    "POSTHOC_DIAGNOSTIC_SPLIT_REUSE"
                ):
                    errors.append(f"diagnostic label changed for {key}/{phase}")
                if key.startswith("enumeration_bullet|"):
                    terminal_diagnostic = (
                        cells.get(key, {})
                        .get("terminal", {})
                        .get("local_diagnostic", {})
                        .get(phase, {})
                    )
                    if terminal_diagnostic.get("analysis_status") != (
                        "POSTHOC_DIAGNOSTIC_SPLIT_REUSE"
                    ):
                        errors.append(
                            f"local terminal diagnostic label changed for {key}/{phase}"
                        )
                if key == "enumeration_bullet|Gemma4-E4B":
                    carrier_diagnostic = (
                        cells.get(key, {})
                        .get("carrier", {})
                        .get("decode_aligned_diagnostic", {})
                        .get(phase, {})
                    )
                    if carrier_diagnostic.get("analysis_status") != (
                        "POSTHOC_DIAGNOSTIC_SPLIT_REUSE"
                    ):
                        errors.append(
                            f"carrier diagnostic label changed for {key}/{phase}"
                        )

        item_end_sensitivity = embedded.get(
            "index_item_end_anchor_sensitivity", {}
        )
        sensitivity_analyses = item_end_sensitivity.get("analyses", {})
        sensitivity_source_hashes = item_end_sensitivity.get(
            "analysis_source_sha256", {}
        )
        if (
            item_end_sensitivity.get("status") != "PASS_COMPLETE"
            or item_end_sensitivity.get("schema_version")
            != "realistic_niah_v6_index_item_end_anchor_sensitivity_report_payload_v1"
            or item_end_sensitivity.get("scientific_scope")
            != "post_hoc_motivated_prospectively_frozen_discovery_only"
            or item_end_sensitivity.get("primary_confirmation_replaced") is not False
            or item_end_sensitivity.get("k_reselected") is not False
            or set(sensitivity_analyses) != {"Qwen3-8B", "Gemma4-E4B"}
            or set(sensitivity_source_hashes) != {"Qwen3-8B", "Gemma4-E4B"}
            or any(len(str(value)) != 64 for value in sensitivity_source_hashes.values())
            or len(str(item_end_sensitivity.get("contract_sha256", ""))) != 64
        ):
            errors.append("Index item-end 2x2 sensitivity report payload is incomplete")
        observed_contract_hashes: set[str] = set()
        for model, expected_k in (("Qwen3-8B", 128), ("Gemma4-E4B", 8)):
            analysis = sensitivity_analyses.get(model, {})
            cells_2x2 = analysis.get("cells", {})
            contrasts_2x2 = analysis.get("contrasts", {})
            if (
                analysis.get("schema_version")
                != "realistic_niah_v6_index_item_end_anchor_sensitivity_analysis_v1"
                or analysis.get("status") != "PASS"
                or analysis.get("scientific_scope")
                != "post_hoc_motivated_prospectively_frozen_sensitivity"
                or analysis.get("model_label") != model
                or analysis.get("prompt_mode") != "enumeration_index"
                or int(analysis.get("fixed_k", -1)) != expected_k
                or analysis.get("decision_is_exploratory") is not True
                or analysis.get("may_replace_primary_result") is not False
                or analysis.get("may_reselect_k") is not False
                or analysis.get("confirmation_authorized") is not False
                or set(cells_2x2) != EXPECTED_SENSITIVITY_CELLS
                or set(contrasts_2x2) != EXPECTED_SENSITIVITY_CONTRASTS
            ):
                errors.append(f"Index item-end sensitivity contract changed for {model}")
            contract_hash_2x2 = str(analysis.get("contract_sha256", ""))
            if len(contract_hash_2x2) == 64:
                observed_contract_hashes.add(contract_hash_2x2)
            for cell_name in EXPECTED_SENSITIVITY_CELLS:
                cell_2x2 = cells_2x2.get(cell_name, {})
                effect = cell_2x2.get("selected_minus_random_failure", {})
                ci95 = effect.get("ci95", [])
                if (
                    cell_2x2.get("cell") != cell_name
                    or int(effect.get("n_analysis_slot_seeds", -1)) != 20
                    or not isinstance(ci95, list)
                    or len(ci95) != 2
                ):
                    errors.append(f"2x2 cell estimand changed for {model}/{cell_name}")
            for contrast_name in EXPECTED_SENSITIVITY_CONTRASTS:
                contrast = contrasts_2x2.get(contrast_name, {})
                ci95 = contrast.get("ci95", [])
                if (
                    int(contrast.get("n_analysis_slot_seeds", -1)) != 20
                    or not isinstance(ci95, list)
                    or len(ci95) != 2
                ):
                    errors.append(
                        f"2x2 contrast estimand changed for {model}/{contrast_name}"
                    )
            if model == "Gemma4-E4B" and analysis.get(
                "generation_container_audit", {}
            ).get("status") != (
                "PASS_AMENDED_APPENDABLE_CONTAINER_WITH_PRE_FREEZE_ROW_IDENTITY"
            ):
                errors.append("Gemma item-end sensitivity row-identity audit is absent")
        if observed_contract_hashes != {
            str(item_end_sensitivity.get("contract_sha256", ""))
        }:
            errors.append("Index item-end sensitivity contract hashes disagree")

        answer_trace_extension = embedded.get("answer_trace_extension", {})
        extension_cells = answer_trace_extension.get("cells", [])
        extension_by_key = {
            f"{cell.get('prompt_mode')}|{cell.get('model_label')}": cell
            for cell in extension_cells
        }
        if (
            answer_trace_extension.get("status") != "PASS_COMPLETE"
            or set(extension_by_key) != set(EXPECTED_K)
            or len(extension_cells) != 4
        ):
            errors.append("Native-isomorphic answer/trace extension is not complete")
        contract_hash = str(
            answer_trace_extension.get("extension_contract_sha256", "")
        )
        if len(contract_hash) != 64:
            errors.append("answer/trace extension lacks one frozen contract hash")
        for key in EXPECTED_K:
            cell = extension_by_key.get(key, {})
            model = key.split("|", 1)[1]
            layer_rows = cell.get("answer_layer_effects", [])
            observed_layers = [int(row.get("layer", -1)) for row in layer_rows]
            if observed_layers != EXPECTED_ANSWER_LAYERS[model]:
                errors.append(f"answer-query frozen layer grid changed for {key}")
            if set(cell.get("relay_gates", {})) != EXPECTED_RELAY_GATES:
                errors.append(f"terminal relay 2x3 gate registry changed for {key}")
            if (
                cell.get("complete_mediation_not_claimed") is not True
                or cell.get("seed_aliasing") is not False
            ):
                errors.append(f"terminal relay claim boundary changed for {key}")
            try:
                relay_planned = int(cell.get("relay_planned_seed_count", -1))
                relay_eligible = int(cell.get("relay_eligible_seed_count", -1))
                relay_estimable = bool(
                    cell.get("relay_estimable", relay_eligible > 0)
                )
                relay_full_na_count = int(
                    cell.get(
                        "relay_geometry_not_applicable_full_seed_count", -1
                    )
                )
                relay_full_na_seeds = [
                    int(seed)
                    for seed in cell.get(
                        "relay_geometry_not_applicable_full_seeds", []
                    )
                ]
            except (TypeError, ValueError):
                relay_planned = relay_eligible = relay_full_na_count = -1
                relay_estimable = True
                relay_full_na_seeds = []
            if (
                relay_planned != 10
                or (relay_estimable and not 0 < relay_eligible <= relay_planned)
                or (not relay_estimable and relay_eligible != 0)
                or relay_full_na_count != len(relay_full_na_seeds)
                or len(set(relay_full_na_seeds)) != len(relay_full_na_seeds)
                or relay_eligible + relay_full_na_count != relay_planned
            ):
                errors.append(
                    f"terminal relay seed accounting is incomplete for {key}"
                )
            if not relay_estimable:
                relay_gates = cell.get("relay_gates", {}).values()
                if not cell.get("relay_not_estimable_reason"):
                    errors.append(
                        f"terminal relay not-estimable reason is absent for {key}"
                    )
                if any(
                    gate.get("pass") is not False
                    or gate.get("estimate") is not None
                    or gate.get("low") is not None
                    or gate.get("high") is not None
                    for gate in relay_gates
                ):
                    errors.append(
                        f"terminal relay has fabricated N/A numerics for {key}"
                    )
            artifact_hashes = cell.get("artifact_hashes", {})
            if set(artifact_hashes) != {
                "completion",
                "answer_audit",
                "layer_effects",
                "relay_audit",
                "claim_gates",
            } or any(len(str(value)) != 64 for value in artifact_hashes.values()):
                errors.append(f"answer/trace artifact provenance is incomplete for {key}")

        followup_protocol = embedded.get("followup_protocol", {})
        if followup_protocol.get("status") != (
            "FROZEN_BEFORE_V2_INTERVENTION_OUTCOMES"
        ):
            errors.append("follow-up V2 protocol is not frozen")
        followup = embedded.get("followup", {})
        if followup.get("status") != "PASS_COMPLETE":
            errors.append("follow-up V2 is not complete")
        index_support = followup.get("index_targeted_city_support", {})
        if set(index_support) != {"Qwen3-8B", "Gemma4-E4B"}:
            errors.append("Index targeted-city follow-up lost a model")
        for model in ("Qwen3-8B", "Gemma4-E4B"):
            value = index_support.get(model, {})
            analysis = value.get("analysis", {})
            position = value.get("position_audit", {})
            if (
                analysis.get("analysis_status")
                != "POSTHOC_DIAGNOSTIC_SPLIT_REUSE"
                or analysis.get("phase") != "confirmation"
                or int(analysis.get("seed_count", -1)) != 10
            ):
                errors.append(f"Index sustained-support analysis changed for {model}")
            if (
                position.get("status") != "PASS"
                or position.get("head_ablation_scope")
                != "registered_query_through_city_prefix"
                or int(position.get("seed_count", -1)) != 10
                or int(position.get("anchor_count", -1)) != 10
            ):
                errors.append(f"Index position audit changed for {model}")

        full_item = followup.get("full_item_greedy", {})
        if (
            full_item.get("status")
            != "POSTHOC_GREEDY_READOUT_EXTENSION_COMPLETE"
            or full_item.get("analysis_status")
            != "POSTHOC_CONFIRMATION_SPLIT_REUSE"
            or bool(full_item.get("frozen_layers_changed", True))
            or bool(full_item.get("frozen_k_changed", True))
            or bool(full_item.get("seed_selection_used_greedy_outcomes", True))
        ):
            errors.append("full-item greedy extension contract changed")
        greedy_cells = {
            f"{row.get('prompt_mode')}|{row.get('model_label')}"
            for row in full_item.get("cell_summaries", [])
        }
        if greedy_cells != set(EXPECTED_K):
            errors.append("full-item greedy extension lost a model×grammar cell")

        fresh = followup.get("fresh_bullet_gemma_carrier", {})
        replication = fresh.get("replication", {})
        fresh_analysis = fresh.get("analysis", {})
        cohort_lock = fresh.get("cohort_lock", {})
        if (
            replication.get("status")
            != "FRESH_CAUSAL_OUTCOME_REPLICATION_COMPLETE"
            or replication.get("head_ablation_scope") != "query_through_carrier"
            or int(replication.get("seed_count", -1)) != 10
            or int(replication.get("selected_k", -1)) != 2
            or bool(replication.get("seed_selection_used_intervention_outcomes", True))
            or not bool(replication.get("original_query_local_null_retained"))
        ):
            errors.append("fresh Bullet-Gemma carrier replication changed")
        if (
            fresh_analysis.get("analysis_status")
            != "POSTHOC_DIAGNOSTIC_SPLIT_REUSE"
            or fresh_analysis.get("head_ablation_scope") != "query_through_carrier"
            or int(fresh_analysis.get("seed_count", -1)) != 10
        ):
            errors.append("fresh Bullet-Gemma carrier analysis changed")
        if (
            cohort_lock.get("status")
            != "FROZEN_BEFORE_CAUSAL_INTERVENTION_OUTCOMES"
            or len(cohort_lock.get("true_source_seeds", [])) != 10
        ):
            errors.append("fresh Bullet-Gemma cohort lock changed")

        followup_v3_protocol = embedded.get("followup_v3_protocol", {})
        if followup_v3_protocol.get("status") != (
            "FROZEN_BEFORE_V3_AGGREGATE_MULTIHOP_REPARSE"
        ):
            errors.append("follow-up V3 protocol is not frozen")
        corrections = followup_v3_protocol.get("freeze_corrections", [])
        if len(corrections) != 2 or any(
            value.get("outcome_dependent_change") is not False
            for value in corrections
        ):
            errors.append("V3 clerical hash corrections are not fully disclosed")
        followup_v3 = embedded.get("followup_v3", {})
        multihop = followup_v3.get("full_item_multihop", {})
        if (
            followup_v3.get("status") != "PASS_COMPLETE"
            or multihop.get("status") != "POSTHOC_MULTIHOP_REPARSE_COMPLETE"
            or multihop.get("analysis_status")
            != "POSTHOC_AGGREGATE_REPARSE_AFTER_ONE_SCHEMA_SMOKE_ROW"
            or int(multihop.get("row_count", -1)) != 240
            or int(multihop.get("seed_effect_row_count", -1)) != 80
            or multihop.get("new_model_forward_used") is not False
            or multihop.get("all_cells_primary_depth4_strong_gate_pass") is not True
        ):
            errors.append("V3 multihop aggregate contract changed")
        manifold = embedded.get("representation_manifold", {})
        if (
            manifold.get("status")
            != "PASS_DISCOVERY_FIT_CONFIRMATION_PROJECTION"
            or manifold.get("heavy_coordinates_embedded_once_in_viewer") is not True
            or not str(manifold.get("payload_sha256", ""))
        ):
            errors.append("V3 representation manifold summary changed")
        aligned = embedded.get("native_aligned_representation", {})
        manifold_manifest = aligned.get("manifold_manifest", {})
        if (
            manifold_manifest.get("status")
            != "PASS_DISCOVERY_FIT_CONFIRMATION_PROJECTION"
            or manifold_manifest.get("confirmation_used_for_fit_or_selection")
            is not False
            or manifold_manifest.get("new_model_forward_used") is not False
            or len(aligned.get("running_candidates", [])) != 156
            or len(aligned.get("final_candidates", [])) != 156
        ):
            errors.append("V3 all-layer representation audit changed")

        parser_appendix = embedded.get("parser_appendix", {})
        if (
            parser_appendix.get("status")
            != "PASS_PARSER_CONTRACT_AND_COHORT_AUDIT"
            or parser_appendix.get("schema_version")
            != "realistic_niah_v6_parser_report_appendix_v1"
            or set(parser_appendix.get("registered_sites", []))
            != EXPECTED_PARSER_SITES
            or set(parser_appendix.get("formal_gate_components", []))
            != EXPECTED_PARSER_GATES
        ):
            errors.append("parser appendix contract or site registry changed")
        strict_grammar = parser_appendix.get("strict_grammar", {})
        if (
            strict_grammar.get("index_record_pattern")
            != r"^\s*(\d+)\.\s*(.+?)\s*:\s*(-?\d+)\s*$"
            or strict_grammar.get("bullet_record_pattern")
            != r"^\s*-\s*(.+?)\s*:\s*(-?\d+)\s*$"
            or set(strict_grammar.get("format_statuses", []))
            != {
                "ok",
                "mixed_markers",
                "wrong_marker",
                "no_records",
                "index_sequence_error",
            }
        ):
            errors.append("strict enumeration grammar changed")
        cohort_summaries = parser_appendix.get("cohort_summaries", [])
        expected_parser_panels = {
            f"{key}|{split}"
            for key in EXPECTED_K
            for split in ("discovery", "confirmation")
        }
        observed_parser_panels = {
            f"{row.get('prompt_mode')}|{row.get('model_label')}|{row.get('split')}"
            for row in cohort_summaries
        }
        if observed_parser_panels != expected_parser_panels or len(cohort_summaries) != 8:
            errors.append("parser cohort summary lost a model×grammar×split panel")
        for row in cohort_summaries:
            expected_count = 200 if row.get("split") == "discovery" else 100
            accuracy_key = f"{row.get('prompt_mode')}|{row.get('model_label')}"
            expected_panel = EXPECTED_RAW_STRICT.get(accuracy_key, {}).get(
                str(row.get("split")), (-1, -1)
            )
            expected_pass, expected_accuracy_total = expected_panel
            if (
                int(row.get("selected_cell_count", -1)) != expected_count
                or int(row.get("original_strict_pass_count", -1)) != expected_pass
                or expected_accuracy_total != expected_count
                or abs(
                    float(row.get("original_strict_accuracy", -1.0))
                    - expected_pass / expected_count
                )
                > 1e-12
                or int(row.get("replacement_count", -1))
                != int(row.get("original_strict_failure_count", -2))
                or int(row.get("final_fixed_quota_eligible_count", -1))
                != expected_count
                or int(row.get("failed_reserve_attempt_count", -1)) < 0
                or row.get("status") != "PASS_STRICT_FIXED_QUOTA"
            ):
                errors.append(
                    "parser fixed-quota summary changed for "
                    f"{row.get('prompt_mode')}|{row.get('model_label')}|{row.get('split')}"
                )
        behavioral_accuracy = parser_appendix.get("behavioral_accuracy", {})
        accuracy_cells = behavioral_accuracy.get("cell_summaries", [])
        accuracy_cell_by_key = {
            f"{row.get('prompt_mode')}|{row.get('model_label')}": row
            for row in accuracy_cells
        }
        if (
            behavioral_accuracy.get("metric_id")
            != "original_fixed_slot_strict_exact_ordered_enumeration_accuracy"
            or behavioral_accuracy.get("reserve_attempts_excluded") is not True
            or behavioral_accuracy.get("replacement_filtered_cohort_not_model_accuracy")
            is not True
            or set(accuracy_cell_by_key) != set(EXPECTED_RAW_STRICT)
            or len(accuracy_cells) != 4
        ):
            errors.append("raw strict behavioral-accuracy contract changed")
        for key, expected in EXPECTED_RAW_STRICT.items():
            row = accuracy_cell_by_key.get(key, {})
            discovery_pass, discovery_total = expected["discovery"]
            confirmation_pass, confirmation_total = expected["confirmation"]
            pooled_pass, pooled_total = expected["pooled"]
            if (
                int(row.get("discovery_pass_count", -1)) != discovery_pass
                or int(row.get("discovery_total_count", -1)) != discovery_total
                or abs(
                    float(row.get("discovery_accuracy", -1.0))
                    - discovery_pass / discovery_total
                )
                > 1e-12
                or int(row.get("confirmation_pass_count", -1))
                != confirmation_pass
                or int(row.get("confirmation_total_count", -1))
                != confirmation_total
                or abs(
                    float(row.get("confirmation_accuracy", -1.0))
                    - confirmation_pass / confirmation_total
                )
                > 1e-12
                or int(row.get("pooled_pass_count", -1)) != pooled_pass
                or int(row.get("pooled_total_count", -1)) != pooled_total
                or abs(
                    float(row.get("pooled_accuracy", -1.0))
                    - pooled_pass / pooled_total
                )
                > 1e-12
                or int(row.get("final_fixed_quota_eligible_count", -1))
                != pooled_total
                or int(row.get("final_fixed_quota_total_count", -1)) != pooled_total
            ):
                errors.append(f"raw strict behavioral accuracy changed for {key}")
        accuracy_modes = {
            str(row.get("prompt_mode")): row
            for row in behavioral_accuracy.get("mode_summaries", [])
        }
        for mode, (expected_pass, expected_total) in EXPECTED_MODE_RAW_STRICT.items():
            row = accuracy_modes.get(mode, {})
            if (
                int(row.get("pass_count", -1)) != expected_pass
                or int(row.get("total_count", -1)) != expected_total
                or abs(
                    float(row.get("accuracy", -1.0))
                    - expected_pass / expected_total
                )
                > 1e-12
            ):
                errors.append(f"mode-level raw strict accuracy changed for {mode}")
        overall_accuracy = behavioral_accuracy.get("overall", {})
        final_eligibility = behavioral_accuracy.get(
            "final_fixed_quota_eligibility", {}
        )
        if (
            int(overall_accuracy.get("pass_count", -1)) != 1076
            or int(overall_accuracy.get("total_count", -1)) != 1200
            or abs(float(overall_accuracy.get("accuracy", -1.0)) - 1076 / 1200)
            > 1e-12
            or int(final_eligibility.get("eligible_count", -1)) != 1200
            or int(final_eligibility.get("total_count", -1)) != 1200
            or float(final_eligibility.get("rate", -1.0)) != 1.0
        ):
            errors.append("overall raw accuracy or final cohort eligibility changed")
        parser_failures = parser_appendix.get("original_failure_ledger", [])
        failed_reserve_attempts = parser_appendix.get(
            "failed_reserve_attempt_ledger", []
        )
        replacement_policy_audit = parser_appendix.get(
            "replacement_policy_audit", {}
        )
        if (
            int(parser_appendix.get("original_strict_failure_count", -1)) != 124
            or len(parser_failures) != 124
            or sum(
                int(row.get("original_strict_failure_count", 0))
                for row in cohort_summaries
            )
            != 124
            or int(parser_appendix.get("failed_reserve_attempt_count", -1)) != 105
            or len(failed_reserve_attempts) != 105
            or sum(
                int(row.get("failed_reserve_attempt_count", 0))
                for row in cohort_summaries
            )
            != 105
            or int(parser_appendix.get("final_fixed_quota_unresolved_count", -1))
            != 0
            or any(
                "fresh_v6_strict_parser_failure"
                not in row.get("failure_reasons", [])
                for row in parser_failures
            )
        ):
            errors.append("parser original-failure/replacement ledger changed")
        if (
            any(
                row.get("intervention_outcomes_read") is not False
                or row.get("eligible") is True
                or row.get("selected") is True
                for row in failed_reserve_attempts
            )
            or int(
                replacement_policy_audit.get(
                    "ordinary_failed_reserve_attempt_count", -1
                )
            )
            != 105
            or int(
                replacement_policy_audit.get(
                    "coherent_failed_reserve_attempt_count", -1
                )
            )
            != 246
            or int(
                replacement_policy_audit.get("all_failed_reserve_attempt_count", -1)
            )
            != 351
            or int(
                replacement_policy_audit.get(
                    "coherent_replacement_trajectory_count", -1
                )
            )
            != 78
            or replacement_policy_audit.get(
                "negative_experimental_results_trigger_replacement"
            )
            is not False
            or replacement_policy_audit.get("silent_sample_exclusion") is not False
        ):
            errors.append("parser reserve/coherent replacement audit changed")
        parser_sources = parser_appendix.get("source_sha256", {})
        if len(parser_sources) != 7 or any(
            len(str(digest)) != 64 for digest in parser_sources.values()
        ):
            errors.append("parser source provenance is incomplete")
        multihop_parser = parser_appendix.get("multihop_endpoint", {})
        taxonomy = multihop_parser.get("failure_taxonomy", {})
        if (
            multihop_parser.get("source_field")
            != "generated_known_city_ordinals_any_surface"
            or multihop_parser.get("registered_depths") != [1, 2, 4]
            or int(multihop_parser.get("all_rows", -1)) != 240
            or int(multihop_parser.get("seed_effect_rows", -1)) != 80
            or multihop_parser.get("failed_and_truncated_rows_retained") is not True
            or len(multihop_parser.get("fixed_lowest_seed_examples", [])) != 24
            or int(taxonomy.get("patched_rows", -1)) != 80
            or int(taxonomy.get("nonconsecutive_ordinal_rows", -1)) != 11
        ):
            errors.append("multihop parser endpoint or failure taxonomy changed")

    external_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    if external_manifest.get("status") != "PASS":
        errors.append("external report manifest is not PASS")
    report_hash = sha256(report)
    if external_manifest.get("output_sha256") != report_hash:
        errors.append("external manifest output hash mismatch")
    if not external_manifest.get("frozen_nulls_retained", False):
        errors.append("external manifest does not retain frozen nulls")
    if not external_manifest.get("posthoc_diagnostics_labeled", False):
        errors.append("external manifest does not label diagnostics")
    for field in (
        "followup_v2_complete",
        "followup_v3_complete",
        "position_audit_labeled",
        "greedy_extension_labeled",
        "fresh_replication_labeled",
        "multihop_reparse_labeled",
        "representation_3d_embedded",
        "parser_appendix_embedded",
        "behavioral_accuracy_embedded",
        "parser_original_failure_ledger_embedded",
        "parser_source_hashes_embedded",
        "native_template_format_mirrored",
        "index_item_end_anchor_sensitivity_complete",
        "answer_trace_extension_complete",
        "narrative_experiment_frames_complete",
        "figure_axis_captions_complete",
        "per_experiment_conclusions_complete",
        "simple_examples_complete",
    ):
        if not external_manifest.get(field, False):
            errors.append(f"external manifest missing {field}")
    if external_manifest.get("format_revision") != (
        "native_mirrored_v6_index_anchor_and_answer_trace_exact"
    ):
        errors.append(
            "external manifest does not seal the v6 Index-anchor + answer/trace exact format"
        )
    if set(external_manifest.get("required_sections", [])) != REQUIRED_SECTIONS:
        errors.append("external manifest required-section registry is stale")
    native_css_source_hash = str(
        external_manifest.get("native_template_css_source_sha256", "")
    )
    if len(native_css_source_hash) != 64:
        errors.append("external manifest lacks the Native template source hash")
    if native_template_report is not None and native_css_source_hash != sha256(
        native_template_report
    ):
        errors.append("external manifest Native template hash mismatch")
    if external_manifest.get("confirmation_used_for_3d_fit_or_selection") is not False:
        errors.append("external manifest allows confirmation leakage into 3D fit")
    if external_manifest.get("new_model_forward_used_for_v3") is not False:
        errors.append("external manifest reports an unexpected V3 model forward")
    if external_manifest.get("new_model_forward_used_for_accuracy_update") is not False:
        errors.append("external manifest reports an unexpected accuracy model forward")
    extension_summary_hash = str(
        external_manifest.get("answer_trace_extension_summary_sha256", "")
    )
    if (
        len(extension_summary_hash) != 64
        or extension_summary_hash
        != str(embedded.get("answer_trace_extension_summary_sha256", ""))
    ):
        errors.append("answer/trace extension summary hash is absent or inconsistent")
    sensitivity_manifest_hashes = external_manifest.get(
        "index_item_end_anchor_sensitivity_source_sha256", {}
    )
    if (
        sensitivity_manifest_hashes
        != embedded.get("index_item_end_anchor_sensitivity", {}).get(
            "analysis_source_sha256", {}
        )
        or set(sensitivity_manifest_hashes) != {"Qwen3-8B", "Gemma4-E4B"}
        or any(len(str(value)) != 64 for value in sensitivity_manifest_hashes.values())
    ):
        errors.append("Index item-end sensitivity source hashes are absent or inconsistent")
    return {
        "schema_version": "realistic_niah_v6_enumeration_narrative_report_validation_v6",
        "status": "PASS" if not errors else "FAIL",
        "report": str(report.resolve()),
        "report_sha256": report_hash,
        "errors": errors,
        "external_assets": bool(parser.external),
        "section_count": len(REQUIRED_SECTIONS - set(missing_sections)),
        "table_count": parser.table_count,
        "figure_count": parser.figure_count,
        "figcaption_count": parser.figcaption_count,
        "figure_title_count": parser.figure_title_count,
        "experiment_frame_count": parser.experiment_frame_count,
        "figure_primer_count": parser.figure_primer_count,
        "section_conclusion_count": parser.section_conclusion_count,
        "subsection_conclusion_count": parser.subsection_conclusion_count,
        "canvas_count": parser.canvas_count,
        "canvas_aria_count": parser.canvas_aria_count,
        "svg_count": parser.svg_count,
        "svg_viewbox_count": parser.svg_viewbox_count,
        "svg_role_img_count": parser.svg_role_img_count,
        "svg_titled_count": parser.svg_titled_count,
        "visual_layout_contract_pass": not visual_errors,
        "model_mode_cell_count": len(set(parser.cell_keys)),
        "unique_id_count": len(set(parser.ids)),
        "native_template_css_inherited": (
            native_template_report is not None
            and style_block(report).startswith(style_block(native_template_report))
        ),
        "native_template_topology_matched": (
            native_template_report is not None
            and re.findall(
                r'<section\b[^>]*\bid="([^"]+)"',
                report.read_text(encoding="utf-8"),
                flags=re.IGNORECASE,
            )
            == re.findall(
                r'<section\b[^>]*\bid="([^"]+)"',
                native_template_report.read_text(encoding="utf-8"),
                flags=re.IGNORECASE,
            )
        ),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--native-template-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.report, args.manifest, args.native_template_report)
    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
