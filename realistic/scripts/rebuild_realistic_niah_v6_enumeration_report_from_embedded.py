#!/usr/bin/env python3
"""Rebuild the V6 Enumeration report from its sealed embedded data.

This path is intentionally report-only: it reuses the scientific payload already
embedded in the validated HTML and refreshes only the parser/cohort audit derived
from the sealed suite-completion audit.  It performs no model forward pass.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

try:
    from build_realistic_niah_v6_enumeration_narrative_report import (
        REQUIRED_SECTIONS,
        atomic_text,
        parser_audit_summary,
        read_style_block,
        render_report,
        sha256,
    )
    from validate_realistic_niah_v6_enumeration_narrative_report import ReportParser
except ModuleNotFoundError:  # Package import under pytest / python -m.
    from scripts.build_realistic_niah_v6_enumeration_narrative_report import (
        REQUIRED_SECTIONS,
        atomic_text,
        parser_audit_summary,
        read_style_block,
        render_report,
        sha256,
    )
    from scripts.validate_realistic_niah_v6_enumeration_narrative_report import (
        ReportParser,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--completion-audit", required=True, type=Path)
    parser.add_argument("--native-template-report", required=True, type=Path)
    parser.add_argument("--full-suite-summary", type=Path)
    parser.add_argument("--full-suite-report", type=Path)
    parser.add_argument("--behavior-manifest-dir", type=Path)
    parser.add_argument("--native-behavior-manifest-root", type=Path)
    parser.add_argument("--index-item-end-sensitivity-qwen-analysis", type=Path)
    parser.add_argument("--index-item-end-sensitivity-gemma-analysis", type=Path)
    parser.add_argument("--answer-trace-extension-summary", type=Path)
    parser.add_argument("--representation-manifold", type=Path)
    parser.add_argument("--representation-manifold-manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser.parse_args()


SENSITIVITY_MODELS = ("Qwen3-8B", "Gemma4-E4B")
SENSITIVITY_CELLS = {
    "p2bank_at_p2",
    "p2bank_at_p0",
    "p0bank_at_p2",
    "p0bank_at_p0",
}
SENSITIVITY_CONTRASTS = {
    "overall_item_end_minus_primary",
    "site_effect_for_p2_bank",
    "site_effect_for_p0_bank",
    "bank_effect_at_p0",
    "bank_effect_at_p2",
}


def _validate_bootstrap_summary(value: Any, *, label: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{label} is not an object")
    ci95 = value.get("ci95")
    if (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or value.get("n_analysis_slot_seeds") != 20
    ):
        raise ValueError(f"{label} lost its frozen 20-seed bootstrap contract")
    for field in ("estimate",):
        if not isinstance(value.get(field), (int, float)):
            raise ValueError(f"{label} has no numeric {field}")
    if not all(isinstance(item, (int, float)) for item in ci95):
        raise ValueError(f"{label} has a non-numeric CI")


def index_item_end_sensitivity_payload(
    qwen_path: Path, gemma_path: Path
) -> dict[str, Any]:
    """Load the two frozen discovery-only 2x2 anchor-sensitivity analyses.

    The payload is deliberately kept separate from confirmation evidence.  It may
    diagnose whether an Index lesion was started too early, but cannot replace the
    primary p2 result, change K, or authorize a new confirmation cohort.
    """

    paths = {"Qwen3-8B": qwen_path, "Gemma4-E4B": gemma_path}
    analyses: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    contract_hash: str | None = None
    for model in SENSITIVITY_MODELS:
        path = paths[model]
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"Index item-end sensitivity is not an object: {path}")
        if (
            value.get("schema_version")
            != "realistic_niah_v6_index_item_end_anchor_sensitivity_analysis_v1"
            or value.get("status") != "PASS"
            or value.get("scientific_scope")
            != "post_hoc_motivated_prospectively_frozen_sensitivity"
            or value.get("model_label") != model
            or value.get("prompt_mode") != "enumeration_index"
            or value.get("decision_is_exploratory") is not True
            or value.get("may_replace_primary_result") is not False
            or value.get("may_reselect_k") is not False
            or value.get("confirmation_authorized") is not False
        ):
            raise ValueError(f"Index item-end sensitivity contract changed for {model}")
        cells = value.get("cells")
        contrasts = value.get("contrasts")
        if not isinstance(cells, dict) or set(cells) != SENSITIVITY_CELLS:
            raise ValueError(f"Index item-end sensitivity cells changed for {model}")
        if not isinstance(contrasts, dict) or set(contrasts) != SENSITIVITY_CONTRASTS:
            raise ValueError(f"Index item-end sensitivity contrasts changed for {model}")
        for cell_name, cell in cells.items():
            if not isinstance(cell, dict) or cell.get("cell") != cell_name:
                raise ValueError(f"Malformed sensitivity cell {model}/{cell_name}")
            _validate_bootstrap_summary(
                cell.get("selected_minus_random_failure"),
                label=f"{model}/{cell_name}",
            )
        for contrast_name, contrast in contrasts.items():
            _validate_bootstrap_summary(
                contrast, label=f"{model}/{contrast_name}"
            )
        observed_contract = str(value.get("contract_sha256", ""))
        if not observed_contract:
            raise ValueError(f"Sensitivity contract hash missing for {model}")
        if contract_hash is None:
            contract_hash = observed_contract
        elif observed_contract != contract_hash:
            raise ValueError("Qwen and Gemma sensitivity analyses use different contracts")
        if model == "Gemma4-E4B" and value.get(
            "generation_container_audit", {}
        ).get("status") != (
            "PASS_AMENDED_APPENDABLE_CONTAINER_WITH_PRE_FREEZE_ROW_IDENTITY"
        ):
            raise ValueError("Gemma sensitivity lost its appendable-container row audit")
        analyses[model] = value
        source_hashes[model] = sha256(path)

    return {
        "schema_version": "realistic_niah_v6_index_item_end_anchor_sensitivity_report_payload_v1",
        "status": "PASS_COMPLETE",
        "scientific_scope": "post_hoc_motivated_prospectively_frozen_discovery_only",
        "contract_sha256": contract_hash,
        "analyses": analyses,
        "analysis_source_sha256": source_hashes,
        "primary_confirmation_replaced": False,
        "k_reselected": False,
    }


def embedded_report_data(report: Path) -> dict[str, Any]:
    parser = ReportParser()
    parser.feed(report.read_text(encoding="utf-8"))
    if not parser.manifest_text:
        raise ValueError(f"Missing embedded report manifest: {report}")
    value = json.loads(parser.manifest_text)
    if not isinstance(value, dict):
        raise TypeError("Embedded report payload is not a JSON object")
    return value


def full_suite_frame_html(report: Path) -> str:
    """Extract the sealed 20 report frames as collapsible, non-section HTML.

    The raw execution report is intentionally preserved rather than paraphrased:
    it carries the four-cell values, source paths, and SHA-256 provenance.  Outer
    ``section`` elements and heading levels are changed only so the Native report's
    twelve top-level section topology remains exact in the narrative document.
    """

    text = report.read_text(encoding="utf-8")
    pattern = re.compile(
        r'<section class="experiment-frame" id="frame-(\d+)"[^>]*>'
        r"(.*?)</section>",
        flags=re.DOTALL,
    )
    frames = pattern.findall(text)
    if [int(number) for number, _ in frames] != list(range(1, 21)):
        raise ValueError(f"Expected sealed frames 1..20 in {report}")
    rendered: list[str] = []
    for number, body in frames:
        title_match = re.search(r"<h2>(.*?)</h2>", body, flags=re.DOTALL)
        if title_match is None:
            raise ValueError(f"Frame {number} has no title")
        title = re.sub(r"<[^>]+>", "", title_match.group(1)).strip()
        body = body.replace("<h4>", "<h5>").replace("</h4>", "</h5>")
        body = body.replace("<h2>", '<h4 class="suite-frame-title">').replace(
            "</h2>", "</h4>"
        )
        rendered.append(
            '<details class="suite-frame-disclosure">'
            f'<summary>Frame {int(number):02d} · {title}</summary>'
            f'<div class="suite-frame-raw" id="suite-raw-frame-{number}">{body}</div>'
            "</details>"
        )
    return "".join(rendered)


def _plain_html(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value))).strip()


def _frame_articles(report_text: str, frame_number: int) -> list[tuple[str, str]]:
    frame_match = re.search(
        rf'<section class="experiment-frame" id="frame-{frame_number}"[^>]*>'
        r"(.*?)</section>",
        report_text,
        flags=re.DOTALL,
    )
    if frame_match is None:
        raise ValueError(f"Missing full-suite frame {frame_number}")
    articles = re.findall(
        r'<article class="cell"><h4>(.*?)</h4>(.*?)</article>',
        frame_match.group(1),
        flags=re.DOTALL,
    )
    if len(articles) != 4:
        raise ValueError(
            f"Expected four grammar/model articles in frame {frame_number}, got {len(articles)}"
        )
    return [(_plain_html(label), body) for label, body in articles]


def _article_definition_list(article_html: str) -> dict[str, str]:
    return {
        _plain_html(key): _plain_html(value)
        for key, value in re.findall(
            r"<dt>(.*?)</dt><dd>(.*?)</dd>", article_html, flags=re.DOTALL
        )
    }


def _article_tables(article_html: str) -> list[list[dict[str, str]]]:
    parsed: list[list[dict[str, str]]] = []
    for table_html in re.findall(r"<table>(.*?)</table>", article_html, flags=re.DOTALL):
        rows = re.findall(r"<tr>(.*?)</tr>", table_html, flags=re.DOTALL)
        if not rows:
            continue
        headers = [
            _plain_html(value)
            for value in re.findall(r"<th>(.*?)</th>", rows[0], flags=re.DOTALL)
        ]
        if not headers:
            continue
        values: list[dict[str, str]] = []
        for row_html in rows[1:]:
            cells = [
                _plain_html(value)
                for value in re.findall(r"<td>(.*?)</td>", row_html, flags=re.DOTALL)
            ]
            if len(cells) == len(headers):
                values.append(dict(zip(headers, cells, strict=True)))
        parsed.append(values)
    return parsed


def full_suite_claims(report: Path) -> dict[str, Any]:
    """Extract compact main-text claims from sealed frames 10 and 13.

    Complete raw tables remain embedded in Appendix A.9.  The extraction is
    deliberately narrow and fail-closed so the narrative cannot silently drift
    away from the sealed execution report.
    """

    text = report.read_text(encoding="utf-8")
    answer_source: dict[str, Any] = {}
    for label, article in _frame_articles(text, 10):
        result_rows: list[dict[str, str]] | None = None
        for rows in _article_tables(article):
            if rows and {
                "condition",
                "mean_delta_exact_count",
                "mean_delta_gold_first_answer_token_log_probability",
            }.issubset(rows[0]):
                result_rows = rows
                break
        if result_rows is None:
            raise ValueError(f"Frame 10 lost answer-source outcome table for {label}")
        by_condition = {row["condition"]: row for row in result_rows}
        required = ("prompt_records_blank", "trace_all_blank")
        if any(condition not in by_condition for condition in required):
            raise ValueError(f"Frame 10 lost registered conditions for {label}")
        answer_source[label] = {
            condition: {
                "mean_delta_exact_count": float(
                    by_condition[condition]["mean_delta_exact_count"]
                ),
                "mean_delta_gold_first_answer_token_log_probability": float(
                    by_condition[condition][
                        "mean_delta_gold_first_answer_token_log_probability"
                    ]
                ),
            }
            for condition in required
        }

    direct_margin: dict[str, Any] = {}
    key_prefix = "endpoint_results.final_answer_sequence_margin.confirmation."
    for label, article in _frame_articles(text, 13):
        values = _article_definition_list(article)
        required_keys = (
            "clean_accuracy",
            "clean_mean_margin",
            "selected_margin_loss.mean_effect",
            "selected_margin_loss.ci_low",
            "selected_margin_loss.ci_high",
            "selected_vs_random_specificity.mean_effect",
            "selected_vs_random_specificity.ci_low",
            "selected_vs_random_specificity.ci_high",
        )
        missing = [key for key in required_keys if key_prefix + key not in values]
        if missing:
            raise ValueError(f"Frame 13 lost {missing} for {label}")
        direct_margin[label] = {
            key: float(values[key_prefix + key]) for key in required_keys
        }
        direct_margin[label]["effect_status"] = values[
            "endpoint_results.final_answer_sequence_margin.effect_status"
        ]

    if len(answer_source) != 4 or len(direct_margin) != 4:
        raise ValueError("Full-suite compact claim extraction did not preserve four cells")
    if any(
        value["prompt_records_blank"]["mean_delta_exact_count"] != 0.0
        or value["trace_all_blank"]["mean_delta_exact_count"] != -1.0
        for value in answer_source.values()
    ):
        raise ValueError("Frame 10 no longer supports the registered four-cell source contrast")
    return {
        "status": "PASS_SEALED_FRAME_10_13_EXTRACTION",
        "source_report_sha256": sha256(report),
        "answer_source": answer_source,
        "direct_count_output_margin": direct_margin,
    }


def head_mask_scope_audit(
    manifest_dir: Path,
    data: dict[str, Any],
    native_manifest_root: Path | None = None,
) -> dict[str, Any]:
    expected = {
        "enumeration_index|Qwen3-8B": ("enumeration_index__Qwen3-8B__k128.json", 128),
        "enumeration_index|Gemma4-E4B": ("enumeration_index__Gemma4-E4B__k8.json", 8),
        "enumeration_bullet|Qwen3-8B": ("enumeration_bullet__Qwen3-8B__k96.json", 96),
        "enumeration_bullet|Gemma4-E4B": ("enumeration_bullet__Gemma4-E4B__k2.json", 2),
    }
    behavior: dict[str, Any] = {}
    for cell_key, (filename, selected_k) in expected.items():
        path = manifest_dir / filename
        value = json.loads(path.read_text(encoding="utf-8"))
        observed = {
            "selected_k": selected_k,
            "branch_policy": value.get("branch_policy"),
            "decode_head_ablation_steps": value.get("decode_head_ablation_steps"),
            "max_new_tokens": value.get("max_new_tokens"),
            "completed_shards": value.get("completed_shards"),
            "manifest_sha256": sha256(path),
        }
        if observed["branch_policy"] != (
            "teacher_force_through_registered_anchor_then_persistent_decode_head_ablation"
        ):
            raise ValueError(f"Behavior mask policy changed for {cell_key}: {observed}")
        if observed["decode_head_ablation_steps"] != -1 or observed["completed_shards"] != 50:
            raise ValueError(f"Behavior mask duration/completion changed for {cell_key}: {observed}")
        behavior[cell_key] = observed

    original_carrier = {
        cell_key: {
            "head_ablation_scopes": cell["carrier"]["original_head_ablation_scopes"],
            "head_ablation_position_counts": cell["carrier"][
                "original_head_ablation_position_counts"
            ],
        }
        for cell_key, cell in data["cells"].items()
    }
    for cell_key, value in original_carrier.items():
        if value["head_ablation_scopes"] != ["query_local"] or value[
            "head_ablation_position_counts"
        ] != [1]:
            raise ValueError(f"Original carrier scope changed for {cell_key}: {value}")

    followup = data["followup"]
    sustained = {
        model: {
            "head_ablation_scope": followup["index_targeted_city_support"][model][
                "trials_manifest"
            ]["head_ablation_scope"],
            "completed_shards": followup["index_targeted_city_support"][model][
                "trials_manifest"
            ]["completed_shards"],
        }
        for model in ("Qwen3-8B", "Gemma4-E4B")
    }
    fresh = followup["fresh_bullet_gemma_carrier"]["replication"]
    native_behavior: dict[str, Any] = {}
    if native_manifest_root is not None:
        for model in ("Qwen3-8B", "Gemma4-E4B"):
            path = (
                native_manifest_root
                / model
                / "targeted_retrieval"
                / "confirmation"
                / "manifest.json"
            )
            value = json.loads(path.read_text(encoding="utf-8"))
            observed = {
                "branch_policy": value.get("branch_policy"),
                "decode_head_ablation_steps": value.get(
                    "decode_head_ablation_steps"
                ),
                "max_new_tokens": value.get("max_new_tokens"),
                "completed_shards": value.get("completed_shards"),
                "manifest_sha256": sha256(path),
            }
            if observed["branch_policy"] != (
                "teacher_force_through_latest_registered_anchor_then_persistent_decode_head_ablation"
            ):
                raise ValueError(
                    f"Native behavior mask policy changed for {model}: {observed}"
                )
            if observed["decode_head_ablation_steps"] != -1 or observed[
                "completed_shards"
            ] != 50:
                raise ValueError(
                    f"Native behavior mask duration/completion changed for {model}: {observed}"
                )
            native_behavior[model] = observed

    return {
        "status": "PASS_DISTINCT_TEMPORAL_SCOPES_AUDITED",
        "native_behavior_persistent_decode": native_behavior,
        "behavior_persistent_decode": behavior,
        "original_carrier_query_local": original_carrier,
        "index_sustained_city_prefix": sustained,
        "bullet_gemma_fresh_carrier": {
            "head_ablation_scope": fresh["head_ablation_scope"],
            "strong_interval_gate_pass": fresh["strong_interval_gate_pass"],
            "manifest_sha256": fresh["artifacts"]["trials_manifest"]["sha256"],
        },
    }


def main() -> int:
    args = parse_args()
    data = embedded_report_data(args.source_report)
    old_parser = data.get("parser_appendix", {})
    if not isinstance(old_parser, dict):
        raise TypeError("Embedded parser appendix is not an object")
    completion_audit = json.loads(args.completion_audit.read_text(encoding="utf-8"))
    parser_appendix = parser_audit_summary(completion_audit)
    for field in ("source_sha256", "multihop_endpoint"):
        if field not in old_parser:
            raise ValueError(f"Embedded parser appendix lost {field}")
        parser_appendix[field] = old_parser[field]
    data["parser_appendix"] = parser_appendix
    data["native_template_css"] = read_style_block(args.native_template_report)
    manifold_paths = (
        args.representation_manifold,
        args.representation_manifold_manifest,
    )
    if any(path is not None for path in manifold_paths):
        if any(path is None for path in manifold_paths):
            raise ValueError(
                "Both representation manifold data and manifest are required"
            )
        manifold = json.loads(
            args.representation_manifold.read_text(encoding="utf-8")
        )
        manifold_manifest = json.loads(
            args.representation_manifold_manifest.read_text(encoding="utf-8")
        )
        old_manifold = data.get("representation_manifold", {})
        payload_sha = sha256(args.representation_manifold)
        manifest_sha = sha256(args.representation_manifold_manifest)
        registered_manifest_hashes = {
            str(digest)
            for path, digest in data.get("source_sha256", {}).items()
            if str(path).endswith(
                "representation_manifold_v3/representation_manifold_manifest.json"
            )
        }
        if (
            manifold.get("status")
            != "PASS_DISCOVERY_FIT_CONFIRMATION_PROJECTION"
            or manifold.get("qualification")
            != "DESCRIPTIVE_DISCOVERY_FIT_CONFIRMATION_PROJECTION"
            or manifold.get("fit_split") != "discovery"
            or manifold.get("display_split") != "confirmation"
            or set(manifold.get("running", {}))
            != {
                "enumeration_index|Qwen3-8B",
                "enumeration_index|Gemma4-E4B",
                "enumeration_bullet|Qwen3-8B",
                "enumeration_bullet|Gemma4-E4B",
            }
            or set(manifold.get("final", {})) != set(manifold.get("running", {}))
            or manifold_manifest.get("status")
            != "PASS_DISCOVERY_FIT_CONFIRMATION_PROJECTION"
            or manifold_manifest.get("confirmation_used_for_fit_or_selection")
            is not False
            or manifold_manifest.get("new_model_forward_used") is not False
            or str(manifold_manifest.get("output", {}).get("sha256")) != payload_sha
            or str(old_manifold.get("payload_sha256", "")) != payload_sha
            or registered_manifest_hashes != {manifest_sha}
        ):
            raise ValueError("Representation manifold payload or lineage changed")
        data["representation_manifold"] = manifold
    elif not {
        "running",
        "final",
    } <= set(data.get("representation_manifold", {})):
        raise ValueError(
            "Report-only rebuild requires the sealed heavy representation manifold "
            "payload; the embedded report intentionally contains only its summary"
        )
    if args.full_suite_summary is not None:
        full_suite_summary = json.loads(
            args.full_suite_summary.read_text(encoding="utf-8")
        )
        if full_suite_summary.get("status") != "PASS" or full_suite_summary.get(
            "report_frame_count"
        ) != 20:
            raise ValueError("Full-suite summary is not the sealed PASS 20-frame report")
        data["full_suite_summary"] = full_suite_summary
        data["full_suite_summary_sha256"] = sha256(args.full_suite_summary)
    if args.full_suite_report is not None:
        data["full_suite_frames_html"] = full_suite_frame_html(args.full_suite_report)
        data["full_suite_report_sha256"] = sha256(args.full_suite_report)
        data["full_suite_claims"] = full_suite_claims(args.full_suite_report)
    if args.behavior_manifest_dir is not None:
        data["head_mask_scope_audit"] = head_mask_scope_audit(
            args.behavior_manifest_dir,
            data,
            args.native_behavior_manifest_root,
        )
    sensitivity_paths = (
        args.index_item_end_sensitivity_qwen_analysis,
        args.index_item_end_sensitivity_gemma_analysis,
    )
    if any(path is not None for path in sensitivity_paths):
        if any(path is None for path in sensitivity_paths):
            raise ValueError(
                "Both Qwen and Gemma Index item-end sensitivity analyses are required"
            )
        data["index_item_end_anchor_sensitivity"] = (
            index_item_end_sensitivity_payload(
                args.index_item_end_sensitivity_qwen_analysis,
                args.index_item_end_sensitivity_gemma_analysis,
            )
        )
    if args.answer_trace_extension_summary is not None:
        extension = json.loads(
            args.answer_trace_extension_summary.read_text(encoding="utf-8")
        )
        extension_cells = extension.get("cells", [])
        expected_cells = {
            (mode, model)
            for mode in ("enumeration_index", "enumeration_bullet")
            for model in ("Qwen3-8B", "Gemma4-E4B")
        }
        observed_cells = {
            (str(cell.get("prompt_mode")), str(cell.get("model_label")))
            for cell in extension_cells
        }
        if extension.get("status") != "PASS_COMPLETE" or observed_cells != expected_cells:
            raise ValueError("Answer/trace extension is not the complete four-cell report")
        if any(
            len(cell.get("answer_layer_effects", [])) != 8
            or set(cell.get("relay_gates", {}))
            != {
                "terminal_state_patch_effect",
                "post_terminal_suffix_specific_mediation",
                "post_terminal_suffix_residual_equivalence",
                "self_reset_is_nondamaging",
                "answer_query_only_mediation",
            }
            for cell in extension_cells
        ):
            raise ValueError("Answer/trace extension lost a frozen layer grid or relay gate")
        for cell in extension_cells:
            planned = int(cell.get("relay_planned_seed_count", -1))
            eligible = int(cell.get("relay_eligible_seed_count", -1))
            estimable = bool(cell.get("relay_estimable", eligible > 0))
            full_na_count = int(
                cell.get("relay_geometry_not_applicable_full_seed_count", -1)
            )
            full_na_seeds = [
                int(seed)
                for seed in cell.get(
                    "relay_geometry_not_applicable_full_seeds", []
                )
            ]
            if (
                planned != 10
                or (estimable and not 0 < eligible <= planned)
                or (not estimable and eligible != 0)
                or full_na_count != len(full_na_seeds)
                or len(set(full_na_seeds)) != len(full_na_seeds)
                or eligible + full_na_count != planned
            ):
                raise ValueError(
                    "Answer/trace extension lost preregistered-versus-eligible "
                    "relay seed accounting"
                )
            if not estimable:
                if not cell.get("relay_not_estimable_reason"):
                    raise ValueError(
                        "Non-estimable terminal relay lost its registered reason"
                    )
                for gate in cell["relay_gates"].values():
                    if (
                        gate.get("pass") is not False
                        or gate.get("estimate") is not None
                        or gate.get("low") is not None
                        or gate.get("high") is not None
                    ):
                        raise ValueError(
                            "Non-estimable terminal relay contains a fabricated "
                            "numeric result"
                        )
        data["answer_trace_extension"] = extension
        data["answer_trace_extension_summary_sha256"] = sha256(
            args.answer_trace_extension_summary
        )

    document = render_report(data)
    for section in REQUIRED_SECTIONS:
        if f'id="{section}"' not in document:
            raise RuntimeError(f"Narrative report lost section {section}")
    atomic_text(args.output, document)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "PASS",
            "format_revision": "native_mirrored_v6_index_anchor_and_answer_trace_exact",
            "output": str(args.output.resolve()),
            "output_sha256": sha256(args.output),
            "required_sections": list(REQUIRED_SECTIONS),
            "behavioral_accuracy_embedded": True,
            "behavioral_accuracy_metric_id": parser_appendix["behavioral_accuracy"][
                "metric_id"
            ],
            "behavioral_accuracy_source_sha256": sha256(args.completion_audit),
            "native_template_css_source_sha256": sha256(args.native_template_report),
            "native_template_format_mirrored": True,
            "native_to_enumeration_alignment_direction": True,
            "full_suite_report_frames_embedded": bool(
                data.get("full_suite_frames_html")
            ),
            "full_suite_report_frame_count": data.get("full_suite_summary", {}).get(
                "report_frame_count"
            ),
            "head_mask_temporal_scope_audited": bool(
                data.get("head_mask_scope_audit")
            ),
            "answer_trace_extension_complete": bool(
                data.get("answer_trace_extension")
            ),
            "answer_trace_extension_summary_sha256": data.get(
                "answer_trace_extension_summary_sha256"
            ),
            "index_item_end_anchor_sensitivity_complete": bool(
                data.get("index_item_end_anchor_sensitivity")
            ),
            "index_item_end_anchor_sensitivity_source_sha256": data.get(
                "index_item_end_anchor_sensitivity", {}
            ).get("analysis_source_sha256"),
            "narrative_experiment_frames_complete": True,
            "figure_axis_captions_complete": True,
            "per_experiment_conclusions_complete": True,
            "simple_examples_complete": True,
            "new_model_forward_used_for_accuracy_update": False,
            "scientific_results_rewritten": False,
        }
    )
    atomic_text(args.manifest, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output.resolve()),
                "output_sha256": manifest["output_sha256"],
                "behavioral_accuracy": parser_appendix["behavioral_accuracy"],
                "new_model_forward_used": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
