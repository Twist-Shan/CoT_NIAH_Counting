from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Mapping

from .spec import MODEL_LABELS, PROMPT_MODES
from .suite import EXPERIMENTS, NATIVE_ANALYSIS_PATH


REPORT_SCHEMA_VERSION = "realistic_niah_v6_enumeration_html_report_v3_native_aligned"


RESULT_SOURCES: dict[str, tuple[str, ...]] = {
    "behavior_and_parser_baseline": (),
    "layerwise_representation": (
        "representation/native_aligned/running_index_selected.csv",
        "representation/native_aligned/final_count_selected.csv",
        "representation/native_aligned/cell_manifest.json",
    ),
    "paired_causal_estimands": (
        "causal/targeted_retrieval/discovery_formal/analysis/selection.json",
    ),
    "trace_scope_layer_sweep": (
        "causal/report_tail/confirmation_formal/natural_selected/confirmation_analysis.json",
    ),
    "targeted_retrieval_bank": (
        "causal/targeted_retrieval/confirmation_formal/analysis.json",
    ),
    "seed_equal_sampling_contract": (),
    "targeted_query_to_carrier": (
        "causal/specialized/confirmation_analysis/targeted_counter_write/confirmation/claim_gates.json",
    ),
    "carrier_to_commit_restore": (
        "causal/specialized/confirmation_analysis/targeted_counter_write/confirmation/claim_gates.json",
    ),
    "progress_state_to_successor": (
        "count_stream/confirmation_formal/trace_patch_analysis/estimands.csv",
    ),
    "answer_token_source_ablation": (
        "causal/specialized/confirmation_analysis/token_ablation_answer/confirmation/seed_equal_summary.csv",
        "causal/specialized/confirmation_analysis/token_ablation_targeting/confirmation/seed_equal_summary.csv",
    ),
    "terminal_state_bridge": (
        "causal/specialized/confirmation_analysis/terminal_state_bridge/confirmation/claim_gates.json",
    ),
    "timing_stratified_ncc": (
        "causal/specialized/confirmation_analysis/stratified_ncc/claim_gates.json",
    ),
    "direct_count_output_margin": (
        "causal/specialized/confirmation_analysis/direct_count_logit_margin/claim_gates.json",
    ),
    "count_geometry_ncc": (
        "causal/specialized/confirmation_analysis/count_geometry_ncc/claim_gates.json",
    ),
    "layer_timing_diagnostic": (
        "causal/specialized/confirmation_analysis/stratified_ncc/layer_metrics.csv",
    ),
    "visible_progress_positive_control": (
        "causal/report_tail/confirmation_formal/natural_selected/confirmation_analysis.json",
    ),
    "commit_to_query_patch": (
        "causal/report_tail/confirmation_formal/native_loop/analysis/claim_gates.json",
    ),
    "single_seed_walkthrough": (
        "causal/report_tail/confirmation_formal/single_seed_walkthrough/analysis/walkthrough_complete.json",
    ),
    "format_specific_source_scrub_restore": (
        "causal/report_tail/confirmation_formal/restoration/analysis/estimands.csv",
    ),
    "scrub_coverage_and_cross_mode_audit": ("suite_audit.json",),
}


INTERESTING_KEY = re.compile(
    r"(?:status|pass|gate|claim|effect|estimate|mean|median|accuracy|rate|margin|"
    r"recovery|interval|ci_|_ci|p_value|selected_k|selected_layer|bank_size|"
    r"seed_count|cell_count|pair_count|directional|negative_result|excluded_rows|"
    r"confirmation_used|reselected|aliasing|source_layer|readout_layer)",
    re.IGNORECASE,
)
IGNORED_KEY = re.compile(
    r"(?:path|sha256|argv|runtime|code|completed_utc|schema_version|request_id|"
    r"stimulus_id|trial_root|registry|freeze$)",
    re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return value


def _logical(path: Path, run_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(run_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        if value == 0:
            return "0"
        if abs(value) < 1e-4 or abs(value) >= 1e4:
            return f"{value:.4e}"
        return f"{value:.5f}".rstrip("0").rstrip(".")
    text = str(value)
    return text if len(text) <= 150 else text[:147] + "..."


def interesting_scalars(value: Any, *, limit: int = 28) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []

    def visit(node: Any, prefix: str, depth: int) -> None:
        if len(rows) >= limit or depth > 6:
            return
        if isinstance(node, Mapping):
            for key, child in node.items():
                name = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(child, (Mapping, list, tuple)):
                    visit(child, name, depth + 1)
                elif (
                    not IGNORED_KEY.search(str(key))
                    and (INTERESTING_KEY.search(str(key)) or str(key) == "model_label")
                ):
                    rows.append((name, _format_scalar(child)))
                    if len(rows) >= limit:
                        return
        elif isinstance(node, (list, tuple)) and len(node) <= 20:
            for index, child in enumerate(node):
                visit(child, f"{prefix}[{index}]", depth + 1)

    visit(value, "", 0)
    return rows


def _float_or_none(value: str | None) -> float | None:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def summarize_csv(path: Path, *, model_label: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    columns = list(rows[0]) if rows else []
    if path.name in {"classification_confirmation.csv", "regression_confirmation.csv"}:
        source_layer = 19 if model_label == "Qwen3-8B" else 16
        focused = [
            row
            for row in rows
            if str(row.get("cohort")) == "one_to_one"
            and _float_or_none(row.get("layer")) == source_layer
            and str(row.get("site_kind")) in {"item_end", "answer_query", "answer_query_v3"}
            and str(row.get("classifier", "logistic")) == "logistic"
        ]
        preview = focused or rows[:12]
    else:
        preview = rows[:12]

    numeric_ranges: dict[str, dict[str, float]] = {}
    for column in columns:
        if not INTERESTING_KEY.search(column) or IGNORED_KEY.search(column):
            continue
        values = [parsed for row in rows if (parsed := _float_or_none(row.get(column))) is not None]
        if values:
            numeric_ranges[column] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }
    preferred_columns = [
        column
        for column in columns
        if INTERESTING_KEY.search(column)
        or column
        in {
            "model_label",
            "prompt_mode",
            "grammar",
            "endpoint",
            "token_site",
            "cohort",
            "site_kind",
            "layer",
            "classifier",
            "probe",
            "experiment_id",
            "condition",
            "contrast",
            "source_group",
            "donor_direction",
        }
    ]
    if not preferred_columns:
        preferred_columns = columns[:10]
    preferred_columns = preferred_columns[:14]
    return {
        "row_count": len(rows),
        "columns": preferred_columns,
        "preview": [
            {column: row.get(column, "") for column in preferred_columns}
            for row in preview[:12]
        ],
        "numeric_ranges": numeric_ranges,
    }


def _cell_lookup(audit: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    result = {}
    for cell in audit.get("cells", []):
        result[(str(cell["prompt_mode"]), str(cell["model_label"]))] = cell
    expected = {(mode, model) for mode in PROMPT_MODES for model in MODEL_LABELS}
    if set(result) != expected:
        raise ValueError("Completion audit does not contain the four V6 cells")
    return result


def _cell_baseline(cell: Mapping[str, Any]) -> dict[str, Any]:
    discovery = cell["cell_replacements"]["discovery"]
    confirmation = cell["cell_replacements"]["confirmation"]
    return {
        "status": "PASS_FIXED_QUOTA_AFTER_FORMAT_ONLY_REPLACEMENT",
        "discovery_original_cells": int(discovery["cell_count"]),
        "discovery_original_failures": int(discovery["replacement_count"]),
        "discovery_original_strict_eligible_rate": 1.0
        - int(discovery["replacement_count"]) / int(discovery["cell_count"]),
        "confirmation_original_cells": int(confirmation["cell_count"]),
        "confirmation_original_failures": int(confirmation["replacement_count"]),
        "confirmation_original_strict_eligible_rate": 1.0
        - int(confirmation["replacement_count"]) / int(confirmation["cell_count"]),
        "resolved_formal_quota_rate": 1.0,
        "negative_experimental_results_trigger_replacement": False,
    }


def _sampling_contract(cell: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "PASS_FIXED_SLOT_TRUE_SOURCE_IDENTITY",
        "discovery_fixed_slots_per_count": 20,
        "confirmation_fixed_slots_per_count": 10,
        "panel_membership_identity": "analysis_slot_seed",
        "statistical_identity": "true_source_seed",
        "seed_aliasing": bool(cell.get("seed_aliasing", False)),
        "discovery_freeze": cell["confirmation_freeze"]["content_freeze_sha256"],
    }


def build_report_document(run_root: str | Path, completion_audit: str | Path) -> tuple[str, dict[str, Any]]:
    run_root = Path(run_root).resolve()
    audit_path = Path(completion_audit).resolve()
    audit = _read_json(audit_path)
    if audit.get("status") != "PASS_FULL_V6_ENUMERATION_SUITE":
        raise ValueError("HTML report requires a PASS full-suite completion audit")
    cells = _cell_lookup(audit)
    if set(RESULT_SOURCES) != {experiment.experiment_id for experiment in EXPERIMENTS}:
        raise ValueError("Report result-source registry changed relative to the 20 frames")

    snapshots: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, str] = {_logical(audit_path, run_root): _sha256(audit_path)}
    for experiment in EXPERIMENTS:
        by_cell: dict[str, Any] = {}
        for mode in PROMPT_MODES:
            for model in MODEL_LABELS:
                key = f"{mode}|{model}"
                cell = cells[(mode, model)]
                if experiment.experiment_id == "behavior_and_parser_baseline":
                    by_cell[key] = {"derived": _cell_baseline(cell), "sources": []}
                    continue
                if experiment.experiment_id == "seed_equal_sampling_contract":
                    by_cell[key] = {"derived": _sampling_contract(cell), "sources": []}
                    continue
                sources = []
                model_root = run_root / mode / model
                for relative in RESULT_SOURCES[experiment.experiment_id]:
                    path = model_root / relative
                    if not path.is_file():
                        raise FileNotFoundError(
                            f"Missing report result source for {experiment.experiment_id}: {path}"
                        )
                    logical = _logical(path, run_root)
                    source_hashes[logical] = _sha256(path)
                    if path.suffix.lower() == ".json":
                        value = _read_json(path)
                        sources.append(
                            {
                                "path": logical,
                                "sha256": source_hashes[logical],
                                "kind": "json",
                                "interesting_scalars": interesting_scalars(value),
                            }
                        )
                    elif path.suffix.lower() == ".csv":
                        sources.append(
                            {
                                "path": logical,
                                "sha256": source_hashes[logical],
                                "kind": "csv",
                                "summary": summarize_csv(path, model_label=model),
                            }
                        )
                    else:
                        raise ValueError(f"Unsupported report result source: {path}")
                by_cell[key] = {"sources": sources}
        snapshots[experiment.experiment_id] = by_cell

    summary = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "PASS",
        "run_root": str(run_root),
        "completion_audit": str(audit_path),
        "completion_audit_sha256": _sha256(audit_path),
        "completion_audit_content_hash": audit["audit_sha256"],
        "report_frame_count": len(EXPERIMENTS),
        "model_mode_cell_count": len(cells),
        "ordinary_failure_count": audit["replacement_policy"][
            "ordinary_cell_failure_count"
        ],
        "failed_reserve_attempt_count": audit["replacement_policy"][
            "failed_reserve_attempt_count"
        ],
        "coherent_replacement_trajectory_count": audit["replacement_policy"][
            "coherent_replacement_trajectory_count"
        ],
        "protocol_amendments": dict(audit.get("protocol_amendments", {})),
        "infrastructure_recovery_count": len(audit["infrastructure_recoveries"]),
        "native_aligned_representation": dict(
            audit["native_aligned_representation"]
        ),
        "targeted_retrieval_selections": {
            f"{mode}|{model}": dict(
                cells[(mode, model)]["targeted_retrieval_selection"]
            )
            for mode in PROMPT_MODES
            for model in MODEL_LABELS
        },
        "specialized_bank_plan_adapters": {
            f"{mode}|{model}": dict(
                cells[(mode, model)]["specialized_bank_plan_adapter"]
            )
            for mode in PROMPT_MODES
            for model in MODEL_LABELS
        },
        "source_sha256": source_hashes,
        "snapshots": snapshots,
    }
    document = _render_html(audit, summary)
    return document, summary


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _mode_label(mode: str) -> str:
    return "Index enumeration" if mode == "enumeration_index" else "Bullet enumeration"


def _render_key_values(rows: Iterable[tuple[str, str]]) -> str:
    rows = list(rows)
    if not rows:
        return '<p class="muted">No scalar claim field; use the hashed evidence path.</p>'
    return '<dl class="kv">' + "".join(
        f"<dt>{_esc(key)}</dt><dd>{_esc(value)}</dd>" for key, value in rows
    ) + "</dl>"


def _render_csv(summary: Mapping[str, Any]) -> str:
    ranges = summary.get("numeric_ranges", {})
    range_rows = [
        (
            key,
            f"min={_format_scalar(value['min'])}; mean={_format_scalar(value['mean'])}; max={_format_scalar(value['max'])}",
        )
        for key, value in ranges.items()
    ][:12]
    preview = list(summary.get("preview", []))
    columns = list(summary.get("columns", []))
    table = ""
    if preview and columns:
        table = (
            '<div class="table-wrap"><table><thead><tr>'
            + "".join(f"<th>{_esc(column)}</th>" for column in columns)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>"
                + "".join(f"<td>{_esc(row.get(column, ''))}</td>" for column in columns)
                + "</tr>"
                for row in preview
            )
            + "</tbody></table></div>"
        )
    return (
        f'<p class="muted">Rows: {_esc(summary.get("row_count", 0))}</p>'
        + _render_key_values(range_rows)
        + table
    )


def _render_cell_snapshot(snapshot: Mapping[str, Any]) -> str:
    if "derived" in snapshot:
        return _render_key_values(interesting_scalars(snapshot["derived"], limit=40))
    blocks = []
    for source in snapshot.get("sources", []):
        blocks.append(
            f'<details><summary>{_esc(source["path"])}</summary>'
            f'<p class="hash">SHA-256 {_esc(source["sha256"])}</p>'
            + (
                _render_key_values(source["interesting_scalars"])
                if source["kind"] == "json"
                else _render_csv(source["summary"])
            )
            + "</details>"
        )
    return "".join(blocks)


def _failure_rows(audit: Mapping[str, Any]) -> str:
    rows = audit["replacement_policy"]["ordinary_failures"]
    if not rows:
        return '<p class="muted">No ordinary original cell failed.</p>'
    return (
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Mode</th><th>Model</th><th>Split</th><th>N</th><th>Slot seed</th>"
        "<th>Failure reasons</th><th>Replacement seed</th><th>Rank</th>"
        "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            f'<td>{_esc(_mode_label(row["prompt_mode"]))}</td>'
            f'<td>{_esc(row["model_label"])}</td><td>{_esc(row["split"])}</td>'
            f'<td>{_esc(row["gold_count"])}</td><td>{_esc(row["analysis_slot_seed"])}</td>'
            f'<td>{_esc("; ".join(row["failure_reasons"]))}</td>'
            f'<td>{_esc(row["replacement_seed"])}</td>'
            f'<td>{_esc(row["replacement_candidate_rank"])}</td></tr>'
            for row in rows
        )
        + "</tbody></table></div>"
    )


def _coherent_rows(audit: Mapping[str, Any]) -> str:
    rows = audit["replacement_policy"]["coherent_replacements"]
    if not rows:
        return '<p class="muted">No coherent trajectory required replacement.</p>'

    def failed_cells(row: Mapping[str, Any]) -> str:
        return " | ".join(
            f"N={value['gold_count']}: {'; '.join(value.get('failure_reasons', []))}"
            for value in row["original_failed_cells"]
        )

    return (
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Mode</th><th>Model</th><th>Split</th><th>Panel</th><th>Slot→source</th>"
        "<th>Required counts</th><th>Failed counts</th><th>Successful counts displaced for coherence</th>"
        "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            f'<td>{_esc(_mode_label(row["prompt_mode"]))}</td>'
            f'<td>{_esc(row["model_label"])}</td><td>{_esc(row["split"])}</td>'
            f'<td>{_esc(row["panel_kind"])}</td>'
            f'<td>{_esc(row["analysis_slot_seed"])} → {_esc(row["replacement_seed"])}</td>'
            f'<td>{_esc(", ".join(map(str, row["required_counts"])))}</td>'
            f'<td>{_esc(failed_cells(row))}</td>'
            f'<td>{_esc(", ".join(map(str, row["successful_original_cells_replaced_for_seed_coherence"])))}</td>'
            "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def _failed_reserve_rows(audit: Mapping[str, Any]) -> str:
    rows = audit["replacement_policy"]["failed_reserve_attempts"]
    if not rows:
        return '<p class="muted">No generated reserve sample failed runtime or strict parsing.</p>'

    def candidate(row: Mapping[str, Any]) -> Any:
        return row.get("candidate_seed", row.get("seed", ""))

    def counts(row: Mapping[str, Any]) -> str:
        if "gold_count" in row:
            return str(row["gold_count"])
        return ", ".join(map(str, row.get("required_counts", [])))

    def eligible_counts(row: Mapping[str, Any]) -> str:
        if row.get("panel_kind") == "ordinary_cell":
            return "PASS" if bool(row.get("eligible")) else "none"
        return ", ".join(map(str, row.get("eligible_counts", []))) or "none"

    return (
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Mode</th><th>Model</th><th>Split</th><th>Panel</th><th>Slot</th>"
        "<th>Reserve seed</th><th>Registered counts</th><th>Eligible counts</th>"
        "<th>Runtime failure</th><th>Failure reasons</th>"
        "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            f'<td>{_esc(_mode_label(str(row["prompt_mode"])))}</td>'
            f'<td>{_esc(row["model_label"])}</td><td>{_esc(row["split"])}</td>'
            f'<td>{_esc(row.get("panel_kind", "ordinary_cell"))}</td>'
            f'<td>{_esc(row.get("analysis_slot_seed", "—"))}</td>'
            f'<td>{_esc(candidate(row))}</td><td>{_esc(counts(row))}</td>'
            f'<td>{_esc(eligible_counts(row))}</td>'
            f'<td>{_esc(bool(row.get("runtime_failure", False)))}</td>'
            f'<td>{_esc("; ".join(map(str, row.get("failure_reasons", []))))}</td>'
            "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def _non_failure_candidate_rows(audit: Mapping[str, Any]) -> str:
    rows = audit["replacement_policy"].get(
        "coherent_non_failure_candidate_rejections", []
    )
    if not rows:
        return '<p class="muted">No reserve candidate was skipped solely for source-identity coherence.</p>'
    return (
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Mode</th><th>Model</th><th>Split</th><th>Panel</th><th>Slot</th>"
        "<th>Candidate seed</th><th>Reason</th><th>Classification</th>"
        "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            f'<td>{_esc(_mode_label(str(row["prompt_mode"])))}</td>'
            f'<td>{_esc(row["model_label"])}</td><td>{_esc(row["split"])}</td>'
            f'<td>{_esc(row["panel_kind"])}</td>'
            f'<td>{_esc(row["analysis_slot_seed"])}</td>'
            f'<td>{_esc(row["candidate_seed"])}</td>'
            f'<td>{_esc("; ".join(map(str, row.get("failure_reasons", []))))}</td>'
            '<td>identity skip; not a failed sample</td></tr>'
            for row in rows
        )
        + "</tbody></table></div>"
    )


def _protocol_amendment_rows(audit: Mapping[str, Any]) -> str:
    rows = audit.get("protocol_amendments", {}).get(
        "pool_exhaustion_amendments", []
    )
    if not rows:
        return '<p class="muted">No post-exhaustion reserve-pool amendment was required.</p>'
    return (
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Mode</th><th>Model</th><th>Panel</th><th>Replacement policy</th>"
        "<th>Policy SHA-256</th><th>Reserve stimuli SHA-256</th>"
        "<th>Pool manifest SHA-256</th><th>Outcome firewall</th>"
        "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            f'<td>{_esc(_mode_label(str(row["prompt_mode"])))}</td>'
            f'<td>{_esc(row["model_label"])}</td>'
            f'<td>{_esc(row["coherent_panel"])}</td>'
            f'<td>{_esc(Path(str(row["replacement_policy"])).name)}</td>'
            f'<td class="hash">{_esc(row["replacement_policy_sha256"])}</td>'
            f'<td class="hash">{_esc(row["replacement_stimuli_sha256"])}</td>'
            f'<td class="hash">{_esc(row["pool_manifest_sha256"])}</td>'
            f'<td>intervention outcomes read = {_esc(row["intervention_outcomes_read"])}</td>'
            "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def _render_html(audit: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    cells = _cell_lookup(audit)
    ordinary_by_cell = Counter(
        (row["prompt_mode"], row["model_label"], row["split"])
        for row in audit["replacement_policy"]["ordinary_failures"]
    )
    nav = "".join(
        f'<a href="#frame-{experiment.report_frame}">{experiment.report_frame:02d}</a>'
        for experiment in EXPERIMENTS
    )
    frame_html = []
    for experiment in EXPERIMENTS:
        native_path = NATIVE_ANALYSIS_PATH[experiment.experiment_id]
        cards = []
        snapshots = summary["snapshots"][experiment.experiment_id]
        for mode in PROMPT_MODES:
            for model in MODEL_LABELS:
                key = f"{mode}|{model}"
                cards.append(
                    '<article class="cell">'
                    f'<h4>{_esc(_mode_label(mode))} · {_esc(model)}</h4>'
                    '<p class="pass">Execution evidence: PASS</p>'
                    + _render_cell_snapshot(snapshots[key])
                    + "</article>"
                )
        frame_html.append(
            f'<section class="experiment-frame" id="frame-{experiment.report_frame}" data-frame="{experiment.report_frame}">'
            f'<p class="eyebrow">Frame {experiment.report_frame:02d} · {_esc(experiment.section)}</p>'
            f'<h2>{_esc(experiment.experiment_id)}</h2>'
            '<div class="contract">'
            f'<p><strong>Native section.</strong> {_esc(native_path["native_report_section"])}</p>'
            f'<p><strong>Evidence tier.</strong> {_esc(native_path["evidence_tier"])}</p>'
            f'<p><strong>Analysis population.</strong> {_esc(native_path["analysis_population"])}</p>'
            f'<p><strong>Role.</strong> {_esc(experiment.claim_role)}</p>'
            f'<p><strong>Primary estimand.</strong> {_esc(experiment.primary_estimand)}</p>'
            f'<p><strong>Controls.</strong> {_esc("; ".join(experiment.controls))}</p>'
            f'<p><strong>Kernel.</strong> <code>{_esc(experiment.kernel_entrypoint)}</code></p>'
            f'<p><strong>Held-out confirmation required.</strong> {_esc(experiment.confirmation_required)}</p>'
            "</div>"
            '<div class="cell-grid">' + "".join(cards) + "</div></section>"
        )

    summary_rows = []
    for mode in PROMPT_MODES:
        for model in MODEL_LABELS:
            cell = cells[(mode, model)]
            broad = sum(
                value["replacement_trajectory_count"]
                for key, value in cell["coherent_replacements"].items()
                if key.startswith("broad_")
            )
            native = sum(
                value["replacement_trajectory_count"]
                for key, value in cell["coherent_replacements"].items()
                if key.startswith("native_loop_")
            )
            targeted = cell["targeted_retrieval_selection"]
            bank_adapter = cell["specialized_bank_plan_adapter"]
            geometry = (
                f'{bank_adapter["replacement_count"]} global controls replaced; '
                f'capture starts L{bank_adapter["capture_start_layer"]}'
                if bank_adapter["random_control_heads_changed"]
                else "frozen source plan unchanged"
            )
            summary_rows.append(
                "<tr>"
                f'<td>{_esc(_mode_label(mode))}</td><td>{_esc(model)}</td>'
                f'<td>{_esc(targeted["selected_k"])}</td>'
                f'<td>{_esc(targeted["selected_random_condition"])}</td>'
                f'<td>{_esc(geometry)}</td>'
                f'<td>{ordinary_by_cell[(mode, model, "discovery")]}</td>'
                f'<td>{ordinary_by_cell[(mode, model, "confirmation")]}</td>'
                f'<td>{broad}</td><td>{native}</td>'
                f'<td class="pass">PASS</td></tr>'
            )

    recoveries = audit.get("infrastructure_recoveries", [])
    recovery_html = (
        '<div class="table-wrap"><table><thead><tr><th>Status</th><th>Model/mode</th><th>Reason</th><th>Recoverable</th></tr></thead><tbody>'
        + "".join(
            "<tr>"
            f'<td>{_esc(row.get("status"))}</td>'
            f'<td>{_esc(row.get("details", {}).get("prompt_mode", ""))} {_esc(row.get("details", {}).get("model_label", ""))}</td>'
            f'<td>{_esc(row.get("details", {}).get("reason", ""))}</td>'
            f'<td>{_esc(row.get("details", {}).get("deletion_performed") is False or row.get("details", {}).get("model_outputs_recomputed") is False)}</td>'
            "</tr>"
            for row in recoveries
        )
        + "</tbody></table></div>"
        if recoveries
        else '<p class="muted">No infrastructure recovery event.</p>'
    )

    # Script elements use raw-text parsing, so HTML entities would corrupt the
    # embedded JSON.  JSON unicode escapes keep it parseable while preventing a
    # literal ``</script>`` or HTML-significant ampersand from being emitted.
    embedded = (
        json.dumps(summary, sort_keys=True)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V6 Index/Bullet · Native-thinking Analysis-Path Replication</title>
<style>
:root{{--ink:#172033;--muted:#667085;--line:#d7dde7;--bg:#f5f7fa;--teal:#0f766e;--blue:#315f7d;--amber:#9a4b00}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);background:var(--bg);font:15px/1.62 Inter,Segoe UI,Arial,sans-serif}}
header,main,footer{{max-width:1380px;margin:auto;padding:28px 42px}}header{{padding-top:58px;background:#fff;border-bottom:1px solid var(--line)}}
h1{{max-width:980px;margin:7px 0 14px;font-size:42px;line-height:1.12}}h2{{margin:4px 0 16px;font-size:27px}}h3{{font-size:20px}}h4{{margin:0 0 8px}}
.eyebrow{{margin:0;color:var(--teal);font:800 11px/1.3 ui-monospace,Consolas,monospace;letter-spacing:.12em;text-transform:uppercase}}
.lead{{max-width:980px;color:#475467;font-size:17px}}nav{{position:sticky;top:0;z-index:2;display:flex;gap:7px;overflow:auto;padding:10px 42px;background:#172033}}nav a{{color:#d8e2ee;text-decoration:none;font:700 12px ui-monospace,Consolas,monospace}}
.audit-banner{{margin:22px 0;padding:16px 20px;border-left:5px solid var(--teal);background:#edf8f5}}.pass{{color:#075e58;font-weight:800}}.muted,.hash{{color:var(--muted);font-size:12px;overflow-wrap:anywhere}}
.metric-grid,.cell-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}.metric{{padding:18px;background:#fff;border-top:4px solid var(--teal)}}.metric strong{{display:block;font-size:28px}}
.experiment-frame{{margin:28px 0;padding:24px;background:#fff;border:1px solid var(--line);scroll-margin-top:52px}}.contract{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 22px;padding:14px 18px;background:#f7f9fc;border-left:4px solid var(--blue)}}.contract p{{margin:5px 0}}
.cell{{min-width:0;padding:17px;border:1px solid var(--line);background:#fcfdff}}details{{margin:9px 0;border-top:1px solid var(--line);padding-top:8px}}summary{{cursor:pointer;color:var(--blue);font-weight:700;overflow-wrap:anywhere}}
.kv{{display:grid;grid-template-columns:minmax(180px,42%) 1fr;margin:8px 0;font-size:12px}}.kv dt,.kv dd{{margin:0;padding:5px 7px;border-bottom:1px solid #edf0f4;overflow-wrap:anywhere}}.kv dt{{color:#475467}}.kv dd{{font-family:ui-monospace,Consolas,monospace}}
.table-wrap{{max-width:100%;overflow:auto}}table{{width:100%;border-collapse:collapse;background:#fff;font-size:12px}}th,td{{padding:8px 9px;border:1px solid var(--line);text-align:left;vertical-align:top}}th{{background:#eef2f7;white-space:nowrap}}
.appendix{{margin:32px 0;padding:24px;background:#fff;border:1px solid var(--line)}}code{{font-family:ui-monospace,Consolas,monospace;font-size:.92em}}footer{{color:var(--muted);font-size:12px}}
@media(max-width:850px){{header,main,footer{{padding-left:18px;padding-right:18px}}nav{{padding-left:18px}}h1{{font-size:32px}}.metric-grid,.cell-grid,.contract{{grid-template-columns:1fr}}}}
</style></head><body>
<header><p class="eyebrow">Realistic CoT NIAH · V6 · Native-aligned</p><h1>Index enumeration 与 Bullet enumeration：沿 Native-thinking 分析路径的语法对照复现</h1>
<p class="lead">结论先行：本页只在四个 model×mode 单元全部通过 fail-closed 验收后生成。表示主分析固定为原始样本上的 <code>item_end</code> 四单元精确共同支持与 <code>answer_query_v3</code> 完整 300 轨迹；其余机制实验沿 Native-thinking 的行为→状态→检索→更新→终端读出→证据综合层级报告。resolved replacement 只维持单元内固定配额，不被当作跨语法的同源样本。</p>
<div class="audit-banner"><strong>Suite status:</strong> {_esc(audit['status'])}<br><span class="hash">Completion audit content hash: {_esc(audit['audit_sha256'])}</span></div></header>
<nav><a href="#summary">Summary</a>{nav}<a href="#failures">Failures</a><a href="#provenance">Provenance</a></nav>
<main><section id="summary"><p class="eyebrow">Execution summary</p><h2>完成度与补 seed 概览</h2>
<div class="metric-grid"><div class="metric"><strong>20 × 4</strong><span>20 frames across four model×mode cells</span></div>
<div class="metric"><strong>2 endpoints</strong><span>item_end common support + answer_query_v3 full 300</span></div>
<div class="metric"><strong>{_esc(summary['ordinary_failure_count'])}</strong><span>original cell failures, all replaced and listed</span></div>
<div class="metric"><strong>{_esc(summary['coherent_replacement_trajectory_count'])}</strong><span>whole-source broad/native trajectories replaced</span></div>
<div class="metric"><strong>{_esc(summary['failed_reserve_attempt_count'])}</strong><span>failed reserve attempts retained in the ledger</span></div></div>
<div class="table-wrap"><table><thead><tr><th>Mode</th><th>Model</th><th>Discovery-selected K</th><th>Random control</th><th>Specialized control geometry</th><th>Discovery cell failures</th><th>Confirmation cell failures</th><th>Broad trajectory replacements</th><th>Native-loop trajectory replacements</th><th>Status</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table></div>
<p class="muted">Targeted K is selected independently in each model×mode cell by the discovery seed-equal primary estimand (exact tie → smaller K), then frozen through confirmation. The intervention matches Native-thinking: index starts at the explicit ordinal marker’s `post_marker` anchor, bullet starts at the previous item’s committed `p0_item_end` anchor, and selected/control head ablation remains active for every cached decode step (`decode_head_ablation_steps=-1`), rather than only at the anchor token. Qwen grid: 32/64/80/96/112/128; Gemma grid: 1/2/4/6/8. Qwen K128 uses the merged-grid source artifact’s global-random controls; all other registered doses use layer-matched random controls. If a K128 global control occupies the final layer, only that random-control head is deterministically replaced from the capture-reachable complement; the selected treatment bank and K remain unchanged, and no result field is consulted. Ordinary cell replacement preserves successful original cells. Broad-K uses one real source seed over its registered five counts. Native-loop uses one real source seed over counts 2–10. `analysis_slot_seed` fixes panel membership; `true_source_seed` is the statistical identity.</p></section>
{''.join(frame_html)}
<section class="appendix" id="failures"><p class="eyebrow">Failure and replacement ledger</p><h2>失败样本、reserve seed 与同源轨迹替换</h2>
<p><strong>边界：</strong>下表中的 failure 是 generation/runtime 或 fresh strict-parser failure；科学上的零效应、反向效应和未过 claim gate 不会触发换 seed。</p>
<h3>普通 cell-level failures</h3>{_failure_rows(audit)}
<h3>失败的 reserve attempts（继续补 seed）</h3>{_failed_reserve_rows(audit)}
<h3>Coherent broad/native replacements</h3>{_coherent_rows(audit)}
<h3>Coherent identity skips（不属于失败样本）</h3>{_non_failure_candidate_rows(audit)}
<h3>Post-exhaustion reserve-pool amendments</h3>{_protocol_amendment_rows(audit)}
<h3>Infrastructure recovery（不计为样本失败）</h3>{recovery_html}</section>
<section class="appendix" id="provenance"><p class="eyebrow">Reproducibility</p><h2>审计与范围</h2>
<dl class="kv"><dt>Audit file SHA-256</dt><dd>{_esc(summary['completion_audit_sha256'])}</dd><dt>Audit content hash</dt><dd>{_esc(summary['completion_audit_content_hash'])}</dd><dt>Stimulus SHA-256</dt><dd>da4dd86142eb8a07f9a7e53497efd3375184c8e68367d4db994370fcb331f090</dd><dt>Base reserve stimulus SHA-256</dt><dd>4f60897ff680d4f977008d0ac7e256d9e4bce50e7d84378ad82511270b673662</dd><dt>Base native-loop policy SHA-256</dt><dd>e9f7e2cd88a6eba7f97342bc9fdfdf14cc9dacc6fbdd14b985105e67a8c571fa</dd></dl>
<p class="muted">本报告展示注册的 outcome fields 与其哈希，不根据结果重新选择 K、layer、scope、cohort 或 seed。CSV 中的 min/mean/max 是描述性摘要；正式判定以各分析器写出的 claim/status 字段为准。</p></section>
<script type="application/json" id="v6-report-summary">{embedded}</script></main>
<footer>Generated from a PASS V6 completion audit. No external assets.</footer></body></html>"""


class _ReportParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.frames: list[str] = []
        self.cell_count = 0
        self.scripts: list[dict[str, str | None]] = []
        self._active_summary_script = False
        self.summary_script_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.append(str(values["id"]))
        classes = set(str(values.get("class", "")).split())
        if "experiment-frame" in classes:
            self.frames.append(str(values.get("data-frame", "")))
        if "cell" in classes:
            self.cell_count += 1
        if tag == "script":
            self.scripts.append(values)
            self._active_summary_script = values.get("id") == "v6-report-summary"

    def handle_data(self, data: str) -> None:
        if self._active_summary_script:
            self.summary_script_chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._active_summary_script = False


def validate_report_html(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    parser = _ReportParser()
    parser.feed(path.read_text(encoding="utf-8"))
    duplicate_ids = sorted(key for key, count in Counter(parser.ids).items() if count > 1)
    errors = []
    if duplicate_ids:
        errors.append(f"duplicate ids: {duplicate_ids}")
    if parser.frames != [str(value) for value in range(1, 21)]:
        errors.append(f"frame sequence changed: {parser.frames}")
    if parser.cell_count != 80:
        errors.append(f"expected 80 model-mode frame cells, found {parser.cell_count}")
    if any(values.get("src") for values in parser.scripts):
        errors.append("external script source found")
    summary_scripts = [
        values for values in parser.scripts if values.get("id") == "v6-report-summary"
    ]
    if len(summary_scripts) != 1:
        errors.append(f"expected one embedded V6 summary, found {len(summary_scripts)}")
    else:
        try:
            embedded_summary = json.loads("".join(parser.summary_script_chunks))
        except json.JSONDecodeError as error:
            errors.append(f"embedded V6 summary is not JSON: {error}")
        else:
            if not isinstance(embedded_summary, dict):
                errors.append("embedded V6 summary is not one JSON object")
    if "http://" in path.read_text(encoding="utf-8") or "https://" in path.read_text(
        encoding="utf-8"
    ):
        errors.append("external URL found")
    return {
        "schema_version": "realistic_niah_v6_html_report_validation_v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "report": str(path.resolve()),
        "report_sha256": _sha256(path),
        "unique_id_count": len(set(parser.ids)),
        "report_frame_count": len(parser.frames),
        "model_mode_frame_cell_count": parser.cell_count,
        "external_assets": False if not errors else None,
    }
