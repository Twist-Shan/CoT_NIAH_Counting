#!/usr/bin/env python3
"""Build the audited narrative V6 Enumeration mechanism report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MODES = ("enumeration_index", "enumeration_bullet")
MODE_LABEL = {
    "enumeration_index": "Index",
    "enumeration_bullet": "Bullet",
}
MODE_GRAMMAR = {
    "enumeration_index": "显式 ordinal → city",
    "enumeration_bullet": "bullet item → city",
}
MODELS = ("Qwen3-8B", "Gemma4-E4B")
CELL_ORDER = tuple((mode, model) for mode in MODES for model in MODELS)
DIAGNOSTIC_LABEL = "POSTHOC_DIAGNOSTIC_SPLIT_REUSE"
FOLLOWUP_GREEDY_LABEL = "POSTHOC_CONFIRMATION_SPLIT_REUSE"
FOLLOWUP_PROTOCOL_STATUS = "FROZEN_BEFORE_V2_INTERVENTION_OUTCOMES"
FOLLOWUP_V3_PROTOCOL_STATUS = "FROZEN_BEFORE_V3_AGGREGATE_MULTIHOP_REPARSE"
FRESH_CARRIER_STATUS = "FRESH_CAUSAL_OUTCOME_REPLICATION_COMPLETE"
REQUIRED_SECTIONS = (
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

PARSER_REGISTERED_SITES = (
    ("pre_marker", "marker 前一字符形成的 prefix endpoint；仅在 marker 左侧存在字符时生成"),
    ("marker_end", "显式 index / invariant bullet marker 的字符跨度终点"),
    ("pre_city", "目标 city 首字符之前的 prefix endpoint"),
    ("city_end", "目标 city 最后字符之后的 prefix endpoint"),
    ("city_unit_end", "hybrid parser 识别的完整 city semantic unit 终点"),
    ("item_end", "完整 semantic item span 的终点；V6 primary running-progress site"),
    ("post_boundary", "item_end 后紧邻的 CRLF/LF/CR 边界；若无换行则等于 item_end"),
    ("list_cut", "selected list 的起点到 parser cut boundary；用于整段审计"),
    ("answer_query", "最后一个行首 Total:，终点位于冒号之后"),
    ("answer_query_v3", "最后一个 Total: 标签及其空白，终点恰在最终整数首字符之前"),
)

PARSER_FORMAL_GATES = (
    ("registered_success", "final Total 等于 gold N、整段响应格式严格合规、且生成未因长度截断"),
    ("enumeration_format_compliant", "只使用注册 marker；Index 必须为连续 1..M，不能混用 marker"),
    ("strict_listed_total_matches_length", "final Total 等于严格格式下实际列出的 item 数"),
    ("exact_ordered_gold_pairs", "去除 city 两侧空白后，city 字符串与整数 score 按 passage 顺序逐项精确相等"),
    ("marker_kind_compliant", "Index 单元必须解析为 indexed；Bullet 单元必须解析为 bullet"),
    ("parser_forward_one_to_one", "semantic span parser 必须一对一、forward，不允许重复、遗漏或逆序"),
    ("item_count_matches_gold", "semantic parser item_count 必须等于冻结 gold pair 数"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return value


def read_style_block(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"<style\b[^>]*>(.*?)</style>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"No style block found in native report: {path}")
    return match.group(1)


def read_csv_one(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one selected row: {path}")
    return dict(rows[0])


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_jsonl_directory(path: Path) -> list[dict[str, Any]]:
    files = sorted((path / "shards").glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No diagnostic shards under {path}")
    return [
        json.loads(line)
        for file in files
        for line in file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite report value: {value}")
    return result


def estimand(
    value: Mapping[str, Any],
    name: str,
    *,
    outcome: str | None = None,
) -> dict[str, Any]:
    rows = value.get("all_estimands", value.get("estimands", []))
    found = [
        dict(row)
        for row in rows
        if str(row.get("estimand")) == name
        and (outcome is None or str(row.get("outcome")) == outcome)
    ]
    if len(found) != 1:
        raise ValueError(f"Expected one estimand {name}/{outcome}, found {len(found)}")
    return found[0]


def logical(path: Path, run_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(run_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def parser_audit_summary(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce the sealed suite audit to a complete parser/replacement ledger."""

    expected_cells = {f"{mode}|{model}" for mode, model in CELL_ORDER}
    expected_split_cells = {"discovery": 200, "confirmation": 100}
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    failed_reserve_attempts: list[dict[str, Any]] = []
    observed_cells: set[str] = set()
    for cell in audit.get("cells", []):
        mode = str(cell.get("prompt_mode", ""))
        model = str(cell.get("model_label", ""))
        key = f"{mode}|{model}"
        if key not in expected_cells or key in observed_cells:
            raise ValueError(f"Unexpected or duplicate parser audit cell: {key}")
        observed_cells.add(key)
        replacements = cell.get("cell_replacements", {})
        for split, expected_count in expected_split_cells.items():
            value = replacements.get(split, {})
            original_failures = list(value.get("failures", []))
            replacement_count = int(value.get("replacement_count", -1))
            failed_reserve_attempt_count = int(
                value.get("failed_reserve_attempt_count", -1)
            )
            split_failed_attempts = list(value.get("failed_reserve_attempts", []))
            if (
                value.get("status") != "PASS_STRICT_FIXED_QUOTA"
                or int(value.get("cell_count", -1)) != expected_count
                or replacement_count != len(original_failures)
                or failed_reserve_attempt_count != len(split_failed_attempts)
            ):
                raise ValueError(f"Parser replacement audit changed for {key}/{split}")
            reason_counts: dict[str, int] = {}
            for failure in original_failures:
                reasons = [str(reason) for reason in failure.get("failure_reasons", [])]
                if "fresh_v6_strict_parser_failure" not in reasons:
                    raise ValueError(
                        f"A replacement is not attributed to the strict parser: {key}/{split}"
                    )
                for reason in reasons:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                failures.append(
                    {
                        "prompt_mode": mode,
                        "model_label": model,
                        "split": split,
                        "analysis_slot_seed": int(failure["analysis_slot_seed"]),
                        "original_seed": int(failure["original_seed"]),
                        "gold_count": int(failure["gold_count"]),
                        "failure_reasons": reasons,
                        "replacement_seed": int(failure["replacement_seed"]),
                        "replacement_candidate_rank": int(
                            failure["replacement_candidate_rank"]
                        ),
                    }
                )
            for attempt in split_failed_attempts:
                attempt_reasons = [
                    str(reason) for reason in attempt.get("failure_reasons", [])
                ]
                if (
                    bool(attempt.get("eligible", True))
                    or bool(attempt.get("selected", True))
                    or bool(attempt.get("intervention_outcomes_read", True))
                ):
                    raise ValueError(
                        f"Failed reserve attempt contract changed: {key}/{split}"
                    )
                failed_reserve_attempts.append(
                    {
                        "prompt_mode": mode,
                        "model_label": model,
                        "split": split,
                        "seed": int(attempt["seed"]),
                        "gold_count": int(attempt["gold_count"]),
                        "candidate_rank": int(attempt["candidate_rank"]),
                        "candidate_kind": str(attempt["candidate_kind"]),
                        "generation_present": bool(attempt["generation_present"]),
                        "runtime_failure": bool(attempt["runtime_failure"]),
                        "failure_reasons": attempt_reasons,
                        "intervention_outcomes_read": False,
                    }
                )
            summaries.append(
                {
                    "prompt_mode": mode,
                    "model_label": model,
                    "split": split,
                    "selected_cell_count": int(value["cell_count"]),
                    "original_strict_pass_count": expected_count
                    - len(original_failures),
                    "original_strict_accuracy": (
                        expected_count - len(original_failures)
                    )
                    / expected_count,
                    "original_strict_failure_count": len(original_failures),
                    "replacement_count": replacement_count,
                    "final_fixed_quota_eligible_count": expected_count,
                    "failed_reserve_attempt_count": failed_reserve_attempt_count,
                    "reason_counts": dict(sorted(reason_counts.items())),
                    "status": str(value["status"]),
                    "mapping_sha256": str(value["mapping_sha256"]),
                    "attempt_ledger_sha256": str(value["attempt_ledger_sha256"]),
                }
            )
    if observed_cells != expected_cells:
        raise ValueError(f"Parser audit lost cells: {sorted(expected_cells - observed_cells)}")
    mode_model_order = {
        f"{mode}|{model}": index for index, (mode, model) in enumerate(CELL_ORDER)
    }
    summaries.sort(
        key=lambda row: (
            mode_model_order[f"{row['prompt_mode']}|{row['model_label']}"],
            0 if row["split"] == "discovery" else 1,
        )
    )
    failures.sort(
        key=lambda row: (
            mode_model_order[f"{row['prompt_mode']}|{row['model_label']}"],
            0 if row["split"] == "discovery" else 1,
            row["analysis_slot_seed"],
            row["gold_count"],
        )
    )
    failed_reserve_attempts.sort(
        key=lambda row: (
            mode_model_order[f"{row['prompt_mode']}|{row['model_label']}"],
            0 if row["split"] == "discovery" else 1,
            row["seed"],
            row["gold_count"],
            row["candidate_rank"],
        )
    )
    summary_by_panel = {
        (str(row["prompt_mode"]), str(row["model_label"]), str(row["split"])): row
        for row in summaries
    }
    accuracy_cells: list[dict[str, Any]] = []
    for mode, model in CELL_ORDER:
        discovery = summary_by_panel[(mode, model, "discovery")]
        confirmation = summary_by_panel[(mode, model, "confirmation")]
        pooled_total = int(discovery["selected_cell_count"]) + int(
            confirmation["selected_cell_count"]
        )
        pooled_pass = int(discovery["original_strict_pass_count"]) + int(
            confirmation["original_strict_pass_count"]
        )
        accuracy_cells.append(
            {
                "prompt_mode": mode,
                "model_label": model,
                "discovery_pass_count": int(discovery["original_strict_pass_count"]),
                "discovery_total_count": int(discovery["selected_cell_count"]),
                "discovery_accuracy": float(discovery["original_strict_accuracy"]),
                "confirmation_pass_count": int(
                    confirmation["original_strict_pass_count"]
                ),
                "confirmation_total_count": int(confirmation["selected_cell_count"]),
                "confirmation_accuracy": float(
                    confirmation["original_strict_accuracy"]
                ),
                "pooled_pass_count": pooled_pass,
                "pooled_total_count": pooled_total,
                "pooled_accuracy": pooled_pass / pooled_total,
                "final_fixed_quota_eligible_count": pooled_total,
                "final_fixed_quota_total_count": pooled_total,
            }
        )
    accuracy_modes: list[dict[str, Any]] = []
    for mode in ("enumeration_index", "enumeration_bullet"):
        mode_rows = [row for row in accuracy_cells if row["prompt_mode"] == mode]
        mode_pass = sum(int(row["pooled_pass_count"]) for row in mode_rows)
        mode_total = sum(int(row["pooled_total_count"]) for row in mode_rows)
        accuracy_modes.append(
            {
                "prompt_mode": mode,
                "pass_count": mode_pass,
                "total_count": mode_total,
                "accuracy": mode_pass / mode_total,
            }
        )
    overall_pass = sum(int(row["pass_count"]) for row in accuracy_modes)
    overall_total = sum(int(row["total_count"]) for row in accuracy_modes)
    replacement_policy = audit["replacement_policy"]
    coherent_failed_attempts = list(
        replacement_policy.get("coherent_failed_reserve_attempts", [])
    )
    coherent_replacements = list(replacement_policy.get("coherent_replacements", []))
    if (
        int(replacement_policy.get("ordinary_cell_failure_count", -1))
        != len(failures)
        or len(list(replacement_policy.get("ordinary_failed_reserve_attempts", [])))
        != len(failed_reserve_attempts)
        or int(replacement_policy.get("failed_reserve_attempt_count", -1))
        != len(failed_reserve_attempts) + len(coherent_failed_attempts)
        or int(replacement_policy.get("coherent_replacement_trajectory_count", -1))
        != len(coherent_replacements)
        or replacement_policy.get("all_failures_reported") is not True
        or replacement_policy.get("all_seed_attempts_accounted_for") is not True
        or replacement_policy.get("negative_experimental_results_trigger_replacement")
        is not False
        or replacement_policy.get("silent_sample_exclusion") is not False
    ):
        raise ValueError("Sealed parser replacement policy audit changed")
    return {
        "status": "PASS_PARSER_CONTRACT_AND_COHORT_AUDIT",
        "schema_version": "realistic_niah_v6_parser_report_appendix_v1",
        "strict_grammar": {
            "index_record_pattern": r"^\s*(\d+)\.\s*(.+?)\s*:\s*(-?\d+)\s*$",
            "bullet_record_pattern": r"^\s*-\s*(.+?)\s*:\s*(-?\d+)\s*$",
            "total_line": "Total: <signed integer>; it must be the final non-empty line",
            "optional_terminal_tokens": [
                "<|im_end|>",
                "<turn|>",
                "<|endoftext|>",
                "<｜end▁of▁sentence｜>",
            ],
            "format_statuses": [
                "ok",
                "mixed_markers",
                "wrong_marker",
                "no_records",
                "index_sequence_error",
            ],
        },
        "formal_gate_components": [name for name, _description in PARSER_FORMAL_GATES],
        "registered_sites": [name for name, _description in PARSER_REGISTERED_SITES],
        "legacy_unregistered_site": {
            "site_id": "answer_query_v2",
            "note": "Inherited relaxed Total: locator may be emitted for audit, but it is not a V6 registered analysis site.",
        },
        "cohort_summaries": summaries,
        "behavioral_accuracy": {
            "metric_id": "original_fixed_slot_strict_exact_ordered_enumeration_accuracy",
            "definition": (
                "A raw frozen slot passes only when strict grammar, final Total, exact "
                "ordered gold city-score pairs, marker kind, one-to-one semantic trace, "
                "and gold item count all pass before any reserve replacement."
            ),
            "reserve_attempts_excluded": True,
            "replacement_filtered_cohort_not_model_accuracy": True,
            "cell_summaries": accuracy_cells,
            "mode_summaries": accuracy_modes,
            "overall": {
                "pass_count": overall_pass,
                "total_count": overall_total,
                "accuracy": overall_pass / overall_total,
            },
            "final_fixed_quota_eligibility": {
                "eligible_count": overall_total,
                "total_count": overall_total,
                "rate": 1.0,
                "interpretation": (
                    "Cohort eligibility after outcome-blind sealed reserve replacement; "
                    "this is not behavioral model accuracy."
                ),
            },
        },
        "original_failure_ledger": failures,
        "original_strict_failure_count": len(failures),
        "failed_reserve_attempt_ledger": failed_reserve_attempts,
        "failed_reserve_attempt_count": len(failed_reserve_attempts),
        "final_fixed_quota_unresolved_count": 0,
        "replacement_policy_audit": {
            "ordinary_cell_failure_count": len(failures),
            "ordinary_failed_reserve_attempt_count": len(failed_reserve_attempts),
            "coherent_failed_reserve_attempt_count": len(coherent_failed_attempts),
            "all_failed_reserve_attempt_count": int(
                replacement_policy["failed_reserve_attempt_count"]
            ),
            "coherent_replacement_trajectory_count": len(coherent_replacements),
            "all_failures_reported": True,
            "all_seed_attempts_accounted_for": True,
            "negative_experimental_results_trigger_replacement": False,
            "silent_sample_exclusion": False,
        },
        "selection_policy": (
            "Strict parser eligibility is determined before causal outcomes. Failed fixed "
            "slots use the sealed reserve mapping; coherent broad/native-loop panels replace "
            "the entire required seed trajectory rather than cherry-picking successful counts."
        ),
    }


def collect_report_data(
    *,
    run_root: Path,
    completion_audit: Path,
    baseline_report: Path,
    native_report: Path,
    protocol: Path,
    followup_protocol: Path,
    followup_baseline_report: Path,
    followup_v3_protocol: Path,
    followup_v3_baseline_report: Path,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    source_hashes: dict[str, str] = {}

    def source(path: Path) -> Path:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        source_hashes[logical(path, run_root)] = sha256(path)
        return path

    audit = read_json(source(completion_audit))
    if audit.get("status") != "PASS_FULL_V6_ENUMERATION_SUITE":
        raise ValueError("Narrative report requires the sealed V6 completion audit")
    repo_root = Path(__file__).resolve().parents[1]
    parser_source_paths = {
        "strict_generation_evaluator": repo_root / "src/realistic_niah/parsing.py",
        "hybrid_semantic_parser": repo_root / "src/realistic_niah_v5/parsing.py",
        "rank_episode_parser": repo_root
        / "src/realistic_niah_v5/hybrid_trace_parser.py",
        "generated_city_endpoint": repo_root
        / "src/realistic_niah_v5/same_site_progress_transplant.py",
        "v6_strict_wrapper": repo_root / "src/realistic_niah_v6/parsing.py",
        "v6_site_registry": repo_root / "src/realistic_niah_v6/spec.py",
        "v3_multihop_reparse": repo_root
        / "scripts/analyze_realistic_niah_v6_full_item_multihop.py",
    }
    parser_source_sha256 = {
        label: sha256(source(path)) for label, path in parser_source_paths.items()
    }
    parser_appendix = parser_audit_summary(audit)
    parser_appendix["source_sha256"] = dict(sorted(parser_source_sha256.items()))
    protocol_data = read_json(source(protocol))
    if protocol_data.get("status") != "FROZEN_BEFORE_NEW_DIAGNOSTIC_MODEL_OUTCOMES":
        raise ValueError("Mechanism diagnostic extension was not frozen")
    baseline_hash = sha256(source(baseline_report))
    if baseline_hash != str(protocol_data["baseline_report"]["sha256"]):
        raise ValueError("Frozen audit-matrix report hash disagrees with protocol")
    native_text = source(native_report).read_text(encoding="utf-8")
    native_hash = sha256(native_report)
    followup_protocol_data = read_json(source(followup_protocol))
    if followup_protocol_data.get("status") != FOLLOWUP_PROTOCOL_STATUS:
        raise ValueError("Mechanism follow-up V2 was not frozen before outcomes")
    followup_baseline_hash = sha256(source(followup_baseline_report))
    expected_followup_baseline_hash = str(
        followup_protocol_data["baseline_reports"]["current_narrative_report"][
            "sha256"
        ]
    )
    if followup_baseline_hash != expected_followup_baseline_hash:
        raise ValueError("Follow-up V2 narrative baseline hash disagrees with protocol")
    if (
        str(
            followup_protocol_data["baseline_reports"][
                "original_frozen_audit_matrix_sha256"
            ]
        )
        != baseline_hash
    ):
        raise ValueError("Follow-up V2 changed the original frozen audit matrix")
    if (
        str(
            followup_protocol_data["baseline_reports"][
                "native_thinking_reference"
            ]["sha256"]
        )
        != native_hash
    ):
        raise ValueError("Follow-up V2 changed the Native-thinking reference")
    followup_v3_protocol_data = read_json(source(followup_v3_protocol))
    if followup_v3_protocol_data.get("status") != FOLLOWUP_V3_PROTOCOL_STATUS:
        raise ValueError("Mechanism follow-up V3 was not frozen at its declared checkpoint")
    followup_v3_baseline_hash = sha256(source(followup_v3_baseline_report))
    if followup_v3_baseline_hash != str(
        followup_v3_protocol_data["baseline_reports"]["enumeration_narrative"][
            "sha256"
        ]
    ):
        raise ValueError("Follow-up V3 narrative baseline hash disagrees with protocol")
    if str(
        followup_v3_protocol_data["baseline_reports"]["native_thinking_reference"][
            "sha256"
        ]
    ) != native_hash:
        raise ValueError("Follow-up V3 changed the Native-thinking reference")
    native_checks = {
        "mechanism_report_title_present": "Native-thinking" in native_text,
        "targeted_retrieval_section_present": "Targeted retrieval" in native_text,
        "ncc_limit_present": "NCC" in native_text,
        "commit_query_language_present": "commit" in native_text and "query" in native_text,
        "distributed_content_bound_state_present": (
            "distributed" in native_text and "content-bound" in native_text
        ),
        "single_register_not_established_present": (
            "content-free scalar register" in native_text and "唯一 circuit" in native_text
        ),
        "terminal_partial_relay_present": (
            "partial relay" in native_text and "不表示唯一或完全中介" in native_text
        ),
        "explicit_index_not_natural_evidence_present": (
            "显式-index" in native_text and "不能作为" in native_text
        ),
    }
    if not all(native_checks.values()):
        missing_native_checks = sorted(
            name for name, passed in native_checks.items() if not passed
        )
        raise ValueError(
            "Native-thinking report lacks required comparison anchors: "
            f"{missing_native_checks}"
        )
    for model in MODELS:
        marker = run_root / "mechanism_diagnostic_extension" / f"{model}.COMPLETE"
        if source(marker).read_text(encoding="utf-8").strip() != "PASS":
            raise ValueError(f"Mechanism diagnostic queue is incomplete for {model}")

    followup_root = run_root / "mechanism_followup_v2"
    for marker_name in ("Qwen3-8B.COMPLETE", "Gemma4-E4B.COMPLETE", "FOLLOWUP_V2.COMPLETE"):
        marker = followup_root / marker_name
        if source(marker).read_text(encoding="utf-8").strip() != "PASS":
            raise ValueError(f"Mechanism follow-up V2 marker is incomplete: {marker_name}")

    index_city_support: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        root = followup_root / "index_targeted_city_support" / model
        claims = read_json(source(root / "analysis/claim_gates.json"))
        position_audit = read_json(source(root / "position_audit.json"))
        trials_manifest = read_json(source(root / "trials/manifest.json"))
        if (
            claims.get("analysis_status") != DIAGNOSTIC_LABEL
            or claims.get("phase") != "confirmation"
            or int(claims.get("seed_count", -1)) != 10
            or claims.get("model_label") != model
        ):
            raise ValueError(f"Index targeted-city follow-up contract changed for {model}")
        if (
            position_audit.get("status") != "PASS"
            or position_audit.get("model_label") != model
            or int(position_audit.get("seed_count", -1)) != 10
            or position_audit.get("head_ablation_scope")
            != "registered_query_through_city_prefix"
        ):
            raise ValueError(f"Index position audit contract changed for {model}")
        if (
            str(trials_manifest.get("head_ablation_scope"))
            != "registered_query_through_city_prefix"
            or int(trials_manifest.get("completed_shards", -1)) != 50
        ):
            raise ValueError(f"Index sustained-support trial geometry changed for {model}")
        index_city_support[model] = {
            "analysis": claims,
            "position_audit": position_audit,
            "trials_manifest": trials_manifest,
        }

    full_item_greedy = read_json(
        source(followup_root / "full_item_greedy/analysis/claim_gates.json")
    )
    if (
        full_item_greedy.get("status")
        != "POSTHOC_GREEDY_READOUT_EXTENSION_COMPLETE"
        or full_item_greedy.get("analysis_status") != FOLLOWUP_GREEDY_LABEL
        or bool(full_item_greedy.get("frozen_layers_changed", True))
        or bool(full_item_greedy.get("frozen_k_changed", True))
        or bool(full_item_greedy.get("seed_selection_used_greedy_outcomes", True))
    ):
        raise ValueError("Full-item greedy follow-up contract changed")
    greedy_by_cell: dict[str, dict[str, Any]] = {}
    for row in full_item_greedy.get("cell_summaries", []):
        key = f"{row.get('prompt_mode')}|{row.get('model_label')}"
        if key in greedy_by_cell:
            raise ValueError(f"Duplicate full-item greedy cell: {key}")
        greedy_by_cell[key] = dict(row)
    expected_cells = {f"{mode}|{model}" for mode, model in CELL_ORDER}
    if set(greedy_by_cell) != expected_cells:
        raise ValueError("Full-item greedy analysis lost a model×grammar cell")

    fresh_root = followup_root / "fresh_bullet_gemma_carrier"
    fresh_carrier = read_json(source(fresh_root / "replication_complete.json"))
    fresh_carrier_analysis_path = source(fresh_root / "analysis/claim_gates.json")
    fresh_carrier_analysis = read_json(fresh_carrier_analysis_path)
    fresh_cohort_lock = read_json(source(fresh_root / "cohort/cohort_lock.json"))
    if (
        fresh_carrier.get("status") != FRESH_CARRIER_STATUS
        or fresh_carrier.get("replication_kind")
        != "fresh_prospective_causal_outcomes_with_earlier_discovery_frozen_bank"
        or int(fresh_carrier.get("seed_count", -1)) != 10
        or int(fresh_carrier.get("selected_k", -1)) != 2
        or int(fresh_carrier.get("source_layer", -1)) != 16
        or fresh_carrier.get("head_ablation_scope") != "query_through_carrier"
        or not bool(fresh_carrier.get("original_query_local_null_retained"))
        or bool(fresh_carrier.get("seed_selection_used_intervention_outcomes", True))
        or bool(fresh_carrier.get("frozen_bank_changed", True))
    ):
        raise ValueError("Fresh Bullet-Gemma carrier replication contract changed")
    if (
        fresh_carrier_analysis.get("analysis_status") != DIAGNOSTIC_LABEL
        or fresh_carrier_analysis.get("phase") != "confirmation"
        or int(fresh_carrier_analysis.get("seed_count", -1)) != 10
        or fresh_carrier_analysis.get("head_ablation_scope")
        != "query_through_carrier"
        or fresh_carrier["artifacts"]["claim_gates"]["sha256"]
        != sha256(fresh_carrier_analysis_path)
    ):
        raise ValueError("Fresh Bullet-Gemma carrier analysis contract changed")
    if (
        fresh_cohort_lock.get("status")
        != "FROZEN_BEFORE_CAUSAL_INTERVENTION_OUTCOMES"
        or len(fresh_cohort_lock.get("true_source_seeds", [])) != 10
    ):
        raise ValueError("Fresh Bullet-Gemma carrier cohort lock changed")

    followup_v3_root = run_root / "mechanism_followup_v3"
    multihop_root = followup_v3_root / "full_item_multihop"
    full_item_multihop = read_json(source(multihop_root / "claim_gates.json"))
    multihop_trial_path = source(multihop_root / "trial_reparse.csv")
    multihop_seed_path = source(multihop_root / "seed_effects.csv")
    if (
        full_item_multihop.get("status") != "POSTHOC_MULTIHOP_REPARSE_COMPLETE"
        or full_item_multihop.get("analysis_status")
        != "POSTHOC_AGGREGATE_REPARSE_AFTER_ONE_SCHEMA_SMOKE_ROW"
        or full_item_multihop.get("new_model_forward_used") is not False
        or full_item_multihop.get("all_ten_seeds_retained") is not True
        or full_item_multihop.get("truncated_and_failed_rows_retained_in_denominators")
        is not True
        or int(full_item_multihop.get("row_count", -1)) != 240
        or str(full_item_multihop.get("protocol_sha256"))
        != sha256(followup_v3_protocol)
    ):
        raise ValueError("V3 full-item multihop contract changed")
    multihop_by_cell = {
        f"{row['prompt_mode']}|{row['model_label']}": dict(row)
        for row in full_item_multihop.get("cell_summaries", [])
    }
    expected_cells = {f"{mode}|{model}" for mode, model in CELL_ORDER}
    if set(multihop_by_cell) != expected_cells:
        raise ValueError("V3 multihop analysis lost a model×grammar cell")
    if len(read_csv_rows(multihop_trial_path)) != 240:
        raise ValueError("V3 multihop trial ledger changed row count")
    if len(read_csv_rows(multihop_seed_path)) != 80:
        raise ValueError("V3 multihop seed ledger changed row count")
    fixed_multihop_examples = list(
        full_item_multihop.get("fixed_lowest_seed_examples", [])
    )
    if len(fixed_multihop_examples) != 24:
        raise ValueError("V3 fixed parser examples changed row count")
    parser_appendix["multihop_endpoint"] = {
        "source_field": "generated_known_city_ordinals_any_surface",
        "surface_rule": (
            "Before the earliest reasoning-close marker, match every registered city "
            "case-insensitively with ASCII-alphanumeric word boundaries and preserve "
            "character order; bullet-line-only matches remain a diagnostic field."
        ),
        "reasoning_close_markers": [
            "</think>",
            "<|im_end|>",
            "<end_of_turn>",
        ],
        "expected_donor_path": "range(donor_successor, gold_count + 1)",
        "expected_receiver_path": "range(receiver_successor, gold_count + 1)",
        "exact_prefix_rule": (
            "Depth is the longest prefix with observed[i] == expected[i]. "
            "No skipping, reordering, deduplication, or repair is allowed."
        ),
        "registered_depths": list(full_item_multihop["registered_depths"]),
        "all_rows": int(full_item_multihop["row_count"]),
        "seed_effect_rows": int(full_item_multihop["seed_effect_row_count"]),
        "failed_and_truncated_rows_retained": bool(
            full_item_multihop[
                "truncated_and_failed_rows_retained_in_denominators"
            ]
        ),
        "failure_taxonomy": dict(full_item_multihop["failure_taxonomy"]),
        "fixed_lowest_seed_examples": fixed_multihop_examples,
    }

    aligned_root = run_root / "native_aligned_representation"
    aligned_manifest_path = source(aligned_root / "analysis_manifest.json")
    aligned_audit_path = source(aligned_root / "alignment_audit.json")
    aligned_manifest = read_json(aligned_manifest_path)
    aligned_audit = read_json(aligned_audit_path)
    geometry_contract = followup_v3_protocol_data["representation_3d"]
    if (
        sha256(aligned_manifest_path)
        != str(geometry_contract["source_analysis_manifest_sha256"])
        or sha256(aligned_audit_path)
        != str(geometry_contract["source_alignment_audit_sha256"])
        or aligned_manifest.get("status") != "PASS_NATIVE_ANALYSIS_PATH_ALIGNED"
        or aligned_audit.get("status") != "PASS_NATIVE_ANALYSIS_PATH_ALIGNED"
    ):
        raise ValueError("Native-aligned representation source changed")
    running_candidates = read_csv_rows(
        source(aligned_root / "running_index_candidate_metrics.csv")
    )
    final_candidates = read_csv_rows(
        source(aligned_root / "final_count_candidate_metrics.csv")
    )
    grammar_contrasts = read_csv_rows(
        source(aligned_root / "grammar_contrasts.csv")
    )
    manifold_root = run_root / "representation_manifold_v3"
    manifold_path = source(manifold_root / "representation_manifold.json")
    manifold_manifest_path = source(
        manifold_root / "representation_manifold_manifest.json"
    )
    manifold = read_json(manifold_path)
    manifold_manifest = read_json(manifold_manifest_path)
    if (
        manifold.get("status") != "PASS_DISCOVERY_FIT_CONFIRMATION_PROJECTION"
        or manifold_manifest.get("status")
        != "PASS_DISCOVERY_FIT_CONFIRMATION_PROJECTION"
        or manifold_manifest.get("confirmation_used_for_fit_or_selection") is not False
        or manifold_manifest.get("new_model_forward_used") is not False
        or str(manifold_manifest["output"]["sha256"]) != sha256(manifold_path)
        or str(manifold_manifest["protocol"]["sha256"])
        != sha256(followup_v3_protocol)
    ):
        raise ValueError("V3 representation manifold contract changed")

    cells: dict[str, dict[str, Any]] = {}
    for mode, model in CELL_ORDER:
        root = run_root / mode / model
        key = f"{mode}|{model}"
        running = read_csv_one(
            source(root / "representation/native_aligned/running_index_selected.csv")
        )
        final = read_csv_one(
            source(root / "representation/native_aligned/final_count_selected.csv")
        )
        target = read_json(
            source(root / "causal/targeted_retrieval/confirmation_formal/analysis.json")
        )
        carrier = read_json(
            source(
                root
                / "causal/specialized/confirmation_analysis/targeted_counter_write/confirmation/claim_gates.json"
            )
        )
        terminal = read_json(
            source(
                root
                / "causal/specialized/confirmation_analysis/terminal_state_bridge/confirmation/claim_gates.json"
            )
        )
        ncc = read_json(
            source(
                root
                / "causal/specialized/confirmation_analysis/count_geometry_ncc/claim_gates.json"
            )
        )
        narrow_loop = read_json(
            source(
                root
                / "causal/report_tail/confirmation_formal/native_loop/analysis/claim_gates.json"
            )
        )
        full_loop = {
            phase: read_json(
                source(
                    root
                    / f"causal/report_tail/{phase}_formal/native_loop/full_commit_to_query_analysis/claim_gates.json"
                )
            )
            for phase in ("discovery", "confirmation")
        }
        targeted_continuous = {
            phase: read_json(
                source(
                    root
                    / f"causal/mechanism_diagnostic_extension/targeted_city_likelihood/{phase}/analysis/claim_gates.json"
                )
            )
            for phase in ("discovery", "confirmation")
        }
        for phase, value in targeted_continuous.items():
            if value.get("analysis_status") != DIAGNOSTIC_LABEL:
                raise ValueError(f"Targeted likelihood label changed for {key}/{phase}")

        original_carrier_trials = read_jsonl_directory(
            root / "causal/specialized/confirmation_formal/targeted_counter_write"
        )
        original_scopes = {
            str(row.get("head_ablation_scope", "query_local"))
            for row in original_carrier_trials
        }
        original_position_counts = {
            int(row.get("head_ablation_position_count", 1))
            for row in original_carrier_trials
        }
        terminal_trials = read_jsonl_directory(
            root / "causal/specialized/confirmation_formal/terminal_state_bridge"
        )
        marker_counts = {
            int(row.get("terminal_marker_token_count", -1)) for row in terminal_trials
        }
        nonmarker_counts = {
            int(row.get("terminal_nonmarker_token_count", -1)) for row in terminal_trials
        }

        cell = {
            "key": key,
            "mode": mode,
            "mode_label": MODE_LABEL[mode],
            "grammar": MODE_GRAMMAR[mode],
            "model": model,
            "selected_k": int(target["selected_k"]),
            "representation": {
                "running_layer": int(running["layer"]),
                "running_confirmation_logistic_ba": finite_float(
                    running["confirmation_logistic_balanced_accuracy"]
                ),
                "running_confirmation_ncc_ba": finite_float(
                    running["confirmation_ncc_balanced_accuracy"]
                ),
                "final_layer": int(final["layer"]),
                "final_confirmation_logistic_ba": finite_float(
                    final["confirmation_logistic_balanced_accuracy"]
                ),
                "final_confirmation_ncc_ba": finite_float(
                    final["confirmation_ncc_balanced_accuracy"]
                ),
            },
            "targeted_retrieval": {
                "binary": dict(target["result"]),
                "continuous": targeted_continuous,
            },
            "carrier": {
                "baseline": carrier,
                "original_head_ablation_scopes": sorted(original_scopes),
                "original_head_ablation_position_counts": sorted(
                    original_position_counts
                ),
            },
            "terminal": {
                "baseline": terminal,
                "marker_token_counts": sorted(marker_counts),
                "nonmarker_token_counts": sorted(nonmarker_counts),
            },
            "ncc": ncc,
            "narrow_loop": narrow_loop,
            "full_commit_to_query": full_loop,
            "full_item_greedy": greedy_by_cell[key],
            "full_item_multihop": multihop_by_cell[key],
        }

        if mode == "enumeration_index":
            cell["targeted_retrieval"]["sustained_city_support"] = (
                index_city_support[model]
            )

        if mode == "enumeration_bullet":
            local_terminal = {
                phase: read_json(
                    source(
                        root
                        / f"causal/mechanism_diagnostic_extension/local_terminal_bridge/{phase}/analysis/claim_gates.json"
                    )
                )
                for phase in ("discovery", "confirmation")
            }
            for phase, value in local_terminal.items():
                if value.get("analysis_status") != DIAGNOSTIC_LABEL:
                    raise ValueError(
                        f"Local terminal diagnostic label changed for {key}/{phase}"
                    )
            cell["terminal"]["local_diagnostic"] = local_terminal
        if mode == "enumeration_bullet" and model == "Gemma4-E4B":
            decode_carrier = {
                phase: read_json(
                    source(
                        root
                        / f"causal/mechanism_diagnostic_extension/decode_aligned_carrier/{phase}/analysis/claim_gates.json"
                    )
                )
                for phase in ("discovery", "confirmation")
            }
            for phase, value in decode_carrier.items():
                if value.get("analysis_status") != DIAGNOSTIC_LABEL:
                    raise ValueError(
                        f"Carrier diagnostic label changed for {key}/{phase}"
                    )
            cell["carrier"]["decode_aligned_diagnostic"] = decode_carrier
            cell["carrier"]["fresh_query_through_carrier_replication"] = {
                "replication": fresh_carrier,
                "analysis": fresh_carrier_analysis,
                "cohort_lock": fresh_cohort_lock,
            }
        cells[key] = cell

    ncc_statuses = {str(value["ncc"].get("ncc_effect_status")) for value in cells.values()}
    if ncc_statuses != {"NO_DIRECTIONAL_SPECIFIC_SUPPORT"}:
        raise ValueError(f"Unexpected NCC status set: {ncc_statuses}")
    bullet_terminal_shapes = {
        model: (
            tuple(cells[f"enumeration_bullet|{model}"]["terminal"]["marker_token_counts"]),
            tuple(cells[f"enumeration_bullet|{model}"]["terminal"]["nonmarker_token_counts"]),
        )
        for model in MODELS
    }
    if any(
        marker_counts != (0,) or not nonmarker_counts or min(nonmarker_counts) <= 0
        for marker_counts, nonmarker_counts in bullet_terminal_shapes.values()
    ):
        raise ValueError(f"Unexpected Bullet terminal token geometry: {bullet_terminal_shapes}")
    gemma_carrier = cells["enumeration_bullet|Gemma4-E4B"]["carrier"]
    if (
        gemma_carrier["original_head_ablation_scopes"] != ["query_local"]
        or gemma_carrier["original_head_ablation_position_counts"] != [1]
    ):
        raise ValueError("Original Bullet-Gemma carrier support mismatch changed")

    return {
        "schema_version": "realistic_niah_v6_enumeration_narrative_report_data_v3",
        "status": "PASS_EVIDENCE_COLLECTED",
        "run_root": str(run_root),
        "completion_audit": {
            "path": logical(completion_audit, run_root),
            "status": audit["status"],
            "audit_sha256": audit["audit_sha256"],
            "ordinary_failure_count": audit["replacement_policy"][
                "ordinary_cell_failure_count"
            ],
            "coherent_replacement_trajectory_count": audit["replacement_policy"][
                "coherent_replacement_trajectory_count"
            ],
        },
        "protocol": protocol_data,
        "followup_protocol": followup_protocol_data,
        "followup_protocol_sha256": sha256(followup_protocol),
        "followup_baseline_report_sha256": followup_baseline_hash,
        "followup_v3_protocol": followup_v3_protocol_data,
        "followup_v3_protocol_sha256": sha256(followup_v3_protocol),
        "followup_v3_baseline_report_sha256": followup_v3_baseline_hash,
        "followup": {
            "status": "PASS_COMPLETE",
            "index_targeted_city_support": index_city_support,
            "full_item_greedy": full_item_greedy,
            "fresh_bullet_gemma_carrier": {
                "replication": fresh_carrier,
                "analysis": fresh_carrier_analysis,
                "cohort_lock": fresh_cohort_lock,
            },
        },
        "followup_v3": {
            "status": "PASS_COMPLETE",
            "full_item_multihop": full_item_multihop,
            "multihop_trial_ledger_sha256": sha256(multihop_trial_path),
            "multihop_seed_ledger_sha256": sha256(multihop_seed_path),
        },
        "native_aligned_representation": {
            "manifest": aligned_manifest,
            "audit": aligned_audit,
            "running_candidates": running_candidates,
            "final_candidates": final_candidates,
            "grammar_contrasts": grammar_contrasts,
            "manifold_manifest": manifold_manifest,
        },
        "representation_manifold": manifold,
        "baseline_report_sha256": baseline_hash,
        "native_report_sha256": native_hash,
        "native_report_checks": native_checks,
        "native_template_css": read_style_block(native_report),
        "parser_appendix": parser_appendix,
        "cells": cells,
        "source_sha256": dict(sorted(source_hashes.items())),
    }


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def num(value: Any, digits: int = 3, *, signed: bool = False) -> str:
    value = finite_float(value)
    prefix = "+" if signed and value > 0 else ""
    return f"{prefix}{value:.{digits}f}"


def pct(value: Any, digits: int = 1) -> str:
    return f"{100 * finite_float(value):.{digits}f}%"


def ci(row: Mapping[str, Any], digits: int = 3) -> str:
    estimate = row.get("mean_effect", row.get("estimate"))
    low = row.get("ci_low", row.get("low"))
    high = row.get("ci_high", row.get("high"))
    return (
        f"{num(estimate, digits, signed=True)} "
        f"[{num(low, digits)}, {num(high, digits)}]"
    )


def pill(label: str, kind: str) -> str:
    return f'<span class="pill {esc(kind)}">{esc(label)}</span>'


def baseline_target_status(cell: Mapping[str, Any]) -> tuple[str, str]:
    value = cell["targeted_retrieval"]["binary"]
    if bool(value["interval_strictly_positive"]):
        return "通过", "support"
    if bool(value["directional_positive"]):
        return "方向性", "partial"
    return "未见效应", "null"


def diagnostic_status(value: Mapping[str, Any], gate: str) -> tuple[str, str]:
    if bool(value.get(gate, False)):
        return "诊断支持", "diagnostic"
    directional = bool(
        value.get("directional_specific_signal", False)
        or value.get("targeted_counter_write_directional_pass", False)
    )
    return ("方向性诊断", "partial") if directional else ("诊断仍为 null", "null")


def table(headers: Iterable[str], rows: Iterable[Iterable[str]], *, cls: str = "") -> str:
    return (
        f'<div class="table-wrap"><table class="{esc(cls)}"><thead><tr>'
        + "".join(f"<th>{esc(header)}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def chain_figure(cells: Mapping[str, Mapping[str, Any]]) -> str:
    lanes = []
    for mode, model in CELL_ORDER:
        cell = cells[f"{mode}|{model}"]
        target_label, target_kind = baseline_target_status(cell)
        continuous = cell["targeted_retrieval"]["continuous"]["confirmation"]
        if target_kind != "support" and continuous.get("strong_interval_gate_pass"):
            target_label, target_kind = "连续读出支持†", "diagnostic"
        sustained = cell["targeted_retrieval"].get("sustained_city_support", {})
        sustained_analysis = sustained.get("analysis", {})
        if sustained_analysis.get("strong_interval_gate_pass"):
            target_label, target_kind = "跨 city-prefix 支持‡", "restored"
        elif sustained_analysis.get("directional_specific_signal"):
            target_label, target_kind = "跨 city-prefix 方向性‡", "partial"
        carrier = cell["carrier"]["baseline"]
        carrier_label = "通过" if carrier["targeted_counter_write_strong_gate_pass"] else "原检验 null"
        carrier_kind = "support" if carrier["targeted_counter_write_strong_gate_pass"] else "null"
        if "decode_aligned_diagnostic" in cell["carrier"]:
            diagnostic = cell["carrier"]["decode_aligned_diagnostic"]["confirmation"]
            if diagnostic["targeted_counter_write_strong_gate_pass"]:
                carrier_label, carrier_kind = "跨 carrier 支持†", "diagnostic"
        fresh_carrier = cell["carrier"].get(
            "fresh_query_through_carrier_replication", {}
        )
        fresh_replication = fresh_carrier.get("replication", {})
        if fresh_replication.get("strong_interval_gate_pass"):
            carrier_label, carrier_kind = "fresh 跨 carrier 支持§", "restored"
        elif fresh_replication.get("directional_gate_pass"):
            carrier_label, carrier_kind = "fresh 跨 carrier 方向性§", "partial"
        commit = cell["full_commit_to_query"]["confirmation"]
        commit_label = "全状态支持*" if commit["strong_direct_gate_pass"] else "未通过"
        commit_kind = "restored" if commit["strong_direct_gate_pass"] else "null"
        greedy = cell["full_item_greedy"]
        if greedy.get("strong_interval_gate_pass"):
            commit_label, commit_kind = "全状态 + greedy 支持‡", "restored"
        elif greedy.get("directional"):
            commit_label = "全状态支持；greedy 方向性‡"
        multihop = cell.get("full_item_multihop", {})
        if multihop.get("primary_depth4_strong_gate_pass"):
            commit_label, commit_kind = "全状态 + 四步 continuation¶", "restored"
        lanes.append(
            f'<div class="chain-lane" data-cell="{esc(cell["key"])}">'
            f'<div class="lane-name"><strong>{esc(cell["mode_label"])} · {esc(model.replace("4-", " ").replace("3-", " "))}</strong>'
            f'<span>K={cell["selected_k"]}</span></div>'
            f'<div class="node"><b>Targeted query</b>{pill(target_label, target_kind)}</div>'
            '<div class="arrow" aria-hidden="true">→</div>'
            f'<div class="node"><b>Grammar carrier</b>{pill(carrier_label, carrier_kind)}</div>'
            '<div class="arrow" aria-hidden="true">→</div>'
            f'<div class="node"><b>Commit state</b>{pill(commit_label, commit_kind)}</div>'
            '<div class="arrow" aria-hidden="true">↺</div>'
            '<div class="node"><b>Next query</b><span class="micro">attention / city likelihood</span></div>'
            '</div>'
        )
    return '<div class="chain-scroll"><div class="chain">' + "".join(lanes) + "</div></div>"


def representation_layer_curves_svg(
    rows: Sequence[Mapping[str, Any]], *, endpoint_label: str
) -> str:
    """Four-panel layerwise discovery/confirmation representation figure."""

    width, height = 1040, 640
    panel_w, panel_h = 465, 230
    origins = ((66, 72), (560, 72), (66, 370), (560, 370))
    parts = [
        f'<svg class="paper-chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{esc(endpoint_label)} layerwise representation comparison">',
        f'<title>{esc(endpoint_label)} layerwise representation comparison</title>',
    ]
    for panel_index, (mode, model) in enumerate(CELL_ORDER):
        panel = sorted(
            (
                row
                for row in rows
                if str(row["prompt_mode"]) == mode
                and str(row["model_label"]) == model
            ),
            key=lambda row: int(row["layer"]),
        )
        if not panel:
            raise ValueError(f"Missing representation curve for {mode}/{model}")
        x0, y0 = origins[panel_index]
        plot_x, plot_y = x0 + 48, y0 + 28
        plot_w, plot_h = panel_w - 66, panel_h - 58
        layer_min = int(panel[0]["layer"])
        layer_max = int(panel[-1]["layer"])
        selected = max(
            panel, key=lambda row: finite_float(row["discovery_selection_score"])
        )

        def sx(layer: int) -> float:
            return plot_x + (layer - layer_min) / max(layer_max - layer_min, 1) * plot_w

        def sy(value: float) -> float:
            return plot_y + (1.0 - max(0.0, min(1.0, value))) * plot_h

        parts.extend(
            [
                f'<text x="{x0}" y="{y0-8}" class="heat-title">{esc(MODE_LABEL[mode])} · {esc(model)}</text>',
                f'<rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" class="plot-bg"/>',
            ]
        )
        for tick in (0.0, 0.5, 1.0):
            y = sy(tick)
            parts.append(
                f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x+plot_w}" y2="{y:.1f}" class="grid"/>'
            )
            parts.append(
                f'<text x="{plot_x-9}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick:.1f}</text>'
            )
        chance_y = sy(0.1)
        parts.append(
            f'<line x1="{plot_x}" y1="{chance_y:.1f}" x2="{plot_x+plot_w}" y2="{chance_y:.1f}" stroke="#98a2b3" stroke-dasharray="5 4"/>'
        )
        selected_x = sx(int(selected["layer"]))
        parts.append(
            f'<line x1="{selected_x:.1f}" y1="{plot_y}" x2="{selected_x:.1f}" y2="{plot_y+plot_h}" stroke="#b42318" stroke-dasharray="3 3"><title>discovery-selected L{int(selected["layer"])}</title></line>'
        )
        series = (
            ("discovery_selection_score", "#667085", "5 4"),
            ("confirmation_logistic_balanced_accuracy", "#0f766e", ""),
            ("confirmation_ncc_balanced_accuracy", "#d97706", ""),
        )
        for field, color, dash in series:
            points = " ".join(
                f'{sx(int(row["layer"])):.1f},{sy(finite_float(row[field])):.1f}'
                for row in panel
            )
            parts.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>'
            )
        parts.extend(
            [
                f'<text x="{plot_x}" y="{plot_y+plot_h+20}" class="tick">L{layer_min}</text>',
                f'<text x="{plot_x+plot_w}" y="{plot_y+plot_h+20}" text-anchor="end" class="tick">L{layer_max}</text>',
                f'<text x="{plot_x+plot_w/2}" y="{plot_y+plot_h+38}" text-anchor="middle" class="axis-label">post-block layer · default L{int(selected["layer"])}</text>',
            ]
        )
    parts.extend(
        [
            '<line x1="225" y1="626" x2="251" y2="626" stroke="#0f766e" stroke-width="2"/><text x="258" y="630" class="legend-label">confirmation logistic BA</text>',
            '<line x1="440" y1="626" x2="466" y2="626" stroke="#d97706" stroke-width="2"/><text x="473" y="630" class="legend-label">confirmation NCC BA</text>',
            '<line x1="638" y1="626" x2="664" y2="626" stroke="#667085" stroke-width="2" stroke-dasharray="5 4"/><text x="671" y="630" class="legend-label">discovery selection score</text>',
            '</svg>',
        ]
    )
    return "".join(parts)


def interval_forest_svg(
    title: str,
    rows: Sequence[tuple[str, Mapping[str, Any], bool]],
    *,
    unit: str,
) -> str:
    """Compact mean/95%-CI forest used for each causal edge."""

    if not rows:
        raise ValueError(f"No rows for interval figure {title}")
    width = 1040
    # Keep a full text line below the tick labels.  The previous axis baseline
    # sat at height-1, which allowed glyph descenders to be clipped by the SVG
    # viewport and left only 15 px between ticks and the axis title.
    height = 118 + 42 * len(rows)
    plot_left, plot_right = 340, 990
    center = (plot_left + plot_right) / 2
    values = [
        abs(finite_float(row[field]))
        for _label, row, _gate in rows
        for field in ("mean_effect", "ci_low", "ci_high")
    ]
    limit = max(max(values, default=0.0) * 1.12, 1e-3)

    def sx(value: float) -> float:
        clipped = max(-limit, min(limit, value))
        return center + clipped / limit * (plot_right - plot_left) / 2

    parts = [
        f'<svg class="paper-chart forest" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
        f'<title>{esc(title)}</title>',
        f'<text x="24" y="27" class="heat-title">{esc(title)}</text>',
        f'<line x1="{center:.1f}" y1="45" x2="{center:.1f}" y2="{height-62}" stroke="#98a2b3" stroke-dasharray="4 4"/>',
    ]
    for index, (label, row, gate) in enumerate(rows):
        y = 64 + index * 42
        mean = finite_float(row["mean_effect"])
        low = finite_float(row["ci_low"])
        high = finite_float(row["ci_high"])
        color = "#0f766e" if gate else "#d97706"
        parts.extend(
            [
                f'<text x="325" y="{y+4}" text-anchor="end" class="chart-axis">{esc(label)}</text>',
                f'<line x1="{sx(low):.1f}" y1="{y}" x2="{sx(high):.1f}" y2="{y}" stroke="#475467" stroke-width="2"/>',
                f'<line x1="{sx(low):.1f}" y1="{y-5}" x2="{sx(low):.1f}" y2="{y+5}" stroke="#475467"/>',
                f'<line x1="{sx(high):.1f}" y1="{y-5}" x2="{sx(high):.1f}" y2="{y+5}" stroke="#475467"/>',
                f'<circle cx="{sx(mean):.1f}" cy="{y}" r="5.3" fill="{color}"><title>{esc(label)}: {mean:+.6g} [{low:+.6g}, {high:+.6g}]</title></circle>',
                f'<text x="1016" y="{y+4}" text-anchor="end" class="chart-value">{mean:+.3g}</text>',
            ]
        )
    parts.extend(
        [
            f'<text x="{plot_left}" y="{height-34}" class="tick">−{limit:.3g}</text>',
            f'<text x="{center}" y="{height-34}" text-anchor="middle" class="tick">0</text>',
            f'<text x="{plot_right}" y="{height-34}" text-anchor="end" class="tick">+{limit:.3g}</text>',
            f'<text x="{center}" y="{height-11}" text-anchor="middle" class="axis-label">{esc(unit)}</text>',
            '</svg>',
        ]
    )
    return "".join(parts)


def behavioral_accuracy_svg(rows: Sequence[Mapping[str, Any]]) -> str:
    width, height = 1040, 390
    left, right = 250, 970
    top, step, bar_height = 54, 66, 28
    plot_width = right - left
    colors = {
        "enumeration_index": "#0f766e",
        "enumeration_bullet": "#46758f",
    }
    parts = [
        f'<svg class="paper-chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="enum-accuracy-title enum-accuracy-desc">',
        '<title id="enum-accuracy-title">Raw strict exact enumeration accuracy</title>',
        '<desc id="enum-accuracy-desc">Horizontal bars compare pre-replacement strict exact pass rates for four grammar by model cells.</desc>',
        f'<rect x="{left}" y="32" width="{plot_width}" height="{step * len(rows) + 12}" class="plot-bg"/>',
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + plot_width * tick
        parts.append(
            f'<line x1="{x:.1f}" y1="32" x2="{x:.1f}" y2="{step * len(rows) + 44}" class="grid"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{step * len(rows) + 68}" text-anchor="middle" class="tick">{tick:.2f}</text>'
        )
    for index, row in enumerate(rows):
        y = top + index * step
        rate = finite_float(row["pooled_accuracy"])
        bar_width = plot_width * rate
        label = f"{MODE_LABEL[str(row['prompt_mode'])]} · {row['model_label']}"
        value_label = (
            f'{pct(rate)} · {esc(row["pooled_pass_count"])}/'
            f'{esc(row["pooled_total_count"])}'
        )
        value_inside = bar_width >= 180
        value_x = left + bar_width - 10 if value_inside else left + bar_width + 10
        value_anchor = "end" if value_inside else "start"
        value_class = "bar-value bar-value-inverse" if value_inside else "bar-value"
        parts.extend(
            (
                f'<text x="{left - 14}" y="{y + 20}" text-anchor="end" class="bar-label">{esc(label)}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" fill="{colors[str(row["prompt_mode"])]}" opacity=".86"/>',
                f'<text x="{value_x:.1f}" y="{y + 20}" text-anchor="{value_anchor}" class="{value_class}">{value_label}</text>',
            )
        )
    parts.append(
        f'<text x="{(left + right) / 2:.1f}" y="{height - 16}" text-anchor="middle" class="axis-label">pre-replacement strict exact pass rate</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def grouped_rate_svg(
    title: str,
    groups: Sequence[str],
    series: Sequence[tuple[str, str, Sequence[float]]],
    *,
    y_label: str,
) -> str:
    """Small deterministic grouped-bar chart for rates or binary gates."""

    if not groups or not series:
        raise ValueError(f"No data for grouped-rate figure {title}")
    if any(len(values) != len(groups) for _label, _color, values in series):
        raise ValueError(f"Grouped-rate series length mismatch for {title}")
    width, height = 1040, 455
    left, right, top, bottom = 94, 1000, 52, 344
    plot_w, plot_h = right - left, bottom - top
    group_w = plot_w / len(groups)
    bar_w = min(34.0, group_w * 0.74 / len(series))
    parts = [
        f'<svg class="paper-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
        f'<title>{esc(title)}</title>',
        f'<text x="{left}" y="28" class="heat-title">{esc(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" class="plot-bg"/>',
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = bottom - tick * plot_h
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>',
                f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick:.2f}</text>',
            ]
        )
    for group_index, group in enumerate(groups):
        group_center = left + (group_index + 0.5) * group_w
        total_w = len(series) * bar_w
        for series_index, (label, color, values) in enumerate(series):
            value = max(0.0, min(1.0, finite_float(values[group_index])))
            x = group_center - total_w / 2 + series_index * bar_w
            y = bottom - value * plot_h
            parts.extend(
                [
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-2:.1f}" height="{value*plot_h:.1f}" fill="{color}" opacity=".86"><title>{esc(group)} · {esc(label)}: {value:.3f}</title></rect>',
                    f'<text x="{x+(bar_w-2)/2:.1f}" y="{max(y-5,top+10):.1f}" text-anchor="middle" class="chart-value">{value:.2f}</text>',
                ]
            )
        parts.append(
            f'<text x="{group_center:.1f}" y="{bottom+24}" text-anchor="middle" class="tick">{esc(group)}</text>'
        )
    legend_x = left
    for label, color, _values in series:
        parts.extend(
            [
                f'<rect x="{legend_x}" y="396" width="14" height="10" fill="{color}"/>',
                f'<text x="{legend_x+20}" y="405" class="legend-label">{esc(label)}</text>',
            ]
        )
        legend_x += 20 + max(120, 7 * len(label))
    parts.extend(
        [
            f'<text x="24" y="{(top+bottom)/2:.1f}" text-anchor="middle" class="axis-label" transform="rotate(-90 24 {(top+bottom)/2:.1f})">{esc(y_label)}</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def mask_scope_timeline_svg(audit: Mapping[str, Any]) -> str:
    if audit.get("status") != "PASS_DISTINCT_TEMPORAL_SCOPES_AUDITED":
        raise ValueError("Head-mask temporal-scope audit is missing or invalid")
    width, height = 1040, 420
    left, right, top = 290, 995, 76
    labels = ("registered query", "city prefix", "grammar carrier", "item commit", "cached decode →")
    lanes = (
        ("Behavior necessity · all 4 cells", 0, 5, "#0f766e", "persistent", "persistent; decode_head_ablation_steps=-1"),
        ("Original carrier assay · all 4", 0, 1, "#d97706", "query-only", "query_local; one teacher-forced position"),
        ("Index sustained likelihood · Q/G", 0, 2, "#46758f", "through city prefix", "registered_query_through_city_prefix"),
        ("Bullet-Gemma fresh carrier", 0, 3, "#7c3aed", "through carrier", "query_through_carrier"),
    )
    step = (right - left) / len(labels)
    parts = [
        f'<svg class="paper-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Audited head-mask temporal scopes">',
        '<title>Audited head-mask temporal scopes</title>',
        f'<rect x="{left}" y="48" width="{right-left}" height="278" class="plot-bg"/>',
    ]
    for index, label in enumerate(labels):
        x = left + (index + 0.5) * step
        parts.extend(
            [
                f'<line x1="{left+index*step:.1f}" y1="48" x2="{left+index*step:.1f}" y2="326" class="grid"/>',
                f'<text x="{x:.1f}" y="368" text-anchor="middle" class="tick">{esc(label)}</text>',
            ]
        )
    for lane_index, (label, start, end, color, visible_note, full_note) in enumerate(lanes):
        y = top + lane_index * 64
        x = left + start * step + 5
        bar_w = (end - start) * step - 10
        parts.extend(
            [
                f'<text x="{left-14}" y="{y+19}" text-anchor="end" class="chart-axis">{esc(label)}</text>',
                f'<rect x="{x:.1f}" y="{y}" width="{bar_w:.1f}" height="28" rx="4" fill="{color}" opacity=".84"><title>{esc(full_note)}</title></rect>',
                f'<text x="{x+8:.1f}" y="{y+19}" class="scope-label">{esc(visible_note)}</text>',
            ]
        )
    parts.extend(
        [
            f'<line x1="{right}" y1="48" x2="{right}" y2="326" class="grid"/>',
            '<text x="642" y="403" text-anchor="middle" class="axis-label">autoregressive intervention support window (categorical token phases; not elapsed time)</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def suite_coverage_svg() -> str:
    labels = ("§1", "§2", "§3", "§4", "§5", "§6", "§7", "§8", "Appx")
    counts = (1, 1, 5, 1, 3, 3, 1, 2, 3)
    width, height = 1040, 390
    left, right, top, bottom = 80, 990, 52, 300
    plot_w, plot_h = right - left, bottom - top
    max_count = max(counts)
    bar_w = plot_w / len(labels) * 0.58
    parts = [
        f'<svg class="paper-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Twenty sealed Enumeration frames mapped to Native-thinking sections">',
        '<title>Twenty sealed Enumeration frames mapped to Native-thinking sections</title>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" class="plot-bg"/>',
    ]
    for tick in range(max_count + 1):
        y = bottom - tick / max_count * plot_h
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>',
                f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick}</text>',
            ]
        )
    for index, (label, count) in enumerate(zip(labels, counts)):
        center = left + (index + 0.5) * plot_w / len(labels)
        y = bottom - count / max_count * plot_h
        parts.extend(
            [
                f'<rect x="{center-bar_w/2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bottom-y:.1f}" fill="#0f766e" opacity=".82"><title>{esc(label)}: {count} sealed frames</title></rect>',
                f'<text x="{center:.1f}" y="{y-7:.1f}" text-anchor="middle" class="chart-value">{count}</text>',
                f'<text x="{center:.1f}" y="{bottom+23}" text-anchor="middle" class="tick">{esc(label)}</text>',
            ]
        )
    parts.extend(
        [
            '<text x="535" y="356" text-anchor="middle" class="axis-label">Native-thinking narrative section / Appendix slot</text>',
            '<text x="24" y="176" text-anchor="middle" class="axis-label" transform="rotate(-90 24 176)">sealed report-frame count</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def answer_query_layer_adoption_svg(
    extension_cells: Mapping[str, Mapping[str, Any]],
) -> str:
    """Four-panel Native-aligned answer-query donor adoption layer sweep."""

    width, height = 1040, 660
    panel_w, panel_h = 420.0, 190.0
    parts = [
        f'<svg class="paper-chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="enum-answer-adoption-title enum-answer-adoption-desc">',
        '<title id="enum-answer-adoption-title">Enumeration answer-query full-state donor-count adoption</title>',
        '<desc id="enum-answer-adoption-desc">Four panels show seed-equal greedy donor-count adoption over the preregistered eight-layer grids for Index and Bullet enumeration with Qwen and Gemma.</desc>',
    ]
    colors = {"Qwen3-8B": "#0f766e", "Gemma4-E4B": "#7c3aed"}
    for index, (mode, model) in enumerate(CELL_ORDER):
        cell = extension_cells[f"{mode}|{model}"]
        rows = sorted(cell["answer_layer_effects"], key=lambda row: int(row["layer"]))
        col, row_index = index % 2, index // 2
        x0 = 65.0 + col * 500.0
        y0 = 58.0 + row_index * 285.0
        max_layer = max(int(row["layer"]) for row in rows)
        parts.extend(
            (
                f'<text x="{x0:.1f}" y="{y0 - 19:.1f}" class="heat-title">'
                f'{esc(MODE_LABEL[mode])} · {esc(model)} · answer_query_v3</text>',
                f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{panel_w:.1f}" height="{panel_h:.1f}" class="plot-bg"/>',
            )
        )
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = y0 + panel_h * (1.0 - tick)
            parts.append(
                f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x0 + panel_w:.1f}" y2="{y:.1f}" class="grid"/>'
            )
            parts.append(
                f'<text x="{x0 - 9:.1f}" y="{y + 4:.1f}" text-anchor="end" class="tick">{tick:.2f}</text>'
            )
        points: list[tuple[float, float, Mapping[str, Any]]] = []
        for value in rows:
            x = x0 + panel_w * int(value["layer"]) / max_layer
            y = y0 + panel_h * (1.0 - finite_float(value["full_donor_adoption"]))
            points.append((x, y, value))
        parts.append(
            '<polyline points="'
            + " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
            + f'" fill="none" stroke="{colors[model]}" stroke-width="3"/>'
        )
        for x, y, value in points:
            low_y = y0 + panel_h * (
                1.0 - finite_float(value["full_donor_adoption_ci95_low"])
            )
            high_y = y0 + panel_h * (
                1.0 - finite_float(value["full_donor_adoption_ci95_high"])
            )
            parts.extend(
                (
                    f'<line x1="{x:.1f}" y1="{high_y:.1f}" x2="{x:.1f}" y2="{low_y:.1f}" stroke="#475467" stroke-width="1.5"/>',
                    f'<line x1="{x - 4:.1f}" y1="{high_y:.1f}" x2="{x + 4:.1f}" y2="{high_y:.1f}" stroke="#475467"/>',
                    f'<line x1="{x - 4:.1f}" y1="{low_y:.1f}" x2="{x + 4:.1f}" y2="{low_y:.1f}" stroke="#475467"/>',
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{colors[model]}"><title>'
                    f'L{esc(value["layer"])}: donor adoption={num(value["full_donor_adoption"])}; '
                    f'95% CI [{num(value["full_donor_adoption_ci95_low"])}, '
                    f'{num(value["full_donor_adoption_ci95_high"])}]</title></circle>',
                    f'<text x="{x:.1f}" y="{y0 + panel_h + 19:.1f}" text-anchor="middle" class="tick">{esc(value["layer"])}</text>',
                )
            )
        parts.extend(
            (
                f'<text x="{x0 + panel_w / 2:.1f}" y="{y0 + panel_h + 42:.1f}" text-anchor="middle" class="axis-label">zero-based post-block layer</text>',
                f'<text transform="translate({x0 - 43:.1f} {y0 + panel_h / 2:.1f}) rotate(-90)" text-anchor="middle" class="axis-label">greedy donor-count adoption</text>',
            )
        )
    parts.extend(
        (
            '<line x1="330" y1="634" x2="360" y2="634" stroke="#0f766e" stroke-width="3"/>',
            '<text x="369" y="638" class="legend-label">seed-equal mean</text>',
            '<line x1="540" y1="624" x2="540" y2="644" stroke="#475467" stroke-width="1.5"/>',
            '<text x="552" y="638" class="legend-label">true-source-seed cluster bootstrap 95% CI</text>',
            "</svg>",
        )
    )
    return "".join(parts)


def multihop_depth_svg(cell_summaries: Sequence[Mapping[str, Any]]) -> str:
    by_cell = {
        (str(row["prompt_mode"]), str(row["model_label"])): row
        for row in cell_summaries
    }
    width, height = 1040, 405
    parts = [
        f'<svg class="paper-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Full-item multihop donor continuation by grammar and model">',
        '<title>Full-item multihop donor continuation by grammar and model</title>',
    ]
    for panel_index, mode in enumerate(MODES):
        x0, y0, plot_w, plot_h = 72 + panel_index * 510, 64, 420, 245

        def sx(index: int) -> float:
            # Inset categorical points so the model-specific horizontal jitter
            # remains inside the plotting rectangle at depth 1 and depth 4.
            return x0 + 12 + index * (plot_w - 24) / 2

        def sy(value: float) -> float:
            return y0 + (1.0 - value) * plot_h

        parts.append(
            f'<text x="{x0}" y="32" class="heat-title">{esc(MODE_LABEL[mode])} · exact donor prefix</text>'
        )
        parts.append(
            f'<rect x="{x0}" y="{y0}" width="{plot_w}" height="{plot_h}" class="plot-bg"/>'
        )
        for tick in (0.0, 0.5, 1.0):
            y = sy(tick)
            parts.append(
                f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+plot_w}" y2="{y:.1f}" class="grid"/>'
            )
            parts.append(
                f'<text x="{x0-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick:.1f}</text>'
            )
        for model_index, (model, color) in enumerate(
            (("Qwen3-8B", "#0f766e"), ("Gemma4-E4B", "#7c3aed"))
        ):
            row = by_cell[(mode, model)]
            values = [finite_float(row[f"depth_{depth}"]["patched_rate"]) for depth in (1, 2, 4)]
            series_offset = -6 if model_index == 0 else 6
            points = " ".join(
                f"{sx(index) + series_offset:.1f},{sy(value):.1f}"
                for index, value in enumerate(values)
            )
            parts.append(
                f'<polyline class="multihop-series" data-series-offset="{series_offset}" points="{points}" fill="none" stroke="{color}" stroke-width="3"/>'
            )
            for index, (depth, value) in enumerate(zip((1, 2, 4), values)):
                point_x = sx(index) + series_offset
                label_x = point_x - 8 if model_index == 0 else point_x + 8
                label_anchor = "end" if model_index == 0 else "start"
                parts.append(
                    f'<circle cx="{point_x:.1f}" cy="{sy(value):.1f}" r="6" fill="{color}"><title>{esc(model)} depth≥{depth}: {value:.2f}</title></circle>'
                )
                parts.append(
                    f'<text x="{label_x:.1f}" y="{sy(value)-11:.1f}" text-anchor="{label_anchor}" class="chart-value">{value:.2f}</text>'
                )
        for index, depth in enumerate((1, 2, 4)):
            parts.append(
                f'<text x="{sx(index):.1f}" y="{y0+plot_h+24}" text-anchor="middle" class="tick">depth ≥ {depth}</text>'
            )
        parts.append(
            f'<text x="{x0+plot_w/2}" y="{y0+plot_h+48}" text-anchor="middle" class="axis-label">donor_to_receiver unconditional rate · 20 direction rows</text>'
        )
    parts.extend(
        [
            '<line x1="378" y1="388" x2="405" y2="388" stroke="#0f766e" stroke-width="3"/><text x="413" y="392" class="legend-label">Qwen3-8B</text>',
            '<line x1="548" y1="388" x2="575" y2="388" stroke="#7c3aed" stroke-width="3"/><text x="583" y="392" class="legend-label">Gemma4-E4B</text>',
            '</svg>',
        ]
    )
    return "".join(parts)


def point_cloud_script(payload: Mapping[str, Any]) -> str:
    """Dependency-free synced 3D viewer for all four Enumeration cells."""

    expected_cells = {f"{mode}|{model}" for mode, model in CELL_ORDER}
    if payload.get("status") != "PASS_DISCOVERY_FIT_CONFIRMATION_PROJECTION":
        raise ValueError("Representation manifold is not a PASS projection payload")
    for endpoint in ("running", "final"):
        endpoint_cells = payload.get(endpoint)
        if not isinstance(endpoint_cells, Mapping) or set(endpoint_cells) != expected_cells:
            raise ValueError(
                f"Representation manifold lost four {endpoint} cells"
            )
        for cell_key, cell in endpoint_cells.items():
            layers = cell.get("layers") if isinstance(cell, Mapping) else None
            default_layer = str(cell.get("default_layer")) if isinstance(cell, Mapping) else ""
            if (
                not isinstance(layers, Mapping)
                or not layers
                or default_layer not in layers
                or not cell.get("token_site")
            ):
                raise ValueError(
                    f"Representation manifold has invalid layer metadata: "
                    f"{endpoint}/{cell_key}"
                )
            for layer, layer_data in layers.items():
                evr = layer_data.get("evr") if isinstance(layer_data, Mapping) else None
                rows = layer_data.get("rows") if isinstance(layer_data, Mapping) else None
                if (
                    not isinstance(evr, list)
                    or len(evr) != 3
                    or not all(isinstance(value, (int, float)) for value in evr)
                    or not isinstance(rows, list)
                    or not rows
                    or any(
                        not isinstance(row, list)
                        or len(row) != 5
                        or not all(isinstance(value, (int, float)) for value in row)
                        for row in rows
                    )
                ):
                    raise ValueError(
                        f"Representation manifold has invalid coordinates: "
                        f"{endpoint}/{cell_key}/L{layer}"
                    )

    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    return "const ENUM_GEOMETRY=" + encoded + r""";
const ENUM_COLORS=['#0b4f6c','#176b87','#228b8d','#3aa17e','#66b56b','#9abe55','#d0bb42','#e69b35','#df6f32','#c9413a'];
const ENUM_CLOUDS=[];
class EnumerationPointCloud3D {
  constructor(canvasId,layerId,statsId,endpoint){
    this.canvas=document.getElementById(canvasId);this.ctx=this.canvas.getContext('2d');
    this.layer=document.getElementById(layerId);this.stats=document.getElementById(statsId);this.endpoint=endpoint;
    this.yaw=-0.58;this.pitch=0.34;this.dragging=false;this.last=null;
    this.layer.addEventListener('change',()=>this.setLayer());
    this.canvas.addEventListener('pointerdown',event=>this.pointerDown(event));
    this.canvas.addEventListener('pointermove',event=>this.pointerMove(event));
    this.canvas.addEventListener('pointerup',event=>this.pointerUp(event));
    this.canvas.addEventListener('pointercancel',event=>this.pointerUp(event));
    this.canvas.addEventListener('dblclick',()=>this.resetView());
    if(window.ResizeObserver)new ResizeObserver(()=>this.resize()).observe(this.canvas);else window.addEventListener('resize',()=>this.resize());
    ENUM_CLOUDS.push(this);this.setCell();this.resize();
  }
  cellKey(){return document.getElementById('enum-geometry-grammar').value+'|'+document.getElementById('enum-geometry-model').value;}
  setCell(){
    this.modelData=ENUM_GEOMETRY[this.endpoint][this.cellKey()];this.layer.innerHTML='';
    Object.keys(this.modelData.layers).map(Number).sort((a,b)=>a-b).forEach(layer=>{
      const option=document.createElement('option');option.value=String(layer);
      option.textContent='L'+layer+(layer===this.modelData.default_layer?' · discovery-selected':'');this.layer.appendChild(option);
    });
    this.layer.value=String(this.modelData.default_layer);this.setLayer();
  }
  setLayer(){
    this.data=this.modelData.layers[this.layer.value];this.rows=this.data.rows;this.prepare();
    const evr=100*this.data.evr.reduce((sum,value)=>sum+value,0);
    this.stats.textContent=this.modelData.token_site+' · L'+this.layer.value+' · '+this.modelData.discovery_rows+' discovery states fit StandardScaler/PCA3 · '+this.rows.length+' confirmation states shown · EVR₁₋₃ '+evr.toFixed(1)+'%';this.draw();
  }
  resetView(){ENUM_CLOUDS.forEach(cloud=>{cloud.yaw=-0.58;cloud.pitch=0.34;cloud.draw();});}
  pointerDown(event){this.dragging=true;this.last=[event.clientX,event.clientY];this.canvas.setPointerCapture(event.pointerId);}
  pointerMove(event){if(!this.dragging)return;const dx=event.clientX-this.last[0],dy=event.clientY-this.last[1];this.last=[event.clientX,event.clientY];ENUM_CLOUDS.forEach(cloud=>{cloud.yaw+=dx*.009;cloud.pitch=Math.max(-1.35,Math.min(1.35,cloud.pitch+dy*.009));cloud.draw();});}
  pointerUp(event){this.dragging=false;this.last=null;if(this.canvas.hasPointerCapture(event.pointerId))this.canvas.releasePointerCapture(event.pointerId);}
  resize(){const rect=this.canvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2),width=Math.max(320,Math.round(rect.width)),height=Math.max(390,Math.round(rect.height));if(this.canvas.width!==Math.round(width*dpr)||this.canvas.height!==Math.round(height*dpr)){this.canvas.width=Math.round(width*dpr);this.canvas.height=Math.round(height*dpr);}this.ctx.setTransform(dpr,0,0,dpr,0,0);this.cssWidth=width;this.cssHeight=height;this.draw();}
  prepare(){
    const coords=[0,1,2].map(axis=>this.rows.map(row=>row[axis+2]));
    this.centers=coords.map(values=>(Math.min(...values)+Math.max(...values))/2);
    this.scales=coords.map(values=>Math.max(Math.max(...values)-Math.min(...values),1e-8)/2);
    this.points=this.rows.map(row=>({seed:row[0],label:row[1],v:[0,1,2].map(axis=>(row[axis+2]-this.centers[axis])/this.scales[axis])}));
    this.centroids=[];for(let label=1;label<=10;label++){const group=this.points.filter(point=>point.label===label);if(group.length)this.centroids.push({label:label,v:[0,1,2].map(axis=>group.reduce((sum,point)=>sum+point.v[axis],0)/group.length)});}
  }
  rotate(v){const cy=Math.cos(this.yaw),sy=Math.sin(this.yaw),cp=Math.cos(this.pitch),sp=Math.sin(this.pitch),x=cy*v[0]+sy*v[2],z=-sy*v[0]+cy*v[2];return[x,cp*v[1]-sp*z,sp*v[1]+cp*z];}
  project(v){const r=this.rotate(v),depth=3.4-r[2]*.34,scale=Math.min(this.cssWidth,510)*.78/depth;return{x:this.cssWidth*.5+r[0]*scale,y:this.cssHeight*.45-r[1]*scale,z:r[2]};}
  line(a,b,color,width,dash){const pa=this.project(a),pb=this.project(b),ctx=this.ctx;ctx.save();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash||[]);ctx.beginPath();ctx.moveTo(pa.x,pa.y);ctx.lineTo(pb.x,pb.y);ctx.stroke();ctx.restore();}
  draw(){
    if(!this.ctx||!this.rows||!this.cssWidth)return;const ctx=this.ctx,w=this.cssWidth,h=this.cssHeight;ctx.clearRect(0,0,w,h);ctx.fillStyle='#fbfcfe';ctx.fillRect(0,0,w,h);
    [[[0,0,0],[1.18,0,0],'PC1'],[[0,0,0],[0,1.18,0],'PC2'],[[0,0,0],[0,0,1.18],'PC3']].forEach(axis=>{this.line(axis[0],axis[1],'#98a2b3',1.25,[]);const p=this.project(axis[1]);ctx.fillStyle='#475467';ctx.font='700 12px system-ui';ctx.fillText(axis[2],p.x+5,p.y-5);});
    this.points.map(point=>Object.assign({},point,this.project(point.v))).sort((a,b)=>a.z-b.z).forEach(point=>{ctx.beginPath();ctx.arc(point.x,point.y,3.1,0,Math.PI*2);ctx.fillStyle=ENUM_COLORS[point.label-1]+'70';ctx.fill();});
    const centroids=this.centroids.map(point=>Object.assign({},point,this.project(point.v))).sort((a,b)=>a.label-b.label);ctx.save();ctx.strokeStyle='#344054';ctx.lineWidth=1.7;ctx.globalAlpha=.75;ctx.beginPath();centroids.forEach((point,index)=>index?ctx.lineTo(point.x,point.y):ctx.moveTo(point.x,point.y));ctx.stroke();ctx.restore();
    centroids.slice().sort((a,b)=>a.z-b.z).forEach(point=>{ctx.beginPath();ctx.arc(point.x,point.y,10.5,0,Math.PI*2);ctx.fillStyle=ENUM_COLORS[point.label-1];ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=1.8;ctx.stroke();ctx.fillStyle='#fff';ctx.font='800 10px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(String(point.label),point.x,point.y+.4);});
    ctx.textAlign='left';ctx.textBaseline='alphabetic';ctx.font='12px system-ui';const legendWidth=Math.min(620,w-36),startX=(w-legendWidth)/2,step=legendWidth/10;for(let label=1;label<=10;label++){const x=startX+(label-.5)*step;ctx.fillStyle=ENUM_COLORS[label-1];ctx.beginPath();ctx.arc(x,h-24,5,0,Math.PI*2);ctx.fill();ctx.fillStyle='#475467';ctx.fillText(String(label),x+9,h-20);}ctx.fillStyle='#667085';ctx.font='11px system-ui';ctx.fillText(this.points.length+' confirmation states · drag to rotate · double-click to reset',16,22);
  }
}
const enumRunning=new EnumerationPointCloud3D('enum-running-canvas','enum-running-layer','enum-running-stats','running');
const enumFinal=new EnumerationPointCloud3D('enum-final-canvas','enum-final-layer','enum-final-stats','final');
function enumSetCell(){ENUM_CLOUDS.forEach(cloud=>cloud.setCell());}
document.getElementById('enum-geometry-grammar').addEventListener('change',enumSetCell);
document.getElementById('enum-geometry-model').addEventListener('change',enumSetCell);
document.getElementById('enum-geometry-reset').addEventListener('click',()=>enumRunning.resetView());
window.__ENUM_CLOUDS=ENUM_CLOUDS;
"""


def render_report(data: Mapping[str, Any]) -> str:
    cells = data["cells"]
    item_end_sensitivity = data.get("index_item_end_anchor_sensitivity", {})
    sensitivity_analyses = item_end_sensitivity.get("analyses", {})
    sensitivity_complete = item_end_sensitivity.get("status") == "PASS_COMPLETE"
    if item_end_sensitivity and (
        not sensitivity_complete
        or set(sensitivity_analyses) != {"Qwen3-8B", "Gemma4-E4B"}
        or item_end_sensitivity.get("primary_confirmation_replaced") is not False
        or item_end_sensitivity.get("k_reselected") is not False
    ):
        raise ValueError(
            "Index item-end sensitivity does not preserve its two-model discovery-only contract"
        )
    answer_trace_extension = data.get("answer_trace_extension", {})
    extension_cells = {
        f"{cell['prompt_mode']}|{cell['model_label']}": cell
        for cell in answer_trace_extension.get("cells", [])
    }
    extension_complete = answer_trace_extension.get("status") == "PASS_COMPLETE"
    if answer_trace_extension and (
        not extension_complete or set(extension_cells) != set(cells)
    ):
        raise ValueError("Answer/trace extension does not preserve all four cells")
    if extension_complete:
        adapted_hashes: set[str] = set()
        adapted_bullet_models: set[str] = set()
        for key, extension_cell in extension_cells.items():
            mode, model = key.split("|", 1)
            geometry = str(extension_cell.get("relay_geometry", "suffix8"))
            evidence_label = str(
                extension_cell.get(
                    "relay_evidence_label", "original_registered_suffix8"
                )
            )
            if mode == "enumeration_index":
                if (
                    geometry != "suffix8"
                    or evidence_label != "original_registered_suffix8"
                ):
                    raise ValueError(
                        "Index relay must remain the original suffix8 assay"
                    )
            elif mode == "enumeration_bullet" and geometry == "suffix8":
                if evidence_label != "original_registered_suffix8":
                    raise ValueError("Original Bullet suffix8 relay has a false label")
            elif mode == "enumeration_bullet" and geometry == "suffix4":
                if evidence_label != (
                    "post_hoc_task_adapted_bullet_relay_replication"
                ):
                    raise ValueError("Bullet suffix4 relay lost its task-adapted label")
                amendment_hash = str(
                    extension_cell.get("relay_geometry_amendment_sha256", "")
                )
                if len(amendment_hash) != 64:
                    raise ValueError("Bullet suffix4 relay lost its amendment hash")
                original_suffix8 = extension_cell.get("original_suffix8_relay", {})
                if original_suffix8.get("geometry") != "suffix8":
                    raise ValueError("Bullet suffix4 relay hid its suffix8 audit")
                adapted_hashes.add(amendment_hash)
                adapted_bullet_models.add(model)
            else:
                raise ValueError("Answer/trace extension has an unsupported geometry")
        if adapted_bullet_models and adapted_bullet_models != set(MODELS):
            raise ValueError("Bullet suffix4 adaptation must cover both models")
        if len(adapted_hashes) > 1:
            raise ValueError("Bullet suffix4 cells use different amendments")
        if adapted_hashes and str(
            answer_trace_extension.get("relay_geometry_amendment_sha256", "")
        ) not in adapted_hashes:
            raise ValueError("Answer/trace summary amendment hash disagrees")
    full_suite_summary = data.get("full_suite_summary", {})
    full_suite_frames_html = str(data.get("full_suite_frames_html", ""))
    full_suite_claims = data.get("full_suite_claims", {})
    head_mask_audit = data.get("head_mask_scope_audit", {})
    if full_suite_summary and (
        full_suite_summary.get("status") != "PASS"
        or full_suite_summary.get("report_frame_count") != 20
    ):
        raise ValueError("Expected a PASS sealed 20-frame Enumeration suite summary")
    if full_suite_claims and full_suite_claims.get("status") != (
        "PASS_SEALED_FRAME_10_13_EXTRACTION"
    ):
        raise ValueError("Full-suite compact claims were not extracted from sealed frames")
    if full_suite_summary and not full_suite_claims:
        raise ValueError("Sealed 20-frame summary requires its audited compact claims")
    parser_data = data["parser_appendix"]
    behavioral_accuracy = parser_data["behavioral_accuracy"]
    accuracy_by_mode = {
        str(row["prompt_mode"]): row for row in behavioral_accuracy["mode_summaries"]
    }
    index_accuracy = accuracy_by_mode["enumeration_index"]
    bullet_accuracy = accuracy_by_mode["enumeration_bullet"]
    overall_accuracy = behavioral_accuracy["overall"]
    full_gate_count = sum(
        bool(cell["full_commit_to_query"]["confirmation"]["strong_direct_gate_pass"])
        for cell in cells.values()
    )
    continuous_gate_count = sum(
        bool(
            cell["targeted_retrieval"]["continuous"]["confirmation"][
                "strong_interval_gate_pass"
            ]
        )
        for cell in cells.values()
    )
    local_terminal_count = sum(
        bool(
            cell.get("terminal", {})
            .get("local_diagnostic", {})
            .get("confirmation", {})
            .get("local_terminal_token_state_mediation_pass", False)
        )
        for cell in cells.values()
    )
    terminal_necessity_count = sum(
        finite_float(
            estimand(
                cell["terminal"]["baseline"],
                "terminal_token_necessity",
                outcome="expected_count_utility",
            )["ci_low"]
        )
        > 0
        for cell in cells.values()
    )
    terminal_global_sufficiency_count = sum(
        finite_float(
            estimand(
                cell["terminal"]["baseline"],
                "terminal_token_sufficiency",
                outcome="expected_count_utility",
            )["ci_low"]
        )
        > 0
        for cell in cells.values()
    )
    sustained_city_gate_count = sum(
        bool(
            cell["targeted_retrieval"]
            .get("sustained_city_support", {})
            .get("analysis", {})
            .get("strong_interval_gate_pass", False)
        )
        for cell in cells.values()
    )
    greedy_strong_count = sum(
        bool(cell["full_item_greedy"].get("strong_interval_gate_pass", False))
        for cell in cells.values()
    )
    greedy_directional_count = sum(
        bool(cell["full_item_greedy"].get("directional", False))
        for cell in cells.values()
    )
    multihop = data["followup_v3"]["full_item_multihop"]
    multihop_strong_count = sum(
        bool(cell["full_item_multihop"].get("primary_depth4_strong_gate_pass", False))
        for cell in cells.values()
    )
    fresh_carrier_replication = data["followup"]["fresh_bullet_gemma_carrier"][
        "replication"
    ]
    fresh_carrier_strong = bool(
        fresh_carrier_replication.get("strong_interval_gate_pass", False)
    )
    position_alias_count = sum(
        bool(
            data["followup"]["index_targeted_city_support"][model][
                "position_audit"
            ]["all_registered_queries_equal_last_pre_city_predictor"]
        )
        for model in MODELS
    )

    representation_rows = []
    retrieval_rows = []
    carrier_rows = []
    commit_rows = []
    terminal_rows = []
    ncc_rows = []
    state_update_rows = []
    multihop_rows = []
    retrieval_forest_rows: list[tuple[str, Mapping[str, Any], bool]] = []
    carrier_forest_rows: list[tuple[str, Mapping[str, Any], bool]] = []
    commit_forest_rows: list[tuple[str, Mapping[str, Any], bool]] = []
    terminal_forest_rows: list[tuple[str, Mapping[str, Any], bool]] = []
    local_terminal_forest_rows: list[tuple[str, Mapping[str, Any], bool]] = []
    ncc_forest_rows: list[tuple[str, Mapping[str, Any], bool]] = []
    for mode, model in CELL_ORDER:
        cell = cells[f"{mode}|{model}"]
        rep = cell["representation"]
        representation_rows.append(
            (
                esc(cell["mode_label"]),
                esc(model),
                f'L{rep["running_layer"]}',
                num(rep["running_confirmation_logistic_ba"]),
                f'L{rep["final_layer"]}',
                num(rep["final_confirmation_logistic_ba"]),
                "0.100",
            )
        )
        binary = cell["targeted_retrieval"]["binary"]
        b_label, b_kind = baseline_target_status(cell)
        cont = cell["targeted_retrieval"]["continuous"]["confirmation"]
        damage = estimand(
            cont, "selected_damage", outcome="target_city_log_probability"
        )
        specificity = estimand(
            cont,
            "selected_vs_random_specificity",
            outcome="target_city_log_probability",
        )
        d_label, d_kind = diagnostic_status(cont, "strong_interval_gate_pass")
        retrieval_rows.append(
            (
                esc(cell["mode_label"]),
                esc(model),
                str(cell["selected_k"]),
                f"{num(binary['seed_equal_selected_minus_random_failure'], 3, signed=True)} "
                f"[{num(binary['seed_bootstrap_95_lo'])}, {num(binary['seed_bootstrap_95_hi'])}] "
                + pill(b_label, b_kind),
                f"{ci(damage)} / specificity {ci(specificity)} " + pill(d_label, d_kind),
                "—",
                "—",
            )
        )
        short_cell = f"{cell['mode_label']}·{'Qwen' if model == 'Qwen3-8B' else 'Gemma'}"
        retrieval_forest_rows.extend(
            (
                (f"{short_cell} · damage", damage, finite_float(damage["ci_low"]) > 0),
                (
                    f"{short_cell} · specificity",
                    specificity,
                    finite_float(specificity["ci_low"]) > 0,
                ),
            )
        )
        if cell["mode"] == "enumeration_index":
            sustained = cell["targeted_retrieval"]["sustained_city_support"]
            sustained_analysis = sustained["analysis"]
            sustained_damage = estimand(
                sustained_analysis,
                "selected_damage",
                outcome="target_city_log_probability",
            )
            sustained_specificity = estimand(
                sustained_analysis,
                "selected_vs_random_specificity",
                outcome="target_city_log_probability",
            )
            s_label, s_kind = diagnostic_status(
                sustained_analysis, "strong_interval_gate_pass"
            )
            position = sustained["position_audit"]
            retrieval_rows[-1] = retrieval_rows[-1][:-2] + (
                f"damage {ci(sustained_damage, 4)}; specificity {ci(sustained_specificity, 4)} "
                + pill(s_label, s_kind),
                (
                    f"query=first-city predictor: "
                    f"{str(bool(position['all_registered_queries_equal_last_pre_city_predictor'])).lower()}; "
                    f"distance={esc(position['query_to_last_pre_city_distances'])}; "
                    f"lesion positions={esc(position['head_ablation_position_counts'])}"
                ),
            )
        carrier = cell["carrier"]["baseline"]
        deformation = estimand(carrier, "selected_carrier_deformation")
        restoration = estimand(carrier, "clean_carrier_restoration")
        baseline_carrier_status = (
            pill("通过", "support")
            if carrier["targeted_counter_write_strong_gate_pass"]
            else pill("未通过", "null")
        )
        carrier_diag = "—"
        carrier_fresh = "—"
        if "decode_aligned_diagnostic" in cell["carrier"]:
            value = cell["carrier"]["decode_aligned_diagnostic"]["confirmation"]
            diag_def = estimand(value, "selected_carrier_deformation")
            diag_restore = estimand(value, "clean_carrier_restoration")
            label, kind = diagnostic_status(
                value, "targeted_counter_write_strong_gate_pass"
            )
            carrier_diag = (
                f"deformation {ci(diag_def)}; restoration {ci(diag_restore)} "
                + pill(label, kind)
            )
        if "fresh_query_through_carrier_replication" in cell["carrier"]:
            value = cell["carrier"]["fresh_query_through_carrier_replication"]
            fresh_analysis = value["analysis"]
            fresh_deformation = estimand(
                fresh_analysis, "selected_carrier_deformation"
            )
            fresh_restoration = estimand(
                fresh_analysis, "clean_carrier_restoration"
            )
            fresh_specificity = estimand(
                fresh_analysis, "restoration_position_specificity"
            )
            if value["replication"]["strong_interval_gate_pass"]:
                fresh_label, fresh_kind = "fresh 强支持", "restored"
            elif value["replication"]["directional_gate_pass"]:
                fresh_label, fresh_kind = "fresh 方向性", "partial"
            else:
                fresh_label, fresh_kind = "fresh null", "null"
            carrier_fresh = (
                f"deformation {ci(fresh_deformation)}; restoration {ci(fresh_restoration)}; "
                f"matched-position specificity {ci(fresh_specificity)} "
                + pill(fresh_label, fresh_kind)
            )
        carrier_rows.append(
            (
                esc(cell["mode_label"]),
                esc(model),
                f"deformation {ci(deformation)}; restoration {ci(restoration)} {baseline_carrier_status}",
                carrier_diag,
                carrier_fresh,
            )
        )
        carrier_forest_rows.extend(
            (
                (
                    f"{short_cell} · deformation",
                    deformation,
                    finite_float(deformation["ci_low"]) > 0,
                ),
                (
                    f"{short_cell} · restoration",
                    restoration,
                    finite_float(restoration["ci_low"]) > 0,
                ),
            )
        )
        full = cell["full_commit_to_query"]["confirmation"]
        full_self = estimand(full, "full_commit_targeted_attention_vs_self_distance_1")
        full_orth = estimand(
            full, "full_commit_targeted_attention_vs_orthogonal_distance_1"
        )
        full_logodds = estimand(full, "full_commit_city_log_odds_vs_self_distance_1")
        commit_forest_rows.extend(
            (
                (
                    f"{short_cell} · attention vs self",
                    full_self,
                    finite_float(full_self["ci_low"]) > 0,
                ),
                (
                    f"{short_cell} · attention vs orthogonal",
                    full_orth,
                    finite_float(full_orth["ci_low"]) > 0,
                ),
            )
        )
        narrow = cell["narrow_loop"]
        greedy = cell["full_item_greedy"]
        multihop_cell = cell["full_item_multihop"]
        if greedy["strong_interval_gate_pass"]:
            greedy_label, greedy_kind = "行为强支持‡", "restored"
        elif greedy["directional"]:
            greedy_label, greedy_kind = "行为方向性‡", "partial"
        else:
            greedy_label, greedy_kind = "行为 null‡", "null"
        commit_rows.append(
            (
                esc(cell["mode_label"]),
                esc(model),
                pill("false", "null") if not narrow["commit_to_retrieval_pass"] else pill("true", "support"),
                f"vs self {ci(full_self)}; vs orthogonal {ci(full_orth)} "
                + pill("支持*", "restored"),
                ci(full_logodds),
                (
                    f"patched adoption {num(greedy['patched_donor_adoption_rate'])}; "
                    f"self {num(greedy['receiver_self_donor_adoption_rate'])}; "
                    f"native donor {num(greedy['native_donor_adoption_rate'])}; "
                    f"paired {ci(greedy['paired_adoption_effect'])} "
                    + pill(greedy_label, greedy_kind)
                ),
                (
                    f"depth≥1/2/4 = {num(multihop_cell['depth_1']['patched_rate'])}/"
                    f"{num(multihop_cell['depth_2']['patched_rate'])}/"
                    f"{num(multihop_cell['depth_4']['patched_rate'])}; "
                    f"paired depth-4 {ci(multihop_cell['depth_4']['paired_effect'])} "
                    + pill(
                        "多步强支持¶"
                        if multihop_cell["primary_depth4_strong_gate_pass"]
                        else "多步未过¶",
                        "restored"
                        if multihop_cell["primary_depth4_strong_gate_pass"]
                        else "null",
                    )
                ),
            )
        )
        multihop_rows.append(
            (
                esc(cell["mode_label"]),
                esc(model),
                num(multihop_cell["depth_1"]["patched_rate"]),
                num(multihop_cell["depth_2"]["patched_rate"]),
                num(multihop_cell["depth_4"]["patched_rate"]),
                num(multihop_cell["depth_4"]["receiver_self_donor_rate"]),
                num(multihop_cell["depth_4"]["native_donor_rate"]),
                ci(multihop_cell["depth_4"]["paired_effect"]),
                pill(
                    "CI>0"
                    if multihop_cell["primary_depth4_strong_gate_pass"]
                    else "CI overlaps 0",
                    "restored"
                    if multihop_cell["primary_depth4_strong_gate_pass"]
                    else "null",
                ),
            )
        )
        natural_backstep = estimand(narrow, "natural_backstep_repeat")
        component_restoration = estimand(narrow, "count_component_restoration")
        terminal_stop = estimand(narrow, "terminal_state_causes_stop")
        nonterminal_continue = estimand(narrow, "nonterminal_state_causes_continue")
        state_update_rows.append(
            (
                esc(cell["mode_label"]),
                esc(model),
                ci(natural_backstep),
                ci(component_restoration),
                ci(terminal_stop),
                ci(nonterminal_continue),
                pill("update / stop 均未通过", "null"),
            )
        )
        terminal = cell["terminal"]["baseline"]
        necessity = estimand(
            terminal, "terminal_token_necessity", outcome="expected_count_utility"
        )
        sufficiency = estimand(
            terminal, "terminal_token_sufficiency", outcome="expected_count_utility"
        )
        written_state_sufficiency = estimand(
            terminal,
            "token_written_state_sufficiency",
            outcome="expected_count_utility",
        )
        token_requires_state = estimand(
            terminal,
            "token_effect_requires_terminal_state",
            outcome="expected_count_utility",
        )
        local = "—（协议未扩展）"
        if "local_diagnostic" in cell["terminal"]:
            value = cell["terminal"]["local_diagnostic"]["confirmation"]
            local_nec = estimand(
                value, "local_terminal_token_necessity", outcome="expected_count_utility"
            )
            local_restore = estimand(
                value,
                "clean_state_restores_ablated_terminal",
                outcome="expected_count_utility",
            )
            local_occlude = estimand(
                value,
                "ablated_state_occludes_clean_terminal",
                outcome="expected_count_utility",
            )
            local = (
                f"necessity {ci(local_nec)}; restore {ci(local_restore)}; "
                f"occlusion {ci(local_occlude)} "
                + (
                    pill("诊断支持", "diagnostic")
                    if value["local_terminal_token_state_mediation_pass"]
                    else pill("诊断未通过", "null")
                )
            )
            local_terminal_forest_rows.extend(
                (
                    (
                        f"{short_cell} · local necessity",
                        local_nec,
                        finite_float(local_nec["ci_low"]) > 0,
                    ),
                    (
                        f"{short_cell} · clean-state restore",
                        local_restore,
                        finite_float(local_restore["ci_low"]) > 0,
                    ),
                    (
                        f"{short_cell} · state occlusion",
                        local_occlude,
                        finite_float(local_occlude["ci_low"]) > 0,
                    ),
                )
            )
        terminal_rows.append(
            (
                esc(cell["mode_label"]),
                esc(model),
                ci(necessity)
                + pill(
                    "支持" if finite_float(necessity["ci_low"]) > 0 else "未通过",
                    "support" if finite_float(necessity["ci_low"]) > 0 else "null",
                ),
                ci(sufficiency)
                + pill(
                    "支持" if finite_float(sufficiency["ci_low"]) > 0 else "未通过",
                    "support" if finite_float(sufficiency["ci_low"]) > 0 else "null",
                ),
                ci(written_state_sufficiency),
                ci(token_requires_state),
                local,
            )
        )
        terminal_forest_rows.extend(
            (
                (
                    f"{short_cell} · necessity",
                    necessity,
                    finite_float(necessity["ci_low"]) > 0,
                ),
                (
                    f"{short_cell} · global sufficiency",
                    sufficiency,
                    finite_float(sufficiency["ci_low"]) > 0,
                ),
            )
        )
        ncc = cell["ncc"]
        ncc_rows.append(
            (
                esc(cell["mode_label"]),
                esc(model),
                f'L{ncc["selected_layer"]}',
                ci(ncc["primary_estimand"]),
                ci(ncc["specificity_estimand"]),
                pill("无方向特异支持", "null"),
            )
        )
        ncc_forest_rows.extend(
            (
                (f"{short_cell} · selected loss", ncc["primary_estimand"], False),
                (
                    f"{short_cell} · selected−random",
                    ncc["specificity_estimand"],
                    False,
                ),
            )
        )

    sensitivity_cell_rows: list[tuple[str, ...]] = []
    sensitivity_contrast_rows: list[tuple[str, ...]] = []
    sensitivity_forest_rows: list[tuple[str, Mapping[str, Any], bool]] = []
    sensitivity_decision_parts: list[str] = []
    sensitivity_decision_labels = {
        "SUPPORTS_ANCHOR_SENSITIVITY": ("支持 anchor sensitivity", "support"),
        "DESCRIPTIVE_ITEM_END_IMPROVEMENT_UNCERTAIN": (
            "描述性改善，区间不确定",
            "partial",
        ),
        "NO_ITEM_END_IMPROVEMENT": ("不支持 item-end 改善", "null"),
    }
    if sensitivity_complete:
        for model in ("Qwen3-8B", "Gemma4-E4B"):
            analysis = sensitivity_analyses[model]
            decision_text, decision_kind = sensitivity_decision_labels.get(
                str(analysis.get("decision")),
                (str(analysis.get("decision")), "diagnostic"),
            )
            short_model = "Qwen" if model == "Qwen3-8B" else "Gemma"
            sensitivity_decision_parts.append(
                f"{short_model}: {decision_text}"
            )
            for cell_name in (
                "p2bank_at_p2",
                "p2bank_at_p0",
                "p0bank_at_p2",
                "p0bank_at_p0",
            ):
                cell = analysis["cells"][cell_name]
                raw_effect = cell["selected_minus_random_failure"]
                effect = {
                    "mean_effect": raw_effect["estimate"],
                    "ci_low": raw_effect["ci95"][0],
                    "ci_high": raw_effect["ci95"][1],
                }
                sensitivity_cell_rows.append(
                    (
                        esc(short_model),
                        str(analysis["fixed_k"]),
                        esc(cell_name),
                        esc(cell["bank_selection_anchor_role"]),
                        esc(cell["intervention_start_anchor_role"]),
                        num(cell["selected_failure_rate"], 3),
                        num(cell["registered_random_failure_rate"], 3),
                        ci(effect, 3),
                    )
                )
                sensitivity_forest_rows.append(
                    (
                        f"{short_model} · {cell_name}",
                        effect,
                        finite_float(effect["ci_low"]) > 0,
                    )
                )
            for contrast_name in (
                "overall_item_end_minus_primary",
                "site_effect_for_p2_bank",
                "site_effect_for_p0_bank",
                "bank_effect_at_p0",
                "bank_effect_at_p2",
            ):
                contrast = analysis["contrasts"][contrast_name]
                contrast_effect = {
                    "mean_effect": contrast["estimate"],
                    "ci_low": contrast["ci95"][0],
                    "ci_high": contrast["ci95"][1],
                }
                sensitivity_contrast_rows.append(
                    (
                        esc(short_model),
                        esc(contrast_name),
                        f"{esc(contrast['left'])} − {esc(contrast['right'])}",
                        ci(contrast_effect, 3),
                        (
                            pill(decision_text, decision_kind)
                            if contrast_name == "overall_item_end_minus_primary"
                            else "—"
                        ),
                    )
                )
            primary = analysis["contrasts"]["overall_item_end_minus_primary"]
            primary_effect = {
                "mean_effect": primary["estimate"],
                "ci_low": primary["ci95"][0],
                "ci_high": primary["ci95"][1],
            }
            sensitivity_forest_rows.append(
                (
                    f"{short_model} · item-end−primary contrast",
                    primary_effect,
                    finite_float(primary_effect["ci_low"]) > 0,
                )
            )
    sensitivity_decision_summary = "；".join(sensitivity_decision_parts)

    retrieval_table = table(
        (
            "语法",
            "模型",
            "冻结 K",
            "原 binary failure contrast",
            "原 query-local 连续 log P†",
            "query-through-city-prefix log P‡",
            "位置审计‡",
        ),
        retrieval_rows,
        cls="wide",
    )
    sensitivity_cell_table = (
        table(
            (
                "模型",
                "冻结 K",
                "2×2 cell",
                "bank 选择 anchor",
                "lesion 起始 anchor",
                "selected failure",
                "random failure",
                "selected−random [95% CI]",
            ),
            sensitivity_cell_rows,
            cls="wide",
        )
        if sensitivity_complete
        else ""
    )
    sensitivity_contrast_table = (
        table(
            ("模型", "冻结 contrast", "left − right", "效应 [95% CI]", "主判定"),
            sensitivity_contrast_rows,
            cls="wide",
        )
        if sensitivity_complete
        else ""
    )
    representation_table = table(
        ("语法", "模型", "Running 层", "Running BA", "Final 层", "Final BA", "Chance"),
        representation_rows,
    )
    carrier_table = table(
        (
            "语法",
            "模型",
            "原 query-local carrier gate",
            "旧 split decode-aligned 诊断†",
            "fresh query-through-carrier 复现§",
        ),
        carrier_rows,
        cls="wide",
    )
    commit_table = table(
        (
            "语法",
            "模型",
            "原 count-subspace loop",
            "full commit→query attention*",
            "city log-odds*",
            "full-item-span greedy adoption‡",
            "连续 donor path¶",
        ),
        commit_rows,
        cls="wide",
    )
    greedy_direction_rows = []
    direction_values = {
        (
            str(row["prompt_mode"]),
            str(row["model_label"]),
            str(row["direction"]),
        ): row
        for row in data["followup"]["full_item_greedy"]["direction_summaries"]
    }
    for mode, model in CELL_ORDER:
        for direction in ("forward_skip", "backward_rewind"):
            row = direction_values[(mode, model, direction)]
            greedy_direction_rows.append(
                (
                    esc(MODE_LABEL[mode]),
                    esc(model),
                    esc("5→6 forward" if direction == "forward_skip" else "7→6 backward"),
                    num(row["patched_donor_adoption_rate"]),
                    num(row["receiver_self_donor_adoption_rate"]),
                    num(row["native_donor_adoption_rate"]),
                    ci(row["paired_adoption_effect"]),
                    pill(
                        "CI>0" if row["positive_95pct_ci"] else "CI overlaps 0",
                        "restored" if row["positive_95pct_ci"] else "null",
                    ),
                )
            )
    greedy_direction_table = table(
        (
            "语法",
            "模型",
            "方向",
            "patched donor adoption",
            "receiver-self adoption",
            "native-donor control",
            "paired effect",
            "方向 gate",
        ),
        greedy_direction_rows,
        cls="wide",
    )
    multihop_direction_rows = []
    for row in multihop["direction_summaries"]:
        depth4 = row["depth_4"]
        multihop_direction_rows.append(
            (
                esc(MODE_LABEL[str(row["prompt_mode"])]),
                esc(row["model_label"]),
                esc(
                    "5→6 forward"
                    if row["direction"] == "forward_skip"
                    else "7→6 backward"
                ),
                num(row["depth_1"]["patched_rate"]),
                num(row["depth_2"]["patched_rate"]),
                num(depth4["patched_rate"]),
                num(depth4["native_donor_rate"]),
                ci(depth4["paired_effect"]),
                pill(
                    "CI>0" if depth4["positive_95pct_ci"] else "CI touches 0",
                    "restored" if depth4["positive_95pct_ci"] else "partial",
                ),
            )
        )
    multihop_direction_table = table(
        (
            "语法",
            "模型",
            "方向",
            "patched depth≥1",
            "depth≥2",
            "depth≥4",
            "native donor depth≥4",
            "paired depth-4 effect",
            "方向判定",
        ),
        multihop_direction_rows,
        cls="wide",
    )
    multihop_table = table(
        (
            "语法",
            "模型",
            "patched depth≥1",
            "patched depth≥2",
            "patched depth≥4",
            "receiver-self donor depth≥4",
            "native-donor depth≥4",
            "paired depth-4 effect",
            "cell gate",
        ),
        multihop_rows,
        cls="wide",
    )
    state_update_table = table(
        (
            "语法",
            "模型",
            "backstep repeat",
            "count-component restore",
            "terminal state→stop",
            "nonterminal state→continue",
            "原低维 gate",
        ),
        state_update_rows,
        cls="wide",
    )
    terminal_table = table(
        (
            "语法",
            "模型",
            "完整 trace 中 necessity",
            "global one-item sufficiency",
            "token-written state sufficiency",
            "token effect requires state",
            "局部 clean-context mediation†",
        ),
        terminal_rows,
        cls="wide",
    )
    suite_cell_label = {
        ("enumeration_index", "Qwen3-8B"): "Index enumeration · Qwen3-8B",
        ("enumeration_index", "Gemma4-E4B"): "Index enumeration · Gemma4-E4B",
        ("enumeration_bullet", "Qwen3-8B"): "Bullet enumeration · Qwen3-8B",
        ("enumeration_bullet", "Gemma4-E4B"): "Bullet enumeration · Gemma4-E4B",
    }
    answer_source_rows: list[tuple[str, ...]] = []
    direct_count_margin_rows: list[tuple[str, ...]] = []
    direct_count_margin_forest_rows: list[tuple[str, Mapping[str, Any], bool]] = []
    if full_suite_claims:
        for mode, model in CELL_ORDER:
            source_label = suite_cell_label[(mode, model)]
            short_cell = f"{MODE_LABEL[mode]}·{'Q' if model == 'Qwen3-8B' else 'G'}"
            source = full_suite_claims["answer_source"][source_label]
            prompt_blank = source["prompt_records_blank"]
            trace_blank = source["trace_all_blank"]
            answer_source_rows.append(
                (
                    esc(MODE_LABEL[mode]),
                    esc(model),
                    num(prompt_blank["mean_delta_exact_count"]),
                    num(trace_blank["mean_delta_exact_count"]),
                    num(
                        prompt_blank[
                            "mean_delta_gold_first_answer_token_log_probability"
                        ]
                    ),
                    num(
                        trace_blank[
                            "mean_delta_gold_first_answer_token_log_probability"
                        ]
                    ),
                )
            )
            margin = full_suite_claims["direct_count_output_margin"][source_label]
            selected = {
                "mean_effect": margin["selected_margin_loss.mean_effect"],
                "ci_low": margin["selected_margin_loss.ci_low"],
                "ci_high": margin["selected_margin_loss.ci_high"],
            }
            specificity = {
                "mean_effect": margin[
                    "selected_vs_random_specificity.mean_effect"
                ],
                "ci_low": margin["selected_vs_random_specificity.ci_low"],
                "ci_high": margin["selected_vs_random_specificity.ci_high"],
            }
            status = str(margin["effect_status"])
            status_class = (
                "support"
                if status.startswith("INTERVAL_CONFIRMED")
                else "diagnostic"
                if status.startswith("VALID_READOUT_DIRECTIONAL")
                else "null"
            )
            direct_count_margin_rows.append(
                (
                    esc(MODE_LABEL[mode]),
                    esc(model),
                    num(margin["clean_accuracy"]),
                    num(margin["clean_mean_margin"]),
                    ci(selected),
                    ci(specificity),
                    pill(status, status_class),
                )
            )
            direct_count_margin_forest_rows.extend(
                (
                    (
                        f"{short_cell} · selected margin loss",
                        selected,
                        finite_float(selected["ci_low"]) > 0,
                    ),
                    (
                        f"{short_cell} · selected−random",
                        specificity,
                        finite_float(specificity["ci_low"]) > 0,
                    ),
                )
            )
    answer_source_table = table(
        (
            "语法",
            "模型",
            "prompt-record blank · Δ exact",
            "full-trace blank · Δ exact",
            "prompt blank · Δ first-token log P",
            "trace blank · Δ first-token log P",
        ),
        answer_source_rows,
        cls="wide",
    )
    direct_count_margin_table = table(
        (
            "语法",
            "模型",
            "clean accuracy",
            "clean margin",
            "selected margin loss",
            "selected−random specificity",
            "Frame 13 status",
        ),
        direct_count_margin_rows,
        cls="wide",
    )
    answer_extension_rows: list[tuple[str, ...]] = []
    relay_extension_rows: list[tuple[str, ...]] = []
    relay_extension_forest_rows: list[
        tuple[str, Mapping[str, Any], bool]
    ] = []
    relay_seed_accounting_summaries: list[str] = []
    extension_partial_mediation_count = 0
    extension_query_mediation_count = 0
    extension_relay_estimable_count = 0
    extension_task_adapted_relay_count = 0
    extension_original_suffix8_na_cells: list[str] = []
    extension_relay_geometry_labels: list[str] = []
    if extension_complete:
        for mode, model in CELL_ORDER:
            extension_cell = extension_cells[f"{mode}|{model}"]
            layer_effects = sorted(
                extension_cell["answer_layer_effects"],
                key=lambda row: int(row["layer"]),
            )
            final_layer = layer_effects[-1]
            layer_grid = ", ".join(f"L{int(row['layer'])}" for row in layer_effects)
            onset = extension_cell.get("answer_descriptive_onset")
            answer_extension_rows.append(
                (
                    esc(MODE_LABEL[mode]),
                    esc(model),
                    f"<code>{esc(layer_grid)}</code>",
                    f"{esc(extension_cell['answer_registered_pairs'])} / "
                    f"{esc(extension_cell['answer_seed_clusters'])}",
                    "NA" if onset is None else f"L{esc(onset)}",
                    f"L{esc(final_layer['layer'])}: "
                    f"{num(final_layer['full_donor_adoption'])} "
                    f"[{num(final_layer['full_donor_adoption_ci95_low'])}, "
                    f"{num(final_layer['full_donor_adoption_ci95_high'])}]",
                    num(final_layer["registered_numeric_valid"]),
                )
            )
            natural = extension_cell["terminal_patch"]
            suffix = extension_cell["suffix_mediation"]
            query = extension_cell["query_mediation"]
            residual_ratio = extension_cell["suffix_residual_ratio"]
            relay_geometry = str(extension_cell.get("relay_geometry", "suffix8"))
            relay_evidence_label = str(
                extension_cell.get(
                    "relay_evidence_label", "original_registered_suffix8"
                )
            )
            relay_is_task_adapted = relay_evidence_label == (
                "post_hoc_task_adapted_bullet_relay_replication"
            )
            extension_task_adapted_relay_count += int(relay_is_task_adapted)
            extension_relay_geometry_labels.append(
                f"{esc(MODE_LABEL[mode])}·{esc(model)}={esc(relay_geometry)}"
            )
            relay_planned = int(extension_cell["relay_planned_seed_count"])
            relay_eligible = int(extension_cell["relay_eligible_seed_count"])
            relay_estimable = bool(
                extension_cell.get("relay_estimable", relay_eligible > 0)
            )
            original_suffix8 = extension_cell.get("original_suffix8_relay", {})
            original_suffix8_estimable = bool(
                original_suffix8.get(
                    "estimable",
                    relay_estimable
                    if relay_evidence_label == "original_registered_suffix8"
                    else True,
                )
            )
            if not original_suffix8_estimable:
                extension_original_suffix8_na_cells.append(
                    f"{MODE_LABEL[mode]}·{model}"
                )
            relay_full_na = [
                int(seed)
                for seed in extension_cell[
                    "relay_geometry_not_applicable_full_seeds"
                ]
            ]
            relay_seed_accounting = f"{relay_eligible}/{relay_planned}"
            if relay_full_na:
                relay_seed_accounting += "; full-NA=" + ", ".join(
                    str(seed) for seed in relay_full_na
                )
            else:
                relay_seed_accounting += "; full-NA=无"
            relay_seed_accounting_summaries.append(
                f"{esc(MODE_LABEL[mode])}·{esc(model)} "
                f"{esc(relay_geometry)} {esc(relay_seed_accounting)}"
                + (f"（{esc(relay_geometry)} 不可估计）" if not relay_estimable else "")
            )
            partial_pass = relay_estimable and bool(
                extension_cell["partial_mediation_pass"]
            )
            query_pass = relay_estimable and bool(query["pass"])
            extension_relay_estimable_count += int(relay_estimable)
            extension_partial_mediation_count += int(partial_pass)
            extension_query_mediation_count += int(query_pass)
            if relay_estimable:
                explained_estimate = 1.0 - finite_float(
                    residual_ratio["estimate"]
                )
                explained_low = 1.0 - finite_float(residual_ratio["high"])
                explained_high = 1.0 - finite_float(residual_ratio["low"])
                relay_extension_rows.append(
                    (
                        esc(MODE_LABEL[mode]),
                        esc(model),
                        f"<code>{esc(relay_geometry)}</code><br>{esc(relay_evidence_label)}",
                        esc(relay_seed_accounting),
                        ci(natural),
                        ci(suffix),
                        ci(query),
                        f"{pct(explained_estimate)} "
                        f"[{pct(explained_low)}, {pct(explained_high)}]",
                        f"{num(residual_ratio['estimate'])} "
                        f"[{num(residual_ratio['low'])}, {num(residual_ratio['high'])}]",
                        pill(
                            "partial mediation supported"
                            if partial_pass
                            else "primary gate not met",
                            "support" if partial_pass else "null",
                        ),
                    )
                )
            else:
                reason = extension_cell.get(
                    "relay_not_estimable_reason",
                    f"all preregistered trace items are shorter than {relay_geometry}",
                )
                relay_extension_rows.append(
                    (
                        esc(MODE_LABEL[mode]),
                        esc(model),
                        f"<code>{esc(relay_geometry)}</code><br>{esc(relay_evidence_label)}",
                        esc(relay_seed_accounting),
                        "NA",
                        "NA",
                        "NA",
                        "NA",
                        "NA",
                        pill(f"不可估计：{reason}", "diagnostic"),
                    )
                )
            short_cell = (
                f"{MODE_LABEL[mode]}·{'Q' if model == 'Qwen3-8B' else 'G'}"
                f"·{relay_geometry}"
            )
            if relay_estimable:
                for label, gate in (
                    ("natural patch damage", natural),
                    ("suffix-specific mediation", suffix),
                    ("answer-query mediation", query),
                ):
                    effect = {
                        "mean_effect": gate["estimate"],
                        "ci_low": gate["low"],
                        "ci_high": gate["high"],
                    }
                    relay_extension_forest_rows.append(
                        (
                            f"{short_cell} · {label}",
                            effect,
                            finite_float(gate["low"]) > 0,
                        )
                    )
    answer_extension_table = table(
        (
            "语法",
            "模型",
            "预冻结八层网格",
            "directed pairs / seed clusters",
            "descriptive onset",
            "最终采样层 donor adoption [95% CI]",
            "最终层可解析率",
        ),
        answer_extension_rows,
        cls="wide",
    )
    relay_extension_table = table(
        (
            "语法",
            "模型",
            "relay geometry / evidence label",
            "relay geometry-eligible / preregistered seeds",
            "natural patch damage",
            "suffix-specific mediation",
            "answer-query-specific mediation",
            "suffix explained fraction",
            "residual / natural ratio",
            "主结论",
        ),
        relay_extension_rows,
        cls="wide",
    )
    relay_seed_accounting_text = "；".join(relay_seed_accounting_summaries)
    relay_geometry_cell_text = "；".join(extension_relay_geometry_labels)
    if extension_task_adapted_relay_count:
        relay_geometry_design_text = (
            "Index 两格保留原注册 suffix8；Bullet 两格统一使用在 suffix4 "
            "intervention outcomes 产生前冻结的 post-hoc task-adapted replication。"
            "两种 geometry 的 source/relay arms、模型特定层、cohort、pair rule、"
            "sequence-margin estimands、bootstrap 与 gates 均不变"
        )
    else:
        relay_geometry_design_text = "四格均使用原注册 suffix8"
    original_suffix8_audit_text = (
        "原 suffix8 审计仍保留，其中不可估计格为："
        + "、".join(extension_original_suffix8_na_cells)
        if extension_original_suffix8_na_cells
        else "原 suffix8 审计仍保留，四格均有数值支持"
    )
    ncc_table = table(
        ("语法", "模型", "层", "Selected margin loss", "Selected−random specificity", "判定"),
        ncc_rows,
    )
    native_main_crosswalk_table = table(
        (
            "Native-thinking 槽位",
            "Enumeration 对应实验",
            "设计对齐等级",
            "Enumeration 结果与边界",
        ),
        (
            (
                "§1 行为基线",
                "Frame 01 · strict ordered enumeration",
                pill("同分析角色", "support"),
                f"原始 strict exact：Index {pct(index_accuracy['accuracy'])}；Bullet {pct(bullet_accuracy['accuracy'])}；replacement 不冒充准确率。",
            ),
            (
                "§2 Representation",
                "Frame 02 + discovery-fit PCA3",
                pill("同测量合同", "support"),
                "running item_end 与 final answer_query_v3 均 discovery-only 选层、confirmation-only 读出；probe/PC 不作 causal claim。",
            ),
            (
                "§3 State scope / formation",
                "Frames 04/12/14/15/16 + full-item follow-up",
                pill("任务适配对应", "diagnostic"),
                "完整 content-bound item state 的 direct/behavior/multihop 为 4/4；低维 loop、NCC、update/stop 为 0/4，不能复制 Native 的纯低维解释。",
            ),
            (
                "§4 Targeted retrieval",
                "Frame 05 + sustained city-prefix + frozen 2×2 item-end sensitivity",
                pill("同 bank-level 因果族", "support"),
                (
                    f"行为级 mask 是持续 decode；Bullet 原 binary 强，Index 的同一冻结 bank 在更宽 city-prefix window 为 {sustained_city_gate_count}/2。"
                    + (
                        f" 事后动机、结果前冻结的 2×2 判定为：{sensitivity_decision_summary}。"
                        if sensitivity_complete
                        else " 2×2 item-end sensitivity 尚未完成，不提前改写 primary。"
                    )
                ),
            ),
            (
                "§5.1 Query→carrier",
                "Frame 07 + fresh Bullet-Gemma",
                pill("同局部 transport estimand", "support"),
                "原 V6 是 query-local 单点；Bullet-Gemma fresh 是 query-through-carrier。两者不可混成一个窗口。",
            ),
            (
                "§5.2 Carrier→commit rescue",
                "Frame 08",
                pill("同受控 rescue 逻辑", "support"),
                "clean carrier 在同一 selected-bank damage context 中恢复 later commit；原三格通过，Bullet-Gemma 由 fresh wider-window 补强。",
            ),
            (
                "§5.3 State→next item",
                "Frames 09/17 + V2 greedy + V3 multihop",
                pill("同闭环角色", "restored"),
                f"teacher-forced direct {full_gate_count}/4；first-city {greedy_strong_count}/4 strong；depth-4 {multihop_strong_count}/4。",
            ),
            (
                "§6.1 Answer source necessity",
                "Frame 10 · answer-token source ablation",
                pill("同源区消融角色", "support"),
                "trace、prompt-record 与 length-matched ordinary blank 分开；完整四格数值在 20-frame Appendix 原样保留。",
            ),
            (
                "§6.2 Terminal bridge",
                "Frame 11 + Bullet local diagnostic",
                pill("同局部 bridge 逻辑", "support"),
                f"necessity {terminal_necessity_count}/4；global sufficiency 是 Index 2/2、Bullet 0/2；Bullet clean-context local mediation {local_terminal_count}/2。",
            ),
            (
                "§6.3 Answer-state execution",
                (
                    "Exact answer_query_v3 full-state layer sweep + Frame 13"
                    if extension_complete
                    else "Frame 13 + global restore arms"
                ),
                (
                    pill("同 site / layer grid / estimand", "support")
                    if extension_complete
                    else pill("功能对应，非同一 assay", "partial")
                ),
                (
                    "四格均使用 Native 的八层网格、self/full arms、greedy-16 donor-count adoption；Frame 13 作为 bank→decoder margin 的补充边界。"
                    if extension_complete
                    else "检验最终 count sequence margin 与 terminal restore；不是 Native 的逐层 donor-count answer-query full-state adoption sweep。"
                ),
            ),
            (
                "§6.4 Same-trial terminal relay",
                (
                    "Exact source-patch × natural/suffix/query-reset factorial + Frame 11"
                    if extension_complete
                    else "Frame 11 mediation arms + local bridge"
                ),
                (
                    pill("同 2×3 estimator / same-trial role", "support")
                    if extension_complete
                    else pill("功能对应，非逐 trial 同构", "partial")
                ),
                (
                    "四格均运行 natural relay、clean suffix reset 与 answer-query-only reset；只允许 partial mediation，不声称 complete mediation。"
                    if extension_complete
                    else "证明 token effect 与 terminal state 的相互依赖；没有把 Native 的 earlier-terminal→suffix→answer-query reset factorial 原样移植。"
                ),
            ),
        ),
        cls="wide mirror-table",
    )
    full_suite_crosswalk_rows = (
        ("01", "behavior_and_parser_baseline", "§1", "define_population", "同分析角色"),
        ("02", "layerwise_representation", "§2", "localization", "同测量合同"),
        ("03", "paired_causal_estimands", "Appendix", "estimand_contract", "设计合同"),
        ("04", "trace_scope_layer_sweep", "§3", "minimal_state_scope", "任务适配对应"),
        ("05", "targeted_retrieval_bank", "§4", "necessity", "同 bank-level 因果族"),
        ("06", "seed_equal_sampling_contract", "Appendix", "independence", "设计合同"),
        ("07", "targeted_query_to_carrier", "§5", "local_transport", "同局部边"),
        ("08", "carrier_to_commit_restore", "§5", "state_execution", "同 rescue 逻辑"),
        ("09", "progress_state_to_successor", "§5", "next_item_control", "任务适配对应"),
        ("10", "answer_token_source_ablation", "§6", "source_necessity", "同源区消融角色"),
        ("11", "terminal_state_bridge", "§6", "terminal_transport", "同局部 bridge"),
        ("12", "timing_stratified_ncc", "§3", "geometry_direction", "诊断扩展"),
        ("13", "direct_count_output_margin", "§6", "decoder_effect", "任务适配对应"),
        ("14", "count_geometry_ncc", "§3", "count_specificity", "诊断扩展"),
        ("15", "layer_timing_diagnostic", "§3", "causal_timing", "discovery 诊断"),
        ("16", "visible_progress_positive_control", "§3 / Appx J", "assay_calibration", "同 positive control 角色"),
        ("17", "commit_to_query_patch", "§7 / Appx J", "loop_edge", "同闭环边角色"),
        ("18", "single_seed_walkthrough", "Appendix", "strict_sufficiency_sanity", "审计示例"),
        ("19", "format_specific_source_scrub_restore", "§8", "state_sufficiency", "任务适配对应"),
        ("20", "scrub_coverage_and_cross_mode_audit", "§8", "confound_audit", "审计合同"),
    )
    full_suite_crosswalk_table = table(
        ("Frame", "Enumeration 实验", "Native 槽位", "Claim role", "对齐类型"),
        (
            (
                frame,
                f"<code>{esc(name)}</code>",
                native_slot,
                f"<code>{esc(role)}</code>",
                pill(
                    alignment,
                    "support"
                    if alignment.startswith("同")
                    else "diagnostic"
                    if "诊断" in alignment or "适配" in alignment
                    else "partial",
                ),
            )
            for frame, name, native_slot, role, alignment in full_suite_crosswalk_rows
        ),
        cls="wide",
    )
    head_mask_rows: list[tuple[str, ...]] = []
    if head_mask_audit:
        for model, value in head_mask_audit.get(
            "native_behavior_persistent_decode", {}
        ).items():
            head_mask_rows.append(
                (
                    "Native-thinking behavior",
                    esc(model),
                    "latest registered anchor → all cached decode forwards",
                    f"<code>{esc(value['branch_policy'])}</code>",
                    f"<code>decode_head_ablation_steps={esc(value['decode_head_ablation_steps'])}</code>",
                    f"{esc(value['completed_shards'])} shards · <code>{esc(value['manifest_sha256'])}</code>",
                )
            )
        for cell_key, value in head_mask_audit.get(
            "behavior_persistent_decode", {}
        ).items():
            head_mask_rows.append(
                (
                    "Enumeration behavior",
                    esc(cell_key),
                    "registered query → all cached decode forwards",
                    f"<code>{esc(value['branch_policy'])}</code>",
                    f"<code>decode_head_ablation_steps={esc(value['decode_head_ablation_steps'])}</code>",
                    f"{esc(value['completed_shards'])} shards · <code>{esc(value['manifest_sha256'])}</code>",
                )
            )
        head_mask_rows.extend(
            (
                (
                    "Enumeration original carrier",
                    "all 4 cells",
                    "registered query only",
                    "<code>query_local</code>",
                    "one teacher-forced position",
                    "10 confirmation seeds/cell",
                ),
                (
                    "Enumeration Index likelihood follow-up",
                    "Qwen + Gemma",
                    "registered query → target-city prefix",
                    "<code>registered_query_through_city_prefix</code>",
                    "1–6 autoregressive predictor positions",
                    "50 shards/model",
                ),
                (
                    "Enumeration Bullet-Gemma fresh carrier",
                    "Gemma4-E4B · K2",
                    "registered query → final carrier token",
                    "<code>query_through_carrier</code>",
                    "8–9 positions/request",
                    "10 fresh shards",
                ),
            )
        )
    head_mask_scope_table = table(
        ("Assay", "Cell/model", "时间支持", "Manifest scope/policy", "长度", "证据"),
        head_mask_rows,
        cls="wide",
    )
    behavioral_accuracy_table = table(
        (
            "语法",
            "模型",
            "Discovery raw strict exact",
            "Confirmation raw strict exact",
            "合并 raw accuracy",
            "补 seed 后 cohort eligibility*",
        ),
        (
            (
                esc(MODE_LABEL[str(row["prompt_mode"])]),
                esc(row["model_label"]),
                (
                    f"{esc(row['discovery_pass_count'])}/{esc(row['discovery_total_count'])} "
                    f"({pct(row['discovery_accuracy'])})"
                ),
                (
                    f"{esc(row['confirmation_pass_count'])}/{esc(row['confirmation_total_count'])} "
                    f"({pct(row['confirmation_accuracy'])})"
                ),
                (
                    f"{esc(row['pooled_pass_count'])}/{esc(row['pooled_total_count'])} "
                    f"({pct(row['pooled_accuracy'])})"
                ),
                (
                    f"{esc(row['final_fixed_quota_eligible_count'])}/"
                    f"{esc(row['final_fixed_quota_total_count'])} (100.0%)*"
                ),
            )
            for row in behavioral_accuracy["cell_summaries"]
        ),
        cls="wide",
    )
    strict_grammar = parser_data["strict_grammar"]
    parser_grammar_table = table(
        ("对象", "严格接受式", "附加约束"),
        (
            (
                "Index item",
                f"<code>{esc(strict_grammar['index_record_pattern'])}</code>",
                "marker 必须是 <code>1.</code> 起始并连续到 M；<code>1)</code> 不属于 V6 strict final grammar。",
            ),
            (
                "Bullet item",
                f"<code>{esc(strict_grammar['bullet_record_pattern'])}</code>",
                "只接受 ASCII hyphen <code>-</code>；<code>•</code>、<code>*</code> 或数字 marker 均不属于该单元的 strict grammar。",
            ),
            (
                "Final total",
                f"<code>{esc(strict_grammar['total_line'])}</code>",
                "必须是最后一个非空行；允许列出的四种 terminal token，不能在 item 或 Total 前后加入 prose。",
            ),
        ),
        cls="wide",
    )
    parser_gate_table = table(
        ("Gate field", "必须满足的条件"),
        (
            (f"<code>{esc(name)}</code>", esc(description))
            for name, description in PARSER_FORMAL_GATES
        ),
        cls="wide",
    )
    parser_site_table = table(
        ("Registered site", "字符边界与分析语义"),
        (
            (f"<code>{esc(name)}</code>", esc(description))
            for name, description in PARSER_REGISTERED_SITES
        ),
        cls="wide",
    )
    parser_cohort_table = table(
        (
            "语法",
            "模型",
            "split",
            "最终固定 cells",
            "原始 strict pass",
            "原始 accuracy",
            "原 strict failures",
            "replacement",
            "最终 eligible",
            "失败原因计数",
            "reserve failures",
        ),
        (
            (
                esc(MODE_LABEL[str(row["prompt_mode"])]),
                esc(row["model_label"]),
                esc(row["split"]),
                esc(row["selected_cell_count"]),
                esc(row["original_strict_pass_count"]),
                pct(row["original_strict_accuracy"]),
                esc(row["original_strict_failure_count"]),
                esc(row["replacement_count"]),
                esc(row["final_fixed_quota_eligible_count"]),
                (
                    "<code>none</code>"
                    if not row["reason_counts"]
                    else "<br>".join(
                        f"<code>{esc(reason)}</code>={esc(count)}"
                        for reason, count in row["reason_counts"].items()
                    )
                ),
                esc(row["failed_reserve_attempt_count"]),
            )
            for row in parser_data["cohort_summaries"]
        ),
        cls="wide",
    )
    parser_failure_table = table(
        (
            "语法",
            "模型",
            "split",
            "slot seed",
            "gold N",
            "failure flags",
            "replacement seed",
            "candidate rank",
        ),
        (
            (
                esc(MODE_LABEL[str(row["prompt_mode"])]),
                esc(row["model_label"]),
                esc(row["split"]),
                esc(row["analysis_slot_seed"]),
                esc(row["gold_count"]),
                "<br>".join(
                    f"<code>{esc(reason)}</code>"
                    for reason in row["failure_reasons"]
                ),
                esc(row["replacement_seed"]),
                esc(row["replacement_candidate_rank"]),
            )
            for row in parser_data["original_failure_ledger"]
        ),
        cls="wide",
    )
    parser_failed_reserve_table = table(
        (
            "语法",
            "模型",
            "split",
            "candidate seed",
            "gold N",
            "rank",
            "failure flags",
            "runtime failure",
            "intervention outcomes read",
        ),
        (
            (
                esc(MODE_LABEL[str(row["prompt_mode"])]),
                esc(row["model_label"]),
                esc(row["split"]),
                esc(row["seed"]),
                esc(row["gold_count"]),
                esc(row["candidate_rank"]),
                "<br>".join(
                    f"<code>{esc(reason)}</code>"
                    for reason in row["failure_reasons"]
                ),
                esc(str(bool(row["runtime_failure"])).lower()),
                esc(str(bool(row["intervention_outcomes_read"])).lower()),
            )
            for row in parser_data["failed_reserve_attempt_ledger"]
        ),
        cls="wide",
    )
    replacement_policy_table = table(
        ("Replacement audit field", "Value"),
        (
            (f"<code>{esc(name)}</code>", f"<code>{esc(value)}</code>")
            for name, value in parser_data["replacement_policy_audit"].items()
        ),
    )
    multihop_endpoint = parser_data["multihop_endpoint"]
    parser_multihop_example_table = table(
        (
            "语法",
            "模型",
            "方向",
            "seed",
            "condition",
            "observed known-city ordinals",
            "donor depth",
            "receiver depth",
            "truncated",
        ),
        (
            (
                esc(MODE_LABEL[str(row["prompt_mode"])]),
                esc(row["model_label"]),
                esc(row["direction"]),
                esc(row["seed"]),
                esc(row["condition"]),
                f"<code>{esc(json.dumps(row['generated_ordinals'], separators=(',', ':')))}</code>",
                esc(row["donor_prefix_depth"]),
                esc(row["receiver_prefix_depth"]),
                esc(str(bool(row["generation_truncated"])).lower()),
            )
            for row in multihop_endpoint["fixed_lowest_seed_examples"]
        ),
        cls="wide",
    )
    parser_multihop_taxonomy_table = table(
        ("Multihop parser audit field", "Count"),
        (
            (f"<code>{esc(name)}</code>", esc(count))
            for name, count in multihop_endpoint["failure_taxonomy"].items()
        ),
    )
    parser_source_table = table(
        ("Parser component", "SHA-256"),
        (
            (f"<code>{esc(name)}</code>", f"<code>{esc(digest)}</code>")
            for name, digest in parser_data["source_sha256"].items()
        ),
        cls="hash-table",
    )

    position_details = "; ".join(
        (
            f"{model}: distance="
            f"{data['followup']['index_targeted_city_support'][model]['position_audit']['query_to_last_pre_city_distances']}, "
            f"lesion-position-count="
            f"{data['followup']['index_targeted_city_support'][model]['position_audit']['head_ablation_position_counts']}"
        )
        for model in MODELS
    )
    greedy_summary = (
        f"{greedy_strong_count}/4 strong CI gates，"
        f"{greedy_directional_count}/4 directionally positive"
    )
    fresh_carrier_summary = (
        "strong gate 通过"
        if fresh_carrier_strong
        else (
            "仅方向性"
            if fresh_carrier_replication.get("directional_gate_pass")
            else "仍为 null"
        )
    )
    fresh_lock = data["followup"]["fresh_bullet_gemma_carrier"]["cohort_lock"]

    running_curve_figure = representation_layer_curves_svg(
        data["native_aligned_representation"]["running_candidates"],
        endpoint_label="Running occurrence at item_end",
    )
    final_curve_figure = representation_layer_curves_svg(
        data["native_aligned_representation"]["final_candidates"],
        endpoint_label="Final count at answer_query_v3",
    )
    accuracy_figure = behavioral_accuracy_svg(behavioral_accuracy["cell_summaries"])
    compact_cell_labels = [
        f"{MODE_LABEL[mode]}·{'Q' if model == 'Qwen3-8B' else 'G'}"
        for mode, model in CELL_ORDER
    ]
    answer_source_figure = (
        grouped_rate_svg(
            "Answer-source blanking · exact-count loss",
            compact_cell_labels,
            (
                (
                    "prompt records blank",
                    "#667085",
                    [
                        -finite_float(
                            full_suite_claims["answer_source"][
                                suite_cell_label[(mode, model)]
                            ]["prompt_records_blank"]["mean_delta_exact_count"]
                        )
                        for mode, model in CELL_ORDER
                    ],
                ),
                (
                    "full trace blank",
                    "#0f766e",
                    [
                        -finite_float(
                            full_suite_claims["answer_source"][
                                suite_cell_label[(mode, model)]
                            ]["trace_all_blank"]["mean_delta_exact_count"]
                        )
                        for mode, model in CELL_ORDER
                    ],
                ),
            ),
            y_label="loss in exact-count rate relative to clean (0–1)",
        )
        if full_suite_claims
        else ""
    )
    frozen_representation_figure = grouped_rate_svg(
        "Frozen report layers · confirmation balanced accuracy",
        compact_cell_labels,
        (
            (
                "running item_end",
                "#0f766e",
                [
                    finite_float(cells[f"{mode}|{model}"]["representation"]["running_confirmation_logistic_ba"])
                    for mode, model in CELL_ORDER
                ],
            ),
            (
                "final answer_query_v3",
                "#46758f",
                [
                    finite_float(cells[f"{mode}|{model}"]["representation"]["final_confirmation_logistic_ba"])
                    for mode, model in CELL_ORDER
                ],
            ),
        ),
        y_label="confirmation balanced accuracy",
    )
    state_scope_gate_figure = grouped_rate_svg(
        "Narrow component versus full-state causal gates",
        compact_cell_labels,
        (
            (
                "narrow loop",
                "#b54708",
                [
                    float(bool(cells[f"{mode}|{model}"]["narrow_loop"]["native_loop_pass"]))
                    for mode, model in CELL_ORDER
                ],
            ),
            (
                "full direct",
                "#315f7d",
                [
                    float(bool(cells[f"{mode}|{model}"]["full_commit_to_query"]["confirmation"]["strong_direct_gate_pass"]))
                    for mode, model in CELL_ORDER
                ],
            ),
            (
                "first city",
                "#7c3aed",
                [
                    float(bool(cells[f"{mode}|{model}"]["full_item_greedy"]["strong_interval_gate_pass"]))
                    for mode, model in CELL_ORDER
                ],
            ),
            (
                "depth-4",
                "#0f766e",
                [
                    float(bool(cells[f"{mode}|{model}"]["full_item_multihop"]["primary_depth4_strong_gate_pass"]))
                    for mode, model in CELL_ORDER
                ],
            ),
        ),
        y_label="registered gate indicator (0/1; not effect size)",
    )
    greedy_adoption_figure = grouped_rate_svg(
        "Full-item first-city donor adoption by condition",
        compact_cell_labels,
        (
            (
                "donor→receiver patch",
                "#0f766e",
                [
                    finite_float(cells[f"{mode}|{model}"]["full_item_greedy"]["patched_donor_adoption_rate"])
                    for mode, model in CELL_ORDER
                ],
            ),
            (
                "receiver-self control",
                "#667085",
                [
                    finite_float(cells[f"{mode}|{model}"]["full_item_greedy"]["receiver_self_donor_adoption_rate"])
                    for mode, model in CELL_ORDER
                ],
            ),
            (
                "native-donor control",
                "#d97706",
                [
                    finite_float(cells[f"{mode}|{model}"]["full_item_greedy"]["native_donor_adoption_rate"])
                    for mode, model in CELL_ORDER
                ],
            ),
        ),
        y_label="first known-city donor-adoption rate",
    )
    retrieval_forest = interval_forest_svg(
        "Target-city log-probability damage and frozen-bank specificity",
        retrieval_forest_rows,
        unit="seed-equal log P effect · 95% bootstrap CI",
    )
    sensitivity_forest = (
        interval_forest_svg(
            "Index 2×2 bank-anchor × lesion-start sensitivity",
            sensitivity_forest_rows,
            unit="selected-minus-random next-city failure · 20-seed bootstrap 95% CI",
        )
        if sensitivity_complete
        else ""
    )
    carrier_forest = interval_forest_svg(
        "Query-local carrier deformation and clean-carrier restoration",
        carrier_forest_rows,
        unit="registered carrier effect · 95% bootstrap CI",
    )
    carrier_deformation_forest = interval_forest_svg(
        "Query-local targeted-bank damage deforms the grammar carrier",
        carrier_forest_rows[0::2],
        unit="registered carrier deformation · 95% bootstrap CI",
    )
    carrier_restoration_forest = interval_forest_svg(
        "Clean-carrier clamp rescues the later item-end commit",
        carrier_forest_rows[1::2],
        unit="registered commit restoration · 95% bootstrap CI",
    )
    commit_forest = interval_forest_svg(
        "Full-state commit patch redirects the next targeted query",
        commit_forest_rows,
        unit="paired targeted-attention effect · 95% bootstrap CI",
    )
    terminal_forest = interval_forest_svg(
        "Terminal necessity versus global one-item sufficiency",
        terminal_forest_rows,
        unit="expected-count utility effect · 95% bootstrap CI",
    )
    local_terminal_forest = interval_forest_svg(
        "Bullet clean-context local terminal mediation",
        local_terminal_forest_rows,
        unit="expected-count utility effect · 95% bootstrap CI",
    )
    direct_count_margin_forest = (
        interval_forest_svg(
            "Frozen targeted-query bank changes final count-sequence margins",
            direct_count_margin_forest_rows,
            unit="gold-vs-best-wrong sequence-margin effect · 95% bootstrap CI",
        )
        if direct_count_margin_forest_rows
        else ""
    )
    answer_extension_figure = (
        answer_query_layer_adoption_svg(extension_cells)
        if extension_complete
        else ""
    )
    relay_extension_forest = (
        interval_forest_svg(
            "Same-trial terminal relay: natural damage and specific mediation",
            relay_extension_forest_rows,
            unit="correct-count sequence-margin effect · true-source-seed bootstrap 95% CI",
        )
        if extension_complete
        else ""
    )
    ncc_forest = interval_forest_svg(
        "NCC directional damage and selected-versus-random specificity",
        ncc_forest_rows,
        unit="NCC margin loss · 95% bootstrap CI",
    )
    multihop_figure = multihop_depth_svg(multihop["cell_summaries"])
    mask_scope_figure = mask_scope_timeline_svg(head_mask_audit)
    full_suite_coverage_figure = suite_coverage_svg()
    manifold = data["representation_manifold"]
    manifold_script = point_cloud_script(manifold)

    index_sensitivity_section = ""
    sensitivity_appendix_block = ""
    if sensitivity_complete:
        sensitivity_provenance_rows = []
        for model in ("Qwen3-8B", "Gemma4-E4B"):
            analysis = sensitivity_analyses[model]
            container_audit = analysis.get("generation_container_audit", {})
            sensitivity_provenance_rows.append(
                (
                    esc(model),
                    str(analysis["fixed_k"]),
                    esc(analysis["decision"]),
                    esc(container_audit.get("status", "not-recorded")),
                    f"<code>{esc(item_end_sensitivity['analysis_source_sha256'][model])}</code>",
                )
            )
        sensitivity_provenance_table = table(
            (
                "模型",
                "冻结 K",
                "探索性判定",
                "generation-container audit",
                "analysis.json SHA-256",
            ),
            sensitivity_provenance_rows,
            cls="wide hash-table",
        )
        sensitivity_appendix_block = f"""
<div class="appendix-block"><h3>A.7 Index item-end 2×2 的冻结与来源审计</h3>{sensitivity_provenance_table}<div class="reading-contract"><div class="contract-row"><strong>共同冻结合同</strong><span><code>{esc(item_end_sensitivity['contract_sha256'])}</code>；20 个原始 analysis-slot seeds、三个 random repeats、fixed K、五个 contrasts。</span></div><div class="contract-row"><strong>科学范围</strong><span><code>{esc(item_end_sensitivity['scientific_scope'])}</code>；primary_confirmation_replaced=false；k_reselected=false；confirmation_authorized=false。</span></div><div class="contract-row"><strong>Gemma appendable container 恢复</strong><span>聚合 generation 文件在冻结后继续追加，因此不以旧 container hash 冒充相同文件；恢复审计逐条验证冻结 200-row cohort 与冻结前 immutable shard object 完全相等，并固定 canonical row digest，随后才运行新增 behavior arms。</span></div></div><div class="subsection-conclusion"><strong>A.7 结论。</strong>Item-end sensitivity 的设计先于新增三格 outcomes 冻结；它是可审计的 discovery-only 位置诊断，不能覆盖原 confirmation。</div></div>"""
        index_sensitivity_section = f"""
<h3>4.3a Index 的弱效应是否只是 lesion 起点选早了：bank anchor × intervention start 的 2×2 敏感性</h3>
<p class="lead">Index 的 primary assay 在 <code>post_marker</code>（实验文件名中的 <code>p2</code>）选择 head bank，也从该点开始持续 mask。这里把“在哪里选 bank”与“从哪里开始 lesion”拆成两个因素，并加入字面 item 结束点 <code>p0_item_end</code>，直接检验 token 位置解释。</p>
<div class="experiment-frame"><div><span class="experiment-label">实验目的</span>判断 Index primary 弱效应是否源于 lesion 在 city predictor 形成之前过早启动，而不是误把显式 ordinal 当成具体 city 内容。实验不改变目标 city，只改变 bank 的选择 anchor 与持续 mask 的起始 anchor。</div><div><span class="experiment-label">实验设定</span>Qwen 固定 K=128，Gemma 固定 K=8；2×2 cells 为 <code>p2bank_at_p2</code>、<code>p2bank_at_p0</code>、<code>p0bank_at_p2</code>、<code>p0bank_at_p0</code>。每格使用同一批 20 个 discovery analysis-slot seeds、gold N=10、1 个 selected bank 与 3 个预注册 random banks；从指定 anchor 起，<code>decode_head_ablation_steps=-1</code>，即后续自由生成全程持续 mask。新增三格在任何 outcome 生成前冻结。</div><div><span class="experiment-label">计算方法</span><span class="formula">E(B,S)=mean<sub>seed</sub>[Fail(selected bank B, start S)−mean<sub>r=1..3</sub>Fail(random bank r, start S)]<br>Δ<sub>item-end</sub>=E(p0 bank,p0 start)−E(p2 bank,p2 start)</span><code>Fail=1</code> 表示下一已知 city 不正确。每个 seed 先平均三个 random controls，再对 20 个原始 analysis-slot seeds 等权；95% CI 用 10,000 次 seed bootstrap。其余四个 factorial contrasts 分别隔离 bank-anchor effect 与 start-site effect。</div><div><span class="experiment-label">结果</span>{esc(sensitivity_decision_summary)}。八个 cell effects 与十个冻结 contrasts 均完整报告；主判定只读取 <code>overall_item_end_minus_primary</code> 的方向和 CI。</div><div><span class="experiment-label">分析</span>若 Δ 的 95% CI 全部高于 0，说明 item-end 对齐使 selected bank 相对 matched random 更具行为必要性，支持“primary window 过早”的位置敏感解释；若 CI 跨 0 或效应不增，则不能把 primary 弱效应仅归因于 token 起点。无论结果如何，该 discovery-only sensitivity 都不替换原 primary/confirmation、不重选 K，也不把显式 ordinal 当作 city memory。</div><div><span class="experiment-label">简单例子</span>假设同一 20 seeds 上，primary <code>p2bank_at_p2</code> 的 selected−random failure 为 0.05，而 <code>p0bank_at_p0</code> 为 0.30，则 Δ=0.25；只有 seed-bootstrap CI 也在 0 以上，才写“支持位置敏感”，不能只看两个点估计大小。</div></div>
<div class="figure-primer"><div><strong>图中画什么</strong>Qwen 与 Gemma 的四个 2×2 cell effects，以及各自预注册的 item-end−primary 主 contrast。</div><div><strong>坐标怎么读</strong>横轴是 selected bank 相对三个 matched random banks 增加的 next-city failure probability；正值表示 selected bank 的持续 mask 更具特异损伤。纵轴是模型×cell/contrast。</div><div><strong>点与误差线</strong>点为 20 个原始 analysis-slot seeds 的等权 mean，横线为 10,000 次 seed bootstrap 95% CI；零线为无 selected-vs-random specificity。</div></div><figure class="paper-figure"><h3 class="figure-title">图 4c · Index item-end anchor 的 2×2 discovery-only 敏感性</h3>{sensitivity_forest}<figcaption><code>p2</code> 对应原 <code>post_marker</code> anchor，<code>p0</code> 对应字面 <code>p0_item_end</code>。前四行/模型给出 E(B,S)，最后一行/模型给出 E(p0,p0)−E(p2,p2)。横轴单位是失败概率差而不是 attention mass；不同 K 的 Qwen/Gemma 只在各自 matched-random 合同内解释。该图是事后动机、结果前冻结的 discovery sensitivity，不是 fresh confirmation。</figcaption></figure>{sensitivity_cell_table}{sensitivity_contrast_table}<div class="subsection-conclusion"><strong>4.3a 结论。</strong>{esc(sensitivity_decision_summary)}。这项实验回答“token 起点是否错位”，但严格保留原 primary 结果并禁止按本结果追加 anchor、重选 seed 或重选 K。</div>"""

    if extension_complete:
        answer_result_summary = "；".join(
            (
                f"{MODE_LABEL[mode]}·{'Qwen' if model == 'Qwen3-8B' else 'Gemma'} "
                f"L{extension_cells[f'{mode}|{model}']['answer_terminal_layer']}="
                f"{num(extension_cells[f'{mode}|{model}']['answer_terminal_adoption'])} "
                f"[{num(extension_cells[f'{mode}|{model}']['answer_terminal_ci_low'])}, "
                f"{num(extension_cells[f'{mode}|{model}']['answer_terminal_ci_high'])}]"
            )
            for mode, model in CELL_ORDER
        )
        answer_execution_section = f"""
<h3>6.3 Answer-query residual 本身是否已经是可执行的 count state</h3>
<p class="lead">这里把 Native §6.3 原样移植到 Enumeration：直接把 donor 的完整 <code>answer_query_v3</code> residual 写入 receiver，按预冻结的八层网格观察实际 greedy 数字是否采用 donor count。Frame 13 的 bank→count-margin 结果保留为下游传播补充，但不再替代本 assay。</p>
<div class="experiment-frame"><div><span class="experiment-label">实验目的</span>区分“answer query 上可以线性解码 count”与“该完整 state 能被后续层执行”。若 donor state 写入后让 receiver 的实际生成数字采用 donor count，则建立 answer-state→number execution。</div><div><span class="experiment-label">实验设定</span>Site 固定为 <code>answer_query_v3</code>，即最终整数首 token 前的字面 trace boundary。每格使用 frozen coherent confirmation cohort、true source seed identity、outcome-blind directed pairs；Qwen 层网格为 L0/5/10/15/20/25/30/35，Gemma 为 L0/6/12/18/23/29/35/41。每层只有 self patch 与 full donor patch 两臂，greedy 最多 16 tokens。</div><div><span class="experiment-label">计算方法</span><span class="formula">Adopt<sub>i,l</sub>=1[ŷ<sup>full donor</sup><sub>i,l</sub>=N<sup>donor</sup><sub>i</sub>]<br>LayerEffect<sub>l</sub>=(1/S) Σ<sub>s=1</sub><sup>S</sup> mean<sub>i∈seed s</sub> Adopt<sub>i,l</sub></span>Self patch 必须逐 pair×layer 重新生成 receiver gold，否则分析 fail closed；不可解析或越界的 greedy 输出保留在分母并计 adoption failure。区间对 true source seed cluster bootstrap 10,000 次。</div><div><span class="experiment-label">结果</span>{esc(answer_result_summary)}。逐层曲线、onset、pair 数与可解析率见图表；onset 只按预注册描述规则汇报，不用于重选层。</div><div><span class="experiment-label">分析</span>该实验与 Native 使用相同 semantic site、两臂、模型特定八层网格、greedy donor adoption 与 seed-cluster estimand。它证明完整 answer state 可执行；由于 patch 是 full residual，仍不能推出一个低维、content-free scalar counter。</div><div><span class="experiment-label">简单例子</span>Receiver 的 clean Total 是 3，同一 source-seed donor 的 clean Total 是 4。若在某层写入 donor answer-query state 后，receiver 实际 continuation 从“3”变为“4”，记 donor adoption；probe 更靠近 4 但输出不变不算。</div></div>
<div class="figure-primer"><div><strong>图中画什么</strong>四格 answer_query_v3 full-state donor patch 的逐层 greedy donor-count adoption。</div><div><strong>坐标怎么读</strong>横轴是 zero-based post-block layer；纵轴 0–1 是先 seed 内平均、再让 true source seeds 等权的 adoption rate。</div><div><strong>点与误差线</strong>点为 seed-equal mean，竖线为 true-source-seed cluster bootstrap 95% CI；不同模型的层号只在各自网络内解释。</div></div><figure class="paper-figure"><h3 class="figure-title">图 6c · Enumeration answer-query full-state patch 的逐层 donor-count adoption</h3>{answer_extension_figure}<figcaption>四个 panel 分别对应 Index-Qwen、Index-Gemma、Bullet-Qwen 与 Bullet-Gemma。横轴严格使用预冻结的模型特定八层网格，纵轴是 deterministic greedy continuation 对 donor gold count 的 adoption。Self patch 不画成曲线，因为它在每个 pair×layer cell 都必须重新生成 receiver gold，否则整个分析失败；不可解析/越界输出没有被删除。误差线的聚类单位是真实 source seed，而不是把同一 seed 的多条 directed pairs 当独立样本。</figcaption></figure>{answer_extension_table}<div class="subsection-conclusion"><strong>6.3 主结论。</strong>Enumeration 已完成 Native §6.3 的 exact answer-state execution assay；允许的结论是“完整 answer-query count state 可被后续层执行为实际数字”，不是“纯低维计数器已被隔离”。</div>
<h4>6.3a 补充边界：frozen targeted-query bank 是否传播到 final count margin</h4><div class="experiment-frame"><div><span class="experiment-label">实验目的</span>补充检验 frozen retrieval bank 的干预是否传播到最终 gold-vs-best-wrong count sequence margin。</div><div><span class="experiment-label">实验设定</span>Sealed Frame 13 比较 clean、selected bank 与三个 matched random banks；confirmation 不按本轮 margin 结果重选 bank。</div><div><span class="experiment-label">计算方法</span><span class="formula">Margin(N)=S(N)−max<sub>j≠N</sub>S(j)<br>specificity=[Margin(clean)−Margin(selected)]−mean random loss</span>完整 autoregressive count sequence 计分，95% CI 对 true source seed 聚类。</div><div><span class="experiment-label">结果</span>Index-Qwen 与 Bullet-Qwen 的 directional/specificity interval 最强；Index-Gemma 仅部分支持，Bullet-Gemma 为零效应。逐格数值见图表及 Appendix Frame 13。</div><div><span class="experiment-label">分析</span>这是一条 bank→decoder-margin 的补充下游边；它与 6.3 主 assay 的 full answer residual→greedy number estimand 不同，不能混成同一效应量。</div><div><span class="experiment-label">简单例子</span>Gold=6；若 selected-bank lesion 把 margin 从 +4 降到 +1，而随机 bank 平均只降到 +3.5，则 specificity=2.5。</div></div><div class="figure-primer"><div><strong>图中画什么</strong>四格 selected margin loss 与 selected−random specificity。</div><div><strong>坐标怎么读</strong>横轴是完整 count sequence 的 log-score margin effect，纵轴为 cell×estimand；零线为无效应。</div><div><strong>不能怎么读</strong>该横轴不是 layer，outcome 也不是 greedy adoption；它只报告 retrieval bank 的下游传播。</div></div><figure class="paper-figure"><h3 class="figure-title">图 6d · Targeted-query bank 对 final count-sequence margin 的传播</h3>{direct_count_margin_forest}<figcaption>点是 true-source-seed 等权 paired mean，线是 bootstrap 95% CI。每格上行为 selected bank 相对 clean 的 margin loss，下行为相对三个 matched random banks 的 specificity。不同模型 clean margin/K 不同，不按横向绝对距离排序；Frame 13 是 registered existing-split extension。</figcaption></figure>{direct_count_margin_table}<div class="subsection-conclusion"><strong>6.3a 结论。</strong>Bank-level final-margin 传播具有 grammar/model 异质性；它补充但不改变 exact answer-state execution 的四格结果。</div>"""

        terminal_relay_section = f"""
<h3>6.4 Earlier terminal state 是否经 post-terminal suffix 与 answer query 传递</h3><p class="lead">本节在每个 Enumeration grammar×model 单元的同一 directed pairs 上运行 Native 的 2×3 factorial：source 为 self/full donor terminal state，relay 为 natural、clean post-terminal suffix reset、clean answer-query-only reset。它直接检验 terminal trace state→suffix/query→count margin 的 same-trial 部分中介。</p>
<div class="experiment-frame"><div><span class="experiment-label">实验目的</span>避免把两个独立 patch 首尾相接：在同一 trial 内测量 upstream terminal-state damage 是否会被 clean suffix/query reset 削弱，从而建立有序的部分 relay。</div><div><span class="experiment-label">实验设定</span>四格均使用 frozen coherent confirmation cohort 与 true source seed。{esc(relay_geometry_design_text)}。逐格 geometry 为：{relay_geometry_cell_text}。Qwen source/relay 层为 L19/L26，Gemma 为 L16/L34；source arms 是 self/full donor，relay arms 是 natural、clean suffix reset、clean answer-query-only reset。每格始终保留预注册的 10 个 seed；若某 seed 的全部 registered pairs 都短于该格 geometry，则在 planned audit 中保留并标作 full-NA，不能悄悄删掉或换 seed。Pair selection 不读取 intervention outcomes。{esc(original_suffix8_audit_text)}。</div><div><span class="experiment-label">计算方法</span><span class="formula">Damage<sub>natural</sub>=M(self,natural)−M(donor,natural)<br>SpecificMediation<sub>reset</sub>=Damage<sub>natural</sub>−Damage<sub>reset</sub><br>ExplainedFraction=1−|Damage<sub>suffix reset</sub>/Damage<sub>natural</sub>|</span><em>M</em> 是 gold count 相对最强错误 count 的完整 sequence log-score margin。数值 estimand 只定义在 geometry-eligible seeds 上，先在 seed 内平均、再令 eligible true source seeds 等权 bootstrap；suffix 与 query estimands嵌套，禁止相加。若某格 eligible seed 为 0，则三个数值 estimand 均记为不可估计，而不是零效应或 gate FAIL。</div><div><span class="experiment-label">结果</span>{extension_relay_estimable_count}/4 格在各自明确标注的 geometry 下具有数值支持；在这些可估计格中，Natural damage 与 suffix-specific mediation 的联合主 gate 为 {extension_partial_mediation_count}/{extension_relay_estimable_count}，answer-query-only secondary gate 为 {extension_query_mediation_count}/{extension_relay_estimable_count}。其余 {4 - extension_relay_estimable_count}/4 格为 geometry 不可估计。Seed accounting 为：{relay_seed_accounting_text}。逐格 effect、CI、explained fraction 与 residual/natural ratio 见图表。</div><div><span class="experiment-label">分析</span>主 gate 通过表示 clean downstream reset 显著削弱 upstream terminal patch damage；residual ratio 仍单独报告，因此即使 partial mediation 为正，也不声称 complete、single 或 exclusive mediation。Full-NA 是注册 geometry 的可估计性边界，不是行为失败、零效应或 outcome-based exclusion。Bullet suffix4 是透明标注的 task adaptation，不回写为原 suffix8 confirmation。Scientific gate FAIL 只适用于有数值支持但区间未通过的格。</div><div><span class="experiment-label">简单例子</span>若 donor terminal patch 让正确 count margin 相对 self 下降 4.0，clean suffix reset 后只下降 1.5，则 specific mediation=2.5、explained fraction=62.5%；剩余 1.5 明确表示仍有旁路或未重置 state。若某格十个 seed 的所有 trace 都达不到其注册 suffix width，它仍计入“10 个已预注册”，但该格不产生均值、CI 或 pass/fail。</div></div>
<div class="figure-primer"><div><strong>图中画什么</strong>在各自注册 geometry 下可估计的 cell 中，natural terminal-patch damage、suffix-specific mediation 与 answer-query-only mediation；全格不可估计者只在表中保留，不伪造零点。</div><div><strong>坐标怎么读</strong>横轴是 correct-count sequence-margin effect 与 95% CI；纵轴是可估计 cell×三个 same-trial estimands；零线为无效应。</div><div><strong>不能怎么读</strong>Suffix 与 query reset 是嵌套定位，不能相加；四格各自估计，suffix width 也不用于比较模型机制强弱。表中的 eligible/planned 分母不可当作准确率。</div></div><figure class="paper-figure"><h3 class="figure-title">图 6e · Enumeration terminal relay 的同一-trial 部分中介</h3>{relay_extension_forest}<figcaption>点为先在 geometry-eligible true source seed 内平均、再让 eligible seeds 等权的 effect；横线为 10,000 次 cluster bootstrap 95% CI。Natural damage 是 upstream full-donor terminal patch 的总作用；suffix/query-specific mediation 是 patch×clean-reset interaction。全格 0/10 eligible 的 cell 不画成零效应，而在表中标为不可估计。表格逐格标注 geometry/evidence label，并保留 eligible/planned 与 full-NA seed，防止 task adaptation 或 geometry 排除被误读为换 seed。</figcaption></figure>{relay_extension_table}<div class="subsection-conclusion"><strong>6.4 主结论。</strong>Enumeration 已执行 Native §6.4 的同一 2×3 same-trial relay estimator；{extension_relay_estimable_count}/4 格可数值估计。原 suffix8 与 post-hoc Bullet suffix4 的证据标签分开，允许的结论只按可估计格写为有序、部分、非唯一通路，不把不可估计写成 null，也不把 task adaptation 冒充原合同。</div>
<h4>6.4a 补充边界：原 terminal token / written-state 局部实验</h4><div class="experiment-frame"><div><span class="experiment-label">实验目的</span>保留原 Frame 11 与 Bullet clean-context assay，检验 terminal token 的效果是否依赖其写入 state。</div><div><span class="experiment-label">实验设定</span>四格报告 token-written-state sufficiency 与 token-effect-requires-state；Bullet 两格另含 local necessity、clean-state restore 与 state occlusion。</div><div><span class="experiment-label">计算方法</span>在受损背景写回 clean terminal state 测 restoration；保留 token 但遮蔽其 state 测 occlusion。Bullet local mediation 要求三个注册方向共同成立。</div><div><span class="experiment-label">结果</span>四格依赖项见表 6.2；Bullet local mediation 为 {local_terminal_count}/2，三项局部效应如下图。</div><div><span class="experiment-label">分析</span>该 assay 使用 expected-count utility 与局部 clean context；它是 exact 2×3 sequence-margin relay 的任务内补充，不能与主 interaction 数值相加。</div><div><span class="experiment-label">简单例子</span>若遮蔽 terminal item state 后答案受损，写回 clean state 后改善，再次遮蔽又失效，则 token 的作用至少部分通过该 state。</div></div><div class="figure-primer"><div><strong>图中画什么</strong>Bullet-Qwen/Gemma 的 local necessity、clean-state restoration 与 state occlusion。</div><div><strong>坐标怎么读</strong>横轴为 expected-count utility effect 与 95% CI；纵轴为模型×三个 paired estimands；零线为无局部作用。</div><div><strong>外推边界</strong>该图没有 Index，也不是 2×3 sequence-margin interaction，只作为局部 token/state 边界。</div></div><figure class="paper-figure"><h3 class="figure-title">图 6f · Bullet clean-context terminal token→state→answer 的局部 relay</h3>{local_terminal_forest}<figcaption>横轴是 expected-count utility 的 paired effect，点为 seed-equal mean，线为 95% CI。三项共同支持局部 token/state relay，但不替代 6.4 主 assay。</figcaption></figure><div class="subsection-conclusion"><strong>6.4a 结论。</strong>Bullet 两格的局部 token/state 结果与 same-trial relay 互为补充；估计量和样本单位分开报告。</div>"""
        terminal_section_conclusion = (
            f"Enumeration 与 Native-thinking 的 §6.3/§6.4 已完成 exact assay 对齐："
            f"answer-query full-state greedy adoption 使用相同八层网格，terminal relay 使用相同 2×3 estimator；"
            f"relay 在 {extension_relay_estimable_count}/4 格可估计，"
            f"其中 partial-mediation 主 gate 为 {extension_partial_mediation_count}/{extension_relay_estimable_count}；"
            f"{relay_geometry_design_text}；{original_suffix8_audit_text}；"
            "Source necessity 与 global sufficiency 仍保留 Enumeration 的 grammar split（Index 2/2、Bullet 0/2），不以模板预期覆盖实际结果。"
        )
    else:
        answer_execution_section = f"""
<h3>6.3 Frozen targeted-query bank 是否改变可执行的 final count margin</h3><div class="experiment-frame"><div><span class="experiment-label">实验目的</span>检验 frozen targeted-query bank 的干预是否传播到最终 gold-vs-best-wrong count sequence margin，而不只改变中间 attention/readout。</div><div><span class="experiment-label">实验设定</span>Frame 13 在每格比较 clean、selected bank 与三个 matched random banks；confirmation 只读取 discovery/registration 后冻结的 bank 与 endpoint，不按本轮 margin 结果重选。</div><div><span class="experiment-label">计算方法</span><span class="formula">Margin(N)=S(N)−max<sub>j≠N</sub>S(j), j∈1,…,10<br>selected loss=Margin(clean)−Margin(selected)<br>specificity=selected loss−mean random loss</span>候选整数按完整 autoregressive sequence 计分，95% CI 对 true source seed 聚类。</div><div><span class="experiment-label">结果</span>Index-Qwen 与 Bullet-Qwen 的两项 interval gate 通过；Index-Gemma 有 selected directional loss，但 selected−random specificity CI 跨 0；Bullet-Gemma 为零效应。</div><div><span class="experiment-label">分析</span>该实验支持 retrieval bank 对 final decoder margin 的传播，但不是 Native §6.3 的逐层 full answer-query residual donor-count greedy adoption。</div><div><span class="experiment-label">简单例子</span>Gold=6。若 clean margin=+4，关闭 selected bank 后=+1，而 matched random bank 平均仍为+3.5，则 selected loss=3、specificity=2.5。</div></div><div class="figure-primer"><div><strong>图中画什么</strong>四格 selected margin loss 与 selected−random specificity。</div><div><strong>坐标怎么读</strong>横轴是完整 count sequence 的 log-score margin effect，纵轴为 cell×estimand。</div><div><strong>边界</strong>该 outcome 不是 donor greedy-number adoption。</div></div><figure class="paper-figure"><h3 class="figure-title">图 6c · Targeted-query bank 对 final count-sequence margin 的传播</h3>{direct_count_margin_forest}<figcaption>点是 true-source-seed 等权 paired mean，线是 bootstrap 95% CI。</figcaption></figure>{direct_count_margin_table}<div class="subsection-conclusion"><strong>6.3 结论。</strong>已有功能对应证据，但尚未逐层复制 Native exact assay。</div>"""
        terminal_relay_section = f"""
<h3>6.4 Terminal token effect 是否依赖 written state，局部 relay 能否恢复</h3><div class="experiment-frame"><div><span class="experiment-label">实验目的</span>检验 terminal token 的效果是否通过其写入 hidden state 传递。</div><div><span class="experiment-label">实验设定</span>Frame 11 四格依赖项与 Bullet 两格 local mediation。</div><div><span class="experiment-label">计算方法</span>比较 local necessity、clean-state restore 与 state occlusion。</div><div><span class="experiment-label">结果</span>Bullet local mediation 为 {local_terminal_count}/2。</div><div><span class="experiment-label">分析</span>尚未实现 Native 2×3 source patch×suffix/query reset。</div><div><span class="experiment-label">简单例子</span>遮蔽 state 后受损、写回后恢复，支持局部 relay。</div></div><div class="figure-primer"><div><strong>图中画什么</strong>Bullet local terminal effects。</div><div><strong>坐标怎么读</strong>横轴为 expected-count utility effect。</div><div><strong>边界</strong>不是 Native exact factorial。</div></div><figure class="paper-figure"><h3 class="figure-title">图 6d · Bullet clean-context terminal relay</h3>{local_terminal_forest}<figcaption>局部 mediation 的 paired effect 与 95% CI。</figcaption></figure><div class="subsection-conclusion"><strong>6.4 结论。</strong>局部 relay 为任务适配对应。</div>"""
        terminal_section_conclusion = (
            "Enumeration 与 Native-thinking 在 source necessity、terminal bridge 和下游 count-margin 上方向对齐；"
            "§6.3/§6.4 仍是功能对应而非 exact assay。"
        )

    source_rows = [
        (f"<code>{esc(path)}</code>", f"<code>{esc(digest)}</code>")
        for path, digest in data["source_sha256"].items()
    ]
    source_table = table(("Evidence path", "SHA-256"), source_rows, cls="hash-table")
    embedded_data = dict(data)
    embedded_data["representation_manifold"] = {
        "status": manifold["status"],
        "qualification": manifold["qualification"],
        "payload_sha256": data["native_aligned_representation"][
            "manifold_manifest"
        ]["output"]["sha256"],
        "heavy_coordinates_embedded_once_in_viewer": True,
    }
    embedded_data.pop("native_template_css", None)
    embedded_data.pop("full_suite_frames_html", None)
    embedded = json.dumps(
        embedded_data, ensure_ascii=False, sort_keys=True
    ).replace("</", "<\\/")

    native_css = str(data.get("native_template_css", "")).strip()
    if not native_css:
        raise ValueError("Native-thinking template CSS is required for the mirrored report")
    extra_css = r"""
.wide{min-width:1050px}.hash-table{min-width:1050px}code{overflow-wrap:anywhere;word-break:break-word}
.pill{display:inline-block;margin:2px;padding:2px 7px;border:1px solid currentColor;border-radius:999px;font:750 10px/1.35 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:nowrap}.support{color:#12665c;background:#eef8f5}.diagnostic{color:#2f6279;background:#edf5f8}.restored{color:#385b84;background:#eef2f9}.partial{color:#82601d;background:#fbf6e9}.null{color:#7a4141;background:#faf0f0}
.chain{min-width:1080px;border-top:1px solid #d9dee7;border-bottom:1px solid #d9dee7}.chain-lane{display:grid;grid-template-columns:160px minmax(170px,1fr) 34px minmax(170px,1fr) 34px minmax(190px,1fr) 34px minmax(155px,1fr);align-items:stretch;border-top:1px solid #e5e9ef;background:#fff}.chain-lane:first-child{border-top:0}.lane-name,.node{padding:15px 13px}.lane-name{background:#eef2f1}.lane-name strong,.lane-name span,.node b,.node .micro{display:block}.lane-name span,.micro{margin-top:4px;color:#667085;font-size:11px}.arrow{display:grid;place-items:center;color:#0f766e;font-size:24px;background:#f7f8fa}.chain-scroll{overflow-x:auto}
.geometry-viewer{margin:22px 0;padding:18px;background:#fff;border:1px solid #d9dee7}.geometry-toolbar{display:flex;flex-wrap:wrap;gap:12px;align-items:end;margin-bottom:16px}.geometry-toolbar label{display:grid;gap:5px;color:#475467;font-size:12px;font-weight:700}.geometry-toolbar select,.geometry-toolbar button,.cloud-head select{padding:7px 9px;border:1px solid #cfd6e2;background:#fff;color:#1d2939}.cloud-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.cloud-panel{min-width:0;overflow:hidden;margin:0;border:1px solid #d9dee7;background:#fbfcfe}.cloud-head{display:flex;flex-wrap:wrap;justify-content:space-between;gap:12px;align-items:center;padding:12px;border-bottom:1px solid #d9dee7}.cloud-head select{max-width:100%}.cloud-panel canvas{display:block;width:100%;height:470px;background:#fff}.cloud-stats{margin:0;padding:10px 12px;color:#667085;font-size:11px;border-top:1px solid #e5e9ef}.figure-stack{display:grid;gap:18px}.figure-stack .paper-chart{width:100%;height:auto}.parser-code{display:block;margin:14px 0;padding:14px 16px;background:#eef2f4;border-left:3px solid #46758f;font:12px/1.7 ui-monospace,SFMono-Regular,Consolas,monospace}.parser-warning{margin:14px 0;padding:14px 16px;background:#fff7ed;border-left:3px solid #d97706}.example-box{margin:14px 0;padding:14px 16px;background:#f8fafc;border:1px solid #d9dee7}.example-box pre{margin:8px 0 0;white-space:pre-wrap;font:12px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace}.claim-tier-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:20px 0}.claim-tier-grid>div{padding:16px;background:#fff;border-top:3px solid #0f766e}.claim-tier-grid h3{margin-top:0}.audit-badge{display:inline-block;padding:3px 8px;border:1px solid #cfd6e2;border-radius:999px;font-size:10px;font-weight:800}.paper-figure{margin:24px 0;overflow-x:auto;overscroll-behavior-inline:contain}.paper-figure>.figure-title{margin:0 0 14px;line-height:1.38}.paper-figure>figcaption{margin:14px 0 0;padding-top:12px;border-top:1px solid #e5e9ef;color:#667085;font-size:12px}.paper-chart{display:block;width:100%;height:auto;background:#fff}.bar-value{font-size:11px;fill:#344054}.bar-value-inverse{fill:#fff}.bar-label{font-size:12px;fill:#344054}.scope-label{font:700 10px ui-monospace,Consolas,monospace;fill:#fff}.experiment-frame .formula{display:block;margin-top:8px}.appendix-block{margin:22px 0;padding-top:12px;border-top:1px solid #d9dee7}.appendix-block>h3{margin-top:0}.source-list{font-size:11px}.meta span{white-space:normal}
@media(max-width:850px){.cloud-grid,.claim-tier-grid{grid-template-columns:1fr}.cloud-panel canvas{height:390px}.chain{min-width:0}.chain-lane{grid-template-columns:1fr}.arrow{min-height:30px;transform:rotate(90deg)}.paper-figure>.paper-chart,.paper-figure>.figure-stack .paper-chart{width:980px;min-width:980px;max-width:none}}
"""
    report_css = native_css + extra_css

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NiaH Enumeration：Native-thinking 同构机制复现报告</title><link rel="icon" href="data:,"><style>{report_css}</style></head>
<body><article class="page"><header><p class="eyebrow">Realistic CoT NiaH · Enumeration mechanism</p>
<h1>Enumeration 如何计数：content-bound state 驱动定向检索与后续枚举</h1>
<p class="dek">本报告严格复刻 Native-thinking 报告的叙事与视觉拓扑：先区分 representation 与 causal evidence，再沿 state formation、targeted retrieval、carrier/write、commit→query、terminal readout 组织证据。每个实验都依次给出目的、设定、计算方法、结果、分析、简单例子与当前结论；Index 与 Bullet 的差异始终分账，不用补 seed 后的 formal cohort eligibility 冒充原始行为准确率。</p>
<div class="meta"><span>Qwen3-8B · Index K128 / Bullet K96</span><span>Gemma4-E4B · Index K8 / Bullet K2</span><span>formal: 200 discovery / 100 confirmation per cell</span><span>sealed V6 + V2/V3 follow-up · no new forward for report rewrite</span></div></header>
<nav><a href="#summary">结论</a><a href="#baseline">1 基线</a><a href="#representation">2 表征</a><a href="#formation">3 State formation</a><a href="#retrieval">4 检索</a><a href="#write">5 写入与闭环</a><a href="#answer">6 终端</a><a href="#ledger">7 证据表</a><a href="#extension-audit">8 扩展审计</a><a href="#limitations">9 边界</a><a href="#appendix">Appendix</a></nav>
<main>
<section id="summary"><p class="eyebrow">Conclusion first</p><h2>先说机制：Enumeration 复现了 Native-thinking 的 content-bound recurrent counting loop</h2>
<div class="core-claim"><strong>本文主张。</strong>四个 grammar×model 单元都支持与 Native-thinking 相同的 recurrent computational loop / shared causal topology：targeted query 读取下一条 city record，检索结果写入 grammar carrier 与 item commit state，完整、content-bound 的 item state 又能重定向下一次 query，并在自由生成中维持 donor-aligned continuation。这里“相同”明确指计算角色与因果拓扑同型，不要求相同的层、head bank、效应量、低维坐标或神经实现；因此是否存在唯一、content-free、memoryless 的标量 <code>c ← c+1</code> 计数器不是本机制 claim 的必要条件。</div>
<div class="plain-language"><strong>可直接用于论文的机制表述。</strong> <em>Enumeration reproduces the same content-bound recurrent counting loop identified in Native-thinking: the committed item state causally redirects the next retrieval step and sustains subsequent donor-aligned enumeration. The correspondence is at the level of computational and causal topology, without requiring an identical low-dimensional representation or neural implementation.</em></div>
<p class="lead">最强证据由三个层级组成。第一，full-state teacher-forced commit→query direct edge 为 {full_gate_count}/4；第二，完整 item-span 对首个 donor city 的行为转移为 {esc(greedy_summary)}；第三，Full-state、content-bound 的多步 continuation 在 240 条冻结 generations 的 outcome-blind exact-prefix 重解析中显示 depth-4 主 gate 为 {multihop_strong_count}/4。与此同时，NCC、低维 loop、update 与 stop 为 0/4；Index-Gemma 的 backward multihop 方向也较弱。报告因此同时保留机制对齐与 grammar/model 异质性。</p>
<div class="reading-contract"><div class="contract-row"><strong>Strict exact accuracy</strong><span>原始冻结 slot 在补 seed 前必须同时通过目标 grammar、最终 Total、有序 city-score pairs、marker kind、一对一 semantic trace 与 gold item count。Index 为 {esc(index_accuracy['pass_count'])}/{esc(index_accuracy['total_count'])}={pct(index_accuracy['accuracy'])}，Bullet 为 {esc(bullet_accuracy['pass_count'])}/{esc(bullet_accuracy['total_count'])}={pct(bullet_accuracy['accuracy'])}。</span></div><div class="contract-row"><strong>Targeted query / carrier</strong><span>Query 是预测下一条 city 的注册位置；carrier 是从实际检索 city 到 item commit tail 的可见/隐状态通路。Bullet 的 invariant hyphen 不是数字 progress token；显式 ordinal 只提供 address/progress cue，不提供 city content。</span></div><div class="contract-row"><strong>Full-state item span</strong><span>在固定 donor/receiver endpoint 上移植完整、content-bound item state；它通常同时含 city、score 与 syntax。它比低维 count projection 更宽，因此能证明 state 使用，不能单独证明纯算术 component。</span></div><div class="contract-row"><strong>Multihop depth</strong><span>从生成文本的首个已知 city ordinal 开始，与 donor successor path 做 exact prefix。Depth d 表示前 d 个生成 city 全部连续且顺序正确；跳过、重排、去重或修复均禁止。</span></div><div class="contract-row"><strong>证据标签</strong><span>冻结 V6 confirmation、既有 raw-arm 恢复、事后 V2 扩展、fresh Bullet-Gemma carrier outcomes 与 V3 aggregate reparse 分开标记；事后机制诊断不是 fresh confirmation，也不冒充 prospective replication。</span></div></div>
<p class="main-note"><strong>最重要的限定。</strong>补 seed 后 1,200/1,200 只是 outcome-blind sealed replacement 得到的 replacement-filtered cohort formal eligibility；模型原始行为准确率仍为 1,076/1,200={pct(overall_accuracy['accuracy'])}。原 124 条失败和所有 replacement attempts 均在 Appendix 保留。</p>
<h3>与 Native-thinking 报告如何逐项对仗</h3><p>对齐方向是 <strong>Native-thinking → Enumeration</strong>：Native 的主文问题、实验角色与证据边界是模板，Enumeration 在 Index/Bullet 的表面语法和 token sites 上做必要适配。表中“同测量合同/同因果族”表示 estimand role 与 controls 同构；“任务适配对应”表示回答同一机制问题但不是逐 token、逐层或逐 trial 的复制。</p>{native_main_crosswalk_table}
<div class="figure-primer"><div><strong>图中画什么</strong>四个 grammar×model 单元从 query 到 carrier、commit 与 next query 的当前最高证据等级。</div><div><strong>坐标怎么读</strong>该机制图没有数值坐标轴；横向箭头表示候选计算顺序，四行分别是 Index/Bullet × Qwen/Gemma。</div><div><strong>标签怎么读</strong>† 事后诊断；* 冻结 raw-arm 恢复；‡ V2 事后扩展；§ fresh causal outcomes；¶ V3 冻结规则重解析。</div></div>
<figure class="paper-figure"><h3 class="figure-title">机制图 S1 · Enumeration counting 通路与当前证据等级</h3>{chain_figure(cells)}<figcaption>这是一张阶段图，没有数值坐标轴。每行从左到右依次表示 registered targeted query、grammar carrier、content-bound commit state 与 next query。格内颜色和文字表示该单元目前最高证据等级，而不是效应量；不同 K、模型架构与 grammar 的标签不可按颜色深浅排序。循环箭头只表示“完整 state 能控制下一次读取并维持枚举”的候选通路，不表示它是唯一 circuit，也不表示低维 <code>c←c+1</code> 算子已经被隔离。</figcaption></figure>
<div class="claim-tier-grid"><div><h3>Established</h3><p>四格 full-state direct edge 与 depth-4 cell-level multihop 主 gate；具体 city 内容不能由 visible ordinal 替代。</p></div><div><h3>Supported, not isolated</h3><p>完整 item state 能重定向并维持后续枚举；state 同时含 content、grammar 与 progress，尚非纯计数分量。</p></div><div><h3>Not established</h3><p>统一线性 centroid geometry、context-free update/stop、唯一 head/cell、或跨 grammar 完全相同的实现。</p></div></div>
<div class="parser-contract"><div class="parser-contract-head"><div><span class="parser-contract-kicker">Pre-experiment contract</span><h3>实验前置 · Parser、token site 与 replacement 合同</h3></div><p>以下规则决定一条输出能否计入行为准确率与 causal cohort。展开项给出接受语法、formal AND gate、token boundary 与 replacement firewall；parser 是测量合同，不是机制结果。</p></div><details class="parser-disclosure"><summary>A · Strict output grammar 与原始行为准确率</summary><div class="parser-disclosure-body">{parser_grammar_table}<p>Index 只接受连续 <code>1..M</code>；Bullet 只接受 ASCII hyphen。唯一 <code>Total:</code> 必须是最后非空行。原始 frozen slots 在 replacement 前计算 accuracy，reserve candidates 不进分母。</p>{behavioral_accuracy_table}<div class="parser-warning"><strong>Claim firewall。</strong>最终 100% formal cohort eligibility 是选择后的分析完整性，不是模型原始准确率。</div></div></details><details class="parser-disclosure"><summary>B · Formal causal cohort：七个条件取 AND</summary><div class="parser-disclosure-body">{parser_gate_table}<span class="parser-code">strict_causal_eligible = registered_success ∧ format_compliant ∧ Total=M ∧ ordered_gold_pairs ∧ marker_kind ∧ forward_one_to_one ∧ item_count=gold</span><p>Gold 只验证和定位，不生成、补齐、排序或去重模型输出。</p></div></details><details class="parser-disclosure"><summary>C · 字符边界如何编译为 causal token sites</summary><div class="parser-disclosure-body">{parser_site_table}<p>每个字符终点只允许 literal baseline prefix 或 text-exact retokenization；endpoint=<code>prefix_token_count−1</code>。无法保持文本精确相等即 fail closed，不用最近 token 猜测。</p></div></details><details class="parser-disclosure"><summary>D · Outcome-blind replacement 与 multihop readout</summary><div class="parser-disclosure-body"><p>原失败 slot 按 sealed reserve mapping 替换；intervention outcomes 不参与 eligibility。Multihop 主 readout 使用 <code>generated_known_city_ordinals_any_surface</code> 的 exact prefix，所有失败和截断保留在 unconditional denominator。</p>{replacement_policy_table}</div></details></div>
<div class="section-conclusion"><strong>Summary 结论。</strong>现有证据足以 claim Enumeration 与 Native-thinking 具有相同的 content-bound recurrent counting loop：full-state commit 控制下一次检索并维持后续枚举。该 claim 位于 computational/causal topology 层级；低维 NCC、update/stop 或唯一标量寄存器是否成立是更窄的实现问题，不是循环机制成立的前提。</div></section>
<section id="baseline"><p class="eyebrow">01 · Task and behavioral baseline</p><h2>1. 任务与行为基线：准确率按原始输出计算，formal replacement 只服务分析完整性</h2>
<div class="experiment-frame"><div><span class="experiment-label">实验目的</span>确定模型在 Index 与 Bullet 两种枚举语法下，能否按 passage 顺序完整列出所有目标 city-score records 并给出正确 Total；这也是后续机制实验的行为入口。</div><div><span class="experiment-label">实验设定</span>每个 grammar×model 单元包含 200 discovery 与 100 confirmation 原始冻结 slots。Index 明示 ordinal；Bullet 只有结构性 hyphen。四格合计 1,200 个原始 slots，补 seed 前统计。</div><div><span class="experiment-label">计算方法</span><span class="formula">StrictExactAccuracy = (# original slots satisfying all 7 strict gates) / (# original frozen slots)</span>Discovery、confirmation 和 pooled 结果分别报告；一个 city、score、顺序、marker 或 Total 错误都会使整条 slot 计 0。</div><div><span class="experiment-label">结果</span>Index 为 {esc(index_accuracy['pass_count'])}/{esc(index_accuracy['total_count'])}={pct(index_accuracy['accuracy'])}；Bullet 为 {esc(bullet_accuracy['pass_count'])}/{esc(bullet_accuracy['total_count'])}={pct(bullet_accuracy['accuracy'])}；总体为 {esc(overall_accuracy['pass_count'])}/{esc(overall_accuracy['total_count'])}={pct(overall_accuracy['accuracy'])}。四格细分见表 1。</div><div><span class="experiment-label">分析</span>Qwen 在两种 grammar 下均为 99.0%；主要下降来自 Gemma，尤其 Bullet-Gemma 的 76.3%。因此 grammar 差异不能只用 aggregate headline 表述；后续因果比较必须按四格分账。</div><div><span class="experiment-label">简单例子</span>Gold 是 <code>[Taipei:51, Nanjing:55]</code>。模型若输出两条都对但 <code>Total: 1</code>，或把 Nanjing 放在 Taipei 前，strict exact 都计失败；parser 不会修复。</div></div>
<div class="figure-primer"><div><strong>图中画什么</strong>四个 grammar×model 的 pooled 原始 strict exact pass rate。</div><div><strong>坐标怎么读</strong>横轴从 0 到 1，是补 seed 前整条输出通过全部 strict gates 的比例；纵轴是四个单元。</div><div><strong>不能怎么读</strong>条长不是 replacement 后的 formal cohort 覆盖率；后者按设计为 1.0。</div></div><figure class="paper-figure"><h3 class="figure-title">图 1 · Index 与 Bullet 的原始 strict exact enumeration accuracy</h3>{accuracy_figure}<figcaption>横轴是原始 frozen slots 的 strict exact pass rate，纵轴是 grammar×model 单元；条后数字同时给出百分比与分子/分母。Index/Bullet 分母各为 600，单格分母为 300。颜色只区分 grammar，不表示证据等级。该图在 reserve replacement 之前计算，因此能作为行为准确率；补 seed 后的 1,200/1,200 formal eligibility 不画在同一轴上，以免误解为模型性能。</figcaption></figure>{behavioral_accuracy_table}
<div class="subsection-conclusion"><strong>Experiment 1 结论。</strong>原始 strict exact accuracy 为 Index 91.7%、Bullet 87.7%；Qwen 两格均为 99.0%，Gemma 尤其在 Bullet 下更易出现格式、数量或有序 pair 失败。</div><div class="section-conclusion"><strong>行为基线结论。</strong>后续 mechanism assays 使用 outcome-blind replacement 形成完整 fixed-quota cohort，但所有 scientific interpretation 必须记住：formal eligibility 与原始模型准确率是两个不同 estimand。</div></section>
<section id="representation"><p class="eyebrow">02 · Measurement framework</p><h2>2. 通用测量框架：Representation 负责定位，Causal test 负责判定</h2>
<div class="experiment-frame"><div><span class="experiment-label">实验目的</span>检验 hidden state 是否携带 running occurrence 与 final count，并确定后续 causal assays 的候选层；可解码性只证明信息存在，不证明模型自然使用该线性方向。</div><div><span class="experiment-label">实验设定</span>Running endpoint 使用 <code>item_end</code>，标签为已完成 occurrence k；final endpoint 使用 <code>answer_query_v3</code>，标签为 gold count N。层只按 discovery-only 规则选择，confirmation 只读出。</div><div><span class="experiment-label">计算方法</span><span class="formula">BalancedAccuracy = (1/10) Σ<sub>c=1</sub><sup>10</sup> TP<sub>c</sub>/(TP<sub>c</sub>+FN<sub>c</sub>)</span>十类等权，chance=0.10。3D 图在每层仅用 discovery states 拟合 StandardScaler 与 PCA3，再投影 confirmation。</div><div><span class="experiment-label">结果</span>四格 running 与 final confirmation logistic balanced accuracy 均明显高于 0.10；冻结层与数值列在表 2。3D 图在所有 post-block layers 可交互查看，但不参与任何 gate。</div><div><span class="experiment-label">分析</span>可读信息同时存在于逐项 commit 与最终 answer query；NCC 的后续 causal null 说明“可分”与“干预沿统一 centroid 方向移动”不是同一个命题。</div><div><span class="experiment-label">简单例子</span>若 held-out seed 中完成第 4 条后的 state 被分类为 4，而非 3/5，则该层含 running-count 信息；只有移植该 state 后下一条 city 真的改变，才说明信息被后续计算使用。</div></div>
<h3>2.1 可解码性如何随层变化</h3><div class="figure-primer"><div><strong>图中画什么</strong>上图是 item-end running occurrence，下图是 answer-query final count；每个图含四个 grammar×model panel。</div><div><strong>坐标怎么读</strong>横轴是 zero-based post-block layer；纵轴是 confirmation balanced accuracy；灰虚线 0.10 是 chance。</div><div><strong>线与竖线</strong>灰线为 discovery selection score，绿/橙为 confirmation logistic/NCC；红色竖线是 discovery-only 默认层。</div></div><figure class="paper-figure"><h3 class="figure-title">图 2a · Enumeration count representation 的逐层 held-out readout</h3><div class="figure-stack">{running_curve_figure}{final_curve_figure}</div><figcaption>两张图的横轴均为 zero-based transformer post-block layer，纵轴为 10-class balanced accuracy（0 到 1）。第一张以 <code>item_end</code> 的 occurrence k 为标签；第二张以 <code>answer_query_v3</code> 的 gold N 为标签。灰虚线水平线为 0.10 chance；红色竖线由 discovery-only 选择，不能因 confirmation 峰值重新移动。不同模型层数不同，Lℓ 只表示各自第 ℓ 个 block 后 residual，不作跨模型同深度假设。</figcaption></figure><div class="subsection-conclusion"><strong>2.1 结论。</strong>四格都存在可泛化的 running/final count information；这一步定位 candidate state，但不贡献 causal-use claim。</div>
<h3>2.2 Count clouds 在低维中长什么样</h3><div class="figure-primer"><div><strong>图中画什么</strong>左侧为 running occurrence，右侧为 final count 的 confirmation point clouds。</div><div><strong>坐标怎么读</strong>PC1、PC2、PC3 是 discovery-fitted 标准化 hidden states 的前三主成分；轴的方向和符号没有机制语义。</div><div><strong>如何交互</strong>grammar/model 同步切换两侧；layer 独立选择；拖动同步旋转，双击重置。</div></div><figure class="paper-figure"><h3 class="figure-title">图 2b · Index/Bullet representation comparison 的逐层 PC1–PC3 manifold</h3><div class="geometry-viewer" id="representation-3d"><div class="geometry-toolbar"><label>Grammar<select id="enum-geometry-grammar"><option value="enumeration_index">Index</option><option value="enumeration_bullet">Bullet</option></select></label><label>Model<select id="enum-geometry-model"><option value="Qwen3-8B">Qwen3-8B</option><option value="Gemma4-E4B">Gemma4-E4B</option></select></label><button id="enum-geometry-reset" type="button">Reset synchronized view</button></div><div class="cloud-grid"><div class="cloud-panel"><div class="cloud-head"><strong>Running occurrence · item_end</strong><select id="enum-running-layer" aria-label="Running representation layer"></select></div><canvas id="enum-running-canvas" aria-label="Interactive running occurrence PCA3 point cloud"></canvas><p class="cloud-stats" id="enum-running-stats"></p></div><div class="cloud-panel"><div class="cloud-head"><strong>Final count · answer_query_v3</strong><select id="enum-final-layer" aria-label="Final representation layer"></select></div><canvas id="enum-final-canvas" aria-label="Interactive final count PCA3 point cloud"></canvas><p class="cloud-stats" id="enum-final-stats"></p></div></div></div><figcaption>左 panel 的点是 confirmation <code>item_end</code> states，颜色/数字表示 occurrence k；右 panel 是生成最终整数之前的 <code>answer_query_v3</code> states，颜色/数字表示 gold N。PC1–PC3 均由该层 discovery states 拟合 StandardScaler/PCA3 后定义；连线连接 confirmation centroids，仅用于观察顺序，不是注册的线性 counter axis。下方状态行给出样本数与前三 PC 的 discovery explained-variance ratio；视觉分簇不能推翻第 8 节 NCC causal null。</figcaption></figure><div class="subsection-conclusion"><strong>2.2 结论。</strong>3D manifold 对四格 count geometry 提供描述性直观；它与 held-out probe 一起证明信息可读，但不证明统一线性方向被模型因果使用。</div>
<h3>2.3 冻结层的数值结果</h3><div class="figure-primer"><div><strong>图中画什么</strong>四格 discovery-only 冻结层上的 running 与 final confirmation logistic balanced accuracy。</div><div><strong>坐标怎么读</strong>横轴是 Index/Bullet×Qwen/Gemma；纵轴 0–1 是十类 balanced accuracy，0.10 为 chance；同组两根柱是不同 endpoint。</div><div><strong>不能怎么读</strong>Running 与 final 的样本单位不同，柱高不能当作同一任务的难度差；跨模型冻结层编号也不可直接比较深浅。</div></div><figure class="paper-figure"><h3 class="figure-title">图 2c · Frozen report layers 的 confirmation balanced accuracy</h3>{frozen_representation_figure}<figcaption>横轴四组依次为 Index-Qwen、Index-Gemma、Bullet-Qwen、Bullet-Gemma；纵轴是 10-class confirmation balanced accuracy。绿色柱使用 <code>item_end</code> occurrence k，蓝色柱使用 <code>answer_query_v3</code> gold N。每个层均只由 discovery selection score 冻结，confirmation 没有参与选层。数值表同时给出层号、两种 endpoint 与统一 0.10 chance。</figcaption></figure>{representation_table}<div class="subsection-conclusion"><strong>2.3 结论。</strong>四格 frozen layers 的 held-out readout 均高于 chance；final count 尤其清楚，但这仍只是定位证据。</div><div class="section-conclusion"><strong>Experiment 2 结论。</strong>Running occurrence 与 final count 在四格 hidden states 中都可解码；真正的机制结论必须由下文 outcome-controlled interventions 给出。</div></section>
<section id="formation"><p class="eyebrow">03 · State formation</p><h2>3. Counter state：低维 commit gate 失败，但完整 content-bound item state 能直接重定向下一次 query</h2><h3>3.1 Discovery/confirmation scope：窄 component 与完整 state 回答的是不同命题</h3>
<div class="experiment-frame"><div><span class="experiment-label">实验目的</span>区分“窄 count-subspace 已足够”与“完整 item state 被使用”：把 donor commit 写入 receiver 后，下一次 targeted query 是否转向 donor successor record。</div><div><span class="experiment-label">实验设定</span>主恢复分析读取既有 confirmation raw shards 的 <code>full_donor_patch</code>、<code>self_patch</code> 与 norm-matched orthogonal arms；只比较相邻 distance-1 pairs，不新增 model forward。</div><div><span class="experiment-label">计算方法</span><span class="formula">Δattn<sub>self</sub> = TargetAttention(full donor patch) − TargetAttention(self patch)<br>Δattn<sub>orth</sub> = TargetAttention(full donor patch) − TargetAttention(orthogonal patch)</span>每个 true source seed 内先配对，再做 seed-cluster bootstrap 95% CI；CI 下界&gt;0 才过 strong direct gate。</div><div><span class="experiment-label">结果</span>四格 full-state direct gate 为 {full_gate_count}/4；原低维 <code>commit_to_retrieval_pass</code> 仍为 0/4。表 3 同时列出 attention、city log-odds、首 city 与 multihop readout。</div><div><span class="experiment-label">分析</span>结果说明能控制下一次 query 的对象至少包含完整 item/event state。由于 patch 含 city、score 与 syntax，它支持 content-bound state 使用，不能声称已隔离纯计数标量。</div><div><span class="experiment-label">简单例子</span>Receiver 已完成第 5 项，本应读 city 6；donor 表示完成第 6 项。若 full donor patch 使 next query 的 attention 从 record 6 转向 record 7，而 self/orthogonal 不会，则 direct edge 成立。</div></div>
<div class="figure-primer"><div><strong>图中画什么</strong>同一四格中，冻结 narrow loop、full-state direct、first-city adoption 与 depth-4 continuation 的注册 gate 指示。</div><div><strong>坐标怎么读</strong>横轴是四个 grammar×model 单元；纵轴只有 0/1，表示对应 strong gate 未过/通过，不是效应量。</div><div><strong>不能怎么读</strong>不同 gate 使用不同 outcome 与 control；1 不代表同样强，0 也不代表进程失败。</div></div><figure class="paper-figure"><h3 class="figure-title">图 3a · 从窄 count component 到完整 item state 的证据梯度</h3>{state_scope_gate_figure}<figcaption>横轴依次为四个 grammar×model 单元，纵轴是注册 gate 的二元结果。橙色 narrow loop 在四格均为 0；full-state teacher-forced direct、自由生成首 city 与 depth-4 exact donor prefix 在四格均为 1。该图只展示证据层级的 scope gradient，不把四类 estimand 压成共同效应量。Frame 04 的 endpoint/tail/full-item 数值和 Frame 16 positive control 在 Appendix 原样展开。</figcaption></figure><div class="subsection-conclusion"><strong>3.1 结论。</strong>证据随 state scope 扩大而稳定：现有正结果属于完整、content-bound state；低维 component 未被隔离。</div>
<h3>3.2 Frozen confirmation：完整 commit 是否直接改变下一次 query</h3><div class="figure-primer"><div><strong>图中画什么</strong>每格 full donor commit 相对 self 与 orthogonal control 的 targeted-attention effect。</div><div><strong>坐标怎么读</strong>横轴是 paired mean effect 与 95% CI；纵轴是 grammar×model×control。零线表示没有重定向。</div><div><strong>跨模型限制</strong>K、residual scale 与 architecture 不同，只读每行 CI 是否支持，不按绝对横距比较强弱。</div></div><figure class="paper-figure"><h3 class="figure-title">图 3b · Full-state commit patch 是否把下一次 query 转向 donor-successor record</h3>{commit_forest}<figcaption>横轴是 full donor patch 相对 self-patch 或 norm-matched orthogonal patch 的 paired targeted-attention effect，点为 true-source-seed 等权均值，线为 seed-cluster bootstrap 95% CI；纵轴列出四个 grammar×model 单元及两种 control。竖直零线表示没有重定向。正区间支持 direct edge，但 full-state scope 同时携带 content 与 progress，因此图证明的是 content-bound commit state 的因果使用，不是低维算术分量。</figcaption></figure>{commit_table}<div class="subsection-conclusion"><strong>3.2 结论。</strong>四格 full-state direct edge 均跨过 self 与 orthogonal controls；这是 Enumeration 闭环的 teacher-forced 内部边。</div><div class="section-conclusion"><strong>Experiment 3 结论。</strong>四格下一次 targeted query 都受完整 commit state 因果控制；原低维 gate 失败意味着目前不能把这一 state 简化为独立线性 count register。</div></section>
<section id="retrieval"><p class="eyebrow">04 · Targeted retrieval</p><h2>4. 下一条 city 仍需 content retrieval：visible ordinal 只减少地址不确定性</h2><h3>4.1 Registered query、target city 与冻结 bank</h3>
<div class="experiment-frame"><div><span class="experiment-label">实验目的</span>检验 frozen targeted-head bank 是否对下一条 city 的检索具有选择性必要性，并排除“Index ordinal 已经包含 city 内容”这一误解。</div><div><span class="experiment-label">实验设定</span>Index-Qwen/Gemma 分别冻结 K128/K8，Bullet-Qwen/Gemma 分别冻结 K96/K2。原 V6 在 registered query 关闭 selected bank；V2 先审计 query→first-city predictor token offset，再把同一 bank 持续关闭到 city prefix 末端。</div><div><span class="experiment-label">计算方法</span><span class="formula">selected damage = log P(target city|clean) − log P(target city|selected off)<br>specificity = mean log P(target city|random off) − log P(target city|selected off)</span>两项 95% CI 都在支持方向才称 strong specificity。</div><div><span class="experiment-label">结果</span>原 free-generation binary endpoint 中 Bullet 两格强支持，Index-Qwen 方向性、Index-Gemma null；连续 query-local 与 V2 sustained city-prefix 结果逐格列在表 4。Index 的 registered query 与 first-city predictor 并非所有 anchor 都 exact alias。</div><div><span class="experiment-label">分析</span>单 token lesion 在 Index 中可能过早，V2 更宽窗口检验的是同一目标的时间支持，而不是按结果更换 target。即使某格严格 gate 未过，也不能推出 ordinal 已替代 city retrieval。</div><div><span class="experiment-label">简单例子</span><code>6.</code> 只告诉模型“现在要第六条”；它不包含第六条实际是 Nanjing。若关闭 targeted bank 后 Nanjing 的 token probability 下降，而相同数量 random heads 不造成同样下降，才说明 content retrieval 具有选择性。</div></div>
<div class="subsection-conclusion"><strong>4.1 结论。</strong>四格 K 均由 discovery seed-equal 规则独立选择并在 confirmation 冻结；query 是预测目标 city 的注册位置，不是可见 ordinal 本身的语义替代。</div>
<h3>4.2 Head mask 到底在哪些 autoregressive steps 保持关闭</h3><div class="figure-primer"><div><strong>图中画什么</strong>Native behavior、Enumeration behavior、原 carrier、Index sustained likelihood 与 Bullet-Gemma fresh carrier 的四种时间支持窗口。</div><div><strong>坐标怎么读</strong>横轴从 registered query 向右依次是 city prefix、grammar carrier、item commit 与后续 cached decode；每行色条覆盖实际保持 mask 的类别区间。</div><div><strong>不能怎么读</strong>横轴是 token-phase 类别，不是毫秒或固定 token 数；不同请求的 city/carrier token 数可变。</div></div><figure class="paper-figure"><h3 class="figure-title">图 4a · Native-thinking 与 Enumeration 的 audited head-mask 时间范围</h3>{mask_scope_figure}<figcaption>行为级 necessity 在 Native 与 Enumeration 中都不是“只 mask 最后一个 query token”：两者均 teacher-force 到注册 anchor 后，把 frozen bank 在所有 cached decode forwards 持续关闭，manifest 中 <code>decode_head_ablation_steps=-1</code>。原 carrier assay 则严格为 query-local 一个 teacher-forced position；Index follow-up 扩到 target-city prefix；Bullet-Gemma fresh 扩到 final carrier token，其准确边界是 <em>every position from the registered query through the final grammar-carrier token, inclusive</em>。四者是不同 estimand，报告不再用“targeted-head bank 一直关闭/打开”一句话混写。</figcaption></figure>{head_mask_scope_table}<div class="subsection-conclusion"><strong>4.2 结论。</strong>你此前记得的“只在 N−1→N query 关闭会 rethink”对应行为 assay 的旧歧义；最终 behavior manifests 确实采用持续 decode mask，而 carrier 基线仍是单点。</div>
<h3>4.3 Frozen bank 的 city-retrieval necessity、specificity 与位置敏感性</h3><div class="figure-primer"><div><strong>图中画什么</strong>selected bank 对 target-city log probability 的损伤与 selected-vs-random specificity。</div><div><strong>坐标怎么读</strong>横轴为 seed-equal log-probability effect 及 95% CI；纵轴为四格的 damage/specificity estimand；零线为无效应。</div><div><strong>颜色怎么读</strong>绿色表示该单项 CI 在支持方向；橙色表示触零或方向不足，不是运行失败。</div></div><figure class="paper-figure"><h3 class="figure-title">图 4b · Frozen targeted bank 的 city-retrieval necessity 与 specificity</h3>{retrieval_forest}<figcaption>横轴是 target city 完整 token sequence 的 seed-equal log-probability effect，点为 paired mean，线为 seed-cluster bootstrap 95% CI；纵轴按 grammar×model 分列 selected damage 与 selected−random specificity。竖直零线表示没有选择性损伤。Qwen/Gemma 的 K 和 residual/attention scale 不同，不能用点的绝对距离排序模型机制强弱；只解释各自方向、区间和 frozen random-bank control。</figcaption></figure>{retrieval_table}<div class="subsection-conclusion"><strong>4.3 结论。</strong>Bullet 的行为级 failure contrast 最清楚；Index 的连续 city likelihood 需要覆盖实际 autoregressive predictor window 才稳定出现，说明的是时间定位，不是 ordinal 替代 retrieval。</div>
{index_sensitivity_section}
<h3>4.4 位置审计与可允许结论</h3><p class="main-note"><strong>位置审计。</strong>{esc(position_details)}。V2 改变的是 lesion 从 registered query 持续到同一 target city 的 autoregressive predictor，并没有按 intervention outcome 重选 K、层或 city。</p><div class="subsection-conclusion"><strong>4.4 结论。</strong>显式 <code>6.</code> 只约束 address/progress；具体 Nanjing 等 city token 仍需从 passage record 读取。bank-level necessity 不等于唯一单头 circuit。</div><div class="section-conclusion"><strong>Experiment 4 结论。</strong>具体 city 内容仍由 targeted retrieval 提供；Index ordinal 是 progress/address cue，不是 city memory。Index 的冻结 2×2 位置判定为：{esc(sensitivity_decision_summary) if sensitivity_complete else '尚未完成，原 primary 不改写'}。</div></section>
<section id="write"><p class="eyebrow">05 · Write and recurrent control</p><h2>5. 从 retrieval 写入 carrier，再由完整 item state 维持后续枚举</h2>
<h3>5.1 关闭 targeted bank 后，检索结果有没有写入 grammar carrier</h3><div class="experiment-frame"><div><span class="experiment-label">实验目的</span>检验 frozen targeted bank 被关闭后，实际 retrieved-event 所在的 grammar carrier hidden state 是否相对 clean 与 matched-random controls 发生选择性形变。</div><div><span class="experiment-label">实验设定</span>原 V6 assay 在 registered query 一个 teacher-forced position 关闭 bank；Bullet-Gemma 另有 fixed K2/source L16 的 fresh query-through-carrier replication。所有 source seeds 与 controls 在 outcome 前冻结。</div><div><span class="experiment-label">计算方法</span><span class="formula">deformation = d(h<sub>selected</sub>, h<sub>clean</sub>) − mean d(h<sub>random</sub>, h<sub>clean</sub>)</span>先在 true source seed 内汇总，再做 seed-cluster bootstrap 95% CI；不同 intervention windows 分开报告。</div><div><span class="experiment-label">结果</span>原 query-local carrier deformation 在三格形成注册通路；Bullet-Gemma 原单点 null 被保留，fresh wider-window replication 为 {esc(fresh_carrier_summary)}。</div><div><span class="experiment-label">分析</span>该结果支持 query→carrier 的局部 transport，但不能说明每个 token 都由同一 head bank 驱动；窗口差异说明写入具有 autoregressive 时间支持。</div><div><span class="experiment-label">简单例子</span>若第六条应写 Nanjing，关闭 bank 后 carrier 偏离 clean，而相同数量的 matched random heads 不造成同等偏离，则 Nanjing 检索结果确实进入了 carrier。</div></div>
<div class="figure-primer"><div><strong>图中画什么</strong>原 query-local selected-bank 对四格 grammar carrier 的 deformation。</div><div><strong>坐标怎么读</strong>横轴为 registered carrier deformation 与 95% CI；纵轴为四格；零线表示 selected mask 不比 control 更能改变 carrier。</div><div><strong>范围限制</strong>这是旧 query-local estimand；fresh Bullet-Gemma wider window 只在表中按独立列报告。</div></div><figure class="paper-figure"><h3 class="figure-title">图 5a · 关闭 targeted heads 后 carrier hidden state 如何变化</h3>{carrier_deformation_forest}<figcaption>横轴是原注册 query-local carrier deformation，点为 true-source-seed 等权均值，线为 95% bootstrap CI；纵轴为四个 grammar×model 单元。不同模型 residual scale 与 bank width 不同，因此只读每行方向和区间。Bullet-Gemma fresh 的 query-through-carrier 结果不投到该旧坐标中。</figcaption></figure><div class="subsection-conclusion"><strong>5.1 结论。</strong>targeted retrieval 的损伤可传播到 grammar carrier；Bullet-Gemma 的单点 null 由 fresh wider-window 支持补充，但不是对旧 estimand 的改写。</div>
<h3>5.2 在同一 query damage 下，恢复 clean carrier 能否救回 later commit</h3><div class="experiment-frame"><div><span class="experiment-label">实验目的</span>区分“carrier 只是伴随变化”与“carrier 是 later commit 的中介”：在同一 selected-bank damaged context 中，恢复 clean carrier 是否减少 item-end commit damage。</div><div><span class="experiment-label">实验设定</span>selected-mask、selected+clean-carrier clamp 与 matched random controls 使用同一 seed、同一 target record 和同一后续 teacher-forced trace；不跨 seed 比较。</div><div><span class="experiment-label">计算方法</span><span class="formula">restoration = damage(commit<sub>selected</sub>) − damage(commit<sub>selected+clean carrier</sub>)</span>正值表示 clean carrier 把 later commit 拉回 clean；fresh assay 另要求 matched-position specificity。</div><div><span class="experiment-label">结果</span>逐格 deformation、restoration、旧 decode-aligned diagnostic 与 fresh Bullet-Gemma 结果见表 5；所有原 null 保留。</div><div><span class="experiment-label">分析</span>这是受控局部 rescue，不是一次端到端完整 mediation；它与 5.1 共用损伤底座，避免把独立运行的正结果首尾拼接。</div><div><span class="experiment-label">简单例子</span>先遮住“查 Nanjing”的检索通道，再只把 clean carrier state 放回；若 item-end commit 比只遮住时更接近 clean，就说明 carrier 对写入有执行作用。</div></div>
<div class="figure-primer"><div><strong>图中画什么</strong>在相同 selected-mask damage 下，clean-carrier clamp 对 later item-end commit 的恢复量。</div><div><strong>坐标怎么读</strong>横轴为 restoration effect 与 95% CI；纵轴为四格；零线表示放回 clean carrier 没有改善 later commit。</div><div><strong>控制怎么读</strong>每个 effect 都在同 seed/target 内相对 selected-mask baseline 配对，不比较跨模型绝对残差距离。</div></div><figure class="paper-figure"><h3 class="figure-title">图 5b · 同一 head damage 下，clean carrier 能否救回 later commit</h3>{carrier_restoration_forest}<figcaption>横轴是 clean-carrier restoration，点和区间按 true source seed 等权聚合；纵轴是四个 grammar×model 单元。正区间支持 carrier→commit 的局部执行边。原 query-local 与 fresh wider-window 的结果在表中分列，不能把 fresh Bullet-Gemma 值代入旧四格 effect。</figcaption></figure>{carrier_table}<div class="subsection-conclusion"><strong>5.2 结论。</strong>carrier 不只是 readout：在受控 damage context 中恢复它可救回 later commit；证据强度与时间窗口仍按单元分账。</div>
<h3>5.3 完整 item state 是否改变下一项，并继续影响后续枚举</h3><div class="experiment-frame"><div><span class="experiment-label">实验目的</span>检验 full item state 的影响是否越过 teacher-forced score，真正改变首个生成 city，并在不再干预的情况下持续到 donor 后续序列。</div><div><span class="experiment-label">实验设定</span>V2 固定 Qwen L0、Gemma L21 的 full item span，测试 forward 5→6 与 backward 7→6；每方向 10 true-source seeds。V3 不新增 forward，对同一 240 条 generations 以冻结 any-surface parser 重算 depth 1/2/4。</div><div><span class="experiment-label">计算方法</span><span class="formula">depth = max d such that observed_known_city_ordinals[:d] = donor_path[:d]<br>paired effect = patched depth≥d rate − receiver-self donor-depth≥d rate</span>失败、非连续和截断保留在 unconditional denominator。</div><div><span class="experiment-label">结果</span>首 city 为 {esc(greedy_summary)}；四格 depth-4 主 gate 为 {multihop_strong_count}/4。Bullet 两格一旦采用 donor successor，后续 persistence 为 1.00；Index-Gemma 的 backward 方向较弱。</div><div><span class="experiment-label">分析</span>多步结果排除“patch 只瞬时提高一个 city token”的较窄解释；但 state 仍是 full content-bound vector，不能据此推断固定、context-free <code>+1</code> operator。Index-Gemma backward 从 depth≥1 的 0.90 降到 depth≥2 的 0.50、depth≥4 的 0.30，而 forward depth≥4 为 0.80。一个与该非对称衰减一致、但尚未被本实验单独验证的解释是：receiver history 中可见的 ordinal/address cue 仍锚定较晚进度，与 patch 注入的较早 donor hidden state 发生冲突；因此 patch 足以触发首个 city rewind，后续生成却可能被可见历史拉回或提前终止。该解释是待检验机制假说，不是由当前 transplant 已识别的原因；验证需要独立操纵 visible ordinal/history 与 patched state 的一致性。</div><div><span class="experiment-label">简单例子</span>预期 donor path 是 <code>[7,8,9,10]</code>：生成完全相同序列得 depth 4；<code>[7,9,10]</code> 只能得 depth 1；<code>[5,7,8,9,10]</code> 因第一项错误得 depth 0。</div></div>
<div class="figure-primer"><div><strong>图中画什么</strong>四格 donor→receiver full-item patch、receiver-self control 与 native-donor control 的首个已知 city donor-adoption rate。</div><div><strong>坐标怎么读</strong>横轴是四格；纵轴 0–1 是首 city 采用 donor successor 的 unconditional rate；每组颜色是不同 condition。</div><div><strong>不能怎么读</strong>native-donor control 本来就在 donor context 中，不是因果 effect；真正 paired contrast 是 patched 与 receiver-self。</div></div><figure class="paper-figure"><h3 class="figure-title">图 5c · Full-item state 是否改变自由生成的第一个 city</h3>{greedy_adoption_figure}<figcaption>横轴依次是 Index-Qwen、Index-Gemma、Bullet-Qwen、Bullet-Gemma；纵轴是首个可识别 city 采用 donor successor 的比例。绿色为 donor→receiver patch，灰色为 receiver-self control，橙色为 native-donor context。失败生成保留在分母；逐方向 paired effect 与 CI 见下表。</figcaption></figure>{greedy_direction_table}<div class="subsection-conclusion"><strong>5.3a 结论。</strong>四格 full-item patch 的首 city strong gate 均通过；这把 teacher-forced direct edge推进到自由生成行为。</div>{multihop_table}
<div class="figure-primer"><div><strong>图中画什么</strong>donor_to_receiver 在 depth≥1、2、4 的 unconditional exact-prefix rate。</div><div><strong>坐标怎么读</strong>横轴是注册 prefix depth；纵轴是 0–1 成功比例；两个 panel 分别为 Index/Bullet。</div><div><strong>线怎么读</strong>绿色 Qwen、紫色 Gemma；每个点以 20 个 seed×direction rows 为分母，不因失败删除。</div></div><figure class="paper-figure"><h3 class="figure-title">图 5d · Full-item transplant 后 donor-aligned continuation 能持续几步</h3>{multihop_figure}<figcaption>横轴是注册的 exact-prefix 深度阈值 d∈{{1,2,4}}，纵轴是 donor_to_receiver condition 在所有 20 个 seed-direction rows 中满足 depth≥d 的 unconditional rate；左 panel 为 Index，右 panel 为 Bullet，绿色/紫色分别是 Qwen/Gemma。点上数字是实际 rate。曲线下降表示首步采用后出现非连续或偏离；不允许跳过错误项、重排、去重或 parser repair。</figcaption></figure>{multihop_direction_table}<div class="subsection-conclusion"><strong>5.3b 结论。</strong>完整 item state 不只改变首个 city，也能在四格维持四步 donor path；方向异质性尤其是 Index-Gemma backward 衰减必须保留。</div><div class="section-conclusion"><strong>Experiment 5 结论。</strong>Enumeration 已形成 retrieval→carrier→content-bound commit→next query→continued enumeration 的候选 recurrent pathway；各边并非都在四格同强度通过，fresh 与 post-hoc 证据也严格分账。</div></section>
<section id="answer"><p class="eyebrow">06 · Logical chain D</p><h2>6. Terminal readout：trace source、terminal bridge 与 count-output execution 分层检验</h2>
<h3>6.1 Answer 到底在读 trace，还是回 prompt records 重数</h3><div class="experiment-frame"><div><span class="experiment-label">实验目的</span>复现 Native §6.1 的 source-necessity 问题：最终 count 主要依赖已经写出的 Enumeration trace，还是在 answer query 重新扫描 prompt records。</div><div><span class="experiment-label">实验设定</span>Frame 10 在四个 grammar×model 单元分别比较 clean、prompt-record blank、full-trace blank、prompt+trace blank，并以 length-matched ordinary blank 作控制；每个 confirmation condition 先在 true source seed 内汇总。</div><div><span class="experiment-label">计算方法</span><span class="formula">ΔExact(condition)=Exact(condition)−Exact(clean)<br>ΔlogP=log P(gold first answer token|condition)−log P(gold|clean)</span>Blank 保持序列长度和 query 位置；负 ΔExact/ΔlogP 表示该 source 被擦除后损伤。</div><div><span class="experiment-label">结果</span>四格的 prompt-record blank 均为 ΔExact=0，而 full-trace blank 均为 ΔExact=−1；gold first-answer-token log probability 的四格数值见表与 Appendix Frame 10。</div><div><span class="experiment-label">分析</span>当 trace 已经存在时，最终答案对 trace content 的行为依赖显著强于对原 prompt records 的依赖；这支持 trace→answer 路径，同时不排除 prompt 提供冗余 attention source。</div><div><span class="experiment-label">简单例子</span>若保留已列出的十条 city-score items、擦掉原 prompt records 后仍答对 10，而擦掉整段 items 后必错，则自然 answer 更像读取已写 trace，而不是在末尾从 prompt 重新枚举。</div></div>
<div class="figure-primer"><div><strong>图中画什么</strong>四格在两种 source blank 下，相对 clean 损失多少 exact-count rate。</div><div><strong>坐标怎么读</strong>横轴是四个 grammar×model 单元；纵轴从 0 到 1，是 <code>−ΔExact</code>，越高表示 source 越必要。</div><div><strong>不能怎么读</strong>两类 source token 数不同，柱高不是每 token 因果强度；它回答的是整类 source 的行为必要性。</div></div><figure class="paper-figure"><h3 class="figure-title">图 6a · Prompt-record 与 full-trace blank 对最终 exact count 的影响</h3>{answer_source_figure}<figcaption>横轴依次为 Index-Qwen、Index-Gemma、Bullet-Qwen、Bullet-Gemma；纵轴是相对 clean 的 exact-count loss，即 <code>−mean_delta_exact_count</code>。灰柱为只擦 prompt records，绿柱为擦完整 Enumeration trace。四格中前者为 0、后者为 1。该图汇总 Frame 10 的行为终点；first-token log-probability 与 source-attention 细节在表和 Appendix 原始帧中保留。</figcaption></figure>{answer_source_table}<div class="subsection-conclusion"><strong>6.1 结论。</strong>四格均支持 trace 是 final answer 的主要自然信息源；这与 Native-thinking 的 source-necessity 结论在分析角色和现象方向上对齐。</div>
<h3>6.2 最后一条 item/state 在完整 trace 中是否必要，单独恢复是否充分</h3><div class="experiment-frame"><div><span class="experiment-label">实验目的</span>把 whole-trace necessity 缩小到 terminal bridge，并严格区分完整上下文中的 necessity 与全局 scrambled baseline 上的 one-item sufficiency。</div><div><span class="experiment-label">实验设定</span>四格使用 Frame 11 的 terminal-token/state arms；Index 与 Bullet 共用 frozen seed-equal contract，但表面 marker、token 数和 state geometry 分别注册。Bullet 另有 clean-context local mediation 事后诊断。</div><div><span class="experiment-label">计算方法</span><span class="formula">necessity = U(clean)−U(terminal occluded)<br>global sufficiency = U(scrambled+terminal restored)−U(scrambled)</span><code>U=−|predicted count−gold count|</code>；换成全称即 <code>necessity = utility(clean) − utility(terminal occluded)</code>。两项基线不同，不能从 necessity 逻辑推出 sufficiency。</div><div><span class="experiment-label">结果</span>Necessity 为 {terminal_necessity_count}/4。Global one-item sufficiency 呈明确 grammar split：Index 2/2 支持，Bullet 0/2 未通过；不是“四格全 null”。Bullet clean-context local mediation 为 {local_terminal_count}/2。</div><div><span class="experiment-label">分析</span>Index 的显式 ordinal/terminal surface 使单 item 在 scrambled 背景中保留更多可执行进度线索；Bullet 缺少该显式地址，故全局 one-item restore 过强，但局部 clean-context relay 仍可成立。</div><div><span class="experiment-label">简单例子</span>Index 的最后一条若明写 <code>10.</code>，恢复它可能同时恢复 terminal progress cue；Bullet 只写 <code>-</code>，单独恢复末项未必足以重建此前累计历史。</div></div>
<div class="figure-primer"><div><strong>图中画什么</strong>四格 terminal necessity 与 global one-item sufficiency 的 paired effects。</div><div><strong>坐标怎么读</strong>横轴为 expected-count utility effect 与 95% CI；纵轴是 cell×estimand；零线表示无改善。</div><div><strong>为何同图</strong>同尺度便于看到 grammar split，但 necessity 与 sufficiency 的干预背景不同，不能比较为同一剂量。</div></div><figure class="paper-figure"><h3 class="figure-title">图 6b · Terminal necessity 与 global one-item sufficiency 的 grammar split</h3>{terminal_forest}<figcaption>横轴是 expected-count utility 的 seed-equal paired effect，线为 bootstrap 95% CI；纵轴分别列四格 necessity 与 global sufficiency。Necessity 在完整 trace 中移除 terminal state；sufficiency 在全局 scrambled trace 中只恢复一个 terminal item。四格 necessity 均为正区间；global sufficiency 仅 Index 两格通过、Bullet 两格不通过。</figcaption></figure>{terminal_table}<div class="subsection-conclusion"><strong>6.2 结论。</strong>Terminal state 在四格完整 trace 中必要；全局充分性不是统一 null，而是 Index positive / Bullet negative 的 grammar-dependent 结果。</div>
{answer_execution_section}
{terminal_relay_section}
<div class="section-conclusion"><strong>Experiment 6 结论。</strong>{esc(terminal_section_conclusion)}</div></section>
<section id="integrated-chain"><p class="eyebrow">Key loop</p><h2>关键闭环：改变完整 commit state 后，模型从 donor 指定的位置继续，并维持后续序列</h2><div class="core-claim"><strong>闭环判据。</strong>只有同时看到 next-query targeted attention/city likelihood 改变、首个生成 city 采用 donor successor、以及无再次 patch 的多步 exact-prefix continuation，才把路径称为行为级闭环。单独的 probe、NCC 或 teacher-forced margin 不足以完成闭环。</div>
<div class="table-wrap"><table><thead><tr><th>共同逻辑阶段</th><th>Enumeration 对应对象</th><th>当前最强结果</th><th>与 Native-thinking 的关系</th><th>仍不能推出</th></tr></thead><tbody><tr><td>Representation</td><td><code>item_end</code> running occurrence 与 <code>answer_query_v3</code> final count</td><td>四格 held-out 可解码；3D discovery-fit/confirmation-display</td><td>同样区分“可读”与“被使用”</td><td>前三 PC 或 probe weight 就是 causal counter axis</td></tr><tr><td>Targeted retrieval</td><td>registered query → next city record</td><td>Bullet 强；Index 2×2：{esc(sensitivity_decision_summary) if sensitivity_complete else '待完成'}</td><td>都需要按 target 变化的 content retrieval</td><td>visible ordinal 包含具体 city</td></tr><tr><td>Write / carrier</td><td>retrieved city → grammar carrier → commit</td><td>部分旧格支持；Bullet-Gemma fresh query-through-carrier 复现</td><td>都存在检索结果写入 later state 的局部边</td><td>所有 grammar/model 使用同一时间窗口</td></tr><tr><td>Commit → next query</td><td>full item state → donor-successor routing</td><td>{full_gate_count}/4 direct；{multihop_strong_count}/4 depth-4 cell gates</td><td>与 Native 的 content-bound state transplant 最接近</td><td>纯标量、memoryless <code>+1</code> operator</td></tr><tr><td>Terminal readout</td><td>terminal item/state → final Total</td><td>necessity 支持；global one-item sufficiency 过强</td><td>同样支持有序、部分、非唯一 relay</td><td>terminal state 单独编码全部历史</td></tr></tbody></table></div><div class="section-conclusion"><strong>关键闭环结论。</strong>Enumeration 与 Native-thinking 可以 claim 同一类 content-bound recurrent computational loop：两者共享 retrieval→write→commit→next-query→continuation 的因果拓扑。这个“same loop”结论不要求每个 grammar/model 的效应量、head bank、层、低维几何或底层 circuit 完全相同。</div></section>
<section id="ledger"><p class="eyebrow">07 · Evidence synthesis</p><h2>7. Evidence synthesis：按 claim 贡献排序，而不是按实验运行时间排序</h2><div class="evidence-ledger"><div><strong>最高等级 · 行为闭环</strong><p>full-state direct {full_gate_count}/4；first-city {esc(greedy_summary)}；depth-4 {multihop_strong_count}/4。支持 content-bound state 控制并维持枚举。</p></div><div><strong>局部边 · Retrieval/write/readout</strong><p>targeted city、carrier 与 terminal 的 selected-vs-control effects；部分为原 frozen confirmation，部分为事后诊断或 fresh outcomes。</p></div><div><strong>定位证据 · Representation</strong><p>running/final count 可解码与 3D manifold；只定位 candidate state，不独立贡献 causal claim。</p></div><div><strong>限制证据 · Nulls</strong><p>NCC、低维 loop、update/stop 与 global sufficiency 的 null 用于收窄表述，不当作流水线失败。</p></div></div><div class="reading-contract"><div class="contract-row"><strong>冻结 V6 基线</strong><span>每格 200 discovery + 100 confirmation，K、随机 bank、parser 与 replacement 规则不因 intervention outcome 改变。</span></div><div class="contract-row"><strong>Raw-arm 恢复*</strong><span>只读取既有 full_donor/self/orthogonal shards；无新增 model forward。</span></div><div class="contract-row"><strong>V2 事后扩展‡</strong><span>Index position/city-prefix 与四格 full-item greedy 在看过相关结果后注册，明确标为 post-hoc。</span></div><div class="contract-row"><strong>Item-end 2×2</strong><span>看过 primary weak effect 后提出，但新增 outcomes 前冻结；20 discovery seeds、fixed K、五个 contrasts，不替换 primary/confirmation。</span></div><div class="contract-row"><strong>Fresh carrier§</strong><span>Bullet-Gemma 十个 source requests 和 causal outcomes fresh；K2 bank discovery 不 fresh。</span></div><div class="contract-row"><strong>V3 reparse¶</strong><span>240 条冻结 generation 的 exact-prefix 聚合前冻结；曾看过一条 schema smoke row，故不是 prospective confirmation。</span></div></div><div class="section-conclusion"><strong>Experiment 7 结论。</strong>最强论文主张来自 full-state direct + behavior adoption + multihop persistence 的组合；probe 与局部 mediation 提供拓扑支持，null experiments 规定主张边界。</div></section>
<section id="extension-audit"><p class="eyebrow">08 · Extension audit</p><h2>8. 扩展问题审计：哪些 null 限制低维 counter 主张？</h2>
<h3>8.1 Low-dimensional loop、update 与 stop</h3><div class="experiment-frame"><div><span class="experiment-label">实验目的</span>检验一个冻结 low-dimensional count component 是否能完成 backstep、更新、继续枚举与 terminal stop，而无需完整 item content。</div><div><span class="experiment-label">实验设定</span>四格沿用 frozen count-subspace、direction、layer 与 controls；不因 full-state positive 结果重新选维度或层。</div><div><span class="experiment-label">计算方法</span>分别评估 backstep repeat、count-component restoration、terminal state→stop 与 nonterminal state→continue；group gate 要求注册方向及区间条件同时满足。</div><div><span class="experiment-label">结果</span>native_loop、update 与 stop 均为 0/4；具体四格状态见表 8a。</div><div><span class="experiment-label">分析</span>Full-state positive 与 low-dimensional null 不矛盾：前者允许 content/grammar/progress 混合，后者要求一个更窄、可迁移的 component 独立驱动更新与停止。</div><div><span class="experiment-label">简单例子</span>把完整“第 6 条 Nanjing:55”state 写入可能让模型继续到第 7 条；只写入投影出的“6”方向却未必同时恢复目标 city、语法边界和停止条件。</div></div>{state_update_table}<div class="subsection-conclusion"><strong>8.1 结论。</strong>目前能写“完整 state 维持枚举”，不能写“发现 context-free <code>c←c+1</code> 与 stop operator”。</div>
<h3>8.2 NCC：线性 centroid geometry 是否沿错误 count 方向移动</h3><div class="experiment-frame"><div><span class="experiment-label">实验目的</span>检验 selected intervention 是否不仅让 state 变化，而且按 discovery-fitted nearest-centroid geometry 朝错误 count centroid 移动，并超过 random-bank control。</div><div><span class="experiment-label">实验设定</span>每格在冻结层使用 discovery-fitted count centroids；confirmation 只计算 selected margin loss 与 selected−random specificity，不重新选择层或方向。</div><div><span class="experiment-label">计算方法</span><span class="formula">margin(h)=d(h,wrong centroid)−d(h,correct centroid)<br>selected loss = margin(clean)−margin(selected)<br>specificity = selected loss−mean random loss</span>clean readout validity 与两条 interval gate 必须同时通过。</div><div><span class="experiment-label">结果</span>四格 NCC geometry 均未通过注册 strong gate；表 8b 与图 7 保留方向、区间和 selected-vs-random 结果。</div><div><span class="experiment-label">分析</span>3D point clouds 看起来分簇只说明 variance geometry；NCC 问的是 intervention displacement 是否沿一个统一 discovery centroid direction，命题更窄。</div><div><span class="experiment-label">简单例子</span>红蓝点云可以在 PC 图上分开，但一次 patch 可能沿第三个未画出的方向移动，或同时改变 city/syntax；因此“看起来分簇”不保证 NCC margin 按预期变化。</div></div>
<div class="figure-primer"><div><strong>图中画什么</strong>四格 selected NCC margin loss 与 selected−random specificity。</div><div><strong>坐标怎么读</strong>横轴为 NCC margin effect 及 95% CI；纵轴为 cell×estimand；零线表示无方向性损伤。</div><div><strong>与 3D 的区别</strong>NCC 用全部注册表示空间和 frozen centroids；PC1–PC3 只用于描述性展示。</div></div><figure class="paper-figure"><h3 class="figure-title">图 7 · Frozen NCC count geometry 的 selected loss 与 specificity</h3>{ncc_forest}<figcaption>横轴是 discovery-fitted NCC geometry 下的 margin loss 或 selected−random specificity，点为 confirmation seed mean，线为 95% bootstrap CI；纵轴是四个 grammar×model 的独立 estimands。竖直零线表示没有支持方向。绿色只表示该单项区间方向，正式 strong gate 还要求 clean readout validity 与两项条件共同满足；本报告四格最终均为 null。NCC 坐标系跨 grammar/model 不同，不能合并 raw margin。</figcaption></figure>{ncc_table}<div class="subsection-conclusion"><strong>8.2 结论。</strong>现有数据不支持统一线性 centroid counter geometry；这限制低维解释，但不否定完整 hidden-state vector 的 causal efficacy。</div><div class="section-conclusion"><strong>Experiment 8 结论。</strong>扩展审计强化的是边界：正结果属于 full-state、content-bound recurrent control；低维 loop、update/stop 与 NCC 仍需新的、预注册实验。</div></section>
<section id="limitations"><p class="eyebrow">09 · What remains</p><h2>9. What remains：与 Native-thinking 对齐的是机制拓扑，不是所有实现细节</h2><ol><li><strong>Full-state 不等于 pure count。</strong>Item span 同时包含 city、score、marker 与局部上下文；需要 factorized content/progress interventions 才能隔离纯 count component；这一更窄问题不影响 same recurrent loop claim。</li><li><strong>事后扩展不是 fresh confirmation。</strong>V2 Index position/full-item greedy、冻结 2×2 item-end sensitivity 与 V3 multihop reparse 都有明确 post-hoc/discovery-only 标签；不能与 frozen V6 confirmation 混写。</li><li><strong>Grammar heterogeneity 必须保留。</strong>Index-Gemma backward persistence 较弱，Bullet-Gemma 原始 strict accuracy也最低；不应用 4/4 cell gate 掩盖方向和行为差异。可见 ordinal/history 与 rewind hidden state 的冲突是与结果一致的候选解释，但在未做二者独立操纵前仍是待检验假说。</li><li><strong>Carrier 时间支持不统一。</strong>Query-local、query-through-city-prefix 与 query-through-carrier 是不同 intervention scope；fresh carrier 目前只覆盖 Bullet-Gemma。</li><li><strong>Terminal relay 不等于完整或唯一中介。</strong>Exact 2×3 assay 报告 partial mediation 与 residual；global one-item sufficiency 的 grammar split 仍禁止把 terminal state 写成独立包含全部历史。</li><li><strong>3D 不是 causal geometry。</strong>PCA3 的轴、符号与视觉分簇是描述性的；NCC null 不能被图形观感覆盖。</li><li><strong>跨模型效应量不可直接排序。</strong>层数、head 数、K、tokenizer 与 residual scale 不同；本文只比较每格的 treatment−control 方向和区间。</li></ol><div class="section-conclusion"><strong>Experiment 9 结论。</strong>现有结果支持 Enumeration 与 Native-thinking 共享同一类 content-bound recurrent counting loop；其科学含义是完整 item state 能重定向下一次检索并维持后续枚举。唯一低维寄存器、memoryless <code>+1/stop</code> operator 或相同底层 circuit 属于额外实现假设，不是该循环 claim 的组成部分。</div></section>
<section id="appendix" class="paper-appendix"><p class="eyebrow">Appendix · Parser, calculations and provenance</p><h2>Appendix：所有新概念、计算规则、失败样本与底层文件</h2>
<div class="appendix-block"><h2>先定义本文所有核心对象</h2><div class="reading-contract"><div class="contract-row"><strong>Original frozen slot</strong><span>主运行预先分配的 seed/count cell；其行为正确与否在任何 reserve replacement 之前判定。</span></div><div class="contract-row"><strong>Formal fixed-quota cell</strong><span>通过 strict parser 的 analysis slot；原失败按 outcome-blind reserve mapping 替换，用于保证每个 mechanism panel 的样本数完整。</span></div><div class="contract-row"><strong>Strict parser</strong><span>决定整条输出是否行为成功；同时验证 grammar、Total、有序 pairs 与 marker。</span></div><div class="contract-row"><strong>Hybrid semantic parser</strong><span>决定 item 字符 spans 与 causal token sites；它不能把 synthetic fallback 伪装成 strict one-to-one trace。</span></div><div class="contract-row"><strong>Relay full-NA seed</strong><span>该预注册 true source seed 的所有 registered pairs 都不满足该 cell 明确标注的 suffix geometry；seed 留在 planned audit 中但不进入仅对 geometry-eligible seeds 定义的 relay 数值 estimand，也不得用 reserve 或 outcome 结果替换。</span></div><div class="contract-row"><strong>Task-adapted relay</strong><span>原 suffix8 因 Bullet token-span 支持不足而产生 geometry N/A 后，为 Bullet-Qwen/Gemma 一致冻结 suffix4；它保留原 cohort、pair rule、layers、2×3 estimands 与 gates，且不能替换或重命名原 suffix8 结果。</span></div><div class="contract-row"><strong>Direct edge</strong><span>teacher-forced full-state patch 对 next-query attention/city likelihood 的配对作用。</span></div><div class="contract-row"><strong>Behavioral continuation</strong><span>自由生成首个 known city 与后续 exact donor prefix；不允许 parser repair。</span></div></div></div>
<div class="appendix-block"><h3>A.1 严格响应语法：接受什么，拒绝什么</h3>{parser_grammar_table}<div class="example-box"><strong>规则示例（不是模型输出）。</strong><pre>ACCEPT Index<br>1. Taipei: 51<br>2. Nanjing: 55<br>Total: 2<br><br>ACCEPT Bullet<br>- Taipei: 51<br>- Nanjing: 55<br>Total: 2<br><br>REJECT<br>1) Taipei: 51<br>- Nanjing: 55<br>Total: 2</pre></div><div class="subsection-conclusion"><strong>A.1 结论。</strong>行为 PASS 是整条精确匹配，不是模糊抽取或只看最终 Total。</div></div>
<div class="appendix-block"><h3>A.2 Formal causal cohort：七个条件取 AND</h3>{parser_gate_table}<span class="parser-code">strict_causal_eligible = registered_success ∧ enumeration_format_compliant ∧ listed_total_matches_length ∧ exact_ordered_gold_pairs ∧ marker_kind_compliant ∧ parser_forward_one_to_one ∧ item_count_matches_gold</span><div class="parser-warning"><strong>Gold firewall。</strong>Gold records 只验证、定位与建立 registry；Gold N 和 final Total 不会构造、补齐或选择 item sequence。</div><div class="subsection-conclusion"><strong>A.2 结论。</strong>缺失、重复、逆序或 marker 错误都保留为失败，不会为了贴近 gold 而被 parser 修复。</div></div>
<div class="appendix-block"><h3>A.3 Hybrid semantic span 的选择顺序</h3><ol><li><strong>Rank-supported episode：</strong>选择最长连续 1..M，长度相同取最早。</li><li><strong>Structural extension：</strong>只有 structural span 以 rank episode 为精确前缀且新增 city 时扩展。</li><li><strong>Structural fallback：</strong>无 rank episode 时保留首个可靠、terminated gold-city list；Bullet 常走此层。</li><li><strong>Synthetic fallback：</strong>只供审计，标记 <code>synthetic_unverified</code>/<code>trace_one_to_one=false</code>，不得进入 causal cohort。</li></ol><div class="subsection-conclusion"><strong>A.3 结论。</strong>Semantic parser 能定位自然表面变化，但 V6 strict grammar 仍是 formal eligibility 的最终 firewall。</div></div>
<div class="appendix-block"><h3>A.4 字符 site 与 token site</h3>{parser_site_table}<span class="parser-code">character site = [char_start,char_end)<br>endpoint token = prefix_token_count − 1<br>alignment ∈ {{literal_baseline_token_prefix, text_exact_boundary_retokenization}}</span><p><code>answer_query_v3</code> 包含 <code>Total:</code> 后空白并停在最终整数首字符之前；即使 running parser miss，answer locators 仍独立执行。</p><div class="subsection-conclusion"><strong>A.4 结论。</strong>所有 causal sites 都由 exact text boundary 编译，无法精确对齐即 ineligible，不做 nearest-token 猜测。</div></div>
<div class="appendix-block"><h3>A.5 1,200 个原始 slots 与 replacement 审计</h3>{parser_cohort_table}<p>原 fixed slots 有 {esc(parser_data['original_strict_failure_count'])} 条 strict failures；寻找 replacement 时另有 {esc(parser_data['failed_reserve_attempt_count'])} 条 ordinary reserve candidates 失败。最终 unresolved=0 只说明 fixed quota 补齐。</p>{replacement_policy_table}<details><summary>展开全部 {esc(parser_data['original_strict_failure_count'])} 条原 strict failure 与 replacement mapping</summary>{parser_failure_table}</details><details><summary>展开全部 {esc(parser_data['failed_reserve_attempt_count'])} 条 ordinary reserve candidate 失败</summary>{parser_failed_reserve_table}</details><div class="subsection-conclusion"><strong>A.5 结论。</strong>失败和补 seed 尝试全部可审计；replacement 不读取 intervention outcomes，也不静默排除负结果。</div></div>
<div class="appendix-block"><h3>A.6 Full-item multihop 的行为 parser</h3><span class="parser-code">expected donor path = [donor_successor,…,gold_count]<br>depth = longest d with observed[:d] = expected[:d]<br>registered d ∈ {{1,2,4}}; skip/reorder/deduplicate/repair forbidden</span>{parser_multihop_taxonomy_table}<details><summary>展开 fixed lowest-seed 的 24 个三条件例子</summary>{parser_multihop_example_table}</details><div class="subsection-conclusion"><strong>A.6 结论。</strong>Multihop 使用 any-surface known-city exact prefix；失败、非连续与截断都留在 unconditional denominator。</div></div>
{sensitivity_appendix_block}
<div class="appendix-block"><h3>A.8 Parser 与 evidence provenance</h3>{parser_source_table}<details><summary>展开全部 evidence path 与 SHA-256</summary>{source_table}</details><div class="reading-contract"><div class="contract-row"><strong>Suite audit</strong><span>{esc(data['completion_audit']['status'])}；ordinary original failures={esc(data['completion_audit']['ordinary_failure_count'])}；coherent replacement trajectories={esc(data['completion_audit']['coherent_replacement_trajectory_count'])}。</span></div><div class="contract-row"><strong>V2/V3</strong><span>V2={esc(data['followup']['status'])}；V3={esc(data['followup_v3']['status'])}；240 trial rows / 80 seed-direction rows；report rewrite 无新 model forward。</span></div><div class="contract-row"><strong>Template provenance</strong><span>CSS 与 section/experiment/figure/conclusion topology 来自本地 Native-thinking report SHA-256 <code>{esc(data['native_report_sha256'])}</code>；科学数字仍只来自 Enumeration frozen artifacts。</span></div></div></div>
<div class="appendix-block"><h3>A.9 Native-thinking ↔ Enumeration：mechanism-claim 对照</h3>
<p>本对照回答的不是“两套模型是否使用相同层或相同 heads”，而是 Enumeration 是否足以作为 Native-thinking 的 <em>mechanistically faithful controlled proxy</em>。比较单位固定为计算角色、干预后的因果方向和 recurrent behavioral consequence；它是一项 claim-level synthesis，不新增统计 gate，也不重新解释任何 frozen null。</p>
<div class="table-wrap"><table class="wide"><thead><tr><th>比较维度</th><th>Native-thinking 的 claim / evidence scope</th><th>Enumeration 的对应 claim / evidence</th><th>claim-level 判断</th></tr></thead><tbody>
<tr><td>核心对象</td><td>分布式、content-bound 的 event/progress state；当前 state 决定从哪里继续。</td><td>完整 item commit state 同时含 city、score、grammar 与 progress，并控制下一次 targeted query。</td><td><span class="pill support">同一机制对象层级</span>；均非单 token 或纯标量定义。</td></tr>
<tr><td>Targeted retrieval</td><td>registered query 通过 model-specific frozen head bank 读取下一条具体 record。</td><td>Bullet 原窗口支持；Index 用同一冻结 bank 在 city-prefix support window 与冻结 2×2 anchor sensitivity 中补足时间定位。</td><td><span class="pill support">同一计算角色</span>；最佳 token window 可不同。</td></tr>
<tr><td>Retrieval → carrier → commit</td><td>检索到的 event 写入 grammar carrier 与 later event state；局部 damage/rescue 支持 transport edge。</td><td>原 V6 三格通过，Bullet-Gemma 由 fresh query-through-carrier damage/rescue 补强。</td><td><span class="pill support">同一局部因果方向</span>；四格证据来源标签不同。</td></tr>
<tr><td>Commit → next retrieval</td><td>content-bound item/event-state transplant 重定向下一项读取。</td><td>full-state teacher-forced direct gate 为 {full_gate_count}/4，均相对 self 与 norm-matched orthogonal controls 比较。</td><td><span class="pill support">循环闭合的直接边对齐</span>。</td></tr>
<tr><td>Recurrent continuation</td><td>成功的 state transplant 不只改变第一项，还会在无再次干预时沿 donor trajectory 继续。</td><td>first-city strong gate 为 {greedy_strong_count}/4；冻结 240-generation reparse 的 depth-4 cell gate 为 {multihop_strong_count}/4。</td><td><span class="pill support">相同 recurrent behavioral consequence</span>。</td></tr>
<tr><td>Terminal readout</td><td>trace state 经 post-terminal suffix / answer query 构成有序、部分、非唯一的 answer pathway。</td><td>source/terminal necessity 为 {terminal_necessity_count}/4；same-trial 2×3 relay 有 {extension_relay_estimable_count}/4 格在逐格标注的 geometry 下可估计，可估计格的 primary partial-mediation gate 为 {extension_partial_mediation_count}/{extension_relay_estimable_count}。{esc(relay_geometry_design_text)}；{esc(original_suffix8_audit_text)}，不把 N/A 写成零效应。</td><td><span class="pill partial">同一 partial-relay estimator / claim boundary</span>；task adaptation 单列，不声称 complete mediation。</td></tr>
<tr><td>自然性与覆盖范围</td><td>自然 no-index 的主因果 claim 主要限于 Qwen；Gemma 的 next-item 结果更多来自受控 surrogate。</td><td>Index/Bullet × Qwen/Gemma 四个严格受控 Enumeration cells 均支持 full-state loop；Index ordinal 只提供 address/progress cue，不提供 city content。</td><td><span class="pill diagnostic">功能 claim 对齐，证据场景不同</span>；Enumeration 是 controlled proxy，不冒充自然 trace。</td></tr>
<tr><td>不属于主 claim 的实现假设</td><td>不要求唯一 circuit、content-free scalar register 或 context-invariant <code>+1</code> operator。</td><td>NCC、low-dimensional loop、update/stop 的 null 被保留，但不作为 same-loop claim 的必要条件。</td><td><span class="pill support">结论边界一致</span>。</td></tr>
</tbody></table></div>
<h4>Mechanistic fidelity 的操作化检查</h4><div class="table-wrap"><table><thead><tr><th>必要现象</th><th>Enumeration 当前状态</th><th>对 proxy claim 的贡献</th></tr></thead><tbody>
<tr><td>F1 · 下一条具体内容需要 targeted retrieval</td><td>支持；Index/Bullet 的 support window 分账报告</td><td>排除“ordinal 自身包含 city content”</td></tr>
<tr><td>F2 · retrieved event 写入 carrier / later state</td><td>支持；原三格 + Bullet-Gemma fresh 补强</td><td>建立 retrieval→write 边</td></tr>
<tr><td>F3 · commit state 因果控制 next query</td><td>{full_gate_count}/4 direct gates</td><td>建立 write→next-retrieval 反馈边</td></tr>
<tr><td>F4 · 无再次 patch 时维持 donor-aligned trajectory</td><td>{multihop_strong_count}/4 depth-4 cell gates</td><td>把单步 steering 升级为 recurrent loop</td></tr>
<tr><td>F5 · trace state 参与 final answer readout</td><td>{terminal_necessity_count}/4 necessity；relay 只作 partial/estimable-scope claim</td><td>把循环状态连接到最终输出，同时保留并行路径</td></tr>
</tbody></table></div>
<div class="core-claim"><strong>对照结论。</strong>Enumeration 满足作为 Native-thinking 受控机制代理所需的核心功能对应：两者共享 targeted retrieval → carrier/write → content-bound commit → next targeted query → donor-aligned continuation 的 recurrent computational loop。这里“same”指 shared causal topology 与相同的干预后行为后果；不要求相同层、heads、效应量、低维表示或底层 neural circuit。</div>
<div class="plain-language"><strong>Paper-ready claim.</strong> <em>Enumeration provides a mechanistically faithful controlled proxy for Native-thinking by reproducing the same distributed, content-bound recurrent counting loop: the committed item state causally redirects the next retrieval step and sustains the subsequent donor-aligned enumeration trajectory. This correspondence is defined at the level of computational and causal topology, not identical neural implementation.</em></div>
<div class="subsection-conclusion"><strong>A.9 结论。</strong>二者可以使用同一个机制主张：完整、content-bound 的 committed state 控制下一次检索并维持后续枚举；Enumeration 的作用是对该 Native-thinking loop 做可控、可重复的机制模拟。</div></div>
<div class="section-conclusion"><strong>Appendix 结论。</strong>报告中的每个 accuracy、PASS/null、token site、replacement 与 multihop depth 都有明确计算合同和 SHA-256 provenance；A.9 进一步明确二者共享的是 recurrent computational loop 与 causal topology，版式同构或 proxy 身份都不意味着借用 Native 的科学数字。</div></section>
</main><footer><p>Self-contained HTML · no external assets · Native-thinking CSS/topology mirrored · Enumeration scientific content from sealed V6 + V2/V3 artifacts.</p></footer><script id="report-manifest" type="application/json">{embedded}</script><script>{manifold_script}</script></article></body></html>"""

    css = r"""
.metrics.v2{grid-template-columns:repeat(auto-fit,minmax(190px,1fr))}
.paper-figure{margin:28px 0;padding:18px;background:var(--surface);border:1px solid var(--line);box-shadow:0 9px 30px var(--shadow)}.paper-figure figcaption{margin:0 0 12px;color:var(--muted);font-size:12px}.paper-chart{display:block;width:100%;height:auto;background:#fff}.plot-bg{fill:#fbfcfe;stroke:#d0d5dd}.grid{stroke:#eaecf0;stroke-width:1}.heat-title{font:750 13px Aptos,"Segoe UI",sans-serif;fill:#344054}.axis-label,.legend-label,.tick,.chart-axis,.chart-value{font:11px Aptos,"Segoe UI",sans-serif;fill:#667085}.chart-value{font-weight:750;fill:#344054}.geometry-viewer{margin:30px 0;padding:18px;background:var(--surface);border:1px solid var(--line);box-shadow:0 9px 30px var(--shadow)}.geometry-toolbar{display:flex;flex-wrap:wrap;gap:12px;align-items:end;margin-bottom:14px}.geometry-toolbar label{display:grid;gap:5px;color:var(--muted);font-size:11px;font-weight:750}.geometry-toolbar select,.geometry-toolbar button,.cloud-head select{min-height:36px;padding:7px 10px;border:1px solid #cfd6da;border-radius:4px;background:#fff;color:var(--ink);font:12px Aptos,"Segoe UI",sans-serif}.geometry-toolbar button{cursor:pointer;background:var(--soft);font-weight:750}.cloud-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.cloud-panel{margin:0;border:1px solid var(--line);background:#fbfcfe}.cloud-head{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:12px 13px;border-bottom:1px solid var(--line);background:#f3f6f5}.cloud-head strong{font-size:13px}.cloud-panel canvas{display:block;width:100%;height:430px;touch-action:none}.cloud-stats{min-height:46px;margin:0;padding:10px 13px;color:var(--muted);font-size:11px;border-top:1px solid var(--line)}.trajectory-strip{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin:20px 0}.trajectory-strip span{padding:12px;text-align:center;background:#fff;border:1px solid var(--line);font:750 12px ui-monospace,SFMono-Regular,Consolas,monospace}.trajectory-strip .active{border-color:var(--accent);background:var(--soft);color:var(--accent)}
.code-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:20px 0}.code-card{background:#18212a;color:#e8eef2;border-top:4px solid var(--accent)}.code-card h4{margin:0;padding:12px 15px;color:#b9d9d3;font-size:12px}.code-card pre{margin:0;padding:0 15px 16px;white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.65 ui-monospace,SFMono-Regular,Consolas,monospace}.parser-flow{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:1px;margin:22px 0;background:var(--line);border:1px solid var(--line)}.parser-flow span{padding:14px;background:var(--surface);font-size:12px;font-weight:750}.parser-flow span+span:before{content:"→";margin-right:8px;color:var(--accent)}
:root{--ink:#17202a;--muted:#5d6875;--line:#d9e0e4;--paper:#f7f8f6;--surface:#ffffff;--soft:#edf4f2;--accent:#176b61;--accent-2:#335f77;--warn:#8a5a16;--null:#7a4141;--shadow:rgba(33,48,58,.08)}
*{box-sizing:border-box}code{overflow-wrap:anywhere;word-break:break-word}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:Aptos,"Segoe UI","Noto Sans SC",sans-serif;line-height:1.72}.page{width:min(1240px,calc(100% - 40px));margin:0 auto;padding:48px 0 92px}.hero{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(280px,.7fr);gap:46px;padding:50px 0 42px;border-bottom:1px solid var(--line)}.eyebrow{margin:0 0 10px;color:var(--accent);font:800 12px/1.3 ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.08em;text-transform:uppercase}h1{max-width:820px;margin:0;font-size:clamp(38px,5.3vw,70px);font-weight:780;letter-spacing:-.045em;line-height:1.02}h1 span{display:block;margin-top:12px;color:var(--muted);font-size:.33em;font-weight:620;letter-spacing:.01em;line-height:1.45}.hero-summary{align-self:end;padding:20px 0;border-top:3px solid var(--accent);border-bottom:1px solid var(--line)}.hero-summary strong{display:block;font-size:28px;line-height:1.1}.hero-summary p{margin:9px 0 0;color:var(--muted);font-size:13px}.toc{position:sticky;top:0;z-index:2;display:flex;gap:4px;overflow-x:auto;margin:0 -12px;padding:10px 12px;background:rgba(247,248,246,.94);border-bottom:1px solid var(--line);backdrop-filter:blur(12px)}.toc a{flex:0 0 auto;padding:7px 10px;color:var(--muted);font-size:12px;text-decoration:none;border-bottom:2px solid transparent}.toc a:hover,.toc a:focus{color:var(--accent);border-color:var(--accent)}section{padding:56px 0;border-top:1px solid var(--line)}section:first-of-type{border-top:0}.section-head{display:grid;grid-template-columns:190px 1fr;gap:26px;margin-bottom:24px}.section-no{color:var(--accent);font:750 12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.06em;text-transform:uppercase}.section-head h2{margin:0;max-width:900px;font-size:clamp(26px,3.2vw,42px);line-height:1.14;letter-spacing:-.025em}.lede{max-width:850px;color:#394655;font-size:17px}.claim{max-width:960px;margin:24px 0;padding:18px 20px;background:var(--soft);border-left:4px solid var(--accent)}.claim strong{color:#0e554e}.qualification{max-width:980px;margin:20px 0;padding:16px 18px;background:#f7f2e9;border-left:4px solid var(--warn);color:#594321}.definition-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1px;margin:24px 0;background:var(--line);border:1px solid var(--line)}.definition{padding:18px;background:var(--surface)}.definition dt{font-weight:800}.definition dd{margin:6px 0 0;color:var(--muted);font-size:13px}.metrics{display:grid;grid-template-columns:1.25fr .85fr .85fr;gap:1px;margin:24px 0;background:var(--line);border:1px solid var(--line)}.metric{padding:22px;background:var(--surface)}.metric b{display:block;font-size:30px;line-height:1.05}.metric span{color:var(--muted);font-size:12px}.table-wrap{overflow-x:auto;margin:22px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}table{width:100%;border-collapse:collapse;background:var(--surface);font-size:13px}th,td{padding:13px 14px;text-align:left;vertical-align:top;border-bottom:1px solid #e7ebed}th{color:#47525e;font-size:11px;letter-spacing:.045em;text-transform:uppercase;background:#f0f3f2}tbody tr:last-child td{border-bottom:0}td code{font-size:11px;overflow-wrap:anywhere}.wide{min-width:1050px}.hash-table{min-width:1050px}.pill{display:inline-block;margin:2px 2px;padding:2px 7px;border:1px solid currentColor;border-radius:999px;font:750 10px/1.35 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:nowrap}.support{color:#12665c;background:#eef8f5}.diagnostic{color:#2f6279;background:#edf5f8}.restored{color:#385b84;background:#eef2f9}.partial{color:#82601d;background:#fbf6e9}.null{color:#7a4141;background:#faf0f0}.chain{margin:28px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}.chain figcaption{padding:13px 0;color:var(--muted);font-size:12px}.chain-lane{display:grid;grid-template-columns:160px minmax(150px,1fr) 32px minmax(150px,1fr) 32px minmax(150px,1fr) 32px minmax(150px,1fr);align-items:stretch;border-top:1px solid var(--line);background:var(--surface)}.lane-name,.node{padding:15px 13px}.lane-name{background:#eef2f1}.lane-name strong,.lane-name span,.node b,.node .micro{display:block}.lane-name span,.micro{margin-top:4px;color:var(--muted);font-size:11px}.node b{margin-bottom:6px}.arrow{display:grid;place-items:center;color:var(--accent);font-size:24px;background:#f4f6f5}.mechanism-note{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin:22px 0}.mechanism-note>div{padding:18px;border-top:3px solid var(--accent-2);background:var(--surface)}.mechanism-note h3{margin:0 0 8px;font-size:16px}.mechanism-note p{margin:0;color:var(--muted);font-size:13px}.formula{display:block;max-width:900px;margin:16px 0;padding:15px 18px;background:#eef2f4;border-left:3px solid var(--accent-2);font:13px/1.65 ui-monospace,SFMono-Regular,Consolas,monospace;overflow-x:auto}.ledger{display:grid;grid-template-columns:160px 1fr;gap:1px;margin:20px 0;background:var(--line);border:1px solid var(--line)}.ledger dt,.ledger dd{margin:0;padding:14px;background:var(--surface)}.ledger dt{font-weight:780}.ledger dd{color:var(--muted)}details{margin:12px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line);background:var(--surface)}summary{cursor:pointer;padding:15px 4px;font-weight:760}details>div{padding:0 4px 18px}.conclusion{max-width:980px;margin-top:24px;padding:18px 20px;background:var(--soft);border-left:4px solid var(--accent)}.conclusion strong{color:#0e554e}.footnote{color:var(--muted);font-size:12px}.footer{padding:30px 0 0;color:var(--muted);font-size:12px;border-top:1px solid var(--line)}@media(max-width:850px){.page{width:min(100% - 24px,1240px);padding-top:20px}.hero{grid-template-columns:1fr;gap:28px;padding-top:30px}.section-head{grid-template-columns:1fr;gap:7px}.definition-grid,.metrics,.mechanism-note,.ledger,.cloud-grid{grid-template-columns:1fr}.chain-lane{grid-template-columns:1fr}.arrow{min-height:30px;transform:rotate(90deg)}.toc{margin:0 -6px;padding-inline:6px}.cloud-panel canvas{height:390px}}@media print{body{background:#fff}.page{width:100%;padding:0}.toc{display:none}section{break-inside:auto}.table-wrap,.chain,details,.paper-figure,.geometry-viewer{break-inside:avoid}details>div{display:block}.hero{padding-top:0}.geometry-viewer{box-shadow:none}}
@media(max-width:850px){.code-grid,.parser-flow{grid-template-columns:1fr}.parser-flow span+span:before{content:"↓"}}
"""

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NiaH Enumeration：完整叙事型机制复现报告</title><link rel="icon" href="data:,"><style>{css}</style></head>
<body><article class="page"><header class="hero"><div><p class="eyebrow">Realistic CoT NiaH · V6 Enumeration mechanism</p>
<h1>Enumeration 如何维持计数并选择下一项<span>Qwen3-8B × Gemma4-E4B · Index × Bullet · 与 Native-thinking 同构的完整机制矩阵、图形与审计</span></h1></div>
<div class="hero-summary"><strong>{full_gate_count}/4 direct · {multihop_strong_count}/4 multihop</strong><p>full-state teacher-forced direct edge、first-city adoption 与四步 free-generation continuation 分层报告；原低维 null 不被覆盖。</p></div></header>
<nav class="toc" aria-label="报告目录"><a href="#summary">结论</a><a href="#contract">合同</a><a href="#task">任务</a><a href="#representation">表示</a><a href="#retrieval">检索</a><a href="#recurrent-loop">闭环</a><a href="#state-update">更新/停止</a><a href="#terminal">读出</a><a href="#geometry">几何</a><a href="#native-comparison">对照</a><a href="#limitations">限制</a><a href="#parser-appendix">Parser</a><a href="#evidence">证据</a></nav>

<section id="summary"><div class="section-head"><span class="section-no">Conclusion first</span><h2>city content 仍需检索；full-state commit→query 已闭合，行为级闭环按 full-item 结果单独判定</h2></div>
<p class="lede">V2 排除了一个关键误解：显式 ordinal 只外显“现在轮到第几项”的 address/progress cue，并不包含目标 city 内容。位置审计检验原 query 是否错一位；随后把冻结 bank 的 lesion 持续到完整 city prefix。V3 不新增 model forward，而是对 240 条已冻结 full-item generations 做 outcome-blind rule 的聚合重解析：从首个生成 city 开始，逐项核对 donor 的第 1、2、4 个后继。四格在 cell-level depth-4 主指标上均通过，但 Index-Gemma backward 单方向较弱并如实保留。原 binary/query-local null、NCC、低维 count-subspace 与 update/stop null 全部保留。</p>
<div class="metrics v2"><div class="metric"><b>{full_gate_count}/4</b><span>full commit→query direct gates*</span></div><div class="metric"><b>{sustained_city_gate_count}/2</b><span>Index sustained city-support gates‡</span></div><div class="metric"><b>{greedy_strong_count}/4</b><span>first-city greedy gates‡</span></div><div class="metric"><b>{multihop_strong_count}/4</b><span>depth-4 donor continuation gates¶</span></div><div class="metric"><b>{'PASS' if fresh_carrier_strong else ('DIR' if fresh_carrier_replication.get('directional_gate_pass') else 'NULL')}</b><span>fresh Bullet-Gemma query→carrier§</span></div></div>
{chain_figure(cells)}
<div class="claim"><strong>允许的主张。</strong>commit→next-query 直接边在 full-state 层级闭合；full-item state 不只改变第一个 donor city，也可在自由生成中维持连续 donor path。组合证据支持一条“targeted retrieval → event/carrier state → content-bound commit → next targeted query → continued enumeration”的候选 recurrent pathway，但各边强度随 grammar / model 而变。它不等价于唯一、排他的 content-free arithmetic register。</div>
<div class="qualification"><strong>证据等级。</strong>原 V6 confirmation gate 保持原判定；† 是首轮同 split 事后诊断；* 是既有 raw shards 的冻结对比恢复；‡ 是在查看相关结果后注册的位置/行为扩展；§ 的十个 source requests 与因果 outcomes 都是 fresh，但 K=2 bank 来自更早 discovery；¶ 是聚合前冻结、但已看过一条 schema smoke row 的 post-hoc multihop reparse。3D 是 discovery-fit / confirmation-display 描述图，不进入任何 gate。</div></section>

<section id="contract"><div class="section-head"><span class="section-no">01 · Experimental contract</span><h2>五层证据分账：冻结基线、恢复分析、事后扩展、fresh outcomes 与聚合重解析</h2></div>
<dl class="definition-grid"><div class="definition"><dt>冻结 V6 基线</dt><dd>20 discovery seeds、10 disjoint confirmation slots；K、随机 bank、parser 与 replacement 规则不因干预结果改变。</dd></div><div class="definition"><dt>Full-state 对比恢复*</dt><dd>使用既有 raw shards 的 full_donor_patch、self_patch 与 norm-matched orthogonal arms；没有新增 model forward。</dd></div><div class="definition"><dt>首轮事后诊断†</dt><dd>连续 query-local likelihood、旧 split query-through-carrier 与 local terminal bridge 在查看基线后注册。</dd></div><div class="definition"><dt>V2 事后扩展‡</dt><dd>Index 的 query-through-city-prefix 与四格 full-item greedy 复用 confirmation cohort；冻结 K、bank 与 item-span layer 不变。</dd></div><div class="definition"><dt>Fresh carrier 复现§</dt><dd>Bullet-Gemma 使用十个未进入任何早期该单元 registry 的 source requests；outcomes fresh，bank discovery 不 fresh。</dd></div><div class="definition"><dt>V3 multihop reparse¶</dt><dd>不做新 forward；固定 exact-prefix depth 1/2/4、240 条分母和 seed-cluster bootstrap。一条 schema smoke row 已事先看过，故不冒充 prospective confirmation。</dd></div><div class="definition"><dt>Representation 3D</dt><dd>每层只在 discovery fit StandardScaler/PCA3，confirmation 仅投影与展示；默认层沿用 discovery-selected layer，图形不反向选层。</dd></div><div class="definition"><dt>保留的 null</dt><dd>原 binary、query-local carrier、global-sufficiency terminal、NCC、low-dimensional loop、update/stop 的失败均不被覆盖。</dd></div></dl>
<p class="formula">Primary statistical unit = true source seed; discovery n=20, confirmation n=10. Repeated heads / arms / tokens are paired conditions, not extra independent samples.</p>
<p class="footnote">冻结 audit matrix SHA-256：<code>{esc(data['baseline_report_sha256'])}</code>；V2 protocol：<code>{esc(data['followup_protocol_sha256'])}</code>；V3 protocol：<code>{esc(data['followup_v3_protocol_sha256'])}</code>。两处输入哈希的人工转录错误在强制 pre-analysis check 中被发现并逐字段记入 protocol；没有改变 endpoint、seed、方向、层或统计规则。</p></section>

<section id="task"><div class="section-head"><span class="section-no">02 · Task and unit</span><h2>模型同时解决两个问题：当前已处理多少项，下一条 city 应从哪里取</h2></div>
<p class="lede"><strong>显式 ordinal 只提供 address/progress cue，不提供 city content。</strong>Index grammar 把 ordinal 明示在 surface form 中；Bullet grammar 只提供结构化 item boundary。两者仍必须从 record/trace 中取得具体目标 city，并在 answer query 输出最终 count。ordinal 可能让 Index 的 binary endpoint 更接近 ceiling，却不可能逻辑上替代内容检索；它也不能与 Native-thinking 的 natural no-index 条件直接等同。</p>
<div class="mechanism-note"><div><h3>Retrieval state</h3><p>回答“下一项读哪条 record”。主要观测是 frozen-bank ablation 后 next-city failure、teacher-forced target-city log probability 与 targeted attention。</p></div><div><h3>Progress / commit state</h3><p>回答“当前走到第几项”。主要观测是 running/final decodability、carrier/commit deformation-restoration，以及 full commit patch 对下一 query 的影响。</p></div></div>
<h3>2.1 原始冻结 slots 的 strict exact enumeration accuracy</h3>
<p>这里的 accuracy 是补 seed <strong>之前</strong>的行为准确率：一个原始 slot 只有同时通过目标 grammar、最终 <code>Total</code>、有序 city-score pairs、marker kind、一对一 semantic trace 与 gold item count 才计正确。Discovery 与 confirmation 分母分别是每格 200 和 100；失败 reserve candidates 不进入这些分母。</p>
<div class="metrics v2"><div class="metric"><b>{pct(index_accuracy['accuracy'])}</b><span>Index raw strict exact · {esc(index_accuracy['pass_count'])}/{esc(index_accuracy['total_count'])}</span></div><div class="metric"><b>{pct(bullet_accuracy['accuracy'])}</b><span>Bullet raw strict exact · {esc(bullet_accuracy['pass_count'])}/{esc(bullet_accuracy['total_count'])}</span></div><div class="metric"><b>{pct(overall_accuracy['accuracy'])}</b><span>Overall raw strict exact · {esc(overall_accuracy['pass_count'])}/{esc(overall_accuracy['total_count'])}</span></div></div>
{behavioral_accuracy_table}
<div class="qualification"><strong>*不要把 replacement-filtered cohort 当作准确率。</strong>最终 1,200/1,200 只表示 outcome-blind sealed reserve replacement 后，每个预定分析槽位都有 strict-eligible trajectory；模型在原始冻结 slots 上的准确率仍是 Index {pct(index_accuracy['accuracy'])}、Bullet {pct(bullet_accuracy['accuracy'])}。完整 124 条原始失败及 replacement mapping 在 Appendix A.5。</div>
<div class="conclusion"><strong>任务层结论。</strong>只看 final exact count 会把检索失败、状态写入失败与末端读出失败混在一起；本报告按因果链逐边解释。</div></section>

<section id="representation"><div class="section-head"><span class="section-no">03 · Representation</span><h2>四个单元都能从 hidden state 解码 running index 与 final count</h2></div>
<p class="lede">层由 discovery-only 规则选择，再在 confirmation 上读取。十分类 chance 为 0.10；下表使用 logistic balanced accuracy。它证明信息存在，但不自动证明模型沿同一线性方向执行计算。</p>
{representation_table}
<figure class="paper-figure"><figcaption>Figure 3A · 每个 grammar×model 的 item_end running-occurrence 全层曲线。红色竖虚线是 discovery-only 默认层；灰/绿/橙分别是 discovery selection、confirmation logistic 与 confirmation NCC。</figcaption>{running_curve_figure}</figure>
<figure class="paper-figure"><figcaption>Figure 3B · answer_query_v3 final-count 全层曲线；与 running panel 使用相同的 exact original sample policy。</figcaption>{final_curve_figure}</figure>
<div class="geometry-viewer" id="representation-3d"><div class="geometry-toolbar"><label>Grammar<select id="enum-geometry-grammar"><option value="enumeration_index">Index</option><option value="enumeration_bullet">Bullet</option></select></label><label>Model<select id="enum-geometry-model"><option value="Qwen3-8B">Qwen3-8B</option><option value="Gemma4-E4B">Gemma4-E4B</option></select></label><button id="enum-geometry-reset" type="button">Reset synchronized view</button></div><div class="cloud-grid"><figure class="cloud-panel"><div class="cloud-head"><strong>Running occurrence · item_end</strong><select id="enum-running-layer" aria-label="Running representation layer"></select></div><canvas id="enum-running-canvas" aria-label="Interactive running occurrence PCA3 point cloud"></canvas><p class="cloud-stats" id="enum-running-stats"></p></figure><figure class="cloud-panel"><div class="cloud-head"><strong>Final count · answer_query_v3</strong><select id="enum-final-layer" aria-label="Final representation layer"></select></div><canvas id="enum-final-canvas" aria-label="Interactive final count PCA3 point cloud"></canvas><p class="cloud-stats" id="enum-final-stats"></p></figure></div><p class="footnote">Figure 3C · 与 Native representation comparison 同构：每层 discovery-only StandardScaler/PCA3，显示全部 confirmation points 与按 occurrence/count 连接的 confirmation centroids。拖动任一面板会同步旋转两个面板；双击重置。PC 轴是任意可视化坐标，不是因果方向。</p></div>
<div class="conclusion"><strong>Experiment 3 结论。</strong>count information 同时存在于 item-end progress state 与 answer-query final state；后续 causal assays 决定它是否被实际使用。</div></section>

<section id="retrieval"><div class="section-head"><span class="section-no">04 · Targeted retrieval</span><h2>先量化 registered query 到 city predictor 的 token offset，再检验 support window</h2></div>
<p class="lede">原 V6 gate 以 free-generation next-city failure 为主 endpoint：Bullet 两格强复现，Index-Qwen 只有方向性，Index-Gemma 为 null。V2 没有因为这些结果重选 head；它先审计 registered query 与“首个 city token 的直接 autoregressive predictor”是否相同，再把同一冻结 bank 从 registered query 持续关闭到最后一个 city-token predictor。位置审计中 {position_alias_count}/2 个 Index 单元为 exact alias；详细几何为 <code>{esc(position_details)}</code>。</p>
{retrieval_table}
<figure class="paper-figure"><figcaption>Figure 4 · 原 registered query-local 连续读出的 selected damage 与 selected-vs-random specificity。V2 的更宽 city-prefix 窗口仍在表中单独列账。</figcaption>{retrieval_forest}</figure>
<p class="formula">selected damage = log P(target city | clean) − log P(target city | selected bank off)<br>specificity = mean log P(target city | random bank off) − log P(target city | selected bank off)</p>
<div class="qualification"><strong>解释边界。</strong>本次 0/2 个 Index 单元在所有 anchors 上都是 exact alias；实际 offset 随 tokenization / visible ordinal 为 0–4。它不是把 target 索引到错误 city，而是说明单点 query-only lesion 会在部分样本中过早结束。V2 改变的是同一 target 的时间支持，不是按结果偷换 target；即使 sustained gate 不通过，也不能推出 city retrieval 不存在。</div>
<div class="conclusion"><strong>Experiment 4 结论。</strong>Index sustained city-support 的严格 gate 为 {sustained_city_gate_count}/2。无论 gate 数量如何，ordinal 只能指定地址/进度，具体 city 仍是 content-bound retrieval 的对象；原 binary 判定继续原样保留。</div></section>

<section id="recurrent-loop"><div class="section-head"><span class="section-no">05 · Recurrent loop</span><h2>query→carrier 用 fresh outcomes 复现；commit→query 再加入 full-item free-generation readout</h2></div>
<h3>5.1 Retrieval→carrier：窗口从 registered query 到 final grammar-carrier token，首尾均包含</h3>
<p>原 carrier assay 在 teacher-forced trace 中只关闭一个 query token；但 Bullet-Gemma 的 K=2 行为干预在每个 cached decode step 都保持关闭。其原始 carrier rows 的 scope 为 <code>{esc(cells['enumeration_bullet|Gemma4-E4B']['carrier']['original_head_ablation_scopes'])}</code>、位置数为 <code>{esc(cells['enumeration_bullet|Gemma4-E4B']['carrier']['original_head_ablation_position_counts'])}</code>。预注册复现固定 K=2、source layer L16，并在 teacher-forced visible path 上关闭 <em>every position from the registered query through the final grammar-carrier token, inclusive</em>；matched-position clean-state clamp 是等 token 数的近深度非 item 对照。</p>
{carrier_table}
<figure class="paper-figure"><figcaption>Figure 5A · 原 query-local carrier deformation / clean-carrier restoration。Fresh Bullet-Gemma query-through-carrier 复现在表内保留其独立 fresh-outcome 身份。</figcaption>{carrier_forest}</figure>
<p class="footnote">Fresh cohort true-source seeds：<code>{esc(fresh_lock['true_source_seeds'])}</code>；outcome-blind cohort lock：<code>{esc(fresh_lock['cohort_lock_sha256'])}</code>。结果为 {esc(fresh_carrier_summary)}；原 query-local null 仍在表中。</p>
<h3>5.2 Commit→next query：full-state teacher-forced edge 与 full-item greedy adoption 分开 gate</h3>
<p>首版 <code>commit_to_retrieval_pass</code> 把低维 count-subspace attention、city log-odds 与 greedy adoption 合成一个过强 gate。恢复 raw shards 中的 full-state arms 后，四格 direct edge 均通过。V2 进一步回到先前的 endpoint-aligned <strong>full item span</strong> 几何：固定 Qwen L0 / Gemma L21，forward 5→6 与 backward 7→6 两个方向都报告，free generation 比较 <code>donor_to_receiver</code> 与 <code>receiver_self</code>，并保留 <code>native_donor</code> 正控制。</p>
{commit_table}
<h3>5.3 两个方向逐项核对</h3>
{greedy_direction_table}
<p class="footnote">八个方向 gate 均使用十个 true source seeds；maximum generation truncation rate = <code>{num(data['followup']['full_item_greedy']['maximum_generation_truncation_rate'])}</code>。</p>
<h3>5.4 Full-state、content-bound 的多步 continuation：不是只看第一个 donor city</h3>
<p>V3 固定从生成序列的<strong>第一个已知 city ordinal</strong>开始做 exact-prefix 解析；不允许跳过错误项、重排或修复。例如预期 donor path 为 <code>[7,8,9,10]</code> 时，<code>[7,8,9,10]</code> 的 depth=4，而 <code>[5,7,8,9,10]</code> 的 depth=0。所有 10 seeds、失败项与截断项保留在 unconditional denominator。本数据没有截断、空输出或歧义行，但有 <code>{esc(multihop['failure_taxonomy']['nonconsecutive_ordinal_rows'])}</code> 条非连续输出；它们按冻结规则计失败。</p>
{multihop_table}
<figure class="paper-figure"><figcaption>Figure 5B · donor_to_receiver 在 depth 1/2/4 的 unconditional exact-prefix rate。Bullet 两格一旦采用首个 donor successor，后续 persistence 为 1.00；Index-Gemma 从 depth 1 到 depth 4 明显衰减。</figcaption>{multihop_figure}</figure>
<h3>5.5 Multihop 的方向级异质性</h3>
{multihop_direction_table}
<p class="footnote">主 gate 按 protocol 在每个 cell 内先对双方向做 true-seed cluster aggregation；单方向只作透明诊断。因而 Index-Gemma backward 的 depth-4 rate=0.30、CI 下限=0.00 不被隐藏，但 cell-level 0.55 [0.35, 0.75] 仍满足预定主 gate。V3 没有新增模型 forward。</p>
<div class="qualification"><strong>闭环边界。</strong>teacher-forced full-state edge 的 4/4、first-city greedy 的 {esc(greedy_summary)} 与 multihop depth-4 的 {multihop_strong_count}/4 是三个 estimand；多步正结果支持 <em>content-bound full-state continuation</em>，不能把原 narrow count-subspace/update/stop null“修成通过”。</div>
<div class="conclusion"><strong>Experiment 5 结论。</strong>完整 commit→query direct edge 为 4/4；full-item first-city adoption 为 {esc(greedy_summary)}；连续四步 donor path 的 cell-level 强 gate 为 {multihop_strong_count}/4。由此可以写“full-state patch 可重定向并维持后续枚举”，但仍不能写“发现 content-free +1 operator”。</div></section>

<section id="state-update"><div class="section-head"><span class="section-no">06 · Update and stop</span><h2>full-state continuation 已补齐；低维 count-subspace 的 update / stop 仍然是严格 null</h2></div>
<p class="lede">这两组实验问的不是同一件事。V3 multihop 保留 donor 的完整 item-span hidden state，并观察自由生成是否沿 donor 内容路径继续；原 update/stop assay 只移植 discovery-fitted count subspace 或注入 terminal/nonterminal 低维组件，要求它像 context-free operator 一样导致 backstep、恢复、停止或继续。后者四格的行为效应均为 0，不能因 full-state 正结果而改判。</p>
<div class="trajectory-strip" aria-label="Full-state versus low-dimensional update distinction"><span class="active">full item state</span><span>→ donor city k+1</span><span>→ donor city k+2</span><span>→ donor city k+3</span><span>→ donor city k+4</span></div>
{state_update_table}
<div class="claim"><strong>机制解释。</strong>Enumeration 能做“更新与继续枚举”，但目前被支持的是 <em>full-state、content-bound、context-dependent</em> 的更新；没有证据把这个状态压缩成可脱离内容和上下文独立运行的低维 <code>+1 / stop</code> 控制量。这与 Native-thinking 报告中“distributed content-bound state，而非唯一 scalar register”的边界一致。</div>
<div class="conclusion"><strong>Experiment 6 结论。</strong>完整状态的行为级多步 continuation 为正；低维 loop、update 与 stop 为 0/4。二者共同界定了现象，而非互相矛盾。</div></section>

<section id="terminal"><div class="section-head"><span class="section-no">07 · Terminal readout</span><h2>terminal item 是必要的；“在全局 scrambled trace 中单独充分”是过强命题</h2></div>
<p class="lede">两个 Bullet 单元的 terminal necessity 很强，但原 sufficiency arm 把除最后一项外的全部 trace item 替换为普通 background tokens，再只恢复一个 terminal item。原 Bullet rows 的 marker-token count 在两模型中均为 <code>[0]</code>；non-marker count 因 tokenizer 而异，Qwen 为 <code>{esc(cells['enumeration_bullet|Qwen3-8B']['terminal']['nonmarker_token_counts'])}</code>，Gemma 为 <code>{esc(cells['enumeration_bullet|Gemma4-E4B']['terminal']['nonmarker_token_counts'])}</code>。marker-only arm 因而是字面上的空干预，真正的 terminal item 分布在完整 non-marker span 上。局部诊断保持 earlier trace clean，只消融完整 terminal item，并在同一位置做 state restoration/occlusion。</p>
{terminal_table}
<figure class="paper-figure"><figcaption>Figure 7 · terminal necessity 与 global one-item sufficiency 的同尺度比较。necessity 回答“完整上下文中是否需要它”；global sufficiency 回答“其余 trace 被破坏后单独恢复它是否足够”。</figcaption>{terminal_forest}</figure>
<div class="qualification"><strong>解释。</strong>若 necessity 为正而 global sufficiency 为负，逻辑上并不矛盾：一个组件可以在完整系统中必需，却不能在被整体破坏的背景中单独重建功能。局部 mediation 才对应 token→terminal state→answer 的窄边。</div>
<div class="conclusion"><strong>Experiment 7 结论。</strong>terminal readout 现象应表述为 context-dependent mediation，而不是“最后一个 token 单独携带最终计数”。</div></section>

<section id="geometry"><div class="section-head"><span class="section-no">08 · Geometry limits</span><h2>NCC 四格全 null；这限制统一线性 code，不否定 full-vector causal state</h2></div>
<p class="lede">NCC 问的是 lesion 是否沿 discovery-fitted count centroid geometry 产生方向特异损伤。它比“hidden state 改变”更强，也比“full state 能改变下一 query”更依赖选定 readout。四格均为 <code>NO_DIRECTIONAL_SPECIFIC_SUPPORT</code>。</p>
{ncc_table}
<figure class="paper-figure"><figcaption>Figure 8 · discovery-fitted NCC centroid geometry 下的 selected loss 与 selected-vs-random specificity。橙色表示注册区间 gate 未满足；它与 3D 可分性不是同一个命题。</figcaption>{ncc_forest}</figure>
<div class="conclusion"><strong>Experiment 8 结论。</strong>V6 与 Native-thinking Appendix H 保持一致：存在可解码与可干预的 distributed state，但没有证据把它压缩成跨模型、跨 grammar 统一的单一线性 counter geometry。</div></section>

<section id="native-comparison"><div class="section-head"><span class="section-no">09 · Cross-report comparison</span><h2>实验矩阵现已补齐；核心机制对齐，但 grammar-specific 强度差异必须保留</h2></div>
{table(('维度','Native-thinking 报告','Enumeration V6','一致性判断'),(
('Representation','running / answer state 可解码，并有 discovery-fit 3D comparison','四格 running / final count 全层曲线、选层表和交互 3D 已按同一模式补齐','实验与可视化口径一致；信息存在 ≠ 统一寄存器'),
('Targeted retrieval','model-specific frozen bank 对下一项具体内容检索有必要性',f'Bullet binary 强；Index 位置审计 {position_alias_count}/2 exact alias，sustained gate {sustained_city_gate_count}/2','机制对象一致；visible ordinal 只改变 surface/support 条件'),
('Carrier / commit','distributed event state 与 carrier 可被 patch；非唯一单 token',f'三格原 query-local 通过；Bullet-Gemma fresh query-through-carrier 为 {fresh_carrier_summary}','一致：更宽时窗 state 比单点解释稳定'),
('Commit→query / continuation','full item state 可改变并延续下一项；Qwen natural no-index 较强，Gemma 条件性更强',f'full-state direct 4/4；first-city {greedy_summary}；depth-4 cell gate {multihop_strong_count}/4','对齐 content-bound continuation；Index-Gemma backward 较弱被保留'),
('Low-dimensional update / stop','未建立 memoryless +1、唯一低维 loop 或单独 stop controller','四格 count-subspace update/stop 均未通过；full-state multihop 不覆盖这些 null','一致的限制：完整内容状态有效，不等于 scalar operator'),
('Terminal readout','trace→answer 是有序的部分通路，不是完全排他中介','terminal necessity 强；global one-item sufficiency 失败，local mediation 另报','一致：context-dependent readout'),
('Linear geometry','线性 centroid/NCC 并未形成跨模型统一正结果','四格 NCC 无方向特异支持','一致的限制；3D 分离不能反推 NCC causal pass'),
))}
<div class="claim"><strong>“对齐”的精确定义。</strong>现在两份报告在 representation、retrieval、carrier、full-state commit、multihop continuation、terminal、NCC 与 update/stop 的实验槽位上可以一一对照；两者都支持 distributed, content-bound progress state、targeted retrieval 与 state-dependent routing，也都不支持唯一 circuit、content-free +1 operator 或统一线性 counter。对齐不要求每格数值相等，更不允许把 null 隐去。</div>
<p class="footnote">比较源 Native-thinking report SHA-256：<code>{esc(data['native_report_sha256'])}</code>；报告结构锚点已程序化核验。</p></section>

<section id="limitations"><div class="section-head"><span class="section-no">10 · Limitations</span><h2>哪些结论仍不能写进论文主张</h2></div>
<ol><li><strong>事后机制诊断不是 fresh confirmation。</strong>旧 continuous diagnostics、V2 Index temporal-support 与 full-item greedy 都是在查看相关 V6 结果后注册；只有新 Bullet-Gemma source requests 的 causal outcomes 是 fresh。</li><li><strong>V3 multihop 不是 prospective replication。</strong>规则在聚合前冻结，但此前看过一条 receiver-self schema smoke row；所以它是强审计的 post-hoc robustness，不是独立 fresh confirmation。</li><li><strong>bank 发现并不 fresh。</strong>fresh carrier replication 沿用 earlier discovery-frozen K=2 bank；其强度只能确认该 bank 在新 requests 上的因果可复现性。</li><li><strong>full-state direct、first-city 与 multihop 必须分账。</strong>teacher-forced 4/4 不自动推出 free-generation；本次 first-city 为 {esc(greedy_summary)}、depth-4 为 {multihop_strong_count}/4，且方向异质性仍存在。</li><li><strong>显式 ordinal 不是 city memory。</strong>它降低 address/progress 不确定性，不提供具体 city，也不能作为 natural no-index internal retrieval 的无混杂替代。</li><li><strong>3D 分离不是 causal geometry gate。</strong>PCA3 是描述图；NCC null 约束 discovery-fitted linear centroid geometry，不能被“看起来分簇”推翻。</li><li><strong>未证明唯一性或算术算子。</strong>多个 head/state 路径可能冗余；原 count-subspace update/stop 全 null，没有隔离 context-free、memoryless <code>+1</code>。</li></ol>
<div class="conclusion"><strong>论文写法。</strong>可以写“完整、content-bound 的 item state 能重定向并维持后续枚举，形成一条可重复干预但随 grammar/model 改变强度的 recurrent counting pathway”；仍不应写“发现唯一低维计数寄存器或 memoryless +1/stop operator”。</div></section>

<section id="parser-appendix"><div class="section-head"><span class="section-no">Appendix A · Parser contract</span><h2>从原始文本到 causal token site：语法、semantic span、token alignment 与 multihop 判定的完整规则</h2></div>
<p class="lede">本报告使用两个彼此独立但必须同时通过的 parser 层：严格输出 parser 决定行为成功和 formal cohort；hybrid semantic parser 决定每个 item 的字符跨度与 token site。最终答案边界又独立于 running-trace 命中，因此 running parser miss 不会静默删除 <code>answer_query_v3</code> 样本。</p>
<div class="parser-flow" aria-label="Parser data flow"><span>raw generation</span><span>channel / wrapper split</span><span>strict grammar + Total audit</span><span>semantic item spans</span><span>exact token-prefix sites</span></div>

<h3>A.1 严格响应语法：接受什么，拒绝什么</h3>
{parser_grammar_table}
<p>空行会被忽略；其余每一个非空行都必须是对应 grammar 的 item，且唯一的 <code>Total:</code> 必须位于最后。Index 与 Bullet 同时出现时状态为 <code>mixed_markers</code>；只出现相反 marker 为 <code>wrong_marker</code>；没有合法 item 为 <code>no_records</code>；Index 不是严格 <code>1..M</code> 时为 <code>index_sequence_error</code>。city 去除两侧空白后按字符串精确比较，score 与 Total 解析为整数；集合 precision/recall 只是诊断，不能替代有序逐项相等。</p>
<div class="code-grid"><div class="code-card"><h4>ACCEPT · Index</h4><pre>1. Taipei: 51
2. Nanjing: 55
Total: 2</pre></div><div class="code-card"><h4>ACCEPT · Bullet</h4><pre>- Taipei: 51
- Nanjing: 55
Total: 2</pre></div><div class="code-card"><h4>REJECT · wrong/mixed marker</h4><pre>1) Taipei: 51
- Nanjing: 55
Total: 2</pre></div><div class="code-card"><h4>REJECT · extra prose / inconsistent Total</h4><pre>1. Taipei: 51
I found one record.
Total: 2</pre></div></div>
<p class="footnote">以上四段是规则示意，不冒充模型输出。非 thinking 模式且没有 channel delimiter 时，整个 raw response（去除完整外层 <code>&lt;response&gt;</code> / <code>[RESPONSE]</code> wrapper 后）作为 final text；若意外出现 Qwen/Gemma/Mistral reasoning delimiter，则按对应 close 分割，未闭合 reasoning 不会伪造 final answer。</p>

<h3>A.2 Formal causal cohort：七个条件取 AND</h3>
{parser_gate_table}
<p class="formula">strict_causal_eligible = registered_success ∧ enumeration_format_compliant ∧ listed_total_matches_length ∧ exact_ordered_gold_pairs ∧ marker_kind_compliant ∧ forward_one_to_one ∧ item_count_matches_gold</p>
<div class="qualification"><strong>Gold firewall。</strong>Gold records 用来验证 city-score 身份、passage order 与一对一覆盖，并为已出现的 city 建立 registry；<strong>Gold N 与 final Total 不会构造、补齐或选择 item sequence</strong>。缺失 item、重复 city、错误顺序或错误 marker 均保留为失败，parser 不会为了贴近答案而去重、排序或 padding。</div>

<h3>A.3 Hybrid semantic span 的选择顺序</h3>
<ol><li><strong>Rank-supported episode。</strong>将局部 city observation 与显式 rank evidence 配对；每次 rank-1 restart 开新 episode，选择最长连续 <code>1..M</code> 序列，长度相同取最早。</li><li><strong>Structural extension。</strong>只有当保守 structural span 以 rank episode 的 city 序列为精确前缀，并且确实新增至少一个不同 city 时，才采用更长 structural span。</li><li><strong>Structural fallback。</strong>没有可用 rank episode 时，保留第一个具有可靠 item/termination boundary 的 gold-city list；Bullet 通常走这一层。</li><li><strong>Synthetic evidence fallback。</strong>前两类都没有命中时，允许输出 score-supported order 供审计，但它被标成 <code>synthetic_unverified</code>、<code>trace_one_to_one=false</code>，因此不能进入 strict causal cohort。</li></ol>
<p>V6 在上述 inherited semantic parser 之外再强制 grammar marker：Index 必须为 <code>indexed</code>，Bullet 必须为 <code>bullet</code>。因此 inherited parser 能识别的 ordinal word、inline count、audit sentence 或 recap 并不会自动成为 V6 formal evidence。</p>

<h3>A.4 字符 site 与 token site</h3>
{parser_site_table}
<p>每个字符 site 保存半开区间 <code>[char_start, char_end)</code>。对其终点先与原始 <code>output_token_ids</code> 做精确 prefix 对齐：若字符边界正好对应 baseline token prefix，策略为 <code>literal_baseline_token_prefix</code>；否则只允许 <code>text_exact_boundary_retokenization</code>，并记录共享 baseline prefix token 数、重新 tokenized suffix 数和 prefix token hash。分析 endpoint 固定为 <code>prefix_token_count - 1</code>；无法保持文本精确相等时标成 ineligible，绝不使用最近 token 猜测。</p>
<p><code>answer_query</code> 截止到冒号，<code>answer_query_v3</code> 包含 <code>Total:</code> 后空白并停在最终整数首字符之前。Inherited relaxed locator <code>answer_query_v2</code> 可作为审计字段出现，但不在 V6 registered site registry 内。即使 running semantic parser 没命中，三个 answer locator 仍独立执行。</p>
<div class="claim"><strong>Grammar timing。</strong>Index 的显式 ordinal 属于 <code>rank_before_city</code>；Bullet 的 invariant hyphen 不能解释成数字进度，主 timing stratum 是 <code>structural_item_end</code>。两种 grammar 的 carrier 都以实际检索到的 city 到 item commit tail 为内容载体。</div>

<h3>A.5 本次 1,200 个固定分析 cells 的实际 parser/replacement 审计</h3>
<div class="metrics v2"><div class="metric"><b>{sum(int(row['selected_cell_count']) for row in parser_data['cohort_summaries'])}</b><span>final strict cells · 4 cells × (200 discovery + 100 confirmation)</span></div><div class="metric"><b>{esc(parser_data['original_strict_failure_count'])}</b><span>original fixed-slot strict failures，全部留在 ledger</span></div><div class="metric"><b>{esc(parser_data['failed_reserve_attempt_count'])}</b><span>ordinary reserve candidates rejected before a valid replacement</span></div><div class="metric"><b>{esc(parser_data['final_fixed_quota_unresolved_count'])}</b><span>unresolved cells after sealed fixed-quota replacement</span></div></div>
{parser_cohort_table}
<p>这里的 “final unresolved=0” 不等于原始输出从未失败：原固定 slots 中共有 {esc(parser_data['original_strict_failure_count'])} 个 strict failure。它们大多已有可识别 semantic span，但在 exact count、exact ordered pairs 或严格格式上失败。寻找合格 replacement 时另有 {esc(parser_data['failed_reserve_attempt_count'])} 个 ordinary reserve candidates 未通过；这些尝试同样不被删除。冻结 reserve policy 逐格替换原失败；broad/native-loop panel 为保持 true-source seed coherence，会替换该 seed 的整条 required-count trajectory，而不是只挑通过的 count。replacement 与候选拒绝不读取 intervention outcomes。</p>
{replacement_policy_table}
<details><summary>展开全部 {esc(parser_data['original_strict_failure_count'])} 条原 strict failure 与 replacement 映射</summary><div>{parser_failure_table}</div></details>
<details><summary>展开全部 {esc(parser_data['failed_reserve_attempt_count'])} 条 ordinary reserve candidate 失败</summary><div>{parser_failed_reserve_table}</div></details>

<h3>A.6 Full-item multihop 的第二套行为 parser</h3>
<p>V3 不重新运行 strict final parser，而是读取已冻结 generation row 的 <code>generated_known_city_ordinals_any_surface</code>。生成器在最早的 <code>&lt;/think&gt;</code>、<code>&lt;|im_end|&gt;</code> 或 <code>&lt;end_of_turn&gt;</code> 之前，对注册 city 做不区分大小写、带 ASCII 字母数字边界的精确字符串匹配，并按字符位置保留每次出现；<code>generated_bullet_city_ordinals</code> 与 ambiguous bullet lines 仅作诊断，主 estimand 使用 any-surface 列表。</p>
<p class="formula">expected donor path = [donor_successor, ..., gold_count]<br>depth = longest d such that observed[:d] == expected[:d]<br>registered depths = {esc(multihop_endpoint['registered_depths'])}; skip / reorder / deduplicate / repair = forbidden</p>
<p>因此 <code>[7,8,9,10]</code> 对预期 <code>[7,8,9,10]</code> 得 depth 4；<code>[5,7,8,9,10]</code> 得 depth 0；<code>[7,9,10]</code> 只得 depth 1。重复、非连续、空输出、歧义行和长度截断都记录在 taxonomy 中，并留在 unconditional denominator。</p>
{parser_multihop_taxonomy_table}
<details><summary>展开固定 lowest-seed 的 24 个三条件 parser 例子</summary><div>{parser_multihop_example_table}</div></details>

<h3>A.7 Parser provenance</h3>
{parser_source_table}
<div class="conclusion"><strong>Parser 审计结论。</strong>报告中的 “PASS” 不是宽松文本匹配：行为 cohort 要通过严格 grammar、Total、一对一有序 city-score、semantic span 与精确 token-prefix alignment；multihop 又使用独立的 any-surface exact-prefix endpoint。两套 endpoint 的差异、原失败 rows 与 replacement mapping 均完整保留。</div></section>

<section id="evidence"><div class="section-head"><span class="section-no">Appendix B · Audit ledger</span><h2>冻结结果、诊断扩展与所有引用源的哈希</h2></div>
<dl class="ledger"><dt>Suite audit</dt><dd>{esc(data['completion_audit']['status'])}; ordinary parser/runtime failures={esc(data['completion_audit']['ordinary_failure_count'])}; coherent replacement trajectories={esc(data['completion_audit']['coherent_replacement_trajectory_count'])}。</dd><dt>Selection</dt><dd>Index-Qwen K128；Index-Gemma K8；Bullet-Qwen K96；Bullet-Gemma K2。V2/V3 没有重选这些 bank。</dd><dt>Frozen layers</dt><dd>Full-item greedy 固定 Qwen L0、Gemma L21；fresh Bullet-Gemma carrier 固定 source layer L16；3D 默认层严格读取 native-aligned discovery-selected CSV。</dd><dt>V2 completion</dt><dd><code>{esc(data['followup']['status'])}</code>；protocol <code>{esc(data['followup_protocol_sha256'])}</code>；fresh cohort lock <code>{esc(fresh_lock['cohort_lock_sha256'])}</code>。</dd><dt>V3 completion</dt><dd><code>{esc(data['followup_v3']['status'])}</code>；protocol <code>{esc(data['followup_v3_protocol_sha256'])}</code>；240 trial rows / 80 seed-direction rows；new model forward=false。</dd><dt>3D firewall</dt><dd><code>{esc(data['representation_manifold']['status'])}</code>；confirmation 用于展示但不用于 fit/selection；坐标 payload SHA-256 <code>{esc(data['native_aligned_representation']['manifold_manifest']['output']['sha256'])}</code>。</dd><dt>Original matrix</dt><dd>原 audit matrix SHA-256 <code>{esc(data['baseline_report_sha256'])}</code>；V2 前叙事报告 <code>{esc(data['followup_baseline_report_sha256'])}</code>；V3 前叙事报告 <code>{esc(data['followup_v3_baseline_report_sha256'])}</code>。</dd><dt>Statistical identity</dt><dd>true source seed；analysis slot 仅承担 panel membership，不伪装成独立 seed；fresh cohort 不读取 intervention outcomes；V3 unconditional denominator 不因生成成功与否改变。</dd></dl>
<details><summary>展开全部 evidence path 与 SHA-256</summary><div>{source_table}</div></details>
<details><summary>如何读取 PASS / null</summary><div><p><strong>Pipeline PASS</strong> 表示数据、freeze、provenance 与预定计算完整；不表示每个科学假设通过。<strong>Null</strong> 表示该 endpoint/gate 未获得所需方向和区间支持；除非有明确 assay mismatch，只保留为限制，不按结果重调参数。</p></div></details>
<div class="conclusion"><strong>审计结论。</strong>叙事重构不改写冻结数字。V2 三条路径分别标记为事后位置诊断、事后行为扩展与 fresh causal-outcome replication；V3 标记为聚合前冻结的 post-hoc multihop reparse 与描述性 3D。所有原 null、方向异质性与新结果同时呈现。</div></section>

<footer class="footer"><p>Self-contained HTML · no external assets · generated from sealed V6 + follow-up V2/V3 artifacts. Narrative topology and interactive representation comparison mirror the Native-thinking report; scientific judgments are determined only by frozen contracts and hashed result files.</p></footer>
<script id="report-manifest" type="application/json">{embedded}</script><script>{manifold_script}</script></article></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--completion-audit", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--followup-protocol", type=Path, required=True)
    parser.add_argument("--followup-baseline-report", type=Path, required=True)
    parser.add_argument("--followup-v3-protocol", type=Path, required=True)
    parser.add_argument("--followup-v3-baseline-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    data = collect_report_data(
        run_root=args.run_root,
        completion_audit=args.completion_audit,
        baseline_report=args.baseline_report,
        native_report=args.native_report,
        protocol=args.protocol,
        followup_protocol=args.followup_protocol,
        followup_baseline_report=args.followup_baseline_report,
        followup_v3_protocol=args.followup_v3_protocol,
        followup_v3_baseline_report=args.followup_v3_baseline_report,
    )
    document = render_report(data)
    for section in REQUIRED_SECTIONS:
        if f'id="{section}"' not in document:
            raise RuntimeError(f"Narrative report lost section {section}")
    atomic_text(args.output, document)
    manifest = {
        "schema_version": "realistic_niah_v6_enumeration_narrative_report_manifest_v3",
        "format_revision": "native_mirrored_v4",
        "status": "PASS",
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "required_sections": list(REQUIRED_SECTIONS),
        "baseline_report_sha256": data["baseline_report_sha256"],
        "native_report_sha256": data["native_report_sha256"],
        "native_template_css_source_sha256": data["native_report_sha256"],
        "native_template_format_mirrored": True,
        "followup_protocol_sha256": data["followup_protocol_sha256"],
        "followup_baseline_report_sha256": data[
            "followup_baseline_report_sha256"
        ],
        "followup_v3_protocol_sha256": data["followup_v3_protocol_sha256"],
        "followup_v3_baseline_report_sha256": data[
            "followup_v3_baseline_report_sha256"
        ],
        "source_sha256": data["source_sha256"],
        "scientific_results_rewritten": False,
        "followup_v2_complete": True,
        "followup_v3_complete": True,
        "position_audit_labeled": True,
        "greedy_extension_labeled": True,
        "fresh_replication_labeled": True,
        "multihop_reparse_labeled": True,
        "representation_3d_embedded": True,
        "parser_appendix_embedded": True,
        "behavioral_accuracy_embedded": True,
        "parser_original_failure_ledger_embedded": True,
        "parser_source_hashes_embedded": True,
        "narrative_experiment_frames_complete": True,
        "figure_axis_captions_complete": True,
        "per_experiment_conclusions_complete": True,
        "simple_examples_complete": True,
        "confirmation_used_for_3d_fit_or_selection": False,
        "new_model_forward_used_for_v3": False,
        "new_model_forward_used_for_accuracy_update": False,
        "posthoc_diagnostics_labeled": True,
        "frozen_nulls_retained": True,
    }
    atomic_text(args.manifest, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
