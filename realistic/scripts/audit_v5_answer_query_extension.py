#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "realistic_niah_v5_answer_query_extension_final_audit_v1"
MODELS = ("Qwen3-8B", "Gemma4-E4B")
REGISTERED_K = {1, 2, 4, 8, 16, 32}
EXECUTION_CONDITIONS = {
    "self_patch",
    "full_donor_patch",
    "projected_donor_patch",
    "orthogonal_norm_matched",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid physical JSONL line {path}:{line_number}: {error}") from error
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def boolish(value: Any) -> bool:
    return value is True or str(value).lower() in {"true", "1", "yes"}


def finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and abs(number) != float("inf")


def audit_model(run_root: Path, model: str) -> tuple[dict[str, Any], list[Path]]:
    root = run_root / model
    ext = root / "answer_query_extension"
    status_path = ext / "logs/supervisor.status"
    require(status_path.read_text(encoding="utf-8").strip() == "complete", f"{model} supervisor is not complete")
    paths = {
        "generations": root / "generations.jsonl",
        "supplement_generations": ext / "supplement/accepted_generations.jsonl",
        "supplement_materialize_audit": ext / "supplement/accepted_generations.audit.json",
        "primary_capture": ext / "representation/capture_primary/capture_index.jsonl",
        "primary_exclusions": ext / "representation/capture_primary/capture_exclusions.jsonl",
        "supplement_capture": ext / "representation/capture_supplement_n10/capture_index.jsonl",
        "supplement_exclusions": ext / "representation/capture_supplement_n10/capture_exclusions.jsonl",
        "representation_audit": ext / "representation/analysis_primary/representation_audit.json",
        "representation_regression": ext / "representation/analysis_primary/regression_confirmation.csv",
        "representation_geometry": ext / "representation/analysis_primary/geometry_summary.csv",
        "attention_primary": ext / "head_ablation/attention_primary.csv",
        "attention_primary_audit": ext / "head_ablation/attention_primary.audit.json",
        "attention_supplement": ext / "head_ablation/attention_supplement_n10.csv",
        "attention_supplement_audit": ext / "head_ablation/attention_supplement_n10.audit.json",
        "head_plan": ext / "head_ablation/plan/answer_query_causal_plan.csv",
        "head_plan_audit": ext / "head_ablation/plan/answer_query_causal_plan_audit.json",
        "head_primary": ext / "head_ablation/trials_primary_confirmation.jsonl",
        "head_supplement": ext / "head_ablation/trials_supplement_n10_confirmation.jsonl",
        "head_primary_statistics": ext / "analysis/head_ablation_primary_confirmation_statistics.csv",
        "head_supplement_statistics": ext / "analysis/head_ablation_supplement_n10_confirmation_statistics.csv",
        "execution_layer": ext / f"answer_execution/plan/{model}__answer_execution_layer_selection.json",
        "execution_pairs": ext / f"answer_execution/plan/{model}__answer_execution_pairs.jsonl",
        "execution_plan_audit": ext / f"answer_execution/plan/{model}__answer_execution_plan_audit.json",
        "execution_trials": ext / "answer_execution/trials_confirmation.jsonl",
        "execution_statistics": ext / "analysis/answer_execution_statistics.csv",
        "analysis_audit": ext / "analysis/answer_query_extension_analysis_audit.json",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    require(not missing, f"{model} missing audit inputs: {missing}")

    original = read_jsonl(paths["generations"])
    supplement = read_jsonl(paths["supplement_generations"])
    primary_capture = read_jsonl(paths["primary_capture"])
    primary_exclusions = read_jsonl(paths["primary_exclusions"])
    supplement_capture = read_jsonl(paths["supplement_capture"])
    supplement_exclusions = read_jsonl(paths["supplement_exclusions"])
    require(len(original) == 300, f"{model} original generations != 300")
    require(len(primary_capture) + len(primary_exclusions) == 300, f"{model} captured+excluded != 300")
    require(len(supplement_capture) + len(supplement_exclusions) == len(supplement), f"{model} supplement capture accounting mismatch")
    require(not supplement_exclusions, f"{model} accepted supplement has capture exclusions")
    original_ids = {str(row.get("request_id", row.get("stimulus_id"))) for row in original}
    supplement_ids = {str(row.get("request_id", row.get("stimulus_id"))) for row in supplement}
    require(original_ids.isdisjoint(supplement_ids), f"{model} supplement overlaps original request IDs")
    require(all(int(row["gold_count"]) == 10 for row in supplement_capture), f"{model} supplement is not N10-only")
    require(all(boolish(row.get("trace_one_to_one")) for row in supplement_capture), f"{model} supplement capture contains non-one-to-one row")

    representation = read_json(paths["representation_audit"])
    require("answer_query_v2" in representation.get("observed_sites", []), f"{model} representation omitted answer_query_v2")
    require("confirmation" not in str(representation.get("selection_policy", "")).lower() or "no" in str(representation.get("selection_policy", "")).lower(), f"{model} representation selection policy is not confirmation-safe")

    attention_primary = read_csv(paths["attention_primary"])
    attention_supplement = read_csv(paths["attention_supplement"])
    for cohort_name, attention in (("primary", attention_primary), ("supplement", attention_supplement)):
        require(attention, f"{model} {cohort_name} attention is empty")
        require(all(row.get("site_id") == "answer_query_v2" for row in attention), f"{model} {cohort_name} wrong answer-query site")
        require(all(boolish(row.get("trace_one_to_one")) for row in attention), f"{model} {cohort_name} attention contains non-one-to-one rows")
        require(all(row.get("alignment_strategy") == "literal_baseline_token_prefix" for row in attention), f"{model} {cohort_name} has non-literal query alignment")
        require(all(finite(row.get("target_needle_raw_mass")) and finite(row.get("target_needle_relative_mass")) for row in attention), f"{model} {cohort_name} has invalid exact-span mass")
        require(all(row.get("target_needle_relative_denominator") == "all_prompt_attention_mass" for row in attention), f"{model} {cohort_name} relative-mass denominator mismatch")
        rows_per_request = Counter(row["request_id"] for row in attention)
        require(len(set(rows_per_request.values())) == 1, f"{model} {cohort_name} attention head rows are incomplete")

    primary_attention_requests = {row["request_id"] for row in attention_primary}
    supplement_attention_requests = {row["request_id"] for row in attention_supplement}
    primary_confirmation_requests = {
        row["request_id"] for row in attention_primary
        if row.get("split") == "confirmation"
    }
    supplement_confirmation_requests = {
        row["request_id"] for row in attention_supplement
        if row.get("split") == "confirmation"
    }
    require(supplement_attention_requests == supplement_ids, f"{model} supplement attention request set mismatch")
    legacy_by_request = {row["request_id"]: row for row in attention_primary}
    legacy_present = sum(boolish(row.get("legacy_answer_query_present")) for row in legacy_by_request.values())
    legacy_same = sum(boolish(row.get("legacy_answer_query_same_endpoint")) for row in legacy_by_request.values())
    require(legacy_same <= legacy_present, f"{model} impossible legacy alias audit")

    plan = read_csv(paths["head_plan"])
    plan_audit = read_json(paths["head_plan_audit"])
    require(plan_audit.get("selection_split") == "discovery", f"{model} head plan not discovery-selected")
    require(plan_audit.get("confirmation_used_for_selection") is False, f"{model} confirmation used to select heads")
    require({int(row["bank_size"]) for row in plan} == REGISTERED_K, f"{model} missing registered K")
    require(all(finite(row.get("target_needle_raw_mass")) and finite(row.get("target_needle_relative_mass")) for row in plan), f"{model} plan missing exact-span mass")

    head_primary = read_jsonl(paths["head_primary"])
    head_supplement = read_jsonl(paths["head_supplement"])
    for cohort_name, trials, expected_requests in (
        ("primary", head_primary, primary_confirmation_requests),
        ("supplement", head_supplement, supplement_confirmation_requests),
    ):
        by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in trials:
            by_request[str(row["request_id"])].append(row)
        require(set(by_request) == expected_requests, f"{model} {cohort_name} head-trial request set mismatch")
        require(all(row.get("split") == "confirmation" for row in trials), f"{model} {cohort_name} head trials include discovery")
        require(all(sum(row.get("condition") == "clean" for row in rows) == 1 for rows in by_request.values()), f"{model} {cohort_name} clean rows not request-unique")
        require(all(row.get("site_id") == "answer_query_v2" for row in trials), f"{model} {cohort_name} head trial wrong site")

    layer = read_json(paths["execution_layer"])
    pairs = read_jsonl(paths["execution_pairs"])
    execution = read_jsonl(paths["execution_trials"])
    require(layer.get("selection_split") == "discovery", f"{model} execution layer not discovery-selected")
    require(layer.get("confirmation_used_for_selection") is False, f"{model} confirmation used for execution layer")
    selected_layer = int(layer["selected_layer"])
    regression_rows = [
        row for row in read_csv(paths["representation_regression"])
        if row.get("site_kind") == "answer_query_v2"
        and row.get("cohort") == "one_to_one"
        and row.get("probe") == "ridge"
        and int(float(row["layer"])) == selected_layer
    ]
    geometry_rows = [
        row for row in read_csv(paths["representation_geometry"])
        if row.get("site_kind") == "answer_query_v2"
        and row.get("cohort") == "one_to_one"
        and int(float(row["layer"])) == selected_layer
    ]
    require(len(regression_rows) == 1 and finite(regression_rows[0].get("confirmation_r2")) and finite(regression_rows[0].get("confirmation_mae")), f"{model} selected-layer representation metrics invalid")
    require(len(geometry_rows) == 1 and finite(geometry_rows[0].get("centroid_rank_3_fraction")), f"{model} selected-layer geometry invalid")
    require(pairs, f"{model} execution pair plan is empty")
    require(all(row.get("split") == "confirmation" and row.get("selection_split") == "discovery" for row in pairs), f"{model} execution pair split violation")
    require(all(int(row["receiver_count"]) != int(row["donor_count"]) for row in pairs), f"{model} execution donor equals receiver")
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in execution:
        by_pair[str(row["pair_id"])].append(row)
    require(set(by_pair) == {str(row["pair_id"]) for row in pairs}, f"{model} execution pair coverage mismatch")
    require(all({row["condition"] for row in rows} == EXECUTION_CONDITIONS and len(rows) == 4 for rows in by_pair.values()), f"{model} execution conditions incomplete")
    require(all(row.get("receiver_site_id") == "answer_query_v2" and row.get("donor_site_id") == "answer_query_v2" for row in execution), f"{model} execution site mismatch")

    analysis = read_json(paths["analysis_audit"])
    require(analysis.get("confirmation_used_for_selection") is False, f"{model} analysis reports confirmation selection")
    require(analysis.get("ov_comparison_status") == "not_run_by_user_request", f"{model} OV scope audit mismatch")
    mass_columns = {
        "ranked_target_needle_raw_mass",
        "ranked_target_needle_relative_mass",
        "random_target_needle_raw_mass",
        "random_target_needle_relative_mass",
    }
    for name in ("head_primary_statistics", "head_supplement_statistics"):
        statistics = read_csv(paths[name])
        require(statistics and mass_columns.issubset(statistics[0]), f"{model} {name} omits exact-span mass")
        for row in statistics:
            if int(float(row.get("seed_clusters", 0))) > 0:
                require(all(finite(row.get(column)) for column in mass_columns), f"{model} {name} has invalid estimable exact-span mass")

    return (
        {
            "model_label": model,
            "status": "PASS",
            "primary_generation_rows": len(original),
            "primary_captured": len(primary_capture),
            "primary_excluded": len(primary_exclusions),
            "supplement_generation_rows": len(supplement),
            "supplement_captured": len(supplement_capture),
            "supplement_excluded": len(supplement_exclusions),
            "attention_primary_requests": len(primary_attention_requests),
            "attention_primary_rows": len(attention_primary),
            "attention_supplement_requests": len(supplement_attention_requests),
            "attention_supplement_rows": len(attention_supplement),
            "legacy_answer_query_present_requests": legacy_present,
            "legacy_answer_query_same_endpoint_requests": legacy_same,
            "head_plan_rows": len(plan),
            "head_primary_rows": len(head_primary),
            "head_supplement_rows": len(head_supplement),
            "execution_selected_layer": selected_layer,
            "execution_pairs": len(pairs),
            "execution_trial_rows": len(execution),
        },
        list(paths.values()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model_audits = []
    inputs: list[Path] = []
    for model in MODELS:
        audit, paths = audit_model(args.run_root, model)
        model_audits.append(audit)
        inputs.extend(paths)
    report = args.report_dir / "v5_native_thinking_integrated_causal_chain_report.html"
    report_manifest = args.report_dir / "v5_native_thinking_integrated_causal_chain_report_manifest.json"
    require(report.exists() and report_manifest.exists(), "Integrated report or manifest missing")
    manifest = read_json(report_manifest)
    require(manifest.get("report_sha256") == sha256(report), "Integrated report hash mismatch")
    inputs.extend([report, report_manifest])
    payload = {
        "schema_version": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "models": model_audits,
        "directory_isolation": "answer_query_extension and one_to_one supplements remain outside original 300",
        "selection_policy": "discovery only; confirmation evaluation only",
        "ov_comparison_status": "not_run_by_user_request",
        "inputs": [
            {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(set(inputs))
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
