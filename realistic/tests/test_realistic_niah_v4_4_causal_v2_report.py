from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_realistic_niah_v4_4_causal_v2_report.py"
REPORT = ROOT / "reports" / "realistic_niah_v4_4_causal_v2_report.html"
DATA = ROOT / "reports" / "v4_non-thinking_causal" / "v4_4_causal_v2"


def _read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_report_is_standalone_structured_and_scientifically_bounded() -> None:
    text = REPORT.read_text(encoding="utf-8")

    assert SCRIPT.is_file()
    assert "<!doctype html>" in text.lower()
    assert 'lang="zh-CN"' in text
    assert text.count("<figure") == 5
    assert text.count("<figcaption") == 5
    assert text.count("本节结论") >= 18
    assert "Clean-correct patching 强化 hidden-state 结论" in text
    assert "Qwen 全样本 n=2、correct-only n=4" in text
    assert "Gemma 两类均 n=1" in text
    assert "不尝试证明唯一计数回路" in text
    assert "fresh-seed discovery" in text
    assert "不能说 CI 严格排除零" in text
    assert "n=6–32 已保存为额外诊断" in text
    assert "random controls 平均与 ranked bank 重叠 1 个 head" in text
    assert "@media(max-width:860px)" in text
    assert "@media print" in text
    assert "linear-gradient" not in text
    assert "TODO" not in text
    assert "@@" not in text
    element_ids = re.findall(r'\bid="([^"]+)"', text)
    assert len(element_ids) == len(set(element_ids))


def test_headline_values_audits_commits_and_hashes_match_sources() -> None:
    summary = json.loads((DATA / "report_summary.json").read_text(encoding="utf-8"))

    assert summary["schema_version"] == "realistic_niah_v4_4_causal_v2_integrated_report_v3"
    assert summary["implementation_commit"] == "dd409f2dff82ccd6400dfc3d7704025cb6939940"
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        assert summary["audits"][model]["status"] == "PASS"
        assert summary["audits"][model]["checks"] == 302
        assert summary["audits"][model]["errors"] == 0
        assert summary["exports"][model]["archive_sha256_verified"] is True
        assert summary["exports"][model]["source_copy_manifests_identical"] is True

    correct = summary["correct_interventions"]
    assert correct["audit"]["status"] == "PASS"
    assert correct["audit"]["checks"] == 98
    assert correct["audit"]["passed"] == 98
    assert correct["audit"]["errors"] == 0
    assert correct["audit"]["definition_sha256"] == "6f7f7760f53a2bab08e5b840aa765dbf70d853a75952eabb5282d108b4315f5e"
    assert correct["models"]["Qwen3-8B"]["design_hash"] == "4c3cdeb48cbf"
    assert correct["models"]["Gemma4-E4B"]["design_hash"] == "d419daff86de"
    assert {row["implementation_commit"] for row in correct["models"].values()} == {
        "cda0d092db424d4bcb712a1402b899df1bee793b"
    }

    pooled = correct["patch_pooled"]
    expected = {
        "Qwen3-8B::prompt_patching": (1161, 1424),
        "Gemma4-E4B::prompt_patching": (1000, 1088),
        "Qwen3-8B::answer_patching": (1628, 1686),
        "Gemma4-E4B::answer_patching": (1809, 1884),
    }
    for key, (successes, denominator) in expected.items():
        row = pooled[key]
        assert row["patching_acc_successes"] == successes
        assert row["patching_acc_denominator"] == denominator
        assert math.isclose(row["pooled_average_patching_acc"], successes / denominator)


def test_pooled_patching_is_recomputed_from_group_successes_and_denominators() -> None:
    aggregate = _read_csv("correct_patching_aggregate.csv")
    pooled = {
        (row["model_label"], row["family"]): row
        for row in _read_csv("correct_patching_pooled.csv")
    }
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in aggregate:
        grouped[(row["model_label"], row["family"])].append(row)

    assert len(aggregate) == 24
    for key, rows in grouped.items():
        assert len(rows) == 6
        assert {int(row["seed_clusters"]) for row in rows} == {5}
        successes = sum(int(row["patching_acc_successes"]) for row in rows)
        denominator = sum(int(row["patching_acc_denominator"]) for row in rows)
        output = pooled[key]
        assert int(output["patching_acc_successes"]) == successes
        assert int(output["patching_acc_denominator"]) == denominator
        assert math.isclose(float(output["pooled_average_patching_acc"]), successes / denominator)
        assert math.isclose(
            float(output["group_min_average_patching_acc"]),
            min(float(row["average_patching_acc"]) for row in rows),
        )


def test_ablation_candidates_are_maxima_only_within_main_n_one_to_five() -> None:
    main = _read_csv("dual_population_ablation_top_n_1_5.csv")
    candidates = {
        (row["model_label"], row["analysis_population"]): row
        for row in _read_csv("ablation_candidate_summary.csv")
    }
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in main:
        assert 1 <= int(row["top_n"]) <= 5
        grouped[(row["model_label"], row["analysis_population"])].append(row)

    assert len(main) == 20
    for key, rows in grouped.items():
        expected = max(rows, key=lambda row: (float(row["primary_effect"]), -int(row["top_n"])))
        output = candidates[key]
        assert int(output["candidate_top_n"]) == int(expected["top_n"])
        assert math.isclose(float(output["primary_effect"]), float(expected["primary_effect"]))
        assert output["selection_status"] == "candidate_from_unfrozen_n_1_to_5_discovery"

    assert int(candidates[("Qwen3-8B", "all_examples_signed")]["candidate_top_n"]) == 2
    assert int(candidates[("Qwen3-8B", "clean_correct_only")]["candidate_top_n"]) == 4
    assert int(candidates[("Gemma4-E4B", "all_examples_signed")]["candidate_top_n"]) == 1
    gemma_correct = candidates[("Gemma4-E4B", "clean_correct_only")]
    assert int(gemma_correct["candidate_top_n"]) == 1
    assert float(gemma_correct["ci95_low"]) == 0.0
    assert gemma_correct["ci95_excludes_zero_positive"] == "False"


def test_supplement_shortages_and_final_coverage_are_explicit() -> None:
    rows = {row["model_label"]: row for row in _read_csv("supplement_seed_summary.csv")}
    assert rows["Qwen3-8B"]["patch_initial_missing"] == "k=3 increase: 1; k=3 decrease: 1; k=5 increase: 3; k=5 decrease: 3"
    assert rows["Gemma4-E4B"]["patch_initial_missing"] == "k=5 increase: 4; k=5 decrease: 4"
    assert rows["Qwen3-8B"]["patch_added_eligible_pair_seeds"] == "1274;1275;1276;1278"
    assert rows["Gemma4-E4B"]["patch_added_eligible_pair_seeds"] == "1275;1277;1281;1295"
    for row in rows.values():
        assert int(row["patch_final_missing_groups"]) == 0
        assert int(row["correct_initial_missing_seed_clusters"]) == 10
        assert int(row["correct_final_seed_clusters"]) == 10
        assert int(row["correct_target_seed_clusters"]) == 10


def test_machine_tables_and_source_ledger_are_complete() -> None:
    expected_files = {
        "ablation_candidate_summary.csv",
        "correct_interventions_audit.csv",
        "correct_interventions_stage_summary.csv",
        "correct_patching_aggregate.csv",
        "correct_patching_pooled.csv",
        "correct_prompt_alignment_summary.csv",
        "dual_population_ablation_diagnostics.csv",
        "dual_population_ablation_top_n_1_5.csv",
        "primary_confirmation_conditions.csv",
        "primary_confirmation_family_summary.csv",
        "report_summary.json",
        "source_ledger.csv",
        "supplement_seed_summary.csv",
    }
    assert expected_files <= {path.name for path in DATA.iterdir()}

    ledger = _read_csv("source_ledger.csv")
    assert len(ledger) >= 55
    assert len({row["source_label"] for row in ledger}) == len(ledger)
    for row in ledger:
        assert row["source_path"]
        assert int(row["size_bytes"]) > 0
        assert len(row["sha256"]) == 64
        int(row["sha256"], 16)


def test_original_family_summary_and_stage_inventory_remain_recomputable() -> None:
    rows = _read_csv("primary_confirmation_conditions.csv")
    summaries = {
        (row["model_label"], row["family"]): row
        for row in _read_csv("primary_confirmation_family_summary.csv")
    }
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_label"], row["family"])].append(row)
    for key, selected in grouped.items():
        effects = [float(row["mean_control_adjusted_transport"]) for row in selected]
        assert int(summaries[key]["conditions"]) == len(selected)
        assert math.isclose(float(summaries[key]["mean_effect"]), statistics.fmean(effects))
        assert math.isclose(float(summaries[key]["median_effect"]), statistics.median(effects))

    stages = _read_csv("stage_inventory.csv")
    assert len(stages) == 12
    for row in stages:
        assert row["status"] == "complete"
        assert int(row["successful_rows"]) == int(row["logical_rows"])
        assert int(row["skipped_rows"]) == 0
