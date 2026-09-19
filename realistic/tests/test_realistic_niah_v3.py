from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_generation.dynamic_niah import TokenizerAdapter
from realistic_niah.runner import build_requests
from realistic_niah.spec import FORMAL_PROMPT_MODES
from realistic_niah.stimuli import FreezeSpec, freeze_stimulus
from realistic_niah_v3.analysis import (
    Candidate,
    _select_candidate,
    accuracy_distribution_diagnostics,
    add_derived_predictors,
    behavior_tables,
    cross_validate_accuracy,
    cross_validate_continuous,
    exclusive_outcome_class,
    load_request_table,
    paired_mode_comparisons,
)
from realistic_niah_v3.accuracy_distributions import ACCURACY_FAMILIES
from realistic_niah_v3.native_thinking import (
    classify_native_thinking_style,
)
from realistic_niah_v3.reporting import (
    build_all_plots,
    write_behavior_report,
    write_empirical_law_report,
)
from realistic_niah_v3.sharding import (
    expected_request_ids,
    formal_shard_plan,
    resource_profile,
)
from realistic_niah_v3.scheduler import allocate_pending_tasks
from realistic_niah_v3.spec import (
    CANONICAL_TOKENIZER_REVISION,
    EXPECTED_REQUESTS,
    EXPECTED_SHARDS,
    EXPECTED_STIMULI,
    INSERTION_DEPTH_MAX_FRACTION,
    INSERTION_DEPTH_MIN_FRACTION,
    MATCHED_CONTROL_MODEL_LABELS,
    MATCHED_REASONING_MODEL_LABELS,
    MODEL_REVISIONS,
    MODEL_SPECS,
    NEEDLE_COUNTS,
    PASSAGE_LENGTHS,
    SEEDS,
    SWITCHABLE_MODEL_LABELS,
    V3_FREEZE_PROTOCOL,
    V3_RUN_PROTOCOL,
)


def test_v3_registered_grid_and_request_accounting() -> None:
    assert PASSAGE_LENGTHS == (
        2_000,
        3_000,
        5_000,
        8_000,
        10_000,
        15_000,
        20_000,
    )
    assert NEEDLE_COUNTS == tuple(range(1, 11)) + (12, 15, 18, 20)
    assert SEEDS == tuple(range(1234, 1244))
    assert EXPECTED_STIMULI == 980
    assert EXPECTED_SHARDS == 48
    assert EXPECTED_REQUESTS == 47_040
    assert len(SWITCHABLE_MODEL_LABELS) == 10
    assert len(MATCHED_CONTROL_MODEL_LABELS) == 2
    assert len(MATCHED_REASONING_MODEL_LABELS) == 2
    assert len(MODEL_SPECS) == len(MODEL_REVISIONS) == 14
    assert (
        CANONICAL_TOKENIZER_REVISION
        == MODEL_REVISIONS["Qwen3-8B"]
    )


def test_v3_formal_plan_has_48_mode_shards() -> None:
    plan = formal_shard_plan()
    assert plan["expected_stimuli_per_shard"] == 980
    assert plan["expected_shards"] == 48
    assert plan["expected_requests"] == 47_040
    assert plan["raw_checkpoint_count"] == 14
    assert plan["behavior_comparison_slots"] == 12
    assert all(task["expected_requests"] == 980 for task in plan["tasks"])
    assert all(
        int(task["gpus_required"])
        == int(task["tensor_parallel_size"])
        for task in plan["tasks"]
    )
    assert len({task["task_id"] for task in plan["tasks"]}) == 48


def test_v3_resource_profiles_and_eight_gpu_packing() -> None:
    assert resource_profile("Gemma4-31B").gpus_required == 2
    assert resource_profile("Gemma4-31B").tensor_parallel_size == 2
    assert resource_profile("Qwen3-32B").gpus_required == 1
    assert resource_profile("Qwen3-32B").gpu_memory_utilization == 0.92

    tasks = formal_shard_plan()["tasks"]
    allocations = allocate_pending_tasks(
        tasks,
        visible_gpu_ids=range(8),
    )
    allocated = [gpu for item in allocations for gpu in item.gpu_ids]
    assert len(allocated) == 8
    assert len(allocated) == len(set(allocated))
    assert all(len(item.gpu_ids) in {1, 2} for item in allocations)
    assert any(len(item.gpu_ids) == 2 for item in allocations)


def test_native_thinking_counting_style_classifier() -> None:
    assert classify_native_thinking_style("1. Paris: 4\n2. Rome: 8") == (
        "indexed_list"
    )
    assert classify_native_thinking_style("- Paris: 4\n- Rome: 8") == (
        "bullet_list"
    )
    assert classify_native_thinking_style("1. Paris\n- Rome") == (
        "mixed_structured_list"
    )
    assert classify_native_thinking_style(
        "First I found Paris; second I found Rome."
    ) == "ordinal_word_enumeration"
    assert classify_native_thinking_style("count so far: 3 -> 4") == (
        "inline_tally_or_arithmetic"
    )
    assert classify_native_thinking_style("I found several records.") == (
        "prose_reasoning"
    )
    assert classify_native_thinking_style("  ") == "no_visible_reasoning"


def test_v3_request_ids_are_namespaced_without_changing_v2_default() -> None:
    stimulus = {
        "stimulus_id": "V3_T2000_N1_seed1234",
        "passage": "Example passage.",
        "seed": 1234,
    }
    v3_requests = build_requests(
        [stimulus],
        model_spec=MODEL_SPECS["Qwen3-4B"],
        protocol=V3_RUN_PROTOCOL,
    )
    v2_requests = build_requests(
        [stimulus],
        model_spec=MODEL_SPECS["Qwen3-4B"],
    )

    assert len(v3_requests) == len(FORMAL_PROMPT_MODES)
    assert all(
        request["request_id"].startswith("v3/Qwen3-4B/")
        for request in v3_requests
    )
    assert all(
        request["request_id"].startswith("Qwen3-4B/")
        for request in v2_requests
    )


def test_v3_expected_request_ids_match_runner_namespace() -> None:
    task = next(
        task
        for task in formal_shard_plan()["tasks"]
        if task["model_label"] == "Qwen3-8B"
        and task["prompt_mode"] == "native_thinking"
    )
    assert expected_request_ids(("V3_T2000_N1_seed1234",), task) == (
        "v3/Qwen3-8B/native_thinking/"
        "cue_before_query_after/V3_T2000_N1_seed1234",
    )


def test_v3_frozen_stimulus_enforces_final_5_to_95_percent_depth() -> None:
    tokenizer = TokenizerAdapter("simple")
    spec = FreezeSpec(
        passage_lengths=(320,),
        needle_counts=(6,),
        seeds=(1234,),
        canonical_tokenizer="simple",
        max_search_attempts=12,
        max_window_retries=2,
        minimum_filler_tokens=80,
        insertion_depth_min_fraction=INSERTION_DEPTH_MIN_FRACTION,
        insertion_depth_max_fraction=INSERTION_DEPTH_MAX_FRACTION,
    )

    row = freeze_stimulus(
        target_passage_tokens=320,
        num_needles=6,
        seed=1234,
        tokenizer=tokenizer,
        spec=spec,
        protocol=V3_FREEZE_PROTOCOL,
    )

    assert row["schema_version"] == "realistic_niah_master_v3"
    assert row["protocol_version"] == "realistic_niah_v3"
    assert row["stimulus_id"].startswith("V3_")
    assert row["canonical_passage_tokens"] == 320
    assert all(
        INSERTION_DEPTH_MIN_FRACTION
        <= needle["normalized_depth"]
        <= INSERTION_DEPTH_MAX_FRACTION
        for needle in row["needles"]
    )
    assert row["insertion_depth_policy"]["minimum_inclusive"] == 0.05
    assert row["insertion_depth_policy"]["maximum_inclusive"] == 0.95


def test_v2_default_frozen_stimulus_does_not_gain_v3_fields() -> None:
    tokenizer = TokenizerAdapter("simple")
    row = freeze_stimulus(
        target_passage_tokens=160,
        num_needles=2,
        seed=1234,
        tokenizer=tokenizer,
        spec=FreezeSpec(
            passage_lengths=(160,),
            needle_counts=(2,),
            seeds=(1234,),
            canonical_tokenizer="simple",
            minimum_filler_tokens=60,
        ),
    )

    assert row["schema_version"] == "realistic_niah_master_v1"
    assert "protocol_version" not in row
    assert "insertion_depth_policy" not in row
    assert all(
        "normalized_text_depth" not in insertion
        for insertion in row["realized_insertions"]
    )


def _synthetic_analysis_rows() -> pd.DataFrame:
    rows: list[dict] = []
    for slot_index, slot in enumerate(("A", "B")):
        for seed in range(10):
            for n in (1, 2, 4, 8):
                for length in (2_000, 5_000, 10_000):
                    signed = (
                        slot_index * 0.75
                        + 0.4 * n
                        - 0.15 * (length / 1000)
                        + (seed - 4.5) * 0.01
                    )
                    exact = bool(
                        n <= 4 + slot_index
                        and length <= 5_000
                        and seed % 5 != 0
                    )
                    rows.append(
                        {
                            "comparison_slot": slot,
                            "prompt_mode": "direct",
                            "seed": seed,
                            "N": n,
                            "L": length,
                            "signed_deviation": signed,
                            "absolute_deviation": abs(signed),
                            "exact_count": exact,
                        }
                    )
    return add_derived_predictors(pd.DataFrame(rows))


def test_grouped_cv_recovers_simple_shared_continuous_law() -> None:
    rows = _synthetic_analysis_rows()
    result, coefficients = cross_validate_continuous(
        rows,
        prompt_mode="direct",
        target="signed_mean_deviation",
        candidate=Candidate("known", ("N", "L_k")),
    )

    assert result["cv_r2_mean"] > 0.99
    assert result["cv_mae_mean"] < 0.05
    estimates = {row["term"]: row["estimate"] for row in coefficients}
    assert np.isclose(estimates["N"], 0.4, atol=0.01)
    assert np.isclose(estimates["L_k"], -0.15, atol=0.01)


def test_accuracy_cv_reports_proper_scoring_rules() -> None:
    rows = _synthetic_analysis_rows()
    result, coefficients = cross_validate_accuracy(
        rows,
        prompt_mode="direct",
        candidate=Candidate("simple", ("N", "L_k")),
    )

    assert result["cv_log_loss_mean"] >= 0
    assert 0 <= result["cv_brier_mean"] <= 1
    assert np.isfinite(result["cv_deviance_explained_mean"])
    assert result["distribution_family"] == "binomial_logit"
    assert result["cv_predictive_nlpd_mean"] >= 0
    assert {row["term"] for row in coefficients} >= {"N", "L_k"}


def test_accuracy_distribution_grid_and_qq_diagnostics() -> None:
    rows = _synthetic_analysis_rows()
    results = []
    for family in ACCURACY_FAMILIES:
        result, _ = cross_validate_accuracy(
            rows,
            prompt_mode="direct",
            candidate=Candidate("linear_additive", ("N", "L_k")),
            family=family,
        )
        results.append(result)
        assert np.isfinite(result["cv_predictive_nlpd_mean"])
        assert 0 <= result["cv_brier_mean"] <= 1
    assert {result["distribution_family"] for result in results} == {
        family.name for family in ACCURACY_FAMILIES
    }

    selected = pd.DataFrame(
        [
            min(
                results,
                key=lambda result: result["cv_predictive_nlpd_mean"],
            )
        ]
    )
    residuals, diagnostics = accuracy_distribution_diagnostics(
        rows,
        selected,
    )
    assert len(residuals) == 24
    assert len(diagnostics) == 1
    assert diagnostics.iloc[0]["qq_correlation_r2"] > 0


def test_nonsignificant_interaction_cannot_win_selection() -> None:
    comparison = pd.DataFrame(
        [
            {
                "candidate": "intercept_only",
                "converged": True,
                "interaction_significant_0_05": True,
                "feature_count": 0,
                "cv_r2_mean": 0.40,
                "cv_r2_sd": 0.01,
                "cv_mae_mean": 1.0,
                "cv_folds": 5,
            },
            {
                "candidate": "linear_N",
                "converged": True,
                "interaction_significant_0_05": True,
                "feature_count": 1,
                "cv_r2_mean": 0.50,
                "cv_r2_sd": 0.01,
                "cv_mae_mean": 0.9,
                "cv_folds": 5,
            },
            {
                "candidate": "linear_interaction",
                "converged": True,
                "interaction_significant_0_05": False,
                "feature_count": 3,
                "cv_r2_mean": 0.99,
                "cv_r2_sd": 0.01,
                "cv_mae_mean": 0.1,
                "cv_folds": 5,
            },
        ]
    )

    assert _select_candidate(comparison, "signed_mean_deviation") == "linear_N"


def test_exclusive_behavior_outcome_priority() -> None:
    base = {
        "truncated": False,
        "parse_success": True,
        "signed_deviation": 0.0,
        "format_compliant": True,
    }
    assert exclusive_outcome_class(pd.Series(base)) == "strict_success"
    assert (
        exclusive_outcome_class(
            pd.Series(base | {"format_compliant": False})
        )
        == "format_only_failure"
    )
    assert (
        exclusive_outcome_class(
            pd.Series(base | {"signed_deviation": -2.0})
        )
        == "undercount"
    )
    assert (
        exclusive_outcome_class(
            pd.Series(base | {"signed_deviation": 3.0})
        )
        == "overcount"
    )
    assert (
        exclusive_outcome_class(
            pd.Series(base | {"parse_success": False})
        )
        == "parse_failure"
    )
    assert (
        exclusive_outcome_class(
            pd.Series(
                base
                | {
                    "truncated": True,
                    "parse_success": False,
                }
            )
        )
        == "truncation"
    )


def test_paired_mode_comparison_uses_shared_stimuli() -> None:
    rows: list[dict] = []
    for seed in range(4):
        for stimulus_index in range(3):
            stimulus_id = f"S{seed}_{stimulus_index}"
            rows.extend(
                [
                    {
                        "comparison_slot": "A",
                        "prompt_mode": "direct",
                        "stimulus_id": stimulus_id,
                        "seed": seed,
                        "exact_count": stimulus_index == 0,
                    },
                    {
                        "comparison_slot": "A",
                        "prompt_mode": "native_thinking",
                        "stimulus_id": stimulus_id,
                        "seed": seed,
                        "exact_count": stimulus_index < 2,
                    },
                ]
            )
    paired = paired_mode_comparisons(
        pd.DataFrame(rows),
        bootstrap_replicates=500,
    )

    assert len(paired) == 1
    row = paired.iloc[0]
    assert row["paired_stimuli"] == 12
    assert row["risk_difference_b_minus_a"] == 1 / 3
    assert row["b_only_correct"] == 4
    assert row["a_only_correct"] == 0
    assert 0 <= row["mcnemar_holm_p_value_within_slot"] <= 1


def test_v3_json_config_matches_python_registry() -> None:
    repo = Path(__file__).resolve().parents[1]
    config = json.loads(
        (repo / "configs" / "realistic_niah_v3.json").read_text(
            encoding="utf-8"
        )
    )
    assert tuple(config["target_passage_tokens"]) == PASSAGE_LENGTHS
    assert tuple(config["needle_counts"]) == NEEDLE_COUNTS
    assert tuple(config["seeds"]) == SEEDS
    assert tuple(config["switchable_models"]) == SWITCHABLE_MODEL_LABELS
    assert (
        config["canonical_tokenizer_revision"]
        == CANONICAL_TOKENIZER_REVISION
    )
    assert config["model_revisions"] == MODEL_REVISIONS
    assert config["expected_stimuli"] == EXPECTED_STIMULI
    assert config["expected_shards"] == EXPECTED_SHARDS
    assert config["expected_requests_total"] == EXPECTED_REQUESTS
    assert config["accuracy_distribution_families"] == [
        family.name for family in ACCURACY_FAMILIES
    ]
    assert config["formal_scheduler"]["maximum_gpus"] == 8
    assert (
        config["formal_scheduler"]["Gemma4-31B"][
            "tensor_parallel_size"
        ]
        == 2
    )


def test_analysis_loader_requires_and_hashes_canonical_provenance(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    dataset = run_root / "dataset"
    orchestration = run_root / "orchestration"
    canonical = run_root / "models" / "Qwen3-4B" / "main"
    dataset.mkdir(parents=True)
    orchestration.mkdir(parents=True)
    canonical.mkdir(parents=True)

    def write_json(path: Path, value: dict) -> None:
        path.write_text(
            json.dumps(value, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    stimulus = {
        "schema_version": "realistic_niah_master_v3",
        "protocol_version": "realistic_niah_v3",
        "stimulus_id": "V3_T2000_N1_seed1234",
    }
    (dataset / "stimuli.jsonl").write_text(
        json.dumps(stimulus, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_json(dataset / "manifest.json", {"protocol_version": "realistic_niah_v3"})
    write_json(
        dataset / "audit_report.json",
        {"protocol_version": "realistic_niah_v3", "passed": True},
    )
    write_json(
        orchestration / "formal_shards.json",
        {"protocol_version": "realistic_niah_v3"},
    )
    write_json(
        orchestration / "final_shard_audit.json",
        {
            "protocol_version": "realistic_niah_v3",
            "passed": True,
            "audit_only": False,
        },
    )
    request = {
        "schema_version": "realistic_niah_request_v3",
        "protocol_version": "realistic_niah_v3",
        "request_id": "v3/Qwen3-4B/direct/x",
        "model_label": "Qwen3-4B",
        "model_id": "Qwen/Qwen3-4B",
        "model_revision": MODEL_REVISIONS["Qwen3-4B"],
        "prompt_mode": "direct",
        "stimulus_id": stimulus["stimulus_id"],
        "seed": 1234,
        "gold_count": 1,
        "target_passage_tokens": 2_000,
        "evaluation": {
            "predicted_count": 1,
            "parse_status": "parsed",
            "exact_count": True,
            "registered_success": True,
            "response_format_compliant": True,
            "truncated": False,
        },
        "output_tokens": 4,
        "finish_reason": "stop",
        "reasoning_expected": False,
    }
    (canonical / "requests.jsonl").write_text(
        json.dumps(request, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_json(
        canonical / "run_manifest.json",
        {
            "protocol_version": "realistic_niah_v3",
            "completed_requests": 1,
        },
    )
    write_json(
        canonical / "qc_report.json",
        {
            "protocol_version": "realistic_niah_v3",
            "passed": True,
            "completed_requests": 1,
        },
    )

    table, sources = load_request_table(run_root)
    assert len(table) == 1
    assert table.iloc[0]["exact_count"]
    assert len(sources) == 8
    assert all(len(source["sha256"]) == 64 for source in sources)


def test_v3_html_reports_and_plots_render(tmp_path: Path) -> None:
    rows: list[dict] = []
    for slot_index, slot in enumerate(("A", "B")):
        for seed in range(4):
            for n in (1, 2, 4):
                for length in (2_000, 5_000):
                    predicted = n + slot_index - int(length == 5_000)
                    signed = float(predicted - n)
                    exact = predicted == n
                    rows.append(
                        {
                            "request_id": (
                                f"v3/{slot}/direct/{seed}/{n}/{length}"
                            ),
                            "comparison_slot": slot,
                            "model_label": slot,
                            "prompt_mode": "direct",
                            "seed": seed,
                            "N": n,
                            "L": length,
                            "predicted_count": predicted,
                            "parse_success": True,
                            "parse_status": "parsed",
                            "exact_count": exact,
                            "strict_registered_success": exact,
                            "format_compliant": True,
                            "truncated": False,
                            "signed_deviation": signed,
                            "absolute_deviation": abs(signed),
                            "output_tokens": 4,
                            "finish_reason": "stop",
                            "reasoning_expected": False,
                            "source_file": "synthetic.jsonl",
                        }
                    )
    requests = add_derived_predictors(pd.DataFrame(rows))
    summary, by_condition, outcomes = behavior_tables(requests)
    paired = pd.DataFrame(
        [
            {
                "comparison_slot": "A",
                "mode_a": "direct",
                "mode_b": "native_thinking",
                "paired_stimuli": 0,
                "risk_difference_b_minus_a": 0.0,
            }
        ]
    )
    selected = pd.DataFrame(
        [
            {
                "target": "signed_mean_deviation",
                "prompt_mode": "direct",
                "candidate": "linear_N",
                "distribution_family": "gaussian_ols",
                "cv_r2_mean": 0.5,
                "cv_mae_mean": 0.5,
                "cv_rmse_mean": 0.7,
            },
            {
                "target": "parseable_exact_accuracy",
                "prompt_mode": "direct",
                "candidate": "linear_L",
                "distribution_family": "binomial_logit",
                "cv_predictive_nlpd_mean": 0.3,
                "cv_log_loss_mean": 0.4,
                "cv_brier_mean": 0.15,
                "cv_deviance_explained_mean": 0.2,
            },
        ]
    )
    comparisons = selected.assign(
        converged=True,
        feature_count=1,
        selected=True,
    )
    coefficients = pd.DataFrame(
        [
            {
                "target": "signed_mean_deviation",
                "prompt_mode": "direct",
                "candidate": "linear_N",
                "distribution_family": "gaussian_ols",
                "term": "N",
                "estimate": 0.0,
                "standard_error": 0.1,
                "p_value": 1.0,
                "ci95_low": -0.196,
                "ci95_high": 0.196,
            },
            {
                "target": "parseable_exact_accuracy",
                "prompt_mode": "direct",
                "candidate": "linear_L",
                "distribution_family": "binomial_logit",
                "term": "L_k",
                "estimate": -0.1,
                "standard_error": 0.05,
                "p_value": 0.04,
                "ci95_low": -0.198,
                "ci95_high": -0.002,
            },
        ]
    )
    accuracy_residuals = pd.DataFrame(
        {
            "prompt_mode": ["direct"] * 5,
            "theoretical_normal_quantile": [-1.2, -0.5, 0.0, 0.5, 1.2],
            "randomized_quantile_residual": [-1.1, -0.4, 0.1, 0.6, 1.1],
        }
    )
    accuracy_diagnostics = pd.DataFrame(
        [
            {
                "prompt_mode": "direct",
                "distribution_family": "binomial_logit",
                "qq_correlation_r2": 0.99,
                "shapiro_w": 0.98,
                "shapiro_p_value": 0.7,
            }
        ]
    )

    behavior_plots, empirical_plots = build_all_plots(
        requests=requests,
        selected=selected,
        output_dir=tmp_path / "figures",
        accuracy_quantile_residuals=accuracy_residuals,
        accuracy_distribution_diagnostics=accuracy_diagnostics,
    )
    behavior_report = write_behavior_report(
        output_path=tmp_path / "reports" / "behavior_report.html",
        summary=summary,
        by_condition=by_condition,
        outcomes=outcomes,
        paired_comparisons=paired,
        plot_paths=behavior_plots,
    )
    empirical_report = write_empirical_law_report(
        output_path=tmp_path / "reports" / "empirical_law_report.html",
        selected=selected,
        comparisons=comparisons,
        coefficients=coefficients,
        plot_paths=empirical_plots,
        accuracy_distribution_diagnostics=accuracy_diagnostics,
    )

    assert behavior_report.stat().st_size > 1_000
    assert empirical_report.stat().st_size > 1_000
    assert all(path.stat().st_size > 1_000 for path in behavior_plots)
    assert all(path.stat().st_size > 1_000 for path in empirical_plots)
    assert any(path.name == "accuracy_distribution_qq.png" for path in empirical_plots)
    assert "MathJax" in empirical_report.read_text(encoding="utf-8")
    assert "<details" in behavior_report.read_text(encoding="utf-8")


def test_v3_inference_environment_is_pinned() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    requirements = (
        repo_root / "requirements-inference-v3.txt"
    ).read_text(encoding="utf-8")
    launcher = (
        repo_root / "scripts" / "launch_realistic_niah_v3.sh"
    ).read_text(encoding="utf-8")
    worker = (
        repo_root / "scripts" / "run_realistic_niah_v3_worker.sh"
    ).read_text(encoding="utf-8")

    assert "transformers==5.14.1" in requirements
    assert "vllm==0.25.1" in requirements
    assert "mistral-common>=1.8.6,<2" in requirements
    assert 'version("transformers") == "5.14.1"' in launcher
    assert 'version("vllm") == "0.25.1"' in launcher
    assert "schedule_realistic_niah_v3.py" in launcher
    assert "--max-gpus" in launcher
    assert '--tensor-parallel-size "${tensor_parallel_size}"' in worker
    assert 'CUDA_VISIBLE_DEVICES="${gpu_ids}"' in worker
