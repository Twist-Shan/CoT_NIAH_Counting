import csv
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from counting.feature_analysis import (
    StageTimer,
    TokenizedCountingExample,
    build_counting_feature_cache_config,
    counting_dataset_count_summary,
    counterfactual_insertion_positions,
    build_counting_feature_run_config,
    load_counting_feature_config_file,
    restore_counting_feature_cache,
    save_counting_feature_cache,
    validate_counting_feature_cache,
    validate_counting_dataset_count,
    build_needle_token_mask,
    build_target_count_matrix,
    build_target_count_vector,
    evaluate_classification_probe,
    evaluate_ridge_probe,
    extract_hidden_states_by_layer,
    filter_successful_rows,
    fit_classification_probe,
    fit_contrastive_success_direction,
    fit_counterfactual_count_direction,
    fit_ridge_probe,
    plot_probe_2d_projection,
    prepare_contrastive_examples_for_position,
    prepare_feature_examples_for_position,
    run_needle_sensitivity_analysis,
    sample_probe_positions,
    run_counting_probe_diagnostics,
    save_contrastive_success_direction,
    save_counterfactual_count_direction,
    save_target_artifacts,
    select_contrastive_success_examples,
    sequence_length_stats,
    train_test_split_indices,
    torch_gpu_memory_snapshot,
    warn_if_large_file,
    write_response_generation_checkpoint_archive,
)


class _FakeEmbedding:
    def __init__(self) -> None:
        self.weight = torch.empty(1, device="cpu")


class _FakeHiddenStateModel:
    def __init__(self, num_layers: int = 4, hidden_dim: int = 2) -> None:
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.calls: list[dict[str, object]] = []

    def get_input_embeddings(self) -> _FakeEmbedding:
        return _FakeEmbedding()

    def __call__(
        self,
        input_ids: torch.Tensor,
        *,
        output_hidden_states: bool = False,
        use_cache: bool | None = None,
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "shape": tuple(input_ids.shape),
                "output_hidden_states": output_hidden_states,
                "use_cache": use_cache,
            }
        )
        batch, seq_len = input_ids.shape
        states = []
        for layer in range(self.num_layers):
            state = torch.full(
                (batch, seq_len, self.hidden_dim),
                float(layer),
                dtype=torch.float32,
                device=input_ids.device,
            )
            states.append(state)
        return SimpleNamespace(hidden_states=tuple(states))


class _TokenSensitiveHiddenStateModel(_FakeHiddenStateModel):
    def __call__(
        self,
        input_ids: torch.Tensor,
        *,
        output_hidden_states: bool = False,
        use_cache: bool | None = None,
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "shape": tuple(input_ids.shape),
                "output_hidden_states": output_hidden_states,
                "use_cache": use_cache,
            }
        )
        values = input_ids.to(dtype=torch.float32)
        seq_sum = values.sum(dim=1, keepdim=True)
        positions = torch.arange(
            input_ids.shape[1], dtype=torch.float32, device=input_ids.device
        ).unsqueeze(0)
        states = []
        for layer in range(self.num_layers):
            state = torch.stack(
                [
                    values + float(layer) + 0.01 * seq_sum.expand_as(values),
                    positions.expand_as(values) + float(layer),
                ],
                dim=-1,
            )
            states.append(state)
        return SimpleNamespace(hidden_states=tuple(states))


def test_build_counting_feature_run_config_resolves_defaults_and_overrides() -> None:
    cfg = build_counting_feature_run_config()

    assert cfg["SMOKE_TEST"] is False
    assert cfg["NUM_EXAMPLES"] == 100
    assert cfg["NUM_MAX_NEEDLES"] is None
    assert cfg["RANDOMIZE_NEEDLE_INSERTION"] is True
    assert cfg["SENTENCE_LEVEL_INSERTION"] is True
    assert cfg["LAYERS"] == [16, 20, 24, 28]
    assert cfg["TOKENIZER_NAME"] == cfg["MODEL_NAME"]
    assert cfg["PROMPT_STYLE"] == "vanilla"
    assert cfg["USE_KV_CACHE_FOR_NONTHINKG"] is True
    assert cfg["SAVE_GENERATED_DATA"] is True
    assert cfg["TARGET_COUNT_TYPE"] == "interpolation"
    assert cfg["FILTER_EXAMPLE"] is True
    assert cfg["RUN_CLASSIFICATION"] is False
    assert cfg["CLASSIFIER_EPOCHS"] == 200
    assert cfg["MAX_TRAIN_TOKENS_PER_LAYER"] == 50_000
    assert cfg["MAX_EVAL_TOKENS_PER_LAYER"] == 50_000
    assert cfg["RUN_COUNTING_FEATURE_CALC"] is True
    assert cfg["RUN_NEEDLE_SENSITIVITY"] is False
    assert cfg["NUM_REMOVAL"] == 3
    assert cfg["TARGET_SENSITIVITY_POSITION"] == "last-token"
    assert cfg["NEEDLE_SENSITIVITY_SEED"] == 42
    assert cfg["MAX_SENSITIVITY_EXAMPLES"] is None
    assert cfg["COUNTING_FEATURE_CALC_METHOD"] == "counterfactual"
    assert cfg["COUNTING_PROBE_MODE"] == "direct"
    assert cfg["FEATURE_CALC_POS"] == "needle-last"
    assert cfg["COUNTERFACTUAL_REMOVED_NEEDLE_INDEX"] == 0
    assert cfg["STEERING_POSITION_MODE"] == "needle_span"
    assert cfg["STEERING_COEFF"] == [-2, -1, -0.5, 0.5, 1, 2, 3, 4, 6]
    assert cfg["STEERING_TEST_EVAL"] is False
    assert cfg["NUM_MAX_NEEDLES_STEERING_EVAL"] == 5
    assert cfg["NUM_EXAMPLES_STEERING_EVAL"] == 20
    assert cfg["STEERING_EVAL_RANDOMIZE_NEEDLE_MIN_SEPARATION"] == 20
    assert (
        cfg["FACT_TEMPLATES_PATH"]
        == "data/templates/niah_fact_single_template.txt"
    )

    contrastive = build_counting_feature_run_config(
        {"COUNTING_FEATURE_CALC_METHOD": "contrastive-success"}
    )
    assert contrastive["COUNTING_FEATURE_CALC_METHOD"] == "contrastive-success"
    assert contrastive["FEATURE_CALC_POS"] == "needle-last"

    needle_last = build_counting_feature_run_config(
        {"FEATURE_CALC_POS": "needle-last"}
    )
    assert needle_last["FEATURE_CALC_POS"] == "needle-last"

    counterfactual = build_counting_feature_run_config(
        {
            "COUNTING_FEATURE_CALC_METHOD": "counterfactual",
            "COUNTERFACTUAL_REMOVED_NEEDLE_INDEX": 1,
            "RUN_COUNTING_FEATURE_CALC": False,
            "STEERING_TEST_EVAL": True,
        }
    )
    assert counterfactual["COUNTING_FEATURE_CALC_METHOD"] == "counterfactual"
    assert counterfactual["COUNTERFACTUAL_REMOVED_NEEDLE_INDEX"] == 1
    assert counterfactual["RUN_COUNTING_FEATURE_CALC"] is False
    assert counterfactual["STEERING_TEST_EVAL"] is True

    probe = build_counting_feature_run_config(
        {
            "COUNTING_FEATURE_CALC_METHOD": "ridge",
            "COUNTING_PROBE_MODE": "final_token",
            "NUM_MAX_NEEDLES": 5,
        }
    )
    assert probe["COUNTING_PROBE_MODE"] == "final_token"
    assert probe["NUM_MAX_NEEDLES"] == 5
    assert probe["RUN_COUNTING_PROBE_BASELINE"] is True
    assert probe["COUNTING_PROBE_BASELINE_MIN_DISTANCE"] == 5

    alias_probe = build_counting_feature_run_config(
        {
            "COUNTING_FEATURE_CALC_METHOD": "ridge",
            "COUNTING_PROBE_MODE": "mean_final_token",
        }
    )
    assert alias_probe["COUNTING_PROBE_MODE"] == "final_token"
    occurrence_probe = build_counting_feature_run_config(
        {
            "COUNTING_FEATURE_CALC_METHOD": "ridge",
            "COUNTING_PROBE_MODE": "occurrence_index_probe",
        }
    )
    assert occurrence_probe["COUNTING_PROBE_MODE"] == "occurrence_index_probe"
    occurrence_alias_probe = build_counting_feature_run_config(
        {
            "COUNTING_FEATURE_CALC_METHOD": "ridge",
            "COUNTING_PROBE_MODE": "mean_across_examples",
        }
    )
    assert occurrence_alias_probe["COUNTING_PROBE_MODE"] == "occurrence_index_probe"

    with pytest.raises(ValueError, match="Unknown counting-feature config override"):
        build_counting_feature_run_config({"TYPO": 1})

    with pytest.raises(ValueError, match="COUNTING_FEATURE_CALC_METHOD"):
        build_counting_feature_run_config({"COUNTING_FEATURE_CALC_METHOD": "mean-diff"})

    with pytest.raises(ValueError, match="FEATURE_CALC_POS"):
        build_counting_feature_run_config({"FEATURE_CALC_POS": "needle"})

    with pytest.raises(ValueError, match="COUNTING_PROBE_MODE"):
        build_counting_feature_run_config({"COUNTING_PROBE_MODE": "occurrence_mean"})

    with pytest.raises(ValueError, match="COUNTING_FEATURE_CALC_METHOD='ridge'"):
        build_counting_feature_run_config({"COUNTING_PROBE_MODE": "final_token"})

    with pytest.raises(ValueError, match="NUM_MAX_NEEDLES"):
        build_counting_feature_run_config({"NUM_MAX_NEEDLES": 0})

    with pytest.raises(ValueError, match="COUNTING_PROBE_BASELINE_MIN_DISTANCE"):
        build_counting_feature_run_config(
            {"COUNTING_PROBE_BASELINE_MIN_DISTANCE": -1}
        )

    with pytest.raises(ValueError, match="NUM_REMOVAL"):
        build_counting_feature_run_config({"NUM_REMOVAL": 0})

    with pytest.raises(ValueError, match="TARGET_SENSITIVITY_POSITION"):
        build_counting_feature_run_config({"TARGET_SENSITIVITY_POSITION": "span-last"})

    with pytest.raises(ValueError, match="MAX_SENSITIVITY_EXAMPLES"):
        build_counting_feature_run_config({"MAX_SENSITIVITY_EXAMPLES": 0})

    with pytest.raises(ValueError, match="NUM_MAX_NEEDLES_STEERING_EVAL"):
        build_counting_feature_run_config({"NUM_MAX_NEEDLES_STEERING_EVAL": 0})

    with pytest.raises(ValueError, match="STEERING_EVAL_RANDOMIZE_NEEDLE_MIN_SEPARATION"):
        build_counting_feature_run_config(
            {"STEERING_EVAL_RANDOMIZE_NEEDLE_MIN_SEPARATION": -1}
        )


def test_stage_timer_records_and_saves_success_skip_and_failure(tmp_path: Path) -> None:
    json_path = tmp_path / "timing_summary.json"
    csv_path = tmp_path / "timing_summary.csv"
    timer = StageTimer(json_path=json_path, csv_path=csv_path)

    with timer.stage("success_stage", detail="ok"):
        pass
    timer.mark_skipped("skipped_stage", reason="not requested")
    with pytest.raises(RuntimeError, match="boom"):
        with timer.stage("failed_stage"):
            raise RuntimeError("boom")

    assert json_path.exists()
    assert csv_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    statuses = {row["stage"]: row["status"] for row in payload["records"]}
    assert statuses == {
        "success_stage": "completed",
        "skipped_stage": "skipped",
        "failed_stage": "failed",
    }
    assert payload["records"][0]["elapsed_seconds"] >= 0
    assert payload["records"][2]["error_type"] == "RuntimeError"
    assert payload["records"][0]["gpu_memory_start"]["cuda_available"] is torch.cuda.is_available()
    assert payload["records"][0]["gpu_memory_end"]["cuda_available"] is torch.cuda.is_available()
    snapshot = torch_gpu_memory_snapshot()
    assert snapshot["cuda_available"] is torch.cuda.is_available()

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["stage"] for row in rows] == [
        "success_stage",
        "skipped_stage",
        "failed_stage",
    ]
    assert '"detail": "ok"' in rows[0]["metadata"]
    assert "cuda_available" in rows[0]["gpu_memory_start"]


def test_write_response_generation_checkpoint_archive_includes_scored_outputs(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    generate_data_dir = run_dir / "generate_data"
    tables_dir = run_dir / "tables"
    generate_data_dir.mkdir(parents=True)
    tables_dir.mkdir()
    dataset_path = generate_data_dir / "dynamic_niah_v2.jsonl"
    config_path = generate_data_dir / "config.used.json"
    predictions_path = tables_dir / "predictions.jsonl"
    metrics_path = tables_dir / "metrics.json"
    metadata_path = tables_dir / "counting_feature_run_metadata.json"
    timing_json_path = tables_dir / "timing_summary.json"
    timing_csv_path = tables_dir / "timing_summary.csv"
    for path, text in [
        (dataset_path, '{"id": "row-0"}\n'),
        (config_path, "{}"),
        (predictions_path, '{"exact_match": true}\n'),
        (metrics_path, '{"exact_match_accuracy": 1.0}'),
        (metadata_path, '{"run_name": "run"}'),
        (timing_json_path, '{"records": []}'),
        (timing_csv_path, "stage,status\n"),
    ]:
        path.write_text(text, encoding="utf-8")

    archive_path = write_response_generation_checkpoint_archive(
        run_dir=run_dir,
        archive_path=run_dir / "response_checkpoint.zip",
        dataset_path=dataset_path,
        config_path=config_path,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        metadata_path=metadata_path,
        timing_json_path=timing_json_path,
        timing_csv_path=timing_csv_path,
    )

    with zipfile.ZipFile(archive_path) as zf:
        names = set(zf.namelist())
    assert names == {
        "generate_data/dynamic_niah_v2.jsonl",
        "generate_data/config.used.json",
        "tables/predictions.jsonl",
        "tables/metrics.json",
        "tables/counting_feature_run_metadata.json",
        "tables/timing_summary.json",
        "tables/timing_summary.csv",
    }


def test_load_counting_feature_config_file_uses_config_and_ignores_notes(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "counting_analysis.json"
    config_path.write_text(
        json.dumps(
            {
                "_description": "Human-readable metadata only.",
                "config": {
                    "NUM_EXAMPLES": 12,
                    "STEERING_TEST_EVAL": True,
                    "FEATURE_CALC_POS": "last",
                },
                "notes": {
                    "NUM_EXAMPLES": "This text must not affect the run config.",
                    "TYPO": "Notes may mention non-config words safely.",
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_counting_feature_config_file(config_path)
    assert loaded == {
        "NUM_EXAMPLES": 12,
        "STEERING_TEST_EVAL": True,
        "FEATURE_CALC_POS": "last",
    }
    cfg = build_counting_feature_run_config(loaded)
    assert cfg["NUM_EXAMPLES"] == 12
    assert cfg["STEERING_TEST_EVAL"] is True
    assert cfg["FEATURE_CALC_POS"] == "last"


def test_counting_feature_config_precedence_and_unknown_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "counting_analysis.json"
    config_path.write_text(
        json.dumps(
            {
                "config": {
                    "NUM_EXAMPLES": 12,
                    "STEERING_TEST_EVAL": False,
                    "FEATURE_CALC_POS": "last",
                },
                "notes": {
                    "FEATURE_CALC_POS": "Notebook overrides should win below."
                },
            }
        ),
        encoding="utf-8",
    )

    file_overrides = load_counting_feature_config_file(config_path)
    notebook_overrides = {
        "FEATURE_CALC_POS": "needle-last",
        "STEERING_TEST_EVAL": True,
    }
    cfg = build_counting_feature_run_config(
        {**file_overrides, **notebook_overrides}
    )

    assert cfg["NUM_EXAMPLES"] == 12
    assert cfg["FEATURE_CALC_POS"] == "needle-last"
    assert cfg["STEERING_TEST_EVAL"] is True

    bad_path = tmp_path / "bad_counting_analysis.json"
    bad_path.write_text(json.dumps({"config": {"TYPO": 1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown counting-feature config override"):
        build_counting_feature_run_config(load_counting_feature_config_file(bad_path))


def test_validate_counting_dataset_count_checks_gold_and_inserted_counts() -> None:
    rows = [
        {
            "gold_answer": {"count": 3},
            "relevant_records": [{}, {}, {}],
            "realized_insertions": [{}, {}, {}],
            "needles": [
                {"is_inserted": True},
                {"is_inserted": True},
                {"is_inserted": True},
            ],
        },
        {
            "gold_answer": {"count": 3},
            "relevant_records": [{}, {}, {}],
            "realized_insertions": [{}, {}, {}],
            "needles": [
                {"is_inserted": True},
                {"is_inserted": True},
                {"is_inserted": True},
            ],
        },
    ]

    summary = validate_counting_dataset_count(
        rows, expected_count=3, label="original"
    )

    assert summary == {
        "num_rows": 2,
        "gold_count_distribution": {3: 2},
        "relevant_record_count_distribution": {3: 2},
        "realized_insertion_count_distribution": {3: 2},
        "inserted_needle_count_distribution": {3: 2},
    }
    assert counting_dataset_count_summary(rows)["gold_count_distribution"] == {3: 2}

    bad = [dict(rows[0], gold_answer={"count": 2})]
    with pytest.raises(ValueError, match="does not uniformly contain expected count 3"):
        validate_counting_dataset_count(bad, expected_count=3, label="bad")


def _segments():
    return [
        {"needle_id": "N1", "start": 2, "end": 5},
        {"needle_id": "N2", "start": 7, "end": 9},
    ]


def test_build_target_count_vector_jump_conventions() -> None:
    left = build_target_count_vector(
        11, _segments(), target_count_type="left_jump", matching_needle_ids=["N1", "N2"]
    )
    right = build_target_count_vector(
        11,
        _segments(),
        target_count_type="right_jump",
        matching_needle_ids=["N1", "N2"],
    )
    interp = build_target_count_vector(
        11,
        _segments(),
        target_count_type="interpolation",
        matching_needle_ids=["N1", "N2"],
    )

    assert left.tolist() == pytest.approx([0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2])
    assert right.tolist() == pytest.approx([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    assert interp.tolist() == pytest.approx([0, 0, 0, 1 / 3, 2 / 3, 1, 1, 1, 1.5, 2, 2])


def test_target_matrix_pads_with_nan_and_uses_matching_needles() -> None:
    examples = [
        TokenizedCountingExample(0, "row0", [1] * 6, _segments(), ["N1"]),
        TokenizedCountingExample(1, "row1", [1] * 11, _segments(), ["N1", "N2"]),
    ]
    target = build_target_count_matrix(examples, target_count_type="right_jump")

    assert target.shape == (2, 11)
    assert torch.isnan(target[0, 6:]).all()
    assert target[0, :6].tolist() == pytest.approx([0, 0, 0, 0, 0, 1])
    assert target[1, :].tolist() == pytest.approx([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2])


def test_sample_probe_positions_keeps_needle_tokens_under_cap() -> None:
    target = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 2, 2, 2]], dtype=torch.float32)
    needle = torch.zeros_like(target, dtype=torch.bool)
    needle[0, 2:5] = True
    sample = sample_probe_positions(target, needle_mask=needle, max_tokens=5, seed=0)

    assert sample.num_sampled == 5
    assert {2, 3, 4}.issubset(set(sample.token_indices.tolist()))


def test_ridge_probe_recovers_synthetic_linear_count_feature() -> None:
    torch.manual_seed(0)
    n, t, d = 5, 12, 4
    target = torch.arange(t, dtype=torch.float32).repeat(n, 1) % 3
    true_u = torch.tensor([1.5, -0.5, 0.25, 2.0])
    hidden = torch.randn(n, t, d) * 0.01
    hidden += target[..., None] * true_u[None, None, :]

    result = fit_ridge_probe(
        hidden, target, alpha=1e-3, max_train_tokens=None, standardize=True
    )
    metrics = evaluate_ridge_probe(result, hidden, target, max_tokens=None)

    assert result.metrics["r2"] > 0.99
    assert metrics["r2"] > 0.99
    assert metrics["mae"] < 0.05


def test_run_counting_probe_diagnostics_saves_all_diagnostic_outputs(
    tmp_path: Path,
) -> None:
    all_rows = []
    all_examples = []
    counts = [1, 2, 3, 2, 3]
    for idx, count in enumerate(counts):
        needle_segments = [
            {"needle_id": f"N{j+1}", "start": 1 + 2 * j, "end": 2 + 2 * j}
            for j in range(count)
        ]
        all_rows.append(
            {
                "id": f"row-{idx}",
                "task_type": "match_count",
                "gold_answer": {"count": count},
                "relevant_records": [
                    {"needle_id": f"N{j+1}"} for j in range(count)
                ],
                "exact_match": idx < 3,
            }
        )
        all_examples.append(
            TokenizedCountingExample(
                example_index=idx,
                row_id=f"row-{idx}",
                input_ids=list(range(10)),
                needle_segments=needle_segments,
                matching_needle_ids=[f"N{j+1}" for j in range(count)],
            )
        )
    rows = [row for row in all_rows if row["exact_match"]]
    examples = [
        example for row, example in zip(all_rows, all_examples) if row["exact_match"]
    ]
    hidden = torch.zeros((len(rows), 10, 3), dtype=torch.float32)
    extra_hidden = torch.zeros((len(all_rows), 10, 3), dtype=torch.float32)
    for idx, count in enumerate(counts):
        extra_hidden[idx, 9, 0] = float(count)
        for segment in all_examples[idx].needle_segments:
            extra_hidden[idx, int(segment["end"]) - 1, 1] = float(count)
    for idx, row in enumerate(rows):
        count = int(row["gold_answer"]["count"])
        hidden[idx, 9, 0] = float(count)
        for segment in examples[idx].needle_segments:
            hidden[idx, int(segment["end"]) - 1, 1] = float(count)
    tensor_dir = tmp_path / "tensors"
    extra_tensor_dir = tmp_path / "extra_tensors"
    table_dir = tmp_path / "tables"
    figure_dir = tmp_path / "figures"
    tensor_dir.mkdir()
    extra_tensor_dir.mkdir()
    torch.save(hidden, tensor_dir / "hidden_layer_1.pt")
    torch.save(extra_hidden, extra_tensor_dir / "hidden_layer_1.pt")

    outputs = run_counting_probe_diagnostics(
        mode="all_diagnostics",
        rows=rows,
        examples=examples,
        split=train_test_split_indices(len(rows), test_fraction=0.34, seed=0),
        layers=[1],
        feature_tensors_dir=tensor_dir,
        feature_tables_dir=table_dir,
        feature_figures_dir=figure_dir,
        filter_summary={"filter_example": True},
        resolved_config={
            "COUNTING_FEATURE_CALC_METHOD": "ridge",
            "FILTER_EXAMPLE": True,
            "NUM_NEEDLES": 3,
            "NUM_MAX_NEEDLES": 3,
        },
        ridge_alpha=0.01,
        standardize_features=True,
        run_baseline=True,
        baseline_min_distance=1,
        split_seed=0,
        scored_rows=all_rows,
        extra_eval_rows=all_rows,
        extra_eval_examples=all_examples,
        extra_eval_feature_tensors_dir=extra_tensor_dir,
    )

    assert set(outputs) == {
        "occurrence_index_probe",
        "mean_across_needles_span_last",
        "mean_across_needles_span_mean",
        "final_token",
        "ridge_vector_similarity",
    }
    occurrence_tensors = tensor_dir / "probe_diagnostics" / "occurrence_index_probe"
    occurrence_tables = table_dir / "probe_diagnostics" / "occurrence_index_probe"
    assert outputs["occurrence_index_probe"]["summary"][0]["status"] == "fit"
    assert (occurrence_tensors / "ridge_probe_layer_1.pt").exists()
    assert (occurrence_tensors / "layer_1_prototypes.pt").exists()
    assert (occurrence_tables / "prototype_metadata_layer_1.csv").exists()
    assert (occurrence_tables / "prototype_geometry_layer_1.csv").exists()
    mode_tensors = tensor_dir / "probe_diagnostics" / "final_token"
    mode_tables = table_dir / "probe_diagnostics" / "final_token"
    mode_figures = figure_dir / "probe_diagnostics" / "final_token"
    assert outputs["final_token"]["summary"][0]["status"] == "fit"
    assert outputs["final_token"]["baseline_summary"][0]["status"] == "fit"
    assert "baseline_test_r2" in outputs["final_token"]["summary"][0]
    assert (mode_tensors / "ridge_probe_layer_1.pt").exists()
    assert (mode_tensors / "baseline_ridge_probe_layer_1.pt").exists()
    assert (mode_tensors / "layer_1_X.pt").exists()
    assert (mode_tensors / "y.pt").exists()
    assert (mode_tables / "predictions_layer_1.csv").exists()
    assert (mode_tables / "baseline_predictions_layer_1.csv").exists()
    assert (mode_tables / "baseline_positions_layer_1.csv").exists()
    assert (mode_tables / "baseline_ridge_metrics.csv").exists()
    assert (mode_tables / "extra_eval_metrics.csv").exists()
    assert (mode_tables / "extra_eval_predictions_unfiltered_all_layer_1.csv").exists()
    assert (mode_tables / "extra_eval_predictions_failed_only_layer_1.csv").exists()
    assert (mode_tables / "failed_only_count_bin_metrics_layer_1.csv").exists()
    assert (mode_tables / "probe_examples.csv").exists()
    assert (mode_figures / "pred_vs_target_layer_1_train.png").exists()
    assert (mode_figures / "pred_vs_target_layer_1_test.png").exists()
    assert (mode_figures / "residual_vs_target_layer_1_test.png").exists()
    assert (mode_figures / "extra_eval_unified_layer_1_prediction_scatter.png").exists()
    assert (mode_figures / "extra_eval_unified_layer_1_residual_scatter.png").exists()
    sequence_tables = mode_tables / "sequence_projection"
    sequence_figures = mode_figures / "sequence_projection"
    assert (
        sequence_tables / "sequence_projection_examples_layer_1.csv"
    ).exists()
    assert (
        sequence_tables / "sequence_projection_values_layer_1.csv"
    ).exists()
    assert list(sequence_figures.glob("sequence_projection_layer_1_example_*.png"))
    with (
        sequence_tables / "sequence_projection_values_layer_1.csv"
    ).open(newline="", encoding="utf-8") as handle:
        sequence_rows = list(csv.DictReader(handle))
    assert sequence_rows
    assert any(row["is_matching_needle_token"] == "True" for row in sequence_rows)
    assert all(
        0.0 <= float(row["normalized_position"]) <= 1.0
        for row in sequence_rows
    )
    with (mode_tables / "extra_eval_predictions_failed_only_layer_1.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        failed_rows = list(csv.DictReader(handle))
    assert failed_rows
    assert all(row["model_exact_match"] == "False" for row in failed_rows)
    metadata = json.loads(
        (mode_tables / "diagnostic_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["NUM_MAX_NEEDLES"] == 3
    assert metadata["RUN_COUNTING_PROBE_BASELINE"] is True
    assert metadata["count_distribution"] == {"1": 1, "2": 1, "3": 1}
    assert metadata["extra_eval"]["attempted"] is True
    assert metadata["extra_eval"]["num_failed_examples"] == 2
    similarity_tables = (
        table_dir / "probe_diagnostics" / "ridge_vector_similarity"
    )
    similarity_figures = (
        figure_dir / "probe_diagnostics" / "ridge_vector_similarity"
    )
    assert (similarity_tables / "layer_1_cosine_similarity.csv").exists()
    assert (
        similarity_figures / "layer_1_cosine_similarity_heatmap.png"
    ).exists()
    with (similarity_tables / "layer_1_cosine_similarity.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        sim_rows = list(csv.DictReader(handle))
    assert [row["mode"] for row in sim_rows] == [
        "occurrence_index_probe",
        "mean_across_needles_span_last",
        "mean_across_needles_span_mean",
        "final_token",
    ]


def test_extract_hidden_states_by_layer_disables_cache_and_saves_selected_layers(
    tmp_path: Path,
) -> None:
    model = _FakeHiddenStateModel(num_layers=4, hidden_dim=2)
    examples = [
        TokenizedCountingExample(
            example_index=0,
            row_id="row-0",
            input_ids=[1, 2, 3],
            needle_segments=[],
            matching_needle_ids=[],
        ),
        TokenizedCountingExample(
            example_index=1,
            row_id="row-1",
            input_ids=[1, 2],
            needle_segments=[],
            matching_needle_ids=[],
        ),
    ]

    paths = extract_hidden_states_by_layer(
        model,
        examples,
        layers=[1, 3],
        output_dir=tmp_path,
        dtype="float32",
    )

    assert set(paths) == {1, 3}
    assert (tmp_path / "hidden_layer_1.pt").exists()
    assert (tmp_path / "hidden_layer_3.pt").exists()
    assert not (tmp_path / "hidden_layer_2.pt").exists()
    assert all(call["use_cache"] is False for call in model.calls)

    layer_1 = torch.load(tmp_path / "hidden_layer_1.pt", map_location="cpu")
    layer_3 = torch.load(tmp_path / "hidden_layer_3.pt", map_location="cpu")
    assert tuple(layer_1.shape) == (2, 3, 2)
    assert torch.all(layer_1[0, :3, :] == 1)
    assert torch.all(layer_3[1, :2, :] == 3)


def test_run_needle_sensitivity_analysis_saves_streamed_outputs(
    tmp_path: Path,
) -> None:
    model = _TokenSensitiveHiddenStateModel(num_layers=3, hidden_dim=2)
    rows = [
        {
            "id": "row-ok",
            "gold_answer": {"count": 3},
            "exact_match": True,
        },
        {
            "id": "row-skip",
            "gold_answer": {"count": 2},
            "exact_match": False,
        },
    ]
    examples = [
        TokenizedCountingExample(
            example_index=0,
            row_id="row-ok",
            input_ids=[10, 11, 101, 12, 13, 102, 14, 15, 103, 16, 17, 18],
            needle_segments=[
                {"needle_id": "N1", "start": 2, "end": 3},
                {"needle_id": "N2", "start": 5, "end": 6},
                {"needle_id": "N3", "start": 8, "end": 9},
            ],
            matching_needle_ids=["N1", "N2", "N3"],
        ),
        TokenizedCountingExample(
            example_index=1,
            row_id="row-skip",
            input_ids=[10, 11, 101, 12, 13, 102, 14, 15],
            needle_segments=[
                {"needle_id": "N1", "start": 2, "end": 3},
                {"needle_id": "N2", "start": 5, "end": 6},
            ],
            matching_needle_ids=["N1", "N2"],
        ),
    ]

    outputs = run_needle_sensitivity_analysis(
        model=model,
        rows=rows,
        examples=examples,
        layers=[1, 2],
        output_tables_dir=tmp_path / "tables",
        output_tensors_dir=tmp_path / "tensors",
        output_figures_dir=tmp_path / "figures",
        num_removal=3,
        seed=7,
        scored_rows=rows,
    )

    table_dir = tmp_path / "tables" / "needle_sensitivity"
    tensor_dir = tmp_path / "tensors" / "needle_sensitivity"
    figure_dir = tmp_path / "figures" / "needle_sensitivity"
    assert outputs["num_processed_examples"] == 1
    assert outputs["exclude_reason_counts"]["fewer_than_num_removal_needles"] == 1
    assert (table_dir / "sensitivity_examples.csv").exists()
    assert (table_dir / "sensitivity_removals_layer_1.csv").exists()
    assert (table_dir / "sensitivity_summary_layer_1.csv").exists()
    assert (table_dir / "sensitivity_metadata.json").exists()
    assert (table_dir / "sensitivity_timing_summary.json").exists()
    assert (table_dir / "sensitivity_timing_summary.csv").exists()
    assert (table_dir / "sensitivity_memory_summary.csv").exists()
    assert (tensor_dir / "mean_sensitivity_layer_1.pt").exists()
    assert (figure_dir / "dist_sensitivity_by_count_layer_1.png").exists()

    with (table_dir / "sensitivity_removals_layer_1.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        removal_rows = list(csv.DictReader(handle))
    assert len(removal_rows) == 3
    for row in removal_rows:
        removed_len = int(row["removed_span_end"]) - int(row["removed_span_start"])
        replacement_len = int(row["replacement_end"]) - int(row["replacement_start"])
        assert removed_len == replacement_len
        assert float(row["l2_distance"]) >= 0.0

    with (table_dir / "sensitivity_summary_layer_1.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        summary_rows = list(csv.DictReader(handle))
    assert len(summary_rows) == 1
    assert summary_rows[0]["target_position"] == "11"
    assert float(summary_rows[0]["dist_sensitivity"]) > 0.0

    payload = torch.load(tensor_dir / "mean_sensitivity_layer_1.pt", map_location="cpu")
    assert payload["normalized"] is True
    assert payload["mean_sensitivity_vectors"].shape == (1, 2)
    assert all(call["use_cache"] is False for call in model.calls)


def test_contrastive_success_direction_uses_last_token_mean_difference(
    tmp_path: Path,
) -> None:
    examples = [
        TokenizedCountingExample(0, "s0", [1, 2, 3], [], []),
        TokenizedCountingExample(1, "s1", [1, 2], [], []),
        TokenizedCountingExample(2, "f0", [1, 2, 3], [], []),
        TokenizedCountingExample(3, "f1", [1], [], []),
    ]
    hidden = torch.zeros(4, 3, 2)
    hidden[0, 2] = torch.tensor([3.0, 0.0])
    hidden[1, 1] = torch.tensor([1.0, 2.0])
    hidden[2, 2] = torch.tensor([1.0, 0.0])
    hidden[3, 0] = torch.tensor([1.0, 0.0])

    result = fit_contrastive_success_direction(
        hidden,
        examples,
        ["successful", "successful", "unsuccessful", "unsuccessful"],
        layer=4,
        position_mode="last",
    )

    assert result.raw_direction.tolist() == pytest.approx([1.0, 1.0])
    assert result.raw_norm == pytest.approx(2**0.5)
    assert result.direction.tolist() == pytest.approx([1 / 2**0.5, 1 / 2**0.5])
    assert result.metrics["num_successful"] == 2
    assert result.metrics["num_unsuccessful"] == 2

    path = save_contrastive_success_direction(
        result,
        tmp_path,
        selected_dataset_indices=[0, 1, 2, 3],
        selected_row_ids=["s0", "s1", "f0", "f1"],
        labels=["successful", "successful", "unsuccessful", "unsuccessful"],
    )
    payload = torch.load(path, map_location="cpu")
    assert payload["method"] == "raw_mean_difference"
    assert payload["position"] == "last"
    assert payload["selected_dataset_indices"] == [0, 1, 2, 3]


def test_contrastive_success_direction_can_use_last_matching_needle_token(
    tmp_path: Path,
) -> None:
    examples = [
        TokenizedCountingExample(0, "s0", [1] * 10, _segments(), ["N1", "N2"]),
        TokenizedCountingExample(1, "f0", [1] * 10, _segments(), ["N1", "N2"]),
    ]
    hidden = torch.zeros(2, 10, 2)
    hidden[0, 9] = torch.tensor([100.0, 100.0])
    hidden[1, 9] = torch.tensor([100.0, 100.0])
    hidden[0, 9 - 1] = torch.tensor([3.0, 1.0])
    hidden[1, 9 - 1] = torch.tensor([1.0, 1.0])

    result = fit_contrastive_success_direction(
        hidden,
        examples,
        ["successful", "unsuccessful"],
        layer=4,
        position_mode="needle-last",
    )

    assert result.raw_direction.tolist() == pytest.approx([2.0, 0.0])
    assert result.metrics["position"] == "needle-last"
    path = save_contrastive_success_direction(
        result,
        tmp_path,
        selected_dataset_indices=[0, 1],
        selected_row_ids=["s0", "f0"],
        labels=["successful", "unsuccessful"],
        position_indices=[9 - 1, 9 - 1],
    )
    payload = torch.load(path, map_location="cpu")
    assert payload["position"] == "needle-last"
    assert payload["selected_position_indices"] == [8, 8]


def test_prepare_contrastive_examples_skips_and_rebalances_for_needle_last() -> None:
    rows = [{"id": "s0"}, {"id": "s1"}, {"id": "f0"}, {"id": "f1"}]
    examples = [
        TokenizedCountingExample(0, "s0", [1] * 10, _segments(), ["N1", "N2"]),
        TokenizedCountingExample(1, "s1", [1] * 10, [], []),
        TokenizedCountingExample(2, "f0", [1] * 10, _segments(), ["N1"]),
        TokenizedCountingExample(3, "f1", [1] * 10, _segments(), ["N1", "N2"]),
    ]

    prepared = prepare_contrastive_examples_for_position(
        rows=rows,
        examples=examples,
        labels=["successful", "successful", "unsuccessful", "unsuccessful"],
        selected_dataset_indices=[10, 11, 12, 13],
        selected_row_ids=["s0", "s1", "f0", "f1"],
        position_mode="needle-last",
    )

    assert prepared.summary["num_skipped"] == 1
    assert prepared.summary["num_selected_per_group_after_rebalance"] == 1
    assert prepared.selected_row_ids == ["s0", "f0"]
    assert prepared.position_indices == [9 - 1, 5 - 1]


def test_counterfactual_insertion_positions_removes_one_non_last_slot() -> None:
    assert counterfactual_insertion_positions(
        [100, 200, 400], removed_needle_index=0
    ) == [None, 200, 400]

    with pytest.raises(ValueError, match="no None values"):
        counterfactual_insertion_positions([None, 200, 400], removed_needle_index=0)

    with pytest.raises(ValueError, match="valid non-last index"):
        counterfactual_insertion_positions([100, 200, 400], removed_needle_index=2)


def test_prepare_feature_examples_keeps_all_resolvable_examples() -> None:
    rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    examples = [
        TokenizedCountingExample(0, "a", [1] * 10, _segments(), ["N1", "N2"]),
        TokenizedCountingExample(1, "b", [1] * 10, [], []),
        TokenizedCountingExample(2, "c", [1] * 10, _segments(), ["N1"]),
    ]

    prepared = prepare_feature_examples_for_position(
        rows=rows,
        examples=examples,
        selected_dataset_indices=[5, 6, 7],
        selected_row_ids=["a", "b", "c"],
        position_mode="needle-last",
        label="original_successful",
    )

    assert prepared.summary["num_selected"] == 2
    assert prepared.summary["num_skipped"] == 1
    assert prepared.selected_row_ids == ["a", "c"]
    assert prepared.position_indices == [8, 4]


def test_counterfactual_count_direction_uses_independent_success_means(
    tmp_path: Path,
) -> None:
    original_examples = [
        TokenizedCountingExample(0, "o0", [1, 2, 3], [], []),
        TokenizedCountingExample(1, "o1", [1, 2], [], []),
    ]
    counterfactual_examples = [
        TokenizedCountingExample(0, "c0", [1, 2, 3], [], []),
        TokenizedCountingExample(1, "c1", [1], [], []),
        TokenizedCountingExample(2, "c2", [1, 2], [], []),
    ]
    original_hidden = torch.zeros(2, 3, 2)
    counterfactual_hidden = torch.zeros(3, 3, 2)
    original_hidden[0, 2] = torch.tensor([4.0, 0.0])
    original_hidden[1, 1] = torch.tensor([2.0, 2.0])
    counterfactual_hidden[0, 2] = torch.tensor([1.0, 0.0])
    counterfactual_hidden[1, 0] = torch.tensor([1.0, 0.0])
    counterfactual_hidden[2, 1] = torch.tensor([1.0, 2.0])

    result = fit_counterfactual_count_direction(
        original_hidden,
        original_examples,
        counterfactual_hidden,
        counterfactual_examples,
        layer=8,
        position_mode="last",
    )

    expected_raw = torch.tensor([2.0, 1.0 / 3.0])
    expected_norm = float(torch.linalg.vector_norm(expected_raw).item())
    assert result.raw_direction.tolist() == pytest.approx(expected_raw.tolist())
    assert result.raw_norm == pytest.approx(expected_norm)
    assert result.direction.tolist() == pytest.approx((expected_raw / expected_norm).tolist())
    assert result.metrics["num_original_successful"] == 2
    assert result.metrics["num_counterfactual_successful"] == 3

    path = save_counterfactual_count_direction(
        result,
        tmp_path,
        original_dataset_indices=[0, 1],
        original_row_ids=["o0", "o1"],
        original_position_indices=[2, 1],
        counterfactual_dataset_indices=[0, 1, 2],
        counterfactual_row_ids=["c0", "c1", "c2"],
        counterfactual_position_indices=[2, 0, 1],
        original_gold_count=3,
        counterfactual_gold_count=2,
    )
    payload = torch.load(path, map_location="cpu")
    assert payload["method"] == "counterfactual_count_difference"
    assert payload["original_gold_count"] == 3
    assert payload["counterfactual_gold_count"] == 2
    assert payload["counterfactual_row_ids"] == ["c0", "c1", "c2"]


def test_select_contrastive_success_examples_balances_full_scored_pool() -> None:
    rows = [{"id": str(i)} for i in range(6)]
    scored = [
        {"id": str(i), "exact_match": i in {0, 1, 2, 3}}
        for i in range(6)
    ]

    selected = select_contrastive_success_examples(rows, scored, min_group_warning=3)

    assert selected.summary["num_successful_available"] == 4
    assert selected.summary["num_unsuccessful_available"] == 2
    assert selected.summary["num_selected_per_group"] == 2
    assert selected.labels == ["successful", "successful", "unsuccessful", "unsuccessful"]
    assert selected.selected_dataset_indices == [0, 1, 4, 5]
    assert selected.summary["warnings"]


def test_classification_probe_handles_integer_counts() -> None:
    n, t, d = 4, 15, 3
    target = torch.arange(t, dtype=torch.float32).repeat(n, 1) % 3
    centers = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
    hidden = centers[target.long()]

    result = fit_classification_probe(
        hidden,
        target,
        max_train_tokens=None,
        standardize=True,
        epochs=80,
        lr=0.05,
    )
    metrics = evaluate_classification_probe(result, hidden, target, max_tokens=None)

    assert metrics["accuracy"] > 0.95
    assert metrics["macro_f1"] > 0.95


def test_classification_probe_rejects_interpolation_targets() -> None:
    hidden = torch.randn(1, 4, 2)
    target = torch.tensor([[0.0, 0.25, 0.5, 1.0]])

    with pytest.raises(ValueError, match="integer targets"):
        fit_classification_probe(hidden, target, max_train_tokens=None)


def test_save_target_artifacts_and_large_file_warning(tmp_path: Path) -> None:
    examples = [TokenizedCountingExample(0, "row0", [1, 2, 3], [], [])]
    target = torch.zeros(1, 3)
    paths = save_target_artifacts(
        target, examples, tmp_path, target_count_type="left_jump"
    )
    assert paths["target"].exists()
    assert paths["metadata"].exists()

    large = tmp_path / "large.bin"
    large.write_bytes(b"12345")
    with pytest.warns(RuntimeWarning, match="Large counting-feature artifact"):
        assert warn_if_large_file(large, threshold_bytes=4)


def test_filter_successful_rows_and_length_stats_and_split() -> None:
    rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    scored = [{"id": "a", "exact_match": True}, {"id": "b", "exact_match": False}]
    kept, summary = filter_successful_rows(rows, scored)

    assert kept == [{"id": "a"}]
    assert summary["num_successful"] == 1
    assert summary["num_failed"] == 1
    assert summary["num_missing_scores"] == 1
    assert summary["num_rows_used"] == 1
    assert summary["filter_example"] is True
    assert sequence_length_stats([2, 4, 6])["mean"] == 4.0
    split = train_test_split_indices(10, test_fraction=0.2, seed=123)
    assert len(split.train_indices) == 8
    assert len(split.test_indices) == 2


def test_filter_successful_rows_can_keep_all_examples() -> None:
    rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    scored = [{"id": "a", "exact_match": True}, {"id": "b", "exact_match": False}]

    kept, summary = filter_successful_rows(rows, scored, filter_example=False)

    assert kept == rows
    assert summary["filter_example"] is False
    assert summary["num_successful"] == 1
    assert summary["num_failed"] == 1
    assert summary["num_missing_scores"] == 1
    assert summary["num_rows_used"] == 3
    assert summary["included_unsuccessful"] is True
    assert summary["included_missing_scores"] is True


def test_plot_projection_accepts_standardized_probe_and_outliers(
    tmp_path: Path,
) -> None:
    n, t, d = 2, 8, 3
    target = torch.arange(t, dtype=torch.float32).repeat(n, 1) % 3
    hidden = torch.randn(n, t, d) * 0.01
    hidden += target[..., None] * torch.tensor([2.0, -1.0, 0.5])
    hidden[0, 0, :] = 1_000.0

    ridge = fit_ridge_probe(
        hidden, target, alpha=1e-3, max_train_tokens=None, standardize=True
    )
    out = plot_probe_2d_projection(
        ridge,
        hidden,
        target,
        tmp_path / "ridge_2d.png",
        max_points=100,
        pca_norm_median_multiplier=5.0,
    )

    assert out.exists()
    assert out.stat().st_size > 0
    metadata = json.loads((tmp_path / "ridge_2d.png.metadata.json").read_text())
    assert metadata["color_mode"] == "continuous_token_positions"
    assert metadata["marker_mode"] == "discrete_integer_counts"
    assert metadata["unique_target_values"] == [0.0, 1.0, 2.0]
    assert metadata["scatter_alpha"] == 0.85
    assert metadata["colorbar_label"] == "token position"
    assert metadata["legend_title"] == "target count"
    assert metadata["marker_by_label"] == {"0": "o", "1": "s", "2": "^"}
    assert metadata["position_min"] == 0
    assert metadata["position_max"] == 7


def test_plot_projection_uses_classifier_count_direction(tmp_path: Path) -> None:
    n, t, d = 3, 9, 3
    target = torch.arange(t, dtype=torch.float32).repeat(n, 1) % 3
    centers = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
    hidden = centers[target.long()]
    clf = fit_classification_probe(
        hidden,
        target,
        max_train_tokens=None,
        standardize=True,
        epochs=40,
        lr=0.05,
    )

    out = plot_probe_2d_projection(
        clf,
        hidden,
        target,
        tmp_path / "clf_2d.png",
        classifier=clf,
        max_points=100,
    )

    assert out.exists()
    assert out.stat().st_size > 0



def _touch_required_feature_cache_files(root: Path, layer: int = 12) -> None:
    tensors = root / "tensors"
    tables = root / "tables"
    figures = root / "figures"
    tensors.mkdir(parents=True)
    tables.mkdir(parents=True)
    figures.mkdir(parents=True)
    torch.save(torch.zeros(1, 2), tensors / "target_count_y_t.pt")
    torch.save(torch.zeros(1, 2, dtype=torch.bool), tensors / "matching_needle_token_mask.pt")
    torch.save(torch.zeros(1, 2, 3), tensors / f"hidden_layer_{layer}.pt")
    torch.save({"coef": torch.ones(3)}, tensors / f"ridge_probe_layer_{layer}.pt")
    (tensors / "target_count_metadata.json").write_text("{}", encoding="utf-8")
    (tensors / "hidden_state_metadata.json").write_text("{}", encoding="utf-8")
    (tensors / f"ridge_probe_layer_{layer}_metrics.json").write_text("{}", encoding="utf-8")
    (tables / "split_and_filter_summary.json").write_text("{}", encoding="utf-8")
    (tables / "probe_summary.csv").write_text("layer,probe\n12,ridge\n", encoding="utf-8")
    (tables / f"ridge_layer_{layer}_eval.json").write_text("{}", encoding="utf-8")
    for name in [
        f"ridge_layer_{layer}_train_line.png",
        f"ridge_layer_{layer}_test_line.png",
        f"ridge_layer_{layer}_train_2d.png",
        f"ridge_layer_{layer}_test_2d.png",
    ]:
        (figures / name).write_bytes(b"png")


def test_counting_feature_cache_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _touch_required_feature_cache_files(source)
    cfg = build_counting_feature_run_config(
        {
            "COUNTING_FEATURE_CALC_METHOD": "ridge",
            "FEATURE_CALC_POS": "last",
            "LAYERS": [12],
            "RUN_CLASSIFICATION": False,
        }
    )
    cache_config = build_counting_feature_cache_config(cfg, setting_name="setting-a")
    cache = tmp_path / "cache"

    saved = save_counting_feature_cache(
        cache,
        feature_tensors_dir=source / "tensors",
        feature_tables_dir=source / "tables",
        feature_figures_dir=source / "figures",
        cache_config=cache_config,
    )
    validated = validate_counting_feature_cache(
        cache,
        cache_config,
        layers=[12],
        run_classification=False,
    )
    restored = tmp_path / "restored"
    restore_counting_feature_cache(
        cache,
        feature_tensors_dir=restored / "tensors",
        feature_tables_dir=restored / "tables",
        feature_figures_dir=restored / "figures",
    )

    assert Path(saved["metadata_path"]).exists()
    assert validated["num_required_files"] > 0
    assert (restored / "tensors" / "ridge_probe_layer_12.pt").exists()
    with pytest.raises(ValueError, match="settings do not match"):
        validate_counting_feature_cache(
            cache,
            {**cache_config, "TARGET_COUNT_TYPE": "other"},
            layers=[12],
            run_classification=False,
        )
