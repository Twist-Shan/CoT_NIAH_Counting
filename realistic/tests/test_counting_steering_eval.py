import csv

import pytest

from dataset_generation.dynamic_niah_v2 import DynamicNiahV2Config
from counting.steering_eval import (
    TUPLE_MEANING,
    build_steering_eval_dataset_config,
    steering_eval_examples,
    steering_eval_insertion_positions,
    summarize_steering_test_eval_details,
    write_steering_test_eval_summary_csv,
)


def test_steering_eval_insertion_positions_extends_to_requested_k() -> None:
    positions = steering_eval_insertion_positions(
        num_needles=5,
        base_positions=[100, 200, 400],
        target_haystack_tokens=1000,
    )

    assert len(positions) == 5
    assert all(pos is not None for pos in positions)
    assert positions[:3] != ()


def test_build_steering_eval_dataset_config_randomizes_and_sets_count(tmp_path) -> None:
    base = DynamicNiahV2Config(
        task_type="match_count",
        num_examples=500,
        num_needles=3,
        insertion_positions=(100, 200, 400),
        randomize_needle_insertion=False,
        output_dir="old",
        data_save_path="old/data.jsonl",
    )

    cfg = build_steering_eval_dataset_config(
        base,
        gold_count=4,
        num_examples=10,
        randomize_needle_min_separation=20,
        output_dir=tmp_path / "dataset_4",
    )

    assert cfg.num_needles == 4
    assert cfg.num_examples == 10
    assert cfg.randomize_needle_insertion is True
    assert cfg.randomize_needle_min_separation == 20
    assert len(cfg.insertion_positions) == 4
    assert cfg.data_save_path.endswith("dynamic_niah_v2.jsonl")


def test_steering_eval_examples_assign_global_indices_and_dataset_groups() -> None:
    datasets = {
        1: [{"id": "a", "messages": [], "gold_answer": {"count": 1}}],
        2: [
            {"id": "b", "messages": [], "gold_answer": {"count": 2}},
            {"id": "c", "messages": [], "gold_answer": {"count": 2}},
        ],
    }

    examples, lookup, summary = steering_eval_examples(datasets)

    assert [ex.dataset_index for ex in examples] == [0, 1, 2]
    assert [ex.group for ex in examples] == ["dataset_1", "dataset_2", "dataset_2"]
    assert examples[1].row_id == "dataset_2:b"
    assert lookup[2]["dataset_k"] == 2
    assert summary["examples_by_dataset"] == {"1": 1, "2": 2}


def test_summarize_steering_test_eval_details_uses_four_tuple_metrics() -> None:
    rows = [
        {
            "layer": 20,
            "beta": 1.0,
            "intervention_target": "needle_1",
            "eval_dataset_k": 1,
            "baseline_exact_match": False,
            "steered_exact_match": True,
            "baseline_prediction_count": 0,
            "steered_prediction_count": 1,
        },
        {
            "layer": 20,
            "beta": 1.0,
            "intervention_target": "needle_1",
            "eval_dataset_k": 1,
            "baseline_exact_match": True,
            "steered_exact_match": True,
            "baseline_prediction_count": 1,
            "steered_prediction_count": 1,
        },
        {
            "layer": 20,
            "beta": 1.0,
            "intervention_target": "needle_1",
            "eval_dataset_k": 2,
            "baseline_exact_match": False,
            "steered_exact_match": False,
            "baseline_prediction_count": 1,
            "steered_prediction_count": 3,
        },
    ]

    summary = summarize_steering_test_eval_details(rows, dataset_ks=[1, 2])

    assert summary == [
        {
            "layer": 20,
            "intervention_target": "needle_1",
            "steering_coeff": 1.0,
            "Dataset 1 Before/After steering": "(0.5, 1, 0.5, 1)",
            "Dataset 2 Before/After steering": "(0, 0, 1, 3)",
        }
    ]


def test_summarize_steering_test_eval_details_leaves_missing_dataset_blank() -> None:
    rows = [
        {
            "layer": 20,
            "beta": 1.0,
            "intervention_target": "needle_2",
            "eval_dataset_k": 2,
            "baseline_exact_match": True,
            "steered_exact_match": True,
            "baseline_prediction_count": 2,
            "steered_prediction_count": 2,
        },
    ]

    summary = summarize_steering_test_eval_details(rows, dataset_ks=[1, 2])

    assert summary[0]["Dataset 1 Before/After steering"] == ""
    assert summary[0]["Dataset 2 Before/After steering"] == "(1, 1, 2, 2)"


def test_write_steering_test_eval_summary_csv_adds_tuple_explanation(tmp_path) -> None:
    path = tmp_path / "summary.csv"
    rows = [
        {
            "layer": 20,
            "intervention_target": "last_token",
            "steering_coeff": 0.5,
            "Dataset 1 Before/After steering": "(1, 1, 1, 1)",
        }
    ]

    write_steering_test_eval_summary_csv(rows, path)

    with path.open(newline="", encoding="utf-8") as handle:
        reader = list(csv.DictReader(handle))

    assert reader[0]["layer"] == "# tuple_meaning"
    assert TUPLE_MEANING in reader[0]["intervention_target"]
    assert reader[1]["intervention_target"] == "last_token"


def test_build_steering_eval_dataset_config_rejects_nonpositive_count(tmp_path) -> None:
    with pytest.raises(ValueError, match="num_needles"):
        build_steering_eval_dataset_config(
            DynamicNiahV2Config(),
            gold_count=0,
            num_examples=10,
            output_dir=tmp_path,
        )
