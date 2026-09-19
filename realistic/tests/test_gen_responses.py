from dataset_generation.response_eval import build_response_result


def test_count_avg_response_result_scores_against_dataset_gold_answer() -> None:
    row = {
        "id": "row-count",
        "task_type": "count_avg",
        "query": "How many cities are rated, and what is their average score?",
        "gold_answer": {"count": 3, "average_score": 80.0},
        "control_gold_answer": {
            "count": 2,
            "average_score": 82.5,
            "has_answer": True,
        },
        "controls": {"control_switch": [True, False, False]},
    }

    control_only_result = build_response_result(
        row, '{"count": 2, "average_score": 82.5}'
    )
    full_dataset_result = build_response_result(
        row, '{"count": 3, "average_score": 80.0}'
    )

    assert control_only_result["gold_answer"] == {"count": 3, "average_score": 80.0}
    assert control_only_result["control_gold_answer"] == {
        "count": 2,
        "average_score": 82.5,
        "has_answer": True,
    }
    assert control_only_result["expected_answer"] == {
        "count": 3,
        "average_score": 80.0,
    }
    assert control_only_result["exact_match"] is False
    assert full_dataset_result["exact_match"] is True


def test_argmax_response_result_scores_against_dataset_gold_answer() -> None:
    row = {
        "id": "row-argmax",
        "task_type": "argmax",
        "query": "Which city has the highest score?",
        "gold_answer": {"city": "Phnom Penh", "score": 92},
        "control_gold_answer": {
            "city": "Islamabad",
            "score": 81,
            "has_answer": True,
        },
        "controls": {"control_switch": [False, True, False]},
    }

    control_only_result = build_response_result(
        row, '{"city": "Islamabad", "score": 81}'
    )
    full_dataset_result = build_response_result(
        row, '{"city": "Phnom Penh", "score": 92}'
    )

    assert control_only_result["gold_answer"] == {"city": "Phnom Penh", "score": 92}
    assert control_only_result["control_gold_answer"] == {
        "city": "Islamabad",
        "score": 81,
        "has_answer": True,
    }
    assert control_only_result["expected_answer"] == {
        "city": "Phnom Penh",
        "score": 92,
    }
    assert control_only_result["exact_match"] is False
    assert full_dataset_result["exact_match"] is True
