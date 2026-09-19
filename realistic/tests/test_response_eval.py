from dataset_generation.response_eval import (
    build_control_gold_answer,
    build_gold_answer,
    build_task_query,
    expected_answer_for_row,
    parse_model_output,
    score_prediction,
    summarize_results,
)
from dataset_generation.niah_prompt_utils import (
    build_messages_easier,
    build_messages_vanilla,
)


def test_argmax_parse_and_score() -> None:
    pred = parse_model_output('prefix {"city": "Paris", "score": 91} suffix', "argmax")
    assert pred == {"city": "Paris", "score": 91, "parse_mode": "json"}
    assert score_prediction(pred, {"city": "Paris", "score": 91}, "argmax")
    assert not score_prediction(pred, {"city": "Lyon", "score": 91}, "argmax")


def test_count_avg_gold_parse_and_tolerant_score() -> None:
    records = [
        {"city": "A", "score": 80},
        {"city": "B", "score": 91},
        {"city": "C", "score": 99},
    ]
    gold = build_gold_answer(records, "count_avg")
    pred = parse_model_output('{"count": 3, "average_score": 90.0}', "count_avg")

    assert gold == {"count": 3, "average_score": 90.0}
    assert score_prediction(pred, gold, "count_avg")
    assert not score_prediction({"count": 2, "average_score": 90.0}, gold, "count_avg")


def test_parse_uses_final_json_object_for_thinking_outputs() -> None:
    pred = parse_model_output(
        'Draft: {"count": 2}\nCorrection: {"count": 4}',
        "match_count",
    )
    assert pred == {"count": 4, "parse_mode": "json"}


def test_count_avg_parse_uses_final_json_object() -> None:
    pred = parse_model_output(
        'First try: {"count": 2, "average_score": 80.0}\n'
        'Final: {"count": 3, "average_score": 90.0}',
        "count_avg",
    )
    assert pred == {"count": 3, "average_score": 90.0, "parse_mode": "json"}


def test_argmax_parse_uses_final_json_object() -> None:
    pred = parse_model_output(
        'Maybe {"city": "Paris", "score": 91}. Final {"city": "Lyon", "score": 95}.',
        "argmax",
    )
    assert pred == {"city": "Lyon", "score": 95, "parse_mode": "json"}


def test_regex_parse_uses_final_answer_when_no_json_is_available() -> None:
    pred = parse_model_output("draft count: 2\nfinal count: 5", "match_count")
    assert pred == {"count": 5, "parse_mode": "regex"}


def test_control_gold_is_separate_from_expected_answer() -> None:
    no_answer = build_control_gold_answer([], "count_avg")
    assert no_answer == {"count": None, "average_score": None, "has_answer": False}

    row = {
        "gold_answer": {"count": 3, "average_score": 80.0},
        "control_gold_answer": {"count": 2, "average_score": 75.0, "has_answer": True},
        "controls": {"control_switch": [True, False, False]},
    }
    assert expected_answer_for_row(row) == {"count": 3, "average_score": 80.0}


def test_summarize_results() -> None:
    metrics = summarize_results(
        [
            {"exact_match": True, "parse_mode": "json"},
            {"exact_match": False, "parse_mode": "parse_fail"},
        ]
    )
    assert metrics["exact_match_accuracy"] == 0.5
    assert metrics["parse_failure_rate"] == 0.5


def test_count_only_tasks_gold_parse_and_score() -> None:
    records = [{"needle_id": "N1"}, {"needle_id": "N2"}, {"needle_id": "N3"}]

    for task_type in ("match_count", "literal_count"):
        gold = build_gold_answer(records, task_type)
        pred = parse_model_output('The answer is {"count": 3}.', task_type)

        assert gold == {"count": 3}
        assert pred == {"count": 3, "parse_mode": "json"}
        assert score_prediction(pred, gold, task_type)
        assert not score_prediction({"count": 2}, gold, task_type)
        assert build_control_gold_answer([], task_type) == {
            "count": None,
            "has_answer": False,
        }


def test_build_task_query_uses_task_specific_question() -> None:
    assert "Which city has the highest score?" in build_task_query("argmax")
    assert "average score" in build_task_query("count_avg")
    assert "How many cities received a score?" in build_task_query("match_count")
    assert "city score audit records" in build_task_query("match_count")
    assert "Make sure to memorize" not in build_task_query("match_count")
    literal_query = build_task_query("literal_count", literal="ABC123")
    assert 'How many exact copies of "ABC123"' in literal_query
    assert "How many cities receive a score?" not in literal_query
    assert "Make sure to memorize" not in literal_query


def test_prompt_instruction_contains_task_aware_memorization_once() -> None:
    city = build_messages_easier(
        "haystack context",
        "task query",
        thinking_mode=False,
        response_schema='{"count":0}',
        task_type="match_count",
    )[0]["content"]

    assert city.count("Some information about cities are inserted") == 1
    assert "Do NOT explain or include reasoning." in city
    assert city.index("Some information about cities are inserted") < city.index(
        "Context:"
    )
    assert city.index("Query:") < city.index("Context:")
    query_block = city.split("Query:\n", 1)[1].split("\n\nContext:", 1)[0]
    assert "Make sure to memorize" not in query_block

    literal = build_messages_vanilla(
        "literal context",
        "literal query",
        thinking_mode=False,
        response_schema='{"count":0}',
        task_type="literal_count",
        literal_text="ABC123",
    )[0]["content"]

    assert literal.count('The exact literal "ABC123" is inserted') == 1
    assert "Some exact literal strings are inserted" not in literal
    assert "Some information about cities are inserted" not in literal
    assert literal.index('The exact literal "ABC123" is inserted') < literal.index(
        "Context:"
    )
    assert literal.index("Context:") < literal.index("Query:")
    literal_query_block = literal.split("Query:\n", 1)[1]
    assert "Make sure to memorize" not in literal_query_block
