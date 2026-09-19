from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

AVG_TOLERANCE = 1e-2


def canonical_task_type(task_type: str) -> str:
    normalized = task_type.strip().lower().replace("-", "_")
    if normalized in {"argmax", "max", "highest_score"}:
        return "argmax"
    if normalized in {
        "count_avg",
        "count_average",
        "count_average_score",
        "avg",
        "average",
    }:
        return "count_avg"
    if normalized in {"match_count", "matched_count", "city_count", "count_match"}:
        return "match_count"
    if normalized in {"literal_count", "canary_count", "exact_count"}:
        return "literal_count"
    raise ValueError(f"Unsupported task_type: {task_type}")


def build_task_query(
    task_type: str,
    *,
    literal: str | None = None,
    counting_needle_kind: str = "city_score",
    marker_text: str = "[dolphin]",
) -> str:
    task_type = canonical_task_type(task_type)
    if task_type == "argmax":
        return (
            "One or more cities and their ratings are provided in a survey or audit. "
            "You need to answer the following question: "
            "Which city has the highest score? Respond as JSON with keys city and score."
        )
    if task_type == "count_avg":
        return (
            "One or more cities and their ratings are provided in a survey or audit. "
            "You need to answer the following question: "
            "How many cities are rated, and what is their average score? Respond as "
            "JSON with keys count and average_score."
        )
    if task_type == "match_count":
        if counting_needle_kind.strip().lower().replace("-", "_") == "marker":
            return (
                f'How many times does the exact marker "{marker_text}" appear in '
                f'the context? Count only exact copies of "{marker_text}". '
                "Respond as JSON with key count."
            )
        return (
            "One or more city score audit records are provided in the context. "
            "You need to answer the following question: "
            "How many cities received a score? Respond as JSON with key count."
        )
    if literal is None:
        raise ValueError("literal_count requires a literal query string")
    return (
        "One or more exact literal strings are provided in the context. "
        "You need to answer the following question: "
        f'How many exact copies of "{literal}" appear in the context? Respond as '
        "JSON with key count."
    )


def build_gold_answer(records: list[dict[str, Any]], task_type: str) -> dict[str, Any]:
    task_type = canonical_task_type(task_type)
    if task_type == "argmax":
        winner = max(records, key=lambda r: (r["score"], r["city"]))
        return {"city": winner["city"], "score": winner["score"]}

    if task_type in {"match_count", "literal_count"}:
        return {"count": len(records)}

    count = len(records)
    if count == 0:
        return {"count": 0, "average_score": 0.0}
    average_score = sum(float(r["score"]) for r in records) / count
    return {"count": count, "average_score": average_score}


def build_missing_control_answer(task_type: str) -> dict[str, Any]:
    task_type = canonical_task_type(task_type)
    if task_type == "argmax":
        return {"city": None, "score": None, "has_answer": False}
    if task_type in {"match_count", "literal_count"}:
        return {"count": None, "has_answer": False}
    return {"count": None, "average_score": None, "has_answer": False}


def build_control_gold_answer(
    records: list[dict[str, Any]], task_type: str
) -> dict[str, Any]:
    if not records:
        return build_missing_control_answer(task_type)
    return build_gold_answer(records, task_type) | {"has_answer": True}


def expected_answer_for_row(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("gold_answer", {})


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    parsed_objects: list[dict[str, Any]] = []
    for match in re.finditer(r"\{[^{}]*\}", text, flags=re.S):
        try:
            obj = json.loads(match.group(0))
        except Exception:
            continue
        if isinstance(obj, dict):
            parsed_objects.append(obj)
    if not parsed_objects:
        return None
    return parsed_objects[-1]


def _last_regex_group(pattern: str, text: str, *, flags: int = 0) -> str | None:
    matches = list(re.finditer(pattern, text, flags=flags))
    if not matches:
        return None
    return matches[-1].group(1)


def _maybe_int(value: Any) -> int | None:
    try:
        if isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> float | None:
    try:
        if isinstance(value, bool):
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def parse_model_output(text: str, task_type: str) -> dict[str, Any]:
    task_type = canonical_task_type(task_type)
    obj = _extract_json_object(text) or {}

    if task_type == "argmax":
        city = obj.get("city", obj.get("entity"))
        score = obj.get("score")
        parse_mode = "json" if obj else "parse_fail"
        if city is None or score is None:
            city_value = _last_regex_group(
                r"(?:city|entity)\s*[:=]\s*[\"']?([^\"',\n}]+)", text, flags=re.I
            )
            score_value = _last_regex_group(
                r"score\s*[:=]\s*(-?\d+)", text, flags=re.I
            )
            if city_value is not None and score_value is not None:
                city = city_value.strip()
                score = score_value
                parse_mode = "regex"
        parsed_score = _maybe_int(score)
        if city is None or parsed_score is None:
            return {"city": None, "score": None, "parse_mode": "parse_fail"}
        return {
            "city": str(city).strip(),
            "score": parsed_score,
            "parse_mode": parse_mode,
        }

    if task_type in {"match_count", "literal_count"}:
        count = obj.get(
            "count",
            obj.get(
                "num_cities",
                obj.get("number_of_cities", obj.get("number_of_copies")),
            ),
        )
        parse_mode = "json" if obj else "parse_fail"
        if count is None:
            count_value = _last_regex_group(
                r"(?:count|num_cities|number_of_cities|number_of_copies|copies)\s*[:=]\s*(\d+)",
                text,
                flags=re.I,
            )
            if count_value is not None:
                count = count_value
                parse_mode = "regex"
        parsed_count = _maybe_int(count)
        if parsed_count is None:
            return {"count": None, "parse_mode": "parse_fail"}
        return {"count": parsed_count, "parse_mode": parse_mode}

    count = obj.get("count", obj.get("num_cities", obj.get("number_of_cities")))
    average_score = obj.get("average_score", obj.get("avg_score", obj.get("average")))
    parse_mode = "json" if obj else "parse_fail"
    if count is None or average_score is None:
        count_value = _last_regex_group(
            r"(?:count|num_cities|number_of_cities)\s*[:=]\s*(\d+)", text, flags=re.I
        )
        average_value = _last_regex_group(
            r"(?:average_score|avg_score|average)\s*[:=]\s*(-?\d+(?:\.\d+)?)",
            text,
            flags=re.I,
        )
        if count_value is not None and average_value is not None:
            count = count_value
            average_score = average_value
            parse_mode = "regex"
    parsed_count = _maybe_int(count)
    parsed_average = _maybe_float(average_score)
    if parsed_count is None or parsed_average is None:
        return {"count": None, "average_score": None, "parse_mode": "parse_fail"}
    return {
        "count": parsed_count,
        "average_score": parsed_average,
        "parse_mode": parse_mode,
    }


def score_prediction(
    prediction: dict[str, Any], gold_answer: dict[str, Any], task_type: str
) -> bool:
    task_type = canonical_task_type(task_type)
    if task_type == "argmax":
        return prediction.get("city") == gold_answer.get("city") and prediction.get(
            "score"
        ) == gold_answer.get("score")

    if task_type in {"match_count", "literal_count"}:
        return prediction.get("count") == gold_answer.get("count")

    pred_avg = _maybe_float(prediction.get("average_score"))
    gold_avg = _maybe_float(gold_answer.get("average_score"))
    return (
        prediction.get("count") == gold_answer.get("count")
        and pred_avg is not None
        and gold_avg is not None
        and abs(pred_avg - gold_avg) <= AVG_TOLERANCE
    )


def build_response_result(row: dict[str, Any], model_output: str) -> dict[str, Any]:
    dataset_gold_answer = row.get("gold_answer", {})
    control_gold_answer = row.get("control_gold_answer")
    expected_answer = expected_answer_for_row(row)
    prediction = parse_model_output(model_output, row["task_type"])
    exact_match = score_prediction(prediction, expected_answer, row["task_type"])
    return {
        "id": row.get("id"),
        "task_type": row.get("task_type"),
        "query": row.get("query"),
        "gold_answer": dataset_gold_answer,
        "control_gold_answer": control_gold_answer,
        "expected_answer": expected_answer,
        "prediction": {k: v for k, v in prediction.items() if k != "parse_mode"},
        "parse_mode": prediction.get("parse_mode"),
        "model_output_text": model_output,
        "exact_match": exact_match,
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    exact = sum(1 for row in results if row.get("exact_match"))
    parse_failures = sum(1 for row in results if row.get("parse_mode") == "parse_fail")
    total = len(results)
    return {
        "num_examples": total,
        "exact_match_count": exact,
        "exact_match_accuracy": exact / total if total else 0.0,
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / total if total else 0.0,
    }


def write_jsonl(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
