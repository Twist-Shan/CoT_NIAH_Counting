from __future__ import annotations

import json
import re

ARGMAX_RESPONSE_SCHEMA = '{"city":"","score":0}'
COUNT_AVG_RESPONSE_SCHEMA = '{"count":0,"average_score":0.0}'
COUNT_RESPONSE_SCHEMA = '{"count":0}'


def response_schema_for_task(task_type: str) -> str:
    if task_type == "argmax":
        return ARGMAX_RESPONSE_SCHEMA
    if task_type == "count_avg":
        return COUNT_AVG_RESPONSE_SCHEMA
    if task_type in {"match_count", "literal_count"}:
        return COUNT_RESPONSE_SCHEMA
    raise ValueError(f"Unsupported task_type: {task_type}")


def _memorization_instruction(
    task_type: str | None,
    *,
    counting_needle_kind: str = "city_score",
    marker_text: str = "[dolphin]",
    literal_text: str | None = None,
) -> str:
    if task_type is None:
        return ""
    normalized = task_type.strip().lower().replace("-", "_")
    if normalized in {"match_count", "matched_count", "city_count", "count_match"}:
        if counting_needle_kind.strip().lower().replace("-", "_") == "marker":
            return (
                f'The exact marker "{marker_text}" is inserted one or more times '
                "within the following text. Make sure to memorize them."
            )
    if normalized in {"literal_count", "canary_count", "exact_count"}:
        if literal_text is not None:
            return (
                f'The exact literal "{literal_text}" is inserted one or more times '
                "within the following text. Make sure to memorize it."
            )
        return (
            "Some exact literal strings are inserted within the following text. "
            "Make sure to memorize them."
        )
    if normalized in {
        "argmax",
        "max",
        "highest_score",
        "count_avg",
        "count_average",
        "count_average_score",
        "avg",
        "average",
        "match_count",
        "matched_count",
        "city_count",
        "count_match",
    }:
        return (
            "Some information about cities are inserted within the following text. "
            "Make sure to memorize them."
        )
    return ""


def _instruction(
    thinking_mode: bool,
    response_schema: str,
    *,
    task_type: str | None = None,
    counting_needle_kind: str = "city_score",
    marker_text: str = "[dolphin]",
    literal_text: str | None = None,
    include_reasoning_ban: bool = True,
    include_memorization_instruction: bool = True,
) -> str:
    instruction = (
        "Return ONLY one JSON object on a single line with schema "
        f"{response_schema}. "
        "No extra text."
    )
    if include_reasoning_ban and not thinking_mode:
        instruction += " Do NOT explain or include reasoning."
    memorization_instruction = (
        _memorization_instruction(
            task_type,
            counting_needle_kind=counting_needle_kind,
            marker_text=marker_text,
            literal_text=literal_text,
        )
        if include_memorization_instruction
        else ""
    )
    if include_memorization_instruction and memorization_instruction:
        instruction += " " + memorization_instruction
    return instruction


def build_messages_easier(
    context: str,
    query: str,
    thinking_mode: bool = False,
    response_schema: str = ARGMAX_RESPONSE_SCHEMA,
    task_type: str | None = None,
    counting_needle_kind: str = "city_score",
    marker_text: str = "[dolphin]",
    literal_text: str | None = None,
):
    instruction = _instruction(
        thinking_mode,
        response_schema,
        task_type=task_type,
        counting_needle_kind=counting_needle_kind,
        marker_text=marker_text,
        literal_text=literal_text,
    )
    user_content = f"{instruction}\n\nQuery:\n{query}\n\nContext:\n{context}"
    return [{"role": "user", "content": user_content}]


def build_messages_vanilla(
    context: str,
    query: str,
    thinking_mode: bool = False,
    response_schema: str = ARGMAX_RESPONSE_SCHEMA,
    task_type: str | None = None,
    counting_needle_kind: str = "city_score",
    marker_text: str = "[dolphin]",
    literal_text: str | None = None,
    include_reasoning_ban: bool = True,
    include_memorization_instruction: bool = True,
):
    instruction = _instruction(
        thinking_mode,
        response_schema,
        task_type=task_type,
        counting_needle_kind=counting_needle_kind,
        marker_text=marker_text,
        literal_text=literal_text,
        include_reasoning_ban=include_reasoning_ban,
        include_memorization_instruction=include_memorization_instruction,
    )
    user_content = f"{instruction}\n\nContext:\n{context}\n\nQuery:\n{query}"
    return [{"role": "user", "content": user_content}]


def parse_prediction(text: str):
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "city" in obj and "score" in obj:
            return str(obj["city"]).strip(), int(obj["score"]), "json"
        if isinstance(obj, dict) and "entity" in obj and "score" in obj:
            return str(obj["entity"]).strip(), int(obj["score"]), "json"
    except Exception:
        pass

    m = re.search(r"\{[^{}]*\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if "city" in obj and "score" in obj:
                return str(obj["city"]).strip(), int(obj["score"]), "json_substr"
            if "entity" in obj and "score" in obj:
                return str(obj["entity"]).strip(), int(obj["score"]), "json_substr"
        except Exception:
            pass

    em = re.search(r"city\s*[:=]\s*[\"\']?([^\"\',\n}]+)", text, flags=re.I)
    sm = re.search(r"score\s*[:=]\s*(-?\d+)", text, flags=re.I)
    if em and sm:
        return em.group(1).strip(), int(sm.group(1)), "regex"

    return None, None, "parse_fail"
