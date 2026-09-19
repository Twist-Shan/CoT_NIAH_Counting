from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from typing import Any

from .spec import MODEL_REVISIONS, MODEL_SPECS, QUERY_LAYOUT

PROMPT_REVISION_PLAN_SCHEMA = "realistic_niah_prompt_revision_shard_plan_v2_1"
PROTOCOL_VERSION = "realistic_niah_v2_1_prompt_revision"
EXPECTED_STIMULI_PER_SHARD = 500
EXPECTED_PROMPT_REVISION_REQUESTS = 7_500

ENUMERATION_MODEL_LABELS = (
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-32B",
    "Gemma4-E4B",
    "Gemma4-12B",
    "GLM-4-9B-0414",
)
ENUMERATION_MODES = ("enumeration_index", "enumeration_bullet")
APPENDIX_MODEL_LABEL = "Gemma4-12B"
APPENDIX_MODE = "direct"

_PRIORITIES: dict[tuple[str, str], int] = {
    ("Qwen3-32B", "enumeration_index"): 100,
    ("Qwen3-32B", "enumeration_bullet"): 99,
    ("Gemma4-12B", "enumeration_index"): 90,
    ("Gemma4-12B", "enumeration_bullet"): 89,
    ("Gemma4-12B", "direct"): 88,
    ("GLM-4-9B-0414", "enumeration_index"): 80,
    ("GLM-4-9B-0414", "enumeration_bullet"): 79,
    ("Qwen3-8B", "enumeration_index"): 70,
    ("Qwen3-8B", "enumeration_bullet"): 69,
    ("Gemma4-E4B", "enumeration_index"): 60,
    ("Gemma4-E4B", "enumeration_bullet"): 59,
    ("Qwen3-4B", "enumeration_index"): 50,
    ("Qwen3-4B", "enumeration_bullet"): 49,
    ("Qwen3-1.7B", "enumeration_index"): 40,
    ("Qwen3-1.7B", "enumeration_bullet"): 39,
}


def _task_id(model_label: str, prompt_mode: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_label).strip("-")
    return f"{normalized}__{prompt_mode}"


def _task(
    model_label: str,
    prompt_mode: str,
    *,
    analysis_role: str,
    output_collection: str,
) -> dict[str, Any]:
    spec = MODEL_SPECS[model_label]
    key = (model_label, prompt_mode)
    return {
        "task_id": _task_id(model_label, prompt_mode),
        "priority": _PRIORITIES[key],
        "model_label": model_label,
        "model_id": spec.model_id,
        "model_revision": MODEL_REVISIONS[model_label],
        "prompt_mode": prompt_mode,
        "reasoning_policy": spec.reasoning_policy,
        "analysis_role": analysis_role,
        "output_collection": output_collection,
        "expected_requests": EXPECTED_STIMULI_PER_SHARD,
    }


def prompt_revision_shard_tasks() -> list[dict[str, Any]]:
    """Return the frozen 15-shard V2.1 prompt-revision execution plan."""

    tasks: list[dict[str, Any]] = []
    for model_label in ENUMERATION_MODEL_LABELS:
        collection = (
            "matched_controls"
            if model_label == "GLM-4-9B-0414"
            else "models"
        )
        for prompt_mode in ENUMERATION_MODES:
            tasks.append(
                _task(
                    model_label,
                    prompt_mode,
                    analysis_role="replacement_enumeration",
                    output_collection=collection,
                )
            )
    tasks.append(
        _task(
            APPENDIX_MODEL_LABEL,
            APPENDIX_MODE,
            analysis_role="appendix_strict_direct",
            output_collection="appendix",
        )
    )
    tasks.sort(key=lambda task: (-int(task["priority"]), str(task["task_id"])))

    task_ids = [str(task["task_id"]) for task in tasks]
    if len(tasks) != 15 or len(task_ids) != len(set(task_ids)):
        raise RuntimeError("The V2.1 prompt-revision plan must have 15 shards")
    if sum(int(task["expected_requests"]) for task in tasks) != (
        EXPECTED_PROMPT_REVISION_REQUESTS
    ):
        raise RuntimeError(
            "The V2.1 prompt-revision plan must contain 7,500 requests"
        )
    return tasks


def prompt_revision_shard_plan() -> dict[str, Any]:
    tasks = prompt_revision_shard_tasks()
    canonical = json.dumps(
        tasks,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": PROMPT_REVISION_PLAN_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "query_layout": QUERY_LAYOUT,
        "expected_stimuli_per_shard": EXPECTED_STIMULI_PER_SHARD,
        "expected_shards": len(tasks),
        "expected_requests": EXPECTED_PROMPT_REVISION_REQUESTS,
        "tasks_sha256": hashlib.sha256(canonical).hexdigest(),
        "tasks": tasks,
    }


def expected_request_ids(
    stimulus_ids: Iterable[str],
    task: dict[str, Any],
) -> tuple[str, ...]:
    model_label = str(task["model_label"])
    prompt_mode = str(task["prompt_mode"])
    return tuple(
        f"{model_label}/{prompt_mode}/{QUERY_LAYOUT}/{stimulus_id}"
        for stimulus_id in stimulus_ids
    )
