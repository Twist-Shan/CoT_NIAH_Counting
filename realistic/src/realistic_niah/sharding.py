from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from typing import Any

from .spec import (
    MATCHED_CONTROL_MODEL_LABELS,
    MODEL_REVISIONS,
    MODEL_SPECS,
    PRIMARY_MODEL_LABELS,
    QUERY_LAYOUT,
)

FORMAL_SHARD_PLAN_SCHEMA = "realistic_niah_formal_shard_plan_v2"
EXPECTED_STIMULI_PER_SHARD = 500
EXPECTED_FORMAL_REQUESTS = 14_500

# Scheduling weights are operational heuristics only. They do not affect the
# scientific configuration, request IDs, prompts, or decoding parameters.
_SCHEDULING_WEIGHTS: dict[tuple[str, str], int] = {
    ("Qwen3-32B", "native_thinking"): 100,
    ("Gemma4-12B", "native_thinking"): 95,
    ("Qwen3-32B", "enumeration_index"): 90,
    ("Qwen3-32B", "enumeration_bullet"): 90,
    ("DeepSeek-R1-0528-Qwen3-8B", "native_thinking"): 85,
    ("GLM-Z1-9B-0414", "native_thinking"): 80,
    ("Qwen3-8B", "native_thinking"): 75,
    ("Qwen3-32B", "direct"): 70,
    ("Gemma4-E4B", "native_thinking"): 65,
    ("Gemma4-12B", "enumeration_index"): 60,
    ("Gemma4-12B", "enumeration_bullet"): 60,
    ("Qwen3-4B", "native_thinking"): 55,
    ("GLM-4-9B-0414", "enumeration_index"): 52,
    ("GLM-4-9B-0414", "enumeration_bullet"): 52,
    ("Qwen3-8B", "enumeration_index"): 50,
    ("Qwen3-8B", "enumeration_bullet"): 50,
    ("Qwen3-1.7B", "native_thinking"): 45,
    ("Gemma4-E4B", "enumeration_index"): 42,
    ("Gemma4-E4B", "enumeration_bullet"): 42,
    ("Gemma4-12B", "direct"): 40,
    ("Qwen3-4B", "enumeration_index"): 35,
    ("Qwen3-4B", "enumeration_bullet"): 35,
    ("GLM-4-9B-0414", "direct"): 32,
    ("Qwen3-8B", "direct"): 30,
    ("Qwen3-1.7B", "enumeration_index"): 25,
    ("Qwen3-1.7B", "enumeration_bullet"): 25,
    ("Gemma4-E4B", "direct"): 22,
    ("Qwen3-4B", "direct"): 18,
    ("Qwen3-1.7B", "direct"): 15,
}


def _task_id(model_label: str, prompt_mode: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_label).strip("-")
    return f"{normalized}__{prompt_mode}"


def formal_shard_tasks() -> list[dict[str, Any]]:
    """Return the immutable 29-shard formal execution plan."""

    tasks: list[dict[str, Any]] = []
    labels = (*PRIMARY_MODEL_LABELS, *MATCHED_CONTROL_MODEL_LABELS)
    for model_label in labels:
        spec = MODEL_SPECS[model_label]
        collection = (
            "matched_controls"
            if model_label in MATCHED_CONTROL_MODEL_LABELS
            else "models"
        )
        for prompt_mode in spec.prompt_modes:
            key = (model_label, prompt_mode)
            if key not in _SCHEDULING_WEIGHTS:
                raise RuntimeError(f"Missing scheduling weight for {key}")
            tasks.append(
                {
                    "task_id": _task_id(model_label, prompt_mode),
                    "priority": _SCHEDULING_WEIGHTS[key],
                    "model_label": model_label,
                    "model_id": spec.model_id,
                    "model_revision": MODEL_REVISIONS[model_label],
                    "prompt_mode": prompt_mode,
                    "reasoning_policy": spec.reasoning_policy,
                    "output_collection": collection,
                    "expected_requests": EXPECTED_STIMULI_PER_SHARD,
                }
            )
    tasks.sort(key=lambda task: (-int(task["priority"]), str(task["task_id"])))

    task_ids = [str(task["task_id"]) for task in tasks]
    if len(tasks) != 29 or len(task_ids) != len(set(task_ids)):
        raise RuntimeError("The formal plan must contain 29 unique shards")
    if sum(int(task["expected_requests"]) for task in tasks) != (
        EXPECTED_FORMAL_REQUESTS
    ):
        raise RuntimeError("The formal shard plan must contain 14,500 requests")
    return tasks


def formal_shard_plan() -> dict[str, Any]:
    tasks = formal_shard_tasks()
    canonical = json.dumps(
        tasks,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": FORMAL_SHARD_PLAN_SCHEMA,
        "protocol_version": "realistic_niah_v2",
        "query_layout": QUERY_LAYOUT,
        "expected_stimuli_per_shard": EXPECTED_STIMULI_PER_SHARD,
        "expected_shards": len(tasks),
        "expected_requests": EXPECTED_FORMAL_REQUESTS,
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
