from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from typing import Any

from .spec import (
    EXTENSION_MODEL_REVISIONS,
    EXTENSION_MODEL_SPECS,
    NONTHINKING_PROMPT_MODES,
    QUERY_LAYOUT,
)

OLMO3_EXTENSION_PLAN_SCHEMA = "realistic_niah_olmo3_extension_plan_v1"
OLMO3_EXTENSION_PROTOCOL = "realistic_niah_v2_olmo3_extension_v1"
OLMO3_LOGICAL_MODEL_LABEL = "Olmo3-7B"
SOURCE_FORMAL_STIMULI_SHA256 = (
    "b739122c96adf73ec6df4abe0266af239a026b4de6f09f309933231f604c7f71"
)
EXPECTED_STIMULI_PER_SHARD = 500
EXPECTED_EXTENSION_SHARDS = 4
EXPECTED_EXTENSION_REQUESTS = 2_000

_TASK_PRIORITIES = {
    ("Olmo3-7B-Think", "native_thinking"): 40,
    ("Olmo3-7B-Instruct", "enumeration_index"): 30,
    ("Olmo3-7B-Instruct", "enumeration_bullet"): 25,
    ("Olmo3-7B-Instruct", "direct"): 20,
}


def _task_id(model_label: str, prompt_mode: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_label).strip("-")
    return f"{normalized}__{prompt_mode}"


def olmo3_extension_tasks() -> list[dict[str, Any]]:
    """Return the immutable four-shard OLMo 3 extension plan."""

    modes_by_label = {
        "Olmo3-7B-Instruct": NONTHINKING_PROMPT_MODES,
        "Olmo3-7B-Think": ("native_thinking",),
    }
    tasks: list[dict[str, Any]] = []
    for model_label, prompt_modes in modes_by_label.items():
        spec = EXTENSION_MODEL_SPECS[model_label]
        for prompt_mode in prompt_modes:
            key = (model_label, prompt_mode)
            tasks.append(
                {
                    "task_id": _task_id(model_label, prompt_mode),
                    "priority": _TASK_PRIORITIES[key],
                    "logical_model_label": OLMO3_LOGICAL_MODEL_LABEL,
                    "model_label": model_label,
                    "model_id": spec.model_id,
                    "model_revision": EXTENSION_MODEL_REVISIONS[model_label],
                    "prompt_mode": prompt_mode,
                    "reasoning_policy": spec.reasoning_policy,
                    "output_collection": "models",
                    "expected_requests": EXPECTED_STIMULI_PER_SHARD,
                }
            )
    tasks.sort(key=lambda task: (-int(task["priority"]), str(task["task_id"])))

    task_ids = [str(task["task_id"]) for task in tasks]
    if len(tasks) != EXPECTED_EXTENSION_SHARDS:
        raise RuntimeError("The OLMo 3 extension must contain four shards")
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("The OLMo 3 extension task IDs must be unique")
    if sum(int(task["expected_requests"]) for task in tasks) != (
        EXPECTED_EXTENSION_REQUESTS
    ):
        raise RuntimeError("The OLMo 3 extension must contain 2,000 requests")
    return tasks


def olmo3_extension_plan() -> dict[str, Any]:
    tasks = olmo3_extension_tasks()
    canonical = json.dumps(
        tasks,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": OLMO3_EXTENSION_PLAN_SCHEMA,
        "protocol_version": OLMO3_EXTENSION_PROTOCOL,
        "source_protocol_version": "realistic_niah_v2",
        "logical_model_label": OLMO3_LOGICAL_MODEL_LABEL,
        "query_layout": QUERY_LAYOUT,
        "source_stimuli_sha256": SOURCE_FORMAL_STIMULI_SHA256,
        "expected_stimuli_per_shard": EXPECTED_STIMULI_PER_SHARD,
        "expected_shards": EXPECTED_EXTENSION_SHARDS,
        "expected_requests": EXPECTED_EXTENSION_REQUESTS,
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
