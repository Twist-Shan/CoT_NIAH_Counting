from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from realistic_niah.spec import QUERY_LAYOUT

from .spec import (
    EXPECTED_REQUESTS,
    EXPECTED_SHARDS,
    EXPECTED_STIMULI,
    MATCHED_CONTROL_MODEL_LABELS,
    MATCHED_REASONING_MODEL_LABELS,
    MODEL_LABELS,
    MODEL_REVISIONS,
    MODEL_SPECS,
    PROTOCOL_VERSION,
    SWITCHABLE_MODEL_LABELS,
    V3_RUN_PROTOCOL,
)

FORMAL_SHARD_PLAN_SCHEMA = (
    "realistic_niah_formal_shard_plan_v3_resource_aware_v1"
)


@dataclass(frozen=True)
class ResourceProfile:
    gpus_required: int
    tensor_parallel_size: int
    request_batch_size: int
    max_num_seqs: int
    gpu_memory_utilization: float
    max_model_len: int = 32_768


MODEL_RESOURCE_PROFILES = {
    "Gemma4-31B": ResourceProfile(2, 2, 1, 1, 0.90),
    "Qwen3-32B": ResourceProfile(1, 1, 1, 1, 0.92),
    "Gemma4-26B-A4B": ResourceProfile(1, 1, 2, 2, 0.92),
    "Qwen3-14B": ResourceProfile(1, 1, 2, 2, 0.92),
    "Gemma4-12B": ResourceProfile(1, 1, 4, 4, 0.90),
    "Nemotron-Nano-v2-9B": ResourceProfile(1, 1, 4, 4, 0.90),
    "GLM-4-9B-0414": ResourceProfile(1, 1, 4, 4, 0.90),
    "GLM-Z1-9B-0414": ResourceProfile(1, 1, 4, 4, 0.90),
    "Qwen3-8B": ResourceProfile(1, 1, 6, 6, 0.90),
    "Gemma4-E4B": ResourceProfile(1, 1, 6, 6, 0.90),
    "Ministral-3-Instruct-8B": ResourceProfile(1, 1, 6, 6, 0.90),
    "Ministral-3-Reasoning-8B": ResourceProfile(1, 1, 6, 6, 0.90),
    "Qwen3-4B": ResourceProfile(1, 1, 8, 8, 0.90),
    "Nemotron-3-Nano-4B": ResourceProfile(1, 1, 8, 8, 0.90),
}


def resource_profile(model_label: str) -> ResourceProfile:
    try:
        profile = MODEL_RESOURCE_PROFILES[model_label]
    except KeyError as exc:
        raise KeyError(
            f"No V3 resource profile registered for {model_label}"
        ) from exc
    if profile.gpus_required != profile.tensor_parallel_size:
        raise RuntimeError(
            f"V3 profile must allocate one GPU per TP rank: {model_label}"
        )
    return profile

_MODEL_SCHEDULING_WEIGHT = {
    "Qwen3-4B": 40,
    "Qwen3-8B": 80,
    "Qwen3-14B": 140,
    "Qwen3-32B": 320,
    "Gemma4-E4B": 80,
    "Gemma4-12B": 120,
    "Gemma4-26B-A4B": 260,
    "Gemma4-31B": 310,
    "Nemotron-Nano-v2-9B": 90,
    "Nemotron-3-Nano-4B": 50,
    "GLM-4-9B-0414": 90,
    "GLM-Z1-9B-0414": 100,
    "Ministral-3-Instruct-8B": 80,
    "Ministral-3-Reasoning-8B": 90,
}
_MODE_SCHEDULING_WEIGHT = {
    "direct": 0,
    "enumeration_index": 25,
    "enumeration_bullet": 25,
    "native_thinking": 50,
}


def _task_id(model_label: str, prompt_mode: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_label).strip("-")
    return f"{normalized}__{prompt_mode}"


def _output_collection(model_label: str) -> str:
    if model_label in MATCHED_CONTROL_MODEL_LABELS:
        return "matched_controls"
    if model_label in MATCHED_REASONING_MODEL_LABELS:
        return "matched_reasoning"
    return "models"


def formal_shard_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for model_label in MODEL_LABELS:
        spec = MODEL_SPECS[model_label]
        resources = resource_profile(model_label)
        for prompt_mode in spec.prompt_modes:
            tasks.append(
                {
                    "task_id": _task_id(model_label, prompt_mode),
                    "priority": (
                        _MODEL_SCHEDULING_WEIGHT[model_label]
                        + _MODE_SCHEDULING_WEIGHT[prompt_mode]
                    ),
                    "model_label": model_label,
                    "model_id": spec.model_id,
                    "model_revision": MODEL_REVISIONS[model_label],
                    "prompt_mode": prompt_mode,
                    "reasoning_policy": spec.reasoning_policy,
                    "output_collection": _output_collection(model_label),
                    "expected_requests": EXPECTED_STIMULI,
                    **asdict(resources),
                }
            )
    tasks.sort(key=lambda task: (-int(task["priority"]), str(task["task_id"])))
    task_ids = [str(task["task_id"]) for task in tasks]
    if len(tasks) != EXPECTED_SHARDS or len(task_ids) != len(set(task_ids)):
        raise RuntimeError("The V3 formal plan must contain 48 unique shards")
    if sum(int(task["expected_requests"]) for task in tasks) != EXPECTED_REQUESTS:
        raise RuntimeError("The V3 formal plan must contain 47,040 requests")
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
        "protocol_version": PROTOCOL_VERSION,
        "query_layout": QUERY_LAYOUT,
        "request_id_namespace": V3_RUN_PROTOCOL.request_id_namespace,
        "expected_stimuli_per_shard": EXPECTED_STIMULI,
        "expected_shards": EXPECTED_SHARDS,
        "expected_requests": EXPECTED_REQUESTS,
        "raw_checkpoint_count": len(MODEL_LABELS),
        "behavior_comparison_slots": (
            len(SWITCHABLE_MODEL_LABELS)
            + len(MATCHED_REASONING_MODEL_LABELS)
        ),
        "scheduler": {
            "algorithm": "resource_aware_greedy_backfill",
            "maximum_supported_gpus": 8,
            "allocation_unit": "whole visible GPU",
            "multi_gpu_tasks": [
                task["task_id"]
                for task in tasks
                if int(task["gpus_required"]) > 1
            ],
        },
        "tasks_sha256": hashlib.sha256(canonical).hexdigest(),
        "tasks": tasks,
    }


def expected_request_ids(
    stimulus_ids: Iterable[str],
    task: dict[str, Any],
) -> tuple[str, ...]:
    model_label = str(task["model_label"])
    prompt_mode = str(task["prompt_mode"])
    namespace = V3_RUN_PROTOCOL.request_id_namespace
    return tuple(
        "/".join(
            (
                str(namespace),
                model_label,
                prompt_mode,
                QUERY_LAYOUT,
                stimulus_id,
            )
        )
        for stimulus_id in stimulus_ids
    )
