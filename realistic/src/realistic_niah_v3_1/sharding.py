from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from typing import Any

from realistic_niah.spec import QUERY_LAYOUT
from realistic_niah_v3.sharding import (
    _MODE_SCHEDULING_WEIGHT,
    _MODEL_SCHEDULING_WEIGHT,
)

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
    V31_RUN_PROTOCOL,
)

FORMAL_SHARD_PLAN_SCHEMA = "realistic_niah_formal_shard_plan_v3_1"
FORMAL_BUNDLE_PLAN_SCHEMA = "realistic_niah_physical_bundle_plan_v3_1"


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
                }
            )
    tasks.sort(key=lambda task: (-int(task["priority"]), str(task["task_id"])))
    task_ids = [str(task["task_id"]) for task in tasks]
    if len(tasks) != EXPECTED_SHARDS or len(task_ids) != len(set(task_ids)):
        raise RuntimeError("The V3.1 plan must contain 48 unique shards")
    if sum(int(task["expected_requests"]) for task in tasks) != EXPECTED_REQUESTS:
        raise RuntimeError("The V3.1 plan must contain 161,280 requests")
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
        "request_id_namespace": V31_RUN_PROTOCOL.request_id_namespace,
        "expected_stimuli_per_shard": EXPECTED_STIMULI,
        "expected_shards": EXPECTED_SHARDS,
        "expected_requests": EXPECTED_REQUESTS,
        "raw_checkpoint_count": len(MODEL_LABELS),
        "behavior_comparison_slots": (
            len(SWITCHABLE_MODEL_LABELS) + len(MATCHED_REASONING_MODEL_LABELS)
        ),
        "tasks_sha256": hashlib.sha256(canonical).hexdigest(),
        "tasks": tasks,
    }


def formal_bundle_tasks() -> list[dict[str, Any]]:
    """Group logical model-mode shards into one physical load per model."""

    logical = formal_shard_tasks()
    bundles: list[dict[str, Any]] = []
    for model_label in MODEL_LABELS:
        tasks = [task for task in logical if task["model_label"] == model_label]
        bundles.append(
            {
                "bundle_id": re.sub(r"[^A-Za-z0-9_.-]+", "-", model_label).strip("-"),
                "priority": max(int(task["priority"]) for task in tasks),
                "model_label": model_label,
                "model_id": tasks[0]["model_id"],
                "model_revision": tasks[0]["model_revision"],
                "prompt_modes": [str(task["prompt_mode"]) for task in tasks],
                "logical_task_ids": [str(task["task_id"]) for task in tasks],
                "expected_logical_shards": len(tasks),
                "expected_requests": sum(
                    int(task["expected_requests"]) for task in tasks
                ),
            }
        )
    bundles.sort(key=lambda task: (-int(task["priority"]), str(task["bundle_id"])))
    if len(bundles) != len(MODEL_LABELS):
        raise RuntimeError("V3.1 bundle plan must contain one bundle per model")
    logical_ids = [
        task_id for bundle in bundles for task_id in bundle["logical_task_ids"]
    ]
    if len(logical_ids) != EXPECTED_SHARDS or len(set(logical_ids)) != EXPECTED_SHARDS:
        raise RuntimeError("V3.1 bundle plan must cover 48 logical shards once")
    if sum(int(bundle["expected_requests"]) for bundle in bundles) != EXPECTED_REQUESTS:
        raise RuntimeError("V3.1 bundle plan request accounting is invalid")
    return bundles


def formal_bundle_plan() -> dict[str, Any]:
    bundles = formal_bundle_tasks()
    canonical = json.dumps(
        bundles,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": FORMAL_BUNDLE_PLAN_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "physical_bundles": len(bundles),
        "logical_shards": EXPECTED_SHARDS,
        "expected_requests": EXPECTED_REQUESTS,
        "physical_model_loads": len(bundles),
        "legacy_model_mode_loads": EXPECTED_SHARDS,
        "loads_avoided": EXPECTED_SHARDS - len(bundles),
        "bundles_sha256": hashlib.sha256(canonical).hexdigest(),
        "bundles": bundles,
    }


def expected_request_ids(
    stimulus_ids: Iterable[str],
    task: dict[str, Any],
) -> tuple[str, ...]:
    return tuple(
        "/".join(
            (
                V31_RUN_PROTOCOL.request_id_namespace,
                str(task["model_label"]),
                str(task["prompt_mode"]),
                QUERY_LAYOUT,
                stimulus_id,
            )
        )
        for stimulus_id in stimulus_ids
    )
